"""Starting, tracking and stopping turns — the piece the HTTP layer talks to.

Two admission rules, and they refuse for different reasons because the client
does different things with each:

* **taken** — this CONVERSATION is already replying. The message was never read
  by a model, so the client must be able to put it back in the composer rather
  than believe it was sent.
* **busy** — this ENDPOINT is at capacity. A backstop: the client is told the
  limit by `/api/sessions` and blocks its own composer, so reaching here is a
  race, not the normal path.

The limit is per endpoint because the real ceiling is the model behind it. A
24 GB card running `--max-running-requests 1` and whatever serves the next
endpoint are different numbers, and one global limit can only be right for one
of them.

`run_conversation` blocks, so each turn owns a thread. Threads are what the
limit counts; the agents themselves outlive their turns in the pool's LRU.
"""

from __future__ import annotations

import logging
import secrets
import threading
from typing import Any, Callable

from hermes_agent import AgentPool, Endpoint, history_for, run_turn
from turn_stream import EventSink, TurnStream

log = logging.getLogger("buda.turns")


class Refused(Exception):
    """Admission refused. `reason` is what the client branches on."""

    def __init__(self, reason: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.extra = extra

    def as_json(self) -> dict:
        return {"error": self.message, self.reason: True, **self.extra}


class TurnManager:
    def __init__(
        self,
        pool: AgentPool,
        *,
        history: Callable[[str], list] = history_for,
        run: Callable[..., dict] = run_turn,
    ) -> None:
        self._pool = pool
        self._history = history
        self._run = run
        self._lock = threading.Lock()
        # By SESSION, not one global: a reader looking at an idle conversation
        # while another streams must see THAT conversation's state. Deriving it
        # from a global is how a busy session's spinner lands on an idle one.
        self._live: dict[str, TurnStream] = {}
        self._streams: dict[str, TurnStream] = {}

    # ── queries ────────────────────────────────────────────────────────────

    def stream(self, stream_id: str) -> TurnStream | None:
        with self._lock:
            return self._streams.get(stream_id)

    def live_for(self, session_id: str) -> TurnStream | None:
        with self._lock:
            s = self._live.get(session_id)
            return s if s is not None and s.running else None

    def running(self) -> dict[str, str]:
        """session_id -> stream_id for every turn still going."""
        with self._lock:
            return {sid: s.stream_id for sid, s in self._live.items() if s.running}

    def running_on(self, endpoint_key: str) -> int:
        with self._lock:
            return sum(
                1
                for s in self._live.values()
                if s.running and getattr(s, "endpoint_key", None) == endpoint_key
            )

    def forget_finished(self, keep: int = 200) -> None:
        """Drop the oldest finished streams. A finished stream is kept so a
        reader who was away can still collect its tail; it is not kept forever."""
        with self._lock:
            done = sorted(
                (s for s in self._streams.values() if not s.running and s.finished_at),
                key=lambda s: s.finished_at or 0.0,
            )
            for s in done[: max(0, len(done) - keep)]:
                self._streams.pop(s.stream_id, None)
                if self._live.get(s.session_id) is s:
                    self._live.pop(s.session_id, None)

    # ── starting ───────────────────────────────────────────────────────────

    def start(
        self,
        *,
        session_id: str,
        text: str,
        endpoint: Endpoint,
        client_id: str = "",
    ) -> TurnStream:
        """Admit and launch one turn. Raises `Refused` if it cannot start."""
        with self._lock:
            current = self._live.get(session_id)
            if current is not None and current.running:
                raise Refused(
                    "taken",
                    "这个会话正在回复中（另一个窗口发起的），消息未发出",
                    sessionId=session_id,
                    streamId=current.stream_id,
                )
            running = sum(
                1
                for s in self._live.values()
                if s.running and getattr(s, "endpoint_key", None) == endpoint.key
            )
            if running >= endpoint.max_concurrent:
                raise Refused(
                    "busy",
                    f"{endpoint.label} 已有 {running} 个会话在回复，达到上限 "
                    f"{endpoint.max_concurrent}",
                    running=running,
                    maxConcurrent=endpoint.max_concurrent,
                )
            stream = TurnStream(secrets.token_hex(8), session_id, client_id)
            # Which endpoint this turn is on, so the per-endpoint count above can
            # be taken without reaching back into the agent.
            stream.endpoint_key = endpoint.key  # type: ignore[attr-defined]
            self._streams[stream.stream_id] = stream
            self._live[session_id] = stream

        # The prompt is echoed into the log FIRST, so a reader attaching to this
        # stream sees the question above the answer even if they arrive late.
        stream.emit("user", text=text)

        threading.Thread(
            target=self._run_turn,
            args=(stream, session_id, text, endpoint),
            name=f"turn-{stream.stream_id}",
            daemon=True,
        ).start()
        return stream

    def _run_turn(
        self, stream: TurnStream, session_id: str, text: str, endpoint: Endpoint
    ) -> None:
        agent = None
        try:
            agent = self._pool.acquire(session_id, endpoint)
            self._pool.note_running(stream.stream_id, agent)
            from hermes_agent import bind_callbacks

            # EVERY turn, cached agent or fresh: a reused agent still carries the
            # previous turn's callbacks, which captured a stream nobody reads.
            bind_callbacks(agent, EventSink(stream))
            history = self._history(session_id)
            self._run(agent, session_id=session_id, user_message=text, history=history)
            stream.finish()
        except Exception as e:  # noqa: BLE001
            # The reader must be told. A turn that dies silently leaves the
            # composer locked and the spinner running until EventSource gives up.
            log.exception("turn %s failed", stream.stream_id)
            stream.finish(error=str(e) or e.__class__.__name__)
        finally:
            self._pool.clear_running(stream.stream_id)

    # ── stopping ───────────────────────────────────────────────────────────

    def cancel(self, stream_id: str) -> bool:
        """Ask the turn behind `stream_id` to stop.

        `stopped` is set BEFORE interrupting so the death it causes is reported
        as a stop rather than a failure — the reader pressed the button and must
        be told it worked.
        """
        stream = self.stream(stream_id)
        if stream is None or not stream.running:
            return False
        stream.stopped = True
        return self._pool.interrupt(stream_id, "user asked to stop")
