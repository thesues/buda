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

`run_conversation` blocks, so a turn occupies a thread for its whole life —
but it runs on `_TurnPool`, a fixed set of daemon workers sized to the
admission budget (Σ `max_concurrent`), not a fresh `threading.Thread` per
turn. Daemon on purpose: `concurrent.futures` workers are non-daemon and
joined at interpreter exit on 3.9+ (bpo-39812), and a rollout's SIGTERM must
drop an in-flight turn, not wait out the approval timeout it may be parked
in. Admission still counts live turns per endpoint — the pool buys reuse and
a named worker set, it is not the limit.
"""

from __future__ import annotations

import logging
import queue
import secrets
import os
import threading
import uuid
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


# How long a permission card waits before the turn gives up on it. Long,
# because the person it is asking may be away from the tab; bounded, because a
# turn blocked forever holds its thread and its agent.
APPROVAL_TIMEOUT_S = int(os.environ.get("BUDA_APPROVAL_TIMEOUT_S", "600"))


class _TurnPool:
    """Fixed daemon workers pulling from a queue.

    A hand-rolled pool, not `concurrent.futures.ThreadPoolExecutor`, for one
    property: these workers are daemon threads. Executor workers are joined at
    interpreter exit since 3.9 (bpo-39812), and SIGTERM here means "lose the
    turn" — a turn parked in a 600 s approval wait must not hold a rollout.

    Sized by the caller to the admission budget, so an admitted turn always
    finds a worker. `submit` still checks: if every worker is somehow busy
    (endpoints grew after boot), the turn spawns its own daemon thread rather
    than queueing behind a turn that may outlast the reader's patience.
    Liveness beats tidiness.
    """

    def __init__(self, workers: int, prefix: str = "turn") -> None:
        self._q: "queue.SimpleQueue[tuple[str, Any, tuple]]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._busy = 0
        self._workers = max(1, workers)
        for i in range(self._workers):
            threading.Thread(target=self._loop, name=f"{prefix}-{i}", daemon=True).start()

    def _loop(self) -> None:
        while True:
            name, target, args = self._q.get()
            with self._lock:
                self._busy += 1
            # Carry the turn's stream id into the thread name while it runs —
            # py-spy dumps and log lines keep pointing at a stream — then
            # restore it, because this worker outlives many turns.
            t = threading.current_thread()
            base = t.name
            try:
                t.name = name
                target(*args)
            except BaseException:  # noqa: BLE001 -- a dead worker must not die
                log.exception("turn worker crashed")
            finally:
                t.name = base
                with self._lock:
                    self._busy -= 1

    def submit(self, name: str, target: Any, *args: Any) -> None:
        """Run target(name-bearing) on a worker, or on a spare daemon thread
        if every worker is busy. Never blocks the caller, never queues behind
        a turn whose stream may already be abandoned."""
        with self._lock:
            if self._busy < self._workers:
                self._q.put((name, target, args))
                return
        log.warning("turn pool (%d workers) saturated; spawning a spare thread", self._workers)
        threading.Thread(target=target, args=args, name=name, daemon=True).start()


class TurnManager:
    def __init__(
        self,
        pool: AgentPool,
        *,
        history: Callable[[str], list] = history_for,
        run: Callable[..., dict] = run_turn,
        workers: int = 8,
    ) -> None:
        # 8 is a tests-and-small-deployments default; main() passes the exact
        # admission budget (Σ max_concurrent), which is the number a turn can
        # never exceed.
        self._turns = _TurnPool(workers)
        self._pool = pool
        self._history = history
        self._run = run
        self._lock = threading.Lock()
        # One contextvar token per live turn, so teardown resets exactly the
        # binding that turn made and not a later one's.
        self._approval_tokens: dict[str, Any] = {}
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

        self._turns.submit(f"turn-{stream.stream_id}", self._run_turn, stream, session_id, text, endpoint)
        return stream

    def _install_approval(self, stream: TurnStream) -> None:
        """Route this turn's permission prompts to this turn's reader.

        Through hermes' GATEWAY path, not `terminal_tool.set_approval_callback`.
        That setter stores the callback in a `threading.local` — deliberately,
        it is a fix for concurrent ACP sessions sharing one slot — and hermes
        dispatches tools on threads of its own. The callback was therefore
        invisible to the thread that actually asked, hermes fell through to its
        `input()` fallback, and that spawns a daemon thread reading a stdin no
        one will ever type into. Measured: a turn stopped at
        `terminal / running`, emitted nothing for minutes, held the endpoint's
        only slot, and `interrupt()` could not touch it because interrupt sets
        a flag the conversation loop reads and the loop was never reached. The
        module's own comment calls this "an invisible 60s deadlock".

        The gateway path is module-global instead: the session key is a
        contextvar (inherited by threads hermes spawns), the pending entry goes
        into `_gateway_queues`, and `resolve_gateway_approval` unblocks it from
        whichever thread the HTTP handler happens to be on. This is what
        hermes-webui does, for the same reason.

        `HERMES_GATEWAY_SESSION` is the switch `_is_gateway_approval_context()`
        reads; `HERMES_INTERACTIVE` tells the non-interactive auto-approve path
        to stay out of it.
        """
        import os

        try:
            from tools import approval as ap

            os.environ["HERMES_GATEWAY_SESSION"] = "1"
            os.environ["HERMES_INTERACTIVE"] = "1"
            key = self._approval_key(stream)
            # Pinned. hermes queues approvals under HERMES_SESSION_KEY, which
            # compression does not move (it moves HERMES_SESSION_ID), so after a
            # rotation the queue, the notify and the teardown all stay here.
            stream.approval_key = key  # type: ignore[attr-defined]

            # BOTH, and the env var is the one that carries.
            #
            # `get_current_session_key()` resolves contextvar → HERMES_SESSION_KEY
            # → "default". A contextvar is NOT inherited by a thread started with
            # `threading.Thread`: a new thread begins with an EMPTY context, not a
            # copy of its parent's. hermes dispatches tools on such threads, so it
            # asked under "default" while the notify callback was registered under
            # the conversation's id — no callback found, no card, and the turn
            # waited forever. That is the same wedge as before wearing a different
            # hat: the first was a thread-local callback, this is a thread-local
            # KEY.
            #
            # os.environ is process-wide and therefore thread-visible. It is
            # correct here because this app runs one turn at a time per endpoint
            # (`maxConcurrent`), and a second concurrent turn WOULD cross the two
            # keys — if that limit is ever raised, this has to become a per-thread
            # binding installed on hermes' side instead.
            os.environ["HERMES_SESSION_KEY"] = key
            self._approval_tokens[stream.stream_id] = ap.set_current_session_key(key)
            ap.register_gateway_notify(key, lambda data: self._notify_approval(stream, key, data))
            log.info("approval hook armed for %s (gateway)", key)
        except Exception:  # noqa: BLE001 -- a missing hook must not fail the turn
            log.warning("could not install the approval hook", exc_info=True)

    @staticmethod
    def _approval_key(stream: TurnStream) -> str:
        """One key per CONVERSATION, so "allow for this session" means what a
        reader thinks it means and a reconnect finds its own pending card."""
        return str(getattr(stream, "approval_key", None) or stream.session_id or stream.stream_id)

    def approval_key_for(self, session_id: str) -> str:
        """The gateway queue a conversation's approvals wait in. The session id
        itself — unless its live turn rotated, in which case the queue is still
        under the id the turn started with."""
        live = self.live_for(session_id)
        return self._approval_key(live) if live is not None else session_id

    def _notify_approval(self, stream: TurnStream, key: str, data: dict) -> None:
        """hermes has something to ask. Put it on this turn's stream.

        Runs on the agent's thread, inside the wait — emitting here is what
        makes the card appear while the turn is still blocked, which is the
        whole point.
        """
        opts = [
            {"optionId": "once", "name": "允许一次"},
            {"optionId": "session", "name": "本次会话都允许"},
        ]
        if data.get("allow_permanent", True):
            opts.append({"optionId": "always", "name": "始终允许"})
        opts.append({"optionId": "deny", "name": "拒绝"})
        title = str(data.get("description") or data.get("command") or "需要确认")
        stream.emit(
            "approval",
            id=key,                      # resolve_gateway_approval keys on this
            title=title,
            command=str(data.get("command") or ""),
            options=opts,
        )
        status_note = f"需要你确认：{title}"
        stream.emit("note", text=status_note)

    def _release_approval(self, stream: TurnStream) -> None:
        """Drop the turn's approval wiring, and deny anything still pending.

        A turn that ended with a card still on screen would otherwise leave an
        entry in `_gateway_queues` that nothing will ever answer, and the next
        approval for this conversation would queue behind it.
        """
        try:
            from tools import approval as ap

            key = self._approval_key(stream)
            ap.resolve_gateway_approval(key, "deny", resolve_all=True)
            if hasattr(ap, "unregister_gateway_notify"):
                ap.unregister_gateway_notify(key)
            tok = self._approval_tokens.pop(stream.stream_id, None)
            if tok is not None:
                ap.reset_current_session_key(tok)
        except Exception:  # noqa: BLE001
            log.debug("approval teardown skipped", exc_info=True)

    def _rotated(self, stream: TurnStream, old: str, new: str) -> None:
        """hermes compressed the context and continued under a new session id.

        Everything keyed by the conversation moves with it: the live map (so
        the sidebar marks the new row, and a second prompt into it is `taken`)
        and the agent cache (so the next turn under the new id continues with
        the agent that rotated, reading the child session's history). The old
        row stays in the store as the parent; nothing here deletes it.
        """
        with self._lock:
            if self._live.get(old) is stream:
                self._live.pop(old, None)
                self._live[new] = stream
        self._pool.rename(old, new)
        log.info("session %s rotated to %s mid-turn (stream %s)", old, new, stream.stream_id)

    def _run_turn(
        self, stream: TurnStream, session_id: str, text: str, endpoint: Endpoint
    ) -> None:
        agent = None
        try:
            agent = self._pool.acquire(session_id, endpoint)
            self._pool.note_running(stream.stream_id, agent)
            self._install_approval(stream)
            # After the approval key is pinned: it stays the id the hook was
            # registered under even if hermes rotates the session.
            stream.follow(
                lambda: getattr(agent, "session_id", None),
                lambda old, new: self._rotated(stream, old, new),
            )
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
            self._release_approval(stream)
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
        # Deny anything this conversation is parked on FIRST. `interrupt` only
        # sets a flag the conversation loop reads, and a turn blocked waiting
        # for permission is not in that loop — it is parked on the approval
        # queue, which is exactly where a reader who pressed Stop most wants it
        # to stop. Releasing it lets the turn reach the loop and see the flag.
        try:
            from tools import approval as ap

            n = ap.resolve_gateway_approval(self._approval_key(stream), "deny", resolve_all=True)
            if n:
                stream.emit("note", text="已停止：拒绝了等待中的确认")
        except Exception:  # noqa: BLE001 -- a failed release must not fail the stop
            log.debug("no pending approval to release for %s", stream_id, exc_info=True)
        return self._pool.interrupt(stream_id, "user asked to stop")
