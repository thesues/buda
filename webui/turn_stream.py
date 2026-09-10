"""One turn's event log — sequence-numbered, bounded, replayable, thread-safe.

Ported from the asyncio version unchanged in SEMANTICS and changed in exactly
one mechanism: the wake-up. The agent now runs on a worker thread and calls the
callbacks from there, while readers sit in their own request threads, so the
`asyncio.Event` that used to be replaced on every emit becomes a
`threading.Condition`. Everything a reader depends on — monotonic `seq`,
read-from-your-own-seq, the eviction gap, the terminal event ordering — is the
same, and the tests below pin each of those rather than the mechanism.

Why a shared condition rather than a queue per subscriber: subscribers re-read
from their own `seq`, so the notification carries no data and a queue would only
add a second place for the two to disagree about what has been delivered.
"""

from __future__ import annotations

import threading
import time
from collections import deque

# How many events one turn keeps for replay. A reader that reconnects asking for
# something older than this is told a gap exists rather than handed a transcript
# with a silent hole in it.
BACKLOG_EVENTS = 2000


class TurnStream:
    def __init__(
        self,
        stream_id: str,
        session_id: str | None,
        client_id: str = "",
        backlog: int = BACKLOG_EVENTS,
    ) -> None:
        self.stream_id = stream_id
        self.session_id = session_id
        # Which browser asked for this turn. Used ONLY to tell a double-click
        # apart from a second person typing into the same conversation; anyone
        # may still READ this stream, which is what makes "go back to the
        # conversation that is replying" work for whoever is looking.
        self.client_id = client_id
        self.seq = 0
        self.events: deque[tuple[int, dict]] = deque(maxlen=backlog)
        self.dropped = 0
        self.running = True
        # Set before a deliberate restart, so the death it causes is not
        # reported as a failure: the reader pressed Stop and must be told it
        # stopped.
        self.stopped = False
        self.finished_at: float | None = None
        self.error: str | None = None
        self._cv = threading.Condition()

    # ── producer side (the agent's worker thread) ──────────────────────────

    def emit(self, kind: str, **data) -> None:
        with self._cv:
            self.seq += 1
            if len(self.events) == self.events.maxlen:
                self.dropped += 1
            # Every event says which conversation it belongs to. The reader can
            # be looking somewhere else — that is the point of browsing
            # mid-turn — and without this the client paints one session's tokens
            # into whatever transcript happens to be on screen.
            self.events.append(
                (self.seq, {"kind": kind, "seq": self.seq, "session": self.session_id, **data})
            )
            self._cv.notify_all()

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self.finished_at = time.time()
        # Emit BEFORE lowering `running`. A reader drains, then tests `running`;
        # lowering it first opens a window where the reader breaks out with the
        # terminal event still unread, and the browser is left believing the
        # turn is live until EventSource reconnects on its own schedule.
        self.emit("end", error=error)
        with self._cv:
            self.running = False
            self._cv.notify_all()

    # ── consumer side (a request thread writing SSE) ───────────────────────

    def after(self, seq: int) -> list[dict]:
        with self._cv:
            return [e for s, e in self.events if s > seq]

    def gap_before(self, seq: int) -> bool:
        """Did eviction eat anything this reader has not seen?"""
        with self._cv:
            if not self.events:
                return False
            return seq < self.events[0][0] - 1

    def wait(self, seq: int, timeout: float) -> bool:
        """Block until there is something after `seq`, or the turn ends.

        Returns True if the caller should look again. The timeout is what lets
        the SSE writer send a keep-alive on an idle turn — a long prefill emits
        nothing for minutes, and a proxy with an idle timeout will drop a
        connection that says nothing at all.
        """
        deadline = time.monotonic() + timeout
        with self._cv:
            while self.running and self.seq <= seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return self.seq > seq or not self.running


class EventSink:
    """What `bind_callbacks` hands the agent: a turn's stream, in agent terms.

    Kept apart from `TurnStream` because the agent's vocabulary is not the
    wire's — it emits text deltas and tool progress, and the mapping to event
    kinds is this server's choice, not hermes'.
    """

    def __init__(self, stream: TurnStream) -> None:
        self.stream = stream

    def delta(self, text: str, thought: bool = False) -> None:
        if not text:
            return
        self.stream.emit("delta", text=text, thought=thought)

    def tool(self, *args, **kwargs) -> None:
        """hermes' tool-progress callback signature differs across versions and
        call sites, so accept anything and pull out what is recognisable rather
        than binding to one arity that a later version quietly changes."""
        info = dict(kwargs)
        for a in args:
            if isinstance(a, dict):
                info.update(a)
            elif isinstance(a, str) and "title" not in info:
                info["title"] = a
        title = str(info.get("title") or info.get("name") or info.get("tool") or "tool")
        self.stream.emit(
            "tool",
            id=str(info.get("id") or info.get("tool_call_id") or title),
            title=title,
            status=str(info.get("status") or ""),
            detail=str(info.get("detail") or ""),
        )

    def step(self, *args, **kwargs) -> None:
        """Steps are progress, not content. Dropped unless they name something:
        a bare tick would push the transcript around for no information."""
        text = next((a for a in args if isinstance(a, str) and a.strip()), "")
        if not text:
            text = str(kwargs.get("text") or kwargs.get("message") or "").strip()
        if text:
            self.stream.emit("note", text=text)

    def note(self, text: str) -> None:
        if text:
            self.stream.emit("note", text=text)
