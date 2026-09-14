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


# How much of a tool's detail crosses the wire. The client clamps its DISPLAY
# further and offers to expand; this is the transport ceiling, and `detailFull`
# carries the true length so "there is more" can be told from "that is all".
DETAIL_MAX = 4000


def _invocation_text(preview, call_args) -> str:
    """What the tool was CALLED with, as one readable line.

    `preview` is hermes' own rendering and is preferred when it says something.
    Falling back to the argument dict matters for tools that supply no preview,
    and a single-valued dict reads better unwrapped: `{'command': 'ls'}` is
    worth showing as `ls`, not as its JSON.
    """
    if isinstance(preview, str) and preview.strip():
        return preview.strip()
    if isinstance(call_args, dict) and call_args:
        if len(call_args) == 1:
            only = next(iter(call_args.values()))
            if isinstance(only, str) and only.strip():
                return only.strip()
        try:
            import json

            return json.dumps(call_args, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(call_args)
    if isinstance(call_args, str) and call_args.strip():
        return call_args.strip()
    return ""


def _detail_for(invocation: str, result) -> str:
    """The row's body: what ran, then what it answered.

    `result` arrives as a JSON STRING from hermes' terminal tool
    (`{"output":…, "exit_code":N, "error":…}`), as a dict from others, and as
    plain text from the rest. All three are shown; a shape this does not
    recognise is printed rather than dropped, because an unreadable result still
    tells the reader more than an empty box.
    """
    import json

    parsed = result
    if isinstance(result, str):
        text = result.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = result

    lines: list[str] = []
    if invocation:
        lines.append(f"$ {invocation}" if "\n" not in invocation else invocation)

    if isinstance(parsed, dict):
        out = parsed.get("output")
        err = parsed.get("error")
        code = parsed.get("exit_code")
        body = "" if out is None else str(out)
        if body:
            lines.append(body)
        if err:
            lines.append(f"error: {err}")
        # Shown always when non-zero, and when there was nothing else to show --
        # "exit 0" with no output beats a box that looks like it failed to load.
        if code not in (None, 0):
            lines.append(f"exit {code}")
        elif code == 0 and not body and not err:
            lines.append("exit 0")
        if out is None and err is None and code is None:
            lines.append(json.dumps(parsed, ensure_ascii=False))
    elif parsed is not None:
        body = str(parsed).strip()
        if body:
            lines.append(body)

    return "\n".join(lines).strip()


class EventSink:
    """What `bind_callbacks` hands the agent: a turn's stream, in agent terms.

    Kept apart from `TurnStream` because the agent's vocabulary is not the
    wire's — it emits text deltas and tool progress, and the mapping to event
    kinds is this server's choice, not hermes'.
    """

    def __init__(self, stream: TurnStream) -> None:
        self.stream = stream
        # Pairing state for tools hermes announces without a call id.
        self._tool_seq = 0
        self._open: dict[str, list[str]] = {}
        # What each open row was invoked with, so the completed event can show
        # the command ABOVE its output -- `completed` carries neither.
        self._invocation: dict[str, str] = {}

    def delta(self, text: str, thought: bool = False) -> None:
        if not text:
            return
        self.stream.emit("delta", text=text, thought=thought)

    # hermes calls this POSITIONALLY:
    #   tool_progress_callback(event, name, preview, args, **extra)
    # e.g. ("tool.started", "skills_list", preview, args)
    #      ("tool.completed", "skills_list", None, None, duration=…, is_error=…)
    #      ("reasoning.available", "_thinking", text, None)
    # The first argument is the EVENT, the second is the tool. Reading the first
    # string as the label is why every row in the activity panel read
    # "tool.started" with no status and no name — the thing the reader actually
    # wants was in the argument after it, and was being thrown away.
    _EVENT_STATUS = {"tool.started": "running", "tool.completed": "completed"}

    def tool(self, *args, **kwargs) -> None:
        """hermes' tool-progress callback. Positional, with the shape above.

        The row carries WHAT RAN and WHAT CAME BACK. It used to carry neither:
        `preview`/`args` were dropped on the way in, `result` was dropped on the
        way out, and `detail` fell back to the duration -- so a reader got a box
        containing "1.0s" and no way to tell a failure's reason from a success's
        output. Both were in the callback the whole time.

        Still tolerant of a dict, but only as a FALLBACK. Merging every dict
        argument into the metadata put the tool's own arguments there, so a tool
        with a parameter named `status` or `title` would have rewritten the row.
        """
        pos = list(args)
        event = str(pos[0]) if pos and isinstance(pos[0], str) else ""
        name = str(pos[1]) if len(pos) > 1 and isinstance(pos[1], str) else ""
        preview = pos[2] if len(pos) > 2 else None
        call_args = pos[3] if len(pos) > 3 else None

        info = dict(kwargs)
        if not event:
            # An older or dict-shaped call site. Only here is merging safe:
            # there is no positional contract to read instead.
            for a in args:
                if isinstance(a, dict):
                    info.update(a)
            event = str(info.get("event") or "")
            name = str(info.get("name") or "")

        # `_thinking` is the reasoning channel wearing the tool callback's
        # signature; it has its own pane and is not a tool call.
        if name == "_thinking" or event == "reasoning.available":
            return

        title = str(info.get("title") or name or info.get("name") or event or "tool")
        status = str(info.get("status") or self._EVENT_STATUS.get(event, ""))

        # hermes supplies no call id on this path, so the id used to fall back to
        # the TITLE -- and two `terminal` calls in one turn then landed on one
        # row, the second overwriting the first. Pair them here instead: a
        # `started` opens a row, the next `completed` for that tool closes the
        # one that is still open.
        given = info.get("id") or info.get("tool_call_id")
        if given:
            row_id = str(given)
        elif status == "running":
            self._tool_seq += 1
            row_id = f"{title}#{self._tool_seq}"
            self._open.setdefault(title, []).append(row_id)
        else:
            waiting = self._open.get(title) or []
            row_id = waiting.pop(0) if waiting else f"{title}#0"

        invocation = _invocation_text(preview, call_args)
        if invocation:
            self._invocation[row_id] = invocation

        detail = str(info.get("detail") or "")
        if not detail:
            detail = _detail_for(self._invocation.get(row_id, ""), info.get("result"))

        # `is_error` is hermes' own verdict and the one we keep. A non-zero exit
        # code is NOT promoted to a failure here: `grep` answers 1 for "no
        # match", and calling that failed would be wrong. The code is written
        # into the detail instead, where the reader can see it and judge.
        if info.get("is_error"):
            status = "failed"

        if status != "running":
            self._open.get(title, [])
            self._invocation.pop(row_id, None)

        # hermes' own measurement, sent as its OWN field. It used to be the
        # fallback VALUE of `detail`, so a completed tool showed a box
        # containing "1.0s" where its output belonged. The row has a slot for
        # elapsed time already -- it was just counting client-side from when the
        # row appeared, which is why the row said 0s while the box said 1.0s.
        dur = ""
        try:
            if info.get("duration") is not None:
                dur = f"{float(info['duration']):.1f}s"
        except (TypeError, ValueError):
            dur = ""

        self.stream.emit(
            "tool",
            id=row_id,
            title=title,
            status=status,
            duration=dur,
            detail=detail[:DETAIL_MAX],
            # The TRUE length, so the client can say how much never arrived
            # rather than silently ending mid-value.
            detailFull=len(detail),
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
