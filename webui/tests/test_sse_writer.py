"""The SSE writer's rules, each one a reader-visible failure it prevents."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sse  # noqa: E402
from turn_stream import TurnStream  # noqa: E402


class Wire:
    """Collects frames; optionally hangs up after N writes."""

    def __init__(self, die_after: int | None = None):
        self.chunks: list[bytes] = []
        self._die_after = die_after

    def __call__(self, b: bytes) -> None:
        if self._die_after is not None and len(self.chunks) >= self._die_after:
            raise ConnectionResetError("peer went away")
        self.chunks.append(b)

    def events(self) -> list[dict]:
        out = []
        for c in self.chunks:
            for line in c.decode().splitlines():
                if line.startswith("data: "):
                    out.append(json.loads(line[6:]))
        return out

    def ids(self) -> list[str]:
        return [
            line[4:]
            for c in self.chunks
            for line in c.decode().splitlines()
            if line.startswith("id: ")
        ]


def _finished(*emits) -> TurnStream:
    s = TurnStream("st", "sess")
    for kind, data in emits:
        s.emit(kind, **data)
    s.finish()
    return s


# ── the resume cursor ───────────────────────────────────────────────────────


def test_a_reconnect_resumes_from_last_event_id_not_the_url():
    """The browser reconnects on its own and replays the id it saw. The query
    parameter is only the opening position of a FRESH attach — honouring it on a
    reconnect re-delivers the whole turn, which is how one prompt rendered
    twice."""
    assert sse.resolve_cursor(after=0, last_event_id="5", stream_seq=9) == 5


def test_a_cursor_we_cannot_honour_is_refused_rather_than_adopted():
    """Unparseable, or AHEAD of anything this stream emitted — a stale id from a
    previous process, or another stream's. Adopting it skips every frame up to
    it, the terminal event included, and the reader stalls on keepalives."""
    assert sse.resolve_cursor(after=2, last_event_id="not-a-number", stream_seq=9) == 2
    assert sse.resolve_cursor(after=2, last_event_id="99", stream_seq=9) == 2
    assert sse.resolve_cursor(after=2, last_event_id="-1", stream_seq=9) == 2


def test_no_last_event_id_keeps_the_fresh_attach_position():
    assert sse.resolve_cursor(after=3, last_event_id=None, stream_seq=9) == 3


# ── framing ─────────────────────────────────────────────────────────────────


def test_every_frame_carries_its_seq_as_the_event_id():
    """Without `id:` the browser's own reconnect has nothing to resume from."""
    s = _finished(("delta", {"text": "a"}), ("delta", {"text": "b"}))
    w = Wire()
    sse.write_stream(s, w)
    assert w.ids() == ["1", "2", "3"]  # two deltas + end


def test_the_reconnect_delay_is_sent_before_anything_else():
    s = _finished(("delta", {"text": "a"}))
    w = Wire()
    sse.write_stream(s, w)
    assert w.chunks[0] == b"retry: 750\n\n"


# ── delivery guarantees ─────────────────────────────────────────────────────


def test_the_reader_is_always_told_the_turn_ended():
    """A reader that attaches AFTER the turn finished has nothing outstanding.
    Closing silently leaves the browser believing the turn is live, reconnecting
    every 750 ms with the composer locked."""
    s = _finished(("delta", {"text": "a"}))
    w = Wire()
    sse.write_stream(s, w, after=s.seq)  # already past everything, `end` included
    kinds = [e["kind"] for e in w.events()]
    assert kinds == ["end"]


def test_a_turn_that_ends_mid_write_still_delivers_its_terminal_event():
    """The pending list is fixed before the first write, so a turn finishing
    during those writes lands `end` outside the pass. Draining again is what
    catches it."""
    s = TurnStream("st", "sess")
    s.emit("delta", text="a")

    class LateFinisher(Wire):
        def __call__(self, b: bytes) -> None:
            super().__call__(b)
            if b'"text": "a"' in b:
                s.finish()

    w = LateFinisher()
    sse.write_stream(s, w)
    assert [e["kind"] for e in w.events()] == ["delta", "end"]


def test_an_event_landing_between_nothing_pending_and_the_conclusion_is_not_lost():
    """The narrow window the second read exists for.

    The writer sees nothing pending, then sees the turn is over — and an event
    can land BETWEEN those two observations. Concluding on the first one drops
    it: the reader is told the turn ended and never receives its last tokens.
    Driven deterministically here, because an ordinary "finish during a write"
    is caught one branch earlier and leaves this guard unexercised — the version
    of this file before this test passed with the guard deleted.
    """

    class Interleaved(TurnStream):
        def __init__(self) -> None:
            super().__init__("st", "sess")
            self.reads = 0

        def after(self, seq: int):
            self.reads += 1
            if self.reads == 1:
                # "nothing pending, and the turn is over"
                self.running = False
            elif self.reads == 2:
                # ...and its final tokens arrive right here.
                TurnStream.emit(self, "delta", text="late")
            return TurnStream.after(self, seq)

    s = Interleaved()
    w = Wire()
    sse.write_stream(s, w)
    kinds = [e["kind"] for e in w.events()]
    assert kinds == ["delta", "end"], f"the late event must still be delivered, got {kinds}"


def test_eviction_is_announced_before_the_replay():
    """A reader resuming from before the window would otherwise receive a
    transcript with a silent hole."""
    s = TurnStream("st", "sess", backlog=3)
    for i in range(6):
        s.emit("delta", text=str(i))
    s.finish()
    w = Wire()
    sse.write_stream(s, w, after=0)
    assert w.events()[0]["kind"] == "gap"


def test_an_idle_turn_is_kept_alive_rather_than_left_silent():
    """A long prefill emits nothing for minutes; a proxy drops a connection that
    says nothing at all."""
    s = TurnStream("st", "sess")
    w = Wire()
    orig = sse.SSE_HEARTBEAT_SEC
    sse.SSE_HEARTBEAT_SEC = 0.02
    try:
        t = threading.Thread(target=lambda: sse.write_stream(s, w), daemon=True)
        t.start()
        time.sleep(0.15)
        s.finish()
        t.join(timeout=2)
    finally:
        sse.SSE_HEARTBEAT_SEC = orig
    assert any(c == b": keepalive\n\n" for c in w.chunks), "an idle turn must be kept alive"
    assert [e["kind"] for e in w.events()] == ["end"]


def test_a_reader_that_hangs_up_does_not_take_the_turn_with_it():
    """The write raises when the peer goes; that is the normal end of a reader
    and must reach the caller rather than being swallowed into a spin."""
    s = _finished(("delta", {"text": "a"}), ("delta", {"text": "b"}))
    w = Wire(die_after=2)
    try:
        sse.write_stream(s, w)
    except ConnectionResetError:
        pass
    else:
        raise AssertionError("the disconnect must surface to the caller")
    assert s.running is False and s.error is None, "the turn itself is untouched"


def test_a_live_turn_streams_incrementally_rather_than_at_the_end():
    """The whole point of SSE here. If the writer only flushed at the end, a
    reader would watch a spinner for the length of the answer."""
    s = TurnStream("st", "sess")
    w = Wire()
    t = threading.Thread(target=lambda: sse.write_stream(s, w), daemon=True)
    t.start()
    s.emit("delta", text="first")
    for _ in range(200):
        if any(b'"first"' in c for c in w.chunks):
            break
        time.sleep(0.005)
    assert any(b'"first"' in c for c in w.chunks), "the first token must arrive before the last"
    s.finish()
    t.join(timeout=2)
