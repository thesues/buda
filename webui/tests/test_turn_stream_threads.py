"""The turn log's invariants, now that producer and consumer are real threads.

Each test below is a reader-visible failure, not a property of the mechanism:
a transcript with a silent hole, a turn that looks live after it ended, tokens
painted into the wrong conversation, or a reader that never wakes.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turn_stream import EventSink, TurnStream  # noqa: E402


def test_a_reader_resumes_from_its_own_seq():
    s = TurnStream("st", "sess")
    s.emit("delta", text="a")
    s.emit("delta", text="b")
    assert [e["text"] for e in s.after(0)] == ["a", "b"]
    assert [e["text"] for e in s.after(1)] == ["b"]
    assert s.after(2) == []


def test_every_event_names_its_conversation():
    """A reader can be looking at a DIFFERENT conversation while this one
    streams — that is the point of browsing mid-turn. Without the session on
    each event the client paints these tokens into whatever is on screen."""
    s = TurnStream("st", "sess-7")
    s.emit("delta", text="x")
    assert s.after(0)[0]["session"] == "sess-7"


def test_the_terminal_event_is_emitted_while_the_turn_still_reads_as_running():
    """A reader drains, then tests `running`. If `running` dropped FIRST, the
    reader could break out with `end` still unread and the browser would go on
    believing the turn is live until EventSource reconnects on its own.

    Observed at the emit itself rather than from a watcher thread: the window
    between the two statements is nanoseconds, so a polling observer proves
    nothing about the ordering — an earlier version of this test passed with the
    order reversed.
    """
    s = TurnStream("st", "sess")
    running_when_end_was_emitted: list[bool] = []
    real_emit = s.emit

    def spy(kind: str, **data):
        if kind == "end":
            running_when_end_was_emitted.append(s.running)
        real_emit(kind, **data)

    s.emit = spy  # type: ignore[method-assign]
    s.finish()
    assert running_when_end_was_emitted == [True]
    assert s.running is False
    assert [e["kind"] for e in s.after(0)] == ["end"]


def test_eviction_is_reported_as_a_gap_not_a_silent_hole():
    s = TurnStream("st", "sess", backlog=4)
    for i in range(10):
        s.emit("delta", text=str(i))
    assert s.dropped == 6
    # A reader that saw nothing cannot be served from a window starting at 7.
    assert s.gap_before(0) is True
    # One that is already inside the window can.
    assert s.gap_before(9) is False


def test_a_waiting_reader_is_woken_by_an_emit():
    s = TurnStream("st", "sess")
    woke: list[bool] = []

    def reader():
        woke.append(s.wait(0, timeout=2.0))

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.02)
    s.emit("delta", text="hi")
    t.join(timeout=3)
    assert woke == [True]


def test_a_waiting_reader_is_woken_by_the_turn_ending():
    """Otherwise a reader on a turn that produced nothing more sits until its
    timeout, and the browser shows a spinner past the end of the answer."""
    s = TurnStream("st", "sess")
    woke: list[bool] = []
    t = threading.Thread(target=lambda: woke.append(s.wait(999, timeout=2.0)))
    t.start()
    time.sleep(0.02)
    s.finish()
    t.join(timeout=3)
    assert woke == [True]


def test_waiting_times_out_so_an_idle_turn_can_be_kept_alive():
    """A long prefill emits nothing for minutes. The writer needs to regain
    control to send a keep-alive, or a proxy drops a connection that has said
    nothing at all."""
    s = TurnStream("st", "sess")
    t0 = time.monotonic()
    assert s.wait(0, timeout=0.05) is False
    assert time.monotonic() - t0 >= 0.05


def test_concurrent_emits_keep_seq_dense_and_ordered():
    """The agent's callbacks fire from a worker thread; `seq` is what the whole
    resume protocol rests on, so a lost or duplicated one is a reader that
    silently skips or repeats output."""
    s = TurnStream("st", "sess", backlog=10000)
    threads = [
        threading.Thread(target=lambda: [s.emit("delta", text="x") for _ in range(200)])
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [e["seq"] for e in s.after(0)]
    assert seqs == list(range(1, 1601))


# ── the sink: agent vocabulary -> wire events ───────────────────────────────


def test_reasoning_and_content_are_the_same_kind_flagged_apart():
    s = TurnStream("st", "sess")
    sink = EventSink(s)
    sink.delta("answer")
    sink.delta("pondering", thought=True)
    got = [(e["text"], e["thought"]) for e in s.after(0)]
    assert got == [("answer", False), ("pondering", True)]


def test_an_empty_delta_emits_nothing():
    """Providers send empty chunks; each one would otherwise cost a `seq` and a
    wake-up for no content."""
    s = TurnStream("st", "sess")
    EventSink(s).delta("")
    assert s.after(0) == []


def test_tool_progress_survives_a_signature_this_server_did_not_predict():
    """hermes' tool callback differs by version and call site. Binding to one
    arity means a later version raises inside the agent's own callback — during
    a turn, where the failure is hardest to attribute."""
    s = TurnStream("st", "sess")
    sink = EventSink(s)
    sink.tool({"id": "t1", "title": "search", "status": "running"})
    sink.tool("grep", status="completed")
    sink.tool(id="t3", name="read")
    kinds = [(e["title"], e["status"]) for e in s.after(0)]
    assert kinds == [("search", "running"), ("grep", "completed"), ("read", "")]


def test_a_step_with_nothing_to_say_is_dropped():
    s = TurnStream("st", "sess")
    sink = EventSink(s)
    sink.step()
    sink.step("")
    assert s.after(0) == []
    sink.step("thinking about it")
    assert s.after(0)[0]["text"] == "thinking about it"
