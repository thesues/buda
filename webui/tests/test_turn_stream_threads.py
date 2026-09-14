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


# ── the tool callback's real signature ──────────────────────────────────────


def _tool_events(calls):
    """Run `sink.tool(*args, **kwargs)` for each call, return the emitted rows."""
    from turn_stream import EventSink, TurnStream

    st = TurnStream("st", "sess")
    sink = EventSink(st)
    for args, kwargs in calls:
        sink.tool(*args, **kwargs)
    return [e for e in st.after(0) if e["kind"] == "tool"]


def test_a_tool_row_is_named_after_the_tool_not_the_event():
    """hermes calls this positionally: `(event, name, preview, args, **extra)`.

    Reading the first string as the label made every row in the activity panel
    read `tool.started`, with no status and no name — the tool the reader
    actually wanted was in the next argument and was discarded. Verbatim from
    `agent/tool_executor.py`:

        agent.tool_progress_callback("tool.started", name, preview, args)
        agent.tool_progress_callback("tool.completed", function_name, None,
                                     None, duration=…, is_error=…)

    Ablation: take `pos[0]` as the title again and both assertions fail.
    """
    rows = _tool_events([
        (("tool.started", "skills_list", "preview", {"a": 1}), {}),
        (("tool.completed", "skills_list", None, None), {"duration": 1.25, "is_error": False}),
    ])
    assert [r["title"] for r in rows] == ["skills_list", "skills_list"]
    assert [r["status"] for r in rows] == ["running", "completed"]
    # Both halves key on the same row, or the panel shows the call twice.
    assert rows[0]["id"] == rows[1]["id"]
    # The duration is its OWN field. It used to be the fallback value of
    # `detail`, which put "1.2s" in the box the tool's output belongs in.
    assert rows[1]["duration"] == "1.2s", rows[1]
    assert "1.2s" not in rows[1]["detail"], rows[1]["detail"]


def test_a_failed_tool_says_so():
    rows = _tool_events([
        (("tool.completed", "search_files", None, None), {"is_error": True}),
    ])
    assert rows[0]["status"] == "failed"


def test_the_reasoning_channel_is_not_a_tool_row():
    """`reasoning.available` arrives through the same callback with `_thinking`
    as its name. It has its own pane; listing it as a tool call was noise."""
    rows = _tool_events([
        (("reasoning.available", "_thinking", "some thinking", None), {}),
    ])
    assert rows == []


# ── what a tool row actually SHOWS ──────────────────────────────────────────
#
# The payloads below are captured verbatim from the installed hermes by driving
# a real turn ("echo HELLO-OK", then "ls /definitely-not-here-xyz") with a
# recording callback. They are not a guess at the contract.
#
# The reader's complaint was that a tool row told them nothing: a failure gave
# no reason, a success gave no output, and the box under the row contained the
# duration. All of it was in the callback and thrown away here.


def test_the_command_and_its_output_both_reach_the_row():
    """Ablation: stop reading `preview`/`args` on started, or `result` on
    completed, and the detail goes back to being empty."""
    rows = _tool_events([
        (("tool.started", "terminal", "echo HELLO-OK", {"command": "echo HELLO-OK"}), {}),
        (("tool.completed", "terminal", None, None),
         {"duration": 0.0296, "is_error": False,
          "result": '{"output": "HELLO-OK", "exit_code": 0, "error": null}'}),
    ])
    assert "echo HELLO-OK" in rows[0]["detail"]
    assert "echo HELLO-OK" in rows[1]["detail"], "the command must survive to the result row"
    assert "HELLO-OK" in rows[1]["detail"]
    assert rows[0]["id"] == rows[1]["id"]


def test_a_nonzero_exit_is_visible_without_being_called_a_failure():
    """`grep` answers 1 for "no match", so a non-zero exit is not promoted to
    `failed` -- hermes' own `is_error` decides that. But the code must be ON the
    row, or a reader cannot tell the two apart at all."""
    rows = _tool_events([
        (("tool.started", "terminal", "ls /nope", {"command": "ls /nope"}), {}),
        (("tool.completed", "terminal", None, None),
         {"duration": 0.011, "is_error": False,
          "result": '{"output": "ls: cannot access \'/nope\': No such file or directory",'
                    ' "exit_code": 2, "error": null}'}),
    ])
    assert "No such file or directory" in rows[1]["detail"]
    assert "exit 2" in rows[1]["detail"]
    assert rows[1]["status"] == "completed"


def test_two_calls_to_one_tool_are_two_rows():
    """hermes sends no call id on this path, so the id fell back to the TITLE
    and a second `terminal` call overwrote the first -- one row for two
    commands, the earlier one gone.

    Ablation: key the row on the title again and these ids collapse."""
    rows = _tool_events([
        (("tool.started", "terminal", "echo one", {"command": "echo one"}), {}),
        (("tool.completed", "terminal", None, None),
         {"result": '{"output": "one", "exit_code": 0}'}),
        (("tool.started", "terminal", "echo two", {"command": "echo two"}), {}),
        (("tool.completed", "terminal", None, None),
         {"result": '{"output": "two", "exit_code": 0}'}),
    ])
    assert rows[0]["id"] == rows[1]["id"]
    assert rows[2]["id"] == rows[3]["id"]
    assert rows[0]["id"] != rows[2]["id"], "two calls collapsed onto one row"
    assert "one" in rows[1]["detail"] and "two" in rows[3]["detail"]


def test_a_tools_own_arguments_cannot_rewrite_its_row():
    """The args dict used to be merged into the event metadata, so a tool with a
    parameter called `status` or `title` rewrote the row describing it."""
    rows = _tool_events([
        (("tool.started", "search_files",
          "look", {"title": "NOT THE TOOL", "status": "failed"}), {}),
    ])
    assert rows[0]["title"] == "search_files"
    assert rows[0]["status"] == "running"


def test_the_true_length_is_sent_so_the_client_can_say_what_was_cut():
    """`detailFull` is a LENGTH. The client computes "还有 N 字" from it, so a
    string there made that arithmetic NaN."""
    big = "x" * 9000
    rows = _tool_events([
        (("tool.started", "terminal", "cat big", {"command": "cat big"}), {}),
        (("tool.completed", "terminal", None, None),
         {"result": '{"output": "' + big + '", "exit_code": 0}'}),
    ])
    assert isinstance(rows[1]["detailFull"], int)
    assert len(rows[1]["detail"]) <= 4000
    assert rows[1]["detailFull"] > len(rows[1]["detail"])


def test_an_unknown_event_kind_does_not_steal_an_open_row():
    """A row is closed by a POSITIVE condition, not by "anything not running".

    hermes already names `tool.failed` on the consuming side, and a delegate
    toolset brings `subagent.start`/`subagent.complete` through this same
    callback. Under "not running" any of them consumed the open row id, so the
    real completion found nothing, opened a SECOND row, and the first was
    stranded at "running" forever.

    Ablation: close the row on `status != "running"` and the ids diverge."""
    rows = _tool_events([
        (("tool.started", "terminal", "sleep 1", {"command": "sleep 1"}), {}),
        (("tool.progress", "terminal", None, None), {}),
        (("tool.completed", "terminal", None, None),
         {"result": '{"output": "done", "exit_code": 0}'}),
    ])
    assert rows[0]["id"] == rows[-1]["id"], "the completion opened a new row"
    assert "sleep 1" in rows[-1]["detail"], "the invocation was lost with the row"


def test_a_result_shape_json_cannot_encode_is_still_shown():
    """An exception here does not fail the turn -- hermes wraps each callback --
    it silently LOSES the row, which is harder to notice. `_invocation_text`
    already guarded its own dump; this one did not."""
    rows = _tool_events([
        (("tool.started", "odd", "go", {"a": 1}), {}),
        (("tool.completed", "odd", None, None), {"result": {"when": {1, 2}}}),
    ])
    assert len(rows) == 2
    assert rows[1]["detail"], "an unencodable result must still say something"


def test_metadata_passed_as_a_dict_beside_the_event_keeps_the_tool_name():
    """`tool("tool.started", {...})` -- event positional, metadata as a dict.
    Gating the merge on an empty event lost the name for this shape, which the
    pre-positional code handled."""
    rows = _tool_events([
        (("tool.started", {"name": "terminal", "status": "running"}), {}),
    ])
    assert rows[0]["title"] == "terminal"
    assert rows[0]["status"] == "running"
