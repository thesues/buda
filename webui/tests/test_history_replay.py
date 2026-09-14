"""What a tool call looks like AFTER a reload.

The reported bug was that a tool row showed nothing useful. The live path was
fixed first, and a review caught that the fix stopped at the live path: a reader
who refreshed got the raw `{"output":…,"exit_code":…}` back, with no command
above it and a failure rendered as a success. These pin the replayed shape
against the messages hermes actually persists -- captured from the store, not
invented:

  call:   {"id": "call_terminal_0_…", "function": {"name": "terminal",
           "arguments": '{"command":"echo HELLO-OK","timeout":15}'}}
  result: {"role": "tool", "tool_call_id": "call_terminal_0_…",
           "tool_name": "terminal",
           "content": '{"output": "HELLO-OK", "exit_code": 0, "error": null}'}

`arguments` is a JSON STRING and `is_error` is NOT persisted -- both shape what
replay can and cannot say.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_session_api as hs  # noqa: E402


def _msgs(command, output, exit_code, error=None, extra_args=None):
    import json

    args = {"command": command}
    if extra_args:
        args.update(extra_args)
    return [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "call_id": "call_1", "type": "function",
            "function": {"name": "terminal", "arguments": json.dumps(args)},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "tool_name": "terminal",
         "content": json.dumps({"output": output, "exit_code": exit_code,
                                "error": error})},
    ]


def _replay(monkeypatch, msgs):
    monkeypatch.setattr(hs, "_db", lambda: type("D", (), {
        "get_messages": staticmethod(lambda sid: msgs)})())
    return [e for e in hs.history("s", 0) if e.get("kind") == "tool"]


def test_a_reloaded_row_shows_the_command_and_the_output(monkeypatch):
    """Ablation: drop the `arguments` parse and the command disappears; drop
    `_detail_for` and the row is raw JSON again."""
    rows = _replay(monkeypatch, _msgs("echo HELLO-OK", "HELLO-OK", 0))
    assert "echo HELLO-OK" in rows[0]["detail"], "the call row lost its command"
    assert "echo HELLO-OK" in rows[1]["detail"], "the result row lost its command"
    assert "HELLO-OK" in rows[1]["detail"]
    assert rows[0]["id"] == rows[1]["id"] == "call_1"


def test_a_second_argument_does_not_turn_the_command_into_json(monkeypatch):
    """`terminal` persists a `timeout` beside the command, so the arguments dict
    has two keys. A rule that only unwrapped SINGLE-valued dicts printed the
    whole JSON — the live row said `echo …` and the reloaded one said
    `{"command": "echo …", "timeout": 15}` for the same call.

    Falls back to the dict rendering without hermes installed, which is what
    this test environment has; the assertion is that the COMMAND is legible
    either way."""
    rows = _replay(monkeypatch,
                   _msgs("echo HELLO-OK", "HELLO-OK", 0, extra_args={"timeout": 15}))
    assert "echo HELLO-OK" in rows[1]["detail"]


def test_a_nonzero_exit_survives_the_reload(monkeypatch):
    rows = _replay(monkeypatch, _msgs("ls /nope", "No such file or directory", 2))
    assert "exit 2" in rows[1]["detail"]
    assert "No such file or directory" in rows[1]["detail"]


def test_an_explicit_error_is_the_only_failure_replay_can_claim(monkeypatch):
    """`is_error` is not persisted, so hermes' verdict cannot be recovered. An
    explicit `error` in the payload is a failure by any reading; a non-zero exit
    alone stays `completed` with the code visible, because `grep` answers 1 for
    "no match" and inventing a failure is the bigger lie."""
    plain = _replay(monkeypatch, _msgs("ls /nope", "nope", 2))
    assert plain[1]["status"] == "completed"
    erred = _replay(monkeypatch, _msgs("boom", "", 1, error="tool exploded"))
    assert erred[1]["status"] == "failed"
    assert "tool exploded" in erred[1]["detail"]


def test_the_true_length_is_a_number_not_the_text(monkeypatch):
    """The client computes "还有 N 字" from `detailFull`. Sending the string made
    that arithmetic NaN on every replayed result."""
    rows = _replay(monkeypatch, _msgs("cat big", "x" * 9000, 0))
    assert isinstance(rows[1]["detailFull"], int)
    assert len(rows[1]["detail"]) <= hs.DETAIL_MAX
    assert rows[1]["detailFull"] > len(rows[1]["detail"])
