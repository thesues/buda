"""Admission and lifecycle of a turn. Every case here is a client-visible one."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import hermes_agent as ha  # noqa: E402
from turns import Refused, TurnManager  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.interrupted = threading.Event()
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.thinking_callback = None

    def interrupt(self, message=None):
        self.interrupted.set()


def _mgr(monkeypatch, run=None, history=None):
    monkeypatch.setattr(ha, "build_agent", lambda session_id, ep: FakeAgent())
    pool = ha.AgentPool()
    return TurnManager(
        pool,
        history=history or (lambda sid: []),
        run=run or (lambda agent, **kw: {"final_response": "ok"}),
    )


def _ep(key="dsv4", max_concurrent=4):
    return ha.Endpoint(key=key, label=key, model="m", base_url="http://x/v1",
                       max_concurrent=max_concurrent)


def _wait_done(stream, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not stream.running:
            return True
        time.sleep(0.005)
    return False


# ── the happy path ──────────────────────────────────────────────────────────


def test_the_prompt_is_in_the_log_before_the_answer(monkeypatch):
    """A reader attaching late must still see the question above the reply."""
    def run(agent, **kw):
        agent.stream_delta_callback("hi")
        return {}

    m = _mgr(monkeypatch, run=run)
    s = m.start(session_id="s1", text="who are you", endpoint=_ep())
    assert _wait_done(s)
    kinds = [(e["kind"], e.get("text")) for e in s.after(0)]
    assert kinds[0] == ("user", "who are you")
    assert ("delta", "hi") in kinds
    assert kinds[-1][0] == "end"


def test_the_history_read_is_what_the_agent_is_given(monkeypatch):
    """hermes does NOT load history itself: a turn handed nothing starts the
    conversation over, silently, and the model answers as if nothing was said."""
    seen = {}

    def run(agent, **kw):
        seen["history"] = kw["history"]
        return {}

    m = _mgr(monkeypatch, run=run, history=lambda sid: [{"role": "user", "content": "before"}])
    s = m.start(session_id="s1", text="and now", endpoint=_ep())
    assert _wait_done(s)
    assert seen["history"] == [{"role": "user", "content": "before"}]


# ── admission ───────────────────────────────────────────────────────────────


def test_a_second_prompt_into_a_replying_conversation_is_refused_as_taken(monkeypatch):
    """`taken`, not `busy`: the text was never read by a model, so the client
    has to be able to put it back in the composer."""
    gate = threading.Event()
    m = _mgr(monkeypatch, run=lambda agent, **kw: gate.wait(3) and {})
    first = m.start(session_id="s1", text="one", endpoint=_ep())
    with pytest.raises(Refused) as e:
        m.start(session_id="s1", text="two", endpoint=_ep())
    assert e.value.reason == "taken"
    assert e.value.as_json()["streamId"] == first.stream_id
    gate.set()
    assert _wait_done(first)


def test_the_limit_is_per_endpoint_not_global(monkeypatch):
    """The ceiling is the model behind the endpoint. One global number can only
    be right for one of them."""
    gate = threading.Event()
    m = _mgr(monkeypatch, run=lambda agent, **kw: gate.wait(3) and {})
    small = _ep("dsv4", max_concurrent=1)
    other = _ep("vision", max_concurrent=1)

    a = m.start(session_id="s1", text="x", endpoint=small)
    with pytest.raises(Refused) as e:
        m.start(session_id="s2", text="y", endpoint=small)
    assert e.value.reason == "busy"
    assert e.value.as_json()["maxConcurrent"] == 1

    # A different endpoint has its own budget and is unaffected.
    b = m.start(session_id="s3", text="z", endpoint=other)
    gate.set()
    assert _wait_done(a) and _wait_done(b)


def test_a_finished_turn_frees_its_conversation(monkeypatch):
    m = _mgr(monkeypatch)
    first = m.start(session_id="s1", text="one", endpoint=_ep())
    assert _wait_done(first)
    second = m.start(session_id="s1", text="two", endpoint=_ep())
    assert _wait_done(second)


# ── failure and cancellation ────────────────────────────────────────────────


def test_a_turn_that_raises_still_tells_the_reader(monkeypatch):
    """Dying silently leaves the composer locked and the spinner running until
    EventSource gives up on its own."""
    def boom(agent, **kw):
        raise RuntimeError("model went away")

    m = _mgr(monkeypatch, run=boom)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert _wait_done(s)
    end = [e for e in s.after(0) if e["kind"] == "end"][0]
    assert "model went away" in end["error"]


def test_a_failed_turn_does_not_wedge_its_conversation(monkeypatch):
    m = _mgr(monkeypatch, run=lambda agent, **kw: (_ for _ in ()).throw(RuntimeError("x")))
    first = m.start(session_id="s1", text="one", endpoint=_ep())
    assert _wait_done(first)
    m.start(session_id="s1", text="two", endpoint=_ep())  # must not raise Refused


def test_cancel_reaches_the_running_agent_and_marks_the_stop(monkeypatch):
    """`stopped` is set before interrupting so the death it causes reads as a
    stop, not a failure — the reader pressed the button and must be told it
    worked."""
    started, release = threading.Event(), threading.Event()

    def run(agent, **kw):
        started.set()
        release.wait(3)
        return {}

    m = _mgr(monkeypatch, run=run)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert started.wait(2)
    assert m.cancel(s.stream_id) is True
    assert s.stopped is True
    release.set()
    assert _wait_done(s)


def test_cancelling_a_finished_turn_is_not_an_error(monkeypatch):
    m = _mgr(monkeypatch)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert _wait_done(s)
    assert m.cancel(s.stream_id) is False
    assert m.cancel("never-existed") is False


# ── bookkeeping ─────────────────────────────────────────────────────────────


def test_a_finished_stream_stays_readable_for_a_reader_who_was_away(monkeypatch):
    m = _mgr(monkeypatch)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert _wait_done(s)
    assert m.stream(s.stream_id) is s, "the tail must still be collectable"
    assert m.live_for("s1") is None, "but the conversation is no longer replying"


def test_finished_streams_are_eventually_dropped(monkeypatch):
    m = _mgr(monkeypatch)
    ids = []
    for i in range(5):
        s = m.start(session_id=f"s{i}", text="x", endpoint=_ep())
        assert _wait_done(s)
        ids.append(s.stream_id)
    m.forget_finished(keep=2)
    assert [i for i in ids if m.stream(i) is not None] == ids[-2:]
