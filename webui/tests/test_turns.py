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


# ── approvals ───────────────────────────────────────────────────────────────


def _fake_approval(monkeypatch):
    """Stand in for `tools.approval`, which lives in hermes' venv.

    Models the GATEWAY surface — a module-global queue keyed by session, a
    notify callback, and a resolve that unblocks from any thread — because that
    is the surface this code uses. The thread-local
    `terminal_tool.set_approval_callback` is NOT modelled: it is what this
    replaced, and modelling it would let the old bug pass.
    """
    import sys
    import threading as _t
    import types

    ap = types.ModuleType("tools.approval")
    ap.queues = {}          # session_key -> list of {"data":…, "event":…, "choice":…}
    ap.notify = {}
    ap._key = {"v": "default"}

    ap.bound = []           # every key this turn bound, in order

    def set_current_session_key(k):
        prev = ap._key["v"]
        ap._key["v"] = k
        ap.bound.append(k)
        return prev

    def reset_current_session_key(tok):
        ap._key["v"] = tok

    def register_gateway_notify(k, cb):
        ap.notify[k] = cb

    def resolve_gateway_approval(k, choice, resolve_all=False):
        q = ap.queues.get(k) or []
        targets = list(q) if resolve_all else q[:1]
        for t in targets:
            t["choice"] = choice
            t["event"].set()
            q.remove(t)
        return len(targets)

    def ask(k, data):
        """What hermes does on the agent thread: queue, notify, block."""
        entry = {"data": data, "event": _t.Event(), "choice": None}
        ap.queues.setdefault(k, []).append(entry)
        cb = ap.notify.get(k)
        if cb:
            cb(data)
        entry["event"].wait(timeout=10)
        return entry["choice"] or "deny"

    ap.set_current_session_key = set_current_session_key
    ap.reset_current_session_key = reset_current_session_key
    ap.register_gateway_notify = register_gateway_notify
    ap.resolve_gateway_approval = resolve_gateway_approval
    ap.ask = ask

    tools = sys.modules.get("tools") or types.ModuleType("tools")
    tools.approval = ap
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "tools.approval", ap)
    return ap


def test_the_approval_hook_is_the_gateway_one_not_the_thread_local(monkeypatch):
    """`terminal_tool.set_approval_callback` stores into a `threading.local`.

    hermes dispatches tools on its own threads, so a callback registered on the
    turn's thread is invisible where the question is actually asked; hermes
    then falls through to an `input()` that nothing will ever answer. Measured
    on the cluster: `terminal / running`, no events for minutes, the endpoint's
    only slot held, and `interrupt()` unable to help because it sets a flag the
    conversation loop reads and the loop was never reached.

    So the turn must bind the session key (a contextvar, inherited by threads)
    and register a gateway notify — not the thread-local callback.

    Ablation: go back to `terminal_tool.set_approval_callback` and this fails.
    """
    ap = _fake_approval(monkeypatch)
    m = _mgr(monkeypatch)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert _wait_done(s)
    # Checked as a HISTORY, not a current value: teardown resets the contextvar
    # when the turn ends, which is correct and would make a live read useless.
    assert "s1" in ap.bound, "the session key was never bound for this turn"
    assert "s1" in ap.notify, "no gateway notify was registered"


def test_a_permission_request_reaches_the_reader_and_the_turn_waits(monkeypatch):
    import threading

    ap = _fake_approval(monkeypatch)
    m = _mgr(monkeypatch)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert _wait_done(s)

    out = {}
    t = threading.Thread(target=lambda: out.setdefault(
        "r", ap.ask("s1", {"command": "rm -rf /tmp/x", "description": "delete a path"})))
    t.start()

    ev = None
    for _ in range(300):
        ev = next((e for e in s.after(0) if e["kind"] == "approval"), None)
        if ev:
            break
        time.sleep(0.01)
    assert ev, "no approval card was emitted"
    assert ev["title"] == "delete a path"
    assert [o["optionId"] for o in ev["options"]] == ["once", "session", "always", "deny"]
    assert t.is_alive(), "the turn carried on without an answer"

    ap.resolve_gateway_approval("s1", "session")
    t.join(timeout=5)
    assert out["r"] == "session"


def test_stop_releases_a_turn_parked_on_an_approval(monkeypatch):
    """Interrupt alone cannot reach it.

    `interrupt` sets a flag the conversation loop reads; a turn waiting for
    permission is parked on the approval queue and never reaches that loop.
    Pressing Stop with a card on screen is exactly when a reader most wants it
    to stop, so cancel denies what is pending first.

    The turn is held RUNNING here on purpose — cancel returns early for a
    finished stream, and a finished turn is not the situation this is about.

    Ablation: drop the `resolve_gateway_approval` call from `cancel` and the
    waiter never returns.
    """
    import threading

    ap = _fake_approval(monkeypatch)
    parked = threading.Event()
    released = {}

    def blocking_run(agent, **kw):
        # What hermes does: ask, and do not come back until answered.
        parked.set()
        released["choice"] = ap.ask("s1", {"command": "rm -rf /", "description": "dangerous"})
        return {"final_response": "stopped"}

    m = _mgr(monkeypatch, run=blocking_run)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert parked.wait(timeout=5), "the turn never reached the approval"
    for _ in range(300):
        if ap.queues.get("s1"):
            break
        time.sleep(0.01)
    assert s.running, "the turn should still be running while it waits"

    m.cancel(s.stream_id)
    assert _wait_done(s), "the turn did not finish after stop"
    assert released.get("choice") == "deny", "stop did not deny the parked approval"




def test_the_session_key_is_visible_from_a_thread_hermes_starts(monkeypatch):
    """A contextvar does not cross a `threading.Thread`.

    A new thread begins with an EMPTY context, not a copy of its parent's, and
    hermes dispatches tools on threads of its own. So the key bound with
    `set_current_session_key` was invisible where the tool actually asked:
    hermes queued the approval under `"default"`, looked for a notify callback
    under `"default"`, found none, and the turn waited forever with no card on
    screen. `get_current_session_key()` falls back to HERMES_SESSION_KEY, which
    is process-wide and therefore does cross.

    Ablation: drop the `os.environ["HERMES_SESSION_KEY"]` line and this fails.
    """
    import os
    import threading

    _fake_approval(monkeypatch)
    m = _mgr(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    s = m.start(session_id="s-thread", text="x", endpoint=_ep())
    assert _wait_done(s)

    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("key", os.environ.get("HERMES_SESSION_KEY")))
    t.start(); t.join(timeout=5)
    assert seen["key"] == "s-thread", (
        "a thread hermes starts cannot see the session key, so it will ask "
        "under 'default' and no card will ever be shown"
    )


def test_the_gateway_context_switch_is_on(monkeypatch):
    """`_is_gateway_approval_context()` reads HERMES_GATEWAY_SESSION. Without
    it hermes takes the CLI branch — the thread-local callback and then the
    `input()` that never returns."""
    import os

    _fake_approval(monkeypatch)
    m = _mgr(monkeypatch)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    s = m.start(session_id="s1", text="x", endpoint=_ep())
    assert _wait_done(s)
    assert os.environ.get("HERMES_GATEWAY_SESSION") == "1"
