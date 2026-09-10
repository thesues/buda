"""The parts of the in-process agent layer that hold without hermes installed.

Everything here is the glue that decides WHICH agent runs a turn and WHAT it is
called with. The agent itself is hermes'; these are the rules this server adds
around it, and each one below is a failure that actually costs a conversation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_agent as ha  # noqa: E402


# ── supported_kwargs ────────────────────────────────────────────────────────


def test_an_argument_the_installed_hermes_lacks_is_dropped():
    """A version skew must cost a feature, not the turn.

    `run_conversation` gains optional parameters between hermes releases. Passing
    one the installed copy has not got raises TypeError before the model is ever
    called, so every conversation dies on an upgrade this server did not make.
    """

    def older(user_message, conversation_history=None):
        return None

    got = ha.supported_kwargs(
        older,
        {"user_message": "hi", "conversation_history": [], "persist_user_timestamp": 1.0},
    )
    assert got == {"user_message": "hi", "conversation_history": []}


def test_a_var_keyword_signature_accepts_everything():
    def newer(user_message, **kw):
        return None

    got = ha.supported_kwargs(newer, {"user_message": "hi", "anything": 1})
    assert got == {"user_message": "hi", "anything": 1}


def test_an_unreadable_signature_passes_nothing_rather_than_guessing():
    """Builtins have no introspectable signature. Sending a guess would be the
    TypeError this function exists to prevent."""
    assert ha.supported_kwargs(len, {"user_message": "hi"}) == {}


# ── endpoints ───────────────────────────────────────────────────────────────


def _ep(key="a", model="m", base_url="http://x/v1", **kw):
    return ha.Endpoint(key=key, label=key, model=model, base_url=base_url, **kw)


def test_the_endpoint_is_in_the_signature_so_switching_rebuilds():
    """The whole multi-endpoint mechanism. If `base_url` were not part of what
    makes a cached agent reusable, a session that switched endpoint would keep
    talking to the old one — with the UI showing the new name."""
    assert _ep(base_url="http://a/v1").signature() != _ep(base_url="http://b/v1").signature()
    assert _ep(model="m1").signature() != _ep(model="m2").signature()
    assert _ep().signature() == _ep().signature()


def test_a_broken_endpoint_list_still_yields_one_working_endpoint(monkeypatch):
    """A chat that starts on one model beats a correct refusal to start."""
    monkeypatch.setenv("LLM_BASE_URL", "http://fallback/v1")
    for raw in ["{not json", "[]", '[{"key":"a","base_url":"u"},{"key":"a","base_url":"v"}]']:
        eps = ha.load_endpoints(raw, default_home_model="dsv4")
        assert len(eps) == 1 and eps[0].key == "default"
        assert eps[0].base_url == "http://fallback/v1"


def test_endpoints_parse_in_order_with_the_first_as_default():
    eps = ha.load_endpoints(
        '[{"key":"dsv4","base_url":"http://a/v1","model":"m1","maxConcurrent":4},'
        ' {"key":"vision","base_url":"http://b/v1","model":"m2","maxConcurrent":1}]'
    )
    assert [e.key for e in eps] == ["dsv4", "vision"]
    assert eps[1].max_concurrent == 1


# ── the pool ────────────────────────────────────────────────────────────────


class FakeAgent:
    def __init__(self, session_id, ep):
        self.session_id = session_id
        self.ep = ep
        self.interrupted = 0
        self.stream_delta_callback = None
        self.tool_progress_callback = None
        self.reasoning_callback = None
        self.step_callback = None
        self.thinking_callback = "left over from the previous turn"

    def interrupt(self, message=None):
        self.interrupted += 1


def _pool(monkeypatch, max_size=25):
    built = []

    def fake_build(session_id, ep):
        a = FakeAgent(session_id, ep)
        built.append(a)
        return a

    monkeypatch.setattr(ha, "build_agent", fake_build)
    return ha.AgentPool(max_size=max_size), built


def test_the_same_session_and_endpoint_reuses_one_agent(monkeypatch):
    pool, built = _pool(monkeypatch)
    ep = _ep()
    assert pool.acquire("s1", ep) is pool.acquire("s1", ep)
    assert len(built) == 1


def test_switching_endpoint_rebuilds_rather_than_mutating(monkeypatch):
    """Rebuilding is the deliberate choice: the cached object also carries the
    previous turn's callbacks and tool surface, and `switch_model` would leave
    all of that pointed at the old endpoint."""
    pool, built = _pool(monkeypatch)
    a = pool.acquire("s1", _ep(base_url="http://a/v1"))
    b = pool.acquire("s1", _ep(base_url="http://b/v1"))
    assert a is not b and len(built) == 2
    assert b.ep.base_url == "http://b/v1"


def test_the_cache_is_bounded_and_evicts_least_recently_used(monkeypatch):
    """Sessions are unbounded — they live in state.db. Agents are not: each one
    pins a transcript, so an uncapped cache turns a long-lived server into a
    memory leak that only shows up on the busiest day."""
    pool, _ = _pool(monkeypatch, max_size=2)
    first = pool.acquire("s1", _ep())
    pool.acquire("s2", _ep())
    pool.acquire("s1", _ep())  # s1 is now the most recent
    pool.acquire("s3", _ep())  # evicts s2
    assert pool.stats()["cached"] == 2
    assert pool.acquire("s1", _ep()) is first, "the recently used one must survive"


def test_interrupt_reaches_the_agent_running_that_stream(monkeypatch):
    pool, _ = _pool(monkeypatch)
    agent = pool.acquire("s1", _ep())
    pool.note_running("stream-1", agent)
    assert pool.interrupt("stream-1") is True and agent.interrupted == 1
    pool.clear_running("stream-1")
    assert pool.interrupt("stream-1") is False, "a finished turn has nothing to stop"


def test_an_agent_that_cannot_be_interrupted_does_not_fail_the_request(monkeypatch):
    pool, _ = _pool(monkeypatch)

    class Stubborn(FakeAgent):
        def interrupt(self, message=None):
            raise RuntimeError("no")

    agent = Stubborn("s1", _ep())
    pool.note_running("s", agent)
    assert pool.interrupt("s") is False


# ── callbacks ───────────────────────────────────────────────────────────────


class Sink:
    def __init__(self):
        self.deltas = []

    def delta(self, text, thought=False):
        self.deltas.append((text, thought))

    def tool(self, *a, **kw):
        pass

    def step(self, *a, **kw):
        pass


def test_rebinding_replaces_the_previous_turns_closures(monkeypatch):
    """The reuse hazard. A cached agent still holds the callbacks of the turn
    before, which captured a stream that is closed — so the reader sees an empty
    transcript while the server logs a healthy turn."""
    pool, _ = _pool(monkeypatch)
    agent = pool.acquire("s1", _ep())

    old = Sink()
    ha.bind_callbacks(agent, old)
    new = Sink()
    ha.bind_callbacks(agent, new)

    agent.stream_delta_callback("hello")
    agent.reasoning_callback("thinking")
    assert new.deltas == [("hello", False), ("thinking", True)]
    assert old.deltas == [], "the previous turn's sink must receive nothing"


def test_binding_skips_callbacks_this_hermes_does_not_have(monkeypatch):
    """Assigning an unknown attribute succeeds in Python and binds a callback
    nothing will ever call — a silent no-op instead of a visible mismatch."""
    pool, _ = _pool(monkeypatch)
    agent = pool.acquire("s1", _ep())
    del agent.step_callback
    ha.bind_callbacks(agent, Sink())
    assert not hasattr(agent, "step_callback")
    assert agent.thinking_callback is None, "hermes' own status chatter stays off"
