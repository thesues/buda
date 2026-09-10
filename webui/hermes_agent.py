"""hermes as a LIBRARY — `run_agent.AIAgent` in this process, no ACP subprocess.

## Why this replaces the ACP client

The ACP adapter shipped inside hermes is a thin wrapper: for one turn it sets
five callback attributes on an `AIAgent` and calls `run_conversation`. It adds
no capability this server did not already have access to — and it REMOVES one.
`AIAgent.switch_model(model, provider, api_key, base_url, api_mode)` and the
`base_url` setter exist on the object; ACP exposes neither. `session/set_model`
is a name in the ACP schema with no handler behind it, `hermes acp` takes no
`--model`, and the model list a session advertises comes from a curated
per-provider table that never reads `custom_providers`.

That absence is what forced "one model per process": a second endpoint meant a
second `hermes acp` with its own HERMES_HOME, hence its own session store, hence
two disjoint conversation lists for what is one application. Talking to the
library directly, an endpoint is just another cache key.

What else goes away with the subprocess: the JSON-RPC framing and its pending
map, the respawn/reconnect logic, `session/cancel not honoured in 12s —
restarting hermes acp` (there is `agent.interrupt()`), the 1.3 s `session/load`
warm-up, and the history-replay protocol that made the same prompt render twice.

## What this module owns, and what it does NOT

Persistence stays hermes'. `run_conversation` writes the turn through
`agent._persist_session` into `state.db`; nothing here does. Reading is the
caller's job — `turn_context.py` starts a turn with
`messages = list(conversation_history) if conversation_history else []`, so a
turn given nothing is a turn with no history, silently. `history_for` is that
read, and it goes straight to `SessionDB` now that this process runs inside
hermes' interpreter.

The agent cache is an OPTIMISATION, never a source of truth. Evicting one costs
the ~1.3 s rebuild and nothing else: a fresh agent recomputes `_user_turn_count`
from the history it is handed. It is capped because each live agent pins a
conversation transcript in RAM.
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("buda.agent")


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


# How many built agents to keep. Each one pins a full transcript, so this is the
# dominant lever on this process' resident memory — hermes' own web UI ships 25
# with a note that fifty large sessions can hold more than a gigabyte. Sessions
# are unbounded (they live in state.db); agents are not.
AGENT_CACHE_MAX = max(1, int(os.environ.get("BUDA_AGENT_CACHE_MAX", "25")))


class _Db:
    """`SessionDB`, opened once. Import is lazy so this module can be imported
    (and its pure helpers tested) outside hermes' interpreter."""

    _lock = threading.Lock()
    _db: Any = None

    @classmethod
    def get(cls) -> Any:
        if cls._db is None:
            with cls._lock:
                if cls._db is None:
                    from hermes_state import SessionDB

                    cls._db = SessionDB(db_path=hermes_home() / "state.db")
        return cls._db


def history_for(session_id: str) -> list[dict]:
    """The conversation to hand `run_conversation`, read from hermes' store.

    Returns `[]` for an unknown session, which is exactly right for a new one.
    A read failure is NOT swallowed: continuing with `[]` would silently start
    an existing conversation over, and the model would answer as if nothing had
    been said.
    """
    if not session_id:
        return []
    return list(_Db.get().get_messages_as_conversation(session_id) or [])


def supported_kwargs(fn: Callable, candidate: dict) -> dict:
    """Drop kwargs the installed hermes does not accept.

    This process is pinned to whatever `run_agent` is installed beside it, and
    that signature moves between versions. Passing an argument it has not got
    raises `TypeError` before the turn starts — one new optional parameter
    upstream would take every conversation down. A `**kwargs` in the signature
    accepts everything, so it short-circuits.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidate)
    return {k: v for k, v in candidate.items() if k in params}


class Endpoint:
    """One model behind one OpenAI-compatible base_url.

    `key` is what the UI sends back; everything else is what the agent is built
    with. Two endpoints differing only in `model` are still two endpoints — the
    cache signature covers both fields.
    """

    __slots__ = ("key", "label", "model", "base_url", "provider", "api_key", "max_concurrent")

    def __init__(
        self,
        key: str,
        label: str,
        model: str,
        base_url: str,
        provider: str = "custom",
        api_key: str = "none",
        max_concurrent: int = 4,
    ) -> None:
        self.key = key
        self.label = label
        self.model = model
        self.base_url = base_url
        self.provider = provider
        self.api_key = api_key
        # Per endpoint, because the ceiling is the model behind it and two
        # models on two GPUs do not share one. `freetoken-l3` runs
        # `--max-running-requests 1`; whatever serves the next endpoint has its
        # own number, and one global limit can only be right for one of them.
        self.max_concurrent = max(1, int(max_concurrent))

    def as_json(self) -> dict:
        return {"key": self.key, "label": self.label, "model": self.model,
                "maxConcurrent": self.max_concurrent}

    def signature(self) -> tuple:
        """What makes a built agent reusable. `base_url` and `provider` are in
        here, which is the whole multi-endpoint mechanism: switching endpoint
        changes the signature, the cached agent misses, and a correctly-wired
        one is built. No separate 'switch model' path to keep correct."""
        return (self.model, self.base_url, self.provider, self.api_key)


def load_endpoints(raw: str | None, default_home_model: str = "") -> list[Endpoint]:
    """Parse `BUDA_ENDPOINTS` (JSON list). First entry is the default.

    Unparseable input yields the single endpoint the environment already
    describes rather than raising: a chat that starts with one model beats a
    correct refusal to start.
    """
    import json

    raw = (raw or "").strip()
    if raw:
        try:
            items = json.loads(raw)
            eps = [
                Endpoint(
                    key=str(e["key"]),
                    label=str(e.get("label") or e["key"]),
                    model=str(e.get("model") or default_home_model),
                    base_url=str(e["base_url"]),
                    provider=str(e.get("provider") or "custom"),
                    api_key=str(e.get("api_key") or "none"),
                    max_concurrent=int(e.get("maxConcurrent") or 4),
                )
                for e in items
            ]
            if not eps:
                raise ValueError("BUDA_ENDPOINTS is empty")
            keys = [e.key for e in eps]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate endpoint keys: {keys}")
            return eps
        except Exception as e:  # noqa: BLE001
            log.error("BUDA_ENDPOINTS ignored (%s); falling back to the env's single model", e)
    return [
        Endpoint(
            key="default",
            label=default_home_model or "default",
            model=default_home_model,
            base_url=os.environ.get("LLM_BASE_URL", ""),
            api_key=os.environ.get("LLM_API_KEY", "none"),
            max_concurrent=max(1, int(os.environ.get("MAX_CONCURRENT_TURNS", "4"))),
        )
    ]


# ── the agent itself ────────────────────────────────────────────────────────


def _toolsets() -> list[str]:
    return [t.strip() for t in os.environ.get("HERMES_ACP_TOOLSETS", "").split(",") if t.strip()]


def build_agent(session_id: str, ep: Endpoint) -> Any:
    """Construct one `AIAgent` wired to `ep`.

    The kwargs mirror what hermes' own ACP adapter passes, minus the parts that
    are about stdio being a JSON-RPC transport. `session_db` is handed in so the
    agent persists into the same store the sidebar reads.
    """
    from run_agent import AIAgent

    mcp_servers = []
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        mcp_servers = [
            name
            for name, c in (cfg.get("mcp_servers") or {}).items()
            if not isinstance(c, dict) or c.get("enabled", True) is not False
        ]
    except Exception:  # noqa: BLE001 -- a missing config must not block a turn
        log.debug("could not read mcp_servers from hermes config", exc_info=True)

    toolsets = list(_toolsets())
    for name in mcp_servers:
        t = f"mcp-{name}"
        if t not in toolsets:
            toolsets.append(t)

    candidate = {
        "platform": "buda",
        "model": ep.model,
        "provider": ep.provider,
        "base_url": ep.base_url,
        "api_key": ep.api_key,
        "quiet_mode": True,
        "session_id": session_id,
        "session_db": _Db.get(),
        "enabled_toolsets": toolsets or None,
        "mcp_server_names": mcp_servers or None,
    }
    return AIAgent(**supported_kwargs(AIAgent.__init__, candidate))


class AgentPool:
    """Built agents, keyed by session and pinned to a configuration.

    Two maps with different lifetimes, and conflating them is a bug:

    * `_cache` — SESSION-scoped, LRU, the agent object. Survives turns.
    * `_running` — TURN-scoped, `stream_id -> agent`, so a cancel arriving with
      a stream id can find the agent to interrupt. Registered when a turn starts
      and removed when it ends; it holds no agent of its own.
    """

    def __init__(self, max_size: int = AGENT_CACHE_MAX) -> None:
        self._cache: OrderedDict[str, tuple[Any, tuple]] = OrderedDict()
        self._running: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max = max(1, max_size)

    def acquire(self, session_id: str, ep: Endpoint) -> Any:
        """The agent for this (session, endpoint), built if the cache misses."""
        sig = ep.signature()
        with self._lock:
            hit = self._cache.get(session_id)
            if hit is not None and hit[1] == sig:
                self._cache.move_to_end(session_id)
                return hit[0]
            # A signature change means the endpoint moved under this session.
            # Drop the old agent rather than mutating it through `switch_model`:
            # the object also carries the previous turn's callbacks, tool
            # surface and reasoning config, and rebuilding is the same 1.3 s the
            # cache exists to avoid paying twice, not a new cost.
            self._cache.pop(session_id, None)
        agent = build_agent(session_id, ep)
        with self._lock:
            self._cache[session_id] = (agent, sig)
            self._cache.move_to_end(session_id)
            while len(self._cache) > self._max:
                evicted_id, _ = self._cache.popitem(last=False)
                log.debug("agent cache: evicted %s", evicted_id)
        return agent

    def evict(self, session_id: str) -> None:
        with self._lock:
            self._cache.pop(session_id, None)

    def note_running(self, stream_id: str, agent: Any) -> None:
        with self._lock:
            self._running[stream_id] = agent

    def clear_running(self, stream_id: str) -> None:
        with self._lock:
            self._running.pop(stream_id, None)

    def interrupt(self, stream_id: str, message: str | None = None) -> bool:
        """Stop the turn behind `stream_id`. True if there was one to stop.

        This is what the ACP path could not do: `session/cancel` could not reach
        a running tool, so a Stop waited out a grace period and then restarted
        the whole process. `interrupt` sets a flag the conversation loop reads.
        """
        with self._lock:
            agent = self._running.get(stream_id)
        if agent is None:
            return False
        try:
            agent.interrupt(message) if message else agent.interrupt()
            return True
        except Exception:  # noqa: BLE001 -- a failed stop must not fail the request
            log.warning("interrupt(%s) raised", stream_id, exc_info=True)
            return False

    def stats(self) -> dict:
        with self._lock:
            return {"cached": len(self._cache), "running": len(self._running), "max": self._max}


def bind_callbacks(agent: Any, sink: Any) -> None:
    """Point the agent's output at THIS turn's sink.

    Callbacks are attributes on a long-lived object, so a reused agent still
    carries the previous turn's closures — which captured the previous turn's
    stream. Left unrebound, a turn's tokens are delivered to a stream nobody is
    reading and the reader watches an empty transcript while the logs look fine.
    Call this on EVERY turn, cached or freshly built: one path, one fewer way to
    be wrong.

    Each assignment is guarded, for the same reason `supported_kwargs` exists —
    the set of callbacks differs between hermes versions, and assigning an
    attribute Python is happy to create would bind a callback nothing ever
    calls.
    """
    wiring = {
        "stream_delta_callback": lambda text: sink.delta(text),
        "reasoning_callback": lambda text: sink.delta(text, thought=True),
        "tool_progress_callback": sink.tool,
        "step_callback": sink.step,
        # hermes' local status chatter has its own pane in the TUI and no home
        # here; ACP silences it the same way.
        "thinking_callback": None,
    }
    for name, fn in wiring.items():
        if hasattr(agent, name):
            setattr(agent, name, fn)


def run_turn(
    agent: Any,
    *,
    session_id: str,
    user_message: Any,
    history: list[dict],
    system_message: str | None = None,
) -> dict:
    """One blocking turn. Call it off the request path if the caller is async.

    `persist_user_message` is what makes the prompt itself land in the store;
    without it the assistant's reply is persisted against a conversation that
    does not contain the question.
    """
    candidate = {
        "user_message": user_message,
        "system_message": system_message,
        "conversation_history": history,
        "task_id": session_id,
        "persist_user_message": user_message if isinstance(user_message, str) else None,
    }
    kwargs = supported_kwargs(agent.run_conversation, candidate)
    return agent.run_conversation(**kwargs) or {}
