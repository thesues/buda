"""The claim this whole design rests on, tested: a turn does not belong to a connection.

Everything here runs against the real aiohttp app with a FAKE acp, so the
transport, the routing and the SSE framing are the production ones and only the
agent is substituted. A test that stubbed the app would pin nothing.

Run:  python -m pytest tests/ -q        (needs aiohttp + pytest-aiohttp)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server as srv  # noqa: E402


class FakeACP:
    """Stands in for `hermes acp`: a turn we can drive from the test.

    `prompt` blocks until `release` is set, which is what lets a test disconnect
    mid-turn and prove the turn kept going without it.
    """

    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.alive = True
        self.release = asyncio.Event()
        self.emitted: list[str] = []
        self.on_update = None
        self.cancelled = False

    async def prompt(self, text, on_update, on_permission):
        for i in range(3):
            await on_update({"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text", "text": f"part{i} "}})
            self.emitted.append(f"part{i}")
        await self.release.wait()
        await on_update({"sessionUpdate": "agent_message_chunk",
                         "content": {"type": "text", "text": "tail"}})
        return {"stopReason": "end_turn"}

    async def cancel(self):
        self.cancelled = True

    async def list_sessions(self):
        return []

    async def new_session(self):
        self.session_id = None


@pytest.fixture
def app():
    a = srv.build_app()
    # Replace the real ACP; drop the lifecycle hooks that would spawn hermes and
    # rewrite a config file on a developer's machine.
    a["state"].acp = FakeACP()
    a.on_startup.clear()
    a.on_cleanup.clear()
    return a


async def read_events(resp, want: int, timeout: float = 5.0) -> list[dict]:
    """Collect `want` SSE data events, ignoring keepalive comments."""
    out: list[dict] = []
    async def pump():
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            out.append(json.loads(line[6:]))
            if len(out) >= want:
                return
    await asyncio.wait_for(pump(), timeout)
    return out


@pytest.mark.asyncio
async def test_a_disconnect_does_not_stop_the_turn(aiohttp_client, app):
    client = await aiohttp_client(app)
    started = await (await client.post("/api/chat/start", json={"text": "hi"})).json()
    sid = started["streamId"]
    assert sid and not started["attached"]

    # Read the first few events, then hang up mid-turn.
    resp = await client.get(f"/api/chat/stream?stream_id={sid}&after_seq=0")
    first = await read_events(resp, 2)          # "user", then the first delta
    resp.close()
    last_seq = first[-1]["seq"]

    # With nobody listening, let the turn finish.
    app["state"].acp.release.set()
    await asyncio.wait_for(app["state"].turn_task, timeout=5)

    # The turn ran to completion while disconnected.
    stream = app["state"].streams[sid]
    assert not stream.running
    assert stream.error is None
    assert "tail" in "".join(e.get("text", "") for _, e in stream.events)


@pytest.mark.asyncio
async def test_a_reconnect_replays_only_the_delta(aiohttp_client, app):
    client = await aiohttp_client(app)
    sid = (await (await client.post("/api/chat/start", json={"text": "hi"})).json())["streamId"]

    resp = await client.get(f"/api/chat/stream?stream_id={sid}&after_seq=0")
    seen = await read_events(resp, 2)
    resp.close()
    resume_from = seen[-1]["seq"]

    app["state"].acp.release.set()
    await asyncio.wait_for(app["state"].turn_task, timeout=5)

    # Reattach from where we left off. Nothing already delivered may repeat —
    # a replayed transcript is what a naive "resend everything" reconnect gives,
    # and it duplicates the answer on screen.
    resp2 = await client.get(f"/api/chat/stream?stream_id={sid}&after_seq={resume_from}")
    rest = []
    async for raw in resp2.content:
        line = raw.decode().strip()
        if line.startswith("data: "):
            rest.append(json.loads(line[6:]))
    resp2.close()

    assert rest, "the reconnect delivered nothing"
    assert all(e["seq"] > resume_from for e in rest), "the reconnect repeated delivered events"
    assert rest[-1]["kind"] == "end", "a finished turn must close the stream with `end`"


@pytest.mark.asyncio
async def test_status_tells_a_reloaded_page_what_to_do(aiohttp_client, app):
    client = await aiohttp_client(app)
    sid = (await (await client.post("/api/chat/start", json={"text": "hi"})).json())["streamId"]

    st = await (await client.get(f"/api/chat/status?stream_id={sid}")).json()
    assert st["known"] and st["running"]

    app["state"].acp.release.set()
    await asyncio.wait_for(app["state"].turn_task, timeout=5)

    st = await (await client.get(f"/api/chat/status?stream_id={sid}")).json()
    assert st["known"] and not st["running"]

    # An id from a previous process must answer honestly, not invent a turn.
    st = await (await client.get("/api/chat/status?stream_id=gone")).json()
    assert st == {"known": False}


@pytest.mark.asyncio
async def test_a_second_send_attaches_instead_of_starting_a_rival_turn(aiohttp_client, app):
    client = await aiohttp_client(app)
    first = await (await client.post("/api/chat/start", json={"text": "one"})).json()
    second = await (await client.post("/api/chat/start", json={"text": "two"})).json()

    # A double-click, or a reload that re-submits, must join the running turn.
    # Starting a rival one would interleave two agents into one transcript.
    assert second["attached"] is True
    assert second["streamId"] == first["streamId"]

    app["state"].acp.release.set()
    await asyncio.wait_for(app["state"].turn_task, timeout=5)


@pytest.mark.asyncio
async def test_an_approval_is_answerable_out_of_band(aiohttp_client, app):
    """The push and the answer travel on different requests, so the answer must
    reach a callback that is parked on a future — and a reloaded page must be
    able to FIND the prompt it never saw."""
    client = await aiohttp_client(app)

    async def prompt_with_permission(text, on_update, on_permission):
        choice = await on_permission({"toolCall": {"title": "rm -rf /"},
                                      "options": [{"optionId": "allow", "name": "允许"}]})
        await on_update({"sessionUpdate": "agent_message_chunk",
                         "content": {"type": "text", "text": f"chose:{choice}"}})
        return {"stopReason": "end_turn"}

    app["state"].acp.prompt = prompt_with_permission
    sid = (await (await client.post("/api/chat/start", json={"text": "go"})).json())["streamId"]

    # Poll the way the browser does — this is the fallback that survives a lost push.
    for _ in range(50):
        pending = (await (await client.get("/api/approval/pending")).json())["pending"]
        if pending:
            break
        await asyncio.sleep(0.02)
    assert pending, "the approval never became pollable"
    assert pending[0]["title"] == "rm -rf /"

    await client.post("/api/approval/answer",
                      json={"id": pending[0]["id"], "optionId": "allow"})
    await asyncio.wait_for(app["state"].turn_task, timeout=5)

    text = "".join(e.get("text", "") for _, e in app["state"].streams[sid].events)
    assert "chose:allow" in text
    # Answered approvals must not linger — a stale prompt would keep re-appearing
    # on every poll.
    assert (await (await client.get("/api/approval/pending")).json())["pending"] == []


def test_the_backlog_cap_reports_a_gap_instead_of_lying():
    """An overflowed window must be reported. Silently handing a reader a
    transcript with a hole in it is worse than telling it the hole exists."""
    s = srv.TurnStream("x", "sess")
    for i in range(srv.BACKLOG_EVENTS + 10):
        s.emit("delta", text=str(i))
    assert s.dropped == 10
    assert s.gap_before(0) is True                 # asking from the start: gap
    assert s.gap_before(s.seq - 1) is False        # asking from the tail: no gap
    assert len(s.after(s.seq - 5)) == 5
