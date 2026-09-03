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


@pytest.mark.asyncio
async def test_an_acp_error_reply_raises_out_of_request():
    """The real choke point: `_request` must RAISE on a JSON-RPC error reply.

    Found live: hermes rejected `session/new`; the read loop hands an error back
    as an ordinary result carrying `_error` (it cannot raise into a future's
    awaiter), `_request` RETURNED it, `session_id` silently became None, the
    next prompt sent `sessionId: null`, and the turn ended with no content and
    `error: None` -- indistinguishable on screen from the agent choosing to say
    nothing.

    This drives the production `HermesACP._request` with the write stubbed, so
    it fails if the check is removed. An earlier version of this test injected a
    prompt that raised directly, which never reached `_request` and passed with
    the fix disabled -- it proved nothing.
    """
    acp = srv.HermesACP()
    written: list[dict] = []

    async def fake_write(obj):
        written.append(obj)
        # Answer the way the read loop does for a JSON-RPC error.
        acp._pending[obj["id"]].set_result({"_error": {"code": -32602, "message": "Invalid params"}})

    acp._write = fake_write
    with pytest.raises(RuntimeError, match="Invalid params"):
        await acp._request("session/new", {"cwd": "/tmp", "mcpServers": []})
    assert written and written[0]["method"] == "session/new"


@pytest.mark.asyncio
async def test_an_acp_ok_reply_is_returned_unchanged():
    acp = srv.HermesACP()

    async def fake_write(obj):
        acp._pending[obj["id"]].set_result({"sessionId": "s-1"})

    acp._write = fake_write
    assert await acp._request("session/new", {}) == {"sessionId": "s-1"}


# --- Regressions, one per failure actually observed against a live stack. -----
# Named after what they pin, in the upstream hermes-webui convention: a test
# whose name says which bug it is stops anyone from "simplifying" it away.

@pytest.mark.asyncio
async def test_an_oversized_acp_frame_is_actually_read():
    """asyncio caps a line at 64 KiB and RAISES past it -- it does not truncate.

    An MCP tool result carrying document text blew through that: the read loop
    died and the turn ended with no output and no error. This drives a REAL
    subprocess emitting a 1 MiB JSON-RPC line through the production spawn
    settings, because a test that greps the source for `limit=` proves only that
    the string is present.
    """
    # Build the payload IN the child: a 1 MiB argv blows past ARG_MAX.
    prog = (
        "import json,sys;"
        "sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':1,"
        "'result':{'t':'x'*(1024*1024)}})+chr(10));"
        "sys.stdout.flush()"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", prog,
        stdout=asyncio.subprocess.PIPE, limit=srv.ACP_LINE_LIMIT,
    )
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    await proc.wait()
    msg = json.loads(line)
    assert len(msg["result"]["t"]) == 1024 * 1024, "the whole frame must survive"


@pytest.mark.asyncio
async def test_the_default_limit_would_have_lost_that_frame():
    """The ablation: at asyncio's default the same line raises rather than truncating.

    This is what makes the fix load-bearing rather than decorative.
    """
    # Build the payload IN the child: a 1 MiB argv blows past ARG_MAX.
    prog = (
        "import json,sys;"
        "sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':1,"
        "'result':{'t':'x'*(1024*1024)}})+chr(10));"
        "sys.stdout.flush()"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", prog, stdout=asyncio.subprocess.PIPE,   # default 64 KiB
    )
    with pytest.raises((ValueError, asyncio.LimitOverrunError)):
        await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    proc.kill()
    await proc.wait()


@pytest.mark.asyncio
async def test_the_mcp_server_is_declared_with_its_union_discriminator(aiohttp_client, app):
    """`type` and `headers` are both required, and omitting either is silent.

    `session/new` takes `HttpMcpServer`, which subclasses `McpServerHttp` ONLY to
    add `type: Literal["http"]`; `headers` has no default. Sending the parent's
    shape is rejected outright (-32602), and sending nothing at all yields an
    agent with no tools while `hermes mcp list` still reports the server enabled.
    """
    old = (srv.MEMORY_MCP_URL, srv.MEMORY_MCP_NAME)
    srv.MEMORY_MCP_URL, srv.MEMORY_MCP_NAME = "http://example:5100/mcp", "corpus"
    try:
        entry = srv._acp_mcp_servers()[0]
    finally:
        srv.MEMORY_MCP_URL, srv.MEMORY_MCP_NAME = old
    assert entry["type"] == "http", "the union discriminator"
    assert entry["headers"] == [], "required, no default"
    assert entry["name"] == "corpus" and entry["url"].endswith("/mcp")


def test_no_mcp_url_declares_no_servers():
    old = srv.MEMORY_MCP_URL
    srv.MEMORY_MCP_URL = ""
    try:
        assert srv._acp_mcp_servers() == []
    finally:
        srv.MEMORY_MCP_URL = old
