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


@pytest.mark.asyncio
async def test_a_stream_lost_with_the_process_is_reported_as_unknown(aiohttp_client, app):
    """A restart drops every live stream AND the turn behind it.

    The browser's EventSource retries forever regardless, so a UI that says
    "reconnecting" on any error keeps saying it long after there is nothing to
    reconnect to. `status` must answer honestly for an id it has never seen, so
    the client can tell "still coming" from "gone with the process".
    """
    client = await aiohttp_client(app)
    st = await (await client.get("/api/chat/status?stream_id=from-a-dead-process")).json()
    assert st == {"known": False}

    # And a stream request for it must 404 rather than hanging or 200-ing empty:
    # an empty 200 is indistinguishable from a turn that has said nothing yet.
    resp = await client.get("/api/chat/stream?stream_id=from-a-dead-process&after_seq=0")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_a_cancel_that_is_ignored_escalates_to_a_restart(aiohttp_client, app):
    """`session/cancel` cannot interrupt a running tool, so Stop must escalate.

    The agent notices a cancel only between steps. While a tool runs the turn
    keeps going, and a Stop that only sends the notification does nothing
    visible for as long as the tool takes -- which is how a working button gets
    reported as broken. This fakes an ACP that ignores cancel entirely and
    asserts the endpoint restarts it and closes the stream, because a turn
    cancelled mid-flight can no longer send its own `end`.
    """
    srv.CANCEL_GRACE_SEC = 0.2          # the real 12 s is a human timescale, not a test one
    client = await aiohttp_client(app)
    acp = app["state"].acp
    acp.restarted = False

    async def never_yields(text, on_update, on_permission):
        await asyncio.sleep(30)         # a tool that will not be interrupted
        return {"stopReason": "end_turn"}

    async def fake_restart():
        acp.restarted = True

    async def fake_load(sid, on_update):
        return None

    acp.prompt = never_yields
    acp.restart = fake_restart
    acp.load_session = fake_load

    sid = (await (await client.post("/api/chat/start", json={"text": "go"})).json())["streamId"]
    body = await (await client.post("/api/chat/cancel")).json()

    assert body["how"] == "restart", "an ignored cancel must not report success and stop there"
    assert acp.restarted, "the process must actually be restarted"

    stream = app["state"].streams[sid]
    assert not stream.running, "the stream must be closed, or the client waits for an `end` forever"
    assert [e for _s, e in stream.events if e["kind"] == "end"], "an `end` event must be emitted"


@pytest.mark.asyncio
async def test_a_cancel_that_is_honoured_does_not_restart(aiohttp_client, app):
    """The graceful path is the normal one -- a restart throws away this turn's output."""
    srv.CANCEL_GRACE_SEC = 5
    client = await aiohttp_client(app)
    acp = app["state"].acp
    acp.restarted = False

    async def fake_restart():
        acp.restarted = True
    acp.restart = fake_restart

    await client.post("/api/chat/start", json={"text": "go"})
    # Release SHORTLY AFTER the cancel goes out, not before: releasing first
    # finishes the turn and the endpoint correctly answers "already idle",
    # which tests nothing about the graceful path.
    async def wind_down():
        await asyncio.sleep(0.05)
        acp.release.set()
    asyncio.get_running_loop().create_task(wind_down())
    body = await (await client.post("/api/chat/cancel")).json()

    assert body["how"] == "graceful"
    assert not acp.restarted, "a cancel that was honoured must not cost the process"


@pytest.mark.asyncio
async def test_cancelling_when_idle_is_not_an_error(aiohttp_client, app):
    client = await aiohttp_client(app)
    body = await (await client.post("/api/chat/cancel")).json()
    assert body["ok"] and body.get("already") == "idle"


@pytest.mark.asyncio
async def test_a_stop_is_not_reported_as_an_error(aiohttp_client, app):
    """The restart Stop escalates to kills the in-flight request, which raises
    "hermes acp exited". Reporting that as the turn's error tells the reader
    their own Stop broke something."""
    srv.CANCEL_GRACE_SEC = 0.2
    client = await aiohttp_client(app)
    acp = app["state"].acp

    async def dies_on_restart(text, on_update, on_permission):
        await asyncio.sleep(30)
        raise RuntimeError("hermes acp exited")

    async def fake_restart():
        # What a real restart does to the in-flight request.
        app["state"].turn_task.cancel()
    acp.prompt = dies_on_restart
    acp.restart = fake_restart
    acp.load_session = lambda sid, cb: asyncio.sleep(0)

    sid = (await (await client.post("/api/chat/start", json={"text": "go"})).json())["streamId"]
    await client.post("/api/chat/cancel")

    stream = app["state"].streams[sid]
    assert not stream.running
    assert stream.error is None, f"a deliberate stop must not surface as an error: {stream.error}"


@pytest.mark.asyncio
async def test_the_page_title_comes_from_the_file_and_env_overrides_it(aiohttp_client, app):
    """One deployment should be able to say what it is without a rebuild, and a
    title with markup in it must not become markup."""
    client = await aiohttp_client(app)

    srv.PAGE_TITLE = ""
    html = await (await client.get("/")).text()
    assert "<title>buda · 佛典检索</title>" in html, "unset env serves the file unchanged"

    srv.PAGE_TITLE = "藏经阁 <script>"
    try:
        html = await (await client.get("/")).text()
        assert "<title>藏经阁 &lt;script&gt;</title>" in html, "the title must be escaped"
        assert "<script>" not in html.split("</title>")[0], "no raw markup from the env"
    finally:
        srv.PAGE_TITLE = ""


def test_the_client_renders_a_loaded_transcript_in_full():
    """The browser-side render ordering, checked in node.

    A loaded session delivers every token in one synchronous pass, so the only
    render is the deferred rAF -- and finalizeSeg() used to clear the segment
    that render needs. The answer vanished. Live turns lost their closing
    sentences the same way, which is subtler and was never noticed.

    Skipped rather than failed without node: this pins client behaviour, and a
    missing runtime is not a broken client.
    """
    import shutil, subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = Path(__file__).parent / "js" / "render_finalize.mjs"
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-400:]


def test_a_tool_update_does_not_rename_the_tool():
    """ACP's `kind` is a CATEGORY (other/read/search), not a name.

    Using it as a title fallback let a tool_call_update -- which carries no
    title -- rename `mcp_buda_corpus_search_docs` to "other" the moment it
    completed, so the transcript credited the work to a tool that does not
    exist.
    """
    call = srv._to_event({"sessionUpdate": "tool_call", "toolCallId": "t1",
                          "title": "mcp_buda_corpus_search_docs", "status": "pending"})
    assert call[1]["title"] == "mcp_buda_corpus_search_docs"

    upd = srv._to_event({"sessionUpdate": "tool_call_update", "toolCallId": "t1",
                         "kind": "other", "status": "completed"})
    assert upd[1]["title"] == "", "an update with no title must not assert one"


def test_the_end_event_is_appended_while_the_stream_still_reads_as_running():
    """The invariant the SSE reader depends on.

    That reader drains the backlog and only then tests `running`. If `finish()`
    lowers the flag before appending the terminal event, a reader that samples
    the flag in between exits with `end` still unread — the browser goes on
    believing the turn is live and the banner sits on "接回中…" until
    EventSource reconnects on its own schedule. Ablation: swap the two lines in
    `finish()` and this goes red.
    """
    st = srv.TurnStream("s", None)
    st.emit("delta", text="hi")

    running_at_emit: list[bool] = []
    inner = st.emit

    def spy(kind: str, **data):
        running_at_emit.append(st.running)
        inner(kind, **data)

    st.emit = spy  # type: ignore[method-assign]
    st.finish()

    assert running_at_emit == [True]


@pytest.mark.asyncio
async def test_a_finished_stream_closes_after_delivering_its_end_event(
    aiohttp_client, app
):
    """A finished stream hands over its whole backlog and then closes.

    This pins the ordinary shape — every frame, terminating with `end`, plus the
    `retry` hint. It does NOT distinguish the drain-loop shapes: both the old
    loop and the new one drain before testing the flag, so it stays green under
    that ablation. What pins the loop fix is
    `test_a_turn_finishing_after_the_drain_list_is_fixed_still_sends_end`.
    """
    st = srv.TurnStream("gone", None)
    st.emit("delta", text="partial")
    st.running = False              # as if the flag had been lowered first
    st.emit("end", error=None)
    app["state"].streams["gone"] = st

    cli = await aiohttp_client(app)
    resp = await cli.get("/api/chat/stream", params={"stream_id": "gone", "after_seq": 0})
    body = (await resp.read()).decode()

    kinds = [json.loads(ln[6:])["kind"] for ln in body.splitlines() if ln.startswith("data: ")]
    assert kinds == ["delta", "end"]
    assert "retry: " in body


@pytest.mark.asyncio
async def test_a_reconnect_resumes_from_last_event_id_not_the_stale_query(
    aiohttp_client, app
):
    """EventSource reuses the URL it was constructed with, so `after_seq` is
    frozen at attach time. Only `Last-Event-ID` knows where the reader got to;
    without honouring it, every reconnect re-delivers the whole turn."""
    st = srv.TurnStream("r", None)
    for i in range(4):
        st.emit("delta", text=f"t{i}")
    st.finish()
    app["state"].streams["r"] = st

    cli = await aiohttp_client(app)
    resp = await cli.get(
        "/api/chat/stream",
        params={"stream_id": "r", "after_seq": 0},   # the stale opening position
        headers={"Last-Event-ID": "3"},              # where the browser really is
    )
    body = (await resp.read()).decode()

    texts = [
        json.loads(ln[6:]).get("text")
        for ln in body.splitlines() if ln.startswith("data: ")
    ]
    assert texts == ["t3", None]  # only the delta it missed, then `end`


@pytest.mark.asyncio
async def test_a_turn_finishing_after_the_drain_list_is_fixed_still_sends_end(
    aiohttp_client, app
):
    """The actual race behind a stuck "接回中…".

    `stream.after()` materialises its list before the send loop awaits, so a turn
    that calls `finish()` during one of those writes lands its terminal event
    OUTSIDE the current pass. A reader that then breaks on the `running` flag
    hangs up with `end` undelivered — and the browser, still believing the turn
    is live, sits on the reconnect banner until EventSource retries on its own
    schedule. (hermes-webui hit the same shape from the other direction: a
    terminal event dropped at the resume boundary stalls the reader on
    keepalives.)

    Reproduced by finishing the stream from inside `after()`, immediately after
    the list it returns has been built — the exact interleaving, without having
    to win a race.

    This no longer reddens on the drain-loop shape alone: the synthetic `end`
    the flag path now sends covers delivery here too. What it still pins is that
    the terminal frame arrives PROMPTLY — the 5 s read timeout fails if it waits
    for a heartbeat.
    """
    st = srv.TurnStream("race", None)
    st.emit("delta", text="a")
    app["state"].streams["race"] = st

    inner_after, calls = st.after, 0

    def after(seq: int):
        nonlocal calls
        out = inner_after(seq)
        calls += 1
        if calls == 1:
            st.finish()  # lands after `out` was built, so it is not in this pass
        return out

    st.after = after  # type: ignore[method-assign]

    cli = await aiohttp_client(app)
    resp = await cli.get(
        "/api/chat/stream", params={"stream_id": "race", "after_seq": 0}
    )
    body = (await asyncio.wait_for(resp.read(), 5)).decode()

    kinds = [
        json.loads(ln[6:])["kind"]
        for ln in body.splitlines() if ln.startswith("data: ")
    ]
    assert kinds == ["delta", "end"], f"the terminal event was dropped: {kinds}"


@pytest.mark.asyncio
async def test_a_cursor_ahead_of_the_stream_replays_instead_of_skipping(
    aiohttp_client, app
):
    """A stale id from a previous process must not become the resume point.

    Adopting it would skip every frame below it — the terminal event included —
    and leave the reader waiting on keepalives for a turn that already ended.
    """
    st = srv.TurnStream("ahead", None)
    st.emit("delta", text="only")
    st.finish()
    app["state"].streams["ahead"] = st

    cli = await aiohttp_client(app)
    resp = await cli.get(
        "/api/chat/stream",
        params={"stream_id": "ahead", "after_seq": 0},
        headers={"Last-Event-ID": "99999"},
    )
    body = (await resp.read()).decode()

    kinds = [
        json.loads(ln[6:])["kind"]
        for ln in body.splitlines() if ln.startswith("data: ")
    ]
    assert kinds == ["delta", "end"]


@pytest.mark.asyncio
async def test_a_reader_that_arrives_caught_up_is_still_told_the_turn_ended(
    aiohttp_client, app
):
    """The reload path, and the one that leaves the banner stuck for real.

    A page reload reattaches at the cursor it last saw. If that turn finished
    while the page was away, the reader arrives with nothing outstanding — and
    a handler that just closes has told it nothing, so the browser goes on
    believing the turn is live and EventSource reconnects forever, every 750 ms,
    with the composer locked. The endpoint owes every reader a terminal frame,
    whatever is left in the backlog. Ablation: drop the synthetic `end` from the
    flag-break path and this goes red.
    """
    st = srv.TurnStream("caught-up", None)
    st.emit("delta", text="all of it")
    st.finish()
    app["state"].streams["caught-up"] = st

    cli = await aiohttp_client(app)
    resp = await cli.get(
        "/api/chat/stream", params={"stream_id": "caught-up", "after_seq": st.seq}
    )
    body = (await resp.read()).decode()

    kinds = [
        json.loads(ln[6:])["kind"]
        for ln in body.splitlines() if ln.startswith("data: ")
    ]
    assert kinds == ["end"], f"a caught-up reader was told nothing: {body!r}"


def test_a_finished_turn_does_not_leave_its_cursor_in_local_storage():
    """The client half of the same bug.

    `apply()` persists the cursor AFTER the switch, and the switch is where
    `end` runs `endTurn()` -> `forget()`. Persisting unconditionally therefore
    undid the forget one line later, and the next reload reattached to a stream
    that had nothing left to send. Ablation: drop the `S.busy` guard and this
    goes red.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    line = next(l for l in src.splitlines() if "remember(S.streamId, ev.seq)" in l)
    assert "S.busy" in line, (
        "apply() persists the stream cursor unconditionally; `end` clears it "
        f"earlier in the same call and this line writes it back: {line.strip()}"
    )


@pytest.mark.asyncio
async def test_a_turn_finishing_between_the_drain_and_the_flag_check_delivers(
    aiohttp_client, app
):
    """The materialisation window, one branch further down.

    The flag path synthesises a terminal frame for a reader with nothing
    outstanding. But `after()` fixed its list before this pass, so a turn that
    finished during it left real events behind — and synthesising `end` without
    re-reading would hang up on them. Ablation: drop the `if stream.after(after):
    continue` guard and the delta never reaches the reader.
    """
    assert srv.SSE_HEARTBEAT_SEC >= 10, "the timeout below must be well under it"
    st = srv.TurnStream("wakeup", None)
    app["state"].streams["wakeup"] = st

    inner_after, fired = st.after, False

    def after(seq: int):
        nonlocal fired
        out = inner_after(seq)
        if not out and not fired:      # the reader is about to park
            fired = True
            st.emit("delta", text="late")
            st.finish()
        return out

    st.after = after  # type: ignore[method-assign]

    cli = await aiohttp_client(app)
    resp = await cli.get(
        "/api/chat/stream", params={"stream_id": "wakeup", "after_seq": 0}
    )
    body = (await asyncio.wait_for(resp.read(), 5)).decode()

    assert fired, "the window was never entered; the test proves nothing"
    kinds = [
        json.loads(ln[6:])["kind"]
        for ln in body.splitlines() if ln.startswith("data: ")
    ]
    assert kinds == ["delta", "end"], kinds


@pytest.mark.asyncio
async def test_an_event_emitted_between_the_drain_and_the_park_is_not_slept_through(
    aiohttp_client, app
):
    """The lost-wakeup window, and why the reader takes the bell up front.

    `emit()` sets the current wakeup and then swaps in a fresh one. A reader
    that reads the backlog, finds it empty, and only THEN reaches for
    `stream._bell` is handed the replacement — while the wakeup meant for it has
    already gone off. It sleeps the full heartbeat with the event sitting in the
    deque. Taking the bell BEFORE reading the backlog closes that: whatever is
    emitted after the snapshot rings the object the reader is holding.

    Unlike the test above, this one leaves the stream RUNNING at the window, so
    the reader actually parks — which is the only state where the bell matters.
    Ablation: await `stream.wait()` instead of the snapshot and the reader sleeps
    until the heartbeat, failing the 5 s read timeout.
    """
    assert srv.SSE_HEARTBEAT_SEC >= 10, "the timeout below must be well under it"
    st = srv.TurnStream("park", None)
    app["state"].streams["park"] = st

    inner_after, hits = st.after, 0

    def after(seq: int):
        nonlocal hits
        out = inner_after(seq)
        if not out:
            hits += 1
            if hits == 1:
                st.emit("delta", text="late")   # stream stays running -> park
            elif hits == 2:
                st.finish()
        return out

    st.after = after  # type: ignore[method-assign]

    cli = await aiohttp_client(app)
    resp = await cli.get(
        "/api/chat/stream", params={"stream_id": "park", "after_seq": 0}
    )
    body = (await asyncio.wait_for(resp.read(), 5)).decode()

    assert hits >= 2, "the reader never parked; the test proves nothing"
    kinds = [
        json.loads(ln[6:])["kind"]
        for ln in body.splitlines() if ln.startswith("data: ")
    ]
    assert kinds == ["delta", "end"], kinds
    assert ": keepalive" not in body, "the reader slept through its own wakeup"
