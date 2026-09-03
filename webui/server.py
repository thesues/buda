#!/usr/bin/env python3
"""hermes webui — session management and a chat box, and deliberately nothing else.

Shape (the one thing worth understanding before reading the code):

    POST /api/chat/start   -> make a TurnStream, spawn a task, return {stream_id}
    GET  /api/chat/stream  -> SSE; replays from ?after_seq and then follows
    GET  /api/chat/status  -> is that stream still running, and at what seq

The agent turn runs in its own task writing into a sequence-numbered buffer. It
is NOT bound to the connection that started it. That is the whole design, and it
is a direct answer to how the lerobot console failed: there, the turn lived on a
WebSocket, so a reload lost the stream id, an idle proxy timeout killed the turn,
and "the UI stopped responding, I had to refresh" was the standard bug report.
Here a browser can close, reload, or reconnect from a different tab; the turn
does not notice, and the reconnect replays only what it missed.

Adapted from `lerobot-agent-console`. What was dropped, and why it is not an
oversight: the PTY terminal, the port proxy, the service discovery and the
lerobot/volcano-specific endpoints. A terminal is where most of that console's
hard-won fixes live (process-group reaping, idle reclamation, output backlogs) —
none of which can regress here, because there is no terminal to regress.

What survives from those fixes is the IDEA behind the terminal's output backlog:
number the events, keep a bounded window, and let a reconnecting reader ask for
the delta. Applied to chat instead of to a shell.

Retrieval is NOT built in here. hermes talks to `memory-mcp` over HTTP MCP as a
configured server (MEMORY_MCP_URL), so this process never spawns it, never holds
an autumn credential for it, and never proxies its traffic.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hmac
import json
import logging
import os
import secrets
import shutil
import time
from collections import deque
from pathlib import Path

from aiohttp import web

log = logging.getLogger("webui")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

PORT = int(os.environ.get("PORT", "8080"))
WORKDIR = os.environ.get("WORKDIR") or os.path.expanduser("~")
HERMES_BIN = os.environ.get("HERMES_BIN") or shutil.which("hermes") or "hermes"
# `hermes_session_api.py` must run in HERMES' interpreter, not ours: the two
# venvs share most packages and disagree on some, so importing hermes_state
# in-process would shadow one's deps with the other's.
HERMES_PY = os.environ.get("HERMES_PY") or str(Path(HERMES_BIN).resolve().with_name("python"))
HERMES_SESSION_API = str(HERE / "hermes_session_api.py")

# The MCP server hermes is pointed at. Empty disables the wiring entirely, which
# is what a local run without a cluster wants.
MEMORY_MCP_URL = os.environ.get("MEMORY_MCP_URL", "http://memory-mcp:5100/mcp")
MEMORY_MCP_NAME = os.environ.get("MEMORY_MCP_NAME", "memory")


AUTH_USER = os.environ.get("AUTH_USER", "")
AUTH_PASS = os.environ.get("AUTH_PASS", "")

# How many events one turn keeps for replay. A turn is a few hundred deltas; the
# cap exists so a runaway tool loop cannot grow this without bound. Overflowing
# it is REPORTED (see `dropped`) rather than silently truncating the transcript —
# a reader that cannot be made whole must be told, not quietly lied to.
BACKLOG_EVENTS = 4000

# Finished turns are kept this long so a reload can still collect the ending.
# Without it, "reload right as the answer lands" shows an empty chat.
TURN_RETENTION_SEC = 900

# An unanswered permission request expires rather than pinning the turn forever.
APPROVAL_TIMEOUT_SEC = 600

# SSE idle comment interval. Proxies and load balancers cut a silent connection;
# a comment line is the cheapest thing that keeps it open and costs the client
# nothing (EventSource ignores comments).
SSE_HEARTBEAT_SEC = 25

# asyncio's StreamReader caps a line at 64 KiB by default, and ACP frames one
# JSON-RPC message per line. A chat reply never comes close; an MCP tool RESULT
# does -- a corpus search returning document text blew straight past it, and
# `readline()` raises rather than truncating, which killed the read loop and
# left the turn dead with no output and no error. Sized for a tool result, not
# for a sentence.
ACP_LINE_LIMIT = 32 * 1024 * 1024

CHAT_DIRECTIVE = os.environ.get(
    "CHAT_DIRECTIVE",
    "请用简洁的 Markdown 回答；代码用围栏代码块并标注语言。",
)


# --------------------------------------------------------------------------- #
# Auth                                                                          #
# --------------------------------------------------------------------------- #
@web.middleware
async def auth_middleware(request: web.Request, handler):
    # The health endpoint stays open so a k8s probe needs no credential, and the
    # static assets are useless without the API behind them.
    if request.path == "/healthz" or not AUTH_USER:
        return await handler(request)
    hdr = request.headers.get("Authorization", "")
    if hdr.startswith("Basic "):
        try:
            user, _, passwd = base64.b64decode(hdr[6:]).decode().partition(":")
        except Exception:  # noqa: BLE001
            user = passwd = ""
        # compare_digest on both halves: a plain `==` on the user leaks its
        # length through timing just as surely as one on the password.
        if hmac.compare_digest(user, AUTH_USER) and hmac.compare_digest(passwd, AUTH_PASS):
            return await handler(request)
    return web.Response(
        status=401,
        headers={"WWW-Authenticate": 'Basic realm="hermes"'},
        text="unauthorized",
    )


# --------------------------------------------------------------------------- #
# TurnStream — a turn's events, owned by the app and not by any connection       #
# --------------------------------------------------------------------------- #
class TurnStream:
    """One agent turn's event log: sequence-numbered, bounded, replayable.

    Every event carries a monotonic `seq`. A reader reconnects with the last seq
    it saw and gets only what came after, so a reload costs a few hundred bytes
    instead of the transcript — and, more to the point, costs the TURN nothing at
    all, because the turn was never reading from the connection.
    """

    def __init__(self, stream_id: str, session_id: str | None) -> None:
        self.stream_id = stream_id
        self.session_id = session_id
        self.seq = 0
        self.events: deque[tuple[int, dict]] = deque(maxlen=BACKLOG_EVENTS)
        # Events evicted by the cap. A reader asking for a seq older than the
        # window gets told the gap exists instead of receiving a transcript with
        # a silent hole in it.
        self.dropped = 0
        self.running = True
        self.finished_at: float | None = None
        self.error: str | None = None
        # One shared Event, replaced on each emit: cheaper than per-subscriber
        # queues and correct because subscribers re-read from their own seq.
        self._bell = asyncio.Event()

    def emit(self, kind: str, **data) -> None:
        self.seq += 1
        if len(self.events) == self.events.maxlen:
            self.dropped += 1
        self.events.append((self.seq, {"kind": kind, "seq": self.seq, **data}))
        self._bell.set()
        self._bell = asyncio.Event()

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self.running = False
        self.finished_at = time.time()
        self.emit("end", error=error)

    def after(self, seq: int) -> list[dict]:
        return [e for s, e in self.events if s > seq]

    def gap_before(self, seq: int) -> bool:
        """Did the cap evict something this reader still needs?"""
        if not self.dropped or not self.events:
            return False
        return seq < self.events[0][0] - 1

    async def wait(self) -> None:
        await self._bell.wait()


class Approval:
    """A pending `session/request_permission`, answerable out of band.

    The ACP callback cannot return until a human decides, and the decision now
    arrives on a DIFFERENT request than the one streaming the turn. So the
    callback parks on a future and the answer endpoint resolves it. This is also
    why `/api/approval/pending` exists: a reloaded browser has no memory of the
    prompt, and the push that told it about the prompt is long gone.
    """

    def __init__(self, approval_id: str, stream_id: str, params: dict) -> None:
        self.id = approval_id
        self.stream_id = stream_id
        self.params = params
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.created_at = time.time()

    def brief(self) -> dict:
        p = self.params
        return {
            "id": self.id,
            "streamId": self.stream_id,
            "title": p.get("toolCall", {}).get("title") or p.get("title") or "permission",
            "options": p.get("options") or [],
        }


class State:
    """Everything that changes while the server runs.

    Held as ONE object stored in the app at build time, rather than as a handful
    of `app["..."]` keys: aiohttp deprecates mutating the application mapping
    after startup, and a handler assigning `app["turn_task"]` was doing exactly
    that. Attributes on a stored object are ordinary state, not app config.
    """

    def __init__(self) -> None:
        self.acp = HermesACP()
        self.streams: dict[str, TurnStream] = {}
        self.approvals: dict[str, Approval] = {}
        self.turn_task: asyncio.Task | None = None
        self.current: TurnStream | None = None
        self.gc: asyncio.Task | None = None


# --------------------------------------------------------------------------- #
# hermes ACP — one warm process, driven over JSON-RPC on stdio                   #
# --------------------------------------------------------------------------- #
# Cold-spawning `hermes chat` per turn costs ~10 s of startup each time, so one
# `hermes acp` process is kept alive: initialize -> session/new (once) ->
# session/prompt (per turn, streaming).
class HermesACP:
    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.session_id: str | None = None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        # Sessions already steered with CHAT_DIRECTIVE. A loaded session is
        # marked on load so the directive is never injected mid-conversation.
        self._directive_sent: set[str] = set()
        self.on_update = None      # async fn(update)
        self.on_permission = None  # async fn(params) -> optionId | None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def ensure(self) -> None:
        async with self._start_lock:
            if not self.alive:
                await self._spawn()
            if not self.session_id:
                await self._new_locked()

    async def ensure_proc(self) -> None:
        async with self._start_lock:
            if not self.alive:
                await self._spawn()

    async def _spawn(self) -> None:
        env = dict(os.environ)
        env["HERMES_ACCEPT_HOOKS"] = "1"
        env.setdefault("NO_COLOR", "1")
        # stderr to a file, never DEVNULL: when this process dies its dying words
        # are the only evidence, and DEVNULL made every "acp exited" undebuggable.
        stderr_path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "logs"
        stderr_path.mkdir(parents=True, exist_ok=True)
        stderr_f = open(stderr_path / "acp_stderr.log", "ab")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                HERMES_BIN, "acp", "--accept-hooks",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=stderr_f, cwd=WORKDIR, env=env, limit=ACP_LINE_LIMIT,
            )
        finally:
            stderr_f.close()  # the child holds its own fd copy
        # A fresh pending map PER PROCESS, with the read loop bound to the pair:
        # a dead process's exit cleanup must never fail a newer process's
        # in-flight requests (that raced, and killed the replacement's
        # `initialize` with "acp exited").
        self._pending = pending = {}
        asyncio.create_task(self._read_loop(self.proc, pending))
        await self._request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        log.info("hermes acp ready")

    async def _new_locked(self) -> str:
        res = await self._request(
            "session/new", {"cwd": WORKDIR, "mcpServers": _acp_mcp_servers()}
        )
        self.session_id = res.get("sessionId")
        log.info("hermes acp session=%s", self.session_id)
        return self.session_id

    async def new_session(self) -> None:
        """Arm a new session WITHOUT creating one; the next prompt creates it.

        Creating it eagerly is what littered the store with titleless zero-message
        ghosts — one per page open, per restart — because most never get a message.
        """
        await self.ensure_proc()
        self.session_id = None

    async def list_sessions(self) -> list[dict]:
        # hermes' own SessionDB via its venv, NOT ACP `session/list`: that caches
        # in memory, so a session deleted through the CLI kept reappearing.
        proc = await asyncio.create_subprocess_exec(
            HERMES_PY, HERMES_SESSION_API, "list", "--limit", "200",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"session list failed (rc={proc.returncode}): "
                               f"{err.decode(errors='replace')[:400]}")
        return json.loads(out.decode() or "[]")

    async def load_session(self, sid: str, on_update) -> None:
        await self.ensure_proc()
        self.on_update = on_update
        try:
            await self._request(
                "session/load",
                {"sessionId": sid, "cwd": WORKDIR, "mcpServers": _acp_mcp_servers()},
            )
        finally:
            self.on_update = None
        self.session_id = sid
        self._directive_sent.add(sid)  # has history — never inject the directive

    async def delete_session(self, sid: str) -> None:
        # Through hermes' CLI, which is schema-aware (it also cleans the FTS
        # index and related tables). No SQL fallback; a failure is surfaced.
        proc = await asyncio.create_subprocess_exec(
            HERMES_BIN, "sessions", "delete", sid, "--yes",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"`hermes sessions delete {sid}` failed "
                               f"(rc={proc.returncode}): {out.decode(errors='replace')[:400]}")
        self._directive_sent.discard(sid)
        if self.session_id == sid:
            self.session_id = None

    async def _write(self, obj: dict) -> None:
        async with self._write_lock:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            await self.proc.stdin.drain()

    async def _request(self, method: str, params: dict):
        self._id += 1
        rid = self._id
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        res = await fut
        # A JSON-RPC error comes back as a RESULT carrying `_error` (the read
        # loop cannot raise into a future's awaiter without this shape). Raise
        # it here, at the one choke point every request goes through: returning
        # it made a failed `session/new` look like success, so `session_id`
        # became None and the next prompt sent `sessionId: null` -- the turn
        # then ended with no content AND no error, which is the worst of both.
        if isinstance(res, dict) and "_error" in res:
            err = res["_error"] or {}
            raise RuntimeError(
                f"hermes acp {method} failed: "
                f"{err.get('message') or err} (code {err.get('code', '?')})"
            )
        return res

    async def _read_loop(self, proc, pending: dict) -> None:
        while True:
            try:
                line = await proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError) as e:
                # One oversized frame must not take the whole connection down.
                # The message is lost either way -- readline() cannot resume mid
                # line -- but the session survives to serve the next turn, and
                # the reason reaches the log instead of vanishing.
                log.error("acp line exceeded %d bytes, dropping the frame: %s",
                          ACP_LINE_LIMIT, e)
                continue
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = pending.pop(msg["id"], None)
                if fut and not fut.done():
                    fut.set_result(msg.get("result") or {"_error": msg.get("error")})
            elif msg.get("method") == "session/update":
                # Drop stragglers from a superseded process.
                if self.on_update and self.proc is proc:
                    try:
                        await self.on_update(msg["params"].get("update", {}))
                    except Exception:  # noqa: BLE001
                        log.debug("on_update failed", exc_info=True)
            elif msg.get("method") == "session/request_permission":
                if self.proc is proc:
                    await self._reply_permission(msg)
            elif "id" in msg:  # a server->client request we do not implement
                if self.proc is proc:
                    await self._write({"jsonrpc": "2.0", "id": msg["id"],
                                       "error": {"code": -32601, "message": "unsupported"}})
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("hermes acp exited"))
        pending.clear()
        if self.proc is proc:
            self.session_id = None
        log.warning("hermes acp exited (rc=%s%s)", proc.returncode,
                    "" if self.proc is proc else ", superseded")

    async def _reply_permission(self, msg: dict) -> None:
        option_id = None
        if self.on_permission:
            try:
                option_id = await self.on_permission(msg.get("params", {}))
            except Exception:  # noqa: BLE001
                option_id = None
        outcome = ({"outcome": "selected", "optionId": option_id} if option_id
                   else {"outcome": "cancelled"})
        await self._write({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})

    async def prompt(self, text: str, on_update, on_permission):
        await self.ensure()
        if self.session_id not in self._directive_sent:
            # APPEND, never prepend: the session's auto-title comes from the
            # user's real first words, not from the steering block.
            text = text + "\n\n" + CHAT_DIRECTIVE
            self._directive_sent.add(self.session_id)
        self.on_update = on_update
        self.on_permission = on_permission
        try:
            return await self._request(
                "session/prompt",
                {"sessionId": self.session_id, "prompt": [{"type": "text", "text": text}]},
            )
        finally:
            self.on_update = None
            self.on_permission = None

    async def cancel(self) -> None:
        if self.alive and self.session_id:
            await self._write({"jsonrpc": "2.0", "method": "session/cancel",
                               "params": {"sessionId": self.session_id}})



def _acp_mcp_servers() -> list[dict]:
    """The MCP servers to register on an ACP session.

    This is the ONLY channel that reaches the agent's toolset. Writing
    `mcp_servers` into hermes' config.yaml registers the server -- `hermes mcp
    list` shows it enabled -- and the ACP session still has no such tool,
    because the adapter builds a session's tools from THIS parameter and
    ignores the file. Verified the hard way: the agent answered that no
    `search_docs` tool existed while the CLI listed the server as enabled.

    Two fields are easy to miss and both are load-bearing. `type` is the union
    DISCRIMINATOR: `session/new` takes `HttpMcpServer`, which subclasses
    `McpServerHttp` only to add `type: Literal["http"]` -- send the parent's
    shape and the whole request is rejected with `Invalid params`. `headers` is
    required with no default, so omitting it fails validation the same way.
    """
    if not MEMORY_MCP_URL:
        return []
    return [{"type": "http", "name": MEMORY_MCP_NAME, "url": MEMORY_MCP_URL, "headers": []}]


# --------------------------------------------------------------------------- #
# Turn events                                                                   #
# --------------------------------------------------------------------------- #
def _chunk_text(update: dict) -> str:
    c = update.get("content") or {}
    return c.get("text", "") if isinstance(c, dict) else ""


def _to_event(update: dict) -> tuple[str, dict] | None:
    """Map one ACP `session/update` to a UI event, or None to ignore it.

    Deliberately lossy: this UI is a chat box, so it renders assistant text and
    a one-line trace of tool activity. Anything else is dropped HERE, where the
    decision is visible, rather than being streamed and ignored by the client.
    """
    kind = update.get("sessionUpdate")
    if kind in ("agent_message_chunk", "agent_thought_chunk"):
        text = _chunk_text(update)
        if not text:
            return None
        return ("delta", {"text": text, "thought": kind == "agent_thought_chunk"})
    if kind == "tool_call":
        return ("tool", {"id": update.get("toolCallId"),
                         "title": update.get("title") or update.get("kind") or "tool",
                         "status": update.get("status") or "pending"})
    if kind == "tool_call_update":
        return ("tool", {"id": update.get("toolCallId"),
                         "title": update.get("title") or "",
                         "status": update.get("status") or ""})
    if kind == "user_message_chunk":
        # Only seen while replaying a loaded session's history.
        text = _chunk_text(update)
        return ("history_user", {"text": text}) if text else None
    return None


# --------------------------------------------------------------------------- #
# Chat — start / stream / status / cancel                                       #
# --------------------------------------------------------------------------- #
async def handle_chat_start(request: web.Request) -> web.Response:
    st = state(request)
    if st.turn_task and not st.turn_task.done():
        # One turn at a time. Returning the RUNNING stream id rather than an
        # error means a double-click, or a reload that re-submits, attaches to
        # the turn in flight instead of being told "busy" with no way back to it.
        cur = st.current
        return web.json_response({"streamId": cur.stream_id if cur else None,
                                  "attached": True})
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid JSON"}, status=400)
    text = (body.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "empty message"}, status=400)

    acp = st.acp
    stream = TurnStream(secrets.token_hex(8), acp.session_id)
    st.streams[stream.stream_id] = stream
    st.current = stream
    stream.emit("user", text=text)

    async def on_update(update: dict) -> None:
        ev = _to_event(update)
        if ev:
            stream.emit(ev[0], **ev[1])

    async def on_permission(params: dict):
        ap = Approval(secrets.token_hex(6), stream.stream_id, params)
        st.approvals[ap.id] = ap
        stream.emit("approval", **ap.brief())
        try:
            return await asyncio.wait_for(ap.future, timeout=APPROVAL_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            # Cancel rather than hold the turn open forever. The agent gets a
            # clean "denied" and the transcript says why.
            stream.emit("approval_expired", id=ap.id)
            return None
        finally:
            st.approvals.pop(ap.id, None)

    async def run() -> None:
        try:
            res = await acp.prompt(text, on_update, on_permission)
            stream.session_id = acp.session_id
            reason = (res or {}).get("stopReason")
            if reason and reason != "end_turn":
                stream.emit("note", text=f"stopped: {reason}")
            stream.finish()
        except Exception as e:  # noqa: BLE001
            log.warning("turn failed: %s", e)
            stream.finish(error=str(e))

    st.turn_task = asyncio.create_task(run())
    return web.json_response({"streamId": stream.stream_id, "attached": False})


async def handle_chat_stream(request: web.Request) -> web.StreamResponse:
    sid = request.query.get("stream_id", "")
    stream: TurnStream | None = state(request).streams.get(sid)
    if stream is None:
        # 404 rather than an empty stream: the client must be able to tell
        # "that turn is gone" from "that turn has said nothing yet".
        return web.json_response({"error": "unknown stream"}, status=404)
    try:
        after = int(request.query.get("after_seq", "0"))
    except ValueError:
        after = 0

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        # Nginx buffers text/event-stream by default, which turns a live stream
        # into one delivery at the end.
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    async def send(obj: dict) -> None:
        await resp.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

    if stream.gap_before(after):
        await send({"kind": "gap", "seq": after,
                    "text": "some output was dropped while disconnected"})
    try:
        while True:
            for ev in stream.after(after):
                await send(ev)
                after = ev["seq"]
            if not stream.running:
                break
            try:
                await asyncio.wait_for(stream.wait(), timeout=SSE_HEARTBEAT_SEC)
            except asyncio.TimeoutError:
                # A comment line: EventSource ignores it, proxies see traffic.
                await resp.write(b": keepalive\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        # The reader went away. The TURN is untouched — that is the point.
        pass
    with contextlib.suppress(Exception):
        await resp.write_eof()
    return resp


async def handle_chat_status(request: web.Request) -> web.Response:
    """What a reloaded page asks before deciding whether to reattach."""
    sid = request.query.get("stream_id", "")
    stream: TurnStream | None = state(request).streams.get(sid)
    if stream is None:
        return web.json_response({"known": False})
    return web.json_response({
        "known": True, "running": stream.running, "lastSeq": stream.seq,
        "sessionId": stream.session_id, "error": stream.error,
    })


async def handle_chat_cancel(request: web.Request) -> web.Response:
    await state(request).acp.cancel()
    return web.json_response({"ok": True})


# --------------------------------------------------------------------------- #
# Approvals — pushed on the stream, and pollable because a push can be missed   #
# --------------------------------------------------------------------------- #
async def handle_approval_pending(request: web.Request) -> web.Response:
    return web.json_response({"pending": [a.brief() for a in state(request).approvals.values()]})


async def handle_approval_answer(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid JSON"}, status=400)
    ap: Approval | None = state(request).approvals.get(body.get("id"))
    if ap is None:
        # Already answered, expired, or from a previous process. Not an error
        # worth surfacing: the client polls and will simply stop seeing it.
        return web.json_response({"ok": False, "reason": "unknown or already answered"})
    if not ap.future.done():
        ap.future.set_result(body.get("optionId"))
    return web.json_response({"ok": True})


# --------------------------------------------------------------------------- #
# Sessions                                                                      #
# --------------------------------------------------------------------------- #
async def handle_sessions(request: web.Request) -> web.Response:
    acp = state(request).acp
    try:
        rows = await acp.list_sessions()
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"sessions": rows, "current": acp.session_id})


async def handle_session_new(request: web.Request) -> web.Response:
    await state(request).acp.new_session()
    return web.json_response({"ok": True, "current": None})


async def handle_session_load(request: web.Request) -> web.Response:
    """Switch to an existing session and hand back its replayed history.

    The replay is collected into the RESPONSE rather than pushed on a stream:
    it is a bounded, already-complete transcript, so a plain request/response is
    the honest shape. Streams are for turns, which are neither.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid JSON"}, status=400)
    sid = (body.get("id") or "").strip()
    if not sid:
        return web.json_response({"error": "missing id"}, status=400)

    history: list[dict] = []

    async def on_update(update: dict) -> None:
        ev = _to_event(update)
        if ev:
            history.append({"kind": ev[0], **ev[1]})

    try:
        await state(request).acp.load_session(sid, on_update)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"ok": True, "current": sid, "history": history})


async def handle_session_delete(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    try:
        await state(request).acp.delete_session(sid)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"ok": True})


# --------------------------------------------------------------------------- #
# Status / health / index                                                       #
# --------------------------------------------------------------------------- #
async def handle_status(request: web.Request) -> web.Response:
    acp = state(request).acp
    cur = state(request).current
    return web.json_response({
        "acpAlive": acp.alive,
        "session": acp.session_id,
        "mcp": {"name": MEMORY_MCP_NAME, "url": MEMORY_MCP_URL} if MEMORY_MCP_URL else None,
        "turn": {"streamId": cur.stream_id, "running": cur.running} if cur else None,
    })


async def handle_health(_request: web.Request) -> web.Response:
    # Deliberately shallow: it must answer while a turn is running and while the
    # acp process is restarting. A probe that fails during either would restart
    # the pod in exactly the moments the design set out to survive.
    return web.json_response({"ok": True})


async def handle_index(_request: web.Request) -> web.StreamResponse:
    return web.FileResponse(STATIC / "index.html")


# --------------------------------------------------------------------------- #
# Lifecycle                                                                     #
# --------------------------------------------------------------------------- #
async def _gc_streams(st: "State") -> None:
    """Drop finished turns after a grace period, and expire stale approvals."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for sid, s in list(st.streams.items()):
            if not s.running and s.finished_at and now - s.finished_at > TURN_RETENTION_SEC:
                st.streams.pop(sid, None)
        for aid, a in list(st.approvals.items()):
            if now - a.created_at > APPROVAL_TIMEOUT_SEC * 2:
                st.approvals.pop(aid, None)


async def _on_startup(app: web.Application) -> None:
    if MEMORY_MCP_URL:
        from hermes_config import ensure_mcp_server
        cfg = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "config.yaml"
        try:
            ensure_mcp_server(cfg, MEMORY_MCP_NAME, MEMORY_MCP_URL)
        except Exception as e:  # noqa: BLE001
            # Loud, but not fatal: chat without retrieval beats no chat, and the
            # /api/status surface reports what was configured either way.
            log.error("could not point hermes at %s: %s", MEMORY_MCP_URL, e)
    st = app["state"]
    st.gc = asyncio.create_task(_gc_streams(st))


async def _on_cleanup(app: web.Application) -> None:
    st: State = app["state"]
    if st.gc:
        st.gc.cancel()
        with contextlib.suppress(Exception):
            await st.gc
    acp = st.acp
    if acp.alive:
        with contextlib.suppress(ProcessLookupError):
            acp.proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(acp.proc.wait(), timeout=5)


def state(request: web.Request) -> "State":
    return request.app["state"]


def build_app() -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app["state"] = State()
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_get("/", handle_index)
    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/api/status", handle_status)
    app.router.add_post("/api/chat/start", handle_chat_start)
    app.router.add_get("/api/chat/stream", handle_chat_stream)
    app.router.add_get("/api/chat/status", handle_chat_status)
    app.router.add_post("/api/chat/cancel", handle_chat_cancel)
    app.router.add_get("/api/approval/pending", handle_approval_pending)
    app.router.add_post("/api/approval/answer", handle_approval_answer)
    app.router.add_get("/api/sessions", handle_sessions)
    app.router.add_post("/api/session/new", handle_session_new)
    app.router.add_post("/api/session/load", handle_session_load)
    app.router.add_delete(r"/api/session/{sid}", handle_session_delete)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if AUTH_USER and not AUTH_PASS:
        raise SystemExit("AUTH_USER is set but AUTH_PASS is empty — refusing to start "
                         "with an unusable credential")
    log.info("hermes webui :%s  hermes=%s  mcp=%s", args.port, HERMES_BIN,
             MEMORY_MCP_URL or "(none)")
    # Plain HTTP: TLS is terminated upstream by the ingress / load balancer.
    web.run_app(build_app(), host="0.0.0.0", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
