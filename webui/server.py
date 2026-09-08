#!/usr/bin/env python3
"""hermes webui — session management and a chat box, and deliberately nothing else.

Shape (the one thing worth understanding before reading the code):

    POST /api/chat/start   -> make a TurnStream, spawn a task, return {stream_id}
    GET  /api/chat/stream  -> SSE; replays from ?after_seq and then follows
    GET  /api/chat/status  -> is that stream still running, and at what seq

The agent turn runs in its own task writing into a sequence-numbered buffer. It
is NOT bound to the connection that started it, and it is keyed by SESSION, so
several conversations can be replying at once over the one `hermes acp` process:
ACP multiplexes on `sessionId`, and the single-turn limit this server used to
report was its own — one session field, one pair of callbacks. That is the whole design, and it
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
import re
import shutil
import time
from html import escape as html_escape
from collections import deque
from pathlib import Path

from aiohttp import web

log = logging.getLogger("webui")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

# The browser tab's name. The page ships with "buda · 佛典检索"; this rewrites it
# at startup so one deployment can say what IT is without a rebuild -- the same
# reason CHAT_DIRECTIVE is an env var. Empty keeps whatever the file has.
PAGE_TITLE = os.environ.get("PAGE_TITLE", "")

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


# Tells one BROWSER from another. Not authentication -- everyone here shares
# one credential and is trusted; this only stops two people sharing a deployment
# from stepping on each other. Anyone can forge it, and forging it buys nothing
# they could not already do.
#
# A cookie rather than a header because `EventSource` cannot set headers, and a
# cookie rides every request including the SSE one with no call site changed.
# The server mints it on the page load, so the client never has to think about
# it either.
CLIENT_COOKIE = "cid"

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

# How long a graceful `session/cancel` gets before the process is restarted.
# `session/cancel` CANNOT interrupt a running tool -- the agent notices only
# when the tool returns -- so without an escalation Stop silently does nothing
# for the whole length of a slow search. But the wait must be REALISTIC: at 1.5s
# nearly every Stop escalated, and a restart throws away what the turn had
# already produced. Stop means "end this, keep what it made", so it is worth
# waiting out the wind-down. Restart is the last resort, not the normal path.
CANCEL_GRACE_SEC = float(os.environ.get("CANCEL_GRACE_SEC", "12"))

# How many turns may be in flight at once, across all sessions.
#
# Four, because that is what the agent side actually offers: hermes' ACP adapter
# runs sessions on a shared `ThreadPoolExecutor(max_workers=4)`. It is not a
# property of the pipe -- one stdio connection multiplexes any number of
# sessions, every session-scoped frame carrying `sessionId` -- and it was never
# one. This server used to say 1, and that was its own limitation: a single
# `session_id` field and a single pair of callbacks on the ACP client.
#
# The limit BELOW this one is the model, not the agent: `freetoken-l3` runs with
# `--max-running-requests 1`, chosen because 4 put a 24 GB card into CUDA OOM.
# Raising this past that does not make two replies arrive at once -- it makes the
# second one queue at the model instead of being refused at the door, which is
# still the better failure.
MAX_CONCURRENT_TURNS = max(1, int(os.environ.get("MAX_CONCURRENT_TURNS", "4")))

# How long a reader must stay on a conversation before it is warmed on the ACP
# process. Negative disables the warming entirely.
#
# The cost being moved is 1.3 s, measured, and it is NOT the cost of reading
# history — `session/new` measures the same 1310 ms as `session/load`'s 1315 ms,
# because both handlers rebuild the agent's whole tool surface
# (`_register_session_mcp_servers` -> `register_mcp_servers` +
# `get_tool_definitions`). A session with no history to load pays it too. It is
# the price of this process touching a session for the FIRST time, once each.
#
# Unwarmed, that 1.3 s lands on the send path — after the reader has typed and
# pressed enter, which is the worst place for it. Reading a transcript costs
# 5-9 ms and a human then types for seconds, so warming on VIEW hides it
# entirely and `ensure_known` at send time returns in ~0 ms.
#
# The debounce is what keeps that from being speculative work on everything the
# reader scrolls past: warm what they are still looking at, not what they went
# through to get there.
PREFETCH_DEBOUNCE_SEC = float(os.environ.get("PREFETCH_DEBOUNCE_SEC", "0.4"))

# SSE idle comment interval. Proxies and load balancers cut a silent connection;
# a comment line is the cheapest thing that keeps it open and costs the client
# nothing (EventSource ignores comments).
SSE_HEARTBEAT_SEC = 25

# asyncio's StreamReader caps a line at 64 KiB by default, and BOTH pipes here
# frame one JSON message per line -- ACP's JSON-RPC, and the session bridge's
# request/response. A chat reply never comes close; an MCP tool RESULT does --
# a corpus search returning document text blew straight past it, and
# `readline()` raises rather than truncating, which killed the read loop and
# left the turn dead with no output and no error. Sized for a tool result, not
# for a sentence.
#
# The bridge hit the identical wall from the other direction and was missed for
# longer, because its failure looked like something else: a transcript is ONE
# line, so any conversation past ~64 KiB could not be opened at all. Measured on
# a real session: 196,503 bytes, three times the default. Any pipe read with
# `readline()` in this file gets this limit.
ACP_LINE_LIMIT = 32 * 1024 * 1024

# Appended to the first prompt of a session, never prepended: the session's
# auto-title is taken from its opening words, and a prepended block titles every
# conversation after the directive instead of after the question.
#
# It says what this deployment is FOR. The default it replaced was inherited
# from a coding console ("use fenced code blocks with a language tag") and
# steered the agent toward long prose, which is the opposite of what a scripture
# lookup wants.
CHAT_DIRECTIVE = os.environ.get(
    "CHAT_DIRECTIVE",
    "你是佛教典籍的检索助手，工作是从已索引的语料库中找出依据来回答问题。\n"
    "- 凡涉及经文内容的问题，先用语料库工具检索，不要凭记忆作答。\n"
    "- 回答时给出经名与原文引文，并标出出处（文件 › 标题路径 › 行号）。\n"
    "- 语料库里没有的，直说没有；可以补充常识背景，但要注明那不是来自语料库。\n"
    "- 简明作答，通常几段以内。除非明确要求，不要写长文、不要综述式铺陈。",
)


# --------------------------------------------------------------------------- #
# Auth                                                                          #
# --------------------------------------------------------------------------- #
@web.middleware
async def client_cookie_middleware(request: web.Request, handler):
    """Give every caller a browser id, on whatever it asked for first.

    Minting only on `/` was not enough: anything that reaches an API endpoint
    without having loaded the page — a second tab restored from history, a
    client whose cookie was cleared, curl — arrives with no id, and every such
    caller shares the SAME empty one. Two of them then read as one person, which
    is the bug this whole mechanism exists to prevent.

    The id is assigned BEFORE the handler runs, not just returned after it: a
    caller whose very first request is an API call must be attributed to the id
    it is about to receive, or that one request is anonymous and the next one is
    somebody else -- which read as two people to everything downstream.

    An SSE response is already on the wire by the time this returns, so it
    cannot carry a cookie. That is fine: a stream is never a first request, and
    the id assigned above is still what the handler saw.
    """
    fresh = "" if request.cookies.get(CLIENT_COOKIE) else secrets.token_hex(8)
    if fresh:
        request["cid"] = fresh
    resp = await handler(request)
    if fresh and not resp.prepared:
        resp.set_cookie(CLIENT_COOKIE, fresh, max_age=31536000,
                        path="/", httponly=True, samesite="Lax")
    return resp


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

    def __init__(self, stream_id: str, session_id: str | None,
                 client_id: str = "") -> None:
        self.stream_id = stream_id
        self.session_id = session_id
        # Which browser asked for this turn. Used ONLY to tell a double-click
        # apart from a second person typing into the same conversation; anyone
        # may still READ this stream, which is what makes "go back to the
        # conversation that is replying" work for whoever is looking.
        self.client_id = client_id
        self.seq = 0
        self.events: deque[tuple[int, dict]] = deque(maxlen=BACKLOG_EVENTS)
        # Events evicted by the cap. A reader asking for a seq older than the
        # window gets told the gap exists instead of receiving a transcript with
        # a silent hole in it.
        self.dropped = 0
        self.running = True
        # Set before a deliberate restart, so the death it causes is not reported
        # as a failure: the reader pressed Stop and must be told it stopped.
        self.stopped = False
        self.finished_at: float | None = None
        self.error: str | None = None
        # One shared Event, replaced on each emit: cheaper than per-subscriber
        # queues and correct because subscribers re-read from their own seq.
        self._bell = asyncio.Event()

    def emit(self, kind: str, **data) -> None:
        self.seq += 1
        if len(self.events) == self.events.maxlen:
            self.dropped += 1
        # Every event says which conversation it belongs to. The reader can be
        # looking somewhere else — that is the point of browsing mid-turn — and
        # without this the client paints one session's tokens into whatever
        # transcript happens to be on screen.
        self.events.append((self.seq, {"kind": kind, "seq": self.seq,
                                       "session": self.session_id, **data}))
        self._bell.set()
        self._bell = asyncio.Event()

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self.finished_at = time.time()
        # Emit BEFORE lowering `running`. A reader's loop drains, then tests
        # `running`; lowering it first opens a window where the reader breaks out
        # with the terminal event still unread, and the browser is left believing
        # the turn is live until EventSource reconnects on its own schedule.
        self.emit("end", error=error)
        self.running = False

    def after(self, seq: int) -> list[dict]:
        return [e for s, e in self.events if s > seq]

    def gap_before(self, seq: int) -> bool:
        """Did the cap evict something this reader still needs?"""
        if not self.dropped or not self.events:
            return False
        return seq < self.events[0][0] - 1

    def bell(self) -> asyncio.Event:
        """The wakeup that is current RIGHT NOW.

        A reader must take this before reading the backlog and await the object
        it took, not `self._bell` at park time: `emit()` sets the current bell
        and then replaces it, so re-reading the attribute later can hand back a
        fresh, never-to-be-set Event while the wakeup it was meant to catch has
        already gone off."""
        return self._bell

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

    def __init__(self, approval_id: str, stream_id: str, params: dict,
                 session_id: str | None = None) -> None:
        self.id = approval_id
        self.stream_id = stream_id
        # Which conversation is asking. The pending list is filtered on it:
        # unfiltered, a second person's poll showed them a prompt from a
        # conversation they were not even looking at, and let them answer it.
        # Scoping to the SESSION rather than to the browser is deliberate --
        # whoever is reading a shared conversation should be able to answer it.
        self.session_id = session_id
        self.params = params
        # get_running_loop, not get_event_loop: this must belong to the loop
        # that will await it, and only the running one is guaranteed to be that.
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.created_at = time.time()

    def brief(self) -> dict:
        p = self.params
        return {
            "id": self.id,
            "streamId": self.stream_id,
            "session": self.session_id,
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
        # Turns in flight, keyed by SESSION. Both used to be single-valued
        # (`turn_task`, `current`), which is what made "one turn at a time"
        # a property of this server rather than of anything underneath it.
        self.turns: dict[str, asyncio.Task] = {}
        self.live: dict[str, TurnStream] = {}
        # Per browser, both of them. Shared, these leaked one person's position
        # into another's page: a fresh load adopted whatever session someone
        # else had just prompted, and two readers opening different
        # conversations within the debounce window cancelled each other's
        # warm-up. Neither is about the agent; both are about where a PERSON is.
        self.last_session: dict[str, str] = {}
        # The conversation each browser most recently opened, and the warming
        # tasks watching them. The tasks are held because asyncio references a
        # task only weakly; a collected one warms nothing and says nothing.
        self.viewing: dict[str, str] = {}
        self.warming: set[asyncio.Task] = set()
        self.gc: asyncio.Task | None = None

    def running(self) -> dict[str, asyncio.Task]:
        """Turns actually still running, reaped of the ones that have finished."""
        for sid, t in [(k, v) for k, v in self.turns.items() if v.done()]:
            self.turns.pop(sid, None)
            self.live.pop(sid, None)
        return self.turns


# --------------------------------------------------------------------------- #
# hermes ACP — one warm process, MULTIPLEXED by sessionId over JSON-RPC on stdio #
# --------------------------------------------------------------------------- #
# Cold-spawning `hermes chat` per turn costs ~10 s of startup each time, so one
# `hermes acp` process is kept alive: initialize -> session/new (per session) ->
# session/prompt (per turn, streaming).
#
# One process does NOT mean one session. ACP addresses every session-scoped
# message by `sessionId` -- `session/prompt`, `session/update`, `session/cancel`,
# and `session/request_permission`, where the field is required rather than
# optional -- and the SDK dispatches each incoming request onto its own task
# rather than serialising them, so two prompts can be in flight on one pipe.
# hermes' adapter is written for it too: `SessionManager._sessions` is a dict,
# and its own comments talk about isolating writes from "other concurrent ACP
# sessions" on a `ThreadPoolExecutor(max_workers=4)`.
#
# This client used to hold ONE `session_id` and ONE pair of callbacks, so
# `session/load` read as "move the agent" and a second turn had nowhere to
# deliver. Both are now keyed by session, which is what the wire always was.
class HermesACP:
    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        # The session-store bridge, held open. See `_bridge`.
        self._bridge_proc: asyncio.subprocess.Process | None = None
        self._bridge_lock = asyncio.Lock()
        # Sessions THIS PROCESS can be prompted about. hermes' adapter keeps its
        # session map in memory, so a session that exists in the store is
        # unknown to a freshly spawned `hermes acp` until it is loaded, and
        # prompting an unknown id fails. This is what makes the load lazy and
        # once per process, instead of once per switch as it used to be.
        self._known: set[str] = set()
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        # Created lazily, inside the loop that will use them. On Python 3.9 an
        # asyncio.Lock BINDS to the running loop at construction time, and this
        # object is built in `build_app()` -- before `run_app` makes the loop it
        # will actually serve on. The mismatch surfaced only on the first path
        # that took the lock after a restart: "got Future attached to a
        # different loop", reported as a session that would not load.
        self._write_lock: asyncio.Lock | None = None
        self._start_lock: asyncio.Lock | None = None
        # Sessions already steered with CHAT_DIRECTIVE. A loaded session is
        # marked on load so the directive is never injected mid-conversation.
        self._directive_sent: set[str] = set()
        # Per-session callbacks, NOT one pair for "the current session". The read
        # loop routes on the `sessionId` every session-scoped frame carries; with
        # a single pair, two concurrent turns delivered into whichever registered
        # last, and a permission prompt was answered by the wrong conversation.
        self._on_update: dict[str, object] = {}      # sid -> async fn(update)
        self._on_permission: dict[str, object] = {}  # sid -> async fn(params)
        # Strong refs for the detached permission replies. asyncio keeps only a
        # weak reference to a task, so a bare create_task may be collected
        # mid-flight and the agent then waits on an answer nobody is writing.
        self._perm_tasks: set[asyncio.Task] = set()

    def _wl(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    def _sl(self) -> asyncio.Lock:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        return self._start_lock

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def ensure_proc(self) -> None:
        async with self._sl():
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

    async def create_session(self) -> str:
        """Create a session and return its id. Called at SEND time, never earlier.

        Creating one on a page open is what littered the store with titleless
        zero-message ghosts — one per open, per restart — because most never got
        a message. So the caller arms the intent and this runs when there is
        actually something to say.

        It returns the id rather than storing it: with sessions multiplexed there
        is no "the" session for this object to be at.
        """
        await self.ensure_proc()
        res = await self._request(
            "session/new", {"cwd": WORKDIR, "mcpServers": _acp_mcp_servers()}
        )
        sid = (res or {}).get("sessionId")
        if not sid:
            raise RuntimeError("session/new returned no sessionId")
        self._known.add(sid)
        log.info("hermes acp session=%s created", sid)
        return sid

    async def ensure_known(self, sid: str) -> None:
        """Make `sid` promptable, loading it once if this process has not seen it.

        `session/load` is not cheap — hermes re-registers the MCP server and
        rebuilds the whole tool surface on every one, about a second regardless
        of transcript length — which is exactly why this is memoised. It used to
        run on every switch because the client had to MOVE its single session
        pointer; now it runs once per session per process, to introduce it.
        """
        if sid in self._known:
            return
        await self.load_session(sid)

    async def _bridge(self, req: dict) -> object:
        """One request to the long-lived `hermes_session_api serve` process.

        Held open because the alternative is measured: a one-shot invocation is
        ~310 ms, of which ~270 ms is `import hermes_state`, and a session switch
        pays it twice. Kept here rather than in the ACP client because it is the
        same separation for the same reason — hermes' venv, not ours.

        A dead or wedged bridge falls back to spawning one, so the slow path is
        the old behaviour rather than an error. The lock is what makes a single
        pipe safe: replies are matched by ORDER, so two callers interleaving
        their writes would each read the other's answer.
        """
        async with self._bridge_lock:
            for attempt in (1, 2):
                try:
                    if self._bridge_proc is None or self._bridge_proc.returncode is not None:
                        self._bridge_proc = await asyncio.create_subprocess_exec(
                            HERMES_PY, HERMES_SESSION_API, "serve",
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.DEVNULL,
                            limit=ACP_LINE_LIMIT,
                        )
                        hello = await self._bridge_proc.stdout.readline()
                        if not hello:
                            raise RuntimeError("bridge died before it said ready")
                    self._bridge_proc.stdin.write((json.dumps(req) + "\n").encode())
                    await self._bridge_proc.stdin.drain()
                    line = await self._bridge_proc.stdout.readline()
                    if not line:
                        raise RuntimeError("bridge closed mid-request")
                    resp = json.loads(line.decode())
                    if "error" in resp:
                        raise RuntimeError(resp["error"])
                    return resp["ok"]
                except Exception:  # noqa: BLE001
                    self._bridge_proc = None
                    if attempt == 2:
                        raise

    async def list_sessions(self) -> list[dict]:
        # hermes' own SessionDB via its venv, NOT ACP `session/list`: that caches
        # in memory, so a session deleted through the CLI kept reappearing.
        try:
            return await self._bridge({"cmd": "list", "limit": 200})
        except Exception:  # noqa: BLE001
            log.warning("session bridge unavailable; falling back to a spawn", exc_info=True)
        proc = await asyncio.create_subprocess_exec(
            HERMES_PY, HERMES_SESSION_API, "list", "--limit", "200",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"session list failed (rc={proc.returncode}): "
                               f"{err.decode(errors='replace')[:400]}")
        return json.loads(out.decode() or "[]")

    async def read_history(self, sid: str) -> list[dict]:
        """One session's transcript, read from the store instead of through ACP.

        Measured on this stack: 2.7 ms here against 1.17 s for a warm ACP
        `session/load`, because hermes re-registers the MCP server and rebuilds
        the agent's whole tool surface on every load regardless of how short the
        transcript is. That factor of ~400 is what made switching conversations
        feel slow, and it is the whole reason this path exists.

        It also used to be a correctness problem, not just a speed one: back
        when the client held a single session pointer, `load_session` relocated
        the agent, so reading another conversation yanked it out from under a
        streaming turn. That half is gone — sessions are addressed by id now —
        but the cost is not, so the read still stays off ACP.
        """
        try:
            return await self._bridge({"cmd": "history", "id": sid})
        except Exception:  # noqa: BLE001
            log.warning("session bridge unavailable; falling back to a spawn", exc_info=True)
        proc = await asyncio.create_subprocess_exec(
            HERMES_PY, HERMES_SESSION_API, "history", "--id", sid,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"session history failed (rc={proc.returncode}): "
                               f"{err.decode(errors='replace')[:400]}")
        return json.loads(out.decode() or "[]")

    async def load_session(self, sid: str, on_update=None) -> None:
        """Introduce `sid` to this process, optionally collecting its replay.

        This no longer MOVES anything. hermes replays the transcript as
        `session/update` notifications carrying this sid, so a caller that wants
        them passes a sink; a caller that only needs the session to become
        promptable passes nothing and the replay is dropped on the floor.
        """
        await self.ensure_proc()
        # Save and restore rather than clear: a turn streaming into this same
        # session would otherwise lose its sink to a concurrent load.
        prev = self._on_update.get(sid)
        if on_update is not None:
            self._on_update[sid] = on_update
        try:
            res = await self._request(
                "session/load",
                {"sessionId": sid, "cwd": WORKDIR, "mcpServers": _acp_mcp_servers()},
            )
            # A session hermes cannot find is reported as SUCCESS with an empty
            # result, not as a JSON-RPC error -- its handler returns None and
            # the SDK serialises that as `{}`. Verified against 0.12 by loading
            # a real id and a bogus one on the same process: the real one comes
            # back carrying `_meta.hermes.sessionProvenance`, the bogus one as
            # `{}` with "session ... not found" in the adapter's log.
            #
            # So the emptiness IS the signal, and it has to be acted on here:
            # letting it through would add the id to `_known` and move the
            # failure to the next prompt, where it reads as the agent losing a
            # conversation it just opened.
            if not res:
                raise RuntimeError(f"hermes acp does not know session {sid}")
        finally:
            if on_update is not None:
                if prev is None:
                    self._on_update.pop(sid, None)
                else:
                    self._on_update[sid] = prev
        self._known.add(sid)
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
        self._known.discard(sid)
        self._on_update.pop(sid, None)
        self._on_permission.pop(sid, None)

    async def _write(self, obj: dict) -> None:
        async with self._wl():
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            await self.proc.stdin.drain()

    async def _request(self, method: str, params: dict):
        self._id += 1
        rid = self._id
        fut = asyncio.get_running_loop().create_future()
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
            # Keep `data`. hermes's generic ACP handler puts the exception's
            # own message there (`data = {"details": str(exc)}`) before
            # `raise err from None` throws the exception itself away, so on the
            # paths that have no logging of their own it is the only place the
            # cause survives at all. `session/new` is one: its stderr log shows
            # the already-converted "RequestError: Internal error" and nothing
            # underneath, so dropping `data` here left a bare "Internal error
            # (code -32603)" with the reason recoverable from neither side.
            # (`load_session` logs with exc_info and does show its cause -- the
            # difference is per-path, not global.) Log the whole error object
            # too, so a shape this code does not read still lands somewhere.
            log.error("acp %s failed, raw error: %s", method,
                      json.dumps(err, ensure_ascii=False)[:2000])
            detail = err.get("data")
            if isinstance(detail, (dict, list)):
                detail = json.dumps(detail, ensure_ascii=False)[:800]
            elif detail is not None:
                detail = str(detail)[:800]
            raise RuntimeError(
                f"hermes acp {method} failed: "
                f"{err.get('message') or err} (code {err.get('code', '?')})"
                + (f": {detail}" if detail else "")
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
                    # Distinguish an ERROR reply from a null RESULT. `result or
                    # {"_error": ...}` conflated them, so hermes answering
                    # `session/load` for a missing session with `result: null`
                    # -- which it does, rather than raising -- surfaced as
                    # "session/load failed: {} (code ?)". Loud, and unreadable.
                    if "error" in msg:
                        fut.set_result({"_error": msg["error"]})
                    else:
                        fut.set_result(msg["result"])
            elif msg.get("method") == "session/update":
                # Drop stragglers from a superseded process.
                if self.proc is proc:
                    params = msg.get("params") or {}
                    # Route by sessionId. Delivering to "the current callback"
                    # is what put one conversation's tokens into another's
                    # transcript the moment two turns overlapped.
                    cb = self._on_update.get(params.get("sessionId"))
                    if cb:
                        try:
                            await cb(params.get("update", {}))
                        except Exception:  # noqa: BLE001
                            log.debug("on_update failed", exc_info=True)
            elif msg.get("method") == "session/request_permission":
                if self.proc is proc:
                    # Not awaited: a permission request parks on a human for up
                    # to APPROVAL_TIMEOUT_SEC, and awaiting it here would stall
                    # the read loop — every other session's tokens included.
                    # Sequential was survivable with one session; it is not now.
                    t = asyncio.create_task(self._reply_permission(msg))
                    self._perm_tasks.add(t)
                    t.add_done_callback(self._perm_tasks.discard)
            elif "id" in msg:  # a server->client request we do not implement
                if self.proc is proc:
                    await self._write({"jsonrpc": "2.0", "id": msg["id"],
                                       "error": {"code": -32601, "message": "unsupported"}})
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("hermes acp exited"))
        pending.clear()
        if self.proc is proc:
            # Nothing this process knew survives it: the adapter's session map
            # died with it, so every id must be re-loaded before it can be
            # prompted again.
            self._known.clear()
            self._on_update.clear()
            self._on_permission.clear()
        log.warning("hermes acp exited (rc=%s%s)", proc.returncode,
                    "" if self.proc is proc else ", superseded")

    async def _reply_permission(self, msg: dict) -> None:
        params = msg.get("params") or {}
        # `sessionId` is REQUIRED on RequestPermissionRequest, not optional, so
        # there is always something to route on — which is what lets two
        # sessions have a prompt outstanding at the same time.
        cb = self._on_permission.get(params.get("sessionId"))
        option_id = None
        if cb:
            try:
                option_id = await cb(params)
            except Exception:  # noqa: BLE001
                option_id = None
        outcome = ({"outcome": "selected", "optionId": option_id} if option_id
                   else {"outcome": "cancelled"})
        await self._write({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})

    async def prompt(self, session_id: str, text: str, on_update, on_permission):
        await self.ensure_known(session_id)
        if session_id not in self._directive_sent:
            # APPEND, never prepend: the session's auto-title comes from the
            # user's real first words, not from the steering block.
            text = text + "\n\n" + CHAT_DIRECTIVE
            self._directive_sent.add(session_id)
        self._on_update[session_id] = on_update
        self._on_permission[session_id] = on_permission
        try:
            return await self._request(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
            )
        finally:
            self._on_update.pop(session_id, None)
            self._on_permission.pop(session_id, None)

    async def cancel(self, session_id: str) -> None:
        if self.alive and session_id:
            await self._write({"jsonrpc": "2.0", "method": "session/cancel",
                               "params": {"sessionId": session_id}})

    async def restart(self) -> None:
        """Kill the process and bring a fresh one up. The hard half of stop."""
        async with self._sl():
            if self.alive:
                old = self.proc
                with contextlib.suppress(ProcessLookupError):
                    old.terminate()
                # WAIT for the old process to actually die before spawning:
                # terminate() alone races, and the dying process's read-loop
                # cleanup then fails the NEW process's in-flight `initialize`
                # with "acp exited". Escalate to SIGKILL -- a stuck tool can
                # ignore SIGTERM, which is the case this path exists for.
                try:
                    await asyncio.wait_for(old.wait(), timeout=5)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        old.kill()
                    await old.wait()
            self.proc = None
            # The replacement process shares nothing with the old one: its
            # session map is empty, so every id has to be introduced again.
            self._known.clear()
            self._on_update.clear()
            self._on_permission.clear()
            self._directive_sent.clear()
            await self._spawn()



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



# A tool card's expandable body: what was asked, and what came back. Bounded --
# a corpus search returns whole passages, and the whole point of the card is
# that the reader can glance at it and open it only if it matters.
# The card's body is CLAMPED by the client, which can reveal the rest -- so this
# is a transport ceiling, not a display budget, and it can be generous. It still
# exists: a search over the whole corpus can return more than anyone will read,
# and an unbounded value would ride every SSE frame and sit in the backlog.
TOOL_DETAIL_MAX = int(os.environ.get("TOOL_DETAIL_MAX", "60000"))

# What the UI shows and what the MODEL sees are different budgets. This one
# only bounds the card; the text the agent actually reasons over comes back
# through ACP untouched, and on a 24 GiB card a few whole passages in the
# prefill context was enough to OOM dsv4's sparse indexer and take the server
# down with it. Bounding the card does NOT bound that -- the fix for the model
# side is `k` on the search and the serving flags, not this number.


def _tool_detail(u: dict) -> tuple[str, int]:
    """Returns `(text, full_len)`. `full_len > len(text)` means the tail was cut
    at the transport ceiling, and the client says so rather than pretending the
    value ended there.

    Best-effort over the ACP shape: `rawInput` (the arguments) plus the text
    in `content` blocks (the output). Truncated head+tail so a long result stays
    readable at both ends rather than being cut off mid-sentence."""
    parts: list[str] = []
    ri = u.get("rawInput")
    if isinstance(ri, dict) and ri:
        try:
            parts.append(json.dumps(ri, ensure_ascii=False, indent=2))
        except (TypeError, ValueError):
            pass
    elif isinstance(ri, str) and ri.strip():
        parts.append(ri)
    out: list[str] = []
    for blk in u.get("content") or []:
        if not isinstance(blk, dict):
            continue
        c = blk.get("content")
        if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
            out.append(c["text"])
        elif blk.get("type") == "diff" and blk.get("path"):
            out.append(f"[diff] {blk['path']}")
    if out:
        parts.append("\n".join(o for o in out if o))
    text = "\n\n".join(p for p in parts if p).strip()
    full = len(text)
    if full > TOOL_DETAIL_MAX:
        # Head only, not head+tail: the client reveals progressively, and a
        # stitched-together middle would make "show more" reveal a seam rather
        # than more of the same value.
        text = text[:TOOL_DETAIL_MAX]
    return text, full


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
    if kind in ("tool_call", "tool_call_update"):
        return ("tool", {
            "id": update.get("toolCallId"),
            # `kind` is ACP's tool CATEGORY (other/read/search), not a name. Using
            # it as a fallback let a tool_call_update -- which carries no title --
            # rename an identified tool to "other" halfway through. An update with
            # no title says nothing about the name, so it sends nothing and the
            # client keeps what it had.
            "title": update.get("title") or ("tool" if kind == "tool_call" else ""),
            "status": update.get("status") or ("pending" if kind == "tool_call" else ""),
            "detail": (_d := _tool_detail(update))[0],
            "detailFull": _d[1],
        })
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
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid JSON"}, status=400)
    text = (body.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "empty message"}, status=400)

    acp = st.acp
    cid = client_id(request)
    want = (body.get("sessionId") or "").strip()
    if not want and not body.get("new"):
        # A caller that names nothing means "carry on where I was". That used to
        # fall out of the single `acp.session_id`; it has to be said now, and it
        # is what keeps a double-click on a BRAND-NEW conversation from opening
        # two of them — the second request has no id to send yet, so without
        # this it reads as a second "start something new".
        want = st.last_session.get(cid, "")
    fresh = bool(body.get("new")) or not want
    running = st.running()

    # Busy is now a question about THIS conversation, not about the server.
    if not fresh and want in running:
        cur = st.live.get(want)
        if cur is not None and cur.client_id == cid:
            # The same browser asking again: a double-click, or a reload that
            # re-submits. Join the turn -- the text is the same text, and
            # starting a rival turn would interleave two agents into one
            # transcript.
            return web.json_response({"streamId": cur.stream_id,
                                      "sessionId": want, "attached": True})
        # A DIFFERENT browser. Same shape, completely different event: someone
        # else typed this, and joining would silently discard what they wrote
        # while showing them a reply to a question they did not ask. That is
        # what this branch used to do to the second person on a shared
        # deployment -- the text was never even read. Refuse, and say so, so the
        # client can put the message back in the composer.
        return web.json_response(
            {"error": "这个会话正在回复中（另一个窗口发起的），消息未发出",
             "taken": True, "sessionId": want,
             "streamId": cur.stream_id if cur else None},
            status=409)

    if len(running) >= MAX_CONCURRENT_TURNS:
        # A backstop, not the normal path — the client is told the number by
        # `/api/sessions` and blocks the composer itself. Refusing is right here
        # because there is no stream of THEIRS to hand back: the old code could
        # only ever return the same conversation's turn, and returning some
        # other session's now would paint a stranger's reply into this one.
        return web.json_response(
            {"error": f"已有 {len(running)} 个会话在回复，达到上限 {MAX_CONCURRENT_TURNS}",
             "busy": True, "running": len(running),
             "maxConcurrent": MAX_CONCURRENT_TURNS},
            status=429)

    # The session id is settled BEFORE the turn starts, in both directions. It
    # used to be learned from the first `session/update` because a new session
    # was created lazily inside `prompt`, and that race is what could relabel a
    # new conversation as the previous one.
    try:
        sid = await acp.create_session() if fresh else want
        if not fresh:
            # Introduce it to this process if it has not been seen. Free after
            # the first time, and no longer "moving" anything.
            await acp.ensure_known(sid)
    except Exception as e:  # noqa: BLE001
        verb = "start a session" if fresh else f"open {want}"
        return web.json_response({"error": f"could not {verb}: {e}"}, status=500)

    st.last_session[cid] = sid
    stream = TurnStream(secrets.token_hex(8), sid, cid)
    st.streams[stream.stream_id] = stream
    st.live[sid] = stream
    stream.emit("user", text=text)

    async def on_update(update: dict) -> None:
        ev = _to_event(update)
        if ev:
            stream.emit(ev[0], **ev[1])

    async def on_permission(params: dict):
        ap = Approval(secrets.token_hex(6), stream.stream_id, params, sid)
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
            res = await acp.prompt(sid, text, on_update, on_permission)
            reason = (res or {}).get("stopReason")
            if reason and reason != "end_turn":
                stream.emit("note", text=f"stopped: {reason}")
            stream.finish()
        except Exception as e:  # noqa: BLE001
            if stream.stopped:
                # The restart that Stop escalated to kills the in-flight request,
                # which surfaces as "hermes acp exited". That is the stop
                # working, not the turn failing, and showing it as an error
                # tells the reader their own action broke something.
                log.info("turn ended by stop")
                stream.finish()
            else:
                log.warning("turn failed: %s", e)
                stream.finish(error=str(e))

    st.turns[sid] = asyncio.create_task(run())
    return web.json_response({"streamId": stream.stream_id,
                              "sessionId": sid, "attached": False})


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
        # `id:` lets the browser resume by itself: EventSource replays the last id
        # it saw as `Last-Event-ID` on reconnect. Without it a reconnect reuses the
        # URL's original `after_seq` and re-delivers the whole turn.
        await resp.write(
            b"id: " + str(obj.get("seq", 0)).encode()
            + b"\ndata: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
        )

    # A browser reconnect carries where it actually got to; the query parameter is
    # only the opening position of a fresh attach.
    resumed = request.headers.get("Last-Event-ID")
    if resumed:
        try:
            cursor = int(resumed)
        except ValueError:
            cursor = -1
        # Fail closed on a cursor we cannot honour. One that is unparseable or
        # AHEAD of what this stream ever emitted (a stale id from a previous
        # process, a foreign stream) must not be adopted: doing so would skip
        # every frame up to it, the terminal event included, and leave the reader
        # stalled on keepalives. Replaying is cheap; a silent hole is not.
        after = cursor if 0 <= cursor <= stream.seq else after

    # Chrome's default reconnect delay is 3s. Everything here is one hop away, so
    # a stall that long is all overhead.
    await resp.write(b"retry: 750\n\n")

    async def _close(r: web.StreamResponse) -> web.StreamResponse:
        with contextlib.suppress(Exception):
            await r.write_eof()
        return r

    if stream.gap_before(after):
        await send({"kind": "gap", "seq": after,
                    "text": "some output was dropped while disconnected"})
    try:
        while True:
            # Snapshot the bell BEFORE reading the backlog. `emit()` sets the
            # current bell and then swaps in a fresh one, so a wakeup that lands
            # between the read below and the park at the bottom would be rung on
            # an Event nobody is holding, and this reader would sleep until the
            # heartbeat with the event already sitting in the deque.
            bell = stream.bell()

            # Drain, then loop back to drain again. `after()` fixes its list
            # before the first write awaits, so a turn that finishes during one
            # of those writes lands its terminal event OUTSIDE this pass. Park
            # only when there is genuinely nothing pending.
            pending = stream.after(after)
            if pending:
                for ev in pending:
                    await send(ev)
                    after = ev["seq"]
                    if ev["kind"] == "end":
                        return await _close(resp)
                continue

            if not stream.running:
                # Nothing pending and the turn is over — and because the branch
                # above returns the moment it writes `end`, reaching here means
                # this reader has been sent no terminal frame. That is the
                # ordinary reload: the client reattaches at the cursor it last
                # saw, so a turn that finished while the page was away leaves it
                # with nothing outstanding. Closing silently would leave the
                # browser believing the turn is live, reconnecting every 750 ms
                # with the composer locked. Every reader leaves this endpoint
                # having been told the turn ended.
                # Re-read before concluding there is nothing left. The list
                # above was fixed before its writes awaited, so a turn that
                # finished during them left its own events behind this pass —
                # the same materialisation window, one branch further down.
                if stream.after(after):
                    continue
                await send({"kind": "end", "seq": stream.seq, "error": stream.error})
                break

            try:
                await asyncio.wait_for(bell.wait(), timeout=SSE_HEARTBEAT_SEC)
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
    """End ONE session's turn: graceful first, restart only if that is safe.

    `session/cancel` alone is not enough. It reaches the agent's loop, which
    notices only between steps -- so while a tool is running (a corpus search, a
    command) the turn keeps going and the button appears to do nothing for as
    long as the tool takes. The escalation is what makes Stop mean stop.

    But the escalation kills the PROCESS, and the process now carries every
    session. So it is conditional: with another turn in flight, restarting to
    stop this one would take a bystander's reply down with it, and the honest
    answer is to report that the cancel was not honoured. That is a worse
    outcome for the person pressing Stop and a much better one for the person
    who did not.
    """
    st = state(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    running = st.running()

    sid = (body.get("sessionId") or "").strip()
    if not sid and body.get("streamId"):
        owner = st.streams.get(body["streamId"])
        sid = owner.session_id if owner else ""
    if not sid:
        # No target named. That was unambiguous when only one turn could exist;
        # now it is only unambiguous when only one happens to be running, and
        # guessing between two would stop the wrong conversation.
        if not running:
            return web.json_response({"ok": True, "already": "idle"})
        if len(running) > 1:
            return web.json_response(
                {"error": "多个会话正在回复，停止请求必须指明 sessionId",
                 "running": list(running)}, status=400)
        sid = next(iter(running))

    task = running.get(sid)
    if task is None:
        return web.json_response({"ok": True, "already": "idle"})
    stream = st.live.get(sid)

    await st.acp.cancel(sid)
    try:
        # `shield` so OUR timeout does not cancel the turn task itself -- the
        # point is to observe whether it winds down, not to tear it down here.
        await asyncio.wait_for(asyncio.shield(task), timeout=CANCEL_GRACE_SEC)
        return web.json_response({"ok": True, "how": "graceful"})
    except asyncio.TimeoutError:
        pass
    except Exception:  # noqa: BLE001
        # The turn ended by failing; that is still ended.
        return web.json_response({"ok": True, "how": "graceful"})

    others = [k for k in st.running() if k != sid]
    if others:
        log.warning("session/cancel not honoured in %.1fs on %s, and %d other "
                    "turn(s) are live — refusing to restart", CANCEL_GRACE_SEC,
                    sid, len(others))
        if stream and stream.running:
            stream.emit("note", text="停止请求已发出，但 agent 仍卡在工具调用里；"
                                     "另有会话正在回复，暂不重启 agent")
        return web.json_response({"ok": False, "how": "unyielding",
                                  "blockedBy": others}, status=409)

    log.warning("session/cancel not honoured in %.1fs — restarting hermes acp",
                CANCEL_GRACE_SEC)
    if stream and stream.running:
        stream.stopped = True
        stream.emit("note", text="取消未被响应，已重启 agent")
    await st.acp.restart()
    if not task.done():
        task.cancel()
    # No re-select. A restart empties the process's session map, and the next
    # prompt re-introduces whatever id it names -- which is what `ensure_known`
    # is for. The old eager reload existed only because the client held a single
    # "where the agent is" pointer that a restart cleared.
    # run() was cancelled before it could finish the stream, so close it here or
    # the client waits forever for an `end` that is never coming.
    if stream and stream.running:
        stream.finish(error=None)
    return web.json_response({"ok": True, "how": "restart"})


# --------------------------------------------------------------------------- #
# Approvals — pushed on the stream, and pollable because a push can be missed   #
# --------------------------------------------------------------------------- #
async def handle_approval_pending(request: web.Request) -> web.Response:
    """Approvals the caller is in a position to answer.

    Filtered by the conversation they say they are looking at. Unfiltered, a
    second person polling saw a prompt raised in a conversation they had never
    opened -- and could answer it, which is not a display bug but someone else
    approving a shell command on your behalf.

    No `session` given means no filter, which is what a probe or an older client
    gets. That is the pre-existing behaviour and it is only reachable by a
    caller that does not say where it is.
    """
    want = (request.query.get("session") or "").strip()
    pending = [a.brief() for a in state(request).approvals.values()
               if not want or a.session_id == want]
    return web.json_response({"pending": pending})


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
def max_concurrent_turns() -> int:
    """How many replies this server can have in flight, across all sessions.

    It used to return 1, and the docstring called that structural — one process,
    one pipe, `session/load` moves it. Only the first of those was true. ACP
    multiplexes: every session-scoped frame carries `sessionId`, the SDK runs
    each incoming request on its own task, and hermes' adapter keeps a dict of
    sessions on a four-worker pool. The limit was this client holding a single
    session pointer and a single pair of callbacks.

    It stays a function so the UI can ASK rather than assume — the frontend used
    to encode the limit as "a turn exists, therefore the composer is blocked",
    which is the conclusion and not the reason.

    See MAX_CONCURRENT_TURNS for where 4 comes from, and for the smaller limit
    underneath it (the model, not the agent).
    """
    return MAX_CONCURRENT_TURNS


async def handle_sessions(request: web.Request) -> web.Response:
    st = state(request)
    acp = st.acp
    try:
        rows = await acp.list_sessions()
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=500)
    # Per session, not one global flag. A reader looking at an idle session
    # while another one streams must see THAT session's state; deriving it from
    # a global is how a busy session's spinner ends up on an idle one. That was
    # already the intent — the flag was just computed from a single global.
    running = st.running()
    for r in rows:
        r["is_streaming"] = r.get("id") in running
    # Every live turn's stream id, so a reader returning to ANY running
    # conversation can reattach and watch it finish rather than seeing a
    # transcript that stops where the committed store does. Singular before,
    # which meant returning to the second one showed a truncated answer.
    streaming = {sid: st.live[sid].stream_id for sid in running if sid in st.live}
    return web.json_response({"sessions": rows,
                              "current": st.last_session.get(client_id(request)),
                              "streaming": streaming,
                              "running": len(running),
                              "maxConcurrent": max_concurrent_turns()})


async def handle_session_new(request: web.Request) -> web.Response:
    """Arm a new conversation. Deliberately creates nothing.

    Kept as an endpoint for the shape of the API, but there is no longer any
    server state to move: a new session is created by the first send, which is
    what stops the store filling with titleless zero-message ghosts.
    """
    return web.json_response({"ok": True, "current": None})


async def handle_session_load(request: web.Request) -> web.Response:
    """Load an existing session through ACP and hand back its replayed history.

    The replay is collected into the RESPONSE rather than pushed on a stream:
    it is a bounded, already-complete transcript, so a plain request/response is
    the honest shape. Streams are for turns, which are neither.

    This no longer switches anything — nothing in this server is "at" a session.
    The UI reads transcripts through `/api/session/{sid}/history`, which never
    touches ACP at all; this endpoint remains for the case where the ACP
    adapter's own rendering of a session is what is wanted.
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


def _warm_after_debounce(st: "State", cid: str, sid: str) -> None:
    """Introduce `sid` to the ACP process, later, if the reader is still there.

    Fire and forget, deliberately: the response this is scheduled from must not
    wait on it. Warming takes 1.3 s and reading the transcript takes 9 ms, so
    awaiting it here would put the cost back on the path it was moved off —
    just at "open" instead of at "send", which is worse, not better.

    The debounce is a plain staleness check rather than a cancellation: a task
    already past its sleep is inside `session/load`, and cancelling THAT would
    abandon a JSON-RPC request mid-flight and leave hermes having done the work
    with nothing recording that it did. Stale tasks return instead.
    """
    if PREFETCH_DEBOUNCE_SEC < 0:
        return
    st.viewing[cid] = sid

    async def warm() -> None:
        await asyncio.sleep(PREFETCH_DEBOUNCE_SEC)
        if st.viewing.get(cid) != sid:
            return          # scrolled past; whoever they landed on gets warmed
        try:
            await st.acp.ensure_known(sid)
        except Exception:  # noqa: BLE001
            # Speculative work: a failure here must not surface anywhere. The
            # send that needs this session will try again and report properly.
            log.debug("warming %s failed", sid, exc_info=True)

    t = asyncio.create_task(warm())
    st.warming.add(t)
    t.add_done_callback(st.warming.discard)


async def handle_session_history(request: web.Request) -> web.Response:
    """Read a transcript. Never touches ACP, so it is safe mid-turn.

    Paired with the lazy `session/load` in `handle_chat_start`: viewing is free,
    and the session is introduced to the ACP process only when it is needed.
    What this schedules is the same introduction, moved EARLIER — off the send
    path, where the reader is waiting on it, into the seconds they spend typing.
    Nothing here waits on it.
    """
    sid = (request.match_info.get("sid") or "").strip()
    if not sid:
        return web.json_response({"error": "missing id"}, status=400)
    st = state(request)
    try:
        history = await st.acp.read_history(sid)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=500)
    _warm_after_debounce(st, client_id(request), sid)
    return web.json_response({"ok": True, "id": sid, "history": history})


async def handle_session_delete(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    st = state(request)
    # Stop any pending warm-up for it, for EVERY browser: loading a session that
    # is being deleted is pure waste, and on an unlucky interleaving it would
    # put the id back into `_known` after `delete_session` took it out.
    for cid, viewed in list(st.viewing.items()):
        if viewed == sid:
            st.viewing.pop(cid, None)
    for cid, last in list(st.last_session.items()):
        if last == sid:
            st.last_session.pop(cid, None)
    try:
        await st.acp.delete_session(sid)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"ok": True})


# --------------------------------------------------------------------------- #
# Status / health / index                                                       #
# --------------------------------------------------------------------------- #
async def handle_status(request: web.Request) -> web.Response:
    st = state(request)
    acp = st.acp
    running = st.running()
    return web.json_response({
        "acpAlive": acp.alive,
        "session": st.last_session.get(client_id(request)),
        "mcp": {"name": MEMORY_MCP_NAME, "url": MEMORY_MCP_URL} if MEMORY_MCP_URL else None,
        # A list, because there can be several. It was a single `turn` object
        # for the same reason everything else here was singular.
        "turns": [{"session": sid, "streamId": st.live[sid].stream_id}
                  for sid in running if sid in st.live],
        "maxConcurrent": max_concurrent_turns(),
    })


async def handle_health(_request: web.Request) -> web.Response:
    # Deliberately shallow: it must answer while a turn is running and while the
    # acp process is restarting. A probe that fails during either would restart
    # the pod in exactly the moments the design set out to survive.
    return web.json_response({"ok": True})


async def handle_index(request: web.Request) -> web.StreamResponse:
    if not PAGE_TITLE:
        resp: web.StreamResponse = web.FileResponse(STATIC / "index.html")
    else:
        # Substituted per request rather than at boot: the file is the source of
        # truth, and a served copy that drifts from it is the kind of thing
        # nobody thinks to check.
        html = (STATIC / "index.html").read_text()
        html = re.sub(r"<title>.*?</title>",
                      f"<title>{html_escape(PAGE_TITLE)}</title>", html, count=1)
        resp = web.Response(text=html, content_type="text/html")
    return resp   # the browser id is minted by `client_cookie_middleware`


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


def client_id(request: web.Request) -> str:
    """Which browser this is.

    Every callers gets one from `client_cookie_middleware`, so "" reaches here
    only for a request whose response could not carry a Set-Cookie -- in
    practice the SSE stream, which never comes first. It is worth being precise
    about what "" means: it is ONE shared anonymous identity, not "unknown".
    Two cookie-less callers are indistinguishable and will be treated as the
    same person, which is exactly why the cookie is minted on every entry point
    rather than only on the page load.
    """
    return request.get("cid") or request.cookies.get(CLIENT_COOKIE, "")


def build_app() -> web.Application:
    # Order matters: auth runs first (a 401 must not hand out an id), and the
    # cookie middleware wraps whatever comes back from it.
    app = web.Application(middlewares=[client_cookie_middleware, auth_middleware])
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
    app.router.add_get(r"/api/session/{sid}/history", handle_session_history)
    app.router.add_delete(r"/api/session/{sid}", handle_session_delete)
    # `must-revalidate` with a zero max-age: the browser may keep the file, but
    # it has to ask before using it. aiohttp already sends an ETag, so a
    # revalidation that finds nothing new costs one 304 and no body.
    #
    # Without this the browser applies heuristic freshness and serves a cached
    # stylesheet after a deploy — the page comes back looking like the previous
    # version, which reads as "the change did not ship" rather than as a cache
    # hit. Cost is one conditional request per asset per load, on a UI that
    # loads a handful of small files.
    async def _no_stale_static(_req: web.Request, resp: web.StreamResponse) -> None:
        if _req.path.startswith("/static/"):
            resp.headers.setdefault("Cache-Control", "no-cache, must-revalidate")

    app.on_response_prepare.append(_no_stale_static)
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
