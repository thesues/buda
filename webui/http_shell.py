"""The HTTP layer: `ThreadingHTTPServer`, stdlib only.

## Why not aiohttp any more

Two reasons, and the second is the one that decides it.

The agent runs in this process now and `run_conversation` BLOCKS, so on an event
loop every turn would have to be handed to an executor and every callback
handed back across `call_soon_threadsafe` — a thread boundary in the middle of
the hottest path, for a server whose concurrency is a handful of turns. A
thread-per-request server has the agent, its callbacks and the SSE writer all on
one thread, and the boundary disappears.

The deciding reason: importing `run_agent` means running inside HERMES' venv,
and hermes' own session bridge exists precisely because "the two venvs share
most of their packages and disagree on some". aiohttp was this server's ONLY
third-party import. Without it there is nothing left to disagree about.

## What this module is not

No routing framework, no middleware stack — a dict of handlers and two checks
that run before them. The handlers themselves live where their logic does; this
file is the transport.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import mimetypes
import secrets
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("buda.http")

# Tells one BROWSER from another. Not authentication — everyone here shares one
# credential; this only separates two people's cursors and double-click
# detection. A cookie rather than a header because `EventSource` cannot set
# headers, and a cookie rides every request including the SSE one.
CLIENT_COOKIE = "buda_cid"


class Request:
    """What a handler is given. Deliberately small."""

    __slots__ = ("method", "path", "query", "headers", "body", "client_id", "_h")

    def __init__(self, h: BaseHTTPRequestHandler, client_id: str) -> None:
        parsed = urlparse(h.path)
        self.method = h.command
        self.path = parsed.path
        self.query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self.headers = h.headers
        self.client_id = client_id
        self.body = b""
        self._h = h
        if h.command in ("POST", "PUT", "PATCH"):
            try:
                n = int(h.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            # Cap it: this endpoint takes prompts, not uploads, and an unbounded
            # read is a way to make the server hold memory on request.
            self.body = h.rfile.read(min(n, 8 << 20)) if n > 0 else b""

    def json(self) -> dict:
        try:
            return json.loads(self.body or b"{}")
        except Exception:  # noqa: BLE001
            return {}


class Response:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int = 200, headers: list | None = None, body: bytes = b"") -> None:
        self.status = status
        self.headers = headers or []
        self.body = body


def json_response(obj: Any, status: int = 200) -> Response:
    body = json.dumps(obj, ensure_ascii=False).encode()
    return Response(status, [("Content-Type", "application/json; charset=utf-8")], body)


class Streaming(Response):
    """A response the handler writes itself.

    `pump(write)` is called with a function that puts bytes on the wire
    immediately — the point of SSE is that the reader sees the first token
    before the last one exists, so nothing here may buffer to the end.
    """

    __slots__ = ("pump",)

    def __init__(self, headers: list, pump: Callable[[Callable[[bytes], None]], None]) -> None:
        super().__init__(200, headers, b"")
        self.pump = pump


class App:
    def __init__(
        self,
        static_dir: Path | None = None,
        auth_user: str = "",
        auth_pass: str = "",
    ) -> None:
        self.routes: dict[tuple[str, str], Callable[[Request], Response]] = {}
        self.static_dir = static_dir
        self.auth_user = auth_user
        self.auth_pass = auth_pass

    def route(self, method: str, path: str):
        def deco(fn):
            self.routes[(method, path)] = fn
            return fn

        return deco

    # ── the two checks that run before any handler ─────────────────────────

    def authorised(self, headers) -> bool:
        """Basic auth, both halves compared in constant time.

        A plain `==` on the USER leaks its length through timing just as surely
        as one on the password.
        """
        if not self.auth_user:
            return True
        hdr = headers.get("Authorization", "")
        if not hdr.startswith("Basic "):
            return False
        try:
            user, _, passwd = base64.b64decode(hdr[6:]).decode().partition(":")
        except Exception:  # noqa: BLE001
            return False
        return hmac.compare_digest(user, self.auth_user) and hmac.compare_digest(
            passwd, self.auth_pass
        )

    @staticmethod
    def client_id_for(cookie_header: str | None) -> tuple[str, bool]:
        """`(id, is_fresh)` for this caller.

        Minted on WHATEVER the caller asked for first, not only on `/`: anything
        reaching an API endpoint without having loaded the page — a second tab
        restored from history, a cleared cookie, curl — would otherwise share
        the same empty id, and two of them read as one person, which is the bug
        this mechanism exists to prevent.

        The id is decided BEFORE the handler runs, so a caller whose very first
        request is an API call is attributed to the id it is about to receive.
        """
        if cookie_header:
            for part in cookie_header.split(";"):
                name, _, value = part.strip().partition("=")
                if name == CLIENT_COOKIE and value:
                    return value, False
        return secrets.token_hex(8), True

    # ── dispatch ───────────────────────────────────────────────────────────

    def handle(self, req: Request) -> Response:
        fn = self.routes.get((req.method, req.path))
        if fn is not None:
            return fn(req)
        if req.method == "GET" and self.static_dir is not None:
            static = self.serve_static(req.path)
            if static is not None:
                return static
        return json_response({"error": "not found"}, status=404)

    def serve_static(self, path: str) -> Response | None:
        """Files under `static_dir`, and nothing else.

        The traversal guard is `resolve()` + `is_relative_to`, not a scan for
        "..": a symlink inside the directory reaches outside it without the
        string ever containing one.
        """
        if self.static_dir is None:
            return None
        rel = path.lstrip("/")
        if not rel:
            return None
        try:
            target = (self.static_dir / rel).resolve()
            if not target.is_relative_to(self.static_dir.resolve()) or not target.is_file():
                return None
            body = target.read_bytes()
        except (OSError, ValueError):
            return None
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        return Response(200, [("Content-Type", ctype)], body)


class _Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _serve(self) -> None:
        # The health endpoint stays open so a k8s probe needs no credential, and
        # the static assets are useless without the API behind them.
        if self.path.split("?")[0] != "/healthz" and not self.app.authorised(self.headers):
            body = b"unauthorized"
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="buda"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        cid, fresh = self.app.client_id_for(self.headers.get("Cookie"))
        try:
            req = Request(self, cid)
            resp = self.app.handle(req)
        except Exception:  # noqa: BLE001
            log.exception("handler raised for %s", self.path)
            resp = json_response({"error": "internal error"}, status=500)

        cookie = (
            [(
                "Set-Cookie",
                f"{CLIENT_COOKIE}={cid}; Max-Age=31536000; Path=/; HttpOnly; SameSite=Lax",
            )]
            if fresh
            else []
        )

        if isinstance(resp, Streaming):
            self.send_response(resp.status)
            for k, v in list(resp.headers) + cookie:
                self.send_header(k, v)
            # Chunked, because the length is unknowable and HTTP/1.1 keep-alive
            # would otherwise wait for a Content-Length that never comes.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write(b: bytes) -> None:
                if not b:
                    return
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()

            try:
                resp.pump(write)
            except (BrokenPipeError, ConnectionResetError):
                # The reader went away. Normal; the turn is unaffected.
                return
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(resp.status)
        for k, v in list(resp.headers) + cookie:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp.body)))
        self.end_headers()
        if resp.body:
            self.wfile.write(resp.body)

    do_GET = _serve
    do_POST = _serve
    do_DELETE = _serve


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # Restarting after a deploy must not fail on a socket still in TIME_WAIT.
    allow_reuse_address = True


def serve(app: App, host: str, port: int) -> Server:
    handler = type("BoundHandler", (_Handler,), {"app": app})
    srv = Server((host, port), handler)
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()
    return srv
