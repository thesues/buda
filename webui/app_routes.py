"""The API, wired onto the transport.

Only wiring lives here: every rule it enforces belongs to `turns`,
`hermes_agent` or `sse`, and every one of those is tested on its own. What this
file owns is the JSON shapes the client already speaks, which are kept
unchanged — the point of the refactor is that the browser cannot tell.

Two contracts worth restating because they are easy to break silently:

* `taken` and `busy` are DIFFERENT refusals. `taken` means this conversation is
  already replying and the text was never read by a model, so the client must be
  able to put it back in the composer. `busy` means the endpoint is at capacity.
* Reading a transcript never touches the agent. Under ACP the only way to read
  one was `session/load`, which MOVED the single agent process — so looking at
  another conversation while one streamed yanked the agent out from under the
  turn in flight. Reading goes to the store, so viewing is free.
"""

from __future__ import annotations

import logging
from pathlib import Path

from http_shell import App, Request, Response, Streaming, json_response
from hermes_agent import Endpoint
from sse import SSE_HEADERS, write_stream
from turns import Refused, TurnManager

log = logging.getLogger("buda.routes")


def build_app(
    *,
    manager: TurnManager,
    endpoints: list[Endpoint],
    static_dir: Path,
    index_html: Path,
    auth_user: str = "",
    auth_pass: str = "",
    sessions: object | None = None,
    mcp: dict | None = None,
) -> App:
    app = App(static_dir=static_dir, auth_user=auth_user, auth_pass=auth_pass)
    by_key = {e.key: e for e in endpoints}
    default_ep = endpoints[0]
    # Which conversation each BROWSER last opened. Per browser, not global:
    # shared, it leaked one person's position into another's page.
    last_session: dict[str, str] = {}

    def _endpoint(req: Request, session_hint: str = "") -> Endpoint:
        return by_key.get(req.json().get("endpoint") or req.query.get("endpoint", ""), default_ep)

    # ── health and status ──────────────────────────────────────────────────

    @app.route("GET", "/healthz")
    def _health(req: Request) -> Response:
        return Response(200, [("Content-Type", "text/plain; charset=utf-8")], b"ok")

    @app.route("GET", "/api/status")
    def _status(req: Request) -> Response:
        running = manager.running()
        return json_response({
            "session": last_session.get(req.client_id),
            "mcp": mcp,
            "turns": [{"session": sid, "streamId": st} for sid, st in running.items()],
            "endpoints": [e.as_json() for e in endpoints],
            "defaultEndpoint": default_ep.key,
        })

    # ── chat ───────────────────────────────────────────────────────────────

    @app.route("POST", "/api/chat/start")
    def _start(req: Request) -> Response:
        body = req.json()
        text = (body.get("text") or "").strip()
        if not text:
            return json_response({"error": "empty message"}, status=400)
        session_id = (body.get("sessionId") or "").strip()
        if body.get("new") or not session_id:
            # A conversation is created by hermes on its first turn; there is
            # nothing to allocate here, and pre-creating one is what used to
            # fill the store with titleless ghosts.
            import secrets

            session_id = secrets.token_hex(8)
        endpoint = by_key.get(body.get("endpoint") or "", default_ep)
        try:
            stream = manager.start(
                session_id=session_id, text=text, endpoint=endpoint, client_id=req.client_id
            )
        except Refused as r:
            # 409 for a conversation already replying, 429 for a full endpoint —
            # the client branches on the flag, not the code, but the codes are
            # the honest ones.
            return json_response(r.as_json(), status=409 if r.reason == "taken" else 429)
        last_session[req.client_id] = session_id
        return json_response({
            "streamId": stream.stream_id,
            "sessionId": session_id,
            "endpoint": endpoint.key,
        })

    @app.route("GET", "/api/chat/stream")
    def _stream(req: Request) -> Response:
        stream = manager.stream(req.query.get("stream_id", ""))
        if stream is None:
            # 404 rather than an empty stream: the client must be able to tell
            # "that turn is gone" from "that turn has said nothing yet".
            return json_response({"error": "unknown stream"}, status=404)
        try:
            after = int(req.query.get("after_seq", "0"))
        except ValueError:
            after = 0
        last_id = req.headers.get("Last-Event-ID")
        return Streaming(
            list(SSE_HEADERS),
            lambda write: write_stream(stream, write, after=after, last_event_id=last_id),
        )

    @app.route("GET", "/api/chat/status")
    def _chat_status(req: Request) -> Response:
        stream = manager.stream(req.query.get("stream_id", ""))
        if stream is None:
            return json_response({"known": False})
        return json_response({
            "known": True,
            "running": stream.running,
            "lastSeq": stream.seq,
            "stopped": stream.stopped,
            "error": stream.error,
        })

    @app.route("POST", "/api/chat/cancel")
    def _cancel(req: Request) -> Response:
        return json_response({"ok": manager.cancel(req.json().get("streamId", ""))})

    # ── sessions (store reads; the agent is never moved) ───────────────────

    @app.route("GET", "/api/sessions")
    def _sessions(req: Request) -> Response:
        rows = []
        if sessions is not None:
            try:
                rows = sessions.list_sessions(limit=100, include_empty=False)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 -- an unreadable sidebar must not 500 the app
                log.exception("could not list sessions")
        running = manager.running()
        for r in rows:
            r["is_streaming"] = r.get("id") in running
        return json_response({
            "sessions": rows,
            "current": last_session.get(req.client_id),
            "streaming": running,
            "endpoints": [e.as_json() for e in endpoints],
        })

    @app.route("GET", "/api/session/history")
    def _history(req: Request) -> Response:
        sid = req.query.get("id", "")
        if not sid or sessions is None:
            return json_response({"events": []})
        try:
            return json_response({"events": sessions.history(sid, limit=400)})  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.exception("could not read history for %s", sid)
            return json_response({"events": [], "error": "could not read this conversation"})

    @app.route("POST", "/api/session/open")
    def _open(req: Request) -> Response:
        """Remember where this browser is. Deliberately does nothing else —
        opening a conversation must not disturb one that is replying."""
        sid = (req.json().get("sessionId") or "").strip()
        if sid:
            last_session[req.client_id] = sid
        return json_response({"ok": True, "current": sid or None})

    # ── the page ───────────────────────────────────────────────────────────

    @app.route("GET", "/")
    def _index(req: Request) -> Response:
        try:
            body = index_html.read_bytes()
        except OSError:
            return json_response({"error": "index missing"}, status=500)
        return Response(200, [("Content-Type", "text/html; charset=utf-8")], body)

    return app
