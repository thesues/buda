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
import hashlib
import json
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
        # A live turn's session row does not exist in the store until hermes
        # persists its first message — which happens when the TURN ends
        # (`_persist_session` sits on the exit paths of the conversation loop).
        # A first turn runs tens of seconds on these engines, and for all of it
        # the sidebar had nothing to show; if the turn died before persisting,
        # nothing ever. Synthesize the row from the live stream's own user
        # event; the real row replaces it when the turn ends and loadSessions
        # reads the store again.
        known = {r.get("id") for r in rows}
        for sid, stream_id in running.items():
            if sid in known:
                continue
            stream = manager.stream(stream_id)
            prompt = next(
                (e.get("text", "") for e in (stream.after(0) if stream else [])
                 if e.get("kind") == "user"),
                "",
            )
            first_line = (prompt or "").strip().splitlines()[0] if prompt.strip() else ""
            rows.insert(0, {
                "id": sid,
                "title": first_line[:40] or "新会话",
                "preview": "回复中…",
                "messageCount": 0,
                "is_streaming": True,
            })
        return json_response({
            "sessions": rows,
            "current": last_session.get(req.client_id),
            "streaming": running,
            # `running` per endpoint, not a global: the composer gates on the
            # picker's CHOICE, and one full endpoint must not grey out a send
            # aimed at another one. `running_on` reads the stream's endpoint_key,
            # so the count is of turns still going, same as the server refuses on.
            "endpoints": [
                {**e.as_json(), "running": manager.running_on(e.key)}
                for e in endpoints
            ],
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

    @app.route("POST", "/api/session/delete")
    def _delete(req: Request) -> Response:
        """Delete a conversation, through hermes' own `SessionDB.delete_session`.

        In-process, not via the `hermes sessions delete` CLI in a subprocess.
        The subprocess cost ~1.2 s — most of it importing the hermes package
        into a fresh interpreter — and ran the SAME method at the end:
        `hermes_cli.main` does exactly
        `SessionDB().delete_session(sid, sessions_dir=get_hermes_home()/"sessions")`.
        This process already holds a `SessionDB` (`_Db.get()`) inside hermes'
        own interpreter, so the call is the same code with none of the
        shipping.

        Not a hand-rolled DELETE — the reason the CLI was consulted originally.
        A session spans several tables and on-disk transcript files; the method
        orphans child sessions, deletes messages and the row, removes the
        files, and the `messages_fts_*` sqlite triggers keep the search index
        consistent no matter who issues the DELETE.

        POST, not DELETE-with-a-path-id: the router matches `(method, path)`
        exactly and has no parameter support, so `/api/session/<id>` can never
        be a route here. The client used to send exactly that and swallow the
        404, which is why deleting appeared to work and the row came back.
        """
        sid = (req.json().get("sessionId") or "").strip()
        if not sid:
            return json_response({"error": "sessionId is required"}, status=400)
        # Deleting a conversation that is mid-reply would leave the turn
        # writing into a store row that no longer exists. `running()` is
        # session_id -> stream_id for exactly the turns still going.
        if sid in manager.running():
            return json_response(
                {"error": "这个会话正在回复中，先停止再删除"}, status=409,
            )
        try:
            from hermes_agent import _Db, hermes_home

            deleted = _Db.get().delete_session(
                sid, sessions_dir=hermes_home() / "sessions"
            )
        except Exception as e:  # noqa: BLE001 -- a failed delete must answer, not 500
            log.exception("could not delete session %s", sid)
            return json_response({"error": f"删除失败: {e}"}, status=502)
        # Idempotent, like the CLI it replaced: `hermes sessions delete` on an
        # already-gone id printed "not found" and exited 0. A second tab's
        # delete racing the first's must read as success — the goal is achieved.
        last_session.pop(req.client_id, None)
        return json_response({"ok": True, "deleted": sid, "found": bool(deleted)})

    # ── approvals ──────────────────────────────────────────────────────────
    #
    # The state is hermes' own (`tools.approval._pending`), not a second map
    # kept here. hermes-webui does the same, and for the same reason: the agent
    # thread blocks on that module, so a copy on this side would be a thing to
    # keep in sync with the thing that actually decides.

    def _approval_state():
        from tools import approval as ap  # noqa: PLC0415

        return ap

    @app.route("GET", "/api/approval/pending")
    def _perm_pending(req: Request) -> Response:
        """What this conversation is waiting on.

        Read from hermes' gateway queue, which is keyed by session and lives in
        `tools.approval` — the same place the blocked turn is parked. Scoped by
        session on purpose: unscoped, this showed one reader a prompt raised in
        a conversation they had never opened, and let them answer it.
        """
        sid = req.query.get("session", "")
        if not sid:
            return json_response({"pending": []})
        try:
            from tools import approval as ap  # noqa: PLC0415
            # A rotated turn still queues under the id it started with.
            key = manager.approval_key_for(sid)
            with ap._lock:
                entries = list(ap._gateway_queues.get(key) or [])
        except Exception:  # noqa: BLE001 -- no approval module, nothing pending
            return json_response({"pending": []})

        out = []
        for e in entries[:1]:          # one card at a time; the queue is FIFO
            data = getattr(e, "data", None) or getattr(e, "approval_data", None) or {}
            if not isinstance(data, dict):
                data = {}
            title = str(data.get("description") or data.get("command") or "需要确认")
            opts = [{"optionId": "once", "name": "允许一次"},
                    {"optionId": "session", "name": "本次会话都允许"}]
            if data.get("allow_permanent", True):
                opts.append({"optionId": "always", "name": "始终允许"})
            opts.append({"optionId": "deny", "name": "拒绝"})
            # The id IS the session key: that is what `resolve_gateway_approval`
            # takes, and the queue is per-conversation FIFO.
            out.append({"id": key, "title": title,
                        "command": str(data.get("command") or ""), "options": opts})
        return json_response({"pending": out})

    @app.route("POST", "/api/approval/answer")
    def _perm_answer(req: Request) -> Response:
        """Hand one choice back to the blocked turn.

        `resolve_gateway_approval` unblocks it from whichever thread this
        handler happens to run on — which is the whole reason this path is used
        instead of `terminal_tool.set_approval_callback`, whose slot is
        thread-local and therefore invisible here.
        """
        body = req.json()
        sid = str(body.get("id") or body.get("sessionId") or "")
        choice = str(body.get("optionId") or "deny")
        if choice not in ("once", "session", "always", "deny"):
            return json_response({"error": f"unknown choice: {choice}"}, status=400)
        if not sid:
            return json_response({"error": "id is required"}, status=400)
        try:
            from tools import approval as ap  # noqa: PLC0415
            n = ap.resolve_gateway_approval(sid, choice)
        except Exception as e:  # noqa: BLE001
            log.exception("could not resolve the approval")
            return json_response({"error": str(e)}, status=503)
        # Nothing pending is not an error: two tabs can both show the card and
        # both press a button, and the second one is simply late.
        return json_response({"ok": True, "resolved": n, "stale": n == 0})

    # ── the page ───────────────────────────────────────────────────────────

    @app.route("GET", "/")
    def _index(req: Request) -> Response:
        try:
            body = index_html.read_bytes()
        except OSError:
            return json_response({"error": "index missing"}, status=500)
        # Same contract as the static handler: no-cache + ETag, so a deploy's
        # new bundle cannot be trapped behind a heuristically-cached index.
        #
        # Version every static URL with the page's own hash. The no-cache
        # header only helps a browser that ASKS again — a copy cached BEFORE
        # any validator existed sits heuristically fresh for hours and never
        # revalidates, which is exactly how a fixed bug kept "not working".
        # A changed URL cannot be served from any cache, by construction.
        #
        # The hash covers the ASSETS too, not only this page. Hashed alone, a
        # deploy that changed just app.js — most of them — kept the same `?v=`,
        # so the URL promised a new bundle and could still be served the old.
        assets = (b"app.js", b"style.css", b"vendor/marked.min.js", b"vendor/purify.min.js")
        h = hashlib.sha256(body)
        for asset in assets:
            try:
                h.update((static_dir / asset.decode()).read_bytes())
            except OSError:
                pass   # a missing asset 404s on its own; it must not 500 the page
        version = h.hexdigest()[:12]
        for asset in assets:
            body = body.replace(
                b"/static/" + asset, b"/static/" + asset + b"?v=" + version.encode()
            )
        body = body.replace(b"@@BUILD@@", version.encode())
        etag = f'"{version}"'
        if req.headers.get("If-None-Match") == etag:
            return Response(304, [("ETag", etag), ("Cache-Control", "no-cache")])
        return Response(200, [("Content-Type", "text/html; charset=utf-8"),
                              ("ETag", etag), ("Cache-Control", "no-cache")], body)

    return app
