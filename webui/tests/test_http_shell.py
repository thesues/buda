"""The transport's rules, exercised over a real socket.

A real server rather than a mocked handler, because the things that break here
are protocol-level: chunked framing, whether a stream reaches the client before
it ends, and whether a cookie is decided before the handler runs.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from http_shell import App, Response, Streaming, json_response, serve  # noqa: E402


def _client_ids_seen(app: App) -> list:
    return app._seen  # type: ignore[attr-defined]


def _app(tmp_static: Path | None = None, auth=("", "")) -> App:
    app = App(static_dir=tmp_static, auth_user=auth[0], auth_pass=auth[1])
    app._seen = []  # type: ignore[attr-defined]

    @app.route("GET", "/healthz")
    def _health(req):
        return Response(200, [("Content-Type", "text/plain")], b"ok")

    @app.route("GET", "/api/who")
    def _who(req):
        app._seen.append(req.client_id)  # type: ignore[attr-defined]
        return json_response({"cid": req.client_id})

    @app.route("POST", "/api/echo")
    def _echo(req):
        return json_response({"got": req.json(), "q": req.query})

    @app.route("GET", "/api/boom")
    def _boom(req):
        raise RuntimeError("handler exploded")

    return app


@pytest.fixture
def server():
    started = []

    def start(app: App):
        srv = serve(app, "127.0.0.1", 0)
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for s in started:
        s.shutdown()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=5)


# ── identity ────────────────────────────────────────────────────────────────


def test_a_browser_id_is_minted_on_whatever_was_asked_for_first(server):
    """Minting only on `/` was not enough: anything reaching an API endpoint
    without loading the page — a restored tab, a cleared cookie, curl — would
    share the same empty id, and two of them read as one person."""
    app = _app()
    base = server(app)
    r = _get(base + "/api/who")
    cookie = r.headers.get("Set-Cookie") or ""
    assert "buda_cid=" in cookie and "HttpOnly" in cookie
    assert json.loads(r.read())["cid"], "the handler must already know the id"


def test_the_handler_sees_the_id_it_is_about_to_hand_out(server):
    """Decided BEFORE the handler runs. Otherwise a caller's first request is
    anonymous and its second is somebody else — two people, downstream."""
    app = _app()
    base = server(app)
    r = _get(base + "/api/who")
    handed_out = [
        p.split("=", 1)[1]
        for p in (r.headers.get("Set-Cookie") or "").split(";")
        if p.strip().startswith("buda_cid=")
    ][0]
    assert _client_ids_seen(app)[0] == handed_out


def test_an_existing_cookie_is_kept_rather_than_reissued(server):
    app = _app()
    base = server(app)
    r = _get(base + "/api/who", {"Cookie": "buda_cid=abc123"})
    assert r.headers.get("Set-Cookie") is None
    assert json.loads(r.read())["cid"] == "abc123"


# ── auth ────────────────────────────────────────────────────────────────────


def test_the_health_probe_needs_no_credential(server):
    """A k8s probe has none, and a server that fails its own liveness check
    because of auth restarts forever."""
    base = server(_app(auth=("u", "p")))
    assert _get(base + "/healthz").status == 200


def test_an_api_call_without_a_credential_is_challenged(server):
    base = server(_app(auth=("u", "p")))
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/who")
    assert e.value.code == 401
    assert e.value.headers.get("WWW-Authenticate", "").startswith("Basic ")


def test_a_correct_credential_passes(server):
    base = server(_app(auth=("u", "p")))
    tok = base64.b64encode(b"u:p").decode()
    assert _get(base + "/api/who", {"Authorization": f"Basic {tok}"}).status == 200


def test_a_wrong_user_is_refused_like_a_wrong_password(server):
    base = server(_app(auth=("u", "p")))
    for cred in (b"wrong:p", b"u:wrong"):
        tok = base64.b64encode(cred).decode()
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/api/who", {"Authorization": f"Basic {tok}"})
        assert e.value.code == 401


# ── requests and errors ─────────────────────────────────────────────────────


def test_a_json_body_and_query_reach_the_handler(server):
    base = server(_app())
    req = urllib.request.Request(
        base + "/api/echo?a=1",
        data=json.dumps({"text": "hi"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    got = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert got == {"got": {"text": "hi"}, "q": {"a": "1"}}


def test_a_handler_that_raises_becomes_a_500_not_a_dropped_connection(server):
    """A dropped connection is indistinguishable from the server being gone,
    and the client retries into it."""
    base = server(_app())
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/boom")
    assert e.value.code == 500


def test_an_unknown_path_is_a_json_404(server):
    base = server(_app())
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/nothing")
    assert e.value.code == 404


# ── static ──────────────────────────────────────────────────────────────────


def test_static_files_are_served_under_the_static_prefix(server, tmp_path):
    """The prefix is the contract the page is written against.

    `index.html` asks for `/static/style.css`, `/static/app.js` and two vendor
    scripts; the aiohttp server this module replaced mounted them with
    `add_static("/static/", STATIC)`. Serving them at the URL root instead
    404s every one while `/` still returns 200 — the page renders blank and
    unstyled, with a working API behind it and only console errors to say so.
    Ablation: drop the prefix handling from `serve_static` and this goes red
    while every other test in this file stays green, which is exactly how it
    reached a browser.
    """
    (tmp_path / "app.js").write_text("console.log(1)")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "marked.min.js").write_text("//marked")
    base = server(_app(tmp_static=tmp_path))

    r = _get(base + "/static/app.js")
    assert r.read() == b"console.log(1)"
    assert "javascript" in r.headers.get("Content-Type", "")

    # Nested, because the vendor scripts live one level down.
    assert _get(base + "/static/vendor/marked.min.js").read() == b"//marked"


def test_a_file_outside_the_static_prefix_is_not_served(server, tmp_path):
    """Only `/static/` reaches the directory. A bare `/app.js` is not an alias
    for it — one file, one URL, so a cache header or a CDN rule written for the
    prefix cannot be sidestepped by addressing the same bytes another way."""
    (tmp_path / "app.js").write_text("console.log(1)")
    base = server(_app(tmp_static=tmp_path))
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/app.js")
    assert e.value.code == 404


def test_a_path_escaping_the_static_directory_is_refused(server, tmp_path):
    """Resolved, not string-scanned: a symlink inside the directory reaches
    outside it without the path ever containing `..`."""
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("private")
    (tmp_path / "link.txt").symlink_to(secret)
    base = server(_app(tmp_static=tmp_path))
    for path in ("/static/../secret.txt", "/static/link.txt"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + path)
        assert e.value.code == 404, f"{path} must not be served"


# ── streaming ───────────────────────────────────────────────────────────────


def test_a_stream_reaches_the_client_before_it_ends(server):
    """The whole point. If the transport buffered to the end, a reader would
    watch a spinner for the length of the answer."""
    release = threading.Event()
    app = _app()

    @app.route("GET", "/api/stream")
    def _stream(req):
        def pump(write):
            write(b"first\n")
            release.wait(3)
            write(b"last\n")

        return Streaming([("Content-Type", "text/plain")], pump)

    base = server(app)
    r = _get(base + "/api/stream")

    # Read on another thread with a deadline. Asserting only the CONTENT would
    # pass on a transport that buffers to the end: the read simply blocks until
    # the pump finishes and then returns the same bytes. What has to be true is
    # that the first chunk arrives WHILE the pump is still blocked.
    got: list[bytes] = []
    reader = threading.Thread(target=lambda: got.append(r.readline()), daemon=True)
    reader.start()
    reader.join(timeout=1.0)
    assert got == [b"first\n"], (
        "the first chunk must arrive before the last exists; a buffered "
        "transport leaves the reader blocked here"
    )

    release.set()
    assert r.readline() == b"last\n"


def test_a_reader_that_hangs_up_mid_stream_does_not_kill_the_server(server):
    finished = threading.Event()
    app = _app()

    @app.route("GET", "/api/stream")
    def _stream(req):
        def pump(write):
            try:
                for _ in range(2000):
                    write(b"x" * 512)
                    time.sleep(0.001)
            finally:
                finished.set()

        return Streaming([("Content-Type", "text/plain")], pump)

    base = server(app)
    r = _get(base + "/api/stream")
    r.read(64)
    r.close()
    assert finished.wait(5), "the pump must unwind rather than spin"
    # The server is still there afterwards.
    assert _get(base + "/healthz").status == 200


def test_requests_are_served_concurrently(server):
    """One blocked turn must not stop anyone else from loading the page — the
    reason this is a threading server at all."""
    gate = threading.Event()
    app = _app()

    @app.route("GET", "/api/slow")
    def _slow(req):
        gate.wait(3)
        return json_response({"ok": True})

    base = server(app)
    t = threading.Thread(target=lambda: _get(base + "/api/slow").read(), daemon=True)
    t.start()
    time.sleep(0.05)
    assert _get(base + "/healthz").status == 200, "a blocked request must not block the server"
    gate.set()
    t.join(timeout=3)
