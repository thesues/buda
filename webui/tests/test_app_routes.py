"""The API's contracts, over a real socket, with a fake agent behind them."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import hermes_agent as ha  # noqa: E402
from app_routes import build_app  # noqa: E402
from http_shell import serve  # noqa: E402
from turns import TurnManager  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.thinking_callback = None

    def interrupt(self, message=None):
        pass


class FakeSessions:
    def __init__(self):
        self.rows = [{"id": "s-old", "title": "昨天的问题", "messageCount": 4}]
        self.moved = []

    def list_sessions(self, limit, include_empty):
        return [dict(r) for r in self.rows]

    def history(self, sid, limit):
        self.moved.append(sid)
        return [{"kind": "history_user", "text": "before"}]


@pytest.fixture
def app_server(monkeypatch, tmp_path):
    monkeypatch.setattr(ha, "build_agent", lambda session_id, ep: FakeAgent())
    (tmp_path / "index.html").write_text("<html>buda</html>")
    gate = threading.Event()
    gate.set()
    state = {"gate": gate, "sessions": FakeSessions()}

    def run(agent, **kw):
        state["gate"].wait(3)
        agent.stream_delta_callback("hello")
        return {}

    mgr = TurnManager(ha.AgentPool(), history=lambda sid: [], run=run)
    eps = [
        ha.Endpoint("dsv4", "DSV4", "m1", "http://a/v1", max_concurrent=1),
        ha.Endpoint("vision", "Vision", "m2", "http://b/v1", max_concurrent=1),
    ]
    app = build_app(
        manager=mgr,
        endpoints=eps,
        static_dir=tmp_path,
        index_html=tmp_path / "index.html",
        sessions=state["sessions"],
        mcp={"name": "memory", "url": "http://mcp"},
    )
    srv = serve(app, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, mgr, state
    srv.shutdown()


def _post(base, path, obj, cookie=""):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json", **({"Cookie": cookie} if cookie else {})},
    )
    try:
        r = urllib.request.urlopen(req, timeout=5)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base, path, cookie=""):
    req = urllib.request.Request(base + path, headers={"Cookie": cookie} if cookie else {})
    r = urllib.request.urlopen(req, timeout=5)
    return json.loads(r.read())


def _drain(stream, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and stream.running:
        time.sleep(0.005)


# ── sending ─────────────────────────────────────────────────────────────────


def test_a_prompt_starts_a_turn_and_names_its_stream(app_server):
    base, mgr, _ = app_server
    code, body = _post(base, "/api/chat/start", {"text": "hi"})
    assert code == 200 and body["streamId"] and body["sessionId"]
    assert body["endpoint"] == "dsv4", "no endpoint named means the default one"
    _drain(mgr.stream(body["streamId"]))


def test_an_empty_prompt_is_refused_before_anything_starts(app_server):
    base, mgr, _ = app_server
    code, _ = _post(base, "/api/chat/start", {"text": "   "})
    assert code == 400 and mgr.running() == {}


def test_the_endpoint_named_by_the_client_is_the_one_used(app_server):
    """The picker is the whole multi-endpoint feature; if this is ignored the UI
    shows one model's name and another model answers."""
    base, mgr, _ = app_server
    code, body = _post(base, "/api/chat/start", {"text": "hi", "endpoint": "vision"})
    assert code == 200 and body["endpoint"] == "vision"
    _drain(mgr.stream(body["streamId"]))


def test_an_unknown_endpoint_falls_back_rather_than_failing(app_server):
    base, mgr, _ = app_server
    _, body = _post(base, "/api/chat/start", {"text": "hi", "endpoint": "nope"})
    assert body["endpoint"] == "dsv4"
    _drain(mgr.stream(body["streamId"]))


# ── the two refusals are different ──────────────────────────────────────────


def test_a_busy_conversation_is_409_taken_and_a_full_endpoint_is_429_busy(app_server):
    base, mgr, state = app_server
    state["gate"] = threading.Event()  # hold the turn open

    _, first = _post(base, "/api/chat/start", {"text": "one"})
    sid = first["sessionId"]

    code, body = _post(base, "/api/chat/start", {"text": "two", "sessionId": sid})
    assert code == 409 and body.get("taken") is True, "the same conversation"
    assert body["streamId"] == first["streamId"]

    code, body = _post(base, "/api/chat/start", {"text": "three"})
    assert code == 429 and body.get("busy") is True, "a different one, same full endpoint"
    assert body["maxConcurrent"] == 1

    # The other endpoint has its own budget.
    code, _ = _post(base, "/api/chat/start", {"text": "four", "endpoint": "vision"})
    assert code == 200

    state["gate"].set()
    for s in list(mgr.running().values()):
        _drain(mgr.stream(s))


# ── streaming ───────────────────────────────────────────────────────────────


def test_the_stream_carries_the_turn_and_ends(app_server):
    base, mgr, _ = app_server
    _, body = _post(base, "/api/chat/start", {"text": "hi"})
    r = urllib.request.urlopen(base + f"/api/chat/stream?stream_id={body['streamId']}", timeout=5)
    kinds = []
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            kinds.append(ev["kind"])
            if ev["kind"] == "end":
                break
    assert kinds[0] == "user" and "delta" in kinds and kinds[-1] == "end"


def test_an_unknown_stream_is_404_not_an_empty_stream(app_server):
    """The client has to tell "that turn is gone" from "it has said nothing"."""
    base, _, _ = app_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/api/chat/stream?stream_id=nope", timeout=5)
    assert e.value.code == 404


def test_chat_status_reports_a_finished_turn(app_server):
    base, mgr, _ = app_server
    _, body = _post(base, "/api/chat/start", {"text": "hi"})
    _drain(mgr.stream(body["streamId"]))
    st = _get(base, f"/api/chat/status?stream_id={body['streamId']}")
    assert st["known"] is True and st["running"] is False and st["error"] is None
    assert _get(base, "/api/chat/status?stream_id=nope")["known"] is False


# ── reading is free ─────────────────────────────────────────────────────────


def test_reading_a_transcript_does_not_disturb_a_running_turn(app_server):
    """Under ACP the only way to read one was `session/load`, which MOVED the
    single agent process — so looking at another conversation while one streamed
    yanked the agent out from under the turn in flight."""
    base, mgr, state = app_server
    state["gate"] = threading.Event()
    _, live = _post(base, "/api/chat/start", {"text": "one"})

    got = _get(base, "/api/session/history?id=s-old")
    assert got["events"][0]["text"] == "before"
    assert mgr.stream(live["streamId"]).running is True, "the live turn is untouched"

    state["gate"].set()
    _drain(mgr.stream(live["streamId"]))


def test_the_sidebar_marks_which_conversations_are_replying(app_server):
    base, mgr, state = app_server
    state["gate"] = threading.Event()
    _, live = _post(base, "/api/chat/start", {"text": "one"})

    body = _get(base, "/api/sessions")
    assert live["sessionId"] in body["streaming"]
    assert [e["key"] for e in body["endpoints"]] == ["dsv4", "vision"]

    state["gate"].set()
    _drain(mgr.stream(live["streamId"]))


def test_a_sessions_read_that_fails_does_not_take_the_app_down(app_server, monkeypatch):
    base, _, state = app_server
    monkeypatch.setattr(
        state["sessions"], "list_sessions",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("db locked")),
    )
    body = _get(base, "/api/sessions")
    assert body["sessions"] == [], "an unreadable sidebar must not 500 the page"


# ── per-browser position ────────────────────────────────────────────────────


def test_each_browser_keeps_its_own_position(app_server):
    """Shared, this leaked one person's position into another's page."""
    base, _, _ = app_server
    _post(base, "/api/session/open", {"sessionId": "s-a"}, cookie="buda_cid=alice")
    _post(base, "/api/session/open", {"sessionId": "s-b"}, cookie="buda_cid=bob")
    assert _get(base, "/api/status", cookie="buda_cid=alice")["session"] == "s-a"
    assert _get(base, "/api/status", cookie="buda_cid=bob")["session"] == "s-b"


def test_status_advertises_the_endpoints_the_client_can_pick(app_server):
    base, _, _ = app_server
    body = _get(base, "/api/status")
    assert body["defaultEndpoint"] == "dsv4"
    assert {e["key"]: e["maxConcurrent"] for e in body["endpoints"]} == {"dsv4": 1, "vision": 1}
    assert body["mcp"]["name"] == "memory"


# ── media ───────────────────────────────────────────────────────────────────
#
# `media` is stubbed rather than mocked at the autumn boundary: these tests are
# about the route's contract — what it refuses, what status it maps a failure
# to, what headers it sets — and wiring a real cluster in would test neither
# that nor autumn.


class _FakeStore:
    """Stands in for `media`, recording what the route handed it."""

    def __init__(self):
        self.put_calls = []
        self.blobs = {}
        self.fail_put = None
        self.fail_get = None

    def put(self, data, ext, session="shared"):
        self.put_calls.append((len(data), ext, session))
        if self.fail_put:
            raise self.fail_put
        mid = f"{session}/deadbeef.{ext}"
        self.blobs[mid] = (data, f"image/{ext}")
        return mid

    def get(self, media_id):
        if self.fail_get:
            raise self.fail_get
        if media_id not in self.blobs:
            raise KeyError(media_id)
        return self.blobs[media_id]


@pytest.fixture
def media_server(app_server, monkeypatch):
    import app_routes

    base, _mgr, _state = app_server     # the fixture yields a 3-tuple
    store = _FakeStore()
    monkeypatch.setattr(app_routes, "media", store)
    return base, store


def _raw_post(base, path, body: bytes, extra=None):
    req = urllib.request.Request(base + path, data=body, method="POST")
    for k, v in (extra or {}).items():
        req.add_header(k, v)
    try:
        r = urllib.request.urlopen(req, timeout=5)
        return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_an_upload_is_stored_and_answered_with_a_url(media_server):
    base, store = media_server
    code, body, _ = _raw_post(base, "/api/media?ext=png&session=s1", b"\x89PNG fake")
    assert code == 200
    j = json.loads(body)
    assert j["id"] == "s1/deadbeef.png"
    assert j["url"] == "/api/media?id=s1/deadbeef.png"
    assert store.put_calls == [(9, "png", "s1")]


def test_an_upload_over_the_body_limit_is_refused_not_truncated(media_server, monkeypatch):
    """`Request` reads at most 8 MiB and says nothing when it cuts.

    Without this check the route would store a short file that looks fine
    until someone renders it. The limit is lowered here rather than sending a
    real 8 MiB: the server replies without draining the rest of the request, so
    a genuinely oversized POST desynchronises the connection and the test hangs
    on a response that is already written. Same code path, no dead socket.

    Ablation: drop the Content-Length check in `_media_put` and this returns
    200 with the store having been handed the bytes.
    """
    import app_routes

    base, store = media_server
    monkeypatch.setattr(app_routes, "BODY_LIMIT", 100)
    code, body, _ = _raw_post(base, "/api/media?ext=png", b"x" * 200)
    assert code == 413, body
    assert "limit" in json.loads(body)["error"]
    assert store.put_calls == [], "an oversized upload must not reach the store"


def test_an_unsupported_type_is_a_400_and_never_reaches_the_store(media_server):
    base, store = media_server
    store.fail_put = ValueError("unsupported media type: 'exe'")
    code, body, _ = _raw_post(base, "/api/media?ext=exe", b"MZ")
    assert code == 400
    assert "unsupported" in json.loads(body)["error"]


def test_a_store_that_is_down_is_503_not_400(media_server):
    """A caller can fix a 400 by sending something else; a 503 says the file was
    fine and the cluster was not. Collapsing them hides an outage as user error."""
    base, store = media_server
    store.fail_put = RuntimeError("AUTUMN_MANAGER is unset")
    code, body, _ = _raw_post(base, "/api/media?ext=png", b"\x89PNG")
    assert code == 503
    assert "could not store" in json.loads(body)["error"]


def test_a_stored_file_is_served_back_with_its_type_and_cached_hard(media_server):
    base, store = media_server
    _raw_post(base, "/api/media?ext=png&session=s1", b"\x89PNG fake")
    r = urllib.request.urlopen(base + "/api/media?id=s1/deadbeef.png", timeout=5)
    assert r.read() == b"\x89PNG fake"
    assert r.headers.get("Content-Type") == "image/png"
    # The id is minted per upload and its bytes never change, so this is safe —
    # and without it a re-rendered transcript refetches every image.
    assert "immutable" in (r.headers.get("Cache-Control") or "")


def test_fetching_without_an_id_is_400_and_an_unknown_id_is_404(media_server):
    base, _ = media_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/api/media", timeout=5)
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e2:
        urllib.request.urlopen(base + "/api/media?id=s1/nope.png", timeout=5)
    assert e2.value.code == 404
