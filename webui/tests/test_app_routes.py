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
    (tmp_path / "index.html").write_text(
        '<html><link rel="stylesheet" href="/static/style.css">'
        '<script src="/static/app.js"></script></html>'
    )
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


def test_each_endpoint_reports_its_own_running_count(app_server):
    """The client's composer gates on the picker's CHOICE, so the counts must
    be per endpoint: one full endpoint greying out a send aimed at another is
    the single-endpoint behaviour wearing a multi-endpoint hat."""
    base, mgr, state = app_server
    state["gate"] = threading.Event()
    _, live = _post(base, "/api/chat/start", {"text": "one"})

    body = _get(base, "/api/sessions")
    counts = {e["key"]: e["running"] for e in body["endpoints"]}
    assert counts == {"dsv4": 1, "vision": 0}, counts

    state["gate"].set()
    _drain(mgr.stream(live["streamId"]))
    body = _get(base, "/api/sessions")
    assert {e["key"]: e["running"] for e in body["endpoints"]} == {"dsv4": 0, "vision": 0}, (
        "a finished turn must not keep the endpoint marked busy")


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


# ── deleting a session ──────────────────────────────────────────────────────


class FakeDb:
    """The slice of SessionDB the delete route touches."""

    def __init__(self, deleted=True, error=None):
        self.deleted = deleted
        self.error = error
        self.calls = []

    def delete_session(self, sid, sessions_dir=None):
        self.calls.append((sid, sessions_dir))
        if self.error:
            raise self.error
        return self.deleted


def test_delete_runs_in_process_against_session_db(app_server, monkeypatch):
    """The delete must reach `SessionDB.delete_session` — the same method the
    CLI subprocess ran — with the sessions dir, and report ok."""
    base, _, state = app_server
    fake = FakeDb(deleted=True)
    monkeypatch.setattr(ha, "_Db", type("_Db", (), {"get": staticmethod(lambda: fake)}))
    code, body = _post(base, "/api/session/delete", {"sessionId": "s-old"})
    assert (code, body["ok"]) == (200, True)
    assert fake.calls and fake.calls[0][0] == "s-old"
    assert fake.calls[0][1] is not None   # transcripts dir passed through


def test_delete_is_idempotent_when_the_row_is_already_gone(app_server, monkeypatch):
    """An already-deleted id must read as success — a second tab racing the
    first's delete is the normal case, and the old CLI exited 0 on it."""
    base, _, state = app_server
    monkeypatch.setattr(ha, "_Db", type("_Db", (), {"get": staticmethod(lambda: FakeDb(deleted=False))}))
    code, body = _post(base, "/api/session/delete", {"sessionId": "s-old"})
    assert (code, body["ok"], body["found"]) == (200, True, False)


def test_delete_refuses_a_session_that_is_mid_reply(app_server, monkeypatch):
    """A turn writing into a deleted row is corruption the reader cannot see."""
    base, mgr, state = app_server
    monkeypatch.setattr(type(mgr), "running", lambda self: {"s-old": "stream-x"})
    monkeypatch.setattr(ha, "_Db", type("_Db", (), {"get": staticmethod(lambda: FakeDb())}))
    code, body = _post(base, "/api/session/delete", {"sessionId": "s-old"})
    assert code == 409
    assert "先停止再删除" in body["error"]


def test_a_live_turns_session_shows_in_the_sidebar_before_the_store_has_it(app_server):
    """hermes persists a session row only when the turn ENDS, so a first turn
    ran its whole life invisible to /api/sessions — "new sessions appear late,
    sometimes never". The route must synthesize the row from the live stream."""
    base, mgr, state = app_server
    state["gate"].clear()   # keep the turn running
    try:
        code, body = _post(base, "/api/chat/start",
                           {"text": "什么是缘起", "sessionId": "s-live", "endpoint": "dsv4"})
        assert code == 200, body
        rows = _get(base, "/api/sessions")["sessions"]
        row = next((r for r in rows if r["id"] == "s-live"), None)
        assert row is not None, "a running turn must be visible in the sidebar"
        assert row["is_streaming"] is True
        assert "什么是缘起" in row["title"]
    finally:
        state["gate"].set()


def test_the_real_row_shadows_the_synthesized_one(app_server, monkeypatch):
    """Once the store has the row (turn ended), the merge must not duplicate it."""
    base, mgr, state = app_server
    fake = FakeSessions()
    fake.rows.append({"id": "s-old", "title": "昨天的问题", "messageCount": 4})
    monkeypatch.setattr(mgr, "running", lambda: {})   # nothing live
    rows = _get(base, "/api/sessions")["sessions"]
    assert sum(1 for r in rows if r["id"] == "s-old") == 1


def test_index_versioned_the_static_urls(app_server):
    """A heuristically-cached, validator-less app.js cannot be revalidated —
    the no-cache fix itself never reached that browser. A URL that changes
    with every build is the only bust that works by construction."""
    base, _, _ = app_server
    html = urllib.request.urlopen(base + "/").read().decode()
    assert "app.js?v=" in html and "style.css?v=" in html, "statics must be versioned"
    assert "@@BUILD@@" not in html, "the build marker must be injected"
    # and the versioned URL still serves
    import re
    v = re.search(r"app\.js\?v=([0-9a-f]+)", html).group(1)
    urllib.request.urlopen(f"{base}/static/app.js?v={v}")
