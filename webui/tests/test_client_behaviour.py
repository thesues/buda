"""Client behaviour — what `static/app.js` does, independent of any transport.

Kept when `server.py` went away with the ACP subprocess: none of these touch it.
They read the client source (or run it under node) and pin behaviour the
refactor did not change. Several are regressions with a name — the prompt
painted twice, a foreign event drawn into the wrong transcript, a refused send
that ate the message, a stop control that belonged to the wrong session.
"""

from __future__ import annotations
import asyncio
import json
import re
import sys
from pathlib import Path
import pytest
from aiohttp import web

def turn(app, sid: str | None = None) -> asyncio.Task:
    """The task behind a running turn.

    Most tests drive one conversation, so the id is optional — but it is an
    assertion, not a shrug: if a test has somehow started two turns, saying
    which one it meant is the point.
    """
    turns = app["state"].turns
    if sid is not None:
        return turns[sid]
    assert len(turns) == 1, f"expected exactly one turn, got {list(turns)}"
    return next(iter(turns.values()))
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
async def _collect(sink: list, update: dict) -> None:
    sink.append(update)

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

def test_the_prompt_is_painted_once():
    """The prompt showed up twice in the transcript.

    Three painters draw that row -- the optimistic echo in send(), the stream's
    own `user` event, and the `history_user` replayed when hermes reloads the
    session -- and skipUserEcho is one boolean, so it cancels one of them. Any
    other pairing renders the prompt twice.

    Skipped rather than failed without node: this pins client behaviour, and a
    missing runtime is not a broken client.
    """
    import shutil, subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = Path(__file__).parent / "js" / "duplicate_user_row.mjs"
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-400:]

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

def test_opening_a_session_is_not_gated_on_a_running_turn():
    """A reader may look wherever they like while a turn streams.

    Ablation: put the `S.busy` early return back into `openSession` and this
    goes red. The guard existed only because opening a session moved the agent.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function openSession("):]
    body = body[:body.index("\nasync function ")]
    # Reading `S.busy` is fine and now necessary — the function reattaches to a
    # live turn and reports it honestly. What must not come back is the early
    # RETURN that refused to open the session at all.
    guard = re.search(r"if\s*\(\s*S\.busy\s*\)\s*\{[^}]*\breturn\b", body)
    assert guard is None, f"openSession still refuses while busy:\n{guard.group(0)}"
    assert "/history" in body, "openSession is not reading the read-only transcript"

def test_a_new_conversation_is_listed_as_soon_as_it_is_asked():
    """Ablation: move `loadSessions()` back to `endTurn` alone and this is red.

    The list used to be refreshed only when a turn ENDED, so a question you just
    asked had no row until the reply landed.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function send()"):]
    # Stop at the next top-level definition of EITHER kind. Slicing only on
    # `function ` ran past the end of send() into a later one that does call
    # `loadSessions`, so the assertion passed with the line removed.
    ends = [body.index(m) for m in ("\nasync function ", "\nfunction ") if m in body]
    body = body[:min(ends)] if ends else body
    assert "loadSessions()" in body, f"send() never refreshes the sidebar:\n{body}"

def test_the_stop_control_belongs_to_the_session_that_is_running():
    """Not to the composer.

    Browsing mid-turn means the composer sits in front of whatever is being
    READ, which is not necessarily what is streaming — so a stop driven by a
    global busy flag offers to stop someone else's turn. It becomes a stop only
    when the reader is looking at the session that owns the turn; from anywhere
    else it is a disabled 发送, and stopping means going to that session.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    busy = src[src.index("function setBusy("):]
    busy = busy[:busy.index("\n}")]
    assert "S.streaming[S.sessionId]" in busy, (
        "the composer's stop is still driven by a global busy flag:\n" + busy
    )

def test_deleting_a_conversation_asks_first():
    """It did not, and the control is invisible until hover.

    `.del` is `opacity:0` until the row is hovered and covers the right 2.4rem
    at full height — the same place a hand lands to click the row. An immediate,
    schema-aware, undoable-by-nothing delete behind an invisible target took
    three conversations out of the store.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function removeSession("):]
    body = body[:body.index("\n}") + 2]
    assert "confirm(" in body, f"delete still fires with no prompt:\n{body}"
    assert body.index("confirm(") < body.index("fetch("), "it asks after deleting"

def test_a_foreign_event_is_not_drawn():
    """Ablation: drop the `mine` check in `apply` and this goes red."""
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("function apply(ev"):]
    body = body[:body.index("\n  switch (ev.kind)")]
    assert "ev.session" in body and "S.sessionId" in body, (
        "apply() draws every event regardless of whose it is:\n" + body
    )

def test_leaving_a_streaming_session_does_not_claim_idle():
    """"就绪" while a reply is still running is a lie the reader acts on.

    Asserts the BEHAVIOUR — readiness is announced conditionally — not the name
    of the flag. This test sat red because it pinned `S.busy ?` while the client
    had moved to `S.blockedElsewhere`, which asks the more precise question:
    THIS view is idle, another conversation is not. The lie it guards against
    was never reintroduced; only the spelling changed.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function newSession()"):]
    body = body[:body.index("\n}") + 2]
    idle = '"就绪"'
    assert idle in body, f"newSession says nothing about readiness:\n{body}"
    line = next(ln for ln in body.splitlines() if idle in ln)
    assert "?" in line and ":" in line, (
        "readiness is announced unconditionally, so a reader who left a running "
        f"turn is told the app is idle:\n{line}"
    )

def test_a_fresh_conversation_looks_different_from_one_with_history():
    """A blank message panel reads as "loading", not as "nothing said yet".

    A new conversation gets its own shape — greeting and composer together in
    the middle of the column — so the state is legible before typing anything.
    Ablation: drop `showFresh()` from `newSession` and this goes red.
    """
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    css = (Path(__file__).resolve().parents[1] / "static" / "style.css").read_text()
    body = src[src.index("async function newSession()"):]
    body = body[:body.index("\n}") + 2]
    assert "showFresh()" in body, f"newSession opens on a blank panel:\n{body}"
    assert ".chat.fresh #composer" in css, "the fresh layout is not distinct"
    # And it must give way the moment anything is said.
    add = src[src.index("function addMsg("):]
    assert "clearFresh()" in add[:add.index("\n}") + 2], "the hero survives the first message"

def test_the_composer_blocks_on_reported_capacity_not_on_a_local_flag():
    """Ablation: go back to `atCapacity = b` and this goes red."""
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("function setBusy("):]
    body = body[:body.index("\n  if (b) {")]
    assert "S.maxConcurrent" in body and "S.running" in body, (
        "the composer still decides the limit for itself:\n" + body
    )

def test_the_page_follows_the_stream_of_the_session_it_is_showing():
    """Ablation: go back to a single `S.streamingStreamId` and returning to the
    SECOND live conversation reattaches to the first one's stream."""
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function openSession("):]
    body = body[:body.index("\nasync function removeSession(")]
    assert "S.streaming[id]" in body, (
        "openSession still reattaches to a single global stream:\n" + body
    )

def test_the_sidebar_keeps_watching_a_turn_this_page_is_not_reading():
    """The page follows one stream — the conversation on screen. A turn running
    anywhere else has no connection to this tab, so nothing would ever tell the
    sidebar it finished."""
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("function watchWhileOthersRun("):]
    body = body[:body.index("\n}") + 2]
    assert "setInterval" in body and "clearInterval" in body, (
        "the watcher never stops, or never starts:\n" + body
    )

def test_the_client_says_which_conversation_it_is_polling_for():
    """Ablation: drop the query and the server has to answer unfiltered."""
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function pollApprovals("):]
    body = body[:body.index("\n}") + 2]
    assert "session=" in body, f"the poll does not say where it is:\n{body}"

def test_a_refused_send_puts_the_message_back():
    """Typed text is the one thing this UI cannot regenerate."""
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    body = src[src.index("async function send()"):]
    ends = [body.index(m) for m in ("\nasync function ", "\nfunction ") if m in body]
    body = body[:min(ends)] if ends else body
    guard = body[body.index("if (j.error)"):]
    guard = guard[:guard.index("input.value = text") + 40]
    assert "j.taken" in guard, (
        "a refused send still eats what was typed:\n" + guard
    )
