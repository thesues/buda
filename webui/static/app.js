"use strict";
/* hermes webui client.
 *
 * Two rules shape this file.
 *
 * 1. A turn belongs to the SERVER, not to this page. The running turn's id and
 *    the last event sequence rendered both live in localStorage, and on load the
 *    page ASKS what happened while it was away. Reattaching is the normal path,
 *    not the error path.
 *
 * 2. A turn is a sequence of SEGMENTS, each either a thought or output. Switching
 *    kind closes the current segment and opens a new one. Rendering the two into
 *    one bubble — which an earlier version did — turns the transcript into the
 *    model's internal monologue glued to its answer, and the reader cannot tell
 *    which is which.
 */

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const LS_STREAM = "hermes.streamId";
const LS_SEQ = "hermes.lastSeq";
const LS_OPEN = "hermes.open";      // which activity groups the reader had open

// Keyed by session + the group's index in the transcript, which is stable for a
// given conversation: reload it, switch away and back, and the rows you opened
// are still open. Upstream persists this per chat and per turn for the same
// reason -- being made to re-open the same trace every visit is what makes a
// disclosure feel like it is fighting you.
function openState() {
  try { return JSON.parse(localStorage.getItem(LS_OPEN) || "{}"); } catch (_) { return {}; }
}
// Keyed by session so one conversation's open groups do not decide another's.
// `S.sessionId` used to be set only by `openSession`, so every conversation
// started from 新会话 — and every boot-time reattach — shared one "-" bucket:
// opening group 0 in one of them pre-opened group 0 in all the others.
function openKey(idx) { return `${S.sessionId || "-"}#${idx}`; }
function isOpen(idx) { return openState()[openKey(idx)] === 1; }
function setOpen(idx, on) {
  try {
    const m = openState();
    if (on) m[openKey(idx)] = 1; else delete m[openKey(idx)];
    // Bounded: one entry per turn per session forever would grow without limit.
    const keys = Object.keys(m);
    if (keys.length > 400) keys.slice(0, keys.length - 400).forEach((k) => delete m[k]);
    localStorage.setItem(LS_OPEN, JSON.stringify(m));
  } catch (_) { /* private mode: the state is a convenience, not a requirement */ }
}

// Open links in a new tab, and never hand the opener over with them.
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A" && node.getAttribute("href")) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});
const renderMD = (src) =>
  DOMPurify.sanitize(marked.parse(src || "", { gfm: true, breaks: true }));

const S = {
  streamId: null,
  lastSeq: 0,
  es: null,
  busy: false,
  seg: null,           // { kind: "think"|"out", body, text, refs }
  tools: new Map(),
  approvalTimer: null,
  awaitingPerm: false,
  skipUserEcho: false,   // we drew this turn's prompt optimistically
  activity: null,        // the current turn's one activity disclosure
  actIndex: 0,           // its position in the transcript, for the open-state key
  sessionId: null,
  switching: null,      // a history read in flight; sending must wait for it
  pendingNew: false,    // "new session" clicked; the ACP move happens on send
  ownStream: null,      // the stream THIS view started, before it has an id
  streamingSession: null,  // which session owns the running turn, if any
  streamingStreamId: null, // and its stream, so returning to it can reattach
  stopping: false,
  startedAt: 0,
  timer: null,
};

/* ---------- just enough state to recover ---------- */
function remember(id, seq) {
  try {
    if (id) localStorage.setItem(LS_STREAM, id);
    if (seq != null) localStorage.setItem(LS_SEQ, String(seq));
  } catch (_) { /* private mode: recovery degrades, chat still works */ }
}
function forget() {
  try { localStorage.removeItem(LS_STREAM); localStorage.removeItem(LS_SEQ); } catch (_) {}
}
function recall() {
  try {
    return { id: localStorage.getItem(LS_STREAM), seq: Number(localStorage.getItem(LS_SEQ) || 0) };
  } catch (_) { return { id: null, seq: 0 }; }
}

/* ---------- chrome ---------- */
function status(text) {
  $("#run-status").textContent = text;
  setPendingText(text);          // the tail row mirrors it, where the eye is
}

const SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
let spinFrame = 0;

function tick() {
  // The spinner is the only thing on screen that says "still working" during a
  // long tool call, where no token arrives for a minute at a time. A static
  // status line reads as a hang.
  const ch = S.busy ? SPIN[spinFrame++ % SPIN.length] : "";
  $("#spin").textContent = ch;
  const ps = $("#pending .p-spin");
  if (ps) ps.textContent = ch;
  // Each running tool times itself from when its row appeared, not from the
  // turn start -- "this search has been going 40s" is the useful number.
  document.querySelectorAll("[data-since]").forEach((n) => {
    const ms = Date.now() - Number(n.dataset.since);
    const sec = Math.floor(ms / 1000);
    n.textContent = sec < 60 ? ` ${sec}s` : ` ${Math.floor(sec / 60)}m${String(sec % 60).padStart(2, "0")}s`;
  });
}

function tickSlow() {
  if (!S.startedAt) { $("#elapsed").textContent = ""; return; }
  const s = Math.floor((Date.now() - S.startedAt) / 1000);
  $("#elapsed").textContent = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

function scroll() {
  const m = $("#messages");
  // Follow the tail only if the reader is already there. Yanking the viewport
  // away from someone reading back is worse than a missed autoscroll.
  if (m.scrollHeight - m.scrollTop - m.clientHeight < 140) m.scrollTop = m.scrollHeight;
}

/* ---------- messages ---------- */
function showFresh() {
  // A brand-new conversation looks nothing like one with history: the greeting
  // and the composer sit together in the middle, so it is obvious at a glance
  // that nothing has been said yet and there is no context behind it.
  const m = $("#messages");
  m.textContent = "";
  const hero = el("div", "fresh-hero");
  hero.append(
    el("div", "fresh-mark", "✳"),
    el("div", "fresh-title", "新的对话"),
    el("div", "fresh-sub", "还没有任何上下文 — 问点什么开始吧"),
  );
  m.appendChild(hero);
  $(".chat").classList.add("fresh");
}

function clearFresh() {
  const c = $(".chat");
  if (!c.classList.contains("fresh")) return;
  c.classList.remove("fresh");
  const hero = $(".fresh-hero");
  if (hero) hero.remove();
}

function addMsg(role, text) {
  clearFresh();
  const wrap = el("div", `msg ${role}`);
  const body = el("div", "bubble");
  if (text) body.textContent = text;
  wrap.appendChild(body);
  $("#messages").appendChild(wrap);
  scroll();
  return body;
}

/* ---------- the tail activity row ---------- */
// The header's spinner is at the TOP of a scrolling transcript, so during a long
// wait the reader is looking at the bottom where nothing moves -- reported as
// "I cannot tell if the model died or is thinking". This row lives at the TAIL,
// where the eye already is.
function showPending() {
  let row = $("#pending");
  if (!row) {
    row = el("div", "msg bot");
    row.id = "pending";
    const b = el("div", "bubble pending");
    b.append(el("span", "p-spin"), el("span", "p-text", "思考中"), el("span", "p-since"));
    b.querySelector(".p-since").dataset.since = String(Date.now());
    row.appendChild(b);
    $("#messages").appendChild(row);
    scroll();
  }
  return row;
}
function setPendingText(t) {
  const n = $("#pending .p-text");
  if (n) n.textContent = t;
}
function clearPending() {
  const row = $("#pending");
  if (row) row.remove();
}

/* ---------- segments ---------- */
function finalizeSeg() {
  if (!S.seg) return;
  const seg = S.seg;
  // Flush any render the rAF has not run yet, BEFORE dropping the reference it
  // needs. scheduleRender bails on `!S.seg`, so clearing first meant the last
  // batch of tokens was never painted: a live turn lost its closing sentences,
  // and a loaded session -- where every token arrives in one synchronous
  // forEach and the only render is the deferred one -- showed an empty bubble
  // where the whole answer should be.
  if (seg.kind === "out") seg.body.innerHTML = renderMD(seg.text);
  S.seg = null;
  if (seg.kind === "think") { activitySummary(); return; }
  if (!seg.text.trim()) { seg.body.closest(".msg").remove(); return; }
  seg.body.classList.remove("streaming");
}

function activityGroup() {
  // One disclosure row per assistant turn, holding the thinking AND every tool.
  // Upstream's design guide is explicit about this: a turn that used ten tools
  // should read as one turn with one compact "Activity: 10 tools" row, not ten
  // chat cards. Ours rendered a card per tool and a separate row for thinking,
  // which is what buried the answer the reader came for.
  if (S.activity && document.body.contains(S.activity.wrap)) return S.activity;
  const wrap = el("div", "msg bot");
  const card = el("div", "bubble activity");
  const head = el("button", "act-head");
  const caret = el("span", "caret", "▸");
  const label = el("span", "act-label", "思考中");
  head.append(caret, label);
  const body = el("div", "act-body");
  const idx = S.actIndex++;
  const open = isOpen(idx);
  body.hidden = !open;
  caret.textContent = open ? "▾" : "▸";
  head.onclick = () => {
    body.hidden = !body.hidden;
    caret.textContent = body.hidden ? "▸" : "▾";
    setOpen(idx, !body.hidden);
  };
  card.append(head, body);
  wrap.appendChild(card);
  $("#messages").appendChild(wrap);
  S.activity = { wrap, card, head, caret, label, body, think: null, tools: 0 };
  scroll();
  return S.activity;
}

function activitySummary() {
  const a = S.activity;
  if (!a) return;
  const bits = [];
  if (a.tools) bits.push(`${a.tools} 个工具`);
  if (a.think && a.think.textContent.trim()) bits.push("思考");
  a.label.textContent = bits.length ? `活动 · ${bits.join(" · ")}` : "思考";
}

function newThinkSeg() {
  const a = activityGroup();
  if (!a.think) {
    a.think = el("div", "act-think");
    a.body.appendChild(a.think);
  }
  S.seg = { kind: "think", body: a.think, text: "", refs: null };
}

function newOutputSeg() {
  clearPending();      // tokens are their own proof of life
  const body = addMsg("bot", "");
  body.classList.add("streaming", "md");
  S.seg = { kind: "out", body, text: "", refs: null };
}

// Coalesce streamed tokens to one render per frame. Re-parsing the whole bubble
// on every token is what makes a long answer crawl.
let renderPending = false;
function scheduleRender() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    if (!S.seg || S.seg.kind !== "out") return;
    S.seg.body.innerHTML = renderMD(S.seg.text);
    scroll();
  });
}

function appendThought(text) {
  if (!text) return;
  clearPending();
  status("思考中");
  if (!S.seg || S.seg.kind !== "think") { finalizeSeg(); newThinkSeg(); }
  S.seg.text += text;
  S.seg.body.textContent = S.seg.text;
  activitySummary();
  scroll();
}

function appendToken(text) {
  if (!text) return;
  status("生成回复");
  if (!S.seg || S.seg.kind !== "out") { finalizeSeg(); newOutputSeg(); }
  S.seg.text += text;
  scheduleRender();
}

/* ---------- tools ---------- */
function toolRow(id, title, st, detail, full) {
  const a = activityGroup();
  let row = S.tools.get(id);
  if (!row) {
    row = el("div", "act-tool");
    row.append(
      el("span", "t-name", title || "tool"),
      el("span", "t-since"),
      el("span", "t-status", st || "")
    );
    row.querySelector(".t-since").dataset.since = String(Date.now());
    a.body.appendChild(row);
    a.tools += 1;
    S.tools.set(id, row);
  }
  if (title) row.querySelector(".t-name").textContent = title;
  row.querySelector(".t-status").textContent = st || "";
  if (detail) {
    let d = row.nextElementSibling;
    if (!d || !d.classList.contains("act-detail")) {
      d = el("pre", "act-detail");
      row.after(d);
      // The row toggles only ITS detail. The group's caret is about the whole
      // activity; a tool's arguments and result are one level further down,
      // which is where the design guide puts them.
      row.classList.add("has-detail");
      row.onclick = () => { d.hidden = !d.hidden; };
      d.hidden = true;
    }
    renderDetail(d, detail, full || detail.length);
  }
  row.dataset.status = st || "";
  if (st === "completed" || st === "failed") {
    const n = row.querySelector(".t-since");
    if (n) n.removeAttribute("data-since");
    if (S.busy) { showPending(); setPendingText("处理检索结果"); }
  } else if (st) { status(title || "工具执行中"); showPending(); }
  activitySummary();
  scroll();
}

// Clamp a tool result and let the reader open it, instead of cutting it and
// hoping the middle did not matter. `full` is the value's TRUE length: when the
// server hit its transport ceiling the tail never arrived, and saying so is the
// difference between "there is more, here it is" and a value that just stops.
const DETAIL_CLAMP = 1200;

function renderDetail(pre, text, full) {
  // A tool emits several `tool_call_update`s, and each one re-renders this
  // block. Read back the reader's own choice first: without it, a detail they
  // opened mid-run snapped shut the moment the tool reported progress.
  let expanded = pre.dataset.expanded === "1";
  pre.textContent = "";
  const short = text.length > DETAIL_CLAMP;
  const body = el("span", null, short ? text.slice(0, DETAIL_CLAMP) : text);
  pre.appendChild(body);
  if (!short && full <= text.length) return;

  const more = el("button", "more");
  const cut = full - text.length;          // never delivered
  const setLabel = (expanded) => {
    more.textContent = expanded
      ? "收起"
      : `显示全部（还有 ${full - DETAIL_CLAMP} 字${cut > 0 ? `，其中 ${cut} 字未传输` : ""}）`;
  };
  more.onclick = (e) => {
    e.stopPropagation();                    // the row's own toggle must not fire
    expanded = !expanded;
    pre.dataset.expanded = expanded ? "1" : "0";
    body.textContent = expanded ? text : text.slice(0, DETAIL_CLAMP);
    setLabel(expanded);
  };
  if (expanded) body.textContent = text;
  setLabel(expanded);
  pre.appendChild(more);
}

/* ---------- approvals ---------- */
function showApproval(a) {
  if (document.querySelector(`[data-perm="${a.id}"]`)) return;   // already on screen
  S.awaitingPerm = true;
  const wrap = el("div", "msg bot");
  const card = el("div", "bubble perm");
  card.dataset.perm = a.id;
  card.appendChild(el("div", "perm-title", a.title || "需要确认"));
  const opts = el("div", "perm-opts");
  (a.options || []).forEach((o) => {
    const b = el("button", "btn opt", o.name || o.optionId);
    b.onclick = () => answerApproval(a.id, o.optionId, card);
    opts.appendChild(b);
  });
  if (!(a.options || []).length) {
    const b = el("button", "btn opt", "取消");
    b.onclick = () => answerApproval(a.id, null, card);
    opts.appendChild(b);
  }
  card.appendChild(opts);
  wrap.appendChild(card);
  $("#messages").appendChild(wrap);
  status("等待你的确认");
  scroll();
}

async function answerApproval(id, optionId, card) {
  S.awaitingPerm = false;
  if (card) { card.classList.add("done"); card.querySelector(".perm-opts").remove(); }
  await fetch("/api/approval/answer", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, optionId }),
  }).catch(() => {});
}

async function pollApprovals() {
  try {
    const j = await (await fetch("/api/approval/pending")).json();
    const p = (j.pending || [])[0];
    if (p) showApproval(p);
    else S.awaitingPerm = false;
  } catch (_) { /* transient; the next tick retries */ }
}

/* ---------- the stream ---------- */
function apply(ev, from) {
  // Does this belong to what the reader is looking at? Browsing mid-turn means
  // the answer can be no, and a stream that paints regardless puts one
  // conversation's tokens into another's transcript. The turn is untouched;
  // only the drawing is skipped. `end` still runs, because the turn really did
  // end and the bookkeeping it does is not about the screen.
  //
  // A conversation that does not exist yet cannot be identified by session id —
  // it has none until hermes assigns one — so for that view the question is
  // "did MY send start this stream", not "does this id match". Asking the id
  // there matched everything, because null matches nothing and the check let it
  // through: another session's reply drew straight into the new one.
  const fresh = S.pendingNew && !S.sessionId;
  const mine = fresh
    ? (!!S.ownStream && from === S.ownStream)
    : (!ev.session || !S.sessionId || ev.session === S.sessionId);
  // The stream is the authority on which conversation this turn belongs to.
  // Reading `acp.session_id` instead raced hermes assigning it and could hand
  // back the PREVIOUS session, relabelling the new conversation as the old one.
  if (mine && fresh && ev.session) {
    S.sessionId = ev.session;
    S.pendingNew = false;
    loadSessions();
  }
  if (!mine && ev.kind !== "end") {
    if (ev.seq) { S.lastSeq = ev.seq; if (S.busy) remember(S.streamId, ev.seq); }
    return;
  }
  switch (ev.kind) {
    case "user":
      // We already drew this optimistically in send(), so the stream's echo of
      // the SAME message would render it twice. A reattach after reload draws
      // nothing first, so there the echo is exactly what paints it.
      finalizeSeg();
      if (S.skipUserEcho) S.skipUserEcho = false;
      else addMsg("user", ev.text);
      break;
    case "history_user": finalizeSeg(); addMsg("user", ev.text); break;
    case "delta": ev.thought ? appendThought(ev.text) : appendToken(ev.text); break;
    case "tool": toolRow(ev.id, ev.title, ev.status, ev.detail, ev.detailFull); break;
    case "approval": showApproval(ev); break;
    case "approval_expired":
      S.awaitingPerm = false;
      finalizeSeg(); addMsg("note", "审批超时，本次操作已取消");
      break;
    case "note": finalizeSeg(); addMsg("note", ev.text); break;
    case "gap": finalizeSeg(); addMsg("note", "（断线期间有部分输出未能保留）"); break;
    case "end": endTurn(ev.error, ev.session); break;
  }
  // Only while a turn is actually live. `endTurn()` clears the saved cursor,
  // and this line runs AFTER the switch that called it — so persisting
  // unconditionally put the finished stream straight back into localStorage,
  // and the next reload reattached to a turn that had nothing left to say.
  if (ev.seq) { S.lastSeq = ev.seq; if (S.busy) remember(S.streamId, ev.seq); }
}

function attach(streamId, afterSeq) {
  detach();
  S.streamId = streamId;
  S.lastSeq = afterSeq || 0;
  setBusy(true);
  const es = new EventSource(
    `/api/chat/stream?stream_id=${encodeURIComponent(streamId)}&after_seq=${S.lastSeq}`
  );
  S.es = es;
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch (_) { return; }
    apply(ev, streamId);
  };
  es.onerror = async () => {
    if (!S.busy) return;
    // EventSource retries forever on its own, so "reconnecting" can outlive the
    // thing it claims to be reconnecting to: a server restart drops every live
    // stream AND the in-process turn behind it, and the banner then sits there
    // permanently while nothing is coming. Ask before claiming.
    status("连接中断,重连中…");
    try {
      const st = await (await fetch(
        `/api/chat/status?stream_id=${encodeURIComponent(streamId)}`
      )).json();
      if (!st.known) {
        // The turn is gone with the process that held it. Say so plainly rather
        // than pretending a reconnect is in progress.
        detach();
        finalizeSeg();
        setBusy(false);
        forget();
        addMsg("note", "服务重启，这一轮的回复已丢失");
        status("就绪");
      } else if (!st.running) {
        // It finished while we were disconnected; the reconnect will collect
        // the tail, so leave the stream alone and stop alarming the reader.
        status("接回中…");
      }
    } catch (_) { /* the server is genuinely unreachable — keep retrying */ }
  };
}

function detach() { if (S.es) { S.es.close(); S.es = null; } }

function endTurn(error, owner) {
  // The turn's own session, which is not necessarily the one on screen.
  const shown = !owner || !S.sessionId || owner === S.sessionId;
  if (owner) HISTORY_CACHE.delete(owner);
  if (S.sessionId) HISTORY_CACHE.delete(S.sessionId);
  detach();
  clearPending();
  finalizeSeg();
  S.activity = null;      // the next turn opens its own group
  setBusy(false);
  S.tools.clear();
  S.awaitingPerm = false;
  forget();
  if (error && shown) addMsg("error", `⚠ ${error}`);
  const wasStopping = S.stopping;
  S.stopping = false;
  status(error ? "出错" : wasStopping ? "已停止" : "就绪");
  loadSessions();   // the turn may have created or retitled a session
}

function cancelTurn() {
  // A REQUEST, not an instant stop: the agent finishes the chunk it is on
  // first, measured at ~12 s here. Without saying so the button reads as
  // broken, which is exactly how it was reported.
  if (!S.busy || S.stopping) return;        // a second click adds nothing
  S.stopping = true;
  $("#send").textContent = "停止中…";
  $("#send").disabled = true;
  status("正在停止,等 agent 收尾…");
  fetch("/api/chat/cancel", { method: "POST" }).catch(() => {
    // The request itself failed — re-arm so the button is usable again.
    S.stopping = false;
    $("#send").textContent = "停止";
    $("#send").disabled = false;
    status("停止请求发送失败");
  });
}

function setBusy(b) {
  S.busy = b;
  if (!b) S.stopping = false;
  // Only when the reader is looking AT the streaming session. Otherwise the
  // composer would offer to stop a turn belonging to a conversation that is
  // not on screen — which is what a global busy flag did as soon as browsing
  // mid-turn became possible.
  const mine = b && (!S.streamingSession || S.streamingSession === S.sessionId);
  $("#send").disabled = false;
  $("#send").textContent = mine ? "停止" : "发送";
  $("#send").classList.toggle("stop", !!mine);
  if (b) {
    S.startedAt = Date.now();
    if (!S.timer) S.timer = setInterval(() => { tick(); tickSlow(); }, 90);
    tick(); tickSlow();
    // Polling backstop for approvals: the push rides the same stream a proxy or
    // a sleeping laptop can drop, and a missed prompt stalls the turn with
    // nothing on screen to explain it.
    if (!S.approvalTimer) S.approvalTimer = setInterval(pollApprovals, 1500);
  } else {
    S.startedAt = 0;
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    if (S.approvalTimer) { clearInterval(S.approvalTimer); S.approvalTimer = null; }
    tick();
  }
}

/* ---------- sessions ---------- */
async function loadSessions() {
  let j;
  try { j = await (await fetch("/api/sessions")).json(); } catch (_) { return; }
  const ul = $("#sessions");
  ul.textContent = "";
  const rows = j.sessions || [];
  // Adopt the id the server says is current. It is the only place the client
  // can learn it for a conversation it did not open from the sidebar — a fresh
  // one gets its id from hermes on its first turn — and without it those
  // conversations all key their disclosure state to the same "-" bucket.
  // `j.current` is where the AGENT is, which is no longer where the reader is
  // looking — that is the whole point of being able to browse mid-turn. Adopt
  // it only when the reader has no session of their own yet.
  // A brand-new conversation has no id at the moment `chat/start` returns —
  // hermes assigns it as the turn begins — so the row appears unselected. If we
  // have none of our own, the streaming session is ours by construction: we
  // just sent to it.
  if (!S.sessionId && !S.pendingNew) S.sessionId = j.streaming || j.current || null;
  S.streamingSession = j.streaming || null;
  S.streamingStreamId = j.streamingStreamId || null;
  $("#sess-count").textContent = rows.length ? String(rows.length) : "";
  rows.forEach((s) => {
    let cls = s.id === S.sessionId ? "sess cur" : "sess";
    if (s.is_streaming) cls += " streaming";
    const li = el("li", cls);
    li.append(el("div", "t", s.title || "(未命名)"), el("div", "p", s.preview || ""));
    li.onclick = () => openSession(s.id);
    if (s.is_streaming) {
      // The stop control belongs to the session that is actually running, not
      // to the composer. With browsing allowed, a global stop button sits in
      // front of whatever you happen to be READING and stops something else.
      const stop = el("button", "stop-row", "■");
      stop.title = "停止这个会话的回复";
      stop.onclick = (e) => { e.stopPropagation(); cancelTurn(); };
      li.appendChild(stop);
    }
    const del = el("button", "del", "×");
    del.title = "删除会话";
    del.onclick = (e) => { e.stopPropagation(); removeSession(s.id, s.title); };
    li.appendChild(del);
    ul.appendChild(li);
  });
}

// Transcripts already replayed in this page's lifetime. A switch back is then
// a repaint, not a round trip — which matters because `session/load` costs
// about a second: hermes re-registers the MCP server and rebuilds its whole
// tool surface on EVERY load, regardless of how short the transcript is.
const HISTORY_CACHE = new Map();

function paintHistory(history) {
  // Skip empty replayed messages. A turn that produced only a tool call, or was
  // cancelled before its first token, leaves a text-less entry that renders as
  // a blank bubble the reader cannot account for.
  history
    .filter((ev) => ev.kind !== "delta" || (ev.text || "").length)
    .forEach(apply);   // history carries no seq — a finished transcript
  finalizeSeg();
}

async function openSession(id) {
  clearFresh();
  // No busy guard. Reading a transcript no longer moves the agent, so a turn
  // in flight is none of this function's business — it keeps streaming into
  // whichever session owns it, and `send()` relocates the agent when the reader
  // actually says something. Telling someone to "wait or stop the current
  // reply" just to LOOK at another conversation was the whole complaint.
  $("#messages").textContent = "";
  S.tools.clear(); S.seg = null; S.activity = null; S.actIndex = 0; S.sessionId = id;

  // Paint what we already have BEFORE asking the server. The request cannot be
  // skipped even on a hit — it is what moves the ACP process to this session,
  // and without it the next prompt would land on the previous one — but the
  // reader does not have to wait for it to see the transcript. Sending is
  // blocked until it lands, so the two can never disagree.
  const cached = HISTORY_CACHE.get(id);
  if (cached) paintHistory(cached);
  S.switching = id;
  status(cached ? "切换中…" : "载入中…");

  let j;
  try {
    j = await (await fetch(`/api/session/${encodeURIComponent(id)}/history`)).json();
  } catch (_) {
    S.switching = null; status("载入会话失败"); return;
  }
  S.switching = null;
  if (j.error) { status(`载入失败: ${j.error}`); return; }
  // A switch the reader started and then abandoned: they are looking at another
  // session now, so painting this one's history would corrupt what they see.
  if (S.sessionId !== id) return;
  HISTORY_CACHE.set(id, j.history || []);
  if (!cached) paintHistory(j.history || []);
  if (S.busy && S.streamingSession === id && S.streamingStreamId) {
    // Back on the conversation that is running. The store stops where the
    // committed transcript does, so replay the live turn from the top rather
    // than showing a reply that appears to have stopped mid-sentence.
    S.skipUserEcho = false;
    attach(S.streamingStreamId, 0);
    status("回复中…");
  } else {
    status(S.busy ? "另一个会话仍在回复中" : "就绪");
  }
  loadSessions();
}

async function removeSession(id, title) {
  // Ask first. This control is invisible until hover and covers the right
  // 2.4rem of the row at full height, so it sits exactly where a hand goes to
  // click the row itself — and the delete is immediate, schema-aware, and has
  // no undo. Three conversations went that way.
  const name = (title || "").trim() || id.slice(0, 8);
  if (!confirm(`删除会话「${name}」?此操作不可撤销。`)) return;
  await fetch(`/api/session/${encodeURIComponent(id)}`, { method: "DELETE" }).catch(() => {});
  if (S.sessionId === id) S.sessionId = null;
  loadSessions();
}

async function newSession() {
  // Purely local. `/api/session/new` moves the one ACP process, which a turn in
  // flight cannot survive, so the intent is recorded and acted on at the next
  // send — where nothing is streaming by construction.
  S.pendingNew = true;
  S.ownStream = null;       // nothing on screen is ours until we send
  // The turn in flight keeps running and keeps its stream; only the drawing
  // stops, because `apply` now checks who each event belongs to. Detaching or
  // calling `endTurn` here would abandon a live reply.
  $("#messages").textContent = "";
  S.seg = null; S.tools.clear(); S.activity = null; S.actIndex = 0; S.sessionId = null;
  showFresh();
  status(S.busy ? "另一个会话仍在回复中" : "就绪");
  loadSessions();
}

/* ---------- sending ---------- */
async function send() {
  const input = $("#input");
  const text = input.value.trim();
  if (!text) return;
  if (S.awaitingPerm) {
    // Never a silent return: you type, press Enter, nothing happens, and the
    // reason (an approval is blocking the agent) is invisible. Point at it.
    const card = document.querySelector("[data-perm]");
    if (card) card.scrollIntoView({ block: "nearest" });
    status("⚠ 先回答上方的确认请求");
    return;
  }
  if (S.busy) return;
  if (S.switching) { status("正在切换会话，稍候"); return; }
  input.value = ""; input.style.height = "auto";
  addMsg("user", text);
  S.seg = null;
  S.activity = null;
  S.skipUserEcho = true;
  let j;
  try {
    j = await (await fetch("/api/chat/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        // The agent is relocated server-side from these, not when a session was
        // clicked. Both are no-ops when it is already in the right place.
        sessionId: S.pendingNew ? "" : (S.sessionId || ""),
        new: !!S.pendingNew,
      }),
    })).json();
  } catch (_) { status("发送失败"); return; }
  if (j.error) { status(j.error); return; }
  if (!j.streamId) { status("没有可用的会话流"); return; }
  S.pendingNew = false;
  // `attached` means a turn was ALREADY running and we joined it -- its prompt
  // is not the one we just drew, so let the echo paint it.
  if (j.attached) S.skipUserEcho = false;
  // A brand-new conversation gets its id from hermes only now, so this is the
  // first moment the sidebar can show it. Refreshing here rather than at the
  // end of the turn is the difference between "it appears when you ask" and
  // "it appears when the answer finishes".
  // Ours, so `apply` can tell our own turn from one still running elsewhere
  // while this view has no id of its own yet.
  S.ownStream = j.streamId;
  if (j.sessionId) { S.sessionId = j.sessionId; S.pendingNew = false; }
  loadSessions();
  remember(j.streamId, 0);
  showPending();
  attach(j.streamId, 0);
}

/* ---------- boot ---------- */
async function boot() {
  loadSessions();
  fetch("/api/status").then((r) => r.json()).then((j) => {
    if (j.mcp) $("#mcp-badge").innerHTML = `mcp <b>${j.mcp.name}</b>`;
    else $("#mcp-badge").textContent = "mcp 未接";
  }).catch(() => {});

  // Reattach BEFORE accepting input: a turn that survived the reload should come
  // up visibly running, not look idle and invite a second prompt.
  const { id, seq } = recall();
  if (id) {
    try {
      const st = await (await fetch(`/api/chat/status?stream_id=${encodeURIComponent(id)}`)).json();
      if (st.known) {
        S.skipUserEcho = false;   // nothing drawn yet — the echo must paint it
        if (st.running) status("接回上一次未完成的回复…");
        attach(id, seq);   // finished-but-unseen still needs its tail collected
      } else forget();
    } catch (_) { forget(); }
  } else if (!$("#messages").firstChild) {
    // Nothing to reattach and nothing on screen: this IS a fresh conversation,
    // so say so rather than opening on a blank panel that reads as loading.
    showFresh();
  }

  // The architecture note. Fetched on first open, not inlined: it is prose that
  // changes on its own schedule, and a reader who never opens it should not pay
  // for it.
  let aboutLoaded = false;
  const openAbout = async () => {
    const box = $("#about");
    if (!aboutLoaded) {
      try {
        box.querySelector("#about-body").innerHTML = renderMD(
          await (await fetch("/static/about.md")).text()
        );
        aboutLoaded = true;
      } catch (_) {
        box.querySelector("#about-body").textContent = "架构说明加载失败";
      }
    }
    box.hidden = false;
  };
  $("#about-btn").onclick = openAbout;
  $("#about-close").onclick = () => { $("#about").hidden = true; };
  $("#about").onclick = (e) => { if (e.target.id === "about") $("#about").hidden = true; };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#about").hidden) $("#about").hidden = true;
  });

  $("#new-session").onclick = newSession;
  // Cmd/Ctrl+K from anywhere, including the composer -- the shortcut is useless
  // if you have to leave the box you are typing in to reach it.
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      newSession();
    }
  });
  $("#send").onclick = (e) => {
    if (!S.busy) return;                    // not busy: let the form submit
    e.preventDefault();
    cancelTurn();
  };

  const input = $("#input");
  const autosize = () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    // Ignore Enter while an IME is composing — that Enter confirms a candidate
    // (pinyin / CJK), and sending on it both fires a half-typed message and
    // leaves residue in the box. `isComposing` is the standard flag; keyCode
    // 229 is the legacy fallback some browsers still report instead.
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      send();
    }
  });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); if (!S.busy) send(); });
}

boot();
