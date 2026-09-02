"use strict";
/* hermes webui client.
 *
 * The one rule that shapes this file: a turn belongs to the SERVER, not to this
 * page. So the page never holds the only copy of anything it would need to
 * recover — the running turn's id and the last event sequence it rendered both
 * live in localStorage, and on load the page asks the server what happened
 * while it was away instead of assuming nothing did.
 *
 * That is why there is no "reconnecting…" spinner that gives up: reattaching is
 * the normal path, not the error path.
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

const S = {
  streamId: null,
  lastSeq: 0,
  es: null,           // EventSource
  busy: false,
  approvalTimer: null,
  bubble: null,       // the assistant bubble currently being appended to
  tools: new Map(),   // toolCallId -> row element
};

/* ---------- persistence of just enough to recover ---------- */
function remember(streamId, seq) {
  try {
    if (streamId) localStorage.setItem(LS_STREAM, streamId);
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

/* ---------- rendering ---------- */
function banner(text) {
  const b = $("#banner");
  if (!text) { b.hidden = true; b.textContent = ""; return; }
  b.hidden = false;
  b.textContent = text;
}

function addBubble(who, text) {
  const wrap = el("div", `msg ${who}`);
  const body = el("div", "body");
  body.textContent = text || "";
  wrap.appendChild(body);
  $("#messages").appendChild(wrap);
  scroll();
  return body;
}

function scroll() {
  const m = $("#messages");
  // Only follow the tail if the reader is already at it — yanking the viewport
  // away from someone reading back through the transcript is worse than a
  // missed autoscroll.
  const nearBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 120;
  if (nearBottom) m.scrollTop = m.scrollHeight;
}

function toolRow(id, title, status) {
  let row = S.tools.get(id);
  if (!row) {
    row = el("div", "tool");
    row.appendChild(el("span", "tool-title", title || "tool"));
    row.appendChild(el("span", "tool-status", status || ""));
    $("#messages").appendChild(row);
    S.tools.set(id, row);
  } else {
    if (title) row.querySelector(".tool-title").textContent = title;
    row.querySelector(".tool-status").textContent = status || "";
  }
  row.dataset.status = status || "";
  scroll();
}

function apply(ev) {
  switch (ev.kind) {
    case "user":
      addBubble("user", ev.text);
      break;
    case "history_user":
      addBubble("user", ev.text);
      break;
    case "delta":
      // Thought chunks share the bubble but are muted; a separate bubble per
      // thought fragment turned the transcript into confetti.
      if (!S.bubble) S.bubble = addBubble("bot", "");
      S.bubble.textContent += ev.text;
      scroll();
      break;
    case "tool":
      toolRow(ev.id, ev.title, ev.status);
      break;
    case "approval":
      showApproval(ev);
      break;
    case "approval_expired":
      hideApproval();
      banner("审批超时，本次操作已取消");
      break;
    case "note":
      addBubble("note", ev.text);
      break;
    case "gap":
      addBubble("note", "（断线期间有部分输出未能保留）");
      break;
    case "end":
      endTurn(ev.error);
      break;
  }
  if (ev.seq) { S.lastSeq = ev.seq; remember(S.streamId, ev.seq); }
}

/* ---------- the stream ---------- */
function attach(streamId, afterSeq) {
  detach();
  S.streamId = streamId;
  S.lastSeq = afterSeq || 0;
  setBusy(true);
  const url = `/api/chat/stream?stream_id=${encodeURIComponent(streamId)}&after_seq=${S.lastSeq}`;
  const es = new EventSource(url);
  S.es = es;
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch (_) { return; }
    apply(ev);
  };
  es.onerror = () => {
    // EventSource retries on its own. Say so rather than showing a dead UI, and
    // do NOT clear the turn: it is still running on the server.
    if (S.busy) banner("连接中断，正在重连…");
  };
}

function detach() {
  if (S.es) { S.es.close(); S.es = null; }
}

function endTurn(error) {
  detach();
  setBusy(false);
  S.bubble = null;
  S.tools.clear();
  forget();
  banner(error ? `出错：${error}` : "");
  loadSessions();  // the turn may have created or retitled a session
}

function setBusy(b) {
  S.busy = b;
  $("#send").disabled = b;
  $("#stop").hidden = !b;
  if (b) {
    banner("");
    // Polling backstop for approvals: the push rides the same stream that a
    // proxy or a sleeping laptop can drop, and a missed approval prompt stalls
    // the turn with nothing on screen to explain it.
    if (!S.approvalTimer) S.approvalTimer = setInterval(pollApprovals, 1500);
  } else if (S.approvalTimer) {
    clearInterval(S.approvalTimer);
    S.approvalTimer = null;
  }
}

/* ---------- approvals ---------- */
function showApproval(a) {
  $("#approval").hidden = false;
  $("#approval-title").textContent = a.title || "需要确认";
  const box = $("#approval-options");
  box.textContent = "";
  (a.options || []).forEach((o) => {
    const b = el("button", "opt", o.name || o.optionId);
    b.onclick = () => answerApproval(a.id, o.optionId);
    box.appendChild(b);
  });
  if (!(a.options || []).length) {
    const b = el("button", "opt", "取消");
    b.onclick = () => answerApproval(a.id, null);
    box.appendChild(b);
  }
}
function hideApproval() { $("#approval").hidden = true; }

async function answerApproval(id, optionId) {
  hideApproval();
  await fetch("/api/approval/answer", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, optionId }),
  }).catch(() => {});
}

async function pollApprovals() {
  try {
    const r = await fetch("/api/approval/pending");
    const j = await r.json();
    const p = (j.pending || [])[0];
    if (p) showApproval(p); else hideApproval();
  } catch (_) { /* transient; the next tick retries */ }
}

/* ---------- sessions ---------- */
async function loadSessions() {
  let j;
  try { j = await (await fetch("/api/sessions")).json(); } catch (_) { return; }
  const ul = $("#sessions");
  ul.textContent = "";
  (j.sessions || []).forEach((s) => {
    const li = el("li", s.id === j.current ? "sess cur" : "sess");
    li.appendChild(el("div", "t", s.title || "(未命名)"));
    li.appendChild(el("div", "p", s.preview || ""));
    li.onclick = () => openSession(s.id);
    const del = el("button", "del", "×");
    del.title = "删除会话";
    del.onclick = (e) => { e.stopPropagation(); removeSession(s.id); };
    li.appendChild(del);
    ul.appendChild(li);
  });
}

async function openSession(id) {
  if (S.busy) { banner("请先等待或停止当前回复"); return; }
  $("#messages").textContent = "";
  S.tools.clear();
  S.bubble = null;
  let j;
  try { j = await (await fetch("/api/session/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  })).json(); } catch (_) { banner("载入会话失败"); return; }
  if (j.error) { banner(`载入会话失败：${j.error}`); return; }
  (j.history || []).forEach((ev) => {
    // History carries no seq — it is a completed transcript, not a live stream.
    if (ev.kind === "delta") { S.bubble = S.bubble || addBubble("bot", ""); S.bubble.textContent += ev.text; }
    else apply(ev);
  });
  S.bubble = null;
  loadSessions();
}

async function removeSession(id) {
  await fetch(`/api/session/${encodeURIComponent(id)}`, { method: "DELETE" }).catch(() => {});
  loadSessions();
}

async function newSession() {
  if (S.busy) { banner("请先等待或停止当前回复"); return; }
  await fetch("/api/session/new", { method: "POST" }).catch(() => {});
  $("#messages").textContent = "";
  S.bubble = null;
  S.tools.clear();
  loadSessions();
}

/* ---------- sending ---------- */
async function send(text) {
  addBubble("user", text);
  S.bubble = null;
  let j;
  try {
    j = await (await fetch("/api/chat/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })).json();
  } catch (_) { banner("发送失败"); return; }
  if (j.error) { banner(j.error); return; }
  if (!j.streamId) { banner("没有可用的会话流"); return; }
  remember(j.streamId, 0);
  // `attached` means a turn was already running and we joined it; start from
  // seq 0 either way, since this page has rendered nothing of it.
  attach(j.streamId, 0);
}

/* ---------- boot ---------- */
async function boot() {
  loadSessions();
  fetch("/api/status").then((r) => r.json()).then((j) => {
    if (j.mcp) $("#mcp-badge").textContent = `mcp: ${j.mcp.name}`;
  }).catch(() => {});

  // Reattach BEFORE accepting input: if a turn survived the reload, the page
  // should come up showing it running rather than looking idle and inviting a
  // second prompt.
  const { id, seq } = recall();
  if (id) {
    try {
      const st = await (await fetch(`/api/chat/status?stream_id=${encodeURIComponent(id)}`)).json();
      if (st.known && st.running) {
        banner("正在接回上一次未完成的回复…");
        attach(id, seq);
      } else if (st.known) {
        // It finished while we were away. Collect the tail so the answer is not
        // silently lost, then let the stream close itself.
        attach(id, seq);
      } else {
        forget();
      }
    } catch (_) { forget(); }
  }

  $("#new-session").onclick = newSession;
  $("#stop").onclick = () => fetch("/api/chat/cancel", { method: "POST" }).catch(() => {});
  const input = $("#input");
  const autosize = () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 200)}px`; };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || S.busy) return;
    input.value = "";
    autosize();
    send(text);
  });
}

boot();
