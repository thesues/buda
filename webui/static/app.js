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
function status(text) { $("#run-status").textContent = text; }

function tick() {
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
function addMsg(role, text) {
  const wrap = el("div", `msg ${role}`);
  const body = el("div", "bubble");
  if (text) body.textContent = text;
  wrap.appendChild(body);
  $("#messages").appendChild(wrap);
  scroll();
  return body;
}

/* ---------- segments ---------- */
function finalizeSeg() {
  if (!S.seg) return;
  const seg = S.seg;
  S.seg = null;
  if (!seg.text.trim()) { seg.body.closest(".msg").remove(); return; }
  if (seg.kind === "think") {
    seg.refs.label.textContent = "思考";
    seg.refs.detail.hidden = true;      // collapse; the header reopens it
    seg.refs.caret.textContent = "▸";
  } else {
    seg.body.classList.remove("streaming");
  }
}

function newThinkSeg() {
  const wrap = el("div", "msg bot");
  const card = el("div", "bubble think");
  const head = el("button", "think-head");
  const caret = el("span", "caret", "▾");
  const label = el("span", "think-label", "思考中…");
  head.append(caret, label);
  const detail = el("div", "think-detail");
  head.onclick = () => {
    detail.hidden = !detail.hidden;
    caret.textContent = detail.hidden ? "▸" : "▾";
  };
  card.append(head, detail);
  wrap.appendChild(card);
  $("#messages").appendChild(wrap);
  S.seg = { kind: "think", body: card, text: "", refs: { head, caret, label, detail } };
  scroll();
}

function newOutputSeg() {
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
  status("思考中");
  if (!S.seg || S.seg.kind !== "think") { finalizeSeg(); newThinkSeg(); }
  S.seg.text += text;
  S.seg.refs.detail.textContent = S.seg.text;
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
function toolRow(id, title, st) {
  let row = S.tools.get(id);
  if (!row) {
    row = el("div", "msg bot");
    const card = el("div", "bubble tool");
    card.append(el("span", "tool-title", title || "tool"), el("span", "tool-status", st || ""));
    row.appendChild(card);
    $("#messages").appendChild(row);
    S.tools.set(id, card);
    row = card;
  } else {
    if (title) row.querySelector(".tool-title").textContent = title;
    row.querySelector(".tool-status").textContent = st || "";
  }
  row.dataset.status = st || "";
  if (st && st !== "completed" && st !== "failed") status(title || "工具执行中");
  scroll();
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
function apply(ev) {
  switch (ev.kind) {
    case "user": finalizeSeg(); addMsg("user", ev.text); break;
    case "history_user": finalizeSeg(); addMsg("user", ev.text); break;
    case "delta": ev.thought ? appendThought(ev.text) : appendToken(ev.text); break;
    case "tool": toolRow(ev.id, ev.title, ev.status); break;
    case "approval": showApproval(ev); break;
    case "approval_expired":
      S.awaitingPerm = false;
      finalizeSeg(); addMsg("note", "审批超时，本次操作已取消");
      break;
    case "note": finalizeSeg(); addMsg("note", ev.text); break;
    case "gap": finalizeSeg(); addMsg("note", "（断线期间有部分输出未能保留）"); break;
    case "end": endTurn(ev.error); break;
  }
  if (ev.seq) { S.lastSeq = ev.seq; remember(S.streamId, ev.seq); }
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
    apply(ev);
  };
  es.onerror = () => {
    // EventSource retries on its own. Say so — and do NOT clear the turn: it is
    // still running on the server.
    if (S.busy) status("连接中断,重连中…");
  };
}

function detach() { if (S.es) { S.es.close(); S.es = null; } }

function endTurn(error) {
  detach();
  finalizeSeg();
  setBusy(false);
  S.tools.clear();
  S.awaitingPerm = false;
  forget();
  if (error) addMsg("error", `⚠ ${error}`);
  status(error ? "出错" : "就绪");
  loadSessions();   // the turn may have created or retitled a session
}

function setBusy(b) {
  S.busy = b;
  $("#send").textContent = b ? "停止" : "发送";
  $("#send").classList.toggle("stop", b);
  if (b) {
    S.startedAt = Date.now();
    if (!S.timer) S.timer = setInterval(tick, 1000);
    tick();
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
  $("#sess-count").textContent = rows.length ? String(rows.length) : "";
  rows.forEach((s) => {
    const li = el("li", s.id === j.current ? "sess cur" : "sess");
    li.append(el("div", "t", s.title || "(未命名)"), el("div", "p", s.preview || ""));
    li.onclick = () => openSession(s.id);
    const del = el("button", "del", "×");
    del.title = "删除会话";
    del.onclick = (e) => { e.stopPropagation(); removeSession(s.id); };
    li.appendChild(del);
    ul.appendChild(li);
  });
}

async function openSession(id) {
  if (S.busy) { status("先等待或停止当前回复"); return; }
  $("#messages").textContent = "";
  S.tools.clear(); S.seg = null;
  let j;
  try {
    j = await (await fetch("/api/session/load", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    })).json();
  } catch (_) { status("载入会话失败"); return; }
  if (j.error) { status(`载入失败: ${j.error}`); return; }
  (j.history || []).forEach(apply);   // history carries no seq — a finished transcript
  finalizeSeg();
  status("就绪");
  loadSessions();
}

async function removeSession(id) {
  await fetch(`/api/session/${encodeURIComponent(id)}`, { method: "DELETE" }).catch(() => {});
  loadSessions();
}

async function newSession() {
  if (S.busy) { status("先等待或停止当前回复"); return; }
  await fetch("/api/session/new", { method: "POST" }).catch(() => {});
  $("#messages").textContent = "";
  S.seg = null; S.tools.clear();
  status("就绪");
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
  input.value = ""; input.style.height = "auto";
  addMsg("user", text);
  S.seg = null;
  let j;
  try {
    j = await (await fetch("/api/chat/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })).json();
  } catch (_) { status("发送失败"); return; }
  if (j.error) { status(j.error); return; }
  if (!j.streamId) { status("没有可用的会话流"); return; }
  remember(j.streamId, 0);
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
        if (st.running) status("接回上一次未完成的回复…");
        attach(id, seq);   // finished-but-unseen still needs its tail collected
      } else forget();
    } catch (_) { forget(); }
  }

  $("#new-session").onclick = newSession;
  $("#send").onclick = (e) => {
    if (S.busy) { e.preventDefault(); fetch("/api/chat/cancel", { method: "POST" }).catch(() => {}); }
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
