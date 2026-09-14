// A turn running in one conversation must survive the reader starting another.
//
// Reported from production: session A (DSV4) is replying, the reader clicks
// 新会话, types, and presses 发送 — and A's reply is gone, its row left at
// "1 条" (the prompt, no answer). The button looked like 发送, but its click
// handler branched on `S.busy`, a TAB-wide flag still raised by A's turn, so it
// ran `cancelTurn()` aimed at A's stream instead of letting the form submit.
//
// Unlike the other files here this runs the REAL `static/app.js`, in a vm, over
// a DOM just large enough for it, a fake server and fake EventSources. The
// source-grep tests all passed while this bug shipped; only running the page's
// own handlers catches which one a click actually reaches.
import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(here, "..", "..", "static", "app.js"), "utf8");

/* ---------- a DOM just large enough for app.js ---------- */
class ClassList {
  constructor(node) { this.node = node; }
  get set() { return new Set(this.node.className.split(/\s+/).filter(Boolean)); }
  write(s) { this.node.className = [...s].join(" "); }
  add(...c) { const s = this.set; c.forEach((x) => s.add(x)); this.write(s); }
  remove(...c) { const s = this.set; c.forEach((x) => s.delete(x)); this.write(s); }
  contains(c) { return this.set.has(c); }
  toggle(c, on) {
    const s = this.set; const want = on === undefined ? !s.has(c) : !!on;
    if (want) s.add(c); else s.delete(c);
    this.write(s); return want;
  }
}

class Node_ {
  constructor(tag) {
    this.tagName = (tag || "div").toUpperCase();
    this.children = []; this.parent = null;
    this.className = ""; this.id = ""; this.dataset = {}; this.style = {};
    this.hidden = false; this.value = ""; this.placeholder = ""; this.disabled = false;
    this.title = ""; this.onclick = null; this._text = ""; this._attrs = {};
    this.listeners = {}; this.classList = new ClassList(this);
    this.scrollHeight = 0; this.scrollTop = 0; this.clientHeight = 0;
  }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(t) { this.children.forEach((c) => { c.parent = null; }); this.children = []; this._text = t == null ? "" : String(t); }
  get innerHTML() { return this.textContent; }
  set innerHTML(h) { this.textContent = h; }
  get firstChild() { return this.children[0] || null; }
  get nextElementSibling() {
    if (!this.parent) return null;
    const sib = this.parent.children; return sib[sib.indexOf(this) + 1] || null;
  }
  appendChild(n) { if (n.parent) n.remove(); n.parent = this; this.children.push(n); return n; }
  append(...ns) { ns.forEach((n) => this.appendChild(n)); }
  replaceChildren(...ns) { this.textContent = ""; this.append(...ns); }
  after(n) { if (n.parent) n.remove(); const sib = this.parent.children; sib.splice(sib.indexOf(this) + 1, 0, n); n.parent = this.parent; }
  remove() { if (this.parent) { const s = this.parent.children; s.splice(s.indexOf(this), 1); this.parent = null; } }
  contains(n) { for (let x = n; x; x = x.parent) if (x === this) return true; return false; }
  closest(sel) { for (let x = this; x; x = x.parent) if (matches(x, sel)) return x; return null; }
  setAttribute(k, v) { this._attrs[k] = String(v); }
  getAttribute(k) { return this._attrs[k] ?? null; }
  removeAttribute(k) { delete this._attrs[k]; if (k.startsWith("data-")) delete this.dataset[camel(k.slice(5))]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type, ev = {}) {
    ev.type = type; ev.target = ev.target || this;
    ev.defaultPrevented = false; ev.preventDefault = () => { ev.defaultPrevented = true; };
    ev.stopPropagation = () => {};
    if (type === "click" && this.onclick) this.onclick(ev);
    (this.listeners[type] || []).forEach((fn) => fn(ev));
    return ev;
  }
  click() {
    const ev = this.dispatch("click");
    // A submit button's default action: submit its form.
    if (!ev.defaultPrevented && this._attrs.type === "submit") {
      let f = this.parent; while (f && f.tagName !== "FORM") f = f.parent;
      if (f) f.dispatch("submit");
    }
  }
  scrollIntoView() {}
  descendants() { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  querySelectorAll(sel) {
    const parts = sel.trim().split(/\s+/);
    return this.descendants().filter((n) => {
      if (!matches(n, parts[parts.length - 1])) return false;
      let i = parts.length - 2;
      for (let a = n.parent; a && i >= 0; a = a.parent) if (matches(a, parts[i])) i--;
      return i < 0;
    });
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
function matches(n, sel) {
  const m = sel.match(/^([a-z]+)?(#[\w-]+)?((?:\.[\w-]+)*)(?:\[([\w-]+)(?:="([^"]*)")?\])?$/i);
  if (!m) throw new Error(`selector not supported by the test DOM: ${sel}`);
  const [, tag, id, cls, attr, val] = m;
  if (tag && n.tagName !== tag.toUpperCase()) return false;
  if (id && n.id !== id.slice(1)) return false;
  if (cls && !cls.split(".").filter(Boolean).every((c) => n.classList.contains(c))) return false;
  if (attr) {
    const v = attr.startsWith("data-") ? n.dataset[camel(attr.slice(5))] : n._attrs[attr];
    if (v === undefined) return false;
    if (val !== undefined && String(v) !== val) return false;
  }
  return true;
}

function page() {
  const body = new Node_("body");
  const mk = (tag, id, cls, parent, attrs = {}) => {
    const n = new Node_(tag); n.id = id || ""; n.className = cls || "";
    Object.assign(n._attrs, attrs); parent.appendChild(n); return n;
  };
  mk("button", "new-session", "btn", body);
  mk("button", "about-btn", "", body);
  mk("span", "mcp-badge", "", body);
  mk("span", "sess-count", "", body);
  mk("ul", "sessions", "", body);
  const chat = mk("section", "", "card chat", body);
  mk("span", "spin", "", chat); mk("span", "run-status", "", chat); mk("span", "elapsed", "", chat);
  mk("div", "messages", "", chat);
  const form = mk("form", "composer", "", chat);
  mk("span", "model-badge", "", form).hidden = true;
  mk("select", "endpoint", "", form).hidden = true;
  mk("textarea", "input", "", form).placeholder = "发消息…";
  mk("button", "send", "btn", form, { type: "submit" });
  const about = mk("div", "about", "", body); about.hidden = true;
  mk("button", "about-close", "", about); mk("div", "about-body", "", about);

  const docListeners = {};
  const document = {
    body,
    querySelector: (s) => body.querySelector(s),
    querySelectorAll: (s) => body.querySelectorAll(s),
    createElement: (t) => new Node_(t),
    addEventListener: (t, fn) => { (docListeners[t] ||= []).push(fn); },
  };
  return { document, body };
}

/* ---------- a fake server ---------- */
function harness({ current = null, streaming = {}, store = new Map(), server: shared = null } = {}) {
  const { document } = page();
  const calls = [];
  const sources = [];
  // Pass `store` and `server` from a previous harness to model a RELOAD: same
  // localStorage, same server, fresh page.
  const server = shared || {
    current, streaming: { ...streaming }, sessions: [], history: {}, events: {},
    endpoints: [
      { key: "dsv4", label: "DSV4", model: "dsv4", maxConcurrent: 1, running: 0 },
      { key: "mm2", label: "MM2", model: "mm2", maxConcurrent: 1, running: 0 },
    ],
    nextStream: 1,
  };
  // A JSON round trip, like the wire: handing the client the server's own
  // objects let a server-side change mutate client state behind its back.
  const reply = (j) => Promise.resolve({ ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(j)), text: async () => "" });
  const fetch = (url, opts = {}) => {
    const u = new URL(String(url), "http://x");
    const body = opts.body ? JSON.parse(opts.body) : null;
    calls.push({ path: u.pathname, query: Object.fromEntries(u.searchParams), body });
    switch (u.pathname) {
      case "/api/sessions":
        return reply({ sessions: server.sessions, current: server.current, streaming: server.streaming, endpoints: server.endpoints });
      case "/api/status":
        return reply({ endpoints: server.endpoints, defaultEndpoint: "dsv4" });
      case "/api/chat/start": return (server.startGate || Promise.resolve()).then(() => {
        const sid = body.new || !body.sessionId ? `new-${server.nextStream}` : body.sessionId;
        const stream = `s${server.nextStream++}`;
        server.streaming[sid] = stream;
        server.current = sid;
        if (!server.sessions.find((r) => r.id === sid)) server.sessions.unshift({ id: sid, title: body.text, messageCount: 0 });
        return reply({ streamId: stream, sessionId: sid, endpoint: body.endpoint });
      });
      case "/api/chat/cancel": return reply({ ok: true });
      case "/api/chat/status": {
        const id = u.searchParams.get("stream_id");
        return reply({ known: true, running: Object.values(server.streaming).includes(id) });
      }
      case "/api/session/history": return reply({ events: server.history[u.searchParams.get("id")] || [] });
      case "/api/approval/pending": return reply({ pending: [] });
      default: return reply({});
    }
  };
  // Like TurnStream: a stream keeps its events, and a (re)connect replays
  // everything after `after_seq` before following.
  class EventSource {
    constructor(url) {
      this.url = url; this.closed = false; sources.push(this);
      const q = new URL(url, "http://x").searchParams;
      const id = q.get("stream_id"), after = Number(q.get("after_seq") || 0);
      setTimeout(() => (server.events[id] || []).filter((e) => e.seq > after)
        .forEach((e) => { if (!this.closed) this.onmessage({ data: JSON.stringify(e) }); }), 0);
    }
    close() { this.closed = true; }
  }
  const ctx = vm.createContext({
    document, fetch, EventSource, console, URL, JSON, Math, Date, Promise, Map, Set, Object, Array, String, Number,
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
      removeItem: (k) => store.delete(k),
    },
    setTimeout, clearTimeout,
    setInterval: () => 0, clearInterval: () => {},     // no background timers
    requestAnimationFrame: (fn) => setTimeout(fn, 0),
    confirm: () => true,
    marked: { parse: (s) => s },
    DOMPurify: { sanitize: (s) => s, addHook: () => {} },
  });
  vm.runInContext(SRC, ctx);
  const $ = (s) => document.querySelector(s);
  const S = () => vm.runInContext("S", ctx);
  const push = (streamId, ev) => {
    (server.events[streamId] ||= []).push(ev);
    sources.filter((s) => !s.closed && s.url.includes(`stream_id=${streamId}&`))
      .forEach((s) => s.onmessage({ data: JSON.stringify(ev) }));
  };
  // The turn ends server-side: its end event, then the store holds the
  // transcript and the session stops streaming.
  const finish = (sid, streamId, question, answer) => {
    const seq = (server.events[streamId] || []).length + 1;
    delete server.streaming[sid];
    server.history[sid] = [{ kind: "history_user", text: question }, { kind: "delta", text: answer, thought: false }];
    const row = server.sessions.find((r) => r.id === sid); if (row) row.messageCount = 2;
    push(streamId, { kind: "end", error: null, seq, session: sid });
  };
  const type = (text) => { $("#input").value = text; };
  const pick = (key) => { $("#endpoint").value = key; $("#endpoint").dispatch("change", { target: $("#endpoint") }); };
  return { ctx, $, S, calls, sources, server, store, push, finish, type, pick, run: (code) => vm.runInContext(code, ctx) };
}
const settle = () => new Promise((r) => setTimeout(r, 20));
const posts = (h, p) => h.calls.filter((c) => c.path === p);

/* ---------- 1. 发送 in a new session does not stop the bystander ---------- */
{
  const h = harness();
  await settle();
  h.pick("dsv4");
  h.type("hello,ds");
  h.$("#send").click();
  await settle();
  const [a] = posts(h, "/api/chat/start");
  assert.ok(a, "the first send must reach the server");
  const sidA = h.S().sessionId, streamA = h.S().streamId;
  h.push(streamA, { kind: "delta", text: "苦谛…", seq: 2, session: sidA });

  h.$("#new-session").click();
  await settle();
  h.pick("mm2");
  h.type("hello mini");
  h.$("#send").click();
  await settle();

  assert.deepStrictEqual(posts(h, "/api/chat/cancel"), [],
    "发送 in a new session cancelled the turn still running in another one");
  const starts = posts(h, "/api/chat/start");
  assert.strictEqual(starts.length, 2, "the new session's message was never sent");
  assert.strictEqual(starts[1].body.new, true, "the second send must open a new conversation");
  assert.strictEqual(starts[1].body.endpoint, "mm2");
  assert.ok(!h.sources.find((s) => s.url.includes(`stream_id=${streamA}&`)).closed,
    "the bystander's feed must stay connected");

  // Enter goes through the same gate as the button.
  const h2 = harness();
  await settle();
  h2.type("A"); h2.$("#send").click(); await settle();
  h2.$("#new-session").click(); await settle();
  h2.pick("mm2");
  h2.type("B");
  h2.$("#input").dispatch("keydown", { key: "Enter", shiftKey: false, isComposing: false, keyCode: 13 });
  await settle();
  assert.strictEqual(posts(h2, "/api/chat/start").length, 2,
    "Enter in a new session was silently swallowed while another session replied");
  assert.deepStrictEqual(posts(h2, "/api/chat/cancel"), []);
}

/* ---------- 2. the bystander's frames do not move the focus cursor ---------- */
{
  const h = harness();
  await settle();
  h.pick("dsv4"); h.type("A"); h.$("#send").click(); await settle();
  const sidA = h.S().sessionId, streamA = h.S().streamId;
  h.$("#new-session").click(); await settle();
  h.pick("mm2"); h.type("B"); h.$("#send").click(); await settle();
  const sidB = h.S().sessionId, streamB = h.S().streamId;
  assert.notStrictEqual(streamA, streamB);
  h.push(streamB, { kind: "delta", text: "b-token", seq: 3, session: sidB });
  h.push(streamA, { kind: "delta", text: "a-token", seq: 40, session: sidA });
  await settle();
  assert.strictEqual(h.S().lastSeq, 3,
    "a background turn's seq became the focus stream's cursor");
  assert.ok(!h.$("#messages").textContent.includes("a-token"),
    "a background turn's tokens were drawn into the conversation on screen");
}

/* ---------- 3. the bystander ending does not repaint the fresh view ---------- */
{
  const h = harness();
  await settle();
  h.pick("dsv4"); h.type("A"); h.$("#send").click(); await settle();
  const sidA = h.S().sessionId, streamA = h.S().streamId;
  h.$("#new-session").click(); await settle();
  h.push(streamA, { kind: "end", error: "boom", seq: 9, session: sidA });
  await settle();
  assert.ok(!h.$("#messages").querySelector(".msg.error"),
    "another session's error was painted into the new, empty conversation");
  assert.ok(h.$(".chat").classList.contains("fresh"), "the fresh view was disturbed");
}

/* ---------- 4. a page that opens on 新的对话 sends a NEW conversation ---------- */
{
  // The server remembers this browser's last session. The page shows the fresh
  // hero, so what is typed must not be appended to that old conversation.
  const h = harness({ current: "old-session" });
  await settle();
  assert.ok(h.$(".chat").classList.contains("fresh"));
  h.type("hi"); h.$("#send").click(); await settle();
  const [start] = posts(h, "/api/chat/start");
  assert.ok(start.body.new === true && !start.body.sessionId,
    `a page showing 新的对话 sent into ${JSON.stringify(start.body.sessionId)}`);
}

/* ---------- 5. a reload shows the conversation, however many times ---------- */
{
  // Production, after the first fix: two conversations answered, the page
  // reloaded a few times — then a reload came up with a row highlighted and
  // an EMPTY transcript. A reload must repaint what the reader was looking at
  // from the store, every time, and follow a turn that is still running.
  const text = (h) => h.$("#messages").textContent;
  const reload = async (prev) => { const h = harness({ store: prev.store, server: prev.server }); await settle(); await settle(); return h; };

  let h = harness();
  await settle();
  h.pick("dsv4"); h.type("hello A"); h.$("#send").click(); await settle();
  const sidA = h.S().sessionId, streamA = h.S().streamId;
  h.push(streamA, { kind: "user", text: "hello A", seq: 1, session: sidA });
  h.push(streamA, { kind: "delta", text: "answer-A", seq: 2, session: sidA });
  h.$("#new-session").click(); await settle();
  h.pick("mm2"); h.type("hello B"); h.$("#send").click(); await settle();
  const sidB = h.S().sessionId, streamB = h.S().streamId;
  h.push(streamB, { kind: "user", text: "hello B", seq: 1, session: sidB });
  h.push(streamB, { kind: "delta", text: "partial-B", seq: 2, session: sidB });

  // Reload while B is mid-answer, A too.
  h = await reload(h);
  assert.ok(text(h).includes("hello B") && text(h).includes("partial-B"),
    `a reload mid-turn lost the conversation on screen: ${JSON.stringify(text(h))}`);
  assert.ok(!text(h).includes("answer-A"), "a reload painted the other conversation's reply");
  assert.strictEqual(h.S().sessionId, sidB);

  // Both finish while this page is open; then reload, several times.
  h.finish(sidA, streamA, "hello A", "answer-A");
  h.push(streamB, { kind: "delta", text: "-done", seq: 3, session: sidB });
  h.finish(sidB, streamB, "hello B", "partial-B-done");
  await settle();
  for (let i = 1; i <= 4; i++) {
    h = await reload(h);
    const shown = text(h);
    const cur = h.$("#sessions").querySelector("li.cur");
    assert.ok(shown.includes("hello B") && shown.includes("partial-B-done"),
      `reload #${i} came up empty (row ${cur ? "highlighted" : "not highlighted"}): ${JSON.stringify(shown)}`);
    assert.strictEqual(h.S().sessionId, sidB, `reload #${i} is looking at the wrong conversation`);
    assert.ok(!h.S().busy, `reload #${i} believes a finished turn is still running`);
  }

  // Open A, reload: A comes back.
  h.run(`openSession(${JSON.stringify(sidA)})`); await settle();
  h = await reload(h);
  assert.ok(text(h).includes("answer-A"), `reload after opening A: ${JSON.stringify(text(h))}`);

  // 新会话 then reload: the fresh view, not the last conversation.
  h.$("#new-session").click(); await settle();
  h = await reload(h);
  assert.ok(h.$(".chat").classList.contains("fresh") && !h.S().sessionId,
    "a reload after 新会话 reopened an old conversation");
}

/* ---------- 6. clicking a conversation in the sidebar draws it ---------- */
{
  const h = harness();
  h.server.sessions.push({ id: "old", title: "old question", messageCount: 2 });
  h.server.history.old = [
    { kind: "history_user", text: "old question" },
    { kind: "delta", text: "old answer", thought: false },
  ];
  await settle();
  h.run("loadSessions()"); await settle();
  const row = h.$("#sessions").querySelector("li");
  row.onclick(); await settle(); await settle();
  const shown = h.$("#messages").textContent;
  assert.ok(shown.includes("old question") && shown.includes("old answer"),
    `opening a conversation drew ${JSON.stringify(shown)} — history replays were dropped as foreign frames`);

  // 新会话, then back to the old conversation, then send: it continues THAT one.
  h.$("#new-session").click(); await settle();
  h.$("#sessions").querySelector("li").onclick(); await settle(); await settle();
  h.type("follow-up"); h.$("#send").click(); await settle();
  const last = posts(h, "/api/chat/start").at(-1);
  assert.ok(last.body.sessionId === "old" && !last.body.new,
    `a send after 新会话 → reopen went to ${JSON.stringify(last.body)}`);
}

/* ---------- 7. leaving a live session does not get pulled back to it ---------- */
{
  // Production: "hello dsv4" replying, the reader clicks "hello mini" — the
  // dsv4 tokens kept drawing under mini's transcript and the view jumped back
  // to dsv4. The rotation rule in apply() adopts a frame's session when it
  // arrives on the FOCUS stream, and leaving a session never cleared the focus,
  // so the next dsv4 token read as "this conversation was renamed".
  const h = harness();
  h.server.sessions.push({ id: "mini", title: "hello mini", messageCount: 2 });
  h.server.history.mini = [
    { kind: "history_user", text: "hello mini" },
    { kind: "delta", text: "mini answer", thought: false },
  ];
  await settle();
  h.pick("dsv4"); h.type("hello dsv4"); h.$("#send").click(); await settle();
  const sidD = h.S().sessionId, streamD = h.S().streamId;
  h.push(streamD, { kind: "user", text: "hello dsv4", seq: 1, session: sidD });
  h.push(streamD, { kind: "delta", text: "dsv4-token-1", seq: 2, session: sidD });

  const rowFor = (title) => h.$("#sessions").querySelectorAll("li").find((li) => li.textContent.includes(title));
  h.run("loadSessions()"); await settle();
  rowFor("hello mini").onclick(); await settle(); await settle();
  h.push(streamD, { kind: "delta", text: "dsv4-token-2", seq: 3, session: sidD });
  await settle(); await settle();

  assert.strictEqual(h.S().sessionId, "mini", "the view was pulled back to the session still replying");
  const shown = h.$("#messages").textContent;
  assert.ok(shown.includes("mini answer") && !shown.includes("dsv4-token-2"),
    `another session's tokens were drawn into the one on screen: ${JSON.stringify(shown)}`);
  assert.strictEqual(h.store.get("hermes.view"), "mini");
  assert.ok(!h.S().owns, "the idle view offers 停止 for the other session's turn");

  // Back to dsv4: its live turn is replayed, and a real rotation on ITS
  // stream is still followed.
  rowFor("hello dsv4").onclick();
  // A token lands while the history read is still in flight: the replay from
  // seq 0 will deliver it, so drawing it now as well would show it twice.
  h.push(streamD, { kind: "delta", text: "|mid-switch|", seq: 4, session: sidD });
  await settle(); await settle();
  const back = h.$("#messages").textContent;
  assert.ok(back.includes("dsv4-token-2"), "returning to the live session lost its reply");
  assert.strictEqual(back.split("|mid-switch|").length - 1, 1,
    `a token that arrived during the switch was drawn twice: ${JSON.stringify(back)}`);
  h.push(streamD, { kind: "delta", text: "after-rotation", seq: 5, session: "rotated" });
  await settle();
  assert.strictEqual(h.S().sessionId, "rotated", "a rotation on the session's own stream is no longer followed");
  assert.ok(h.$("#messages").textContent.includes("after-rotation"));
}

/* ---------- 8. races and leftovers the pi review found ---------- */
const withOld = (h) => {
  h.server.sessions.push({ id: "old", title: "old question", messageCount: 2 });
  h.server.history.old = [
    { kind: "history_user", text: "old question" },
    { kind: "delta", text: "old answer", thought: false },
  ];
};
const rowOf = (h, title) => h.$("#sessions").querySelectorAll("li").find((li) => li.textContent.includes(title));
const count = (hay, needle) => hay.split(needle).length - 1;

// #1 an approval left behind must not lock the composer everywhere else
{
  const h = harness(); await settle();
  h.type("run a command"); h.$("#send").click(); await settle();
  const sid = h.S().sessionId, stream = h.S().streamId;
  h.push(stream, { kind: "approval", id: sid, title: "rm -rf /tmp/x", options: [], seq: 2, session: sid });
  assert.ok(h.S().awaitingPerm);
  h.$("#new-session").click(); await settle();
  h.pick("mm2"); h.type("elsewhere"); h.$("#send").click(); await settle();
  assert.strictEqual(posts(h, "/api/chat/start").length, 2,
    `a pending approval in another conversation blocked this send: ${h.$("#run-status").textContent}`);
}

// #4 a double-click on one row paints the transcript once
{
  const h = harness(); withOld(h); await settle();
  h.run("loadSessions()"); await settle();
  rowOf(h, "old question").onclick(); rowOf(h, "old question").onclick();
  await settle(); await settle();
  assert.strictEqual(count(h.$("#messages").textContent, "old answer"), 1,
    `a double-click drew the transcript twice: ${JSON.stringify(h.$("#messages").textContent)}`);
}

// #5 a stream the sidebar still lists as live, but which has finished, is not
//    replayed on top of the transcript that already contains its turn
{
  const h = harness(); await settle();
  h.type("q1"); h.$("#send").click(); await settle();
  const sid = h.S().sessionId, stream = h.S().streamId;
  h.push(stream, { kind: "user", text: "q1", seq: 1, session: sid });
  h.push(stream, { kind: "delta", text: "a1-final", seq: 2, session: sid });
  h.$("#new-session").click(); await settle();
  // The turn ends server-side between two sidebar polls: this page never hears
  // it (its feed dropped), and S.streaming still says the session is live.
  h.sources.filter((es) => es.url.includes(`stream_id=${stream}&`)).forEach((es) => es.close());
  delete h.server.streaming[sid];
  h.server.history[sid] = [{ kind: "history_user", text: "q1" }, { kind: "delta", text: "a1-final", thought: false }];
  h.server.events[stream].push({ kind: "end", error: null, seq: 3, session: sid });
  assert.ok(h.S().streaming[sid], "precondition: the client's streaming map is stale");
  rowOf(h, "q1").onclick(); await settle(); await settle();
  assert.strictEqual(count(h.$("#messages").textContent, "a1-final"), 1,
    `the finished turn was painted from the store AND replayed: ${JSON.stringify(h.$("#messages").textContent)}`);
  assert.ok(!h.S().busy, "a finished turn left the view busy");
}

// #6 a click during boot is not overridden by boot reopening the saved view
{
  const store = new Map([["hermes.view", "saved"]]);
  const h = harness({ store });
  withOld(h);
  h.server.sessions.push({ id: "saved", title: "saved question", messageCount: 2 });
  h.server.history.saved = [{ kind: "history_user", text: "saved question" }];
  h.run('openSession("old")');          // the reader clicks before boot settles
  await settle(); await settle();
  assert.strictEqual(h.S().sessionId, "old", "boot reopened the saved view over the reader's click");
  assert.ok(!h.$("#messages").textContent.includes("saved question"));
}

// #7 another session's end does not remove this view's 思考中 row
{
  const h = harness(); await settle();
  h.type("A"); h.$("#send").click(); await settle();
  const sidA = h.S().sessionId, streamA = h.S().streamId;
  h.$("#new-session").click(); await settle();
  h.pick("mm2"); h.type("B"); h.$("#send").click(); await settle();
  assert.ok(h.$("#pending"), "precondition: B shows its pending row");
  h.push(streamA, { kind: "end", error: null, seq: 5, session: sidA });
  await settle();
  assert.ok(h.$("#pending"), "a background turn's end removed this conversation's 思考中 row");
}

// #3 leaving the view while the send is in flight keeps its reply out of the new view
{
  const h = harness(); withOld(h); await settle();
  h.run("loadSessions()"); await settle();
  let release; h.server.startGate = new Promise((r) => { release = r; });
  h.type("asked from the fresh view"); h.$("#send").click();
  await settle();
  rowOf(h, "old question").onclick(); await settle(); await settle();
  release(); await settle(); await settle();
  assert.strictEqual(h.S().sessionId, "old", "the send's response took over the view the reader moved to");
  assert.strictEqual(h.store.get("hermes.view"), "old");
  const [start] = posts(h, "/api/chat/start");
  const stream = `s1`;
  h.push(stream, { kind: "delta", text: "reply-to-fresh", seq: 2, session: start && "new-1" });
  await settle();
  assert.ok(!h.$("#messages").textContent.includes("reply-to-fresh"),
    "the moved-away send's reply was drawn into the conversation on screen");
  assert.ok(!h.S().owns, "the view offers 停止 for a turn it does not show");
  // ...and the conversation it started is still reachable and live.
  h.run("loadSessions()"); await settle();
  assert.ok(h.S().streaming["new-1"], "the started turn vanished from the sidebar");

  // 新会话 during the send: the fresh view stays fresh and pending.
  const h2 = harness(); await settle();
  let rel2; h2.server.startGate = new Promise((r) => { rel2 = r; });
  h2.type("first"); h2.$("#send").click(); await settle();
  h2.$("#new-session").click(); await settle();
  rel2(); await settle(); await settle();
  assert.ok(h2.S().pendingNew && !h2.S().sessionId && h2.$(".chat").classList.contains("fresh"),
    "新会话 during a send was overridden by the send's response");
}

console.log("ok - a new session leaves the running one alone");
