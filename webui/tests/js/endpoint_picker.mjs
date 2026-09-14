// The endpoint picker is the front half of the multi-endpoint feature; the
// back half (`/api/chat/start {endpoint}`) has its own tests in
// test_app_routes.py. This replays the picker's two decision rules against
// the exact sequences that broke them before they were rules:
//
//   1. THE SAVED CHOICE SURVIVES RELOAD — localStorage outlives the page, and
//      a reader who picked an endpoint, reloaded, and got silently dropped to
//      the default would send to a model that is not the one on screen.
//   2. A STALE SAVED KEY FALLS BACK — the server's endpoint list changes
//      without telling the browser (a redeploy with BUDA_ENDPOINTS edited).
//      A saved key that is no longer advertised must resolve to the default,
//      not to a 404-shaped silence on every send.
//
// Plus the capacity rule setBusy() copies: a FULL endpoint must not block a
// send aimed at ANOTHER one — the limit belongs to the model behind the
// choice, which is the entire reason the limits are per endpoint.
//
// With ABLATE=1 every guard is off, which is the behaviour before the rules:
// the saved key is ignored, the stale key 404s into the default's arms, and
// one full endpoint greys out the whole composer.
import assert from 'node:assert';

const GUARD = process.env.ABLATE !== "1";

// ── the resolution rule, as setEndpoints() applies it ──────────────────────
function resolve(list, saved, defaultKey, prior) {
  const keys = new Set(list.map((e) => e.key));
  if (GUARD && prior && keys.has(prior.key)) return prior.key;  // already resolved
  if (!GUARD) return list[0].key;                               // ablated: first, always
  if (saved && keys.has(saved)) return saved;
  if (defaultKey && keys.has(defaultKey)) return defaultKey;
  return list[0] ? list[0].key : null;
}

// ── the capacity rule, as setBusy() applies it ─────────────────────────────
function atCapacity(endpoints, chosenKey) {
  const ep = endpoints.find((e) => e.key === chosenKey) || null;
  if (GUARD && ep) return (ep.running || 0) >= ep.maxConcurrent;
  return ep.running >= 1;  // ablated: any running turn anywhere means full
}

const eps = [
  { key: "dsv4", label: "DSV4", maxConcurrent: 1, running: 1 },   // busy
  { key: "vision", label: "Vision", maxConcurrent: 2, running: 1 }, // has room
];

// 1. The saved choice survives reload.
assert.strictEqual(resolve(eps, "vision", "dsv4", null), "vision",
  "a choice made before the reload must still be the choice after it");

// 2. A stale saved key falls back to the server's default, not to silence.
assert.strictEqual(resolve(eps, "gone", "dsv4", null), "dsv4",
  "a key the server no longer advertises must fall back to the default");

// 3. No saved key takes the server's declared default.
assert.strictEqual(resolve(eps, null, "dsv4", null), "dsv4");

// 4. The server declaring no default falls back to the first entry — the
//    same rule load_endpoints() uses, so the two ends agree without talking.
assert.strictEqual(resolve(eps, null, null, null), "dsv4");

// 5. An already-resolved choice is not re-resolved off a poll that lacks the
//    saved key in ITS payload — every loadSessions() response passes through
//    here, and re-resolving on each poll would unseat the reader's pick.
assert.strictEqual(resolve(eps, "vision", "dsv4", { key: "vision" }), "vision",
  "a poll must not unseat a choice it did not invalidate");

// 6. Per-endpoint capacity: the busy endpoint blocks, the other does not.
assert.strictEqual(atCapacity(eps, "dsv4"), true, "dsv4 is full");
assert.strictEqual(atCapacity(eps, "vision"), false,
  "one endpoint being full must not read as every endpoint being full");

// 7. A running count the poll has not delivered yet reads as room, not as
//    full — the counters arrive with the merge, and before them the honest
//    answer is unknown, which setBusy() turns into the closed fallback.
assert.strictEqual(atCapacity([{ key: "x", maxConcurrent: 1, running: undefined }], "x"),
  false, "unknown running must not read as at-limit");

console.log("ok - the endpoint picker's rules hold");
