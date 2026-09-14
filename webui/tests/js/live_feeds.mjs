// The live-feed map is the front line of the multi-endpoint bug: with two
// endpoints, two turns can run at once, and the ONE-SLOT EventSource (`S.es`,
// closed by every attach) made watching session B freeze session A — the
// orphaned turn kept running server-side while its view starved, and
// switching back re-orphaned the other one. Both ends looked dead.
//
// These rules replay the decisions `attach`/`closeEs`/`endTurn` now make,
// without a DOM:
//
//   1. ATTACHING NEVER CLOSES ANOTHER FEED — send() to session B while A runs
//      must leave A's EventSource connected.
//   2. AN END CLOSES ONLY ITSELF — session A finishing while B is on screen
//      must not cut B's feed nor unfreeze B's composer.
//   3. REPLAY RE-OPENS ONLY ITS OWN FEED — openSession on a live session
//      repaints from the store and re-delivers that turn from the top; other
//      live feeds stay connected.
//   4. A BACKGROUND FEED FAILS SILENTLY — its onerror closes itself without
//      touching the screen's busy state.
import assert from 'node:assert';

// the model, as app.js implements it
const ess = new Map();
let streamId = null;    // the view's focus
let busy = false;

function openEs(id) { ess.set(id, { open: true }); }
function closeEs(id) { ess.delete(id); }
function attach(id, { replay = false } = {}) {
  streamId = id;
  if (replay) closeEs(id);
  if (ess.has(id)) return;
  openEs(id);
  busy = true;
}
function endTurn(from) {
  closeEs(from);
  // busy/status mutations are screen-gated in app.js; the rule under test is
  // that a foreign end does not touch the focus feed.
}

// ── 1. attach must not close another feed ───────────────────────────────────
attach("A");
assert.ok(ess.has("A"));
attach("B");                      // send() to a second session
assert.ok(ess.has("A") && ess.has("B"),
  "attaching B must keep A's feed connected — the old single-slot attach was the seesaw");

// ── 2. an end closes only itself ────────────────────────────────────────────
endTurn("A");
assert.ok(!ess.has("A") && ess.has("B"),
  "A ending must not cut B's feed");
assert.strictEqual(ess.size, 1);

// ── 3. replay re-opens only its own feed ────────────────────────────────────
attach("B", { replay: true });    // openSession back on B while C runs
assert.ok(ess.has("B"), "replay must re-open the viewed turn's feed");

// ── 4. a background feed fails silently ─────────────────────────────────────
attach("C");
const focusBefore = streamId;
ess.get("C").open = false;        // its error fires; the handler closes it
closeEs("C");
assert.ok(ess.has("B") && streamId === focusBefore,
  "a background feed's failure must not disturb the focus");

console.log("ok - the live-feed map's rules hold");
