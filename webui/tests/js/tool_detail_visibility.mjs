// A tool row's detail is collapsed by default -- ten successful calls should
// read as a compact activity list, which is the whole point of that panel.
//
// But a row that says only "failed" makes the reader hunt for the row to click
// before they learn why, and the reason is sitting right there. So a failure
// opens itself. The rule has to survive the several updates one tool emits, and
// it must never override a reader who has already made a choice: a detail they
// closed must stay closed, including when a later update re-renders it.
//
// With ABLATE=1 the "touched" guard is off and a closed failure springs back
// open under the reader's hands.
import assert from 'node:assert';

const GUARD = process.env.ABLATE !== "1";

function makeRow() {
  const d = { hidden: true, dataset: {} };
  return {
    d,
    click() { d.hidden = !d.hidden; d.dataset.touched = "1"; },
    // mirrors the tail of toolRow(): render, then the failure rule
    update(st) {
      if (st === "failed" && (!GUARD || d.dataset.touched !== "1")) d.hidden = false;
    },
  };
}

// 1. A successful call stays compact.
let r = makeRow();
r.update("running");
r.update("completed");
assert.equal(r.d.hidden, true, "a successful tool should not expand itself");

// 2. A failure shows its reason without being clicked.
r = makeRow();
r.update("running");
r.update("failed");
assert.equal(r.d.hidden, false, "a failed tool must show why");

// 3. A reader who closes a failure keeps it closed, through later updates.
r = makeRow();
r.update("failed");
r.click();                       // reader closes it
assert.equal(r.d.hidden, true);
r.update("failed");              // a later update re-renders the same row
assert.equal(r.d.hidden, true,
  "a detail the reader closed must not spring back open");

// 4. A reader who opens a successful call keeps it open.
r = makeRow();
r.update("completed");
r.click();
assert.equal(r.d.hidden, false);
r.update("completed");
assert.equal(r.d.hidden, false, "an opened detail must survive re-render");

console.log("ok");
