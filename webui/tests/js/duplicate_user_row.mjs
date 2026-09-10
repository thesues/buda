// The prompt appeared twice on screen. Three independent painters put that row
// there -- send()'s optimistic echo, the stream's own `user` event, and the
// `history_user` replayed when hermes reloads a session -- and skipUserEcho is
// a one-shot boolean, so it cancels exactly one of them. Whichever two line up
// render the prompt twice.
//
// This replays that pairing against the tail rule that now guards the two
// stream-side painters. With ABLATE=1 the rule is off and the row doubles,
// which is what the screenshot showed.
import assert from 'node:assert';

const GUARD = process.env.ABLATE !== "1";

// A transcript is a list of rows; `pending` is the tail spinner, not something
// said, so the rule has to look past it the way the DOM version does.
const rows = [];
const addMsg = (role, text) => { rows.push({ role, text }); };
function addUserMsg(text) {
  if (GUARD) {
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i].role === "pending") continue;
      if (rows[i].role === "user" && rows[i].text === text) return;
      break;
    }
  }
  addMsg("user", text);
}

let skipUserEcho = false;

// send(): draws before the server has accepted it, and arms the one-shot.
addMsg("user", "知道了");
skipUserEcho = true;
rows.push({ role: "pending", text: "" });          // the tail activity row

// the stream's own echo of the turn -- the one the flag was meant for
if (skipUserEcho) skipUserEcho = false; else addUserMsg("知道了");

// hermes reloads the session and replays it; the flag is already spent
addUserMsg("知道了");

assert.strictEqual(rows.filter((r) => r.role === "user").length, 1,
  "the prompt must be drawn once, whichever two painters line up");

// The rule must not swallow a prompt that is genuinely repeated -- asking the
// same thing again after an answer is an ordinary turn, not a double paint.
addMsg("bot", "好。");
addUserMsg("知道了");
assert.strictEqual(rows.filter((r) => r.role === "user").length, 2,
  "a repeat after a reply is a real turn and must be drawn");

console.log("ok - the prompt is painted once no matter which painters line up");
