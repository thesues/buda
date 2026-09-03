// A loaded session delivers every token in ONE synchronous pass, so the only
// render is the deferred one -- and finalizeSeg() used to clear the segment the
// deferred render needs, which made the whole answer vanish. The same ordering
// silently dropped a live turn's closing sentences.
//
// This replays a real transcript through the render logic with the rAF NEVER
// firing, which is the worst case and the one that was broken.
import assert from 'node:assert';

const FLUSH_ON_FINALIZE = process.env.ABLATE !== "1";

const hist = [
  { kind: 'history_user', text: '详细说说剃度' },
  { kind: 'delta', thought: true, text: 'let me search' },
  { kind: 'tool', id: 't1', status: 'completed' },
  { kind: 'delta', thought: false, text: '关于剃度，' },
  { kind: 'delta', thought: false, text: '语料库中的依据如下。' },
];

const bubbles = [];
const renderMD = (s) => s;
let seg = null;
const newOut = () => { const b = { innerHTML: '' }; bubbles.push(b); seg = { kind: 'out', body: b, text: '' }; };
const newThink = () => { seg = { kind: 'think', text: '' }; };
function finalize() {
  if (!seg) return;
  if (FLUSH_ON_FINALIZE && seg.kind === 'out') seg.body.innerHTML = renderMD(seg.text);
  seg = null;
}
for (const e of hist) {
  if (e.kind === 'delta' && !(e.text || '').length) continue;
  if (e.kind === 'delta') {
    const want = e.thought ? 'think' : 'out';
    if (!seg || seg.kind !== want) { finalize(); (want === 'think' ? newThink : newOut)(); }
    seg.text += e.text;               // scheduleRender()'s rAF never runs here
  } else finalize();
}
finalize();

assert.strictEqual(bubbles.length, 1, 'one assistant bubble');
assert.strictEqual(bubbles[0].innerHTML, '关于剃度，语料库中的依据如下。',
  'the whole answer must be painted; an empty bubble is the bug this pins');
console.log('ok - a loaded transcript renders in full');
