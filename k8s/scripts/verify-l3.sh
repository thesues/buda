#!/usr/bin/env bash
# Does the L3 tier actually restore a prefix across a restart?
#
# The experiment is built so that a pass has one explanation. A prefix cache hit
# after a pod delete cannot have come from VRAM, because the VRAM went with the
# pod; and the ablation shows the same request missing entirely when the tier is
# switched off. Either half alone would be weak: the first could be some other
# cache, the second could be an unrelated regression.
#
# Reads only from freetoken-l3. `freetoken-0` is not touched.
set -euo pipefail
NS=autumn
POD=freetoken-l3-0
PROMPT_FILE=${PROMPT_FILE:-/tmp/l3-prompt.json}

say() { printf '\n=== %s ===\n' "$*"; }

# A long, unique prompt: long enough to span many pages, unique so no earlier
# run can be supplying the hit.
if [ ! -f "$PROMPT_FILE" ]; then
  python3 - "$PROMPT_FILE" <<'PY'
import json, random, sys
random.seed(20260904)
body = " ".join(
    f"第{i}条：这是一段用于填充上下文的语料文本，编号{random.randint(100000,999999)}。"
    for i in range(1200)
)
json.dump({"model": "dsv4-flash", "max_tokens": 8,
           "messages": [{"role": "user", "content": body + "\n\n请回答：好。"}]},
          open(sys.argv[1], "w"), ensure_ascii=False)
PY
fi

ask() {   # -> prints the served response, or the error
  kubectl -n "$NS" exec "$POD" -c freetoken -- sh -c \
    "cat > /tmp/p.json <<'EOF'
$(cat "$PROMPT_FILE")
EOF
     curl -s -m 600 -X POST http://127.0.0.1:1919/v1/chat/completions \
       -H 'content-type: application/json' --data-binary @/tmp/p.json" | head -c 200
}

cached() {  # the last prefill batch's cached-token count
  kubectl -n "$NS" logs "$POD" -c freetoken --tail=400 2>/dev/null \
    | grep -oE '#cached-token: [0-9]+' | tail -1 | grep -oE '[0-9]+' || echo 0
}

wait_serving() {
  for _ in $(seq 1 60); do
    if kubectl -n "$NS" exec "$POD" -c freetoken -- sh -c \
        "curl -s -m 20 -X POST http://127.0.0.1:1919/v1/chat/completions \
         -H 'content-type: application/json' \
         -d '{\"model\":\"dsv4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}'" \
        2>/dev/null | grep -q '"choices"'; then return 0; fi
    sleep 20
  done
  echo "  the pod never started serving"; return 1
}

say "1. warm: send the prompt so its prefix is written to autumn"
wait_serving
ask >/dev/null
echo "  cached-token on the cold run: $(cached)   (expected 0 — nothing was stored yet)"
sleep 10   # the write is asynchronous; give the queue a moment

say "2. delete the pod — VRAM goes with it, autumn keeps the pages"
kubectl -n "$NS" delete pod "$POD" --wait=true >/dev/null
kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=900s >/dev/null
wait_serving

say "3. send the SAME prompt to the replacement"
ask >/dev/null
HIT=$(cached)
echo "  cached-token after the restart: $HIT"

say "RESULT"
if [ "$HIT" -gt 0 ]; then
  echo "  PASS — L1 was empty after the restart, so this hit came from L3."
  echo "  Now run the ablation: remove --hicache-storage-backend from the manifest,"
  echo "  re-apply, and repeat. cached-token must return to 0. A pass here without"
  echo "  that is only half the argument."
else
  echo "  FAIL — nothing was restored. Look for, in order:"
  echo "    L3 tier attached      : kubectl -n $NS logs $POD -c freetoken | grep 'L3 tier attached'"
  echo "    write outcomes        : ... | grep -i 'L3 write'"
  echo "    prefetch/adoption     : ... | grep -iE 'L3 prefetch|adoption failed'"
  echo "  'L3 tier attached' missing means the backend never built — that is a"
  echo "  configuration failure, not a cache miss, and the log says why."
fi
