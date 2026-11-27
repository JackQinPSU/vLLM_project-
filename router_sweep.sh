#!/usr/bin/env bash
# Compare routing policies on shared-prefix traces.
# Start the workers first, in another terminal:  ./workers.sh
#   ./router_sweep.sh
#   POLICIES="round_robin kv" RATES="2 4" ./router_sweep.sh
# Then:  python router_summary.py
# Results per trace: results/router-sweep/<trace>/<policy>.*
#   .jsonl        per-request results (lab2.py replay)
#   .stats.json   router counters (routed per worker, estimated hit rate)
#   .vllm.json    vLLM prefix-cache counters per worker, before/after the run
#   .route.jsonl  every routing decision
# Finished runs are skipped, so the script can be re-run after a crash.
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

POLICIES="${POLICIES:-round_robin random least_loaded prefix kv}"
FAMILIES="${FAMILIES:-multiturn docs}"
RATES="${RATES:-2 4 8}"
N="${N:-120}"
DOC_GROUPS="${DOC_GROUPS:-8}"         # docs: number of documents
WORKERS="${WORKERS:-http://localhost:8001 http://localhost:8002}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TOKENIZER="${TOKENIZER:-$MODEL}"
PORT="${PORT:-8000}"
OUT="${OUT:-results/router-sweep}"
TRACE_DIR="${TRACE_DIR:-traces}"

for w in $WORKERS; do
  curl -s -m 2 "$w/v1/models" >/dev/null || { echo "worker $w is not up; run ./workers.sh first" >&2; exit 1; }
done
if curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null; then
  echo "port $PORT is busy; stop the running router first" >&2; exit 1
fi

# vLLM prefix-cache counters (tokens) per worker, as JSON [[hits, queries], ...]
scrape() {
  python - $WORKERS <<'EOF'
import json, re, sys, urllib.request
out = []
for url in sys.argv[1:]:
    text = urllib.request.urlopen(url + "/metrics", timeout=2).read().decode()
    get = lambda name: sum(float(v) for v in re.findall(rf"^{name}(?:{{[^}}]*}})? (\S+)$", text, re.M))
    out.append([get("vllm:prefix_cache_hits_total"), get("vllm:prefix_cache_queries_total")])
print(json.dumps(out))
EOF
}

router=""
trap 'kill $router 2>/dev/null || true' EXIT

for fam in $FAMILIES; do
  for rate in $RATES; do
    # Generate the trace once; same trace for every policy.
    if [[ $fam == docs ]]; then
      name="docs-g${DOC_GROUPS}-n${N}-rate-${rate}"; extra=(--groups "$DOC_GROUPS")
    else
      name="${fam}-n${N}-rate-${rate}"; extra=()
    fi
    trace="$TRACE_DIR/$name.jsonl"
    [[ -f $trace ]] || python lab2.py generate --family "$fam" --rate "$rate" --n "$N" "${extra[@]}" --output "$trace"
    out="$OUT/$name"; mkdir -p "$out"

    for policy in $POLICIES; do
      if [[ -f "$out/$policy.jsonl" ]]; then echo "== $name $policy already done, skipping"; continue; fi
      echo "== $name $policy"

      # Fresh router per run, so its cache guesses and counters start empty.
      rm -f "$out/$policy.route.jsonl"
      python router.py --workers $WORKERS --policy "$policy" --port "$PORT" \
        --tokenizer "$TOKENIZER" --log "$out/$policy.route.jsonl" > "$out/$policy.router.log" 2>&1 &
      router=$!
      until curl -s -m 1 "localhost:$PORT/router/stats" >/dev/null; do
        kill -0 "$router" 2>/dev/null || { echo "router failed; see $out/$policy.router.log" >&2; exit 1; }
        sleep 0.5
      done

      before=$(scrape)
      # Unique run tag: this run can't reuse cache left over from earlier runs.
      python lab2.py replay --trace "$trace" --output "$out/$policy.jsonl.tmp" \
        --model "$MODEL" --url "http://localhost:$PORT/v1/chat/completions" \
        --run-tag "$name-$policy-$(date +%s)" | grep -v '^Trace'
      after=$(scrape)

      curl -s "localhost:$PORT/router/stats" > "$out/$policy.stats.json"
      echo "{\"before\": $before, \"after\": $after}" > "$out/$policy.vllm.json"
      mv "$out/$policy.jsonl.tmp" "$out/$policy.jsonl"   # mark run as done
      kill "$router"; wait "$router" 2>/dev/null || true; router=""
    done
  done
done

python router_summary.py --dir "$OUT"
echo ALL_DONE
