#!/usr/bin/env bash
# Step 2: sweep existing vLLM scheduler settings.
#   token budget (--max-num-batched-tokens): 512, 1024, 2048
#   concurrency  (--max-num-seqs):           8, 16
# Each config gets a fresh vLLM server; results go to results/sweep/<config>/.
# Valid in vLLM 0.32: budget >= max-num-seqs; budgets below max-model-len (4096)
# rely on chunked prefill, which is on by default.
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

BUDGETS=(512 1024 2048)
SEQS=(8 16)
RATES=(0.5 1 2 4 6)
PORT=8000
SWEEP_DIR="${SWEEP_DIR:-results/sweep}"  # e.g. SWEEP_DIR=results/sweep-mon ./sweep.sh
export SWEEP_DIR                          # plot_sweep.py reads it too
mon=""

if curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null; then
  echo "Something is already serving on port $PORT. Stop your vLLM server first." >&2
  exit 1
fi

for budget in "${BUDGETS[@]}"; do
  for seqs in "${SEQS[@]}"; do
    cfg="tok${budget}-seq${seqs}"
    out="$SWEEP_DIR/$cfg"
    if [[ -f "$out/stock-rate-${RATES[-1]}.jsonl" ]]; then
      echo "== $cfg already done, skipping"; continue
    fi
    mkdir -p "$out"
    echo "== $cfg: starting vLLM"

    vllm serve Qwen/Qwen2.5-1.5B-Instruct \
      --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
      --tokenizer-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
      --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.80 \
      --max-num-batched-tokens "$budget" --max-num-seqs "$seqs" \
      --port "$PORT" > "$out/server.log" 2>&1 &
    server=$!
    trap 'kill $mon $server 2>/dev/null; wait $server 2>/dev/null' EXIT

    until curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null; do
      if ! kill -0 "$server" 2>/dev/null; then
        echo "vLLM failed to start for $cfg; see $out/server.log" >&2; exit 1
      fi
      sleep 2
    done

    for rate in "${RATES[@]}"; do
      echo "-- $cfg rate $rate"
      python monitor.py --output "$out/monitor-rate-${rate}.csv" \
        --pid "$server" --port "$PORT" &
      mon=$!
      python lab2.py replay --trace "traces/mixed-rate-${rate}.jsonl" \
        --output "$out/stock-rate-${rate}.jsonl" --run-tag "$cfg-$rate"
      kill "$mon"; wait "$mon" 2>/dev/null || true
    done

    kill "$server"; wait "$server" 2>/dev/null || true
    trap - EXIT
  done
done

python plot_sweep.py
echo ALL_DONE
