#!/usr/bin/env bash
# Start N vLLM workers of the same model on one GPU, for router.py.
#   ./workers.sh                  # 2 workers on ports 8001, 8002
#   N=3 MEM=0.25 ./workers.sh
# Ctrl+C stops them all. Logs: results/workers/worker-<port>.log
# Workers start one at a time: each profiles free GPU memory at startup.
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

N="${N:-2}"
MEM="${MEM:-0.38}"                      # fraction of the whole GPU, per worker
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE_PORT="${BASE_PORT:-8001}"
LOG_DIR="${LOG_DIR:-results/workers}"
mkdir -p "$LOG_DIR"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  log="$LOG_DIR/worker-$port.log"
  echo "== starting worker $i on port $port"
  vllm serve "$MODEL" --dtype float16 --max-model-len 4096 \
    --gpu-memory-utilization "$MEM" --max-num-seqs 16 \
    --enable-prefix-caching --port "$port" > "$log" 2>&1 &
  pids+=($!)
  until curl -s -m 2 "localhost:$port/v1/models" >/dev/null; do
    if ! kill -0 "${pids[-1]}" 2>/dev/null; then
      echo "worker on port $port died; see $log" >&2; exit 1
    fi
    sleep 2
  done
  grep -m1 "GPU KV cache size" "$log" || true
done

echo "== all $N workers up. Ctrl+C to stop."
wait
