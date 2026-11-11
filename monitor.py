"""Sample GPU, CPU and vLLM scheduler state into a CSV until killed.

    python monitor.py --output results/x/monitor.csv [--pid SERVER_PID] [--port 8000]

Columns:
  t_s, unix_s         seconds since monitor start / wall clock
  gpu_util            % of the sample window a kernel was running (NVML; not SM occupancy)
  gpu_mem_util        % of the window the memory controller was busy
  gpu_mem_mb          VRAM in use (vLLM preallocates, so this is mostly flat)
  gpu_power_w, gpu_sm_mhz
  cpu_total           system-wide CPU %, averaged over all cores (0-100)
  cpu_vllm            CPU % of the vLLM server process tree (100 = one full core)
  running, waiting    vLLM scheduler queue (from /metrics)
  kv_cache            vLLM KV cache usage, 0-1
  queue_time_sum      cumulative seconds requests spent queued (vllm:request_queue_time_seconds)
  queue_time_count    cumulative requests that left the queue
  preemptions         cumulative preemptions (vllm:num_preemptions_total; KV cache ran out)
  gen_tokens          cumulative output tokens (vllm:generation_tokens_total)
  prompt_tokens       cumulative prompt tokens (vllm:prompt_tokens_total)
Cumulative counters reset when vLLM restarts; take last - first within one run.
"""
import argparse
import csv
import re
import signal
import time
import urllib.request

import psutil
import pynvml

FIELDS = ["t_s", "unix_s", "gpu_util", "gpu_mem_util", "gpu_mem_mb", "gpu_power_w",
          "gpu_sm_mhz", "cpu_total", "cpu_vllm", "running", "waiting", "kv_cache",
          "queue_time_sum", "queue_time_count", "preemptions", "gen_tokens", "prompt_tokens"]
METRICS = {"running": "vllm:num_requests_running",
           "waiting": "vllm:num_requests_waiting",
           "kv_cache": "vllm:kv_cache_usage_perc",
           "queue_time_sum": "vllm:request_queue_time_seconds_sum",
           "queue_time_count": "vllm:request_queue_time_seconds_count",
           "preemptions": "vllm:num_preemptions_total",
           "gen_tokens": "vllm:generation_tokens_total",
           "prompt_tokens": "vllm:prompt_tokens_total"}


def scrape(port):
    try:
        text = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=1).read().decode()
    except OSError:
        return {k: "" for k in METRICS}
    out = {}
    for key, name in METRICS.items():
        vals = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? (\S+)$", text, re.M)
        out[key] = sum(float(v) for v in vals) if vals else ""
    return out


class TreeCPU:
    """CPU % of a process and all its children (vLLM runs the engine in a child)."""

    def __init__(self, pid):
        self.root = psutil.Process(pid) if pid else None
        self.procs = {}

    def sample(self):
        if self.root is None:
            return ""
        try:
            tree = [self.root] + self.root.children(recursive=True)
        except psutil.NoSuchProcess:
            return ""
        total = 0.0
        for p in tree:
            proc = self.procs.setdefault(p.pid, p)
            try:
                total += proc.cpu_percent(None)  # first call per process returns 0
            except psutil.NoSuchProcess:
                self.procs.pop(p.pid, None)
        return round(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--pid", type=int, help="vLLM server PID (for per-process CPU)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    stop = False

    def handle(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    tree = TreeCPU(args.pid)
    psutil.cpu_percent(None)
    tree.sample()

    # monotonic clock for timing: WSL2 re-syncs its wall clock with Windows,
    # which made t_s jump backwards/forwards by up to ~2 s
    start = time.monotonic()
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        next_t = start + args.interval
        while not stop:
            time.sleep(max(0, next_t - time.monotonic()))
            next_t += args.interval
            now = time.monotonic()
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu)
            try:
                power = round(pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000, 1)
            except pynvml.NVMLError:
                power = ""
            w.writerow({
                "t_s": round(now - start, 3),
                "unix_s": round(time.time(), 3),
                "gpu_util": util.gpu,
                "gpu_mem_util": util.memory,
                "gpu_mem_mb": pynvml.nvmlDeviceGetMemoryInfo(gpu).used // 2**20,
                "gpu_power_w": power,
                "gpu_sm_mhz": pynvml.nvmlDeviceGetClockInfo(gpu, pynvml.NVML_CLOCK_SM),
                "cpu_total": psutil.cpu_percent(None),
                "cpu_vllm": tree.sample(),
                **scrape(args.port),
            })
            f.flush()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
