"""Plot every metric monitor.py records during a sweep.

    SWEEP_DIR=results/sweep-mon python plot_monitor.py

Writes:
  plots/monitor_summary.png        every metric averaged per run, vs arrival rate
  plots/monitor/<config>.png       every metric over time, one line per rate
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path(os.environ.get("SWEEP_DIR", "results/sweep-mon"))
RATES = [0.5, 1, 2, 4, 6]
NAN = float("nan")


def num(v):
    return float(v) if v not in ("", None) else NAN


def load(path):
    rows = list(csv.DictReader(path.open()))
    cols = {k: [num(r.get(k)) for r in rows] for k in rows[0]} if rows else {}
    # Older CSVs timed samples with the wall clock, which WSL2 re-syncs (jumps of
    # up to ~2 s). Rebuild t_s with irregular intervals replaced by the nominal
    # interval, and drop rates computed across those intervals.
    raw = cols.get("t_s", [])
    nominal = sorted(b - a for a, b in zip(raw, raw[1:]))[len(raw) // 2] if len(raw) > 2 else 0.5
    ok = [False] + [abs((b - a) - nominal) < 0.2 for a, b in zip(raw, raw[1:])]
    t = [0.0] * len(raw)
    for i in range(1, len(raw)):
        t[i] = t[i - 1] + (raw[i] - raw[i - 1] if ok[i] else nominal)
    if raw:
        t = [x + raw[0] for x in t]
    cols["t_s"] = t
    # per-interval rates from the cumulative vLLM counters
    for src, dst in (("gen_tokens", "gen_tps"), ("prompt_tokens", "prompt_tps")):
        v = cols.get(src)
        cols[dst] = [(v[i] - v[i - 1]) / (t[i] - t[i - 1]) if ok[i] else NAN
                     for i in range(len(t))] if v else [NAN] * len(t)
    qs, qn = cols.get("queue_time_sum"), cols.get("queue_time_count")
    cols["queue_ms"] = [
        (qs[i] - qs[i - 1]) / (qn[i] - qn[i - 1]) * 1000 if i and qn[i] > qn[i - 1] else NAN
        for i in range(len(t))] if qs else [NAN] * len(t)
    cols["kv_pct"] = [v * 100 for v in cols.get("kv_cache", [])]
    return cols


def clean(v):
    return [x for x in v if x == x]  # drop NaN


def mean(v):
    v = clean(v)
    return sum(v) / len(v) if v else NAN


def delta(v):
    v = clean(v)
    return v[-1] - v[0] if len(v) > 1 else NAN


configs = sorted((d for d in SWEEP.iterdir() if d.is_dir() and any(d.glob("monitor-rate-*.csv"))),
                 key=lambda d: (int(d.name.split("-seq")[1]), int(d.name.split("-seq")[0][3:])))
if not configs:
    raise SystemExit(f"No monitor-rate-*.csv under {SWEEP}/")
runs = {d.name: {r: load(d / f"monitor-rate-{r}.csv") for r in RATES
                 if (d / f"monitor-rate-{r}.csv").exists()} for d in configs}

# ---------- summary: one number per run, vs arrival rate ----------
def summarize(c):
    secs = delta(c["t_s"])
    gen = delta(c.get("gen_tokens", []))
    q_n = delta(c.get("queue_time_count", []))
    power = mean(c.get("gpu_power_w", []))
    return {
        "gpu_util": mean(c["gpu_util"]),
        "gpu_mem_util": mean(c["gpu_mem_util"]),
        "gpu_power_w": power,
        "gpu_sm_mhz": mean(c["gpu_sm_mhz"]),
        "gpu_mem_mb": mean(c["gpu_mem_mb"]),
        "cpu_total": mean(c["cpu_total"]),
        "cpu_vllm": mean(c["cpu_vllm"]),
        "running": mean(c.get("running", [])),
        "waiting": mean(c.get("waiting", [])),
        "waiting_peak": max(clean(c.get("waiting", [])), default=NAN),
        "kv_peak": max(clean(c["kv_pct"]), default=NAN),
        "queue_ms": delta(c.get("queue_time_sum", [])) / q_n * 1000 if q_n and q_n > 0 else NAN,
        "preempt": delta(c.get("preemptions", [])),
        "gen_tps": gen / secs,
        "prompt_tps": delta(c.get("prompt_tokens", [])) / secs,
        "j_per_tok": power * secs / gen if gen and gen > 0 else NAN,
    }


SUMMARY = [
    ("gpu_util", "GPU utilization (%, mean)"),
    ("gpu_mem_util", "GPU memory-bus busy (%, mean)"),
    ("gpu_power_w", "GPU power (W, mean)"),
    ("gpu_sm_mhz", "GPU SM clock (MHz, mean)"),
    ("gpu_mem_mb", "VRAM used (MB, mean)"),
    ("cpu_total", "System CPU (% of all cores, mean)"),
    ("cpu_vllm", "vLLM CPU (%, 100 = one core, mean)"),
    ("kv_peak", "KV cache usage (%, peak)"),
    ("running", "Requests running (mean)"),
    ("waiting", "Requests waiting (mean)"),
    ("waiting_peak", "Requests waiting (peak)"),
    ("queue_ms", "Queue time per request (ms, mean)"),
    ("gen_tps", "Output tokens/s (vLLM)"),
    ("prompt_tps", "Prompt tokens/s (vLLM)"),
    ("j_per_tok", "GPU energy per output token (J)"),
    ("preempt", "Preemptions per run"),
]

Path("plots/monitor").mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(4, 4, figsize=(20, 15))
for cfg, by_rate in runs.items():
    style = "-" if cfg.endswith("seq8") else "--"
    rates = list(by_rate)
    s = [summarize(by_rate[r]) for r in rates]
    for ax, (key, _) in zip(axes.flat, SUMMARY):
        ax.plot(rates, [x[key] for x in s], style, marker="o", label=cfg)
for ax, (_, label) in zip(axes.flat, SUMMARY):
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Arrival rate (req/s)")
    ax.grid(alpha=0.3)
axes.flat[0].legend(title="token budget / max seqs", fontsize=8)
fig.suptitle(f"All monitored metrics per run ({SWEEP})", fontsize=14)
fig.tight_layout()
fig.savefig("plots/monitor_summary.png", dpi=130)
plt.close(fig)
print("Saved plots/monitor_summary.png")

# ---------- timelines: every metric over time, per config ----------
TIMELINE = [
    ("gpu_util", "GPU utilization (%)"),
    ("gpu_mem_util", "GPU memory-bus busy (%)"),
    ("gpu_power_w", "GPU power (W)"),
    ("gpu_sm_mhz", "GPU SM clock (MHz)"),
    ("gpu_mem_mb", "VRAM used (MB)"),
    ("cpu_total", "System CPU (% of all cores)"),
    ("cpu_vllm", "vLLM CPU (%, 100 = one core)"),
    ("kv_pct", "KV cache usage (%)"),
    ("running", "Requests running"),
    ("waiting", "Requests waiting"),
    ("queue_ms", "Queue time, requests leaving queue (ms)"),
    ("gen_tps", "Output tokens/s (vLLM)"),
    ("prompt_tps", "Prompt tokens/s (vLLM)"),
    ("preemptions", "Preemptions (cumulative)"),
]

for cfg, by_rate in runs.items():
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    for rate, c in by_rate.items():
        for ax, (key, _) in zip(axes.flat, TIMELINE):
            if key in c:
                marker = "." if key == "queue_ms" else None
                ls = "none" if key == "queue_ms" else "-"
                ax.plot(c["t_s"], c[key], ls=ls, marker=marker, lw=0.9, label=f"{rate} req/s")
    for ax, (_, label) in zip(axes.flat, TIMELINE):
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Seconds since monitor start")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(TIMELINE):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{cfg}: monitored metrics over time", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"plots/monitor/{cfg}.png", dpi=110)
    plt.close(fig)
    print(f"Saved plots/monitor/{cfg}.png")

# ---------- table ----------
keys = ["gpu_util", "gpu_power_w", "gpu_sm_mhz", "cpu_total", "cpu_vllm",
        "running", "waiting_peak", "j_per_tok"]
print(f"\n{'config':<14} {'rate':>5} " + " ".join(f"{k:>12}" for k in keys))
for cfg, by_rate in runs.items():
    for rate, c in by_rate.items():
        s = summarize(c)
        print(f"{cfg:<14} {rate:>5} " + " ".join(f"{s[k]:>12.2f}" for k in keys))
