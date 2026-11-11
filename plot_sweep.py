import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATES = [0.5, 1, 2, 4, 6]
TTFT_SLO = 0.5
E2E_SLO = 5.0
SWEEP = Path(os.environ.get("SWEEP_DIR", "results/sweep"))


def p95(values):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = 0.95 * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def cfg_key(path):
    tok, seq = path.name.replace("tok", "").split("-seq")
    return int(seq), int(tok)


summary = {}  # cfg -> {"rates", "goodput", "ttft", "tpot"}
for cfg_dir in sorted((d for d in SWEEP.iterdir() if d.is_dir()), key=cfg_key):
    s = {"rates": [], "goodput": [], "ttft": [], "tpot": [],
         "req_tput": [], "out_tps": [], "in_tps": []}
    for rate in RATES:
        path = cfg_dir / f"stock-rate-{rate}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        ok = [r for r in rows if r["success"]]
        good = [r for r in ok if r["ttft_s"] <= TTFT_SLO and r["e2e_s"] <= E2E_SLO]
        end = max(r["finish_s"] for r in rows)
        s["rates"].append(rate)
        s["goodput"].append(len(good) / end)
        s["req_tput"].append(len(ok) / end)
        s["out_tps"].append(sum(r["output_tokens"] for r in ok) / end)
        s["in_tps"].append(sum(r["input_tokens"] for r in ok) / end)

        # resource usage from monitor.py, if this run was monitored
        mon = cfg_dir / f"monitor-rate-{rate}.csv"
        if mon.exists():
            m = list(csv.DictReader(mon.open()))
            col = lambda k: [float(x[k]) for x in m if x.get(k) not in ("", None)]
            mean = lambda v: sum(v) / len(v) if v else float("nan")
            delta = lambda v: v[-1] - v[0] if len(v) > 1 else float("nan")
            secs = delta(col("t_s"))
            s.setdefault("mon_rates", []).append(rate)
            s.setdefault("gpu_util", []).append(mean(col("gpu_util")))
            s.setdefault("cpu_vllm", []).append(mean(col("cpu_vllm")))
            s.setdefault("kv_peak", []).append(max(col("kv_cache"), default=float("nan")) * 100)
            # cumulative vLLM counters -> per-run values (counters reset per server)
            q_n = delta(col("queue_time_count"))
            s.setdefault("queue_ms", []).append(
                delta(col("queue_time_sum")) / q_n * 1000 if q_n and q_n > 0 else float("nan"))
            s.setdefault("preempt", []).append(delta(col("preemptions")))
            s.setdefault("srv_gen_tps", []).append(delta(col("gen_tokens")) / secs)
        s["ttft"].append(p95([r["ttft_s"] for r in ok]))
        s["tpot"].append(p95([r["tpot_s"] for r in ok if r["tpot_s"] is not None]) * 1000)
    if s["rates"]:
        summary[cfg_dir.name] = s

if not summary:
    raise SystemExit(f"No sweep results under {SWEEP}/")

print(f"{'config':<14} {'rate':>5} {'goodput':>8} {'p95 TTFT':>9} {'p95 TPOT':>9}")
for cfg, s in summary.items():
    for i, rate in enumerate(s["rates"]):
        print(f"{cfg:<14} {rate:>5} {s['goodput'][i]:>8.3f} "
              f"{s['ttft'][i]:>8.3f}s {s['tpot'][i]:>7.1f}ms")

Path("plots").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for cfg, s in summary.items():
    style = "-" if cfg.endswith("seq8") else "--"
    axes[0].plot(s["rates"], s["goodput"], style, marker="o", label=cfg)
    axes[1].plot(s["rates"], s["ttft"], style, marker="o", label=cfg)
    axes[2].plot(s["rates"], s["tpot"], style, marker="o", label=cfg)

axes[0].set_ylabel("Goodput (requests/s)")
axes[1].set_ylabel("p95 TTFT (s)")
axes[1].axhline(TTFT_SLO, color="gray", lw=0.8, ls=":")
axes[2].set_ylabel("p95 TPOT (ms/token)")
for ax in axes:
    ax.set_xlabel("Configured average arrival rate (requests/s)")
    ax.grid(alpha=0.3)
axes[0].legend(title="token budget / max seqs")
fig.suptitle("vLLM scheduler sweep: --max-num-batched-tokens x --max-num-seqs")
fig.tight_layout()
fig.savefig("plots/sweep.png", dpi=160)
print("Saved plots/sweep.png")

# Throughput: everything completed, regardless of SLO
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for cfg, s in summary.items():
    style = "-" if cfg.endswith("seq8") else "--"
    axes[0].plot(s["rates"], s["req_tput"], style, marker="o", label=cfg)
    axes[1].plot(s["rates"], s["out_tps"], style, marker="o", label=cfg)
    axes[2].plot(s["rates"], s["in_tps"], style, marker="o", label=cfg)

axes[0].plot(RATES, RATES, color="gray", lw=0.8, ls=":", label="offered load")
axes[0].set_ylabel("Completed requests/s")
axes[1].set_ylabel("Output tokens/s")
axes[2].set_ylabel("Prompt tokens/s")
for ax in axes:
    ax.set_xlabel("Configured average arrival rate (requests/s)")
    ax.grid(alpha=0.3)
axes[0].legend(title="token budget / max seqs")
fig.suptitle("vLLM scheduler sweep: throughput (all successful requests, no SLO filter)")
fig.tight_layout()
fig.savefig("plots/sweep_throughput.png", dpi=160)
print("Saved plots/sweep_throughput.png")

monitored = {cfg: s for cfg, s in summary.items() if s.get("mon_rates")}
if monitored:
    panels = [("gpu_util", "Mean GPU utilization (%)"),
              ("cpu_vllm", "Mean vLLM CPU (%, 100 = one core)"),
              ("kv_peak", "Peak KV cache usage (%)"),
              ("queue_ms", "Mean queue time (ms, vLLM)"),
              ("preempt", "Preemptions per run"),
              ("srv_gen_tps", "Output tokens/s (vLLM counter)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for cfg, s in monitored.items():
        style = "-" if cfg.endswith("seq8") else "--"
        for ax, (key, _) in zip(axes, panels):
            ax.plot(s["mon_rates"], s[key], style, marker="o", label=cfg)
    for ax, (_, label) in zip(axes, panels):
        ax.set_ylabel(label)
    for ax in axes:
        ax.set_xlabel("Configured average arrival rate (requests/s)")
        ax.grid(alpha=0.3)
    axes[0].legend(title="token budget / max seqs")
    fig.suptitle("vLLM scheduler sweep: resource usage (monitor.py)")
    fig.tight_layout()
    fig.savefig("plots/sweep_resources.png", dpi=160)
    print("Saved plots/sweep_resources.png")

    print(f"\n{'config':<14} {'rate':>5} {'gpu%':>5} {'cpu%':>6} {'kv%':>5} "
          f"{'queue ms':>9} {'preempt':>8} {'gen tok/s':>10}")
    for cfg, s in monitored.items():
        for i, rate in enumerate(s["mon_rates"]):
            print(f"{cfg:<14} {rate:>5} {s['gpu_util'][i]:>5.0f} {s['cpu_vllm'][i]:>6.0f} "
                  f"{s['kv_peak'][i]:>5.1f} {s['queue_ms'][i]:>9.1f} "
                  f"{s['preempt'][i]:>8.0f} {s['srv_gen_tps'][i]:>10.1f}")
else:
    print(f"No monitor-rate-*.csv under {SWEEP}/; skipping resource plot")

print(f"\n{'config':<14} {'rate':>5} {'req/s':>7} {'out tok/s':>10} {'in tok/s':>10}")
for cfg, s in summary.items():
    for i, rate in enumerate(s["rates"]):
        print(f"{cfg:<14} {rate:>5} {s['req_tput'][i]:>7.3f} "
              f"{s['out_tps'][i]:>10.1f} {s['in_tps'][i]:>10.1f}")
