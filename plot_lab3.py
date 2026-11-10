import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATES = [0.5, 1, 2, 4, 6]
TTFT_SLO = 0.5
E2E_SLO = 5.0


def p95(values):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = 0.95 * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


goodputs = []
ttft = {"short": [], "long": []}
tpot = {"short": [], "long": []}

for rate in RATES:
    path = Path(f"results/stock-rate-{rate}.jsonl")
    if not path.exists():
        raise SystemExit(f"Missing {path}; check your filenames with: rg --files results")

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    duration = max(row["finish_s"] for row in rows)
    good = [
        row for row in rows
        if row["success"]
        and row["ttft_s"] <= TTFT_SLO
        and row["e2e_s"] <= E2E_SLO
    ]
    goodputs.append(len(good) / duration)

    for kind in ("short", "long"):
        completed = [
            row for row in rows
            if row["class"] == kind and row["success"]
        ]
        ttft[kind].append(p95([row["ttft_s"] for row in completed]))
        tpot[kind].append(p95([
            row["tpot_s"] for row in completed
            if row["tpot_s"] is not None
        ]) * 1000)  # seconds/token -> milliseconds/token

    print(f"{rate} req/s: {len(rows)} offered, {len(good)} good, "
          f"goodput={goodputs[-1]:.3f} req/s")

Path("plots").mkdir(exist_ok=True)

plt.figure(figsize=(7, 4))
plt.plot(RATES, goodputs, marker="o")
plt.xlabel("Configured average arrival rate (requests/s)")
plt.ylabel("Goodput (requests/s)")
plt.title("Stock vLLM: goodput versus offered load")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/goodput.png", dpi=160)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
for kind in ("short", "long"):
    axes[0].plot(RATES, ttft[kind], marker="o", label=kind)
    axes[1].plot(RATES, tpot[kind], marker="o", label=kind)

axes[0].set_ylabel("p95 TTFT (s)")
axes[1].set_ylabel("p95 TPOT (ms/token)")
axes[1].set_xlabel("Configured average arrival rate (requests/s)")
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("Stock vLLM: latency versus offered load")
fig.tight_layout()
fig.savefig("plots/latency.png", dpi=160)
print("Saved plots/goodput.png and plots/latency.png")