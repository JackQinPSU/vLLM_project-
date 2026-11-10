import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATES = [0.5, 1, 2, 4, 6]
BIN_S = 1.0  # time-bucket width in seconds


def load(rate):
    path = Path(f"results/stock-rate-{rate}.jsonl")
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timeline(rows):
    """Derive per-second series straight from the per-request JSON results."""
    end = max(r["finish_s"] for r in rows)
    n = int(end / BIN_S) + 1
    t = [i * BIN_S for i in range(n)]
    running = [0] * n   # first token received, still decoding
    waiting = [0] * n   # sent, no first token yet (queued / prefilling)
    prompt_tps = [0.0] * n
    gen_tps = [0.0] * n

    for r in rows:
        if not r["success"] or r["ttft_s"] is None:
            continue
        send = r["actual_send_s"]
        first = send + r["ttft_s"]
        finish = r["finish_s"]
        for i, ti in enumerate(t):
            if send <= ti < first:
                waiting[i] += 1
            elif first <= ti < finish:
                running[i] += 1
        # prompt tokens are processed by the time the first token arrives
        prompt_tps[min(int(first / BIN_S), n - 1)] += r["input_tokens"] / BIN_S
        # spread generated tokens evenly over the decode window
        decode = max(finish - first, 1e-9)
        per_s = r["output_tokens"] / decode
        for i, ti in enumerate(t):
            overlap = min(finish, ti + BIN_S) - max(first, ti)
            if overlap > 0:
                gen_tps[i] += per_s * overlap / BIN_S

    return t, running, waiting, prompt_tps, gen_tps


def pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


data = {}
ttft = {}  # rate -> [(send time, ttft)] for successful requests
for rate in RATES:
    rows = load(rate)
    if rows:
        data[rate] = timeline(rows)
        ttft[rate] = sorted(
            (r["actual_send_s"], r["ttft_s"])
            for r in rows if r["success"] and r["ttft_s"] is not None
        )

if not data:
    raise SystemExit("No results/stock-rate-*.jsonl files found.")

Path("plots").mkdir(exist_ok=True)


def graph(filename, title, ylabel, column):
    plt.figure(figsize=(10, 4))
    for rate, series in data.items():
        plt.plot(series[0], series[column], label=f"{rate} req/s")
    plt.title(title)
    plt.xlabel("Seconds since run start")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=160)
    plt.close()


graph("running_requests.png", "Requests decoding", "Requests", 1)
graph("waiting_requests.png", "Requests waiting for first token", "Requests", 2)
graph("prompt_throughput.png", "Prompt token throughput", "Tokens/second", 3)
graph("generation_throughput.png", "Generation token throughput", "Tokens/second", 4)

# TTFT of every request over the run, one line per rate
plt.figure(figsize=(10, 4))
for rate, points in ttft.items():
    plt.plot([p[0] for p in points], [p[1] for p in points],
             marker=".", label=f"{rate} req/s")
plt.title("Time to first token per request")
plt.xlabel("Request send time (s since run start)")
plt.ylabel("TTFT (s)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plots/ttft_over_time.png", dpi=160)
plt.close()

# TTFT summary versus rate
rates = list(ttft)
stats = {name: [pct([p[1] for p in ttft[r]], q) for r in rates]
         for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}
stats["mean"] = [sum(p[1] for p in ttft[r]) / len(ttft[r]) for r in rates]

print(f"{'rate':>6} {'n':>4} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}  (TTFT, s)")
for i, r in enumerate(rates):
    print(f"{r:>6} {len(ttft[r]):>4} {stats['mean'][i]:>8.3f} {stats['p50'][i]:>8.3f} "
          f"{stats['p95'][i]:>8.3f} {stats['p99'][i]:>8.3f}")

plt.figure(figsize=(7, 4))
for name in ("mean", "p50", "p95", "p99"):
    plt.plot(rates, stats[name], marker="o", label=name)
plt.title("TTFT versus arrival rate")
plt.xlabel("Configured average arrival rate (requests/s)")
plt.ylabel("TTFT (s)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plots/ttft_vs_rate.png", dpi=160)
plt.close()

print(f"Made 6 graphs from results for rates {list(data)} in plots/")
