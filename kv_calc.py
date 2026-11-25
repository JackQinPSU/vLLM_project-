"""Predict KV cache size for a model and check it against vLLM.

    python kv_calc.py --model Qwen/Qwen2.5-1.5B-Instruct --kv-mem-gib 3.16

Every token stores one key and one value vector per layer, so
    bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_element
kv_heads is num_key_value_heads (GQA), not num_attention_heads.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

from transformers import AutoConfig

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
GIB = 1024 ** 3


def kv_shape(config):
    """Return (layers, kv_heads, head_dim) from a Hugging Face config."""
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads  # no GQA -> MHA
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return layers, kv_heads, head_dim


def bytes_per_token(layers, kv_heads, head_dim, dtype):
    return 2 * layers * kv_heads * head_dim * DTYPE_BYTES[dtype]


def capacity_tokens(mem_bytes, per_token, block_size):
    """Tokens that fit, counting only whole blocks (vLLM allocates by block)."""
    return int(mem_bytes // (per_token * block_size)) * block_size


def parse_server_log(path):
    """Pull vLLM's KV cache memory, token capacity and scheduler flags from a server.log."""
    text = Path(path).read_text(errors="replace")
    mem = re.search(r"Available KV cache memory: ([\d.]+) GiB", text)
    tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if not (mem and tokens):
        raise ValueError(f"{path}: no KV cache lines found (did the server finish starting?)")
    budget = re.search(r"'max_num_batched_tokens': (\d+)", text)
    seqs = re.search(r"'max_num_seqs': (\d+)", text)
    return {
        "mem_gib": float(mem.group(1)),
        "tokens": int(tokens.group(1).replace(",", "")),
        "budget": int(budget.group(1)) if budget else None,
        "seqs": int(seqs.group(1)) if seqs else None,
    }


def check_logs(paths, per_token, block_size):
    """Compare predicted capacity with what vLLM reported in each log.

    vLLM prints the memory rounded to 0.01 GiB, so the prediction is a range:
    the reported token count should land inside it.
    """
    print(f"\n{'log':<42} {'budget':>6} {'seqs':>4} {'KV GiB':>6} "
          f"{'vLLM tokens':>11} {'predicted range':>17} {'error':>6}  ok")
    half = 0.005 * GIB
    all_ok = True
    for path in paths:
        log = parse_server_log(path)
        mem = log["mem_gib"] * GIB
        lo = capacity_tokens(mem - half, per_token, block_size)
        hi = capacity_tokens(mem + half, per_token, block_size)
        mid = capacity_tokens(mem, per_token, block_size)
        ok = lo <= log["tokens"] <= hi
        all_ok &= ok
        err = (mid - log["tokens"]) / log["tokens"] * 100
        print(f"{str(path):<42} {log['budget'] or '?':>6} {log['seqs'] or '?':>4} "
              f"{log['mem_gib']:>6.2f} {log['tokens']:>11,} {lo:>8,}-{hi:<8,} "
              f"{err:>+5.2f}%  {'yes' if ok else 'NO'}")
    print("\nall within rounding" if all_ok else "\nsome logs fall outside the predicted range")


def request_intervals(results_path):
    """(t_first_token, t_finish, prompt_tokens, output_tokens) per successful request.

    Times are seconds from the start of the replay (lab2.py's clock).
    """
    out = []
    for line in Path(results_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["success"]:
            first = r["actual_send_s"] + r["ttft_s"]
            out.append((first, r["finish_s"], r["input_tokens"], r["output_tokens"]))
    return out


def predict_usage(intervals, times, total_blocks, block_size):
    """Predicted KV usage (0-1) and running count at each time.

    Simple model: a request's whole prompt is in the cache from its first token
    (prefill done) until it finishes, and its output grows linearly over that
    time. Each request rounds up to whole blocks. Queued requests hold nothing.
    Ignores prefix sharing between requests (the chat template's system prompt
    is the same for all, so this slightly overestimates).
    """
    usage, running = [], []
    for t in times:
        blocks, n = 0, 0
        for first, finish, prompt, output in intervals:
            if first <= t < finish:
                done = (t - first) / (finish - first) if finish > first else 1
                tokens = prompt + max(1, round(output * done))
                blocks += math.ceil(tokens / block_size)
                n += 1
        usage.append(blocks / total_blocks)
        running.append(n)
    return usage, running


def read_monitor(path):
    rows = list(csv.DictReader(open(path)))
    keep = [r for r in rows if r["kv_cache"] not in ("", None)]
    raw = [float(r["t_s"]) for r in keep]
    # Older CSVs used the wall clock, which WSL2 re-syncs (jumps of up to ~2 s).
    # Same fix as plot_monitor.py: replace irregular steps with the nominal interval.
    steps = sorted(b - a for a, b in zip(raw, raw[1:]))
    nominal = steps[len(steps) // 2] if steps else 0.5
    t = raw[:1]
    for a, b in zip(raw, raw[1:]):
        t.append(t[-1] + (b - a if abs((b - a) - nominal) < 0.2 else nominal))
    return t, [float(r["kv_cache"]) for r in keep], [float(r["gen_tokens"]) for r in keep]


def predict_gen_tokens(intervals, t):
    """Cumulative output tokens produced by time t (same linear decode model)."""
    total = 0.0
    for first, finish, _, output in intervals:
        if t >= finish:
            total += output
        elif t >= first:
            total += output * (t - first) / (finish - first)
    return total


def align(intervals, mon_t, mon_gen, step=0.05):
    """Offset (s) to add to replay time to get monitor time.

    Slides the predicted cumulative output tokens over vLLM's generation_tokens
    counter. The counter only goes up, so it stays informative even when the
    running count is pinned at max-num-seqs, and it doesn't depend on the KV
    prediction being tested. Warmup tokens before the replay are removed by
    anchoring the counter's final value to the trace's total output.
    """
    baseline = mon_gen[-1] - sum(o for _, _, _, o in intervals)
    measured = [g - baseline for g in mon_gen]
    end = max(f for _, f, _, _ in intervals)
    best, best_err = 0.0, float("inf")
    offset = 0.0
    while offset <= mon_t[-1] - end + 1:
        err = sum((predict_gen_tokens(intervals, t - offset) - m) ** 2
                  for t, m in zip(mon_t, measured))
        if err < best_err:
            best, best_err = offset, err
        offset += step
    return best


def check_sweep(sweep_dir, per_token, block_size, plot_path):
    """Predicted vs measured KV usage for every config and rate in a sweep."""
    sweep = Path(sweep_dir)
    runs = []  # (cfg, rate, mon_t, measured, predicted)
    print(f"\n{'config':<14} {'rate':>5} {'offset':>7} {'peak measured':>14} "
          f"{'peak predicted':>15} {'mean abs err':>13}")
    for cfg_dir in sorted(d for d in sweep.iterdir() if (d / "server.log").exists()):
        total_blocks = parse_server_log(cfg_dir / "server.log")["tokens"] // block_size
        for mon in sorted(cfg_dir.glob("monitor-rate-*.csv"),
                          key=lambda p: float(p.stem.split("-")[-1])):
            rate = mon.stem.split("-")[-1]
            res = cfg_dir / f"stock-rate-{rate}.jsonl"
            if not res.exists():
                continue
            intervals = request_intervals(res)
            mon_t, measured, mon_gen = read_monitor(mon)
            offset = align(intervals, mon_t, mon_gen)
            predicted, _ = predict_usage(intervals, [t - offset for t in mon_t],
                                         total_blocks, block_size)
            active = [(p, m) for p, m in zip(predicted, measured) if p > 0 or m > 0]
            mae = sum(abs(p - m) for p, m in active) / len(active) if active else 0
            print(f"{cfg_dir.name:<14} {rate:>5} {offset:>6.2f}s {max(measured) * 100:>13.1f}% "
                  f"{max(predicted) * 100:>14.1f}% {mae * 100:>11.2f}pp")
            runs.append((cfg_dir.name, rate, mon_t, measured, predicted))

    if plot_path and runs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cfgs = sorted({r[0] for r in runs}, key=lambda c: runs.index(next(x for x in runs if x[0] == c)))
        rates = sorted({r[1] for r in runs}, key=float)
        fig, axes = plt.subplots(len(cfgs), len(rates), figsize=(3 * len(rates), 1.9 * len(cfgs)),
                                 sharey=True, squeeze=False)
        for cfg, rate, t, meas, pred in runs:
            ax = axes[cfgs.index(cfg)][rates.index(rate)]
            ax.plot(t, [m * 100 for m in meas], lw=1.2, label="measured (vLLM /metrics)")
            ax.plot(t, [p * 100 for p in pred], lw=1.2, ls="--", label="predicted")
            if cfg == cfgs[0]:
                ax.set_title(f"{rate} req/s", fontsize=9)
            if rate == rates[0]:
                ax.set_ylabel(f"{cfg}\nKV %", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0][0].legend(fontsize=7, loc="upper left")
        fig.supxlabel("seconds (monitor clock)", fontsize=9)
        fig.tight_layout()
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=130)
        print(f"\nSaved {plot_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    ap.add_argument("--dtype", choices=DTYPE_BYTES, default="fp16",
                    help="KV cache dtype (fp8 = --kv-cache-dtype fp8 in vLLM)")
    ap.add_argument("--kv-mem-gib", type=float,
                    help="memory vLLM gives the KV cache ('Available KV cache memory' in server.log)")
    ap.add_argument("--block-size", type=int, default=16, help="vLLM tokens per KV block")
    ap.add_argument("--context", type=int, default=4096,
                    help="tokens per request, for the max-concurrency estimate")
    ap.add_argument("--server-log", nargs="+", metavar="LOG",
                    help="vLLM server.log file(s) to check the prediction against")
    ap.add_argument("--sweep", metavar="DIR",
                    help="sweep dir (e.g. results/sweep-mon): predict KV usage over time "
                         "from each run's results and compare with its monitor CSV")
    ap.add_argument("--plot", default="plots/kv_predicted_vs_measured.png")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    layers, kv_heads, head_dim = kv_shape(config)
    per_token = bytes_per_token(layers, kv_heads, head_dim, args.dtype)

    print(f"model        {args.model}")
    print(f"shape        {layers} layers x {kv_heads} kv heads x {head_dim} head_dim "
          f"({config.num_attention_heads} attention heads)")
    print(f"per token    {per_token:,} bytes ({per_token / 1024:.1f} KiB) in {args.dtype}")
    print(f"per request  {per_token * args.context / 2**20:,.0f} MiB at {args.context:,} tokens")

    if args.kv_mem_gib:
        tokens = capacity_tokens(args.kv_mem_gib * GIB, per_token, args.block_size)
        print(f"capacity     {tokens:,} tokens ({tokens // args.block_size:,} blocks of "
              f"{args.block_size}) in {args.kv_mem_gib} GiB")
        print(f"concurrency  {tokens / args.context:.2f}x requests of {args.context:,} tokens")

    if args.server_log:
        check_logs(args.server_log, per_token, args.block_size)

    if args.sweep:
        check_sweep(args.sweep, per_token, args.block_size, args.plot)


if __name__ == "__main__":
    main()
