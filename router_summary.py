"""Summarize router_sweep.sh results: one table per trace, one row per policy.

    python router_summary.py [--dir results/router-sweep] [--ttft-slo 0.5] [--e2e-slo 5]

Columns:
  TTFT mean/p95, E2E p95   seconds, completed requests only
  goodput                  requests/s meeting both SLOs
  vLLM hit                 prefix-cache hit rate measured by vLLM (tokens, all workers)
  router est               hit rate the router predicted (blocks)
  routed                   requests sent to each worker
  vs RR                    TTFT p95 change relative to round_robin
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from lab2 import percentile, read_jsonl

POLICY_ORDER = ["round_robin", "random", "least_loaded", "prefix", "kv"]


def trace_key(path):
    """Sort traces by family, then numeric rate."""
    match = re.search(r"rate-([\d.]+)$", path.name)
    return (path.name.split("-rate-")[0], float(match.group(1)) if match else 0)


def summarize(trace_dir, policy, ttft_slo, e2e_slo):
    rows = read_jsonl(trace_dir / f"{policy}.jsonl")
    done = [r for r in rows if r["success"]]
    ttft = [r["ttft_s"] for r in done]
    e2e = [r["e2e_s"] for r in done]
    duration = max(r["finish_s"] for r in rows)
    good = [r for r in done if r["ttft_s"] <= ttft_slo and r["e2e_s"] <= e2e_slo]

    stats = json.loads((trace_dir / f"{policy}.stats.json").read_text())
    vllm = json.loads((trace_dir / f"{policy}.vllm.json").read_text())
    hits = sum(a[0] - b[0] for a, b in zip(vllm["after"], vllm["before"]))
    queries = sum(a[1] - b[1] for a, b in zip(vllm["after"], vllm["before"]))
    return {
        "failed": len(rows) - len(done),
        "ttft_mean": statistics.mean(ttft) if ttft else None,
        "ttft_p95": percentile(ttft, 0.95),
        "e2e_p95": percentile(e2e, 0.95),
        "goodput": len(good) / duration,
        "vllm_hit": hits / queries if queries else None,
        "router_hit": stats["estimated_hit_rate"],
        "routed": [w["routed"] for w in stats["workers"]],
    }


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/router-sweep")
    parser.add_argument("--ttft-slo", type=float, default=0.5)
    parser.add_argument("--e2e-slo", type=float, default=5.0)
    args = parser.parse_args()

    for trace_dir in sorted((p for p in Path(args.dir).iterdir() if p.is_dir()), key=trace_key):
        policies = [p for p in POLICY_ORDER if (trace_dir / f"{p}.jsonl").exists()]
        if not policies:
            continue
        results = {p: summarize(trace_dir, p, args.ttft_slo, args.e2e_slo) for p in policies}
        base = results.get("round_robin", {}).get("ttft_p95")

        print(f"\n### {trace_dir.name}\n")
        print("| policy | TTFT mean | TTFT p95 | E2E p95 | goodput | vLLM hit | router est | routed | vs RR |")
        print("|---|---|---|---|---|---|---|---|---|")
        for p, r in results.items():
            change = (f"{(r['ttft_p95'] / base - 1):+.0%}"
                      if base and r["ttft_p95"] is not None and p != "round_robin" else "")
            failed = f" ({r['failed']} failed)" if r["failed"] else ""
            print(f"| {p}{failed} | {fmt(r['ttft_mean'], '.3f')} | {fmt(r['ttft_p95'], '.3f')} "
                  f"| {fmt(r['e2e_p95'], '.2f')} | {r['goodput']:.2f} | {fmt(r['vllm_hit'], '.1%')} "
                  f"| {fmt(r['router_hit'], '.1%')} | {r['routed']} | {change} |")


if __name__ == "__main__":
    main()
