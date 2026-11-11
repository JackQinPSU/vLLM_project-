import argparse
import asyncio
import hashlib
import json
import random
import statistics
import uuid
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
URL = "http://localhost:8000/v1/chat/completions"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def generate(args):
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    if args.family == "short":
        kinds = ["short"] * args.n
    elif args.family in ("mixed", "burst"):
        kinds = ["long"] * round(args.n * 0.20)
        kinds += ["short"] * (args.n - len(kinds))
        rng.shuffle(kinds)
    else:
        kinds = ["generation"] * round(args.n * 0.60)
        kinds += ["short"] * (args.n - len(kinds))
        rng.shuffle(kinds)

    rows = []
    arrival = 0.0
    for i, kind in enumerate(kinds):
        if i:
            rate = args.rate
            if args.family == "burst" and i >= args.n //2:
                rate *= 4
            arrival += rng.expovariate(rate)

        if kind == "long":
            target, output = rng.randint(2200, 2900), 128
        elif kind == "generation":
            target, output = rng.randint(180, 320), 640
        else:
            target, output = rng.randint(180, 320), 96

        request_id = f"{args.family}-{i:04d}"
        material = (
            f"Request {request_id}. Summarize the following notes: "
            + "GPU scheduling balances prefill, decoding, and KV cache capacity. " * 600
        )
        ids = tokenizer.encode(material, add_special_tokens=False)[: target - 20]
        prompt = tokenizer.decode(ids)
        messages = [{"role": "user", "content": prompt}]
        # transformers 5 returns a dict (input_ids, attention_mask) when
        # tokenize=True, so len() of it was always 2. Render text, then encode.
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        actual_input = len(tokenizer.encode(chat_text, add_special_tokens=False))
        if actual_input + output > 4096:
            raise ValueError(f"{request_id} exceeds the server context limit")

        rows.append({
            "id": request_id,
            "class": kind,
            "arrival_s": round(arrival, 6),
            "input_tokens": actual_input,
            "max_tokens": output,
            "prompt": prompt,
        })

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} requests to {args.output}")
    print("Classes:", {kind: kinds.count(kind) for kind in set(kinds)})


async def send(session, request, start, trace_hash, run_tag=""):
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0, start + request["arrival_s"] - loop.time()))
    sent = loop.time()
    first = None
    usage = {}
    done = False
    error = None

    payload = {
        "model": MODEL,
        # run_tag goes first so prompts never share a cached prefix across runs
        "messages": [{"role": "user", "content": run_tag + request["prompt"]}],
        "temperature": 0,
        "max_tokens": request["max_tokens"],
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        async with session.post(URL, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:300]}"
                )
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("delta", {}).get("content") and first is None:
                        first = loop.time()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    finished = loop.time()
    tokens = usage.get("completion_tokens")
    success = done and first is not None and tokens is not None
    if not success and error is None:
        error = "Stream ended without first content, usage, or [DONE]"

    return {
        "id": request["id"],
        "class": request["class"],
        "trace_sha256": trace_hash,
        "planned_arrival_s": request["arrival_s"],
        "actual_send_s": round(sent - start, 6),
        "send_lateness_s": round(sent - start - request["arrival_s"], 6),
        "finish_s": round(finished - start, 6),
        "input_tokens": usage.get("prompt_tokens", request["input_tokens"]),
        "requested_output_tokens": request["max_tokens"],
        "output_tokens": tokens,
        "success": success,
        "error": error,
        "ttft_s": round(first - sent, 6) if first else None,
        "e2e_s": round(finished - sent, 6),
        "tpot_s": (
            round((finished - first) / (tokens - 1), 6)
            if success and tokens > 1 else None
        ),
    }


async def replay_async(args):
    trace_bytes = Path(args.trace).read_bytes()
    trace_hash = hashlib.sha256(trace_bytes).hexdigest()
    requests = read_jsonl(args.trace)
    run_tag = f"[run {args.run_tag or uuid.uuid4().hex[:12]}] "
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Separate warmup requests; they are excluded from the measured trace.
        for kind in ("short", "long", "generation"):
            example = next((r for r in requests if r["class"] == kind), None)
            if example:
                warmup = dict(example, 
                              id="warmup-" + kind, 
                              arrival_s=0,
                              prompt=f"Seperate warmup request for {kind}." + example["prompt"],
                              )
                warmup_result = await send(session, warmup, asyncio.get_running_loop().time(),
                           trace_hash, run_tag)
                if not warmup_result["success"]:
                    raise RuntimeError(f"Warmup failed: {warmup_result['error']}")

        start = asyncio.get_running_loop().time()
        tasks = [
            asyncio.create_task(send(session, request, start, trace_hash, run_tag))
            for request in requests
        ]
        results = await asyncio.gather(*tasks)

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} results to {args.output}")
    print(f"Trace SHA-256: {trace_hash}")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def report(args):
    rows = read_jsonl(args.results)
    duration = max((r["finish_s"] for r in rows), default=0)
    for kind in ["ALL", "short", "long", "generation"]:
        group = rows if kind == "ALL" else [r for r in rows if r["class"] == kind]
        if not group:
            continue
        completed = [r for r in group if r["success"]]
        good = [
            r for r in completed
            if r["ttft_s"] <= args.ttft_slo
            and r["e2e_s"] <= args.e2e_slo
        ]
        ttfts = [r["ttft_s"] for r in completed]
        tpots = [r["tpot_s"] for r in completed if r["tpot_s"] is not None]
        latencies = [r["e2e_s"] for r in completed]
        late_sends = [r["send_lateness_s"] for r in group]

        print(f"\n{kind}: offered={len(group)} completed={len(completed)} "
              f"failed={len(group)-len(completed)}")
        print(f"  TTFT mean={statistics.mean(ttfts) if ttfts else None} s")
        print(f"  TPOT mean={statistics.mean(tpots) if tpots else None} s/token")
        print(f"  E2E p95={percentile(latencies, .95)} s")
        print(f"  send lateness p95={percentile(late_sends, .95)} s")
        print(f"  SLO attainment={len(good)/len(group):.1%}")
        print(f"  goodput={len(good)/duration:.3f} requests/s")
        print(f"  completed throughput={len(completed)/duration:.3f} requests/s")

    for r in rows:
        if not r["success"]:
            print(f"FAIL {r['id']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    g = commands.add_parser("generate")
    g.add_argument("--family", choices=["short", "mixed", "burst", "generation"],
                   required=True)
    g.add_argument("--rate", type=float, default=0.7, 
                    help="Average request per second")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", required=True)

    r = commands.add_parser("replay")
    r.add_argument("--trace", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--run-tag", default=None,
                   help="Prefix added to every prompt so runs never hit each other's prefix cache (default: random per run)")

    p = commands.add_parser("report")
    p.add_argument("--results", required=True)
    p.add_argument("--ttft-slo", type=float, default=2.0)
    p.add_argument("--e2e-slo", type=float, default=30.0)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "replay":
        asyncio.run(replay_async(args))
    else:
        report(args)


if __name__ == "__main__":
    main()
