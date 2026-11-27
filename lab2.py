"""Generate request traces, replay them against an OpenAI-compatible server, report latency.

    python lab2.py generate --family mixed --rate 2 --n 60 --output traces/x.jsonl
    python lab2.py replay   --trace traces/x.jsonl --output results/x.jsonl
    python lab2.py report   --results results/x.jsonl
"""
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
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # pinned for reproducibility
URL = "http://localhost:8000/v1/chat/completions"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


# ---- trace generation: shared-prefix families (docs, multiturn) ----

# Vocabulary for random filler text.
WORDS = (
    "scheduler cache token block prefill decode batch latency throughput memory "
    "kernel tensor layer attention queue worker router request stream replica "
    "cluster bandwidth compute model weight context window prompt output budget "
    "policy priority deadline fairness eviction capacity utilization pipeline "
    "stage shard node network storage index hash tree prefix suffix server "
    "client trace metric sample profile report"
).split()


def filler(rng, tokenizer, header, n_tokens):
    """`header` + random words, cut to ~n_tokens tokens."""
    words = " ".join(rng.choice(WORDS) for _ in range(n_tokens))
    ids = tokenizer.encode(f"{header} {words}", add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids)


def make_row(tokenizer, request_id, kind, arrival, max_tokens,
             prompt=None, messages=None, **extra):
    """One trace row. Pass `prompt` (single turn) or `messages` (multi-turn)."""
    # Count input tokens the way the server sees them: with the chat template.
    chat = messages or [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    actual_input = len(tokenizer.encode(chat_text, add_special_tokens=False))
    if actual_input + max_tokens > 4096:  # server's --max-model-len
        raise ValueError(f"{request_id} exceeds the server context limit")
    row = {
        "id": request_id,
        "class": kind,
        "arrival_s": round(arrival, 6),
        "input_tokens": actual_input,
        "max_tokens": max_tokens,
        **extra,  # e.g. group, conversation, turn
    }
    if messages:
        row["messages"] = messages
    else:
        row["prompt"] = prompt
    return row


def generate_docs(args, rng, tokenizer):
    """Each request = one of --groups long documents + a short question.
    Same document -> shared prefix. --zipf > 0 makes low-numbered docs hot."""
    docs = [filler(rng, tokenizer, f"Document {g}.", rng.randint(1500, 2500))
            for g in range(args.groups)]
    weights = [1 / (g + 1) ** args.zipf for g in range(args.groups)]  # zipf=0 -> uniform
    rows, arrival = [], 0.0
    for i in range(args.n):
        if i:
            arrival += rng.expovariate(args.rate)  # Poisson arrivals
        group = rng.choices(range(args.groups), weights)[0]
        question = filler(rng, tokenizer, f"Question {i}:", rng.randint(20, 60))
        rows.append(make_row(tokenizer, f"docs-{i:04d}", "doc", arrival, 96,
                             prompt=f"{docs[group]}\n\n{question}", group=group))
    return rows


def generate_multiturn(args, rng, tokenizer):
    """Conversations of --turns turns; each turn resends the full history,
    so turn t reuses all of turn t-1's prompt.
    Assistant replies are fixed trace text (not the model's real output),
    so only prompt blocks are reusable, never generated ones."""
    conversations = max(1, args.n // args.turns)
    rows, start = [], 0.0
    for c in range(conversations):
        if c:
            # Conversations start at rate/turns, so total request rate ~ --rate.
            start += rng.expovariate(args.rate / args.turns)
        arrival, messages = start, []
        for t in range(args.turns):
            if t == 0:
                content = filler(rng, tokenizer, f"Conversation {c}. Background:",
                                 rng.randint(400, 800))
            else:
                # User "thinks" 2 s + exp(mean --think-s) before the next turn.
                arrival += 2.0 + rng.expovariate(1 / args.think_s)
                content = filler(rng, tokenizer, f"Follow-up {t}:", rng.randint(40, 120))
            messages = messages + [{"role": "user", "content": content}]
            rows.append(make_row(tokenizer, f"conv{c:03d}-t{t}", "turn", arrival, 96,
                                 messages=messages, conversation=c, turn=t))
            reply = filler(rng, tokenizer, "Answer:", rng.randint(80, 150))
            messages = messages + [{"role": "assistant", "content": reply}]
    rows.sort(key=lambda r: r["arrival_s"])  # interleave conversations by time
    return rows


def ideal_reuse(rows, tokenizer, block_size=16):
    """Max fraction of prompt blocks that could hit the prefix cache
    (one shared infinite cache, timing ignored). Upper bound for any router."""
    seen, hits, total = set(), 0, 0
    for row in sorted(rows, key=lambda r: r["arrival_s"]):
        chat = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        parent, matching = None, True
        for start in range(0, len(ids) - block_size + 1, block_size):
            # Chained hash: block i's hash covers all blocks before it.
            parent = hash((parent, tuple(ids[start:start + block_size])))
            total += 1
            matching = matching and parent in seen  # stop at first miss
            hits += matching
            seen.add(parent)
    return hits / total if total else 0.0


# ---- trace generation: entry point + original families ----

def generate(args):
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    if args.family in ("docs", "multiturn"):
        make = generate_docs if args.family == "docs" else generate_multiturn
        rows = make(args, rng, tokenizer)
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} requests to {args.output}")
        print(f"Upper bound on prefix reuse (one shared cache, no timing): "
              f"{ideal_reuse(rows, tokenizer):.1%}")
        return

    # Original families. Request mix:
    #   short: all short | mixed/burst: 20% long | generation: 60% long-output
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
                rate *= 4  # second half arrives 4x faster
            arrival += rng.expovariate(rate)  # Poisson arrivals

        # (prompt tokens, output tokens) per class
        if kind == "long":
            target, output = rng.randint(2200, 2900), 128
        elif kind == "generation":
            target, output = rng.randint(180, 320), 640
        else:
            target, output = rng.randint(180, 320), 96

        # Request id comes first, so no two requests share a cached prefix.
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


# ---- replay ----

async def send(session, request, start, trace_hash, run_tag="", model=MODEL, url=URL):
    """Send one request at its planned time, stream the reply, return timings."""
    loop = asyncio.get_running_loop()
    # Open loop: wait for the planned arrival time, regardless of earlier requests.
    await asyncio.sleep(max(0, start + request["arrival_s"] - loop.time()))
    sent = loop.time()
    first = None   # time of first generated token
    usage = {}
    done = False
    error = None

    messages = request.get("messages") or [{"role": "user", "content": request["prompt"]}]
    # run_tag goes first so prompts never share a cached prefix across runs
    messages = [dict(messages[0], content=run_tag + messages[0]["content"])] + messages[1:]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": request["max_tokens"],
        "ignore_eos": True,  # always generate exactly max_tokens
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:300]}"
                )
            # Server-sent events: one "data: {...}" line per chunk.
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
        "send_lateness_s": round(sent - start - request["arrival_s"], 6),  # client overload check
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
    trace_hash = hashlib.sha256(trace_bytes).hexdigest()  # proves which trace was used
    requests = read_jsonl(args.trace)
    run_tag = f"[run {args.run_tag or uuid.uuid4().hex[:12]}] "
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)  # no client-side connection cap

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Separate warmup requests; they are excluded from the measured trace.
        for kind in class_order(requests):
            example = next((r for r in requests if r["class"] == kind), None)
            if example:
                # Warmup text goes first so it never pre-fills the trace's shared prefixes.
                text = example.get("prompt") or example["messages"][0]["content"]
                warmup = {k: v for k, v in example.items() if k != "messages"}
                warmup.update(id="warmup-" + kind,
                              arrival_s=0,
                              prompt=f"Seperate warmup request for {kind}." + text)
                warmup_result = await send(session, warmup, asyncio.get_running_loop().time(),
                           trace_hash, run_tag, args.model, args.url)
                if not warmup_result["success"]:
                    raise RuntimeError(f"Warmup failed: {warmup_result['error']}")

        # All requests are scheduled at once; each sleeps until its arrival time.
        start = asyncio.get_running_loop().time()
        tasks = [
            asyncio.create_task(send(session, request, start, trace_hash, run_tag,
                                     args.model, args.url))
            for request in requests
        ]
        results = await asyncio.gather(*tasks)

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} results to {args.output}")
    print(f"Trace SHA-256: {trace_hash}")


# ---- report ----

def class_order(rows):
    """Request classes present, original classes first in their original order."""
    present = {r["class"] for r in rows}
    original = [k for k in ("short", "long", "generation") if k in present]
    return original + sorted(present - set(original))


def percentile(values, fraction):
    """Linear-interpolated percentile, e.g. fraction=.95 for p95."""
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
    for kind in ["ALL"] + class_order(rows):
        group = rows if kind == "ALL" else [r for r in rows if r["class"] == kind]
        if not group:
            continue
        completed = [r for r in group if r["success"]]
        # "Good" = finished and met both SLOs.
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


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    g = commands.add_parser("generate")
    g.add_argument("--family", required=True,
                   choices=["short", "mixed", "burst", "generation", "docs", "multiturn"])
    g.add_argument("--groups", type=int, default=8,
                   help="docs: number of distinct documents")
    g.add_argument("--zipf", type=float, default=0.0,
                   help="docs: popularity skew; 0 = uniform, 1 = document k asked 1/k as often")
    g.add_argument("--turns", type=int, default=5, help="multiturn: turns per conversation")
    g.add_argument("--think-s", type=float, default=4.0,
                   help="multiturn: mean extra seconds between turns (on top of 2 s)")
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
    r.add_argument("--model", default=MODEL,
                   help="model name sent in each request; must match what the server serves")
    r.add_argument("--url", default=URL, help="chat completions endpoint")

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
