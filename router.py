"""Router in front of several vLLM workers: baselines plus a small Python
re-implementation of NVIDIA Dynamo's KV-aware routing.

    python router.py --workers http://localhost:8001 http://localhost:8002 --policy kv

Clients talk to the router (default port 8000) exactly as they would to vLLM.

Policies:
  round_robin   next worker in turn
  random        uniform random worker
  least_loaded  fewest in-flight requests (load only, no cache awareness)
  prefix        most cached prefix blocks, ties -> fewest in-flight requests
                (cache only, ignores load, so a hot prefix piles onto one worker)
  kv            Dynamo's cost, from dynamo docs router/routing-concepts.md:
                  prefill = max(0, active_prefill_blocks + request_blocks - overlap_blocks)
                  decode  = active_decode_blocks + request blocks not already active
                  cost    = prefill_load_scale * prefill + decode
                Lowest cost wins. --temperature > 0 samples softmax(-cost) instead.

Cache state (v1): the router assumes a worker holds every prompt block the
router has sent it, until the --cache-blocks LRU capacity pushes it out. It
never sees vLLM's own evictions. v2 will read vLLM's KV events instead.

Load state: in-flight requests and blocks, tracked by the router from the
requests it proxies. A request's uncached blocks count as prefill until its
first token streams back; its prompt blocks count as decode until it finishes.

GET /router/stats returns per-worker counts and the router's estimated hit rate.
"""
import argparse
import itertools
import json
import math
import random
import time
from collections import Counter, OrderedDict

import aiohttp
from aiohttp import web
from transformers import AutoTokenizer

POLICIES = ["round_robin", "random", "least_loaded", "prefix", "kv"]


def block_hashes(token_ids, block_size):
    """Chained hash of every full block. Equal hash at block i means the first
    i+1 blocks are identical, because KV values depend on all earlier tokens."""
    hashes, parent = [], None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = hash((parent, tuple(token_ids[start:start + block_size])))
        hashes.append(parent)
    return hashes


class Worker:
    def __init__(self, url, cache_blocks):
        self.url = url.rstrip("/")
        self.cache = OrderedDict()      # block hash -> None, least recently used first
        self.cache_blocks = cache_blocks
        self.active_requests = 0
        self.active_prefill_blocks = 0
        self.active_blocks = Counter()  # block hash -> in-flight requests using it
        self.routed = 0

    def overlap(self, hashes):
        """Leading blocks of this request already cached here; stops at the first miss."""
        n = 0
        for h in hashes:
            if h not in self.cache:
                break
            n += 1
        return n

    def remember(self, hashes):
        for h in hashes:
            self.cache[h] = None
            self.cache.move_to_end(h)
        if self.cache_blocks:
            while len(self.cache) > self.cache_blocks:
                self.cache.popitem(last=False)

    def start(self, hashes, prefill_blocks):
        self.routed += 1
        self.active_requests += 1
        self.active_prefill_blocks += prefill_blocks
        self.active_blocks.update(hashes)
        self.remember(hashes)

    def finish(self, hashes):
        self.active_requests -= 1
        for h in hashes:
            self.active_blocks[h] -= 1
            if self.active_blocks[h] <= 0:
                del self.active_blocks[h]


class Router:
    def __init__(self, args):
        self.workers = [Worker(url, args.cache_blocks) for url in args.workers]
        self.policy = args.policy
        self.block_size = args.block_size
        self.prefill_load_scale = args.prefill_load_scale
        self.temperature = args.temperature
        self.rng = random.Random(args.seed)
        self.next_rr = itertools.cycle(range(len(self.workers)))
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.log = open(args.log, "a", buffering=1) if args.log else None
        self.totals = Counter()
        self.session = None

    # ---- tokens -> blocks ----------------------------------------------------

    def token_ids(self, body):
        if "messages" in body:
            # Render text first, then encode (same as lab2.py; transformers 5
            # returns a dict from apply_chat_template(tokenize=True)).
            text = self.tokenizer.apply_chat_template(
                body["messages"], tokenize=False, add_generation_prompt=True
            )
        else:
            text = body.get("prompt", "")
            if not isinstance(text, str):
                text = json.dumps(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ---- worker selection ----------------------------------------------------

    def kv_cost(self, worker, hashes, overlap):
        prefill = max(0, worker.active_prefill_blocks + len(hashes) - overlap)
        new_active = sum(1 for h in hashes if h not in worker.active_blocks)
        decode = len(worker.active_blocks) + new_active
        return self.prefill_load_scale * prefill + decode

    def argmin(self, keys):
        best = min(keys)
        return self.rng.choice([i for i, k in enumerate(keys) if k == best])

    def softmax_sample(self, costs):
        top = max(costs) or 1.0
        logits = [-c / top / self.temperature for c in costs]
        peak = max(logits)
        weights = [math.exp(x - peak) for x in logits]
        return self.rng.choices(range(len(costs)), weights)[0]

    def choose(self, hashes):
        workers = self.workers
        overlaps = [w.overlap(hashes) for w in workers]
        costs = None
        if self.policy == "round_robin":
            index = next(self.next_rr)
        elif self.policy == "random":
            index = self.rng.randrange(len(workers))
        elif self.policy == "least_loaded":
            index = self.argmin([w.active_requests for w in workers])
        elif self.policy == "prefix":
            index = self.argmin([(-o, w.active_requests) for o, w in zip(overlaps, workers)])
        else:
            costs = [self.kv_cost(w, hashes, o) for w, o in zip(workers, overlaps)]
            index = (self.softmax_sample(costs) if self.temperature > 0
                     else self.argmin(costs))
        return index, overlaps, costs

    # ---- HTTP ------------------------------------------------------------------

    async def proxy(self, request):
        body = await request.json()
        hashes = block_hashes(self.token_ids(body), self.block_size)

        # No await between choosing and recording, so concurrent requests see
        # each other's load.
        index, overlaps, costs = self.choose(hashes)
        worker = self.workers[index]
        prefill_blocks = len(hashes) - overlaps[index]
        worker.start(hashes, prefill_blocks)
        self.totals["requests"] += 1
        self.totals["prompt_blocks"] += len(hashes)
        self.totals["hit_blocks"] += overlaps[index]
        if self.log:
            self.log.write(json.dumps({
                "t": round(time.time(), 6), "policy": self.policy, "worker": index,
                "prompt_blocks": len(hashes), "overlaps": overlaps, "costs": costs,
                "active_requests": [w.active_requests for w in self.workers],
            }) + "\n")

        prefill_open = True

        def end_prefill():
            nonlocal prefill_open
            if prefill_open:
                worker.active_prefill_blocks -= prefill_blocks
                prefill_open = False

        try:
            async with self.session.post(worker.url + request.path, json=body) as upstream:
                if not body.get("stream"):
                    data = await upstream.read()
                    end_prefill()
                    return web.Response(body=data, status=upstream.status,
                                        content_type=upstream.content_type)

                response = web.StreamResponse(status=upstream.status, headers={
                    "Content-Type": upstream.headers.get("Content-Type", "text/event-stream"),
                })
                await response.prepare(request)
                async for line in upstream.content:
                    if prefill_open and first_token(line):
                        end_prefill()
                    await response.write(line)
                await response.write_eof()
                return response
        except aiohttp.ClientError as exc:
            return web.json_response({"error": f"worker {worker.url}: {exc}"}, status=502)
        finally:
            end_prefill()
            worker.finish(hashes)

    async def models(self, request):
        async with self.session.get(self.workers[0].url + "/v1/models") as upstream:
            return web.Response(body=await upstream.read(), status=upstream.status,
                                content_type=upstream.content_type)

    async def stats(self, request):
        t = self.totals
        return web.json_response({
            "policy": self.policy,
            "requests": t["requests"],
            "prompt_blocks": t["prompt_blocks"],
            "hit_blocks": t["hit_blocks"],
            "estimated_hit_rate": t["hit_blocks"] / t["prompt_blocks"] if t["prompt_blocks"] else None,
            "workers": [{
                "url": w.url,
                "routed": w.routed,
                "active_requests": w.active_requests,
                "active_prefill_blocks": w.active_prefill_blocks,
                "active_blocks": len(w.active_blocks),
                "cached_blocks": len(w.cache),
            } for w in self.workers],
        })


def first_token(line):
    """True if this SSE line carries generated text (the first one marks end of prefill)."""
    if not line.startswith(b"data: {"):
        return False
    try:
        event = json.loads(line[6:])
    except json.JSONDecodeError:
        return False
    for choice in event.get("choices", []):
        if choice.get("delta", {}).get("content") or choice.get("text"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", nargs="+", required=True,
                        help="worker base URLs, e.g. http://localhost:8001")
    parser.add_argument("--policy", choices=POLICIES, default="kv")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--block-size", type=int, default=16,
                        help="must match vLLM's block size (default 16)")
    parser.add_argument("--prefill-load-scale", type=float, default=1.0,
                        help="kv: weight of prefill blocks vs decode blocks (Dynamo default 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="kv: 0 = always lowest cost; >0 = softmax sampling")
    parser.add_argument("--cache-blocks", type=int, default=0,
                        help="per-worker LRU capacity in blocks; 0 = unlimited. Use vLLM's "
                             "'GPU KV cache size' tokens / block size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", help="append one JSON line per routing decision")
    args = parser.parse_args()

    router = Router(args)

    async def on_startup(app):
        router.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),
            connector=aiohttp.TCPConnector(limit=0),
        )

    async def on_cleanup(app):
        await router.session.close()

    app = web.Application(client_max_size=64 * 1024 ** 2)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/chat/completions", router.proxy)
    app.router.add_post("/v1/completions", router.proxy)
    app.router.add_get("/v1/models", router.models)
    app.router.add_get("/router/stats", router.stats)
    print(f"policy={args.policy} workers={args.workers} port={args.port}")
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
