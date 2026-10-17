"""Run model x condition x subset episodes against OpenRouter, with caching, JSONL logging and a budget guard.

    python src/run.py --models A F --conditions C0 C2 --subsets FLIP [--limit 20] [--prompt-version v1]

One episode = one claim under one model and one condition: up to 4 requests (3 tool calls + final answer).
Episodes are cached in runs/cache/<sha256(model|condition|prompt_version|claim_id)>.json; a cached episode is never
re-billed. Every request is appended to runs/calls.jsonl.
"""
import argparse, asyncio, hashlib, json, os, random, re, sys, time
from datetime import datetime, timezone
import httpx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vintage_tool import get_values, META  # noqa: E402

MODELS = {
    "A": {"id": "meta-llama/llama-3.1-8b-instruct", "extra": {}, "max_tokens": 1000, "conc": 6},  # upstream rate limits
    # pinned to the two cheapest DeepSeek-V4-Pro hosts (same open weights, both fp8); neither accepts a seed
    "F": {"id": "deepseek/deepseek-v4-pro", "extra": {"reasoning": {"enabled": False}}, "max_tokens": 1000, "no_seed": True,
          "provider": {"order": ["Baidu", "StreamLake"], "allow_fallbacks": False}},
    "S": {"id": "anthropic/claude-sonnet-5.5", "extra": {}, "max_tokens": 1000, "no_seed": True, "no_temperature": True},  # Sonnet 5.5 accepts neither seed nor temperature
    # pinned to one host (DEVIATIONS D3.15): Darkbloom, fp8, no fallback
    "B": {"id": "openai/gpt-oss-20b", "extra": {"reasoning": {"effort": "low"}}, "max_tokens": 2500,
          "provider": {"order": ["Darkbloom"], "allow_fallbacks": False}},
}
SEED = 2027
MAX_TOOL_CALLS = 3
BUDGET_STOP = float(os.environ.get("BUDGET_STOP", "27"))  # raised 18 -> 23 -> 27 by the author (DEVIATIONS D3.8, D3.16)
URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = os.path.join(ROOT, "runs", "cache")
# one call log per process: parallel lanes appending to a single file interleaved their writes (DEVIATIONS D5.3)
CALLS = os.path.join(ROOT, "runs", f"calls_{os.getpid()}.jsonl")
LABELS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")


def read_key():
    p = os.environ["OPENROUTER_KEY_FILE"]  # a local file holding an OpenRouter API key
    return open(p, encoding="utf-8").read().strip()


def measure_text(kind, horizon):
    kind = "" if pd.isna(kind) else kind; horizon = "" if pd.isna(horizon) else horizon
    m = re.fullmatch(r"(avg|sum)_change_(\d+)m", kind)
    if m:
        what = "average monthly change" if m.group(1) == "avg" else "sum of monthly changes"
        return f"{what} over the {m.group(2)} months ending in the reference period"
    return {
        ("change", ""): "change from the prior period",
        ("pct", "mom"): "percent change from the prior month",
        ("pct", "yoy"): "percent change from a year earlier",
        ("level", ""): "level",
        ("saar", ""): "annualized growth rate from the prior quarter",
        ("saar", "yoy"): "growth from a year earlier",
        ("saar_3m", ""): "annualized percent change over the 3 months ending in the reference period",
        ("annual_sum_of_changes", ""): "sum of monthly changes over the year ending in the reference period",
    }.get((kind, horizon), "not specified")


def build_messages(row, cond, pv):
    d = os.path.join(ROOT, "prompts", pv)
    system = open(os.path.join(d, "system.txt"), encoding="utf-8").read()
    system = system.replace("{CONDITION}", open(os.path.join(d, f"condition_{cond}.txt"), encoding="utf-8").read().strip())
    name, units = META.get(row["series"], (row["series"], ""))
    cons = row.get("consensus")
    consensus_line = "" if pd.isna(cons) or cons in ("", None) else \
        f"\nConsensus forecast for this figure (median across coverage, in the units of the measure): {cons}"
    user = open(os.path.join(d, "user.txt"), encoding="utf-8").read().strip().format(
        claim_text=row["claim_text"], as_of=row["as_of"], series=row["series"], series_name=name, units=units,
        ref=row["ref"], measure=measure_text(row["kind"], row["horizon"]), consensus_line=consensus_line)
    tools = None
    if cond in ("C1", "C2"):
        tools = [json.load(open(os.path.join(d, f"tool_{cond}.json"), encoding="utf-8"))]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], tools


def run_tool(cond, args):
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}
    if cond == "C1":
        return get_values(args.get("series"), args.get("start_period"), args.get("end_period"))
    return get_values(args.get("series"), args.get("start_period"), args.get("end_period"), args.get("as_of"))


def parse_verdict(text):
    if not text:
        return None, "empty"
    for m in reversed(list(re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.S))):
        try:
            v = str(json.loads(m.group(0)).get("verdict", "")).strip().upper()
            if v in LABELS: return v, "json"
        except Exception:
            pass
        m2 = re.search(r"\"verdict\"\s*:\s*\"?(SUPPORTED|CONTRADICTED|UNVERIFIABLE)", m.group(0), re.I)
        if m2: return m2.group(1).upper(), "json_loose"
    hits = re.findall(r"\b(SUPPORTED|CONTRADICTED|UNVERIFIABLE)\b", text)
    if hits: return hits[-1], "label_last"
    hits = re.findall(r"verdict\W{0,5}(supported|contradicted|unverifiable)\b", text, re.I)
    if hits: return hits[-1].upper(), "verdict_word"
    return None, "unparsed"


def text_tool_call(content):
    """Fallback JSON tool format for models that emit the call as text: {"name": "get_values", "parameters": {...}}."""
    if not content: return None
    m = re.search(r"\{.*\}", content, re.S)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(obj, dict) and (obj.get("name") == "get_values" or obj.get("tool") == "get_values"):
        args = obj.get("parameters") or obj.get("arguments") or {k: v for k, v in obj.items() if k not in ("name", "tool")}
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: return None
        return args
    return None


class Budget:
    def __init__(self):
        self.spent = 0.0
        # all spend counts, including superseded pilot episodes
        for d in (CACHE, os.path.join(ROOT, "runs", "superseded_cache")):
            for f in (os.listdir(d) if os.path.isdir(d) else []):
                try: self.spent += json.load(open(os.path.join(d, f), encoding="utf-8")).get("cost", 0.0) or 0.0
                except Exception: pass
        self.stopped = False


async def post(client, key, body, log_ctx):
    delay = 2.0
    for attempt in range(7):
        t = time.time()
        try:
            r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}", "X-Title": "anon-eval"}, timeout=180)
            lat = time.time() - t
            if r.status_code == 200:
                j = r.json()
                if "choices" in j and j["choices"]:
                    return j, lat
                err = j.get("error")
            else:
                err = r.text[:500]
            if r.status_code in (400, 401, 402, 403, 404):
                with open(CALLS, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**log_ctx, "fatal": r.status_code, "error": str(err)[:1500], "request_messages": body["messages"],
                                        "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
                raise RuntimeError(f"HTTP {r.status_code}: {err}")
        except (httpx.TimeoutException, httpx.TransportError) as e:
            err = repr(e)
        with open(CALLS, "a", encoding="utf-8") as f:
            f.write(json.dumps({**log_ctx, "retry": attempt, "error": str(err)[:500], "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
        await asyncio.sleep(delay + random.random()); delay = min(delay * 2, 60)
    raise RuntimeError(f"gave up after retries: {err}")


async def episode(client, key, sem, budget, mkey, cond, row, pv):
    model = MODELS[mkey]
    ck = hashlib.sha256(f"{model['id']}|{cond}|{pv}|{row['claim_id']}".encode()).hexdigest()
    path = os.path.join(CACHE, ck + ".json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    async with sem:
        if budget.spent >= BUDGET_STOP:
            budget.stopped = True
            return None
        messages, tools = build_messages(row, cond, pv)
        ep = {"claim_id": row["claim_id"], "subset": row["subset"], "model_key": mkey, "model": model["id"], "condition": cond,
              "prompt_version": pv, "cache_key": ck, "requests": [], "tool_calls": [], "cost": 0.0,
              "prompt_tokens": 0, "completion_tokens": 0, "latency": 0.0, "started": datetime.now(timezone.utc).isoformat()}
        n_tools, final_text = 0, None
        for turn in range(MAX_TOOL_CALLS + 2):
            body = {"model": model["id"], "messages": messages, "temperature": 0,
                    "max_tokens": model["max_tokens"], "usage": {"include": True}, **model["extra"]}
            if model.get("no_temperature"):
                body.pop("temperature")
            if not model.get("no_seed"):
                body["seed"] = SEED
            body["provider"] = {"require_parameters": True, **model.get("provider", {})}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto" if n_tools < MAX_TOOL_CALLS else "none"
            ctx = {"model": model["id"], "condition": cond, "claim_id": row["claim_id"], "turn": turn, "prompt_version": pv}
            j, lat = await post(client, key, body, ctx)
            msg = j["choices"][0]["message"]; usage = j.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            ep["cost"] += cost; budget.spent += cost; ep["latency"] += lat
            ep["prompt_tokens"] += usage.get("prompt_tokens") or 0; ep["completion_tokens"] += usage.get("completion_tokens") or 0
            rec = {**ctx, "provider": j.get("provider"), "model_returned": j.get("model"), "request_messages": messages,
                   "response": msg, "finish_reason": j["choices"][0].get("finish_reason"), "usage": usage,
                   "latency": round(lat, 3), "ts": datetime.now(timezone.utc).isoformat()}
            with open(CALLS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            ep["requests"].append({"provider": j.get("provider"), "model_returned": j.get("model"), "usage": usage,
                                   "finish_reason": rec["finish_reason"], "latency": round(lat, 3)})
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls") or []
            if tools and tcs:
                # echo back well-formed arguments: small models sometimes emit two JSON objects glued together, which
                # providers then reject on the next turn; the first object is used and the raw text is logged
                clean = []
                for tc in tcs:
                    raw = tc.get("function", {}).get("arguments") or "{}"
                    try: args = json.JSONDecoder().raw_decode(raw.strip())[0] if isinstance(raw, str) else raw
                    except Exception: args = None
                    clean.append((tc, raw, args))
                echoed = [{"id": tc.get("id"), "type": "function",
                           "function": {"name": tc.get("function", {}).get("name", "get_values"),
                                        "arguments": json.dumps(args if isinstance(args, dict) else {})}} for tc, raw, args in clean]
                messages = messages + [{"role": "assistant", "content": content, "tool_calls": echoed}]
                for (tc, raw, args), e in zip(clean, echoed):
                    if args is None:
                        res = {"error": "arguments were not valid JSON"}
                    elif n_tools < MAX_TOOL_CALLS:
                        res = run_tool(cond, args); n_tools += 1
                    else:
                        res = {"error": "tool call limit reached; give your final answer now"}
                    ep["tool_calls"].append({"args": args, "raw_args": raw, "result": res, "mode": "native"})
                    messages.append({"role": "tool", "tool_call_id": e["id"], "content": json.dumps(res)})
                continue
            targs = text_tool_call(content) if tools and parse_verdict(content)[0] is None else None
            if targs is not None and n_tools < MAX_TOOL_CALLS:
                res = run_tool(cond, targs); n_tools += 1
                ep["tool_calls"].append({"args": targs, "result": res, "mode": "text"})
                messages = messages + [{"role": "assistant", "content": content},
                                       {"role": "user", "content": "Tool result: " + json.dumps(res)}]
                continue
            final_text = content
            break
        v, mode = parse_verdict(final_text)
        ep.update({"final_text": final_text, "verdict": v, "parse_mode": mode, "n_tool_calls": n_tools,
                   "finished": datetime.now(timezone.utc).isoformat()})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ep, f)
        return ep


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--subsets", nargs="+", required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-version", default="v1"); ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--sample", default=os.path.join(ROOT, "data", "sample.csv"))
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    # cut-off (DEVIATIONS D3.4/D5.1): after 02:45 local time on 2026-10-18 only the retry pass of a block started earlier may run
    sig = "|".join([" ".join(a.models), " ".join(a.conditions), " ".join(a.subsets), str(a.limit), a.prompt_version])
    started = os.path.join(ROOT, "runs", "started_blocks.txt")
    seen = set(open(started, encoding="utf-8").read().splitlines()) if os.path.exists(started) else set()
    now = datetime.now()
    if now >= datetime(2026, 10, 18, 2, 45) and sig not in seen and not os.environ.get("ALLOW_AFTER_CUTOFF"):
        print(f"REFUSED: new block after the 02:45 cut-off: {sig}", flush=True); return
    if sig not in seen:
        with open(started, "a", encoding="utf-8") as f: f.write(sig + "\n")
    s = pd.read_csv(a.sample, dtype=str)
    s = s[s.subset.isin(a.subsets)]
    if a.limit:
        s = s.sample(n=min(a.limit, len(s)), random_state=SEED)
    key = read_key(); budget = Budget(); sem = asyncio.Semaphore(a.concurrency)
    print(f"start: {len(s)} claims x {a.models} x {a.conditions}; spent so far ${budget.spent:.4f}", flush=True)
    async with httpx.AsyncClient(http2=False, limits=httpx.Limits(max_connections=a.concurrency + 5)) as client:
        sems = {m: asyncio.Semaphore(min(a.concurrency, MODELS[m].get("conc", a.concurrency))) for m in a.models}
        jobs = [episode(client, key, sems[m], budget, m, c, r, a.prompt_version)
                for m in a.models for c in a.conditions for r in s.to_dict("records")]
        done, failed, t0 = 0, 0, time.time()
        for fut in asyncio.as_completed(jobs):
            try:
                ep = await fut
            except Exception as e:
                failed += 1; print("EPISODE FAILED:", repr(e)[:300], flush=True); continue
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} done, {failed} failed, ${budget.spent:.3f}, {time.time() - t0:.0f}s", flush=True)
    print(f"end: {done} done, {failed} failed, spent ${budget.spent:.4f}, budget_stopped={budget.stopped}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
