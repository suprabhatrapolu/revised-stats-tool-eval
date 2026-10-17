# Prompt and harness changelog

## v1 (2026-10-17, pilot only; superseded, never reported)
Pilot of 20 claims (10 FLIP, 10 STABLE; random_state 2027) × C0/C1/C2 × A, F, S. Found:
- S (Claude Sonnet 5.5) accepts neither `seed` nor `temperature`; with strict parameter routing, every tool request 404'd.
  Fix: omit both for S. S runs at the model's default sampling.
- S hit the 400-token cap before writing its JSON (finish_reason=length): 6/20 C0 and 13/20 C1 episodes had empty output.
  Fix: max_tokens 1000 for A, F, S; v2 adds the line "Keep your reasoning brief: a few short sentences at most."
- A (Llama-3.1-8B) sometimes emitted two glued JSON objects as tool arguments; echoing them back verbatim made the
  provider reject the next turn (HTTP 400). Fix: echo the first parsed object; log the raw text as `raw_args`.
- A wrote FRED-style periods (`2025M12`); the tool rejected them (26 tool errors in 20 C2 episodes).
  Fix: the tool also accepts `YYYYMmm`.
- A hit upstream 429 rate limits at concurrency 20. Fix: concurrency 6 for A.
- F (DeepSeek-V4-Pro) routed to hosts priced up to 9× the cheapest because the cheapest hosts do not accept `seed`.
  Fix: no seed for F; pinned to Baidu then StreamLake (both fp8), no other fallback.
v1 episodes moved to runs/superseded_cache/.

## v2 (2026-10-17): FROZEN
v1 plus the brevity line. Pilot v2 (same 20 claims, all four models): 0 parse failures, 0 truncations.
F's v2 pilot episodes ran on mixed hosts before pinning; they were moved to runs/superseded_cache/ and rerun pinned.
From here on only bug fixes are allowed, each logged below.
