# DEVIATIONS: every change after FREEZE.md was first written

FREEZE.md was written at 23:17 local time on 2026-10-17, before any model call. Every change made after that point is listed
here, with its reason and evidence. Since 2026-10-17 ~23:58 local time, **prompt v2, the tool (`src/vintage_tool.py`) and the
harness (`src/run.py`) are frozen**. Only crash fixes are allowed from now on, each logged in the last section.

## D1. Models changed before any call (budget)
- **What:** F changed from Claude Sonnet 5.5 to DeepSeek-V4-Pro (open weights), which runs the full design. Sonnet 5.5
  became S, a closed-frontier check on FLIP and HOLDOUT.
- **Why:** budget. All-Sonnet was estimated at about $27, against a $20 credit.
- **When:** before any model call; the author approved it.

## D2. Pilot fixes (pilot v1 → prompt v2)
The v1 pilot ran 10 FLIP and 10 STABLE claims (random_state 2027) × C0/C1/C2 × A, F, S. Its episodes were moved to
`runs/superseded_cache/` and are never reported. All costs are counted in the budget.

| # | Change | Affects | Evidence | Reason |
|---|---|---|---|---|
| D2.1 | `seed` and `temperature` not sent to Sonnet 5.5 | S | With strict parameter routing, every S tool request returned HTTP 404 "No endpoints found that can handle the requested parameters". A direct test showed that dropping `temperature` alone fixes it. | Sonnet 5.5 accepts neither parameter, so S runs at the model's default sampling. In C0 without strict routing, OpenRouter had already dropped them silently, so S-C0 behaviour is unchanged. |
| D2.2 | `max_tokens` 400 → 1000 | A, F, S (B already 2500) | S stopped at the cap (finish_reason=length) before writing its JSON in 6/20 C0 and 13/20 C1 episodes. | Truncation is a harness artefact, not model behaviour. |
| D2.3 | Prompt v2 adds one line: "Keep your reasoning brief: a few short sentences at most." | all | Same as D2.2. Also keeps cost down. | |
| D2.4 | Tool arguments: when a model emits two JSON objects glued together, the first object is used and echoed back; the raw text is logged as `raw_args` | all (seen in A) | Echoing the malformed text back made the host reject the next turn with HTTP 400 "Extra data: line 1 column 96". | Harness bug. Malformed calls are now counted and reported as a result (see the tool-reliability table). |
| D2.5 | The tool accepts FRED-style periods (`2026M01`) as well as `YYYY-MM` and `YYYYQn` | **all models**: the change is in `vintage_tool._ts`, the single parser every tool call passes through | A produced 26 "unrecognised period" errors in 20 C2 episodes. | A tool that rejects a standard notation tests format-guessing, not verification. |
| D2.6 | Llama concurrency 20 → 6 | A | Upstream HTTP 429 rate limits (Groq, CoreWeave). | Throughput only. |
| D2.7 | DeepSeek-V4-Pro: no `seed`; requests pinned to hosts Baidu then StreamLake (both fp8), `allow_fallbacks=false`; reasoning disabled (`reasoning.enabled=false`, as in FREEZE) | F | With `seed` required, routing went to hosts priced up to 9× higher, including an fp4 host (AtlasCloud). The v2 pilot F episodes ran on mixed hosts; they were moved to `runs/superseded_cache/` and rerun pinned. | Cost, and one fixed precision (fp8). |
| D2.8 | Per-turn `provider.require_parameters=true` for every request (it was set only for tool requests before) | all | n/a | No model can silently ignore a parameter we send. |

Prompt v2 pilot (same 20 claims, all four models): 0 parse failures, 0 truncations.

## D3. Changes requested by the author after the pilot, before any full-block result was seen

- **D3.1 Pilot claims excluded from the main results.** The 20 pilot claims were used to tune prompt v2, so the main
  results exclude them; the including-pilot numbers appear once, as a sensitivity check. The ids are listed in
  FREEZE.md.
- **D3.2 Headline H1 contrast changed.** It is now A-C2 (Llama-3.1-8B with the as-of tool) vs **S-C0** (Claude Sonnet
  5.5, no tool) on FLIP, with F-C0 (DeepSeek-V4-Pro) reported alongside. FREEZE had A-C2 vs F-C0 as primary and S as
  secondary.
  - Disclosure: the v2 pilot had already been run when this change was made. On the 10 pilot FLIP claims, S-C0 said
    SUPPORTED on 10 of 10, F-C0 on 2 of 10, and A-C2 on 6 of 10.
  - The change therefore makes H1 *harder* to support, not easier.
  - The pilot claims are excluded from the test (D3.1).
  - **Revised at 00:55 local time, before any S full-block result existed: H1 stays as frozen** (A-C2 vs F-C0, confirmatory,
    in the Holm family with H2 and H3 for A and F). A-C2 vs S-C0 is reported prominently but labelled a *secondary,
    exploratory* comparison, because it was introduced after the pilot. It is never called preregistered.
- **D3.3 Missing episodes are reported, never dropped.** After each block's retry pass, the remaining failures are
  counted per model × condition and reported. Every comparison runs only on claims that every compared cell completed,
  so it stays paired.
- **D3.4 Cut-off.** No new block is launched after 02:45 local time. The paper reports completed blocks only.
- **D3.5 Tool-call reliability is a reported result.** Per model × condition: the share of malformed or concatenated
  tool calls (from `raw_args`), the tool error rate, and the share of C2 calls with `as_of` equal to the publication
  date.
- **D3.6 DeepSeek host stalls.** If the pinned hosts stall for more than 10 minutes, one more fp8 host may be added and
  logged here. An unpinned route is never used.
- **D3.7 S's HOLDOUT cells.** Limited to C0 and C2 (the H4 contrast), because S costs about $0.026 per claim across
  three conditions (pilot evidence). S-HOLDOUT-C1 runs only if the budget allows.

- **D3.8 Budget cap raised from $18 to $23** (the author added $5 of credit). The trimmed S cell (HOLDOUT × C1) runs
  after block 7, as block 8. S × C1 on FLIP was already in block 4. This changes the cap only; the harness is otherwise
  untouched.
- **D3.9 Knowledge-cutoff split (analysis only).** Every model × condition is split by release date: releases on or
  before the end of the model's cutoff month, and releases after it. Cutoff months: A 2023-12, B 2024-06, F 2025-05
  (unofficial), S 2026-06. Only S has releases on both sides: Jan to Jun 2026 before its cutoff, Jul and Aug 2026 after.
  **Exploratory**, added after the pilot. It is not a natural experiment, and a difference between the two sides is not
  evidence of memorisation on its own: the sides also differ in release month, series mix and revision history, and
  the cutoffs themselves are uncertain (S: Anthropic's stated month; F: unofficial third-party listings). The
  A-C2 vs S-C0 contrast is reported within each side, descriptively.
- **D3.10 Host sensitivity for A (analysis only).** A's accuracy, contradicted rate and unverifiable rate by serving
  host, using only episodes served entirely by one host, and only hosts with n >= 30.
- **D3.11 Retry sweep: superseded by D3.12.** Each block now runs a second pass that retries its failures; nothing
  else re-requests episodes.
- **D3.12 Run order changed at the block-2/3 boundary (00:50 local time).** The author approved it; cells, prompts, sample and
  settings are unchanged, and nothing completed was restarted. New order: 2-retry, 3, 4 (S on FLIP), 6a (HOLDOUT × A,
  F), 7 (HOLDOUT × S × C0/C2), 8 (HOLDOUT × S × C1), 5 (B on FLIP+STABLE), 6b (HOLDOUT × B). Optional model B goes
  last, at its tested concurrency of 20. If B misses the 02:45 cut-off, it is reported as not run.
  The old scripts (`runs/blocks.sh`, `runs/blocks_extra.sh`) were stopped. Block 2's first-pass process was left
  running; `runs/blocks2.sh` waited for it to exit before starting.

- **D3.13 Parallel lanes by model (00:55 local time, the author asked for it).** Three lanes run at once; blocks within a lane
  run in order, each with its retry pass and metrics. Lane 1 (A, F): 3 then 6a. Lane 2 (S): 4, 7, 8. Lane 3 (B): 5 then
  6b. Upstream rate limits are per model, so the lanes do not compete. Cells, prompts and settings are unchanged.
  Known side effect: each process's $23 guard sees the spend of cached episodes and of its own calls, not the live spend
  of the other lanes, so the binding limit is the OpenRouter balance. The analysis reader now skips a cache file that
  another lane is still writing (analysis only); the final analysis runs after every lane has finished.

- **D3.14 Block 8 gated on credit (00:58 local time).** Block 4 cost about $9.50, and the author's $5 top-up had not yet
  reached the account: $9.04 remained. A credit failure (HTTP 402) is fatal and not retried, so running block 8 while
  lanes 1 and 3 were active could have left essential episodes missing. Lane 2 was replaced by `runs/lane2b.sh`. It
  waits for block 7's first pass, runs block 7's retry pass, then waits for lanes 1 and 3 to finish, and starts block 8
  only if at least $4.50 of credit remains (and before the 02:45 cut-off). Otherwise block 8 is reported as not run.

- **D3.15 B pinned to one host (01:00 local time, the author asked for it).** `openai/gpt-oss-20b` now runs only on Darkbloom
  (fp8), with no fallback. Before this, block 5 had been running unpinned since 00:52. Its 980 B episodes (including
  the v2 pilot) were spread over 8 hosts with mixed precision: Darkbloom fp8 served 76% of requests; Parasail, AkashML
  and CoreWeave are fp4; DekaLLM bf16; Google, Groq and DeepInfra a few each. They were moved to
  `runs/superseded_cache/` and are never reported (cost under $0.10, counted in the budget). The block was restarted
  pinned, at concurrency 20, under the same signature.
  - Why: A's host-sensitivity table shows large differences between hosts. B is optional and had not been analysed, so
    removing the confound costs little.
  - Disclosure: a development table printed at 00:53 showed partial B cells (n = 48 to 172). The decision was based on
    A's host variance, not on B's numbers.
  - Darkbloom was chosen because it served most of B's traffic and accepted our parameters. DeepInfra (bf16) returned
    HTTP 429 on a test call.
  - If Darkbloom cannot sustain the load, B's unfinished cells are reported as missing. No second host is added.

- **D3.16 Budget cap raised from $23 to $27 (01:05 local time).** The author added credit: total credits are now $30.50, with
  $1.67 of usage that predates this project. Our spend was $16.98, and the remaining plan (block 7 rest, blocks 3, 6a,
  5 and 6b rest, block 8) is projected at about $6. At $23 the guard would have stopped block 8 part-way. $27 still
  leaves about $1.80 of credit unused as a margin for in-flight reservations.

- **D3.17 Post-hoc analyses, added after results were seen (01:15 local time; all exploratory, analysis only).**
  - `results/integrity.csv`: expected, cached and missing episodes per design cell, read from `runs/cache/` only.
    0 unreadable cache files. The 211 garbled lines (D5.3) affect only the audit call log; no metric reads it.
  - Truncation: `truncated_rate` (no verdict, finish_reason = length) and `accuracy_excl_truncated`, per cell. Scoring
    is unchanged: as FREEZE specifies, a missing verdict counts as wrong and never as UNVERIFIABLE. The
    excluding-truncation accuracy is a sensitivity column only.
  - `results/flip_by_claimtype.csv`: FLIP results split into revision claims (claim type `revision_statement`, 82 of
    322) and all other claims. For C2 it also gives the share of episodes that queried two or more distinct vintages,
    and accuracy by that split. That split is self-selected, so it is descriptive and not causal. Framing: prompt v2
    tells the model to set `as_of` to the publication date. That under-specifies revision claims, which need two
    vintages, and the 3-call budget compounds it. We report it as a limitation of our tool and prompt design.
  - `results/label_noise.csv`: from the 40 sampled C2 failures (`label_flag` column in
    `results/error_analysis_manual.csv`), 1 is a plausible reference-label error and 2 are ambiguous. This estimates how
    much wrong labels lower *measured* C2 accuracy (A about 1.8 pp; F 0 to 1.3 pp). It is **not** a bound on total
    label noise: a wrong label that the model also accepts cannot appear among failures. The overall label-noise
    reference is the audit's blind validation (11 of 153 disagreements, 7.2%).

## D4. Facts recorded for the paper (not changes)
- Model IDs (OpenRouter):
  - A: `meta-llama/llama-3.1-8b-instruct`, served by several hosts (Groq, CoreWeave, DeepInfra, Novita, Cloudflare;
    fp8, bf16 or unstated). Logged per request.
  - B: `openai/gpt-oss-20b`, pinned to Darkbloom (fp8) from 01:00 local time (D3.15).
  - F: `deepseek/deepseek-v4-pro`, which OpenRouter lists as "DeepSeek V4 Pro 0423" (as of 2026-10-17); fp8 on Baidu.
  - S: `anthropic/claude-sonnet-5.5` via "Claude Platform on AWS".
- Knowledge cutoffs:
  - Llama-3.1-8B: December 2023 (Meta model card: "The pretraining data has a cutoff of December 2023").
  - gpt-oss-20b: June 2024 (OpenAI developer docs).
  - DeepSeek-V4-Pro: not stated by DeepSeek; third-party listings give May 2025.
  - Claude Sonnet 5.5: June 2026 (Anthropic). This is **inside** the claim window (Jan to Aug 2026), so S-C0 may recall
    some of these releases, as first printed or as revised.
- C0 results depend on each model's training cutoff relative to Jan to Aug 2026.
- Reference labels, stated exactly: FLIP and STABLE labels are the audit pipeline's automated *supported* verdict at
  the publication vintage. That verdict agreed with blind (LLM) adjudication on 142 of 153 claims from two held-out
  releases (0.928; 95% CI 0.876 to 0.959). This validates the verdict's accuracy on unseen releases; it does not
  establish that each FLIP or STABLE label is individually correct. Human coding of a subset is pending.

## D5. Crash-fix and operations log (after the freeze)
- **D5.1 (00:35 local time): cut-off guard in `src/run.py`.** An invocation records its block signature (models, conditions,
  subsets, limit, prompt version) in `runs/started_blocks.txt`. After 02:45 local time a signature not already recorded is
  refused, so only the retry pass of a block that began earlier can run. Blocks 1 and 2 were recorded by hand, since
  they started before the guard existed. Operations only: no prompt, tool, parameter or model change.
- **D5.2 (00:30 local time): `src/numbers.py` renamed to `src/paper_numbers.py`.** It shadowed Python's standard `numbers`
  module and crashed numpy's import inside `src/`. No run was affected: block 1's process had already imported its
  modules, and the file was renamed before block 1's retry pass.
- **D5.3 (01:04 local time): one call log per process.** With three lanes appending to one `runs/calls.jsonl`, writes
  interleaved and 211 lines became unparseable. Records in those lines are lost from the per-call log. Episode results
  are not affected: each episode is its own file in `runs/cache/`, with per-request usage, host and tool calls, and the
  full request messages can be rebuilt from the frozen prompts, the sample and the logged tool results. From 01:04 each
  process writes `runs/calls_<pid>.jsonl`. Logging only.
- **D5.4 (01:04 local time): block 7 rerun at concurrency 5, then 3.** Its first pass died at 00:59 when the parent lane
  script was replaced (its output pipe closed). The retry pass then lost 190 episodes to HTTP 402 "would exceed your
  available credits given your current in-flight requests": OpenRouter reserves the worst-case cost of every in-flight
  request, and only $5.58 remained. Lower concurrency keeps the reservation small. Throughput only; same signature,
  prompts and settings.
- **D5.5 (noted): Llama host rejects parallel tool calls.** 5 Llama C1 requests got HTTP 400 "This model only supports
  single tool-calls at once!" from one host, after the model emitted two tool calls in one turn. This is fatal for the
  episode. Retry passes may route to another host; anything still failing is reported as missing. Not fixed (frozen
  harness).

- **D5.6 (02:05 local time): correction to D2.4.** The concatenated tool-call arguments were **not** model behaviour. Every
  one of the 3,836 concatenated Llama calls (C1 and C2, pilot excluded) holds two *identical* JSON objects, and they
  come from one host: 1,442 of 1,443 calls in episodes served only by CoreWeave, against 0 of 4,762 on Groq. It is a
  serving artifact. The D2.4 fix (use the first object) made it harmless, and no tool result changed. The paper reports
  it as a host artifact, not as a model tool-use failure. Analysis only; nothing was rerun.
- **D5.7 (01:57 local time): block 8 passed its credit gate** ($10.50 remaining) and started before the 02:45 cut-off.
- **D5.8 Text corrections before submission (no reruns, no new analyses).** Contributions now give 2,132 claims
  (pilot excluded) instead of 2,152. H1 notes that F-C0 scores 51.5% on all 322 claims against 51.4% on the 321 paired
  claims. DeepSeek's cutoff is described as "reported" (it comes from third-party listings). "Larger models reached
  94–97%" became "DeepSeek and Sonnet". The error analysis is described as covering Llama and DeepSeek only. The
  duplicate-argument counts are restricted to FLIP + STABLE, to match the tool table: 3,269 calls, all identical
  duplicates; 1,262 of 1,263 on CoreWeave-only episodes and 0 of 4,160 on Groq-only.
