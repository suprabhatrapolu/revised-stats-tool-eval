# FREEZE: evaluation set and hypotheses

Frozen 2026-10-17, before any model call. After this file is approved, the sample, hypotheses, metrics and tests below
do not change. Prompts are frozen separately after the 20-claim pilot (see `prompts/PROMPTS_FROZEN.md`); after that
only logged bug fixes are allowed.

## Source (read-only)

The companion preregistered audit's repository (path given by `GA_ROOT`; cited in the paper as Anonymous, 2026, under
review). Nothing in it is written to. Inputs:

- `data/processed/all_verdicts.csv`: 4,669 claims from 19 preregistered US releases (Jan to Aug 2026).
- `data/processed/{es-2026-09-04,cpi-2025-12-18}/verdicts.csv`: the two held-out releases.
- `analysis/holdout3/scored.csv` and `analysis/holdout4/adjudications/b*.csv`: blind adjudication labels.
- `data/vintages/*_vintages.csv`: the ALFRED vintage matrices for 8 series. The latest vintage is 2026-09-04 for
  PAYEMS, UNRATE and CES0500000003; 2026-08-12 for the CPI series; and 2026-08-26 for GDPC1.

Built by `src/build_sample.py`; output in `data/sample.csv`.

## Evaluation set: censuses, not samples

| subset | definition | n | documents | reference label |
|---|---|---|---|---|
| FLIP | checkable; pipeline verdict *supported* at the publication vintage; *contradicted* at the latest vintage | 332 | 141 | SUPPORTED |
| STABLE | checkable; *supported* at the publication vintage and at the latest vintage | 1,529 | 302 | SUPPORTED |
| HOLDOUT | every checkable claim in the two held-out releases | 291 | 36 | blind adjudication |
| **total** | | **2,152** | | |

- STABLE is 1,529, not 1,530. Claim `3f0fe57768fb14fd` appears twice, byte-identical, in `all_verdicts.csv` and is
  evaluated once. FLIP plus STABLE is therefore 1,861 unique claims; the audit counts 1,862 supported rows.
- FLIP by series: PAYEMS 276, CES0500000003 (average hourly earnings) 28, GDPC1 28. CPI and UNRATE have no flips.
- HOLDOUT adjudicated labels: supported 216; contradicted 6; unverifiable 49; non-checkable 20. For scoring,
  "unverifiable" and "non-checkable" both map to UNVERIFIABLE.
- Why these reference labels: FLIP and STABLE use the pipeline's *supported* verdict at the publication vintage, which
  agreed with blind adjudication on 142 of 153 unseen claims (0.928; 95% CI 0.876 to 0.959). The pipeline's
  *contradicted* verdict is not used as a reference anywhere (its precision on unseen releases is 0.167).
- For expectation claims ("beat/miss" type), the consensus figure the pipeline used is passed to the model in every
  condition. 30 HOLDOUT expectation claims have no pipeline consensus and are passed without one.
- Known property: for the Employment Situation half of HOLDOUT (`es-2026-09-04`), the publication vintage *is* the
  latest vintage in the archive, so C1 and C2 receive the same numbers there. HOLDOUT tests calibration, not revision.

`data/claim_ids.txt`: one `subset,claim_id` line per claim, sorted, LF line endings.
**SHA-256: `e90eedd913f1b7cae99034356a3a1df14f46028cd3a863bbda3b73aa688019c2`**

## Task, conditions, models

Question, the same in every condition: *"Was this claim accurate according to the official record available when it
was made?"* Answer: SUPPORTED, CONTRADICTED or UNVERIFIABLE, plus one sentence. Inputs: claim text, publication date,
series, reference period, measure, and the consensus figure for expectation claims.

- **C0, no tool.** The model answers from its own knowledge.
- **C1, latest-value tool.** `get_values(series, start_period, end_period)` returns the latest vintage in the archive.
- **C2, as-of tool.** `get_values(series, start_period, end_period, as_of)` returns the vintage in force on `as_of`.
  The prompt tells the model to use the publication date.
- C3 (both tools, no hint) is dropped.

Both tools read only the local ALFRED archive. They return level, period-on-period change, percent change and
year-on-year percent change (SAAR growth for GDP), all computed within one vintage. At most 3 tool calls per claim.

Models (exact OpenRouter IDs recorded in every log line):

- **A, small open:** `meta-llama/llama-3.1-8b-instruct` (8B, dense).
- **F, frontier-scale open:** `deepseek/deepseek-v4-pro` (open weights, `deepseek-ai/DeepSeek-V4-Pro`; 1.6T total,
  49B active parameters, MoE), reasoning off. It runs the full design.
- **S, closed frontier check:** `anthropic/claude-sonnet-5.5`, reasoning off. It runs FLIP and HOLDOUT under every
  condition, not STABLE (budget).
- **B, small open, if time allows:** `openai/gpt-oss-20b` (open weights; 21B total, 3.6B active), reasoning effort low.

Temperature 0, seed 2027, max 3 tool calls.

## Run order

Each block is completed, with its metrics, before the next one starts. The paper reports only completed blocks.

1. FLIP × {C0, C2} × {A, F}
2. STABLE × {C0, C2} × {A, F}
3. FLIP + STABLE × C1 × {A, F}
4. FLIP × {C0, C1, C2} × S
5. FLIP + STABLE × {C0, C1, C2} × B
6. HOLDOUT × {C0, C1, C2} × {A, F, B}
7. HOLDOUT × {C0, C2} × S (the H4 contrast); HOLDOUT × C1 × S only if the budget allows

Amendment after the pilot (before any full block, recorded with the cost evidence in `prompts/CHANGELOG.md`): S costs
about $0.026 per claim across three conditions, so S's HOLDOUT cells are limited to the H4 contrast.

## Hypotheses (directional, stated before any run)

- **H1 (design beats scale).** On FLIP, small model A with the as-of tool (A-C2) is more accurate than frontier-scale F
  with no tool (F-C0). The same comparison against S-C0 is reported as a secondary test.
- **H2 (the latest-value tool hurts).** On FLIP, for each model, the latest-value tool (C1) gives a higher
  false-contradiction rate than no tool (C0): models anchor on today's revised number.
- **H3 (vintage is the mechanism).** On FLIP, for each model, C2 gives a lower false-contradiction rate than C1.
- **H4 (no free lunch).** C2 does not win by answering SUPPORTED to everything. On HOLDOUT, C2's accuracy is no lower
  than C0's for the same model. Point estimates are reported for both, plus C2's accuracy on HOLDOUT claims whose
  label is not SUPPORTED (75 claims).

STABLE accuracy is reported for every model × condition. It is not a hypothesis test, because its reference is always
SUPPORTED. That is why H4 uses HOLDOUT.

## Metrics and tests

- **Headline:** the FLIP false-contradiction rate, i.e. the share of FLIP claims a model labels CONTRADICTED.
- Accuracy against the reference label for each model × condition × subset. A parse failure counts as wrong and is
  reported separately.
- Paired McNemar tests (exact binomial on discordant pairs) on FLIP correctness for: A-C2 vs F-C0 (H1); C1 vs C0 per
  model (H2); C2 vs C1 per model (H3). Holm correction across the primary family (H1, H2, H3; up to 5 tests with A
  and F). Reported with and without correction.
- 95% CIs: bootstrap with 2,000 resamples, resampling source documents (`doc_id`), seed 2027.
- FLIP false-contradiction rate broken down by series: payroll (PAYEMS), average hourly earnings (CES0500000003),
  GDP (GDPC1).
- Cost per 100 claims in USD (from OpenRouter's reported cost), median latency per claim, and tool calls per claim.
- C2 error analysis: 20 random C2 failures on FLIP (seed 2027), classified as wrong tool arguments, rounding,
  direction, parse failure, or other.

## Guardrails

- Budget: $20 of credit. Stop and report if spend passes $18. Estimate before the pilot: about $13 for all six blocks.
- Responses are cached by hash(model, condition, prompt version, claim_id). A rerun is never billed twice.
- Every call is logged as JSONL: model, provider, condition, claim_id, full messages, tool calls and results,
  tokens, cost, latency, timestamp.
- No number reaches the paper unless it is in `results/`; `paper/NUMBERS.md` maps each number to its source.

## Pilot claims (excluded from the main results; see DEVIATIONS.md D3.1)
Drawn with `DataFrame.sample(n=10, random_state=2027)` within FLIP and within STABLE.

- FLIP: 138cdd1c7cf70697 151872e98c472519 228d5bc7900007de 29a3cc972c983566 3d119ec34aed4048 424d6ed582447872
  95da6be68805df1d c48d5cf255f35622 d1761a6ab195c102 e025a0e93f02d0db
- STABLE: 005041f1faa0586c 2569ea42a9eb068e 2b83eccd9f91686d 3e7a8f4a8eaeeaae 8c32301b9d48cea3 92078107a518bbc5
  bf15e0b0d781c83f ccdf75601939638a dad97b1c6056254f ee0254a9f423fd4b

Main-result populations are therefore FLIP 322, STABLE 1,519, HOLDOUT 291 (the pilot drew no HOLDOUT claims).

## Amendments after the pilot (full detail in DEVIATIONS.md)
- H1 stays as frozen (A-C2 vs F-C0, confirmatory). A-C2 vs S-C0 is a secondary, exploratory comparison (D3.2, revised).
- The knowledge-cutoff split is exploratory (D3.9). The run order puts optional model B last (D3.12).
- Missing episodes are reported per cell; every comparison is paired on the claims all compared cells completed (D3.3).
- No new block is launched after 02:45 local time (D3.4).
