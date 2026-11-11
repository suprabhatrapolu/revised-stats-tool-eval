# Today's Number, Yesterday's Claim

**How tool design shapes LLM checks of claims about revised statistics.**

By Suprabhat Rapolu (Dirmacs Private Limited, dirmacs.com).

Code, frozen protocol and aggregate results for a short paper accepted at the OE-Agent workshop at ACML 2026 (*Open
and Efficient Agentic AI for Resource-Constrained Environments*; workshop is non-archival, 1 Dec 2026, Melbourne).

---

## The problem in one paragraph

Official statistics are revised after they are published. A news story that correctly reported "employers added
50,000 jobs in December" can be contradicted a month later, once the figure is revised to 48,000. A fact-checking
agent that looks up *today's* number will call an accurate claim false. The right check is against the figure **as it
was published on the day the claim was made** (its *vintage*). This repository tests how much that design choice
matters for LLM agents, and whether getting the tool right lets a small open model compete with much larger ones.

## What we tested

**Claims.** News claims about US jobs, earnings and GDP from 19 preregistered releases (January to August 2026), each
already linked to an official series, reference period and measure by a companion study. We evaluate whole
populations, not samples:

| Subset | Claims | What it is |
|---|---|---|
| FLIP | 322 | Accurate when published (by the record of that day), contradicted by today's record |
| STABLE | 1,519 | Accurate when published and still matching today's record |
| HOLDOUT | 291 | Every checkable claim from two held-out releases, labelled by blind adjudication; 75 of its labels are CONTRADICTED or UNVERIFIABLE |

Counts exclude 20 pilot claims that were used to tune the prompt.

**Three tool conditions.** Each model is asked: *was this claim accurate according to the official record available
when it was made?*

| Condition | Tool |
|---|---|
| C0 | No tool; the model answers from what it knows |
| C1 | `get_values(series, start, end)`: today's (latest) figures |
| C2 | `get_values(series, start, end, as_of)`: figures exactly as published on `as_of`, set to the claim's date |

Both tools read only a local copy of ALFRED, the St. Louis Fed's public archive of past data vintages.

**Four models,** called through OpenRouter: Llama-3.1-8B (8B parameters), gpt-oss-20b (3.6B active), DeepSeek-V4-Pro
(1.6T total, 49B active) and Claude Sonnet 5.5 (closed frontier model).

## Main results (FLIP claims)

| Model | C0 no tool | C1 today's figures | C2 figures as of the claim date |
|---|---|---|---|
| | accuracy / false contradictions | accuracy / false contradictions | accuracy / false contradictions |
| Llama-3.1-8B | 46.6% / 49.4% | 29.8% / 66.8% | **64.5%** / 30.2% |
| gpt-oss-20b | 31.7% / 46.9% | 2.2% / 92.9% | **75.8%** / 19.6% |
| DeepSeek-V4-Pro | 51.5% / 38.8% | 0.9% / 59.3% | **87.0%** / 8.1% |
| Claude Sonnet 5.5 | 77.6% / 2.5% | 46.9% / 17.4% | **95.3%** / 3.7% |

- **Preregistered H1 holds.** The 8B model with the as-of tool beats DeepSeek-V4-Pro without tools (64.5% vs 51.4% on
  paired claims; Holm-adjusted p = 0.0009). DeepSeek's reported training cutoff predates these claims, so without a tool it
  cannot know the figures. What it does instead is the interesting part: it labels 38.8% of accurate claims CONTRADICTED and
  says UNVERIFIABLE only 9.6% of the time.
- **The latest-value tool makes every model worse.** Given today's figures, every model rejects accurate reporting
  more often than with no tool at all.
- **The as-of tool fixes most of it.** With DeepSeek and Sonnet, 94–97% of claims that need a single vintage are
  checked correctly.
- **What remains is revision claims** ("December was revised down by 2,000"). These need two vintages, which our
  prompt did not ask for.
- **Exploratory:** Claude Sonnet 5.5, whose training extends into 2026, scores 89.6% without tools on releases before
  its knowledge cutoff and 37.0% after it.

- **What the study does not show.** On HOLDOUT, always answering SUPPORTED would score 74.2%, and no model reaches
  that under any condition (best: Claude Sonnet 5.5 with the as-of tool, 69.1%). On the 75 HOLDOUT claims that are not
  SUPPORTED, the as-of tool raises accuracy for Llama-3.1-8B (20.0% to 31.1%), barely moves gpt-oss-20b and
  DeepSeek-V4-Pro, and lowers it for Claude Sonnet 5.5 (44.0% to 29.3%). So the as-of tool fixes the revision failure,
  but this study does not show that it helps a model separate accurate claims from inaccurate ones. These two
  analyses were added after peer review and are marked post-hoc (`DEVIATIONS.md`, D6).

Full tables, confidence intervals and tests are in `results/` and in the paper.

## How the study was run (and how to check it)

1. **Frozen before any model call.** `data/FREEZE.md` fixes the populations, hypotheses (H1–H4), metrics and tests.
   `data/claim_ids.txt` is the exact claim list, and its SHA-256 is recorded in FREEZE.md.
2. **Pilot.** 20 claims were used to fix harness bugs and add one line to the prompt. `prompts/CHANGELOG.md` records
   prompt v1 → v2, and those claims (`data/pilot_ids.txt`) are excluded from all results.
3. **Every later change is logged.** `DEVIATIONS.md` lists each change with its time, reason and evidence: model and
   host choices, budget, run order, and corrections of our own mistakes.
4. **Every number is traceable.** `paper/NUMBERS.md` maps each number in the paper to the results file and row it
   comes from.
5. **Nothing is silently dropped.** `results/integrity.csv` lists expected, completed and missing episodes for every
   cell. 3 of about 23,400 episodes are missing after retries, and a missing verdict counts as wrong.

## Repository map

```
data/
  FREEZE.md            frozen protocol: populations, hypotheses, metrics, tests, pilot ids, amendments
  claim_ids.txt        frozen claim list (subset, claim_id); hash in FREEZE.md
  pilot_ids.txt        the 20 pilot claims
DEVIATIONS.md          every change after the freeze, with reasons
prompts/
  v1/, v2/             system prompt, condition texts, user template, tool schemas (v2 = frozen)
  CHANGELOG.md         what the pilot changed and why
src/
  build_sample.py      builds the evaluation set from the companion study's data
  vintage_tool.py      the get_values tool over the ALFRED vintage archive
  run.py               runs model x condition x claim episodes (caching, retries, logging, budget guard)
  analyze.py           metrics, bootstrap CIs, McNemar tests, cutoff split, host and claim-type breakdowns
  error_analysis.py    samples and tags C2 failures
  paper_numbers.py     writes every number used in the paper, with its source (paper/NUMBERS.md)
  tables.py            writes the paper's tables from results/
  final_checks.py      page, anonymity and placeholder checks on the PDF
results/
  metrics.csv          accuracy, contradiction and abstention rates (with 95% CIs) per model x condition x subset
  tests.csv            paired McNemar tests, Holm-adjusted where preregistered, plus HOLDOUT paired differences
  integrity.csv        expected vs completed vs missing episodes per cell
  tool_use.csv         tool calls per claim, errors, malformed arguments, as_of use
  flip_by_claimtype.csv  revision claims vs other claims
  cutoff_split.csv     results by release date relative to each model's knowledge cutoff (exploratory)
  flip_by_series.csv   false contradictions by statistic (payrolls, earnings, GDP)
  a_by_host.csv        Llama-3.1-8B results by serving host
  label_noise.csv, error_analysis_C2_summary.csv, holdout_nonsupported.csv, missing.csv
  sensitivity_incl_pilot/   the same metrics and tests with the pilot claims included
paper/NUMBERS.md       each paper number -> source file and row
```

## Reproducing

```
pip install pandas numpy scipy matplotlib httpx pypdf
export GA_ROOT=<path to the companion study's data>      # read-only
export OPENROUTER_KEY_FILE=<file containing an OpenRouter API key>
python src/build_sample.py        # the claim-list hash must match data/FREEZE.md
python src/run.py --prompt-version v2 --models A F --conditions C0 C2 --subsets FLIP
python src/analyze.py && python src/paper_numbers.py && python src/tables.py
```

Model keys: A = Llama-3.1-8B, B = gpt-oss-20b, F = DeepSeek-V4-Pro, S = Claude Sonnet 5.5. Responses are cached by
hash(model, condition, prompt version, claim), so a rerun is never billed twice. The full study cost about US$21.

## What is not included, and why

- **Claim texts, reference labels and per-claim model outputs.** They come from the companion study's dataset
  and will be released with it. Without them, `build_sample.py` and `run.py` cannot be run end to
  end; every aggregate result is here.
- **The vintage archive itself.** It is public (ALFRED, Federal Reserve Bank of St. Louis); `data/FREEZE.md` lists the
  series and vintages used.
- **API keys and raw request logs.**

The code is the code that produced the results. The only changes are that default local paths were replaced by the
`GA_ROOT` and `OPENROUTER_KEY_FILE` environment variables, and the anonymity checker reads its term list from
`ANON_TERMS`.
