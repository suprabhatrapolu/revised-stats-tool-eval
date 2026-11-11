"""Write paper/build/numbers.tex (LaTeX macros) and paper/NUMBERS.md (each macro -> value -> source file and row).

Every number in the paper's prose is a macro defined here, read from results/ or from a named file in the audit repo.
Tables are generated separately by src/tables.py from the same results files.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
GA = os.environ["GA_ROOT"]  # path to the companion audit's data (released with that study)
RES = os.path.join(ROOT, "results")
macros = []  # (name, value, source)


def add(name, value, source):
    assert name.isalpha(), name
    macros.append((name, value, source))


def pct(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def pval(p):
    """p-value with its relation, for use as $p<macro>$ in the text."""
    if pd.isna(p): return "=\\text{--}"
    if p < 1e-4: return "<10^{-4}"
    return f"={p:.4f}" if p < 0.01 else f"={p:.3f}"


W = {"A": "LA", "B": "GO", "F": "DS", "S": "SO", "C0": "Zero", "C1": "One", "C2": "Two"}


def main():
    s = pd.read_csv(os.path.join(ROOT, "data", "sample.csv"), dtype=str)
    pilot = set(open(os.path.join(ROOT, "data", "pilot_ids.txt")).read().split())
    add("SampleN", f"{len(s):,}", "data/sample.csv: row count")
    for sub, nm in [("FLIP", "FlipN"), ("STABLE", "StableN"), ("HOLDOUT", "HoldoutN")]:
        add(nm, f"{(s.subset == sub).sum():,}", f"data/sample.csv: subset=={sub}")
        add(nm + "Main", f"{((s.subset == sub) & ~s.claim_id.isin(pilot)).sum():,}", f"data/sample.csv: subset=={sub}, minus data/pilot_ids.txt")
    add("PilotN", f"{len(pilot)}", "data/pilot_ids.txt")
    add("SampleNExPilot", f"{(~s.claim_id.isin(pilot)).sum():,}", "data/sample.csv: all subsets minus data/pilot_ids.txt")
    add("HoldoutNonSuppN", f"{((s.subset == 'HOLDOUT') & (s.reference != 'SUPPORTED')).sum()}", "data/sample.csv: HOLDOUT & reference!=SUPPORTED")
    # post-hoc (D6): per-population accuracy definitions and the majority baseline on the mixed HOLDOUT
    add("HoldoutSuppN", f"{int(((s.subset == 'HOLDOUT') & (s.reference == 'SUPPORTED')).sum()):,}", "data/sample.csv: HOLDOUT & reference==SUPPORTED")
    add("HoldoutNonSuppUnvN", f"{int(((s.subset == 'HOLDOUT') & (s.reference == 'UNVERIFIABLE')).sum())}", "data/sample.csv: HOLDOUT & reference==UNVERIFIABLE")
    add("HoldoutNonSuppConN", f"{int(((s.subset == 'HOLDOUT') & (s.reference == 'CONTRADICTED')).sum())}", "data/sample.csv: HOLDOUT & reference==CONTRADICTED")
    nh = int((s.subset == "HOLDOUT").sum())
    hsup = int(((s.subset == "HOLDOUT") & (s.reference == "SUPPORTED")).sum())
    add("HoldoutBaseline", f"{100 * hsup / nh:.1f}", "data/sample.csv: share of HOLDOUT labels SUPPORTED (always-SUPPORTED majority baseline)")
    # post-hoc (D6): per-class accuracy on the non-SUPPORTED HOLDOUT claims (discrimination), from the saved analysis
    ns = pd.read_csv(os.path.join(RES, "holdout_nonsupported.csv"))
    for mk in ["A", "B", "F", "S"]:
        c2 = ns[(ns.model_key == mk) & (ns.condition == "C2")].iloc[0]
        add(f"NsAcc{W[mk]}", pct(c2.accuracy), f"results/holdout_nonsupported.csv: {mk}, C2, non-SUPPORTED HOLDOUT, accuracy")
    s0 = ns[(ns.model_key == "S") & (ns.condition == "C0")].iloc[0]
    add("NsZeroSO", pct(s0.accuracy), "results/holdout_nonsupported.csv: S, C0, non-SUPPORTED HOLDOUT, accuracy")
    # post-hoc (D7): Llama-3.1-8B rerun on one pinned host
    pm_path = os.path.join(RES, "pinned_rerun_metrics.csv")
    if os.path.exists(pm_path):
        pm = pd.read_csv(pm_path); pt = pd.read_csv(os.path.join(RES, "pinned_rerun_tests.csv"))
        ap = pm[pm.model_key == "AP"]
        for c in ["C0", "C1", "C2"]:
            for sub, sn in [("FLIP", "Flip"), ("STABLE", "Stable"), ("HOLDOUT", "Holdout")]:
                r = ap[(ap.condition == c) & (ap.subset == sub)].iloc[0]
                src = f"results/pinned_rerun_metrics.csv: AP, {c}, {sub}"
                add(f"PinAcc{W[c]}{sn}", pct(r.accuracy), src + ", accuracy")
                if sub == "FLIP": add(f"PinFcr{W[c]}Flip", pct(r.contradicted_rate), src + ", contradicted_rate")
        ag = pd.read_csv(os.path.join(RES, "pinned_rerun_agreement.csv")).set_index("condition")
        for c in ["C0", "C1", "C2"]:
            add(f"PinSame{W[c]}", pct(ag.loc[c, "same_verdict"]), f"results/pinned_rerun_agreement.csv: {c}, same_verdict")
        # billed amount for the whole rerun, read from the API key's usage on 2026-11-12; it includes the stopped
        # first-host run and the requests of episodes that failed before the D7.5 fix, which the episode files do not
        add("PinCost", "5.24", "OpenRouter usage of the rerun's API key on 2026-11-12 (DEVIATIONS D7.6)")
        add("PinMissing", f"{int(ap.missing.sum())}", "results/pinned_rerun_metrics.csv: AP, sum of missing")
        add("PinEpisodes", f"{int(ap.n.sum()):,}", "results/pinned_rerun_metrics.csv: AP, sum of n")
        for key, nm in [("H1", "PinHOne"), ("H2", "PinHTwo"), ("H3", "PinHThree"), ("H4", "PinHFour")]:
            r = pt[pt.test.str.startswith(key + " ")].iloc[0]; src = f"results/pinned_rerun_tests.csv: {r.test}"
            add(nm + "First", pct(r.first_rate), src + ", first_rate"); add(nm + "Second", pct(r.second_rate), src + ", second_rate")
            add(nm + "N", f"{int(r.n_pairs)}", src + ", n_pairs"); add(nm + "P", pval(r.p_exact), src + ", p_exact (unadjusted)")
            add(nm + "Diff", f"{100 * r['diff']:+.1f}", src + ", diff"); add(nm + "DiffCI", f"{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}", src + ", diff_lo, diff_hi")
    fm = s[(s.subset == "FLIP") & ~s.claim_id.isin(pilot)]
    add("FlipRevN", f"{(fm.type == 'revision_statement').sum()}", "data/sample.csv: FLIP main, type==revision_statement")

    fr = pd.read_csv(os.path.join(GA, "analysis", "tables", "an2_vintage_flip_rates.csv"))
    a = fr[fr.stratum == "all"].iloc[0]
    add("AuditFlipN", f"{int(a.contradicted_by_the_latest_vintage)}", "audit analysis/tables/an2_vintage_flip_rates.csv: stratum=all")
    add("AuditSuppN", f"{int(a.supported_on_the_claim_date):,}", "audit an2_vintage_flip_rates.csv: stratum=all")
    add("AuditFlipPct", pct(a.flip_rate), "audit an2_vintage_flip_rates.csv: stratum=all, flip_rate")
    add("AuditFlipPayrollPct", pct(fr[fr.stratum == "series PAYEMS"].iloc[0].flip_rate), "audit an2_vintage_flip_rates.csv: series PAYEMS")

    # audit validation constants: each is asserted to appear verbatim in the audit file it is quoted from
    hv = open(os.path.join(GA, "analysis", "heldout_validation.md"), encoding="utf-8").read()
    for frag in ["(142/153; 0.876–0.959)", "**0.167** (3/18", "| documents / claims / checkable | 20 / 230 / 150 | 20 / 233 / 141 |"]:
        assert frag in hv, frag
    add("ValAgreeK", "142", "audit analysis/heldout_validation.md: agreement on the supported verdict 142/153")
    add("ValAgreeN", "153", "audit analysis/heldout_validation.md: 142/153")
    add("ValAgree", "0.928", "audit analysis/heldout_validation.md: 0.928 pooled")
    add("ValAgreeCI", "0.876--0.959", "audit analysis/heldout_validation.md: (142/153; 0.876-0.959)")
    add("ValDisagreePct", f"{100 * 11 / 153:.1f}", "computed: (153-142)/153 from audit analysis/heldout_validation.md")
    add("ValContraPrec", "0.167", "audit analysis/heldout_validation.md: precision of the contradiction verdict 0.167 (3/18)")
    av = pd.read_csv(os.path.join(GA, "data", "processed", "all_verdicts.csv"), dtype=str)
    add("AuditClaimsN", f"{len(av):,}", "audit data/processed/all_verdicts.csv: row count")
    add("AuditDocsN", f"{av.doc_id.nunique()}", "audit data/processed/all_verdicts.csv: distinct doc_id")
    add("AuditReleasesN", f"{av.event_id.nunique()}", "audit data/processed/all_verdicts.csv: distinct event_id")
    met = pd.read_csv(os.path.join(RES, "metrics.csv"))
    for _, r in met.iterrows():
        key = f"{W[r.model_key]}{W[r.condition]}{r.subset.capitalize()}"
        src = f"results/metrics.csv: model_key={r.model_key}, condition={r.condition}, subset={r.subset}"
        add(f"Acc{key}", pct(r.accuracy), src + ", accuracy")
        add(f"AccCI{key}", f"{pct(r.accuracy_lo)}--{pct(r.accuracy_hi)}", src + ", accuracy_lo/hi")
        add(f"Fcr{key}", pct(r.contradicted_rate), src + ", contradicted_rate")
        add(f"FcrCI{key}", f"{pct(r.contradicted_rate_lo)}--{pct(r.contradicted_rate_hi)}", src + ", contradicted_rate_lo/hi")
        add(f"Unv{key}", pct(r.unverifiable_rate), src + ", unverifiable_rate")
        add(f"Sup{key}", pct(1 - r.contradicted_rate - r.unverifiable_rate - r.parse_fail_rate), src + ", 1 - contradicted - unverifiable - no-verdict")
        add(f"Trunc{key}", pct(r.truncated_rate), src + ", truncated_rate")
        add(f"Cost{key}", f"{r.cost_per_100:.3f}", src + ", cost_per_100")
        add(f"N{key}", f"{int(r.n):,}", src + ", n")

    t = pd.read_csv(os.path.join(RES, "tests.csv"))
    names = {("H1", "A-C2", "F-C0"): "HOne", ("H1-S (exploratory)", "A-C2", "S-C0"): "HOneS",
             ("H2", "A-C1", "A-C0"): "HTwoLA", ("H2", "F-C1", "F-C0"): "HTwoDS", ("H2", "S-C1", "S-C0"): "HTwoSO",
             ("H3", "A-C2", "A-C1"): "HThreeLA", ("H3", "F-C2", "F-C1"): "HThreeDS", ("H3", "S-C2", "S-C1"): "HThreeSO",
             ("H4", "A-C2", "A-C0"): "HFourLA", ("H4", "F-C2", "F-C0"): "HFourDS", ("H4", "S-C2", "S-C0"): "HFourSO", ("H4", "B-C2", "B-C0"): "HFourGO",
             ("H2", "B-C1", "B-C0"): "HTwoGO", ("H3", "B-C2", "B-C1"): "HThreeGO",
             ("H1-S (after S cutoff, exploratory)", "A-C2", "S-C0"): "HOneSafter",
             ("H1-S (on/before S cutoff, exploratory)", "A-C2", "S-C0"): "HOneSbefore"}
    for _, r in t.iterrows():
        nm = names.get((r.hypothesis, r["first"], r["second"]))
        if not nm: continue
        src = f"results/tests.csv: hypothesis={r.hypothesis}, {r['first']} vs {r['second']}"
        add(nm + "N", f"{int(r.n_pairs)}", src + ", n_pairs")
        add(nm + "First", pct(r.first_rate), src + ", first_rate")
        add(nm + "Second", pct(r.second_rate), src + ", second_rate")
        add(nm + "P", pval(r.p_exact), src + ", p_exact")
        add(nm + "PHolm", pval(r.p_holm), src + ", p_holm")
        add(nm + "B", f"{int(r.first_only)}", src + ", first_only"); add(nm + "C", f"{int(r.second_only)}", src + ", second_only")
        if "diff" in r and not pd.isna(r.get("diff")):
            add(nm + "Diff", f"{100 * r['diff']:+.1f}", src + ", diff (first - second), points")
            add(nm + "DiffCI", f"{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}", src + ", diff_lo/diff_hi (document bootstrap)")

    ts = pd.read_csv(os.path.join(RES, "sensitivity_incl_pilot", "tests.csv"))
    r = ts[ts.hypothesis == "H1"].iloc[0]; src = "results/sensitivity_incl_pilot/tests.csv: H1"
    add("SensHOneFirst", pct(r.first_rate), src + ", first_rate"); add("SensHOneSecond", pct(r.second_rate), src + ", second_rate")
    add("SensHOneN", f"{int(r.n_pairs)}", src + ", n_pairs"); add("SensHOneP", pval(r.p_exact), src + ", p_exact")

    ct = pd.read_csv(os.path.join(RES, "flip_by_claimtype.csv"))
    for _, r in ct.iterrows():
        nm = f"Ct{W[r.model_key]}{W[r.condition]}{'Rev' if r.claims.startswith('revision') else 'Oth'}"
        src = f"results/flip_by_claimtype.csv: {r.model_key}, {r.condition}, {r.claims}"
        add(nm + "Acc", pct(r.accuracy), src + ", accuracy")
        if r.condition == "C2" and not pd.isna(r.get("share_querying_2plus_vintages")):
            add(nm + "Multi", pct(r.share_querying_2plus_vintages), src + ", share_querying_2plus_vintages")

    cs = pd.read_csv(os.path.join(RES, "cutoff_split.csv"))
    for _, r in cs[(cs.model_key == "S") & (cs.subset == "FLIP")].iterrows():
        nm = f"Cut{W[r.condition]}{'Before' if r.side.startswith('on') else 'After'}"
        src = f"results/cutoff_split.csv: S, {r.condition}, FLIP, {r.side}"
        add(nm + "Acc", pct(r.accuracy), src + ", accuracy"); add(nm + "N", f"{int(r.n)}", src + ", n")
        add(nm + "Fcr", pct(r.contradicted_rate), src + ", contradicted_rate")
        add(nm + "CI", f"{pct(r.acc_lo)}--{pct(r.acc_hi)}", src + ", acc_lo/hi")

    tu = pd.read_csv(os.path.join(RES, "tool_use.csv"))
    for mk in ["A", "B", "F", "S"]:
        for c in ["C1", "C2"]:
            g = tu[(tu.model_key == mk) & (tu.condition == c) & tu.subset.isin(["FLIP", "STABLE"])]
            if g.empty: continue
            calls = g.tool_calls.sum(); w = g.tool_calls / calls
            src = f"results/tool_use.csv: {mk}, {c}, FLIP+STABLE, call-weighted"
            add(f"Mal{W[mk]}{W[c]}", pct((g.share_calls_malformed * w).sum()), src + ", share_calls_malformed")
            add(f"Err{W[mk]}{W[c]}", pct((g.tool_error_rate_per_call * w).sum()), src + ", tool_error_rate_per_call")
            if c == "C2": add(f"Asof{W[mk]}", pct((g.share_asof_equals_publication * w).sum()), src + ", share_asof_equals_publication")

    ea = pd.read_csv(os.path.join(RES, "error_analysis_C2_summary.csv"))
    for mk in ["A", "F"]:
        g = ea[ea.model_key == mk]; src = f"results/error_analysis_C2_summary.csv: {mk}"
        add(f"EaRev{W[mk]}", f"{int(g[g.final_code.str.startswith('revision')].n.sum())}", src + ", revision claim: earlier print not consulted")
        add(f"EaAbst{W[mk]}", f"{int(g[g.final_code.str.startswith('abstained')].n.sum())}", src + ", abstained")
        add(f"EaArith{W[mk]}", f"{int(g[g.final_code.str.startswith('aggregation')].n.sum())}", src + ", aggregation/arithmetic/units")
        add(f"EaBudget{W[mk]}", f"{int(g[g.final_code.str.contains('3-call budget')].n.sum())}", src + ", tool-call budget")
        add(f"EaPrec{W[mk]}", f"{int(g[g.final_code.str.startswith('precision')].n.sum())}", src + ", precision over-strict")
    ln = pd.read_csv(os.path.join(RES, "label_noise.csv"))
    add("LabelPlausible", f"{int(ln.plausible_label_errors.sum())}", "results/label_noise.csv: sum plausible_label_errors")
    add("LabelAmbiguous", f"{int(ln.ambiguous.sum())}", "results/label_noise.csv: sum ambiguous")
    add("LabelDepLA", f"{ln[ln.model_key == 'A'].iloc[0].est_accuracy_depression_pp_hi:.1f}", "results/label_noise.csv: A, est_accuracy_depression_pp_hi")
    add("LabelDepDS", f"{ln[ln.model_key == 'F'].iloc[0].est_accuracy_depression_pp_hi:.1f}", "results/label_noise.csv: F, est_accuracy_depression_pp_hi")
    add("LabelDepDSlo", f"{ln[ln.model_key == 'F'].iloc[0].est_accuracy_depression_pp_lo:.1f}", "results/label_noise.csv: F, est_accuracy_depression_pp_lo")

    # concatenated tool-call arguments by serving host (A, C1+C2, pilot excluded), from the episode cache
    import glob as _g, json as _j
    pil = set(open(os.path.join(ROOT, "data", "pilot_ids.txt")).read().split())
    dup = {"CoreWeave": [0, 0], "Groq": [0, 0]}; ident = [0, 0]
    for pth in _g.glob(os.path.join(ROOT, "runs", "cache", "*.json")):
        try: e = _j.load(open(pth, encoding="utf-8"))
        except Exception: continue
        # FLIP + STABLE only, to match the tool-reliability table it annotates
        if e.get("model_key") != "A" or e.get("condition") not in ("C1", "C2") or e.get("claim_id") in pil \
                or e.get("subset") not in ("FLIP", "STABLE"): continue
        hs = {r.get("provider") for r in e.get("requests", [])}
        for t in e.get("tool_calls", []):
            raw = (t.get("raw_args") or "").strip() if isinstance(t.get("raw_args"), str) else ""
            try:
                o, end = _j.JSONDecoder().raw_decode(raw); bad = raw[end:].strip() != ""
                if bad:
                    ident[1] += 1
                    try: ident[0] += int(_j.JSONDecoder().raw_decode(raw[end:].strip())[0] == o)
                    except Exception: pass
            except Exception: bad = False
            if len(hs) == 1 and next(iter(hs)) in dup:
                dup[next(iter(hs))][0] += int(bad); dup[next(iter(hs))][1] += 1
    add("DupCWk", f"{dup['CoreWeave'][0]:,}", "runs/cache: A C1+C2 calls in CoreWeave-only episodes with concatenated arguments")
    add("DupCWn", f"{dup['CoreWeave'][1]:,}", "runs/cache: A C1+C2 calls in CoreWeave-only episodes")
    add("DupGQk", f"{dup['Groq'][0]:,}", "runs/cache: A C1+C2 calls in Groq-only episodes with concatenated arguments")
    add("DupGQn", f"{dup['Groq'][1]:,}", "runs/cache: A C1+C2 calls in Groq-only episodes")
    add("DupIdentK", f"{ident[0]:,}", "runs/cache: concatenated A calls whose two objects are identical")
    add("DupAllN", f"{ident[1]:,}", "runs/cache: all concatenated A calls (C1+C2, pilot excluded)")

    ig = pd.read_csv(os.path.join(RES, "integrity.csv"))
    ran = ig[ig.status != "not run"]
    add("MissingTotal", f"{int(ran.missing_main.sum())}", "results/integrity.csv: sum missing_main over cells that ran")
    add("CellsNotRun", f"{int((ig.status == 'not run').sum())}", "results/integrity.csv: count status==not run")

    cost = 0.0
    import glob, json
    for d in ("cache", "superseded_cache"):
        for p in glob.glob(os.path.join(ROOT, "runs", d, "*.json")):
            try: cost += json.load(open(p, encoding="utf-8")).get("cost", 0.0) or 0.0
            except Exception: pass
    add("TotalCost", f"{cost:.2f}", "sum of cost over runs/cache and runs/superseded_cache episode files")

    with open(os.path.join(ROOT, "paper", "build", "numbers.tex"), "w", encoding="utf-8", newline="\n") as f:
        for n, v, _ in macros:
            f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    with open(os.path.join(ROOT, "paper", "NUMBERS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Every number in the paper, with its source\n\nGenerated by `src/paper_numbers.py` (prose macros) and "
                "`src/tables.py` (tables, read directly from `results/metrics.csv`, `results/tool_use.csv`, "
                "`results/flip_by_series.csv`, `results/cutoff_split.csv`, `results/a_by_host.csv`, `results/missing.csv`).\n\n"
                "| macro | value | source |\n|---|---|---|\n")
        for n, v, src in macros:
            f.write(f"| `\\{n}` | {v} | {src} |\n")
    print(f"{len(macros)} macros written")


if __name__ == "__main__":
    main()
