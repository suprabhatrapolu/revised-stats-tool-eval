"""Regenerate every number in results/ from data/sample.csv and the cached episodes in runs/cache/.

    python src/analyze.py [--prompt-version v1]

Outputs:
  results/episodes.csv          one row per episode (claim x model x condition)
  results/metrics.csv           model x condition x subset: n, accuracy, false-contradiction rate, ... with 95% CIs
  results/flip_by_series.csv    FLIP false-contradiction rate by series
  results/tests.csv             paired McNemar tests (exact) with Holm correction over the primary family
  results/tool_use.csv          tool calls per episode, as_of compliance (C2), tool errors
  results/holdout_nonsupported.csv  accuracy on HOLDOUT claims whose label is not SUPPORTED (H4)
  results/fig_flip_fcr.pdf/.png the headline figure
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SEED, B = 2027, 2000
SERIES_NAME = {"PAYEMS": "payroll change", "CES0500000003": "avg hourly earnings", "GDPC1": "GDP"}


def malformed(t):
    """A tool call is malformed if its arguments were not exactly one JSON object (e.g. two objects glued together)."""
    raw = t.get("raw_args")
    if t.get("args") is None: return 1
    if isinstance(raw, str):
        txt = raw.strip()
        try:
            _, end = json.JSONDecoder().raw_decode(txt)
            return int(txt[end:].strip() != "")
        except Exception:
            return 1
    return 0


def load(pv, include_pilot=False, cache_dir="cache"):
    s = pd.read_csv(os.path.join(ROOT, "data", "sample.csv"), dtype=str)
    if not include_pilot:
        pilot = set(open(os.path.join(ROOT, "data", "pilot_ids.txt")).read().split())
        s = s[~s.claim_id.isin(pilot)]
    rows = []
    for p in glob.glob(os.path.join(ROOT, "runs", cache_dir, "*.json")):
        try:
            e = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # another lane is still writing this file; the final analysis rerun picks it up
        if e.get("prompt_version") != pv: continue
        tc = e.get("tool_calls") or []
        rows.append({"claim_id": e["claim_id"], "model_key": e["model_key"], "model": e["model"], "condition": e["condition"],
                     "verdict": e.get("verdict"), "parse_mode": e.get("parse_mode"), "cost": e.get("cost", 0.0),
                     "latency": e.get("latency", 0.0), "n_tool_calls": e.get("n_tool_calls", 0),
                     "tool_errors": sum(1 for t in tc if isinstance(t.get("result"), dict) and "error" in t["result"]),
                     "tool_calls_total": len(tc), "tool_calls_malformed": sum(malformed(t) for t in tc),
                     "tool_calls_text_format": sum(1 for t in tc if t.get("mode") == "text"),
                     "tool_asofs": json.dumps([t["args"].get("as_of") if isinstance(t.get("args"), dict) else None for t in tc]),
                     "providers": "|".join(sorted({str(r.get("provider")) for r in e.get("requests", [])})),
                     "hosts_seq": json.dumps([str(r.get("provider")) for r in e.get("requests", [])]),
                     "last_finish": (e.get("requests") or [{}])[-1].get("finish_reason"),
                     "budget_text": int(e.get("verdict") is None and any(k in (e.get("final_text") or "") for k in ("DSML", "tool_call", "invoke name"))),
                     "n_distinct_asof": len({str(t["args"].get("as_of"))[:10] for t in tc if isinstance(t.get("args"), dict) and t["args"].get("as_of")}),
                     "prompt_tokens": e.get("prompt_tokens", 0), "completion_tokens": e.get("completion_tokens", 0)})
    ep = pd.DataFrame(rows)
    ep = ep.merge(s[["claim_id", "subset", "doc_id", "event_id", "series", "as_of", "reference", "type"]], on="claim_id", how="inner")
    ep["release_date"] = ep.event_id.str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    ep["first_host"] = [json.loads(h)[0] if h else None for h in ep.hosts_seq]
    ep["correct"] = (ep.verdict == ep.reference).astype(int)
    ep["said_contra"] = (ep.verdict == "CONTRADICTED").astype(int)
    ep["said_unver"] = (ep.verdict == "UNVERIFIABLE").astype(int)
    ep["parse_fail"] = ep.verdict.isna().astype(int)
    ep["truncated"] = (ep.verdict.isna() & (ep.last_finish == "length")).astype(int)
    ep["is_revision"] = (ep.type == "revision_statement").astype(int)
    return s, ep


def cluster_ci(df, col, docs, rng_w):
    """95% percentile CI of mean(col) resampling documents: rng_w is a (B x n_docs) matrix of draw counts."""
    g = df.groupby("doc_id")[col].agg(["sum", "count"]).reindex(docs).fillna(0)
    num = rng_w @ g["sum"].to_numpy(); den = rng_w @ g["count"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = r[np.isfinite(r)]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def draws(docs):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(docs), size=(B, len(docs)))
    w = np.zeros((B, len(docs)))
    np.add.at(w, (np.repeat(np.arange(B), len(docs)), idx.ravel()), 1)
    return w


def mcnemar(x, y):
    """Exact McNemar on paired 0/1 vectors: b = x1 & y0, c = x0 & y1."""
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


DESIGN = {"A": {"C0": "FSH", "C1": "FSH", "C2": "FSH"}, "F": {"C0": "FSH", "C1": "FSH", "C2": "FSH"},
          "S": {"C0": "FH", "C1": "FH", "C2": "FH"}, "B": {"C0": "FSH", "C1": "FSH", "C2": "FSH"}}
SUBS = {"F": "FLIP", "S": "STABLE", "H": "HOLDOUT"}


def integrity(s):
    """Expected vs cached vs missing episodes per design cell, read from runs/cache only (never from the call logs)."""
    pilot = set(open(os.path.join(ROOT, "data", "pilot_ids.txt")).read().split())
    full = pd.read_csv(os.path.join(ROOT, "data", "sample.csv"), dtype=str)
    have, unreadable = {}, 0
    for p in glob.glob(os.path.join(ROOT, "runs", "cache", "*.json")):
        try: e = json.load(open(p, encoding="utf-8"))
        except Exception: unreadable += 1; continue
        if e.get("prompt_version") != "v2": continue
        have.setdefault((e["model_key"], e["condition"], e["subset"]), set()).add(e["claim_id"])
    rows = []
    for mk, conds in DESIGN.items():
        for c, subs in conds.items():
            for k in subs:
                sub = SUBS[k]; ids = set(full[full.subset == sub].claim_id); main_ids = ids - pilot
                got = have.get((mk, c, sub), set())
                rows.append({"model_key": mk, "condition": c, "subset": sub, "expected_all": len(ids), "cached_all": len(got & ids),
                             "expected_main_excl_pilot": len(main_ids), "cached_main": len(got & main_ids),
                             "missing_main": len(main_ids - got), "status": "not run" if not got else ("complete" if not (main_ids - got) else "incomplete")})
    d = pd.DataFrame(rows); d["unreadable_cache_files_total"] = unreadable
    d.to_csv(os.path.join(RES, "integrity.csv"), index=False, lineterminator="\n")


def main():
    global RES
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt-version", default="v2")
    ap.add_argument("--include-pilot", action="store_true"); a = ap.parse_args()
    if a.include_pilot: RES = os.path.join(RES, "sensitivity_incl_pilot")
    os.makedirs(RES, exist_ok=True)
    s, ep = load(a.prompt_version, a.include_pilot)
    ep.to_csv(os.path.join(RES, "episodes.csv"), index=False, lineterminator="\n")
    W = {sub: (sorted(s[s.subset == sub].doc_id.unique()),) for sub in s.subset.unique()}
    W = {sub: (d[0], draws(d[0])) for sub, d in W.items()}

    # metrics per model x condition x subset; only complete cells are reported as complete
    out = []
    for (mk, m, c, sub), g in ep.groupby(["model_key", "model", "condition", "subset"]):
        docs, w = W[sub]; n_pop = int((s.subset == sub).sum())
        row = {"model_key": mk, "model": m, "condition": c, "subset": sub, "n": len(g), "n_population": n_pop,
               "missing": n_pop - len(g), "complete": len(g) == n_pop}
        for col, name in [("correct", "accuracy"), ("said_contra", "contradicted_rate"), ("said_unver", "unverifiable_rate")]:
            row[name] = g[col].mean(); row[name + "_lo"], row[name + "_hi"] = cluster_ci(g, col, docs, w)
        row["parse_fail_rate"] = g.parse_fail.mean()  # no verdict; scored as wrong (FREEZE), never as UNVERIFIABLE
        row["truncated_rate"] = g.truncated.mean()  # no verdict because max_tokens was hit
        row["tool_budget_text_rate"] = g.budget_text.mean()  # no verdict: tried another tool call after the budget
        nt = g[g.truncated == 0]
        row["accuracy_excl_truncated"] = nt.correct.mean() if len(nt) else np.nan  # sensitivity only
        row["cost_usd"] = g.cost.sum(); row["cost_per_100"] = 100 * g.cost.mean()
        row["median_latency_s"] = g.latency.median(); row["mean_tool_calls"] = g.n_tool_calls.mean()
        row["tool_error_rate"] = (g.tool_errors > 0).mean()
        row["providers"] = "|".join(sorted(set("|".join(g.providers).split("|"))))
        out.append(row)
    met = pd.DataFrame(out).sort_values(["subset", "model_key", "condition"])
    met.to_csv(os.path.join(RES, "metrics.csv"), index=False, lineterminator="\n", float_format="%.6f")

    # FLIP false-contradiction by series
    f = ep[ep.subset == "FLIP"]
    bs = []
    for (mk, c, se), g in f.groupby(["model_key", "condition", "series"]):
        docs = sorted(g.doc_id.unique()); w = draws(docs)
        lo, hi = cluster_ci(g, "said_contra", docs, w)
        bs.append({"model_key": mk, "condition": c, "series": se, "series_name": SERIES_NAME.get(se, se), "n": len(g),
                   "false_contradiction_rate": g.said_contra.mean(), "lo": lo, "hi": hi, "accuracy": g.correct.mean()})
    pd.DataFrame(bs).to_csv(os.path.join(RES, "flip_by_series.csv"), index=False, lineterminator="\n", float_format="%.6f")

    # paired tests on FLIP
    def vec(mk, c, col):
        g = f[(f.model_key == mk) & (f.condition == c)].set_index("claim_id")[col]
        return g
    tests = []
    def add(hyp, family, mk1, c1, mk2, c2, col):
        x, y = vec(mk1, c1, col), vec(mk2, c2, col)
        ids = x.index.intersection(y.index)
        if len(ids) == 0: return
        b, cc, p = mcnemar(x[ids].to_numpy(), y[ids].to_numpy())
        tests.append({"hypothesis": hyp, "family": family, "metric": col, "first": f"{mk1}-{c1}", "second": f"{mk2}-{c2}",
                      "n_pairs": len(ids), "first_rate": x[ids].mean(), "second_rate": y[ids].mean(),
                      "first_only": b, "second_only": cc, "p_exact": p})
    # confirmatory family as frozen in FREEZE.md: H1 (A-C2 vs F-C0), H2 and H3 for A and F (Holm over these 5).
    # A-C2 vs S-C0 was added after the pilot: exploratory (DEVIATIONS D3.2, revised)
    add("H1", "primary", "A", "C2", "F", "C0", "correct")
    add("H1-S (exploratory)", "exploratory", "A", "C2", "S", "C0", "correct")
    for mk in sorted(f.model_key.unique()):
        fam = "primary" if mk in ("A", "F") else "secondary"  # S and B: secondary
        add("H2", fam, mk, "C1", mk, "C0", "said_contra")
        add("H3", fam, mk, "C2", mk, "C1", "said_contra")
    for mk in sorted(f.model_key.unique()):
        add("C2 vs C0", "descriptive", mk, "C2", mk, "C0", "said_contra")
    # H4 on HOLDOUT: C2 vs C0 accuracy, paired
    hd = ep[ep.subset == "HOLDOUT"]
    for mk in sorted(hd.model_key.unique()):
        x = hd[(hd.model_key == mk) & (hd.condition == "C2")].set_index("claim_id")["correct"]
        y = hd[(hd.model_key == mk) & (hd.condition == "C0")].set_index("claim_id")["correct"]
        ids = x.index.intersection(y.index)
        if len(ids):
            b, cc, p = mcnemar(x[ids].to_numpy(), y[ids].to_numpy())
            # paired difference C2 - C0 with a 95% document-bootstrap CI (claims resampled by source document)
            dd = pd.DataFrame({"diff": (x[ids] - y[ids]).to_numpy()}, index=ids)
            dd["doc_id"] = hd.drop_duplicates("claim_id").set_index("claim_id").loc[ids, "doc_id"].to_numpy()
            docs = sorted(dd.doc_id.unique()); lo, hi = cluster_ci(dd, "diff", docs, draws(docs))
            tests.append({"hypothesis": "H4", "family": "secondary", "metric": "correct (HOLDOUT)", "first": f"{mk}-C2",
                          "second": f"{mk}-C0", "n_pairs": len(ids), "first_rate": x[ids].mean(), "second_rate": y[ids].mean(),
                          "first_only": b, "second_only": cc, "p_exact": p, "diff": dd["diff"].mean(), "diff_lo": lo, "diff_hi": hi})
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = np.nan
        prim = t[t.family == "primary"].sort_values("p_exact")
        k = len(prim); running = 0.0
        for i, (ix, r) in enumerate(prim.iterrows()):
            running = max(running, min(1.0, (k - i) * r.p_exact)); t.loc[ix, "p_holm"] = running
    t.to_csv(os.path.join(RES, "tests.csv"), index=False, lineterminator="\n", float_format="%.6g")

    # tool use: as_of compliance in C2
    tu = []
    for (mk, c, sub), g in ep[ep.condition.isin(["C1", "C2"])].groupby(["model_key", "condition", "subset"]):
        calls = matches = 0
        for _, r in g.iterrows():
            for v in json.loads(r.tool_asofs):
                calls += 1; matches += int(str(v).strip()[:10] == str(r.as_of))
        tu.append({"model_key": mk, "condition": c, "subset": sub, "episodes": len(g),
                   "share_with_tool_call": (g.n_tool_calls > 0).mean(), "mean_tool_calls": g.n_tool_calls.mean(),
                   "tool_calls": calls, "share_asof_equals_publication": (matches / calls) if (c == "C2" and calls) else np.nan,
                   "share_episodes_with_tool_error": (g.tool_errors > 0).mean(),
                   "tool_error_rate_per_call": (g.tool_errors.sum() / calls) if calls else np.nan,
                   "share_calls_malformed": (g.tool_calls_malformed.sum() / calls) if calls else np.nan,
                   "share_calls_text_format": (g.tool_calls_text_format.sum() / calls) if calls else np.nan})
    pd.DataFrame(tu).to_csv(os.path.join(RES, "tool_use.csv"), index=False, lineterminator="\n", float_format="%.6f")

    # H4 support: HOLDOUT non-SUPPORTED claims
    h = ep[(ep.subset == "HOLDOUT") & (ep.reference != "SUPPORTED")]
    if len(h):
        h.groupby(["model_key", "condition"]).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                                  said_supported=("verdict", lambda v: (v == "SUPPORTED").mean())) \
            .reset_index().to_csv(os.path.join(RES, "holdout_nonsupported.csv"), index=False, lineterminator="\n", float_format="%.6f")

    # exploratory knowledge-cutoff split (DEVIATIONS D3.9); cutoffs are uncertain, no causal reading: releases on/before vs after each model's cutoff month
    CUTOFF_END = {"A": "2023-12-31", "B": "2024-06-30", "F": "2025-05-31", "S": "2026-06-30"}
    cs = []
    for (mk, c, sub), g in ep.groupby(["model_key", "condition", "subset"]):
        for side, gg in [("on_or_before_cutoff", g[g.release_date <= CUTOFF_END[mk]]), ("after_cutoff", g[g.release_date > CUTOFF_END[mk]])]:
            if gg.empty: continue
            docs = sorted(gg.doc_id.unique()); w = draws(docs)
            lo, hi = cluster_ci(gg, "correct", docs, w); flo, fhi = cluster_ci(gg, "said_contra", docs, w)
            cs.append({"model_key": mk, "condition": c, "subset": sub, "cutoff_end": CUTOFF_END[mk], "side": side,
                       "releases": ",".join(sorted(gg.release_date.unique())), "n": len(gg),
                       "accuracy": gg.correct.mean(), "acc_lo": lo, "acc_hi": hi,
                       "contradicted_rate": gg.said_contra.mean(), "fcr_lo": flo, "fcr_hi": fhi})
    pd.DataFrame(cs).to_csv(os.path.join(RES, "cutoff_split.csv"), index=False, lineterminator="\n", float_format="%.6f")
    # H1 restricted to releases after S's cutoff (the clean test) and on/before it
    for side, cond in [("after S cutoff", f.release_date > CUTOFF_END["S"]), ("on/before S cutoff", f.release_date <= CUTOFF_END["S"])]:
        ff = f[cond]
        x = ff[(ff.model_key == "A") & (ff.condition == "C2")].set_index("claim_id")["correct"]
        y = ff[(ff.model_key == "S") & (ff.condition == "C0")].set_index("claim_id")["correct"]
        ids = x.index.intersection(y.index)
        if len(ids):
            b, cc, p = mcnemar(x[ids].to_numpy(), y[ids].to_numpy())
            t = pd.concat([t, pd.DataFrame([{"hypothesis": f"H1-S ({side}, exploratory)", "family": "exploratory", "metric": "correct",
                          "first": "A-C2", "second": "S-C0", "n_pairs": len(ids), "first_rate": x[ids].mean(),
                          "second_rate": y[ids].mean(), "first_only": b, "second_only": cc, "p_exact": p}])], ignore_index=True)
    t.to_csv(os.path.join(RES, "tests.csv"), index=False, lineterminator="\n", float_format="%.6g")

    # A by serving host (D3.10): episodes whose every request went to one host; hosts with n >= 30 only
    a_ = ep[(ep.model_key == "A")].copy()
    a_["single_host"] = [len(set(json.loads(h))) == 1 for h in a_.hosts_seq]
    hs = []
    for (c, sub, host), g in a_[a_.single_host].groupby(["condition", "subset", "first_host"]):
        if len(g) < 30: continue
        hs.append({"condition": c, "subset": sub, "host": host, "n": len(g), "accuracy": g.correct.mean(),
                   "contradicted_rate": g.said_contra.mean(), "unverifiable_rate": g.said_unver.mean()})
    hs.append({"condition": "all", "subset": "all", "host": "mixed-host episodes (excluded above)", "n": int((~a_.single_host).sum()),
               "accuracy": np.nan, "contradicted_rate": np.nan, "unverifiable_rate": np.nan})
    pd.DataFrame(hs).to_csv(os.path.join(RES, "a_by_host.csv"), index=False, lineterminator="\n", float_format="%.6f")

    integrity(s)
    ct = []
    for (mk, c), g in ep[ep.subset == "FLIP"].groupby(["model_key", "condition"]):
        for lab, gg in [("revision claims", g[g.is_revision == 1]), ("all other claims", g[g.is_revision == 0])]:
            if gg.empty: continue
            docs = sorted(gg.doc_id.unique()); w = draws(docs); lo, hi = cluster_ci(gg, "correct", docs, w)
            r = {"model_key": mk, "condition": c, "claims": lab, "n": len(gg), "accuracy": gg.correct.mean(), "acc_lo": lo,
                 "acc_hi": hi, "contradicted_rate": gg.said_contra.mean(), "unverifiable_rate": gg.said_unver.mean(),
                 "no_verdict_rate": gg.parse_fail.mean()}
            if c == "C2":
                multi = gg[gg.n_distinct_asof >= 2]; single = gg[gg.n_distinct_asof < 2]
                r.update({"share_querying_2plus_vintages": (gg.n_distinct_asof >= 2).mean(),
                          "accuracy_if_2plus_vintages": multi.correct.mean() if len(multi) else np.nan,
                          "accuracy_if_1_vintage": single.correct.mean() if len(single) else np.nan})
            ct.append(r)
    pd.DataFrame(ct).to_csv(os.path.join(RES, "flip_by_claimtype.csv"), index=False, lineterminator="\n", float_format="%.6f")
    met[["model_key", "model", "condition", "subset", "n_population", "n", "missing"]].to_csv(
        os.path.join(RES, "missing.csv"), index=False, lineterminator="\n")
    figure(met)
    print(met[["model_key", "condition", "subset", "n", "complete", "accuracy", "contradicted_rate", "unverifiable_rate",
               "parse_fail_rate", "cost_per_100", "mean_tool_calls"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(t): print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def figure(met):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = met[met.subset == "FLIP"]
    if d.empty: return
    conds = [c for c in ["C0", "C1", "C2"] if c in set(d.condition)]
    models = [m for m in ["A", "B", "F", "S"] if m in set(d.model_key)]
    names = {"A": "Llama-3.1-8B", "B": "gpt-oss-20b", "F": "DeepSeek-V4-Pro", "S": "Claude Sonnet 5.5"}
    cname = {"C0": "no tool", "C1": "latest-value tool", "C2": "as-of tool"}
    colors = {"C0": "#9aa0a6", "C1": "#d95f02", "C2": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    wdt = 0.8 / len(conds)
    for j, c in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(models):
            r = d[(d.model_key == m) & (d.condition == c)]
            if r.empty: continue
            r = r.iloc[0]; xs.append(i + (j - (len(conds) - 1) / 2) * wdt)
            ys.append(100 * r.contradicted_rate); lo.append(100 * (r.contradicted_rate - r.contradicted_rate_lo)); hi.append(100 * (r.contradicted_rate_hi - r.contradicted_rate))
        ax.bar(xs, ys, wdt * 0.95, yerr=[lo, hi], capsize=2, color=colors[c], label=cname[c], error_kw={"lw": 0.8})
    ax.set_xticks(range(len(models))); ax.set_xticklabels([names[m] for m in models], fontsize=8)
    ax.set_ylabel("FLIP claims labelled\nCONTRADICTED (%)", fontsize=8); ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False, ncol=len(conds), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig_flip_fcr.{ext}"), dpi=200, metadata={"Author": "", "Creator": "", "Producer": ""} if ext == "pdf" else None)


if __name__ == "__main__":
    main()
