"""Post-review robustness check (DEVIATIONS D7): Llama-3.1-8B rerun on one pinned host (model key AP, runs/cache_pinned).

    python src/pinned_rerun.py

Scores the pinned-host episodes exactly as analyze.py scores the October ones and writes
results/pinned_rerun_metrics.csv and results/pinned_rerun_tests.csv. Nothing in the other results files changes.
"""
import os
import numpy as np, pandas as pd
import analyze as an

RES = an.RES


def main():
    s, ep = an.load("v2")                                   # October episodes (frozen)
    _, pp = an.load("v2", cache_dir="cache_pinned")          # pinned-host rerun
    pp = pp[pp.model_key == "AP"]
    W = {sub: sorted(s[s.subset == sub].doc_id.unique()) for sub in s.subset.unique()}
    W = {sub: (d, an.draws(d)) for sub, d in W.items()}
    rows = []
    for label, df, mk in [("A (October, mixed hosts)", ep, "A"), ("AP (pinned host)", pp, "AP")]:
        for (c, sub), g in df[df.model_key == mk].groupby(["condition", "subset"]):
            docs, w = W[sub]; n_pop = int((s.subset == sub).sum())
            r = {"run": label, "model_key": mk, "condition": c, "subset": sub, "n": len(g), "n_population": n_pop, "missing": n_pop - len(g)}
            for col, name in [("correct", "accuracy"), ("said_contra", "contradicted_rate"), ("said_unver", "unverifiable_rate")]:
                r[name] = g[col].mean(); r[name + "_lo"], r[name + "_hi"] = an.cluster_ci(g, col, docs, w)
            r["no_verdict_rate"] = g.parse_fail.mean()
            r["mean_tool_calls"] = g.n_tool_calls.mean()
            r["malformed_call_share"] = g.tool_calls_malformed.sum() / max(1, g.tool_calls_total.sum())
            r["cost_per_100"] = 100 * g.cost.mean()
            r["providers"] = "|".join(sorted(set("|".join(g.providers).split("|"))))
            rows.append(r)
    pd.DataFrame(rows).sort_values(["subset", "condition", "model_key"]).to_csv(
        os.path.join(RES, "pinned_rerun_metrics.csv"), index=False, lineterminator="\n", float_format="%.6f")

    both = pd.concat([ep, pp]); f = both[both.subset == "FLIP"]; hd = both[both.subset == "HOLDOUT"]
    tests = []

    def add(name, df, mk1, c1, mk2, c2, col):
        x = df[(df.model_key == mk1) & (df.condition == c1)].set_index("claim_id")[col]
        y = df[(df.model_key == mk2) & (df.condition == c2)].set_index("claim_id")[col]
        ids = x.index.intersection(y.index)
        if not len(ids): return
        b, cc, p = an.mcnemar(x[ids].to_numpy(), y[ids].to_numpy())
        row = {"test": name, "metric": col, "first": f"{mk1}-{c1}", "second": f"{mk2}-{c2}", "n_pairs": len(ids),
               "first_rate": x[ids].mean(), "second_rate": y[ids].mean(), "first_only": b, "second_only": cc, "p_exact": p}
        dd = pd.DataFrame({"diff": (x[ids] - y[ids]).to_numpy()}, index=ids)
        dd["doc_id"] = df.drop_duplicates("claim_id").set_index("claim_id").loc[ids, "doc_id"].to_numpy()
        docs = sorted(dd.doc_id.unique()); row["diff"] = dd["diff"].mean()
        row["diff_lo"], row["diff_hi"] = an.cluster_ci(dd, "diff", docs, an.draws(docs))
        tests.append(row)

    add("H1 with pinned host (post-hoc)", f, "AP", "C2", "F", "C0", "correct")
    add("H2 with pinned host (post-hoc)", f, "AP", "C1", "AP", "C0", "said_contra")
    add("H3 with pinned host (post-hoc)", f, "AP", "C2", "AP", "C1", "said_contra")
    add("H4 with pinned host (post-hoc)", hd, "AP", "C2", "AP", "C0", "correct")
    for c in ["C0", "C1", "C2"]:
        add(f"pinned vs October, FLIP accuracy, {c}", f, "AP", c, "A", c, "correct")
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(RES, "pinned_rerun_tests.csv"), index=False, lineterminator="\n", float_format="%.6g")
    # same-verdict share between the two runs
    agree = []
    for c in ["C0", "C1", "C2"]:
        x = pp[pp.condition == c].set_index("claim_id").verdict; y = ep[(ep.model_key == "A") & (ep.condition == c)].set_index("claim_id").verdict
        ids = x.index.intersection(y.index)
        agree.append({"condition": c, "n_pairs": len(ids), "same_verdict": (x[ids].fillna("-") == y[ids].fillna("-")).mean()})
    pd.DataFrame(agree).to_csv(os.path.join(RES, "pinned_rerun_agreement.csv"), index=False, lineterminator="\n", float_format="%.6f")
    print(pd.DataFrame(agree).to_string())
    print(pd.read_csv(os.path.join(RES, "pinned_rerun_metrics.csv"))[["run", "condition", "subset", "n", "missing", "accuracy", "contradicted_rate", "unverifiable_rate", "no_verdict_rate", "malformed_call_share", "providers"]].to_string())
    print(t.to_string())


if __name__ == "__main__":
    main()
