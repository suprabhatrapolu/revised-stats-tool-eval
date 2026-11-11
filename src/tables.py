"""Generate every LaTeX table in the paper from results/*.csv (nothing is typed by hand).

    python src/tables.py
Writes paper/build/table_main.tex, table_tools.tex, table_series.tex, table_cutoff.tex, table_hosts.tex, table_missing.tex
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "paper", "build")
NAME = {"A": "Llama-3.1-8B", "B": "gpt-oss-20b", "F": "DeepSeek-V4-Pro", "S": "Claude Sonnet 5.5"}
COND = {"C0": "C0 no tool", "C1": "C1 latest", "C2": "C2 as-of"}
ORDER = ["A", "B", "F", "S"]


def p(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def ci(r, k):
    return f"{p(r[k])} {{\\scriptsize[{p(r[k + '_lo'])}, {p(r[k + '_hi'])}]}}"


def cell(met, mk, c, sub):
    r = met[(met.model_key == mk) & (met.condition == c) & (met.subset == sub)]
    return None if r.empty else r.iloc[0]


def main_table(met, baseline=""):
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Main results (pilot excluded; \%). FCR, UNV: share of FLIP claims (reference SUPPORTED) labelled CONTRADICTED, UNVERIFIABLE; brackets: 95\% CI. \$/100: US\$ per 100 FLIP claims. --: not run; $^\dagger$: over 5\% of episodes missing. First row: post-hoc baseline (D6); 100 on FLIP and STABLE by definition.}\label{tab:main}",
         r"\adjustbox{max width=\textwidth}{\begin{tabular}{llcccccc}", r"\toprule",
         r"Model & Cond. & FLIP acc. & FLIP FCR & FLIP UNV & STABLE acc. & HOLDOUT acc. & \$/100 \\", r"\midrule",
         r"Majority baseline & always SUPPORTED & 100 & 0 & 0 & 100 & " + baseline + r" & 0 \\", r"\midrule"]
    def v(x, k):
        if x is None: return "--"
        return p(x[k]) + (r"$^\dagger$" if x.missing > 0.05 * x.n_population else "")
    for mk in ORDER:
        if mk not in set(met.model_key): continue
        first = True
        for c in ["C0", "C1", "C2"]:
            f, s, h = cell(met, mk, c, "FLIP"), cell(met, mk, c, "STABLE"), cell(met, mk, c, "HOLDOUT")
            if f is None and s is None and h is None: continue
            L.append(" & ".join([NAME[mk] if first else "", COND[c], v(f, "accuracy"),
                                 (ci(f, "contradicted_rate") if f is not None else "--"), v(f, "unverifiable_rate"),
                                 v(s, "accuracy"), v(h, "accuracy"),
                                 "--" if f is None else f"{f.cost_per_100:.3f}"]) + r" \\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def tools_table(tu, met):
    t = tu[tu.subset.isin(["FLIP", "STABLE"])].copy()
    rows = []
    for (mk, c), g in t.groupby(["model_key", "condition"]):
        calls = g.tool_calls.sum()
        w = g.tool_calls / calls if calls else 0
        rows.append({"mk": mk, "c": c, "calls": calls, "mean": (g.mean_tool_calls * g.episodes).sum() / g.episodes.sum(),
                     "mal": (g.share_calls_malformed * w).sum(), "err": (g.tool_error_rate_per_call * w).sum(),
                     "asof": (g.share_asof_equals_publication * w).sum() if c == "C2" else np.nan})
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Tool-call reliability on FLIP+STABLE (pilot excluded). Malformed: arguments not exactly one JSON object (e.g.\ two objects concatenated). Error: the tool returned an error (bad series, period or date). as\_of: share of C2 calls whose \texttt{as\_of} equals the publication date.}\label{tab:tools}",
         r"\begin{tabular}{llrcccc}", r"\toprule", r"Model & Cond. & calls & calls/claim & malformed \% & error \% & as\_of = pub.\ \% \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: (ORDER.index(r["mk"]), r["c"])):
        L.append(f"{NAME[r['mk']]} & {COND[r['c']]} & {int(r['calls']):,} & {r['mean']:.2f} & {p(r['mal'])} & {p(r['err'])} & {p(r['asof'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def series_table(fs):
    L = [r"\begin{tabular}{llccc}", r"\toprule", r"Model & Cond. & payroll & avg.\ hourly earnings & GDP \\", r"\midrule"]
    for (mk, c), g in fs.groupby(["model_key", "condition"], sort=False):
        vals = {r.series: f"{p(r.false_contradiction_rate)} (n={r.n})" for r in g.itertuples()}
        L.append(f"{NAME[mk]} & {COND[c]} & {vals.get('PAYEMS', '--')} & {vals.get('CES0500000003', '--')} & {vals.get('GDPC1', '--')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def cutoff_table(cs):
    d = cs[(cs.subset == "FLIP")]
    L = [r"\begin{tabular}{llllrcc}", r"\toprule", r"Model & Cond. & cutoff (end) & releases & n & acc.\ \% & FCR \% \\", r"\midrule"]
    for r in d.itertuples():
        side = "on/before cutoff" if r.side == "on_or_before_cutoff" else "after cutoff"
        L.append(f"{NAME[r.model_key]} & {COND[r.condition]} & {r.cutoff_end} & {side} & {r.n} & {p(r.accuracy)} {{\\scriptsize[{p(r.acc_lo)}, {p(r.acc_hi)}]}} & {p(r.contradicted_rate)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def hosts_table(h):
    L = [r"\begin{tabular}{lllrccc}", r"\toprule", r"Cond. & Subset & Host & n & acc.\ \% & contradicted \% & unverifiable \% \\", r"\midrule"]
    for r in h.itertuples():
        if r.condition == "all":
            L.append(r"\midrule" + f"\n\\multicolumn{{7}}{{l}}{{Episodes served by more than one host (excluded above): {int(r.n)}}} \\\\"); continue
        L.append(f"{COND[r.condition]} & {r.subset} & {r.host} & {r.n} & {p(r.accuracy)} & {p(r.contradicted_rate)} & {p(r.unverifiable_rate)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def missing_table(m):
    d = m.copy()
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"Model & Cond. & FLIP & STABLE & HOLDOUT \\", r"\midrule"]
    for (mk, c), g in d.groupby(["model_key", "condition"]):
        v = {r.subset: f"{r.n}/{r.n_population}" for r in g.itertuples()}
        L.append(f"{NAME[mk]} & {COND[c]} & {v.get('FLIP', 'not run')} & {v.get('STABLE', 'not run')} & {v.get('HOLDOUT', 'not run')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def discrim_table(ns):
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Accuracy (\%) on the \HoldoutNonSuppN{} non-SUPPORTED HOLDOUT claims, and share of them answered SUPPORTED, without a tool (C0) and with the as-of tool (C2). Post-hoc (D6). Always answering SUPPORTED scores 0 here.}\label{tab:discrim}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Said SUPPORTED} \\",
         r"Model & C0 & C2 & C0 & C2 \\",
         r"\midrule"]
    for mk in ORDER:
        r0 = ns[(ns.model_key == mk) & (ns.condition == "C0")].iloc[0]
        r = ns[(ns.model_key == mk) & (ns.condition == "C2")].iloc[0]
        L.append(f"{NAME[mk]} & {p(r0.accuracy)} & {p(r.accuracy)} & {p(r0.said_supported)} & {p(r.said_supported)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def error_table(ea):
    L = [r"\begin{tabular}{lrr}", r"\toprule", r"Cause (20 random C2 FLIP failures each) & Llama-3.1-8B & DeepSeek-V4-Pro \\", r"\midrule"]
    piv = ea.pivot_table(index="final_code", columns="model_key", values="n", aggfunc="sum").fillna(0).astype(int)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    for code, r in piv.iterrows():
        L.append(f"{code} & {r.get('A', 0)} & {r.get('F', 0)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def trunc_table(met):
    d = met[met.parse_fail_rate > 0]
    L = [r"\begin{tabular}{lllrrr}", r"\toprule", r"Model & Cond. & Subset & n & no verdict \% (truncated \%) & acc.\ \%: scored / excl.\ truncated \\", r"\midrule"]
    for r in d.sort_values(["model_key", "condition", "subset"]).itertuples():
        L.append(f"{NAME[r.model_key]} & {COND[r.condition]} & {r.subset} & {r.n} & {p(r.parse_fail_rate)} ({p(r.truncated_rate)}) & {p(r.accuracy)} / {p(r.accuracy_excl_truncated)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    met = pd.read_csv(os.path.join(RES, "metrics.csv"))
    # post-hoc majority baseline on the mixed HOLDOUT (D6): share of HOLDOUT labels that are SUPPORTED
    samp = pd.read_csv(os.path.join(HERE, os.pardir, "data", "sample.csv"), dtype=str)
    hold = samp[samp.subset == "HOLDOUT"]
    baseline = f"{100 * (hold.reference == 'SUPPORTED').mean():.1f}"
    w = lambda n, t: open(os.path.join(OUT, n), "w", encoding="utf-8", newline="\n").write(t + "\n")
    w("table_main.tex", main_table(met, baseline))
    tt = tools_table(pd.read_csv(os.path.join(RES, "tool_use.csv")), met)
    w("table_tools.tex", tt)
    w("table_tools_body.tex", tt[tt.index(r"\begin{tabular}"):tt.index(r"\end{tabular}") + len(r"\end{tabular}")])  # appendix copy
    w("table_series.tex", series_table(pd.read_csv(os.path.join(RES, "flip_by_series.csv"))))
    w("table_cutoff.tex", cutoff_table(pd.read_csv(os.path.join(RES, "cutoff_split.csv"))))
    w("table_hosts.tex", hosts_table(pd.read_csv(os.path.join(RES, "a_by_host.csv"))))
    w("table_missing.tex", missing_table(pd.read_csv(os.path.join(RES, "missing.csv"))))
    w("table_errors.tex", error_table(pd.read_csv(os.path.join(RES, "error_analysis_C2_summary.csv"))))
    w("table_discrim.tex", discrim_table(pd.read_csv(os.path.join(RES, "holdout_nonsupported.csv"))))
    w("table_trunc.tex", trunc_table(met))
    print("tables written")


if __name__ == "__main__":
    main()
