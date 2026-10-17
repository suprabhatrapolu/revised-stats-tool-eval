"""C2 error analysis (FREEZE: 20 random C2 failures on FLIP, seed 2027; extended to A and F, 20 each).

Mechanical tags are computed here; the remaining failures are coded by reading the transcript, recorded in
results/error_analysis_manual.csv (claim_id, model_key, manual_code, note), and merged by this script.

    python src/error_analysis.py            # writes results/error_analysis_C2.csv (+ summary)
"""
import glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SEED, N = 2027, 20


def norm_period(p):
    p = str(p or "").strip().upper().replace("M", "-") if "Q" not in str(p).upper() else str(p).strip().upper().replace("-", "").replace(" ", "")
    return p


def main():
    s = pd.read_csv(os.path.join(ROOT, "data", "sample.csv"), dtype=str).set_index("claim_id")
    pilot = set(open(os.path.join(ROOT, "data", "pilot_ids.txt")).read().split())
    eps = {}
    for p in glob.glob(os.path.join(ROOT, "runs", "cache", "*.json")):
        try: e = json.load(open(p, encoding="utf-8"))
        except Exception: continue
        if e.get("prompt_version") == "v2" and e["condition"] == "C2" and e["model_key"] in ("A", "F") and e["subset"] == "FLIP" \
                and e["claim_id"] not in pilot:
            eps[(e["model_key"], e["claim_id"])] = e
    rows = []
    for mk in ("A", "F"):
        fails = sorted(cid for (m, cid), e in eps.items() if m == mk and e.get("verdict") != "SUPPORTED")
        pick = pd.Series(fails).sample(n=min(N, len(fails)), random_state=SEED).tolist()
        for cid in pick:
            e = eps[(mk, cid)]; c = s.loc[cid]; tc = e.get("tool_calls") or []
            args = [t.get("args") if isinstance(t.get("args"), dict) else {} for t in tc]
            asofs = [str(a.get("as_of", ""))[:10] for a in args]
            hit = [a for a in args if str(a.get("series", "")).upper() == c.series]
            last_fr = (e.get("requests") or [{}])[-1].get("finish_reason")
            ft = e.get("final_text") or ""
            if e.get("verdict") is None and ("DSML" in ft or "tool_call" in ft or "invoke name" in ft):
                tag = "no verdict: tried another tool call after the 3-call budget"
            elif e.get("verdict") is None and last_fr == "length": tag = "no verdict: truncated at max_tokens"
            elif e.get("verdict") is None: tag = "no verdict in answer"
            elif e["verdict"] == "UNVERIFIABLE": tag = "abstained (UNVERIFIABLE)"
            elif not tc: tag = "no tool call"
            elif any("error" in (t.get("result") or {}) for t in tc) and not any("rows" in (t.get("result") or {}) for t in tc): tag = "tool error only"
            elif not hit: tag = "wrong series"
            elif c.as_of not in asofs: tag = "wrong as_of (never the publication date)"
            else: tag = "needs reading"
            rows.append({"model_key": mk, "claim_id": cid, "series": c.series, "ref": c.ref, "kind": c.kind, "horizon": c.horizon,
                         "type": c.type, "as_of": c.as_of, "claim_text": c.claim_text, "verdict": e.get("verdict"),
                         "auto_tag": tag, "n_tool_calls": len(tc), "tool_args": json.dumps(args),
                         "tool_rows": json.dumps([(t.get("result") or {}).get("rows") or (t.get("result") or {}).get("error") for t in tc])[:1500],
                         "vintages_used": json.dumps([(t.get("result") or {}).get("vintage_date") for t in tc]),
                         "final_text": (e.get("final_text") or "")[-700:]})
    d = pd.DataFrame(rows)
    man_p = os.path.join(RES, "error_analysis_manual.csv")
    if os.path.exists(man_p):
        man = pd.read_csv(man_p, dtype=str)
        d = d.merge(man[[c for c in ["model_key", "claim_id", "manual_code", "note", "label_flag"] if c in man.columns]], on=["model_key", "claim_id"], how="left")
        d["final_code"] = d.manual_code.where(d.auto_tag == "needs reading", d.auto_tag)
    else:
        d["final_code"] = d.auto_tag
    d.to_csv(os.path.join(RES, "error_analysis_C2.csv"), index=False, lineterminator="\n")
    # label flags (from results/error_analysis_manual.csv, column label_flag: plausible / ambiguous / no)
    if "label_flag" in d.columns:
        d["label_flag"] = d.label_flag.fillna("no")
        nfail = {mk: sum(1 for (m, c), e in eps.items() if m == mk and e.get("verdict") != "SUPPORTED") for mk in ("A", "F")}
        nflip = {mk: sum(1 for (m, c), e in eps.items() if m == mk) for mk in ("A", "F")}
        ln = []
        for mk, g in d.groupby("model_key"):
            k_pl = int((g.label_flag == "plausible").sum()); k_amb = int((g.label_flag == "ambiguous").sum()); n = len(g)
            lo = nfail[mk] * k_pl / n; hi = nfail[mk] * (k_pl + k_amb) / n
            ln.append({"model_key": mk, "sampled_failures": n, "plausible_label_errors": k_pl, "ambiguous": k_amb,
                       "c2_failures_total": nfail[mk], "flip_claims": nflip[mk],
                       "est_label_errors_among_failures_lo": round(lo, 1), "est_label_errors_among_failures_hi": round(hi, 1),
                       "est_accuracy_depression_pp_lo": round(100 * lo / nflip[mk], 2), "est_accuracy_depression_pp_hi": round(100 * hi / nflip[mk], 2),
                       "note": "estimates how much wrong labels lower measured C2 accuracy; not a bound on total label noise "
                               "(a wrong label the model also accepts is invisible here). Overall label-noise reference: audit "
                               "blind validation, 11/153 = 7.2% disagreement on supported verdicts."})
        pd.DataFrame(ln).to_csv(os.path.join(RES, "label_noise.csv"), index=False, lineterminator="\n")
        print(pd.DataFrame(ln).drop(columns="note").to_string(index=False))
    summ = d.groupby(["model_key", "final_code"]).size().rename("n").reset_index()
    summ.to_csv(os.path.join(RES, "error_analysis_C2_summary.csv"), index=False, lineterminator="\n")
    print(summ.to_string(index=False))
    print("failures available:", {mk: sum(1 for (m, c), e in eps.items() if m == mk and e.get('verdict') != 'SUPPORTED') for mk in ('A', 'F')})


if __name__ == "__main__":
    main()
