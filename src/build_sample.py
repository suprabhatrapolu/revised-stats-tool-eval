"""Build the frozen evaluation set from the audit repo (read-only).

FLIP    = every checkable claim supported on its publication date and contradicted by the latest vintage (census).
STABLE  = every checkable claim supported on its publication date and still supported now (census).
HOLDOUT = every checkable claim of the two held-out releases, labelled by blind adjudication (census).
"""
import glob, hashlib, json, os, sys
import pandas as pd

GA = os.environ["GA_ROOT"]  # path to the companion audit's data (released with that study)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

COLS = ["claim_id", "doc_id", "event_id", "tier", "as_of", "claim_text", "type", "series", "ref", "kind", "horizon"]


def consensus_from_log(log):
    try:
        c = json.loads(log).get("consensus")
    except Exception:
        return None
    return None if c is None else round(float(c), 2)


def main():
    d = pd.read_csv(os.path.join(GA, "data", "processed", "all_verdicts.csv"), dtype=str)
    s = d[(d.checkable == "True") & (d.verdict_asof == "supported")].copy()
    n0 = len(s)
    s = s.drop_duplicates()  # one claim row appears twice verbatim in the audit table (3f0fe57768fb14fd)
    print("supported rows", n0, "after exact-duplicate removal", len(s), file=sys.stderr)
    s["consensus"] = s.verdict_log_asof.map(consensus_from_log)
    s["subset"] = s.flip.map({"True": "FLIP", "False": "STABLE"})
    s["reference"] = "SUPPORTED"
    s["reference_source"] = "pipeline verdict at publication vintage"

    # held-out releases: claim rows from each event's verdicts.csv, labels from the blind adjudications
    rows = []
    for ev in ["es-2026-09-04", "cpi-2025-12-18"]:
        v = pd.read_csv(os.path.join(GA, "data", "processed", ev, "verdicts.csv"), dtype=str)
        v["event_id"] = ev
        rows.append(v)
    hv = pd.concat(rows)
    hv = hv[hv.checkable == "True"].copy()
    lab3 = pd.read_csv(os.path.join(GA, "analysis", "holdout3", "scored.csv"), dtype=str)[["claim_id", "verdict"]]
    lab4 = pd.concat(pd.read_csv(p, dtype=str)[["claim_id", "verdict"]]
                     for p in sorted(glob.glob(os.path.join(GA, "analysis", "holdout4", "adjudications", "b*.csv"))))
    lab = pd.concat([lab3, lab4])
    assert lab.claim_id.is_unique, "duplicate adjudication"
    h = hv.merge(lab, on="claim_id", how="inner")
    missing = set(hv.claim_id) - set(lab.claim_id)
    print("holdout checkable", len(hv), "labelled", len(h), "unlabelled", len(missing), file=sys.stderr)
    h["consensus"] = h.verdict_log_asof.map(consensus_from_log)
    h["subset"] = "HOLDOUT"
    h["adjudicated_verdict"] = h.verdict
    h["reference"] = h.verdict.map({"supported": "SUPPORTED", "contradicted": "CONTRADICTED"}).fillna("UNVERIFIABLE")
    h["reference_source"] = "blind adjudication"

    keep = COLS + ["consensus", "subset", "reference", "reference_source", "verdict_asof", "verdict_latest"]
    out = pd.concat([s[keep], h[keep + ["adjudicated_verdict"]]], ignore_index=True)
    out = out.sort_values(["subset", "claim_id"]).reset_index(drop=True)
    assert out.claim_id.is_unique
    out.to_csv(os.path.join(OUT, "sample.csv"), index=False, lineterminator="\n")

    ids = "\n".join(out.subset + "," + out.claim_id) + "\n"
    with open(os.path.join(OUT, "claim_ids.txt"), "w", newline="\n") as f:
        f.write(ids)
    print("sha256(claim_ids.txt) =", hashlib.sha256(ids.encode()).hexdigest())
    print(out.subset.value_counts().to_string())
    print(pd.crosstab(out.subset, out.reference).to_string())
    print(pd.crosstab(out.series, out.subset).to_string())
    print("documents per subset:", out.groupby("subset").doc_id.nunique().to_dict())
    print("holdout adjudicated:", h.adjudicated_verdict.value_counts().to_dict())
    print("expectation claims w/o consensus:",
          out[(out.type == "comparison_to_expectation") & out.consensus.isna()].groupby("subset").size().to_dict())


if __name__ == "__main__":
    main()
