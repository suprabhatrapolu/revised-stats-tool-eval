"""The get_values tool: reads only the local ALFRED vintage archive (copied read-only semantics of the audit's vintage.py).

The record for an as-of date is the most recent vintage whose date is <= as_of (PREREG s6 of the audit).
Every derived figure (change, percent change, YoY, SAAR) is computed within that one vintage.
"""
import glob, os, re
import pandas as pd

GA = os.environ["GA_ROOT"]  # path to the companion audit's data (released with that study)
VINT_DIR = os.path.join(GA, "data", "vintages")

QUARTERLY = {"GDPC1"}
META = {
    "PAYEMS": ("All employees, total nonfarm (seasonally adjusted)", "thousands of persons"),
    "UNRATE": ("Unemployment rate (seasonally adjusted)", "percent"),
    "CES0500000003": ("Average hourly earnings of all employees, total private (seasonally adjusted)", "dollars per hour"),
    "CPIAUCSL": ("CPI-U, all items (seasonally adjusted)", "index 1982-84=100"),
    "CPIAUCNS": ("CPI-U, all items (not seasonally adjusted)", "index 1982-84=100"),
    "CPILFESL": ("CPI-U, all items less food and energy (seasonally adjusted)", "index 1982-84=100"),
    "CPILFENS": ("CPI-U, all items less food and energy (not seasonally adjusted)", "index 1982-84=100"),
    "GDPC1": ("Real gross domestic product (seasonally adjusted annual rate)", "billions of chained 2017 dollars"),
}
MAX_PERIODS = 25


def _load():
    frames = {}
    for p in sorted(glob.glob(os.path.join(VINT_DIR, "*_vintages.csv"))):
        s = os.path.basename(p).split("_vintages")[0]
        df = pd.read_csv(p, parse_dates=["observation_date"]).set_index("observation_date")
        df.columns = [pd.Timestamp(c.split("_")[-1]).strftime("%Y-%m-%d") for c in df.columns]
        frames[s] = df[sorted(df.columns)]
    return frames


FRAMES = _load()


def _ts(ref):
    ref = str(ref).strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", ref)
    if m: return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.fullmatch(r"(\d{4})M(\d{1,2})", ref, re.I)  # FRED-style 2026M01 (pilot fix, logged in prompts/CHANGELOG.md)
    if m: return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.fullmatch(r"(\d{4})\s*-?\s*Q([1-4])", ref, re.I)
    if m: return pd.Timestamp(int(m.group(1)), 3 * (int(m.group(2)) - 1) + 1, 1)
    raise ValueError(f"unrecognised period {ref!r}; use YYYY-MM for monthly series or YYYYQn for GDPC1")


def _label(series, ts):
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}" if series in QUARTERLY else ts.strftime("%Y-%m")


def get_values(series, start_period, end_period=None, as_of=None):
    """Return the rows of `series` from start_period to end_period as published in the vintage in force on as_of
    (latest vintage if as_of is None). Errors come back as {"error": ...} so the model can correct its call."""
    series = str(series).strip().upper()
    if series not in FRAMES:
        return {"error": f"unknown series {series!r}; available: {sorted(FRAMES)}"}
    df = FRAMES[series]
    try:
        t0 = _ts(start_period); t1 = _ts(end_period or start_period)
    except ValueError as e:
        return {"error": str(e)}
    if t1 < t0: t0, t1 = t1, t0
    if as_of is None:
        vint = df.columns[-1]
    else:
        try:
            a = pd.Timestamp(str(as_of).strip()).strftime("%Y-%m-%d")
        except Exception:
            return {"error": f"unrecognised as_of {as_of!r}; use YYYY-MM-DD"}
        cands = [v for v in df.columns if v <= a]
        if not cands:
            return {"error": f"no vintage of {series} on or before {a}; earliest is {df.columns[0]}"}
        vint = cands[-1]
    col = df[vint]
    step = 3 if series in QUARTERLY else 1
    rows, ts = [], t0
    while ts <= t1:
        if len(rows) >= MAX_PERIODS:
            return {"error": f"range too long; at most {MAX_PERIODS} periods per call"}
        x = col.get(ts)
        row = {"period": _label(series, ts)}
        if x is None or pd.isna(x):
            row["value"] = None; row["note"] = "not yet published in this vintage"
        else:
            row["value"] = round(float(x), 3)
            prev = col.get(ts - pd.DateOffset(months=step))
            if prev is not None and pd.notna(prev) and prev != 0:
                row["change_from_prior_period"] = round(float(x - prev), 3)
                if series in QUARTERLY:
                    row["annualized_growth_pct"] = round(((x / prev) ** 4 - 1) * 100, 3)
                else:
                    row["pct_change_from_prior_month"] = round((x / prev - 1) * 100, 3)
            yo = col.get(ts - pd.DateOffset(months=12))
            if yo is not None and pd.notna(yo) and yo != 0:
                row["pct_change_from_year_earlier"] = round((x / yo - 1) * 100, 3)
        rows.append(row)
        ts = ts + pd.DateOffset(months=step)
    name, units = META.get(series, (series, ""))
    return {"series": series, "name": name, "units": units, "vintage_date": vint, "rows": rows}
