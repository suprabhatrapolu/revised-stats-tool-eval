"""Pre-submission checks on paper/build/main.pdf and the .tex sources; writes paper/submission/ (clean PDF + fields).

    python src/final_checks.py
"""
import os, re, sys
from pypdf import PdfReader, PdfWriter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
B = os.path.join(ROOT, "paper", "build"); OUT = os.path.join(ROOT, "paper", "submission")
FORBID = [t.strip().lower() for t in os.environ.get("ANON_TERMS", "").split(",") if t.strip()]  # identifying terms, supplied locally
PLACEHOLDER = ["pending", "[headline", "todo", "xx", "tbd", "??", "anonurlplaceholder"]
PROSE = ["abstract.tex", "contributions.tex", "body.tex", "results.tex", "discussion.tex", "pending_h4.tex"]
problems = []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    r = PdfReader(os.path.join(B, "main.pdf"))
    pages = [p.extract_text() for p in r.pages]
    full = "\n".join(pages); low = full.lower()
    # anonymity (word-boundary match for short terms)
    for w in FORBID:
        pat = r"\b" + re.escape(w) + r"\b" if w.isalpha() else re.escape(w)
        for m in re.finditer(pat, low):
            problems.append(f"identifying term '{w}': ...{full[max(0, m.start() - 40):m.end() + 40]!r}")
    for w in PLACEHOLDER:
        for m in re.finditer(re.escape(w), low):
            problems.append(f"placeholder '{w}': ...{full[max(0, m.start() - 50):m.end() + 50]!r}")
    # layout: body must end on page 4; references start no later than page 5
    ref_page = next(i + 1 for i, t in enumerate(pages) if re.search(r"^References$", t, re.M))
    lim_page = next(i + 1 for i, t in enumerate(pages) if "Limitations." in t)
    body_ok = ref_page <= 5 and (ref_page == 5 and pages[4].split("References")[0].count("\n") <= 1 or ref_page <= 4)
    print(f"pages={len(pages)} limitations_on={lim_page} references_on={ref_page} body_within_4_pages={body_ok}")
    if not body_ok: problems.append("body exceeds 4 pages")
    # metadata
    print("metadata:", dict(r.metadata or {}))
    # raw numbers in prose not coming from macros (for manual review)
    allowed = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "40", "2026", "2001", "2,000", "2027", "1.6", "8",
               "21", "3.6", "49", "2023", "2024", "2025", "1.6T", "8B", "3.1", "5.5", "95", "1000", "1,000"}
    raw = []
    for f in PROSE:
        t = open(os.path.join(B, f), encoding="utf-8").read()
        t = re.sub(r"\\[A-Za-z]+", " ", t)  # drop macro names
        t = re.sub(r"\\ref\{[^}]*\}|\\label\{[^}]*\}|\\citep?\{[^}]*\}", " ", t)
        for m in re.finditer(r"(?<![A-Za-z])\d[\d,.]*", t):
            v = m.group(0).rstrip(".,")
            if v not in allowed: raw.append((f, v, t[max(0, m.start() - 30):m.end() + 15].replace("\n", " ")))
    print(f"raw numbers in prose needing a source ({len(raw)}):")
    for f, v, ctx in raw: print(f"  {f}: {v}  | {ctx}")
    # abstract word count and expanded text
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", open(os.path.join(B, "numbers.tex"), encoding="utf-8").read()))
    def expand(t):
        t = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), t)
        t = re.sub(r"\\([A-Za-z]+)", lambda m: macros.get(m.group(1), "\\" + m.group(1)), t)
        t = t.replace("\\%", "%").replace("--", "–").replace("$", "").replace("<10^{-4}", "<0.0001").replace("\n", " ")
        return re.sub(r"\s+", " ", t).strip()
    abstract = expand(open(os.path.join(B, "abstract.tex"), encoding="utf-8").read())
    print("abstract words:", len(abstract.split()))
    if len(abstract.split()) > 200: problems.append("abstract over 200 words")
    if "\\" in abstract: problems.append("unexpanded macro in abstract")
    os.makedirs(OUT, exist_ok=True)
    title = "Today's Number, Yesterday's Claim: How Tool Design Shapes LLM Checks of Revised Statistics"
    fields = f"""# OpenReview fields (OE-Agent @ ACML 2026)

**Title:** {title}

**Abstract ({len(abstract.split())} words):**
{abstract}

**Keywords:** tool use, small language models, fact verification, temporal grounding, data revisions, agent evaluation

**TL;DR:** A data tool's time index decides LLM checks of claims about revised statistics: an 8B model given figures as published beats a 1.6T model without data, while a latest-value tool makes every model reject accurate reporting.

**Comments / notes field:** Short paper: 4 pages of main text; references and appendices follow. If short papers must fit within 4 pages including the appendix, we will move or drop the appendices. Builds on an anonymised companion study under review elsewhere.

**PDF:** paper/submission/main.pdf (metadata cleared)
"""
    open(os.path.join(OUT, "OPENREVIEW_FIELDS.md"), "w", encoding="utf-8", newline="\n").write(fields)
    if len("A data tool's time index decides LLM checks of claims about revised statistics: an 8B model given figures as published beats a 1.6T model without data, while a latest-value tool makes every model reject accurate reporting.") > 250:
        problems.append("TL;DR over 250 chars")
    # clean metadata copy
    w = PdfWriter()
    for pg in r.pages: w.add_page(pg)
    w.add_metadata({"/Title": title, "/Author": "", "/Creator": "", "/Producer": "", "/Subject": "", "/Keywords": ""})
    with open(os.path.join(OUT, "main.pdf"), "wb") as f: w.write(f)
    print("clean metadata:", dict(PdfReader(os.path.join(OUT, "main.pdf")).metadata))
    print("\nPROBLEMS:" if problems else "\nNO PROBLEMS FOUND")
    for p in problems: print(" -", p)


if __name__ == "__main__":
    main()
