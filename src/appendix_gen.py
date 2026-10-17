"""Write paper/build/appendix.tex: the frozen v2 prompts and the C2 tool schema, verbatim."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "prompts", "v2")
rd = lambda f: open(os.path.join(D, f), encoding="utf-8").read().strip()
BL, EL = r"\begin{lstlisting}", r"\end{lstlisting}"
out = [r"The system prompt is shared by all conditions; \texttt{\{CONDITION\}} is replaced by one of the three condition "
       r"texts below. Files are reproduced verbatim (prompt version v2, frozen after the pilot)." + "\n",
       r"\paragraph{System prompt.}" + "\n" + BL + "\n" + rd("system.txt") + "\n" + EL + "\n"]
for c, n in [("C0", "no tool"), ("C1", "latest-value tool"), ("C2", "as-of tool")]:
    out.append(rf"\paragraph{{Condition text, {c} ({n}).}}" + "\n" + BL + "\n" + rd(f"condition_{c}.txt") + "\n" + EL + "\n")
out.append(r"\paragraph{User message.} Fields in braces are filled from the claim record; the consensus line appears only "
           r"for claims about forecasts." + "\n" + BL + "\n" + rd("user.txt") + "\n" + EL + "\n")
t = json.load(open(os.path.join(D, "tool_C2.json"), encoding="utf-8"))
out.append(r"\paragraph{Tool schema (C2).} The C1 schema is identical without \texttt{as\_of}, and describes the figures "
           r"as the latest available." + "\n" + BL + "\n" + json.dumps(t, indent=1) + "\n" + EL + "\n")
open(os.path.join(ROOT, "paper", "build", "appendix.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(out))
