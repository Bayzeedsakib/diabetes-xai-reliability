r"""Word count per section of main.tex: text only (floats, \input tables, comments and LaTeX commands removed).
Sections are delimited by '%% WC:<name>' markers; inline math counts as one word.
Usage: python -I wordcount.py main.tex [out.csv]"""
import re, sys, csv
src = open(sys.argv[1], encoding="utf-8").read()
src = src.replace("%% WC:", "@@WC:")
src = re.sub(r"(?<!\\)%.*", "", src)                                # comments
src = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", " ", src, flags=re.S)
src = re.sub(r"\\input\{[^}]*\}", " ", src)
parts = re.split(r"@@WC:(.*)", src)
rows, tot = [], 0
for name, body in zip(parts[1::2], parts[2::2]):
    if name.strip() == "END":
        break
    t = re.sub(r"\\(cite|ref|label|bibliography|bibliographystyle)\{[^}]*\}", " ", body)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\(section|subsection)\*?\{([^}]*)\}", r" \2 ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}\[\]~\\]", " ", t)
    n = len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])
    rows.append((name.strip(), n)); tot += n
rows.append(("TOTAL text", tot))
for r in rows:
    print(f"{r[0]:<28}{r[1]:>6}")
if len(sys.argv) > 2:
    with open(sys.argv[2], "w", newline="") as f:
        csv.writer(f).writerows([("section", "words")] + rows)
