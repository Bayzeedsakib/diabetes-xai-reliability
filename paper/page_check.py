"""Page check for the conference draft (D42 amendment 4).
Reports: page count; page of each float caption; text bottom of each column on the last page (fraction of the
text block 54-738 pt); and the smallest printed font of each figure = smallest char size in the figure PDF
(pdfplumber, upright characters; rotated text sized from its text matrix) x include scale (included width / figure PDF width). Widths come from LaTeX (columnwidth, textwidth
printed by a probe in main.log via \\typeout). Usage: python -I page_check.py  (after compiling main.tex)"""
import re, pdfplumber, csv
tex = open("main.tex", encoding="utf-8").read()
log = open("main.log", errors="ignore").read()
cw, tw = [float(x) for x in re.search(r"PAGECHECK columnwidth=([\d.]+)pt textwidth=([\d.]+)pt", log).groups()]
rows = []
with pdfplumber.open("main.pdf") as pdf:
    n = len(pdf.pages); rows.append(("pages", n))
    for k, pg in enumerate(pdf.pages, 1):
        ws = pg.extract_words()
        for i, w in enumerate(ws):  # caption tokens: "TABLEIV", "Fig.1." or "Fig." + "3."
            m = re.match(r"^TABLE([IVX]+)$", w["text"])
            f = re.match(r"^Fig\.(\d)\.$", w["text"]) or (w["text"] == "Fig." and i + 1 < len(ws) and re.match(r"^(\d)\.$", ws[i + 1]["text"]))
            if m: rows.append((f"float TABLE {m.group(1)}", f"page {k}, y {w['top']:.0f}, {'left' if w['x0'] < 306 else 'right/full'}"))
            if f and (w["text"] != "Fig." or ws[i + 1]["x0"] - w["x1"] < 6):
                n_ = f.group(1) if w["text"] != "Fig." else re.match(r"^(\d)", ws[i + 1]["text"]).group(1)
                rows.append((f"float Fig. {n_}", f"page {k}, y {w['top']:.0f}, {'left' if w['x0'] < 306 else 'right'}"))
    last = pdf.pages[-1]; words = last.extract_words()
    for side, sel in (("left", lambda w: w["x0"] < 306), ("right", lambda w: w["x0"] >= 306)):
        b = max([w["bottom"] for w in words if sel(w)] or [54])
        rows.append((f"page {n} {side} column bottom", f"{b:.1f} pt ({(b - 54) / (738 - 54):.3f} of text block)"))
for name in ["fig1_pipeline", "fig2_within_between", "fig3_shap_lime_settings"]:
    m = re.search(r"\\includegraphics\[width=([\d.]*)\\(columnwidth|textwidth)\]\{" + name + r"\.pdf\}", tex)
    w = (float(m.group(1)) if m.group(1) else 1.0) * (cw if m.group(2) == "columnwidth" else tw)
    with pdfplumber.open(name + ".pdf") as f:
        p = f.pages[0]; sc = w / float(p.width)
        up = [c for c in p.chars if c["text"].strip() and c["upright"]]
        rot = [c for c in p.chars if c["text"].strip() and not c["upright"]]
        smin = min(c["size"] for c in up); who = "".join(c["text"] for c in up if abs(c["size"] - smin) < 1e-3)[:40]
        if rot:  # rotated text: pdfplumber's size is not the font size; take it from the text matrix
            rsz = min(c["width"] for c in rot)  # 90-degree text: the glyph box width is the font size
            who += f"; rotated text {rsz:.2f} pt source -> {rsz * sc:.2f} pt printed"
        rows.append((f"{name}: PDF {float(p.width):.1f} x {float(p.height):.1f} pt, printed {w:.1f} x {float(p.height) * sc:.1f} pt (scale {sc:.3f})",
                     f"smallest source {smin:.2f} pt -> printed {smin * sc:.2f} pt ('{who}')"))
with open("page_check.csv", "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows([("item", "value")] + rows)
for a, b in rows: print(f"{a}: {b}")
