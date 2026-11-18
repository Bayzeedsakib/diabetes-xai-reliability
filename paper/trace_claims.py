r"""Claim trace for main.tex: every number in the text (abstract to reproducibility statement) and, since the
D42 amendment, in every figure caption (D42 amendment 2: claims updated for the O35 steps; D42 amendment 3: reproducibility, limitations; D42 amendment 4: reviewer patch D42d) and in the captions and footnotes of the \input tables is tied to a source.
Table body cells are traced by build_tables.py (table_trace.csv), figure data by make_figures.py.

Categories
  R  registry value: the substring must occur in the named column of results_registry.csv
  D  derived from registry values: recomputed here from the registry strings and compared
  C  configuration or design (pre-stated setting, search space): source file and item named
  L  literature value (cited reference; wording approved by the reviewer)
  S  structural (split/section/rank index, years, 'top-5', file names): no result
  T  table cell value repeated in a footnote: asserted against the registry by build_tables.py (table_trace.csv)

Checks
  1. each claim snippet occurs in the traced text: body (comments, floats, \cite, \ref, \label, \input removed)
     plus captions of figures and captions/footnotes (\parbox) of the \input tables;
  2. each R/D check passes against the registry;
  3. every numeric token in the text lies inside at least one claim snippet (no untraced number).
Usage: python -I trace_claims.py main.tex results_registry.csv claim_trace.csv  (table files read from main.tex's folder)
"""
import csv, re, sys

tex_path, reg_path, out_path = sys.argv[1:4]
REG = {r["result_id"]: r for r in csv.DictReader(open(reg_path, encoding="utf-8"))}
import os
full = open(tex_path, encoding="utf-8").read()


def arg(t, i):
    """Content of the brace group starting at t[i] == '{'."""
    assert t[i] == "{"
    d = 0
    for k in range(i, len(t)):
        d += {"{": 1, "}": -1}.get(t[k], 0)
        if d == 0:
            return t[i + 1:k]
    raise ValueError("unbalanced")


def float_text(t):
    """Captions and \\parbox footnotes (second argument) of a LaTeX fragment."""
    out = []
    for m in re.finditer(r"\\caption\{", t):
        out.append(arg(t, m.end() - 1))
    for m in re.finditer(r"\\parbox\{", t):
        a = arg(t, m.end() - 1); j = m.end() + len(a) + 1
        out.append(arg(t, j))
    return " | ".join(out)


src = full[full.index("\\begin{abstract}"):full.index("\\bibliographystyle")]
src = re.sub(r"(?<!\\)%.*", "", src)
floats = [float_text(m.group(0)) for m in re.finditer(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}", src, flags=re.S)]
for m in re.finditer(r"\\input\{([^}]*)\}", src):
    floats.append(float_text(open(os.path.join(os.path.dirname(os.path.abspath(tex_path)), m.group(1) + ".tex"), encoding="utf-8").read()))
src = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", " ", src, flags=re.S)
src = src + " || FLOATS: " + " || ".join(floats)
src = re.sub(r"\\(cite|ref|label|input)\{[^}]*\}", " ", src)
T = re.sub(r"\s+", " ", src)


def reg(rid, sub, col="value"):
    assert rid in REG, rid
    assert sub in REG[rid][col], f"{rid}[{col}] lacks '{sub}': {REG[rid][col][:300]}"
    return f"{rid}[{col}] contains '{sub}'"


def nums(rid, pat, col="value"):
    return [float(x) for x in re.findall(pat, REG[rid][col])]


def rng(vals, nd):
    return f"{min(vals):.{nd}f}--{max(vals):.{nd}f}"


def d_prediab():
    v = [nums(r, r"^([\d.]+) \[")[0] for r in ["P3-001", "P3-003", "P3-007", "P3-008", "P3-009", "P3-010"]]
    assert rng(v, 3) == "0.740--0.747", v
    return "min/max of the ROC-AUC in P3-001, P3-003, P3-007-P3-010 (six paper models)"


def d_h2_cross():
    v = nums("P5-006", r"split \d [\d.]+ / ([\d.]+)")
    assert rng(v, 2) == "0.94--0.97", v
    return "min/max cross-family mean RBO over splits, P5-006"


def d_h2_native():
    v = [float(x) for x in re.findall(r"split \d ([-\d.]+) \[", REG["P5-010"]["value"].split("native:")[1])]
    big = [x for x in v if x > 0]
    assert len(big) == 4 and rng(big, 2) == "0.11--0.14", v
    return "native within-cross difference per split, P5-010 (4 positive splits 0.110-0.143; split 3 -0.006)"


def d_h2_common():
    v = [float(x) for x in re.findall(r"split \d ([-\d.]+) \[", REG["P5-010"]["value"].split("|")[0])]
    assert (min(v), max(v)) == (-0.012, 0.049), v
    return "common within-cross difference per split, P5-010"


def _h3():
    within, between = {}, {}
    for rid in ["P6-004", "P6-005", "P6-006"]:
        for block in REG[rid]["value"].split(" | "):
            pair = block.split(":")[0].strip() if ":" in block.split("split")[0] else "CAT-MLP"
            for sp, a, b, bt in re.findall(r"split (\d) ([\d.]+) / ([\d.]+) / ([\d.]+)", block):
                A, B = pair.split("-")
                within.setdefault(A, {})[sp] = float(a); within.setdefault(B, {})[sp] = float(b)
                between[(pair, sp)] = float(bt)
    return within, between


def d_mlp_within():
    w, _ = _h3(); v = list(w["MLP"].values())
    assert rng(v, 2) == "0.77--0.94", v
    return "MLP median within-refit RBO, splits 1-5, P6-004"


def d_between():
    _, b = _h3(); v = [x for (p, s), x in b.items() if p.endswith("MLP")]
    assert len(v) == 15 and (min(v), max(v)) == (0.885, 0.963), v
    lo = REG["P6-004"]["value"].split("split 3 ")[1].split(",")[0].split(" / ")[2]
    assert lo == "0.885", lo  # CAT-MLP split 3; the unrounded value 0.88499... (h3_pairs.csv) rounds to 0.88
    return "booster-MLP median between RBO, 15 pair x split cells, P6-004/P6-005 (min 0.88499... CAT-MLP split 3 -> 0.88; max 0.963 -> 0.96); D42 amendment A5: 0.89 came from rounding 0.885 a second time"


def d_booster_within():
    w, _ = _h3(); v = [x for m in ["CAT", "XGB", "LGBM"] for x in w[m].values()]
    assert (min(v), max(v)) == (0.915, 0.994), v
    return "booster median within-refit RBO, P6-004/P6-005/P6-006 (0.915-0.994; minimum CAT split 2)"


def d_refit_dis():
    v = nums("P6-008", r"([\d.]+)(?= /| ;|;)")
    v = [float(x) for x in re.findall(r"\d\.\d{3}", REG["P6-008"]["value"].split("; above")[0])]
    assert (min(v), max(v)) == (0.006, 0.233), v
    return "min/max refit disagreement, P6-008"


def d_12of15():
    reg("P6-010", "the MLP's in 3/15")
    return "15 cells - 3 where between exceeds the MLP's refit disagreement (P6-010); note field states 12/15"


def d_refit_ap():
    sel = {m: [float(x) for x in re.search(r"AP [\d.]+ ± [\d.]+ \(([\d.]+); ([\d.]+); ([\d.]+); ([\d.]+); ([\d.]+)\)", REG[r]["value"]).groups()]
           for m, r in [("CAT", "P4-033"), ("XGB", "P4-029"), ("LGBM", "P4-030"), ("MLP", "P4-031")]}
    loss = {m: [] for m in sel}
    for k, part in enumerate(REG["P6-011"]["value"].split(" | ")):
        for m in sel:
            mu = float(re.search(m + r" ([\d.]+) ±", part).group(1))
            loss[m].append(sel[m][k] - mu)
    b = [x for m in ["CAT", "XGB", "LGBM"] for x in loss[m]]
    assert rng(b, 3) == "0.002--0.004" and rng(loss["MLP"], 3) == "0.006--0.018", (b, loss["MLP"])
    return "selected-model test AP per split (P4-029-P4-033) minus bootstrap-refit mean AP (P6-011)"


def d_slopes():
    v = [nums(r, r"b ([\d.]+)")[0] for r in ["P7-001", "P7-002", "P7-003"]]
    assert rng(v, 3) == "0.994--0.999", v
    return "Platt slopes b, P7-001-P7-003"


def d_platt():
    v = {r: REG[r]["value"] for r in ["P7-001", "P7-002", "P7-003"]}
    assert "Brier 0.1389 -> 0.0967" in v["P7-001"] and "Brier 0.1062 -> 0.0969" in v["P7-002"] and "Brier 0.1061 -> 0.0968" in v["P7-003"]
    ece = [float(re.search(r"ECE10 [\d.]+ -> ([\d.]+)", x).group(1)) for x in v.values()]
    assert max(ece) <= 0.005, ece
    return "P7-001-P7-003: Brier 0.1389->0.0967, 0.1062->0.0969, 0.1061->0.0968 (3 decimals 0.139->0.097, 0.106->0.097); ECE10 after 0.0039/0.0044/0.0047 <= 0.005"


def d_r2():
    disc = [float(REG[r]["value"].split("; ")[4]) for r in [f"P2-0{i:02d}" for i in range(8, 14)]]
    cont = [float(REG[r]["value"].split("; ")[4]) for r in ["P2-014", "P2-015", "P2-016"]]
    t5 = float(REG["P2-006"]["value"].split("; ")[4])
    assert rng(disc, 2) == "0.31--0.36" and rng(cont, 2) == "0.46--0.48" and f"{t5:.2f}" == "0.50", (disc, cont, t5)
    return "median LIME R2 (5th field): P2-008-P2-013 discretised, P2-014-P2-016 continuous, P2-006 thesis setting"


def d_neg_ratio():
    r = int(REG["S1-006"]["value"]) / int(REG["S1-004"]["value"])
    assert f"{r:.2f}" == "6.18", r
    return "S1-006 / S1-004 = 218,334 / 35,346"


def d_h5():
    reg("P2-022", "0.1128; 0.0046")
    return "P2-022: 0.1128 -> 0.113, 0.0046 -> 0.005"


C = []  # (snippet, category, source)
def add(snip, cat, *src):
    C.append((snip, cat, " ; ".join(s if isinstance(s, str) else s() for s in src)))

S = "structural"
# ---------------- Abstract
add("BRFSS 2015 health indicators file", "S", S)
add("holds 253,680 respondents and 21 self-reported predictors", "R", reg("S1-001", "253680"), reg("S1-002", "21"))
add("(prediabetes is coded 0)", "R", reg("P0-004", "4631"), "code 0 = negative class")
add("and LIME in 11 settings with 3 seeds", "R", reg("P2-005", "12 runs (T5, T10, N_{quartile,decile,none}_{0.5,1,2}, T5_chol50)"), reg("P2-005", "3 (thesis LIME seeds)"), "11 grid settings + T5_chol50 (design section 10 item 6)")
add("at $\\delta=0.01$ average precision in 5/5 splits (margin set after the first split), not at 0.005", "R", reg("P4-051", "5/5; supported"), reg("P4-052", "0/5; not supported"), reg("P4-051", "delta = 0.01 was set after the split-1 (thesis) results were known", "notes"))
add("mainly because MLP refits disagreed", "R", reg("P6-010", "the MLP's in 3/15", ), reg("P6-008", "CAT 0.013 / 0.085"), "mainly: the split-2 CatBoost failure is CatBoost's own refit instability (D42 amendment A7a)")
add("CatBoost--MLP passed in 1/5 splits", "R", reg("P6-004", "passes 1/5"))
add("on sign (median 1.00)", "R", reg("P2-018", "1.000"))
add("for 81\\% of case--model pairs", "R", reg("P2-019", "0.810"))
# ---------------- Introduction
add("The BRFSS 2015 diabetes health indicators file", "S", S)
add("near-equivalent at $\\delta=0.01$ AP in 5/5 splits, not at 0.005", "R", reg("P4-051", "5/5"), reg("P4-052", "0/5"))
add("(except CatBoost in split 2)", "R", reg("P6-008", "CAT 0.013 / 0.085"))
add("than on its seed (81\\%)", "R", reg("P2-019", "0.810"))
add("(55\\% with variances over the nine categorical-ordinal settings)", "R", reg("P2-030", "9 new: 0.554"), "P2-030 = variance comparison")
add("(ROC-AUC 0.740--0.747 across the six models)", "D", d_prediab)
add("includes BRFSS 2015", "S", S)
add("merged BRFSS 2015 with", "S", S)
# ---------------- Methods A, B
add("We use the BRFSS 2015 diabetes health indicators file (253,680 respondents)", "R", reg("S1-001", "253680"))
add("all 21 predictors", "R", reg("S1-002", "21"))
add("the row-aligned 3-class file", "R", reg("P0-002", "253680 of 253680"))
add("respondents are coded 0, so", "R", reg("P0-004", "4631"))
add("Five stratified 80:20 outer splits", "R", reg("S1-025", "202944"), reg("S1-027", "50736"), "design section 2")
add("split 1 reuses", "S", S)
add("the same 30 frozen candidate configurations by 5-fold stratified", "C", "design section 2; S7(3) (30 configurations); thesis_common.py N_ITER")
add("a 50,000-row stratified subsample", "C", "D14; design section 2")
add("$C$ log-uniform $10^{-4}$--$10^{2}$", "C", "thesis_common.py SEARCH_SPACES['LR']")
add("liblinear, 1,000 iterations", "C", "thesis_common.py LR pipeline; thesis Appendix Table A.2")
add("trees \\{100, 200, 300\\}, maximum depth \\{8, 12, 16\\}, minimum leaf size \\{1, 5, 10, 20, 50\\}, features per split \\{$\\sqrt{p}$, $\\log_2 p$, 0.5\\}, bootstrap fraction \\{0.5, 0.8, 1\\}", "C", "SEARCH_SPACES['RF']")
add("trees 100--600, learning rate log-uniform 0.01--0.3, depth 3--8, minimum child weight 1--10, row and column subsampling 0.6--1.0, L2 penalty log-uniform 0.01--10, positive-class weight \\{1, 2, 4, 6.18\\}", "C", "SEARCH_SPACES['XGB'] (randint upper bounds exclusive; SPW = [1, 2, 4, POS_WEIGHT_MAX])")
add("leaves 15--255 and minimum child samples 10--200", "C", "SEARCH_SPACES['LGBM']")
add("iterations 200--600, learning rate log-uniform 0.02--0.3, depth 4--8, L2 leaf penalty 1--10", "C", "SEARCH_SPACES['CAT']")
add("hidden layers \\{(32), (64), (128), (64, 32), (128, 64)\\}, L2 penalty log-uniform $10^{-6}$--$10^{-2}$, initial learning rate log-uniform $10^{-4}$--$10^{-2}$, batch \\{256, 512, 1,024\\}", "C", "SEARCH_SPACES['MLP']")
add("early stopping on a 10\\% inner validation set", "C", "thesis_common.py MLP pipeline (validation_fraction 0.1); outline v2 III-B")
add("The value 6.18 is the negative-to-positive ratio", "D", d_neg_ratio)
# ---------------- Methods C-F
add("(paired class-stratified bootstrap, $B=2{,}000$), Holm-adjusted across the five candidates, is below 0.05 at $\\delta=0.01$; $\\delta=0.005$ and 0.02 are sensitivity analyses", "C", "design section 1 (H1), section 5 and section 12; D37 amendment A1 (decision = Holm-adjusted TOST p)")
add("after the split-1 results were known", "S", S)
add("TreeSHAP with 500 background rows", "C", "design section 3; S15(b)")
add("one 100-row background", "C", "design section 3 / section 9 item 1")
add("order the 21 predictors", "R", reg("S1-002", "21"))
add("each split's 500-row test core", "R", reg("P6-018", "500 / 70 in each of splits 1-5"))
add("(RBO, $p=0.8$)", "C", "design section 12 item 12")
add("share one top-5 set", "S", S)
add("10 refits in split 1, 5 in splits 2--5", "C", "design section 9 item 2; P6-001 (140 bootstrap refits)")
add("permutation $p<0.05$", "C", "design section 1 (H3)")
add("validation split (10\\%)", "C", "D40 methods sentence (fix e)")
add("all 252 assignments in splits 2--5", "C", "design section 1 (H3); D40")
add("at most 126 distinct values of $T$", "C", "design section 9 item 8")
add("observed 40--110", "R", reg("P6-017", "40-110"))
add("in split 1, 9,999 random assignments", "C", "D40 part 1")
add("within 0.005 CV AP of the selected one", "C", "design section 3 / section 9 item 2")
add("are descriptive, in split 1.", "S", S)
add("class-weighted boosters (split 1)", "S", S)
add("In split 1, the core and a 30-case panel (530 cases)", "R", reg("P2-005", "530 (500-row core + 30-case panel)"))
add("by LIME in 11 settings with 3 seeds each", "R", reg("P2-005", "3 (thesis LIME seeds)"), "11 grid settings: design section 10 item 6")
add("treats the 14 binary predictors as categorical and discretises the 7 others", "R", reg("S1-002", "14 binary, 4 ordinal, 3 numerical", "notes"))
add("T10 doubles its 5,000 samples", "C", "design section 10 item 6 (T5 5,000; T10 10,000)")
add("treat the 14 binary and 4 ordinal predictors", "R", reg("S1-002", "14 binary, 4 ordinal", "notes"))
add("SHAP's top 5 is at least 0.95", "C", "design section 1 (H4)")
# ---------------- Results A
add("H1 was supported at $\\delta=0.01$ in 5/5 splits; not at 0.005 (0/5, boosting only); at 0.02 RF joined in 5/5 splits", "R", reg("P4-051", "5/5; supported"), reg("P4-052", "0/5; not supported; members XGB, LGBM, CAT"), reg("P4-053", "5/5; supported; members XGB, LGBM, MLP, RF, CAT"))
add("$\\delta$ was set after the split-1 results were known", "S", S)
add("lower by 0.0045 (Nadeau--Bengio $p=0.009$)", "R", reg("P4-057", "-0.0045 (0.0014)"), reg("P4-057", "; 0.0094;"))
add("TOST $p=0.002$; Holm 0.0066", "R", reg("P4-057", "0.304 / 0.00219 / 4.15e-05"), reg("P4-057", "0.911 / 0.00657 / 0.000124", "notes"))
add("ALT\\_RF 0.4332 vs RF 0.4333; ALT\\_XGB within 0.001 AP of CatBoost on average, per split $-0.0027$ to $+0.0010$", "R", reg("P4-039", "0.4332 ± 0.0056"), reg("P4-032", "AP 0.4333"), reg("P4-061", "mean -0.0008"), reg("P4-061", "-0.0027"), reg("P4-061", "+0.0010"))
add("matches the reported 86\\% accuracy (0.863 in split 1), against 0.861", "R", reg("P4-060", "0.8634 (AHMED_LR_default, split 1)"), reg("S5-152", "0.861"), "86%: Ahmed et al. [4] reported value (design section 4)")
# ---------------- Results B
add("small ($-0.012$ to 0.049, about one swap of adjacent predictors at ranks 3--5)", "D", d_h2_common, reg("P5-011", "ranks 3", "value") if "ranks 3" in REG["P5-011"]["value"] else "ranks 3-5: D38 second amendment (b) mechanism")
add("the 500 explained rows", "R", reg("P5-010", "paired class-stratified bootstrap of the 500 core rows", "description"))
add("(mean RBO 0.94--0.97)", "D", d_h2_cross)
add("(0.11--0.14 in four of five splits)", "D", d_h2_native)
add("exchanging ranks 2--3", "S", "D38 second amendment (b); P5-011 top-6 orders")
add("(RBO 0.907)", "R", reg("P5-008", "log-odds vs probability 0.907"))
add("The top-5 set is agreement", "S", S)
add("LR and RF shared it in every split", "R", reg("P5-009", "5/5 (native 5/5)"), reg("P5-009", "the top-5 set is GenHlth, BMI, Age, HighBP, HighChol for every model, split and setting", "notes"))
# ---------------- Results C
add("(split 5: median between-model RBO 0.923, below the within-refit medians of 0.994 for CatBoost and 0.938 for the MLP; permutation $p=0.008$)", "R", reg("P6-004", "split 5 0.994 / 0.938 / 0.923, T +0.0143, p 0.0079 PASS"))
add("(in 12 of 15 booster--MLP pair $\\times$ split cells; median within-refit RBO 0.77--0.94 against median between-model RBO 0.88--0.96)", "D", d_12of15, d_mlp_within, d_between)
add("(0.92--0.99; the lowest value is CatBoost in split 2)", "D", d_booster_within)
add("($1-$median within-refit RBO; 0.006--0.233)", "D", d_refit_dis)
add("floor (0.000, measured for XGBoost in split 1)", "R", reg("P5-008", "perm vs exact 1.000"), reg("P6-008", "above 0.000 in 20/20"))
add("All 359 refit explanations", "R", reg("P6-009", "359/359"))
add("In the native setting, only CatBoost--LightGBM passed among booster pairs (split 3)", "R", reg("P6-007", "split 3 0.979 / 0.976 / 0.976, T +0.0006, p 0.0476 PASS; split 4 0.980 / 0.989 / 0.983, T -0.0030, p 0.7619; split 5 0.985 / 0.986 / 0.988, T -0.0027, p 0.6587; passes 1/5"), reg("P6-007", "XGB-LGBM: split 1"), "native CAT-XGB passes 0/5 and XGB-LGBM passes 0/5 (P6-007); D42 amendment 2, O35 step 3a")
add("(CatBoost--XGBoost and XGBoost--LightGBM, splits 3 and 5)", "R", reg("P6-006", "split 3 0.984 / 0.988 / 0.944, T +0.0403, p 0.0079 PASS"), reg("P6-006", "split 5 0.994 / 0.988 / 0.948, T +0.0400, p 0.0079 PASS"))
add("(an RBO change of about 0.04)", "R", reg("P6-006", "T +0.0403"), "H3_results_paragraph.md (reviewed)")
add("lowered test AP by 0.002--0.004 for the boosters and by 0.006--0.018 for the MLP", "D", d_refit_ap)
add("In split 1, MLP refits", "S", S)
add("test AP 0.4246 $\\pm$ 0.0028 (CatBoost 0.4316 $\\pm$ 0.0003", "R", reg("P6-013", "MLP seed 0.4246 ± 0.0028"), reg("P6-013", "CAT seed 0.4316 ± 0.0003"))
add("(median within-set RBO 0.940 against at least 0.9996", "R", reg("P6-014", "MLP 0.911 / 0.940"), reg("P6-014", "boosters >= 0.9996", "notes"))
add("8 of the 10 LR configurations are within $5\\times10^{-5}$ CV AP", "R", reg("P6-015", "LR 20 (gap < 5e-5: 8)"), "10 = capped set incl. the selected configuration (P6-014 description)")
add("within-set RBO of 1.000", "R", reg("P6-014", "LR 0.992 / - / 1.000"))
# ---------------- Results D
add("(median 1.00)", "R", reg("P2-018", "1.000"))
add("for 81\\% of case--model pairs (76\\% when variances are compared)", "R", reg("P2-019", "0.810"), reg("P2-030", "11: 0.764"))
add("(55\\% with variances)", "R", reg("P2-030", "9 new: 0.554"))
add("H4 was supported", "R", reg("P2-018", "supported"), reg("P2-019", "supported"))
add("(46\\% over all settings and 33\\% over the nine, with variances)", "R", reg("P2-030", "LR 0.455 / 0.328"))
add("(median $R^2$ 0.31--0.36 against 0.50), but not when they were left continuous (0.46--0.48)", "D", d_r2)
add("(gap 0.113 natively, 0.005 rescaled)", "D", d_h5)
add("swung 23-fold", "R", reg("P2-028", "swing 23.0-fold -> 1.05"))
add("and 1.05-fold after rescaling", "R", reg("P2-028", "-> 1.05"))
add("sampling CholCheck at 50\\% raised LIME's share in all 3,180 pairs, median $\\times$1.27", "R", reg("P2-029", "3180 of 3180; 1.27"), "50%: design section 3 (H5 intervention c)")
# ---------------- Results E
add("ROC-AUC to 0.9701 on a test set with 50\\% prevalence and 84\\% synthetic positives, against 0.8268", "R", reg("P1-028", "0.9701"), reg("P1-033", "36700 (84.0%)"), reg("P1-028", "50% prevalence", "description"), reg("P1-032", "0.8268"))
add("(e.g., ROC-AUC 0.975 with SMOTE", "L", "Khokhar et al. [5] arXiv:2501.18071 reported value; wording fixed by the reviewer (D41 amendment B)")
add("changed AP by $+0.0000$ (LR) and $-0.0002$ (CAT) on average", "R", reg("P1-034", "AP grouped 0.4048 ± 0.0061, row-level 0.4048 ± 0.0052, paired diff +0.0000"), reg("P1-035", "paired diff -0.0002"))
add("(slopes 0.994--0.999)", "D", d_slopes)
add("Brier score from 0.139 to 0.097 (CAT) and from 0.106 to 0.097 (XGB, LGBM), with ECE at most 0.005 after", "D", d_platt)
add("refitting without them changed AP by $-0.0008$ to $+0.0014$ across the six models", "R", reg("P3-017", "range -0.0008 to +0.0014"), reg("P3-017", "LR +0.00005; RF -0.00006; XGB +0.00019; LGBM -0.00075; CAT -0.00007; MLP +0.00138"))
add("all 359 refit explanations share the five leading predictors", "R", reg("P6-009", "359/359"))
# ---------------- Discussion, Conclusion, Reproducibility
add("A top-5 statement from any one", "S", S)
add("that top-5 claims hold", "S", S)
add("H3 has limited power (with 5+5 refits the smallest $p$ is 0.008)", "C", "design section 9 item 8 (2/252)")
add("studied in split 1 only", "S", S)
add("On the BRFSS 2015 file", "S", S)
add("(SHA-256 in the repository)", "R", reg("S1-047", "19f367e3"))
add("(AMD Ryzen 5 5500U, 12 logical CPUs, 16 GB RAM, CPU only, Python 3.13)", "R", reg("S3-021", "Ryzen 5 5500U", "notes"), reg("S7-001", "12 logical CPUs", "notes"), reg("S3-045", "3.13.0"), "16 GB: decisions_log D-row of 2026-10-12 (Stage 3 plan, PC 6 cores / 12 threads, 16 GB); CPU only: no GPU in any run")
add("master seed 42 (paper\\_outer 2128606514, paper\\_r6 1741519417)", "R", reg("S1-046", "42"), reg("P5-010", "paper_r6 = 16th stream 1741519417", "notes"), "paper_outer = SeedSequence(42).generate_state(16)[14] = 2128606514 (recomputed; r4 env_log_r4_pc.json field paper_outer)")
add("the LIME grid, the ablations, reproduction checks and calibration ran in a Linux workspace", "R", reg("P2-005", "workspace", "machine"), reg("P1-028", "workspace", "machine"), reg("P1-005", "workspace", "machine"), reg("P7-001", "workspace", "machine"))
add("CatBoost used 12 threads", "R", reg("P1-007", "12 = PC logical CPUs reproduces the frozen model", "notes"), reg("P1-007", "2.2e-16 / 0.146 / 0.119"))
add("by up to 0.17 at identical AP", "R", reg("P1-005", "0.167; 0.4319 / 0.4319"), reg("P1-007", "thread_count", "notes"))

# ---------------- Captions and footnotes (D42 amendment, reviewer item C)
add("H1--H5 decision rules were stated before the analyses ($\\delta$ for H1 after the split-1 results)", "C", "design section 1; P4-051 notes (delta set after split 1, D32)")
add("Refits: 10 per model in split 1, 5 in splits 2--5", "C", "design section 9 item 2; P6-001 (140 bootstrap refits)")
add("for the 11 LIME settings and the CholCheck intervention, split 1, 530 cases", "R", reg("P2-005", "530 (500-row core + 30-case panel)"), reg("P2-005", "12 runs (T5, T10, N_{quartile,decile,none}_{0.5,1,2}, T5_chol50)"))
add("median LIME local $R^2$", "S", S)
add("T5: thesis setting (14 binary predictors categorical, 7 others quartile-discretised, 5,000 samples); T10: T5 with 10,000 samples; N: Section~III-F", "R", reg("P2-005", "T = thesis categorical scheme (14 binary categorical, 7 non-binary quartile)", "notes"), "5,000 / 10,000: design section 10 item 6 (D42 correction 3)")
add("width: multiple of $0.75\\sqrt{21}$", "R", reg("P2-005", "kernel width as multiple of 0.75*sqrt(21)", "notes"))
add("Dataset and partition (BRFSS 2015 diabetes", "S", S)
add("Prediabetes respondents are coded 0 (checked against the 3-class file)", "R", reg("P0-004", "4631"), reg("P0-002", "253680 of 253680"))
add("Five stratified 80:20 splits by row (split 1: thesis partition)", "R", reg("S1-025", "202944"), reg("S1-027", "50736"), reg("P4-002", "260/260 pass"))
add("(mean $\\pm$ SD, ddof 1)", "R", reg("P4-033", "mean ± SD (ddof 1)", "description"))
add("Nadeau--Bengio TOST (4 df)", "R", reg("P4-057", "t with 4 df", "notes"))
add("RF 0.072, MLP 0.0066", "T", reg("P4-058", "1 / 0.0716 / 0.000384", "notes"), reg("P4-057", "0.911 / 0.00657 / 0.000124", "notes"), "asserted by build_tables.py")
add("Splits equivalent at $\\delta=0.01$ / $0.005$", "R", reg("P4-051", "5/5"), reg("P4-052", "0/5"), "per-split flags P4-046-P4-050 counted by build_tables.py")
add("$B=2{,}000$); $\\delta=0.01$ set after the split-1 results", "C", "design section 1 (H1); P4-046 notes (B = 2,000)")
add("thresholds 0.50 / Youden", "C", "design section 4 (baselines)")
add("AP 0.4069$\\pm$0.0062 / 0.4070$\\pm$0.0060, ROC-AUC 0.8238$\\pm$0.0030 / 0.8181$\\pm$0.0034", "T", reg("P4-035", "0.4069 ± 0.0062; 0.8238 ± 0.0030"), reg("P4-037", "0.4070 ± 0.0060; 0.8181 ± 0.0034"), "D42 amendment cut (g)")
add("(extrapolated RBO, $p=0.8$, 21 predictors), splits 1--5", "C", "design section 12 item 12; S1-002 (21)")
add("Ablations and robustness (split 1 unless noted)", "S", S)
add("50\\% prevalence; 84\\% of test positives synthetic", "R", reg("P1-028", "50% prevalence", "description"), reg("P1-033", "36700 (84.0%)"))
add("removing the 910 test prediabetes respondents minus that from removing 910 random negatives (200 draws)", "R", reg("P3-018", "200 draws", "description"), reg("P0-006", "3721 / 910"))

# ---------------- checks 1 and 3
covered = [False] * len(T)
rows = []
for snip, cat, source in C:
    n = re.sub(r"\s+", " ", snip)
    hits = [m.start() for m in re.finditer(re.escape(n), T)]
    assert hits, f"snippet not in text: {snip}"
    for h in hits:
        for k in range(h, h + len(n)): covered[k] = True
    rows.append((cat, snip, source, len(hits)))
untraced = []
for m in re.finditer(r"(?<![A-Za-z\\{])[-+]?\d[\d,.]*", T):
    if not covered[m.start()]:
        untraced.append((m.group(0), T[max(0, m.start() - 40):m.end() + 20]))
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["category", "text_snippet", "source_and_check", "occurrences"]); w.writerows(rows)
print(f"{len(rows)} claims; registry checks passed; untraced numeric tokens: {len(untraced)}")
for t, ctx in untraced: print("  UNTRACED", repr(t), "|", ctx)
sys.exit(1 if untraced else 0)
