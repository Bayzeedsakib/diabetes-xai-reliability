"""Build Tables I-IV (IEEEtran LaTeX) for the conference paper.

Every cell is declared as (registry_id, value_string). Before any LaTeX is written,
the script asserts that each value_string occurs in the 'value' field of that registry
row (claude/results_registry.csv), so no number enters a table without its source.
Usage: python -I build_tables.py <results_registry.csv> <out_dir>
D42 amendment (2026-11-17): registry IDs removed from the Table I footnote; Table III footnote points to Section III-E
instead of repeating the permutation details; cut (f): refit-disagreement rows removed from Table III;
cut (g): ALT_LR and ALT_DT rows of Table II moved to one footnote sentence.
D42 amendment 4 (2026-11-18): Table I and Table IV footnotes shortened as in the reviewer's D42d patch.
"""
import csv, os, sys

reg_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
REG = {r["result_id"]: r for r in csv.DictReader(open(reg_path, encoding="utf-8"))}
TRACE = []


def v(rid, s, shown=None, field="value", comment=""):
    """Return the LaTeX text for a cell after checking s occurs in registry row rid (column `field`)."""
    assert rid in REG, f"{rid} not in registry"
    assert s in REG[rid][field], f"{rid}: '{s}' not found in {field}: {REG[rid][field][:200]}"
    TRACE.append((rid, field, s, comment))
    return shown if shown is not None else s


def pm(rid, s):  # "0.4085 ± 0.0060" -> LaTeX
    return v(rid, s, s.replace(" ± ", r"$\pm$"))


def neg(x):
    return x.replace("-", "$-$") if x.startswith("-") else x


# ---------------------------------------------------------------- Table I: dataset
t1 = [
    ("Respondents", v("S1-001", "253680", "253,680")),
    ("Predictors", v("S1-002", "21") + ": " + v("S1-002", "14 binary, 4 ordinal, 3 numerical", "14 binary, 4 ordinal, 3 numeric", field="notes",
     comment="type split is in the notes column of S1-002 (value holds only 21); predictor lists by type: phase4_design.md section 9 item 10; recounted from the data file by check_predictor_types.py")),
    ("\\quad ordinal", "GenHlth, Age, Education, Income"),
    ("\\quad numeric", "BMI, MentHlth, PhysHlth"),
    ("Positive class (self-reported diagnosed diabetes)", v("S1-004", "35346", "35,346") + " (" + v("S1-005", "13.93") + "\\%)"),
    ("Prediabetes respondents (coded 0)", v("P0-004", "4631", "4,631")),
    ("Development / test rows per split", v("S1-025", "202944", "202,944") + " / " + v("S1-027", "50736", "50,736")),
    ("Explanation core per split", v("P6-018", "500 / 70 in each of splits 1-5", "500 rows, 70 positive")),
    ("Exact repeated rows (kept)", v("S1-008", "24206", "24,206") + " (" + v("S1-009", "9.54") + "\\%)"),
]
L = [r"\begin{table}[!t]", r"\centering", r"\caption{Dataset and partition (BRFSS 2015 diabetes health indicators file)}",
     r"\label{tab:data}", r"\footnotesize", r"\begin{tabular}{@{}p{0.46\columnwidth}p{0.48\columnwidth}@{}}", r"\toprule", r"Item & Value \\", r"\midrule"]
L += [f"{a} & {b} \\\\" for a, b in t1]
L += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
      r"\parbox{\columnwidth}{\scriptsize Five stratified 80:20 splits by row (split 1: thesis partition). Prediabetes respondents are coded 0 (checked against the 3-class file).}",
      r"\end{table}"]
open(os.path.join(out, "table1_data.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Table II: performance
P = {"LR": ("P4-028", "0.4085 ± 0.0060", "0.8232 ± 0.0029"), "RF": ("P4-032", "0.4333 ± 0.0055", "0.8282 ± 0.0027"),
     "XGB": ("P4-029", "0.4399 ± 0.0072", "0.8321 ± 0.0033"), "LGBM": ("P4-030", "0.4394 ± 0.0066", "0.8318 ± 0.0030"),
     "CAT": ("P4-033", "0.4405 ± 0.0068", "0.8324 ± 0.0030"), "MLP": ("P4-031", "0.4360 ± 0.0068", "0.8307 ± 0.0030")}
# across-split Nadeau-Bengio TOST p at delta = 0.01 (middle of the three TOST values; unadjusted, as pre-stated)
D = {"LR": ("P4-054", "-0.0320 (0.0022)", "; 1 / 1 / 0.999", "1.00"), "RF": ("P4-058", "-0.0072 (0.0017)", "0.932 / 0.0358 / 0.000192", "0.036"),
     "XGB": ("P4-055", "-0.0006 (0.0011)", "0.00204 / 0.000111 / 6.26e-06", r"$<$0.001"), "LGBM": ("P4-056", "-0.0011 (0.0008)", "0.000803 / 3.26e-05 / 1.64e-06", r"$<$0.001"),
     "MLP": ("P4-057", "-0.0045 (0.0014)", "0.304 / 0.00219 / 4.15e-05", "0.002")}
HOLM_RF = v("P4-058", "Holm across the 5 candidates (reported beside it, D37 amendment A2): 1 / 0.0716 / 0.000384", "0.072", field="notes")
HOLM_MLP = v("P4-057", "Holm across the 5 candidates (reported beside it, D37 amendment A2): 0.911 / 0.00657 / 0.000124", "0.0066", field="notes")
# splits with equivalence (Holm TOST) at delta = 0.01 / 0.005, counted from the per-split rows
EQ = {}
for m in ["LR", "RF", "XGB", "LGBM", "MLP"]:
    c01 = c005 = 0
    for rid in ["P4-046", "P4-047", "P4-048", "P4-049", "P4-050"]:
        seg = [s for s in REG[rid]["value"].split("; ") if s.startswith(m + " ")][0]
        flags = seg.split()[-1].split("/")  # Y/Y/Y = 0.005 / 0.01 / 0.02
        c005 += flags[0] == "Y"; c01 += flags[1] == "Y"
        TRACE.append((rid, "value", seg, "equivalence flags counted"))
    EQ[m] = (c01, c005)
FAM = {"LR": "linear", "RF": "bagging", "XGB": "boosting", "LGBM": "boosting", "CAT": "boosting", "MLP": "neural"}
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Test performance over five outer splits (mean $\pm$ SD, ddof 1) and near-equivalence to CatBoost}",
     r"\label{tab:perf}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
     r"Model & Family & AP & ROC-AUC & $\Delta$AP vs CAT (SD) & NB TOST $p$ ($\delta$ 0.01)$^\ddagger$ & Equiv.\ splits$^\dagger$ \\", r"\midrule"]
for m in ["LR", "RF", "XGB", "LGBM", "CAT", "MLP"]:
    rid, ap, auc = P[m]
    apc, aucc = pm(rid, "AP " + ap)[3:], pm(rid, "ROC-AUC " + auc)[8:]
    if m == "CAT":
        apc, aucc = r"\textbf{" + apc + "}", r"\textbf{" + aucc + "}"
        L.append(f"CAT & {FAM[m]} & {apc} & {aucc} & reference & -- & -- \\\\")
    else:
        drid, d, p, pshow = D[m]
        dd = v(drid, d, neg(d)); pp = v(drid, p, pshow)
        L.append(f"{m} & {FAM[m]} & {apc} & {aucc} & {dd} & {pp} & {EQ[m][0]}/5 / {EQ[m][1]}/5 \\\\")
L += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Baselines on the same splits (published settings); accuracy and F1 at 0.50 / Youden}} \\",
      r"Model & & AP & ROC-AUC & Accuracy & F1 & \\", r"\midrule"]
# D42 amendment cut (g): ALT_LR and ALT_DT rows moved to one footnote sentence (values checked below)
ALTLD = (f"ALT\\_LR / ALT\\_DT~\\cite{{altamimi2026}} (not shown): AP {pm('P4-035', '0.4069 ± 0.0062')} / {pm('P4-037', '0.4070 ± 0.0060')}, "
         f"ROC-AUC {pm('P4-035', '0.8238 ± 0.0030')} / {pm('P4-037', '0.8181 ± 0.0034')}.")
B = [("ALT\\_RF~\\cite{altamimi2026}", "P4-039", "P4-040", "0.4332 ± 0.0056", "0.8280 ± 0.0026", "0.8665", "0.7184", "0.1810", "0.4398"),
     ("ALT\\_XGB~\\cite{altamimi2026}", "P4-041", "P4-042", "0.4397 ± 0.0068", "0.8320 ± 0.0029", "0.8679", "0.7193", "0.2547", "0.4436")]
for name, r50, ry, ap, auc, a50, ay, f50, fy in B:
    L.append(f"{name} & & {pm(r50, ap)} & {pm(r50, auc)} & {v(r50, a50)} / {v(ry, ay)} & {v(r50, f50)} / {v(ry, fy)} & \\\\")
L.append(f"Ahmed et al.~\\cite{{ahmed2025}} (default LR) & & {pm('P4-043', '0.4085 ± 0.0060')} & {pm('P4-043', '0.8232 ± 0.0029')} & "
         f"{v('P4-043', '0.8641')} / -- & {v('P4-043', '0.2396')} / -- & \\\\")
L.append(f"Majority class & & {v('P4-044', '0.1393')} & {v('P4-044', '0.5000')} & {v('P4-044', '0.8607')} / -- & 0 / -- & \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
      rf"\parbox{{\textwidth}}{{\scriptsize Bold: highest mean, not a significant difference. $\Delta$AP: mean difference to CAT over splits (SD). $^\ddagger$Nadeau--Bengio TOST (4 df), unadjusted as pre-stated; Holm (five candidates): RF {HOLM_RF}, MLP {HOLM_MLP}. $^\dagger$Splits equivalent at $\delta=0.01$ / $0.005$ (per-split Holm-adjusted bootstrap TOST, $B=2{{,}}000$); $\delta=0.01$ set after the split-1 results. Accuracy: baselines only; ALT\_* thresholds 0.50 / Youden (out-of-fold). {ALTLD}}}",
      r"\end{table*}"]
open(os.path.join(out, "table2_performance.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Table III: agreement and stability
H2c = [("split 1 0.994 / 0.945", "0.994", "0.945"), ("split 2 0.964 / 0.948", "0.964", "0.948"), ("split 3 0.948 / 0.960", "0.948", "0.960"),
       ("split 4 0.996 / 0.972", "0.996", "0.972"), ("split 5 0.966 / 0.944", "0.966", "0.944")]
c = [v("P5-006", s, f"{a} / {b}") for s, a, b in H2c]
cm = [("split 1 0.987 / 0.911 / 0.955", "0.4542"), ("split 2 0.915 / 0.938 / 0.925", "0.2381"), ("split 3 0.984 / 0.767 / 0.885", "0.6587"),
      ("split 4 0.993 / 0.935 / 0.959", "0.5635"), ("split 5 0.994 / 0.938 / 0.923", "0.0079")]
cmw = [v("P6-004", s, s.split(" ", 2)[2]) for s, _ in cm]
cmp_ = [v("P6-004", "p " + p, ("\\textbf{" + p + "}" if float(p) < 0.05 else p)) for _, p in cm]
RF = {"CAT": "CAT 0.013 / 0.085 / 0.016 / 0.007 / 0.006", "XGB": "XGB 0.023 / 0.024 / 0.012 / 0.011 / 0.012",
      "LGBM": "LGBM 0.021 / 0.061 / 0.017 / 0.014 / 0.009", "MLP": "MLP 0.089 / 0.062 / 0.233 / 0.065 / 0.062"}
PASS = [("CAT--MLP", "CAT-MLP", "passes 1/5", "passes 4/5"), ("XGB--MLP", "XGB-MLP", "passes 0/5", "passes 4/5"),
        ("LGBM--MLP", "LGBM-MLP", "passes 1/5", "passes 4/5")]
CON = [("CAT--XGB", "passes 2/5", "passes 0/5"), ("CAT--LGBM", "passes 0/5", "passes 1/5"), ("XGB--LGBM", "passes 2/5", "passes 0/5")]
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Agreement and stability of global SHAP rankings (extrapolated RBO, $p=0.8$, 21 predictors), splits 1--5. The common setting decides}",
     r"\label{tab:agree}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r" & Split 1 & Split 2 & Split 3 & Split 4 & Split 5 \\", r"\midrule",
     r"\multicolumn{6}{@{}l}{\emph{H2: mean RBO within boosting family / across families (CAT, XGB, LGBM, MLP); top-5 set identical in 5/5 splits}} \\",
     "Common & " + " & ".join(c) + r" \\", r"\midrule",
     r"\multicolumn{6}{@{}l}{\emph{H3 primary pair CAT--MLP, common: median RBO within CAT / within MLP / between refits}} \\",
     "Medians & " + " & ".join(cmw) + r" \\", "Permutation $p$ & " + " & ".join(cmp_) + r" \\", r"\midrule",
     ]
# D42 amendment cut (f): the four refit-disagreement rows (P6-008) are no longer in the table; their range is in the text
# (0.006-0.233, trace_claims.py d_refit_dis) and the within-refit medians are in Fig. 2.
# D42 amendment 2, O35 step 3a: "H3 splits passed" block removed; all pass counts are in the H3 text (trace_claims.py)
L[-1:] = []  # drop the \midrule that opened the removed block
# split-1 extras block cut from the table (outline overflow rule 3); seed-only values are in the text (P6-013, P6-014)
L += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
      r"\parbox{\textwidth}{\scriptsize No ``best'' value is marked: these are agreement measures, not scores. $T$ and the permutation test: Section~III-E; pass counts per pair: Section~IV-C.}",
      r"\end{table*}"]
open(os.path.join(out, "table3_agreement.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Table IV: ablations
L = [r"\begin{table}[!t]", r"\centering", r"\caption{Ablations and robustness (split 1 unless noted)}", r"\label{tab:abl}",
     r"\footnotesize", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r" & CAT & LR \\", r"\midrule",
     r"\multicolumn{3}{@{}l}{\emph{Leakage (ROC-AUC only; threshold-free)}} \\",
     f"No resampling & {v('P1-026', '0.8296')} & {v('P1-021', '0.8205')} \\\\",
     f"SMOTE in development only & {v('P1-027', '0.8264')} & {v('P1-022', '0.8188')} \\\\",
     f"Leaky SMOTE, resampled test$^a$ & {v('P1-028', '0.9701')} & {v('P1-023', '0.8304')} \\\\",
     f"\\quad same model, real test rows & {v('P1-032', '0.8268')} & {v('P1-031', '0.8183')} \\\\",
     r"\multicolumn{3}{@{}l}{\emph{Profile-grouped $-$ row-level split, 5 folds}} \\",
     f"$\\Delta$AP & {v('P1-035', 'paired diff -0.0002 ± 0.0124', '$-$0.0002$\\pm$0.0124')} & {v('P1-034', 'paired diff +0.0000 ± 0.0093', '+0.0000$\\pm$0.0093')} \\\\",
     # D42 amendment 2, O35 step 5: profile-grouped Delta ROC-AUC row and the "Explainer setting" block removed (margins in IV-B)
     ]
# D42 amendment 2, O35 step 8: Platt block folded into Results IV-E (values traced in trace_claims.py from P7-001-P7-004)
L += [r"\midrule", r"\multicolumn{3}{@{}l}{\emph{Prediabetes respondents (coded 0; exploratory)}} \\",
      f"ROC-AUC vs other negatives & {v('P3-009', '0.747 [0.732, 0.761]', '0.747')} & {v('P3-001', '0.740 [0.726, 0.755]', '0.740')} \\\\",
      f"Excess AP gain$^b$ & {v('P3-026', 'excess +0.0096', '+0.0096')} & {v('P3-018', 'excess +0.0086', '+0.0086')} \\\\",
      f"Refit without them, $\\Delta$AP & {v('P3-017', 'CAT -0.00007', '$-$0.00007')} & {v('P3-017', 'LR +0.00005', '+0.00005')} \\\\",
      r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
      r"\parbox{\columnwidth}{\scriptsize $^a$50\% prevalence; 84\% of test positives synthetic. $^b$AP gain from removing the 910 test prediabetes respondents minus that from removing 910 random negatives (200 draws). Profile-grouped: mean $\pm$ SD of paired differences.}",
      r"\end{table}"]
open(os.path.join(out, "table4_ablations.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

with open(os.path.join(out, "table_trace.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["registry_id", "registry_column", "checked_string", "comment"]); w.writerows(TRACE)
print(f"4 tables written; {len(TRACE)} cell values checked against the registry")
