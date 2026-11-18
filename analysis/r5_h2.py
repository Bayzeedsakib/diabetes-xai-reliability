"""H2 (paper Phase 5; phase4_design.md section 1 and section 12, decided 2026-11-13 before any split 2-5 SHAP existed).
Members (O33, option b): CAT, XGB, LGBM, MLP in every split. Within-family pairs CAT-XGB, CAT-LGBM, XGB-LGBM; cross-family
pairs CAT-MLP, XGB-MLP, LGBM-MLP. Decision setting: common (B2, both parts); native reported beside it.
Global ranking per model and split: mean |SHAP| over the split's 500-row core (normalized; the order is what matters),
ties broken by feature order (stable sort). RBO = extrapolated RBO on the full 21-predictor rankings, p = 0.8
(the R2 function, r2_analysis.py; sensitivity p = 0.9).
- Top-5 part: all six member pairs have the identical top-5 set; holds if true in >= 4 of 5 splits.
- RBO part: mean RBO over the 3 within-family pairs > mean RBO over the 3 cross-family pairs; holds in >= 4 of 5 splits.
- Descriptive: share of the 9 (within, cross) combinations with within > cross; Kendall tau-b; top-3 Jaccard.
- Approximation floor (section 12 item 9; split 1): RBO and top-5 Jaccard of XGB_perm_logodds vs XGB_tree100 and vs common XGB.
- Exploratory (labelled): all six models, families boosting / bagging (RF) / linear (LR) / neural (MLP).
usage: python -I r5_h2.py CODE_DIR R5_DIR OUTDIR"""
import itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau

code_dir, O, out = sys.argv[1:4]
sys.path.insert(0, code_dir); import thesis_common as tc
os.makedirs(out, exist_ok=True)
F = list(tc.FEATURES)
MEM = ["CAT", "XGB", "LGBM", "MLP"]
WITHIN = [("CAT", "XGB"), ("CAT", "LGBM"), ("XGB", "LGBM")]
CROSS = [("CAT", "MLP"), ("XGB", "MLP"), ("LGBM", "MLP")]
ALL6 = ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]
FAM = {"XGB": "boosting", "LGBM": "boosting", "CAT": "boosting", "RF": "bagging", "LR": "linear", "MLP": "neural"}


def rbo(a, b, p):                                   # identical to r2_analysis.py
    sa, sb, s = set(), set(), 0.0
    for d in range(1, len(a) + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1]); s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s + p ** len(a) * len(sa & sb) / len(a)


def load(k, s, m):
    rec = json.load(open(os.path.join(O, f"split{k}", s, f"SHAP_{m}.done.json")))
    df = pd.read_csv(os.path.join(O, rec["file"]), float_precision="round_trip")
    assert len(df) == 500
    return df


def imp(df):
    a = df[F].abs().mean().to_numpy()
    return a / a.sum()


def order(v):
    return np.argsort(-v, kind="stable")


def pair_stats(ia, ib):
    oa, ob = order(ia), order(ib)
    return {"rbo08": rbo(oa, ob, 0.8), "rbo09": rbo(oa, ob, 0.9), "tau_b": kendalltau(ia, ib).statistic,
            "top3_jaccard": len(set(oa[:3]) & set(ob[:3])) / len(set(oa[:3]) | set(ob[:3])),
            "top5_jaccard": len(set(oa[:5]) & set(ob[:5])) / len(set(oa[:5]) | set(ob[:5]))}


rows, imps, tops = [], [], []
for k in [1, 2, 3, 4, 5]:
    for s in ["common", "native"]:
        I = {m: imp(load(k, s, m)) for m in ALL6}
        for m in ALL6:
            imps += [{"split": k, "setting": s, "model": m, "feature": F[j], "norm_importance": I[m][j]} for j in range(len(F))]
            tops.append({"split": k, "setting": s, "model": m, **{f"top{j + 1}": F[i] for j, i in enumerate(order(I[m])[:5])}})
        for a, b in itertools.combinations(ALL6, 2):
            kind = ("within" if (a, b) in WITHIN or (b, a) in WITHIN else "cross" if (a, b) in CROSS or (b, a) in CROSS else
                    "exploratory")
            rows.append({"split": k, "setting": s, "model_a": a, "model_b": b, "pair_type": kind,
                         "same_family": FAM[a] == FAM[b], **pair_stats(I[a], I[b])})
P = pd.DataFrame(rows); P.to_csv(f"{out}/h2_pairs.csv", index=False, float_format="%.10g")
pd.DataFrame(imps).to_csv(f"{out}/h2_global_importance.csv", index=False, float_format="%.10g")
T = pd.DataFrame(tops); T.to_csv(f"{out}/h2_top5_sets.csv", index=False)

per = []
for (k, s), g in P.groupby(["split", "setting"]):
    w, c = g[g.pair_type == "within"], g[g.pair_type == "cross"]
    mem = g[g.pair_type.isin(["within", "cross"])]
    combos = [(x, y) for x in w.rbo08 for y in c.rbo08]
    # exploratory all-six: same family (the three booster pairs) vs different families (12 pairs)
    sf, df_ = g[g.same_family], g[~g.same_family]
    per.append({"split": k, "setting": s, "top5_identical_all_member_pairs": bool((mem.top5_jaccard == 1).all()),
                "member_top5_jaccard_min": mem.top5_jaccard.min(),
                "mean_rbo_within": w.rbo08.mean(), "mean_rbo_cross": c.rbo08.mean(), "rbo_part_holds": bool(w.rbo08.mean() > c.rbo08.mean()),
                "diff_within_minus_cross": w.rbo08.mean() - c.rbo08.mean(),
                "share_within_gt_cross_combos": np.mean([x > y for x, y in combos]),
                "mean_rbo09_within": w.rbo09.mean(), "mean_rbo09_cross": c.rbo09.mean(),
                "mean_tau_within": w.tau_b.mean(), "mean_tau_cross": c.tau_b.mean(),
                "expl_all6_mean_rbo_same_family": sf.rbo08.mean(), "expl_all6_mean_rbo_diff_family": df_.rbo08.mean(),
                "expl_all6_top5_identical_all_15_pairs": bool((g.top5_jaccard == 1).all())})
S = pd.DataFrame(per).sort_values(["setting", "split"]); S.to_csv(f"{out}/h2_per_split.csv", index=False, float_format="%.10g")

res = {}
for s in ["common", "native"]:
    x = S[S.setting == s]
    res[s] = {"top5_part_splits": int(x.top5_identical_all_member_pairs.sum()), "top5_part_holds": bool(x.top5_identical_all_member_pairs.sum() >= 4),
              "rbo_part_splits": int(x.rbo_part_holds.sum()), "rbo_part_holds": bool(x.rbo_part_holds.sum() >= 4)}
res["decision"] = {"setting": "common", **res["common"],
                   "H2_supported_both_parts": res["common"]["top5_part_holds"] and res["common"]["rbo_part_holds"]}
# approximation floor (split 1)
fl = {}
xt = imp(load(1, "xgbcheck", "XGB_tree100")); xp = imp(load(1, "xgbcheck", "XGB_perm_logodds")); xc = imp(load(1, "common", "XGB"))
fl["perm_logodds_vs_tree100"] = pair_stats(xp, xt)
fl["perm_logodds_vs_common_XGB"] = pair_stats(xp, xc)
fl["tree100_vs_native_XGB_500bg"] = pair_stats(xt, imp(load(1, "native", "XGB")))
res["approximation_floor_split1"] = fl
res["alt_xgb_top5"] = pd.read_csv(f"{O}/alt_xgb_top5.csv").to_dict("records")
json.dump(res, open(f"{out}/h2_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(S[["split", "setting", "top5_identical_all_member_pairs", "member_top5_jaccard_min", "mean_rbo_within", "mean_rbo_cross",
         "rbo_part_holds", "share_within_gt_cross_combos", "expl_all6_mean_rbo_same_family", "expl_all6_mean_rbo_diff_family"]].round(4).to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k != "alt_xgb_top5"}, indent=1, default=float))
