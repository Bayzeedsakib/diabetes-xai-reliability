"""R2 follow-up (D36): (2b) post-hoc exploratory CholCheck ratio by the case's own value; (2c) LIME share change
under 50/50 sampling; (3) H4 sensitivity with variances (unbiased under the null) instead of SDs.
usage: python -I r2_followup.py OUT_R2_DIR"""
import json, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
O = sys.argv[1]
H = pd.read_csv(f"{O}/analysis/r2_cholcheck_shares.csv")
res = {"ratio_by_cholcheck_value": {}, "ratio_by_model_cholcheck1": {}}
H["l_nat"] = np.log10(H.lime_T5 / H.shap_native); H["l_a"] = np.log10(H.lime_rescaled_a / H.shap_native)
for v in [1.0, 0.0]:
    d = H[H.CholCheck == v]
    res["ratio_by_cholcheck_value"][int(v)] = {"pairs": len(d), "median_log10_unrescaled": d.l_nat.median(), "ratio_unrescaled": 10 ** d.l_nat.median(),
        "median_log10_rescaled": d.l_a.median(), "ratio_rescaled": 10 ** d.l_a.median(),
        "spearman_rescaled_vs_shap": spearmanr(d.lime_rescaled_a, d.shap_native).statistic,
        "spearman_unrescaled_vs_shap": spearmanr(d.lime_T5, d.shap_native).statistic}
r1, r0 = res["ratio_by_cholcheck_value"][1], res["ratio_by_cholcheck_value"][0]
res["swing_unrescaled_fold"] = r1["ratio_unrescaled"] / r0["ratio_unrescaled"]
res["swing_rescaled_fold"] = r1["ratio_rescaled"] / r0["ratio_rescaled"]
for m, d in H[H.CholCheck == 1].groupby("model"):
    res["ratio_by_model_cholcheck1"][m] = {"unrescaled": 10 ** d.l_nat.median(), "rescaled": 10 ** d.l_a.median()}
res["ratio_by_model_cholcheck0"] = {m: {"unrescaled": 10 ** d.l_nat.median(), "rescaled": 10 ** d.l_a.median(), "pairs": len(d)} for m, d in H[H.CholCheck == 0].groupby("model")}
# rule slack: largest rescaled median share that would still pass gap_a <= 0.5 gap_native
R = json.load(open(f"{O}/analysis/r2_results.json"))["H5"]["all"]
res["rule_pass_max_rescaled_median_share"] = R["median_share"]["shap_native"] + 0.5 * R["gap_native"]
res["share_of_pairs_CholCheck1"] = float((H.CholCheck == 1).mean())
# (2c) LIME share 50/50 sampling vs T5
r = H.lime_chol50_c / H.lime_T5
res["chol50_lime_ratio"] = {"pairs_increase": int((r > 1).sum()), "pairs": len(r), "median_ratio": r.median(), "min_ratio": r.min()}
# (3) H4 with variances
P = pd.read_csv(f"{O}/analysis/r2_agreement_per_seed.csv")
GRID = ["T5", "T10"] + [f"N_{d}_{k}" for d in ["quartile", "decile", "none"] for k in ["0.5", "1", "2"]]
def var_test(settings, stat="rbo08", model=None):
    Q = P[P.setting.isin(settings)]
    if model: Q = Q[Q.model == model]
    vs = Q.groupby(["model", "row_id", "seed_index"])[stat].var(ddof=1).groupby(["model", "row_id"]).mean()
    vd = Q.groupby(["model", "row_id", "setting"])[stat].var(ddof=1).groupby(["model", "row_id"]).mean()
    d = (vs - vd).dropna(); k, n = int((d > 0).sum()), int((d != 0).sum())
    return {"pairs": len(d), "pairs_var_set_gt_var_seed": k, "share": k / len(d), "sign_test_p_one_sided": binomtest(k, n, 0.5, "greater").pvalue}
res["H4_var_11settings"] = var_test(GRID)
res["H4_var_9new"] = var_test(GRID[2:])
res["H4_var_11settings_by_model"] = {m: var_test(GRID, model=m)["share"] for m in sorted(P.model.unique())}
res["H4_var_9new_by_model"] = {m: var_test(GRID[2:], model=m)["share"] for m in sorted(P.model.unique())}
json.dump(res, open(f"{O}/analysis/r2_followup.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
