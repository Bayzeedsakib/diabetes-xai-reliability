"""Part A checks of D37 (2026-11-04): recompute the reporting points from the returned files.
usage: python -I a_checks.py BOOT_DIR H1_PER_SPLIT_CSV NB_CSV MODELS_SUMMARY BASELINES_SUMMARY TEST_PREDICTIONS OUT_JSON"""
import hashlib, json, sys
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

boot, h1p, nbp, msum, bsum, tpp, out = sys.argv[1:8]
B = 2000
MODELS = ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]
R = pd.read_csv(h1p)
res = {}

# 1. every (split, model, delta) where the 90% CI lies inside (-delta, delta) but Holm-adjusted TOST p >= 0.05,
#    recomputed from the bootstrap AP matrices (not from h1_per_split.csv)
mismatch, maxdiff = [], 0.0
for k in range(1, 6):
    BA = pd.read_csv(f"{boot}/bootstrap_ap_split{k}.csv", float_precision="round_trip")
    top = R[R.split == k].top.iloc[0]
    for dl in [0.005, 0.01, 0.02]:
        raw, inside, ci = {}, {}, {}
        for m in [x for x in MODELS if x != top]:
            d = (BA[m] - BA[top]).to_numpy()
            lo, hi = np.percentile(d, 5), np.percentile(d, 95)
            raw[m] = max(((d <= -dl).sum() + 1) / (B + 1), ((d >= dl).sum() + 1) / (B + 1))
            inside[m], ci[m] = (-dl < lo and hi < dl), (lo, hi)
            r = R[(R.split == k) & (R.model == m)].iloc[0]
            maxdiff = max(maxdiff, abs(raw[m] - r[f"p_tost_{dl}"]), abs(lo - r.ci90_lo), abs(hi - r.ci90_hi))
        ms = list(raw)
        holm = dict(zip(ms, multipletests([raw[m] for m in ms], method="holm")[1]))
        for m in ms:
            if inside[m] != (holm[m] < 0.05):
                mismatch.append({"split": k, "model": m, "delta": dl, "ci90": [round(ci[m][0], 4), round(ci[m][1], 4)],
                                 "raw_tost_p": round(raw[m], 4), "holm_p": round(holm[m], 4),
                                 "ci_inside": inside[m], "equivalent_by_holm": bool(holm[m] < 0.05)})
res["ci_inside_vs_holm_mismatches"] = mismatch
res["max_abs_diff_vs_h1_per_split"] = maxdiff

# 2. Nadeau-Bengio: Holm across the 5 candidates
NB = pd.read_csv(nbp)
for dl in ["0.005", "0.01", "0.02"]:
    res[f"nb_tost_{dl}"] = {m: {"unadjusted": float(p), "holm": float(h)} for m, p, h in
                            zip(NB.model, NB[f"p_tost_nb_{dl}"], multipletests(NB[f"p_tost_nb_{dl}"], method="holm")[1])}

# 3. baseline accuracy at 0.50 (unweighted baselines), per-split range and means
Bs = pd.read_csv(bsum)
u = Bs[(Bs.thr_rule == "0.50") & Bs.baseline.isin(["ALT_DT", "ALT_RF", "ALT_XGB", "AHMED_LR_default"])]
res["baseline_acc050_per_split_min"] = u.loc[u.accuracy.idxmin(), ["split", "baseline", "accuracy"]].to_dict()
res["baseline_acc050_per_split_max"] = u.loc[u.accuracy.idxmax(), ["split", "baseline", "accuracy"]].to_dict()
res["baseline_acc050_means"] = u.groupby("baseline").accuracy.mean().to_dict()

# 4. ALT_XGB - CAT (and - XGB) test AP per split
S = pd.read_csv(msum)
ax = Bs[(Bs.baseline == "ALT_XGB") & (Bs.thr_rule == "0.50")].set_index("split").AP
cat = S[S.model == "CAT"].set_index("split").AP
xgb = S[S.model == "XGB"].set_index("split").AP
res["altxgb_minus_cat"] = {int(k): float(ax[k] - cat[k]) for k in ax.index}
res["altxgb_minus_xgb"] = {int(k): float(ax[k] - xgb[k]) for k in ax.index}
res["altxgb_minus_cat_mean"] = float((ax - cat).mean())
res["altxgb_split1_AP_pc"] = float(ax[1])

# 5. hash of the split-1 test predictions used by r4_h1.py
h = hashlib.sha256(open(tpp, "rb").read()).hexdigest()
res["test_predictions_sha256"] = h
res["test_predictions_hash_ok"] = h == "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"
json.dump(res, open(out, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
