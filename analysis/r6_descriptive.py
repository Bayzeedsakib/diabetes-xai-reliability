# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""R6 descriptive summaries (no decision): test AP of the refits per split x model x kind (boot, seed, rash),
MLP spread vs the boosters, MLP vs CAT bootstrap means, seed-only identity check vs R5 models, Rashomon counts
from the split-1 Stage 3 cv_results. usage: python -I r6_descriptive.py PROJECT_ROOT CV_DIR OUTDIR   (PROJECT_ROOT = repository root; CV_DIR = stage3_outputs/models)"""
import json, os, sys
import numpy as np, pandas as pd
root, cvd, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
R = pd.read_csv(os.path.join(root, "paper_r6_outputs", "refits_r6.csv"))
assert len(R) == 219
g = R.groupby(["split", "model", "kind"]).test_ap
S = pd.DataFrame({"n": g.size(), "ap_mean": g.mean(), "ap_sd": g.std(ddof=1), "ap_min": g.min(), "ap_max": g.max(),
                  "ap_range": g.max() - g.min()}).reset_index()
g2 = R.groupby(["split", "model", "kind"]).test_roc_auc
S["roc_mean"] = g2.mean().values; S["roc_sd"] = g2.std(ddof=1).values
S.to_csv(f"{out}/r6_refit_ap_summary.csv", index=False, float_format="%.6g")
b = S[S.kind == "boot"].pivot(index="split", columns="model", values="ap_mean")
sd = S[S.kind == "boot"].pivot(index="split", columns="model", values="ap_sd")
cmp = pd.DataFrame({"MLP_boot_mean": b.MLP, "CAT_boot_mean": b.CAT, "MLP_minus_CAT": b.MLP - b.CAT,
                    "MLP_sd": sd.MLP, "CAT_sd": sd.CAT, "XGB_sd": sd.XGB, "LGBM_sd": sd.LGBM,
                    "MLP_sd_over_max_booster_sd": sd.MLP / sd[["CAT", "XGB", "LGBM"]].max(axis=1)}).reset_index()
cmp.to_csv(f"{out}/r6_mlp_vs_boosters.csv", index=False, float_format="%.6g")
# Rashomon: configurations within 0.005 of the best mean CV AP (split 1, Stage 3)
rows = []
units = json.load(open(os.path.join(root, "paper_r6_outputs", "seeds_r6.json")))["units"]
for m in ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]:
    c = pd.read_csv(os.path.join(cvd, m, f"cv_results_{m}.csv"))
    best = c.mean_test_ap.max(); gap = best - c.mean_test_ap
    within = gap <= 0.005
    used = sorted(u["config_id"] for u in units if u["kind"] == "rash" and u["model"] == m)
    rows.append({"model": m, "n_configs": len(c), "n_within_0.005": int(within.sum()), "capped_at_10": bool(within.sum() > 10),
                 "n_gap_lt_5e-5": int((gap < 5e-5).sum()), "max_gap_in_set_of_10": float(np.sort(gap)[9]),
                 "rashomon_refits_r6": len(used), "rashomon_config_ids": " ".join(map(str, used))})
pd.DataFrame(rows).to_csv(f"{out}/r6_rashomon_counts.csv", index=False, float_format="%.6g")
pd.set_option("display.width", 250)
print(S.round(4).to_string(index=False)); print(cmp.round(4).to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False))
