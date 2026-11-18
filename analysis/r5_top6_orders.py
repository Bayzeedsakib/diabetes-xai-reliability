"""P5-011 (descriptive): top-6 order of global importance (mean |SHAP| on the 500-row core) of each member,
per split and setting, and the RBO (p = 0.8) cost of one swap of adjacent predictors at ranks d, d+1.
Repository version: the computation is the P5-011 block of the workspace registry builder (build_registry_r5.py),
copied unchanged into a stand-alone script; it reads the r5_h2.py output h2_global_importance.csv.
usage: python -I r5_top6_orders.py H2_DIR"""
import sys
import numpy as np
import pandas as pd

H = sys.argv[1]
GI = pd.read_csv(f"{H}/h2_global_importance.csv")
AB = {"GenHlth": "GH", "BMI": "BMI", "Age": "Age", "HighBP": "BP", "HighChol": "Chol", "Sex": "Sex", "Income": "Inc"}
def top6(k, s, m):
    g = GI[(GI.split == k) & (GI.setting == s) & (GI.model == m)]
    o = np.argsort(-g.norm_importance.to_numpy(), kind="stable")[:6]
    return "-".join(AB.get(f, f) for f in g.feature.to_numpy()[o])
val = " | ".join(f"{s} split {k}: " + ", ".join(f"{m} {top6(k, s, m)}" for m in ["CAT", "XGB", "LGBM", "MLP"])
                 for s in ["native", "common"] for k in [1, 2, 3, 4, 5])
cost = {d: (1 - 0.8) * 0.8 ** (d - 1) / d for d in [2, 3, 4, 5]}
print(val + f" | swap cost ranks 2-3 {cost[2]:.3f}, 3-4 {cost[3]:.3f}, 4-5 {cost[4]:.3f}, 5-6 {cost[5]:.3f}")
