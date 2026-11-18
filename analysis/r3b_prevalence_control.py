"""R3b (paper Phase 5, D35 item 6, exploratory): how much of the AP rise after removing the 910 test prediabetes
respondents (P3-001..010) is prevalence alone? Control: remove 910 randomly chosen test negatives WITHOUT
prediabetes (200 draws, no fitting, frozen thesis predictions) and record the mean AP / ROC-AUC change.
Excess = observed change - mean control change (the part due to prediabetes respondents scoring high).
usage: python -I r3b_prevalence_control.py BINARY_CSV THREECLASS_CSV TEST_PREDICTIONS_CSV OUTDIR"""
import hashlib, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

b_path, t_path, p_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
assert hashlib.sha256(open(p_path, "rb").read()).hexdigest() == "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"
tp = pd.read_csv(p_path, float_precision="round_trip")
code = pd.read_csv(t_path)["Diabetes_012"].to_numpy().astype(int)[tp.row_id.to_numpy()]
y = tp.y.to_numpy()
pre = code == 1
nodiab_neg = np.where((y == 0) & ~pre)[0]
assert pre.sum() == 910 and len(nodiab_neg) == 42757
rng = np.random.default_rng([2705014006, 4])          # eval_bootstrap seed, new sub-stream (R3 used [.., 3])
B = 200
draws = [np.sort(rng.choice(nodiab_neg, 910, replace=False)) for _ in range(B)]
models = [c[2:] for c in tp.columns if c.startswith("p_")]
rows = []
for m in models:
    p = tp["p_" + m].to_numpy()
    ap0, auc0 = ap(y, p), auc(y, p)
    keep = ~pre
    d_ap_obs, d_auc_obs = ap(y[keep], p[keep]) - ap0, auc(y[keep], p[keep]) - auc0
    cap, cauc = [], []
    for dr in draws:
        k = np.ones(len(y), bool); k[dr] = False
        cap.append(ap(y[k], p[k]) - ap0); cauc.append(auc(y[k], p[k]) - auc0)
    cap, cauc = np.array(cap), np.array(cauc)
    rows.append({"model": m, "d_ap_observed": d_ap_obs, "d_ap_control_mean": cap.mean(), "d_ap_control_sd": cap.std(ddof=1),
                 "d_ap_control_p2_5": np.percentile(cap, 2.5), "d_ap_control_p97_5": np.percentile(cap, 97.5),
                 "d_ap_excess": d_ap_obs - cap.mean(), "share_of_d_ap_from_prevalence": cap.mean() / d_ap_obs,
                 "d_rocauc_observed": d_auc_obs, "d_rocauc_control_mean": cauc.mean(), "d_rocauc_control_sd": cauc.std(ddof=1),
                 "d_rocauc_excess": d_auc_obs - cauc.mean(), "draws_with_d_ap_ge_observed": int((cap >= d_ap_obs).sum())})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(out, "r3b_prevalence_control.csv"), index=False, float_format="%.6f")
print(R.round(5).to_string(index=False))
