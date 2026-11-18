"""R3 (paper Phase 5): exploratory prediabetes analyses E2 with the frozen thesis predictions (split 1).
usage: python -I r3_prediabetes.py BINARY_CSV THREECLASS_CSV TEST_PREDICTIONS_CSV OUTDIR
No model is fitted. test_predictions.csv must have SHA-256 b9813461...ed (S5-136)."""
import hashlib, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.model_selection import train_test_split

b_path, t_path, p_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
assert hashlib.sha256(open(p_path, "rb").read()).hexdigest().startswith("b9813461cd03e455")
b, t, tp = pd.read_csv(b_path), pd.read_csv(t_path), pd.read_csv(p_path)
y_all = b["Diabetes_binary"].to_numpy().astype(int)
seed = int(np.random.SeedSequence(42).generate_state(13)[0])
dev, test = train_test_split(np.arange(len(y_all)), test_size=0.2, stratify=y_all, random_state=seed)
assert np.array_equal(np.sort(test), np.sort(tp["row_id"].to_numpy())), "partition reconstruction differs"
assert np.array_equal(tp["y"].to_numpy(), y_all[tp["row_id"].to_numpy()])
code = t["Diabetes_012"].to_numpy().astype(int)[tp["row_id"].to_numpy()]
models = [c[2:] for c in tp.columns if c.startswith("p_")]
neg = tp["y"].to_numpy() == 0
pre = code == 1
rng = np.random.default_rng([2705014006, 3])           # eval_bootstrap seed, new sub-stream for R3
B = 2000
i_pre, i_no = np.where(pre)[0], np.where(neg & ~pre)[0]
rows = []
for m in models:
    p = tp["p_" + m].to_numpy()
    # (1) within test negatives: does the score separate reported prediabetes from no diabetes?
    s = np.r_[p[i_pre], p[i_no]]; lab = np.r_[np.ones(len(i_pre)), np.zeros(len(i_no))]
    a = auc(lab, s)
    boot = []
    for _ in range(B):
        bp, bn = rng.choice(i_pre, len(i_pre)), rng.choice(i_no, len(i_no))
        boot.append(auc(np.r_[np.ones(len(bp)), np.zeros(len(bn))], np.r_[p[bp], p[bn]]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pct = pd.Series(p).rank(pct=True).to_numpy()        # percentile among all test rows
    # (2) test discrimination with the 910 prediabetes respondents removed from the negatives
    keep = ~pre
    rows.append({"model": m, "auc_prediab_vs_nodiab": a, "auc_ci_lo": lo, "auc_ci_hi": hi,
                 "median_p_nodiab": np.median(p[i_no]), "median_p_prediab": np.median(p[i_pre]),
                 "median_p_diabetes": np.median(p[tp["y"].to_numpy() == 1]),
                 "median_pct_prediab": np.median(pct[i_pre]),
                 "ap_full": ap(tp["y"], p), "ap_without_prediab": ap(tp["y"][keep], p[keep]),
                 "rocauc_full": auc(tp["y"], p), "rocauc_without_prediab": auc(tp["y"][keep], p[keep])})
res = pd.DataFrame(rows)
res["d_ap"] = res.ap_without_prediab - res.ap_full
res["d_rocauc"] = res.rocauc_without_prediab - res.rocauc_full
res.to_csv(os.path.join(out, "r3_prediabetes.csv"), index=False, float_format="%.6f")
pd.Series({"test_rows": len(tp), "test_negatives": int(neg.sum()), "prediab_in_test": int(pre.sum()),
           "nodiab_in_test": len(i_no), "prediab_also_positive": int((pre & ~neg).sum()),
           "prevalence_without_prediab": tp["y"][~pre].mean()}).to_csv(os.path.join(out, "r3_counts.csv"), header=["value"])
print(res.round(4).to_string())
