"""P0 label checks (paper phase, D30/D33). usage: python -I p0_label_check.py BINARY_CSV THREECLASS_CSV OUT_CSV
Checks that the binary and 3-class BRFSS 2015 files are row-aligned, cross-tabulates their labels, and counts
the prediabetes respondents in the thesis development/test partition (partition reconstructed with the S2
partition seed; it reproduces S1-025, S1-027, S1-028 and S1-035)."""
import hashlib, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

b_path, t_path, out = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
b, t = pd.read_csv(b_path), pd.read_csv(t_path)
X = [c for c in b.columns if c != "Diabetes_binary"]
assert list(t.columns[1:]) == X
seeds = dict(zip("partition folds".split(), np.random.SeedSequence(42).generate_state(13)[:2]))
y = b["Diabetes_binary"].to_numpy().astype(int)
dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=int(seeds["partition"]))
key = b[X].astype(int).astype(str).agg("|".join, axis=1)
pre = t["Diabetes_012"].to_numpy() == 1
ct = pd.crosstab(t["Diabetes_012"], b["Diabetes_binary"])
rows = {
    "binary_sha256": sha(b_path), "threeclass_sha256": sha(t_path),
    "rows_binary": len(b), "rows_threeclass": len(t),
    "rows_identical_21_predictors_same_position": int((b[X].to_numpy() == t[X].to_numpy()).all(axis=1).sum()),
    "012=0 & bin=0": int(ct.loc[0.0, 0.0]), "012=1 & bin=0": int(ct.loc[1.0, 0.0]),
    "012=2 & bin=1": int(ct.loc[2.0, 1.0]), "off-diagonal": int(ct.to_numpy().sum() - ct.loc[0.0, 0.0] - ct.loc[1.0, 0.0] - ct.loc[2.0, 1.0]),
    "thesis_dev_rows": len(dev), "thesis_test_rows": len(test), "thesis_test_positives": int(y[test].sum()),
    "thesis_test_profile_in_dev": int(key.iloc[test].isin(set(key.iloc[dev])).sum()),
    "prediabetes_in_thesis_dev": int(pre[dev].sum()), "prediabetes_in_thesis_test": int(pre[test].sum()),
    "prediabetes_distinct_profiles": int(key[pre].nunique()),
    "prediabetes_profiles_also_in_diabetes_rows": len(set(key[pre]) & set(key[y == 1])),
    "prediabetes_profiles_also_in_nodiabetes_rows": len(set(key[pre]) & set(key[t["Diabetes_012"].to_numpy() == 0])),
}
pd.Series(rows).to_csv(out, header=["value"])
print(pd.Series(rows).to_string())
