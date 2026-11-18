"""Rebuild the Stage 1 index files needed in the paper phase and verify each against MANIFEST.csv (SHA-256).
Procedure identified 2026-11-03: every stratified sample = sklearn train_test_split(pool, train_size=n,
stratify=y[pool], random_state=<stage seed>)[0], sorted; pools: development rows (500 background), the 500
background (100 permutation background), test rows (2,000 global sample), the global sample (500 core).
File format: columns row_id,y; '\\n' line ends. Files whose hash does not match are not written.
usage: python -I rebuild_indices_v2.py CODE_DIR BINARY_CSV TEST_PREDICTIONS_CSV MANIFEST_CSV OUTDIR"""
import hashlib, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import thesis_common as tc
from sklearn.model_selection import train_test_split, StratifiedKFold
b, p, mf, out = sys.argv[2:6]; os.makedirs(out, exist_ok=True)
assert tc.sha256_file(b) == tc.DATA_SHA256
assert tc.sha256_file(p) == "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"
y = pd.read_csv(b)[tc.TARGET].astype(int).to_numpy()
dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=tc.SEEDS["partition"])
dev, test = np.sort(dev), np.sort(test)
assert np.array_equal(test, np.sort(pd.read_csv(p, usecols=["row_id"]).row_id.to_numpy()))
fold = np.empty(len(dev), int)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=tc.SEEDS["folds"]).split(dev, y[dev])):
    fold[v] = k
samp = lambda pool, n, s: np.sort(train_test_split(pool, train_size=n, stratify=y[pool], random_state=tc.SEEDS[s])[0])
bg500 = samp(dev, 500, "shap_background"); bg100 = samp(bg500, 100, "shap_perm_background")
glob = samp(test, 2000, "global_sample"); core = samp(glob, 500, "perm_core")
man = pd.read_csv(mf).set_index("file").sha256
files = {"dev_folds.csv": pd.DataFrame({"row_id": dev, "fold": fold, "y": y[dev]}),
         "shap_background_500_dev.csv": pd.DataFrame({"row_id": bg500, "y": y[bg500]}),
         "shap_perm_background_100_dev.csv": pd.DataFrame({"row_id": bg100, "y": y[bg100]}),
         "shap_global_sample_2000_test.csv": pd.DataFrame({"row_id": glob, "y": y[glob]}),
         "shap_perm_core_500_test.csv": pd.DataFrame({"row_id": core, "y": y[core]})}
res = []
for f, d in files.items():
    blob = d.to_csv(index=False, lineterminator="\n").encode()
    if f == "dev_folds.csv" and hashlib.sha256(blob).hexdigest() != man[f]:   # alternative column layouts
        for alt in [d[["row_id", "fold"]], d.rename(columns={"y": tc.TARGET})]:
            b2 = alt.to_csv(index=False, lineterminator="\n").encode()
            if hashlib.sha256(b2).hexdigest() == man[f]: blob = b2
    h = hashlib.sha256(blob).hexdigest(); ok = h == man[f]
    res.append({"file": f, "rows": len(d), "sha256": h, "manifest": man[f], "match": ok})
    if ok:
        open(os.path.join(out, f), "wb").write(blob)
r = pd.DataFrame(res); r.to_csv(os.path.join(out, "index_check_workspace.csv"), index=False)
print(r[["file", "rows", "match"]].to_string(index=False))
