"""R4 index-file check (D37 amendment A8): every split 2-5 index file against MANIFEST_splits2_3_4_5.csv, and an
independent rebuild of the 50k tuning subsample and the 100-row permutation background (not covered by r4_verify.py),
plus the R4 zips available here against SHA256SUMS_r4.txt.  Output: r4_rebuild_check.csv (one row per check).
usage: python -I r4_rebuild_check.py CODE_DIR BINARY_CSV R4_OUT_DIR SUMS_TXT ZIP_DIR OUTCSV"""
import hashlib, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import thesis_common as tc
from sklearn.model_selection import StratifiedKFold, train_test_split
b_path, O, sums, zdir, out = sys.argv[2:7]
assert tc.sha256_file(b_path) == tc.DATA_SHA256
y = pd.read_csv(b_path)[tc.TARGET].astype(int).to_numpy()
po = int(np.random.SeedSequence(42).generate_state(15)[14])
NAMES = ["partition", "folds", "subsample_50k", "shap_background", "shap_perm_background", "test_core", "eval_bootstrap"]
rows = []
def add(kind, split, item, ok, detail=""):
    rows.append({"kind": kind, "split": split, "item": item, "pass": bool(ok), "detail": detail})
man = pd.read_csv(f"{O}/indices/MANIFEST_splits2_3_4_5.csv")
for r in man.itertuples():
    h = tc.sha256_file(f"{O}/indices/split{r.split}/{r.file}")
    add("index file vs MANIFEST", r.split, r.file, h == r.sha256, h[:16])
for k in [2, 3, 4, 5]:
    s = dict(zip(NAMES, map(int, np.random.SeedSequence([po, k]).generate_state(7))))
    dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s["partition"])
    dev = np.sort(dev)
    fold = np.empty(len(dev), int)
    for j, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s["folds"]).split(dev, y[dev])):
        fold[v] = j
    sub = np.sort(train_test_split(np.arange(len(dev)), train_size=50000, stratify=y[dev] * 5 + fold,
                                   random_state=s["subsample_50k"])[0])
    blob = pd.DataFrame({"row_id": dev[sub], "fold": fold[sub], "y": y[dev[sub]]}).to_csv(index=False, lineterminator="\n").encode()
    f50 = open(f"{O}/indices/split{k}/dev_subsample_50k.csv", "rb").read()
    add("rebuild", k, "dev_subsample_50k.csv", blob == f50, f"{len(sub)} rows, {int(y[dev[sub]].sum())} positive")
    bg = np.sort(dev[train_test_split(np.arange(len(dev)), train_size=500, stratify=y[dev], random_state=s["shap_background"])[0]])
    bg100 = np.sort(bg[train_test_split(np.arange(500), train_size=100, stratify=y[bg], random_state=s["shap_perm_background"])[0]])
    blob = pd.DataFrame({"row_id": bg100, "y": y[bg100]}).to_csv(index=False, lineterminator="\n").encode()
    add("rebuild", k, "shap_perm_background_100_dev.csv", blob == open(f"{O}/indices/split{k}/shap_perm_background_100_dev.csv", "rb").read(),
        f"{len(bg100)} rows, {int(y[bg100].sum())} positive; subset of the 500 background")
listed = dict(l.split("  ", 1)[::-1] for l in open(sums, encoding="utf-8").read().splitlines())
for z in sorted(f for f in os.listdir(zdir) if f.startswith("r4_") and f.endswith(".zip") and f in listed):
    h = hashlib.sha256(open(os.path.join(zdir, z), "rb").read()).hexdigest()
    add("zip vs SHA256SUMS_r4.txt", "-", z, h == listed[z], h[:16])
R = pd.DataFrame(rows); R.to_csv(out, index=False, lineterminator="\n")
print(R.groupby("kind")["pass"].agg(["sum", "count"]).to_string())
print(R[R.kind != "index file vs MANIFEST"].to_string(index=False))
