"""Independent refit check of R4 split 2 in the workspace (D37 amendment A9): LR, LGBM and RF are refitted from the
selected configurations on the split-2 development rows and compared with the PC test_predictions.csv (all 50,736
test rows). XGBoost and CatBoost are not refitted here (machine-dependent, P1-007/P1-008).
usage: python -I r4_split2_refit_check.py CODE_DIR BINARY_CSV R4_OUT_DIR OUTCSV"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import thesis_common as tc
from sklearn.metrics import average_precision_score as AP
b_path, O, out = sys.argv[2:5]
assert tc.sha256_file(b_path) == tc.DATA_SHA256
df = pd.read_csv(b_path); X, y = df[tc.FEATURES], df[tc.TARGET].astype(int).to_numpy()
dev = pd.read_csv(f"{O}/indices/split2/dev_folds.csv").row_id.to_numpy()
test = pd.read_csv(f"{O}/indices/split2/test_rows.csv").row_id.to_numpy()
rows = []
for m in ["LR", "LGBM", "RF"]:
    sm = json.load(open(f"{O}/split2/{m}/summary.json")); cid = sm["config_id"]
    tp = pd.read_csv(f"{O}/split2/{m}/test_predictions.csv", float_precision="round_trip")
    assert np.array_equal(tp.row_id.to_numpy(), test)
    t0 = time.time()
    pipe = tc.make_pipeline(m).set_params(**tc.sampled_configs(m)[cid]).fit(X.iloc[dev], y[dev])
    if m == "RF":
        pipe.set_params(clf__n_jobs=1)
    p = pipe.predict_proba(X.iloc[test])[:, 1]
    d = np.abs(p - tp.p.to_numpy())
    rows.append({"split": 2, "model": m, "config_id": cid, "max_abs_diff": float(d.max()), "rows_differing": int((d > 0).sum()),
                 "AP_workspace": AP(y[test], p), "AP_pc": AP(y[test], tp.p), "fit_s": round(time.time() - t0, 1)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g", lineterminator="\n")
