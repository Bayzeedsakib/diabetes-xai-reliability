# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""R2 input checks (student item 11): frozen pipelines vs test_predictions.csv, and workspace refits
(CatBoost thread_count = 12, D35) vs test_predictions.csv.
usage: python -I r2_input_check.py CODE_DIR BINARY_CSV TEST_PREDICTIONS_CSV FROZEN_JSON PIPE_DIR OUTDIR"""
import json, os, sys, warnings
import joblib, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import thesis_common as tc
from sklearn.metrics import average_precision_score as AP
b, p, fz, pdir, out = sys.argv[2:7]; os.makedirs(out, exist_ok=True)
assert tc.sha256_file(b) == tc.DATA_SHA256
assert tc.sha256_file(p) == "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"
assert tc.sha256_file(fz) == "1dd5bf06708d148399675d9fc657857df1c86f015cbd83c24e79e8754b48c0d8"
FZ = json.load(open(fz, encoding="utf-8"))
df = pd.read_csv(b); X, y = df[tc.FEATURES], df[tc.TARGET].astype(int).to_numpy()
tp = pd.read_csv(p, float_precision="round_trip"); test = tp.row_id.to_numpy()
dev = np.setdiff1d(np.arange(len(y)), test)
CFG = {"LR": 20, "RF": 6, "XGB": 2, "LGBM": 19, "CAT": 7, "MLP": 21}
rows = []
for m in CFG:
    path = os.path.join(pdir, m, f"pipeline_{m}.joblib")
    assert tc.sha256_file(path) == FZ["models"][m]["pipeline_sha256"], m
    pipe = joblib.load(path)
    if m == "RF": pipe.set_params(clf__n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = pipe.predict_proba(X.iloc[test])[:, 1]
    rows.append({"model": m, "check": "frozen pipeline predict", "max_abs_diff": float(np.abs(pf - tp["p_" + m]).max()),
                 "AP": AP(y[test], pf)})
    if m != "XGB":
        pr = tc.make_pipeline(m, n_jobs=12 if m == "CAT" else -1)
        pr.set_params(**tc.sampled_configs(m)[CFG[m]])
        # params equal the freeze
        for k, v in FZ["models"][m]["params_repr"].items():
            assert repr(pr.get_params()[k]) == v or str(pr.get_params()[k]) == v, (m, k, v, pr.get_params()[k])
        pr.fit(X.iloc[dev], y[dev])
        if m == "RF": pr.set_params(clf__n_jobs=1)
        q = pr.predict_proba(X.iloc[test])[:, 1]
        rows.append({"model": m, "check": "workspace refit" + (" (thread_count=12)" if m == "CAT" else ""),
                     "max_abs_diff": float(np.abs(q - tp["p_" + m]).max()), "AP": AP(y[test], q)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "r2_input_check.csv"), index=False)
