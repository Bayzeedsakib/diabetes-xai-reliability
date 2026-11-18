"""R1 (paper Phase 5, split 1 = thesis partition): reproduction check, baselines, leaky/profile ablations,
and the prediabetes refit row of R3. Workspace run on the canonical pins (D14: no timings reported).
Amended 2026-11-03 (D35): test_predictions.csv hash assert (item 10c); CatBoost pinned to thread_count = 12
(item 1; the 2026-11-03 R1 run used n_jobs = -1 = 2 threads, see P1-005/P1-007). The CatBoost rows of the
first run are superseded by r1b_cat12_profile5.py (P1-027-P1-029, P3-015 replaced).
usage: python -I r1_baselines_ablations.py CODE_DIR BINARY_CSV THREECLASS_CSV TEST_PREDICTIONS_CSV OUTDIR"""
import json, os, sys, time
import numpy as np, pandas as pd

code_dir, b_path, t_path, p_path, out = sys.argv[1:6]
sys.path.insert(0, code_dir)
import thesis_common as tc
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score as AP, roc_auc_score as AUC, f1_score, accuracy_score,
                             roc_curve, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

os.makedirs(out, exist_ok=True)
assert tc.sha256_file(b_path) == tc.DATA_SHA256
assert tc.sha256_file(p_path) == "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"   # S5-136 (D35 item 10c)
df = pd.read_csv(b_path); X = df[tc.FEATURES]; y = df[tc.TARGET].astype(int).to_numpy()
code012 = pd.read_csv(t_path)["Diabetes_012"].to_numpy().astype(int)
tp = pd.read_csv(p_path, float_precision="round_trip")
dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=tc.SEEDS["partition"])
dev, test = np.sort(dev), np.sort(test)
assert np.array_equal(test, np.sort(tp.row_id.to_numpy()))
tp = tp.set_index("row_id").loc[test]
fold = np.empty(len(dev), int)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=tc.SEEDS["folds"]).split(dev, y[dev])):
    fold[v] = k
Xd, yd, Xt, yt = X.iloc[dev], y[dev], X.iloc[test], y[test]
log = {"fold_sizes": np.bincount(fold).tolist(), "fold_pos": [int(yd[fold == k].sum()) for k in range(5)]}

def metrics(yy, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"AP": AP(yy, p), "ROC_AUC": AUC(yy, p), "accuracy": accuracy_score(yy, pred),
            "F1": f1_score(yy, pred, zero_division=0), "MCC": matthews_corrcoef(yy, pred),
            "sensitivity": pred[yy == 1].mean(), "specificity": 1 - pred[yy == 0].mean(), "threshold": thr}

# ---- 1. frozen thesis configurations (Stage 3 config ids, registry S3-013..S3-020), refit and reproduction check
CFG = {"LR": 20, "RF": 6, "XGB": 2, "LGBM": 19, "CAT": 7, "MLP": 21}
def thesis_pipe(m):
    pipe = tc.make_pipeline(m, n_jobs=12 if m == "CAT" else -1)   # D35 item 1: CatBoost thread_count = 12
    pipe.set_params(**tc.sampled_configs(m)[CFG[m]])
    return pipe
def predict(pipe, m, Xn):
    if m == "RF":
        pipe.named_steps["clf"].set_params(n_jobs=1)            # S12e: single-threaded RF prediction
    return pipe.predict_proba(Xn)[:, 1]
params = {m: {k: (repr(v) if isinstance(v, tuple) else v) for k, v in tc.sampled_configs(m)[CFG[m]].items()} for m in CFG}
json.dump(params, open(os.path.join(out, "thesis_params.json"), "w"), indent=1, default=str)
rep, fitted = [], {}
for m in CFG:
    t0 = time.time(); pipe = thesis_pipe(m).fit(Xd, yd); p = predict(pipe, m, Xt); fitted[m] = pipe
    d = np.abs(p - tp["p_" + m].to_numpy())
    rep.append({"model": m, "max_abs_diff": d.max(), "n_diff": int((d > 0).sum()), "AP_refit": AP(yt, p),
                "AP_frozen": AP(yt, tp["p_" + m]), "fit_s_workspace_not_reported": round(time.time() - t0, 1)})
    print(rep[-1], flush=True)
pd.DataFrame(rep).to_csv(os.path.join(out, "r1_reproduction.csv"), index=False)

# ---- 2. prediabetes refit row (R3 part): drop the development prediabetes respondents, same configurations
keep = code012[dev] != 1
pre_rows = []
nopre_test = code012[test] != 1
for m in CFG:
    pipe = thesis_pipe(m).fit(Xd[keep], yd[keep]); p = predict(pipe, m, Xt)
    pre_rows.append({"model": m, "dev_rows": int(keep.sum()), "AP_test": AP(yt, p), "ROC_AUC_test": AUC(yt, p),
                     "AP_test_without_prediab": AP(yt[nopre_test], p[nopre_test]),
                     "ROC_AUC_test_without_prediab": AUC(yt[nopre_test], p[nopre_test]),
                     "AP_frozen": AP(yt, tp["p_" + m]), "ROC_AUC_frozen": AUC(yt, tp["p_" + m])})
    print(pre_rows[-1], flush=True)
pd.DataFrame(pre_rows).to_csv(os.path.join(out, "r3_prediabetes_refit.csv"), index=False)

# ---- 3. baselines on split 1 (retrained on development data, scored on the thesis test set)
def oof_youden(make, Xa, ya):
    oof = np.zeros(len(ya))
    for k in range(5):
        tr, va = fold != k, fold == k
        oof[va] = make().fit(Xa[tr], ya[tr]).predict_proba(Xa[va])[:, 1]
    fpr, tpr, thr = roc_curve(ya, oof)
    return float(thr[np.argmax(tpr - fpr)])
scaled = lambda est: Pipeline([("scale", StandardScaler()), ("clf", est)])
ALT = {  # Altamimi et al. 2026, reported settings (JIDMIS 3(7s)); seed = thesis estimator seed
    "ALT_LR": lambda: scaled(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "ALT_DT": lambda: DecisionTreeClassifier(max_depth=8, random_state=tc.MODEL_SEED),
    "ALT_RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=tc.MODEL_SEED),
    "ALT_XGB": lambda: XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                                     random_state=tc.MODEL_SEED, verbosity=0)}
# D35 item 3: one row, "Ahmed et al. (stated reading) = default-parameter LR"
REF = {"AHMED_LR_default": lambda: scaled(LogisticRegression()),
       "MAJORITY": lambda: DummyClassifier(strategy="most_frequent")}
Xd_np, Xt_np = Xd.to_numpy(), Xt.to_numpy()
base = []
for name, make in {**ALT, **REF}.items():
    est = make().fit(Xd_np, yd); p = est.predict_proba(Xt_np)[:, 1]
    r = {"baseline": name, "thr_rule": "0.50", **metrics(yt, p, 0.5)}; base.append(r); print(r, flush=True)
    if name.startswith("ALT"):
        t = oof_youden(make, Xd_np, yd)
        r = {"baseline": name, "thr_rule": "Youden (OOF dev)", **metrics(yt, p, t)}; base.append(r); print(r, flush=True)
for m in ["LR", "CAT"]:   # thesis models for reference, same split
    p = tp["p_" + m].to_numpy(); base.append({"baseline": "THESIS_" + m, "thr_rule": "0.50", **metrics(yt, p, 0.5)})
pd.DataFrame(base).to_csv(os.path.join(out, "r1_baselines_split1.csv"), index=False)

# ---- 4. ablations for CAT and LR (thesis configurations fixed); leaky real-row scoring: r1b_cat12_profile5.py
abl = []
sm_seed = int(np.random.SeedSequence([tc.MASTER_SEED, 101]).generate_state(1)[0])   # paper sub-stream for SMOTE
for m in ["LR", "CAT"]:
    abl.append({"model": m, "setting": "row-level split, no resampling (thesis)", **metrics(yt, tp["p_" + m].to_numpy(),
                0.5)})
    # (a) SMOTE inside training only (correct use; plain SMOTE, not SMOTENC)
    Xs, ys = SMOTE(random_state=sm_seed).fit_resample(Xd, yd)
    p = predict(thesis_pipe(m).fit(Xs, ys), m, Xt)
    abl.append({"model": m, "setting": "SMOTE on development data only; real test set", **metrics(yt, p, 0.5)}); print(abl[-1], flush=True)
    # (b) SMOTE before the split (leaky): resample all rows, then split the resampled set 80/20
    Xa, ya = SMOTE(random_state=sm_seed).fit_resample(X, y)
    i_tr, i_te = train_test_split(np.arange(len(ya)), test_size=0.2, stratify=ya, random_state=tc.SEEDS["partition"])
    p = predict(thesis_pipe(m).fit(Xa.iloc[i_tr], ya[i_tr]), m, Xa.iloc[i_te])
    abl.append({"model": m, "setting": "SMOTE before the split (leaky); resampled test set", **metrics(ya[i_te], p, 0.5)}); print(abl[-1], flush=True)
    # (c) profile-grouped split: no 21-predictor profile in both partitions
    key = X.astype(int).astype(str).agg("|".join, axis=1).to_numpy()
    g = pd.factorize(key)[0]
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=tc.SEEDS["partition"]).split(X, y, g))
    p = predict(thesis_pipe(m).fit(X.iloc[tr], y[tr]), m, X.iloc[te])
    abl.append({"model": m, "setting": f"profile-grouped split (test {len(te)} rows, prevalence {y[te].mean():.4f})",
                **metrics(y[te], p, 0.5)}); print(abl[-1], flush=True)
    # (d) thesis novel-profile subset of the row-level test set, for comparison with (c)
    nov = tp["profile_in_dev"].to_numpy() == 0
    abl.append({"model": m, "setting": "row-level split, novel-profile test subset (thesis S5)",
                **metrics(yt[nov], tp["p_" + m].to_numpy()[nov], 0.5)})
pd.DataFrame(abl).to_csv(os.path.join(out, "r1_ablations_split1.csv"), index=False)
json.dump(log, open(os.path.join(out, "r1_log.json"), "w"), indent=1)
print("done", log)
