"""R1b (paper Phase 5, D35): items 1, 2 and 4 of the student's 2026-11-03 review.
 1. CatBoost rows of R1/R3 rerun with thread_count = 12 (the pinned PC value, P1-007): SMOTE on development
    data only, SMOTE before the split (leaky), profile-grouped split (first fold), prediabetes refit.
 4. Leaky-SMOTE models also scored on the real (original) rows of the resampled test split only; share of
    synthetic rows reported. imblearn SMOTE returns the original rows first, then the synthetic rows.
 2. Profile-grouped check over 5 splits: the 5 folds of StratifiedGroupKFold on 21-predictor profiles vs the
    5 folds of a row-level StratifiedKFold, both shuffle=True with the partition seed; each fold is the test
    set once; fixed thesis configurations; CAT (12 threads) and LR. Robustness row, not confirmatory.
Workspace run on the canonical pins; no timings reported (D14).
usage: python -I r1b_cat12_profile5.py CODE_DIR BINARY_CSV THREECLASS_CSV TEST_PREDICTIONS_CSV OUTDIR"""
import json, os, sys
import numpy as np, pandas as pd

code_dir, b_path, t_path, p_path, out = sys.argv[1:6]
sys.path.insert(0, code_dir)
import thesis_common as tc
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from imblearn.over_sampling import SMOTE

os.makedirs(out, exist_ok=True)
assert tc.sha256_file(b_path) == tc.DATA_SHA256
assert tc.sha256_file(p_path) == "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"
df = pd.read_csv(b_path); X = df[tc.FEATURES]; y = df[tc.TARGET].astype(int).to_numpy()
code012 = pd.read_csv(t_path)["Diabetes_012"].to_numpy().astype(int)
tp = pd.read_csv(p_path, float_precision="round_trip")
dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=tc.SEEDS["partition"])
dev, test = np.sort(dev), np.sort(test)
assert np.array_equal(test, np.sort(tp.row_id.to_numpy()))
tp = tp.set_index("row_id").loc[test]
Xd, yd, Xt, yt = X.iloc[dev], y[dev], X.iloc[test], y[test]
CFG = {"LR": 20, "CAT": 7}
THREADS = {"CAT": 12, "LR": -1}                     # D35 item 1: CatBoost pinned to 12 threads


def pipe_for(m):
    p = tc.make_pipeline(m, n_jobs=THREADS[m]); p.set_params(**tc.sampled_configs(m)[CFG[m]]); return p


def metrics(yy, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"n": len(yy), "prevalence": yy.mean(), "AP": AP(yy, p), "ROC_AUC": AUC(yy, p),
            "accuracy": accuracy_score(yy, pred), "F1": f1_score(yy, pred, zero_division=0),
            "MCC": matthews_corrcoef(yy, pred), "sensitivity": pred[yy == 1].mean(), "specificity": 1 - pred[yy == 0].mean()}


# reproduction guard: CAT at 12 threads must reproduce the frozen probabilities
pc = pipe_for("CAT").fit(Xd, yd).predict_proba(Xt)[:, 1]
d = float(np.abs(pc - tp["p_CAT"].to_numpy()).max()); print("CAT 12-thread refit max |diff|", d, flush=True)
assert d < 1e-12, "CatBoost at thread_count=12 does not reproduce the frozen model"
sm_seed = int(np.random.SeedSequence([tc.MASTER_SEED, 101]).generate_state(1)[0])   # as in R1
key = X.astype(int).astype(str).agg("|".join, axis=1).to_numpy(); g = pd.factorize(key)[0]

rows, info = [], {"cat_repro_max_abs_diff": d, "smote_seed": sm_seed}
# ---- item 1 and 4: SMOTE rows and first profile-grouped fold
Xs, ys = SMOTE(random_state=sm_seed).fit_resample(Xd, yd)
Xa, ya = SMOTE(random_state=sm_seed).fit_resample(X, y)
n0 = len(y); synth = np.arange(len(ya)) >= n0
i_tr, i_te = train_test_split(np.arange(len(ya)), test_size=0.2, stratify=ya, random_state=tc.SEEDS["partition"])
info.update({"leaky_total_rows": len(ya), "leaky_positives": int(ya.sum()), "leaky_synthetic": int(synth.sum()),
             "leaky_synthetic_share_of_positives": synth.sum() / ya.sum(),
             "leaky_test_rows": len(i_te), "leaky_test_positives": int(ya[i_te].sum()),
             "leaky_test_synthetic": int(synth[i_te].sum()),
             "leaky_test_synthetic_share_of_positives": synth[i_te].sum() / ya[i_te].sum(),
             "leaky_test_real_rows": int((~synth[i_te]).sum()), "leaky_test_real_positives": int(ya[i_te][~synth[i_te]].sum()),
             "leaky_train_real_rows": int((~synth[i_tr]).sum())})
# fractional values: share of synthetic rows with a non-integer value in at least one binary/ordinal predictor
disc = [f for f in tc.FEATURES if f not in ("BMI", "MentHlth", "PhysHlth")]
Za = np.asarray(Xa[disc] if hasattr(Xa, "columns") else Xa)
frac = (np.abs(Za - np.round(Za)) > 0).any(axis=1)
info["synthetic_rows_with_fractional_binary_or_ordinal"] = float(frac[synth].mean())
info["real_rows_with_fractional_binary_or_ordinal"] = float(frac[~synth].mean())
print(json.dumps(info, indent=1, default=float), flush=True)
tr0, te0 = next(StratifiedGroupKFold(5, shuffle=True, random_state=tc.SEEDS["partition"]).split(X, y, g))
ABL = os.path.join(out, "r1b_ablations_cat12.csv")
for m in ([] if os.path.exists(ABL) else ["LR", "CAT"]):   # rerun skips finished parts
    p = pipe_for(m).fit(Xs, ys).predict_proba(Xt)[:, 1]
    rows.append({"model": m, "setting": "SMOTE on development data only (plain SMOTE, not SMOTENC); real test set", **metrics(yt, p)})
    pa = pipe_for(m).fit(Xa.iloc[i_tr], ya[i_tr]).predict_proba(Xa.iloc[i_te])[:, 1]
    rows.append({"model": m, "setting": "SMOTE before the split (leaky); resampled test set", **metrics(ya[i_te], pa)})
    real = ~synth[i_te]
    rows.append({"model": m, "setting": "SMOTE before the split (leaky); real rows of the resampled test set only",
                 **metrics(ya[i_te][real], pa[real])})
    pg = pipe_for(m).fit(X.iloc[tr0], y[tr0]).predict_proba(X.iloc[te0])[:, 1]
    rows.append({"model": m, "setting": "profile-grouped split, fold 1 (as P1-024/P1-029)", **metrics(y[te0], pg)})
    print(rows[-4:], flush=True)
if rows:
    pd.DataFrame(rows).to_csv(ABL, index=False)

# ---- item 1: prediabetes refit for CAT at 12 threads (compare with the frozen model, which the 12-thread refit equals)
keep = code012[dev] != 1; nopre = code012[test] != 1
PRE = os.path.join(out, "r3_prediabetes_refit_cat12.csv")
p = pc if os.path.exists(PRE) else pipe_for("CAT").fit(Xd[keep], yd[keep]).predict_proba(Xt)[:, 1]
if not os.path.exists(PRE):
  pre = {"model": "CAT", "threads": 12, "dev_rows": int(keep.sum()), "AP_test": AP(yt, p), "ROC_AUC_test": AUC(yt, p),
       "AP_test_without_prediab": AP(yt[nopre], p[nopre]), "ROC_AUC_test_without_prediab": AUC(yt[nopre], p[nopre]),
       "AP_same_machine_full_refit": AP(yt, pc), "ROC_AUC_same_machine_full_refit": AUC(yt, pc)}
  print(pre, flush=True)
  pd.DataFrame([pre]).to_csv(PRE, index=False)

# ---- item 2: 5 profile-grouped folds vs 5 row-level folds
five = []
splitters = {"profile-grouped": StratifiedGroupKFold(5, shuffle=True, random_state=tc.SEEDS["partition"]).split(X, y, g),
             "row-level": StratifiedKFold(5, shuffle=True, random_state=tc.SEEDS["partition"]).split(X, y)}
for kind, it in splitters.items():
    for k, (tr, te) in enumerate(it):
        shared = np.isin(g[te], np.unique(g[tr]))
        for m in ["LR", "CAT"]:
            p = pipe_for(m).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            five.append({"split_type": kind, "fold": k + 1, "model": m, "test_rows": len(te), "prevalence": y[te].mean(),
                         "test_rows_profile_in_train": int(shared.sum()), "AP": AP(y[te], p), "ROC_AUC": AUC(y[te], p)})
            print(five[-1], flush=True)
F = pd.DataFrame(five); F.to_csv(os.path.join(out, "r1b_profile5_folds.csv"), index=False)
summ = []
for m in ["LR", "CAT"]:
    a = F[(F.model == m) & (F.split_type == "profile-grouped")].sort_values("fold")
    b = F[(F.model == m) & (F.split_type == "row-level")].sort_values("fold")
    for met in ["ROC_AUC", "AP"]:
        dlt = a[met].to_numpy() - b[met].to_numpy()
        summ.append({"model": m, "metric": met, "grouped_mean": a[met].mean(), "grouped_sd": a[met].std(ddof=1),
                     "row_mean": b[met].mean(), "row_sd": b[met].std(ddof=1), "paired_diff_mean": dlt.mean(),
                     "paired_diff_sd": dlt.std(ddof=1), "paired_diff_min": dlt.min(), "paired_diff_max": dlt.max(),
                     "grouped_prev_mean": a.prevalence.mean(), "row_prev_mean": b.prevalence.mean()})
S = pd.DataFrame(summ); S.to_csv(os.path.join(out, "r1b_profile5_summary.csv"), index=False)
json.dump(info, open(os.path.join(out, "r1b_info.json"), "w"), indent=1, default=float)
print(S.round(5).to_string(index=False))
