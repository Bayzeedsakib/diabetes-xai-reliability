"""Independent check of returned R4 outputs (paper Phase 5). Written separately from the notebook.
usage: python -I r4_verify.py CODE_DIR BINARY_CSV R4_OUT_DIR SPLITS(e.g. 2,3,4,5) [TEST_FRACTION]"""
import hashlib, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import thesis_common as tc
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, matthews_corrcoef, brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
b_path, O = sys.argv[2], sys.argv[3]; SPLITS = [int(s) for s in sys.argv[4].split(",")]
frac = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
y = pd.read_csv(b_path)[tc.TARGET].astype(int).to_numpy()
rows_all = np.arange(len(y))
if frac < 1:
    rows_all = np.sort(train_test_split(rows_all, train_size=frac, stratify=y, random_state=0)[0])
po = int(np.random.SeedSequence(42).generate_state(15)[14])
NAMES = ["partition", "folds", "subsample_50k", "shap_background", "shap_perm_background", "test_core", "eval_bootstrap"]

def full_thresholds(yy, p):
    """Independent of the notebook's thresholds(): every distinct OOF value is a candidate; counts come from
    binary search on the sorted scores of each class (not cumulative sums over a joint sort). Rule (S12c):
    positive if p >= t; MCC-max with ties -> higher sensitivity -> closer to 0.50 -> lower t;
    screening = highest t with sensitivity >= 0.80."""
    pos, neg = np.sort(p[yy == 1]), np.sort(p[yy == 0])
    c = np.unique(p)
    TP = len(pos) - np.searchsorted(pos, c, side="left"); FP = len(neg) - np.searchsorted(neg, c, side="left")
    FN, TN = len(pos) - TP, len(neg) - FP
    TP, FP, FN, TN = (a.astype(np.float64) for a in (TP, FP, FN, TN))
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = np.divide(TP * TN - FP * FN, den, out=np.zeros_like(den), where=den > 0)
    sens = TP / len(pos)
    i = np.lexsort((c, np.abs(c - 0.5), -sens, -np.round(mcc, 15)))[0]      # last key = primary
    return float(c[i]), float(c[sens >= 0.80].max()), mcc
checks = []
def ck(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
for k in SPLITS:
    s = dict(zip(NAMES, map(int, np.random.SeedSequence([po, k]).generate_state(7))))
    dev, test = train_test_split(rows_all, test_size=0.2, stratify=y[rows_all], random_state=s["partition"]); dev, test = np.sort(dev), np.sort(test)
    D = pd.read_csv(f"{O}/indices/split{k}/dev_folds.csv"); T = pd.read_csv(f"{O}/indices/split{k}/test_rows.csv")
    ck(f"split{k} partition rebuilt", np.array_equal(D.row_id, dev) and np.array_equal(T.row_id, test))
    f = np.empty(len(dev), int)
    for j, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s["folds"]).split(dev, y[dev])): f[v] = j
    ck(f"split{k} folds rebuilt", np.array_equal(D.fold, f))
    for nm, n, pool, st in [("shap_background_500_dev.csv", 500, dev, "shap_background"), ("test_core_500.csv", 500, test, "test_core")]:
        r = np.sort(pool[train_test_split(np.arange(len(pool)), train_size=n, stratify=y[pool], random_state=s[st])[0]])
        ck(f"split{k} {nm} rebuilt", np.array_equal(pd.read_csv(f"{O}/indices/split{k}/{nm}").row_id, r))
    for m in ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]:
        cv = pd.read_csv(f"{O}/split{k}/{m}/cv_results.csv", float_precision="round_trip")
        sm = json.load(open(f"{O}/split{k}/{m}/summary.json"))
        mean = cv[[f"fold{j}_ap" for j in range(5)]].mean(axis=1)
        ck(f"split{k} {m} mean CV AP recomputed", np.allclose(mean, cv.mean_ap, atol=1e-12))
        ck(f"split{k} {m} selected = highest mean AP, lowest id", sm["config_id"] == int(np.flatnonzero(cv.mean_ap.to_numpy() == cv.mean_ap.max())[0]))
        oof = pd.read_csv(f"{O}/split{k}/{m}/oof.csv", float_precision="round_trip")
        ck(f"split{k} {m} OOF rows = development rows", np.array_equal(oof.row_id, dev))
        fa = [AP(oof.y[oof.fold == j], oof.p[oof.fold == j]) for j in range(5)]
        ck(f"split{k} {m} dev mean AP recomputed", abs(np.mean(fa) - sm["dev_mean_ap"]) < 1e-12)
        p, yy = oof.p.to_numpy(), oof.y.to_numpy()
        tm, ts, mcc_c = full_thresholds(yy, p)
        ck(f"split{k} {m} MCC threshold, all {len(mcc_c)} distinct OOF values", np.isclose(tm, sm["threshold_mcc"], rtol=0, atol=0),
           f"independent {tm!r} vs reported {sm['threshold_mcc']!r}")
        ck(f"split{k} {m} MCC at the threshold equals sklearn", abs(matthews_corrcoef(yy, (p >= sm["threshold_mcc"]).astype(int)) - mcc_c.max()) < 1e-12)
        ck(f"split{k} {m} screening threshold, all distinct OOF values", ts == sm["threshold_screen"], f"independent {ts!r} vs reported {sm['threshold_screen']!r}")
        tp = pd.read_csv(f"{O}/split{k}/{m}/test_predictions.csv", float_precision="round_trip")
        ck(f"split{k} {m} test rows = test partition", np.array_equal(tp.row_id, test))
        ck(f"split{k} {m} test AP / ROC-AUC / Brier recomputed", abs(AP(tp.y, tp.p) - sm["test"]["AP"]) < 1e-12 and abs(AUC(tp.y, tp.p) - sm["test"]["ROC_AUC"]) < 1e-12
           and abs(brier_score_loss(tp.y, tp.p) - sm["test"]["Brier"]) < 1e-12)
        pm = (tp.p >= sm["threshold_mcc"]).astype(int)
        ck(f"split{k} {m} test MCC recomputed", abs(matthews_corrcoef(tp.y, pm) - sm["test"]["MCC_mcc"]) < 1e-12)
    bm = pd.read_csv(f"{O}/split{k}/baselines/baselines_metrics.csv"); bp = pd.read_csv(f"{O}/split{k}/baselines/baselines_test_predictions.csv", float_precision="round_trip")
    ok = all(abs(AP(bp.y, bp[f"p_{r.baseline}"]) - r.AP) < 1e-9 for r in bm.itertuples())
    ck(f"split{k} baselines AP recomputed", ok)
C = pd.DataFrame(checks); C.to_csv(os.path.join(O, "r4_verify.csv"), index=False)
print(C.to_string(index=False)); print(f"{C['pass'].sum()}/{len(C)} pass")
