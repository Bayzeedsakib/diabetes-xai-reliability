"""P7 (D41): Platt calibration of the class-weighted boosters, split 1 only.

Pre-stated in decisions_log.md D41 before any run on the real files.
Platt map fitted on the stored pooled OOF development probabilities (Stage 4),
applied to the frozen test probabilities (Stage 5). No model is refitted.

Usage:
  python -I p7_platt.py --project-dir <thesis project root on the PC>
                        --dataset <diabetes_binary_health_indicators_BRFSS2015.csv>
                        --out <output folder> [--fixture]

Reads (read-only):
  <project>/stage4_outputs/frozen_models.json          (SHA asserted)
  <project>/<oof_file per model from the freeze>        (SHA asserted against the freeze)
  <project>/stage5_outputs/test_predictions.csv         (SHA asserted)
Writes:
  <out>/p7_platt.csv, <out>/p7_platt.json, <out>/p7_report.md
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ["CAT", "XGB", "LGBM"]
SHA = {"dataset": "19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0",   # S1-047
       "freeze": "1dd5bf06708d148399675d9fc657857df1c86f015cbd83c24e79e8754b48c0d8",    # S4-047
       "test_pred": "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"}  # S5-136
BRIER_REG = {"CAT": 0.1389, "XGB": 0.1062, "LGBM": 0.1061}   # S5-071..S5-073 (4 dp)
N_ROWS, N_DEV, N_TEST = 253680, 202944, 50736
EPS = 1e-15


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def platt_fit(z, y, tol=1e-12, max_iter=100):
    """Unpenalised logistic regression of y on (1, z) by Newton-Raphson; returns (a, b).
    Written out so the result does not depend on the scikit-learn version."""
    X = np.column_stack([np.ones_like(z), z]); w = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ w; p = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (y - p); H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g); w = w + step
        if np.max(np.abs(step)) < tol:
            return float(w[0]), float(w[1])
    raise SystemExit("Platt fit did not converge")


def ece10(p, y):
    """Expected calibration error, 10 equal-width bins on [0, 1]; bin = min(floor(10p), 9);
    count-weighted mean of |mean p - mean y| over non-empty bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum(np.floor(10 * p).astype(int), 9)
    tot = 0.0
    for k in range(10):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(p)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def gate(ok, msg, gates):
    gates.append({"gate": msg, "pass": bool(ok)})
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        pd.DataFrame(gates).to_csv(os.path.join(ARGS.out, "p7_gates.csv"), index=False)
        sys.exit(f"Gate failed: {msg}. Nothing reported.")


ap = argparse.ArgumentParser()
ap.add_argument("--project-dir", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fixture", action="store_true", help="synthetic test: skip real hashes, sizes and registry Brier")
ARGS = ap.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
G = []
real = not ARGS.fixture

fz_path = os.path.join(ARGS.project_dir, "stage4_outputs", "frozen_models.json")
tp_path = os.path.join(ARGS.project_dir, "stage5_outputs", "test_predictions.csv")
if real:
    gate(sha256(ARGS.dataset) == SHA["dataset"], "dataset SHA-256 = S1-047", G)
    gate(sha256(fz_path) == SHA["freeze"], "frozen_models.json SHA-256 = S4-047", G)
    gate(sha256(tp_path) == SHA["test_pred"], "test_predictions.csv SHA-256 = S5-136", G)
with open(fz_path, encoding="utf-8") as f:
    FZ = json.load(f)

y_all = pd.read_csv(ARGS.dataset, usecols=["Diabetes_binary"])["Diabetes_binary"].to_numpy().astype(int)
if real:
    gate(len(y_all) == N_ROWS, f"dataset rows = {N_ROWS}", G)
tp = pd.read_csv(tp_path)
gate(tp.row_id.is_unique and np.array_equal(tp.y.to_numpy().astype(int), y_all[tp.row_id.to_numpy()]),
     "test labels equal dataset labels by row_id", G)
if real:
    gate(len(tp) == N_TEST, f"test rows = {N_TEST}", G)

rows, fits = [], {}
for m in MODELS:
    r = FZ["models"][m]
    oof_path = os.path.join(ARGS.project_dir, r["oof_file"])
    gate(sha256(oof_path) == r["oof_sha256"], f"{m}: OOF file SHA-256 = freeze", G)
    oof = pd.read_csv(oof_path)
    gate(oof.row_id.is_unique and set(oof.row_id).isdisjoint(set(tp.row_id)),
         f"{m}: OOF row_ids unique and disjoint from test", G)
    if real:
        gate(len(oof) == N_DEV and len(oof) + len(tp) == N_ROWS, f"{m}: OOF rows = {N_DEV}; dev + test = all rows", G)
    y_dev = y_all[oof.row_id.to_numpy()]
    # Platt: unpenalised logistic regression of y on logit(p), fitted on development OOF only
    a, b = platt_fit(logit(oof.score.to_numpy()), y_dev.astype(float))
    gate(b > 0, f"{m}: Platt slope > 0 (monotone increasing map)", G)
    p0 = tp[f"p_{m}"].to_numpy(float)
    p1 = 1.0 / (1.0 + np.exp(-(a + b * logit(p0))))
    y = tp.y.to_numpy().astype(int)
    b0, b1 = brier(p0, y), brier(p1, y)
    if real:
        gate(round(b0, 4) == BRIER_REG[m], f"{m}: test Brier before = registry S5 value {BRIER_REG[m]}", G)
    ap0, ap1 = average_precision_score(y, p0), average_precision_score(y, p1)
    au0, au1 = roc_auc_score(y, p0), roc_auc_score(y, p1)
    gate(abs(ap0 - ap1) < 1e-12 and abs(au0 - au1) < 1e-12, f"{m}: AP and ROC-AUC unchanged (|diff| < 1e-12)", G)
    rows.append({"model": m, "platt_intercept": a, "platt_slope": b, "n_dev_oof": len(oof), "n_test": len(tp),
                 "brier_before": b0, "brier_after": b1, "ece10_before": ece10(p0, y), "ece10_after": ece10(p1, y),
                 "mean_p_before": p0.mean(), "mean_p_after": p1.mean(), "observed_prevalence": y.mean(),
                 "ap_before": ap0, "ap_after": ap1, "roc_auc_before": au0, "roc_auc_after": au1,
                 "max_abs_ap_diff": abs(ap0 - ap1), "max_abs_auc_diff": abs(au0 - au1)})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ARGS.out, "p7_platt.csv"), index=False, float_format="%.10g")
pd.DataFrame(G).to_csv(os.path.join(ARGS.out, "p7_gates.csv"), index=False)
meta = {"design": "D41", "split": 1, "models": MODELS, "fit_data": "pooled Stage 4 OOF development probabilities",
        "map": "p' = sigmoid(a + b * logit(clip(p, 1e-15, 1 - 1e-15))); unpenalised logistic regression, Newton-Raphson (tol 1e-12)",
        "ece": "10 equal-width bins, bin = min(floor(10p), 9), count-weighted |mean p - mean y|",
        "fixture": ARGS.fixture, "inputs_sha256": {"dataset": sha256(ARGS.dataset), "freeze": sha256(fz_path),
                                                   "test_predictions": sha256(tp_path)},
        "gates_passed": len(G)}
with open(os.path.join(ARGS.out, "p7_platt.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
L = ["# P7 Platt calibration (D41), split 1", "", "| Model | a | b | Brier before | after | ECE before | after | mean p before | after | prevalence |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in res.itertuples():
    L.append(f"| {r.model} | {r.platt_intercept:.3f} | {r.platt_slope:.3f} | {r.brier_before:.4f} | {r.brier_after:.4f} | "
             f"{r.ece10_before:.4f} | {r.ece10_after:.4f} | {r.mean_p_before:.4f} | {r.mean_p_after:.4f} | {r.observed_prevalence:.4f} |")
L += ["", f"All {len(G)} gates passed. AP and ROC-AUC unchanged for every model (monotone map)."]
with open(os.path.join(ARGS.out, "p7_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
