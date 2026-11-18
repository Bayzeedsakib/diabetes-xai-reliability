"""Independent check of the R5 outputs (paper Phase 5), written separately from the notebook. Runs in the workspace on
the returned files with the data, the Stage 1 procedure, the R4 index files and predictions, and the split-1 frozen
pipelines (the R4 pipelines stay on the PC). Splits 2-5: LR, LGBM and RF are refitted here from the selected configurations
(they reproduce the PC predictions in the workspace, P4-062; checked again below) and cached next to OUTCSV, for the LR
closed form, native TreeSHAP (LGBM, RF) on 3 core rows, the native base values and the common setting on 3 rows.
usage: python -I r5_verify.py CODE_DIR PROJECT_DIR R4_DIR R5_DIR SPLITS(e.g. 1,2,3,4,5) OUTCSV [TEST]"""
import json, os, sys
import joblib
import numpy as np, pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split

code_dir, P, R4, O, sp, outcsv = sys.argv[1:7]; TEST = len(sys.argv) > 7
sys.path.insert(0, code_dir); import thesis_common as tc
SPLITS = [int(s) for s in sp.split(",")]
X, y = tc.load_data(P); F = list(tc.FEATURES)
M6 = ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]
FZ = json.load(open(f"{P}/stage4_outputs/frozen_models.json"))
PRED = pd.read_csv(f"{P}/stage5_outputs/test_predictions.csv", float_precision="round_trip").set_index("row_id")
checks, SKIP = [], []
def ck(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
sig = lambda z: 1 / (1 + np.exp(-z))
CACHE = os.path.join(os.path.dirname(os.path.abspath(outcsv)), "r5_verify_cache")
os.makedirs(CACHE, exist_ok=True)


def config_id(k, m):
    sj = f"{R4}/split{k}/{m}/summary.json"
    if os.path.exists(sj):
        return int(json.load(open(sj))["config_id"])
    # R4 split folder not here: selected configuration from r4_split_model_summaries.csv
    SS = pd.read_csv(os.environ["R5V_SUMMARIES"]); return int(SS[(SS.split == k) & (SS.model == m)].config_id.iloc[0])


# LR configurations that do not refit bit-exactly in the workspace (liblinear l2, primal trust-region Newton solver;
# Linux vs Windows, as P1-001). Only these get the relaxed LR tolerance (D38 amendment, student item 3).
LR_RELAXED = {(3, 18), (4, 18)}


def refit(k, m, dev):
    """selected configuration of split k refitted on its development rows; cached (built once)"""
    cid = config_id(k, m)
    path = os.path.join(CACHE, f"split{k}_{m}_config{cid}.joblib")
    if not os.path.exists(path):
        pipe = tc.make_pipeline(m).set_params(**tc.sampled_configs(m)[cid]).fit(X.iloc[dev], y[dev])
        joblib.dump(pipe, path)
    pipe = joblib.load(path)
    if m == "RF":
        pipe.set_params(clf__n_jobs=1)
    return pipe

# seeds, derived independently of the notebook
po = int(np.random.SeedSequence(42).generate_state(15)[14])
SEED = {1: int(np.random.SeedSequence(int(np.random.SeedSequence(42).generate_state(14)[13])).generate_state(2)[0])}
for k in [2, 3, 4, 5]:
    w = np.random.SeedSequence([po, k]).generate_state(8)
    assert list(w[:7]) == list(np.random.SeedSequence([po, k]).generate_state(7))
    SEED[k] = int(w[7])
ck("split-1 permutation seed = Stage 6 main seed 2941821818", SEED[1] == 2941821818)

# cores and backgrounds rebuilt (Stage 1 procedure for split 1; R4 procedure for splits 2-5)
def rebuild(k):
    if k == 1:
        s = tc.SEEDS
        dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s["partition"])
        dev, test = np.sort(dev), np.sort(test)
        smp = lambda pool, n, st: np.sort(train_test_split(pool, train_size=n, stratify=y[pool], random_state=s[st])[0])
        bg = smp(dev, 500, "shap_background"); bg100 = smp(bg, 100, "shap_perm_background")
        core = smp(smp(test, 2000, "global_sample"), 500, "perm_core")
        return dev, test, bg, bg100, core
    s = dict(zip(["partition", "folds", "subsample_50k", "shap_background", "shap_perm_background", "test_core", "eval_bootstrap"],
                 map(int, np.random.SeedSequence([po, k]).generate_state(7))))
    dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s["partition"])
    dev, test = np.sort(dev), np.sort(test)
    pick = lambda pool, n, st: np.sort(pool[train_test_split(np.arange(len(pool)), train_size=n, stratify=y[pool], random_state=s[st])[0]])
    bg = pick(dev, 500, "shap_background"); bg100 = pick(bg, 100, "shap_perm_background"); core = pick(test, 500, "test_core")
    return dev, test, bg, bg100, core


def load(k, s, m):
    d = os.path.join(O, f"split{k}", s)
    rec = json.load(open(os.path.join(d, f"SHAP_{m}.done.json")))
    f = os.path.join(O, rec["file"])
    ok = tc.sha256_file(f) == rec["sha256"]
    return pd.read_csv(f, float_precision="round_trip"), rec, ok


TOL_ADD = {"tree": 1e-5, "tree_xgb": 1e-4, "linear": 1e-12, "perm": 1e-10}   # XGBoost sums float32 leaf values (Stage 6 max 6.8e-6)
n_files = 0
for k in SPLITS:
    dev, test, bg, bg100, core = rebuild(k)
    if k == 1:
        P_test = {m: PRED[f"p_{m}"] for m in M6}
    else:
        P_test = ({m: pd.read_csv(f"{R4}/split{k}/{m}/test_predictions.csv", float_precision="round_trip").set_index("row_id").p for m in M6}
                  if os.path.exists(f"{R4}/split{k}") else None)
    if P_test is None:
        SKIP.append(f"split{k}: model output = stored R4 test prediction (13 checks) and the LGBM/RF full-test refit check: "
                    "the R4 split folder is on the PC only; the refits are compared with the R5 outputs on the 500 core rows instead")
    else:
        bp = f"{R4}/split{k}/baselines/baselines_test_predictions.csv"
        if os.path.exists(bp):
            P_test["ALT_XGB"] = pd.read_csv(bp, float_precision="round_trip").set_index("row_id").p_ALT_XGB
        else:
            SKIP.append(f"split{k}: native ALT_XGB model output = stored R4 baseline prediction (R4 split folder on the PC only)")
    files = [("native", m) for m in M6 + ["ALT_XGB"]] + [("common", m) for m in M6]
    if k == 1:
        files += [("xgbcheck", "XGB_tree100"), ("xgbcheck", "XGB_perm_logodds")]
    D = {}
    for s, m in files:
        df, rec, ok = load(k, s, m); D[(s, m)] = df; n_files += 1
        ck(f"split{k} {s} {m}: file = .done.json hash", ok)
        rows = df.row_id.to_numpy()
        ck(f"split{k} {s} {m}: rows = first {len(rows)} rows of the rebuilt core", np.array_equal(rows, core[:len(rows)]) and (TEST or len(rows) == 500))
        add = np.abs(df.base_value + df[F].sum(axis=1) - df.model_output).to_numpy()
        kind = "linear" if (s == "native" and m == "LR") else "perm" if (s == "common" or m in ("MLP", "XGB_perm_logodds")) else "tree"
        kind = "tree_xgb" if (kind == "tree" and "XGB" in m) else kind
        ck(f"split{k} {s} {m}: additivity recomputed = column; max <= {TOL_ADD[kind]:g}", np.allclose(add, df.additivity_error, atol=1e-12, rtol=0) and add.max() <= TOL_ADD[kind], f"{add.max():.2e}")
        if s != "xgbcheck" and P_test is not None and ("ALT_XGB" in P_test or m != "ALT_XGB"):
            mm = "ALT_XGB" if m == "ALT_XGB" else m
            p = P_test[mm].loc[rows].to_numpy()
            fx = df.model_output.to_numpy()
            if s == "common" or m in ("RF", "MLP"):
                d_, tol = np.abs(fx - p).max(), 1e-12
            else:
                d_, tol = np.abs(sig(fx) - p).max(), 1e-6
            known = TEST and k == 2 and mm in ("XGB", "ALT_XGB")        # PC XGBoost cannot be reproduced in the workspace
            ck(f"split{k} {s} {m}: model output = stored test prediction" + (" (known workspace difference)" if known else ""), d_ <= tol or known, f"{d_:.2e}")
        if "seed" in rec and rec["seed"] is not None:
            ck(f"split{k} {s} {m}: seed = derived permutation seed", rec["seed"] == SEED[k], rec["seed"])
    # MLP: the common setting is the native explainer, so the values must be identical
    a, b = D[("native", "MLP")], D[("common", "MLP")]
    n = min(len(a), len(b)); a, b = a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)
    ck(f"split{k} MLP native and common: same number of rows", len(D[("native", "MLP")]) == len(D[("common", "MLP")]) or TEST)
    same = a[["row_id", "base_value"] + F].equals(b[["row_id", "base_value"] + F])
    dfx = float(np.abs(a.model_output - b.model_output).max())
    ck(f"split{k} MLP common = MLP native (SHAP and base identical; output within 1e-12)", same and dfx <= 1e-12, f"output {dfx:.1e}")
    # LR closed form on the standardized inputs: phi_j = beta_j (z_j - mean_bg z_j)
    if k == 1:
        lr = joblib.load(f"{P}/{FZ['models']['LR']['pipeline']}")
    else:
        lr = refit(k, "LR", dev)
        # LR config 18 (splits 3-4; liblinear l2) does not refit bit-exactly in the workspace (Linux vs Windows, as
        # P1-001): its LR checks allow 10x the measured refit-vs-PC difference on the core; every other LR check keeps 1e-9
        dcl = D[("common", "LR")]
        d_lr = float(np.abs(lr.predict_proba(X.iloc[dcl.row_id])[:, 1] - dcl.model_output.to_numpy()).max())
        cid = config_id(k, "LR")
        TOL_LR = max(1e-9, 10 * d_lr) if (k, cid) in LR_RELAXED else 1e-9
        ck(f"split{k} LR (config {cid}): workspace refit vs the PC model on the core (max |dp|; tolerance of the LR checks {TOL_LR:.1e})",
           d_lr <= (1e-6 if (k, cid) in LR_RELAXED else 1e-12), f"{d_lr:.1e}")
    if k != 1:
        dfn = D[("native", "LR")]
        Z, Zb = lr.named_steps["prep"].transform(X.iloc[dfn.row_id]), lr.named_steps["prep"].transform(X.iloc[bg])
        names = list(lr.named_steps["prep"].get_feature_names_out()); beta = lr.named_steps["clf"].coef_[0]
        phi = (beta * (Z - Zb.mean(axis=0)))[:, [names.index(f) for f in F]]
        ck(f"split{k} native LR = closed form beta (z - mean bg z)", np.abs(phi - dfn[F].to_numpy()).max() <= TOL_LR, f"{np.abs(phi - dfn[F].to_numpy()).max():.1e}")
    # permutation recomputation (independent wrapper, reverse order, per-row seed): LR (refit or frozen) on 3 rows
    Xb = X.iloc[bg100].to_numpy()
    mods = {"LR": lr}
    if k == 1:
        mods.update({m: joblib.load(f"{P}/{FZ['models'][m]['pipeline']}") for m in ["LGBM", "CAT", "MLP", "RF"]})
    else:
        # LGBM and RF refits (workspace reproduces the PC models, P4-062): refit check, then native TreeSHAP on 3 core rows
        # and the native base value = mean raw output (LGBM) / probability (RF) over the split's 500-row background
        Xbg500 = X.iloc[bg]
        for m in ["LGBM", "RF"]:
            pipe = refit(k, m, dev); mods[m] = pipe
            if P_test is not None:
                dp = np.abs(pipe.predict_proba(X.iloc[test])[:, 1] - P_test[m].loc[test].to_numpy()).max()
                ck(f"split{k} {m}: workspace refit reproduces the stored test predictions (<= 1e-15)", dp <= 1e-15, f"{dp:.1e}")
            else:
                dc = D[("common", m)]
                dp = np.abs(pipe.predict_proba(X.iloc[dc.row_id])[:, 1] - dc.model_output.to_numpy()).max()
                ck(f"split{k} {m}: workspace refit = R5 model output on the 500 core rows (<= 1e-15)", dp <= 1e-15, f"{dp:.1e}")
            clf = pipe.named_steps["clf"]
            dfn = D[("native", m)].set_index("row_id")
            rows3 = dfn.index.to_numpy()[:3][::-1]
            te = shap.TreeExplainer(clf, shap.maskers.Independent(Xbg500, max_samples=500), feature_perturbation="interventional", model_output="raw")
            e = te(X.iloc[rows3], check_additivity=False)
            v = np.asarray(e.values); v = v[:, :, 1] if v.ndim == 3 else v
            dv = np.abs(v - dfn.loc[rows3, F].to_numpy()).max()
            ck(f"split{k} native {m}: TreeSHAP on 3 core rows recomputed (reverse order, <= 1e-9)", dv <= 1e-9, f"{dv:.1e}")
            raw = clf.predict(Xbg500, raw_score=True) if m == "LGBM" else clf.predict_proba(Xbg500)[:, 1]
            db = np.abs(dfn.base_value.to_numpy() - raw.mean()).max()
            ck(f"split{k} native {m}: base value = mean {'raw output' if m == 'LGBM' else 'probability'} over the 500 background (<= 1e-9)",
               db <= 1e-9, f"{db:.1e}")
    for m, pipe in mods.items():
        if m == "RF":
            pipe.set_params(clf__n_jobs=1)
        f = lambda Z, pipe=pipe: pipe.predict_proba(pd.DataFrame(Z, columns=F))[:, 1]
        ex = shap.PermutationExplainer(f, shap.maskers.Independent(Xb, max_samples=100))
        dfc = D[("common", m)]
        worst = 0.0
        for r in dfc.row_id.to_numpy()[:3][::-1]:
            np.random.seed(int(np.random.SeedSequence([SEED[k], int(r)]).generate_state(1)[0]))
            e = ex(X.iloc[[r]].to_numpy(), max_evals=250, silent=True)
            worst = max(worst, np.abs(e.values[0] - dfc.set_index("row_id").loc[r, F].to_numpy()).max())
        tol = 1e-12 if k == 1 else TOL_LR if m == "LR" else 1e-9         # splits 2-5: workspace refits
        ck(f"split{k} common {m}: 3 rows recomputed (independent wrapper, reverse order)", worst <= tol, f"{worst:.1e}")
        # base value of the common setting = mean model output over the 100-row background
        bb = f(Xb).mean()
        ck(f"split{k} common {m}: base value = mean output over the 100-row background", np.abs(dfc.base_value - bb).max() <= (1e-12 if k == 1 else TOL_LR if m == "LR" else 1e-9), f"{np.abs(dfc.base_value - bb).max():.1e}")
    if k == 1:
        t, q = D[("xgbcheck", "XGB_tree100")], D[("xgbcheck", "XGB_perm_logodds")]
        ck("split1 XGB check: same rows and same base value (mean log-odds over the 100 background)",
           np.array_equal(t.row_id, q.row_id) and np.abs(t.base_value - q.base_value).max() < 1e-6, f"{np.abs(t.base_value - q.base_value).max():.1e}")
        reg = pd.read_csv(os.path.join(O, "split1", "regression_check.csv"))
        ck("split1 regression check vs Stage 6: all pass", bool(reg["pass"].all()) and len(reg) == 6)
    # ALT_XGB top-5 set recomputed
    alt = D[("native", "ALT_XGB")]; imp = alt[F].abs().mean().to_numpy()
    t5 = list(np.array(F)[np.argsort(-imp / imp.sum(), kind="stable")[:5]])
    T = pd.read_csv(os.path.join(O, "alt_xgb_top5.csv")); tr = T[T.split == k].iloc[0]
    ck(f"split{k} ALT_XGB top-5 set recomputed", t5 == [tr[f"top{j}"] for j in range(1, 6)], ", ".join(t5))
rep = pd.read_csv(os.path.join(O, "reproduction_all.csv"))
ck("every model passed its reproduction gate", bool(rep["pass"].all()))
ck("every model reproduced its predictions exactly" + (" (test: see detail)" if TEST else ""), bool(rep.exact.all()) or TEST,
   "; ".join(f"split{r.split} {r.model} {r.max_abs_diff:.1e}" for r in rep.itertuples() if not r.exact))
ck(f"file count ({n_files})", n_files == 13 * len(SPLITS) + (2 if 1 in SPLITS else 0))
C = pd.DataFrame(checks); C.to_csv(outcsv, index=False)
print(C.to_string(index=False)); print(f"{C['pass'].sum()}/{len(C)} pass")
for x in SKIP:
    print("NOT CHECKED HERE:", x)
