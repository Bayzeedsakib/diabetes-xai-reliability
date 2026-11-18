# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""R2 (paper Phase 5, split 1), H5 intervention (b): SHAP with a background in which CholCheck is 50/50.
Background = every row of the native background twice, once with CholCheck = 0 and once with CholCheck = 1;
all other predictors keep their background values (500 -> 1,000 rows for tree models and LR; the 100-row
permutation background -> 200 rows for MLP). Explainers, output scales and seeds as in Stage 6 (S15b, S15k).
Before the intervention, each model's native explainer is rerun on the first 10 core rows and compared with the
Stage 6 SHAP file (implementation check).
usage: python -I r2_shap_chol50.py CODE_DIR BINARY_CSV FROZEN_JSON PIPE_DIR INDEX_DIR CASE_PANEL SHAP_DIR OUTDIR [N_JOBS]"""
import contextlib, io, json, os, sys, warnings
import numpy as np, pandas as pd

code_dir, b_path, fz_path, pdir, idir, panel_path, sdir, out = sys.argv[1:9]
N_JOBS = int(sys.argv[9]) if len(sys.argv) > 9 else 2
sys.path.insert(0, code_dir)
import thesis_common as tc

FEATURES = list(tc.FEATURES); CHOL = FEATURES.index("CholCheck")
MODELS = ["LR", "XGB", "LGBM", "CAT", "MLP", "RF"]
SS14 = np.random.SeedSequence(tc.MASTER_SEED).generate_state(14)
PERM_SEED = int(np.random.SeedSequence(int(SS14[13])).generate_state(2)[0])
assert PERM_SEED == 2941821818
row_seed = lambda base, r: int(np.random.SeedSequence([int(base), int(r)]).generate_state(1)[0])


def load():
    assert tc.sha256_file(b_path) == tc.DATA_SHA256
    FZ = json.load(open(fz_path, encoding="utf-8"))
    for m in MODELS:
        assert tc.sha256_file(os.path.join(pdir, m, f"pipeline_{m}.joblib")) == FZ["models"][m]["pipeline_sha256"], m
    man = pd.read_csv(os.path.join(idir, "MANIFEST.csv")).set_index("file").sha256
    ix = {}
    for f in ["shap_background_500_dev.csv", "shap_perm_background_100_dev.csv", "shap_perm_core_500_test.csv"]:
        assert tc.sha256_file(os.path.join(idir, f)) == man[f], f
        ix[f] = pd.read_csv(os.path.join(idir, f)).row_id.to_numpy()
    X = pd.read_csv(b_path)[FEATURES]
    rows = np.r_[ix["shap_perm_core_500_test.csv"], pd.read_csv(panel_path).row_id.to_numpy()]
    return X, ix["shap_background_500_dev.csv"], ix["shap_perm_background_100_dev.csv"], rows


def chol50(B):
    a, b = B.copy(), B.copy(); a["CholCheck"] = 0.0; b["CholCheck"] = 1.0
    return pd.concat([a, b], ignore_index=True)


def tree_chunk(m, Xc, Xbg):
    import joblib, shap
    pipe = joblib.load(os.path.join(pdir, m, f"pipeline_{m}.joblib")); clf = pipe.named_steps["clf"]
    if m in ("RF", "XGB", "LGBM"):
        clf.set_params(n_jobs=1)
    obj = clf.get_booster() if m == "XGB" else clf
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        ex = shap.TreeExplainer(obj, shap.maskers.Independent(Xbg, max_samples=len(Xbg)),
                                feature_perturbation="interventional", model_output="raw")
        e = ex(Xc, check_additivity=False)
    v, b = np.asarray(e.values), np.asarray(e.base_values, float)
    if v.ndim == 3:
        v, b = v[:, :, 1], b[:, 1]
    return v.astype(float), np.broadcast_to(b, (len(Xc),)).astype(float)


def linear(Xc, Xbg):
    import joblib, shap
    pipe = joblib.load(os.path.join(pdir, "pipeline_LR.joblib")); prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = list(prep.get_feature_names_out()); Zbg, Z = prep.transform(Xbg), prep.transform(Xc)
    e = shap.LinearExplainer(clf, shap.maskers.Independent(Zbg, max_samples=len(Zbg)))(Z)
    order = [names.index(f) for f in FEATURES]
    return np.asarray(e.values)[:, order].astype(float), np.broadcast_to(np.asarray(e.base_values, float), (len(Xc),)).astype(float)


def perm(Xc, ids, Xbg):
    import joblib, shap
    pipe = joblib.load(os.path.join(pdir, "pipeline_MLP.joblib"))

    def f(Z):
        U, inv = np.unique(Z, axis=0, return_inverse=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = pipe.predict_proba(pd.DataFrame(U, columns=FEATURES))[:, 1]
        return p[inv.ravel()]
    ex = shap.PermutationExplainer(f, shap.maskers.Independent(Xbg.to_numpy(), max_samples=len(Xbg)))
    V, Bv = [], []
    for x, r in zip(Xc.to_numpy(), ids):
        np.random.seed(row_seed(PERM_SEED, r))
        e = ex(x[None, :], max_evals=250, silent=True)
        V.append(e.values[0]); Bv.append(float(np.asarray(e.base_values).ravel()[0]))
    return np.array(V, float), np.array(Bv)


def explain(m, rows, X, Xbg):
    Xc = X.iloc[rows].reset_index(drop=True)
    if m == "LR":
        return linear(Xc, Xbg)
    if m == "MLP":
        return perm(Xc, rows, Xbg)
    from joblib import Parallel, delayed
    ch = np.array_split(np.arange(len(rows)), max(1, int(np.ceil(len(rows) / 25))))
    res = Parallel(n_jobs=N_JOBS)(delayed(tree_chunk)(m, Xc.iloc[c], Xbg) for c in ch)
    return np.vstack([r[0] for r in res]), np.concatenate([r[1] for r in res])


if __name__ == "__main__":
    os.makedirs(out, exist_ok=True)
    X, bg500, bg100, rows = load()
    chk = []
    for m in MODELS:
        nat_bg = X.iloc[bg100 if m == "MLP" else bg500].reset_index(drop=True)
        ref = pd.read_csv(os.path.join(sdir, f"SHAP_{m}.csv"), float_precision="round_trip").set_index("row_id")
        v, _ = explain(m, rows[:10], X, nat_bg)
        d = float(np.abs(v - ref.loc[rows[:10], FEATURES].to_numpy()).max())
        chk.append({"model": m, "native_rerun_max_abs_diff_vs_stage6": d}); print(chk[-1], flush=True)
        path = os.path.join(out, f"shap_chol50_{m}.csv")
        if os.path.exists(path):
            continue
        bgc = chol50(nat_bg)
        assert bgc.CholCheck.mean() == 0.5
        V, Bv = explain(m, rows, X, bgc)
        dfo = pd.DataFrame(V, columns=FEATURES); dfo.insert(0, "base_value", Bv); dfo.insert(0, "row_id", rows)
        dfo.to_csv(path, index=False, float_format="%.17g", lineterminator="\n"); print("written", path, flush=True)
    pd.DataFrame(chk).to_csv(os.path.join(out, "shap_native_rerun_check.csv"), index=False)
