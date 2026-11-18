# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""R2 (paper Phase 5, split 1 = thesis partition): LIME settings grid and the CholCheck LIME intervention (H4, H5c).
Models: LR, RF, XGB, LGBM, CAT, MLP (frozen thesis pipelines, hashes checked against the Stage 4 freeze).
Cases: the 500-row core (shap_perm_core_500_test.csv, verified against MANIFEST.csv) + the 30-case thesis panel.
Settings (phase4_design.md section 3 and section 9 items 2 and 10):
  T5        thesis setting, unchanged (14 binary categorical; 7 non-binary incl. 4 ordinals quartile-discretized;
            default kernel width 0.75*sqrt(21); 5,000 samples) = reference row, must reproduce Stage 6 on the panel
  T10       thesis setting at 10,000 samples
  N_<disc>_<kw>  new settings: 14 binary + 4 ordinal predictors categorical; discretizer {quartile, decile, none}
            acts on BMI, MentHlth, PhysHlth only; kernel width {0.5, 1, 2} x default; 5,000 samples
  T5_chol50 thesis setting with CholCheck perturbation frequency set to 50/50 (H5 intervention c)
Seeds: the three thesis LIME seeds (S15k) for every setting, case and model; before each explanation the
explainer's random state is reset to RandomState(seed) (S15m).
Checkpoint: one file per (setting, model); a rerun skips finished files.
usage: python -I r2_lime_grid.py CODE_DIR BINARY_CSV FROZEN_JSON PIPE_DIR INDEX_DIR CASE_PANEL OUTDIR [N_JOBS] [SETTINGS]"""
import hashlib, json, os, sys, time, warnings
import numpy as np, pandas as pd

code_dir, b_path, fz_path, pdir, idir, panel_path, out = sys.argv[1:8]
N_JOBS = int(sys.argv[8]) if len(sys.argv) > 8 else 2
ONLY = sys.argv[9].split(",") if len(sys.argv) > 9 else None
sys.path.insert(0, code_dir)
import thesis_common as tc

FEATURES = list(tc.FEATURES)
BIN = [i for i, f in enumerate(FEATURES) if f not in tc.NON_BINARY]                 # 14 binary
ORD = [FEATURES.index(f) for f in ["GenHlth", "Age", "Education", "Income"]]       # 4 ordinals
CHOL = FEATURES.index("CholCheck")
KW0 = 0.75 * np.sqrt(len(FEATURES))                                                 # LIME default kernel width
MODELS = ["LR", "RF", "XGB", "LGBM", "CAT", "MLP"]
LIME_SEEDS = [int(s) for s in np.random.SeedSequence(tc.SEEDS["lime_perturbation"]).generate_state(3)]
assert LIME_SEEDS == [2792219722, 1652670301, 1080463383]
SETTINGS = {"T5": dict(cat="thesis", disc="quartile", kw=1.0, n=5000),
            "T10": dict(cat="thesis", disc="quartile", kw=1.0, n=10000)}
for d in ["quartile", "decile", "none"]:
    for k in [0.5, 1.0, 2.0]:
        SETTINGS[f"N_{d}_{k:g}"] = dict(cat="new", disc=d, kw=k, n=5000)
SETTINGS["T5_chol50"] = dict(cat="thesis", disc="quartile", kw=1.0, n=5000, chol50=True)


def setup():
    assert tc.sha256_file(b_path) == tc.DATA_SHA256
    FZ = json.load(open(fz_path, encoding="utf-8"))
    for m in MODELS:
        assert tc.sha256_file(os.path.join(pdir, m, f"pipeline_{m}.joblib")) == FZ["models"][m]["pipeline_sha256"], m
    man = pd.read_csv(os.path.join(idir, "MANIFEST.csv")).set_index("file").sha256
    for f in ["dev_folds.csv", "shap_perm_core_500_test.csv"]:
        assert tc.sha256_file(os.path.join(idir, f)) == man[f], f
    df = pd.read_csv(b_path)
    X = df[FEATURES]
    dev = pd.read_csv(os.path.join(idir, "dev_folds.csv")).row_id.to_numpy()
    core = pd.read_csv(os.path.join(idir, "shap_perm_core_500_test.csv")).row_id.to_numpy()
    panel = pd.read_csv(panel_path)
    assert hashlib.sha256(open(panel_path, "rb").read()).hexdigest().startswith("dd74d70a")
    cases = pd.concat([pd.DataFrame({"row_id": core, "case_set": "core", "case_id": -1}),
                       pd.DataFrame({"row_id": panel.row_id, "case_set": "panel", "case_id": panel.case_id})], ignore_index=True)
    assert cases.row_id.is_unique and len(cases) == 530
    return X, dev, cases


def explainer(X, dev, s):
    from lime.lime_tabular import LimeTabularExplainer
    cat = BIN if s["cat"] == "thesis" else sorted(BIN + ORD)
    kw = dict(feature_names=FEATURES, categorical_features=cat, mode="classification", class_names=["0", "1"],
              random_state=0, kernel_width=s["kw"] * KW0)
    if s["disc"] == "none":
        LE = LimeTabularExplainer(X.iloc[dev].to_numpy(), discretize_continuous=False, **kw)
    else:
        LE = LimeTabularExplainer(X.iloc[dev].to_numpy(), discretize_continuous=True, discretizer=s["disc"], **kw)
    if s.get("chol50"):
        vals = list(LE.feature_values[CHOL]); assert sorted(vals) == [0.0, 1.0], vals
        LE.feature_frequencies[CHOL] = np.array([0.5, 0.5])
    return LE


def run_task(sname, m):
    import joblib
    path = os.path.join(out, f"lime_{sname}_{m}.csv")
    if os.path.exists(path):
        return path, "skipped"
    s = SETTINGS[sname]
    X, dev, cases = setup()
    LE = explainer(X, dev, s)
    pipe = joblib.load(os.path.join(pdir, m, f"pipeline_{m}.joblib"))
    if m == "RF":
        pipe.set_params(clf__n_jobs=1)
    if m in ("XGB", "LGBM"):
        pipe.named_steps["clf"].set_params(n_jobs=1)

    def f(Z):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pipe.predict_proba(pd.DataFrame(Z, columns=FEATURES))
    rows, t0 = [], time.time()
    for c in cases.itertuples(index=False):
        x = X.iloc[int(c.row_id)].to_numpy().astype(float)
        p_model = float(f(x[None, :])[0, 1])
        for si, seed in enumerate(LIME_SEEDS):
            rs = np.random.RandomState(seed)
            LE.random_state = rs
            if LE.discretizer is not None:
                LE.discretizer.random_state = rs
            e = LE.explain_instance(x, f, labels=(1,), num_features=len(FEATURES), num_samples=s["n"])
            w = dict(e.as_map()[1])
            rows.append({"setting": sname, "model": m, "row_id": int(c.row_id), "case_set": c.case_set, "case_id": int(c.case_id),
                         "seed_index": si + 1, **{FEATURES[j]: w.get(j, 0.0) for j in range(len(FEATURES))},
                         "r2": float(e.score), "local_pred": float(np.ravel(e.local_pred)[0]), "intercept": float(e.intercept[1]),
                         "model_p": p_model})
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False, float_format="%.17g", lineterminator="\n")
    os.replace(path + ".tmp", path)
    return path, f"{time.time() - t0:.0f}s (workspace, not reported)"


if __name__ == "__main__":
    from joblib import Parallel, delayed
    os.makedirs(out, exist_ok=True)
    names = ONLY or list(SETTINGS)
    tasks = [(s, m) for s in names for m in MODELS]
    # expensive models first so the two workers finish together
    order = {"RF": 0, "LGBM": 1, "XGB": 2, "MLP": 3, "CAT": 4, "LR": 5}
    tasks.sort(key=lambda t: (order[t[1]], names.index(t[0])))
    for r in Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(delayed(run_task)(s, m) for s, m in tasks):
        print(r, flush=True)
