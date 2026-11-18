# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""Independent checks of R2 outputs (written separately from r2_analysis.py)."""
import sys, json, warnings, numpy as np, pandas as pd, joblib
sys.path.insert(0, "."); import thesis_common as tc
from scipy.stats import kendalltau
warnings.simplefilter("ignore")
F = list(tc.FEATURES); out = {}
X = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")[F]
PER = pd.read_csv("analysis_outputs/r2/analysis/r2_agreement_per_seed.csv")
rng = np.random.default_rng(7)
# 1. RBO and tau-b recomputed with an independent formula (Webber eq. 32, closed sets) on 50 random rows
def rbo_ext(S, T, p):
    k = len(S); X_ = [len(set(S[:d]) & set(T[:d])) for d in range(1, k + 1)]
    return X_[-1] / k * p ** k + (1 - p) / p * sum(X_[d - 1] / d * p ** d for d in range(1, k + 1))
mx = 0; mt = 0
for i in rng.choice(len(PER), 50, replace=False):
    r = PER.iloc[i]
    g = pd.read_csv(f"analysis_outputs/r2/grid/lime_{r.setting}_{r.model}.csv")
    w = g[(g.row_id == r.row_id) & (g.seed_index == r.seed_index)][F].to_numpy()[0]
    phi = pd.read_csv(f"stage6_outputs/shap/SHAP_{r.model}.csv").set_index("row_id").loc[r.row_id, F].to_numpy()
    S = list(np.lexsort((np.arange(21), -np.abs(phi)))); T = list(np.lexsort((np.arange(21), -np.abs(w))))
    mx = max(mx, abs(rbo_ext(S, T, 0.8) - r.rbo08)); mt = max(mt, abs(kendalltau(np.abs(phi), np.abs(w))[0] - r.tau_b))
out["rbo_recompute_max_abs_diff"] = mx; out["taub_recompute_max_abs_diff"] = mt
# 2. SHAP chol50 additivity on the explained scale and CholCheck mean of the background used
add = {}
for m in ["LR", "XGB", "LGBM", "CAT", "RF", "MLP"]:
    s = pd.read_csv(f"analysis_outputs/r2/chol50/shap_chol50_{m}.csv", float_precision="round_trip").set_index("row_id")
    pipe = joblib.load(f"stage3_outputs/models/{m}/pipeline_{m}.joblib"); clf = pipe.named_steps["clf"]
    Xr = X.loc[s.index]
    if m == "LR": f = pipe.decision_function(Xr)
    elif m == "XGB": f = clf.predict(Xr, output_margin=True).astype(float)
    elif m == "LGBM": f = clf.predict(Xr, raw_score=True)
    elif m == "CAT": f = clf.predict(Xr, prediction_type="RawFormulaVal")
    else:
        if m == "RF": clf.set_params(n_jobs=1)
        f = pipe.predict_proba(Xr)[:, 1]
    add[m] = float(np.abs(s.base_value + s[F].sum(axis=1) - f).max())
out["chol50_additivity_max"] = add
# LR closed form: phi_Chol = beta_Chol * (x - 0.5) on the standardized input (CholCheck is passthrough, not scaled)
pipe = joblib.load("stage3_outputs/models/LR/pipeline_LR.joblib"); names = list(pipe.named_steps["prep"].get_feature_names_out())
beta = pipe.named_steps["clf"].coef_[0][names.index("CholCheck")]
s = pd.read_csv("analysis_outputs/r2/chol50/shap_chol50_LR.csv").set_index("row_id")
out["LR_chol50_closed_form_max_diff"] = float(np.abs(s.CholCheck - beta * (X.loc[s.index].CholCheck - 0.5)).max())
# 3. LIME intervention (c): share of CholCheck = 1 among the 5,000 perturbation rows for one case
from lime.lime_tabular import LimeTabularExplainer
dev = pd.read_csv("stage1_outputs/indices/dev_folds.csv").row_id.to_numpy()
BIN = [i for i, f in enumerate(F) if f not in tc.NON_BINARY]
for tag, ff in [("T5", None), ("T5_chol50", [0.5, 0.5])]:
    LE = LimeTabularExplainer(X.iloc[dev].to_numpy(), feature_names=F, categorical_features=BIN, discretize_continuous=True, discretizer="quartile", mode="classification", random_state=0)
    if ff: LE.feature_frequencies[2] = np.array(ff)
    LE.random_state = LE.discretizer.random_state = np.random.RandomState(2792219722)
    seen = {}
    def f(Z):
        seen["chol"] = Z[1:, 2].mean(); return np.c_[1 - Z[:, 0] * 0 - 0.5, np.full(len(Z), 0.5)]
    LE.explain_instance(X.iloc[int(dev[0])].to_numpy().astype(float), f, labels=(1,), num_features=21, num_samples=5000)
    out[f"perturbation_share_CholCheck1_{tag}"] = float(seen["chol"])
# 4. H5 shares recomputed for 3 random pairs directly from files
H = pd.read_csv("analysis_outputs/r2/analysis/r2_cholcheck_shares.csv")
pf = X.iloc[dev].mean().to_numpy(); d5 = 0
for i in rng.choice(len(H), 3, replace=False):
    r = H.iloc[i]; g = pd.read_csv(f"analysis_outputs/r2/grid/lime_T5_{r.model}.csv"); w = g[g.row_id == r.row_id][F].mean().to_numpy()
    x = X.loc[r.row_id].to_numpy(); wa = w.copy(); wa[BIN] = w[BIN] * np.abs(x[BIN] - pf[BIN])
    d5 = max(d5, abs(abs(w[2]) / np.abs(w).sum() - r.lime_T5), abs(abs(wa[2]) / np.abs(wa).sum() - r.lime_rescaled_a))
out["h5_share_recompute_max_diff"] = d5
# 5. medians in r2_results.json equal medians of the shares file
R = json.load(open("analysis_outputs/r2/analysis/r2_results.json"))
out["h5_medians_match_1e-9 (CSV written with 10 significant digits)"] = all(abs(H[c].median() - R["H5"]["all"]["median_share"][c]) < 1e-9 for c in R["H5"]["all"]["median_share"])
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("analysis_outputs/r2/analysis/r2_verify.json", "w"), indent=1, default=float)
