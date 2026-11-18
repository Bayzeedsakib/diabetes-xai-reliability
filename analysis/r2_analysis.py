"""R2 analysis (paper Phase 5, split 1): LIME settings grid (H4) and the CholCheck gap (H5); E1 exploratory.
Definitions fixed by Claude on 2026-11-03 BEFORE any R2 LIME/SHAP output existed (phase4_design.md section 10):
- Cases x models: 530 rows (500-row core + 30-case panel) x 6 models (LR, RF, XGB, LGBM, CAT, MLP) = 3,180 pairs.
- SHAP = the Stage 6 native values (500-row background; MLP 100-row permutation background), from the
  returned stage6_outputs/shap/SHAP_<m>.csv files.
- Rankings by |attribution|; ties broken by feature order (stable sort). Kendall tau-b on |values| (as Stage 6);
  top-3/top-5 Jaccard; extrapolated RBO on the full 21-predictor rankings, p = 0.8 (sensitivity p = 0.9);
  sign agreement on SHAP's top 5. Per-seed LIME weights unless stated; "seed-mean" = mean of the 3 seeds (Stage 6).
- H4 sign part: median over the 3,180 pairs of the seed-mean sign agreement in the reference setting T5 >= 0.95.
- H4 rank part: per pair, SD_set = mean over the 3 seeds of the SD (ddof 1) of SHAP-LIME RBO across the 11 grid
  settings (T5, T10, 9 new); SD_seed = mean over the 11 settings of the SD (ddof 1) across the 3 seeds.
  Supported if SD_set > SD_seed for more than half of the 3,180 pairs, one-sided exact sign test p < 0.05.
  Sensitivity: the 9 new settings only; tau-b instead of RBO. The intervention T5_chol50 is not a grid setting.
- H5: share_j = |a_j| / sum_k |a_k| per pair. gap = |median LIME share - median SHAP share| of CholCheck over
  the 3,180 pairs. native: LIME T5 seed-mean vs SHAP native. (a) rescaled LIME: every binary-predictor weight
  x |x_j - p_j|, p_j = share of 1s among the development rows (LIME's training frequency); non-binary weights
  unchanged; supported if gap_a <= 0.5 x gap_native (decision rule, D33 item 9). (b) SHAP with the CholCheck 50/50
  background vs LIME T5. (c) LIME T5_chol50 vs SHAP native. (b) and (c) are reported separately.
  Predictions of the scaling account (stated before the runs): (b) the gap shrinks, because SHAP's CholCheck
  share rises; (c) the gap stays close to native, because LIME's weight estimates the effect of switching and
  should not depend on the sampling frequency.
- E1 (exploratory): per predictor x model, rank gap = median over cases of (rank by |SHAP| - rank by |LIME T5|)
  (positive = LIME ranks it higher); imbalance = mean over cases of LIME's training frequency of the case's own
  value or bin; Spearman over 21 x 6 points; also over the 14 binary predictors.
usage: python -I r2_analysis.py CODE_DIR BINARY_CSV INDEX_DIR CASE_PANEL STAGE6_DIR GRID_DIR CHOL50_DIR OUTDIR"""
import itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr, binomtest

code_dir, b_path, idir, panel_path, s6, gdir, cdir, out = sys.argv[1:9]
sys.path.insert(0, code_dir)
import thesis_common as tc
os.makedirs(out, exist_ok=True)
FEATURES = list(tc.FEATURES); F = len(FEATURES)
BIN = [i for i, f in enumerate(FEATURES) if f not in tc.NON_BINARY]
CHOL = FEATURES.index("CholCheck")
MODELS = ["LR", "RF", "XGB", "LGBM", "CAT", "MLP"]
GRID = ["T5", "T10"] + [f"N_{d}_{k}" for d in ["quartile", "decile", "none"] for k in ["0.5", "1", "2"]]
ALL = GRID + ["T5_chol50"]
X = pd.read_csv(b_path)[FEATURES]
dev = pd.read_csv(os.path.join(idir, "dev_folds.csv")).row_id.to_numpy()
pfreq = X.iloc[dev].mean().to_numpy()                                  # share of 1s (binary predictors)


def order(v):
    return np.argsort(-np.abs(v), kind="stable")


def rbo(a, b, p):
    sa, sb, s = set(), set(), 0.0
    for d in range(1, len(a) + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1]); s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s + p ** len(a) * len(sa & sb) / len(a)


def jac(a, b, k):
    return len(set(a[:k]) & set(b[:k])) / len(set(a[:k]) | set(b[:k]))


def agree(phi, w):
    oa, ob = order(phi), order(w)
    t5 = oa[:5]
    return {"tau_b": kendalltau(np.abs(phi), np.abs(w)).statistic, "rbo08": rbo(oa, ob, 0.8), "rbo09": rbo(oa, ob, 0.9),
            "top3_jaccard": jac(oa, ob, 3), "top5_jaccard": jac(oa, ob, 5),
            "sign_top5": float(np.mean(np.sign(phi[t5]) == np.sign(w[t5])))}


def share(v):
    a = np.abs(v); return a / a.sum()


# ---- inputs
panel = pd.read_csv(panel_path)
SH = {m: pd.read_csv(os.path.join(s6, "shap", f"SHAP_{m}.csv"), float_precision="round_trip").set_index("row_id") for m in MODELS}
G = pd.concat([pd.read_csv(os.path.join(gdir, f"lime_{s}_{m}.csv"), float_precision="round_trip") for s in ALL for m in MODELS],
              ignore_index=True)
rows530 = G[(G.setting == "T5") & (G.model == "LR") & (G.seed_index == 1)].row_id.to_numpy()
assert len(rows530) == 530
checks = {}
# ---- 0. reference reproduction: T5 on the 30 panel cases = Stage 6 LIME weights (thesis setting, PC)
mx = {}
for m in MODELS:
    ref = pd.read_csv(os.path.join(s6, "lime", f"LIME_{m}.csv"), float_precision="round_trip")
    ref = ref.pivot_table(index=["row_id", "seed_index"], columns="feature", values="weight")[FEATURES]
    g = G[(G.setting == "T5") & (G.model == m) & (G.case_set == "panel")].set_index(["row_id", "seed_index"])[FEATURES]
    mx[m] = float(np.abs(g.loc[ref.index].to_numpy() - ref.to_numpy()).max())
checks["T5_vs_stage6_lime_max_abs_diff"] = mx
# ties in |SHAP| (stable order used)
checks["shap_rows_with_tied_abs_values"] = {m: int(sum(len(np.unique(np.abs(SH[m].loc[r, FEATURES].to_numpy()))) < F for r in rows530)) for m in MODELS}

# ---- 1. per (setting, model, row, seed) agreement with SHAP; seed-mean agreement; seed stability
W = G.set_index(["setting", "model", "row_id", "seed_index"])[FEATURES].sort_index()
meta = G.set_index(["setting", "model", "row_id", "seed_index"])[["case_set", "case_id", "r2", "local_pred", "model_p"]].sort_index()
per, mean_rows, seedst = [], [], []
for s in ALL:
    for m in MODELS:
        for r in rows530:
            phi = SH[m].loc[r, FEATURES].to_numpy()
            ws = [W.loc[(s, m, r, k)].to_numpy() for k in (1, 2, 3)]
            for k, w in enumerate(ws, 1):
                mt = meta.loc[(s, m, r, k)]
                per.append({"setting": s, "model": m, "row_id": r, "seed_index": k, **agree(phi, w),
                            "lime_r2": mt.r2, "lime_gap": abs(mt.local_pred - mt.model_p), "chol_in_lime_top5": CHOL in order(w)[:5]})
            wm = np.mean(ws, axis=0)
            mean_rows.append({"setting": s, "model": m, "row_id": r, "case_set": meta.loc[(s, m, r, 1)].case_set, **agree(phi, wm),
                              "chol_share_lime": share(wm)[CHOL], "chol_in_lime_top5": CHOL in order(wm)[:5]})
            for a, b in itertools.combinations(range(3), 2):
                oa, ob = order(ws[a]), order(ws[b])
                seedst.append({"setting": s, "model": m, "row_id": r, "pair": f"{a + 1}-{b + 1}",
                               "tau_b": kendalltau(np.abs(ws[a]), np.abs(ws[b])).statistic, "rbo08": rbo(oa, ob, 0.8),
                               "top5_jaccard": jac(oa, ob, 5)})
PER, MEAN, SEED = pd.DataFrame(per), pd.DataFrame(mean_rows), pd.DataFrame(seedst)
PER.to_csv(os.path.join(out, "r2_agreement_per_seed.csv"), index=False, float_format="%.10g")
MEAN.to_csv(os.path.join(out, "r2_agreement_seedmean.csv"), index=False, float_format="%.10g")
SEED.to_csv(os.path.join(out, "r2_lime_seed_stability.csv"), index=False, float_format="%.10g")

# ---- 2. summary by setting (all models) and by setting x model
def summ(keys):
    a = PER.groupby(keys)[["tau_b", "rbo08", "rbo09", "top3_jaccard", "top5_jaccard", "sign_top5", "lime_r2", "lime_gap"]].median()
    a["chol_in_lime_top5_share"] = PER.groupby(keys).chol_in_lime_top5.mean()
    b = SEED.groupby(keys)[["tau_b", "rbo08", "top5_jaccard"]].median().add_prefix("seed_")
    c = MEAN.groupby(keys)[["tau_b", "rbo08", "top5_jaccard", "sign_top5"]].median().add_prefix("seedmean_")
    return a.join(b).join(c)
S1 = summ(["setting"]).reindex(ALL); S1.to_csv(os.path.join(out, "r2_summary_by_setting.csv"), float_format="%.6g")
S2 = summ(["setting", "model"]); S2.to_csv(os.path.join(out, "r2_summary_by_setting_model.csv"), float_format="%.6g")

# ---- 3. H4
h4 = {}
t5 = MEAN[MEAN.setting == "T5"]
h4["sign_agreement_T5_seedmean_median"] = float(t5.sign_top5.median())
h4["sign_agreement_T5_seedmean_mean"] = float(t5.sign_top5.mean())
h4["sign_part_supported"] = bool(t5.sign_top5.median() >= 0.95)
def sd_test(settings, stat):
    P = PER[PER.setting.isin(settings)]
    sd_set = P.groupby(["model", "row_id", "seed_index"])[stat].std(ddof=1).groupby(["model", "row_id"]).mean()
    sd_seed = P.groupby(["model", "row_id", "setting"])[stat].std(ddof=1).groupby(["model", "row_id"]).mean()
    d = (sd_set - sd_seed).dropna()
    k, n = int((d > 0).sum()), int((d != 0).sum())
    return {"pairs": int(len(d)), "pairs_sd_set_gt_sd_seed": k, "share": k / len(d), "sign_test_p_one_sided": binomtest(k, n, 0.5, "greater").pvalue,
            "median_sd_set": float(sd_set.median()), "median_sd_seed": float(sd_seed.median())}
h4["rank_part_main_11settings_rbo08"] = sd_test(GRID, "rbo08")
h4["rank_part_main_supported"] = bool(h4["rank_part_main_11settings_rbo08"]["share"] > 0.5 and h4["rank_part_main_11settings_rbo08"]["sign_test_p_one_sided"] < 0.05)
h4["sens_9new_rbo08"] = sd_test(GRID[2:], "rbo08")
h4["sens_11settings_tau_b"] = sd_test(GRID, "tau_b")
bm = {}
for m in MODELS:
    P = PER[(PER.setting.isin(GRID)) & (PER.model == m)]
    sd_set = P.groupby(["row_id", "seed_index"]).rbo08.std(ddof=1).groupby("row_id").mean()
    sd_seed = P.groupby(["row_id", "setting"]).rbo08.std(ddof=1).groupby("row_id").mean()
    bm[m] = float(((sd_set - sd_seed) > 0).mean())
h4["share_sd_set_gt_sd_seed_by_model"] = bm

# ---- 4. H5 CholCheck shares
def lime_mean(s, m, r):
    return np.mean([W.loc[(s, m, r, k)].to_numpy() for k in (1, 2, 3)], axis=0)
C50 = {m: pd.read_csv(os.path.join(cdir, f"shap_chol50_{m}.csv"), float_precision="round_trip").set_index("row_id") for m in MODELS}
hrows = []
for m in MODELS:
    for r in rows530:
        x = X.iloc[r].to_numpy(); phi = SH[m].loc[r, FEATURES].to_numpy(); w = lime_mean("T5", m, r)
        wa = w.copy(); wa[BIN] = w[BIN] * np.abs(x[BIN] - pfreq[BIN])
        hrows.append({"model": m, "row_id": r, "case_set": "panel" if r in set(panel.row_id) else "core", "CholCheck": x[CHOL],
                      "shap_native": share(phi)[CHOL], "lime_T5": share(w)[CHOL], "lime_rescaled_a": share(wa)[CHOL],
                      "shap_chol50_b": share(C50[m].loc[r, FEATURES].to_numpy())[CHOL], "lime_chol50_c": share(lime_mean("T5_chol50", m, r))[CHOL],
                      "lime_T5_top5": CHOL in order(w)[:5], "lime_a_top5": CHOL in order(wa)[:5], "shap_top5": CHOL in order(phi)[:5],
                      "shap_b_top5": CHOL in order(C50[m].loc[r, FEATURES].to_numpy())[:5],
                      "lime_c_top5": CHOL in order(lime_mean("T5_chol50", m, r))[:5]})
H = pd.DataFrame(hrows); H.to_csv(os.path.join(out, "r2_cholcheck_shares.csv"), index=False, float_format="%.10g")
med = lambda c, d=H: float(d[c].median())
def h5block(d):
    g_nat = abs(med("lime_T5", d) - med("shap_native", d))
    return {"n_pairs": len(d), "median_share": {c: med(c, d) for c in ["shap_native", "lime_T5", "lime_rescaled_a", "shap_chol50_b", "lime_chol50_c"]},
            "top5_count": {c: int(d[c].sum()) for c in ["shap_top5", "lime_T5_top5", "lime_a_top5", "shap_b_top5", "lime_c_top5"]},
            "gap_native": g_nat, "gap_a": abs(med("lime_rescaled_a", d) - med("shap_native", d)),
            "gap_b": abs(med("lime_T5", d) - med("shap_chol50_b", d)), "gap_c": abs(med("lime_chol50_c", d) - med("shap_native", d))}
h5 = {"all": h5block(H), "panel": h5block(H[H.case_set == "panel"]), "CholCheck_1": h5block(H[H.CholCheck == 1]),
      "CholCheck_0": h5block(H[H.CholCheck == 0]), "by_model": {m: h5block(H[H.model == m]) for m in MODELS}}
for k in ["a", "b", "c"]:
    h5["all"][f"ratio_{k}"] = h5["all"][f"gap_{k}"] / h5["all"]["gap_native"]
h5["supported_on_a"] = bool(h5["all"]["gap_a"] <= 0.5 * h5["all"]["gap_native"])
h5["p_CholCheck_dev"] = float(pfreq[CHOL]); h5["cases_with_CholCheck_0"] = int((X.iloc[rows530].CholCheck == 0).sum())

# ---- 5. E1 exploratory: rank gap vs frequency of the case's own value / bin (LIME T5 discretization)
from lime.lime_tabular import LimeTabularExplainer
LE = LimeTabularExplainer(X.iloc[dev].to_numpy(), feature_names=FEATURES, categorical_features=BIN, discretize_continuous=True,
                          discretizer="quartile", mode="classification", random_state=0)
freq_own = np.zeros((len(rows530), F))
for i, r in enumerate(rows530):
    xd = LE.discretizer.discretize(X.iloc[r].to_numpy().astype(float))
    for j in range(F):
        vals = list(LE.feature_values[j]); freq_own[i, j] = LE.feature_frequencies[j][vals.index(xd[j])]
e1 = []
for m in MODELS:
    rk = []
    for r in rows530:
        rs = np.empty(F); rs[order(SH[m].loc[r, FEATURES].to_numpy())] = np.arange(1, F + 1)
        rl = np.empty(F); rl[order(lime_mean("T5", m, r))] = np.arange(1, F + 1)
        rk.append(rs - rl)
    rk = np.median(np.array(rk), axis=0)
    e1 += [{"model": m, "feature": FEATURES[j], "binary": j in BIN, "rank_gap_median": rk[j], "mean_freq_own_value": freq_own[:, j].mean()} for j in range(F)]
E1 = pd.DataFrame(e1); E1.to_csv(os.path.join(out, "r2_e1_rank_gap.csv"), index=False, float_format="%.6g")
e1s = {"spearman_all_21x6": spearmanr(E1.mean_freq_own_value, E1.rank_gap_median).statistic,
       "spearman_binary_14x6": spearmanr(E1[E1.binary].mean_freq_own_value, E1[E1.binary].rank_gap_median).statistic,
       "spearman_binary_14_model_mean": spearmanr(E1[E1.binary].groupby("feature").mean_freq_own_value.mean(),
                                                  E1[E1.binary].groupby("feature").rank_gap_median.mean()).statistic}
res = {"checks": checks, "H4": h4, "H5": h5, "E1": e1s}
json.dump(res, open(os.path.join(out, "r2_results.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
print(S1.round(3).to_string())
