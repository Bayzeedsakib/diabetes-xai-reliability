"""H1 (paper Phase 5, pre-stated in phase4_design.md section 1): near-equivalent set per split.
Per split: top model = highest mean development AP (D7 rule); for each of the other 5 models, paired
class-stratified bootstrap of dAP = AP(m) - AP(top) on the split's test set (B = 2,000; per resample positives
then negatives drawn with replacement by numpy default_rng(eval_bootstrap seed of the split); split 1 uses the
thesis seed 2705014006, as S13e); TOST p = max(p_low, p_high), p_low = (#{dAP* <= -delta} + 1)/(B + 1),
p_high = (#{dAP* >= delta} + 1)/(B + 1); Holm across the 5 candidates of the split; equivalent if Holm p < 0.05
(the 90% percentile interval inside (-delta, delta) is reported beside it, for information only; the decision is
the Holm-adjusted TOST p, D37 amendment A1). H1 holds in a split if the set has >= 2 families; overall if it holds
in >= 4 of 5 splits; members = models equivalent in >= 4 of 5 splits.
delta = 0.01 (primary), 0.005 and 0.02 (sensitivity). Across splits: mean and SD (ddof 1) of dAP and a TOST with
the Nadeau-Bengio corrected variance (1/J + n_test/n_train) s^2, t with J - 1 = 4 df (unadjusted, as pre-stated in
phase4_design.md section 5; Holm across the 5 candidates is reported beside it, D37 amendment A2).
2026-11-04 (D37 amendment A7): the split-1 test predictions are checked against the Stage 5 hash before use.
usage: python -I r4_h1.py CODE_DIR R4_OUT_DIR TEST_PREDICTIONS_CSV OUTDIR"""
import hashlib, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score as AP
from statsmodels.stats.multitest import multipletests

code_dir, O, tp_path, out = sys.argv[1:5]
sys.path.insert(0, code_dir); import thesis_common as tc
os.makedirs(out, exist_ok=True)
MODELS = ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]
FAM = {"XGB": "boosting", "LGBM": "boosting", "CAT": "boosting", "RF": "bagging", "LR": "linear", "MLP": "neural"}
DELTAS = [0.01, 0.005, 0.02]; B = 2000; J = 5
TP1_SHA256 = "b9813461cd03e4552b50c245c8750648d08b9c48c1960d249642982bc15114ed"     # S5-136
h_tp = hashlib.sha256(open(tp_path, "rb").read()).hexdigest()
print("split-1 test_predictions.csv SHA-256:", h_tp)
assert h_tp == TP1_SHA256, "split-1 test_predictions.csv is not the Stage 5 file (S5-136)"
S = pd.read_csv(f"{O}/r4_models_summary.csv")
po = int(np.random.SeedSequence(tc.MASTER_SEED).generate_state(15)[14])
seed = {1: tc.SEEDS["eval_bootstrap"]}
for k in [2, 3, 4, 5]:
    seed[k] = int(np.random.SeedSequence([po, k]).generate_state(7)[6])
tp1 = pd.read_csv(tp_path, float_precision="round_trip")


def preds(k):
    if k == 1:
        return tp1.y.to_numpy(), {m: tp1[f"p_{m}"].to_numpy() for m in MODELS}
    d = {m: pd.read_csv(f"{O}/split{k}/{m}/test_predictions.csv", float_precision="round_trip") for m in MODELS}
    y = d["LR"].y.to_numpy()
    assert all(np.array_equal(d[m].row_id, d["LR"].row_id) for m in MODELS)
    return y, {m: d[m].p.to_numpy() for m in MODELS}


rows, boot_all, n_train = [], {}, {}
for k in [1, 2, 3, 4, 5]:
    y, P = preds(k)
    top = S[(S.split == k) & S.top_model].model.tolist(); assert len(top) == 1; top = top[0]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed[k])
    BA = np.empty((B, len(MODELS)))
    for b in range(B):
        w = np.bincount(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))], minlength=len(y))
        BA[b] = [AP(y, P[m], sample_weight=w) for m in MODELS]
    boot_all[k] = BA
    pd.DataFrame(BA, columns=MODELS).to_csv(f"{out}/bootstrap_ap_split{k}.csv", index=False, float_format="%.17g")
    pt = {m: AP(y, P[m]) for m in MODELS}
    for m in MODELS:
        if m == top:
            continue
        d_star = BA[:, MODELS.index(m)] - BA[:, MODELS.index(top)]
        r = {"split": k, "top": top, "model": m, "family": FAM[m], "dAP": pt[m] - pt[top],
             "ci90_lo": np.percentile(d_star, 5), "ci90_hi": np.percentile(d_star, 95),
             "ci95_lo": np.percentile(d_star, 2.5), "ci95_hi": np.percentile(d_star, 97.5), "seed": seed[k]}
        for dl in DELTAS:
            p_lo, p_hi = ((d_star <= -dl).sum() + 1) / (B + 1), ((d_star >= dl).sum() + 1) / (B + 1)
            r[f"p_tost_{dl}"] = max(p_lo, p_hi)
            r[f"ci90_inside_{dl}"] = bool(-dl < r["ci90_lo"] and r["ci90_hi"] < dl)
        rows.append(r)
R = pd.DataFrame(rows)
for dl in DELTAS:
    R[f"p_holm_{dl}"] = np.nan
    for k, g in R.groupby("split"):
        R.loc[g.index, f"p_holm_{dl}"] = multipletests(g[f"p_tost_{dl}"], method="holm")[1]
    R[f"equiv_{dl}"] = R[f"p_holm_{dl}"] < 0.05
R.to_csv(f"{out}/h1_per_split.csv", index=False, float_format="%.10g")
res = {"bootstrap_seeds": seed, "test_predictions_sha256": h_tp}
for dl in DELTAS:
    per = {}
    for k, g in R.groupby("split"):
        mem = [g.top.iloc[0]] + g[g[f"equiv_{dl}"]].model.tolist()
        fams = sorted({FAM[m] for m in mem})
        per[int(k)] = {"set": mem, "families": fams, "h1_holds": len(fams) >= 2}
    cnt = {m: int(sum(m in v["set"] for v in per.values())) for m in MODELS}
    res[f"delta_{dl}"] = {"per_split": per, "splits_h1_holds": int(sum(v["h1_holds"] for v in per.values())),
                          "H1_supported": sum(v["h1_holds"] for v in per.values()) >= 4,
                          "membership_count": cnt, "members_ge4of5": [m for m in MODELS if cnt[m] >= 4]}
# across splits, Nadeau-Bengio (all five splits have the same top model if 'top' is constant)
n_test, n_tr = 50736, 202944; corr = 1 / J + n_test / n_tr
nb = []
tops = R.groupby("split").top.first().unique()
for m in [x for x in MODELS if x not in tops]:
    d = R[R.model == m].sort_values("split").dAP.to_numpy()
    mu, s = d.mean(), d.std(ddof=1); se = np.sqrt(corr * s ** 2)
    r = {"model": m, "vs": "/".join(tops), "mean_dAP": mu, "sd_dAP": s, "se_corrected": se,
         "t_diff": mu / se, "p_diff_two_sided": 2 * stats.t.sf(abs(mu / se), J - 1)}
    for dl in DELTAS:
        r[f"p_tost_nb_{dl}"] = max(stats.t.sf((mu + dl) / se, J - 1), stats.t.sf((dl - mu) / se, J - 1))
    nb.append(r)
NB = pd.DataFrame(nb); NB.to_csv(f"{out}/h1_across_splits_nb.csv", index=False, float_format="%.10g")
res["nadeau_bengio"] = NB.to_dict("records"); res["correction_factor"] = corr
json.dump(res, open(f"{out}/h1_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(R[["split", "top", "model", "dAP", "ci90_lo", "ci90_hi", "p_holm_0.01", "equiv_0.01", "equiv_0.005", "equiv_0.02"]].round(4).to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k.startswith("delta")}, indent=1))
print(NB.round(5).to_string(index=False))
