"""H2 core-row bootstrap (POST-HOC, EXPLORATORY; approved by the student 2026-11-13 after the H2 result was known).
It does not change the H2 decision (D38). Per split, a paired, class-stratified bootstrap of the split's 500 core rows
(70 positives and 430 negatives resampled with replacement within class, as the design's paired class-stratified
bootstrap): the SAME resampled rows for all four members (CAT, XGB, LGBM, MLP) and for both settings (common, native).
In each draw: global importance = mean |SHAP| over the resampled rows (with multiplicity), ranking by stable sort
(ties by feature order), RBO = the r2_analysis.py / r5_h2.py function (extrapolated RBO on the full 21-predictor
rankings, p = 0.8); statistic = mean RBO of the 3 within-family pairs - mean RBO of the 3 cross-family pairs.
Reported: observed value, bootstrap mean, 95% percentile interval, share of draws with statistic > 0.
B = 2,000 draws. Seed: 16th stream paper_r6 = SeedSequence(42).generate_state(16)[15] (append-only; the first 15
words are asserted unchanged); per split SeedSequence([paper_r6, k]).generate_state(17)[16] (word 16 = H2 core
bootstrap; words 0-15 are the R6 seeds, see the R6 seeds file).
usage: python -I r5_h2_boot.py CODE_DIR R5_DIR DATA_PROJECT_DIR R4_INDEX_DIR OUTDIR"""
import json, os, sys
import numpy as np, pandas as pd

code_dir, O, proj, r4idx, out = sys.argv[1:6]
sys.path.insert(0, code_dir); import thesis_common as tc
os.makedirs(out, exist_ok=True)
F = list(tc.FEATURES)
MEM = ["CAT", "XGB", "LGBM", "MLP"]
WITHIN = [("CAT", "XGB"), ("CAT", "LGBM"), ("XGB", "LGBM")]
CROSS = [("CAT", "MLP"), ("XGB", "MLP"), ("LGBM", "MLP")]
B = 2000

SS16 = np.random.SeedSequence(tc.MASTER_SEED).generate_state(16)
assert [int(v) for v in SS16[:15]] == [int(v) for v in np.random.SeedSequence(tc.MASTER_SEED).generate_state(15)]
PAPER_R6 = int(SS16[15]); assert PAPER_R6 == 1741519417


def boot_seed(k):
    return int(np.random.SeedSequence([PAPER_R6, k]).generate_state(17)[16])


def rbo(a, b, p):                                   # identical to r2_analysis.py
    sa, sb, s = set(), set(), 0.0
    for d in range(1, len(a) + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1]); s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s + p ** len(a) * len(sa & sb) / len(a)


def load(k, s, m):
    rec = json.load(open(os.path.join(O, f"split{k}", s, f"SHAP_{m}.done.json")))
    df = pd.read_csv(os.path.join(O, rec["file"]), float_precision="round_trip")
    assert len(df) == 500
    return df


def stat(A, w):
    # A: model -> |SHAP| (500 x 21); w: counts per row (500,)
    o = {m: np.argsort(-(w @ A[m]) / w.sum(), kind="stable") for m in MEM}
    win = np.mean([rbo(o[a], o[b], 0.8) for a, b in WITHIN]); cr = np.mean([rbo(o[a], o[b], 0.8) for a, b in CROSS])
    return win - cr, win, cr


X, y = tc.load_data(proj)
rows, draws = [], []
for k in [1, 2, 3, 4, 5]:
    D = {s: {m: load(k, s, m) for m in MEM} for s in ["common", "native"]}
    ids = D["common"]["CAT"].row_id.to_numpy()
    assert all(np.array_equal(D[s][m].row_id.to_numpy(), ids) for s in D for m in MEM)
    yc = y[ids]; assert int(yc.sum()) == 70
    pos, neg = np.flatnonzero(yc == 1), np.flatnonzero(yc == 0)
    rng = np.random.default_rng(boot_seed(k))
    W = np.zeros((B, 500))
    for b in range(B):
        idx = np.concatenate([pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]])
        W[b] = np.bincount(idx, minlength=500)
    for s in ["common", "native"]:
        A = {m: np.abs(D[s][m][F].to_numpy()) for m in MEM}
        obs, ow, oc = stat(A, np.ones(500))
        st = np.array([stat(A, W[b]) for b in range(B)])
        lo, hi = np.percentile(st[:, 0], [2.5, 97.5])
        rows.append({"split": k, "setting": s, "seed": boot_seed(k), "draws": B, "observed_within": ow, "observed_cross": oc,
                     "observed_diff": obs, "boot_mean_diff": st[:, 0].mean(), "boot_sd_diff": st[:, 0].std(ddof=1),
                     "ci95_lo": lo, "ci95_hi": hi, "share_draws_diff_gt_0": float((st[:, 0] > 0).mean()),
                     "ci_excludes_0": bool(lo > 0 or hi < 0)})
        draws += [{"split": k, "setting": s, "draw": b, "diff": st[b, 0], "within": st[b, 1], "cross": st[b, 2]} for b in range(B)]
        print(rows[-1], flush=True)
R = pd.DataFrame(rows).sort_values(["setting", "split"])
R.to_csv(f"{out}/h2_boot_summary.csv", index=False, float_format="%.10g")
pd.DataFrame(draws).to_csv(f"{out}/h2_boot_draws.csv", index=False, float_format="%.10g")
json.dump({"label": "post-hoc, exploratory; does not change the H2 decision (D38)", "B": B, "paper_r6": PAPER_R6,
           "seeds": {k: boot_seed(k) for k in range(1, 6)}, "resampling": "paired (all members, both settings), class-stratified (70/430)",
           "summary": R.to_dict("records")}, open(f"{out}/h2_boot_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(R[["split", "setting", "observed_diff", "boot_mean_diff", "ci95_lo", "ci95_hi", "share_draws_diff_gt_0"]].round(4).to_string(index=False))
