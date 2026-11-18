# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""Independent check of h3_pairs.csv: recompute RBO (vectorised formula) from the raw SHAP files,
medians, T and exact/sampled p with a separate implementation. usage: python -I r6_h3_check.py PROJECT_ROOT H3_OUT   (PROJECT_ROOT = repository root)"""
import itertools, os, sys, json
import numpy as np, pandas as pd
root, out = sys.argv[1:3]
P = pd.read_csv(f"{out}/h3_pairs.csv")
def rk(k, s, tag, m):
    df = pd.read_csv(os.path.join(root, "paper_r6_outputs", f"split{k}", s, f"SHAP_{tag}_{m}.csv"))
    v = df.iloc[:, 4:].abs().mean().to_numpy(); return np.argsort(-v, kind="stable")
def rbo2(a, b, p=0.8):
    n = len(a); A = np.array([len(set(a[:d]) & set(b[:d])) / d for d in range(1, n + 1)])
    return (1 - p) * (p ** np.arange(n) * A).sum() + p ** n * A[-1]
bad = 0
for _, r in P.iterrows():
    k, s, a, b = r.split, r.setting, r.model_a, r.model_b; n = 10 if k == 1 else 5
    X = [rk(k, s, f"boot{i:02d}", a) for i in range(1, n + 1)] + [rk(k, s, f"boot{i:02d}", b) for i in range(1, n + 1)]
    M = np.array([[rbo2(x, y) for y in X] for x in X])
    def T(S):
        S = np.array(S); C = np.setdiff1d(np.arange(2 * n), S)
        iu = np.triu_indices(n, 1)
        wa = M[np.ix_(S, S)][iu]; wb = M[np.ix_(C, C)][iu]; bt = M[np.ix_(S, C)].ravel()
        return min(np.median(wa), np.median(wb)) - np.median(bt), np.median(wa), np.median(wb), np.median(bt)
    t0, ma, mb, mbt = T(range(n))
    if k == 1:
        g = np.random.default_rng(1012652155); Ts = [t0] + [T(np.sort(g.permutation(2 * n)[:n]))[0] for _ in range(9999)]
    else:
        Ts = [T(c)[0] for c in itertools.combinations(range(2 * n), n)]
    p = np.mean(np.array(Ts) >= t0 - 1e-12)
    d = max(abs(t0 - r["T"]), abs(ma - r.median_within_a), abs(mb - r.median_within_b), abs(mbt - r.median_between), abs(p - r.p_perm))
    N = len(Ts); globals()["bp"] = globals().get("bp", 0) + (round(p * N) != round(r.p_perm * N))
    if d > 1e-9: bad += 1; print("DIFF", k, s, a, b, d)
print(f"checked {len(P)} rows (values stored at 10 significant digits), mismatches > 1e-9: {bad}; permutation counts differing: {globals().get('bp', 0)}")
