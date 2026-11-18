"""Number of distinct T values among the 252 label assignments (splits 2-5), per pair x setting
(D40 wording fix (e)). Uses the RBO values in h3_rbo_values.csv. usage: python -I r6_h3_distinct_T.py H3_DIR"""
import itertools, sys
import numpy as np, pandas as pd
h3 = sys.argv[1]
V = pd.read_csv(f"{h3}/h3_rbo_values.csv")
rows = []
for (k, s, pr), g in V[V.split > 1].groupby(["split", "setting", "pair"]):
    M = np.ones((10, 10))
    for _, r in g.iterrows():
        M[int(r.i), int(r.j)] = M[int(r.j), int(r.i)] = r.rbo08
    Ts = []
    for A in itertools.combinations(range(10), 5):
        B = [x for x in range(10) if x not in A]
        wa = [M[i, j] for i, j in itertools.combinations(A, 2)]; wb = [M[i, j] for i, j in itertools.combinations(B, 2)]
        bt = [M[i, j] for i in A for j in B]
        Ts.append(min(np.median(wa), np.median(wb)) - np.median(bt))
    rows.append({"split": k, "setting": s, "pair": pr, "distinct_T": len(np.unique(np.round(Ts, 12)))})
D = pd.DataFrame(rows); D.to_csv(f"{h3}/h3_distinct_T.csv", index=False)
print(D.distinct_T.describe()); print(D.groupby("setting").distinct_T.agg(["min", "max"]))
