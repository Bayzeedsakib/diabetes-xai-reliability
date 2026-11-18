"""Does the XGB / CAT refit depend on the thread count? Refit the thesis configurations with n_jobs = 12
(the PC's logical CPU count, S7-001 note) and compare with the frozen test probabilities.
usage: thread_check.py CODE_DIR BINARY_CSV TEST_PREDICTIONS_CSV"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import thesis_common as tc
from sklearn.model_selection import train_test_split
df = pd.read_csv(sys.argv[2]); X = df[tc.FEATURES]; y = df[tc.TARGET].astype(int).to_numpy()
tp = pd.read_csv(sys.argv[3], float_precision="round_trip").set_index("row_id")
dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=tc.SEEDS["partition"])
dev, test = np.sort(dev), np.sort(test)
for m, cid in [("XGB", 2), ("CAT", 7), ("LR", 20)]:
    for nj in [12, 6, 1]:
        pipe = tc.make_pipeline(m, n_jobs=nj); pipe.set_params(**tc.sampled_configs(m)[cid])
        p = pipe.fit(X.iloc[dev], y[dev]).predict_proba(X.iloc[test])[:, 1]
        print(m, "n_jobs", nj, "max abs diff", np.abs(p - tp.loc[test, "p_" + m].to_numpy()).max(), flush=True)
        if m == "LR": break
