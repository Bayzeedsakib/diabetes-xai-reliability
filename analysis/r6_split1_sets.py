# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""Split 1, native setting (MLP native = common): descriptive within-set RBO for the three refit kinds
(bootstrap 10, seed-only 5, Rashomon <= 9 + the selected R5 model), and seed-only identity vs the R5 model
(D39 rule 1). Descriptive only; not part of the H3 decision. usage: python -I r6_split1_sets.py PROJECT_ROOT OUTDIR   (PROJECT_ROOT = repository root, holding paper_r5_outputs/ and paper_r6_outputs/)"""
import itertools, os, sys
import numpy as np, pandas as pd
root, out = sys.argv[1:3]
d6 = os.path.join(root, "paper_r6_outputs/split1/native"); d5 = os.path.join(root, "paper_r5_outputs/split1/native")
def rbo(a, b, p=0.8):
    sa, sb, s = set(), set(), 0.0
    for d in range(1, len(a) + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1]); s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s + p ** len(a) * len(sa & sb) / len(a)
def rd(path):
    df = pd.read_csv(path, float_precision="round_trip"); v = df.iloc[:, 4:].abs().mean().to_numpy()
    return np.argsort(-v, kind="stable"), df.model_output.to_numpy(), df.iloc[:, 4:].to_numpy()
rows, ident = [], []
for m in ["CAT", "XGB", "LGBM", "MLP", "RF", "LR"]:
    o5, out5, phi5 = rd(os.path.join(d5, f"SHAP_{m}.csv"))
    for kind in ["boot", "seed", "rash"]:
        fs = sorted(f for f in os.listdir(d6) if f.startswith(f"SHAP_{kind}") and f.endswith(f"_{m}.csv"))
        if not fs: continue
        R = [rd(os.path.join(d6, f)) for f in fs]
        ords = [r[0] for r in R] + ([o5] if kind == "rash" else [])
        v = [rbo(a, b) for a, b in itertools.combinations(ords, 2)]
        vs5 = [rbo(r[0], o5) for r in R]
        rows.append({"model": m, "kind": kind, "n_sets": len(ords), "n_pairs": len(v), "median_within_rbo": np.median(v),
                     "min_within_rbo": min(v), "median_rbo_vs_R5_model": np.median(vs5), "min_rbo_vs_R5_model": min(vs5)})
        if kind == "seed":
            for f, r in zip(fs, R):
                ident.append({"model": m, "file": f, "max_abs_output_diff_vs_R5": np.abs(r[1] - out5).max(),
                              "max_abs_shap_diff_vs_R5": np.abs(r[2] - phi5).max(), "identical_to_R5": bool(np.abs(r[1] - out5).max() == 0)})
S = pd.DataFrame(rows); S.to_csv(f"{out}/r6_split1_refit_kinds_rbo.csv", index=False, float_format="%.6g")
I = pd.DataFrame(ident); I.to_csv(f"{out}/r6_seed_only_vs_r5.csv", index=False, float_format="%.6g")
pd.set_option("display.width", 250); print(S.round(4).to_string(index=False)); print(I.groupby("model")[["max_abs_output_diff_vs_R5", "identical_to_R5"]].agg(["min", "max"]))
