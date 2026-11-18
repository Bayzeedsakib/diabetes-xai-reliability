# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""Split-1 LR Rashomon configurations: recover raw-scale coefficients from the native LinearExplainer SHAP
values on the 500 core rows (phi_ij = b_j (x_ij - mean_bg_j), so b_j = slope of phi_j on x_j; exact for a linear
model), compare with the selected configuration 20 (R5/Stage 6 file). Descriptive only.
usage: python -I r6_lr_rashomon_coef.py PROJECT_ROOT DATA_CSV CV_LR OUTDIR   (PROJECT_ROOT = repository root)"""
import json, os, sys
import numpy as np, pandas as pd
root, data, cvf, out = sys.argv[1:5]
F = None
X = pd.read_csv(data)
cv = pd.read_csv(cvf).set_index("config_id"); gap = cv.mean_test_ap.max() - cv.mean_test_ap
def coefs(path):
    df = pd.read_csv(path, float_precision="round_trip"); F = list(df.columns[4:])
    x = X.loc[df.row_id, F].to_numpy(float); phi = df[F].to_numpy()
    b = []; r = []
    for j in range(len(F)):
        xc = x[:, j] - x[:, j].mean(); b.append((xc @ (phi[:, j] - phi[:, j].mean())) / (xc @ xc))
        r.append(np.abs(phi[:, j] - (phi[:, j].mean() + b[-1] * xc)).max())
    return np.array(b), df.model_output.to_numpy(), F, max(r), np.abs(phi).mean(0)
b0, o0, F, r0, i0 = coefs(os.path.join(root, "paper_r5_outputs/split1/native/SHAP_LR.csv"))
rows = [{"config_id": 20, "source": "R5 (selected)", "C": cv.C[20], "penalty": cv.penalty[20], "cv_gap": gap[20],
         "max_abs_coef_diff_vs_c20": 0.0, "max_rel_coef_diff_vs_c20": 0.0, "max_abs_logodds_diff_core": 0.0, "fit_resid": r0,
         "same_ranking_as_c20": True, "n_zero_coefs": int((np.abs(b0) < 1e-10).sum())}]
coef = {"feature": F, "c20": b0}
order0 = np.argsort(-i0, kind="stable")
for fn in sorted(os.listdir(os.path.join(root, "paper_r6_outputs/split1/native"))):
    if fn.startswith("SHAP_rash_") and fn.endswith("_LR.csv"):
        c = int(fn.split("_")[2][1:])
        b, o, _, r, i1 = coefs(os.path.join(root, "paper_r6_outputs/split1/native", fn))
        coef[f"c{c}"] = b
        rows.append({"config_id": c, "source": "R6 Rashomon", "C": cv.C[c], "penalty": cv.penalty[c], "cv_gap": gap[c],
                     "max_abs_coef_diff_vs_c20": np.abs(b - b0).max(),
                     "max_rel_coef_diff_vs_c20": (np.abs(b - b0) / np.maximum(np.abs(b0), 1e-12)).max(),
                     "max_abs_logodds_diff_core": np.abs(o - o0).max(), "fit_resid": r,
                     "same_ranking_as_c20": bool((np.argsort(-i1, kind="stable") == order0).all()),
                     "n_zero_coefs": int((np.abs(b) < 1e-10).sum())})
T = pd.DataFrame(rows).sort_values("cv_gap"); T.to_csv(f"{out}/r6_lr_rashomon_coefs_summary.csv", index=False, float_format="%.6g")
pd.DataFrame(coef).to_csv(f"{out}/r6_lr_rashomon_coefs.csv", index=False, float_format="%.10g")
pd.set_option("display.width", 250); print(T.to_string(index=False))
