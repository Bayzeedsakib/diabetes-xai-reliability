# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""P6-018: size and positives of the 500-row explanation core in each split, read from the R5 common SHAP files."""
import sys, pandas as pd
data, x, out = sys.argv[1], sys.argv[2], sys.argv[3]
y = pd.read_csv(data, usecols=["Diabetes_binary"]).Diabetes_binary.to_numpy().astype(int)
rows = []
for s in range(1, 6):
    ids = {m: pd.read_csv(f"{x}/paper_r5_outputs/split{s}/common/SHAP_{m}.csv", usecols=["row_id"]).row_id for m in ["CAT", "XGB", "LGBM", "MLP", "LR", "RF"]}
    ref = ids["CAT"]
    assert all(v.equals(ref) for v in ids.values()), "core rows differ between models"
    rows.append({"split": s, "core_rows": ref.nunique(), "core_positives": int(y[ref.unique()].sum()), "same_rows_all_six_models": True})
pd.DataFrame(rows).to_csv(out, index=False); print(pd.DataFrame(rows).to_string(index=False))
