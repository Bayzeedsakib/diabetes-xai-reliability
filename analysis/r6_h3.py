# Repository version: only file paths were adapted to the repository layout (run from the repository root); the analysis logic is unchanged.
"""H3 (paper Phase 5; phase4_design.md sections 1, 5 and 12 item 4; implementation details D40 part 1,
recorded in decisions_log.md on 2026-11-15 before this script was run).

H3: disagreement between members exceeds within-model refit disagreement.
- Refit sets: R6 bootstrap refits only (10 per model in split 1, 5 in splits 2-5); the R5 model is not included.
- Ranking per refit: mean |SHAP| over the split's 500-row core, ties by feature order (stable sort).
- RBO: extrapolated RBO, p = 0.8, full 21-predictor rankings (r2_analysis.py / r5_h2.py function, unchanged).
- within = all unordered refit pairs i < j of one model; between = all A x B refit pairs.
- T = min(median within A, median within B) - median between.
- Pass in a split: median between < both within medians AND one-sided permutation p < 0.05.
  Splits 2-5: all C(10,5) = 252 label assignments. Split 1: fresh default_rng(1012652155) per pair x setting,
  9,999 assignments (first 10 of rng.permutation(20) = A) + the observed one; p = #(T >= T_obs - 1e-12) / N.
- Decision: CAT-MLP passes in >= 4 of 5 splits, common setting. Native reported. All six pairs reported.
- Approximation floor (design 12 item 9, D40 item 3): refit disagreement = 1 - within median vs 0.000 (P5-008).
usage: python -I r6_h3.py PROJECT_ROOT OUTDIR   (PROJECT_ROOT = repository root, holding paper_r6_outputs/)
"""
import hashlib, itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau

root, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
F = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
     "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
     "Sex", "Age", "Education", "Income"]           # thesis_common.FEATURES order (= SHAP file columns, asserted)
PAIRS = [("CAT", "MLP", "primary"), ("XGB", "MLP", "secondary"), ("LGBM", "MLP", "secondary"),
         ("CAT", "XGB", "expected_to_fail"), ("CAT", "LGBM", "expected_to_fail"), ("XGB", "LGBM", "expected_to_fail")]
MEM = ["CAT", "XGB", "LGBM", "MLP"]
TOL = 1e-12
NB = {1: 10, 2: 5, 3: 5, 4: 5, 5: 5}

seeds = json.load(open(os.path.join(root, "paper_r6_outputs", "seeds_r6.json")))
paper_r6 = seeds["paper_r6"]
assert paper_r6 == int(np.random.SeedSequence(42).generate_state(16)[15]) == 1741519417
w18 = np.random.SeedSequence([paper_r6, 1]).generate_state(18)
assert [int(x) for x in w18[:17]] == seeds["per_split_words"]["1"]
PERM_SEED_S1 = int(w18[17]); assert PERM_SEED_S1 == 1012652155


def rbo(a, b, p):                                   # identical to r2_analysis.py / r5_h2.py
    sa, sb, s = set(), set(), 0.0
    for d in range(1, len(a) + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1]); s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s + p ** len(a) * len(sa & sb) / len(a)


def order(v):
    return np.argsort(-v, kind="stable")


core_ids = {}


def load(k, setting, tag, m):
    rec = json.load(open(os.path.join(root, "paper_r6_outputs", f"split{k}", setting, f"SHAP_{tag}_{m}.done.json")))
    path = os.path.join(root, "paper_r6_outputs", rec["file"])
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == rec["sha256"]
    df = pd.read_csv(path, float_precision="round_trip")
    assert list(df.columns[4:]) == F and len(df) == 500 and rec["row_set"] == "core"
    ids = tuple(df.row_id)
    assert core_ids.setdefault(k, ids) == ids, "core rows differ between refits"
    v = df[F].abs().mean().to_numpy()
    return v / v.sum(), rec


# ---- per-refit global rankings (all R6 SHAP files: boot, seed, rash; both settings where present) ----
rank_rows, I = [], {}
for k in range(1, 6):
    for setting in ["common", "native"]:
        d = os.path.join(root, "paper_r6_outputs", f"split{k}", setting)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".done.json"):
                continue
            tag, m = fn[len("SHAP_"):-len(".done.json")].rsplit("_", 1)
            v, rec = load(k, setting, tag, m)
            I[(k, setting, tag, m)] = v
            o = order(v)
            rank_rows.append({"split": k, "setting": setting, "unit": rec["unit"], "kind": rec["kind"], "tag": tag,
                              "model": m, "config_id": rec["config_id"],
                              **{f"rank{j + 1:02d}": F[i] for j, i in enumerate(o)},
                              **{f"imp_{f}": v[j] for j, f in enumerate(F)}})
R = pd.DataFrame(rank_rows)
R.to_csv(f"{out}/h3_refit_rankings.csv", index=False, float_format="%.10g")


def boot_set(k, setting, m):
    return [order(I[(k, setting, f"boot{i:02d}", m)]) for i in range(1, NB[k] + 1)]


def stat_from_matrix(M, A, B):
    wa = [M[i, j] for i, j in itertools.combinations(A, 2)]
    wb = [M[i, j] for i, j in itertools.combinations(B, 2)]
    bt = [M[i, j] for i in A for j in B]
    ma, mb, mbt = np.median(wa), np.median(wb), np.median(bt)
    return min(ma, mb) - mbt, ma, mb, mbt, wa, wb, bt


pair_rows, rbo_rows = [], []
for k in range(1, 6):
    n = NB[k]
    for setting in ["common", "native"]:
        for a, b, role in PAIRS:
            pool = boot_set(k, setting, a) + boot_set(k, setting, b)
            N2 = len(pool)
            M = np.ones((N2, N2))
            for i, j in itertools.combinations(range(N2), 2):
                M[i, j] = M[j, i] = rbo(pool[i], pool[j], 0.8)
            A, B = list(range(n)), list(range(n, 2 * n))
            T, ma, mb, mbt, wa, wb, bt = stat_from_matrix(M, A, B)
            if k == 1:
                rng = np.random.default_rng(PERM_SEED_S1)
                Ts = [T]
                for _ in range(9999):
                    pa = sorted(rng.permutation(N2)[:n].tolist())
                    pb = [x for x in range(N2) if x not in pa]
                    Ts.append(stat_from_matrix(M, pa, pb)[0])
                method = "sampled 9,999 + observed"
            else:
                Ts = []
                for pa in itertools.combinations(range(N2), n):
                    pb = [x for x in range(N2) if x not in pa]
                    Ts.append(stat_from_matrix(M, list(pa), pb)[0])
                method = "exact 252"
            Ts = np.array(Ts)
            p = float(np.mean(Ts >= T - TOL))
            med_ok = bool(mbt < ma and mbt < mb)
            pair_rows.append({"split": k, "setting": setting, "model_a": a, "model_b": b, "role": role,
                              "n_refits_each": n, "n_within_a": len(wa), "n_between": len(bt),
                              "median_within_a": ma, "median_within_b": mb, "median_between": mbt,
                              "min_within_a": min(wa), "min_within_b": min(wb), "min_between": min(bt),
                              "T": T, "p_perm": p, "n_assignments": len(Ts), "perm_method": method,
                              "median_condition": med_ok, "pass": bool(med_ok and p < 0.05),
                              "refit_disagreement_a": 1 - ma, "refit_disagreement_b": 1 - mb,
                              "between_disagreement": 1 - mbt})
            for (i, j) in itertools.combinations(range(N2), 2):
                typ = f"within_{a}" if j < n else (f"within_{b}" if i >= n else "between")
                rbo_rows.append({"split": k, "setting": setting, "pair": f"{a}-{b}", "i": i, "j": j, "type": typ,
                                 "rbo08": M[i, j]})
P = pd.DataFrame(pair_rows)
P.to_csv(f"{out}/h3_pairs.csv", index=False, float_format="%.10g")
pd.DataFrame(rbo_rows).to_csv(f"{out}/h3_rbo_values.csv", index=False, float_format="%.10g")

# ---- decision ----
res = {"permutation_seed_split1": PERM_SEED_S1, "tie_tolerance": TOL, "refit_sets": "R6 bootstrap refits only"}
for setting in ["common", "native"]:
    res[setting] = {}
    for a, b, role in PAIRS:
        x = P[(P.setting == setting) & (P.model_a == a) & (P.model_b == b)]
        res[setting][f"{a}-{b}"] = {"role": role, "splits_passed": int(x["pass"].sum()),
                                    "passed_in": x.loc[x["pass"], "split"].tolist(),
                                    "holds_ge4of5": bool(x["pass"].sum() >= 4)}
res["decision"] = {"setting": "common", "primary_pair": "CAT-MLP",
                   "H3_supported": res["common"]["CAT-MLP"]["holds_ge4of5"]}

# ---- approximation floor (design 12 item 9; D40 item 3): within-refit disagreement per model ----
fl = []
for k in range(1, 6):
    for setting in ["common", "native"]:
        for m in MEM + (["LR", "RF"] if k == 1 else []):
            s = boot_set(k, setting, m)
            r = [rbo(s[i], s[j], 0.8) for i, j in itertools.combinations(range(len(s)), 2)]
            fl.append({"split": k, "setting": setting, "model": m, "n_refits": len(s), "median_within_rbo": np.median(r),
                       "min_within_rbo": min(r), "refit_disagreement": 1 - np.median(r),
                       "share_pairs_identical_ranking": np.mean(np.isclose(r, 1.0, atol=0, rtol=0) | (np.array(r) == 1.0)),
                       "approx_floor_disagreement": 0.0, "above_floor": bool(1 - np.median(r) > 0)})
FL = pd.DataFrame(fl); FL.to_csv(f"{out}/h3_refit_floor.csv", index=False, float_format="%.10g")
res["floor"] = {"approx_disagreement_P5_008": 0.0,
                "common_member_refit_disagreement_range": [float(FL[(FL.setting == "common") & FL.model.isin(MEM)].refit_disagreement.min()),
                                                           float(FL[(FL.setting == "common") & FL.model.isin(MEM)].refit_disagreement.max())]}
json.dump(res, open(f"{out}/h3_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(P[["split", "setting", "model_a", "model_b", "median_within_a", "median_within_b", "median_between", "T", "p_perm",
         "median_condition", "pass"]].round(4).to_string(index=False))
print(json.dumps(res, indent=1, default=float))
