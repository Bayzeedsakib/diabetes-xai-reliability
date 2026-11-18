"""Shared code for all thesis stages (Stage 2 onward).

Explainable AI for Early Diabetes Prediction — CDC BRFSS 2015.
Every stage imports this module so that data loading, index files, seeds,
pipelines and search spaces are defined once.

Decisions implemented: Ch3 3.2.3-3.2.5, 3.4; D5/P4 (search spaces), D14 (KNN/MLP
subsample), S1/S5 (frozen index files, verified by hash), S2 (seeds), S7 (pilot design).
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- constants
CONFIG_VERSION = "v1.0"
MASTER_SEED = 42
DATA_SHA256 = "19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0"
TARGET = "Diabetes_binary"
FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
# Ch3 3.2.5: the seven non-binary predictors are standardized for scale-sensitive models
NON_BINARY = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
N_ITER = 30           # Ch3 3.4.2
N_FOLDS = 5
POS_WEIGHT_MAX = 6.18  # 218,334 / 35,346 (registry S1-054)

SEED_STREAMS = ["partition", "folds", "subsample_50k", "shap_background", "shap_perm_background",
                "global_sample", "perm_core", "stability_bootstrap", "search_sampling",
                "estimator_init", "eval_bootstrap", "lime_case_selection", "lime_perturbation"]
SEEDS = {k: int(v) for k, v in zip(SEED_STREAMS,
                                    np.random.SeedSequence(MASTER_SEED).generate_state(len(SEED_STREAMS)))}
# LightGBM and CatBoost take a signed 32-bit seed; reduce every estimator seed the same way (S7).
MODEL_SEED = SEEDS["estimator_init"] % (2 ** 31 - 1)

CANDIDATE_NAMES = ["diabetes_binary_health_indicators_BRFSS2015.csv",
                   "diabetes_binary_health_indicators_BRFSS2015 1.csv",
                   "diabetes_binary_health_indicators_BRFSS2015_1.csv"]

MODELS = ["LR", "DT", "RF", "SVC", "KNN", "GNB", "XGB", "LGBM", "CAT", "MLP"]
MODEL_NAMES = {"LR": "Logistic Regression", "DT": "Decision Tree", "RF": "Random Forest",
               "SVC": "Linear SVC", "KNN": "K-Nearest Neighbors", "GNB": "Gaussian Naive Bayes",
               "XGB": "XGBoost", "LGBM": "LightGBM", "CAT": "CatBoost", "MLP": "Multilayer Perceptron"}
SCALED = {"LR", "SVC", "KNN", "MLP"}          # Ch3 3.2.5, Table 3.1
SUBSAMPLE_TUNED = {"KNN", "MLP"}              # D14
# Models whose estimator uses all cores internally (search runs folds sequentially);
# the others are single-threaded and can run folds in parallel.
MULTITHREADED = {"RF", "XGB", "LGBM", "CAT", "KNN", "MLP"}


# ---------------------------------------------------------------- data and indices
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_data(project_dir):
    dirs = [os.path.join(project_dir, "data"), project_dir, "/content"]
    for d in dirs:
        for n in CANDIDATE_NAMES:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"Dataset not found in {dirs}")


def load_data(project_dir):
    """Read the raw CSV (never modified) and check it is the canonical file."""
    path = find_data(project_dir)
    if sha256_file(path) != DATA_SHA256:
        raise SystemExit(f"{path} is not the canonical dataset (SHA-256 mismatch).")
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET].astype(int).to_numpy()


def load_indices(project_dir):
    """Load the frozen Stage 1 index files after checking each against MANIFEST.csv (S1a, S5)."""
    idx_dir = os.path.join(project_dir, "stage1_outputs", "indices")
    manifest = pd.read_csv(os.path.join(idx_dir, "MANIFEST.csv"))
    out = {}
    for r in manifest.itertuples(index=False):
        p = os.path.join(idx_dir, r.file)
        if sha256_file(p) != r.sha256:
            raise SystemExit(f"Index file {r.file} differs from MANIFEST.csv. Restore the canonical file.")
        out[r.file] = pd.read_csv(p)
    return out


def fold_splits(fold_df):
    """(train_positions, val_positions) per fold, positions relative to fold_df rows."""
    f = fold_df["fold"].to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(N_FOLDS)]


# ---------------------------------------------------------------- pipelines
def make_estimator(model, n_jobs=-1):
    """Unfitted classifier with fixed seed and threading; hyperparameters come from the search."""
    if model == "LR":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver="liblinear", max_iter=1000, random_state=MODEL_SEED)
    if model == "DT":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=MODEL_SEED)
    if model == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=n_jobs, random_state=MODEL_SEED)
    if model == "SVC":
        from sklearn.svm import LinearSVC
        return LinearSVC(loss="squared_hinge", dual=False, max_iter=5000, random_state=MODEL_SEED)
    if model == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(algorithm="brute", n_jobs=n_jobs)
    if model == "GNB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    if model == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", n_jobs=n_jobs, random_state=MODEL_SEED, verbosity=0)
    if model == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(subsample_freq=1, n_jobs=n_jobs, random_state=MODEL_SEED, verbose=-1)
    if model == "CAT":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(thread_count=n_jobs, random_seed=MODEL_SEED, verbose=0,
                                  allow_writing_files=False)
    if model == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(activation="relu", solver="adam", early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=10, max_iter=200,
                             random_state=MODEL_SEED)
    raise ValueError(model)


def make_pipeline(model, n_jobs=-1):
    """Leakage-safe pipeline: scaling (fit inside each training fold) only for scale-sensitive models."""
    if model in SCALED:
        prep = ColumnTransformer([("scale", StandardScaler(), NON_BINARY)], remainder="passthrough",
                                 verbose_feature_names_out=False)
    else:
        prep = "passthrough"
    return Pipeline([("prep", prep), ("clf", make_estimator(model, n_jobs))])


def positive_score(pipe, X):
    """Score for the positive class: probability, or the decision function for the uncalibrated SVC."""
    if hasattr(pipe, "predict_proba"):   # False for LinearSVC, which only has decision_function
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


# ---------------------------------------------------------------- search spaces (D5, P4)
# Integers for counts, depths and leaves; continuous ranges for real-valued regularizers (S7).
SPW = [1.0, 2.0, 4.0, POS_WEIGHT_MAX]
SEARCH_SPACES = {
    "LR": {"clf__C": loguniform(1e-4, 1e2), "clf__penalty": ["l1", "l2"],
           "clf__class_weight": [None, "balanced"]},
    "DT": {"clf__max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20],
           "clf__min_samples_leaf": [1, 5, 10, 20, 50, 100, 200], "clf__min_samples_split": [2, 10, 50],
           "clf__criterion": ["gini", "entropy"], "clf__class_weight": [None, "balanced"]},
    "RF": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [8, 12, 16],
           "clf__min_samples_leaf": [1, 5, 10, 20, 50], "clf__max_features": ["sqrt", "log2", 0.5],
           "clf__max_samples": [0.5, 0.8, None],
           "clf__class_weight": [None, "balanced", "balanced_subsample"]},
    "SVC": {"clf__C": loguniform(1e-4, 1e1), "clf__penalty": ["l1", "l2"],
            "clf__class_weight": [None, "balanced"]},
    "KNN": {"clf__n_neighbors": [5, 11, 25, 51, 101, 151, 201, 251], "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]},
    "GNB": {"clf__var_smoothing": loguniform(1e-12, 1e-2)},
    "XGB": {"clf__n_estimators": randint(100, 601), "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__max_depth": randint(3, 9), "clf__min_child_weight": uniform(1, 9),
            "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__reg_lambda": loguniform(0.01, 10), "clf__scale_pos_weight": SPW},
    "LGBM": {"clf__n_estimators": randint(100, 601), "clf__learning_rate": loguniform(0.01, 0.3),
             "clf__num_leaves": randint(15, 256), "clf__min_child_samples": randint(10, 201),
             "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.6, 0.4),
             "clf__reg_lambda": loguniform(0.01, 10), "clf__scale_pos_weight": SPW},
    "CAT": {"clf__iterations": randint(200, 601), "clf__learning_rate": loguniform(0.02, 0.3),
            "clf__depth": randint(4, 9), "clf__l2_leaf_reg": uniform(1, 9), "clf__scale_pos_weight": SPW},
    "MLP": {"clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
            "clf__alpha": loguniform(1e-6, 1e-2), "clf__learning_rate_init": loguniform(1e-4, 1e-2),
            "clf__batch_size": [256, 512, 1024]},
}

# Boundary configurations for the timing pilot (S7): cheapest and most expensive corner of each space.
# Values a model's run time does not depend on are set to a mid value.
PILOT_CONFIGS = {
    "LR": {"min": {"clf__C": 1e-4, "clf__penalty": "l2", "clf__class_weight": None},
           "max": {"clf__C": 1e2, "clf__penalty": "l1", "clf__class_weight": "balanced"}},
    "DT": {"min": {"clf__max_depth": 3, "clf__min_samples_leaf": 200, "clf__min_samples_split": 50,
                   "clf__criterion": "gini", "clf__class_weight": None},
           "max": {"clf__max_depth": 20, "clf__min_samples_leaf": 1, "clf__min_samples_split": 2,
                   "clf__criterion": "entropy", "clf__class_weight": "balanced"}},
    "RF": {"min": {"clf__n_estimators": 100, "clf__max_depth": 8, "clf__min_samples_leaf": 50,
                   "clf__max_features": "log2", "clf__max_samples": 0.5, "clf__class_weight": None},
           "max": {"clf__n_estimators": 300, "clf__max_depth": 16, "clf__min_samples_leaf": 1,
                   "clf__max_features": 0.5, "clf__max_samples": None,
                   "clf__class_weight": "balanced_subsample"}},
    "SVC": {"min": {"clf__C": 1e-4, "clf__penalty": "l2", "clf__class_weight": None},
            "max": {"clf__C": 1e1, "clf__penalty": "l1", "clf__class_weight": "balanced"}},
    "KNN": {"min": {"clf__n_neighbors": 5, "clf__weights": "uniform", "clf__p": 2},
            "max": {"clf__n_neighbors": 251, "clf__weights": "distance", "clf__p": 1}},
    "GNB": {"min": {"clf__var_smoothing": 1e-9}, "max": {"clf__var_smoothing": 1e-9}},
    "XGB": {"min": {"clf__n_estimators": 100, "clf__learning_rate": 0.1, "clf__max_depth": 3,
                    "clf__min_child_weight": 5.0, "clf__subsample": 0.6, "clf__colsample_bytree": 0.6,
                    "clf__reg_lambda": 1.0, "clf__scale_pos_weight": 1.0},
            "max": {"clf__n_estimators": 600, "clf__learning_rate": 0.1, "clf__max_depth": 8,
                    "clf__min_child_weight": 1.0, "clf__subsample": 1.0, "clf__colsample_bytree": 1.0,
                    "clf__reg_lambda": 1.0, "clf__scale_pos_weight": POS_WEIGHT_MAX}},
    "LGBM": {"min": {"clf__n_estimators": 100, "clf__learning_rate": 0.1, "clf__num_leaves": 15,
                     "clf__min_child_samples": 200, "clf__subsample": 0.6, "clf__colsample_bytree": 0.6,
                     "clf__reg_lambda": 1.0, "clf__scale_pos_weight": 1.0},
             "max": {"clf__n_estimators": 600, "clf__learning_rate": 0.1, "clf__num_leaves": 255,
                     "clf__min_child_samples": 10, "clf__subsample": 1.0, "clf__colsample_bytree": 1.0,
                     "clf__reg_lambda": 1.0, "clf__scale_pos_weight": POS_WEIGHT_MAX}},
    "CAT": {"min": {"clf__iterations": 200, "clf__learning_rate": 0.1, "clf__depth": 4,
                    "clf__l2_leaf_reg": 3.0, "clf__scale_pos_weight": 1.0},
            "max": {"clf__iterations": 600, "clf__learning_rate": 0.1, "clf__depth": 8,
                    "clf__l2_leaf_reg": 3.0, "clf__scale_pos_weight": POS_WEIGHT_MAX}},
    "MLP": {"min": {"clf__hidden_layer_sizes": (32,), "clf__alpha": 1e-4, "clf__learning_rate_init": 1e-2,
                    "clf__batch_size": 1024},
            "max": {"clf__hidden_layer_sizes": (128, 64), "clf__alpha": 1e-6, "clf__learning_rate_init": 1e-4,
                    "clf__batch_size": 256}},
}


def sampled_configs(model):
    """The exact candidate list RandomizedSearchCV(n_iter=30, random_state=SEEDS['search_sampling'])
    will draw (fewer if the grid has fewer than 30 combinations, Ch3 3.4.2)."""
    space = SEARCH_SPACES[model]
    n = N_ITER
    if all(isinstance(v, list) for v in space.values()):
        n = min(N_ITER, int(np.prod([len(v) for v in space.values()])))
    return list(ParameterSampler(space, n_iter=n, random_state=SEEDS["search_sampling"]))


def configs_frame(model):
    rows = []
    for i, c in enumerate(sampled_configs(model)):
        rows.append({"config_id": i, **{k.replace("clf__", ""): (repr(v) if isinstance(v, tuple) else v)
                                        for k, v in c.items()}})
    return pd.DataFrame(rows)


def write_or_verify(frame, path):
    """Create a frozen CSV, or verify it is byte-identical to the existing one (never overwrite)."""
    data = frame.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode()
    h = hashlib.sha256(data).hexdigest()
    if os.path.exists(path):
        old = hashlib.sha256(open(path, "rb").read()).hexdigest()
        return h, ("verified: identical to existing file" if old == h else "MISMATCH: existing file kept")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return h, "created"


def environment(machine):
    import platform
    from importlib import metadata
    pkgs = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap", "lime"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    return {"machine": machine, "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "packages": pkgs}


def check_pins(project_dir, versions=None):
    """Differences from the canonical requirements.txt (S6), {package: (pinned, installed)}.
    Every pinned package is checked; a missing package counts as a difference (S10).
    `versions` is ignored and kept only for compatibility with the Stage 2 notebook."""
    import env_setup
    return env_setup.pin_differences(project_dir)
