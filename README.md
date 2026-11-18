# Do Near-Equivalent Models Tell the Same Story? Reliability of SHAP and LIME Explanations for Self-Reported Diabetes in BRFSS 2015

Bayzeed Shakib, Gazi MD Rafayet Hossen, A.K.M Ashiqur Rahaman Shaikat, Syed Ashraf
Department of Computer Science, American International University-Bangladesh (AIUB), Dhaka, Bangladesh

Code and result records for the paper. Every number in the paper comes from a row of `results_registry.csv`, and each row names the script or notebook that produced it.

## Contents

| Folder / file | Contents |
|---|---|
| `stage1_*.ipynb` … `stage7_*.ipynb` | Thesis pipeline, split 1: partition, timing pilot, tuning (30 configurations per model, 5-fold CV, AP), re-evaluation and freezing, test evaluation, SHAP/LIME, timing |
| `paper_r4_splits.ipynb`, `paper_r5_shap.ipynb`, `paper_r6_refits.ipynb` | Paper phase: splits 2–5 (H1), SHAP on the 500-row cores (H2), bootstrap, seed-only and Rashomon refits (H3) |
| `thesis_common.py`, `env_setup.py` | Shared functions (data loading, features, seeds) and environment check |
| `pack_r4_outputs.py`, `pack_r5_outputs.py`, `pack_r6_outputs.py` | Pack the R4–R6 outputs with SHA-256 lists |
| `requirements.txt` | Package versions |
| `search_configs/`, split index files (splits 1–5), selected configurations, small result tables | Inputs and outputs of the notebooks |
| `analysis/` | Scripts run outside the notebooks (P0–P7): label checks, baselines and ablations (R1), the LIME grid and CholCheck test (R2: H4, H5, E1), prediabetes rows (R3: E2), H1 tests and checks (R4), H2 agreement and bootstrap (R5), the H3 permutation test and R6 summaries, Platt calibration (P7); `run_analysis.sh` gives every command line |
| `paper/` | `build_tables.py` (Tables I–IV, each cell asserted against the registry), `make_figures.py` (Figs. 1–3), `trace_claims.py`, `wordcount.py`, `page_check.py`, `fig2_plotted_values.csv`, `fig3_plotted_values.csv`, `table_trace.csv`; `run_paper.sh` |
| `results_registry.csv` | 671 result rows (ID, description, value, script, output file, date, machine) |
| `phase4_design.md` | Research design with hypotheses H1–H5 and decision rules, stated before the tests were run, with dated amendments |

The notebooks, `thesis_common.py`, the pack scripts, `requirements.txt` and the data/index folders form the PC part; `analysis/`, `paper/`, the registry and the design form the workspace part. In the registry, `output_file` paths of the form `claude/paper/<phase>/…` correspond to `analysis_outputs/<phase>/…` written by `analysis/run_analysis.sh`. Decision codes (D…, O…) in comments refer to the authors' internal project log, which is not included.

## Data

The data file is not redistributed. Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the Kaggle "Diabetes Health Indicators Dataset" (A. Teboul, cleaned BRFSS 2015): https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset . The same data are described in the UCI Machine Learning Repository, "CDC Diabetes Health Indicators", https://archive.ics.uci.edu/dataset/891 (doi:10.24432/C53919); UCI serves them through `ucimlrepo`, not as this CSV file, so use the Kaggle file to match the hash.

Put the file in `data/` and check it:

```
SHA-256  19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0  diabetes_binary_health_indicators_BRFSS2015.csv
```

The label checks (P0) and the prediabetes analyses (R1, R3) also need the three-class file from the same Kaggle page:

```
SHA-256  a971809d0786d2d7d7f6070f83dfe8af9d860ea33a79e0e2701c8a49f78f9861  diabetes_012_health_indicators_BRFSS2015.csv
```

## How to run

All commands run from the repository root.

1. `pip install -r requirements.txt` (the workspace scripts also need `statsmodels` and, for `page_check.py`, `pdfplumber`, if they are not already listed).
2. Notebooks in order: `stage1` → `stage2` → `stage3` → `stage4` → `stage5` → `stage6` → `stage7`, then `paper_r4_splits` → `paper_r5_shap` → `paper_r6_refits`.
3. `bash analysis/run_analysis.sh` (set `N_JOBS` for the LIME grid; the grid is the longest step).
4. `bash paper/run_paper.sh` (tables and figures into `paper_build/`). `trace_claims.py`, `wordcount.py` and `page_check.py` run on the LaTeX source of the paper, which will be added after acceptance.

## Hardware and reproducibility

- PC: AMD Ryzen 5 5500U, 12 logical CPUs, 16 GB RAM, CPU only, Python 3.13. Seeds are fixed in `thesis_common.py` (master seed 42; the paper splits derive their seeds from it with `numpy.random.SeedSequence`).
- CatBoost needs `thread_count = 12` to reproduce the reported models.
- The XGBoost fits were made on Windows; LightGBM and random forest reproduce exactly on Linux.
- Saved models (`.joblib`) and per-row SHAP files (about 4 GB) are not included; the notebooks rebuild them. The analysis scripts that read models or SHAP files need these rebuilt outputs.
- In printed paths in notebook outputs, the Windows user name was replaced by `USER`.

## How to cite

[Paper reference to be completed after acceptance.]

## License

Code: MIT (see `LICENSE`). The dataset is not redistributed; its own terms apply.
