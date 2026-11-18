#!/usr/bin/env bash
# Workspace analyses of the paper, in dependency order. Run from the repository root after the notebooks
# (stage1-stage7, paper_r4_splits, paper_r5_shap, paper_r6_refits) have produced their output folders.
# Outputs go to analysis_outputs/<phase>/ (the registry's output_file paths claude/paper/<phase>/... map to these).
set -euo pipefail
PY="python -I"
BIN=data/diabetes_binary_health_indicators_BRFSS2015.csv
TRI=data/diabetes_012_health_indicators_BRFSS2015.csv
TP=stage5_outputs/test_predictions.csv
FZ=stage4_outputs/frozen_models.json
PIPES=stage3_outputs/models
IDX=stage1_outputs/indices
PANEL=stage6_outputs/case_panel.csv
O=analysis_outputs
N_JOBS=${N_JOBS:-2}
mkdir -p $O/p0 $O/r1 $O/r1b $O/r2/input_check $O/r2/grid $O/r2/chol50 $O/r2/analysis $O/r3 $O/r3b \
         $O/indices_v2 $O/r4 $O/r5/boot $O/r6/descriptive $O/r6/split1_sets $O/r6/lr_rashomon $O/p7

# P0 label checks
$PY analysis/p0_label_check.py $BIN $TRI $O/p0/p0_label_check.csv
# Split-1 checks: thread reproduction and rebuilt index files
$PY analysis/thread_check.py . $BIN $TP
$PY analysis/rebuild_indices_v2.py . $BIN $TP $IDX/MANIFEST.csv $O/indices_v2
# R1 baselines and ablations (split 1)
$PY analysis/r1_baselines_ablations.py . $BIN $TRI $TP $O/r1
$PY analysis/r1b_cat12_profile5.py . $BIN $TRI $TP $O/r1b
# R2 SHAP-LIME (H4, H5, E1): input check, LIME grid, CholCheck SHAP, analysis, follow-up, independent check
$PY analysis/r2_input_check.py . $BIN $TP $FZ $PIPES $O/r2/input_check
$PY analysis/r2_lime_grid.py . $BIN $FZ $PIPES $IDX $PANEL $O/r2/grid $N_JOBS
$PY analysis/r2_shap_chol50.py . $BIN $FZ $PIPES $IDX $PANEL stage6_outputs/shap $O/r2/chol50 $N_JOBS
$PY analysis/r2_analysis.py . $BIN $IDX $PANEL stage6_outputs $O/r2/grid $O/r2/chol50 $O/r2/analysis
$PY analysis/r2_followup.py $O/r2
$PY analysis/r2_verify.py
# R3 prediabetes rows (E2) and prevalence control
$PY analysis/r3_prediabetes.py $BIN $TRI $TP $O/r3
$PY analysis/r3b_prevalence_control.py $BIN $TRI $TP $O/r3b
# R4 (H1): equivalence tests, independent checks
$PY analysis/r4_h1.py . paper_r4_outputs $TP $O/r4
$PY analysis/a_checks.py $O/r4 $O/r4/h1_per_split.csv $O/r4/h1_across_splits_nb.csv \
    paper_r4_outputs/r4_models_summary.csv paper_r4_outputs/r4_baselines_summary.csv $TP $O/r4/a_checks.json
$PY analysis/r4_verify.py . $BIN paper_r4_outputs 2,3,4,5
$PY analysis/r4_split2_refit_check.py . $BIN paper_r4_outputs $O/r4/r4_split2_refit_check.csv
# Optional: needs the returned R4 zip files (r4_*.zip) in r4_upload/
# $PY analysis/r4_rebuild_check.py . $BIN paper_r4_outputs r4_upload/SHA256SUMS_r4.txt r4_upload $O/r4/r4_rebuild_check.csv
# R5 (H2): agreement, descriptive top-6 orders, bootstrap, independent check
$PY analysis/r5_h2.py . paper_r5_outputs $O/r5
$PY analysis/r5_top6_orders.py $O/r5
$PY analysis/r5_h2_boot.py . paper_r5_outputs . paper_r4_outputs/indices $O/r5/boot
$PY analysis/r5_verify.py . . paper_r4_outputs paper_r5_outputs 1,2,3,4,5 $O/r5/r5_verify.csv
# R6 (H3): permutation test, independent check, distinct-T count, descriptive summaries
$PY analysis/r6_h3.py . $O/r6
$PY analysis/r6_h3_check.py . $O/r6
$PY analysis/r6_h3_distinct_T.py $O/r6
$PY analysis/r6_descriptive.py . $PIPES $O/r6/descriptive
$PY analysis/r6_split1_sets.py . $O/r6/split1_sets
$PY analysis/r6_lr_rashomon_coef.py . $BIN $PIPES/LR/cv_results_LR.csv $O/r6/lr_rashomon
$PY analysis/r6_core_positives.py $BIN . $O/r6/r6_core_positives.csv
# P7 Platt calibration (split 1)
$PY analysis/p7_platt.py --project-dir . --dataset $BIN --out $O/p7
