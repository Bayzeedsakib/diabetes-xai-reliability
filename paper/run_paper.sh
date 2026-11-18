#!/usr/bin/env bash
# Paper tables, figures and checks. Run from the repository root after analysis/run_analysis.sh.
set -euo pipefail
PY="python -I"
mkdir -p paper_build/figure_inputs
# Tables I-IV: every cell is asserted against results_registry.csv; writes table*.tex and table_trace.csv
$PY paper/build_tables.py results_registry.csv paper_build
# Figures 1-3: inputs are R6 (H3) and R2 outputs; writes fig*.pdf/.png and fig2/fig3_plotted_values.csv
cp analysis_outputs/r6/h3_rbo_values.csv analysis_outputs/r6/h3_pairs.csv \
   analysis_outputs/r2/analysis/r2_summary_by_setting.csv analysis_outputs/r2/analysis/r2_summary_by_setting_model.csv \
   paper_build/figure_inputs/
$PY paper/make_figures.py paper_build/figure_inputs paper_build
# The next three need the paper's LaTeX source (main.tex), which is added to the repository after acceptance:
#   $PY paper/trace_claims.py main.tex results_registry.csv claim_trace.csv   # every number in the text vs the registry
#   $PY paper/wordcount.py main.tex word_count.csv
#   (cd <latex dir> && $PY page_check.py)                                      # after compiling main.tex
