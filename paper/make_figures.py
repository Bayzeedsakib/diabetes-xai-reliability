"""Figures 1-3 for the conference paper (300-dpi PNG + vector PDF).

Fig. 1  pipeline diagram (no data).
Fig. 2  within-refit vs between-model RBO per split, 4 pairs x 2 settings
        (data: claude/paper/r6/h3_rbo_values.csv, h3_pairs.csv; registry P6-004-P6-010).
Fig. 3  SHAP-LIME agreement by LIME setting (data: claude/paper/r2/r2_summary_by_setting.csv
        and r2_summary_by_setting_model.csv; registry P2-006-P2-017).
Usage: python -I make_figures.py <data_dir> <out_dir>
Palette: categorical slots 1-3 (#2a78d6, #eb6834, #1baf7a), validated with the dataviz
validator (light mode; CVD dE 9.2; slot 3 contrast 2.74:1, so every series also has its own
marker shape and a legend).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

data, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1 pipeline
def box(ax, x, y, w, h, text, fc="#f3f2ee", ec=INK2, bold=False, fs=6.3, ls=1.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            linespacing=ls)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2, mutation_scale=6))


# D42 amendment 4 (O37): the two smallest texts (Bootstrap box, footer) raised from 5.7 / 5.9 pt to FS1 = 6.2 pt, which prints at
# 6.06 pt at the 0.9\columnwidth scale (0.977); FIG1_BBOX_IN = v4 tight bbox (inches), measured once from the v4 layout.
FS1 = 6.2
FIG1_BBOX_IN = (0.18183593750000002 + (3.2238281250000003 - 232.021 / 72) / 2, 0.2604999999999999, 232.021 / 72, 144.252 / 72)
fig, ax = plt.subplots(figsize=(3.5, 2.55))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.86, 0.96, 0.12, "BRFSS 2015 file: 253,680 respondents, 21 self-reported predictors\n"
    "positive = self-reported diagnosed diabetes (prediabetes coded 0)", fc="#e6effb", ec=BLUE)
box(ax, 0.02, 0.67, 0.96, 0.12, "Five stratified 80:20 outer splits (split first; test set used once)\n"
    "per split: 30 frozen configurations \u00d7 5-fold CV on AP, six models")
arrow(ax, 0.5, 0.86, 0.5, 0.79)
cols = [(0.015, "Test evaluation\nAP, ROC-AUC\npaired bootstrap,\nTOST, Holm", "H1\nequivalence"),
        (0.2575, "Global SHAP on\n500-row core\nnative and\ncommon setting", "H2 ranking\nagreement"),
        (0.50, "Bootstrap refits\n10 (split 1) /\n5 (splits 2\u20135)\nper model;\nwithin/between,\npermutation test", "H3 beyond\nrefit noise"),
        (0.7425, "LIME grid,\nsplit 1 only:\n11 settings \u00d7\n3 seeds, 530\ncases, vs SHAP", "H4 / H5\nSHAP\u2013LIME")]
for x, t, h in cols:
    arrow(ax, x + 0.1175, 0.67, x + 0.1175, 0.60)  # D42 amendment 4 (O37): column boxes 0.225 -> 0.235 wide
    box(ax, x, 0.30, 0.235, 0.30, t, fs=FS1 if t.startswith("Bootstrap") else 6.3, ls=1.08 if t.startswith("Bootstrap") else 1.15)
    arrow(ax, x + 0.1175, 0.30, x + 0.1175, 0.235)
    box(ax, x, 0.08, 0.235, 0.155, h, fc="#fdf0ea", ec=ORANGE, bold=True)
ax.text(0.5, 0.01, "H1\u2013H5 rules pre-stated; ablations: leakage, profile split, explainer setting, calibration",
        ha="center", va="bottom", fontsize=FS1, color=INK2, style="italic")
# D42 amendment 4 (O37): saved with the v4 tight bounding box (232.0 x 144.3 pt PDF), so the printed size is unchanged
from matplotlib.transforms import Bbox
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig1_pipeline.{ext}"), bbox_inches=Bbox.from_bounds(*FIG1_BBOX_IN))
plt.close(fig)

# ------------------------------------------------------------------ Fig. 2 within vs between
rv = pd.read_csv(os.path.join(data, "h3_rbo_values.csv"))
pairs = pd.read_csv(os.path.join(data, "h3_pairs.csv"))
PAIRS = [("CAT", "MLP"), ("XGB", "MLP"), ("LGBM", "MLP"), ("CAT", "XGB")]
SETS = ["common", "native"]
fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.35), sharex=True, sharey=True)
off = {"a": -0.22, "b": 0.0, "between": 0.22}
style = {"a": (BLUE, "o"), "b": (ORANGE, "s"), "between": (AQUA, "D")}
summary = []
for r, setting in enumerate(SETS):
    for c, (A, B) in enumerate(PAIRS):
        ax = axes[r, c]
        key = f"{A}-{B}"
        for s in range(1, 6):
            d = rv[(rv.split == s) & (rv.setting == setting) & (rv.pair == key)]
            groups = {"a": d[d.type == f"within_{A}"].rbo08, "b": d[d.type == f"within_{B}"].rbo08,
                      "between": d[d.type == "between"].rbo08}
            for g, vals in groups.items():
                col, mk = style[g]
                med, lo, hi = np.median(vals), vals.min(), vals.max()
                ax.plot([s + off[g]] * 2, [lo, hi], color=col, lw=1.0, solid_capstyle="round", zorder=2)
                ax.plot(s + off[g], med, marker=mk, ms=3.6, color=col, mec="white", mew=0.5, zorder=3)
                summary.append({"setting": setting, "pair": key, "split": s, "group": g, "n": len(vals),
                                "median": med, "min": lo, "max": hi})
            pr = pairs[(pairs.split == s) & (pairs.setting == setting) & (pairs.model_a == A) & (pairs.model_b == B)].iloc[0]
            # cross-check: medians drawn = medians in h3_pairs.csv (P6-004-P6-007 source)
            assert abs(np.median(groups["between"]) - pr.median_between) < 1e-9
            assert abs(np.median(groups["a"]) - pr.median_within_a) < 1e-9 and abs(np.median(groups["b"]) - pr.median_within_b) < 1e-9
            if bool(pr["pass"]):
                ax.text(s, 1.005, "*", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0.6, 1.03)
        ax.set_xticks(range(1, 6))
        ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if r == 0:
            ax.set_title(f"A = {A}, B = {B}", pad=7)
        if c == 0:
            ax.set_ylabel(f"{setting.capitalize()} setting\nRBO ($p=0.8$)")
        if r == 1:
            ax.set_xlabel("Split")
h = [plt.Line2D([], [], color=BLUE, marker="o", ms=3.6, lw=1.0, label="within A (refits of A)"),
     plt.Line2D([], [], color=ORANGE, marker="s", ms=3.6, lw=1.0, label="within B (refits of B)"),
     plt.Line2D([], [], color=AQUA, marker="D", ms=3.6, lw=1.0, label="between A and B"),
     plt.Line2D([], [], color="none", marker="$*$", ms=5, mec=INK, label="H3 pass (median condition and $p<0.05$)")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.045), handlelength=1.6)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.4)
save(fig, "fig2_within_between")
pd.DataFrame(summary).to_csv(os.path.join(out, "fig2_plotted_values.csv"), index=False, float_format="%.6f")

# ------------------------------------------------------------------ Fig. 3 SHAP-LIME by setting
S = pd.read_csv(os.path.join(data, "r2_summary_by_setting.csv"))
_mf = os.path.join(data, "r2_summary_by_setting_model.csv")
M = pd.read_csv(_mf if os.path.exists(_mf) else os.path.join(data, "r2_summary_by_setting_model_extract.csv"))
order = ["T5", "T10", "N_quartile_0.5", "N_quartile_1", "N_quartile_2", "N_decile_0.5", "N_decile_1", "N_decile_2",
         "N_none_0.5", "N_none_1", "N_none_2", "T5_chol50"]
lab = {"T5": "T5 (thesis)", "T10": "T10", "T5_chol50": "T5, CholCheck 50/50"}
for k in order:
    if k.startswith("N_"):
        _, disc, w = k.split("_")
        lab[k] = f"N {disc}, width \u00d7{w}"
# D42 amendment 4 (O37): drawn at its printed size (\columnwidth = 252 pt; height 166.8 pt, the v4 printed aspect) and saved
# without a tight bounding box, so it is included at scale 1.0; every text is >= 6.5 pt (FS3); R^2 uses the character \u00b2.
FS3 = 6.5
fig, ax = plt.subplots(figsize=(252 / 72, 166.8 / 72))
y = np.arange(len(order))[::-1]
rows = []
for yi, k in zip(y, order):
    s = S[S.setting == k].iloc[0]
    m = M[M.setting == k]
    lo, hi = m.rbo08.min(), m.rbo08.max()
    ax.plot([lo, hi], [yi + 0.12] * 2, color=BLUE, lw=1.0, alpha=0.55, solid_capstyle="round")
    ax.plot(s.rbo08, yi + 0.12, "o", color=BLUE, ms=4, mec="white", mew=0.5)
    slo, shi = m.seed_rbo08.min(), m.seed_rbo08.max()
    ax.plot([slo, shi], [yi - 0.12] * 2, color=ORANGE, lw=1.0, alpha=0.55, solid_capstyle="round")
    ax.plot(s.seed_rbo08, yi - 0.12, "s", color=ORANGE, ms=3.6, mec="white", mew=0.5)
    ax.text(1.005, yi, f"{s.lime_r2:.2f}", va="center", ha="left", fontsize=FS3, color=INK2)
    rows.append({"setting": k, "shap_lime_rbo_median": s.rbo08, "model_min": lo, "model_max": hi,
                 "seed_rbo_median": s.seed_rbo08, "seed_model_min": slo, "seed_model_max": shi, "lime_r2_median": s.lime_r2})
ax.text(1.005, len(order) - 0.3, "LIME R\u00b2", va="bottom", ha="left", fontsize=FS3, color=INK2)
ax.axhline(0.5, color=GRID, lw=0.6); ax.axhline(len(order) - 2.5, color=GRID, lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order], fontsize=FS3); ax.tick_params(axis="x", labelsize=FS3, pad=1.5); ax.tick_params(axis="y", pad=1.5)
ax.set_xlim(0.40, 1.0); ax.set_xlabel("Median RBO ($p=0.8$) over 530 cases; bar = range over six models", fontsize=FS3, labelpad=1.5)
ax.grid(axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
h = [plt.Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.0, label="SHAP vs LIME"),
     plt.Line2D([], [], color=ORANGE, marker="s", ms=3.6, lw=1.0, label="LIME seed vs seed")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=2, frameon=False, fontsize=FS3, handlelength=1.6, borderaxespad=0.2)
fig.tight_layout(pad=0.15)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig3_shap_lime_settings.{ext}"), pad_inches=0)  # exact figsize, no tight bbox
plt.close(fig)
pd.DataFrame(rows).to_csv(os.path.join(out, "fig3_plotted_values.csv"), index=False, float_format="%.6f")
print("figures written:", sorted(f for f in os.listdir(out) if f.startswith("fig")))
