#!/usr/bin/env python3
"""
Supplementary Figure S4 — Magnitude-Coherence Correlation by Distance Metric

Prefer the standalone Colab driver, which scores all six datasets:

    python fig_s4_distance_metrics.py

This file only plots from existing CSVs (Euclidean / Whitened / k-NN).

INPUTS (from OUTPUT_DIR):
  crispr_correlations_with_ci.csv    — Adamson, Dixit, Norman, Replogle
                                       (from geometric_stability_main_analysis.py)
  papalexi_method_correlations.csv   — Papalexi 2021
                                       (from papalexi_method_comparison.py)

OUTPUT:
  fig_s4_method_comparison.pdf / .png
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import seaborn as sns
except ImportError:
    class _Sns:
        @staticmethod
        def despine(ax=None, **kwargs):
            if ax is not None:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
    sns = _Sns()


import pipeline_config as cfg

def _first_existing(*candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    return None


DATA_DIR = cfg.OUTPUT_DIR
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})

# =============================================================================
# LOAD DATA
# =============================================================================

main_csv = _first_existing(
    DATA_DIR / "crispr_correlations_with_ci.csv",
    Path("/content/shesha-crispr/crispr_correlations_with_ci.csv"),
    Path("./shesha-crispr/crispr_correlations_with_ci.csv"),
)
papalexi_csv = _first_existing(
    DATA_DIR / "papalexi_method_correlations.csv",
    Path("/content/shesha-crispr/papalexi_method_correlations.csv"),
    Path("./shesha-crispr/papalexi_method_correlations.csv"),
)

if main_csv is None:
    raise FileNotFoundError(
        "Could not find crispr_correlations_with_ci.csv. "
        "Run geometric_stability_main_analysis.py first."
    )

df_main = pd.read_csv(main_csv)
df_main = df_main[~df_main['dataset'].str.contains('Papalexi', case=False, na=False)].copy()
df_main['method'] = df_main['method'].str.strip().str.replace(
    r'(?i)^euclidean$', 'Euclidean', regex=True).str.replace(
    r'(?i)^whitened$',  'Whitened',  regex=True).str.replace(
    r'(?i)^k.?nn$',     'k-NN',      regex=True)

frames = [df_main]
if papalexi_csv is not None:
    df_pap = pd.read_csv(papalexi_csv)
    df_pap['dataset'] = 'Papalexi 2021 (CRISPR-KO)'
    df_pap['method'] = df_pap['method'].str.strip().str.replace(
        r'(?i)^euclidean$', 'Euclidean', regex=True).str.replace(
        r'(?i)^whitened$',  'Whitened',  regex=True).str.replace(
        r'(?i)^k.?nn$',     'k-NN',      regex=True)
    df_pap = df_pap[['dataset', 'method', 'n', 'rho', 'ci_low', 'ci_high', 'p']].copy()
    frames.append(df_pap)
    print(f"Papalexi methods: {papalexi_csv}")
else:
    print("Papalexi method CSV not found — plotting the other datasets only.")

df_all = pd.concat(frames, ignore_index=True)
if 'dataset' in df_all.columns:
    df_all['dataset'] = df_all['dataset'].map(cfg.resolve_dataset_name)

# Confirm we have data
print("Datasets found:", df_all['dataset'].unique().tolist())
print("Methods found: ", df_all['method'].unique().tolist())

# =============================================================================
# DATASET ORDER AND SHORT NAMES
# =============================================================================

DATASET_ORDER = [
    ('Adamson 2016 UPR (CRISPRi)',  'Adamson UPR'),
    ('Adamson 2016 pilot (CRISPRi)', 'Adamson pilot'),
    ('Adamson 2016 (CRISPRi)',  'Adamson'),  # legacy CSV key
    ('Dixit 2016 (CRISPR-KO)',    'Dixit'),
    ('Norman 2019 (CRISPRa)',   'Norman'),
    ('Papalexi 2021 (CRISPR-KO)',  'Papalexi'),
    ('Replogle 2022 (CRISPRi)', 'Replogle'),
]

METHOD_ORDER = ['Euclidean', 'Whitened', 'k-NN']

# Keep only datasets that are present in the data
DATASET_ORDER = [(full, short) for full, short in DATASET_ORDER
                 if full in df_all['dataset'].values]

print("Plotting datasets:", [s for _, s in DATASET_ORDER])

# =============================================================================
# FIGURE
# =============================================================================

# Neutral gray palette matching the original figure
BAR_COLORS = {
    'Euclidean': '#888888',
    'Whitened':  '#555555',
    'k-NN':      '#AAAAAA',
}

n_datasets = len(DATASET_ORDER)
n_methods  = len(METHOD_ORDER)
bar_width  = 0.22
group_gap  = 0.08
group_width = n_methods * bar_width + group_gap

fig, ax = plt.subplots(figsize=(max(8, 1.8 * n_datasets), 5.5))

x_centers = np.arange(n_datasets)

for mi, method in enumerate(METHOD_ORDER):
    # Offset: centre the group of bars around each x_center
    offsets = np.linspace(
        -(n_methods - 1) / 2 * bar_width,
         (n_methods - 1) / 2 * bar_width,
        n_methods
    )
    x_pos = x_centers + offsets[mi]

    rhos, ci_lows, ci_highs = [], [], []
    for ds_full, _ in DATASET_ORDER:
        row = df_all[(df_all['dataset'] == ds_full) &
                     (df_all['method']  == method)]
        if len(row) == 0:
            rhos.append(np.nan)
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)
        else:
            row = row.iloc[0]
            rhos.append(row['rho'])
            ci_lows.append(row['ci_low'])
            ci_highs.append(row['ci_high'])

    rhos      = np.array(rhos,      dtype=float)
    ci_lows   = np.array(ci_lows,   dtype=float)
    ci_highs  = np.array(ci_highs,  dtype=float)
    err_low   = rhos - ci_lows
    err_high  = ci_highs - rhos

    valid = ~np.isnan(rhos)
    ax.bar(
        x_pos[valid], rhos[valid],
        width=bar_width,
        color=BAR_COLORS[method],
        label=method,
        edgecolor='white', linewidth=0.5,
        zorder=3,
    )
    ax.errorbar(
        x_pos[valid], rhos[valid],
        yerr=[err_low[valid], err_high[valid]],
        fmt='none',
        ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2,
        zorder=4,
    )

# Axes formatting
ax.set_xticks(x_centers)
ax.set_xticklabels([short for _, short in DATASET_ORDER],
                   fontsize=12, fontweight='bold')
ax.set_ylabel('Spearman $\\rho$', fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_xlim(-0.55, n_datasets - 0.45)
ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

ax.set_title('Magnitude-Coherence Correlation by Distance Metric',
             fontsize=13, fontweight='bold', pad=12)

ax.legend(fontsize=10, framealpha=0.9, edgecolor='#CCCCCC',
          loc='lower right')

sns.despine(ax=ax)
plt.tight_layout()

# =============================================================================
# SAVE
# =============================================================================

out = OUT_DIR / "fig_s4_method_comparison"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved -> {out}.pdf / .png")
plt.show()

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

print("\n--- Correlation table (rho [95% CI]) ---")
header = f"{'Dataset':<28s}" + "".join(f"{m:>22s}" for m in METHOD_ORDER)
print(header)
print("-" * (28 + 22 * n_methods))

for ds_full, ds_short in DATASET_ORDER:
    row_str = f"{ds_short:<28s}"
    for method in METHOD_ORDER:
        row = df_all[(df_all['dataset'] == ds_full) &
                     (df_all['method']  == method)]
        if len(row) == 0:
            row_str += f"{'N/A':>22s}"
        else:
            r = row.iloc[0]
            row_str += f"{r['rho']:>+.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]".rjust(22)
    print(row_str)
