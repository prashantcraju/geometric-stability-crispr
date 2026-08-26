#!/usr/bin/env python3
"""
fig_replogle.py

Replogle 2022 CRISPRi discordance scatter with LOESS-corrected gene ranking.

Data:
  shesha_crispr_results_euclidean.csv   — magnitude, stability, n_cells per perturbation
  nonlinear_discordance_comparison.csv  — disc_linear, disc_loess, rank_linear, rank_loess

Merge on perturbation, filter to Replogle rows, plot magnitude vs stability.
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import os
from revision_io import data_search_dirs, resolve_out_dir


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from scipy.stats import spearmanr, linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

SEED = 320
np.random.seed(SEED)

plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})

_CSV_ROOTS = data_search_dirs()
DATA_DIR = resolve_out_dir()


def _find_csv(*names):
    for name in names:
        for root in _CSV_ROOTS:
            p = root / name
            if p.exists():
                return p
    return None

# ==============================================================================
# GENE SETS  (derived from rank_loess in nonlinear_discordance_comparison.csv)
# ==============================================================================

REPLOGLE_RED    = {'CHMP2A', 'SF3B3', 'SF3B2', 'PSMD7', 'CHMP3'}   # discordant  LOESS top-5
REPLOGLE_BLUE   = {'CASP8AP2', 'CHAF1B', 'LSG1'}                    # concordant  LOESS top-3
REPLOGLE_ORANGE = {'BUB3', 'CENPW'}                                  # cell cycle
REPLOGLE_GRAY   = {'GATA1', 'AQR'}                                   # former flagships (dimmed)

ALL_LABELED = REPLOGLE_RED | REPLOGLE_BLUE | REPLOGLE_ORANGE | REPLOGLE_GRAY

# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_data():
    euclid_csv = _find_csv(
        "shesha_crispr_results_euclidean.csv",
        "frozen_sp_scores.csv",
    )
    disc_csv = _find_csv("nonlinear_discordance_comparison.csv")
    if euclid_csv is None:
        raise FileNotFoundError(
            "Need frozen_sp_scores.csv or shesha_crispr_results_euclidean.csv "
            "in CRISPR/pathway or shesha-crispr."
        )
    print(f"Sp table: {euclid_csv}")

    df_euclid = pd.read_csv(euclid_csv)
    df_euclid.columns = df_euclid.columns.str.strip().str.lower()

    def _pert_col(df):
        for c in ['perturbation', 'gene', 'pert', 'perturbation_name']:
            if c in df.columns:
                return c
        raise KeyError(f"No perturbation column found in {list(df.columns)}")

    pc_e = _pert_col(df_euclid)
    df_euclid = df_euclid.rename(columns={pc_e: 'perturbation'})
    if 'dataset' in df_euclid.columns:
        df_euclid = df_euclid[df_euclid['dataset'].str.contains('Replogle', case=False, na=False)]

    if disc_csv is not None:
        print(f"Discordance: {disc_csv}")
        df_disc = pd.read_csv(disc_csv)
        df_disc.columns = df_disc.columns.str.strip().str.lower()
        pc_d = _pert_col(df_disc)
        df_disc = df_disc.rename(columns={pc_d: 'perturbation'})
        if 'dataset' in df_disc.columns:
            df_disc = df_disc[df_disc['dataset'].str.contains('Replogle', case=False, na=False)]
        df = df_euclid.merge(df_disc, on='perturbation', how='inner',
                             suffixes=('', '_disc'))
    else:
        print("No nonlinear_discordance_comparison.csv — using frozen Sp only.")
        df = df_euclid.copy()
    df = df.set_index('perturbation')
    rename = {}
    lower = {c.lower(): c for c in df.columns}
    if 'stability' not in df.columns:
        for o in ('sp', 'shesha', 'coherence'):
            if o in lower:
                rename[lower[o]] = 'stability'
                break
    if 'magnitude' not in df.columns:
        for o in ('mp', 'mag'):
            if o in lower:
                rename[lower[o]] = 'magnitude'
                break
    if rename:
        df = df.rename(columns=rename)

    # Ensure required columns exist
    for col in ('magnitude', 'stability'):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found after merge. "
                           f"Available: {list(df.columns)}")

    df = df.dropna(subset=['magnitude', 'stability'])
    print(f"Replogle: {len(df)} perturbations after merge")
    print("Genes found:  ",
          {g for g in ALL_LABELED if g in df.index})
    print("Genes missing:",
          {g for g in ALL_LABELED if g not in df.index})
    return df


df_r = load_data()

# ==============================================================================
# COMPUTE FITS
# ==============================================================================

mag   = df_r['magnitude'].values
stab  = df_r['stability'].values

# Linear fit (dashed gray)
slope, intercept, *_ = linregress(mag, stab)
x_lo, x_hi = mag.min(), mag.max()
x_line = np.linspace(x_lo, x_hi, 300)
y_linear = slope * x_line + intercept

# LOESS fit
loess_result = lowess(stab, mag, frac=0.3, return_sorted=True)

rho, _ = spearmanr(mag, stab)

# ==============================================================================
# PLOT
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

is_red    = df_r.index.isin(REPLOGLE_RED)
is_blue   = df_r.index.isin(REPLOGLE_BLUE)
is_orange = df_r.index.isin(REPLOGLE_ORANGE)
is_gray_l = df_r.index.isin(REPLOGLE_GRAY)
is_other  = ~is_red & ~is_blue & ~is_orange & ~is_gray_l

# --- background scatter
ax.scatter(df_r.loc[is_other,  'magnitude'], df_r.loc[is_other,  'stability'],
           c='lightgray', s=20, alpha=0.4, edgecolor='none',
           zorder=1, label='_nolegend_')

# --- highlighted genes
ax.scatter(df_r.loc[is_red,    'magnitude'], df_r.loc[is_red,    'stability'],
           c='#d73027', s=80, alpha=0.9, edgecolor='white', linewidth=0.5,
           zorder=3, label='Discordant (LOESS top 5)')

ax.scatter(df_r.loc[is_blue,   'magnitude'], df_r.loc[is_blue,   'stability'],
           c='#4575b4', s=80, alpha=0.9, edgecolor='white', linewidth=0.5,
           zorder=3, label='Concordant (LOESS top 3)')

ax.scatter(df_r.loc[is_orange, 'magnitude'], df_r.loc[is_orange, 'stability'],
           c='#f49d37', s=80, alpha=0.9, edgecolor='white', linewidth=0.5,
           zorder=3, label='Cell cycle')

ax.scatter(df_r.loc[is_gray_l, 'magnitude'], df_r.loc[is_gray_l, 'stability'],
           c='#999999', s=55, alpha=0.6, edgecolor='white', linewidth=0.4,
           zorder=2, label='Linear flagships (GATA1, AQR)')

# --- fits
ax.plot(x_line, y_linear,
        '--', color='gray', linewidth=1.2, alpha=0.6, zorder=2,
        label='Linear fit')
ax.plot(loess_result[:, 0], loess_result[:, 1],
        '-', color='black', linewidth=2, alpha=0.8, zorder=4,
        label='LOESS fit')

# --- gene annotations
label_genes = {
    # discordant (red) — below LOESS at moderate magnitudes
    'CHMP2A': ('#d73027', ( 45,  -20)),
    'SF3B3':  ('#d73027', ( 30, -20)),
    'SF3B2':  ('#d73027', ( 30, -0)),
    'PSMD7':  ('#d73027', ( 30,  15)),
    'CHMP3':  ('#d73027', (-30,  15)),
    # concordant (blue)
    'CASP8AP2': ('#4575b4', ( 40,  15)),
    'CHAF1B':   ('#4575b4', (-40,  15)),
    'LSG1':     ('#4575b4', (-30, -20)),
    # cell cycle (orange)
    'BUB3':  ('#f49d37', ( 35, -25)),
    'CENPW': ('#f49d37', (-35,  25)),
    # former flagships (gray, italic, smaller)
    'GATA1': ('#888888', ( 25,  15)),
    'AQR':   ('#888888', ( 25, -15)),
}

for gene, (color, offset) in label_genes.items():
    if gene not in df_r.index:
        continue
    x, y = df_r.loc[gene, ['magnitude', 'stability']]
    italic = gene in REPLOGLE_GRAY
    ax.annotate(
        gene, xy=(x, y), xytext=offset, textcoords='offset points',
        fontsize=8 if italic else 9,
        fontstyle='italic' if italic else 'normal',
        fontweight='bold' if not italic else 'normal',
        color=color, ha='center', zorder=5,
        arrowprops=dict(arrowstyle='-', color=color, lw=0.7,
                        shrinkA=0, shrinkB=3),
    )

# --- rho annotation
ax.text(0.97, 0.97,
        f'$\\rho$ = {rho:.3f}\n$n$ = {len(df_r):,}',
        transform=ax.transAxes,
        fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.9))

ax.set_title('Replogle 2022 CRISPRi ($n$=1,832): LOESS-corrected discordance',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Effect Magnitude (Euclidean)', fontweight='bold', fontsize=12)
ax.set_ylabel('Shesha Coherence (Cosine)', fontweight='bold', fontsize=12)
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9,
          edgecolor='#CCCCCC')
sns.despine(ax=ax)

plt.tight_layout()
out = DATA_DIR / "fig_replogle"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {out}.pdf and .png")
plt.show()
