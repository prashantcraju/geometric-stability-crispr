#!/usr/bin/env python3
"""
Figure 1 panels d & e:
  d) Norman 2019: CEBP family (red) vs KLF1 combos (blue)
  e) Replogle 2022: key discordant genes highlighted

Same clean style as shesha_crispr_discordant_norman.pdf:
solid dots, white edges, gray background, dashed regression line.
"""


import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress, zscore
import scanpy as sc
import pertpy as pt
from anndata import AnnData
from shesha.bio import compute_stability, compute_magnitude
from pathlib import Path

import pipeline_config as cfg
from revision_io import resolve_out_dir

SEED = cfg.SEED
np.random.seed(SEED)

DATA_DIR = resolve_out_dir()

# ==============================================================================
# GENE SETS
# ==============================================================================

CEBP_ALL = {
    'CEBPA', 'CEBPB', 'CEBPE',
    'CEBPA+JUN', 'CEBPA+CEBPB', 'CEBPA+CEBPE', 'CEBPA+ZC3HAV1',
    'CEBPB+FOSB', 'CEBPB+LYL1', 'CEBPB+PTPN12', 'CEBPB+MAPK1',
    'CEBPB+OSR2', 'CEBPB+CEBPE', 'CEBPB+JUN',
    'CEBPE+RUNX1T1', 'CEBPE+SET', 'CEBPE+ZC3HAV1',
    'CEBPE+FOSB', 'CEBPE+ETS2', 'CEBPE+CNN1', 'CEBPE+SPI1', 'CEBPE+PTPN12',
}

KLF1_ALL = {
    'KLF1', 'KLF1+MAP2K6', 'KLF1+SET', 'KLF1+TGFBR2',
    'BAK1+KLF1', 'AHR+KLF1', 'DUSP9+KLF1',
    'FOXA1+KLF1', 'COL2A1+KLF1', 'CLDN6+KLF1',
}

# Replogle: genes to highlight
REPLOGLE_RED = {'GATA1', 'CEBPB', 'ACTB', 'CHMP3', 'AQR'}
REPLOGLE_BLUE = {'BLVRB', 'LSG1', 'ISG20L2', 'KRI1'}
REPLOGLE_ORANGE = {'BUB3', 'CENPW'}

# ==============================================================================
# PROCESS NORMAN
# ==============================================================================
print("Loading Norman 2019...")
adata_n = pt.dt.norman_2019()
sc.pp.normalize_total(adata_n)
sc.pp.log1p(adata_n)
sc.pp.highly_variable_genes(adata_n, n_top_genes=2000, subset=True)
sc.pp.pca(adata_n, n_comps=50)

adata_n_pca = AnnData(X=adata_n.obsm['X_pca'], obs=adata_n.obs)
stab_n = compute_stability(adata_n_pca, perturbation_key='perturbation_name',
                            control_label='control', metric='cosine')
mag_n = compute_magnitude(adata_n_pca, perturbation_key='perturbation_name',
                           control_label='control', metric='euclidean')

df_n = pd.DataFrame({'stability': pd.Series(stab_n), 'magnitude': pd.Series(mag_n)})
counts_n = adata_n.obs['perturbation_name'].value_counts()
df_n['n_cells'] = df_n.index.map(counts_n)
if 'control' in df_n.index: df_n = df_n.drop('control')
df_n = df_n[df_n['n_cells'] > 50].copy()
print(f"Norman: {len(df_n)} perturbations")

# ==============================================================================
# PROCESS REPLOGLE
# ==============================================================================
print("Loading Replogle 2022...")
adata_r = pt.dt.replogle_2022_k562_essential()
adata_r.obs['perturbation'] = adata_r.obs['perturbation'].astype(str)

def clean_label(x):
    if 'non-targeting' in x or x.startswith('chr'): return 'control'
    if 'pos_control' in x: return 'POS_CONTROL'
    return x.split('_')[0]

adata_r.obs['condition'] = adata_r.obs['perturbation'].apply(clean_label)
adata_r_clean = adata_r[
    (adata_r.obs['condition'] != 'POS_CONTROL') &
    (adata_r.obs['condition'] != 'nan')
].copy()
counts_r = adata_r_clean.obs['condition'].value_counts()
valid_r = counts_r[counts_r >= 50].index
adata_r_final = adata_r_clean[adata_r_clean.obs['condition'].isin(valid_r)].copy()

sc.pp.normalize_total(adata_r_final, target_sum=1e4)
sc.pp.log1p(adata_r_final)
sc.pp.highly_variable_genes(adata_r_final, n_top_genes=2000, subset=True)
sc.tl.pca(adata_r_final, n_comps=50)

adata_r_pca = AnnData(X=adata_r_final.obsm['X_pca'], obs=adata_r_final.obs)
stab_r = compute_stability(adata_r_pca, perturbation_key='condition',
                            control_label='control', metric='cosine')
mag_r = compute_magnitude(adata_r_pca, perturbation_key='condition',
                           control_label='control', metric='euclidean')

df_r = pd.DataFrame({'stability': pd.Series(stab_r), 'magnitude': pd.Series(mag_r)})
if 'control' in df_r.index: df_r = df_r.drop('control')
df_r['n_cells'] = df_r.index.map(counts_r)
print(f"Replogle: {len(df_r)} perturbations")

# ==============================================================================
# PLOT
# ==============================================================================
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4))

# ======================================================================
# PANEL D: Norman (same style as shesha_crispr_discordant_norman.pdf)
# ======================================================================

is_cebp = df_n.index.isin(CEBP_ALL)
is_klf1 = df_n.index.isin(KLF1_ALL)
is_other = ~is_cebp & ~is_klf1

slope_n, intercept_n, _, _, _ = linregress(df_n['magnitude'], df_n['stability'])
x_line_n = np.array([df_n['magnitude'].min(), df_n['magnitude'].max()])

# Other (gray)
ax0.scatter(df_n.loc[is_other, 'magnitude'], df_n.loc[is_other, 'stability'],
            c='lightgray', s=50, alpha=0.6, edgecolor='white', linewidth=0.5,
            zorder=1, label='Other')

# CEBP family (red)
ax0.scatter(df_n.loc[is_cebp, 'magnitude'], df_n.loc[is_cebp, 'stability'],
            c='#d62728', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=2, label='CEBP family')

# KLF1 combinations (blue)
ax0.scatter(df_n.loc[is_klf1, 'magnitude'], df_n.loc[is_klf1, 'stability'],
            c='#1f77b4', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=2, label='KLF1 combinations')

# Regression line
ax0.plot(x_line_n, slope_n * x_line_n + intercept_n, '--',
         color='gray', linewidth=2, alpha=0.7, zorder=3)

# Label CEBPA and KLF1 single genes
for gene, color in [('CEBPA', '#d62728'), ('KLF1', '#1f77b4')]:
    if gene in df_n.index:
        x, y = df_n.loc[gene, ['magnitude', 'stability']]
        ax0.annotate(gene, xy=(x, y), xytext=(0, 10), textcoords='offset points',
                     fontsize=11, fontweight='bold', color=color, ha='center',
                     zorder=5)

rho_n, _ = spearmanr(df_n['magnitude'], df_n['stability'])
ax0.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')
ax0.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
ax0.set_ylabel('Shesha Coherence (Cosine)', fontsize=12)
sns.despine(ax=ax0)

# ======================================================================
# PANEL E: Replogle (same clean dot style)
# ======================================================================

is_red = df_r.index.isin(REPLOGLE_RED)
is_blue = df_r.index.isin(REPLOGLE_BLUE)
is_orange = df_r.index.isin(REPLOGLE_ORANGE)
is_other_r = ~is_red & ~is_blue & ~is_orange

slope_r, intercept_r, _, _, _ = linregress(df_r['magnitude'], df_r['stability'])
x_line_r = np.array([df_r['magnitude'].min(), df_r['magnitude'].max()])

# Other (gray)
ax1.scatter(df_r.loc[is_other_r, 'magnitude'], df_r.loc[is_other_r, 'stability'],
            c='lightgray', s=25, alpha=0.4, edgecolor='white', linewidth=0.3,
            zorder=1, label='Other')

# Discordant high (red): large magnitude, low stability relative to fit
ax1.scatter(df_r.loc[is_red, 'magnitude'], df_r.loc[is_red, 'stability'],
            c='#d62728', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=3, label='Discordant (pleiotropic)')

# Concordant high (blue): high stability for magnitude
ax1.scatter(df_r.loc[is_blue, 'magnitude'], df_r.loc[is_blue, 'stability'],
            c='#1f77b4', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=3, label='Concordant (specific)')

# Cell cycle (orange)
ax1.scatter(df_r.loc[is_orange, 'magnitude'], df_r.loc[is_orange, 'stability'],
            c='orange', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=3, label='Cell cycle')

# Regression line
ax1.plot(x_line_r, slope_r * x_line_r + intercept_r, '--',
         color='gray', linewidth=2, alpha=0.7, zorder=2)

# Label key genes
label_genes = {
    'GATA1': ('#d62728', (30, 15)),
    'CEBPB': ('#d62728', (25, 15)),
    'ACTB':  ('#d62728', (-40, -30)),
    'CHMP3': ('#d62728', (20, 15)),
    'BLVRB': ('#1f77b4', (-30, 15)),
    'LSG1':  ('#1f77b4', (-30, 10)),
    'BUB3':  ('orange', (35, -25)),
    'CENPW': ('orange', (-35, 25)),
}

for gene, (color, offset) in label_genes.items():
    if gene in df_r.index:
        x, y = df_r.loc[gene, ['magnitude', 'stability']]
        ax1.annotate(
            gene, xy=(x, y), xytext=offset, textcoords='offset points',
            fontsize=9, fontweight='bold', color=color, ha='center',
            zorder=5,
            arrowprops=dict(
                arrowstyle='-',
                color=color,
                lw=0.8,
                shrinkA=0,
                shrinkB=3,
            )
        )
# label_genes = {
#     'GATA1': ('#d62728', (0, 10)),
#     'CEBPB': ('#d62728', (0, 10)),
#     'ACTB':  ('#d62728', (-12, -14)),
#     'CHMP3': ('#d62728', (0, 10)),
#     'BLVRB': ('#1f77b4', (0, 10)),
#     'LSG1':  ('#1f77b4', (0, 10)),
#     'BUB3':  ('orange', (12, 18)),
#     'CENPW': ('orange', (18, -12)),
# }
# for gene, (color, offset) in label_genes.items():
#     if gene in df_r.index:
#         x, y = df_r.loc[gene, ['magnitude', 'stability']]
#         ax1.annotate(gene, xy=(x, y), xytext=offset, textcoords='offset points',
#                      fontsize=9, fontweight='bold', color=color, ha='center',
#                      zorder=5)

rho_r, _ = spearmanr(df_r['magnitude'], df_r['stability'])
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')
ax1.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
ax1.set_ylabel('Shesha Coherence (Cosine)', fontsize=12)
sns.despine(ax=ax1)

# ======================================================================
# PANEL LABELS & SAVE
# ======================================================================
# for ax, label in zip([ax0, ax1], ['d', 'e']):
#     ax.text(-0.08, 1.05, label, transform=ax.transAxes,
#             fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
out = DATA_DIR / "fig1_de_panels"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {out}.pdf and .png")
plt.show()