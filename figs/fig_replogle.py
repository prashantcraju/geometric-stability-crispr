#!/usr/bin/env python3
"""
Figure: Replogle 2022 K562 — key discordant genes highlighted
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
import scanpy as sc
import pertpy as pt
from anndata import AnnData
from shesha.bio import compute_stability, compute_magnitude
from pathlib import Path

SEED = 320
np.random.seed(SEED)

DATA_DIR = Path("./shesha-crispr")

# ==============================================================================
# GENE SETS
# ==============================================================================

REPLOGLE_RED    = {'GATA1', 'CHMP3', 'AQR'}
REPLOGLE_BLUE   = {'LSG1', 'ISG20L2'}
REPLOGLE_ORANGE = {'BUB3', 'CENPW'}

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
print("Genes found:", {g for g in REPLOGLE_RED | REPLOGLE_BLUE | REPLOGLE_ORANGE if g in df_r.index})
print("Genes missing:", {g for g in REPLOGLE_RED | REPLOGLE_BLUE | REPLOGLE_ORANGE if g not in df_r.index})

# ==============================================================================
# PLOT
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

is_red    = df_r.index.isin(REPLOGLE_RED)
is_blue   = df_r.index.isin(REPLOGLE_BLUE)
is_orange = df_r.index.isin(REPLOGLE_ORANGE)
is_other  = ~is_red & ~is_blue & ~is_orange

slope_r, intercept_r, _, _, _ = linregress(df_r['magnitude'], df_r['stability'])
x_line_r = np.array([df_r['magnitude'].min(), df_r['magnitude'].max()])

# Other (gray)
ax.scatter(df_r.loc[is_other, 'magnitude'], df_r.loc[is_other, 'stability'],
           c='lightgray', s=25, alpha=0.4, edgecolor='white', linewidth=0.3,
           zorder=1, label='Other')

# Discordant (red)
ax.scatter(df_r.loc[is_red, 'magnitude'], df_r.loc[is_red, 'stability'],
           c='#b63a54', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
           zorder=3, label='Discordant (high magnitude)')

# Concordant (blue)
ax.scatter(df_r.loc[is_blue, 'magnitude'], df_r.loc[is_blue, 'stability'],
           c='#56a4c8', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
           zorder=3, label='Concordant (specific)')

# Cell cycle (orange)
ax.scatter(df_r.loc[is_orange, 'magnitude'], df_r.loc[is_orange, 'stability'],
           c='orange', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
           zorder=3, label='Cell cycle')

# Regression line
ax.plot(x_line_r, slope_r * x_line_r + intercept_r, '--',
        color='gray', linewidth=2, alpha=0.7, zorder=2)

# Label key genes
label_genes = {
    'GATA1': ('#56a4c8', (30,  15)),
    'AQR':   ('#56a4c8', (25,  15)),
    'CHMP3': ('#56a4c8', (20,  15)),
    'LSG1':  ('#b63a54', (-20,  15)),
    'ISG20L2': ('#b63a54', (-30, 10)),
    'BUB3':  ('orange',  (35,  -25)),
    'CENPW': ('orange',  (-35,  25)),
}
for gene, (color, offset) in label_genes.items():
    if gene in df_r.index:
        x, y = df_r.loc[gene, ['magnitude', 'stability']]
        ax.annotate(
            gene, xy=(x, y), xytext=offset, textcoords='offset points',
            fontsize=9, fontweight='bold', color='k', ha='center', zorder=5,
            arrowprops=dict(arrowstyle='-', color='k', lw=0.8, shrinkA=0, shrinkB=3)
        )

rho_r, _ = spearmanr(df_r['magnitude'], df_r['stability'])
ax.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')
ax.set_xlabel('Effect Magnitude (Euclidean)', fontweight='bold', fontsize=12)
ax.set_ylabel('Shesha Stability (Cosine)', fontweight='bold',fontsize=12)
sns.despine(ax=ax)

plt.tight_layout()
out = DATA_DIR / "fig_replogle"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {out}.pdf and .png")
plt.show()
