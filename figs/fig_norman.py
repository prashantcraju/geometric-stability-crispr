#!/usr/bin/env python3
"""
Figure: Norman 2019 — CEBPA combinations (red) vs KLF1 combinations (blue)
"""

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scanpy", "pertpy", "matplotlib", "seaborn"])

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

if IN_COLAB:
    DATA_DIR = Path("/content/drive/MyDrive/shesha-crispr")
else:
    DATA_DIR = Path(".")

# ==============================================================================
# GENE SETS
# ==============================================================================

CEBP_ALL = {
    'CEBPA','CEBPA+JUN', 'CEBPA+CEBPB', 'CEBPA+CEBPE', 'CEBPA+ZC3HAV1',
}

KLF1_ALL = {
    'KLF1', 'KLF1+MAP2K6', 'KLF1+SET', 'KLF1+TGFBR2',
    'BAK1+KLF1', 'AHR+KLF1', 'DUSP9+KLF1',
    'FOXA1+KLF1', 'COL2A1+KLF1', 'CLDN6+KLF1',
}

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
# PLOT
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

is_cebp  = df_n.index.isin(CEBP_ALL)
is_klf1  = df_n.index.isin(KLF1_ALL)
is_other = ~is_cebp & ~is_klf1

slope_n, intercept_n, _, _, _ = linregress(df_n['magnitude'], df_n['stability'])
x_line_n = np.array([df_n['magnitude'].min(), df_n['magnitude'].max()])

# Other (gray)
ax.scatter(df_n.loc[is_other, 'magnitude'], df_n.loc[is_other, 'stability'],
           c='lightgray', s=50, alpha=0.6, edgecolor='white', linewidth=0.5,
           zorder=1, label='Other')

# CEBP family (red)
# ax.scatter(df_n.loc[is_cebp, 'magnitude'], df_n.loc[is_cebp, 'stability'],
#            c='#d62728', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
#            zorder=2, label='CEBP family')
ax.scatter(df_n.loc[is_cebp, 'magnitude'], df_n.loc[is_cebp, 'stability'],
           c='#b63a54', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
           zorder=2, label='CEBP family')


# KLF1 combinations (blue)
# ax.scatter(df_n.loc[is_klf1, 'magnitude'], df_n.loc[is_klf1, 'stability'],
#            c='#1f77b4', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
#            zorder=2, label='KLF1 combinations')

ax.scatter(df_n.loc[is_klf1, 'magnitude'], df_n.loc[is_klf1, 'stability'],
           c='#56a4c8', s=80, alpha=0.85, edgecolor='white', linewidth=0.5,
           zorder=2, label='KLF1 combinations')


# Regression line
ax.plot(x_line_n, slope_n * x_line_n + intercept_n, '--',
        color='gray', linewidth=2, alpha=0.7, zorder=3)

# # Label CEBPA and KLF1
# for gene, color in [('CEBPA', '#d62728'), ('KLF1', '#1f77b4')]:
#     if gene in df_n.index:
#         x, y = df_n.loc[gene, ['magnitude', 'stability']]
#         ax.annotate(gene, xy=(x, y), xytext=(0, 10), textcoords='offset points',
#                     fontsize=11,color=color, ha='center', zorder=5)

#                     # fontsize=11, fontweight='bold', color=color, ha='center', zorder=5)

norman_labels = {
    'CEBPA': ('#a50026', (30, -20)),
    'KLF1':  ('#1f77b4', (-30, 15)),
}

for gene, (color, offset) in norman_labels.items():
    if gene in df_n.index:
        x, y = df_n.loc[gene, ['magnitude', 'stability']]
        ax.annotate(
            gene, xy=(x, y), xytext=offset, textcoords='offset points',
            fontsize=11, fontweight='bold',color='k', ha='center',
            zorder=5,
            arrowprops=dict(
                arrowstyle='-',
                color='k',
                lw=0.8,
                shrinkA=0,
                shrinkB=3,
            )
        )



rho_n, _ = spearmanr(df_n['magnitude'], df_n['stability'])
ax.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')
ax.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12, fontweight='bold')
ax.set_ylabel('Shesha Stability (Cosine)', fontsize=12,fontweight='bold')
sns.despine(ax=ax)

plt.tight_layout()
out = DATA_DIR / "fig_norman"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {out}.pdf and .png")
plt.show()
