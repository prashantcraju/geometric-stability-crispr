#!/usr/bin/env python3
"""
Fig 2 — Magnitude–Stability Correlation Across Datasets (with 10k Bootstrap CIs)

Loads all five CRISPR datasets live from pertpy, computes Shesha stability
and magnitude in PCA space, reports Spearman rho with 10,000-replicate bootstrap
95% CIs, and produces the 6-panel figure (5 datasets + pooled z-scored).

OUTPUT (saved to OUTPUT_DIR):
  fig2_magnitude_stability_ci.pdf / .png   — main figure
  magnitude_stability_correlations_ci.csv  — per-dataset rho + CI table
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

import scanpy as sc
import pertpy as pt

from anndata import AnnData
from scipy.stats import spearmanr
from scipy.sparse import issparse
from statsmodels.nonparametric.smoothers_lowess import lowess
from shesha.bio import compute_stability, compute_magnitude
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 320
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
LOESS_FRAC = 0.4
MIN_CELLS = 50
REPLOGLE_MIN_CELLS = 50

OUTPUT_DIR = Path("./shesha-crispr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# Colors matching fig2,4,5.py
BLUE  = '#4C72B0'
RED   = '#C44E52'
GREEN = '#2CA02C'
BROWN = '#8C564B'
GOLD  = '#E5A84B'

DATASETS_INFO = [
    ('Norman 2019 (CRISPRa)',   'Norman 2019',   'CRISPRa', BLUE),
    ('Adamson 2016 (CRISPRi)',  'Adamson 2016',  'CRISPRi', RED),
    ('Dixit 2016 (CRISPRi)',    'Dixit 2016',    'CRISPRi', GREEN),
    ('Papalexi 2021 (CRISPR)',  'Papalexi 2021', 'Pooled',  BROWN),
    ('Replogle 2022 (CRISPRi)', 'Replogle 2022', 'CRISPRi', GOLD),
]

# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Spearman rho with percentile bootstrap 95% CI."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return dict(rho=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=len(x))
    rng = np.random.default_rng(seed)
    boot = np.array([
        spearmanr(x[idx := rng.choice(len(x), len(x), replace=True)],
                  y[idx])[0]
        for _ in range(n_bootstrap)
    ])
    valid = boot[~np.isnan(boot)]
    alpha = 1 - CI_LEVEL
    return dict(
        rho=rho, p=p, n=len(x),
        ci_low=float(np.percentile(valid, 100 * alpha / 2)),
        ci_high=float(np.percentile(valid, 100 * (1 - alpha / 2))),
        n_boot_valid=len(valid),
    )

# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def _to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def _preprocess(adata, min_cells, pert_col, ctrl_label, n_pcs=50):
    """Normalize, HVGs, PCA; return df with stability/magnitude."""
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    counts = adata.obs[pert_col].value_counts()
    valid = [p for p in counts[counts >= min_cells].index if p != ctrl_label]
    print(f"    {len(valid)} perturbations with >= {min_cells} cells")

    sub = adata[adata.obs[pert_col].isin(valid + [ctrl_label])].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, subset=True)
    sc.tl.pca(sub, n_comps=min(n_pcs, sub.n_vars - 1), random_state=SEED)

    adata_pca = AnnData(X=sub.obsm['X_pca'], obs=sub.obs)
    stab = compute_stability(adata_pca, perturbation_key=pert_col,
                             control_label=ctrl_label, metric='cosine')
    mag  = compute_magnitude(adata_pca, perturbation_key=pert_col,
                             control_label=ctrl_label, metric='euclidean')

    df = pd.DataFrame({'stability': pd.Series(stab),
                       'magnitude': pd.Series(mag)})
    if ctrl_label in df.index:
        df = df.drop(ctrl_label)
    df = df[df.index.isin(valid)].copy()
    df['n_cells'] = df.index.map(counts)
    return df


def load_norman():
    print("\n>>> Norman 2019 (CRISPRa)...")
    adata = pt.dt.norman_2019()
    return _preprocess(adata, MIN_CELLS, 'perturbation_name', 'control')


def load_adamson():
    print("\n>>> Adamson 2016 (CRISPRi)...")
    adata = pt.dt.adamson_2016_pilot()
    src = next((c for c in ['perturbation_name', 'perturbation', 'gene',
                             'target', 'guide_id', 'condition']
                if c in adata.obs.columns), None)
    if src is None:
        src = next((c for c in adata.obs.columns
                    if 'pert' in c.lower() or 'gene' in c.lower()), None)
    adata.obs[src] = adata.obs[src].astype(str)
    ctrl_kws = ['gal4', 'gfp', 'neg', 'scramble', 'unperturbed', 'nan']
    adata.obs['condition'] = adata.obs[src].apply(
        lambda x: 'control' if any(kw in x.lower() for kw in ctrl_kws) else x
    )
    adata = adata[adata.obs['condition'] != 'nan'].copy()
    return _preprocess(adata, MIN_CELLS, 'condition', 'control')


def load_dixit():
    print("\n>>> Dixit 2016 (CRISPRi)...")
    adata = pt.dt.dixit_2016()
    return _preprocess(adata, MIN_CELLS, 'perturbation_name', 'control')


def load_papalexi():
    print("\n>>> Papalexi 2021 (CRISPR)...")
    raw = pt.dt.papalexi_2021()
    if type(raw).__name__ != 'MuData':
        raise TypeError(f"Expected MuData for Papalexi, got {type(raw)}")
    adata = raw.mod['rna'].copy()
    if 'gene_target' not in raw.obs.columns:
        raise KeyError("'gene_target' not in Papalexi MuData.obs")
    adata.obs['gene_target'] = raw.obs['gene_target'].values
    print(f"    NT control cells: {(adata.obs['gene_target'] == 'NT').sum()}")
    return _preprocess(adata, MIN_CELLS, 'gene_target', 'NT')


def load_replogle():
    print("\n>>> Replogle 2022 (CRISPRi)...")
    adata = pt.dt.replogle_2022_k562_essential()
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)

    def _label(x):
        if 'non-targeting' in x or x.startswith('chr'): return 'control'
        if 'pos_control' in x: return 'POS_CONTROL'
        return x.split('_')[0]

    adata.obs['condition'] = adata.obs['perturbation'].apply(_label)
    adata = adata[
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    ].copy()
    return _preprocess(adata, REPLOGLE_MIN_CELLS, 'condition', 'control')

# =============================================================================
# LOAD ALL DATASETS
# =============================================================================

print("=" * 80)
print("MAGNITUDE–STABILITY CORRELATIONS WITH 10k BOOTSTRAP CIs")
print(f"Bootstrap replicates: {N_BOOTSTRAP}  |  Seed: {SEED}")
print("=" * 80)

loaders = {
    'Norman 2019 (CRISPRa)':   load_norman,
    'Adamson 2016 (CRISPRi)':  load_adamson,
    'Dixit 2016 (CRISPRi)':    load_dixit,
    'Papalexi 2021 (CRISPR)':  load_papalexi,
    'Replogle 2022 (CRISPRi)': load_replogle,
}

dfs = {}
for ds_full, loader in loaders.items():
    try:
        dfs[ds_full] = loader()
        print(f"    -> {len(dfs[ds_full])} perturbations loaded")
    except Exception as e:
        print(f"    ! Failed: {e}")

# =============================================================================
# COMPUTE CORRELATIONS
# =============================================================================

print("\n" + "=" * 80)
print("SPEARMAN CORRELATIONS (magnitude vs stability)")
print("=" * 80)

corr_results = []
seed_counter = SEED + 500

print(f"\n{'Dataset':<30s}  {'n':>4s}  {'rho':>6s}  {'95% CI':>20s}  {'p':>10s}")
print("-" * 78)

for ds_full, ds_short, modality, _ in DATASETS_INFO:
    if ds_full not in dfs:
        continue
    df = dfs[ds_full]
    ci = bootstrap_spearman_ci(df['magnitude'], df['stability'], seed=seed_counter)
    seed_counter += 1

    ci_str = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
    print(f"{ds_full:<30s}  {ci['n']:>4d}  {ci['rho']:>+.3f}  {ci_str:>20s}  {ci['p']:>10.2e}")

    corr_results.append({
        'dataset': ds_full,
        'dataset_short': ds_short,
        'modality': modality,
        'n': ci['n'],
        'rho': ci['rho'],
        'ci_low': ci['ci_low'],
        'ci_high': ci['ci_high'],
        'p': ci['p'],
    })

# Pooled z-scored
all_z = []
for ds_full, ds_short, _, _ in DATASETS_INFO:
    if ds_full not in dfs:
        continue
    sub = dfs[ds_full][['magnitude', 'stability']].copy()
    sub['mag_z']  = (sub['magnitude'] - sub['magnitude'].mean()) / sub['magnitude'].std()
    sub['stab_z'] = (sub['stability'] - sub['stability'].mean()) / sub['stability'].std()
    sub['dataset_short'] = ds_short
    all_z.append(sub)

pooled = pd.concat(all_z, ignore_index=True)
ci_pooled = bootstrap_spearman_ci(pooled['mag_z'], pooled['stab_z'], seed=seed_counter)
ci_str = f"[{ci_pooled['ci_low']:.3f}, {ci_pooled['ci_high']:.3f}]"
print(f"{'Pooled (z-scored)':<30s}  {ci_pooled['n']:>4d}  {ci_pooled['rho']:>+.3f}  "
      f"{ci_str:>20s}  {ci_pooled['p']:>10.2e}")

corr_results.append({
    'dataset': 'Pooled (z-scored)',
    'dataset_short': 'Pooled',
    'modality': 'All',
    'n': ci_pooled['n'],
    'rho': ci_pooled['rho'],
    'ci_low': ci_pooled['ci_low'],
    'ci_high': ci_pooled['ci_high'],
    'p': ci_pooled['p'],
})

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(OUTPUT_DIR / "magnitude_stability_correlations_ci.csv", index=False)
print(f"\nSaved -> magnitude_stability_correlations_ci.csv")

# =============================================================================
# FIGURE: 2x3 grid (5 datasets + pooled)
# =============================================================================

print("\n>>> Generating figure...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

all_for_pooled = []

for i, (ds_full, ds_short, modality, color) in enumerate(DATASETS_INFO):
    ax = axes[i]

    if ds_full not in dfs or len(dfs[ds_full]) < 3:
        ax.text(0.5, 0.5, f'{ds_short}\n(no data)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='gray')
        sns.despine(ax=ax)
        continue

    sub = dfs[ds_full].copy()
    ci = next((r for r in corr_results if r['dataset'] == ds_full), None)

    # Scatter
    ax.scatter(sub['magnitude'], sub['stability'],
               c=color, s=40, alpha=0.6, edgecolor='white', linewidth=0.3)

    # LOESS fit line
    order = np.argsort(sub['magnitude'].values)
    x_sorted = sub['magnitude'].values[order]
    y_sorted = sub['stability'].values[order]
    fitted = lowess(y_sorted, x_sorted, frac=LOESS_FRAC, return_sorted=False)
    ax.plot(x_sorted, fitted, '--', color='gray', linewidth=2, alpha=0.7)

    # Annotation: rho + 95% CI
    if ci:
        ann = (f"$\\rho$ = {ci['rho']:.3f}\n"
               f"95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    else:
        ann = ''
    ax.text(0.97, 0.03, ann,
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax.set_title(f'{ds_short}\n({modality}, n={len(sub)})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Effect Magnitude', fontsize=10, fontweight='bold')
    ax.set_ylabel('Shesha Stability', fontsize=10, fontweight='bold')
    sns.despine(ax=ax)

    # Accumulate for pooled panel
    sub_z = sub[['magnitude', 'stability']].copy()
    sub_z['mag_z']  = (sub_z['magnitude'] - sub_z['magnitude'].mean()) / sub_z['magnitude'].std()
    sub_z['stab_z'] = (sub_z['stability'] - sub_z['stability'].mean()) / sub_z['stability'].std()
    sub_z['dataset_short'] = ds_short
    sub_z['color'] = color
    all_for_pooled.append(sub_z)

# Panel 6: pooled z-scored
ax5 = axes[5]
pooled_plot = pd.concat(all_for_pooled, ignore_index=True)
for ds in pooled_plot['dataset_short'].unique():
    mask = pooled_plot['dataset_short'] == ds
    c = pooled_plot.loc[mask, 'color'].iloc[0]
    ax5.scatter(pooled_plot.loc[mask, 'mag_z'], pooled_plot.loc[mask, 'stab_z'],
                c=c, s=20, alpha=0.5, edgecolor='none', label=ds)

order_p = np.argsort(pooled_plot['mag_z'].values)
fitted_p = lowess(pooled_plot['stab_z'].values[order_p],
                  pooled_plot['mag_z'].values[order_p],
                  frac=LOESS_FRAC, return_sorted=False)
ax5.plot(pooled_plot['mag_z'].values[order_p], fitted_p,
         '--', color='gray', linewidth=2, alpha=0.7)

ann_p = (f"$\\rho$ = {ci_pooled['rho']:.3f}\n"
         f"95% CI [{ci_pooled['ci_low']:.3f}, {ci_pooled['ci_high']:.3f}]")
ax5.text(0.97, 0.03, ann_p,
         transform=ax5.transAxes, fontsize=9, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#CCCCCC', alpha=0.9))

ax5.set_title(f'Pooled (z-scored)\n(n={ci_pooled["n"]})',
              fontsize=11, fontweight='bold')
ax5.set_xlabel('Magnitude (z)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Stability (z)', fontsize=10, fontweight='bold')
ax5.legend(fontsize=7, framealpha=0.8)
sns.despine(ax=ax5)

# Panel labels a–f
for i, label in enumerate('abcdef'):
    axes[i].text(-0.08, 1.08, label, transform=axes[i].transAxes,
                 fontsize=14, fontweight='bold', va='top', ha='right')

plt.tight_layout()

out = OUTPUT_DIR / "fig2_magnitude_stability_ci"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved figure -> {out}.pdf / .png")
plt.show()

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"\nOutput files in {OUTPUT_DIR}:")
print("  - fig2_magnitude_stability_ci.pdf / .png")
print("  - magnitude_stability_correlations_ci.csv")
