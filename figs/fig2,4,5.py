#!/usr/bin/env python3
"""
Paper 3 Figures 2, 4, 5

Fig 2: Magnitude-stability across datasets (PCA only)
  - Norman, Adamson, Dixit from current PCA CSV
  - Papalexi from original PCA CSV
  - Replogle: computed live from pertpy (or pass --skip-replogle)

Fig 4: scGPT validation (3-panel)
  - All from scgpt_all_datasets.csv
  - PCA rho comparison from PCA CSV where available

Fig 5: Combinatorial vs single-gene (Norman, from PCA CSV)

USAGE:
  # In Colab (with pertpy for Replogle):
  python paper3_figs_245.py

  # Locally (skip Replogle in Fig 2):
  python paper3_figs_245.py --skip-replogle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress, mannwhitneyu
from pathlib import Path
import sys

SEED = 320
np.random.seed(SEED)

# Colors
RED = '#C44E52'
BLUE = '#4C72B0'
GOLD = '#E5A84B'
GREEN = '#2CA02C'
BROWN = '#8C564B'
GRAY = '#D3D3D3'
DARK_GRAY = '#555555'

SKIP_REPLOGLE = '--skip-replogle' in sys.argv

# --- PATHS (update as needed) ---
PCA_CSV = '/content/drive/MyDrive/shesha-crispr/shesha_crispr_results_euclidean.csv'
PCA_CSV_ORIGINAL = '/content/drive/MyDrive/shesha-crispr/shesha_crispr_results_euclidean_original.csv'
SCGPT_CSV = '/content/drive/MyDrive/shesha-crispr/scgpt_all_datasets.csv'
OUT_DIR = Path('/content/drive/MyDrive/shesha-crispr')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df_pca = pd.read_csv(PCA_CSV)
    df_scgpt = pd.read_csv(SCGPT_CSV)

    # Papalexi from original CSV
    df_orig = pd.read_csv(PCA_CSV_ORIGINAL)
    df_papalexi = df_orig[df_orig['dataset'] == 'Papalexi 2021 (CRISPR)'].copy()
    if 'Papalexi 2021 (CRISPR)' not in df_pca['dataset'].values:
        df_pca = pd.concat([df_pca, df_papalexi], ignore_index=True)

    return df_pca, df_scgpt


def load_replogle_pca():
    """Compute Replogle PCA stability/magnitude live from pertpy."""
    import scanpy as sc
    import pertpy as pt
    from anndata import AnnData
    from shesha.bio import compute_stability, compute_magnitude

    print("Loading Replogle for Fig 2 (PCA)...")
    adata = pt.dt.replogle_2022_k562_essential()
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)

    def clean_label(x):
        if 'non-targeting' in x or x.startswith('chr'): return 'control'
        if 'pos_control' in x: return 'POS_CONTROL'
        return x.split('_')[0]

    adata.obs['condition'] = adata.obs['perturbation'].apply(clean_label)
    adata = adata[
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    ].copy()

    counts = adata.obs['condition'].value_counts()
    valid = counts[counts >= 50].index
    adata = adata[adata.obs['condition'].isin(valid)].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.tl.pca(adata, n_comps=50)

    adata_pca = AnnData(X=adata.obsm['X_pca'], obs=adata.obs)
    stab = compute_stability(adata_pca, perturbation_key='condition',
                              control_label='control', metric='cosine')
    mag = compute_magnitude(adata_pca, perturbation_key='condition',
                             control_label='control', metric='euclidean')

    df = pd.DataFrame({
        'perturbation': list(stab.keys()),
        'stability': list(stab.values()),
        'magnitude': [mag[k] for k in stab.keys()],
    })
    df = df[df['perturbation'] != 'control'].copy()
    df['dataset'] = 'Replogle 2022 (CRISPRi)'
    df['n_cells'] = df['perturbation'].map(counts)
    print(f"Replogle PCA: {len(df)} perturbations")
    return df


# ==============================================================================
# FIG 2: Magnitude-stability across datasets (PCA)
# ==============================================================================

def fig2(df_pca, df_replogle_pca=None):
    """5 datasets in a 2x3 grid, 6th panel = pooled z-scored."""

    datasets_info = [
        ('Norman 2019 (CRISPRa)', 'Norman 2019', 'CRISPRa', BLUE),
        ('Adamson 2016 (CRISPRi)', 'Adamson 2016', 'CRISPRi', RED),
        ('Dixit 2016 (CRISPRi)', 'Dixit 2016', 'CRISPRi', GREEN),
        ('Papalexi 2021 (CRISPR)', 'Papalexi 2021', 'Pooled', BROWN),
        ('Replogle 2022 (CRISPRi)', 'Replogle 2022', 'CRISPRi', GOLD),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    all_for_pooled = []

    for i, (ds_full, ds_short, modality, color) in enumerate(datasets_info):
        ax = axes[i]

        if ds_full == 'Replogle 2022 (CRISPRi)':
            if df_replogle_pca is not None:
                sub = df_replogle_pca.copy()
            else:
                ax.text(0.5, 0.5, f'{ds_short}\n(skipped, run in Colab)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='gray')
                sns.despine(ax=ax)
                continue
        else:
            sub = df_pca[df_pca['dataset'] == ds_full].copy()

        if len(sub) < 3:
            ax.text(0.5, 0.5, f'{ds_short}\nn={len(sub)}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
            sns.despine(ax=ax)
            continue

        # Scatter
        ax.scatter(sub['magnitude'], sub['stability'],
                   c=color, s=40, alpha=0.6, edgecolor='white', linewidth=0.3)

        # Regression
        slope, intercept, _, _, _ = linregress(sub['magnitude'], sub['stability'])
        x_line = np.array([sub['magnitude'].min(), sub['magnitude'].max()])
        ax.plot(x_line, slope * x_line + intercept, '--', color='gray',
                linewidth=2, alpha=0.7)

        rho, _ = spearmanr(sub['magnitude'], sub['stability'])
        n = len(sub)

        ax.set_title(f'{ds_short}\n({modality}, n={n})', fontsize=11, fontweight='bold')
        ax.text(0.97, 0.03, f'$\\rho$ = {rho:.3f}',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9))

        ax.set_xlabel('Effect Magnitude', fontsize=10,fontweight='bold')
        ax.set_ylabel('Shesha Stability', fontsize=10,fontweight='bold')
        sns.despine(ax=ax)

        # For pooled panel
        if len(sub) > 5:
            sub_z = sub[['magnitude', 'stability']].copy()
            sub_z['mag_z'] = (sub_z['magnitude'] - sub_z['magnitude'].mean()) / sub_z['magnitude'].std()
            sub_z['stab_z'] = (sub_z['stability'] - sub_z['stability'].mean()) / sub_z['stability'].std()
            sub_z['dataset_short'] = ds_short
            sub_z['color'] = color
            all_for_pooled.append(sub_z)

    # Panel 6: pooled z-scored
    ax5 = axes[5]
    if all_for_pooled:
        pooled = pd.concat(all_for_pooled, ignore_index=True)
        for ds in pooled['dataset_short'].unique():
            mask = pooled['dataset_short'] == ds
            c = pooled.loc[mask, 'color'].iloc[0]
            ax5.scatter(pooled.loc[mask, 'mag_z'], pooled.loc[mask, 'stab_z'],
                        c=c, s=20, alpha=0.5, edgecolor='none', label=ds)

        rho_pooled, _ = spearmanr(pooled['mag_z'], pooled['stab_z'])
        slope_p, intercept_p, _, _, _ = linregress(pooled['mag_z'], pooled['stab_z'])
        x_p = np.array([pooled['mag_z'].min(), pooled['mag_z'].max()])
        ax5.plot(x_p, slope_p * x_p + intercept_p, '--', color='gray', linewidth=2, alpha=0.7)

        ax5.set_title(f'Pooled (z-scored)\n$\\rho$ = {rho_pooled:.3f}', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Magnitude (z)', fontsize=10,fontweight='bold')
        ax5.set_ylabel('Stability (z)', fontsize=10,fontweight='bold')
        ax5.legend(fontsize=7, framealpha=0.8)
        sns.despine(ax=ax5)

    for i, label in enumerate('abcdef'):
        axes[i].text(-0.08, 1.08, label, transform=axes[i].transAxes,
                     fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig2_magnitude_stability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig2_magnitude_stability.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fig 2 saved to {OUT_DIR}")
    plt.show()


# ==============================================================================
# FIG 4: scGPT validation (3-panel)
# ==============================================================================

def fig4(df_pca, df_scgpt):
    """3-panel: Norman, Dixit, Replogle in scGPT embeddings."""

    datasets = [
        ('Norman 2019 (CRISPRa)', 'Norman 2019'),
        ('Dixit 2016 (CRISPRi)', 'Dixit 2016'),
        ('Replogle 2022 (CRISPRi)', 'Replogle 2022'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (ds_full, ds_short) in enumerate(datasets):
        ax = axes[i]
        sub = df_scgpt[df_scgpt['dataset'] == ds_full].copy()

        if len(sub) < 3:
            ax.text(0.5, 0.5, f'{ds_short}\nNo scGPT data', transform=ax.transAxes,
                    ha='center', fontsize=10, color='gray')
            continue
        dataset_colors = {
            'Norman 2019 (CRISPRa)': BLUE,
            'Dixit 2016 (CRISPRi)': GREEN,
            'Replogle 2022 (CRISPRi)': GOLD,
        }
        # Then in the loop:
        color = dataset_colors.get(ds_full, BLUE)
        ax.scatter(sub['magnitude'], sub['stability'],
                  c=color, s=40, alpha=0.6, edgecolor='white', linewidth=0.3)


        # ax.scatter(sub['magnitude'], sub['stability'],
        #            c=BLUE, s=40, alpha=0.6, edgecolor='white', linewidth=0.3)

        slope, intercept, _, _, _ = linregress(sub['magnitude'], sub['stability'])
        x_line = np.array([sub['magnitude'].min(), sub['magnitude'].max()])
        ax.plot(x_line, slope * x_line + intercept, '--', color='gray',
                linewidth=2, alpha=0.7)

        rho_scgpt, _ = spearmanr(sub['magnitude'], sub['stability'])

        # PCA rho for comparison (from PCA CSV, NOT scGPT CSV)
        pca_sub = df_pca[df_pca['dataset'] == ds_full]
        if len(pca_sub) > 5:
            rho_pca, _ = spearmanr(pca_sub['magnitude'], pca_sub['stability'])
            rho_text = f'scGPT $\\rho$ = {rho_scgpt:.3f}\nPCA $\\rho$ = {rho_pca:.3f}'
        else:
            rho_text = f'scGPT $\\rho$ = {rho_scgpt:.3f}\nPCA $\\rho$: N/A'

        ax.set_title(f'{ds_short} (n={len(sub)})', fontsize=12, fontweight='bold')
        ax.text(0.97, 0.03, rho_text,
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9))

        ax.set_xlabel('Effect Magnitude (scGPT)', fontsize=11,fontweight='bold')
        ax.set_ylabel('Shesha Stability (scGPT)', fontsize=11,fontweight='bold')
        sns.despine(ax=ax)

    for i, label in enumerate('abc'):
        axes[i].text(-0.08, 1.08, label, transform=axes[i].transAxes,
                     fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig4_scgpt_validation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig4_scgpt_validation.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fig 4 saved to {OUT_DIR}")
    plt.show()


# ==============================================================================
# FIG 5: Combinatorial vs single-gene
# ==============================================================================

def fig5(df_pca):
    """Norman: single-gene vs combinatorial stability distributions."""

    df_n = df_pca[df_pca['dataset'] == 'Norman 2019 (CRISPRa)'].copy()
    df_n['type'] = df_n['perturbation'].apply(
        lambda x: 'Combinatorial' if '+' in str(x) else 'Single gene'
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.5),
                                     gridspec_kw={'width_ratios': [1, 1.5]})

    # Panel A: Violin
    palette = {'Single gene': BLUE, 'Combinatorial': RED}
    sns.violinplot(data=df_n, x='type', y='stability', palette=palette,
                   inner='box', alpha=0.7, ax=ax0)

    single = df_n[df_n['type'] == 'Single gene']['stability']
    combo = df_n[df_n['type'] == 'Combinatorial']['stability']
    u_stat, p_val = mannwhitneyu(single, combo, alternative='two-sided')

    y_max = df_n['stability'].max() + 0.05
    ax0.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
    ax0.text(0.5, y_max + 0.01, f'{sig}\np = {p_val:.2e}', ha='center', fontweight='bold',fontsize=9)

    ax0.set_xlabel('')
    ax0.set_ylabel('Shesha Stability (Cosine)', fontweight='bold',fontsize=11)
    ax0.text(0.03, 0.97, f'Single: n={len(single)}\nCombo: n={len(combo)}',
             transform=ax0.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCC', alpha=0.9))
    sns.despine(ax=ax0)

    # Panel B: Scatter by type
    for typ, color, z in [('Single gene', BLUE, 2), ('Combinatorial', RED, 3)]:
        sub = df_n[df_n['type'] == typ]
        ax1.scatter(sub['magnitude'], sub['stability'],
                    c=color, s=50, alpha=0.6, edgecolor='white', linewidth=0.3,
                    zorder=z, label=typ)
        if len(sub) > 5:
            slope, intercept, _, _, _ = linregress(sub['magnitude'], sub['stability'])
            x_line = np.array([sub['magnitude'].min(), sub['magnitude'].max()])
            ls = '--' if typ == 'Single gene' else ':'
            ax1.plot(x_line, slope * x_line + intercept, ls, color=color,
                     linewidth=2, alpha=0.7)

    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.set_xlabel('Effect Magnitude (Euclidean)', fontweight='bold',fontsize=11)
    ax1.set_ylabel('Shesha Stability (Cosine)', fontweight='bold',fontsize=11)
    sns.despine(ax=ax1)

    for ax, label in zip([ax0, ax1], ['a', 'b']):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig5_combinatorial.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig5_combinatorial.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fig 5 saved to {OUT_DIR}")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    df_pca, df_scgpt = load_data()
    print(f"PCA data: {len(df_pca)} rows, datasets: {df_pca['dataset'].unique().tolist()}")
    print(f"scGPT data: {len(df_scgpt)} rows, datasets: {df_scgpt['dataset'].unique().tolist()}")

    # Fig 2: Replogle PCA needs live computation
    df_replogle_pca = None
    if not SKIP_REPLOGLE:
        try:
            df_replogle_pca = load_replogle_pca()
        except Exception as e:
            print(f"Could not load Replogle PCA: {e}")
            print("Run with --skip-replogle to skip, or run in Colab.")

    fig2(df_pca, df_replogle_pca)
    fig4(df_pca, df_scgpt)
    fig5(df_pca)

    print("\nAll figures generated!")