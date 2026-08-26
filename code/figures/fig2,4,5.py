#!/usr/bin/env python3
"""
Figures 2, 4, 5

Fig 2: unified PCA + scGPT (3 rows)
  - Row 1 (4): Norman, Adamson UPR, Adamson pilot, Dixit (PCA)
  - Row 2 (3): Papalexi, Replogle, pooled z-scored (PCA)
  - Row 3 (4): Norman, Adamson UPR, Dixit, Replogle (within-scGPT)
  - Original density cmaps (Blues / GnBu / Purples / Greens / Oranges / Reds)
    - No density colorbars, no fit lines; coherence capped at 1; Shesha Coherence

Fig 4: three embedding arms vs frozen coherence — not this script (still to build)

Fig 5: Combinatorial vs single-gene (Norman, from PCA CSV)

USAGE:
  # In Colab (with pertpy for Replogle):
  python fig2,4,5.py

  # Locally (frozen CSV; skip live Replogle):
  python "fig2,4,5.py" --fig2-only --skip-replogle
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

from fig_1 import DATASETS_INFO, despine, perturbation_density
from revision_io import find_data_file, resolve_out_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress, mannwhitneyu
from pathlib import Path
import sys

try:
    import seaborn as sns
except ImportError:
    sns = None

import pipeline_config as cfg

SEED = cfg.SEED
np.random.seed(SEED)

# Colors
RED = '#C44E52'
BLUE = '#4C72B0'
GOLD = '#E5A84B'
GREEN = '#2CA02C'
BROWN = '#8C564B'
PURPLE = '#6A51A3'  # Adamson 2016 UPR (new; pilot keeps RED)
GRAY = '#D3D3D3'
DARK_GRAY = '#555555'

SKIP_REPLOGLE = '--skip-replogle' in sys.argv

# --- PATHS ---
def _need(path, label):
    if path is None:
        raise FileNotFoundError(f"Need {label} (set SHESHA_OUT or place it in ./shesha-crispr)")
    return path

PCA_CSV = str(_need(
    find_data_file("frozen_sp_scores.csv", "shesha_crispr_results_euclidean.csv"),
    "frozen_sp_scores.csv",
))
PCA_CSV_ORIGINAL = str(find_data_file("shesha_crispr_results_euclidean.csv") or PCA_CSV)
SCGPT_CSV = str(_need(find_data_file("scgpt_all_datasets.csv"), "scgpt_all_datasets.csv"))
OUT_DIR = resolve_out_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df_pca = pd.read_csv(PCA_CSV)
    df_scgpt = pd.read_csv(SCGPT_CSV)

    # Papalexi from original CSV
    df_orig = pd.read_csv(PCA_CSV_ORIGINAL)
    df_papalexi = df_orig[df_orig['dataset'] == 'Papalexi 2021 (CRISPR-KO)'].copy()
    if 'Papalexi 2021 (CRISPR-KO)' not in df_pca['dataset'].values:
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


# Original sequential cmaps (Norman Blues, Adamson UPR GnBu, pilot Purples,
# Dixit Greens, Papalexi Oranges, Replogle Reds). One swatch per panel;
# no colorbar, no OLS/identity line (annotated statistic is Spearman).
SCGPT_DATASETS = [
    'Norman 2019 (CRISPRa)',
    'Adamson 2016 UPR (CRISPRi)',
    'Dixit 2016 (CRISPR-KO)',
    'Replogle 2022 (CRISPRi)',
]


def _scatter_cloud(ax, x, y, cmap_name):
    """Same density fill as fig_1.py (original cmaps), no colorbar."""
    z = perturbation_density(x, y)
    order = np.argsort(z)
    n = len(x)
    large = n > 400
    ax.scatter(
        x[order], y[order], c=z[order], cmap=cmap_name,
        s=22 if large else 40, alpha=0.8,
        edgecolor='white', linewidth=0.35 if large else 0.5,
        rasterized=large, zorder=2,
    )


def _coherence_floor(*arrays):
    floor = 0.0
    for a in arrays:
        if a is None or len(a) == 0:
            continue
        floor = min(floor, float(np.nanmin(a)))
    return min(-0.08, floor - 0.03)


def _mag_coh_axes(ax, x, y_lo):
    xmax = float(np.nanmax(x)) if len(x) else 1.0
    ax.set_xlim(0, max(xmax * 1.06, 1.0))
    ax.set_ylim(y_lo, 1.0)
    if y_lo < 0:
        ax.axhline(0.0, color='#B0B0B0', linewidth=0.7, zorder=1)


def _rho_box(ax, text):
    ax.text(
        0.97, 0.03, text, transform=ax.transAxes,
        fontsize=8.5, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.9),
    )


# ==============================================================================
# FIG 2: PCA (4+3) + within-scGPT (4)
# ==============================================================================

def fig2(df_pca, df_scgpt=None, df_replogle_pca=None):
    """Three-row Fig 2: PCA row of 4, PCA row of 3, scGPT row of 4."""

    fig, axes2d = plt.subplots(3, 4, figsize=(15.6, 12.2))
    axes2d[1, 3].axis('off')
    pca_axes = [axes2d[0, i] for i in range(4)] + [axes2d[1, i] for i in range(3)]
    scgpt_axes = [axes2d[2, i] for i in range(4)]

    ds_col = df_pca['dataset'].map(cfg.resolve_dataset_name)
    pca_y = []
    for ds_full, *_ in DATASETS_INFO:
        sub = df_pca[ds_col == ds_full]
        if ds_full == 'Replogle 2022 (CRISPRi)' and len(sub) < 3 and df_replogle_pca is not None:
            sub = df_replogle_pca
        if len(sub):
            pca_y.append(sub['stability'].to_numpy(dtype=float))
    y_lo_pca = _coherence_floor(*pca_y)

    all_for_pooled = []
    print('\nFig 2 PCA:')
    for i, (ds_full, ds_short, modality, cmap_name, legend_color) in enumerate(DATASETS_INFO):
        ax = pca_axes[i]
        sub = df_pca[ds_col == ds_full].copy()
        if ds_full == 'Replogle 2022 (CRISPRi)' and len(sub) < 3 and df_replogle_pca is not None:
            sub = df_replogle_pca.copy()

        if len(sub) < 3:
            ax.text(0.5, 0.5, f'{ds_short}\nn={len(sub)}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
            despine(ax)
            continue

        x = sub['magnitude'].to_numpy(dtype=float)
        y = sub['stability'].to_numpy(dtype=float)
        _scatter_cloud(ax, x, y, cmap_name)
        _mag_coh_axes(ax, x, y_lo_pca)
        rho, _ = spearmanr(x, y)
        n = len(sub)
        print(f'  {ds_short}: n={n}  ρ={rho:.3f}')
        ax.set_title(f'{ds_short}\n({modality}, n={n})', fontsize=11, fontweight='bold')
        _rho_box(ax, f'$\\rho$ = {rho:.3f}')
        ax.set_xlabel('Effect Magnitude', fontsize=10, fontweight='bold')
        ax.set_ylabel('Shesha Coherence', fontsize=10, fontweight='bold')
        despine(ax)

        if len(sub) > 5:
            sub_z = sub[['magnitude', 'stability']].copy()
            sub_z['mag_z'] = (sub_z['magnitude'] - sub_z['magnitude'].mean()) / sub_z['magnitude'].std()
            sub_z['stab_z'] = (sub_z['stability'] - sub_z['stability'].mean()) / sub_z['stability'].std()
            sub_z['dataset_short'] = ds_short
            sub_z['color'] = legend_color
            all_for_pooled.append(sub_z)

    ax_p = pca_axes[6]
    if all_for_pooled:
        pooled = pd.concat(all_for_pooled, ignore_index=True)
        for ds in pooled['dataset_short'].unique():
            mask = pooled['dataset_short'] == ds
            c = pooled.loc[mask, 'color'].iloc[0]
            ax_p.scatter(pooled.loc[mask, 'mag_z'], pooled.loc[mask, 'stab_z'],
                         color=c, s=16, alpha=0.4, edgecolor='none', label=ds)
        rho_pooled, _ = spearmanr(pooled['mag_z'], pooled['stab_z'])
        print(f'  Pooled: n={len(pooled)}  ρ={rho_pooled:.3f}')
        ax_p.set_title(f'Pooled (z-scored)\n(n={len(pooled)})',
                       fontsize=11, fontweight='bold')
        _rho_box(ax_p, f'$\\rho$ = {rho_pooled:.3f}')
        ax_p.set_xlabel('Magnitude (z)', fontsize=10, fontweight='bold')
        ax_p.set_ylabel('Coherence (z)', fontsize=10, fontweight='bold')
        ax_p.legend(fontsize=6.5, framealpha=0.85, loc='upper left')
        despine(ax_p)

    if df_scgpt is not None:
        _draw_scgpt_row(scgpt_axes, df_pca, df_scgpt, labels='hijk')
    else:
        for ax in scgpt_axes:
            ax.axis('off')

    for i, label in enumerate('abcdefg'):
        pca_axes[i].text(-0.10, 1.10, label, transform=pca_axes[i].transAxes,
                         fontsize=14, fontweight='bold', va='top', ha='right')

    fig.text(0.012, 0.72, 'PCA', rotation=90, va='center', ha='center',
             fontsize=12, fontweight='bold')
    fig.text(0.012, 0.20, 'scGPT', rotation=90, va='center', ha='center',
             fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    fig.savefig(OUT_DIR / 'fig2_magnitude_stability.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig2_magnitude_stability.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f'Fig 2 (unified) saved to {OUT_DIR}')
    plt.close(fig)


def _draw_scgpt_row(axes, df_pca, df_scgpt, labels='hijk'):
    info = {row[0]: row for row in DATASETS_INFO}
    y_lo = _coherence_floor(*(
        df_scgpt.loc[df_scgpt['dataset'] == ds, 'stability'].to_numpy(dtype=float)
        for ds in SCGPT_DATASETS
    ))
    print('\nFig 2 scGPT (within-embedding):')
    for i, ds_full in enumerate(SCGPT_DATASETS):
        ax = axes[i]
        ds_full, ds_short, modality, cmap_name, legend_color = info[ds_full]
        sub = df_scgpt[df_scgpt['dataset'] == ds_full]
        if len(sub) < 3:
            ax.text(0.5, 0.5, f'{ds_short}\nNo scGPT data',
                    transform=ax.transAxes, ha='center', color='gray')
            despine(ax)
            continue
        x = sub['magnitude'].to_numpy(dtype=float)
        y = sub['stability'].to_numpy(dtype=float)
        _scatter_cloud(ax, x, y, cmap_name)
        _mag_coh_axes(ax, x, y_lo)
        rho_scgpt, _ = spearmanr(x, y)
        pca_sub = df_pca[df_pca['dataset'].map(cfg.resolve_dataset_name) == ds_full]
        if len(pca_sub) > 5:
            rho_pca, _ = spearmanr(pca_sub['magnitude'], pca_sub['stability'])
            rho_text = f'scGPT $\\rho$ = {rho_scgpt:.3f}\nPCA $\\rho$ = {rho_pca:.3f}'
            print(f'  {ds_short}: n={len(sub)}  scGPT ρ={rho_scgpt:.3f}  PCA ρ={rho_pca:.3f}')
        else:
            rho_text = f'scGPT $\\rho$ = {rho_scgpt:.3f}'
            print(f'  {ds_short}: n={len(sub)}  scGPT ρ={rho_scgpt:.3f}')
        ax.set_title(f'{ds_short}\n({modality}, n={len(sub)})',
                     fontsize=11, fontweight='bold')
        _rho_box(ax, rho_text)
        ax.set_xlabel('Effect Magnitude (scGPT)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Shesha Coherence (scGPT)', fontsize=10, fontweight='bold')
        despine(ax)
        ax.text(-0.10, 1.10, labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')


# ==============================================================================
# FIG 2 companion: within-scGPT magnitude–coherence
# (Norman, Adamson UPR, Dixit, Replogle). Not Fig 4.
# ==============================================================================

def fig2_scgpt_redundancy(df_pca, df_scgpt):
    """Standalone 1x4 of the Fig 2 scGPT row (same colors and axis rules)."""
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.2), sharey=True)
    _draw_scgpt_row(axes, df_pca, df_scgpt, labels='abcd')
    fig.tight_layout()
    stem = OUT_DIR / 'fig2_scgpt_redundancy'
    fig.savefig(str(stem) + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Fig 2 scGPT companion saved to {stem}.pdf / .png')
    plt.close(fig)


def fig4(df_pca, df_scgpt):
    """Backward-compatible alias. This plot is the Fig 2 scGPT companion."""
    fig2_scgpt_redundancy(df_pca, df_scgpt)


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
    if sns is None:
        raise ImportError("fig5 requires seaborn. pip install seaborn")
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
    ax0.set_ylabel('Shesha Coherence (Cosine)', fontweight='bold',fontsize=11)
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
    ax1.set_ylabel('Shesha Coherence (Cosine)', fontweight='bold',fontsize=11)
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
    print(f"PCA_CSV={PCA_CSV}")
    print(f"SCGPT_CSV={SCGPT_CSV}")
    print(f"OUT_DIR={OUT_DIR}")

    if '--fig4-only' in sys.argv or '--fig2-scgpt' in sys.argv:
        fig2_scgpt_redundancy(df_pca, df_scgpt)
        print("\nFig 2 scGPT companion generated (not Fig 4).")
        raise SystemExit(0)

    if '--fig5-only' in sys.argv:
        fig5(df_pca)
        print("\nFig 5 generated!")
        raise SystemExit(0)

    ds_resolved = df_pca['dataset'].map(cfg.resolve_dataset_name)
    have_replogle = (ds_resolved == 'Replogle 2022 (CRISPRi)').sum() >= 3
    df_replogle_pca = None
    if not SKIP_REPLOGLE and not have_replogle:
        try:
            df_replogle_pca = load_replogle_pca()
        except Exception as e:
            print(f"Could not load Replogle PCA: {e}")
            print("Run with --skip-replogle to skip, or run in Colab.")

    if '--fig2-only' in sys.argv:
        fig2(df_pca, df_scgpt, df_replogle_pca)
        print("\nFig 2 generated!")
        raise SystemExit(0)

    fig2(df_pca, df_scgpt, df_replogle_pca)
    fig2_scgpt_redundancy(df_pca, df_scgpt)
    fig5(df_pca)

    print("\nAll figures generated!")