#!/usr/bin/env python3
"""
Paper 3 Figure 3: Geometric Instability and Cellular Stress

3-panel figure:
  a) HSPA5 vs stability scatter in Replogle (n=1832) with quadrant annotation
  b) HSPA5 vs stability scatter in Dixit (n=153) with quadrant annotation
  c) Raw vs partial rho dot plot (from stress CSVs, no pertpy needed)

Panels a/b: need Colab with pertpy (extract HSPA5 expression from raw adata)
Panel c: reads from stress_partial_correlations.csv (run locally or Colab)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
from matplotlib.lines import Line2D
from pathlib import Path
import scanpy as sc
import pertpy as pt
from anndata import AnnData
from shesha.bio import compute_stability, compute_magnitude

SEED = 320
np.random.seed(SEED)

DATA_DIR = Path("./shesha-crispr")

OUT_DIR = DATA_DIR

# Colors
RED = '#C44E52'
BLUE = '#4C72B0'
GREEN = '#2CA02C'
DARK_GRAY = '#555555'
HSPA5_COLOR = '#8B0000'  # dark red for HSPA5 emphasis


def extract_stress_and_shesha(loader_func, perturbation_key, control_label,
                                clean_func=None, min_cells=50):
    """Load dataset, compute stability/magnitude, extract HSPA5 expression."""
    print("Loading dataset...")
    adata = loader_func()
    if clean_func:
        adata = clean_func(adata)
    adata.obs[perturbation_key] = adata.obs[perturbation_key].astype(str)

    # Extract HSPA5 BEFORE subsetting genes
    print("Extracting HSPA5 expression...")
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    counts = adata_norm.obs[perturbation_key].value_counts()
    valid = counts[counts >= min_cells].index
    valid = [v for v in valid if v != control_label]

    hspa5_map = {}
    if 'HSPA5' in adata_norm.var_names:
        for pert in valid:
            mask = adata_norm.obs[perturbation_key] == pert
            val = adata_norm[mask, 'HSPA5'].X.mean()
            if hasattr(val, "item"): val = val.item()
            hspa5_map[pert] = val
        print(f"  HSPA5 extracted for {len(hspa5_map)} perturbations")
    else:
        print("  WARNING: HSPA5 not found!")

    # Compute stability/magnitude
    print("Computing Shesha metrics...")
    adata_proc = adata[adata.obs[perturbation_key].isin(list(valid) + [control_label])].copy()
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    sc.pp.highly_variable_genes(adata_proc, n_top_genes=2000, subset=True)
    sc.tl.pca(adata_proc, n_comps=50)

    adata_pca = AnnData(X=adata_proc.obsm['X_pca'], obs=adata_proc.obs)
    stab = compute_stability(adata_pca, perturbation_key=perturbation_key,
                              control_label=control_label, metric='cosine')
    mag = compute_magnitude(adata_pca, perturbation_key=perturbation_key,
                             control_label=control_label, metric='euclidean')

    df = pd.DataFrame({'stability': pd.Series(stab), 'magnitude': pd.Series(mag)})
    if control_label in df.index:
        df = df.drop(control_label)
    df = df[df.index.isin(valid)].copy()
    df['hspa5'] = df.index.map(hspa5_map)

    return df


def clean_replogle(adata):
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    def clean_label(x):
        if 'non-targeting' in x or x.startswith('chr'): return 'control'
        if 'pos_control' in x: return 'POS_CONTROL'
        return x.split('_')[0]
    adata.obs['condition'] = adata.obs['perturbation'].apply(clean_label)
    return adata[
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    ].copy()


def panel_stress_scatter(ax, df, dataset_name, raw_rho, partial_rho,
                          hh_obs, hh_exp, hh_p_str):
    """Panels a/b: HSPA5 vs stability scatter with quadrant annotation."""
    sub = df.dropna(subset=['hspa5']).copy()
    x = sub['stability'].values
    y = sub['hspa5'].values

    # Scatter
    ax.scatter(x, y, c=BLUE, s=40, alpha=0.5, edgecolor='white', linewidth=0.3)

    # Regression + CI
    slope, intercept, _, _, _ = linregress(x, y)
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_pred + intercept
    n = len(x)
    mean_x = np.mean(x)
    se_y = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
    se_pred = se_y * np.sqrt(1/n + (x_pred - mean_x)**2 / np.sum((x - mean_x)**2))
    ci = 1.96 * se_pred
    ax.fill_between(x_pred, y_pred - ci, y_pred + ci, color='gray', alpha=0.15)
    ax.plot(x_pred, y_pred, color=RED, linewidth=2)

    # Quadrant lines
    med_x, med_y = np.median(x), np.median(y)
    ax.axvline(med_x, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(med_y, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    # HH depletion annotation (top-right quadrant)
    ax.text(0.97, 0.97,
            f'HH: {hh_obs} / {hh_exp:.0f} exp.\np = {hh_p_str}',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3F3',
                      edgecolor=RED, alpha=0.9))

    # Correlation annotations
    ax.text(0.03, 0.03,
            f'raw $\\rho$ = {raw_rho:.3f}\npartial $\\rho$ = {partial_rho:.3f}',
            transform=ax.transAxes, fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCC', alpha=0.9))

    short = dataset_name.split(' (')[0]
    ax.set_title(f'{short} (n={len(sub)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Shesha Stability', fontweight='bold', fontsize=11)
    ax.set_ylabel('HSPA5 (BiP) Expression', fontweight='bold', fontsize=11)
    sns.despine(ax=ax)


def panel_raw_vs_partial(ax, partial_csv_path):
    """Panel c: dot plot from stress_partial_correlations.csv."""
    df_corr = pd.read_csv(partial_csv_path)

    markers = ['DDIT3', 'ATF4', 'XBP1', 'HSPA5']
    datasets_order = ['Dixit 2016 (CRISPRi)', 'Norman 2019 (CRISPRa)', 'Replogle 2022 (CRISPRi)']
    ds_short = {
        'Dixit 2016 (CRISPRi)': 'Dixit',
        'Norman 2019 (CRISPRa)': 'Norman',
        'Replogle 2022 (CRISPRi)': 'Replogle',
    }
    ds_colors = {
        'Dixit 2016 (CRISPRi)': BLUE,
        'Norman 2019 (CRISPRa)': RED,
        'Replogle 2022 (CRISPRi)': GREEN,
    }

    # Build y-positions
    y_positions = {}
    y = 0
    for marker in markers:
        for ds in datasets_order:
            y_positions[(marker, ds)] = y
            y += 1
        y += 0.8  # gap between markers

    for _, row in df_corr.iterrows():
        key = (row['marker'], row['dataset'])
        if key not in y_positions:
            continue
        yp = y_positions[key]
        color = ds_colors.get(row['dataset'], DARK_GRAY)

        raw = row['rho_raw']
        partial = row['rho_partial']

        # Open circle: raw
        ax.scatter(raw, yp, s=80, facecolors='none',
                   edgecolors=color, linewidth=1.5, zorder=3)
        # Filled circle: partial
        fill_alpha = 1.0 if row['survives_magnitude_control'] else 0.3
        ax.scatter(partial, yp, s=80, facecolors=color,
                   edgecolors=color, linewidth=1, alpha=fill_alpha, zorder=4)
        # Arrow
        ax.annotate('', xy=(partial, yp), xytext=(raw, yp),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4))

    # Y-axis labels
    yticks = []
    ylabels = []
    for marker in markers:
        for ds in datasets_order:
            yticks.append(y_positions[(marker, ds)])
            ylabels.append(ds_short.get(ds, ds))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)

    # Marker group labels on right
    for marker in markers:
        mid_y = np.mean([y_positions[(marker, ds)] for ds in datasets_order])
        is_headline = marker == 'HSPA5'
        ax.text(1.02, mid_y, marker, transform=ax.get_yaxis_transform(),
                fontsize=10, fontweight='bold', va='center', ha='left',
                color=HSPA5_COLOR if is_headline else DARK_GRAY)

    # Horizontal separators between marker groups
    for i, marker in enumerate(markers[1:], 1):
        first_y = y_positions[(marker, datasets_order[0])]
        ax.axhline(first_y - 0.4, color='#EEEEEE', linewidth=1, zorder=0)

    ax.axvline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.set_xlabel('Spearman $\\rho$', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Partial Correlation\n(controlling for magnitude)', fontsize=11, fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=DARK_GRAY, markeredgewidth=1.5, markersize=9,
               label='Raw $\\rho$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DARK_GRAY,
               markeredgecolor=DARK_GRAY, markersize=9,
               label='Partial $\\rho$ (survives)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DARK_GRAY,
               markeredgecolor=DARK_GRAY, markersize=9, alpha=0.3,
               label='Partial $\\rho$ (n.s.)'),
        Line2D([0], [0], marker='', color='w', label=''),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=BLUE,
               markersize=8, label='Dixit'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=RED,
               markersize=8, label='Norman'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=GREEN,
               markersize=8, label='Replogle'),
    ]
    # ax.legend(handles=legend_elements, loc='lower left', fontsize=7.5, framealpha=0.9)
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 0.25), fontsize=7.5, framealpha=0.9)
    ax.invert_yaxis()
    sns.despine(ax=ax)


def main():
    # =========================================================================
    # PROCESS REPLOGLE (panels a)
    # =========================================================================
    print("=== REPLOGLE ===")
    df_replogle = extract_stress_and_shesha(
        loader_func=pt.dt.replogle_2022_k562_essential,
        perturbation_key='condition',
        control_label='control',
        clean_func=clean_replogle,
        min_cells=50
    )
    print(f"Replogle: {len(df_replogle)} perturbations, HSPA5 available: {df_replogle['hspa5'].notna().sum()}")

    # =========================================================================
    # PROCESS DIXIT (panel b)
    # =========================================================================
    print("\n=== DIXIT ===")
    df_dixit = extract_stress_and_shesha(
        loader_func=pt.dt.dixit_2016,
        perturbation_key='perturbation_name',
        control_label='control',
        min_cells=10
    )
    print(f"Dixit: {len(df_dixit)} perturbations, HSPA5 available: {df_dixit['hspa5'].notna().sum()}")

    # =========================================================================
    # LOAD PARTIAL CORRELATION CSV (panel c)
    # =========================================================================
    partial_csv = DATA_DIR / "stress_partial_correlations.csv"
    print(f"\nLoading partial correlations from {partial_csv}")

    # =========================================================================
    # FIGURE 3
    # =========================================================================
    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

    # Panel A: Replogle HSPA5
    ax0 = fig.add_subplot(gs[0, 0])
    panel_stress_scatter(ax0, df_replogle, 'Replogle 2022 (CRISPRi)',
                          raw_rho=-0.403, partial_rho=-0.206,
                          hh_obs=301, hh_exp=458.0, hh_p_str='< 0.0001')

    # Panel B: Dixit HSPA5
    ax1 = fig.add_subplot(gs[0, 1])
    panel_stress_scatter(ax1, df_dixit, 'Dixit 2016 (CRISPRi)',
                          raw_rho=-0.313, partial_rho=-0.338,
                          hh_obs=29, hh_exp=38.8, hh_p_str='0.040')

    # Panel C: Raw vs partial dot plot
    ax2 = fig.add_subplot(gs[0, 2])
    panel_raw_vs_partial(ax2, str(partial_csv))

    # Panel labels
    for ax, label in zip([ax0, ax1, ax2], ['a', 'b', 'c']):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig3_stress.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'fig3_stress.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFig 3 saved to {OUT_DIR}")
    plt.show()


if __name__ == '__main__':
    main()