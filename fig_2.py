#!/usr/bin/env python3
"""
Figure 2: Geometric Stability Validation

Generates a three-panel figure comparing Shesha stability and magnitude
metrics across CRISPR perturbation datasets, with biological validation
via the DDIT3 stress marker.

USAGE:
    python fig_2.py

REQUIRES:
    - shesha_crispr_results_euclidean.csv (Norman 2019 results)
    - Replogle 2022 K562 dataset (downloaded via pertpy)

PANELS:
    a) Norman 2019 (CRISPRa): Stability vs Magnitude scatter
    b) Replogle K562: Stability vs Magnitude with discordance coloring
    c) Stress Validation: Stability vs DDIT3 expression (biological proxy)

OUTPUT:
    - raju_2026_figure_2.pdf
    - raju_2026_figure_2.png
"""

def main(csv_path: str = 'shesha_crispr_results_euclidean.csv'):
    import scanpy as sc
    import pertpy as pt
    import pandas as pd
    import numpy as np
    from scipy import stats
    from anndata import AnnData
    from shesha.bio import compute_stability, compute_magnitude
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    import os
    import sys
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'.")
        print("Please update the path to the csv file containing the data.")
        sys.exit(1)


    # ==============================================================================
    # PROCESS REPLOGLE DATA
    # ==============================================================================

    # 1. LOAD & CLEAN DATA
    print("Loading Replogle data...")
    adata = pt.dt.replogle_2022_k562_essential()
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)

    # Clean labels
    def clean_label(x):
        if 'non-targeting' in x or x.startswith('chr'): return 'control'
        if 'pos_control' in x: return 'POS_CONTROL'
        return x.split('_')[0]

    adata.obs['condition'] = adata.obs['perturbation'].apply(clean_label)

    # Filter out controls and multiplets
    adata_clean = adata[
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    ].copy()

    # Filter for size (>50 cells)
    counts = adata_clean.obs['condition'].value_counts()
    valid = counts[counts >= 50].index
    adata_final = adata_clean[adata_clean.obs['condition'].isin(valid)].copy()

    # 2. EXTRACT STRESS DATA (CRITICAL STEP)
    # We fetch DDIT3 from the raw data BEFORE any PCA filtering to ensure we have it
    print("Extracting real DDIT3 expression...")
    sc.pp.normalize_total(adata_clean, target_sum=1e4)
    sc.pp.log1p(adata_clean)

    stress_map = {}
    if 'DDIT3' in adata_clean.var_names:
        # Calculate mean DDIT3 for every perturbation
        for pert in valid:
            # Fast boolean indexing on the full object
            mask = adata_clean.obs['condition'] == pert
            val = adata_clean[mask, 'DDIT3'].X.mean()
            if hasattr(val, "item"): val = val.item()
            stress_map[pert] = val
    else:
        print("ERROR: DDIT3 not found in dataset!")

    # 3. RUN SHESHA (GEOMETRY)
    print("Running Shesha geometry...")
    # Process for PCA
    sc.pp.normalize_total(adata_final, target_sum=1e4)
    sc.pp.log1p(adata_final)
    sc.pp.highly_variable_genes(adata_final, n_top_genes=2000, subset=True)
    sc.tl.pca(adata_final, n_comps=50)

    # Compute metrics
    adata_pca = AnnData(X=adata_final.obsm['X_pca'], obs=adata_final.obs)
    stability = compute_stability(adata_pca, perturbation_key='condition', control_label='control', metric='cosine')
    magnitude = compute_magnitude(adata_pca, perturbation_key='condition', control_label='control', metric='euclidean')

    # 4. ASSEMBLE THE DATAFRAME
    df = pd.DataFrame({'stability': pd.Series(stability), 'magnitude': pd.Series(magnitude)})
    if 'control' in df.index: df = df.drop('control')

    # Map the Stress data we calculated earlier
    df['Stress_DDIT3'] = df.index.map(stress_map)
    df['n_cells'] = df.index.map(counts)
    df['discordance'] = stats.zscore(df['magnitude']) - stats.zscore(df['stability'])

    print("\n--- DONE ---")
    print(df[['stability', 'Stress_DDIT3']].head())

    # ==============================================================================
    # PLOT FIGURE 2
    # ==============================================================================

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    from matplotlib.lines import Line2D

    # 1. Setup Figure
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    df_uploaded = pd.read_csv('shesha_crispr_results_euclidean.csv')
    df_norman = df_uploaded[df_uploaded['dataset'] == 'Norman 2019 (CRISPRa)'].copy()

    # PANEL A: Norman 2019 (Colorbar Legend)
    ax0 = fig.add_subplot(gs[0, 0])
    sc0 = ax0.scatter(
        df_norman['magnitude'], df_norman['stability'],
        c=df_norman['stability'], cmap='Blues',
        s=60, alpha=0.7, edgecolor='grey', linewidth=0.5
    )
    slope, intercept, r_val, _, _ = stats.linregress(df_norman['magnitude'], df_norman['stability'])
    x_vals = np.array([df_norman['magnitude'].min(), df_norman['magnitude'].max()])
    # ax0.plot(x_vals, slope * x_vals + intercept, '--', color='#514949', linewidth=3, alpha=0.8)
    ax0.plot(x_vals, slope * x_vals + intercept, '--', color='gray', linewidth=3, alpha=0.5)

    cbar = plt.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label('Perturbation Density', rotation=90, labelpad=15)
    cbar.ax.yaxis.set_label_position('left')

    rho, _ = stats.spearmanr(df_norman['magnitude'], df_norman['stability'])
    # ax0.set_title(f'Norman 2019 (CRISPRa)\n(n={len(df_norman)}, ρ={rho:.3f})', fontsize=14, weight='bold')
    ax0.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
    ax0.set_ylabel('Shesha Stability (Cosine)', fontsize=12)
    sns.despine(ax=ax0)

    # PANEL B: Replogle K562 (Manual Shifts)
    ax1 = fig.add_subplot(gs[0, 1])

    # Scatter
    sns.scatterplot(
        data=df, x='magnitude', y='stability',
        hue='discordance', palette='RdBu_r', hue_norm=(-2.5, 2.5),
        size='n_cells', sizes=(50, 400),
        alpha=0.7, edgecolor='k', linewidth=0.5, legend=False, ax=ax1
    )
    sns.regplot(
        data=df, x='magnitude', y='stability',
        scatter=False, color='gray', line_kws={'linestyle': '--', 'alpha': 0.5}, ax=ax1
    )

    # --- MANUAL LABEL SHIFTS ---
    # (dx, dy) -> Positive Y is UP, Positive X is RIGHT
    shifts = {
        'CEBPB': (0.00,  0.008),  # Red: Shift Up
        'ACTB':  (0.00,  0.008),  # Red: Shift Up
        'BLVRB': (0.00,  0.008),  # Blue: Shift Up
        'BUB3':  (0.04, -0.005),  # Yellow: Right & Down
        'CENPW': (0.04, -0.005)   # Yellow: Right & Down
    }

    highlights_b = {
        'CEBPB': '#a50026', 'BLVRB': '#313695',
        'BUB3': 'orange', 'CENPW': 'orange', 'ACTB': '#a50026'
    }

    for gene, color in highlights_b.items():
        if gene in df.index:
            x, y = df.loc[gene, ['magnitude', 'stability']]
            # Get offset for this gene, default to 0 if not listed
            dx, dy = shifts.get(gene, (0, 0))

            # Draw Ring
            ax1.scatter(x, y, s=400, facecolors='none', edgecolors=color, linewidth=2.5)
            # Draw Text with Offset
            ax1.text(x + dx, y + dy, gene, weight='bold', color=color, fontsize=11, ha='left' if dx > 0 else 'center')

    # Legend
    disc_breaks = [-1.6, -0.8, 0.0, 0.8, 1.6]
    cell_breaks = [600, 800, 1000, 1200, 1400]
    norm = plt.Normalize(-2.5, 2.5)
    cmap = plt.cm.RdBu_r
    legend_elements = (
        [Line2D([0], [0], marker='', color='w', label=r'$\bf{Discordance}$')] +
        [Line2D([0], [0], marker='o', color='w', label=f'{x}', markerfacecolor=cmap(norm(x)), markersize=8, markeredgecolor='gray') for x in disc_breaks] +
        [Line2D([0], [0], marker='', color='w', label='')] +
        [Line2D([0], [0], marker='', color='w', label=r'$\bf{number of cells}$')] +
        [Line2D([0], [0], marker='o', color='w', label=f'{x}', markerfacecolor='gray', markersize=s, markeredgecolor='k') for x, s in zip(cell_breaks, [6, 8, 10, 12, 14])]
    )
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)

    # ax1.set_title('Geometric Stability vs. Magnitude\n(Replogle K562)', fontsize=14, weight='bold')
    ax1.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
    ax1.set_ylabel('Shesha Stability (Cosine)', fontsize=12)
    sns.despine(ax=ax1)

    # PANEL C: Stress Validation (with Manual Label Positioning)
    ax2 = fig.add_subplot(gs[0, 2])
    df_stress = df.dropna(subset=['Stress_DDIT3'])

    # Define colors
    nature_blue = '#2C5AA0'
    point_color = '#4c72b0'
    ci_color = '#BEBEBE'

    # Get stability values for color gradient
    stab_norm = (df_stress['stability'] - df_stress['stability'].min()) / \
                (df_stress['stability'].max() - df_stress['stability'].min())

    # Create custom scatter with gradient fill
    scatter = ax2.scatter(
        df_stress['stability'], 
        df_stress['Stress_DDIT3'],
        s=110,
        c=stab_norm,
        cmap='Blues',
        alpha=0.75,
        edgecolor='#888888',
        linewidth=0.8,
        zorder=3
    )

    # Regression line and CI
    from scipy import stats as sp_stats
    import numpy as np

    x = df_stress['stability'].values
    y = df_stress['Stress_DDIT3'].values

    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_pred + intercept

    n = len(x)
    mean_x = np.mean(x)
    se_y = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
    se_pred = se_y * np.sqrt(1/n + (x_pred - mean_x)**2 / np.sum((x - mean_x)**2))
    ci = 1.96 * se_pred

    ax2.fill_between(x_pred, y_pred - ci, y_pred + ci, 
                    color=ci_color, alpha=0.25, zorder=1)
    ax2.plot(x_pred, y_pred, color=nature_blue, linewidth=2.2, zorder=2)

    r_stress, p_stress = sp_stats.spearmanr(df_stress['stability'], df_stress['Stress_DDIT3'])

    ax2.set_xlabel('Shesha Stability', fontsize=13, fontfamily='sans-serif', fontweight='medium')
    ax2.set_ylabel('Cellular Stress (DDIT3 Expression)', fontsize=13, fontfamily='sans-serif', fontweight='medium')
    ax2.tick_params(axis='both', labelsize=10)

    # Expand axes
    y_min, y_max = df_stress['Stress_DDIT3'].min(), df_stress['Stress_DDIT3'].max()
    y_padding = (y_max - y_min) * 0.12
    ax2.set_ylim(y_min - y_padding, y_max + y_padding)

    x_min, x_max = df_stress['stability'].min(), df_stress['stability'].max()
    x_padding = (x_max - x_min) * 0.08
    ax2.set_xlim(x_min - x_padding, x_max + x_padding)

    sns.despine(ax=ax2)

    # MANUAL LABEL POSITIONING
    # Define manual offsets: (x_offset, y_offset) relative to data point
    # Positive y = up, negative y = down
    # Positive x = right, negative x = left

    label_offsets = {
        'CALM3':   (0.000,  0.004),   # up a hair
        'ATP5J2':  (0.000,  0.004),   # up a hair
        'BTF3L4':  (0.000,  0.004),   # up a hair
        'BTF3':    (0.000,  0.004),   # up a hair
        'CEBPB':   (0.000,  0.003),   # up a hair
        'ARID1A':  (0.000,  0.003),   # up a hair
        'ACTB':    (0.000, -0.004),   # down a hair
        'BLVRB':   (0.004, -0.003),   # right a hair, down a little
    }

    # Target genes to label
    targets_c = ['CALM3', 'ATP5J2', 'BTF3', 'BTF3L4', 'BLVRB', 'ACTB', 'ARID1A', 'CEBPB']

    for gene in targets_c:
        if gene in df_stress.index:
            x_pos = df_stress.loc[gene, 'stability']
            y_pos = df_stress.loc[gene, 'Stress_DDIT3']
            
            # Get offset (default to no offset if not specified)
            x_off, y_off = label_offsets.get(gene, (0, 0))
            
            # Add annotation with leader line
            ax2.annotate(
                gene,
                xy=(x_pos, y_pos),  # Point location
                xytext=(x_pos + x_off, y_pos + y_off),  # Text location
                fontsize=9,
                fontfamily='sans-serif',
                color='#333333',
                ha='center',
                va='bottom' if y_off >= 0 else 'top',
                arrowprops=dict(
                    arrowstyle='-',
                    color='#888888',
                    lw=0.5,
                    shrinkA=0,
                    shrinkB=3
                ) if (abs(x_off) > 0.003 or abs(y_off) > 0.003) else None
            )
    # ADD LABELS (a, b, c)
    # We place text relative to the Axes (0,0 is bottom-left, 1,1 is top-right)
    # (-0.1, 1.1) places it outside top-left.
    for ax, label in zip([ax0, ax1, ax2], ['a', 'b', 'c']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')

    # For both formats
    plt.savefig('raju_2026_figure_2.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('raju_2026_figure_2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == "__main__":
    # Change the csv_path to the path to the csv file containing the data
    main(csv_path='shesha_crispr_results_euclidean.csv')