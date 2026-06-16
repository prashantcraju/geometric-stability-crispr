#!/usr/bin/env python3
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
from pathlib import Path

SEED = 320
np.random.seed(SEED)

DATA_DIR = Path("./shesha-crispr")

csv_path = str(DATA_DIR / "shesha_crispr_results_euclidean.csv")

# ==============================================================================
# LOAD NORMAN FROM CSV (for Panel D)
# ==============================================================================
df_uploaded = pd.read_csv(csv_path)
df_norman = df_uploaded[df_uploaded['dataset'] == 'Norman 2019 (CRISPRa)'].copy()

# ==============================================================================
# PROCESS NORMAN DATA via Shesha pipeline (for Panel E)
# ==============================================================================

# 1. LOAD & CLEAN DATA
print("Loading Norman data...")
adata_n = pt.dt.norman_2019()

# perturbation_name already uses 'control' for NegCtrl cells
adata_n.obs['condition'] = adata_n.obs['perturbation_name'].astype(str)

# Filter for size (>50 cells)
counts_n = adata_n.obs['condition'].value_counts()
valid_n = counts_n[counts_n >= 50].index
adata_n_final = adata_n[adata_n.obs['condition'].isin(valid_n)].copy()

# 2. RUN SHESHA (GEOMETRY)
# Norman .X is already normalized; restore raw counts from layers['counts']
print("Running Shesha geometry on Norman...")
adata_n_raw = adata_n_final.copy()
adata_n_raw.X = adata_n_raw.layers['counts'].copy()
sc.pp.normalize_total(adata_n_raw, target_sum=1e4)
sc.pp.log1p(adata_n_raw)
sc.pp.highly_variable_genes(adata_n_raw, n_top_genes=2000, subset=True)
sc.tl.pca(adata_n_raw, n_comps=50)

# 3. COMPUTE METRICS
adata_n_pca = AnnData(X=adata_n_raw.obsm['X_pca'], obs=adata_n_raw.obs)
stability_n = compute_stability(adata_n_pca, perturbation_key='condition', control_label='control', metric='cosine')
magnitude_n = compute_magnitude(adata_n_pca, perturbation_key='condition', control_label='control', metric='euclidean')

# 4. ASSEMBLE THE DATAFRAME
df_norman_e = pd.DataFrame(
    {'stability': list(stability_n.values()), 'magnitude': list(magnitude_n.values())},
    index=list(stability_n.keys())
)
if 'control' in df_norman_e.index: df_norman_e = df_norman_e.drop('control')
df_norman_e = df_norman_e[~df_norman_e.index.duplicated(keep='first')]
df_norman_e['n_cells'] = pd.to_numeric(df_norman_e.index.map(counts_n), errors='coerce')
df_norman_e['discordance'] = stats.zscore(df_norman_e['magnitude']) - stats.zscore(df_norman_e['stability'])

print(f"Norman df shape: {df_norman_e.shape}, magnitude range: {df_norman_e['magnitude'].min():.2f} - {df_norman_e['magnitude'].max():.2f}")

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
df = df[~df.index.duplicated(keep='first')]

# Map the Stress data we calculated earlier
df['Stress_DDIT3'] = df.index.map(stress_map)
df['n_cells'] = pd.to_numeric(df.index.map(counts), errors='coerce')
df['discordance'] = stats.zscore(df['magnitude']) - stats.zscore(df['stability'])

print("\n--- DONE ---")
# Print top discordant (high magnitude, low stability) and concordant genes
df_sorted_disc = df.sort_values('discordance', ascending=False)
df_sorted_conc = df.sort_values('discordance', ascending=True)
print("Top discordant (red candidates):", list(df_sorted_disc.index[:10]))
print("Top concordant (blue candidates):", list(df_sorted_conc.index[:10]))
# Check specific genes
for g in ['CEBPB', 'ACTB', 'BLVRB', 'BUB3', 'CENPW', 'GATA1', 'CHMP3', 'LSG1']:
    print(f"  {g}: {'FOUND' if g in df.index else 'MISSING'}")

# ==============================================================================
# PLOT
# ==============================================================================
fig = plt.figure(figsize=(20, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3)

# ============================================================
# PANEL D: Norman single-gene CEBPA vs KLF1
# ============================================================
ax0 = fig.add_subplot(gs[0, 0])

is_cebpa = df_norman['perturbation'] == 'CEBPA'
is_klf1 = df_norman['perturbation'] == 'KLF1'
is_other = ~is_cebpa & ~is_klf1

ax0.scatter(df_norman.loc[is_other, 'magnitude'],
            df_norman.loc[is_other, 'stability'],
            c='#CCCCCC', s=50, alpha=0.5, edgecolor='none', zorder=1)
ax0.scatter(df_norman.loc[is_klf1, 'magnitude'],
            df_norman.loc[is_klf1, 'stability'],
            c='#2166AC', s=120, alpha=0.9, edgecolor='white', linewidth=1,
            zorder=3, label='KLF1 (Lineage Specific)')
ax0.scatter(df_norman.loc[is_cebpa, 'magnitude'],
            df_norman.loc[is_cebpa, 'stability'],
            c='#B2182B', s=120, alpha=0.9, edgecolor='white', linewidth=1,
            zorder=3, label='CEBPA (Pleiotropic)')

slope, intercept, _, _, _ = stats.linregress(df_norman['magnitude'], df_norman['stability'])
x_vals = np.array([df_norman['magnitude'].min(), df_norman['magnitude'].max()])
ax0.plot(x_vals, slope * x_vals + intercept, '--', color='gray', linewidth=3, alpha=0.5)

for gene, color in [('CEBPA', '#B2182B'), ('KLF1', '#2166AC')]:
    row = df_norman[df_norman['perturbation'] == gene]
    if len(row) > 0:
        ax0.annotate(gene, xy=(row['magnitude'].values[0], row['stability'].values[0]),
                     xytext=(0, 8), textcoords='offset points',
                     fontsize=10, fontweight='bold', color=color, ha='center')

ax0.legend(loc='upper left', fontsize=8, framealpha=0.9, edgecolor='#CCCCCC')
ax0.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
ax0.set_ylabel('Shesha Stability (Cosine)', fontsize=12)
sns.despine(ax=ax0)

# ============================================================
# PANEL E: Norman gene families — CEBP (red) vs KLF1 (blue)
# Same visual encoding as fig_2.py Panel B, Norman data
# ============================================================
ax1 = fig.add_subplot(gs[0, 1])

# Scatter
sns.scatterplot(
    data=df_norman_e, x='magnitude', y='stability',
    hue='discordance', palette='RdBu_r', hue_norm=(-2.5, 2.5),
    size='n_cells', sizes=(50, 400),
    alpha=0.7, edgecolor='k', linewidth=0.5, legend=False, ax=ax1
)
sns.regplot(
    data=df_norman_e, x='magnitude', y='stability',
    scatter=False, color='gray', line_kws={'linestyle': '--', 'alpha': 0.5}, ax=ax1
)

# All CEBP family members and their combinations (red rings)
cebp_genes = [g for g in df_norman_e.index
              if 'CEBP' in g and 'KLF1' not in g]

# All KLF1 combinations (blue rings), excluding shared CEBP+KLF1 combos
klf1_genes = [g for g in df_norman_e.index
              if 'KLF1' in g and 'CEBP' not in g]

for gene in cebp_genes:
    row = df_norman_e.loc[[gene]]
    ax1.scatter(row['magnitude'].values[0], row['stability'].values[0],
                s=300, facecolors='none', edgecolors='#a50026', linewidth=2.0)

for gene in klf1_genes:
    row = df_norman_e.loc[[gene]]
    ax1.scatter(row['magnitude'].values[0], row['stability'].values[0],
                s=300, facecolors='none', edgecolors='#313695', linewidth=2.0)

# --- MANUAL LABEL SHIFTS ---
# (dx, dy) -> Positive Y is UP, Positive X is RIGHT
shifts_e = {
    'CEBPA':     (0.00,  0.015),
    'CEBPB':     (0.00,  0.015),
    'CEBPE':     (0.00,  0.015),
    'CEBPA+JUN': (0.00,  0.015),
    'KLF1':      (0.00,  0.015),
    'KLF1+SET':  (0.00,  0.015),
}

labels_e = {
    'CEBPA': '#a50026', 'CEBPB': '#a50026', 'CEBPE': '#a50026',
    'CEBPA+JUN': '#a50026',
    'KLF1': '#313695', 'KLF1+SET': '#313695',
}

for gene, color in labels_e.items():
    if gene in df_norman_e.index:
        row = df_norman_e.loc[[gene]]
        x, y = row['magnitude'].values[0], row['stability'].values[0]
        dx, dy = shifts_e.get(gene, (0, 0))
        ax1.text(x + dx, y + dy, gene, weight='bold', color=color, fontsize=11,
                 ha='left' if dx > 0 else 'center')

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

ax1.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
ax1.set_ylabel('Shesha Stability (Cosine)', fontsize=12)
sns.despine(ax=ax1)

# PANEL F: Replogle K562 (Manual Shifts)
ax2 = fig.add_subplot(gs[0, 2])

# Scatter
sns.scatterplot(
    data=df, x='magnitude', y='stability',
    hue='discordance', palette='RdBu_r', hue_norm=(-2.5, 2.5),
    size='n_cells', sizes=(50, 400),
    alpha=0.7, edgecolor='k', linewidth=0.5, legend=False, ax=ax2
)
sns.regplot(
    data=df, x='magnitude', y='stability',
    scatter=False, color='gray', line_kws={'linestyle': '--', 'alpha': 0.5}, ax=ax2
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
        row = df.loc[[gene]]
        x, y = row['magnitude'].values[0], row['stability'].values[0]
        # Get offset for this gene, default to 0 if not listed
        dx, dy = shifts.get(gene, (0, 0))

        # Draw Ring
        ax2.scatter(x, y, s=400, facecolors='none', edgecolors=color, linewidth=2.5)
        # Draw Text with Offset
        ax2.text(x + dx, y + dy, gene, weight='bold', color=color, fontsize=11, ha='left' if dx > 0 else 'center')

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
ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)

# ax2.set_title('Geometric Stability vs. Magnitude\n(Replogle K562)', fontsize=14, weight='bold')
ax2.set_xlabel('Effect Magnitude (Euclidean)', fontsize=12)
ax2.set_ylabel('Shesha Stability (Cosine)', fontsize=12)
sns.despine(ax=ax2)

# ============================================================
# PANEL LABELS
# ============================================================
# for ax, label in zip([ax0, ax1, ax2], ['d', 'e', 'f']):
#     ax.text(-0.1, 1.05, label, transform=ax.transAxes,
#             fontsize=16, fontweight='bold', va='top', ha='right')

# Save
plt.savefig(str(DATA_DIR / 'fig1_bottom_row.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(str(DATA_DIR / 'fig1_bottom_row.png'), dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {DATA_DIR}")
plt.show()