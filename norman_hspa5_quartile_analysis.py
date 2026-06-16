#!/usr/bin/env python3
"""
Norman 2019 CRISPRa — HSPA5 × Discordance Stratified by Magnitude Quartile

Reviewer-requested analysis: stratify Norman perturbations by magnitude
quartile, split each bin at median LOESS-residual discordance, and test
whether high-discordance perturbations show elevated HSPA5 expression.

OUTPUT (saved to OUTPUT_DIR):
  norman_hspa5_quartile_analysis.csv   — per-perturbation table
  norman_hspa5_quartile_summary.csv    — quartile-level summary
  norman_hspa5_quartile_figure.pdf/png — panel figure
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

import scanpy as sc
import pertpy as pt

from anndata import AnnData
from scipy.stats import mannwhitneyu, spearmanr
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
np.random.seed(SEED)
MIN_CELLS = 50
LOESS_FRAC = 0.3

OUTPUT_DIR = Path("./shesha-crispr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD AND PROCESS NORMAN 2019
# =============================================================================

print("=" * 80)
print("NORMAN 2019 — HSPA5 × DISCORDANCE STRATIFIED BY MAGNITUDE QUARTILE")
print("=" * 80)

print("\n>>> Loading Norman 2019 (CRISPRa)...")
adata = pt.dt.norman_2019()
pert_col = 'perturbation_name'
ctrl_label = 'control'
adata.obs[pert_col] = adata.obs[pert_col].astype(str)

# Normalize and log-transform (keep all genes for HSPA5 extraction)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Filter perturbations by cell count
counts = adata.obs[pert_col].value_counts()
valid_perts = [p for p in counts[counts >= MIN_CELLS].index if p != ctrl_label]
print(f"    {len(valid_perts)} perturbations with >= {MIN_CELLS} cells")

# =============================================================================
# EXTRACT PER-PERTURBATION MEAN HSPA5
# =============================================================================

print("\n>>> Extracting mean HSPA5 expression per perturbation...")

if 'HSPA5' not in adata.var_names:
    raise ValueError("HSPA5 not found in Norman var_names. "
                     f"Available UPR genes: {[g for g in ['HSPA5','HSP90B1','CALR','XBP1','DDIT3'] if g in adata.var_names]}")

hspa5_idx = list(adata.var_names).index('HSPA5')
hspa5_scores = {}
for pert in valid_perts:
    mask = adata.obs[pert_col] == pert
    expr = adata[mask, hspa5_idx].X
    if issparse(expr):
        expr = expr.toarray()
    hspa5_scores[pert] = float(np.mean(expr))

ctrl_mask = adata.obs[pert_col] == ctrl_label
ctrl_expr = adata[ctrl_mask, hspa5_idx].X
if issparse(ctrl_expr):
    ctrl_expr = ctrl_expr.toarray()
ctrl_hspa5 = float(np.mean(ctrl_expr))
print(f"    Control mean HSPA5 = {ctrl_hspa5:.4f}")
print(f"    Perturbation HSPA5 range: [{min(hspa5_scores.values()):.4f}, {max(hspa5_scores.values()):.4f}]")

# =============================================================================
# COMPUTE SHESHA STABILITY AND MAGNITUDE IN PCA SPACE
# =============================================================================

print("\n>>> Computing Shesha stability and magnitude...")

adata_hvg = adata[adata.obs[pert_col].isin(valid_perts + [ctrl_label])].copy()
sc.pp.highly_variable_genes(adata_hvg, n_top_genes=2000, subset=True)
sc.tl.pca(adata_hvg, n_comps=50, random_state=SEED)

adata_pca = AnnData(X=adata_hvg.obsm['X_pca'], obs=adata_hvg.obs)
stab = compute_stability(adata_pca, perturbation_key=pert_col,
                         control_label=ctrl_label, metric='cosine')
mag = compute_magnitude(adata_pca, perturbation_key=pert_col,
                        control_label=ctrl_label, metric='euclidean')

# =============================================================================
# BUILD PER-PERTURBATION TABLE
# =============================================================================

df = pd.DataFrame({
    'Sp': pd.Series(stab),
    'Mp': pd.Series(mag),
})
if ctrl_label in df.index:
    df = df.drop(ctrl_label)
df = df[df.index.isin(valid_perts)].copy()
df['mean_HSPA5'] = df.index.map(hspa5_scores)
df['n_cells'] = df.index.map(counts)
df = df.dropna().copy()

print(f"    {len(df)} perturbations in final table")

# =============================================================================
# COMPUTE LOESS-RESIDUAL DISCORDANCE
# =============================================================================

print("\n>>> Computing LOESS-residual discordance (frac={})...".format(LOESS_FRAC))

fitted = lowess(df['Sp'].values, df['Mp'].values, frac=LOESS_FRAC,
                return_sorted=False)
resid = df['Sp'].values - fitted
disc_loess = -resid
disc_loess_z = (disc_loess - disc_loess.mean()) / disc_loess.std()
df['disc_loess'] = disc_loess_z

print(f"    Discordance range: [{df['disc_loess'].min():.3f}, {df['disc_loess'].max():.3f}]")

# =============================================================================
# MAGNITUDE QUARTILE STRATIFICATION
# =============================================================================

print("\n>>> Stratifying by magnitude quartiles...")

df['mag_quartile'] = pd.qcut(df['Mp'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(f"\n{'='*80}")
print("RESULTS: HSPA5 × DISCORDANCE WITHIN MAGNITUDE QUARTILES")
print(f"{'='*80}")

summary_rows = []

print(f"\n{'Quartile':<10s}  {'Mp range':<22s}  {'n':>3s}  "
      f"{'High-disc HSPA5':>15s}  {'Low-disc HSPA5':>14s}  "
      f"{'Delta':>8s}  {'p (MWU)':>10s}  {'Effect':>8s}")
print("-" * 100)

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    bin_df = df[df['mag_quartile'] == q].copy()
    if len(bin_df) < 4:
        print(f"  {q}: skipped (n={len(bin_df)} < 4)")
        continue

    mp_lo, mp_hi = bin_df['Mp'].min(), bin_df['Mp'].max()
    median_disc = bin_df['disc_loess'].median()

    high_disc = bin_df[bin_df['disc_loess'] >= median_disc]
    low_disc = bin_df[bin_df['disc_loess'] < median_disc]

    h_mean = high_disc['mean_HSPA5'].mean()
    l_mean = low_disc['mean_HSPA5'].mean()
    delta = h_mean - l_mean

    if len(high_disc) >= 2 and len(low_disc) >= 2:
        stat, pval = mannwhitneyu(high_disc['mean_HSPA5'],
                                   low_disc['mean_HSPA5'],
                                   alternative='greater')
        pooled_std = df['mean_HSPA5'].std()
        cohen_d = delta / pooled_std if pooled_std > 0 else 0.0
    else:
        pval = np.nan
        cohen_d = np.nan

    sig = ""
    if not np.isnan(pval):
        if pval < 0.05:
            sig = " *"
        if pval < 0.01:
            sig = " **"
        if pval < 0.001:
            sig = " ***"

    p_str = f"{pval:.3e}" if not np.isnan(pval) else "N/A"
    print(f"{q:<10s}  [{mp_lo:>8.3f}, {mp_hi:>8.3f}]  {len(bin_df):>3d}  "
          f"{h_mean:>15.4f}  {l_mean:>14.4f}  {delta:>+8.4f}  "
          f"{p_str:>10s}{sig}  {cohen_d:>+8.3f}")

    summary_rows.append({
        'quartile': q,
        'Mp_low': mp_lo,
        'Mp_high': mp_hi,
        'n_total': len(bin_df),
        'n_high_disc': len(high_disc),
        'n_low_disc': len(low_disc),
        'disc_median': median_disc,
        'HSPA5_high_disc': h_mean,
        'HSPA5_low_disc': l_mean,
        'HSPA5_delta': delta,
        'MWU_p': pval,
        'cohen_d': cohen_d,
    })

# =============================================================================
# OVERALL (UNSTRATIFIED) TEST
# =============================================================================

print(f"\n{'='*80}")
print("OVERALL (unstratified) comparison")
print(f"{'='*80}")

median_disc_all = df['disc_loess'].median()
high_all = df[df['disc_loess'] >= median_disc_all]
low_all = df[df['disc_loess'] < median_disc_all]

stat_all, p_all = mannwhitneyu(high_all['mean_HSPA5'],
                                low_all['mean_HSPA5'],
                                alternative='greater')
delta_all = high_all['mean_HSPA5'].mean() - low_all['mean_HSPA5'].mean()

print(f"  High-discordance (n={len(high_all)}): mean HSPA5 = {high_all['mean_HSPA5'].mean():.4f}")
print(f"  Low-discordance  (n={len(low_all)}):  mean HSPA5 = {low_all['mean_HSPA5'].mean():.4f}")
print(f"  Delta = {delta_all:+.4f},  MWU p = {p_all:.3e}")

rho_disc_hspa5, p_disc_hspa5 = spearmanr(df['disc_loess'], df['mean_HSPA5'])
print(f"\n  Spearman(discordance, mean_HSPA5) = {rho_disc_hspa5:+.3f}, p = {p_disc_hspa5:.3e}")

# =============================================================================
# TOP DISCORDANT PERTURBATIONS WITH HSPA5
# =============================================================================

print(f"\n{'='*80}")
print("TOP 10 HIGH-DISCORDANCE PERTURBATIONS — HSPA5 EXPRESSION")
print(f"{'='*80}")

top_disc = df.nlargest(10, 'disc_loess')
print(f"\n{'Perturbation':<20s}  {'Mp':>8s}  {'Sp':>8s}  {'Disc':>8s}  "
      f"{'HSPA5':>8s}  {'n_cells':>7s}  {'Quartile':>8s}")
print("-" * 75)
for _, row in top_disc.iterrows():
    print(f"{row.name:<20s}  {row['Mp']:>8.3f}  {row['Sp']:>8.3f}  "
          f"{row['disc_loess']:>8.3f}  {row['mean_HSPA5']:>8.4f}  "
          f"{int(row['n_cells']):>7d}  {row['mag_quartile']:>8s}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

df_out = df.copy()
df_out.index.name = 'perturbation'
df_out.to_csv(OUTPUT_DIR / "norman_hspa5_quartile_analysis.csv")
print(f"\nSaved per-perturbation table -> norman_hspa5_quartile_analysis.csv")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "norman_hspa5_quartile_summary.csv", index=False)
print(f"Saved quartile summary       -> norman_hspa5_quartile_summary.csv")

# =============================================================================
# FIGURE
# =============================================================================

print("\n>>> Generating figure...")

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 6))

# --- Panel A: Sp vs Mp colored by HSPA5 ---
sc_plot = ax_a.scatter(df['Mp'], df['Sp'], c=df['mean_HSPA5'],
                       cmap='RdYlBu_r', s=40, alpha=0.8,
                       edgecolor='white', linewidth=0.3)
cbar = plt.colorbar(sc_plot, ax=ax_a, shrink=0.8, pad=0.02)
cbar.set_label('Mean HSPA5', fontsize=10)

fitted_plot = lowess(df['Sp'].values, df['Mp'].values, frac=LOESS_FRAC)
ax_a.plot(fitted_plot[:, 0], fitted_plot[:, 1], '--', color='gray',
          linewidth=2, alpha=0.7, label='LOESS fit')

ax_a.set_xlabel('Effect Magnitude (Mp)', fontsize=11, fontweight='bold')
ax_a.set_ylabel('Shesha Stability (Sp)', fontsize=11, fontweight='bold')
ax_a.set_title('A. Stability vs Magnitude\n(colored by HSPA5)', fontsize=12,
               fontweight='bold')
ax_a.legend(fontsize=9, loc='upper left')
sns.despine(ax=ax_a)

# --- Panel B: HSPA5 by discordance group within each quartile ---
plot_data = []
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    bin_df = df[df['mag_quartile'] == q]
    median_disc = bin_df['disc_loess'].median()
    for _, row in bin_df.iterrows():
        group = 'High disc.' if row['disc_loess'] >= median_disc else 'Low disc.'
        plot_data.append({
            'Quartile': q,
            'Group': group,
            'HSPA5': row['mean_HSPA5'],
        })

plot_df = pd.DataFrame(plot_data)
palette = {'High disc.': '#d62728', 'Low disc.': '#1f77b4'}
sns.boxplot(data=plot_df, x='Quartile', y='HSPA5', hue='Group',
            palette=palette, ax=ax_b, fliersize=3, linewidth=1.2)
ax_b.set_xlabel('Magnitude Quartile', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Mean HSPA5 Expression', fontsize=11, fontweight='bold')
ax_b.set_title('B. HSPA5 by Discordance Group\n(within magnitude quartiles)',
               fontsize=12, fontweight='bold')
ax_b.legend(fontsize=9, loc='upper right', framealpha=0.9)

for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    row = next((r for r in summary_rows if r['quartile'] == q), None)
    if row and not np.isnan(row['MWU_p']):
        p = row['MWU_p']
        if p < 0.001:
            label = '***'
        elif p < 0.01:
            label = '**'
        elif p < 0.05:
            label = '*'
        else:
            label = 'ns'
        ymax = plot_df[plot_df['Quartile'] == q]['HSPA5'].max()
        ax_b.text(i, ymax + 0.02, label, ha='center', fontsize=10,
                  fontweight='bold', color='#333333')

sns.despine(ax=ax_b)

# --- Panel C: Discordance vs HSPA5 scatter ---
quartile_colors = {'Q1': '#4daf4a', 'Q2': '#377eb8', 'Q3': '#ff7f00', 'Q4': '#e41a1c'}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    q_df = df[df['mag_quartile'] == q]
    ax_c.scatter(q_df['disc_loess'], q_df['mean_HSPA5'],
                 c=quartile_colors[q], s=40, alpha=0.7,
                 edgecolor='white', linewidth=0.3, label=q)

ax_c.axhline(ctrl_hspa5, ls=':', color='gray', alpha=0.7, label='Control HSPA5')
ax_c.axvline(0, ls=':', color='gray', alpha=0.4)

ax_c.set_xlabel('LOESS Discordance (z-scored)', fontsize=11, fontweight='bold')
ax_c.set_ylabel('Mean HSPA5 Expression', fontsize=11, fontweight='bold')
ax_c.set_title(f'C. Discordance vs HSPA5\n'
               f'(Spearman ρ = {rho_disc_hspa5:+.3f}, p = {p_disc_hspa5:.2e})',
               fontsize=12, fontweight='bold')
ax_c.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2)
sns.despine(ax=ax_c)

plt.suptitle('Norman 2019 (CRISPRa): HSPA5 × Discordance Stratified by Magnitude',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

out_path = OUTPUT_DIR / "norman_hspa5_quartile_figure"
plt.savefig(str(out_path) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out_path) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved figure -> {out_path}.pdf / .png")
plt.show()

# =============================================================================
# INTERPRETATION
# =============================================================================

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

n_sig = sum(1 for r in summary_rows if not np.isnan(r['MWU_p']) and r['MWU_p'] < 0.05)
n_pos_delta = sum(1 for r in summary_rows if r['HSPA5_delta'] > 0)
n_quartiles = len(summary_rows)

if n_sig > 0:
    print(f"\n  {n_sig}/{n_quartiles} quartiles show SIGNIFICANT (p<0.05) elevation "
          f"of HSPA5 in high-discordance perturbations.")
    print("  -> Directionally consistent with the reviewer's hypothesis.")
elif n_pos_delta > n_quartiles // 2:
    print(f"\n  {n_pos_delta}/{n_quartiles} quartiles show POSITIVE delta (high-disc > low-disc)")
    print("  but none reach p<0.05.")
    print("  -> Directionally consistent but underpowered.")
else:
    print(f"\n  {n_pos_delta}/{n_quartiles} quartiles show positive delta.")
    print("  -> No evidence of systematic HSPA5 elevation in high-discordance CRISPRa.")
    print("  This confirms that the CRISPRa attenuation is not driven by UPR stress.")

print(f"\n  Overall Spearman(discordance, HSPA5) = {rho_disc_hspa5:+.3f}, p = {p_disc_hspa5:.3e}")

if abs(rho_disc_hspa5) < 0.1:
    print("  -> Negligible correlation: HSPA5 does not track discordance.")
elif abs(rho_disc_hspa5) < 0.3:
    print("  -> Weak correlation: minor signal, likely underpowered.")
else:
    print("  -> Notable correlation worth further investigation.")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\nOutput files in {OUTPUT_DIR}:")
print("  - norman_hspa5_quartile_analysis.csv  (per-perturbation)")
print("  - norman_hspa5_quartile_summary.csv   (quartile summary)")
print("  - norman_hspa5_quartile_figure.pdf/png")
