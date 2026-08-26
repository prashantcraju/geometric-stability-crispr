#!/usr/bin/env python3
"""
Split-Half Reproducibility & Cross-Dataset Concordance Tests
(same implementation as song_ps_replication.py)

CORE QUESTION: Does Sp predict which screen hits will replicate?

This version mirrors song_ps_replication.py exactly:
  * Sp / Mp are computed with compute_stability_magnitude (geometric port).
  * The Perturbation-response Score (PS) is the REAL Song et al. PS,
    computed via the official scMAGeCK R package when Rscript is
    available, otherwise via the faithful pure-Python port
    (compute_real_ps_python).  Two distance proxies (Euclidean,
    Mahalanobis) are also reported, giving the same three-tier PS
    structure used throughout song_ps_replication.py.
  * Datasets, cleaning, preprocessing and bootstrap CIs are all reused
    directly from song_ps_replication.py.

TEST 1 - SPLIT-HALF REPRODUCIBILITY (primary: Replogle):
  For each perturbation with enough cells, randomly split cells 50/50,
  compute shift_A and shift_B independently (relative to the same control
  centroid), then measure cosine(shift_A, shift_B), averaged over N_SPLITS
  random splits.  This is a direct measure of effect-direction
  reproducibility.

  Prediction: perturbations with higher Sp should show higher split-half
  cosine similarity.

  Controls for magnitude confounds:
    * Partial correlation: split-half repro vs Sp | magnitude
    * Partial correlation: split-half repro vs PS (each tier) | magnitude
    * AUC / precision@k for predicting "reproducible hits"
    * Magnitude-matched binned analysis

TEST 2 - CROSS-DATASET REPRODUCIBILITY (Norman intersect Replogle):
  Both are K562 cells.  Compute Sp / PS independently in each dataset and,
  for shared single-gene perturbations, correlate the per-dataset values.

OUTPUT:
  split_half_reproducibility_<dataset>.csv
  split_half_reproducibility_<dataset>.pdf/.png
  cross_dataset_reproducibility.csv
  cross_dataset_reproducibility.pdf/.png

USAGE:
  python split_half_reproducibility.py [--datasets replogle,norman]
                                       [--out_dir ./shesha-crispr]
                                       [--r_executable Rscript]
                                       [--max_perts_per_batch 100]
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# Reuse the EXACT song_ps_replication.py implementation (data loading,
# bootstrap helpers, Sp/Mp, proxy PS, and the real Song et al. PS via the
# official scMAGeCK R package / pure-Python port).
from song_ps_replication import (
    SEED,
    N_BOOTSTRAP,
    CI_LEVEL,
    DATASET_CONFIGS,
    bootstrap_spearman_ci,
    bootstrap_partial_corr_ci,
    compute_stability_magnitude,
    compute_proxy_ps,
    compute_real_ps_python,
    export_for_r,
    run_scmageck_r,
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

N_SPLITS = 50
MIN_CELLS_SPLIT_HALF = 30  # need >= 15 cells per half


# ============================================================================
# SPLIT-HALF COSINE
# ============================================================================

def compute_split_half_cosine(X_pert, ctrl_centroid, n_splits=N_SPLITS,
                              seed=SEED):
    """
    Split perturbation cells 50/50 repeatedly and return the mean cosine
    similarity between the two half-shift vectors (relative to the same
    control centroid).
    """
    n_cells = X_pert.shape[0]
    if n_cells < MIN_CELLS_SPLIT_HALF:
        return np.nan

    rng = np.random.default_rng(seed=seed)
    cosines = np.zeros(n_splits)

    for i in range(n_splits):
        perm = rng.permutation(n_cells)
        half = n_cells // 2
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]

        mean_shift_a = (X_pert[idx_a] - ctrl_centroid).mean(axis=0)
        mean_shift_b = (X_pert[idx_b] - ctrl_centroid).mean(axis=0)

        norm_a = np.linalg.norm(mean_shift_a)
        norm_b = np.linalg.norm(mean_shift_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            cosines[i] = 0.0
        else:
            cosines[i] = np.dot(mean_shift_a, mean_shift_b) / (norm_a * norm_b)

    return float(np.mean(cosines))


# ============================================================================
# DATA PROCESSING PIPELINE (identical to song_ps_replication.load_and_process,
# extended with the per-perturbation split-half cosine)
# ============================================================================

def load_and_process(cfg, r_executable='Rscript', max_perts_per_batch=100,
                     skip_r=False):
    """
    Load + preprocess a dataset and compute, per perturbation:
      stability (Sp), magnitude (Mp), PS_euclid, PS_mahal, PS_real,
      split_half_cosine, n_cells.

    Sp/Mp, proxies and real PS use the same functions as
    song_ps_replication.py.
    """
    name = cfg['name']
    print()
    print("=" * 72)
    print(f"DATASET: {name}")
    print("=" * 72)

    adata = cfg['loader']()
    if cfg['clean_func']:
        adata = cfg['clean_func'](adata)

    pert_col = cfg['pert_col']
    ctrl_label = cfg['ctrl_label']
    min_cells = cfg['min_cells']
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    counts = adata.obs[pert_col].value_counts()
    valid = [v for v in counts[counts >= min_cells].index if v != ctrl_label]
    print(f"  {len(valid)} perturbations with >= {min_cells} cells")

    adata_sub = adata[adata.obs[pert_col].isin(valid + [ctrl_label])].copy()

    # Log-normalised data (for PS) ------------------------------------------
    adata_norm = adata_sub.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    # PCA for Sp/Mp + proxy PS + split-half ---------------------------------
    adata_pca = adata_norm.copy()
    sc.pp.highly_variable_genes(adata_pca, n_top_genes=2000, subset=True)
    sc.tl.pca(adata_pca, n_comps=50, random_state=SEED)

    X_pca = adata_pca.obsm['X_pca']
    pca_labels = adata_pca.obs[pert_col].values.astype(str)

    df = compute_stability_magnitude(X_pca, pca_labels, ctrl_label, valid)
    print(f"  Sp/Mp computed for {len(df)} perturbations")

    euclid_ps, mahal_ps = compute_proxy_ps(X_pca, pca_labels, ctrl_label,
                                           list(df.index))
    df['PS_euclid'] = df.index.map(euclid_ps)
    df['PS_mahal'] = df.index.map(mahal_ps)

    # Real PS (Tier 3) ------------------------------------------------------
    if skip_r:
        print("  Computing Song et al. PS via pure Python port "
              "(R not available)...")
        try:
            real_ps = compute_real_ps_python(
                adata_norm, pert_col, ctrl_label, list(df.index),
                logfc_threshold=0.1, target_gene_max=500)
            df['PS_real'] = df.index.map(real_ps)
        except Exception as e:
            print(f"  ERROR computing Python PS: {e}")
            df['PS_real'] = np.nan
    else:
        print("  Computing Song et al. PS via official scMAGeCK R package...")
        tmpdir = tempfile.mkdtemp(prefix='scmageck_ps_')
        try:
            export_for_r(adata_sub, pert_col, ctrl_label, valid, tmpdir)
            real_ps = run_scmageck_r(
                tmpdir, ctrl_label, list(df.index),
                r_executable=r_executable, max_per_batch=max_perts_per_batch)
            df['PS_real'] = df.index.map(real_ps)
        except Exception as e:
            print(f"  ERROR computing real PS: {e}")
            df['PS_real'] = np.nan
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Split-half cosine -----------------------------------------------------
    ctrl_centroid = X_pca[pca_labels == ctrl_label].mean(axis=0)
    sh = {}
    for pert in df.index:
        X_pert = X_pca[pca_labels == pert]
        sh[pert] = compute_split_half_cosine(
            X_pert, ctrl_centroid, n_splits=N_SPLITS,
            seed=SEED + hash(pert) % 100000)
    df['split_half_cosine'] = df.index.map(sh)

    n_real = df['PS_real'].notna().sum()
    print(f"  Final: {len(df)} perturbations "
          f"({n_real} with real PS, all with Euclidean + Mahalanobis proxy "
          f"and split-half cosine)")

    return df


# ============================================================================
# TEST 1: SPLIT-HALF REPRODUCIBILITY
# ============================================================================

PS_TIERS = [
    ('Sp', 'stability'),
    ('PS_real', 'PS_real'),
    ('PS_euclid', 'PS_euclid'),
    ('PS_mahal', 'PS_mahal'),
    ('Magnitude', 'magnitude'),
]


def run_split_half_test(df, out_dir, ds_label='Replogle'):
    """
    Does Sp predict split-half reproducibility after controlling for
    magnitude, and how does it compare to the real PS (and proxies)?
    """
    print()
    print("=" * 80)
    print(f"TEST 1: SPLIT-HALF REPRODUCIBILITY ({ds_label})")
    print("=" * 80)

    df = df.dropna(subset=['split_half_cosine', 'stability', 'magnitude']).copy()
    print(f"  {len(df)} perturbations with valid split-half estimates")

    seed_ctr = SEED + 1000

    # --- Raw correlations ---
    print()
    print("  Raw Spearman correlations vs split-half cosine:")
    for name, col in PS_TIERS:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if len(sub) < 10:
            print(f"    {name:12s}: only {len(sub)} non-NaN, skipped")
            continue
        ci = bootstrap_spearman_ci(
            sub[col].values, sub['split_half_cosine'].values, seed=seed_ctr)
        seed_ctr += 1
        print(f"    {name:12s}: rho = {ci['rho']:+.4f} "
              f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  p = {ci['p']:.2e}")

    # --- Partial correlations controlling for magnitude ---
    print()
    print("  Partial correlations (controlling for magnitude):")
    partial_store = {}
    for name, col in [('Sp', 'stability'), ('PS_real', 'PS_real'),
                      ('PS_euclid', 'PS_euclid'), ('PS_mahal', 'PS_mahal')]:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if len(sub) < 10:
            print(f"    {name:12s}: only {len(sub)} non-NaN, skipped")
            continue
        pc = bootstrap_partial_corr_ci(
            sub[col].values, sub['split_half_cosine'].values,
            sub['magnitude'].values, seed=seed_ctr)
        seed_ctr += 1
        partial_store[name] = pc['rho_partial']
        print(f"    {name:12s}| magnitude: rho = {pc['rho_partial']:+.4f} "
              f"[{pc['ci_low']:.4f}, {pc['ci_high']:.4f}]  p = {pc['p']:.2e}")

    if 'Sp' in partial_store:
        print()
        for ps_name in ['PS_real', 'PS_euclid', 'PS_mahal']:
            if ps_name in partial_store:
                delta = partial_store['Sp'] - partial_store[ps_name]
                print(f"    Sp vs {ps_name} partial-rho difference: {delta:+.4f}")

    # --- AUC / precision@k for reproducible hits ---
    print()
    print("  BENCHMARK: predicting 'reproducible hits' (top quartile cos):")
    threshold = df['split_half_cosine'].quantile(0.75)
    df['reproducible_hit'] = (df['split_half_cosine'] >= threshold).astype(int)
    n_hits = int(df['reproducible_hit'].sum())
    print(f"    Threshold (75th pct): {threshold:.4f}  |  "
          f"hits: {n_hits}/{len(df)} ({100 * n_hits / len(df):.1f}%)")

    for name, col in PS_TIERS:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if len(sub) < 10 or sub['reproducible_hit'].nunique() < 2:
            continue
        labels_bin = sub['reproducible_hit'].values
        scores = sub[col].values
        auc_val = roc_auc_score(labels_bin, scores)
        k = int(labels_bin.sum())
        sorted_idx = np.argsort(-scores)
        prec_at_k = labels_bin[sorted_idx[:k]].mean() if k > 0 else np.nan
        print(f"    {name:12s}: AUC = {auc_val:.4f}, "
              f"Precision@{k} = {prec_at_k:.4f}")

    # Magnitude-residualized AUC for Sp and PS_real
    base = df.dropna(subset=['stability', 'magnitude'])
    if len(base) >= 10 and base['reproducible_hit'].nunique() >= 2:
        Z = sm.add_constant(base['magnitude'].values)
        sp_resid = sm.OLS(base['stability'].values, Z).fit().resid
        auc_sp = roc_auc_score(base['reproducible_hit'].values, sp_resid)
        print(f"    {'Sp|mag':12s}: AUC = {auc_sp:.4f} "
              f"(magnitude-residualized)")
    if 'PS_real' in df.columns:
        base_r = df.dropna(subset=['PS_real', 'magnitude'])
        if len(base_r) >= 10 and base_r['reproducible_hit'].nunique() >= 2:
            Zr = sm.add_constant(base_r['magnitude'].values)
            ps_resid = sm.OLS(base_r['PS_real'].values, Zr).fit().resid
            auc_ps = roc_auc_score(base_r['reproducible_hit'].values, ps_resid)
            print(f"    {'PS_real|mag':12s}: AUC = {auc_ps:.4f} "
                  f"(magnitude-residualized)")

    # --- Magnitude-matched binned analysis (high-Sp vs low-Sp) ---
    print()
    print("  MAGNITUDE-MATCHED BINNED ANALYSIS (high-Sp vs low-Sp):")
    bin_results = []
    if len(df) >= 16:
        df['mag_quartile'] = pd.qcut(df['magnitude'], q=4,
                                     labels=['Q1', 'Q2', 'Q3', 'Q4'])
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            subset = df[df['mag_quartile'] == q].copy()
            if len(subset) < 6:
                continue
            sp_median = subset['stability'].median()
            high_sp = subset[subset['stability'] >= sp_median]
            low_sp = subset[subset['stability'] < sp_median]
            mean_high = high_sp['split_half_cosine'].mean()
            mean_low = low_sp['split_half_cosine'].mean()
            diff = mean_high - mean_low
            rho_within, p_within = spearmanr(subset['stability'],
                                             subset['split_half_cosine'])
            mag_range = (f"[{subset['magnitude'].min():.2f}, "
                         f"{subset['magnitude'].max():.2f}]")
            bin_results.append({
                'mag_quartile': q, 'n': len(subset), 'mag_range': mag_range,
                'high_Sp_mean_cos': mean_high, 'low_Sp_mean_cos': mean_low,
                'difference': diff, 'within_bin_rho': rho_within,
                'within_bin_p': p_within,
            })
            print(f"    {q} (n={len(subset)}, mag {mag_range}): "
                  f"high-Sp={mean_high:.4f}, low-Sp={mean_low:.4f}, "
                  f"d={diff:+.4f}, rho={rho_within:+.3f} p={p_within:.2e}")

        diffs = [r['difference'] for r in bin_results]
        if diffs:
            n_pos = sum(1 for d in diffs if d > 0)
            print(f"    high-Sp mean cos higher in {n_pos}/{len(diffs)} bins")

    # --- Save ---
    csv_path = out_dir / f"split_half_reproducibility_{ds_label}.csv"
    df.to_csv(csv_path)
    print()
    print(f"  Saved -> {csv_path.name} ({len(df)} rows)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(df['stability'], df['split_half_cosine'], alpha=0.3, s=15,
               c='steelblue')
    rho0 = spearmanr(df['stability'], df['split_half_cosine'])[0]
    ax.set_xlabel('Sp (Stability)')
    ax.set_ylabel('Split-Half Cosine Similarity')
    ax.set_title(f'Sp vs Reproducibility\nrho={rho0:.3f}')

    ax = axes[1]
    ps_plot_col = 'PS_real' if df['PS_real'].notna().sum() >= 10 \
        else 'PS_euclid'
    sub_ps = df.dropna(subset=[ps_plot_col])
    ax.scatter(sub_ps[ps_plot_col], sub_ps['split_half_cosine'], alpha=0.3,
               s=15, c='darkorange')
    rho1 = spearmanr(sub_ps[ps_plot_col], sub_ps['split_half_cosine'])[0]
    ax.set_xlabel(ps_plot_col)
    ax.set_ylabel('Split-Half Cosine Similarity')
    ax.set_title(f'{ps_plot_col} vs Reproducibility\nrho={rho1:.3f}')

    ax = axes[2]
    if bin_results:
        bar_data = pd.DataFrame(bin_results)
        x_pos = np.arange(len(bar_data))
        width = 0.35
        ax.bar(x_pos - width / 2, bar_data['high_Sp_mean_cos'], width,
               label='High Sp', color='steelblue', alpha=0.8)
        ax.bar(x_pos + width / 2, bar_data['low_Sp_mean_cos'], width,
               label='Low Sp', color='lightcoral', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_data['mag_quartile'])
        ax.set_xlabel('Magnitude Quartile')
        ax.set_ylabel('Mean Split-Half Cosine')
        ax.set_title('Magnitude-Matched Comparison')
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"split_half_reproducibility_{ds_label}.pdf",
                bbox_inches='tight')
    plt.savefig(out_dir / f"split_half_reproducibility_{ds_label}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> split_half_reproducibility_{ds_label}.pdf/.png")

    return df


# ============================================================================
# TEST 2: CROSS-DATASET REPRODUCIBILITY (Norman intersect Replogle)
# ============================================================================

CROSS_METRICS = [
    ('Sp', 'stability'),
    ('PS_real', 'PS_real'),
    ('PS_euclid', 'PS_euclid'),
    ('PS_mahal', 'PS_mahal'),
    ('Magnitude', 'magnitude'),
]


def run_cross_dataset_test(df_norman, df_replogle, out_dir):
    """
    For single-gene perturbations shared between Norman (CRISPRa) and
    Replogle (CRISPRi), correlate Sp and PS computed independently in each
    dataset.
    """
    print()
    print("=" * 80)
    print("TEST 2: CROSS-DATASET REPRODUCIBILITY (Norman intersect Replogle)")
    print("=" * 80)
    print("  Both datasets: K562 cells (Norman CRISPRa, Replogle CRISPRi)")

    norman_single = set(g for g in df_norman.index if '+' not in g)
    replogle_genes = set(df_replogle.index)
    shared = sorted(norman_single & replogle_genes)
    print(f"  Norman single-gene perturbations: {len(norman_single)}")
    print(f"  Replogle perturbations: {len(replogle_genes)}")
    print(f"  Shared genes: {len(shared)}")

    if len(shared) < 10:
        print("  WARNING: too few shared genes for meaningful analysis")
        return None

    metric_cols = ['stability', 'magnitude', 'PS_real', 'PS_euclid',
                   'PS_mahal']
    rows = []
    for gene in shared:
        row = {'gene': gene}
        for c in metric_cols:
            row[f'{c}_norman'] = df_norman.loc[gene, c] \
                if c in df_norman.columns else np.nan
            row[f'{c}_replogle'] = df_replogle.loc[gene, c] \
                if c in df_replogle.columns else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)

    print(f"  Genes assembled: {len(df)}")

    # --- Cross-dataset correlations ---
    print()
    print(f"  CROSS-DATASET CORRELATIONS (Norman vs Replogle):")
    seed_ctr = SEED + 5000
    for name, col in CROSS_METRICS:
        cn, cr = f'{col}_norman', f'{col}_replogle'
        sub = df.dropna(subset=[cn, cr])
        if len(sub) < 10:
            print(f"    {name:12s}: only {len(sub)} shared non-NaN, skipped")
            continue
        ci = bootstrap_spearman_ci(sub[cn].values, sub[cr].values,
                                   seed=seed_ctr)
        seed_ctr += 1
        print(f"    {name:12s}: rho = {ci['rho']:+.4f} "
              f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  p = {ci['p']:.2e} "
              f"(n={len(sub)})")

    # --- Partial correlations controlling for magnitude in both ---
    print()
    print("  Partial correlations (controlling for magnitude in both):")
    partial_store = {}
    for name, col in [('Sp', 'stability'), ('PS_real', 'PS_real'),
                      ('PS_euclid', 'PS_euclid'), ('PS_mahal', 'PS_mahal')]:
        cn, cr = f'{col}_norman', f'{col}_replogle'
        sub = df.dropna(subset=[cn, cr, 'magnitude_norman',
                                'magnitude_replogle'])
        if len(sub) < 10:
            continue
        z_mag = np.column_stack([sub['magnitude_norman'].values,
                                 sub['magnitude_replogle'].values])
        pc = bootstrap_partial_corr_ci(sub[cn].values, sub[cr].values, z_mag,
                                       seed=seed_ctr)
        seed_ctr += 1
        partial_store[name] = pc['rho_partial']
        print(f"    {name:12s}| magnitude: rho = {pc['rho_partial']:+.4f} "
              f"[{pc['ci_low']:.4f}, {pc['ci_high']:.4f}]  p = {pc['p']:.2e}")

    if 'Sp' in partial_store:
        print()
        for ps_name in ['PS_real', 'PS_euclid', 'PS_mahal']:
            if ps_name in partial_store:
                delta = partial_store['Sp'] - partial_store[ps_name]
                print(f"    Sp vs {ps_name} cross-dataset partial-rho difference: {delta:+.4f}")

    # --- Save ---
    csv_path = out_dir / "cross_dataset_reproducibility.csv"
    df.to_csv(csv_path, index=False)
    print()
    print(f"  Saved -> {csv_path.name} ({len(df)} rows)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_specs = [
        ('stability', 'Sp', 'steelblue'),
        ('PS_real' if df['PS_real_norman'].notna().sum() >= 10
         else 'PS_euclid', 'PS', 'darkorange'),
        ('magnitude', 'Magnitude', 'seagreen'),
    ]
    for ax, (col, label, color) in zip(axes, panel_specs):
        cn, cr = f'{col}_norman', f'{col}_replogle'
        sub = df.dropna(subset=[cn, cr])
        ax.scatter(sub[cn], sub[cr], alpha=0.6, s=30, c=color)
        rho = spearmanr(sub[cn], sub[cr])[0] if len(sub) >= 3 else np.nan
        ax.set_xlabel(f'{label} (Norman, CRISPRa)')
        ax.set_ylabel(f'{label} (Replogle, CRISPRi)')
        ax.set_title(f'{label} Cross-Dataset\nrho={rho:.3f}')

    plt.tight_layout()
    plt.savefig(out_dir / "cross_dataset_reproducibility.pdf",
                bbox_inches='tight')
    plt.savefig(out_dir / "cross_dataset_reproducibility.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved -> cross_dataset_reproducibility.pdf/.png")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split-half reproducibility & cross-dataset concordance "
                    "(same implementation as song_ps_replication.py)")
    parser.add_argument('--datasets', type=str, default='replogle,norman',
                        help='Comma-separated dataset keys')
    import pipeline_config as cfg
    parser.add_argument('--out_dir', type=str, default=str(cfg.OUTPUT_DIR),
                        help='Output directory')
    parser.add_argument('--r_executable', type=str, default='Rscript',
                        help='Path to Rscript executable')
    parser.add_argument('--max_perts_per_batch', type=int, default=100,
                        help='Max perturbations per R batch call')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SPLIT-HALF REPRODUCIBILITY & CROSS-DATASET CONCORDANCE TESTS")
    print("=" * 80)
    print(f"  N_SPLITS = {N_SPLITS}")
    print(f"  N_BOOTSTRAP = {N_BOOTSTRAP}")
    print(f"  SEED = {SEED}")
    print(f"  Output: {out_dir}")
    print(f"  R executable: {args.r_executable}")

    # Verify R availability (Tier 3 PS)
    r_available = False
    try:
        r_check = subprocess.run([args.r_executable, '--version'],
                                 capture_output=True, text=True, timeout=30)
        if r_check.returncode == 0:
            print(f"  R version: {r_check.stdout.split(chr(10))[0]}")
            r_available = True
    except FileNotFoundError:
        print(f"  NOTE: '{args.r_executable}' not found; "
              f"Tier 3 PS uses the pure-Python scMAGeCK port.")

    if r_available:
        print("  Tier 3: official scMAGeCK R package")
    else:
        print("  Tier 3: pure Python port of scMAGeCK algorithm")

    dataset_keys = [k.strip() for k in args.datasets.split(',')]
    print(f"  Datasets: {dataset_keys}")

    # --- Load + process all requested datasets ---
    datasets = {}
    for key in dataset_keys:
        if key not in DATASET_CONFIGS:
            print(f"\nWARNING: unknown dataset key '{key}', skipping")
            continue
        try:
            df = load_and_process(
                DATASET_CONFIGS[key],
                r_executable=args.r_executable,
                max_perts_per_batch=args.max_perts_per_batch,
                skip_r=not r_available)
            datasets[key] = df
        except Exception as e:
            print(f"\nERROR loading {key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not datasets:
        print("\nFATAL: no datasets loaded successfully")
        sys.exit(1)

    # --- TEST 1: split-half (run on every loaded dataset) ---
    split_results = {}
    for key, df in datasets.items():
        split_results[key] = run_split_half_test(
            df, out_dir, ds_label=key)

    # --- TEST 2: cross-dataset (Norman intersect Replogle) ---
    df_cross = None
    if 'norman' in datasets and 'replogle' in datasets:
        df_cross = run_cross_dataset_test(
            datasets['norman'], datasets['replogle'], out_dir)
    else:
        print()
        print("TEST 2 skipped: requires both 'norman' and 'replogle'.")

    # --- Final summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for key, df in split_results.items():
        sub = df.dropna(subset=['split_half_cosine', 'stability'])
        rho_sp = spearmanr(sub['stability'], sub['split_half_cosine'])[0]
        line = f"  [{key}] Sp-repro rho = {rho_sp:+.4f}"
        if df['PS_real'].notna().sum() >= 10:
            subr = df.dropna(subset=['split_half_cosine', 'PS_real'])
            rho_ps = spearmanr(subr['PS_real'], subr['split_half_cosine'])[0]
            line += f", PS_real-repro rho = {rho_ps:+.4f}"
        print(line)

    if df_cross is not None:
        sub = df_cross.dropna(subset=['stability_norman', 'stability_replogle'])
        if len(sub) >= 3:
            rho = spearmanr(sub['stability_norman'],
                            sub['stability_replogle'])[0]
            print(f"  [cross] Sp Norman vs Replogle rho = {rho:+.4f} "
                  f"(n={len(sub)})")

    print()
    print(f"Output files in {out_dir}:")
    print("  - split_half_reproducibility_<dataset>.csv / .pdf / .png")
    print("  - cross_dataset_reproducibility.csv / .pdf / .png")
    print()
    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
