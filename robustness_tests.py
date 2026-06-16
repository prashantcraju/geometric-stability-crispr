#!/usr/bin/env python3
"""
Methodological Robustness Tests

Two targeted analyses addressing specific reviewer objections:

TEST 1 — NONLINEAR DISCORDANCE RESIDUALS (Improvement 4):
  The magnitude–stability relationship is visibly curved at low magnitudes.
  Linear OLS residuals systematically bias discordance rankings.  Recompute
  discordance using:
    (a) Original linear:  Dp = magnitude_z − stability_z
    (b) Rank-based:       Dp = rank(Mp) − rank(Sp), z-scored
    (c) LOESS residual:   residual from lowess(Sp, Mp, frac=0.3), sign-flipped
  Compare top-discordant rankings across all three methods.  Report whether
  GATA1, CHMP3, AQR retain their positions (Table 7 robustness).

TEST 2 — EMPIRICAL COMPARISON TO SONG et al. PS (Improvement 5):
  Compute perturbation-response score (PS = mean per-cell Euclidean distance
  from control centroid in PCA space) on Norman and Replogle.  Show the joint
  distribution of Sp vs PS controlling for magnitude.  Identify specific
  perturbations where Sp provides non-redundant information.

INPUT:  Raw datasets via pertpy
OUTPUT: nonlinear_discordance_comparison.csv   (Test 1)
        song_ps_comparison.csv                 (Test 2)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt

from anndata import AnnData
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from shesha.bio import compute_stability, compute_magnitude
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 320
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95

OUTPUT_DIR = Path("./shesha-crispr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_MAP = {
    'Norman 2019 (CRISPRa)':   'CRISPRa',
    'Adamson 2016 (CRISPRi)':  'CRISPRi',
    'Dixit 2016 (CRISPRi)':    'CRISPRi',
    'Papalexi 2021 (CRISPR)':  'CRISPR',
    'Replogle 2022 (CRISPRi)': 'CRISPRi',
}

REPLOGLE_MIN_CELLS = 50
NORMAN_MIN_CELLS = 50
DIXIT_MIN_CELLS = 10
ADAMSON_MIN_CELLS = 10
PAPALEXI_MIN_CELLS = 10

# MSigDB Hallmark UPR gene set (for incremental variance test in Test 2)
HALLMARK_UPR = [
    'HSPA5', 'HSP90B1', 'HYOU1', 'CALR', 'CANX', 'P4HB', 'PDIA3',
    'PDIA4', 'PDIA5', 'PDIA6', 'PPIB', 'ERP29', 'ERP44', 'SIL1',
    'FKBP14', 'DNAJB9', 'DNAJB11', 'DNAJC3', 'DNAJC10',
    'ATF6', 'ATF6B', 'ERN1', 'EIF2AK3', 'XBP1', 'DDIT3', 'CREB3L2',
    'EDEM1', 'DERL1', 'OS9', 'SEL1L', 'SYVN1', 'UBE2J1', 'UBE2D1',
    'VIMP', 'YOD1', 'VCP',
    'SEC61A1', 'SEC61B', 'SEC11C', 'SEC24D', 'TRAM1', 'SRPRB',
    'SPCS1', 'SPCS2', 'SPCS3', 'SSR1', 'SSR3', 'SSR4',
    'LMAN1', 'GOSR2', 'KDELR3', 'SURF4',
    'DDOST', 'STT3A', 'STT3B', 'RPN1', 'RPN2', 'MOGS', 'UGGT1',
    'SRD5A3',
    'HERPUD1', 'MANF', 'CRELD2', 'SDF2L1', 'NUCB1', 'RCN1',
    'SERP1', 'WIPI1', 'UFM1', 'BAX', 'ERO1A', 'MBTPS1', 'MBTPS2',
    'ARCN1', 'PREB', 'GANAB', 'TMX1', 'ERLEC1',
]

MIN_GENE_OVERLAP = 10


# =============================================================================
# BOOTSTRAP HELPERS
# =============================================================================

def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=N_BOOTSTRAP,
                                     ci_level=CI_LEVEL, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = len(x)

    def _partial(x, y, z):
        Z_aug = sm.add_constant(z)
        x_resid = sm.OLS(x, Z_aug).fit().resid
        y_resid = sm.OLS(y, Z_aug).fit().resid
        return spearmanr(x_resid, y_resid)

    rho_partial, p = _partial(x, y, z)

    rng = np.random.default_rng(seed=seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = _partial(x[idx], y[idx], z[idx])
        boot[i] = r

    valid = boot[~np.isnan(boot)]
    if len(valid) < 10:
        return {'rho_partial': rho_partial, 'ci_low': np.nan,
                'ci_high': np.nan, 'p': p}

    alpha = 1 - ci_level
    return {
        'rho_partial': rho_partial,
        'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
        'p': p,
    }


def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL,
                           seed=42):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return {'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p': np.nan}
    rng = np.random.default_rng(seed=seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    if len(valid) < 10:
        return {'rho': rho, 'ci_low': np.nan, 'ci_high': np.nan, 'p': p}
    alpha = 1 - ci_level
    return {
        'rho': rho, 'p': p,
        'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def clean_replogle(adata):
    """Label-clean Replogle 2022: merge non-targeting / chr -> control."""
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    def _label(x):
        if 'non-targeting' in x or x.startswith('chr'):
            return 'control'
        if 'pos_control' in x:
            return 'POS_CONTROL'
        return x.split('_')[0]
    adata.obs['condition'] = adata.obs['perturbation'].apply(_label)
    return adata[
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    ].copy()


def clean_adamson(adata):
    """Label-clean Adamson 2016: consolidate control labels."""
    src_col = None
    for candidate in ['perturbation_name', 'perturbation', 'gene', 'target',
                       'guide_id', 'condition']:
        if candidate in adata.obs.columns:
            src_col = candidate
            break
    if src_col is None:
        src_col = next((c for c in adata.obs.columns
                        if 'pert' in c.lower() or 'gene' in c.lower()), None)
    if src_col is None:
        raise ValueError(f"Adamson: no perturbation column found in "
                         f"{list(adata.obs.columns)}")

    adata.obs[src_col] = adata.obs[src_col].astype(str)
    ctrl_keywords = ['gal4', 'gfp', 'neg', 'scramble', 'unperturbed', 'nan']
    def _label(x):
        xl = x.lower().strip()
        for kw in ctrl_keywords:
            if kw in xl:
                return 'control'
        return x
    adata.obs['condition'] = adata.obs[src_col].apply(_label)
    print(f"    Adamson: source column = '{src_col}', mapped to 'condition'")
    return adata[adata.obs['condition'] != 'nan'].copy()


def load_papalexi_rna():
    """Load Papalexi 2021 MuData, extract RNA modality + gene_target annotation."""
    raw = pt.dt.papalexi_2021()
    if type(raw).__name__ != 'MuData':
        raise TypeError(f"Expected MuData for Papalexi, got {type(raw)}")
    if 'rna' not in raw.mod:
        raise KeyError("No 'rna' modality in Papalexi MuData")
    adata = raw.mod['rna'].copy()
    if 'gene_target' in raw.obs.columns:
        adata.obs['gene_target'] = raw.obs['gene_target'].values
        n_ctrl = (adata.obs['gene_target'] == 'NT').sum()
        print(f"    Papalexi: synced gene_target from MuData.obs, "
              f"{n_ctrl} NT control cells")
    else:
        raise KeyError("'gene_target' not found in Papalexi MuData.obs")
    return adata


def load_and_process(name, loader_func, pert_col, ctrl_label,
                     clean_func=None, min_cells=50):
    """
    Load dataset -> normalise -> optionally compute UPR pathway score +
    Shesha stability / magnitude.

    Returns (df, adata_pca) where df has one row per perturbation with
    stability, magnitude (and pw_UPR if gene overlap is sufficient).
    adata_pca is the PCA-space AnnData (X = PC coordinates) needed for PS.
    """
    print(f"\n>>> Loading {name} ...")
    adata = loader_func()
    if clean_func:
        adata = clean_func(adata)
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_norm = adata

    counts = adata_norm.obs[pert_col].value_counts()
    valid = [v for v in counts[counts >= min_cells].index if v != ctrl_label]
    print(f"    {len(valid)} perturbations with >= {min_cells} cells")

    # UPR pathway score (for incremental variance test in Test 2)
    upr_scores = {}
    overlap = [g for g in HALLMARK_UPR if g in adata_norm.var_names]
    pct = 100 * len(overlap) / len(HALLMARK_UPR)
    print(f"    UPR: {len(overlap)}/{len(HALLMARK_UPR)} genes ({pct:.0f}% overlap)")
    if len(overlap) >= MIN_GENE_OVERLAP:
        sc.tl.score_genes(adata_norm, gene_list=overlap,
                          score_name='score_UPR',
                          ctrl_size=50, random_state=SEED)
        for pert in valid:
            mask = adata_norm.obs[pert_col] == pert
            upr_scores[pert] = float(adata_norm[mask].obs['score_UPR'].mean())

    # PCA on HVGs for Shesha metrics
    adata_proc = adata_norm[
        adata_norm.obs[pert_col].isin(valid + [ctrl_label])
    ].copy()
    sc.pp.highly_variable_genes(adata_proc, n_top_genes=2000, subset=True)
    sc.tl.pca(adata_proc, n_comps=50)

    adata_pca = AnnData(X=adata_proc.obsm['X_pca'], obs=adata_proc.obs)
    stab = compute_stability(adata_pca, perturbation_key=pert_col,
                             control_label=ctrl_label, metric='cosine')
    mag = compute_magnitude(adata_pca, perturbation_key=pert_col,
                            control_label=ctrl_label, metric='euclidean')

    df = pd.DataFrame({'stability': pd.Series(stab),
                       'magnitude': pd.Series(mag)})
    if ctrl_label in df.index:
        df = df.drop(ctrl_label)
    df = df[df.index.isin(valid)].copy()

    if upr_scores:
        df['pw_UPR'] = df.index.map(upr_scores)

    print(f"    -> {len(df)} perturbations in final table")
    return df, adata_pca


# =============================================================================
# TEST 1: NONLINEAR DISCORDANCE RESIDUALS (Improvement 4)
#
# The magnitude–stability relationship is visibly curved at low magnitudes.
# Linear OLS residuals systematically bias discordance rankings.
# Recompute discordance with:
#   (a) Original linear:  magnitude_z - stability_z
#   (b) Rank-based:       Dp = rank(Mp) − rank(Sp), z-scored
#   (c) LOESS:            residual from local polynomial fit (frac=0.3)
# Compare top-discordant rankings across all three methods.
# =============================================================================

def _compute_discordance_linear(mag, stab):
    """Original linear z-score difference discordance."""
    mag_z = (mag - mag.mean()) / mag.std()
    stab_z = (stab - stab.mean()) / stab.std()
    return mag_z - stab_z


def _compute_discordance_rank(mag, stab):
    """Rank-based discordance: Dp = rank(Mp) - rank(Sp), z-scored."""
    rank_m = rankdata(mag)
    rank_s = rankdata(stab)
    dp = rank_m - rank_s
    return (dp - dp.mean()) / dp.std()


def _compute_discordance_loess(mag, stab, frac=0.3):
    """LOESS residual-based discordance: sign-flipped, z-scored."""
    fitted = lowess(stab, mag, frac=frac, return_sorted=False)
    resid = stab - fitted
    d = -resid
    return (d - d.mean()) / d.std()


def run_nonlinear_discordance(datasets_dict):
    """
    Recompute discordance rankings using rank-based and LOESS residuals
    alongside the original linear method.  Report whether top-discordant
    perturbations (Table 7 candidates) retain their rankings.

    Parameters
    ----------
    datasets_dict : dict
        {dataset_name: DataFrame} with columns stability, magnitude.
    """
    print("\n" + "=" * 80)
    print("TEST 1: NONLINEAR DISCORDANCE RESIDUALS")
    print("=" * 80)
    print("The magnitude–stability relationship is curved at low magnitudes.")
    print("Linear OLS residuals may bias discordance rankings.  Comparing:")
    print("  (a) Linear z-score difference (original)")
    print("  (b) Rank-based: Dp = rank(Mp) - rank(Sp), z-scored")
    print("  (c) LOESS residual (frac=0.3)\n")

    highlight_genes = {'GATA1', 'CHMP3', 'AQR'}
    all_results = []

    for ds_name in sorted(datasets_dict.keys()):
        df = datasets_dict[ds_name].copy()
        if len(df) < 20:
            print(f"  {ds_name}: skipped (n={len(df)} < 20)\n")
            continue

        mag = df['magnitude'].values
        stab = df['stability'].values
        perts = df.index.tolist()

        disc_linear = _compute_discordance_linear(mag, stab)
        disc_rank = _compute_discordance_rank(mag, stab)
        disc_loess = _compute_discordance_loess(mag, stab, frac=0.3)

        comp = pd.DataFrame({
            'perturbation': perts,
            'magnitude': mag,
            'stability': stab,
            'disc_linear': disc_linear,
            'disc_rank': disc_rank,
            'disc_loess': disc_loess,
        })

        comp['rank_linear'] = comp['disc_linear'].rank(ascending=False).astype(int)
        comp['rank_rank'] = comp['disc_rank'].rank(ascending=False).astype(int)
        comp['rank_loess'] = comp['disc_loess'].rank(ascending=False).astype(int)

        rho_lin_rank = spearmanr(disc_linear, disc_rank)
        rho_lin_loess = spearmanr(disc_linear, disc_loess)
        rho_rank_loess = spearmanr(disc_rank, disc_loess)

        print(f"  --- {ds_name} (n={len(df)}) ---")
        print(f"  Spearman correlation between methods:")
        print(f"    Linear vs Rank:  rho = {rho_lin_rank[0]:+.4f}  "
              f"p = {rho_lin_rank[1]:.2e}")
        print(f"    Linear vs LOESS: rho = {rho_lin_loess[0]:+.4f}  "
              f"p = {rho_lin_loess[1]:.2e}")
        print(f"    Rank vs LOESS:   rho = {rho_rank_loess[0]:+.4f}  "
              f"p = {rho_rank_loess[1]:.2e}")

        n_top = min(10, len(comp))
        for method, rank_col, disc_col in [
            ('Linear (original)', 'rank_linear', 'disc_linear'),
            ('Rank-based', 'rank_rank', 'disc_rank'),
            ('LOESS', 'rank_loess', 'disc_loess'),
        ]:
            top = comp.nsmallest(n_top, rank_col)
            print(f"\n  Top {n_top} discordant ({method}):")
            print(f"    {'Rank':>4s}  {'Perturbation':<16s}  {'Magnitude':>9s}  "
                  f"{'Stability':>9s}  {'Discordance':>11s}")
            for _, row in top.iterrows():
                marker = ' <<<' if row['perturbation'] in highlight_genes else ''
                print(f"    {int(row[rank_col]):>4d}  {row['perturbation']:<16s}  "
                      f"{row['magnitude']:>9.3f}  {row['stability']:>9.3f}  "
                      f"{row[disc_col]:>11.3f}{marker}")

        # Report tracked genes
        found_genes = highlight_genes & set(perts)
        if found_genes:
            print(f"\n  Tracked genes across methods:")
            print(f"    {'Gene':<10s}  {'Linear rank':>11s}  "
                  f"{'Rank-based':>10s}  {'LOESS':>10s}")
            for g in sorted(found_genes):
                row = comp[comp['perturbation'] == g].iloc[0]
                print(f"    {g:<10s}  {int(row['rank_linear']):>11d}  "
                      f"{int(row['rank_rank']):>10d}  "
                      f"{int(row['rank_loess']):>10d}")

        # Top-k overlap
        top_k = min(20, len(comp) // 4)
        top_linear = set(comp.nsmallest(top_k, 'rank_linear')['perturbation'])
        top_rank = set(comp.nsmallest(top_k, 'rank_rank')['perturbation'])
        top_loess = set(comp.nsmallest(top_k, 'rank_loess')['perturbation'])

        overlap_lin_rank = len(top_linear & top_rank) / top_k
        overlap_lin_loess = len(top_linear & top_loess) / top_k
        overlap_rank_loess = len(top_rank & top_loess) / top_k

        print(f"\n  Top-{top_k} overlap (Jaccard-style fraction):")
        print(f"    Linear & Rank:  {overlap_lin_rank:.2f}  "
              f"({len(top_linear & top_rank)}/{top_k})")
        print(f"    Linear & LOESS: {overlap_lin_loess:.2f}  "
              f"({len(top_linear & top_loess)}/{top_k})")
        print(f"    Rank & LOESS:   {overlap_rank_loess:.2f}  "
              f"({len(top_rank & top_loess)}/{top_k})")

        for _, row in comp.iterrows():
            all_results.append({
                'dataset': ds_name,
                'perturbation': row['perturbation'],
                'magnitude': row['magnitude'],
                'stability': row['stability'],
                'disc_linear': row['disc_linear'],
                'disc_rank': row['disc_rank'],
                'disc_loess': row['disc_loess'],
                'rank_linear': int(row['rank_linear']),
                'rank_rank': int(row['rank_rank']),
                'rank_loess': int(row['rank_loess']),
            })

        print()

    out = pd.DataFrame(all_results)
    out.to_csv(OUTPUT_DIR / "nonlinear_discordance_comparison.csv", index=False)
    print(f"Saved -> nonlinear_discordance_comparison.csv  ({len(out)} rows)\n")

    # Cross-dataset summary
    if not out.empty:
        print("--- SUMMARY: Method agreement across datasets ---")
        for ds_name in out['dataset'].unique():
            ds = out[out['dataset'] == ds_name]
            r_lr = spearmanr(ds['disc_linear'], ds['disc_rank'])[0]
            r_ll = spearmanr(ds['disc_linear'], ds['disc_loess'])[0]
            print(f"  {ds_name}: Linear-Rank rho={r_lr:+.3f}, "
                  f"Linear-LOESS rho={r_ll:+.3f}")

            found = highlight_genes & set(ds['perturbation'])
            if found:
                n = len(ds)
                for g in sorted(found):
                    row = ds[ds['perturbation'] == g].iloc[0]
                    pct_lin = 100 * row['rank_linear'] / n
                    pct_rank = 100 * row['rank_rank'] / n
                    pct_loess = 100 * row['rank_loess'] / n
                    print(f"    {g}: linear={row['rank_linear']}({pct_lin:.0f}%), "
                          f"rank={row['rank_rank']}({pct_rank:.0f}%), "
                          f"LOESS={row['rank_loess']}({pct_loess:.0f}%)")
        print()

    return out


# =============================================================================
# TEST 2: EMPIRICAL COMPARISON TO SONG et al. PS (Improvement 5)
#
# Compute the perturbation-response score (PS) — mean per-cell Euclidean
# distance from control centroid in PCA space — on all datasets.
# Show the Sp vs PS joint distribution controlling for magnitude.
# Identify specific perturbations where Sp provides non-redundant information.
# =============================================================================

def _compute_song_ps(adata_pca, pert_col, ctrl_label, valid_perts):
    """
    Compute Song et al. perturbation-response score:
    mean per-cell Euclidean distance from control centroid in PCA space.

    Accepts AnnData where PCA coords are in .X (from load_and_process's
    AnnData(X=X_pca, ...)) or in .obsm['X_pca'].
    """
    if 'X_pca' in adata_pca.obsm:
        X = adata_pca.obsm['X_pca']
    else:
        X = np.asarray(adata_pca.X)
    labels = adata_pca.obs[pert_col].values
    ctrl_mask = labels == ctrl_label
    ctrl_centroid = X[ctrl_mask].mean(axis=0)

    ps_dict = {}
    for pert in valid_perts:
        mask = labels == pert
        cells = X[mask]
        if len(cells) == 0:
            continue
        dists = np.linalg.norm(cells - ctrl_centroid, axis=1)
        ps_dict[pert] = float(dists.mean())
    return ps_dict


def run_song_ps_comparison(datasets_dict, adata_pca_dict):
    """
    Compute Song et al. PS on each dataset, show the Sp vs PS
    joint distribution controlling for magnitude, and identify perturbations
    where Sp provides non-redundant information.

    Parameters
    ----------
    datasets_dict : dict
        {dataset_name: DataFrame} with columns stability, magnitude (and
        optionally pw_UPR).
    adata_pca_dict : dict
        {dataset_name: (adata_pca, pert_col, ctrl_label)}.
    """
    print("\n" + "=" * 80)
    print("TEST 2: EMPIRICAL COMPARISON TO SONG et al. PS")
    print("=" * 80)
    print("Compute perturbation-response score (PS = mean per-cell Euclidean")
    print("distance from control centroid in PCA space) on each dataset.")
    print("Compare Sp vs PS controlling for magnitude.\n")

    all_results = []
    seed_ctr = SEED + 11000

    for ds_name in sorted(datasets_dict.keys()):
        df = datasets_dict[ds_name].copy()
        if ds_name not in adata_pca_dict:
            print(f"  {ds_name}: no AnnData available, skipped\n")
            continue

        adata_pca, pert_col, ctrl_label = adata_pca_dict[ds_name]
        valid_perts = list(df.index)

        ps_scores = _compute_song_ps(adata_pca, pert_col, ctrl_label,
                                     valid_perts)
        df['PS'] = df.index.map(ps_scores)
        df = df.dropna(subset=['PS']).copy()

        if len(df) < 15:
            print(f"  {ds_name}: only {len(df)} perturbations with PS, "
                  f"skipped\n")
            continue

        n = len(df)
        modality = MODALITY_MAP.get(ds_name, '?')
        print(f"  --- {ds_name} ({modality}, n={n}) ---")

        # --- Raw correlations ---
        rho_sp_ps = bootstrap_spearman_ci(
            df['stability'].values, df['PS'].values,
            n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
        seed_ctr += 1

        rho_mag_ps = bootstrap_spearman_ci(
            df['magnitude'].values, df['PS'].values,
            n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
        seed_ctr += 1

        rho_sp_mag = bootstrap_spearman_ci(
            df['stability'].values, df['magnitude'].values,
            n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
        seed_ctr += 1

        print(f"  Raw Spearman correlations:")
        print(f"    Sp vs PS:   rho = {rho_sp_ps['rho']:+.3f}  "
              f"[{rho_sp_ps['ci_low']:.3f}, {rho_sp_ps['ci_high']:.3f}]  "
              f"p = {rho_sp_ps['p']:.2e}")
        print(f"    Mp vs PS:   rho = {rho_mag_ps['rho']:+.3f}  "
              f"[{rho_mag_ps['ci_low']:.3f}, {rho_mag_ps['ci_high']:.3f}]  "
              f"p = {rho_mag_ps['p']:.2e}")
        print(f"    Sp vs Mp:   rho = {rho_sp_mag['rho']:+.3f}  "
              f"[{rho_sp_mag['ci_low']:.3f}, {rho_sp_mag['ci_high']:.3f}]  "
              f"p = {rho_sp_mag['p']:.2e}")

        # --- Partial correlation: Sp vs PS | magnitude ---
        partial_sp_ps = bootstrap_partial_correlation_ci(
            df['stability'].values, df['PS'].values,
            df['magnitude'].values,
            n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
        seed_ctr += 1

        print(f"\n  Partial Spearman (Sp vs PS | magnitude):")
        print(f"    rho = {partial_sp_ps['rho_partial']:+.3f}  "
              f"[{partial_sp_ps['ci_low']:.3f}, "
              f"{partial_sp_ps['ci_high']:.3f}]  "
              f"p = {partial_sp_ps['p']:.2e}")

        r2 = partial_sp_ps['rho_partial'] ** 2
        shared_var = rho_sp_ps['rho'] ** 2
        print(f"    Shared variance (raw R^2): "
              f"{shared_var:.3f} ({100*shared_var:.1f}%)")
        print(f"    Residual R^2 after magnitude: "
              f"{r2:.3f} ({100*r2:.1f}%)")
        if shared_var < 0.25:
            redundancy = 'LOW'
        elif shared_var < 0.50:
            redundancy = 'MODERATE'
        else:
            redundancy = 'HIGH'
        print(f"    -> {redundancy} redundancy between Sp and PS")

        # --- Identify non-redundant perturbations ---
        Z = sm.add_constant(df['magnitude'].values)
        sp_resid = sm.OLS(df['stability'].values, Z).fit().resid
        ps_resid = sm.OLS(df['PS'].values, Z).fit().resid

        sp_resid_z = (sp_resid - sp_resid.mean()) / sp_resid.std()
        ps_resid_z = (ps_resid - ps_resid.mean()) / ps_resid.std()

        df['Sp_resid_z'] = sp_resid_z
        df['PS_resid_z'] = ps_resid_z

        hi_sp = sp_resid_z > 1.0
        lo_sp = sp_resid_z < -1.0
        hi_ps = ps_resid_z > 1.0
        lo_ps = ps_resid_z < -1.0

        n_hi_sp_lo_ps = int((hi_sp & lo_ps).sum())
        n_lo_sp_hi_ps = int((lo_sp & hi_ps).sum())
        n_hi_both = int((hi_sp & hi_ps).sum())
        n_lo_both = int((lo_sp & lo_ps).sum())

        print(f"\n  Magnitude-residualized quadrants (|z| > 1):")
        print(f"    High-Sp / Low-PS (Sp captures, PS misses):  "
              f"{n_hi_sp_lo_ps}")
        print(f"    Low-Sp / High-PS (PS captures, Sp misses):  "
              f"{n_lo_sp_hi_ps}")
        print(f"    High both (redundant signal):               "
              f"{n_hi_both}")
        print(f"    Low both  (redundant signal):               "
              f"{n_lo_both}")

        non_redundant_sp = df[hi_sp & lo_ps].copy()
        non_redundant_ps = df[lo_sp & hi_ps].copy()

        if len(non_redundant_sp) > 0:
            print(f"\n  Perturbations where Sp provides non-redundant "
                  f"info (high-Sp, low-PS):")
            nr = non_redundant_sp.sort_values('Sp_resid_z', ascending=False)
            for g, r in nr.head(10).iterrows():
                print(f"    {g:<16s}  Sp_z={r['Sp_resid_z']:+.2f}  "
                      f"PS_z={r['PS_resid_z']:+.2f}  "
                      f"Sp={r['stability']:.3f}  PS={r['PS']:.3f}  "
                      f"Mp={r['magnitude']:.3f}")

        if len(non_redundant_ps) > 0:
            print(f"\n  Perturbations where PS captures but Sp does not "
                  f"(low-Sp, high-PS):")
            nr = non_redundant_ps.sort_values('PS_resid_z', ascending=False)
            for g, r in nr.head(10).iterrows():
                print(f"    {g:<16s}  Sp_z={r['Sp_resid_z']:+.2f}  "
                      f"PS_z={r['PS_resid_z']:+.2f}  "
                      f"Sp={r['stability']:.3f}  PS={r['PS']:.3f}  "
                      f"Mp={r['magnitude']:.3f}")

        # --- Incremental variance: does Sp predict UPR beyond PS + magnitude? ---
        if 'pw_UPR' in df.columns:
            sub = df.dropna(subset=['pw_UPR']).copy()
            if len(sub) >= 15:
                Z_mag = sm.add_constant(sub['magnitude'].values)
                Z_mag_ps = sm.add_constant(
                    np.column_stack([sub['magnitude'].values,
                                     sub['PS'].values]))

                upr = sub['pw_UPR'].values

                upr_resid_mag = sm.OLS(upr, Z_mag).fit().resid
                sp_resid_mag = sm.OLS(
                    sub['stability'].values, Z_mag).fit().resid
                rho_incr_sp, p_incr_sp = spearmanr(
                    upr_resid_mag, sp_resid_mag)

                upr_resid_mag_ps = sm.OLS(upr, Z_mag_ps).fit().resid
                sp_resid_mag_ps = sm.OLS(
                    sub['stability'].values, Z_mag_ps).fit().resid
                rho_incr_sp_over_ps, p_incr_sp_over_ps = spearmanr(
                    upr_resid_mag_ps, sp_resid_mag_ps)

                ps_resid_mag = sm.OLS(
                    sub['PS'].values, Z_mag).fit().resid
                rho_incr_ps, p_incr_ps = spearmanr(
                    upr_resid_mag, ps_resid_mag)

                print(f"\n  Incremental predictive power for UPR pathway "
                      f"score:")
                print(f"    Sp | magnitude:          rho = "
                      f"{rho_incr_sp:+.3f}  p = {p_incr_sp:.2e}")
                print(f"    PS | magnitude:          rho = "
                      f"{rho_incr_ps:+.3f}  p = {p_incr_ps:.2e}")
                print(f"    Sp | magnitude + PS:     rho = "
                      f"{rho_incr_sp_over_ps:+.3f}  "
                      f"p = {p_incr_sp_over_ps:.2e}")
                if (abs(rho_incr_sp_over_ps) > 0.1
                        and p_incr_sp_over_ps < 0.05):
                    print(f"    -> Sp provides SIGNIFICANT incremental "
                          f"info beyond PS")
                else:
                    print(f"    -> Sp does NOT add significant info beyond "
                          f"PS for UPR")

        for _, r in df.iterrows():
            all_results.append({
                'dataset': ds_name,
                'modality': modality,
                'perturbation': r.name,
                'stability': r['stability'],
                'magnitude': r['magnitude'],
                'PS': r['PS'],
                'Sp_resid_z': r.get('Sp_resid_z', np.nan),
                'PS_resid_z': r.get('PS_resid_z', np.nan),
            })

        print()

    out = pd.DataFrame(all_results)
    out.to_csv(OUTPUT_DIR / "song_ps_comparison.csv", index=False)
    print(f"Saved -> song_ps_comparison.csv  ({len(out)} rows)\n")

    if not out.empty:
        print("--- CROSS-DATASET SUMMARY ---")
        for ds_name in out['dataset'].unique():
            ds = out[out['dataset'] == ds_name]
            rho_raw = spearmanr(ds['stability'], ds['PS'])[0]
            print(f"  {ds_name}: Sp-PS raw rho = {rho_raw:+.3f}, "
                  f"n = {len(ds)}")
        print()

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("METHODOLOGICAL ROBUSTNESS TESTS")
    print("=" * 80)

    datasets = {}
    adata_pca_dict = {}

    # --- Replogle 2022 ---
    df_r, adata_pca_r = load_and_process(
        'Replogle 2022 (CRISPRi)',
        pt.dt.replogle_2022_k562_essential,
        pert_col='condition', ctrl_label='control',
        clean_func=clean_replogle, min_cells=REPLOGLE_MIN_CELLS,
    )
    datasets['Replogle 2022 (CRISPRi)'] = df_r
    adata_pca_dict['Replogle 2022 (CRISPRi)'] = (
        adata_pca_r, 'condition', 'control')

    # --- Norman 2019 ---
    df_n, adata_pca_n = load_and_process(
        'Norman 2019 (CRISPRa)',
        pt.dt.norman_2019,
        pert_col='perturbation_name', ctrl_label='control',
        min_cells=NORMAN_MIN_CELLS,
    )
    datasets['Norman 2019 (CRISPRa)'] = df_n
    adata_pca_dict['Norman 2019 (CRISPRa)'] = (
        adata_pca_n, 'perturbation_name', 'control')

    # --- Dixit 2016 ---
    df_d, adata_pca_d = load_and_process(
        'Dixit 2016 (CRISPRi)',
        pt.dt.dixit_2016,
        pert_col='perturbation_name', ctrl_label='control',
        min_cells=DIXIT_MIN_CELLS,
    )
    datasets['Dixit 2016 (CRISPRi)'] = df_d
    adata_pca_dict['Dixit 2016 (CRISPRi)'] = (
        adata_pca_d, 'perturbation_name', 'control')

    # --- Papalexi 2021 ---
    df_p, adata_pca_p = load_and_process(
        'Papalexi 2021 (CRISPR)',
        load_papalexi_rna,
        pert_col='gene_target', ctrl_label='NT',
        min_cells=PAPALEXI_MIN_CELLS,
    )
    datasets['Papalexi 2021 (CRISPR)'] = df_p
    adata_pca_dict['Papalexi 2021 (CRISPR)'] = (
        adata_pca_p, 'gene_target', 'NT')

    # --- Adamson 2016 ---
    df_a, adata_pca_a = load_and_process(
        'Adamson 2016 (CRISPRi)',
        pt.dt.adamson_2016_pilot,
        pert_col='condition', ctrl_label='control',
        clean_func=clean_adamson, min_cells=ADAMSON_MIN_CELLS,
    )
    datasets['Adamson 2016 (CRISPRi)'] = df_a
    adata_pca_dict['Adamson 2016 (CRISPRi)'] = (
        adata_pca_a, 'condition', 'control')

    # ==================================================================
    # TEST 1: Nonlinear discordance residuals (Improvement 4)
    # ==================================================================
    run_nonlinear_discordance(datasets)

    # ==================================================================
    # TEST 2: Song PS comparison (Improvement 5)
    # ==================================================================
    run_song_ps_comparison(datasets, adata_pca_dict)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("=" * 80)
    print("ROBUSTNESS TESTS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  Test 1 (nonlinear discordance comparison):")
    print("    - nonlinear_discordance_comparison.csv")
    print("  Test 2 (Song PS comparison):")
    print("    - song_ps_comparison.csv")


if __name__ == "__main__":
    main()
