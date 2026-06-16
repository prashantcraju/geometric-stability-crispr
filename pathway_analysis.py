#!/usr/bin/env python3
"""
Pathway-Level Analysis of Geometric Instability

Approach A (Main Text): Perturbation-level gene set activity scores
  For each perturbation, compute mean expression of MSigDB Hallmark gene sets.
  Run Spearman + partial correlation (controlling magnitude) between Sp and
  each pathway score.  Directly extends the Fig 3c stress marker framework.

Approach B (Supplementary): Differential pathway enrichment, discordance Q1 vs Q4
  Within a magnitude-matched band (middle two magnitude quartiles), split
  perturbations by discordance into Q1 (most concordant) vs Q4 (most discordant).
  Run Wilcoxon DE between Q4 and Q1 cells, then preranked GSEA against MSigDB
  Hallmark, KEGG, and Reactome collections.

Datasets:
  Approach A — Replogle 2022, Norman 2019, Dixit 2016
  Approach B — Replogle 2022, Norman 2019

INPUT:  Raw datasets loaded via pertpy
OUTPUT: pathway_signature_correlations.csv        (Approach A)
        pathway_gsea_Q4_vs_Q1_replogle.csv        (Approach B)
        pathway_gsea_Q4_vs_Q1_norman.csv          (Approach B)
        pathway_de_Q4_vs_Q1_replogle.csv          (Approach B — full DE table)
        pathway_de_Q4_vs_Q1_norman.csv            (Approach B — full DE table)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt

from anndata import AnnData
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from shesha.bio import compute_stability, compute_magnitude

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False
    print("WARNING: gseapy not installed — Approach B (GSEA) will be skipped.")
    print("Install with:  pip install gseapy")

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 320
np.random.seed(SEED)
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95

OUTPUT_DIR = Path("./shesha-crispr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPLOGLE_MIN_CELLS = 50
NORMAN_MIN_CELLS   = 50
DIXIT_MIN_CELLS    = 10

MIN_GENE_OVERLAP = 5      # require ≥5 genes from a set present in the data
GSEA_PERMUTATIONS = 1000

# Enrichr / gseapy library names for Approach B
GSEA_LIBRARIES = [
    'MSigDB_Hallmark_2020',
    'KEGG_2021_Human',
    'Reactome_2022',
]

# =============================================================================
# MSigDB HALLMARK GENE SETS  (canonical v2023.2, curated core members)
#
# Each list contains the well-characterised members most commonly detected in
# scRNA-seq.  The analysis reports overlap with each dataset's var_names so the
# effective gene-set size is transparent.
# =============================================================================

HALLMARK_GENE_SETS = {
    'UPR': [
        # ER chaperones / foldases
        'HSPA5', 'HSP90B1', 'HYOU1', 'CALR', 'CANX', 'P4HB', 'PDIA3',
        'PDIA4', 'PDIA5', 'PDIA6', 'PPIB', 'ERP29', 'ERP44', 'SIL1',
        'FKBP14', 'DNAJB9', 'DNAJB11', 'DNAJC3', 'DNAJC10',
        # UPR sensors / effectors
        'ATF6', 'ATF6B', 'ERN1', 'EIF2AK3', 'XBP1', 'DDIT3', 'CREB3L2',
        # ERAD
        'EDEM1', 'DERL1', 'OS9', 'SEL1L', 'SYVN1', 'UBE2J1', 'UBE2D1',
        'VIMP', 'YOD1', 'VCP',
        # translocon / signal peptidase
        'SEC61A1', 'SEC61B', 'SEC11C', 'SEC24D', 'TRAM1', 'SRPRB',
        'SPCS1', 'SPCS2', 'SPCS3', 'SSR1', 'SSR3', 'SSR4',
        # ER-to-Golgi / vesicle
        'LMAN1', 'GOSR2', 'KDELR3', 'SURF4',
        # glycosylation / lipid
        'DDOST', 'STT3A', 'STT3B', 'RPN1', 'RPN2', 'MOGS', 'UGGT1',
        'SRD5A3',
        # other UPR-associated
        'HERPUD1', 'MANF', 'CRELD2', 'SDF2L1', 'NUCB1', 'RCN1',
        'SERP1', 'WIPI1', 'UFM1', 'BAX', 'ERO1A', 'MBTPS1', 'MBTPS2',
        'ARCN1', 'PREB', 'GANAB', 'TMX1', 'ERLEC1',
    ],
    'mTORC1': [
        # lipid / cholesterol synthesis
        'ACLY', 'ACSS2', 'SCD', 'FASN', 'ELOVL5', 'ELOVL6', 'FADS1',
        'FADS2', 'HMGCR', 'HMGCS1', 'MVK', 'MVD', 'IDI1', 'FDPS',
        'FDFT1', 'SQLE', 'LSS', 'DHCR7', 'DHCR24', 'SC5D', 'NSDHL',
        'TM7SF2', 'HSD17B7', 'INSIG1', 'INSIG2', 'LDLR', 'STARD4',
        'VLDLR',
        # nucleotide synthesis
        'IMPDH2', 'RRM1', 'RRM2', 'PRPS1', 'UNG',
        # amino acid / one-carbon
        'PHGDH', 'PSPH', 'SHMT2', 'MTHFD2', 'MTHFD1L', 'SLC7A5',
        'SLC7A11', 'SLC1A4', 'BCAT1', 'GOT1', 'IARS1', 'AARS1',
        # glycolysis
        'HK2', 'GPI', 'PGK1', 'PGM1', 'TPI1', 'PKM', 'SLC2A1',
        # proteasome
        'PSMA3', 'PSMA4', 'PSMB2', 'PSMB4', 'PSMC2', 'PSMC4',
        'PSMC6', 'PSMD1', 'PSMD12', 'PSMD13', 'PSMD14', 'PSME3',
        # ribosome biogenesis / translation
        'NOP14', 'NOP56', 'PNO1', 'GNL3',
        # redox / chaperones
        'PRDX1', 'PRDX6', 'GLRX', 'G6PD', 'PPIA', 'PPIB', 'CALR',
        'CANX', 'HSPA5', 'HSPA9', 'SERP1', 'SSR1',
        # signalling
        'MYC', 'DDIT4', 'TRIB3', 'CDKN1A', 'CCNF', 'BHLHE40',
        'NAMPT', 'AK4', 'ME1', 'PC', 'IDH1',
    ],
    'p53': [
        'CDKN1A', 'MDM2', 'MDM4', 'BAX', 'BBC3', 'PMAIP1', 'PERP',
        'FAS', 'TNFRSF10B', 'PIDD1', 'APAF1', 'CASP1',
        'GADD45A', 'GADD45B', 'GADD45G', 'DDIT4', 'SESN1', 'SESN2',
        'DDB2', 'RRM2B', 'PCNA', 'POLK', 'RPA1',
        'TP53I3', 'TP53INP1', 'ZMAT3', 'EI24', 'TRIAP1', 'AEN',
        'BTG2', 'CCNG1', 'CCNG2', 'PLK2', 'PLK3', 'SFN',
        'GDF15', 'ATF3', 'FDXR', 'SCO2', 'TIGAR', 'GLS2',
        'SAT1', 'STEAP3', 'RPS27L', 'PTEN', 'TSC2',
        'IGFBP4', 'HDAC1', 'ISG15', 'GPX1', 'ING1',
    ],
    'Apoptosis': [
        # BCL-2 family
        'BAX', 'BAK1', 'BCL2', 'BCL2L1', 'BCL2L11', 'BCL2L2', 'MCL1',
        'BID', 'BAD', 'BIK', 'HRK', 'BNIP3', 'BNIP3L', 'PMAIP1',
        # caspases
        'CASP1', 'CASP2', 'CASP3', 'CASP4', 'CASP6', 'CASP7', 'CASP8',
        'CASP9',
        # death receptors / ligands
        'FAS', 'FADD', 'TNFRSF10A', 'TNFRSF10B', 'TNFRSF1A', 'TNFSF10',
        'RIPK1', 'RIPK2', 'CFLAR', 'TRAF1', 'TRAF2',
        # mitochondrial
        'CYCS', 'DIABLO', 'APAF1', 'AIFM1', 'ENDOG', 'HTRA2',
        # inhibitors
        'BIRC2', 'BIRC3', 'XIAP',
        # DNA fragmentation / structural
        'DFFA', 'DFFB', 'LMNA', 'LMNB1', 'LMNB2', 'SPTAN1',
        # other pro-apoptotic / regulators
        'CDKN1A', 'JUN', 'MYC', 'NFKBIA', 'SQSTM1', 'CLU', 'TXNIP',
        'ATF3', 'FDXR', 'SAT1', 'GPX1', 'GPX3', 'GPX4', 'GSR',
        'SOD1', 'SOD2', 'ETF1', 'ADD1', 'APP', 'ANXA1', 'DCN',
        'CD44', 'CCND1', 'CCND2', 'RHOB',
    ],
    'ROS': [
        # superoxide dismutases
        'SOD1', 'SOD2',
        # catalase
        'CAT',
        # glutathione peroxidases
        'GPX1', 'GPX3', 'GPX4',
        # thioredoxin system
        'TXN', 'TXNRD1', 'TXNIP',
        # peroxiredoxins
        'PRDX1', 'PRDX2', 'PRDX3', 'PRDX4', 'PRDX5', 'PRDX6',
        # glutathione synthesis / conjugation
        'GCLC', 'GCLM', 'GSR', 'GSS', 'GLRX', 'GLRX2',
        'GSTP1', 'GSTM1', 'GSTM2', 'GSTM4', 'GSTO1', 'MGST1',
        # Nrf2 / KEAP1
        'NFE2L2', 'KEAP1', 'NQO1', 'HMOX1', 'HMOX2', 'SRXN1',
        # iron / ferritin
        'FTH1', 'FTL',
        # other redox
        'G6PD', 'MSRA', 'PARK7', 'SQSTM1', 'EPHX1',
        'ALDH1A1', 'AKR1B1', 'OXSR1', 'STK25', 'JUNB',
        'ABCC1', 'PRNP', 'PDLIM1', 'CLU', 'CYBA', 'NOX4',
        # mitochondrial complex I (ROS source)
        'NDUFA6', 'NDUFB4', 'NDUFC2', 'NDUFS2',
    ],
}

PATHWAY_FULL_NAMES = {
    'UPR':       'HALLMARK_UNFOLDED_PROTEIN_RESPONSE',
    'mTORC1':    'HALLMARK_MTORC1_SIGNALING',
    'p53':       'HALLMARK_P53_PATHWAY',
    'Apoptosis': 'HALLMARK_APOPTOSIS',
    'ROS':       'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY',
}


# =============================================================================
# BOOTSTRAP HELPERS  (same framework as stress_marker_tests.py)
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=42):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return {'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p': np.nan}
    rng = np.random.default_rng(seed=seed)
    boot = np.empty(n_bootstrap)
    n = len(x)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
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


def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=N_BOOTSTRAP,
                                     ci_level=CI_LEVEL, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1, 1) if np.asarray(z).ndim == 1 \
        else np.asarray(z, dtype=float)
    n = len(x)

    def _partial(x, y, z):
        Z_aug = sm.add_constant(z)
        x_r = sm.OLS(x, Z_aug).fit().resid
        y_r = sm.OLS(y, Z_aug).fit().resid
        return spearmanr(x_r, y_r)

    rho_partial, p = _partial(x, y, z)
    rng = np.random.default_rng(seed=seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot[i], _ = _partial(x[idx], y[idx], z[idx])
    valid = boot[~np.isnan(boot)]
    if len(valid) < 10:
        return {'rho_partial': rho_partial, 'ci_low': np.nan,
                'ci_high': np.nan, 'p': p}
    alpha = 1 - ci_level
    return {
        'rho_partial': rho_partial, 'p': p,
        'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
    }


# =============================================================================
# DATA LOADING  (mirrors fig3.py / fig_replogle.py patterns)
# =============================================================================

def clean_replogle(adata):
    """Label-clean Replogle 2022: merge non-targeting / chr → control."""
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


def load_and_process(name, loader_func, pert_col, ctrl_label,
                     clean_func=None, min_cells=50):
    """
    Load dataset → normalize (all genes) → compute pathway scores +
    Shesha stability / magnitude.

    Returns
    -------
    df : pd.DataFrame   — one row per perturbation (stability, magnitude,
                           pw_UPR, pw_mTORC1, …)
    adata_norm : AnnData — normalised expression with ALL genes (for Approach B DE)
    """
    print(f"\n>>> Loading {name} ...")
    adata = loader_func()
    if clean_func:
        adata = clean_func(adata)
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    # --- Normalise (keep all genes for pathway scoring) ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_norm = adata  # alias; all genes retained

    # --- Identify valid perturbations ---
    counts = adata_norm.obs[pert_col].value_counts()
    valid = [v for v in counts[counts >= min_cells].index if v != ctrl_label]
    print(f"    {len(valid)} perturbations with ≥{min_cells} cells")

    # --- Compute pathway scores per perturbation ---
    # Uses sc.tl.score_genes() which subtracts a size-matched reference gene set,
    # preventing highly expressed housekeeping genes from dominating the score.
    # This provides genuine pathway-level evidence beyond individual gene effects.
    pw_scores = {}
    for pw_short, pw_genes in HALLMARK_GENE_SETS.items():
        overlap = [g for g in pw_genes if g in adata_norm.var_names]
        pct = 100 * len(overlap) / len(pw_genes) if pw_genes else 0
        print(f"    {pw_short}: {len(overlap)}/{len(pw_genes)} genes "
              f"({pct:.0f}% overlap)")
        if len(overlap) < MIN_GENE_OVERLAP:
            print(f"      ↳ skipped (< {MIN_GENE_OVERLAP} genes)")
            continue

        score_col = f'score_{pw_short}'
        sc.tl.score_genes(adata_norm, gene_list=overlap,
                          score_name=score_col,
                          ctrl_size=50, random_state=SEED)

        scores = {}
        for pert in valid:
            mask = adata_norm.obs[pert_col] == pert
            scores[pert] = float(adata_norm[mask].obs[score_col].mean())
        pw_scores[pw_short] = scores

    # --- Compute Shesha stability / magnitude via HVG + PCA ---
    adata_proc = adata_norm[
        adata_norm.obs[pert_col].isin(valid + [ctrl_label])
    ].copy()
    sc.pp.highly_variable_genes(adata_proc, n_top_genes=2000, subset=True)
    sc.tl.pca(adata_proc, n_comps=50)

    adata_pca = AnnData(X=adata_proc.obsm['X_pca'], obs=adata_proc.obs)
    stab = compute_stability(adata_pca, perturbation_key=pert_col,
                             control_label=ctrl_label, metric='cosine')
    mag  = compute_magnitude(adata_pca, perturbation_key=pert_col,
                             control_label=ctrl_label, metric='euclidean')

    # --- Build per-perturbation DataFrame ---
    df = pd.DataFrame({'stability': pd.Series(stab),
                       'magnitude': pd.Series(mag)})
    if ctrl_label in df.index:
        df = df.drop(ctrl_label)
    df = df[df.index.isin(valid)].copy()

    for pw_short, scores in pw_scores.items():
        df[f'pw_{pw_short}'] = df.index.map(scores)

    print(f"    → {len(df)} perturbations in final table")
    return df, adata_norm


# =============================================================================
# APPROACH A: Perturbation-level pathway-score ↔ stability correlations
# =============================================================================

def approach_a(df, dataset_name, results_list):
    """
    For each pathway score, compute raw Spearman and partial Spearman
    (controlling for magnitude) against stability.  Appends rows to
    *results_list* (mutated in place).
    """
    print(f"\n{'=' * 70}")
    print(f"APPROACH A — Pathway Signature Correlations: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"  n = {len(df)} perturbations\n")

    pw_cols = sorted(c for c in df.columns if c.startswith('pw_'))
    # Deterministic per-dataset seed derived from global SEED
    seed_ctr = SEED + sum(ord(c) for c in dataset_name)

    for col in pw_cols:
        pw_name = col[3:]  # strip 'pw_'
        sub = df.dropna(subset=[col]).copy()
        if len(sub) < 15:
            print(f"  {pw_name}: skipped (n = {len(sub)} < 15)")
            continue

        raw = bootstrap_spearman_ci(
            sub['stability'].values, sub[col].values,
            seed=seed_ctr)
        seed_ctr += 1

        partial = bootstrap_partial_correlation_ci(
            sub['stability'].values, sub[col].values,
            sub['magnitude'].values,
            seed=seed_ctr)
        seed_ctr += 1

        abs_rho = abs(partial['rho_partial'])
        ci_excludes_zero = (
            not np.isnan(partial['ci_low'])
            and np.sign(partial['ci_low']) == np.sign(partial['ci_high'])
        )

        if abs_rho >= 0.3:
            effect = 'medium–large'
        elif abs_rho >= 0.2:
            effect = 'small–medium'
        elif abs_rho >= 0.1:
            effect = 'small'
        else:
            effect = 'negligible'

        survives = abs_rho > 0.1 and ci_excludes_zero

        print(f"  {pw_name} (n={len(sub)}):")
        print(f"    Raw:     ρ = {raw['rho']:+.3f}  "
              f"[{raw['ci_low']:.3f}, {raw['ci_high']:.3f}]  "
              f"p = {raw['p']:.2e}")
        print(f"    Partial: ρ = {partial['rho_partial']:+.3f}  "
              f"[{partial['ci_low']:.3f}, {partial['ci_high']:.3f}]  "
              f"p = {partial['p']:.2e}")
        print(f"    Effect: {effect}  |  "
              f"Survives magnitude control: {'YES' if survives else 'no'}")

        results_list.append({
            'dataset':               dataset_name,
            'pathway':               pw_name,
            'pathway_full':          PATHWAY_FULL_NAMES.get(pw_name, ''),
            'n':                     len(sub),
            'rho_raw':               raw['rho'],
            'rho_raw_ci_low':        raw['ci_low'],
            'rho_raw_ci_high':       raw['ci_high'],
            'p_raw':                 raw['p'],
            'rho_partial':           partial['rho_partial'],
            'rho_partial_ci_low':    partial['ci_low'],
            'rho_partial_ci_high':   partial['ci_high'],
            'p_partial':             partial['p'],
            'abs_rho_partial':       abs_rho,
            'effect_size':           effect,
            'ci_excludes_zero':      ci_excludes_zero,
            'survives_magnitude_control': survives,
        })


# =============================================================================
# APPROACH B: Discordance-quartile GSEA  (Q4 discordant vs Q1 concordant)
# =============================================================================

def approach_b(adata_norm, pert_col, ctrl_label, shesha_df,
               dataset_name, min_cells=50):
    """
    1. Restrict to middle 50 % of magnitude (Q2–Q3).
    2. Within that band, split by discordance into Q1 (concordant) / Q4 (discordant).
    3. Wilcoxon DE between Q4 cells and Q1 cells (scanpy).
    4. Pre-ranked GSEA against Hallmark / KEGG / Reactome.
    """
    if not HAS_GSEAPY:
        print(f"\n  Approach B skipped for {dataset_name}: gseapy not installed.")
        return None, None

    print(f"\n{'=' * 70}")
    print(f"APPROACH B — Discordance-Quartile GSEA: {dataset_name}")
    print(f"{'=' * 70}")

    df = shesha_df.copy()

    # --- Compute discordance as standardised regression residual ---
    # Residual-based: deviation of stability from the magnitude–stability
    # regression fit. Negative residual = lower stability than expected.
    # Inverted so that positive discordance = "too unstable for its magnitude".
    reg = LinearRegression().fit(df[['magnitude']], df['stability'])
    df['predicted_stab'] = reg.predict(df[['magnitude']])
    resid = df['stability'] - df['predicted_stab']
    df['discordance'] = -(resid - resid.mean()) / resid.std()

    # --- Restrict to middle magnitude band (Q2–Q3) ---
    q1_mag = df['magnitude'].quantile(0.25)
    q3_mag = df['magnitude'].quantile(0.75)
    df_mid = df[(df['magnitude'] >= q1_mag) & (df['magnitude'] <= q3_mag)].copy()
    print(f"  Perturbations total: {len(df)}")
    print(f"  After magnitude restriction (Q2–Q3): {len(df_mid)}")

    if len(df_mid) < 20:
        print("  Too few perturbations after filtering — skipping.")
        return None, None

    # --- Split by discordance ---
    df_mid['disc_q'] = pd.qcut(df_mid['discordance'], q=4,
                                labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q1_perts = set(df_mid.loc[df_mid['disc_q'] == 'Q1'].index)
    q4_perts = set(df_mid.loc[df_mid['disc_q'] == 'Q4'].index)

    print(f"  Q1 (concordant):  {len(q1_perts)} perturbations")
    print(f"  Q4 (discordant):  {len(q4_perts)} perturbations")

    # Verify magnitude matching
    q1_m = df_mid.loc[df_mid.index.isin(q1_perts), 'magnitude']
    q4_m = df_mid.loc[df_mid.index.isin(q4_perts), 'magnitude']
    print(f"  Q1 magnitude: {q1_m.mean():.3f} ± {q1_m.std():.3f}")
    print(f"  Q4 magnitude: {q4_m.mean():.3f} ± {q4_m.std():.3f}")

    # --- Label cells & subset ---
    all_wanted = q1_perts | q4_perts
    keep_mask = adata_norm.obs[pert_col].isin(all_wanted)
    adata_sub = adata_norm[keep_mask].copy()

    group_map = {}
    for p in q1_perts:
        group_map[p] = 'Q1_concordant'
    for p in q4_perts:
        group_map[p] = 'Q4_discordant'
    adata_sub.obs['disc_group'] = adata_sub.obs[pert_col].map(group_map)
    adata_sub = adata_sub[adata_sub.obs['disc_group'].notna()].copy()

    n_q1 = (adata_sub.obs['disc_group'] == 'Q1_concordant').sum()
    n_q4 = (adata_sub.obs['disc_group'] == 'Q4_discordant').sum()
    print(f"  Q1 cells: {n_q1:,}   Q4 cells: {n_q4:,}")

    # --- Wilcoxon DE: Q4 vs Q1 ---
    print("  Running Wilcoxon rank-sum (Q4 vs Q1) ...")
    sc.tl.rank_genes_groups(adata_sub, groupby='disc_group',
                            reference='Q1_concordant', method='wilcoxon',
                            key_added='de_q4_vs_q1')
    de_df = sc.get.rank_genes_groups_df(adata_sub, group='Q4_discordant',
                                        key='de_q4_vs_q1')
    de_df = de_df.dropna(subset=['names', 'scores']).copy()
    de_df = de_df.sort_values('scores', ascending=False).reset_index(drop=True)

    # Save full DE table
    tag = dataset_name.split('(')[0].strip().lower().replace(' ', '_')
    de_path = OUTPUT_DIR / f"pathway_de_Q4_vs_Q1_{tag}.csv"
    de_df.to_csv(de_path, index=False)
    print(f"  Saved DE table ({len(de_df)} genes) → {de_path.name}")

    print(f"\n  Top 10 UP in Q4 (discordant / low-stability):")
    _show = de_df.head(10)[['names', 'scores', 'logfoldchanges', 'pvals_adj']]
    print(_show.to_string(index=False))
    print(f"\n  Top 10 DOWN in Q4:")
    _show = de_df.tail(10)[['names', 'scores', 'logfoldchanges', 'pvals_adj']]
    print(_show.to_string(index=False))

    # --- Prepare ranked gene list for GSEA ---
    rnk = de_df[['names', 'scores']].drop_duplicates(subset='names')
    rnk = rnk.rename(columns={'names': 'Gene', 'scores': 'Score'})
    rnk = rnk.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Pre-ranked GSEA ---
    all_gsea = []
    for lib in GSEA_LIBRARIES:
        print(f"\n  GSEA: {lib} ...")
        try:
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=lib,
                min_size=5,
                max_size=500,
                permutation_num=GSEA_PERMUTATIONS,
                outdir=None,
                seed=SEED,
                verbose=False,
            )
            res = pre_res.res2d.copy()
            res['library'] = lib

            sig = res[res['FDR q-val'].astype(float) < 0.25].sort_values('NES')
            if len(sig) > 0:
                print(f"    {len(sig)} terms with FDR < 0.25:")
                for _, r in sig.iterrows():
                    print(f"      NES={float(r['NES']):+.2f}  "
                          f"FDR={float(r['FDR q-val']):.3f}  {r['Term']}")
            else:
                print(f"    No terms with FDR < 0.25")
            all_gsea.append(res)
        except Exception as e:
            print(f"    Error: {e}")

    gsea_df = None
    if all_gsea:
        gsea_df = pd.concat(all_gsea, ignore_index=True)
        gsea_df['dataset'] = dataset_name
        gsea_path = OUTPUT_DIR / f"pathway_gsea_Q4_vs_Q1_{tag}.csv"
        gsea_df.to_csv(gsea_path, index=False)
        print(f"\n  Saved GSEA results → {gsea_path.name}")

    return de_df, gsea_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_corr = []   # Approach A rows

    # =================================================================
    # REPLOGLE 2022 (CRISPRi)
    # =================================================================
    print("\n" + "=" * 80)
    print("REPLOGLE 2022 (CRISPRi)")
    print("=" * 80)
    df_r, adata_r = load_and_process(
        'Replogle 2022 (CRISPRi)',
        pt.dt.replogle_2022_k562_essential,
        pert_col='condition', ctrl_label='control',
        clean_func=clean_replogle, min_cells=REPLOGLE_MIN_CELLS,
    )
    approach_a(df_r, 'Replogle 2022 (CRISPRi)', all_corr)
    approach_b(adata_r, 'condition', 'control',
               df_r, 'Replogle 2022 (CRISPRi)', REPLOGLE_MIN_CELLS)

    # =================================================================
    # NORMAN 2019 (CRISPRa)
    # =================================================================
    print("\n" + "=" * 80)
    print("NORMAN 2019 (CRISPRa)")
    print("=" * 80)
    df_n, adata_n = load_and_process(
        'Norman 2019 (CRISPRa)',
        pt.dt.norman_2019,
        pert_col='perturbation_name', ctrl_label='control',
        min_cells=NORMAN_MIN_CELLS,
    )
    approach_a(df_n, 'Norman 2019 (CRISPRa)', all_corr)
    approach_b(adata_n, 'perturbation_name', 'control',
               df_n, 'Norman 2019 (CRISPRa)', NORMAN_MIN_CELLS)

    # =================================================================
    # DIXIT 2016 (CRISPRi)  — Approach A only
    # =================================================================
    print("\n" + "=" * 80)
    print("DIXIT 2016 (CRISPRi)")
    print("=" * 80)
    df_d, _ = load_and_process(
        'Dixit 2016 (CRISPRi)',
        pt.dt.dixit_2016,
        pert_col='perturbation_name', ctrl_label='control',
        min_cells=DIXIT_MIN_CELLS,
    )
    approach_a(df_d, 'Dixit 2016 (CRISPRi)', all_corr)

    # =================================================================
    # SAVE APPROACH A SUMMARY
    # =================================================================
    if all_corr:
        corr_df = pd.DataFrame(all_corr)
        out_path = OUTPUT_DIR / "pathway_signature_correlations.csv"
        corr_df.to_csv(out_path, index=False)
        print(f"\n{'=' * 80}")
        print("APPROACH A — SUMMARY")
        print(f"{'=' * 80}")
        cols_show = ['dataset', 'pathway', 'n', 'rho_raw', 'rho_partial',
                     'effect_size', 'survives_magnitude_control']
        print(corr_df[cols_show].to_string(index=False))
        print(f"\nSaved → {out_path}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("PATHWAY ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("  Approach A:")
    print("    pathway_signature_correlations.csv")
    print("  Approach B:")
    if HAS_GSEAPY:
        print("    pathway_de_Q4_vs_Q1_replogle_2022.csv")
        print("    pathway_de_Q4_vs_Q1_norman_2019.csv")
        print("    pathway_gsea_Q4_vs_Q1_replogle_2022.csv")
        print("    pathway_gsea_Q4_vs_Q1_norman_2019.csv")
    else:
        print("    (skipped — install gseapy)")


if __name__ == '__main__':
    main()
