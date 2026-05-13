#!/usr/bin/env python3
"""
Shesha Crispr Analysis

All correlations include 10,000 bootstrap 95% CIs

ADDRESSES:
1. Theoretical derivation / null-model simulation for stability-magnitude correlation
2. k-NN matched local control centroids (neighborhood matching)
3. Whitening transform (Sigma^{-1/2}) for Mahalanobis-scaled coordinates
4. Cross-dataset calibration (control-spread normalization)
5. Mixed-effects modeling for sample size confound
6. Systematic enrichment analysis for discordant cases
7. Bootstrap CIs for all correlations
8. Ablation studies (PCA dims, random seeds, leave-one-out)
"""

import subprocess
import sys
import os

# Set thread counts BEFORE importing numpy/scanpy to ensure deterministic behavior
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scanpy as sc
import pertpy as pt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import re
import hashlib
import random
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 320
np.random.seed(SEED)
sc.settings.seed = SEED

# Bootstrap configuration
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
BOOTSTRAP_NAN_WARN_THRESHOLD = 0.05

# Output directory (parameterized for portability)
OUTPUT_DIR = Path("./shesha-crispr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ablation parameters
PCA_DIMS = [10, 20, 30, 50, 100]
RANDOM_SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

# Dataset configuration
DATASETS = {
    'Norman 2019 (CRISPRa)': pt.dt.norman_2019,
    'Adamson 2016 (CRISPRi)': pt.dt.adamson_2016_pilot,
    'Dixit 2016 (CRISPRi)': pt.dt.dixit_2016,
    'Papalexi 2021 (CRISPR)': pt.dt.papalexi_2021
}

# Manual control keywords per dataset
MANUAL_CONTROLS = {
    'Adamson 2016 (CRISPRi)': ['gal4', 'gfp', 'neg', 'scramble', 'unperturbed', 'nan'],
    'Dixit 2016 (CRISPRi)': ['nan', 'control', 'neg', 'intergenic'],
    'Papalexi 2021 (CRISPR)': ['nt', 'non-targeting', 'control']
}

# Ablation-specific datasets (subset for faster ablations)
ABLATION_DATASETS = {
    'Norman_2019': {
        'loader': pt.dt.norman_2019,
    },
    'Dixit_2016': {
        'loader': pt.dt.dixit_2016,
    },
}


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL FUNCTIONS
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=10000, ci_level=0.95, seed=42, verbose=False):
    """
    Compute bootstrap confidence interval for Spearman correlation.

    Parameters:
    -----------
    x, y : array-like
        Data vectors
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    seed : int
        Random seed for reproducibility
    verbose : bool
        If True, print warnings about dropped bootstraps

    Returns:
    --------
    dict with keys: rho, ci_low, ci_high, p, n_dropped, warning
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Point estimate
    rho, p = spearmanr(x, y)

    # Short-circuit if point estimate is NaN
    if np.isnan(rho):
        return {
            'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
            'p': np.nan, 'n_dropped': 0, 'warning': "Point estimate is NaN"
        }

    # Bootstrap
    rng = np.random.default_rng(seed=seed)
    bootstrap_rhos = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        bootstrap_rhos[i] = r

    # Filter NaNs before computing percentiles
    valid_rhos = bootstrap_rhos[~np.isnan(bootstrap_rhos)]
    n_dropped = n_bootstrap - len(valid_rhos)
    
    # Check for warnings
    ci_warning = None
    drop_rate = n_dropped / n_bootstrap
    
    if len(valid_rhos) < 100:
        ci_warning = f"Only {len(valid_rhos)} valid bootstraps; CI unreliable"
        if verbose:
            print(f"    WARNING: {ci_warning}")
        return {
            'rho': rho, 'ci_low': np.nan, 'ci_high': np.nan,
            'p': p, 'n_dropped': n_dropped, 'warning': ci_warning
        }
    
    if drop_rate > BOOTSTRAP_NAN_WARN_THRESHOLD:
        ci_warning = f"{n_dropped} bootstraps dropped ({drop_rate*100:.1f}%)"
        if verbose:
            print(f"    WARNING: {ci_warning}")

    # Percentile CI
    alpha = 1 - ci_level
    ci_low = np.percentile(valid_rhos, 100 * alpha / 2)
    ci_high = np.percentile(valid_rhos, 100 * (1 - alpha / 2))

    return {
        'rho': rho, 'ci_low': ci_low, 'ci_high': ci_high,
        'p': p, 'n_dropped': n_dropped, 'warning': ci_warning
    }


def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=10000, ci_level=0.95, seed=42):
    """
    Compute bootstrap CI for partial correlation of x and y controlling for z.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    n = len(x)

    def compute_partial_corr(x, y, z):
        Z_aug = sm.add_constant(z)
        x_resid = sm.OLS(x, Z_aug).fit().resid
        y_resid = sm.OLS(y, Z_aug).fit().resid
        return spearmanr(x_resid, y_resid)

    # Point estimate
    rho_partial, p = compute_partial_corr(x, y, z)

    # Bootstrap
    rng = np.random.default_rng(seed=seed)
    bootstrap_rhos = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = compute_partial_corr(x[idx], y[idx], z[idx])
        bootstrap_rhos[i] = r

    # Filter NaNs
    valid_rhos = bootstrap_rhos[~np.isnan(bootstrap_rhos)]
    if len(valid_rhos) < 10:
        return {
            'rho_partial': rho_partial,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'p': p
        }

    alpha = 1 - ci_level
    ci_low = np.percentile(valid_rhos, 100 * alpha / 2)
    ci_high = np.percentile(valid_rhos, 100 * (1 - alpha / 2))

    return {
        'rho_partial': rho_partial,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p': p
    }


def get_bootstrap_seed(dataset_name, ablation_type, param):
    """Generate unique bootstrap seed for each (dataset, ablation, param) combo."""
    s = f"{dataset_name}_{ablation_type}_{param}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


# =============================================================================
# SECTION 1: THEORETICAL NULL MODEL SIMULATION
# =============================================================================

def theoretical_null_model_simulation(n_per_condition=25, n_samples=100, dim=50):
    """Simulate the expected Shesha-magnitude relationship under a null Gaussian model."""
    print("=" * 80)
    print("THEORETICAL NULL MODEL: Gaussian Shift + Isotropic Noise")
    print("=" * 80)

    results = []
    magnitudes = np.linspace(0.5, 10, 20)
    noise_levels = [0.5, 1.0, 2.0, 3.0]

    total_sims = n_per_condition * len(magnitudes) * len(noise_levels)
    print(f"Running {total_sims} simulations ({n_per_condition} per condition)")

    for sigma in noise_levels:
        for mag in magnitudes:
            for _ in range(n_per_condition):
                control_centroid = np.zeros(dim)
                mu_direction = np.random.randn(dim)
                mu_direction = mu_direction / np.linalg.norm(mu_direction)
                mu = mu_direction * mag

                epsilon = np.random.randn(n_samples, dim) * sigma
                X_pert = control_centroid + mu + epsilon

                shift_vectors = X_pert - control_centroid
                mean_shift = np.mean(shift_vectors, axis=0)
                mean_magnitude = np.linalg.norm(mean_shift)

                if mean_magnitude < 1e-6:
                    continue

                norms = np.linalg.norm(shift_vectors, axis=1)
                valid_idx = norms > 1e-6
                unit_mean = mean_shift / mean_magnitude
                cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
                stability = np.mean(cosine_sims)

                snr = mag / sigma
                spread = np.mean(np.linalg.norm(X_pert - np.mean(X_pert, axis=0), axis=1))

                results.append({
                    'true_magnitude': mag,
                    'observed_magnitude': mean_magnitude,
                    'stability': stability,
                    'snr': snr,
                    'sigma': sigma,
                    'spread': spread
                })

    df_null = pd.DataFrame(results)

    print("\n--- Correlations with Stability (with bootstrap CIs) ---")

    mag_ci = bootstrap_spearman_ci(df_null['observed_magnitude'], df_null['stability'],
                                    n_bootstrap=N_BOOTSTRAP, seed=SEED)
    print(f"rho(magnitude, stability) = {mag_ci['rho']:.3f} [{mag_ci['ci_low']:.3f}, {mag_ci['ci_high']:.3f}]")

    snr_ci = bootstrap_spearman_ci(df_null['snr'], df_null['stability'],
                                    n_bootstrap=N_BOOTSTRAP, seed=SEED + 1)
    print(f"rho(SNR, stability)       = {snr_ci['rho']:.3f} [{snr_ci['ci_low']:.3f}, {snr_ci['ci_high']:.3f}]")

    partial_ci = bootstrap_partial_correlation_ci(
        df_null['observed_magnitude'], df_null['stability'], df_null['snr'],
        n_bootstrap=N_BOOTSTRAP, seed=SEED + 2
    )
    print(f"\nPartial rho(magnitude, stability | SNR) = {partial_ci['rho_partial']:.3f} [{partial_ci['ci_low']:.3f}, {partial_ci['ci_high']:.3f}]")

    return df_null


# =============================================================================
# SECTION 2: ENHANCED SHESHA METRICS
# =============================================================================

def calculate_metrics_enhanced(control_matrix, pert_matrix, use_whitening=False,
                                control_cov=None, regularization=1e-6):
    """Enhanced Shesha calculation with optional whitening."""
    control_centroid = np.mean(control_matrix, axis=0)

    if use_whitening:
        if control_cov is None:
            control_cov = np.cov(control_matrix.T)

        control_cov_reg = control_cov + regularization * np.eye(control_cov.shape[0])

        try:
            eigvals, eigvecs = np.linalg.eigh(control_cov_reg)
            eigvals = np.maximum(eigvals, regularization)
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            control_centroid_w = W @ control_centroid
            pert_matrix_w = (W @ pert_matrix.T).T

            shift_vectors = pert_matrix_w - control_centroid_w
        except np.linalg.LinAlgError:
            shift_vectors = pert_matrix - control_centroid
    else:
        shift_vectors = pert_matrix - control_centroid

    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_shift)

    if mean_magnitude < 1e-6:
        return {'stability': 0.0, 'magnitude': 0.0, 'spread': 0.0, 'snr': 0.0}

    norms = np.linalg.norm(shift_vectors, axis=1)
    valid_idx = norms > 1e-6
    if np.sum(valid_idx) < 5:
        return {'stability': 0.0, 'magnitude': 0.0, 'spread': 0.0, 'snr': 0.0}

    unit_mean = mean_shift / mean_magnitude
    cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
    stability = np.mean(cosine_sims)

    pert_centroid = np.mean(shift_vectors, axis=0)
    internal_spread = np.mean(np.linalg.norm(shift_vectors - pert_centroid, axis=1))
    snr = mean_magnitude / (internal_spread + 1e-6)

    return {
        'stability': stability,
        'magnitude': mean_magnitude,
        'spread': internal_spread,
        'snr': snr
    }


# =============================================================================
# SECTION 3: K-NN MATCHED CONTROL CENTROIDS
# =============================================================================

def calculate_metrics_knn_control(control_matrix, pert_matrix, k=50):
    """Calculate Shesha using k-nearest neighbor matched controls."""
    if control_matrix.shape[0] < k:
        k = control_matrix.shape[0]

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(control_matrix)

    _, indices = nn.kneighbors(pert_matrix)

    shift_vectors = []
    for i, idx in enumerate(indices):
        local_ctrl_centroid = np.mean(control_matrix[idx], axis=0)
        shift_vectors.append(pert_matrix[i] - local_ctrl_centroid)

    shift_vectors = np.array(shift_vectors)

    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_shift)

    if mean_magnitude < 1e-6:
        return {'stability': 0.0, 'magnitude': 0.0, 'spread': 0.0, 'snr': 0.0}

    norms = np.linalg.norm(shift_vectors, axis=1)
    valid_idx = norms > 1e-6
    if np.sum(valid_idx) < 5:
        return {'stability': 0.0, 'magnitude': 0.0, 'spread': 0.0, 'snr': 0.0}

    unit_mean = mean_shift / mean_magnitude
    cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
    stability = np.mean(cosine_sims)

    pert_centroid = np.mean(shift_vectors, axis=0)
    internal_spread = np.mean(np.linalg.norm(shift_vectors - pert_centroid, axis=1))
    snr = mean_magnitude / (internal_spread + 1e-6)

    return {
        'stability': stability,
        'magnitude': mean_magnitude,
        'spread': internal_spread,
        'snr': snr
    }


# =============================================================================
# SECTION 4: CROSS-DATASET CALIBRATION
# =============================================================================

def run_cross_dataset_calibrated_analysis(all_results_df, control_scales):
    """Perform cross-dataset analysis with calibration and bootstrap CIs."""
    df = all_results_df.copy()

    df['calibrated_magnitude'] = df.apply(
        lambda row: row['magnitude'] / (control_scales.get(row['dataset'], 1) + 1e-6),
        axis=1
    )

    df['stability_z'] = df.groupby('dataset')['stability'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    df['magnitude_z'] = df.groupby('dataset')['magnitude'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    print("\n--- CALIBRATED CROSS-DATASET ANALYSIS (Euclidean method, with bootstrap CIs) ---")

    z_ci = bootstrap_spearman_ci(df['magnitude_z'], df['stability_z'],
                                  n_bootstrap=N_BOOTSTRAP, seed=SEED + 100)
    print(f"Pooled rho(magnitude_z, stability_z) = {z_ci['rho']:.3f} [{z_ci['ci_low']:.3f}, {z_ci['ci_high']:.3f}]")

    calib_ci = bootstrap_spearman_ci(df['calibrated_magnitude'], df['stability'],
                                      n_bootstrap=N_BOOTSTRAP, seed=SEED + 101)
    print(f"Pooled rho(calibrated_magnitude, stability) = {calib_ci['rho']:.3f} [{calib_ci['ci_low']:.3f}, {calib_ci['ci_high']:.3f}]")

    return df, z_ci, calib_ci


# =============================================================================
# SECTION 5: MIXED-EFFECTS MODEL
# =============================================================================

def mixed_effects_analysis(df):
    """Mixed-effects model with proper CI reporting and fallback."""
    print("\n" + "=" * 80)
    print("MIXED-EFFECTS MODEL ANALYSIS")
    print("=" * 80)

    df_model = df.copy()
    df_model['log_n_cells'] = np.log10(df_model['n_cells'])

    for col in ['magnitude', 'spread', 'log_n_cells']:
        df_model[col + '_z'] = (df_model[col] - df_model[col].mean()) / df_model[col].std()

    model = MixedLM.from_formula(
        'stability ~ magnitude_z + spread_z + log_n_cells_z',
        groups='dataset',
        data=df_model
    )

    result = None
    for method in ['lbfgs', 'powell', 'cg', 'bfgs']:
        try:
            result = model.fit(method=method, reml=False)
            print(f"\nMixed-Effects Model Results (optimizer: {method}, ML estimation):")
            print("-" * 40)
            print(result.summary())

            print("\n--- KEY FINDINGS ---")
            coefs = result.fe_params
            pvals = result.pvalues
            conf_int = result.conf_int()

            print(f"Magnitude effect:    beta = {coefs['magnitude_z']:.3f} [{conf_int.loc['magnitude_z', 0]:.3f}, {conf_int.loc['magnitude_z', 1]:.3f}], p = {pvals['magnitude_z']:.2e}")
            print(f"Spread effect:       beta = {coefs['spread_z']:.3f} [{conf_int.loc['spread_z', 0]:.3f}, {conf_int.loc['spread_z', 1]:.3f}], p = {pvals['spread_z']:.2e}")
            print(f"Sample size effect:  beta = {coefs['log_n_cells_z']:.3f} [{conf_int.loc['log_n_cells_z', 0]:.3f}, {conf_int.loc['log_n_cells_z', 1]:.3f}], p = {pvals['log_n_cells_z']:.2e}")

            return result
        except Exception as e:
            print(f"  Optimizer '{method}' failed: {e}")
            continue

    # Fallback
    print("\n!!! Mixed-effects model failed. Falling back to partial correlation.")
    
    def compute_fallback_partial(mag, stab, covariates):
        Z_aug = sm.add_constant(covariates)
        mag_resid = sm.OLS(mag, Z_aug).fit().resid
        stab_resid = sm.OLS(stab, Z_aug).fit().resid
        return spearmanr(mag_resid, stab_resid)

    Z_controls = df_model[['spread_z', 'log_n_cells_z']].values
    rho_partial, p_partial = compute_fallback_partial(
        df_model['magnitude_z'].values,
        df_model['stability'].values,
        Z_controls
    )

    if np.isnan(rho_partial):
        print("Warning: Point estimate is NaN.")
        return None

    # Bootstrap CI for fallback
    n = len(df_model)
    rng = np.random.default_rng(seed=SEED + 9999)
    bootstrap_rhos = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = compute_fallback_partial(
            df_model['magnitude_z'].values[idx],
            df_model['stability'].values[idx],
            Z_controls[idx]
        )
        bootstrap_rhos[i] = r

    valid_rhos = bootstrap_rhos[~np.isnan(bootstrap_rhos)]
    if len(valid_rhos) >= 10:
        alpha = 1 - CI_LEVEL
        ci_low = np.percentile(valid_rhos, 100 * alpha / 2)
        ci_high = np.percentile(valid_rhos, 100 * (1 - alpha / 2))
    else:
        ci_low, ci_high = np.nan, np.nan

    p_str = f"{p_partial:.2e}" if p_partial < 0.001 else f"{p_partial:.3f}"
    print(f"Partial rho(magnitude, stability | spread, n_cells) = {rho_partial:.3f} [{ci_low:.3f}, {ci_high:.3f}], p = {p_str}")

    return None


# =============================================================================
# SECTION 6: SYSTEMATIC DISCORDANT ANALYSIS
# =============================================================================

def systematic_discordant_analysis(df, gene_annotations=None):
    """Systematic analysis of discordant cases."""
    print("\n" + "=" * 80)
    print("SYSTEMATIC DISCORDANT CASE ANALYSIS")
    print("=" * 80)

    results = []

    for ds_name in df['dataset'].unique():
        subset = df[df['dataset'] == ds_name].copy()

        subset['magnitude_z'] = (subset['magnitude'] - subset['magnitude'].mean()) / subset['magnitude'].std()
        subset['stability_z'] = (subset['stability'] - subset['stability'].mean()) / subset['stability'].std()
        subset['discordance'] = subset['magnitude_z'] - subset['stability_z']

        subset['discordance_quartile'] = pd.qcut(subset['discordance'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        print(f"\n--- {ds_name} ---")
        print(f"N perturbations: {len(subset)}")

        quartile_stats = subset.groupby('discordance_quartile').agg({
            'stability': ['mean', 'std'],
            'magnitude': ['mean', 'std'],
            'n_cells': 'median',
            'perturbation': 'count'
        }).round(3)
        print("\nQuartile Statistics:")
        print(quartile_stats)

        print("\nTop 5 High Magnitude / Low Stability (potential pleiotropic):")
        high_disc = subset.nlargest(5, 'discordance')[['perturbation', 'magnitude', 'stability', 'discordance']]
        print(high_disc.to_string(index=False))

        print("\nTop 5 Low Magnitude / High Stability (potential lineage-specific):")
        low_disc = subset.nsmallest(5, 'discordance')[['perturbation', 'magnitude', 'stability', 'discordance']]
        print(low_disc.to_string(index=False))

        results.append({
            'dataset': ds_name,
            'data': subset
        })

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name, loader_func, n_pcs=50, seed=320):
    """
    Load and preprocess dataset with special handling for Papalexi 2021.
    
    The Papalexi fix:
    - Papalexi 2021 stores perturbation metadata at MuData level, not in RNA modality
    - We must copy 'gene_target' from mdata.obs to adata.obs
    - 'gene_target' groups all NT guides into 'NT' (2,386 cells)
    """
    random.seed(seed)
    np.random.seed(seed)
    sc.settings.seed = seed
    
    print(f"\n>>> LOADING: {dataset_name}...")
    
    try:
        raw_data = loader_func()
    except Exception as e:
        print(f"    ! Load Failed: {e}")
        return None, None, None

    if raw_data is None:
        print(f"    ! Skipped: Dataset is None.")
        return None, None, None

    # Special handling for Papalexi 2021 (MuData with metadata at top level)
    # Use substring match to avoid silent failures from whitespace/typos
    if 'papalexi' in dataset_name.lower():
        if type(raw_data).__name__ == 'MuData':
            print(f"    - Detected MuData object. Applying Papalexi fix...")
            mdata = raw_data
            
            # Extract RNA modality
            if 'rna' in mdata.mod:
                adata = mdata.mod['rna'].copy()
            else:
                print(f"    ! No 'rna' modality found in Papalexi MuData")
                return None, None, None
            
            # Copy gene_target from MuData level to RNA obs
            if 'gene_target' in mdata.obs.columns:
                adata.obs['gene_target'] = mdata.obs['gene_target'].values
                print(f"    * Synced 'gene_target' from MuData.obs to RNA.obs")
                print(f"    * Control 'NT' has {(adata.obs['gene_target'] == 'NT').sum()} cells")
            else:
                print(f"    ! WARNING: 'gene_target' not found in MuData.obs")
                return None, None, None
            
            # Force perturbation column and control
            pert_col = 'gene_target'
            ctrl_label = 'NT'
            
        else:
            print(f"    ! Expected MuData for Papalexi, got {type(raw_data)}")
            return None, None, None
    else:
        # Standard handling for other datasets
        adata = raw_data
        
        # Handle MuData format
        if type(adata).__name__ == 'MuData':
            print(f"    - Detected MuData object. Extracting 'rna' modality...")
            try:
                if 'rna' in adata.mod:
                    adata = adata.mod['rna']
                elif 'gex' in adata.mod:
                    adata = adata.mod['gex']
                else:
                    key = list(adata.mod.keys())[0]
                    adata = adata.mod[key]
            except Exception as e:
                print(f"    ! Failed to extract modality: {e}")
                return None, None, None

        if not isinstance(adata, sc.AnnData):
            if isinstance(adata, dict) and len(adata) > 0:
                adata = list(adata.values())[0]
            else:
                return None, None, None

        # Find perturbation column
        possible_cols = ['perturbation_name', 'perturbation', 'gene', 'target', 'guide_id', 'sgRNA', 'gene_target']
        pert_col = next((c for c in possible_cols if c in adata.obs.columns), None)

        if not pert_col:
            pert_col = next((c for c in adata.obs.columns if 'pert' in c.lower() or 'guide' in c.lower() or 'gene' in c.lower() or 'target' in c.lower()), None)

        if not pert_col:
            print(f"    ! No perturbation column found")
            return None, None, None

        # Normalize labels
        adata.obs[pert_col] = adata.obs[pert_col].astype(str).replace('nan', 'NaN_Control')

        # Find control
        labels = adata.obs[pert_col].unique()
        exact_match_ctrls = ['control', 'ctrl', 'non-targeting', 'scrambled', 'nt', 'gal4', 'gfp', 'nan_control']
        substring_match_ctrls = ['control', 'ctrl', 'non-targeting', 'scrambled', 'gal4', 'gfp', 'nan_control']
        
        if dataset_name in MANUAL_CONTROLS:
            manual = MANUAL_CONTROLS[dataset_name]
            exact_match_ctrls = manual + exact_match_ctrls
            substring_match_ctrls = [c for c in manual if len(c) >= 3] + substring_match_ctrls

        ctrl_label = next((x for x in labels if x.lower() in [c.lower() for c in exact_match_ctrls]), None)

        if ctrl_label is None:
            ctrl_label = next((x for x in labels if any(c in x.lower() for c in substring_match_ctrls)), None)

        if not ctrl_label:
            ctrl_label = adata.obs[pert_col].value_counts().idxmax()
            print(f"    ! WARNING: Fell back to most frequent label as control")

    if adata.n_obs < 10 or adata.n_vars < 10:
        print(f"    ! Dataset too small")
        return None, None, None

    # Preprocessing
    try:
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=seed)
    except Exception as e:
        print(f"    ! Preprocessing Failed: {e}")
        return None, None, None

    print(f"    - Control: '{ctrl_label}'")
    n_ctrl = (adata.obs[pert_col] == ctrl_label).sum()
    print(f"    - Control cells: {n_ctrl}")

    return adata, pert_col, ctrl_label


# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

def run_main_analysis():
    """Main analysis loop with all enhancements and bootstrap CIs."""

    all_results = []
    all_results_whitened = []
    all_results_knn = []
    control_scales = {}

    print("\n" + "=" * 80)
    print("SHESHA CRISPR ANALYSIS - UNIFIED WITH BOOTSTRAP CIs")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # Run null model simulation
    print("\n>>> Running null model simulation...")
    df_null = theoretical_null_model_simulation(n_per_condition=25, n_samples=100, dim=50)
    df_null.to_csv(os.path.join(OUTPUT_DIR, "null_model_simulation.csv"), index=False)
    print(f"    Saved null model results to {OUTPUT_DIR}/null_model_simulation.csv")

    print(f"\n>>> Starting analysis on {len(DATASETS)} datasets...")

    for dataset_name, loader_func in DATASETS.items():
        adata, pert_col, ctrl_label = load_dataset(dataset_name, loader_func)
        
        if adata is None:
            continue

        ctrl_mask = adata.obs[pert_col] == ctrl_label
        X_ctrl = adata.obsm['X_pca'][ctrl_mask]

        if X_ctrl.shape[0] < 5:
            print(f"    ! Insufficient control cells")
            continue

        ctrl_cov = np.cov(X_ctrl.T)
        ctrl_centroid = np.mean(X_ctrl, axis=0)
        control_scale = np.mean(np.linalg.norm(X_ctrl - ctrl_centroid, axis=1))
        control_scales[dataset_name] = control_scale

        unique_perts = adata.obs[pert_col].unique()

        for pert in tqdm(unique_perts, desc=f"    - Processing"):
            if pert == ctrl_label:
                continue

            pert_mask = adata.obs[pert_col] == pert
            X_pert = adata.obsm['X_pca'][pert_mask]

            n_cells = X_pert.shape[0]
            if n_cells < 10:
                continue

            # Method 1: Standard Euclidean
            metrics = calculate_metrics_enhanced(X_ctrl, X_pert, use_whitening=False)

            if metrics['magnitude'] > 0:
                all_results.append({
                    'dataset': dataset_name,
                    'perturbation': str(pert),
                    'stability': metrics['stability'],
                    'magnitude': metrics['magnitude'],
                    'spread': metrics['spread'],
                    'snr': metrics['snr'],
                    'n_cells': n_cells,
                    'method': 'euclidean'
                })

            # Method 2: Whitened
            metrics_w = calculate_metrics_enhanced(X_ctrl, X_pert, use_whitening=True, control_cov=ctrl_cov)

            if metrics_w['magnitude'] > 0:
                all_results_whitened.append({
                    'dataset': dataset_name,
                    'perturbation': str(pert),
                    'stability': metrics_w['stability'],
                    'magnitude_mahalanobis': metrics_w['magnitude'],
                    'spread': metrics_w['spread'],
                    'snr': metrics_w['snr'],
                    'n_cells': n_cells,
                    'method': 'whitened'
                })

            # Method 3: k-NN matched
            metrics_knn = calculate_metrics_knn_control(X_ctrl, X_pert, k=50)

            if metrics_knn['magnitude'] > 0:
                all_results_knn.append({
                    'dataset': dataset_name,
                    'perturbation': str(pert),
                    'stability': metrics_knn['stability'],
                    'magnitude': metrics_knn['magnitude'],
                    'spread': metrics_knn['spread'],
                    'snr': metrics_knn['snr'],
                    'n_cells': n_cells,
                    'method': 'knn'
                })

    if not all_results:
        print("!!! CRITICAL: No results generated.")
        return None, None, None

    df = pd.DataFrame(all_results)
    df_whitened = pd.DataFrame(all_results_whitened) if all_results_whitened else None
    df_knn = pd.DataFrame(all_results_knn) if all_results_knn else None

    # Save results
    df.to_csv(os.path.join(OUTPUT_DIR, "shesha_crispr_results_euclidean.csv"), index=False)
    if df_whitened is not None:
        df_whitened.to_csv(os.path.join(OUTPUT_DIR, "shesha_crispr_results_whitened.csv"), index=False)
    if df_knn is not None:
        df_knn.to_csv(os.path.join(OUTPUT_DIR, "shesha_crispr_results_knn.csv"), index=False)

    # ==========================================================================
    # COMPARATIVE ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS WITH BOOTSTRAP CIs")
    print("=" * 80)

    correlation_results = []
    seed_counter = 1000

    for ds_name in df['dataset'].unique():
        print(f"\n--- {ds_name} ---")

        subset_euc = df[df['dataset'] == ds_name]
        euc_ci = bootstrap_spearman_ci(subset_euc['magnitude'], subset_euc['stability'],
                                        n_bootstrap=N_BOOTSTRAP, seed=seed_counter)
        seed_counter += 1
        print(f"  Euclidean:  rho = {euc_ci['rho']:.3f} [{euc_ci['ci_low']:.3f}, {euc_ci['ci_high']:.3f}], n = {len(subset_euc)}")

        correlation_results.append({
            'dataset': ds_name,
            'method': 'Euclidean',
            'n': len(subset_euc),
            'rho': euc_ci['rho'],
            'ci_low': euc_ci['ci_low'],
            'ci_high': euc_ci['ci_high'],
            'p': euc_ci['p']
        })

        if df_whitened is not None:
            subset_wht = df_whitened[df_whitened['dataset'] == ds_name]
            if len(subset_wht) > 5:
                wht_ci = bootstrap_spearman_ci(subset_wht['magnitude_mahalanobis'], subset_wht['stability'],
                                               n_bootstrap=N_BOOTSTRAP, seed=seed_counter)
                seed_counter += 1
                print(f"  Whitened:   rho = {wht_ci['rho']:.3f} [{wht_ci['ci_low']:.3f}, {wht_ci['ci_high']:.3f}], n = {len(subset_wht)}")

                correlation_results.append({
                    'dataset': ds_name,
                    'method': 'Whitened',
                    'n': len(subset_wht),
                    'rho': wht_ci['rho'],
                    'ci_low': wht_ci['ci_low'],
                    'ci_high': wht_ci['ci_high'],
                    'p': wht_ci['p']
                })

        if df_knn is not None:
            subset_knn = df_knn[df_knn['dataset'] == ds_name]
            if len(subset_knn) > 5:
                knn_ci = bootstrap_spearman_ci(subset_knn['magnitude'], subset_knn['stability'],
                                               n_bootstrap=N_BOOTSTRAP, seed=seed_counter)
                seed_counter += 1
                print(f"  k-NN:       rho = {knn_ci['rho']:.3f} [{knn_ci['ci_low']:.3f}, {knn_ci['ci_high']:.3f}], n = {len(subset_knn)}")

                correlation_results.append({
                    'dataset': ds_name,
                    'method': 'k-NN',
                    'n': len(subset_knn),
                    'rho': knn_ci['rho'],
                    'ci_low': knn_ci['ci_low'],
                    'ci_high': knn_ci['ci_high'],
                    'p': knn_ci['p']
                })

    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "crispr_correlations_with_ci.csv"), index=False)

    # ==========================================================================
    # PARTIAL CORRELATION ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PARTIAL CORRELATION ANALYSIS (controlling for SNR)")
    print("=" * 80)

    partial_results = []
    seed_counter = 2000

    for ds_name in df['dataset'].unique():
        subset = df[df['dataset'] == ds_name]
        if len(subset) > 20:
            partial_ci = bootstrap_partial_correlation_ci(
                subset['magnitude'].values,
                subset['stability'].values,
                subset['snr'].values,
                n_bootstrap=N_BOOTSTRAP,
                seed=seed_counter
            )
            seed_counter += 1
            print(f"{ds_name}: rho_partial = {partial_ci['rho_partial']:.3f} [{partial_ci['ci_low']:.3f}, {partial_ci['ci_high']:.3f}]")

            partial_results.append({
                'dataset': ds_name,
                'rho_partial': partial_ci['rho_partial'],
                'ci_low': partial_ci['ci_low'],
                'ci_high': partial_ci['ci_high'],
                'p': partial_ci['p']
            })

    print("\n--- Pooled ---")
    pooled_partial = bootstrap_partial_correlation_ci(
        df['magnitude'].values,
        df['stability'].values,
        df['snr'].values,
        n_bootstrap=N_BOOTSTRAP,
        seed=seed_counter
    )
    print(f"Pooled: rho_partial = {pooled_partial['rho_partial']:.3f} [{pooled_partial['ci_low']:.3f}, {pooled_partial['ci_high']:.3f}]")

    partial_results.append({
        'dataset': 'Pooled',
        'rho_partial': pooled_partial['rho_partial'],
        'ci_low': pooled_partial['ci_low'],
        'ci_high': pooled_partial['ci_high'],
        'p': pooled_partial['p']
    })

    partial_df = pd.DataFrame(partial_results)
    partial_df.to_csv(os.path.join(OUTPUT_DIR, "crispr_partial_correlations_with_ci.csv"), index=False)

    # ==========================================================================
    # OTHER ANALYSES
    # ==========================================================================
    mixed_result = mixed_effects_analysis(df)
    df_calibrated, z_ci, calib_ci = run_cross_dataset_calibrated_analysis(df, control_scales)
    discordant_results = systematic_discordant_analysis(df)
    return df, df_whitened, df_knn


# =============================================================================
# ABLATION STUDIES
# =============================================================================

def run_leave_one_out_analysis(df):
    """Compute leave-one-perturbation-out influence on correlation."""
    if len(df) < 10:
        return None
    
    x = df['magnitude'].values
    y = df['stability'].values
    perts = df['perturbation'].values
    
    full_rho, _ = spearmanr(x, y)
    
    loo_results = []
    for i in range(len(df)):
        mask = np.ones(len(df), dtype=bool)
        mask[i] = False
        rho_without, _ = spearmanr(x[mask], y[mask])
        loo_results.append({
            'perturbation': perts[i],
            'rho_without': rho_without,
            'delta': full_rho - rho_without
        })
    
    loo_df = pd.DataFrame(loo_results)
    
    most_helpful_idx = loo_df['delta'].idxmax()
    most_helpful = loo_df.loc[most_helpful_idx]
    
    most_harmful_idx = loo_df['delta'].idxmin()
    most_harmful = loo_df.loc[most_harmful_idx]
    
    return {
        'full_rho': full_rho,
        'loo_df': loo_df,
        'min_rho': loo_df['rho_without'].min(),
        'max_rho': loo_df['rho_without'].max(),
        'most_helpful_pert': most_helpful['perturbation'],
        'most_helpful_delta': most_helpful['delta'],
        'most_harmful_pert': most_harmful['perturbation'],
        'most_harmful_delta': most_harmful['delta'],
        'n_perturbations': len(df)
    }


def load_and_preprocess_ablation(loader_func, n_pcs=30, seed=42):
    """Load dataset for ablation studies."""
    random.seed(seed)
    np.random.seed(seed)
    sc.settings.seed = seed
    sc.settings.verbosity = 0

    adata = loader_func()

    if type(adata).__name__ == 'MuData':
        if 'rna' in adata.mod:
            adata = adata.mod['rna']
        elif 'gex' in adata.mod:
            adata = adata.mod['gex']
        else:
            adata = adata.mod[list(adata.mod.keys())[0]]

    if isinstance(adata, dict):
        adata = list(adata.values())[0]

    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=seed)

    return adata


def find_pert_column_ablation(adata):
    """Find perturbation column for ablations."""
    possible_cols = ['perturbation_name', 'perturbation', 'gene', 'target', 'guide_id']
    for col in possible_cols:
        if col in adata.obs.columns:
            return col
    for col in adata.obs.columns:
        if 'pert' in col.lower() or 'gene' in col.lower():
            return col
    return None


def find_control_ablation(adata, pert_col, dataset_name):
    """Find control for ablation datasets."""
    labels = adata.obs[pert_col].astype(str).unique()
    
    exact_matches = ['control', 'ctrl', 'non-targeting', 'nan', 'intergenic']
    for exact in exact_matches:
        for label in labels:
            if label.lower() == exact.lower():
                return label
    
    token_patterns = {
        'neg': re.compile(r'(^|[_\-\s])neg($|[_\-\s])', re.IGNORECASE),
        'nt':  re.compile(r'(^|[_\-\s])nt($|[_\-\s])', re.IGNORECASE),
    }
    for token, pattern in token_patterns.items():
        for label in labels:
            if pattern.search(label):
                return label
    
    substring_keywords = ['control', 'ctrl', 'non-targeting', 'scramble', 'intergenic']
    for label in labels:
        label_lower = label.lower()
        if any(kw in label_lower for kw in substring_keywords):
            return label

    return adata.obs[pert_col].value_counts().idxmax()


def compute_all_metrics_ablation(adata, pert_col, ctrl_label):
    """Compute metrics for ablation studies."""
    adata.obs[pert_col] = adata.obs[pert_col].astype(str).replace('nan', 'NaN_Control')
    
    ctrl_mask = adata.obs[pert_col] == ctrl_label
    X_ctrl = adata.obsm['X_pca'][ctrl_mask]

    if X_ctrl.shape[0] < 5:
        return pd.DataFrame()

    results = []
    unique_perts = adata.obs[pert_col].unique()

    for pert in unique_perts:
        if pert == ctrl_label:
            continue

        pert_mask = adata.obs[pert_col] == pert
        X_pert = adata.obsm['X_pca'][pert_mask]

        n_cells = X_pert.shape[0]
        if n_cells < 10:
            continue

        control_centroid = np.mean(X_ctrl, axis=0)
        shift_vectors = X_pert - control_centroid
        mean_shift = np.mean(shift_vectors, axis=0)
        mean_magnitude = np.linalg.norm(mean_shift)

        if mean_magnitude < 1e-6:
            continue

        norms = np.linalg.norm(shift_vectors, axis=1)
        valid_idx = norms > 1e-6
        if np.sum(valid_idx) < 5:
            continue

        unit_mean = mean_shift / mean_magnitude
        cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
        stability = np.mean(cosine_sims)

        results.append({
            'perturbation': str(pert),
            'stability': stability,
            'magnitude': mean_magnitude,
            'n_cells': n_cells
        })

    return pd.DataFrame(results)


def run_pca_ablation(dataset_name, dataset_info):
    """Run PCA dimensionality ablation."""
    print(f"\n{'='*70}")
    print(f"PCA DIMENSIONALITY ABLATION: {dataset_name}")
    print(f"{'='*70}")

    results = []
    stability_vectors = {}
    magnitude_vectors = {}

    for n_pcs in PCA_DIMS:
        print(f"  Testing n_pcs = {n_pcs}...", end=" ")

        try:
            adata = load_and_preprocess_ablation(dataset_info['loader'], n_pcs=n_pcs, seed=42)
            pert_col = find_pert_column_ablation(adata)
            if pert_col is None:
                raise ValueError("No perturbation column found.")
            
            ctrl_label = find_control_ablation(adata, pert_col, dataset_name)
            df = compute_all_metrics_ablation(adata, pert_col, ctrl_label)

            if len(df) >= 10:
                stability_vectors[n_pcs] = df.set_index('perturbation')['stability']
                magnitude_vectors[n_pcs] = df.set_index('perturbation')['magnitude']
                
                boot_seed = get_bootstrap_seed(dataset_name, 'pca', n_pcs)
                ci_result = bootstrap_spearman_ci(
                    df['magnitude'].values, df['stability'].values,
                    n_bootstrap=N_BOOTSTRAP, seed=boot_seed
                )

                results.append({
                    'n_pcs': n_pcs,
                    'rho': ci_result['rho'],
                    'ci_low': ci_result['ci_low'],
                    'ci_high': ci_result['ci_high'],
                    'p': ci_result['p'],
                    'n_perturbations': len(df),
                    'dataset': dataset_name
                })
                print(f"rho = {ci_result['rho']:.3f} [{ci_result['ci_low']:.3f}, {ci_result['ci_high']:.3f}], n = {len(df)}")
            else:
                print("insufficient data")

        except Exception as e:
            print(f"Error: {e}")

    return pd.DataFrame(results)


def run_seed_ablation(dataset_name, dataset_info):
    """Run random seed ablation."""
    print(f"\n{'='*70}")
    print(f"RANDOM SEED ABLATION: {dataset_name}")
    print(f"{'='*70}")

    results = []
    all_dfs = []

    for seed in RANDOM_SEEDS:
        print(f"  Testing seed = {seed}...", end=" ")

        try:
            adata = load_and_preprocess_ablation(dataset_info['loader'], n_pcs=30, seed=seed)
            pert_col = find_pert_column_ablation(adata)
            if pert_col is None:
                raise ValueError("No perturbation column found.")
            
            ctrl_label = find_control_ablation(adata, pert_col, dataset_name)
            df = compute_all_metrics_ablation(adata, pert_col, ctrl_label)

            if len(df) >= 10:
                df['seed'] = seed
                all_dfs.append((seed, df))

                boot_seed = get_bootstrap_seed(dataset_name, 'seed', seed)
                ci_result = bootstrap_spearman_ci(
                    df['magnitude'].values, df['stability'].values,
                    n_bootstrap=N_BOOTSTRAP, seed=boot_seed
                )

                results.append({
                    'seed': seed,
                    'rho': ci_result['rho'],
                    'ci_low': ci_result['ci_low'],
                    'ci_high': ci_result['ci_high'],
                    'p': ci_result['p'],
                    'n_perturbations': len(df),
                    'dataset': dataset_name
                })
                print(f"rho = {ci_result['rho']:.3f} [{ci_result['ci_low']:.3f}, {ci_result['ci_high']:.3f}]")
            else:
                print("insufficient data")

        except Exception as e:
            print(f"Error: {e}")

    results_df = pd.DataFrame(results)

    # Cross-seed correlation
    if len(all_dfs) >= 2:
        seed0, df0 = all_dfs[0]
        merged = df0[['perturbation', 'stability']].rename(columns={'stability': f'stab_{seed0}'})
        for seed_i, dfi in all_dfs[1:]:
            merged = merged.merge(
                dfi[['perturbation', 'stability']].rename(columns={'stability': f'stab_{seed_i}'}),
                on='perturbation', how='inner'
            )

        if len(merged) > 5:
            stab_cols = [c for c in merged.columns if c.startswith('stab_')]
            corrs = []
            for i in range(len(stab_cols)):
                for j in range(i+1, len(stab_cols)):
                    r, _ = spearmanr(merged[stab_cols[i]], merged[stab_cols[j]])
                    corrs.append(r)

            mean_corr = np.mean(corrs)
            print(f"\n  Cross-seed stability correlation: {mean_corr:.3f}")
            results_df['cross_seed_corr'] = mean_corr

    return results_df


def run_loo_ablation(dataset_name, dataset_info):
    """Run leave-one-out ablation."""
    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-OUT ANALYSIS: {dataset_name}")
    print(f"{'='*70}")

    try:
        adata = load_and_preprocess_ablation(dataset_info['loader'], n_pcs=30, seed=42)
        pert_col = find_pert_column_ablation(adata)
        if pert_col is None:
            raise ValueError("No perturbation column found.")
        
        ctrl_label = find_control_ablation(adata, pert_col, dataset_name)
        df = compute_all_metrics_ablation(adata, pert_col, ctrl_label)

        if len(df) < 10:
            print("  Insufficient perturbations")
            return None

        loo_results = run_leave_one_out_analysis(df)
        
        print(f"  Full rho: {loo_results['full_rho']:.3f}")
        print(f"  LOO range: [{loo_results['min_rho']:.3f}, {loo_results['max_rho']:.3f}]")
        print(f"  Most helpful: {loo_results['most_helpful_pert']} (delta = +{loo_results['most_helpful_delta']:.4f})")
        print(f"  Most harmful: {loo_results['most_harmful_pert']} (delta = {loo_results['most_harmful_delta']:.4f})")
        
        loo_results['loo_df']['dataset'] = dataset_name
        loo_results['loo_df'].to_csv(os.path.join(OUTPUT_DIR, f'ablation_loo_{dataset_name}.csv'), index=False)
        
        return {
            'dataset': dataset_name,
            'full_rho': loo_results['full_rho'],
            'loo_min_rho': loo_results['min_rho'],
            'loo_max_rho': loo_results['max_rho'],
            'most_helpful_pert': loo_results['most_helpful_pert'],
            'most_helpful_delta': loo_results['most_helpful_delta'],
            'most_harmful_pert': loo_results['most_harmful_pert'],
            'most_harmful_delta': loo_results['most_harmful_delta'],
            'n_perturbations': loo_results['n_perturbations']
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_all_ablations():
    """Run all ablation studies."""
    print("\n" + "=" * 80)
    print("ABLATION STUDIES")
    print("=" * 80)

    all_pca_results = {}
    all_seed_results = {}
    all_loo_results = {}

    for name, info in ABLATION_DATASETS.items():
        print(f"\n>>> Processing {name}...")

        try:
            pca_results = run_pca_ablation(name, info)
            if len(pca_results) > 0:
                all_pca_results[name] = pca_results
                pca_results.to_csv(os.path.join(OUTPUT_DIR, f'ablation_pca_{name}.csv'), index=False)

            seed_results = run_seed_ablation(name, info)
            if len(seed_results) > 0:
                all_seed_results[name] = seed_results
                seed_results.to_csv(os.path.join(OUTPUT_DIR, f'ablation_seed_{name}.csv'), index=False)

            loo_results = run_loo_ablation(name, info)
            if loo_results is not None:
                all_loo_results[name] = loo_results

        except Exception as e:
            print(f"  Dataset failed: {e}")
            continue

    # Save LOO summary
    if all_loo_results:
        loo_summary = pd.DataFrame(list(all_loo_results.values()))
        loo_summary.to_csv(os.path.join(OUTPUT_DIR, 'ablation_loo_summary.csv'), index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    print("\n--- PCA Dimensionality ---")
    for name, df in all_pca_results.items():
        print(f"\n{name}:")
        for _, row in df.iterrows():
            print(f"  n_pcs={row['n_pcs']}: rho={row['rho']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")

    print("\n--- Seed Stability ---")
    for name, df in all_seed_results.items():
        print(f"\n{name}:")
        print(f"  rho range: [{df['rho'].min():.3f}, {df['rho'].max():.3f}]")
        print(f"  rho mean: {df['rho'].mean():.3f} +/- {df['rho'].std():.4f}")
        if 'cross_seed_corr' in df.columns:
            print(f"  Cross-seed r: {df['cross_seed_corr'].iloc[0]:.3f}")

    print("\n--- Leave-One-Out ---")
    for name, res in all_loo_results.items():
        print(f"\n{name}:")
        print(f"  Full rho: {res['full_rho']:.3f}, LOO range: [{res['loo_min_rho']:.3f}, {res['loo_max_rho']:.3f}]")

    return all_pca_results, all_seed_results, all_loo_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run main analysis
    df, df_whitened, df_knn = run_main_analysis()

    if df is not None:
        print("\n" + "=" * 80)
        print("MAIN ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files in {OUTPUT_DIR}:")
        print("  - null_model_simulation.csv")
        print("  - shesha_crispr_results_euclidean.csv")
        print("  - shesha_crispr_results_whitened.csv")
        print("  - shesha_crispr_results_knn.csv")
        print("  - crispr_correlations_with_ci.csv")
        print("  - crispr_partial_correlations_with_ci.csv")

    # Run ablations
    print("\n\n")
    all_pca, all_seed, all_loo = run_all_ablations()

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)