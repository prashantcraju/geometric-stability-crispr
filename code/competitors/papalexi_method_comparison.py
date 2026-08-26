#!/usr/bin/env python3
"""
Papalexi 2021 — Magnitude vs Stability: Euclidean, Whitened, k-NN
with 10,000-replicate Bootstrap CIs

Implements the three metric methods from geometric_stability_main_analysis.py
applied exclusively to the Papalexi 2021 CRISPR dataset:
  1. Euclidean (standard cosine stability, Euclidean magnitude)
  2. Whitened / Mahalanobis (control-covariance whitening transform)
  3. k-NN matched control centroids (k=50)

Spearman rho + 10k bootstrap 95% CI reported for each method.
Partial correlation controlling for SNR also computed.

OUTPUT (saved to OUTPUT_DIR):
  papalexi_euclidean.csv             — per-perturbation results, Method 1
  papalexi_whitened.csv              — per-perturbation results, Method 2
  papalexi_knn.csv                   — per-perturbation results, Method 3
  papalexi_method_correlations.csv   — summary rho + CI table
  papalexi_method_comparison.pdf/.png — 3-panel figure
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import subprocess
import sys
import os
import types
import importlib.util


from pathlib import Path
import numpy as np
import pandas as pd

_cache = Path("/tmp/pertpy_data")
try:
    _cache.mkdir(parents=True, exist_ok=True)
except OSError:
    _cache = Path.home() / ".cache" / "pertpy_data"
    _cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SCVERSE_DATADIR", str(_cache))
os.environ.setdefault("PERTPY_CACHE_DIR", str(_cache))

# Load pertpy dataset loaders without triggering pertpy.__init__ (avoids JAX)
for _mod in list(sys.modules):
    if _mod == "pertpy" or _mod.startswith("pertpy."):
        del sys.modules[_mod]

_pertpy_spec = importlib.util.find_spec("pertpy")
if _pertpy_spec is None or not _pertpy_spec.submodule_search_locations:
    raise ImportError("pertpy is not installed. Run: pip install pertpy==1.0.6")
_pertpy_path = _pertpy_spec.submodule_search_locations[0]
_pertpy_pkg = types.ModuleType("pertpy")
_pertpy_pkg.__path__ = [_pertpy_path]
_pertpy_pkg.__spec__ = _pertpy_spec
sys.modules["pertpy"] = _pertpy_pkg

import scanpy as sc
sc.settings.datasetdir = _cache
_pt_datasets = importlib.import_module("pertpy.data._datasets")
_pt_datasets.settings.datasetdir = _cache

import random
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (frozen — see pipeline_config.py)
# =============================================================================

import pipeline_config as cfg

SEED = cfg.SEED
N_BOOTSTRAP = cfg.N_BOOTSTRAP
CI_LEVEL = cfg.CI_LEVEL
MIN_CELLS = cfg.MIN_CELLS  # frozen at 50 (was 10)
N_PCS = cfg.N_PCS
KNN_K = 50
LOESS_FRAC = 0.4
REGULARIZATION = 1e-6

OUTPUT_DIR = cfg.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
sc.settings.seed = SEED

BROWN = '#8C564B'   # Papalexi color matching fig2,4,5.py

# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Spearman rho with percentile bootstrap 95% CI (mirrors main analysis)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return dict(rho=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=len(x))
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    alpha = 1 - CI_LEVEL
    return dict(
        rho=rho, p=p, n=len(x),
        ci_low=float(np.percentile(valid, 100 * alpha / 2)),
        ci_high=float(np.percentile(valid, 100 * (1 - alpha / 2))),
        n_boot_valid=len(valid),
    )


def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Partial Spearman rho(x, y | z) with bootstrap CI."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    def _partial(x, y, z):
        Z = add_constant(z)
        return spearmanr(OLS(x, Z).fit().resid, OLS(y, Z).fit().resid)

    rho_p, p = _partial(x, y, z)
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        boot[i] = _partial(x[idx], y[idx], z[idx])[0]
    valid = boot[~np.isnan(boot)]
    if len(valid) < 10:
        return dict(rho_partial=rho_p, ci_low=np.nan, ci_high=np.nan, p=p)
    alpha = 1 - CI_LEVEL
    return dict(
        rho_partial=rho_p, p=p,
        ci_low=float(np.percentile(valid, 100 * alpha / 2)),
        ci_high=float(np.percentile(valid, 100 * (1 - alpha / 2))),
    )

# =============================================================================
# METRIC COMPUTATION (exact copies from geometric_stability_main_analysis.py)
# =============================================================================

def calculate_metrics_enhanced(control_matrix, pert_matrix, use_whitening=False,
                                control_cov=None, regularization=REGULARIZATION):
    """Euclidean or Whitened (Mahalanobis) Shesha metrics."""
    control_centroid = np.mean(control_matrix, axis=0)

    if use_whitening:
        if control_cov is None:
            control_cov = np.cov(control_matrix.T)
        cov_reg = control_cov + regularization * np.eye(control_cov.shape[0])
        try:
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            eigvals = np.maximum(eigvals, regularization)
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            control_centroid = W @ control_centroid
            pert_matrix = (W @ pert_matrix.T).T
        except np.linalg.LinAlgError:
            pass  # fall through to Euclidean

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
    stability = np.mean(np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx])

    spread = np.mean(np.linalg.norm(shift_vectors - mean_shift, axis=1))
    snr = mean_magnitude / (spread + 1e-6)

    return {'stability': stability, 'magnitude': mean_magnitude,
            'spread': spread, 'snr': snr}


def calculate_metrics_knn_control(control_matrix, pert_matrix, k=KNN_K):
    """k-NN matched local control centroid Shesha metrics."""
    k = min(k, control_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(control_matrix)
    _, indices = nn.kneighbors(pert_matrix)

    shift_vectors = np.array([
        pert_matrix[i] - np.mean(control_matrix[idx], axis=0)
        for i, idx in enumerate(indices)
    ])

    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_shift)

    if mean_magnitude < 1e-6:
        return {'stability': 0.0, 'magnitude': 0.0, 'spread': 0.0, 'snr': 0.0}

    norms = np.linalg.norm(shift_vectors, axis=1)
    valid_idx = norms > 1e-6
    if np.sum(valid_idx) < 5:
        return {'stability': 0.0, 'magnitude': 0.0, 'spread': 0.0, 'snr': 0.0}

    unit_mean = mean_shift / mean_magnitude
    stability = np.mean(np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx])

    spread = np.mean(np.linalg.norm(shift_vectors - mean_shift, axis=1))
    snr = mean_magnitude / (spread + 1e-6)

    return {'stability': stability, 'magnitude': mean_magnitude,
            'spread': spread, 'snr': snr}

# =============================================================================
# LOAD PAPALEXI 2021
# =============================================================================

print("=" * 80)
print("PAPALEXI 2021 — MAGNITUDE vs STABILITY: EUCLIDEAN / WHITENED / k-NN")
print(f"Bootstrap replicates: {N_BOOTSTRAP}  |  Min cells: {MIN_CELLS}  |  k-NN k={KNN_K}")
print("=" * 80)

print("\n>>> Loading Papalexi 2021 (CRISPR-KO)...")
raw = _pt_datasets.papalexi_2021()

if type(raw).__name__ != 'MuData':
    raise TypeError(f"Expected MuData for Papalexi 2021, got {type(raw)}")
if 'rna' not in raw.mod:
    raise KeyError("No 'rna' modality found in Papalexi MuData")

adata = raw.mod['rna'].copy()

if 'gene_target' not in raw.obs.columns:
    raise KeyError("'gene_target' not found in Papalexi MuData.obs")

adata.obs['gene_target'] = raw.obs['gene_target'].values
pert_col  = 'gene_target'
ctrl_label = 'NT'

n_ctrl = (adata.obs[pert_col] == ctrl_label).sum()
print(f"    Cells: {adata.n_obs}  |  Genes: {adata.n_vars}  |  NT control cells: {n_ctrl}")

# Preprocessing
print("\n>>> Preprocessing (normalize → log1p → HVGs → PCA)...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

counts = adata.obs[pert_col].value_counts()
valid_perts = [p for p in counts[counts >= MIN_CELLS].index if p != ctrl_label]
print(f"    {len(valid_perts)} perturbations with >= {MIN_CELLS} cells")

adata_sub = adata[adata.obs[pert_col].isin(valid_perts + [ctrl_label])].copy()
sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, subset=True)
sc.tl.pca(adata_sub, n_comps=min(N_PCS, adata_sub.n_vars - 1), random_state=SEED)
print(f"    PCA shape: {adata_sub.obsm['X_pca'].shape}")

# Control PCA matrix + covariance
ctrl_mask = adata_sub.obs[pert_col] == ctrl_label
X_ctrl = adata_sub.obsm['X_pca'][ctrl_mask]
ctrl_cov = np.cov(X_ctrl.T)
ctrl_centroid = np.mean(X_ctrl, axis=0)
ctrl_scale = np.mean(np.linalg.norm(X_ctrl - ctrl_centroid, axis=1))
print(f"    Control cells: {X_ctrl.shape[0]}  |  Control spread (PCA): {ctrl_scale:.4f}")

# =============================================================================
# COMPUTE ALL THREE METHODS
# =============================================================================

print("\n>>> Computing metrics for all perturbations (3 methods)...")

results_euc, results_wht, results_knn = [], [], []

for pert in tqdm(valid_perts, desc="    Perturbations"):
    pmask = adata_sub.obs[pert_col] == pert
    X_pert = adata_sub.obsm['X_pca'][pmask]
    n_cells = X_pert.shape[0]

    if n_cells < MIN_CELLS:
        continue

    # Method 1: Euclidean
    m = calculate_metrics_enhanced(X_ctrl, X_pert, use_whitening=False)
    if m['magnitude'] > 0:
        results_euc.append({
            'perturbation': pert, 'n_cells': n_cells,
            'stability': m['stability'], 'magnitude': m['magnitude'],
            'spread': m['spread'], 'snr': m['snr'],
        })

    # Method 2: Whitened (Mahalanobis)
    m_w = calculate_metrics_enhanced(X_ctrl, X_pert, use_whitening=True,
                                     control_cov=ctrl_cov)
    if m_w['magnitude'] > 0:
        results_wht.append({
            'perturbation': pert, 'n_cells': n_cells,
            'stability': m_w['stability'], 'magnitude_mahalanobis': m_w['magnitude'],
            'spread': m_w['spread'], 'snr': m_w['snr'],
        })

    # Method 3: k-NN matched
    m_k = calculate_metrics_knn_control(X_ctrl, X_pert, k=KNN_K)
    if m_k['magnitude'] > 0:
        results_knn.append({
            'perturbation': pert, 'n_cells': n_cells,
            'stability': m_k['stability'], 'magnitude': m_k['magnitude'],
            'spread': m_k['spread'], 'snr': m_k['snr'],
        })

df_euc = pd.DataFrame(results_euc)
df_wht = pd.DataFrame(results_wht)
df_knn = pd.DataFrame(results_knn)

print(f"\n    Euclidean: {len(df_euc)} perturbations")
print(f"    Whitened:  {len(df_wht)} perturbations")
print(f"    k-NN:      {len(df_knn)} perturbations")

# Save per-perturbation CSVs
df_euc.to_csv(OUTPUT_DIR / "papalexi_euclidean.csv", index=False)
df_wht.to_csv(OUTPUT_DIR / "papalexi_whitened.csv", index=False)
df_knn.to_csv(OUTPUT_DIR / "papalexi_knn.csv", index=False)
print(f"\n    Saved per-perturbation CSVs to {OUTPUT_DIR}")

# =============================================================================
# SPEARMAN RHO + 10k BOOTSTRAP CI
# =============================================================================

print("\n" + "=" * 80)
print("SPEARMAN CORRELATIONS — magnitude vs stability (10k bootstrap CIs)")
print("=" * 80)

mag_col = {
    'Euclidean': ('magnitude',            df_euc),
    'Whitened':  ('magnitude_mahalanobis', df_wht),
    'k-NN':      ('magnitude',            df_knn),
}

ci_results = {}
seed_ctr = SEED + 1000

print(f"\n{'Method':<12s}  {'n':>4s}  {'rho':>6s}  {'95% CI':>22s}  {'p':>10s}")
print("-" * 62)

for method, (mcol, df_m) in mag_col.items():
    ci = bootstrap_spearman_ci(df_m[mcol], df_m['stability'], seed=seed_ctr)
    seed_ctr += 1
    ci_results[method] = ci
    ci_str = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
    print(f"{method:<12s}  {ci['n']:>4d}  {ci['rho']:>+.3f}  {ci_str:>22s}  {ci['p']:>10.2e}")

# =============================================================================
# PARTIAL CORRELATION (controlling for SNR)
# =============================================================================

print("\n" + "=" * 80)
print("PARTIAL CORRELATIONS — rho(magnitude, stability | SNR) (10k bootstrap CIs)")
print("=" * 80)

partial_results = {}

print(f"\n{'Method':<12s}  {'rho_partial':>11s}  {'95% CI':>22s}  {'p':>10s}")
print("-" * 62)

for method, (mcol, df_m) in mag_col.items():
    pc = bootstrap_partial_correlation_ci(
        df_m[mcol].values, df_m['stability'].values, df_m['snr'].values,
        seed=seed_ctr
    )
    seed_ctr += 1
    partial_results[method] = pc
    ci_str = f"[{pc['ci_low']:.3f}, {pc['ci_high']:.3f}]"
    print(f"{method:<12s}  {pc['rho_partial']:>+11.3f}  {ci_str:>22s}  {pc['p']:>10.2e}")

# Save summary
summary_rows = []
for method, (mcol, df_m) in mag_col.items():
    ci = ci_results[method]
    pc = partial_results[method]
    summary_rows.append({
        'method': method,
        'n': ci['n'],
        'rho': ci['rho'],
        'ci_low': ci['ci_low'],
        'ci_high': ci['ci_high'],
        'p': ci['p'],
        'rho_partial_snr': pc['rho_partial'],
        'partial_ci_low': pc['ci_low'],
        'partial_ci_high': pc['ci_high'],
        'partial_p': pc['p'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "papalexi_method_correlations.csv", index=False)
print(f"\n    Saved summary -> papalexi_method_correlations.csv")

# =============================================================================
# FIGURE: 3-panel (1x3), one scatter per method
# =============================================================================

print("\n>>> Generating figure...")

METHOD_PANELS = [
    ('Euclidean',   'magnitude',             df_euc, 'Euclidean magnitude'),
    ('Whitened',    'magnitude_mahalanobis',  df_wht, 'Mahalanobis magnitude'),
    ('k-NN',        'magnitude',             df_knn, f'Magnitude (k-NN, k={KNN_K})'),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (method, mcol, df_m, xlabel) in zip(axes, METHOD_PANELS):
    ci = ci_results[method]

    ax.scatter(df_m[mcol], df_m['stability'],
               c=BROWN, s=40, alpha=0.6, edgecolor='white', linewidth=0.3)

    # LOESS fit line
    order = np.argsort(df_m[mcol].values)
    x_s = df_m[mcol].values[order]
    y_s = df_m['stability'].values[order]
    fitted = lowess(y_s, x_s, frac=LOESS_FRAC, return_sorted=False)
    ax.plot(x_s, fitted, '--', color='gray', linewidth=2, alpha=0.7)

    ann = (f"$\\rho$ = {ci['rho']:.3f}\n"
           f"95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]\n"
           f"p = {ci['p']:.2e}")
    ax.text(0.97, 0.03, ann,
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax.set_title(f'Papalexi 2021 — {method}\n(n={ci["n"]})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel('Shesha Coherence (cosine)', fontsize=10, fontweight='bold')
    sns.despine(ax=ax)

for ax, label in zip(axes, 'abc'):
    ax.text(-0.08, 1.08, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')

plt.tight_layout()

out = OUTPUT_DIR / "papalexi_method_comparison"
plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"    Saved figure -> {out}.pdf / .png")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY — Papalexi 2021 (CRISPR-KO)")
print("=" * 80)
print(f"\n{'Method':<12s}  {'rho':>6s}  {'95% CI':>22s}  {'partial rho|SNR':>16s}")
print("-" * 65)
for row in summary_rows:
    pc_str = (f"{row['rho_partial_snr']:>+.3f} [{row['partial_ci_low']:.3f}, "
              f"{row['partial_ci_high']:.3f}]")
    print(f"{row['method']:<12s}  {row['rho']:>+.3f}  "
          f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  "
          f"{pc_str:>16s}")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
print(f"\nOutput files in {OUTPUT_DIR}:")
print("  - papalexi_euclidean.csv")
print("  - papalexi_whitened.csv")
print("  - papalexi_knn.csv")
print("  - papalexi_method_correlations.csv")
print("  - papalexi_method_comparison.pdf / .png")
