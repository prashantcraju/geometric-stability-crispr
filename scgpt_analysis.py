#!/usr/bin/env python3
"""
scGPT Geometric Stability Analysis -- All Datasets

Computes geometric stability and magnitude metrics using scGPT embeddings
on all 5 CRISPR perturbation datasets:
    - Norman 2019 (CRISPRa)
    - Adamson 2016 (CRISPRi)
    - Dixit 2016 (CRISPRi)
    - Papalexi 2021 (CRISPR)
    - Replogle 2022 (CRISPRi)

Saves per-dataset CSVs + combined CSV + correlation summary to Google Drive
(MyDrive/CRISPR/) when running in Colab, otherwise to ./scgpt-results/.

REQUIRES:
    - Pre-downloaded scGPT pretrained model (https://github.com/bowang-lab/scGPT)
    - GPU recommended but CPU fallback supported
"""

import subprocess
import sys
import os

# =============================================================================
# GOOGLE COLAB SETUP -- mount Drive and install dependencies
# =============================================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scanpy", "pertpy", "statsmodels", "tqdm",
                           "scikit-learn", "mudata", "anndata", "scgpt"])

import numpy as np
import pandas as pd
import torch
import random
import warnings
import re
from pathlib import Path
from scipy.stats import spearmanr
from anndata import AnnData
import scanpy as sc
import pertpy as pt
from shesha.bio import compute_stability, compute_magnitude

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 320
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_DIR = Path("/content/drive/MyDrive/shesha-crispr") if IN_COLAB else Path("./scgpt-results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPLOGLE_MIN_CELLS = 50
MIN_CELLS_PER_PERT = 10

# Manual control keywords per dataset
MANUAL_CONTROLS = {
    'Adamson 2016 (CRISPRi)': ['gal4', 'gfp', 'neg', 'scramble', 'unperturbed', 'nan'],
    'Dixit 2016 (CRISPRi)':   ['nan', 'control', 'neg', 'intergenic'],
    'Papalexi 2021 (CRISPR)': ['nt', 'non-targeting', 'control'],
    'Replogle 2022 (CRISPRi)': ['control'],
}


# =============================================================================
# REPLOGLE LOADER
# =============================================================================

def load_replogle_2022():
    print("    Loading Replogle 2022 K562 essential genes...")
    adata = pt.dt.replogle_2022_k562_essential()
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)

    def clean_label(x):
        if 'non-targeting' in x or x.startswith('chr'):
            return 'control'
        if 'pos_control' in x:
            return 'POS_CONTROL'
        return x.split('_')[0]

    adata.obs['condition'] = adata.obs['perturbation'].apply(clean_label)
    mask = (
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    )
    adata = adata[mask].copy()
    counts = adata.obs['condition'].value_counts()
    valid = counts[counts >= REPLOGLE_MIN_CELLS].index
    adata = adata[adata.obs['condition'].isin(valid)].copy()
    n_perts = len(adata.obs['condition'].unique()) - 1
    n_ctrl = (adata.obs['condition'] == 'control').sum()
    print(f"    Replogle: {adata.n_obs} cells, {n_perts} perturbations, {n_ctrl} control cells")
    return adata


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASETS = {
    'Norman 2019 (CRISPRa)':   {'loader': pt.dt.norman_2019,        'is_replogle': False},
    'Adamson 2016 (CRISPRi)':  {'loader': pt.dt.adamson_2016_pilot,  'is_replogle': False},
    'Dixit 2016 (CRISPRi)':    {'loader': pt.dt.dixit_2016,          'is_replogle': False},
    'Papalexi 2021 (CRISPR)':  {'loader': pt.dt.papalexi_2021,       'is_replogle': False},
    'Replogle 2022 (CRISPRi)': {'loader': load_replogle_2022,        'is_replogle': True},
}


# =============================================================================
# DATASET LOADING & CONTROL DETECTION
# =============================================================================

def find_pert_col_and_ctrl(adata, dataset_name, is_replogle=False):
    """Return (pert_col, ctrl_label) for a loaded AnnData."""
    if is_replogle:
        return 'condition', 'control'

    possible_cols = ['perturbation_name', 'perturbation', 'gene_target',
                     'gene', 'target', 'guide_id', 'sgRNA']
    pert_col = next((c for c in possible_cols if c in adata.obs.columns), None)
    if not pert_col:
        pert_col = next(
            (c for c in adata.obs.columns
             if any(k in c.lower() for k in ['pert', 'guide', 'gene', 'target'])),
            None
        )
    if not pert_col:
        raise ValueError("No perturbation column found")

    adata.obs[pert_col] = adata.obs[pert_col].astype(str).replace('nan', 'NaN_Control')
    labels = adata.obs[pert_col].unique()

    exact_ctrls = ['control', 'ctrl', 'non-targeting', 'scrambled', 'nt',
                   'gal4', 'gfp', 'nan_control']
    sub_ctrls   = ['control', 'ctrl', 'non-targeting', 'scrambled', 'gal4',
                   'gfp', 'nan_control', 'intergenic']

    if dataset_name in MANUAL_CONTROLS:
        manual = MANUAL_CONTROLS[dataset_name]
        exact_ctrls = manual + exact_ctrls
        sub_ctrls   = [c for c in manual if len(c) >= 3] + sub_ctrls

    ctrl = next((x for x in labels if x.lower() in [c.lower() for c in exact_ctrls]), None)
    if ctrl is None:
        ctrl = next((x for x in labels if any(c in x.lower() for c in sub_ctrls)), None)
    if ctrl is None:
        ctrl = adata.obs[pert_col].value_counts().idxmax()
        print(f"    WARNING: fell back to most-frequent label '{ctrl}' as control")

    return pert_col, ctrl


def load_dataset_raw(dataset_name, info):
    """
    Load raw AnnData (or MuData -> RNA) for a dataset.
    Returns (adata, pert_col, ctrl_label) with raw counts in .X.
    """
    print(f"\n>>> Loading {dataset_name}...")
    loader = info['loader']
    is_replogle = info['is_replogle']

    try:
        raw = loader()
    except Exception as e:
        print(f"    ! Load failed: {e}")
        return None, None, None

    # --- Papalexi: MuData -> RNA + copy gene_target ---
    if 'papalexi' in dataset_name.lower():
        if type(raw).__name__ != 'MuData':
            print("    ! Expected MuData for Papalexi")
            return None, None, None
        if 'rna' not in raw.mod:
            print("    ! No 'rna' modality in Papalexi MuData")
            return None, None, None
        adata = raw.mod['rna'].copy()
        if 'gene_target' not in raw.obs.columns:
            print("    ! 'gene_target' not in Papalexi MuData.obs")
            return None, None, None
        adata.obs['gene_target'] = raw.obs['gene_target'].values
        pert_col, ctrl = 'gene_target', 'NT'
        return adata, pert_col, ctrl

    # --- MuData (non-Papalexi): extract RNA/GEX modality ---
    if type(raw).__name__ == 'MuData':
        mdata = raw
        if 'rna' in mdata.mod:
            raw = mdata.mod['rna'].copy()
        elif 'gex' in mdata.mod:
            raw = mdata.mod['gex'].copy()
        else:
            raw = mdata.mod[list(mdata.mod.keys())[0]].copy()

    if isinstance(raw, dict):
        raw = list(raw.values())[0]

    if not isinstance(raw, AnnData):
        print(f"    ! Unexpected type: {type(raw)}")
        return None, None, None

    adata = raw
    try:
        pert_col, ctrl = find_pert_col_and_ctrl(adata, dataset_name, is_replogle)
    except Exception as e:
        print(f"    ! {e}")
        return None, None, None

    print(f"    pert_col='{pert_col}', ctrl='{ctrl}', "
          f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    return adata, pert_col, ctrl


# =============================================================================
# scGPT EMBEDDING
# =============================================================================

def embed_with_scgpt(adata, model_dir, device):
    """
    Prepare raw counts and generate scGPT cell embeddings.
    Returns a new AnnData whose .X contains the embeddings.
    """
    from scgpt.tasks import embed_data

    # Use raw counts if available
    if 'counts' in adata.layers:
        adata_raw = adata.copy()
        adata_raw.X = adata_raw.layers['counts'].copy()
    else:
        print("    WARNING: no 'counts' layer found -- using .X as-is (should be raw counts)")
        adata_raw = adata.copy()

    embedded = embed_data(
        adata_raw,
        model_dir=model_dir,
        gene_col='index',
        batch_size=64,
        device=device,
        use_fast_transformer=False,
    )
    return embedded


# =============================================================================
# GEOMETRIC STABILITY METRICS (via shesha)
# =============================================================================

def compute_metrics_from_embeddings(embedded_adata, obs_source, pert_col, ctrl_label):
    """
    Given an AnnData whose .X are scGPT embeddings, compute per-perturbation
    stability and magnitude using shesha.bio.

    Returns a DataFrame with columns:
        perturbation, stability, magnitude, n_cells
    """
    proxy = AnnData(X=embedded_adata.X, obs=obs_source.obs[[pert_col]].copy())

    stability_scores = compute_stability(
        proxy,
        perturbation_key=pert_col,
        control_label=ctrl_label,
        metric='cosine',
    )

    magnitude_scores = compute_magnitude(
        proxy,
        perturbation_key=pert_col,
        control_label=ctrl_label,
        metric='euclidean',
    )

    counts = proxy.obs[pert_col].value_counts()

    results = []
    for pert in stability_scores:
        n_cells = int(counts.get(pert, 0))
        if n_cells < MIN_CELLS_PER_PERT:
            continue
        results.append({
            'perturbation': str(pert),
            'stability': stability_scores[pert],
            'magnitude': magnitude_scores[pert],
            'n_cells': n_cells,
        })

    return pd.DataFrame(results)


# =============================================================================
# BOOTSTRAP SPEARMAN CI
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=10000, ci_level=0.95, seed=320):
    x, y = np.asarray(x), np.asarray(y)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed=seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    alpha = 1 - ci_level
    return {
        'rho': rho, 'p': p,
        'ci_low':  float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
        'n': len(x),
    }


# =============================================================================
# MAIN
# =============================================================================

def run_all(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")

    all_dfs = []
    corr_results = []

    for dataset_name, info in DATASETS.items():
        # --- Load ---
        adata, pert_col, ctrl_label = load_dataset_raw(dataset_name, info)
        if adata is None:
            continue

        # --- Embed ---
        print(f"    Embedding with scGPT...")
        try:
            embedded = embed_with_scgpt(adata, model_dir, device)
        except Exception as e:
            print(f"    ! Embedding failed: {e}")
            continue

        # --- Metrics ---
        print(f"    Computing stability & magnitude...")
        df = compute_metrics_from_embeddings(embedded, adata, pert_col, ctrl_label)
        if df.empty:
            print(f"    ! No results for {dataset_name}")
            continue

        df['dataset'] = dataset_name

        # --- Save per-dataset CSV ---
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name)
        csv_path = OUTPUT_DIR / f"scgpt_{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved {len(df)} perturbations -> {csv_path.name}")

        all_dfs.append(df)

        # --- Correlation ---
        if len(df) >= 10:
            ci = bootstrap_spearman_ci(df['magnitude'], df['stability'], seed=SEED)
            print(f"    rho = {ci['rho']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], "
                  f"p = {ci['p']:.2e}, n = {ci['n']}")
            corr_results.append({'dataset': dataset_name, **ci})

    # --- Combined CSV ---
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "scgpt_all_datasets.csv", index=False)
        print(f"\nSaved combined results ({len(combined)} rows) -> scgpt_all_datasets.csv")

    # --- Correlation summary CSV ---
    if corr_results:
        corr_df = pd.DataFrame(corr_results)
        corr_df.to_csv(OUTPUT_DIR / "scgpt_correlations.csv", index=False)
        print("\n=== scGPT Correlation Summary ===")
        for _, row in corr_df.iterrows():
            print(f"  {row['dataset']}: rho={row['rho']:.3f} "
                  f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                  f"p={row['p']:.2e}, n={int(row['n'])}")
        print(f"\nSaved correlation summary -> scgpt_correlations.csv")

    return combined if all_dfs else None, pd.DataFrame(corr_results)


if __name__ == "__main__":
    # Set model_dir to your downloaded scGPT pretrained model path
    MODEL_DIR = '/path/to/scGPT_model'

    combined_df, corr_df = run_all(model_dir=MODEL_DIR)
