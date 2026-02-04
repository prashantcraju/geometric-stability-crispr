#!/usr/bin/env python3
"""
scGPT Geometric Stability Analysis

Computes geometric stability and magnitude metrics using scGPT embeddings
on the Norman 2019 CRISPR perturbation dataset.

USAGE:
    python scgpt_analysis.py

REQUIRES:
    - Pre-downloaded scGPT pretrained model (https://github.com/bowang-lab/scGPT)
    - GPU recommended but CPU fallback supported

WORKFLOW:
    1. Load Norman 2019 perturbation dataset via pertpy
    2. Prepare raw counts for scGPT embedding
    3. Generate cell embeddings using scGPT pretrained model
    4. Compute geometric stability (directional consistency) via Shesha
    5. Compute magnitude (effect size) via Shesha
    6. Report Spearman correlation between stability and magnitude
"""

def main(model_dir: str):

    import numpy as np
    import pandas as pd
    import scgpt
    from scgpt.tasks import embed_data
    from anndata import AnnData
    import torch
    from shesha.bio import compute_stability, compute_magnitude
    import scanpy as sc
    import pertpy as pt
    import random
    import os

    # Configuration: Enforce Bit-Exact Reproducibility
    SEED = 320

    # Python & NumPy seeds
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # PyTorch seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # For multi-GPU

    # Enforce deterministic algorithms (The "Nuclear Option" for consistency)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seeds set to {SEED}. Deterministic mode engaged.")

    # Load dataset (this may take a minute)
    print("Loading Norman 2019 dataset...")
    adata = pt.dt.norman_2019()
    print(f"Dataset shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Check available metadata
    print("Available columns:")
    print(adata.obs.columns.tolist())


    # 1. Prepare Data (CRITICAL: Use Raw Counts)
    if "counts" in adata.layers:
        adata_scgpt = adata.copy()
        adata_scgpt.X = adata_scgpt.layers["counts"].copy() # Revert to raw counts
    else:
        # If no counts layer exists, use the current object but WARN the user
        # Ideally, you should reload the dataset here to ensure it's raw.
        print("WARNING: Using current .X. Ensure this is RAW COUNTS, not log-normalized data.")
        adata_scgpt = adata.copy()

    # 2. Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Generate Embeddings
    # CHANGE: gene_col="index" tells scGPT to use the gene names from the index.
    adata_embedded = embed_data(
        adata_scgpt,
        model_dir=model_dir,
        gene_col="index",        
        batch_size=64,
        device=device,
        use_fast_transformer=False 
    )

    # 4. Create Proxy for Shesha
    # The output is a new AnnData object with embeddings in .X
    adata_proxy = AnnData(X=adata_embedded.X, obs=adata.obs)

    print("Success! Proxy shape:", adata_proxy.shape)


    print("Computing geometric stability...")
    # 5. Compute Stability (Precision)
    # Measures how consistent the perturbation direction is across cells.
    stability_scores = compute_stability(
        adata_proxy,
        perturbation_key='perturbation_name',
        control_label='control',
        metric='cosine'
    )

    # 6. Compute Magnitude (Strength)
    # Measures how far the perturbation shifts cells from the control state.
    magnitude_scores = compute_magnitude(
        adata_proxy,
        perturbation_key='perturbation_name',
        control_label='control',
        metric='euclidean'
    )

    # 7. Organize results into a DataFrame
    df = pd.DataFrame({
        'perturbation': stability_scores.keys(),
        'stability': stability_scores.values(),
        'magnitude': [magnitude_scores[k] for k in stability_scores.keys()]
    })

    # Add cell counts (critical for filtering noise later)
    counts = adata.obs['perturbation_name'].value_counts()
    df['n_cells'] = df['perturbation'].map(counts)

    print(f"Computed metrics for {len(df)} perturbations.")
    print(df.head())

    return df


def bootstrap_spearman_ci(x, y, n_bootstrap=10000, ci_level=0.95, seed=320, verbose=True):
    import numpy as np
    from scipy.stats import spearmanr

    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Point estimate
    rho, p = spearmanr(x, y)

    # Bootstrap
    rng = np.random.default_rng(seed=seed)
    bootstrap_rhos = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        bootstrap_rhos[i] = r

    # Filter NaNs
    valid_rhos = bootstrap_rhos[~np.isnan(bootstrap_rhos)]
    
    # Percentile CI
    alpha = 1 - ci_level
    ci_low = np.percentile(valid_rhos, 100 * alpha / 2)
    ci_high = np.percentile(valid_rhos, 100 * (1 - alpha / 2))

    return {
        'rho': rho, 'ci_low': ci_low, 'ci_high': ci_high,
        'p': p
    }


if __name__ == "__main__":
    # Change the model_dir to the path to your scGPT model
    df = main(model_dir='/path/to/scGPT_model')
    # Run it on scGPT dataframe
    results = bootstrap_spearman_ci(
        df['stability'], 
        df['magnitude'], 
        n_bootstrap=10000, 
        seed=320 # Using master seed for consistency
    )

    print("-" * 30)
    print("scGPT Results")
    print("-" * 30)
    print(f"Spearman correlation (Stability vs Magnitude):")
    print(f"rho: {results['rho']:.3f}")
    print(f"p-value: {results['p']:.2e}")
    print(f"95% CI: [{results['ci_low']:.3f}, {results['ci_high']:.3f}]")
    print("-" * 30)
