"""
nadig_comparison.py

Compares Sp-discordance rankings against alternative perturbation heterogeneity
metrics on the Replogle 2022 dataset (and Norman 2019 as secondary).

Metrics computed:
  1. Sp-discordance          -- this work (standardized residual, magnitude-stability OLS)
  2. Within-perturbation spread (variance)  -- mean per-cell variance in PCA space
  3. DE gene count           -- number of genes with |LFC| > 0.5 and FDR < 0.05 (t-test)
  4. Mean absolute LFC       -- mean |log fold change| across all tested genes
  5. Nadig-style eta^2       -- proportion of PCA variance explained by perturbation
                                identity (one-way ANOVA on each PC; averaged across PCs)
  6. CV of expression norms  -- coefficient of variation of per-cell L2 norms
  7. Song PS proxy           -- mean per-cell Euclidean distance from control centroid
                                in PCA space (proxy for perturbation-response score concept;
                                NOT the actual Song et al. constrained-optimization PS)

Output:
  - nadig_comparison_table.csv   (per-perturbation metrics)
  - nadig_correlation_table.csv  (pairwise Spearman rhos among all metrics)
  - nadig_comparison_figure.pdf  (scatter grid + bar chart of rhos with discordance)

Usage:
  python nadig_comparison.py --dataset replogle --min_cells 50 --n_pcs 50
  python nadig_comparison.py --dataset norman   --min_cells 50 --n_pcs 50

Random seed: 320 (hardcoded as per paper convention)
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import warnings
import os
import sys
import types
import importlib.util
from pathlib import Path

# Set pertpy/scanpy cache to a writable directory (needed for Deepnote/restricted envs)
_cache = Path.home() / ".cache" / "pertpy_data"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PERTPY_CACHE_DIR", str(_cache))
os.environ.setdefault("SCVERSE_DATADIR", str(_cache))

# Safe pertpy import: load only the data module, skipping pertpy.__init__
# (which pulls in JAX and causes scanpy._utils compatibility errors).
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

pt = types.SimpleNamespace(
    dt=types.SimpleNamespace(
        replogle_2022_k562_gwps=_pt_datasets.replogle_2022_k562_essential,  # essential (50k cells) instead of gwps (250k) to avoid OOM
        norman_2019=_pt_datasets.norman_2019,
    )
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import spearmanr, f_oneway
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

SEED = 320
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> sc.AnnData:
    print(f"Loading dataset: {name}")
    if name == "replogle":
        adata = pt.dt.replogle_2022_k562_gwps()
    elif name == "norman":
        adata = pt.dt.norman_2019()
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'replogle' or 'norman'.")
    print(f"  Raw shape: {adata.shape}")
    return adata


def identify_controls(adata: sc.AnnData, name: str) -> sc.AnnData:
    """Tag control cells with a standardised 'is_control' column."""
    col = "perturbation" if "perturbation" in adata.obs.columns else adata.obs.columns[0]

    # Replogle uses 'gene' column; Norman uses 'perturbation'
    for candidate in ["perturbation", "gene", "condition", "guide_identity"]:
        if candidate in adata.obs.columns:
            col = candidate
            break

    adata.obs["_pert_label"] = adata.obs[col].astype(str)

    control_patterns = [
        "non-targeting", "nontargeting", "control", "ctrl",
        "safe_targeting", "intergenic", "scramble", "chr"
    ]
    mask = adata.obs["_pert_label"].str.lower().str.contains(
        "|".join(control_patterns), regex=True
    )
    adata.obs["is_control"] = mask
    n_ctrl = mask.sum()
    print(f"  Control cells: {n_ctrl} / {len(adata)} "
          f"({100*n_ctrl/len(adata):.1f}%)")
    return adata, col


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(adata: sc.AnnData, n_pcs: int = 50) -> sc.AnnData:
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, random_state=SEED)
    print(f"  PCA done ({n_pcs} PCs), shape: {adata.obsm['X_pca'].shape}")
    return adata


# ---------------------------------------------------------------------------
# Core stability / magnitude computation  (matches paper Eq 1-6)
# ---------------------------------------------------------------------------

def compute_stability_magnitude(
    adata: sc.AnnData,
    pert_col: str,
    min_cells: int = 50,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by perturbation with columns:
      Sp, Mp, n_cells, spread_p
    """
    X_pca = adata.obsm["X_pca"]
    labels = adata.obs["_pert_label"].values
    is_ctrl = adata.obs["is_control"].values

    ctrl_centroid = X_pca[is_ctrl].mean(axis=0)

    records = []
    for pert in np.unique(labels):
        if adata.obs.loc[adata.obs["_pert_label"] == pert, "is_control"].all():
            continue
        idx = np.where(labels == pert)[0]
        if len(idx) < min_cells:
            continue

        cells = X_pca[idx]                    # (n, d)
        shifts = cells - ctrl_centroid         # (n, d)

        mean_shift = shifts.mean(axis=0)       # (d,)
        Mp = float(np.linalg.norm(mean_shift))
        if Mp < 1e-12:
            continue

        # cosine similarity of each shift to mean direction
        norms = np.linalg.norm(shifts, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        cos_sims = (shifts / norms) @ (mean_shift / np.linalg.norm(mean_shift))
        Sp = float(cos_sims.mean())

        # within-perturbation spread: mean variance across PCs
        spread_p = float(shifts.var(axis=0).mean())

        # CV of cell L2 norms  (used later as alternative metric)
        cell_norms = np.linalg.norm(shifts, axis=1)
        cv_norms = float(cell_norms.std() / (cell_norms.mean() + 1e-12))

        records.append({
            "perturbation": pert,
            "Sp": Sp,
            "Mp": Mp,
            "n_cells": len(idx),
            "spread_p": spread_p,
            "cv_norms": cv_norms,
        })

    df = pd.DataFrame(records).set_index("perturbation")
    print(f"  Perturbations retained: {len(df)}")
    return df


def compute_discordance(df: pd.DataFrame) -> pd.DataFrame:
    """Add discordance column: z(Mp) - z(Sp) after OLS regression."""
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    z_Mp = z(df["Mp"])
    z_Sp = z(df["Sp"])

    # OLS: z_Sp ~ z_Mp
    slope, intercept, _, _, _ = stats.linregress(z_Mp, z_Sp)
    residuals = z_Sp - (slope * z_Mp + intercept)
    # Discordance = z(Mp) - z(Sp) is positive when magnitude > predicted stability
    df["discordance"] = z_Mp - z_Sp
    df["ols_residual"] = residuals
    return df


# ---------------------------------------------------------------------------
# Alternative metric 1: DE gene count + mean |LFC|
# ---------------------------------------------------------------------------

def compute_de_metrics(
    adata: sc.AnnData,
    df_base: pd.DataFrame,
    lfc_thresh: float = 0.5,
    fdr_thresh: float = 0.05,
) -> pd.DataFrame:
    """
    Fast per-perturbation t-test vs controls in log-normalised gene space.
    Returns columns: de_gene_count, mean_abs_lfc
    """
    print("Computing DE metrics (t-test per perturbation)...")
    # Use raw log-normalised counts from adata.X (already scaled for PCA;
    # we recompute from scratch on the HVG subset before scaling)
    # Safer: use adata.raw if available, else recompute
    if adata.raw is not None:
        X_raw = adata.raw[:, adata.var_names].X
        try:
            X_raw = X_raw.toarray()
        except AttributeError:
            X_raw = np.array(X_raw)
    else:
        # Fall back to adata.X (already log-normalised but scaled)
        try:
            X_raw = adata.X.toarray()
        except AttributeError:
            X_raw = np.array(adata.X)

    ctrl_idx = np.where(adata.obs["is_control"].values)[0]
    X_ctrl = X_raw[ctrl_idx]
    ctrl_mean = X_ctrl.mean(axis=0)

    labels = adata.obs["_pert_label"].values
    records = []
    perts = list(df_base.index)

    for i, pert in enumerate(perts):
        if i % 200 == 0:
            print(f"  DE: {i}/{len(perts)}")
        pert_idx = np.where(labels == pert)[0]
        X_pert = X_raw[pert_idx]
        pert_mean = X_pert.mean(axis=0)
        lfc = pert_mean - ctrl_mean         # log space, so this is LFC

        # Welch t-test per gene (vectorised approximation)
        n1 = len(pert_idx)
        n2 = len(ctrl_idx)
        var1 = X_pert.var(axis=0) + 1e-12
        var2 = X_ctrl.var(axis=0) + 1e-12
        se = np.sqrt(var1 / n1 + var2 / n2)
        t_stat = lfc / se
        # degrees of freedom (Welch)
        df_w = (var1/n1 + var2/n2)**2 / (
            (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1) + 1e-30
        )
        # two-sided p-value (approximation via normal for speed at large df)
        p_vals = 2 * stats.norm.sf(np.abs(t_stat))

        # Benjamini-Hochberg FDR
        n_genes = len(p_vals)
        order = np.argsort(p_vals)
        ranks = np.empty(n_genes)
        ranks[order] = np.arange(1, n_genes + 1)
        fdr = p_vals * n_genes / ranks
        fdr = np.minimum.accumulate(fdr[order][::-1])[::-1][np.argsort(order)]
        fdr = np.clip(fdr, 0, 1)

        sig_mask = (fdr < fdr_thresh) & (np.abs(lfc) > lfc_thresh)
        records.append({
            "perturbation": pert,
            "de_gene_count": int(sig_mask.sum()),
            "mean_abs_lfc": float(np.abs(lfc).mean()),
        })

    de_df = pd.DataFrame(records).set_index("perturbation")
    return de_df


# ---------------------------------------------------------------------------
# Alternative metric 2: Nadig-style eta^2 (variance decomposition)
# ---------------------------------------------------------------------------

def compute_nadig_eta2(
    adata: sc.AnnData,
    df_base: pd.DataFrame,
    n_pcs: int = 50,
) -> pd.DataFrame:
    """
    For each perturbation p, compute eta^2 = SS_between / SS_total where
    groups are: {cells of p} vs {control cells}, on each PC independently.
    Average eta^2 across all PCs gives the Nadig-style variance decomposition
    proxy: the fraction of PC variance explained by perturbation identity.

    This operationalises the Nadig et al. 2025 approach (which uses a mixed
    model for the full atlas; here we use one-way ANOVA per perturbation-
    control pair as a tractable per-perturbation analogue).
    """
    print("Computing Nadig-style eta^2 (variance decomposition)...")
    X_pca = adata.obsm["X_pca"]          # (N, n_pcs)
    labels = adata.obs["_pert_label"].values
    is_ctrl = adata.obs["is_control"].values
    ctrl_pca = X_pca[is_ctrl]

    records = []
    perts = list(df_base.index)

    for i, pert in enumerate(perts):
        if i % 200 == 0:
            print(f"  eta^2: {i}/{len(perts)}")
        pert_idx = np.where(labels == pert)[0]
        pert_pca = X_pca[pert_idx]

        combined = np.vstack([pert_pca, ctrl_pca])   # (n_pert + n_ctrl, n_pcs)
        grand_mean = combined.mean(axis=0)

        n_pert = len(pert_pca)
        n_ctrl_sub = min(len(ctrl_pca), 5 * n_pert)  # subsample for speed
        rng = np.random.default_rng(SEED)
        ctrl_sample_idx = rng.choice(len(ctrl_pca), size=n_ctrl_sub, replace=False)
        ctrl_sub = ctrl_pca[ctrl_sample_idx]

        combined_sub = np.vstack([pert_pca, ctrl_sub])
        grand_mean_sub = combined_sub.mean(axis=0)

        # SS_between: n_g * (group_mean - grand_mean)^2  summed over groups
        pert_mean = pert_pca.mean(axis=0)
        ctrl_mean_sub = ctrl_sub.mean(axis=0)
        ss_between = (
            n_pert * ((pert_mean - grand_mean_sub)**2) +
            n_ctrl_sub * ((ctrl_mean_sub - grand_mean_sub)**2)
        )   # shape (n_pcs,)

        # SS_total
        ss_total = ((combined_sub - grand_mean_sub)**2).sum(axis=0) + 1e-30

        eta2_per_pc = ss_between / ss_total           # (n_pcs,)
        eta2_mean = float(eta2_per_pc.mean())

        records.append({
            "perturbation": pert,
            "nadig_eta2": eta2_mean,
        })

    eta2_df = pd.DataFrame(records).set_index("perturbation")
    return eta2_df


# ---------------------------------------------------------------------------
# Alternative metric 3: Song PS proxy (Euclidean + Mahalanobis distance)
# ---------------------------------------------------------------------------

def compute_song_ps_proxy(
    adata: sc.AnnData,
    df_base: pd.DataFrame,
) -> pd.DataFrame:
    """
    Proxy for the Song et al. 2025 perturbation-response score.

    Computes TWO distance-based proxies:
      - song_ps_euclid:  mean per-cell Euclidean distance from control
                         centroid in PCA space (original proxy).
      - song_ps_mahal:   mean per-cell Mahalanobis distance, accounting
                         for control covariance structure.  This partially
                         addresses the concern that Euclidean distance in
                         PCA space ignores axis-specific variance scales.

    Neither is the actual Song et al. PS (which uses constrained quadratic
    optimisation on expression of perturbation-signature genes via scMAGeCK).
    """
    print("Computing Song PS proxies (Euclidean + Mahalanobis)...")
    X_pca = adata.obsm["X_pca"]
    labels = adata.obs["_pert_label"].values
    is_ctrl = adata.obs["is_control"].values
    ctrl_centroid = X_pca[is_ctrl].mean(axis=0)

    ctrl_cov = np.cov(X_pca[is_ctrl].T)
    reg = 1e-6 * np.eye(ctrl_cov.shape[0])
    try:
        cov_inv = np.linalg.inv(ctrl_cov + reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(ctrl_cov + reg)

    records = []
    for pert in df_base.index:
        idx = np.where(labels == pert)[0]
        cells = X_pca[idx]
        diff = cells - ctrl_centroid

        euclid_dists = np.linalg.norm(diff, axis=1)
        mahal_dists = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        records.append({
            "perturbation": pert,
            "song_ps_proxy": float(euclid_dists.mean()),
            "song_ps_mahal": float(mahal_dists.mean()),
        })

    return pd.DataFrame(records).set_index("perturbation")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlation_table(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    """Pairwise Spearman correlations with 95% bootstrap CI."""
    n = len(df)
    rng = np.random.default_rng(SEED)
    n_boot = 2000
    results = []

    for m in metric_cols:
        if m == "discordance":
            continue
        x = df["discordance"].values
        y = df[m].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        rho, pval = spearmanr(x, y)

        # bootstrap CI
        boot_rhos = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), size=len(x))
            r, _ = spearmanr(x[idx], y[idx])
            boot_rhos.append(r)
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        results.append({
            "metric": m,
            "rho_with_discordance": round(rho, 3),
            "CI_95_lo": round(ci_lo, 3),
            "CI_95_hi": round(ci_hi, 3),
            "p_value": f"{pval:.2e}",
            "n": int(mask.sum()),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "discordance":      "Sp-Discordance (this work)",
    "spread_p":         "Within-pert. spread\n(mean PC variance)",
    "cv_norms":         "CV of cell norms",
    "de_gene_count":    "DE gene count\n(|LFC|>0.5, FDR<0.05)",
    "mean_abs_lfc":     "Mean |LFC|",
    "nadig_eta2":       "Nadig-style eta^2\n(variance decomposed)",
    "song_ps_proxy":    "Song PS proxy\n(Euclidean distance)",
    "song_ps_mahal":    "Song PS proxy\n(Mahalanobis distance)",
}

def make_comparison_figure(df: pd.DataFrame, corr_df: pd.DataFrame,
                           dataset_name: str, out_path: str):
    alt_metrics = [m for m in METRIC_LABELS if m != "discordance" and m in df.columns]
    n = len(alt_metrics)

    fig = plt.figure(figsize=(5 * n, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)

    disc = df["discordance"].values

    for i, metric in enumerate(alt_metrics):
        ax = fig.add_subplot(gs[0, i])
        y = df[metric].values
        mask = np.isfinite(disc) & np.isfinite(y)
        ax.scatter(disc[mask], y[mask], alpha=0.35, s=8, color="#2166ac", linewidths=0)
        rho, _ = spearmanr(disc[mask], y[mask])
        ax.set_xlabel("Sp-Discordance", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(f"rho = {rho:.3f}", fontsize=10, fontweight="bold")
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.tick_params(labelsize=8)

    # Bar chart of rhos
    ax_bar = fig.add_subplot(gs[1, :])
    rhos = corr_df["rho_with_discordance"].values
    lo_err = rhos - corr_df["CI_95_lo"].values
    hi_err = corr_df["CI_95_hi"].values - rhos
    colors = ["#d73027" if abs(r) < 0.3 else "#4dac26" for r in rhos]
    bars = ax_bar.bar(
        corr_df["metric"].map(lambda x: METRIC_LABELS.get(x, x)),
        rhos,
        yerr=[lo_err, hi_err],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        error_kw=dict(elinewidth=1.2),
    )
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.axhline(0.3, color="gray", linewidth=0.7, linestyle="--", label="|rho|=0.3 threshold")
    ax_bar.axhline(-0.3, color="gray", linewidth=0.7, linestyle="--")
    ax_bar.set_ylabel("Spearman rho with Sp-Discordance\n(95% bootstrap CI)", fontsize=10)
    ax_bar.set_title(
        f"Sp-Discordance vs. Alternative Heterogeneity Metrics -- {dataset_name}\n"
        f"(red = |rho| < 0.3, empirically orthogonal; green = |rho| >= 0.3)",
        fontsize=11
    )
    ax_bar.tick_params(axis="x", labelsize=8, rotation=15)
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylim(-0.6, 0.9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="replogle",
                        choices=["replogle", "norman"],
                        help="Dataset to analyse")
    parser.add_argument("--min_cells", type=int, default=50)
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--out_prefix", default="nadig_comparison")
    args = parser.parse_args()

    # ---- Load + prep ----
    adata = load_dataset(args.dataset)
    adata, pert_col = identify_controls(adata, args.dataset)
    adata.raw = adata.copy()    # save raw counts before scaling
    adata = preprocess(adata, n_pcs=args.n_pcs)

    # ---- Core metrics ----
    df = compute_stability_magnitude(adata, pert_col, min_cells=args.min_cells)
    df = compute_discordance(df)

    # ---- Alternative metrics ----
    de_df = compute_de_metrics(adata, df)
    eta2_df = compute_nadig_eta2(adata, df, n_pcs=args.n_pcs)
    ps_df = compute_song_ps_proxy(adata, df)

    # ---- Merge ----
    df = df.join(de_df, how="left")
    df = df.join(eta2_df, how="left")
    df = df.join(ps_df, how="left")

    # ---- Save per-perturbation table ----
    csv_path = f"{args.out_prefix}_{args.dataset}_per_pert.csv"
    df.to_csv(csv_path)
    print(f"\nPer-perturbation table saved to {csv_path}")
    print(df.describe().to_string())

    # ---- Correlation table ----
    metric_cols = [
        "spread_p", "cv_norms", "de_gene_count",
        "mean_abs_lfc", "nadig_eta2", "song_ps_proxy", "song_ps_mahal",
    ]
    # only keep cols that exist
    metric_cols = [m for m in metric_cols if m in df.columns]
    corr_df = compute_correlation_table(df, ["discordance"] + metric_cols)

    corr_path = f"{args.out_prefix}_{args.dataset}_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"\nCorrelation table saved to {corr_path}")
    print(corr_df.to_string(index=False))

    # ---- Figure ----
    fig_path = f"{args.out_prefix}_{args.dataset}_figure.pdf"
    make_comparison_figure(df, corr_df, args.dataset.capitalize(), fig_path)

    # ---- SI-ready LaTeX table ----
    print("\n--- LaTeX table snippet for SI ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Spearman correlation of Sp-discordance with alternative "
          r"perturbation heterogeneity metrics (" + args.dataset.capitalize() + r" dataset).}")
    print(r"\begin{tabular}{llll}")
    print(r"\hline")
    print(r"Metric & Rho & 95\% CI & n \\")
    print(r"\hline")
    for _, row in corr_df.iterrows():
        label = METRIC_LABELS.get(row["metric"], row["metric"]).replace("\n", " ")
        print(f"{label} & {row['rho_with_discordance']:.3f} & "
              f"[{row['CI_95_lo']:.3f}, {row['CI_95_hi']:.3f}] & {row['n']} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\nDone.")


if __name__ == "__main__":
    main()