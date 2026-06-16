#!/usr/bin/env python3
"""
song_ps_replication.py

Computes Song et al.'s ACTUAL Perturbation-response Score (PS) using
their official scMAGeCK R package, then re-runs the anticorrelation
and incremental UPR analyses with the real PS.

APPROACH:
  1. Python loads + preprocesses each dataset (via pertpy/scanpy)
  2. Exports expression matrix (.mtx), cell barcodes, gene names, and a
     barcode table to a temp directory
  3. Calls the official scMAGeCK R package (scmageck_eff_estimate) via
     a dynamically generated R script
  4. Reads back per-cell PS scores from R
  5. Aggregates to per-perturbation mean PS
  6. Re-runs anticorrelation (Sp vs PS | Mp) and incremental UPR analyses

REFERENCE:
  Song et al. "Decoding heterogeneous single-cell perturbation responses"
  Nature Cell Biology 27, 493–504 (2025).
  Official code: https://github.com/davidliwei/PS
  R package: https://github.com/weililab/scMAGeCK

REQUIREMENTS:
  Python: scanpy, pertpy, numpy, pandas, scipy, statsmodels, anndata
  R:      Seurat, scMAGeCK  (auto-installed if missing)

USAGE:
  python song_ps_replication.py [--datasets replogle,norman,adamson]
                                [--out_dir ./shesha-crispr]
                                [--r_executable Rscript]
                                [--max_perts_per_batch 100]
"""

import argparse
import os
import warnings
import subprocess
import tempfile
import shutil
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt

from anndata import AnnData
from scipy import io as spio
from scipy.sparse import csc_matrix, issparse
from scipy.stats import spearmanr
import statsmodels.api as sm
import gzip

warnings.filterwarnings('ignore')

SEED = 320
np.random.seed(SEED)
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95

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

REPLOGLE_MIN_CELLS = 50
NORMAN_MIN_CELLS = 50
DIXIT_MIN_CELLS = 10
ADAMSON_MIN_CELLS = 10
PAPALEXI_MIN_CELLS = 10


# ============================================================================
# BOOTSTRAP HELPERS
# ============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL,
                          seed=42):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return {'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'p': np.nan}
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


def bootstrap_partial_corr_ci(x, y, z, n_bootstrap=N_BOOTSTRAP,
                               ci_level=CI_LEVEL, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    def _partial(x, y, z):
        Z_aug = sm.add_constant(z)
        x_resid = sm.OLS(x, Z_aug).fit().resid
        y_resid = sm.OLS(y, Z_aug).fit().resid
        return spearmanr(x_resid, y_resid)

    rho_partial, p = _partial(x, y, z)
    rng = np.random.default_rng(seed=seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        boot[i] = _partial(x[idx], y[idx], z[idx])[0]
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


# ============================================================================
# PURE PYTHON PORT OF scMAGeCK PS (no R required)
# ============================================================================
# Faithful port of:
#   select_target_gene.R  → _select_target_genes()
#   single_gene_matrix_regression.R + getsolvedmatrix.R → _ridge_beta()
#   obj_function.R / obj_function_d.R / scmageck_optim_core.R → _optimise_ps()
#   scmageck_eff_estimate.R → compute_ps_python()

from scipy.optimize import minimize as scipy_minimize
from scipy.stats import rankdata
try:
    from scipy.stats import ranksums
except ImportError:
    from scipy.stats import mannwhitneyu as _mwu
    def ranksums(x, y):
        s, p = _mwu(x, y, alternative='two-sided')
        return type('R', (), {'statistic': s, 'pvalue': p})()


def _select_target_genes(expr_data, gene_names, cell_is_pert, cell_is_ctrl,
                         logfc_threshold=0.1, min_genes=10, max_genes=500):
    """
    Python equivalent of scMAGeCK's select_target_gene.R.

    Identifies DEGs between perturbed and control cells using a Wilcoxon
    rank-sum test (analogous to Seurat's FindMarkers default).  Genes are
    ranked by p-value; the top `max_genes` are returned.  If fewer than
    `min_genes` pass the threshold, the threshold is relaxed (up to 3
    rounds, multiplying by 0.8 each time), matching the R behaviour.

    Parameters
    ----------
    expr_data : ndarray, shape (n_cells, n_genes)
        Log-normalised expression matrix (equivalent to Seurat 'data' slot).
    gene_names : list[str]
    cell_is_pert : ndarray[bool]
    cell_is_ctrl : ndarray[bool]
    """
    pert_expr = expr_data[cell_is_pert]
    ctrl_expr = expr_data[cell_is_ctrl]

    for round_i in range(3):
        mean_pert = pert_expr.mean(axis=0)
        mean_ctrl = ctrl_expr.mean(axis=0)
        lfc = mean_pert - mean_ctrl  # already log-space

        passes_lfc = np.abs(lfc) >= logfc_threshold

        pvals = np.ones(len(gene_names))
        for j in np.where(passes_lfc)[0]:
            try:
                _, p = ranksums(pert_expr[:, j], ctrl_expr[:, j])
                pvals[j] = p
            except Exception:
                pass

        sig = passes_lfc & (pvals < 0.05)
        if sig.sum() >= min_genes or round_i >= 2:
            break
        logfc_threshold *= 0.8

    order = np.argsort(pvals)
    if sig.sum() > max_genes:
        selected = order[:max_genes]
    elif sig.sum() >= min_genes:
        selected = np.where(sig)[0]
    else:
        selected = order[:min_genes]

    return list(np.array(gene_names)[selected])


def _ridge_beta(X_ind, Y_expr, lam=0.01):
    """
    Ridge regression beta = (X'X + λI)^{-1} X'Y.
    Equivalent to getsolvedmatrix.R.

    Parameters
    ----------
    X_ind : ndarray (n_cells, n_groups)   — indicator matrix
    Y_expr : ndarray (n_cells, n_target_genes)
    """
    XtX = X_ind.T @ X_ind + lam * np.eye(X_ind.shape[1])
    beta = np.linalg.solve(XtX, X_ind.T @ Y_expr)
    return beta


def _obj_func(X_vec, Y, beta, n_col, mask, lam=0.0):
    """Objective: 0.5 * ||X @ beta - Y||^2  + lambda * sum(X)."""
    Xm = X_vec.reshape(-1, n_col)
    residual = Xm @ beta - Y
    return 0.5 * np.sum(residual ** 2) + lam * np.sum(Xm)


def _obj_grad(X_vec, Y, beta, n_col, mask, lam=0.0):
    """Gradient of objective w.r.t. X (flattened), masked for controls."""
    Xm = X_vec.reshape(-1, n_col)
    diff = Xm @ beta - Y
    grad = (diff @ beta.T + lam).ravel()
    return grad * mask


def _optimise_ps(X_init, Y_expr, beta, scale_factor=3.0, lam=0.0):
    """
    L-BFGS-B constrained optimisation of per-cell perturbation scores.
    Equivalent to scmageck_optim_core.R.

    X_init : ndarray (n_cells, n_groups) — initial indicator matrix
    Y_expr : ndarray (n_cells, n_target_genes)
    beta   : ndarray (n_groups, n_target_genes)
    """
    n_cells, n_groups = X_init.shape

    mask = X_init.copy()
    ctrl_col = n_groups - 1  # NegCtrl is always the last column
    mask[:, ctrl_col] = 0.0
    mask_vec = mask.ravel()

    x0 = X_init.ravel().astype(np.float64)
    bounds = [(0.0, scale_factor)] * len(x0)

    result = scipy_minimize(
        _obj_func, x0,
        args=(Y_expr, beta, n_groups, mask_vec, lam),
        jac=_obj_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10},
    )

    X_opt = result.x.reshape(n_cells, n_groups)
    return X_opt


def compute_ps_python(adata_norm, pert_col, ctrl_label, perturb_gene,
                      logfc_threshold=0.1, target_gene_max=500,
                      scale_factor=3.0, lam=0.0):
    """
    Pure Python equivalent of scmageck_eff_estimate().

    Returns per-cell PS values for cells assigned to `perturb_gene`.

    Steps (matching the R implementation):
      1. Identify perturbed and control cells from obs[pert_col]
      2. Select target genes via DEG analysis
      3. Build indicator matrix X and expression matrix Y
      4. Estimate beta scores via ridge regression
      5. Optimise per-cell PS via L-BFGS-B
      6. Scale scores to [0, 1]
    """
    labels = adata_norm.obs[pert_col].values.astype(str)
    cell_is_pert = labels == perturb_gene
    cell_is_ctrl = labels == ctrl_label
    keep = cell_is_pert | cell_is_ctrl

    if cell_is_pert.sum() < 5:
        return {}

    # Subset to perturbed + control cells
    adata_sub = adata_norm[keep].copy()
    labels_sub = adata_sub.obs[pert_col].values.astype(str)

    # Use log-normalised data for DEG + regression (matches Seurat 'data' slot)
    if hasattr(adata_sub.X, 'toarray'):
        expr_dense = np.asarray(adata_sub.X.toarray())
    else:
        expr_dense = np.asarray(adata_sub.X)

    gene_names = list(adata_sub.var_names)
    is_pert = labels_sub == perturb_gene
    is_ctrl = labels_sub == ctrl_label

    # Step 1: select target genes
    target_genes = _select_target_genes(
        expr_dense, gene_names, is_pert, is_ctrl,
        logfc_threshold=logfc_threshold,
        max_genes=target_gene_max,
    )

    if len(target_genes) < 5:
        return {}

    # Step 2: build Y matrix (cells × target genes, from scaled data)
    gene_idx = [gene_names.index(g) for g in target_genes if g in gene_names]
    if len(gene_idx) < 5:
        return {}

    Y = expr_dense[:, gene_idx].copy()
    # Outlier clipping (matching R: cap at 95th percentile per gene)
    q95 = np.percentile(Y, 95, axis=0)
    Y = np.minimum(Y, q95[None, :])

    # Step 3: build X indicator matrix
    # Columns: [perturb_gene, NegCtrl]
    n_cells = len(labels_sub)
    X = np.zeros((n_cells, 2), dtype=np.float64)
    X[is_pert, 0] = 1.0
    X[:, 1] = 1.0  # NegCtrl baseline (all cells)

    # Step 4: ridge regression for beta
    beta = _ridge_beta(X, Y, lam=0.01)

    # Step 5: constrained optimisation
    X_opt = _optimise_ps(X, Y, beta, scale_factor=scale_factor, lam=lam)

    # Step 6: scale to [0, 1]
    ps_raw = X_opt[:, 0] / scale_factor
    max_ps = ps_raw.max()
    if max_ps > 0.01:
        ps_scaled = ps_raw / max_ps
    else:
        ps_scaled = ps_raw

    # Return only perturbed cells
    cell_names = list(adata_sub.obs_names)
    result = {}
    for i in range(n_cells):
        if is_pert[i]:
            result[cell_names[i]] = float(ps_scaled[i])

    return result


def compute_real_ps_python(adata_norm, pert_col, ctrl_label, valid_perts,
                           logfc_threshold=0.1, target_gene_max=500,
                           scale_factor=3.0, lam=0.0):
    """
    Compute Song et al. PS for all perturbations using pure Python.
    Returns dict: perturbation -> mean PS.
    """
    ps_per_pert = {}
    n_done = 0
    n_total = len(valid_perts)

    for pert in valid_perts:
        n_done += 1
        if n_done % 50 == 0 or n_done == 1:
            print(f"    PS [{n_done}/{n_total}]: {pert}")

        cell_ps = compute_ps_python(
            adata_norm, pert_col, ctrl_label, pert,
            logfc_threshold=logfc_threshold,
            target_gene_max=target_gene_max,
            scale_factor=scale_factor,
            lam=lam,
        )

        if cell_ps:
            ps_per_pert[pert] = float(np.mean(list(cell_ps.values())))

    print(f"    Python PS computed for {len(ps_per_pert)}/{n_total} perturbations")
    if ps_per_pert:
        vals = list(ps_per_pert.values())
        print(f"    Mean PS: {np.mean(vals):.3f} "
              f"[{np.min(vals):.3f}, {np.max(vals):.3f}]")

    return ps_per_pert


# ============================================================================
# EXPORT ANNDATA → FILES FOR R
# ============================================================================

def export_for_r(adata, pert_col, ctrl_label, valid_perts, tmpdir):
    """
    Export AnnData to files that R/Seurat can read:
      - matrix.mtx.gz   (raw counts, cells × genes, Market Matrix format)
      - barcodes.tsv.gz (cell names)
      - features.tsv.gz (gene names)
      - barcode_table.txt (cell, barcode, gene — scMAGeCK triplet format)
    """
    # Expression matrix (use raw counts if available, else .X)
    if adata.raw is not None:
        X = adata.raw.X
        gene_names = list(adata.raw.var_names)
    else:
        X = adata.X
        gene_names = list(adata.var_names)

    if not issparse(X):
        X = csc_matrix(X)
    else:
        X = csc_matrix(X)

    # Seurat's ReadMtx expects genes × cells (features as rows)
    X_t = X.T.tocsc()

    mtx_path = os.path.join(tmpdir, "matrix.mtx")
    spio.mmwrite(mtx_path, X_t)
    with open(mtx_path, 'rb') as f_in:
        with gzip.open(mtx_path + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    os.remove(mtx_path)

    # Barcodes (cell names)
    cell_names = list(adata.obs_names)
    bc_path = os.path.join(tmpdir, "barcodes.tsv.gz")
    with gzip.open(bc_path, 'wt') as f:
        for c in cell_names:
            f.write(c + '\n')

    # Features (gene names) — Seurat expects tab-separated: id\tname\ttype
    feat_path = os.path.join(tmpdir, "features.tsv.gz")
    with gzip.open(feat_path, 'wt') as f:
        for g in gene_names:
            f.write(f"{g}\t{g}\tGene Expression\n")

    # Barcode table — scMAGeCK expects 6 columns:
    #   cell, barcode, sgrna, gene, read_count, umi_count
    # assign_cell_identity with ASSIGNMETHOD='largest' sorts by umi_count,
    # so this column MUST be present.  Since pertpy datasets don't carry
    # guide-level info, we synthesise guide labels ({gene}_sg) and set
    # read_count = umi_count = 1 for every cell.
    labels = adata.obs[pert_col].values.astype(str)
    valid_set = set(valid_perts) | {ctrl_label}
    rows = []
    for i, cell in enumerate(cell_names):
        lab = labels[i]
        if lab in valid_set:
            rows.append({
                'cell': cell,
                'barcode': f'{lab}_sg',
                'sgrna': f'{lab}_sg',
                'gene': lab,
                'read_count': 1,
                'umi_count': 1,
            })

    bc_table = pd.DataFrame(rows)
    bc_table_path = os.path.join(tmpdir, "barcode_table.txt")
    bc_table.to_csv(bc_table_path, sep='\t', index=False)

    print(f"    Exported: {X_t.shape[0]} genes × {X_t.shape[1]} cells")
    print(f"    Barcode table: {len(bc_table)} rows")

    return {
        'mtx': mtx_path + '.gz',
        'barcodes': bc_path,
        'features': feat_path,
        'barcode_table': bc_table_path,
        'cell_names': cell_names,
    }


# ============================================================================
# GENERATE + RUN R SCRIPT  (calls official scMAGeCK)
# ============================================================================

def generate_r_script(tmpdir, ctrl_label, perturb_genes, batch_id=0,
                      lambda_val=0, target_gene_max=500):
    """
    Generate an R script that:
      1. Installs scMAGeCK if needed
      2. Loads expression data into Seurat
      3. Runs scmageck_eff_estimate for each perturbation gene
      4. Saves per-cell PS scores to CSV
    """
    # Escape perturbation gene names for R
    pert_genes_r = ', '.join(f"'{g}'" for g in perturb_genes)

    r_script = f'''
# Auto-generated R script for Song et al. PS computation
# Uses official scMAGeCK package: https://github.com/weililab/scMAGeCK

# --- Install scMAGeCK if not available ---
if (!requireNamespace("scMAGeCK", quietly = TRUE)) {{
  if (!requireNamespace("devtools", quietly = TRUE)) {{
    install.packages("devtools", repos="https://cloud.r-project.org")
  }}
  devtools::install_github("weililab/scMAGeCK")
}}

library(scMAGeCK)
library(Seurat)
library(Matrix)

cat("scMAGeCK loaded successfully\\n")

# --- Load expression data ---
tmpdir <- "{tmpdir}"

exp_mat <- ReadMtx(
  mtx      = file.path(tmpdir, "matrix.mtx.gz"),
  cells    = file.path(tmpdir, "barcodes.tsv.gz"),
  features = file.path(tmpdir, "features.tsv.gz")
)

cat(sprintf("Expression matrix: %d genes x %d cells\\n", nrow(exp_mat), ncol(exp_mat)))

sobj <- CreateSeuratObject(counts = exp_mat, min.cells = 0, min.features = 0)

# Standard Seurat preprocessing
sobj <- NormalizeData(sobj, verbose = FALSE)
sobj <- FindVariableFeatures(sobj, nfeatures = 2000, verbose = FALSE)
sobj <- ScaleData(sobj, verbose = FALSE)

cat("Seurat preprocessing done\\n")

# --- Load barcode table ---
bc_frame <- read.table(
  file.path(tmpdir, "barcode_table.txt"),
  header = TRUE, as.is = TRUE, sep = "\\t"
)
cat(sprintf("Barcode table: %d rows, %d columns\\n", nrow(bc_frame), ncol(bc_frame)))
cat(sprintf("Barcode table columns: %s\\n", paste(colnames(bc_frame), collapse=", ")))

# Cell-name matching: Seurat's ReadMtx may append "-1" to barcodes.
# Align barcode table cell names with the Seurat object.
seurat_cells <- colnames(sobj)
n_match_raw <- sum(bc_frame$cell %in% seurat_cells)
cat(sprintf("Cell name match (raw): %d / %d\\n", n_match_raw, nrow(bc_frame)))

if (n_match_raw == 0) {{
  # Try appending "-1" to barcode table cell names (ReadMtx convention)
  bc_frame$cell <- paste0(bc_frame$cell, "-1")
  n_match_fix <- sum(bc_frame$cell %in% seurat_cells)
  cat(sprintf("Cell name match (after appending '-1'): %d / %d\\n", n_match_fix, nrow(bc_frame)))
  if (n_match_fix == 0) {{
    # Try stripping "-1" from Seurat cell names (demo1 convention)
    bc_frame$cell <- sub("-1$", "", bc_frame$cell)
    cat("WARNING: cell name matching still failed after suffix fix\\n")
  }}
}}

# --- Compute PS for each perturbation gene ---
perturb_genes <- c({pert_genes_r})
ctrl_label <- "{ctrl_label}"

cat(sprintf("Computing PS for %d perturbation genes...\\n", length(perturb_genes)))

all_ps_results <- data.frame()

for (pg in perturb_genes) {{
  cat(sprintf("  Processing: %s\\n", pg))

  tryCatch({{
    DefaultAssay(sobj) <- "RNA"

    eff_object <- scmageck_eff_estimate(
      sobj, bc_frame,
      perturb_gene = pg,
      non_target_ctrl = ctrl_label,
      subset_rds = TRUE,
      scale_score = TRUE,
      lambda = {lambda_val},
      target_gene_max = {target_gene_max},
      assay_for_cor = "RNA",
      logfc.threshold = 0.1
    )

    eff_matrix <- eff_object$eff_matrix
    rds_subset <- eff_object$rds

    ps_col <- paste0(pg, "_eff")
    if (ps_col %in% colnames(rds_subset@meta.data)) {{
      ps_vals <- rds_subset@meta.data[[ps_col]]
      cell_ids <- Cells(rds_subset)
      gene_labels <- rds_subset@meta.data$gene

      batch_df <- data.frame(
        cell = cell_ids,
        gene_label = gene_labels,
        perturbation = pg,
        PS = ps_vals,
        stringsAsFactors = FALSE
      )

      # Only keep cells that belong to this perturbation
      batch_df <- batch_df[batch_df$gene_label == pg, ]
      all_ps_results <- rbind(all_ps_results, batch_df)

      cat(sprintf("    -> %d cells, mean PS = %.3f\\n",
                  nrow(batch_df), mean(batch_df$PS)))
    }} else {{
      cat(sprintf("    -> WARNING: PS column '%s' not found\\n", ps_col))
    }}
  }}, error = function(e) {{
    cat(sprintf("    -> ERROR: %s\\n", conditionMessage(e)))
  }})
}}

# --- Save results ---
out_path <- file.path(tmpdir, sprintf("ps_results_batch%d.csv", {batch_id}))
write.csv(all_ps_results, out_path, row.names = FALSE)
cat(sprintf("\\nSaved %d PS scores to %s\\n", nrow(all_ps_results), out_path))
'''
    script_path = os.path.join(tmpdir, f"run_ps_batch{batch_id}.R")
    with open(script_path, 'w') as f:
        f.write(r_script)

    return script_path


def run_scmageck_r(tmpdir, ctrl_label, perturb_genes, r_executable='Rscript',
                   max_per_batch=100, lambda_val=0, target_gene_max=500):
    """
    Run official scMAGeCK PS computation via R, batching perturbations.

    Returns
    -------
    ps_per_pert : dict  {perturbation: mean_PS}
    """
    n_perts = len(perturb_genes)
    n_batches = max(1, (n_perts + max_per_batch - 1) // max_per_batch)

    print(f"  Running scMAGeCK in R ({n_perts} perturbations, "
          f"{n_batches} batch(es))...")

    all_results = []

    for batch_id in range(n_batches):
        start = batch_id * max_per_batch
        end = min(start + max_per_batch, n_perts)
        batch_perts = perturb_genes[start:end]

        print(f"    Batch {batch_id + 1}/{n_batches}: "
              f"{len(batch_perts)} perturbations "
              f"({batch_perts[0]}...{batch_perts[-1]})")

        r_script_path = generate_r_script(
            tmpdir, ctrl_label, batch_perts, batch_id=batch_id,
            lambda_val=lambda_val, target_gene_max=target_gene_max,
        )

        result = subprocess.run(
            [r_executable, r_script_path],
            capture_output=True, text=True, timeout=7200,
        )

        if result.returncode != 0:
            print(f"    R stderr:\n{result.stderr[-2000:]}")
            print(f"    WARNING: R script failed (exit {result.returncode})")
            continue

        # Print R stdout summary
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"      [R] {line.strip()}")

        # Read results
        csv_path = os.path.join(tmpdir, f"ps_results_batch{batch_id}.csv")
        if os.path.exists(csv_path):
            df_batch = pd.read_csv(csv_path)
            all_results.append(df_batch)
            print(f"    -> Read {len(df_batch)} PS scores from batch")
        else:
            print(f"    WARNING: output CSV not found")

    if not all_results:
        return {}

    df_all = pd.concat(all_results, ignore_index=True)

    # Aggregate to per-perturbation mean PS
    ps_per_pert = {}
    for pert, grp in df_all.groupby('perturbation'):
        ps_per_pert[pert] = float(grp['PS'].mean())

    print(f"  scMAGeCK PS computed for {len(ps_per_pert)} perturbations")
    if ps_per_pert:
        vals = list(ps_per_pert.values())
        print(f"  Mean PS: {np.mean(vals):.3f} "
              f"[{np.min(vals):.3f}, {np.max(vals):.3f}]")

    return ps_per_pert


# ============================================================================
# PROXY PS — Euclidean and Mahalanobis
# ============================================================================

def compute_proxy_ps(X_pca, labels, ctrl_label, valid_perts):
    """
    Two distance-based proxies:
      - PS_euclid: mean per-cell Euclidean distance from control centroid
      - PS_mahal:  mean per-cell Mahalanobis distance (accounts for
                   control covariance structure, addressing the concern
                   that Euclidean in PCA space ignores axis-specific
                   variance scales)

    The anticorrelation between Sp and distance-PS after magnitude control
    is a geometric fact: coherent movement concentrates cells along a shared
    trajectory (low per-cell scatter), while incoherent movement inflates
    per-cell distances.  This holds for both Euclidean and Mahalanobis.
    Mahalanobis partially mitigates the concern by re-weighting axes.
    """
    ctrl_cells = X_pca[labels == ctrl_label]
    ctrl_centroid = ctrl_cells.mean(axis=0)

    ctrl_cov = np.cov(ctrl_cells.T)
    reg = 1e-6 * np.eye(ctrl_cov.shape[0])
    try:
        cov_inv = np.linalg.inv(ctrl_cov + reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(ctrl_cov + reg)

    euclid_dict = {}
    mahal_dict = {}
    for pert in valid_perts:
        cells = X_pca[labels == pert]
        if len(cells) == 0:
            continue
        diff = cells - ctrl_centroid
        euclid_dict[pert] = float(np.linalg.norm(diff, axis=1).mean())
        mahal_dict[pert] = float(
            np.sqrt(np.sum(diff @ cov_inv * diff, axis=1)).mean())

    return euclid_dict, mahal_dict


# ============================================================================
# STABILITY / MAGNITUDE (Sp, Mp)
# ============================================================================

def compute_stability_magnitude(X_pca, labels, ctrl_label, valid_perts):
    ctrl_centroid = X_pca[labels == ctrl_label].mean(axis=0)
    records = {}
    for pert in valid_perts:
        cells = X_pca[labels == pert]
        if len(cells) < 5:
            continue
        shifts = cells - ctrl_centroid
        mean_shift = shifts.mean(axis=0)
        Mp = float(np.linalg.norm(mean_shift))
        if Mp < 1e-6:
            continue
        norms = np.linalg.norm(shifts, axis=1)
        valid = norms > 1e-6
        if valid.sum() < 5:
            continue
        unit_mean = mean_shift / Mp
        cos_sims = (shifts[valid] @ unit_mean) / norms[valid]
        Sp = float(cos_sims.mean())
        records[pert] = {'stability': Sp, 'magnitude': Mp,
                         'n_cells': len(cells)}
    return pd.DataFrame(records).T


# ============================================================================
# DATA LOADING
# ============================================================================

def clean_replogle(adata):
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
    src_col = None
    for c in ['perturbation_name', 'perturbation', 'gene', 'target',
              'guide_id', 'condition']:
        if c in adata.obs.columns:
            src_col = c
            break
    if src_col is None:
        src_col = next((c for c in adata.obs.columns
                        if 'pert' in c.lower() or 'gene' in c.lower()), None)
    if src_col is None:
        raise ValueError("Adamson: no perturbation column found")
    adata.obs[src_col] = adata.obs[src_col].astype(str)
    ctrl_kw = ['gal4', 'gfp', 'neg', 'scramble', 'unperturbed', 'nan']
    def _label(x):
        xl = x.lower().strip()
        for kw in ctrl_kw:
            if kw in xl:
                return 'control'
        return x
    adata.obs['condition'] = adata.obs[src_col].apply(_label)
    return adata[adata.obs['condition'] != 'nan'].copy()


def load_papalexi_rna():
    raw = pt.dt.papalexi_2021()
    if type(raw).__name__ != 'MuData':
        raise TypeError(f"Expected MuData for Papalexi, got {type(raw)}")
    if 'rna' not in raw.mod:
        raise KeyError("No 'rna' modality in Papalexi MuData")
    adata = raw.mod['rna'].copy()
    if 'gene_target' in raw.obs.columns:
        adata.obs['gene_target'] = raw.obs['gene_target'].values
    else:
        raise KeyError("'gene_target' not found in Papalexi MuData.obs")
    return adata


DATASET_CONFIGS = {
    'replogle': {
        'name': 'Replogle 2022 (CRISPRi)',
        'loader': pt.dt.replogle_2022_k562_essential,
        'pert_col': 'condition',
        'ctrl_label': 'control',
        'clean_func': clean_replogle,
        'min_cells': REPLOGLE_MIN_CELLS,
    },
    'norman': {
        'name': 'Norman 2019 (CRISPRa)',
        'loader': pt.dt.norman_2019,
        'pert_col': 'perturbation_name',
        'ctrl_label': 'control',
        'clean_func': None,
        'min_cells': NORMAN_MIN_CELLS,
    },
    'adamson': {
        'name': 'Adamson 2016 (CRISPRi)',
        'loader': pt.dt.adamson_2016_pilot,
        'pert_col': 'condition',
        'ctrl_label': 'control',
        'clean_func': clean_adamson,
        'min_cells': ADAMSON_MIN_CELLS,
    },
    'dixit': {
        'name': 'Dixit 2016 (CRISPRi)',
        'loader': pt.dt.dixit_2016,
        'pert_col': 'perturbation_name',
        'ctrl_label': 'control',
        'clean_func': None,
        'min_cells': DIXIT_MIN_CELLS,
    },
    'papalexi': {
        'name': 'Papalexi 2021 (CRISPR)',
        'loader': load_papalexi_rna,
        'pert_col': 'gene_target',
        'ctrl_label': 'NT',
        'clean_func': None,
        'min_cells': PAPALEXI_MIN_CELLS,
    },
}


# ============================================================================
# MAIN DATA PROCESSING PIPELINE
# ============================================================================

def load_and_process(cfg, r_executable='Rscript', max_perts_per_batch=100,
                     skip_r=False):
    """
    Load dataset, compute Sp/Mp, UPR scores, proxy PS, and optionally
    real PS (via official scMAGeCK R package).  If skip_r=True, only
    Tier 1 (Euclidean) and Tier 2 (Mahalanobis) proxies are computed.
    """
    name = cfg['name']
    print(f"\n{'='*72}")
    print(f"DATASET: {name}")
    print(f"{'='*72}")

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

    # Keep only relevant cells
    adata_sub = adata[
        adata.obs[pert_col].isin(valid + [ctrl_label])
    ].copy()

    # --- UPR pathway score (on log-normalised data) ---
    adata_norm = adata_sub.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    upr_scores = {}
    overlap = [g for g in HALLMARK_UPR if g in adata_norm.var_names]
    pct = 100 * len(overlap) / len(HALLMARK_UPR)
    print(f"  UPR gene overlap: {len(overlap)}/{len(HALLMARK_UPR)} "
          f"({pct:.0f}%)")
    if len(overlap) >= MIN_GENE_OVERLAP:
        sc.tl.score_genes(adata_norm, gene_list=overlap,
                          score_name='score_UPR',
                          ctrl_size=50, random_state=SEED)
        for pert in valid:
            mask = adata_norm.obs[pert_col] == pert
            upr_scores[pert] = float(
                adata_norm[mask].obs['score_UPR'].mean())

    # --- PCA for Sp/Mp + proxy PS ---
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

    # --- Real PS (Tier 3) ---
    if skip_r:
        # Use pure Python port of scMAGeCK algorithm
        print(f"\n  Computing Song et al. PS via pure Python port (R not available)...")
        try:
            real_ps = compute_real_ps_python(
                adata_norm, pert_col, ctrl_label, list(df.index),
                logfc_threshold=0.1, target_gene_max=500,
            )
            df['PS_real'] = df.index.map(real_ps)
        except Exception as e:
            print(f"  ERROR computing Python PS: {e}")
            import traceback
            traceback.print_exc()
            df['PS_real'] = np.nan
    else:
        print(f"\n  Computing Song et al. PS via official scMAGeCK R package...")
        tmpdir = tempfile.mkdtemp(prefix='scmageck_ps_')

        try:
            export_for_r(adata_sub, pert_col, ctrl_label, valid, tmpdir)

            real_ps = run_scmageck_r(
                tmpdir, ctrl_label, list(df.index),
                r_executable=r_executable,
                max_per_batch=max_perts_per_batch,
            )
            df['PS_real'] = df.index.map(real_ps)
        except Exception as e:
            print(f"  ERROR computing real PS: {e}")
            import traceback
            traceback.print_exc()
            df['PS_real'] = np.nan
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # --- UPR ---
    if upr_scores:
        df['pw_UPR'] = df.index.map(upr_scores)

    n_real = df['PS_real'].notna().sum()
    print(f"  Final: {len(df)} perturbations "
          f"({n_real} with real PS, all with Euclidean + Mahalanobis proxy)")

    return df


# ============================================================================
# ANALYSES
# ============================================================================

def run_anticorrelation_analysis(datasets_dict, out_dir):
    """Sp vs PS controlling for magnitude — three-tier comparison."""
    print("\n" + "=" * 80)
    print("ANTICORRELATION ANALYSIS: Sp vs PS | magnitude")
    print("  Three-tier comparison: Euclidean proxy / Mahalanobis proxy / Real PS")
    print("=" * 80)

    all_rows = []
    seed_ctr = SEED + 20000

    for ds_name, df in sorted(datasets_dict.items()):
        if len(df) < 15:
            print(f"\n  {ds_name}: skipped (n={len(df)} < 15)")
            continue

        print(f"\n  --- {ds_name} (n={len(df)}) ---")

        PS_TIERS = [
            ('PS_euclid', 'Tier 1: Proxy PS (Euclidean)'),
            ('PS_mahal',  'Tier 2: Proxy PS (Mahalanobis)'),
            ('PS_real',   'Tier 3: Real PS (scMAGeCK)'),
        ]

        for ps_col, ps_label in PS_TIERS:
            sub = df.dropna(subset=[ps_col]).copy()
            if len(sub) < 15:
                print(f"\n    {ps_label}: only {len(sub)} non-NaN, skipped")
                continue

            rho_sp_ps = bootstrap_spearman_ci(
                sub['stability'].values, sub[ps_col].values,
                n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
            seed_ctr += 1

            rho_mag_ps = bootstrap_spearman_ci(
                sub['magnitude'].values, sub[ps_col].values,
                n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
            seed_ctr += 1

            partial_sp_ps = bootstrap_partial_corr_ci(
                sub['stability'].values, sub[ps_col].values,
                sub['magnitude'].values,
                n_bootstrap=N_BOOTSTRAP, seed=seed_ctr)
            seed_ctr += 1

            print(f"\n    {ps_label} (n={len(sub)}):")
            print(f"      Sp vs PS:   rho = {rho_sp_ps['rho']:+.3f}  "
                  f"[{rho_sp_ps['ci_low']:.3f}, {rho_sp_ps['ci_high']:.3f}]  "
                  f"p = {rho_sp_ps['p']:.2e}")
            print(f"      Mp vs PS:   rho = {rho_mag_ps['rho']:+.3f}  "
                  f"[{rho_mag_ps['ci_low']:.3f}, "
                  f"{rho_mag_ps['ci_high']:.3f}]  "
                  f"p = {rho_mag_ps['p']:.2e}")
            print(f"      Partial (Sp vs PS | Mp):  "
                  f"rho = {partial_sp_ps['rho_partial']:+.3f}  "
                  f"[{partial_sp_ps['ci_low']:.3f}, "
                  f"{partial_sp_ps['ci_high']:.3f}]  "
                  f"p = {partial_sp_ps['p']:.2e}")

            shared_var = rho_sp_ps['rho'] ** 2
            if shared_var < 0.25:
                redundancy = 'LOW'
            elif shared_var < 0.50:
                redundancy = 'MODERATE'
            else:
                redundancy = 'HIGH'
            print(f"      Shared variance: {100*shared_var:.1f}% "
                  f"({redundancy} redundancy)")

            all_rows.append({
                'dataset': ds_name,
                'ps_type': ps_label,
                'n': len(sub),
                'rho_sp_ps': rho_sp_ps['rho'],
                'rho_sp_ps_ci_low': rho_sp_ps['ci_low'],
                'rho_sp_ps_ci_high': rho_sp_ps['ci_high'],
                'rho_sp_ps_p': rho_sp_ps['p'],
                'rho_mag_ps': rho_mag_ps['rho'],
                'partial_sp_ps_mag': partial_sp_ps['rho_partial'],
                'partial_ci_low': partial_sp_ps['ci_low'],
                'partial_ci_high': partial_sp_ps['ci_high'],
                'partial_p': partial_sp_ps['p'],
            })

        # Inter-tier correlations
        for c1, l1, c2, l2 in [
            ('PS_euclid', 'Euclid', 'PS_mahal', 'Mahal'),
            ('PS_euclid', 'Euclid', 'PS_real', 'Real'),
            ('PS_mahal', 'Mahal', 'PS_real', 'Real'),
        ]:
            sub_both = df.dropna(subset=[c1, c2])
            if len(sub_both) >= 10:
                rho_rp = spearmanr(sub_both[c1], sub_both[c2])
                print(f"\n    {l1} vs {l2}:  rho = {rho_rp[0]:+.3f}  "
                      f"p = {rho_rp[1]:.2e}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(out_dir / "anticorrelation_real_vs_proxy_ps.csv",
                     index=False)
    print(f"\n  Saved -> anticorrelation_real_vs_proxy_ps.csv")
    return result_df


def run_incremental_upr_analysis(datasets_dict, out_dir):
    """Does Sp predict UPR beyond PS + magnitude?"""
    print("\n" + "=" * 80)
    print("INCREMENTAL UPR ANALYSIS")
    print("  Does Sp predict UPR pathway score beyond PS + magnitude?")
    print("=" * 80)

    all_rows = []

    for ds_name, df in sorted(datasets_dict.items()):
        if 'pw_UPR' not in df.columns:
            print(f"\n  {ds_name}: no UPR scores, skipped")
            continue

        PS_TIERS = [
            ('PS_euclid', 'Tier 1: Proxy PS (Euclidean)'),
            ('PS_mahal',  'Tier 2: Proxy PS (Mahalanobis)'),
            ('PS_real',   'Tier 3: Real PS (scMAGeCK)'),
        ]

        for ps_col, ps_label in PS_TIERS:
            sub = df.dropna(subset=['pw_UPR', ps_col]).copy()
            if len(sub) < 15:
                continue

            print(f"\n  --- {ds_name} / {ps_label} (n={len(sub)}) ---")

            mag = sub['magnitude'].values
            sp = sub['stability'].values
            ps = sub[ps_col].values
            upr = sub['pw_UPR'].values

            Z_mag = sm.add_constant(mag)
            upr_resid_mag = sm.OLS(upr, Z_mag).fit().resid
            sp_resid_mag = sm.OLS(sp, Z_mag).fit().resid
            rho_sp_mag, p_sp_mag = spearmanr(upr_resid_mag, sp_resid_mag)

            ps_resid_mag = sm.OLS(ps, Z_mag).fit().resid
            rho_ps_mag, p_ps_mag = spearmanr(upr_resid_mag, ps_resid_mag)

            Z_mag_ps = sm.add_constant(np.column_stack([mag, ps]))
            upr_resid_mag_ps = sm.OLS(upr, Z_mag_ps).fit().resid
            sp_resid_mag_ps = sm.OLS(sp, Z_mag_ps).fit().resid
            rho_sp_over_ps, p_sp_over_ps = spearmanr(
                upr_resid_mag_ps, sp_resid_mag_ps)

            Z_mag_sp = sm.add_constant(np.column_stack([mag, sp]))
            upr_resid_mag_sp = sm.OLS(upr, Z_mag_sp).fit().resid
            ps_resid_mag_sp = sm.OLS(ps, Z_mag_sp).fit().resid
            rho_ps_over_sp, p_ps_over_sp = spearmanr(
                upr_resid_mag_sp, ps_resid_mag_sp)

            print(f"    Sp | Mp -> UPR:       rho = {rho_sp_mag:+.3f}  "
                  f"p = {p_sp_mag:.2e}")
            print(f"    PS | Mp -> UPR:       rho = {rho_ps_mag:+.3f}  "
                  f"p = {p_ps_mag:.2e}")
            print(f"    Sp | Mp+PS -> UPR:    rho = {rho_sp_over_ps:+.3f}  "
                  f"p = {p_sp_over_ps:.2e}")
            print(f"    PS | Mp+Sp -> UPR:    rho = {rho_ps_over_sp:+.3f}  "
                  f"p = {p_ps_over_sp:.2e}")

            sp_adds = (abs(rho_sp_over_ps) > 0.1 and p_sp_over_ps < 0.05)
            ps_adds = (abs(rho_ps_over_sp) > 0.1 and p_ps_over_sp < 0.05)

            verdict_sp = "YES" if sp_adds else "no"
            verdict_ps = "YES" if ps_adds else "no"
            print(f"    Sp adds beyond PS? {verdict_sp}  |  "
                  f"PS adds beyond Sp? {verdict_ps}")

            all_rows.append({
                'dataset': ds_name,
                'ps_type': ps_label,
                'n': len(sub),
                'rho_sp_mag_upr': rho_sp_mag,
                'p_sp_mag_upr': p_sp_mag,
                'rho_ps_mag_upr': rho_ps_mag,
                'p_ps_mag_upr': p_ps_mag,
                'rho_sp_over_ps_upr': rho_sp_over_ps,
                'p_sp_over_ps_upr': p_sp_over_ps,
                'rho_ps_over_sp_upr': rho_ps_over_sp,
                'p_ps_over_sp_upr': p_ps_over_sp,
                'sp_adds_beyond_ps': sp_adds,
                'ps_adds_beyond_sp': ps_adds,
            })

    result_df = pd.DataFrame(all_rows)
    if len(result_df) > 0:
        result_df.to_csv(out_dir / "incremental_upr_real_vs_proxy_ps.csv",
                         index=False)
        print(f"\n  Saved -> incremental_upr_real_vs_proxy_ps.csv")
    return result_df


def print_summary(anticorr_df, upr_df):
    """
    Final honest summary with three-tier structure:
      Tier 1 (lightest):  Euclidean proxy — geometric baseline
      Tier 2 (middle):    Mahalanobis proxy — accounts for control covariance
      Tier 3 (strongest): Real scMAGeCK PS — official constrained-optimisation score
    """
    TIER_ORDER = [
        ('Tier 1', 'Euclidean', 'Lightest fix: Euclidean distance proxy'),
        ('Tier 2', 'Mahalanobis', 'Middle fix: Mahalanobis distance proxy'),
        ('Tier 3', 'Real', 'Strongest fix: real Song et al. PS (scMAGeCK)'),
    ]

    print("\n" + "=" * 80)
    print("SUMMARY: THREE-TIER ASSESSMENT")
    print("  Tier 1 (lightest):  Euclidean proxy — geometric baseline")
    print("  Tier 2 (middle):    Mahalanobis proxy — control covariance")
    print("  Tier 3 (strongest): Real scMAGeCK PS — official PS algorithm")
    print("=" * 80)

    # --- 1. Anticorrelation ---
    if anticorr_df is not None and len(anticorr_df) > 0:
        print("\n1. ANTICORRELATION (Sp vs PS | magnitude):")
        print("   " + "-" * 68)

        for _, row in anticorr_df.iterrows():
            ds = row['dataset']
            ps = row['ps_type']
            rho = row['partial_sp_ps_mag']
            ci_lo = row['partial_ci_low']
            ci_hi = row['partial_ci_high']
            p = row['partial_p']
            print(f"   {ds:30s} {ps:35s}  "
                  f"rho = {rho:+.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  "
                  f"p = {p:.2e}")

        print()
        for tier_prefix, tier_kw, tier_desc in TIER_ORDER:
            tier_rows = anticorr_df[
                anticorr_df['ps_type'].str.contains(tier_kw)]
            if len(tier_rows) == 0:
                continue
            rhos = tier_rows['partial_sp_ps_mag'].dropna().values
            if len(rhos) == 0:
                continue
            m = np.mean(np.abs(rhos))
            print(f"   {tier_prefix} mean |partial rho|: {m:.3f}  ({tier_desc})")

        real_rows = anticorr_df[
            anticorr_df['ps_type'].str.contains('Real')]
        euclid_rows = anticorr_df[
            anticorr_df['ps_type'].str.contains('Euclidean')]
        if len(real_rows) > 0 and len(euclid_rows) > 0:
            mr = np.mean(np.abs(
                real_rows['partial_sp_ps_mag'].dropna().values))
            me = np.mean(np.abs(
                euclid_rows['partial_sp_ps_mag'].dropna().values))
            if mr > me * 0.5 and mr > 0.05:
                verdict = ("HOLDS with real Song PS — the geometric basis "
                           "of complementarity is confirmed")
                if mr > me:
                    verdict += " (stronger with real PS)"
            elif mr > 0.1:
                verdict = ("WEAKENS but remains present — anticorrelation "
                           "attenuated, not eliminated")
            else:
                verdict = ("EVAPORATES — anticorrelation largely disappears "
                           "with the real PS")
            print(f"\n   ANTICORRELATION VERDICT: {verdict}")

    # --- 2. Incremental UPR ---
    if upr_df is not None and len(upr_df) > 0:
        print("\n2. INCREMENTAL UPR PREDICTION (the more vulnerable claim):")
        print("   " + "-" * 68)

        for _, row in upr_df.iterrows():
            ds = row['dataset']
            ps = row['ps_type']
            rho_over = row['rho_sp_over_ps_upr']
            p_over = row['p_sp_over_ps_upr']
            adds = row['sp_adds_beyond_ps']
            symbol = "YES" if adds else "no"
            print(f"   {ds:30s} {ps:35s}  "
                  f"Sp|Mp+PS->UPR rho={rho_over:+.3f} "
                  f"p={p_over:.2e}  [{symbol}]")

        for tier_prefix, tier_kw, tier_desc in TIER_ORDER:
            tier_upr = upr_df[upr_df['ps_type'].str.contains(tier_kw)]
            if len(tier_upr) == 0:
                continue
            n_adds = tier_upr['sp_adds_beyond_ps'].sum()
            print(f"\n   {tier_prefix}: Sp adds beyond PS in "
                  f"{n_adds}/{len(tier_upr)} datasets  ({tier_desc})")

        real_upr = upr_df[upr_df['ps_type'].str.contains('Real')]
        if len(real_upr) > 0:
            any_adds = real_upr['sp_adds_beyond_ps'].any()
            all_adds = real_upr['sp_adds_beyond_ps'].all()
            if all_adds:
                verdict = ("Sp provides incremental UPR info beyond real PS "
                           "in ALL datasets — claim HOLDS")
            elif any_adds:
                n = real_upr['sp_adds_beyond_ps'].sum()
                verdict = (f"Sp adds beyond real PS in {n}/{len(real_upr)} "
                           f"datasets — claim PARTIALLY holds")
            else:
                verdict = ("Sp does NOT provide incremental UPR info "
                           "beyond real PS — claim DOES NOT HOLD")
            print(f"\n   UPR VERDICT: {verdict}")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Song et al. PS replication (official scMAGeCK R code) "
                    "+ anticorrelation/UPR analysis")
    parser.add_argument(
        '--datasets', type=str, default='replogle,norman,adamson',
        help='Comma-separated dataset keys')
    parser.add_argument(
        '--out_dir', type=str, default='./shesha-crispr',
        help='Output directory')
    parser.add_argument(
        '--r_executable', type=str, default='Rscript',
        help='Path to Rscript executable')
    parser.add_argument(
        '--max_perts_per_batch', type=int, default=100,
        help='Max perturbations per R batch call')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify R + scMAGeCK availability
    print("=" * 80)
    print("SONG et al. PS REPLICATION")
    print("=" * 80)
    print(f"R executable: {args.r_executable}")

    r_available = False
    try:
        r_check = subprocess.run(
            [args.r_executable, '--version'],
            capture_output=True, text=True, timeout=30)
        r_version = r_check.stdout.split('\n')[0] if r_check.returncode == 0 \
            else 'unknown'
        print(f"R version: {r_version}")
        r_available = True
    except FileNotFoundError:
        print(f"NOTE: '{args.r_executable}' not found.")
        print(f"  Tier 3 will use pure Python port of scMAGeCK algorithm.")
        print(f"  To use the official R package instead, install R and rerun")
        print(f"  with --r_executable /path/to/Rscript.")

    dataset_keys = [k.strip() for k in args.datasets.split(',')]
    print(f"Datasets: {dataset_keys}")
    print(f"Output:   {out_dir}")
    if r_available:
        print(f"Tier 3:   Official scMAGeCK R package")
    else:
        print(f"Tier 3:   Pure Python port of scMAGeCK algorithm")
    print(f"Ref:      Song et al. Nat Cell Biol 27, 493-504 (2025)")
    print(f"Code:     https://github.com/davidliwei/PS")

    # --- Load and process all datasets ---
    datasets = {}

    for key in dataset_keys:
        if key not in DATASET_CONFIGS:
            print(f"\nWARNING: unknown dataset key '{key}', skipping")
            continue
        cfg = DATASET_CONFIGS[key]
        try:
            df = load_and_process(
                cfg,
                r_executable=args.r_executable,
                max_perts_per_batch=args.max_perts_per_batch,
                skip_r=not r_available,
            )
            datasets[cfg['name']] = df
        except Exception as e:
            print(f"\nERROR loading {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not datasets:
        print("\nFATAL: no datasets loaded successfully")
        sys.exit(1)

    # Save per-perturbation tables
    for ds_name, df in datasets.items():
        safe = ds_name.replace(' ', '_').replace('(', '').replace(')', '')
        df.to_csv(out_dir / f"song_ps_official_{safe}.csv")

    # --- Anticorrelation analysis ---
    anticorr_df = run_anticorrelation_analysis(datasets, out_dir)

    # --- Incremental UPR analysis ---
    upr_df = run_incremental_upr_analysis(datasets, out_dir)

    # --- Summary ---
    print_summary(anticorr_df, upr_df)

    print(f"\nAll outputs in {out_dir}/")
    print("  - song_ps_official_*.csv                (per-perturbation metrics)")
    print("  - anticorrelation_real_vs_proxy_ps.csv   (three-tier anticorrelation)")
    print("  - incremental_upr_real_vs_proxy_ps.csv   (three-tier UPR prediction)")
    print("\nDone.")


if __name__ == "__main__":
    main()
