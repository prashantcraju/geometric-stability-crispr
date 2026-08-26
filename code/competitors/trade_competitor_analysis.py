#!/usr/bin/env python3
"""
TRADE competitor analysis for directional coherence (Sp).

Computes transcriptome-wide impact (TWI) from gene-level DE statistics
on the frozen cell set, then Spearman correlations with Sp, centroid
magnitude, and (if present) E-distance. Optionally correlates TWI with
Hallmark pathway scores and runs TWI-conditioned pathway partials.

TWI uses TRADE's ashr mixture; TWI = Var(β). DE is Welch log2FC on the
frozen log-normalized transcriptome (no HVG subset, no PCA).

Required inputs:
  - frozen_sp_scores.csv from run_frozen_main.py
  - pathway_scores_per_pert.csv for pathway diagnostics or partials
  - edistance_scores_per_pert.csv optional, joined when present

Examples:
  python trade_competitor_analysis.py --probe \\
      --frozen-sp /path/to/frozen_sp_scores.csv

  python trade_competitor_analysis.py --correlations-only \\
      --frozen-sp /path/to/frozen_sp_scores.csv

  python trade_competitor_analysis.py \\
      --reuse-trade-scores --twi-pathway-diagnostic \\
      --frozen-sp /path/to/frozen_sp_scores.csv \\
      --pathway-scores /path/to/pathway_scores_per_pert.csv

  python trade_competitor_analysis.py \\
      --reuse-trade-scores --run-pathway-partials \\
      --frozen-sp /path/to/frozen_sp_scores.csv \\
      --pathway-scores /path/to/pathway_scores_per_pert.csv

Methods citations:
  Nadig, Replogle, et al., Nature Genetics (2025), TRADE.
  Stephens, Biostatistics (2017), 18:275-294 (ashr).
  TRADEtools: https://github.com/ajaynadig/TRADEtools
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata, spearmanr, ttest_ind_from_stats

import pipeline_config as cfg
import stats_utils as _stats_utils
from pipeline_core import (
    _expression_matrix,
    _extract_adata,
    _filter_cells_min_genes,
    _log1p_inplace,
    _normalize_total_numpy,
    assert_frozen_sp_compatible,
    ensure_in_memory,
    load_raw,
    materialize_min_cells,
    resolve_matrix_is_log,
    setup_cache,
)
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import (
    bootstrap_partial_spearman_ci,
    bootstrap_spearman_ci,
    pathway_bootstrap_seed,
    survival_status,
)


SCORES_NAME = "trade_scores_per_pert.csv"
DATASET_TABLE_NAME = "trade_dataset_correlations.csv"
PATHWAY_TABLE_NAME = "pathway_signature_correlations_trade.csv"
TWI_PATHWAY_CORR_NAME = "trade_twi_pathway_correlations.csv"
SCOPE_NAME = "trade_competitor_scope.json"
SUMMARY_NAME = "trade_competitor_summary.json"
PARTIAL_SCORES_NAME = "trade_scores_per_pert.partial.csv"
DATASET_STATUS_NAME = "trade_dataset_status.csv"

METHOD_ID = "trade_univariate_ashr_halfuniform"
DE_METHOD_ID = "welch_log2fc_on_frozen_log_transcriptome"
LN2 = float(np.log(2.0))
PATHWAY_MIN_N = 15
DEFAULT_MIN_GENES_TRADE = 100
DEFAULT_ASH_MAX_ABS_LFC = 10.0
DEFAULT_SP_VERIFY_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _major_minor(version: str) -> tuple[int, int]:
    parts = []
    for token in str(version).split(".")[:2]:
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0])[:2])


def _guard_backed_sparse_versions() -> None:
    import anndata
    import scipy

    scipy_mm = _major_minor(scipy.__version__)
    anndata_mm = _major_minor(anndata.__version__)
    if scipy_mm >= (1, 17) and anndata_mm < (0, 13):
        raise RuntimeError(
            "Incompatible Colab stack for backed h5ad slicing: "
            f"scipy={scipy.__version__}, anndata={anndata.__version__}. "
            "Install scipy==1.14.1 (and numpy>=2.0,<2.3), then restart the "
            "runtime before rerunning."
        )


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    tmp.replace(path)


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("BH-FDR input must be one-dimensional")
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def _validate_frozen_metadata(frozen: pd.DataFrame) -> None:
    expected = {
        "seed": cfg.SEED,
        "n_pcs": cfg.N_PCS,
        "min_cells": cfg.MIN_CELLS,
    }
    for col, wanted in expected.items():
        if col not in frozen.columns:
            raise ValueError(f"Frozen Sp table lacks required provenance column {col!r}")
        got = sorted(pd.to_numeric(frozen[col], errors="coerce").dropna().unique())
        if got != [wanted]:
            raise ValueError(
                f"Frozen Sp {col}={got}, expected [{wanted}]. "
                "Refusing to mix analysis versions."
            )


def _parse_h5ad_overrides(values: Iterable[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --h5ad {value!r}; expected exact dataset name=/path/file"
            )
        dataset, raw_path = value.split("=", 1)
        dataset = cfg.resolve_dataset_name(dataset.strip())
        if dataset not in cfg.DATASETS:
            raise KeyError(f"Unknown dataset in --h5ad: {dataset!r}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        out[dataset] = path
    return out


def _resolve_datasets(requested: list[str] | None, frozen: pd.DataFrame) -> list[str]:
    present = set(frozen["dataset"].astype(str).map(cfg.resolve_dataset_name))
    if requested:
        selected = [cfg.resolve_dataset_name(x) for x in requested]
        unknown = [x for x in selected if x not in cfg.DATASETS]
        absent = [x for x in selected if x not in present]
        if unknown:
            raise KeyError(f"Unknown --dataset values: {unknown}")
        if absent:
            raise ValueError(f"Requested datasets absent from frozen table: {absent}")
        return list(dict.fromkeys(selected))
    return [name for name in cfg.DATASETS if name in present]


def _target_gene(perturbation: str) -> str:
    return str(perturbation).upper().split("_")[0]


def _gene_symbols(adata) -> np.ndarray:
    for col in ("gene_symbols", "gene_symbol", "symbol", "gene"):
        if col in adata.var.columns:
            values = adata.var[col].astype(str).str.upper().to_numpy()
            if len(np.unique(values)) > 0.5 * len(values):
                return values
    return adata.var_names.astype(str).str.upper().to_numpy()


# ---------------------------------------------------------------------------
# Transcriptome log-normalization (preprocess without HVG / PCA)
# ---------------------------------------------------------------------------

def _log_normalize_transcriptome(adata, dataset_name: str, sc, pert_col: str, ctrl_label: str, valid, counts):
    """Same normalize / log1p / min-gene QC as preprocess, all genes kept."""
    adata = ensure_in_memory(adata)
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)
    large = adata.n_obs >= cfg.LARGE_DATASET_N_OBS
    already_log, log_src = resolve_matrix_is_log(
        dataset_name=dataset_name, adata=adata
    )
    _expression_matrix(adata, allow_counts=not already_log)
    print(
        f"    TRADE transcriptome: n_obs={adata.n_obs} n_vars={adata.n_vars} "
        f"large={large} matrix_is_log={already_log} (source={log_src})",
        flush=True,
    )
    if already_log:
        print(
            f"    skipping normalize/log1p (matrix_is_log=True via {log_src})",
            flush=True,
        )
        use_scanpy_pp = False
    elif large:
        print("    sparse normalize + log1p…", flush=True)
        if adata.n_vars > 500:
            adata = _filter_cells_min_genes(adata, cfg.MIN_GENES_PER_CELL)
        _normalize_total_numpy(adata, cfg.NORMALIZE_TARGET_SUM)
        if cfg.LOG1P:
            _log1p_inplace(adata)
        use_scanpy_pp = False
    else:
        print("    filter_cells → normalize → log1p…", flush=True)
        use_scanpy_pp = True
        try:
            sc.pp.filter_cells(adata, min_genes=cfg.MIN_GENES_PER_CELL)
            if cfg.NORMALIZE_TARGET_SUM is None:
                sc.pp.normalize_total(adata)
            else:
                sc.pp.normalize_total(adata, target_sum=cfg.NORMALIZE_TARGET_SUM)
            if cfg.LOG1P:
                sc.pp.log1p(adata)
        except (AttributeError, ImportError) as exc:
            print(
                f"    scanpy.pp failed ({exc}); using numpy preprocess path",
                flush=True,
            )
            use_scanpy_pp = False
            adata = _filter_cells_min_genes(adata, cfg.MIN_GENES_PER_CELL)
            _normalize_total_numpy(adata, cfg.NORMALIZE_TARGET_SUM)
            if cfg.LOG1P:
                _log1p_inplace(adata)

    counts = adata.obs[pert_col].value_counts()
    valid = [p for p in valid if counts.get(p, 0) >= cfg.MIN_CELLS]
    keep = list(valid) + [ctrl_label]
    adata = adata[adata.obs[pert_col].isin(keep)].copy()
    return adata, valid, counts, already_log, log_src, use_scanpy_pp


def _csr_mean_var(X) -> tuple[np.ndarray, np.ndarray]:
    from scipy import sparse

    if not sparse.issparse(X):
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        var = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        return np.asarray(mean, dtype=np.float64), np.asarray(var, dtype=np.float64)
    X = X.tocsr()
    n = float(X.shape[0])
    mean = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    if n <= 1:
        return mean, np.zeros_like(mean)
    mean_sq = np.asarray(X.multiply(X).mean(axis=0), dtype=np.float64).ravel()
    var = np.maximum(mean_sq - mean**2, 0.0) * n / (n - 1.0)
    return mean, var


# ---------------------------------------------------------------------------
# DE and TRADE
# ---------------------------------------------------------------------------

def welch_log2fc(
    mean_pert: np.ndarray,
    var_pert: np.ndarray,
    n_pert: int,
    mean_ctrl: np.ndarray,
    var_ctrl: np.ndarray,
    n_ctrl: int,
    *,
    log_is_natural: bool = True,
) -> dict[str, np.ndarray]:
    """Welch log2FC and SE on a log-normalized matrix. scanpy log1p is ln."""
    scale = LN2 if log_is_natural else 1.0
    lfc = (mean_pert - mean_ctrl) / scale
    se = np.sqrt(
        np.maximum(var_pert, 0.0) / max(n_pert, 1)
        + np.maximum(var_ctrl, 0.0) / max(n_ctrl, 1)
    ) / scale
    _, pvalue = ttest_ind_from_stats(
        mean1=mean_pert,
        std1=np.sqrt(np.maximum(var_pert, 0.0)),
        nobs1=n_pert,
        mean2=mean_ctrl,
        std2=np.sqrt(np.maximum(var_ctrl, 0.0)),
        nobs2=n_ctrl,
        equal_var=False,
        alternative="two-sided",
    )
    pvalue = np.asarray(pvalue, dtype=np.float64)
    pvalue[~np.isfinite(pvalue)] = np.nan
    return {"log2FoldChange": lfc, "lfcSE": se, "pvalue": pvalue}


def _filter_trade_genes(
    lfc: np.ndarray,
    se: np.ndarray,
    pvalue: np.ndarray,
    symbols: np.ndarray,
    *,
    exclude: set[str],
    max_abs_lfc: float,
) -> dict[str, np.ndarray]:
    finite = np.isfinite(lfc) & np.isfinite(se) & (se > 0)
    extreme = np.abs(lfc) > max_abs_lfc
    excluded = np.array([s in exclude for s in symbols], dtype=bool)
    keep = finite & ~extreme & ~excluded
    return {
        "log2FoldChange": lfc[keep],
        "lfcSE": se[keep],
        "pvalue": pvalue[keep],
        "gene": symbols[keep],
        "n_input": int(len(lfc)),
        "n_kept": int(keep.sum()),
        "n_na": int((~finite).sum()),
        "n_extreme": int(extreme.sum()),
        "n_excluded_target": int(excluded.sum()),
    }


def _ash_mixsd(betahat: np.ndarray, sebetahat: np.ndarray, mult: float = np.sqrt(2.0)) -> np.ndarray:
    """ashr::autoselect.mixsd (Stephens 2017), used by TRADE_univariate."""
    sigma_min = float(np.min(sebetahat)) / 10.0
    excess = betahat**2 - sebetahat**2
    if np.all(excess <= 0):
        sigma_max = 8.0 * sigma_min
    else:
        sigma_max = 2.0 * float(np.sqrt(np.max(excess)))
    sigma_min = max(sigma_min, 1e-8)
    sigma_max = max(sigma_max, sigma_min * mult)
    npoint = int(np.ceil(np.log2(sigma_max / sigma_min) / np.log2(mult)))
    npoint = max(npoint, 1)
    return (mult ** np.arange(-npoint, 1)) * sigma_max


def _uniform_normal_likelihood(x: np.ndarray, se: np.ndarray, a: float, b: float) -> np.ndarray:
    width = b - a
    if width <= 0:
        return np.zeros_like(x)
    z_a = (x - a) / se
    z_b = (x - b) / se
    return (ndtr(z_a) - ndtr(z_b)) / width


def fit_ash_halfuniform(
    betahat: np.ndarray,
    sebetahat: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    """TRADE's ashr call: mixcompdist='halfuniform', prior='uniform'."""
    betahat = np.asarray(betahat, dtype=np.float64)
    sebetahat = np.asarray(sebetahat, dtype=np.float64)
    if len(betahat) < 5:
        raise ValueError("ashr needs at least 5 genes")
    mixsd = _ash_mixsd(betahat, sebetahat)
    components: list[tuple[str, float, float]] = [("point", 0.0, 0.0)]
    for scale in mixsd:
        scale = float(scale)
        if scale <= 0:
            continue
        components.append(("left", -scale, 0.0))
        components.append(("right", 0.0, scale))
    k = len(components)
    lik = np.empty((len(betahat), k), dtype=np.float64)
    inv_se = 1.0 / sebetahat
    lik[:, 0] = inv_se * np.exp(-0.5 * (betahat * inv_se) ** 2) / np.sqrt(2.0 * np.pi)
    for j, (kind, a, b) in enumerate(components[1:], start=1):
        lik[:, j] = _uniform_normal_likelihood(betahat, sebetahat, a, b)
    lik = np.maximum(lik, 1e-300)
    pi = np.full(k, 1.0 / k)
    loglik = -np.inf
    for _ in range(max_iter):
        weighted = lik * pi
        denom = weighted.sum(axis=1, keepdims=True)
        gamma = weighted / denom
        pi = gamma.mean(axis=0)
        pi = np.maximum(pi, 0.0)
        pi_sum = pi.sum()
        if pi_sum <= 0:
            break
        pi /= pi_sum
        new_loglik = float(np.sum(np.log(denom.ravel())))
        if abs(new_loglik - loglik) < tol * (1.0 + abs(new_loglik)):
            loglik = new_loglik
            break
        loglik = new_loglik
    a = np.array([c[1] for c in components], dtype=np.float64)
    b = np.array([c[2] for c in components], dtype=np.float64)
    return {"pi": pi, "a": a, "b": b, "loglik": loglik, "n_components": k}


def mixture_variance(pi: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """TRADE_univariate mixture mean and Var(β) = transcriptome-wide impact."""
    means = (a + b) / 2.0
    vars_ = (1.0 / 12.0) * (b - a) ** 2
    mixture_mean = float(np.sum(pi * means))
    variance_expectation = float(np.sum(pi * (means - mixture_mean) ** 2))
    expectation_variance = float(np.sum(pi * vars_))
    return mixture_mean, variance_expectation + expectation_variance


def moments_twi(lfc: np.ndarray, se: np.ndarray) -> float:
    return float(max(0.0, np.mean(lfc**2) - np.mean(se**2)))


_R_ASH_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
infile <- args[[1]]
outfile <- args[[2]]
if (!requireNamespace("ashr", quietly = TRUE)) {
  quit(save = "no", status = 2)
}
suppressPackageStartupMessages(library(ashr))
df <- read.csv(infile, stringsAsFactors = FALSE)
needed <- c("perturbation", "log2FoldChange", "lfcSE")
if (!all(needed %in% names(df))) {
  stop("DE table missing required columns")
}
rows <- list()
for (pert in unique(df$perturbation)) {
  sub <- df[df$perturbation == pert, , drop = FALSE]
  ok <- is.finite(sub$log2FoldChange) & is.finite(sub$lfcSE) &
        sub$lfcSE > 0 & abs(sub$log2FoldChange) <= 10
  sub <- sub[ok, , drop = FALSE]
  if (nrow(sub) < 5) {
    rows[[length(rows) + 1]] <- data.frame(
      perturbation = pert, twi = NA_real_, mixture_mean = NA_real_,
      loglik = NA_real_, n_genes = nrow(sub), backend = "r_ashr",
      error = "too_few_genes", stringsAsFactors = FALSE
    )
    next
  }
  fit <- tryCatch(
    ash(
      betahat = sub$log2FoldChange,
      sebetahat = sub$lfcSE,
      mixcompdist = "halfuniform",
      outputlevel = 3,
      grange = c(min(sub$log2FoldChange), max(sub$log2FoldChange)),
      prior = "uniform"
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    rows[[length(rows) + 1]] <- data.frame(
      perturbation = pert, twi = NA_real_, mixture_mean = NA_real_,
      loglik = NA_real_, n_genes = nrow(sub), backend = "r_ashr",
      error = conditionMessage(fit), stringsAsFactors = FALSE
    )
    next
  }
  means <- (fit$fitted_g$a + fit$fitted_g$b) / 2
  vars <- (1 / 12) * (fit$fitted_g$b - fit$fitted_g$a)^2
  mixture_mean <- sum(fit$fitted_g$pi * means)
  twi <- sum(fit$fitted_g$pi * (means - mixture_mean)^2) +
         sum(fit$fitted_g$pi * vars)
  rows[[length(rows) + 1]] <- data.frame(
    perturbation = pert,
    twi = as.numeric(twi),
    mixture_mean = as.numeric(mixture_mean),
    loglik = as.numeric(fit$loglik),
    n_genes = nrow(sub),
    backend = "r_ashr",
    error = "",
    stringsAsFactors = FALSE
  )
}
write.csv(do.call(rbind, rows), outfile, row.names = FALSE)
"""


def _ashr_available() -> tuple[bool, str]:
    rscript = shutil.which("Rscript")
    if not rscript:
        return False, "Rscript not on PATH"
    probe = "if (!requireNamespace('ashr', quietly=TRUE)) quit(status=2)"
    try:
        completed = subprocess.run(
            [rscript, "-e", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Rscript ashr probe failed: {exc}"
    if completed.returncode != 0:
        return False, "R package ashr is not installed"
    return True, rscript


def _fit_trade_python(lfc: np.ndarray, se: np.ndarray) -> dict:
    fit = fit_ash_halfuniform(lfc, se)
    mean, twi = mixture_variance(fit["pi"], fit["a"], fit["b"])
    return {
        "twi": float(twi),
        "mixture_mean": float(mean),
        "loglik": float(fit["loglik"]),
        "n_genes": int(len(lfc)),
        "backend": "python_ashr_halfuniform",
        "error": "",
    }


def _fit_trade_r_batch(de_tables: dict[str, dict[str, np.ndarray]], rscript: str) -> dict[str, dict]:
    frames = []
    for pert, de in de_tables.items():
        frames.append(
            pd.DataFrame(
                {
                    "perturbation": pert,
                    "log2FoldChange": de["log2FoldChange"],
                    "lfcSE": de["lfcSE"],
                }
            )
        )
    if not frames:
        return {}
    with tempfile.TemporaryDirectory(prefix="trade_ashr_") as tmp:
        tmp_path = Path(tmp)
        infile = tmp_path / "de.csv"
        outfile = tmp_path / "twi.csv"
        script = tmp_path / "fit_ash.R"
        pd.concat(frames, ignore_index=True).to_csv(infile, index=False)
        script.write_text(_R_ASH_SCRIPT)
        completed = subprocess.run(
            [rscript, str(script), str(infile), str(outfile)],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(120, 20 * len(de_tables)),
        )
        if completed.returncode != 0 or not outfile.exists():
            raise RuntimeError(
                "R ashr batch failed "
                f"(exit {completed.returncode}): {completed.stderr[-2000:]}"
            )
        out = pd.read_csv(outfile)
    results = {}
    for row in out.itertuples(index=False):
        results[str(row.perturbation)] = {
            "twi": float(row.twi) if pd.notna(row.twi) else np.nan,
            "mixture_mean": (
                float(row.mixture_mean) if pd.notna(row.mixture_mean) else np.nan
            ),
            "loglik": float(row.loglik) if pd.notna(row.loglik) else np.nan,
            "n_genes": int(row.n_genes),
            "backend": str(row.backend),
            "error": "" if pd.isna(row.error) else str(row.error),
        }
    return results


def self_check_trade() -> dict:
    rng = np.random.default_rng(cfg.SEED)
    se_null = np.full(2000, 0.4)
    null = rng.normal(0.0, se_null)
    null_fit = _fit_trade_python(null, se_null)
    true_beta = rng.normal(0.0, 0.8, size=2000)
    se_sig = np.full(2000, 0.05)
    obs = true_beta + rng.normal(0.0, se_sig)
    sig_fit = _fit_trade_python(obs, se_sig)
    true_var = float(np.var(true_beta))
    report = {
        "null_twi": null_fit["twi"],
        "signal_twi": sig_fit["twi"],
        "signal_true_var": true_var,
        "null_ok": bool(null_fit["twi"] < 0.05),
        "signal_ok": bool(abs(sig_fit["twi"] - true_var) / true_var < 0.35),
    }
    if not (report["null_ok"] and report["signal_ok"]):
        raise RuntimeError(f"TRADE self-check failed: {report}")
    return report


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def _dataset_trade_scores(
    dataset_name: str,
    frozen_sub: pd.DataFrame,
    *,
    h5ad_path: Path | None,
    trade_backend: str,
    max_perts: int | None,
    min_genes_trade: int,
    rscript: str | None,
) -> tuple[pd.DataFrame, dict]:
    import scanpy as sc

    cache = setup_cache()
    sc.settings.datasetdir = cache
    print(f"\n{'=' * 80}\n{dataset_name}\n{'=' * 80}", flush=True)
    raw = load_raw(
        dataset_name,
        sc=sc,
        prefer_local=True,
        h5ad_path=h5ad_path,
    )
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset_name, sc)
    adata, valid, counts = materialize_min_cells(
        adata,
        pert_col,
        ctrl_label,
        min_cells=cfg.MIN_CELLS,
        max_cells_per_pert=cfg.MAX_CELLS_PER_PERT,
        max_control_cells=cfg.MAX_CONTROL_CELLS,
        seed=cfg.SEED,
    )
    adata, valid, counts, already_log, log_src, _ = _log_normalize_transcriptome(
        adata, dataset_name, sc, pert_col, ctrl_label, valid, counts
    )

    labels = adata.obs[pert_col].astype(str).to_numpy()
    symbols = _gene_symbols(adata)
    X = adata.X
    from scipy import sparse

    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X, dtype=np.float64))
    else:
        X = X.tocsr()

    ctrl_mask = labels == str(ctrl_label)
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl < cfg.MIN_CONTROL_CELLS:
        raise ValueError(f"{dataset_name}: only {n_ctrl} controls after transcriptome QC")
    mean_ctrl, var_ctrl = _csr_mean_var(X[ctrl_mask])

    wanted = list(frozen_sub["perturbation"].astype(str))
    available = set(map(str, valid))
    missing = sorted(set(wanted) - available)
    if missing:
        raise RuntimeError(
            f"{dataset_name}: {len(missing)} frozen perturbations missing from "
            f"the recreated cell set (first: {missing[:5]})."
        )
    if max_perts is not None:
        wanted = wanted[: int(max_perts)]
        print(
            f"  --max-perts {max_perts}: scoring {len(wanted)} / "
            f"{len(frozen_sub)} frozen perturbations",
            flush=True,
        )

    frozen_by_pert = frozen_sub.set_index("perturbation", drop=False)
    de_tables: dict[str, dict[str, np.ndarray]] = {}
    prep_rows: list[dict] = []
    for i, pert in enumerate(wanted, start=1):
        mask = labels == pert
        n_pert = int(mask.sum())
        frozen_row = frozen_by_pert.loc[pert]
        if isinstance(frozen_row, pd.DataFrame):
            raise ValueError(f"Duplicate frozen key: {dataset_name}/{pert}")
        frozen_n = (
            int(frozen_row["n_cells"])
            if "n_cells" in frozen_row and pd.notna(frozen_row["n_cells"])
            else None
        )
        if frozen_n is not None and frozen_n != n_pert:
            raise RuntimeError(
                f"{dataset_name}/{pert}: transcriptome n_cells={n_pert} "
                f"but frozen n_cells={frozen_n}."
            )
        mean_pert, var_pert = _csr_mean_var(X[mask])
        de = welch_log2fc(
            mean_pert, var_pert, n_pert, mean_ctrl, var_ctrl, n_ctrl
        )
        filtered = _filter_trade_genes(
            de["log2FoldChange"],
            de["lfcSE"],
            de["pvalue"],
            symbols,
            exclude={_target_gene(pert)},
            max_abs_lfc=DEFAULT_ASH_MAX_ABS_LFC,
        )
        status = "ok"
        error = ""
        if filtered["n_kept"] < min_genes_trade:
            status = "too_few_genes"
            error = (
                f"kept {filtered['n_kept']} genes < min_genes_trade="
                f"{min_genes_trade}"
            )
        else:
            de_tables[pert] = {
                "log2FoldChange": filtered["log2FoldChange"],
                "lfcSE": filtered["lfcSE"],
            }
        prep_rows.append(
            {
                "dataset": dataset_name,
                "perturbation": pert,
                "stability": float(frozen_row["stability"]),
                "centroid_magnitude": float(frozen_row["magnitude"]),
                "n_cells": n_pert,
                "n_control": n_ctrl,
                "n_genes_input": filtered["n_input"],
                "n_genes_trade": filtered["n_kept"],
                "n_genes_na": filtered["n_na"],
                "n_genes_extreme": filtered["n_extreme"],
                "fit_status": status,
                "fit_error": error,
                "twi_moments": (
                    moments_twi(filtered["log2FoldChange"], filtered["lfcSE"])
                    if filtered["n_kept"] >= 5
                    else np.nan
                ),
                "seed": cfg.SEED,
                "config_version": cfg.CONFIG_VERSION,
                "sp_digest": str(frozen_row.get("sp_digest", "")),
                "de_method": DE_METHOD_ID,
                "trade_method": METHOD_ID,
                "matrix_is_log": bool(already_log),
                "matrix_is_log_source": log_src,
            }
        )
        if i % 50 == 0 or i == len(wanted):
            print(f"  DE: {i}/{len(wanted)} perturbations", flush=True)

    backend_used = "python_ashr_halfuniform"
    trade_fits: dict[str, dict] = {}
    if de_tables and trade_backend == "r":
        if not rscript:
            raise RuntimeError("R ashr requested but Rscript/ashr is unavailable")
        print(f"  fitting TRADE via R ashr for {len(de_tables)} perturbations…", flush=True)
        trade_fits = _fit_trade_r_batch(de_tables, rscript)
        backend_used = "r_ashr"
    elif de_tables:
        print(
            f"  fitting TRADE via Python ashr fallback for {len(de_tables)} "
            "perturbations…",
            flush=True,
        )
        for j, (pert, de) in enumerate(de_tables.items(), start=1):
            try:
                trade_fits[pert] = _fit_trade_python(de["log2FoldChange"], de["lfcSE"])
            except Exception as exc:  # noqa: BLE001
                trade_fits[pert] = {
                    "twi": np.nan,
                    "mixture_mean": np.nan,
                    "loglik": np.nan,
                    "n_genes": int(len(de["log2FoldChange"])),
                    "backend": "python_ashr_halfuniform",
                    "error": str(exc),
                }
            if j % 50 == 0 or j == len(de_tables):
                print(f"  TRADE: {j}/{len(de_tables)} perturbations", flush=True)

    rows = []
    for row in prep_rows:
        pert = row["perturbation"]
        fit = trade_fits.get(pert)
        if fit is None:
            row.update(
                {
                    "twi": np.nan,
                    "twi_mixture_mean": np.nan,
                    "twi_loglik": np.nan,
                    "trade_backend": backend_used,
                }
            )
        else:
            row["twi"] = fit["twi"]
            row["twi_mixture_mean"] = fit["mixture_mean"]
            row["twi_loglik"] = fit["loglik"]
            row["trade_backend"] = fit["backend"]
            row["n_genes_trade"] = fit["n_genes"]
            if fit["error"]:
                row["fit_status"] = "trade_failed"
                row["fit_error"] = fit["error"]
            elif not np.isfinite(fit["twi"]):
                row["fit_status"] = "trade_failed"
                row["fit_error"] = row["fit_error"] or "non-finite TWI"
            else:
                row["fit_status"] = "ok"
        rows.append(row)

    out = pd.DataFrame(rows)
    n_ok = int((out["fit_status"] == "ok").sum())
    status = {
        "dataset": dataset_name,
        "status": "ok" if n_ok else "failed",
        "n_frozen": int(len(frozen_sub)),
        "n_attempted": int(len(out)),
        "n_ok": n_ok,
        "n_failed": int(len(out) - n_ok),
        "trade_backend": backend_used,
        "de_method": DE_METHOD_ID,
        "n_genes_input": int(out["n_genes_input"].median()) if len(out) else 0,
        "n_control": n_ctrl,
        "matrix_is_log": bool(already_log),
        "reason": (
            ""
            if n_ok
            else "TRADE failed for every attempted perturbation; see fit_error"
        ),
    }
    print(
        f"  TRADE {status['status']}: {n_ok}/{len(out)} perturbations "
        f"(backend={backend_used}, n_genes~{status['n_genes_input']})",
        flush=True,
    )
    del adata, raw, X
    gc.collect()
    return out, status


# ---------------------------------------------------------------------------
# Correlations and pathway tables
# ---------------------------------------------------------------------------

def _rank_residual_diagnostics(
    sp: np.ndarray,
    covariates: np.ndarray,
    outcome: np.ndarray | None = None,
) -> dict:
    sp = np.asarray(sp, dtype=float).reshape(-1)
    z = np.asarray(covariates, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    arrays = [sp, *[z[:, j] for j in range(z.shape[1])]]
    y = None if outcome is None else np.asarray(outcome, dtype=float).reshape(-1)
    if y is not None:
        arrays.append(y)
    mask = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    sp, z = sp[mask], z[mask]
    if y is not None:
        y = y[mask]
    if len(sp) < 5:
        return {
            "frac_sp_variance_remaining": np.nan,
            "r2_sp_on_covariates": np.nan,
            "partial_r2": np.nan,
            "covariate_rank": np.nan,
            "covariate_condition_number": np.nan,
        }
    rsp = rankdata(sp).astype(float)
    rz = np.column_stack([rankdata(z[:, j]).astype(float) for j in range(z.shape[1])])
    design = np.column_stack([np.ones(len(sp)), rz])
    coef_sp, _, rank, singular = np.linalg.lstsq(design, rsp, rcond=None)
    e_sp = rsp - design @ coef_sp
    ss_tot = float(np.sum((rsp - np.mean(rsp)) ** 2))
    ss_res = float(np.sum(e_sp**2))
    frac = ss_res / ss_tot if ss_tot > 0 else np.nan
    partial_r2 = np.nan
    if y is not None:
        ry = rankdata(y).astype(float)
        coef_y, _, _, _ = np.linalg.lstsq(design, ry, rcond=None)
        e_y = ry - design @ coef_y
        if np.std(e_sp) >= 1e-15 and np.std(e_y) >= 1e-15:
            partial_r2 = float(np.corrcoef(e_sp, e_y)[0, 1] ** 2)
    condition = (
        float(singular[0] / singular[-1])
        if len(singular) and singular[-1] > 0
        else np.inf
    )
    return {
        "frac_sp_variance_remaining": float(frac),
        "r2_sp_on_covariates": float(1.0 - frac),
        "partial_r2": partial_r2,
        "covariate_rank": int(rank),
        "covariate_condition_number": condition,
    }


def _spearman_block(x: np.ndarray, y: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    return bootstrap_spearman_ci(x, y, n_bootstrap=n_bootstrap, ci_level=cfg.CI_LEVEL, seed=seed)


def _dataset_correlations(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for dataset, sub in scores.groupby("dataset", sort=False):
        ok = sub.dropna(subset=["stability", "centroid_magnitude", "twi"]).copy()
        n_ok = len(ok)
        row = {
            "dataset": dataset,
            "n_frozen": int(len(sub)),
            "n_trade": n_ok,
            "n_trade_failed": int(len(sub) - n_ok),
            "trade_backend": (
                str(sub["trade_backend"].dropna().iloc[0])
                if "trade_backend" in sub.columns and sub["trade_backend"].notna().any()
                else ""
            ),
            "de_method": DE_METHOD_ID,
            "config_version": cfg.CONFIG_VERSION,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": cfg.SEED,
            "method": "Spearman; frac left from rank-OLS",
        }
        if n_ok < 5:
            row.update(
                {
                    "rho_Sp_centroid_magnitude": np.nan,
                    "rho_Sp_twi": np.nan,
                    "rho_centroid_magnitude_twi": np.nan,
                    "delta_rho_Sp_twi_minus_centroid_twi": np.nan,
                }
            )
            rows.append(row)
            continue
        sp = ok["stability"].to_numpy(float)
        mag = ok["centroid_magnitude"].to_numpy(float)
        twi = ok["twi"].to_numpy(float)
        seed = int(
            hashlib.sha256(f"{cfg.SEED}|trade|{dataset}|corr".encode()).hexdigest()[:8],
            16,
        ) % (2**31 - 1)
        sp_mag = _spearman_block(sp, mag, n_bootstrap, seed)
        sp_twi = _spearman_block(sp, twi, n_bootstrap, seed + 1)
        mag_twi = _spearman_block(mag, twi, n_bootstrap, seed + 2)
        d_mag = _rank_residual_diagnostics(sp, mag)
        d_twi = _rank_residual_diagnostics(sp, twi)
        has_ed = "edistance" in ok.columns and ok["edistance"].notna().sum() >= 5
        if has_ed:
            ed = ok["edistance"].to_numpy(float)
            mask = np.isfinite(ed)
            sp_ed = _spearman_block(sp[mask], ed[mask], n_bootstrap, seed + 3)
            mag_ed = _spearman_block(mag[mask], ed[mask], n_bootstrap, seed + 4)
            twi_ed = _spearman_block(twi[mask], ed[mask], n_bootstrap, seed + 5)
            d_ed = _rank_residual_diagnostics(sp[mask], ed[mask])
            row.update(
                {
                    "rho_Sp_edistance": sp_ed["rho"],
                    "rho_Sp_edistance_ci_low": sp_ed["ci_low"],
                    "rho_Sp_edistance_ci_high": sp_ed["ci_high"],
                    "p_Sp_edistance": sp_ed["p"],
                    "rho_centroid_magnitude_edistance": mag_ed["rho"],
                    "rho_twi_edistance": twi_ed["rho"],
                    "frac_Sp_var_left_after_edistance": d_ed[
                        "frac_sp_variance_remaining"
                    ],
                }
            )
        row.update(
            {
                "rho_Sp_centroid_magnitude": sp_mag["rho"],
                "rho_Sp_centroid_magnitude_ci_low": sp_mag["ci_low"],
                "rho_Sp_centroid_magnitude_ci_high": sp_mag["ci_high"],
                "p_Sp_centroid_magnitude": sp_mag["p"],
                "rho_Sp_twi": sp_twi["rho"],
                "rho_Sp_twi_ci_low": sp_twi["ci_low"],
                "rho_Sp_twi_ci_high": sp_twi["ci_high"],
                "p_Sp_twi": sp_twi["p"],
                "rho_centroid_magnitude_twi": mag_twi["rho"],
                "rho_centroid_magnitude_twi_ci_low": mag_twi["ci_low"],
                "rho_centroid_magnitude_twi_ci_high": mag_twi["ci_high"],
                "p_centroid_magnitude_twi": mag_twi["p"],
                "delta_rho_Sp_twi_minus_centroid": sp_twi["rho"] - sp_mag["rho"],
                "delta_rho_Sp_twi_minus_centroid_twi": (
                    sp_twi["rho"] - mag_twi["rho"]
                ),
                "frac_Sp_var_left_after_centroid_magnitude": d_mag[
                    "frac_sp_variance_remaining"
                ],
                "frac_Sp_var_left_after_twi": d_twi["frac_sp_variance_remaining"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _join_edistance(scores: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return scores
    ed = pd.read_csv(path)
    required = {"dataset", "perturbation", "edistance", "config_version"}
    missing = required - set(ed.columns)
    if missing:
        raise ValueError(f"{path} is not a reusable E-distance table; missing {sorted(missing)}")
    ed["dataset"] = ed["dataset"].astype(str).map(cfg.resolve_dataset_name)
    ed["perturbation"] = ed["perturbation"].astype(str)
    versions = {
        cfg.resolve_config_version(str(x))
        for x in ed["config_version"].dropna().unique()
    }
    if versions != {cfg.CONFIG_VERSION}:
        raise ValueError(f"{path} config versions {versions} != {cfg.CONFIG_VERSION}")
    left = scores.drop(columns=["edistance"], errors="ignore")
    merged = left.merge(
        ed[["dataset", "perturbation", "edistance"]],
        on=["dataset", "perturbation"],
        how="left",
        validate="many_to_one",
    )
    if "edistance" not in merged.columns:
        raise RuntimeError(
            f"E-distance join from {path} did not produce an 'edistance' column. "
            f"Left columns={sorted(left.columns)}; "
            f"merged columns={sorted(merged.columns)}."
        )
    n_join = int(merged["edistance"].notna().sum())
    print(
        f"Joined E-distance for {n_join}/{len(merged)} TRADE rows from {path}.",
        flush=True,
    )
    return merged


def _load_pathway_scores(path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Pathway score file not found: {path}. Run pathway_analysis.py first "
            "or omit pathway flags."
        )
    pw = pd.read_csv(path)
    required = {"dataset", "perturbation"}
    missing = required - set(pw.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    pw["dataset"] = pw["dataset"].astype(str).map(cfg.resolve_dataset_name)
    pw["perturbation"] = pw["perturbation"].astype(str)
    if "config_version" not in pw.columns:
        raise ValueError(f"{path} has no config_version; refusing an unstamped join")
    versions = sorted(
        {
            cfg.resolve_config_version(str(v))
            for v in pw["config_version"].dropna().unique()
        }
    )
    if versions != [cfg.CONFIG_VERSION]:
        raise ValueError(
            f"{path} config_version resolves to {versions}, expected [{cfg.CONFIG_VERSION!r}]"
        )
    pathway_cols = sorted(c for c in pw.columns if c.startswith("pw_"))
    if not pathway_cols:
        raise ValueError(f"{path} has no pw_* pathway columns")
    keep = ["dataset", "perturbation", *pathway_cols]
    geometry_cols = [
        c
        for c in ("dataset", "perturbation", "stability", "centroid_magnitude", "twi", "edistance")
        if c in scores.columns
    ]
    merged = scores[geometry_cols].merge(
        pw[keep],
        on=["dataset", "perturbation"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("Pathway scores and TRADE scores have no shared keys")
    return merged


def _feature_descriptor(feature_col: str) -> tuple[str, str, str]:
    if feature_col.startswith("pw_"):
        return feature_col[3:], "pathway", feature_col[3:]
    raise ValueError(f"Unsupported outcome column {feature_col!r}")


def _twi_pathway_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Spearman of TWI (and centroid) with each Hallmark score."""
    pathway_cols = sorted(c for c in merged.columns if c.startswith("pw_"))
    rows = []
    for dataset, ds in merged.groupby("dataset", sort=False):
        for pathway_col in pathway_cols:
            pathway, _, _ = _feature_descriptor(pathway_col)
            sub = ds.dropna(subset=["twi", pathway_col]).copy()
            n = len(sub)
            if n < 5:
                rho_twi = p_twi = rho_mag = p_mag = np.nan
            else:
                rho_twi, p_twi = spearmanr(
                    sub["twi"].to_numpy(float),
                    sub[pathway_col].to_numpy(float),
                )
                if "centroid_magnitude" in sub.columns:
                    rho_mag, p_mag = spearmanr(
                        sub["centroid_magnitude"].to_numpy(float),
                        sub[pathway_col].to_numpy(float),
                    )
                else:
                    rho_mag = p_mag = np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "pathway": pathway,
                    "n": n,
                    "rho_twi_pathway": float(rho_twi) if np.isfinite(rho_twi) else np.nan,
                    "p_twi_pathway": float(p_twi) if np.isfinite(p_twi) else np.nan,
                    "rho_centroid_pathway": (
                        float(rho_mag) if np.isfinite(rho_mag) else np.nan
                    ),
                    "p_centroid_pathway": (
                        float(p_mag) if np.isfinite(p_mag) else np.nan
                    ),
                    "delta_abs_rho_twi_minus_centroid": (
                        abs(rho_twi) - abs(rho_mag)
                        if np.isfinite(rho_twi) and np.isfinite(rho_mag)
                        else np.nan
                    ),
                    "config_version": cfg.CONFIG_VERSION,
                }
            )
    return pd.DataFrame(rows)


def _print_twi_pathway_diagnostic(table: pd.DataFrame) -> None:
    cols = [
        c
        for c in (
            "dataset",
            "pathway",
            "n",
            "rho_twi_pathway",
            "rho_centroid_pathway",
            "delta_abs_rho_twi_minus_centroid",
        )
        if c in table.columns
    ]
    print("\n--- TWI vs pathway-score correlations ---", flush=True)
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _pathway_partials(merged: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    if _stats_utils.pg is not None:
        print(
            "Outcome partials: using NumPy rank-OLS backend "
            "(disabling pingouin inside bootstrap loop).",
            flush=True,
        )
        _stats_utils.pg = None

    models: dict[str, list[str]] = {"twi": ["twi"]}
    pathway_cols = sorted(c for c in merged.columns if c.startswith("pw_"))
    rows = []
    for dataset, ds in merged.groupby("dataset", sort=False):
        needed = ["stability", "twi"]
        for pathway_col in pathway_cols:
            pathway, feature_type, seed_key = _feature_descriptor(pathway_col)
            sub = ds.dropna(subset=[*needed, pathway_col]).copy()
            if len(sub) < PATHWAY_MIN_N:
                continue
            raw_seed = pathway_bootstrap_seed(
                dataset, seed_key, "raw", n_bootstrap=n_bootstrap
            )
            print(
                f"  bootstrap {dataset} / {pathway}: raw ({n_bootstrap:,} resamples)",
                flush=True,
            )
            raw = bootstrap_spearman_ci(
                sub["stability"].to_numpy(float),
                sub[pathway_col].to_numpy(float),
                n_bootstrap=n_bootstrap,
                ci_level=cfg.CI_LEVEL,
                seed=raw_seed,
            )
            for model, covar_cols in models.items():
                z = sub[covar_cols].to_numpy(float)
                if z.shape[1] == 1:
                    z = z[:, 0]
                seed = pathway_bootstrap_seed(
                    dataset, seed_key, "partial_twi", n_bootstrap=n_bootstrap
                )
                print(
                    f"    partial | {model} ({n_bootstrap:,} resamples)",
                    flush=True,
                )
                partial = bootstrap_partial_spearman_ci(
                    sub["stability"].to_numpy(float),
                    sub[pathway_col].to_numpy(float),
                    z,
                    n_bootstrap=n_bootstrap,
                    ci_level=cfg.CI_LEVEL,
                    seed=seed,
                    method="rank",
                )
                diag = _rank_residual_diagnostics(
                    sub["stability"].to_numpy(float),
                    z,
                    sub[pathway_col].to_numpy(float),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "pathway": pathway,
                        "outcome": pathway,
                        "feature": pathway_col,
                        "feature_type": feature_type,
                        "covariate_model": model,
                        "covariates": "|".join(covar_cols),
                        "n": len(sub),
                        "rho_raw": raw["rho"],
                        "p_raw": raw["p"],
                        "rho_raw_ci_low": raw["ci_low"],
                        "rho_raw_ci_high": raw["ci_high"],
                        "raw_bootstrap_seed": raw_seed,
                        "rho_partial": partial["rho_partial"],
                        "rho_partial_ci_low": partial["ci_low"],
                        "rho_partial_ci_high": partial["ci_high"],
                        "p_partial": partial["p"],
                        "partial_r2": diag["partial_r2"],
                        "r2_Sp_on_covariates": diag["r2_sp_on_covariates"],
                        "frac_Sp_var_left": diag["frac_sp_variance_remaining"],
                        "covariate_rank": diag["covariate_rank"],
                        "covariate_condition_number": diag[
                            "covariate_condition_number"
                        ],
                        "bootstrap_seed": seed,
                        "n_bootstrap": partial.get("n_bootstrap", 0),
                        "bootstrap_frac_valid": partial.get(
                            "bootstrap_frac_valid", np.nan
                        ),
                        "partial_method": partial.get(
                            "method", "partial_spearman_rank"
                        ),
                        "config_version": cfg.CONFIG_VERSION,
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_partial_fdr_bh"] = np.nan
    for (_, _), idx in result.groupby(
        ["dataset", "covariate_model"], sort=False
    ).groups.items():
        p = result.loc[idx, "p_partial"].to_numpy(float)
        result.loc[idx, "p_partial_fdr_bh"] = _fdr_bh(
            np.where(np.isfinite(p), p, 1.0)
        )
    status_rows = [survival_status(
        row["rho_partial"],
        row["rho_partial_ci_low"],
        row["rho_partial_ci_high"],
        fdr=row["p_partial_fdr_bh"],
    ) for _, row in result.iterrows()]
    result["survival_status"] = [s["status"] for s in status_rows]
    result["survives_covariate_control"] = [s["survives"] for s in status_rows]
    result["knife_edge_ci"] = [s["knife_edge"] for s in status_rows]
    result["ci_fdr_disagree"] = [s["ci_fdr_disagree"] for s in status_rows]
    result["survival_criterion_id"] = [s["criterion_id"] for s in status_rows]
    return result


def _load_reusable_trade_scores(
    path: Path,
    *,
    frozen: pd.DataFrame,
    frozen_info: dict,
    datasets: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Reusable TRADE score file not found: {path}. "
            "Run --correlations-only first."
        )
    scores = pd.read_csv(path)
    required = {
        "dataset",
        "perturbation",
        "config_version",
        "sp_digest",
        "twi",
        "de_method",
        "trade_method",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{path} is not reusable; missing columns {sorted(missing)}")
    scores["dataset"] = scores["dataset"].astype(str).map(cfg.resolve_dataset_name)
    scores["perturbation"] = scores["perturbation"].astype(str)
    versions = {
        cfg.resolve_config_version(str(x))
        for x in scores["config_version"].dropna().unique()
    }
    digests = set(scores["sp_digest"].dropna().astype(str).unique())
    if versions != {cfg.CONFIG_VERSION}:
        raise ValueError(f"{path} config versions {versions} != {cfg.CONFIG_VERSION}")
    if digests != {frozen_info["sp_digest"]}:
        raise ValueError(f"{path} Sp digest {digests} != {frozen_info['sp_digest']}")
    if set(scores["de_method"].dropna().astype(str).unique()) != {DE_METHOD_ID}:
        raise ValueError(f"{path} DE method does not match {DE_METHOD_ID}")
    selected = scores[scores["dataset"].isin(datasets)].copy()
    print(
        f"Reusing {len(selected):,} TRADE rows from {path}; "
        "skipping dataset loading / DE / ashr.",
        flush=True,
    )
    return selected


def _print_dataset_table(table: pd.DataFrame) -> None:
    cols = [
        c
        for c in (
            "dataset",
            "n_trade",
            "rho_Sp_twi",
            "rho_centroid_magnitude_twi",
            "delta_rho_Sp_twi_minus_centroid_twi",
            "rho_Sp_centroid_magnitude",
            "rho_Sp_edistance",
        )
        if c in table.columns
    ]
    print("\n--- TRADE competitor table ---", flush=True)
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _scope_payload(
    *,
    datasets: list[str],
    status_rows: list[dict],
    dataset_table: pd.DataFrame,
    pathway_table: pd.DataFrame,
    backend: str,
    max_perts: int | None,
    probe: bool,
    twi_pathway_corr: pd.DataFrame | None = None,
) -> dict:
    survivors = []
    if not pathway_table.empty and "survives_covariate_control" in pathway_table.columns:
        survivors = pathway_table.loc[
            pathway_table["survives_covariate_control"],
            ["dataset", "pathway", "covariate_model"],
        ].to_dict("records")
    return {
        "config_version": cfg.CONFIG_VERSION,
        "trade_backend": backend,
        "de_method": DE_METHOD_ID,
        "probe": probe,
        "max_perts": max_perts,
        "datasets_requested": datasets,
        "dataset_status": status_rows,
        "pathway_survivors": survivors,
        "n_twi_pathway_corr_rows": (
            int(len(twi_pathway_corr))
            if twi_pathway_corr is not None and not twi_pathway_corr.empty
            else 0
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frozen-sp", type=Path, default=None)
    parser.add_argument("--pathway-scores", type=Path, default=None)
    parser.add_argument("--edistance-scores", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Exact dataset display name; repeat to run a subset (default: all frozen).",
    )
    parser.add_argument(
        "--h5ad",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override one dataset file; repeat as needed.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Papalexi, --max-perts 8, correlations only.",
    )
    parser.add_argument(
        "--correlations-only",
        action="store_true",
        help="Stop after the dataset competitor table.",
    )
    parser.add_argument(
        "--run-pathway-partials",
        action="store_true",
        help="TWI-conditioned pathway partials (ci_and_fdr.v1).",
    )
    parser.add_argument(
        "--twi-pathway-diagnostic",
        action="store_true",
        help="Spearman of TWI and centroid with each Hallmark score. No bootstrap.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=cfg.N_BOOTSTRAP,
        help=f"Bootstrap replicates (default: frozen {cfg.N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--max-perts",
        type=int,
        default=None,
        help="Score only the first N frozen perturbations per dataset.",
    )
    parser.add_argument(
        "--min-genes-trade",
        type=int,
        default=DEFAULT_MIN_GENES_TRADE,
        help="Minimum genes retained after TRADE filters (default: 100).",
    )
    parser.add_argument(
        "--trade-backend",
        choices=("auto", "r", "python"),
        default="auto",
        help="ashr implementation: auto, r (R/ashr), or python.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume compatible dataset rows from {PARTIAL_SCORES_NAME}.",
    )
    parser.add_argument(
        "--reuse-trade-scores",
        action="store_true",
        help=f"Reuse a completed, validated {SCORES_NAME} from --out-dir.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run the simulated null/signal TWI guard and exit.",
    )
    parser.add_argument(
        "--skip-self-check",
        action="store_true",
        help="Skip the startup numerical guard.",
    )
    args = parser.parse_args()

    if args.self_check:
        report = self_check_trade()
        print(json.dumps(report, indent=2))
        return

    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be >=100")
    if args.min_genes_trade < 5:
        raise ValueError("--min-genes-trade must be >=5")
    if args.max_perts is not None and args.max_perts < 1:
        raise ValueError("--max-perts must be positive")
    if args.run_pathway_partials and args.correlations_only:
        raise ValueError("--run-pathway-partials cannot be combined with --correlations-only")

    if not args.skip_self_check:
        check = self_check_trade()
        print(
            f"TRADE self-check OK: null TWI={check['null_twi']:.4f}, "
            f"signal TWI={check['signal_twi']:.3f} "
            f"(true var={check['signal_true_var']:.3f})",
            flush=True,
        )

    r_ok, r_info = _ashr_available()
    if args.trade_backend == "r" and not r_ok:
        raise RuntimeError(
            " --trade-backend r requested but ashr is unavailable "
            f"({r_info}). Install with: "
            "Rscript -e \"install.packages('ashr')\""
        )
    if args.trade_backend == "auto":
        trade_backend = "r" if r_ok else "python"
    else:
        trade_backend = args.trade_backend
    rscript = r_info if (trade_backend == "r" and r_ok) else None
    print(
        f"TRADE backend={trade_backend} "
        f"({'R ashr: ' + r_info if trade_backend == 'r' else 'Python half-uniform EM'})",
        flush=True,
    )

    if not args.reuse_trade_scores:
        _guard_backed_sparse_versions()

    out_dir = resolve_out_dir(args.out_dir)
    frozen_path = find_sp_csv(out_dir, args.frozen_sp)
    frozen_info = assert_frozen_sp_compatible(frozen_path)
    frozen = load_sp_table(frozen_path)
    _validate_frozen_metadata(frozen)

    if args.probe:
        probe_ds = "Papalexi 2021 (CRISPR-KO)"
        if probe_ds not in set(frozen["dataset"]):
            raise ValueError(f"--probe requires {probe_ds} in the frozen table")
        args.dataset = [probe_ds]
        if args.max_perts is None:
            args.max_perts = 8
        args.correlations_only = True
        args.run_pathway_partials = False
        print(
            f"--probe: {probe_ds}, max_perts={args.max_perts}, correlations-only.",
            flush=True,
        )

    datasets = _resolve_datasets(args.dataset, frozen)
    h5ad_overrides = _parse_h5ad_overrides(args.h5ad)
    setup_cache()
    print(
        f"config={cfg.CONFIG_VERSION} seed={cfg.SEED} "
        f"datasets={datasets}\nfrozen_sp={frozen_path}\nout_dir={out_dir}",
        flush=True,
    )

    partial_path = out_dir / PARTIAL_SCORES_NAME
    completed: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    done: set[str] = set()
    reuse_path = out_dir / SCORES_NAME
    if (
        args.reuse_trade_scores
        and partial_path.exists()
        and reuse_path.exists()
    ):
        try:
            n_partial = len(pd.read_csv(partial_path, usecols=["dataset"]))
            n_done = len(pd.read_csv(reuse_path, usecols=["dataset"]))
            if n_partial > n_done:
                print(
                    f"Partial checkpoint {partial_path} has {n_partial} rows "
                    f"> {n_done} in {reuse_path}; using the partial.",
                    flush=True,
                )
                reuse_path = partial_path
        except Exception as exc:  # noqa: BLE001
            print(f"Could not compare TRADE checkpoints ({exc}); using {reuse_path}.", flush=True)
    if args.reuse_trade_scores:
        reused = _load_reusable_trade_scores(
            reuse_path,
            frozen=frozen,
            frozen_info=frozen_info,
            datasets=datasets,
        )
        completed.append(reused)
        done = set(reused["dataset"].astype(str).unique())
        for dataset in datasets:
            sub = reused[reused["dataset"] == dataset]
            status_rows.append(
                {
                    "dataset": dataset,
                    "status": "reused" if len(sub) else "missing",
                    "n_ok": int((sub["fit_status"] == "ok").sum()) if len(sub) else 0,
                    "n_attempted": int(len(sub)),
                    "reason": "reused existing TRADE scores",
                }
            )
    elif args.resume and partial_path.exists():
        prior = pd.read_csv(partial_path)
        required = {"dataset", "config_version", "sp_digest", "twi"}
        if not required.issubset(prior.columns):
            raise ValueError(f"Incompatible resume file: {partial_path}")
        versions = {
            cfg.resolve_config_version(str(x))
            for x in prior["config_version"].dropna().unique()
        }
        digests = set(prior["sp_digest"].dropna().astype(str).unique())
        if versions != {cfg.CONFIG_VERSION} or digests != {frozen_info["sp_digest"]}:
            raise ValueError(
                f"Resume file does not match config/digest: versions={versions}, "
                f"digests={digests}"
            )
        completed.append(prior)
        done = set(prior["dataset"].astype(str).unique())
        print(f"Resuming completed datasets: {sorted(done)}.", flush=True)

    for dataset in datasets:
        if dataset in done:
            continue
        frozen_sub = frozen[frozen["dataset"] == dataset].copy()
        try:
            result, status = _dataset_trade_scores(
                dataset,
                frozen_sub,
                h5ad_path=h5ad_overrides.get(dataset),
                trade_backend=trade_backend,
                max_perts=args.max_perts,
                min_genes_trade=args.min_genes_trade,
                rscript=rscript,
            )
            completed.append(result)
            status_rows.append(status)
            checkpoint = pd.concat(completed, ignore_index=True)
            _atomic_csv(checkpoint, partial_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  SCOPE FAILURE: {dataset}: {exc}", flush=True)
            status_rows.append(
                {
                    "dataset": dataset,
                    "status": "failed",
                    "n_frozen": int(len(frozen_sub)),
                    "n_attempted": 0,
                    "n_ok": 0,
                    "reason": str(exc),
                }
            )

    if not completed:
        scope = _scope_payload(
            datasets=datasets,
            status_rows=status_rows,
            dataset_table=pd.DataFrame(),
            pathway_table=pd.DataFrame(),
            backend=trade_backend,
            max_perts=args.max_perts,
            probe=args.probe,
        )
        _atomic_json(scope, out_dir / SCOPE_NAME)
        _atomic_csv(pd.DataFrame(status_rows), out_dir / DATASET_STATUS_NAME)
        raise RuntimeError(
            "TRADE produced no scores on any requested dataset. "
            f"See {out_dir / SCOPE_NAME}."
        )

    scores = pd.concat(completed, ignore_index=True)
    scores = scores[scores["dataset"].isin(datasets)].copy()
    score_path = out_dir / SCORES_NAME
    _atomic_csv(scores.drop(columns=["edistance"], errors="ignore"), score_path)
    if args.edistance_scores is not None:
        edistance_path = Path(args.edistance_scores)
    else:
        edistance_path = next(
            (
                path
                for path in (
                    out_dir / "edistance_scores_per_pert.csv",
                    cfg.ROOT / "edistance_scores_per_pert.csv",
                )
                if path.exists()
            ),
            None,
        )
    scores = _join_edistance(scores, edistance_path)
    scores = scores.sort_values(["dataset", "perturbation"], kind="mergesort")
    _atomic_csv(scores, score_path)

    correlation_n_bootstrap = min(2000, args.n_bootstrap)
    dataset_table = _dataset_correlations(scores, correlation_n_bootstrap)
    dataset_path = out_dir / DATASET_TABLE_NAME
    _atomic_csv(dataset_table, dataset_path)
    _print_dataset_table(dataset_table)

    pathway_path = None
    pathway_table = pd.DataFrame()
    twi_pathway_corr = pd.DataFrame()
    twi_pathway_corr_path = None
    if args.twi_pathway_diagnostic or args.run_pathway_partials:
        source = args.pathway_scores or (out_dir / "pathway_scores_per_pert.csv")
        merged = _load_pathway_scores(source, scores)
        twi_pathway_corr = _twi_pathway_correlations(merged)
        twi_pathway_corr_path = out_dir / TWI_PATHWAY_CORR_NAME
        _atomic_csv(twi_pathway_corr, twi_pathway_corr_path)
        _print_twi_pathway_diagnostic(twi_pathway_corr)
    if args.run_pathway_partials:
        if twi_pathway_corr.empty:
            source = args.pathway_scores or (out_dir / "pathway_scores_per_pert.csv")
            merged = _load_pathway_scores(source, scores)
        pathway_table = _pathway_partials(merged, args.n_bootstrap)
        if pathway_table.empty:
            raise RuntimeError("No pathway partials were scoreable")
        pathway_path = out_dir / PATHWAY_TABLE_NAME
        _atomic_csv(pathway_table, pathway_path)
        print("\n--- Pathway partials (covariate_model=twi) ---", flush=True)
        show = [
            c
            for c in (
                "dataset",
                "pathway",
                "n",
                "rho_partial",
                "rho_partial_ci_low",
                "rho_partial_ci_high",
                "p_partial_fdr_bh",
                "survival_status",
                "survives_covariate_control",
            )
            if c in pathway_table.columns
        ]
        print(pathway_table[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    scope = _scope_payload(
        datasets=datasets,
        status_rows=status_rows,
        dataset_table=dataset_table,
        pathway_table=pathway_table,
        backend=trade_backend,
        max_perts=args.max_perts,
        probe=args.probe,
        twi_pathway_corr=twi_pathway_corr,
    )
    scope_path = out_dir / SCOPE_NAME
    status_path = out_dir / DATASET_STATUS_NAME
    _atomic_json(scope, scope_path)
    _atomic_csv(pd.DataFrame(status_rows), status_path)

    summary = {
        "config_version": cfg.CONFIG_VERSION,
        "seed": cfg.SEED,
        "frozen_sp": str(frozen_path),
        "sp_digest": frozen_info["sp_digest"],
        "datasets": datasets,
        "trade_method": METHOD_ID,
        "de_method": DE_METHOD_ID,
        "trade_backend": trade_backend,
        "n_bootstrap": args.n_bootstrap,
        "dataset_correlation_n_bootstrap": correlation_n_bootstrap,
        "max_perts": args.max_perts,
        "probe": bool(args.probe),
        "outputs": {
            "scores": str(score_path),
            "dataset_correlations": str(dataset_path),
            "dataset_status": str(status_path),
            "scope": str(scope_path),
            "pathway_partials": str(pathway_path) if pathway_path else None,
            "twi_pathway_diagnostic": (
                str(twi_pathway_corr_path) if twi_pathway_corr_path else None
            ),
        },
    }
    _atomic_json(summary, out_dir / SUMMARY_NAME)
    partial_path.unlink(missing_ok=True)

    print(
        f"\nWrote:\n  {score_path}\n  {dataset_path}\n  {status_path}\n  {scope_path}"
        + (f"\n  {pathway_path}" if pathway_path else "")
        + (f"\n  {twi_pathway_corr_path}" if twi_pathway_corr_path else "")
        + f"\n  {out_dir / SUMMARY_NAME}",
        flush=True,
    )
    failed = [s for s in status_rows if s.get("status") == "failed"]
    if failed:
        print("\nDatasets TRADE could not score:", flush=True)
        for row in failed:
            print(f"  {row['dataset']}: {row.get('reason', '')}", flush=True)


if __name__ == "__main__":
    main()
