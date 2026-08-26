#!/usr/bin/env python3
"""
E-distance competitor analysis for directional coherence (Sp).

This is a standalone manuscript-sensitivity driver. It:

1. Recreates each dataset's frozen, hash-selected cell set and 50-PC embedding.
2. Computes scPerturb/pertpy E-distance from every perturbation to control.
3. Reports, per dataset, centroid-magnitude~E-distance, Sp~centroid-magnitude,
   and Sp~E-distance Spearman correlations plus rank-OLS frac_Sp_var_left.
4. Re-runs pathway partials with (a) centroid magnitude, (b) E-distance, and
   (c) both covariates.
5. Optionally adds E-distance+QC and centroid+E-distance+QC beside the existing
   centroid+QC gate, reporting the intersection across all three as the only
   primary survivor set.
6. Applies that identical model family and intersection rule to section-4
   stress markers when --include-stress-qc is requested.

The full frozen control set (up to 5,000 cells) is retained by default.
Calling pertpy independently for every perturbation would recompute the same
control-control distance term thousands of times, so the script evaluates the
same energy-distance formula in blocks and caches that term. For every dataset,
the implementation is checked against an independent sklearn reference that
matches pertpy's Edistance (off-diagonal within-group means). Live
``import pertpy`` is optional and often broken on Colab when scanpy drifts.

E-distance is never labelled "magnitude" in outputs. Centroid magnitude remains
the frozen primary measure; E-distance is an added competitor.

Required inputs:
  - frozen_sp_scores.csv from run_frozen_main.py
  - pathway_scores_per_pert.csv from pathway_analysis.py (unless
    --correlations-only or --reuse-pathway-partials)
  - cell_quality_per_perturbation.csv and cell_quality_partials.csv when
    --run-qc-models is requested; these must have been generated with
    cell_quality_partial.py --include-stress for --include-stress-qc

Examples:
  python edistance_competitor_analysis.py \
      --frozen-sp /path/to/frozen_sp_scores.csv \
      --pathway-scores /path/to/pathway_scores_per_pert.csv

  # Stage 1 only: inspect E-distance~centroid-magnitude before pathway partials.
  python edistance_competitor_analysis.py --correlations-only \
      --frozen-sp /path/to/frozen_sp_scores.csv

  # Fast QC extension after the E-distance and cell-quality runs exist.
  python edistance_competitor_analysis.py \
      --frozen-sp /path/to/frozen_sp_scores.csv \
      --out-dir /path/to/results \
      --reuse-edistance-scores --reuse-pathway-partials --run-qc-models \
      --include-stress-qc

Methods citations:
  Peidli et al., Nature Methods (2024), 21:531-540 (scPerturb method).
  Garbulowski et al. (2024), lqae121
  (GeneSPIDER2 usage precedent).
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
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

import pipeline_config as cfg
import stats_utils as _stats_utils
from pipeline_core import (
    _extract_adata,
    assert_frozen_sp_compatible,
    calculate_sp,
    load_raw,
    materialize_min_cells,
    preprocess,
    setup_cache,
)
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import (
    bootstrap_partial_spearman_ci,
    bootstrap_spearman_ci,
    partial_spearman_rank,
    pathway_bootstrap_seed,
    survival_status,
)


SCORES_NAME = "edistance_scores_per_pert.csv"
DATASET_TABLE_NAME = "edistance_dataset_correlations.csv"
PATHWAY_TABLE_NAME = "pathway_signature_correlations_edistance.csv"
QC_TABLE_NAME = "pathway_qc_partials_edistance.csv"
QC_INTERSECTION_NAME = "pathway_qc_all_model_survivors.csv"
STRESS_QC_TABLE_NAME = "stress_marker_qc_partials_edistance.csv"
STRESS_QC_INTERSECTION_NAME = "stress_marker_qc_all_model_survivors.csv"
QC_SUMMARY_NAME = "pathway_qc_edistance_summary.json"
SUMMARY_NAME = "edistance_competitor_summary.json"
PARTIAL_SCORES_NAME = "edistance_scores_per_pert.partial.csv"

METHOD_ID = "energy_distance_u_statistic_euclidean"  # pertpy Edistance / Peidli
VALIDATION_CELL_CAP = 100
DEFAULT_BLOCK_SIZE = 2048
DEFAULT_SP_VERIFY_ATOL = 1e-3
DEFAULT_MAG_VERIFY_ATOL = 1e-3
PATHWAY_MIN_N = 15

INTERPRETATION_RULES = {
    "comparable_and_partials_survive": (
        "Robustness paragraph; no reframe."
    ),
    "Sp_edistance_substantially_higher_and_partials_collapse": (
        "Coherence is largely recoverable from a dispersion-aware effect size; "
        "the pathway result becomes E-distance-conditional."
    ),
    "Sp_edistance_lower": (
        "Sp captures directional structure that the distributional effect size misses."
    ),
    "guard": (
        "Centroid magnitude remains primary. E-distance is the added competitor; "
        "do not select whichever result is friendlier."
    ),
    "qc_caveat": (
        "E-distance is cell-quality-sensitive because noisy cells inflate "
        "within-group dispersion; conditioning on it can absorb QC variation."
    ),
}


def _major_minor(version: str) -> tuple[int, int]:
    """Parse the numeric major/minor prefix without adding a dependency."""
    parts = []
    for token in str(version).split(".")[:2]:
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0])[:2])


def _guard_backed_sparse_versions() -> None:
    """
    Refuse the known anndata-backed/SciPy combination before loading large data.

    SciPy 1.17 removed sparse ``_validate_indices``; anndata 0.11.x still calls
    it when slicing backed CSR matrices. The frozen environment pins 1.14.1.
    """
    import anndata
    import scipy

    scipy_mm = _major_minor(scipy.__version__)
    anndata_mm = _major_minor(anndata.__version__)
    if scipy_mm >= (1, 17) and anndata_mm < (0, 13):
        raise RuntimeError(
            "Incompatible Colab stack for backed h5ad slicing: "
            f"scipy={scipy.__version__}, anndata={anndata.__version__}. "
            "Install scipy==1.14.1 (and numpy>=2.0,<2.3), then restart the "
            "runtime before rerunning. This guard prevents anndata's "
            "'backed_csr_matrix has no _validate_indices' crash."
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
    """Benjamini-Hochberg adjusted p values, preserving input order."""
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


def _mean_between_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> float:
    """Mean Euclidean distance over all ordered x-y pairs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Incompatible distance arrays: x={x.shape}, y={y.shape}")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Cannot compute a group distance with an empty group")

    total = 0.0
    n_pairs = 0
    for start in range(0, len(x), block_size):
        d = cdist(x[start : start + block_size], y, metric="euclidean")
        total += float(np.sum(d, dtype=np.float64))
        n_pairs += int(d.size)
    return total / n_pairs


def _mean_within_distance(
    x: np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> float:
    """
    Mean Euclidean distance over distinct pairs (i ≠ j).

    Matches pertpy Edistance / Peidli: within-group means exclude the
    diagonal. Equivalent to sklearn pairwise_distances(X,X).sum() / (n*(n-1)).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 0.0
    total = 0.0
    for start in range(0, n, block_size):
        d = cdist(x[start : start + block_size], x, metric="euclidean")
        total += float(np.sum(d, dtype=np.float64))
    # Diagonal contributions are exactly 0, so divide by off-diagonal count.
    return total / (n * (n - 1))


def energy_distance(
    perturbation: np.ndarray,
    control: np.ndarray,
    *,
    control_within_mean: float | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> float:
    """
    scPerturb / pertpy Edistance:
      2 E||X-Y|| - E_{i≠j}||X_i-X_j|| - E_{i≠j}||Y_i-Y_j||.

    This is the unbiased finite-sample U-statistic used by pertpy. It can be
    negative when the two distributions are close; pertpy returns that signed
    estimate unchanged. Do not clamp it to zero.
    """
    pert = np.asarray(perturbation, dtype=np.float64)
    ctrl = np.asarray(control, dtype=np.float64)
    between = _mean_between_distance(pert, ctrl, block_size=block_size)
    within_pert = _mean_within_distance(pert, block_size=block_size)
    within_ctrl = (
        _mean_within_distance(ctrl, block_size=block_size)
        if control_within_mean is None
        else float(control_within_mean)
    )
    return float(2.0 * between - within_pert - within_ctrl)


def _reference_energy_distance_sklearn(pert: np.ndarray, ctrl: np.ndarray) -> float:
    """Independent reference matching pertpy Edistance.__call__ (no pertpy import)."""
    from sklearn.metrics import pairwise_distances

    pert = np.asarray(pert, dtype=np.float64)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    n_p, n_c = len(pert), len(ctrl)
    if n_p < 1 or n_c < 1:
        raise ValueError("Empty group in reference E-distance")
    between = float(pairwise_distances(pert, ctrl, metric="euclidean").mean())
    within_p = (
        0.0
        if n_p < 2
        else float(pairwise_distances(pert, pert, metric="euclidean").sum())
        / (n_p * (n_p - 1))
    )
    within_c = (
        0.0
        if n_c < 2
        else float(pairwise_distances(ctrl, ctrl, metric="euclidean").sum())
        / (n_c * (n_c - 1))
    )
    # Match pertpy exactly: the unbiased estimate is allowed to be negative.
    return float(2.0 * between - within_p - within_c)


def _try_pertpy_version() -> str | None:
    """Best-effort version stamp; never required for scoring."""
    try:
        import importlib.metadata as md

        return md.version("pertpy")
    except Exception:  # noqa: BLE001
        return None


def _deterministic_rows(
    matrix: np.ndarray, names: np.ndarray, cap: int, salt: str
) -> np.ndarray:
    if len(matrix) <= cap:
        return matrix
    keyed = []
    for i, name in enumerate(names.astype(str)):
        digest = hashlib.blake2b(
            f"{cfg.SEED}|{salt}|{name}".encode("utf-8"), digest_size=8
        ).digest()
        keyed.append((digest, i))
    keyed.sort()
    idx = [i for _, i in keyed[:cap]]
    return np.asarray(matrix)[idx]


def _validate_energy_distance_formula(
    pert: np.ndarray,
    pert_names: np.ndarray,
    ctrl: np.ndarray,
    ctrl_names: np.ndarray,
    *,
    dataset_name: str,
    block_size: int,
) -> dict:
    """
    Check blocked E-distance against an independent sklearn reference.

    Does not import pertpy (Colab often has scanpy/pertpy API drift that makes
    ``import pertpy`` fail in Mixscape while datasets still load via
    pipeline_core.import_pertpy_datasets).
    """
    p = _deterministic_rows(
        pert, pert_names, VALIDATION_CELL_CAP, f"{dataset_name}|pert"
    )
    c = _deterministic_rows(
        ctrl, ctrl_names, VALIDATION_CELL_CAP, f"{dataset_name}|ctrl"
    )
    ours = energy_distance(p, c, block_size=block_size)
    ref = _reference_energy_distance_sklearn(p, c)
    delta = abs(ours - ref)
    tolerance = 1e-8 * max(1.0, abs(ref))
    if not np.isfinite(ref) or delta > tolerance:
        raise RuntimeError(
            f"E-distance formula validation failed for {dataset_name}: "
            f"blocked={ours:.12g}, sklearn_ref={ref:.12g}, |delta|={delta:.3g}, "
            f"tol={tolerance:.3g}"
        )
    return {
        "pertpy_version": _try_pertpy_version(),
        "validation_method": "sklearn_pairwise_u_statistic",
        "n_pert_validation": int(len(p)),
        "n_control_validation": int(len(c)),
        "cached_edistance": ours,
        "reference_edistance": ref,
        "abs_delta": delta,
        "tolerance": tolerance,
    }


def _rank_residual_diagnostics(
    sp: np.ndarray,
    covariates: np.ndarray,
    outcome: np.ndarray | None = None,
) -> dict:
    """Rank-OLS fraction of Sp variance left, optionally partial outcome R²."""
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


def _dataset_embedding_and_scores(
    dataset_name: str,
    frozen_sub: pd.DataFrame,
    *,
    h5ad_path: Path | None,
    block_size: int,
    edistance_control_cap: int | None,
    sp_verify_atol: float,
    mag_verify_atol: float,
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
    adata, valid, counts = preprocess(
        adata,
        pert_col,
        ctrl_label,
        sc,
        n_pcs=cfg.N_PCS,
        min_cells=cfg.MIN_CELLS,
        seed=cfg.SEED,
        valid_perts=valid,
        counts=counts,
        dataset_name=dataset_name,
    )

    labels = adata.obs[pert_col].astype(str).to_numpy()
    names = adata.obs_names.astype(str).to_numpy()
    x_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    if x_pca.ndim != 2 or x_pca.shape[1] != cfg.N_PCS:
        raise RuntimeError(
            f"{dataset_name}: recreated embedding has shape {x_pca.shape}; "
            f"expected exactly {cfg.N_PCS} PCs."
        )
    ctrl_mask = labels == str(ctrl_label)
    x_ctrl_full = x_pca[ctrl_mask]
    ctrl_names_full = names[ctrl_mask]
    if len(x_ctrl_full) < cfg.MIN_CONTROL_CELLS:
        raise ValueError(f"{dataset_name}: only {len(x_ctrl_full)} controls after preprocessing")

    if edistance_control_cap is not None and len(x_ctrl_full) > edistance_control_cap:
        x_ctrl = _deterministic_rows(
            x_ctrl_full,
            ctrl_names_full,
            edistance_control_cap,
            f"{dataset_name}|edistance-control",
        )
        # Recover names in the same deterministic order for validation provenance.
        # Names affect only validation sampling; use stable synthetic names here.
        ctrl_names = np.array([f"selected_control_{i}" for i in range(len(x_ctrl))])
    else:
        x_ctrl, ctrl_names = x_ctrl_full, ctrl_names_full

    wanted = set(frozen_sub["perturbation"].astype(str))
    available = set(map(str, valid))
    missing = sorted(wanted - available)
    if missing:
        raise RuntimeError(
            f"{dataset_name}: {len(missing)} frozen perturbations missing from recreated "
            f"cell set (first: {missing[:5]})."
        )

    first_pert = sorted(wanted)[0]
    first_mask = labels == first_pert
    validation = _validate_energy_distance_formula(
        x_pca[first_mask],
        names[first_mask],
        x_ctrl,
        ctrl_names,
        dataset_name=dataset_name,
        block_size=block_size,
    )
    print(
        f"  E-distance formula validation: "
        f"method={validation['validation_method']} "
        f"|delta|={validation['abs_delta']:.3g} "
        f"(pertpy package version={validation['pertpy_version']!r})",
        flush=True,
    )

    print(
        f"  caching control-control term: n_control={len(x_ctrl)} "
        f"(frozen controls={len(x_ctrl_full)})",
        flush=True,
    )
    ctrl_within = _mean_within_distance(x_ctrl, block_size=block_size)
    frozen_by_pert = frozen_sub.set_index("perturbation", drop=False)
    rows = []
    for i, pert in enumerate(sorted(wanted), start=1):
        mask = labels == pert
        x_pert = x_pca[mask]
        metrics = calculate_sp(x_ctrl_full, x_pert)
        e = energy_distance(
            x_pert,
            x_ctrl,
            control_within_mean=ctrl_within,
            block_size=block_size,
        )
        frozen_row = frozen_by_pert.loc[pert]
        if isinstance(frozen_row, pd.DataFrame):
            raise ValueError(f"Duplicate frozen key: {dataset_name}/{pert}")
        if "n_cells" in frozen_row and pd.notna(frozen_row["n_cells"]):
            frozen_n_cells = int(frozen_row["n_cells"])
            if frozen_n_cells != len(x_pert):
                raise RuntimeError(
                    f"{dataset_name}/{pert}: recreated n_cells={len(x_pert)} "
                    f"but frozen n_cells={frozen_n_cells}."
                )
        if "n_control" in frozen_row and pd.notna(frozen_row["n_control"]):
            frozen_n_control = int(frozen_row["n_control"])
            if frozen_n_control != len(x_ctrl_full):
                raise RuntimeError(
                    f"{dataset_name}/{pert}: recreated n_control={len(x_ctrl_full)} "
                    f"but frozen n_control={frozen_n_control}."
                )
        rows.append(
            {
                "dataset": dataset_name,
                "perturbation": pert,
                "stability": float(frozen_row["stability"]),
                "centroid_magnitude": float(frozen_row["magnitude"]),
                "edistance": e,
                "recomputed_stability": float(metrics["stability"]),
                "recomputed_centroid_magnitude": float(metrics["magnitude"]),
                "abs_delta_stability": abs(
                    float(metrics["stability"]) - float(frozen_row["stability"])
                ),
                "abs_delta_centroid_magnitude": abs(
                    float(metrics["magnitude"]) - float(frozen_row["magnitude"])
                ),
                "n_cells": int(len(x_pert)),
                "n_control_frozen_embedding": int(len(x_ctrl_full)),
                "n_control_edistance": int(len(x_ctrl)),
                "n_pcs": cfg.N_PCS,
                "seed": cfg.SEED,
                "config_version": cfg.CONFIG_VERSION,
                "sp_digest": str(frozen_row.get("sp_digest", "")),
                "edistance_method": METHOD_ID,
                "pertpy_version": validation["pertpy_version"],
                "edistance_validation_abs_delta": validation["abs_delta"],
                "edistance_validation_method": validation["validation_method"],
            }
        )
        if i % 100 == 0 or i == len(wanted):
            print(f"  E-distance: {i}/{len(wanted)} perturbations", flush=True)

    out = pd.DataFrame(rows)
    max_sp = float(out["abs_delta_stability"].max())
    max_mag = float(out["abs_delta_centroid_magnitude"].max())
    if max_sp > sp_verify_atol or max_mag > mag_verify_atol:
        raise RuntimeError(
            f"{dataset_name}: frozen embedding verification failed: "
            f"max |delta Sp|={max_sp:.3g} (tol={sp_verify_atol}), "
            f"max |delta centroid magnitude|={max_mag:.3g} (tol={mag_verify_atol})."
        )
    print(
        f"  frozen verification OK: max |delta Sp|={max_sp:.3g}, "
        f"max |delta centroid magnitude|={max_mag:.3g}",
        flush=True,
    )
    validation.update(
        {
            "dataset": dataset_name,
            "n_control_frozen_embedding": int(len(x_ctrl_full)),
            "n_control_edistance": int(len(x_ctrl)),
            "control_within_mean": ctrl_within,
            "max_abs_delta_frozen_sp": max_sp,
            "max_abs_delta_frozen_centroid_magnitude": max_mag,
        }
    )
    del adata, raw, x_pca, x_ctrl_full, x_ctrl
    gc.collect()
    return out, validation


def _dataset_correlations(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for dataset, sub in scores.groupby("dataset", sort=False):
        sub = sub.dropna(
            subset=["stability", "centroid_magnitude", "edistance"]
        ).copy()
        sp = sub["stability"].to_numpy(float)
        mag = sub["centroid_magnitude"].to_numpy(float)
        ed = sub["edistance"].to_numpy(float)

        mag_ed = bootstrap_spearman_ci(
            mag,
            ed,
            n_bootstrap=n_bootstrap,
            seed=cfg.SEED,
        )
        sp_mag = bootstrap_spearman_ci(
            sp,
            mag,
            n_bootstrap=n_bootstrap,
            seed=cfg.SEED,
        )
        sp_ed = bootstrap_spearman_ci(
            sp,
            ed,
            n_bootstrap=n_bootstrap,
            seed=cfg.SEED,
        )
        d_mag = _rank_residual_diagnostics(sp, mag)
        d_ed = _rank_residual_diagnostics(sp, ed)
        rows.append(
            {
                "dataset": dataset,
                "n": len(sub),
                "rho_centroid_magnitude_edistance": mag_ed["rho"],
                "rho_centroid_magnitude_edistance_ci_low": mag_ed["ci_low"],
                "rho_centroid_magnitude_edistance_ci_high": mag_ed["ci_high"],
                "p_centroid_magnitude_edistance": mag_ed["p"],
                "rho_Sp_centroid_magnitude": sp_mag["rho"],
                "rho_Sp_centroid_magnitude_ci_low": sp_mag["ci_low"],
                "rho_Sp_centroid_magnitude_ci_high": sp_mag["ci_high"],
                "p_Sp_centroid_magnitude": sp_mag["p"],
                "rho_Sp_edistance": sp_ed["rho"],
                "rho_Sp_edistance_ci_low": sp_ed["ci_low"],
                "rho_Sp_edistance_ci_high": sp_ed["ci_high"],
                "p_Sp_edistance": sp_ed["p"],
                "delta_abs_rho_Sp_edistance_vs_centroid_magnitude": (
                    abs(sp_ed["rho"]) - abs(sp_mag["rho"])
                ),
                "frac_Sp_var_left_after_centroid_magnitude": d_mag[
                    "frac_sp_variance_remaining"
                ],
                "frac_Sp_var_left_after_edistance": d_ed[
                    "frac_sp_variance_remaining"
                ],
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": cfg.SEED,
                "config_version": cfg.CONFIG_VERSION,
                "method": "Spearman; frac left from rank-OLS",
            }
        )
    return pd.DataFrame(rows)


def _load_pathway_scores(path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Pathway score file not found: {path}. Run pathway_analysis.py first "
            "or use --correlations-only."
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
    if pw.duplicated(["dataset", "perturbation"]).any():
        dup = pw.loc[
            pw.duplicated(["dataset", "perturbation"], keep=False),
            ["dataset", "perturbation"],
        ].head()
        raise ValueError(f"Duplicate pathway-score keys:\n{dup}")

    pathway_cols = sorted(c for c in pw.columns if c.startswith("pw_"))
    if not pathway_cols:
        raise ValueError(f"{path} has no pw_* pathway columns")

    geometry = scores[
        ["dataset", "perturbation", "stability", "centroid_magnitude", "edistance"]
    ]
    merged = geometry.merge(
        pw[["dataset", "perturbation", *pathway_cols]],
        on=["dataset", "perturbation"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_rows = merged["_merge"] != "both"
    if missing_rows.any():
        examples = merged.loc[
            missing_rows, ["dataset", "perturbation"]
        ].head().to_dict("records")
        raise RuntimeError(
            f"Pathway scores missing for {int(missing_rows.sum())} E-distance rows; "
            f"examples={examples}"
        )
    return merged.drop(columns="_merge")


def _feature_descriptor(feature_col: str) -> tuple[str, str, str]:
    if feature_col.startswith("pw_"):
        return feature_col[3:], "pathway", feature_col[3:]
    if feature_col.startswith("stress_"):
        marker = feature_col[7:]
        return marker, "stress_marker", feature_col
    raise ValueError(f"Unsupported outcome column {feature_col!r}")


def _pathway_partials(
    merged: pd.DataFrame,
    n_bootstrap: int,
    *,
    feature_prefixes: tuple[str, ...] = ("pw_",),
) -> pd.DataFrame:
    # stats_utils prefers pingouin when installed. That is point-estimate
    # equivalent, but its DataFrame construction inside every bootstrap draw
    # makes 75 × 10,000 resamples take hours. Force the canonical NumPy
    # rank→OLS-residualize→Pearson implementation documented as equivalent in
    # stats_utils.partial_spearman_rank.
    if _stats_utils.pg is not None:
        print(
            "Outcome partials: using equivalent NumPy rank-OLS backend "
            "(disabling pingouin inside bootstrap loop).",
            flush=True,
        )
        _stats_utils.pg = None

    models = {
        "centroid_magnitude": ["centroid_magnitude"],
        "edistance": ["edistance"],
        "centroid_magnitude+edistance": ["centroid_magnitude", "edistance"],
    }
    pathway_cols = sorted(
        c for c in merged.columns if c.startswith(feature_prefixes)
    )
    rows = []
    for dataset, ds in merged.groupby("dataset", sort=False):
        for pathway_col in pathway_cols:
            pathway, feature_type, seed_key = _feature_descriptor(pathway_col)
            # Common complete-case set makes the three covariate models directly
            # comparable and avoids repeating the identical raw bootstrap.
            sub = ds.dropna(
                subset=[
                    "stability",
                    pathway_col,
                    "centroid_magnitude",
                    "edistance",
                ]
            ).copy()
            if len(sub) < PATHWAY_MIN_N:
                continue
            raw_seed = pathway_bootstrap_seed(
                dataset,
                seed_key,
                "raw",
                n_bootstrap=n_bootstrap,
            )
            print(
                f"  bootstrap {dataset} / {pathway}: raw "
                f"({n_bootstrap:,} resamples)",
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
                stage = {
                    "centroid_magnitude": "partial_mag",
                    "edistance": "partial_edistance",
                    "centroid_magnitude+edistance": "partial_mag_edistance",
                }[model]
                seed = pathway_bootstrap_seed(
                    dataset,
                    seed_key,
                    stage,
                    n_bootstrap=n_bootstrap,
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

    status_rows = []
    for _, row in result.iterrows():
        status_rows.append(
            survival_status(
                row["rho_partial"],
                row["rho_partial_ci_low"],
                row["rho_partial_ci_high"],
                fdr=row["p_partial_fdr_bh"],
            )
        )
    result["survival_status"] = [s["status"] for s in status_rows]
    result["survives_covariate_control"] = [s["survives"] for s in status_rows]
    result["knife_edge_ci"] = [s["knife_edge"] for s in status_rows]
    result["ci_fdr_disagree"] = [s["ci_fdr_disagree"] for s in status_rows]
    result["survival_criterion_id"] = [
        s["criterion_id"] for s in status_rows
    ]
    return result


def _load_qc_gate_inputs(
    qc_per_path: Path,
    existing_gate_path: Path,
    scores: pd.DataFrame,
    *,
    n_bootstrap: int,
    include_stress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the frozen QC table and existing centroid+QC gate, then join E-distance."""
    if not qc_per_path.exists():
        raise FileNotFoundError(
            f"QC per-perturbation table not found: {qc_per_path}. "
            "Run cell_quality_partial.py once to create it."
        )
    if not existing_gate_path.exists():
        raise FileNotFoundError(
            f"Existing centroid+QC gate not found: {existing_gate_path}. "
            "Run cell_quality_partial.py once to create it."
        )

    qc = pd.read_csv(qc_per_path)
    qc_required = {
        "dataset",
        "perturbation",
        "stability",
        "magnitude",
        "qc_percent_mito",
        "qc_n_genes",
        "qc_n_counts",
    }
    missing = qc_required - set(qc.columns)
    if missing:
        raise ValueError(f"{qc_per_path} lacks columns {sorted(missing)}")
    qc["dataset"] = qc["dataset"].astype(str).map(cfg.resolve_dataset_name)
    qc["perturbation"] = qc["perturbation"].astype(str)
    if qc.duplicated(["dataset", "perturbation"]).any():
        raise ValueError(f"{qc_per_path} has duplicate dataset/perturbation keys")

    geometry = scores[
        ["dataset", "perturbation", "stability", "centroid_magnitude", "edistance"]
    ].copy()
    joined = qc.merge(
        geometry,
        on=["dataset", "perturbation"],
        how="left",
        validate="one_to_one",
        suffixes=("_qc", "_frozen"),
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        examples = joined.loc[
            joined["_merge"] != "both", ["dataset", "perturbation"]
        ].head().to_dict("records")
        raise RuntimeError(
            f"{qc_per_path} has rows without E-distance geometry: {examples}"
        )
    joined = joined.drop(columns="_merge")
    if not np.allclose(
        joined["stability_qc"], joined["stability_frozen"], atol=1e-10, rtol=0
    ):
        raise ValueError("QC table stability does not match frozen E-distance scores")
    if not np.allclose(
        joined["magnitude"], joined["centroid_magnitude"], atol=1e-10, rtol=0
    ):
        raise ValueError("QC table magnitude does not match frozen centroid magnitude")
    joined = joined.rename(columns={"stability_frozen": "stability"}).drop(
        columns="stability_qc"
    )

    baseline = pd.read_csv(existing_gate_path)
    baseline_required = {
        "dataset",
        "feature",
        "n",
        "rho_partial_mag_qc",
        "rho_partial_mag_qc_ci_low",
        "rho_partial_mag_qc_ci_high",
        "p_partial_mag_qc",
        "p_partial_mag_qc_fdr_bh",
        "survival_status_qc",
        "survives_mag_qc",
        "gate",
        "qc_descriptive_only",
        "covariates_mag_qc",
        "config_version",
        "n_bootstrap",
    }
    missing = baseline_required - set(baseline.columns)
    if missing:
        raise ValueError(f"{existing_gate_path} lacks columns {sorted(missing)}")
    baseline["dataset"] = baseline["dataset"].astype(str).map(cfg.resolve_dataset_name)
    baseline["feature"] = baseline["feature"].astype(str)
    versions = {
        cfg.resolve_config_version(str(x))
        for x in baseline["config_version"].dropna().unique()
    }
    if versions != {cfg.CONFIG_VERSION}:
        raise ValueError(
            f"{existing_gate_path} config versions {versions} != {cfg.CONFIG_VERSION}"
        )
    baseline_boot = set(
        pd.to_numeric(baseline["n_bootstrap"], errors="coerce").dropna().astype(int)
    )
    if baseline_boot != {int(n_bootstrap)}:
        raise ValueError(
            f"Existing centroid+QC gate used n_bootstrap={sorted(baseline_boot)}, "
            f"but this run requests {n_bootstrap}. Use the same value."
        )
    feature_mask = baseline["feature"].str.startswith("pw_")
    if include_stress:
        feature_mask |= baseline["feature"].str.startswith("stress_")
        qc_stress = sorted(c for c in joined.columns if c.startswith("stress_"))
        gate_stress = sorted(
            baseline.loc[
                baseline["feature"].str.startswith("stress_"), "feature"
            ].unique()
        )
        if "stress_DDIT3" not in qc_stress or "stress_DDIT3" not in gate_stress:
            raise ValueError(
                "--include-stress-qc requires stress_DDIT3 in both "
                "cell_quality_per_perturbation.csv and cell_quality_partials.csv. "
                "Re-run cell_quality_partial.py with --include-stress using the "
                "stress-enriched frozen Sp table."
            )
        missing_stress = sorted(set(gate_stress) - set(qc_stress))
        if missing_stress:
            raise ValueError(
                f"Existing stress gates lack per-perturbation values: {missing_stress}"
            )
    baseline = baseline[feature_mask].copy()
    if baseline.duplicated(["dataset", "feature"]).any():
        raise ValueError(f"{existing_gate_path} has duplicate dataset/feature keys")
    return joined, baseline


def _qc_point_or_bootstrap(
    sp: np.ndarray,
    pathway: np.ndarray,
    covariates: np.ndarray,
    *,
    descriptive_only: bool,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if not descriptive_only:
        return bootstrap_partial_spearman_ci(
            sp,
            pathway,
            covariates,
            n_bootstrap=n_bootstrap,
            ci_level=cfg.CI_LEVEL,
            seed=seed,
            method="rank",
        )
    point = partial_spearman_rank(sp, pathway, covariates)
    return {
        "rho_partial": point["rho_partial"],
        "p": point["p"],
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": point["n"],
        "method": point.get("method", "partial_spearman_rank"),
        "n_bootstrap": 0,
        "bootstrap_frac_valid": np.nan,
        "bootstrap_seed": seed,
    }


def _qc_gate_from_statuses(
    non_qc_status: str,
    qc_status: str,
    *,
    descriptive_only: bool,
) -> str:
    """Mirror cell_quality_partial's primary-gate semantics."""
    if descriptive_only:
        return "descriptive_small_n"
    if "indeterminate" in {non_qc_status, qc_status}:
        return "indeterminate"
    if qc_status == "survives":
        return "survives_qc" if non_qc_status == "survives" else "qc_conditional"
    if non_qc_status == "survives" and qc_status == "does_not_survive":
        return "collapses_under_qc"
    return "no_effect_size_partial"


def _csv_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"Cannot interpret CSV boolean value {value!r}")
    if pd.isna(value):
        return False
    return bool(value)


def _run_qc_edistance_models(
    qc: pd.DataFrame,
    existing_baseline: pd.DataFrame,
    non_qc_partials: pd.DataFrame,
    *,
    n_bootstrap: int,
    include_stress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Add E-distance+QC and centroid+E-distance+QC, preserving centroid+QC baseline.

    Primary survivors are the intersection of ``gate == survives_qc`` across
    all three models. No union-across-models claim is produced.
    """
    if _stats_utils.pg is not None:
        _stats_utils.pg = None
    qc_cols = ["qc_percent_mito", "qc_n_genes", "qc_n_counts"]
    mito_only_n = int(getattr(cfg, "SURVIVAL_QC_MITO_ONLY_MAX_N", 40))
    descriptive_n = int(getattr(cfg, "SURVIVAL_QC_DESCRIPTIVE_MAX_N", 30))
    model_specs = {
        "edistance+QC": ("edistance", "partial_edistance_qc"),
        "centroid_magnitude+edistance+QC": (
            "centroid_magnitude+edistance",
            "partial_mag_edistance_qc",
        ),
    }
    feature_prefixes = ("pw_", "stress_") if include_stress else ("pw_",)
    feature_cols = sorted(
        c for c in qc.columns if c.startswith(feature_prefixes)
    )
    rows: list[dict] = []

    if include_stress:
        stress_non_qc = _pathway_partials(
            qc,
            n_bootstrap,
            feature_prefixes=("stress_",),
        )
        if stress_non_qc.empty:
            raise RuntimeError("No stress-marker non-QC models were scoreable")
        non_qc_partials = pd.concat(
            [non_qc_partials, stress_non_qc], ignore_index=True
        )
        # Match the existing centroid+QC gate's family: pathways and stress
        # markers were BH-adjusted together when --include-stress was used.
        for (_, _), idx in non_qc_partials.groupby(
            ["dataset", "covariate_model"], sort=False
        ).groups.items():
            p = non_qc_partials.loc[idx, "p_partial"].to_numpy(float)
            non_qc_partials.loc[idx, "p_partial_fdr_bh"] = _fdr_bh(
                np.where(np.isfinite(p), p, 1.0)
            )
        statuses = [
            survival_status(
                row["rho_partial"],
                row["rho_partial_ci_low"],
                row["rho_partial_ci_high"],
                fdr=row["p_partial_fdr_bh"],
            )
            for _, row in non_qc_partials.iterrows()
        ]
        non_qc_partials["survival_status"] = [
            status["status"] for status in statuses
        ]
        non_qc_partials["survives_covariate_control"] = [
            status["survives"] for status in statuses
        ]
    non_qc_lookup = non_qc_partials.set_index(
        ["dataset", "pathway", "covariate_model"]
    )
    for dataset in sorted(qc["dataset"].dropna().unique()):
        ds = qc[qc["dataset"] == dataset]
        for feature in feature_cols:
            pathway, feature_type, seed_key = _feature_descriptor(feature)
            sub = ds.dropna(
                subset=[
                    "stability",
                    "centroid_magnitude",
                    "edistance",
                    feature,
                    *qc_cols,
                ]
            ).copy()
            if len(sub) < PATHWAY_MIN_N:
                continue
            selected_qc = (
                ["qc_percent_mito"] if len(sub) < mito_only_n else qc_cols
            )
            descriptive = len(sub) < descriptive_n
            sp = sub["stability"].to_numpy(float)
            y = sub[feature].to_numpy(float)

            for model, (base_model, seed_stage) in model_specs.items():
                effect_cols = (
                    ["edistance"]
                    if model == "edistance+QC"
                    else ["centroid_magnitude", "edistance"]
                )
                covar_cols = [*effect_cols, *selected_qc]
                z = sub[covar_cols].to_numpy(float)
                seed = pathway_bootstrap_seed(
                    dataset, seed_key, seed_stage, n_bootstrap=n_bootstrap
                )
                print(
                    f"  QC bootstrap {dataset} / {pathway} | {model}: "
                    + (
                        "descriptive point estimate"
                        if descriptive
                        else f"{n_bootstrap:,} resamples"
                    ),
                    flush=True,
                )
                result = _qc_point_or_bootstrap(
                    sp,
                    y,
                    z,
                    descriptive_only=descriptive,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                diag = _rank_residual_diagnostics(sp, z, y)
                rows.append(
                    {
                        "dataset": dataset,
                        "pathway": pathway,
                        "outcome": pathway,
                        "feature": feature,
                        "feature_type": feature_type,
                        "covariate_model": model,
                        "covariates": "|".join(covar_cols),
                        "non_qc_covariate_model": base_model,
                        "n": len(sub),
                        "rho_partial": result["rho_partial"],
                        "rho_partial_ci_low": result["ci_low"],
                        "rho_partial_ci_high": result["ci_high"],
                        "p_partial": result["p"],
                        "partial_r2": diag["partial_r2"],
                        "r2_Sp_on_covariates": diag["r2_sp_on_covariates"],
                        "frac_Sp_var_left": diag["frac_sp_variance_remaining"],
                        "covariate_rank": diag["covariate_rank"],
                        "covariate_condition_number": diag[
                            "covariate_condition_number"
                        ],
                        "qc_descriptive_only": descriptive,
                        "bootstrap_seed": seed,
                        "n_bootstrap": result.get("n_bootstrap", 0),
                        "bootstrap_frac_valid": result.get(
                            "bootstrap_frac_valid", np.nan
                        ),
                        "partial_method": result.get(
                            "method", "partial_spearman_rank"
                        ),
                        "config_version": cfg.CONFIG_VERSION,
                        "model_source": "edistance_competitor_analysis",
                    }
                )

    added = pd.DataFrame(rows)
    if added.empty:
        raise RuntimeError("No E-distance+QC pathway models were scoreable")
    added["p_partial_fdr_bh"] = np.nan
    for (_, _), idx in added.groupby(
        ["dataset", "covariate_model"], sort=False
    ).groups.items():
        p = added.loc[idx, "p_partial"].to_numpy(float)
        added.loc[idx, "p_partial_fdr_bh"] = _fdr_bh(
            np.where(np.isfinite(p), p, 1.0)
        )

    statuses = []
    gates = []
    for _, row in added.iterrows():
        if row["qc_descriptive_only"]:
            qc_status = "descriptive_small_n"
            survives = False
        else:
            status = survival_status(
                row["rho_partial"],
                row["rho_partial_ci_low"],
                row["rho_partial_ci_high"],
                fdr=row["p_partial_fdr_bh"],
            )
            qc_status = status["status"]
            survives = status["survives"]
        key = (
            row["dataset"],
            row["pathway"],
            row["non_qc_covariate_model"],
        )
        if key not in non_qc_lookup.index:
            raise RuntimeError(f"Missing non-QC pathway result needed for gate: {key}")
        non_qc_status = str(non_qc_lookup.loc[key, "survival_status"])
        statuses.append((qc_status, survives, non_qc_status))
        gates.append(
            _qc_gate_from_statuses(
                non_qc_status,
                qc_status,
                descriptive_only=bool(row["qc_descriptive_only"]),
            )
        )
    added["survival_status_qc"] = [x[0] for x in statuses]
    added["survives_qc_model"] = [x[1] for x in statuses]
    added["survival_status_non_qc"] = [x[2] for x in statuses]
    added["gate"] = gates
    added["survives_primary_gate"] = added["gate"] == "survives_qc"

    # Standardize the existing, frozen centroid+QC gate into the same long table.
    baseline_rows = []
    baseline = existing_baseline[
        existing_baseline["dataset"].isin(qc["dataset"].unique())
    ].copy()
    for _, row in baseline.iterrows():
        pathway, feature_type, _ = _feature_descriptor(str(row["feature"]))
        sub = qc[
            (qc["dataset"] == row["dataset"])
        ].dropna(
            subset=[
                "stability",
                "centroid_magnitude",
                str(row["feature"]),
                *qc_cols,
            ]
        )
        selected_qc = (
            ["qc_percent_mito"] if len(sub) < mito_only_n else qc_cols
        )
        covar_cols = ["centroid_magnitude", *selected_qc]
        diag = _rank_residual_diagnostics(
            sub["stability"].to_numpy(float),
            sub[covar_cols].to_numpy(float),
            sub[str(row["feature"])].to_numpy(float),
        )
        baseline_rows.append(
            {
                "dataset": row["dataset"],
                "pathway": pathway,
                "outcome": pathway,
                "feature": row["feature"],
                "feature_type": feature_type,
                "covariate_model": "centroid_magnitude+QC",
                "covariates": "|".join(covar_cols),
                "non_qc_covariate_model": "centroid_magnitude",
                "n": int(row["n"]),
                "rho_partial": row["rho_partial_mag_qc"],
                "rho_partial_ci_low": row["rho_partial_mag_qc_ci_low"],
                "rho_partial_ci_high": row["rho_partial_mag_qc_ci_high"],
                "p_partial": row["p_partial_mag_qc"],
                "p_partial_fdr_bh": row["p_partial_mag_qc_fdr_bh"],
                "partial_r2": diag["partial_r2"],
                "r2_Sp_on_covariates": diag["r2_sp_on_covariates"],
                "frac_Sp_var_left": diag["frac_sp_variance_remaining"],
                "covariate_rank": diag["covariate_rank"],
                "covariate_condition_number": diag[
                    "covariate_condition_number"
                ],
                "qc_descriptive_only": _csv_bool(row["qc_descriptive_only"]),
                "bootstrap_seed": row.get("bootstrap_seed_qc", np.nan),
                "n_bootstrap": int(row["n_bootstrap"]),
                "bootstrap_frac_valid": np.nan,
                "partial_method": "partial_spearman_rank",
                "config_version": cfg.CONFIG_VERSION,
                "model_source": "existing_cell_quality_partials",
                "survival_status_qc": row["survival_status_qc"],
                "survives_qc_model": _csv_bool(row["survives_mag_qc"]),
                "survival_status_non_qc": row.get(
                    "survival_status_mag", "unknown"
                ),
                "gate": row["gate"],
                "survives_primary_gate": row["gate"] == "survives_qc",
            }
        )
    combined = pd.concat([pd.DataFrame(baseline_rows), added], ignore_index=True)

    model_order = [
        "centroid_magnitude+QC",
        "edistance+QC",
        "centroid_magnitude+edistance+QC",
    ]
    intersection_rows = []
    for (dataset, feature), group in combined.groupby(["dataset", "feature"]):
        pathway = str(group.iloc[0]["outcome"])
        feature_type = str(group.iloc[0]["feature_type"])
        by_model = group.set_index("covariate_model")
        missing_models = [m for m in model_order if m not in by_model.index]
        if missing_models:
            raise RuntimeError(
                f"Missing QC models for {dataset}/{pathway}: {missing_models}"
            )
        gates_by_model = {m: str(by_model.loc[m, "gate"]) for m in model_order}
        survivors_by_model = {
            m: bool(by_model.loc[m, "survives_primary_gate"]) for m in model_order
        }
        intersection_rows.append(
            {
                "dataset": dataset,
                "pathway": pathway,
                "outcome": pathway,
                "feature": feature,
                "feature_type": feature_type,
                "gate_centroid_magnitude_qc": gates_by_model[
                    "centroid_magnitude+QC"
                ],
                "gate_edistance_qc": gates_by_model["edistance+QC"],
                "gate_centroid_magnitude_edistance_qc": gates_by_model[
                    "centroid_magnitude+edistance+QC"
                ],
                "survives_centroid_magnitude_qc": survivors_by_model[
                    "centroid_magnitude+QC"
                ],
                "survives_edistance_qc": survivors_by_model["edistance+QC"],
                "survives_centroid_magnitude_edistance_qc": survivors_by_model[
                    "centroid_magnitude+edistance+QC"
                ],
                "survives_all_models": all(survivors_by_model.values()),
                "survivor_policy": (
                    "intersection: gate==survives_qc in centroid+QC, "
                    "E-distance+QC, and centroid+E-distance+QC"
                ),
                "primary_effect_size": "centroid_magnitude",
                "config_version": cfg.CONFIG_VERSION,
            }
        )
    intersection = pd.DataFrame(intersection_rows)
    all_survivors = intersection[intersection["survives_all_models"]]
    summary = {
        "config_version": cfg.CONFIG_VERSION,
        "n_bootstrap": n_bootstrap,
        "models_reported": model_order,
        "primary_effect_size": "centroid_magnitude",
        "survivor_policy": "intersection_across_all_qc_models_not_union",
        "small_n_policy": {
            "mito_only_below_n": mito_only_n,
            "descriptive_only_below_n": descriptive_n,
        },
        "stress_markers_included": include_stress,
        "all_model_survivors": all_survivors[
            ["dataset", "feature_type", "outcome"]
        ].to_dict("records"),
        "survivor_counts_by_outcome": {
            f"{feature_type}:{outcome}": int(group["survives_all_models"].sum())
            for (feature_type, outcome), group in intersection.groupby(
                ["feature_type", "outcome"]
            )
        },
        "interpretation": (
            "Report every covariate model. Centroid magnitude remains primary "
            "because it is frozen and pre-specified. The primary survivor set "
            "is the intersection across all three QC-conditioned models."
        ),
    }
    combined = combined.sort_values(
        ["feature_type", "outcome", "dataset", "covariate_model"],
        kind="mergesort",
    )
    intersection = intersection.sort_values(
        ["feature_type", "outcome", "dataset"], kind="mergesort"
    )
    return combined, intersection, summary


def _print_dataset_table(table: pd.DataFrame) -> None:
    cols = [
        "dataset",
        "n",
        "rho_centroid_magnitude_edistance",
        "rho_Sp_centroid_magnitude",
        "rho_Sp_edistance",
        "delta_abs_rho_Sp_edistance_vs_centroid_magnitude",
        "frac_Sp_var_left_after_centroid_magnitude",
        "frac_Sp_var_left_after_edistance",
    ]
    print("\n--- E-distance competitor table ---", flush=True)
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


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


def _load_reusable_edistance_scores(
    path: Path,
    *,
    frozen: pd.DataFrame,
    frozen_info: dict,
    datasets: list[str],
    edistance_control_cap: int | None,
) -> pd.DataFrame:
    """Load a completed score file only after strict freeze/key validation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Reusable E-distance score file not found: {path}. "
            "Run --correlations-only first."
        )
    scores = pd.read_csv(path)
    required = {
        "dataset",
        "perturbation",
        "config_version",
        "sp_digest",
        "n_pcs",
        "seed",
        "edistance",
        "edistance_method",
        "n_control_edistance",
        "n_control_frozen_embedding",
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
    methods = set(scores["edistance_method"].dropna().astype(str).unique())
    if versions != {cfg.CONFIG_VERSION}:
        raise ValueError(f"{path} config versions {versions} != {cfg.CONFIG_VERSION}")
    if digests != {frozen_info["sp_digest"]}:
        raise ValueError(f"{path} Sp digest {digests} != {frozen_info['sp_digest']}")
    if methods != {METHOD_ID}:
        raise ValueError(f"{path} E-distance methods {methods} != {METHOD_ID}")
    if set(pd.to_numeric(scores["n_pcs"]).dropna().astype(int)) != {cfg.N_PCS}:
        raise ValueError(f"{path} does not use n_pcs={cfg.N_PCS}")
    if set(pd.to_numeric(scores["seed"]).dropna().astype(int)) != {cfg.SEED}:
        raise ValueError(f"{path} does not use seed={cfg.SEED}")

    selected = scores[scores["dataset"].isin(datasets)].copy()
    if edistance_control_cap is None:
        cap_ok = (
            selected["n_control_edistance"].astype(int)
            == selected["n_control_frozen_embedding"].astype(int)
        ).all()
    else:
        cap_ok = (
            selected["n_control_edistance"].astype(int)
            == np.minimum(
                selected["n_control_frozen_embedding"].astype(int),
                edistance_control_cap,
            )
        ).all()
    if not cap_ok:
        raise ValueError(
            f"{path} was generated with a different --edistance-control-cap"
        )

    expected = frozen[frozen["dataset"].isin(datasets)][
        ["dataset", "perturbation"]
    ].drop_duplicates()
    got = selected[["dataset", "perturbation"]].drop_duplicates()
    keys = expected.merge(
        got, on=["dataset", "perturbation"], how="outer", indicator=True
    )
    if len(selected) != len(got) or not (keys["_merge"] == "both").all():
        raise ValueError(
            f"{path} keys do not exactly match the selected frozen rows: "
            f"{keys['_merge'].value_counts().to_dict()}"
        )
    print(
        f"Reusing {len(selected):,} verified E-distance rows from {path}; "
        "skipping all dataset loading/PCA/distance computation.",
        flush=True,
    )
    return selected


def _load_reusable_pathway_partials(
    path: Path,
    *,
    datasets: list[str],
    n_bootstrap: int,
) -> pd.DataFrame:
    """Load the completed non-QC E-distance pathway table with strict guards."""
    if not path.exists():
        raise FileNotFoundError(
            f"Reusable pathway partial table not found: {path}. "
            "Run the non-QC pathway analysis first or omit "
            "--reuse-pathway-partials."
        )
    table = pd.read_csv(path)
    required = {
        "dataset",
        "pathway",
        "covariate_model",
        "survival_status",
        "survives_covariate_control",
        "config_version",
        "n_bootstrap",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is not reusable; missing columns {sorted(missing)}")
    table["dataset"] = table["dataset"].astype(str).map(cfg.resolve_dataset_name)
    versions = {
        cfg.resolve_config_version(str(x))
        for x in table["config_version"].dropna().unique()
    }
    if versions != {cfg.CONFIG_VERSION}:
        raise ValueError(f"{path} config versions {versions} != {cfg.CONFIG_VERSION}")
    boot = pd.to_numeric(table["n_bootstrap"], errors="coerce").dropna().astype(int)
    if boot.empty or (boot > int(n_bootstrap)).any() or (
        boot < int(0.8 * n_bootstrap)
    ).any():
        raise ValueError(
            f"{path} valid bootstrap counts span "
            f"{int(boot.min()) if len(boot) else 'n/a'}–"
            f"{int(boot.max()) if len(boot) else 'n/a'}, incompatible with "
            f"requested {n_bootstrap}"
        )
    expected_models = {
        "centroid_magnitude",
        "edistance",
        "centroid_magnitude+edistance",
    }
    got_models = set(table["covariate_model"].astype(str).unique())
    if got_models != expected_models:
        raise ValueError(
            f"{path} covariate models {sorted(got_models)} != "
            f"{sorted(expected_models)}"
        )
    table = table[table["dataset"].isin(datasets)].copy()
    if table.duplicated(["dataset", "pathway", "covariate_model"]).any():
        raise ValueError(f"{path} has duplicate dataset/pathway/model rows")
    print(
        f"Reusing {len(table)} verified non-QC pathway rows from {path}.",
        flush=True,
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-sp", type=Path, default=None)
    parser.add_argument("--pathway-scores", type=Path, default=None)
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
        "--correlations-only",
        action="store_true",
        help="Stop after the dataset competitor table; do not require pathway scores.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=cfg.N_BOOTSTRAP,
        help=f"Bootstrap replicates (default: frozen {cfg.N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Rows per cdist block (memory/performance only; does not change values).",
    )
    parser.add_argument(
        "--edistance-control-cap",
        type=int,
        default=None,
        help=(
            "Optional deterministic control subsample after the frozen PCA is fit. "
            "Default uses all frozen controls; any cap is a sensitivity run."
        ),
    )
    parser.add_argument(
        "--sp-verify-atol", type=float, default=DEFAULT_SP_VERIFY_ATOL
    )
    parser.add_argument(
        "--magnitude-verify-atol", type=float, default=DEFAULT_MAG_VERIFY_ATOL
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume compatible dataset rows from {PARTIAL_SCORES_NAME}.",
    )
    parser.add_argument(
        "--reuse-edistance-scores",
        action="store_true",
        help=(
            f"Reuse a completed, strictly validated {SCORES_NAME} from --out-dir "
            "and run only correlations/pathway partials."
        ),
    )
    parser.add_argument(
        "--reuse-pathway-partials",
        action="store_true",
        help=(
            f"Reuse a completed, validated {PATHWAY_TABLE_NAME} from --out-dir "
            "instead of repeating non-QC pathway bootstraps."
        ),
    )
    parser.add_argument(
        "--run-qc-models",
        action="store_true",
        help=(
            "Add E-distance+QC and centroid+E-distance+QC, compare with the "
            "existing centroid+QC gate, and write the all-model intersection."
        ),
    )
    parser.add_argument(
        "--include-stress-qc",
        action="store_true",
        help=(
            "Extend the same three QC-conditioned models and intersection rule "
            "to stress_* outcomes (including DDIT3). Requires QC artifacts "
            "created by cell_quality_partial.py --include-stress."
        ),
    )
    parser.add_argument(
        "--qc-per-perturbation",
        type=Path,
        default=None,
        help=(
            "cell_quality_per_perturbation.csv (default: --out-dir). "
            "Produced by cell_quality_partial.py."
        ),
    )
    parser.add_argument(
        "--existing-qc-partials",
        type=Path,
        default=None,
        help=(
            "Existing cell_quality_partials.csv containing the frozen "
            "centroid+QC gate (default: --out-dir)."
        ),
    )
    args = parser.parse_args()

    if not args.reuse_edistance_scores:
        _guard_backed_sparse_versions()
    if args.run_qc_models and args.correlations_only:
        raise ValueError("--run-qc-models cannot be combined with --correlations-only")
    if args.include_stress_qc and not args.run_qc_models:
        raise ValueError("--include-stress-qc requires --run-qc-models")
    if args.reuse_pathway_partials and args.correlations_only:
        raise ValueError(
            "--reuse-pathway-partials cannot be combined with --correlations-only"
        )
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be >=100")
    if args.block_size < 1:
        raise ValueError("--block-size must be positive")
    if args.edistance_control_cap is not None and args.edistance_control_cap < 5:
        raise ValueError("--edistance-control-cap must be >=5")

    out_dir = resolve_out_dir(args.out_dir)
    frozen_path = find_sp_csv(out_dir, args.frozen_sp)
    frozen_info = assert_frozen_sp_compatible(frozen_path)
    frozen = load_sp_table(frozen_path)
    _validate_frozen_metadata(frozen)
    datasets = _resolve_datasets(args.dataset, frozen)
    h5ad_overrides = _parse_h5ad_overrides(args.h5ad)
    setup_cache()

    print(
        f"config={cfg.CONFIG_VERSION} seed={cfg.SEED} n_pcs={cfg.N_PCS} "
        f"datasets={datasets}\nfrozen_sp={frozen_path}\nout_dir={out_dir}",
        flush=True,
    )
    if args.edistance_control_cap is not None:
        print(
            "WARNING: --edistance-control-cap is a sensitivity run; PCA still uses "
            "the full frozen controls, but E-distance does not.",
            flush=True,
        )

    partial_path = out_dir / PARTIAL_SCORES_NAME
    completed: list[pd.DataFrame] = []
    validations: list[dict] = []
    done = set()
    if args.reuse_edistance_scores:
        reused = _load_reusable_edistance_scores(
            out_dir / SCORES_NAME,
            frozen=frozen,
            frozen_info=frozen_info,
            datasets=datasets,
            edistance_control_cap=args.edistance_control_cap,
        )
        completed.append(reused)
        done = set(datasets)
    elif args.resume and partial_path.exists():
        prior = pd.read_csv(partial_path)
        required = {"dataset", "config_version", "sp_digest", "n_pcs", "seed"}
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
        if args.edistance_control_cap is None:
            cap_ok = (
                prior["n_control_edistance"].astype(int)
                == prior["n_control_frozen_embedding"].astype(int)
            ).all()
        else:
            cap_ok = (
                prior["n_control_edistance"].astype(int)
                == np.minimum(
                    prior["n_control_frozen_embedding"].astype(int),
                    args.edistance_control_cap,
                )
            ).all()
        if not cap_ok:
            raise ValueError(
                "Resume file was generated with a different "
                "--edistance-control-cap."
            )
        completed.append(prior)
        done = set(prior["dataset"].astype(str).unique())
        print(
            f"Resuming completed datasets: {sorted(done)}. "
            "Resume is opt-in and trusts the checkpoint's already-verified input files.",
            flush=True,
        )

    for dataset in datasets:
        if dataset in done:
            continue
        frozen_sub = frozen[frozen["dataset"] == dataset].copy()
        result, validation = _dataset_embedding_and_scores(
            dataset,
            frozen_sub,
            h5ad_path=h5ad_overrides.get(dataset),
            block_size=args.block_size,
            edistance_control_cap=args.edistance_control_cap,
            sp_verify_atol=args.sp_verify_atol,
            mag_verify_atol=args.magnitude_verify_atol,
        )
        completed.append(result)
        validations.append(validation)
        checkpoint = pd.concat(completed, ignore_index=True)
        _atomic_csv(checkpoint, partial_path)

    if not completed:
        raise RuntimeError("No E-distance rows were computed")
    scores = pd.concat(completed, ignore_index=True)
    scores = scores[scores["dataset"].isin(datasets)].copy()
    expected_keys = frozen[frozen["dataset"].isin(datasets)][
        ["dataset", "perturbation"]
    ].drop_duplicates()
    got_keys = scores[["dataset", "perturbation"]].drop_duplicates()
    key_check = expected_keys.merge(
        got_keys,
        on=["dataset", "perturbation"],
        how="outer",
        indicator=True,
    )
    if not (key_check["_merge"] == "both").all():
        raise RuntimeError(
            "Final E-distance keys do not exactly match selected frozen keys: "
            f"{key_check['_merge'].value_counts().to_dict()}"
        )

    scores = scores.sort_values(["dataset", "perturbation"], kind="mergesort")
    score_path = out_dir / SCORES_NAME
    _atomic_csv(scores, score_path)

    # Preserve run_frozen_main.summarize() CI convention for the section-5
    # baseline: seed=320 and at most 2,000 resamples. Pathway partials retain
    # the frozen 10,000-resample default and canonical feature/model seeds.
    correlation_n_bootstrap = min(2000, args.n_bootstrap)
    dataset_table = _dataset_correlations(scores, correlation_n_bootstrap)
    dataset_path = out_dir / DATASET_TABLE_NAME
    _atomic_csv(dataset_table, dataset_path)
    _print_dataset_table(dataset_table)

    pathway_path = None
    pathway_table = pd.DataFrame()
    qc_path = None
    qc_intersection_path = None
    stress_qc_path = None
    stress_qc_intersection_path = None
    qc_summary_path = None
    if not args.correlations_only:
        pathway_path = out_dir / PATHWAY_TABLE_NAME
        if args.reuse_pathway_partials:
            pathway_table = _load_reusable_pathway_partials(
                pathway_path,
                datasets=datasets,
                n_bootstrap=args.n_bootstrap,
            )
        else:
            source = args.pathway_scores or (out_dir / "pathway_scores_per_pert.csv")
            merged = _load_pathway_scores(source, scores)
            pathway_table = _pathway_partials(merged, args.n_bootstrap)
            if pathway_table.empty:
                raise RuntimeError("No pathway partials were scoreable")
            _atomic_csv(pathway_table, pathway_path)
        print("\n--- Pathway partial survivor counts ---", flush=True)
        print(
            pathway_table.groupby("covariate_model")["survives_covariate_control"]
            .agg(["sum", "count"])
            .to_string()
        )

    if args.run_qc_models:
        qc_per_path = args.qc_per_perturbation or (
            out_dir / "cell_quality_per_perturbation.csv"
        )
        existing_gate_path = args.existing_qc_partials or (
            out_dir / "cell_quality_partials.csv"
        )
        qc_per, existing_gate = _load_qc_gate_inputs(
            qc_per_path,
            existing_gate_path,
            scores,
            n_bootstrap=args.n_bootstrap,
            include_stress=args.include_stress_qc,
        )
        qc_table, qc_intersection, qc_summary = _run_qc_edistance_models(
            qc_per,
            existing_gate,
            pathway_table,
            n_bootstrap=args.n_bootstrap,
            include_stress=args.include_stress_qc,
        )
        qc_path = out_dir / QC_TABLE_NAME
        qc_intersection_path = out_dir / QC_INTERSECTION_NAME
        qc_summary_path = out_dir / QC_SUMMARY_NAME
        pathway_qc = qc_table[qc_table["feature_type"] == "pathway"].copy()
        pathway_intersection = qc_intersection[
            qc_intersection["feature_type"] == "pathway"
        ].copy()
        _atomic_csv(pathway_qc, qc_path)
        _atomic_csv(pathway_intersection, qc_intersection_path)
        if args.include_stress_qc:
            stress_qc = qc_table[
                qc_table["feature_type"] == "stress_marker"
            ].copy()
            stress_intersection = qc_intersection[
                qc_intersection["feature_type"] == "stress_marker"
            ].copy()
            if stress_qc.empty or stress_intersection.empty:
                raise RuntimeError(
                    "--include-stress-qc requested but no stress-marker outputs "
                    "were produced"
                )
            stress_qc_path = out_dir / STRESS_QC_TABLE_NAME
            stress_qc_intersection_path = (
                out_dir / STRESS_QC_INTERSECTION_NAME
            )
            _atomic_csv(stress_qc, stress_qc_path)
            _atomic_csv(stress_intersection, stress_qc_intersection_path)
        _atomic_json(qc_summary, qc_summary_path)
        print("\n--- Primary survivors: intersection across all QC models ---")
        survivors = qc_intersection[qc_intersection["survives_all_models"]]
        if survivors.empty:
            print("  (none)")
        else:
            print(
                survivors[
                    ["dataset", "feature_type", "outcome"]
                ].to_string(index=False)
            )
        print(
            "\nPre-committed reporting: all models are retained; centroid magnitude "
            "remains primary; the headline survivor set is the intersection, "
            "never the union.",
            flush=True,
        )

    summary = {
        "config_version": cfg.CONFIG_VERSION,
        "seed": cfg.SEED,
        "n_pcs": cfg.N_PCS,
        "frozen_sp": str(frozen_path),
        "sp_digest": frozen_info["sp_digest"],
        "datasets": datasets,
        "edistance_method": METHOD_ID,
        "cell_wise_metric": "euclidean",
        "full_frozen_control_set_used": args.edistance_control_cap is None,
        "edistance_control_cap": args.edistance_control_cap,
        "pertpy_validation": validations,
        "n_bootstrap": args.n_bootstrap,
        "dataset_correlation_n_bootstrap": correlation_n_bootstrap,
        "dataset_correlation_bootstrap_seed": cfg.SEED,
        "outputs": {
            "scores": str(score_path),
            "dataset_correlations": str(dataset_path),
            "pathway_partials": str(pathway_path) if pathway_path else None,
            "qc_partials": str(qc_path) if qc_path else None,
            "qc_all_model_intersection": (
                str(qc_intersection_path) if qc_intersection_path else None
            ),
            "stress_qc_partials": (
                str(stress_qc_path) if stress_qc_path else None
            ),
            "stress_qc_all_model_intersection": (
                str(stress_qc_intersection_path)
                if stress_qc_intersection_path
                else None
            ),
            "qc_summary": str(qc_summary_path) if qc_summary_path else None,
        },
        "interpretation_rules": INTERPRETATION_RULES,
        "notes": [
            "Centroid magnitude is the frozen primary measure.",
            "E-distance is a dispersion-aware competitor, not a magnitude label.",
            "Both-covariate condition numbers are reported to expose collinearity.",
            "Pathway outcomes are joined from pathway_analysis.py so all three "
            "covariate models use the identical frozen section-3 pathway scores.",
        ],
    }
    _atomic_json(summary, out_dir / SUMMARY_NAME)
    partial_path.unlink(missing_ok=True)
    print(
        f"\nWrote:\n  {score_path}\n  {dataset_path}"
        + (f"\n  {pathway_path}" if pathway_path else "")
        + (f"\n  {qc_path}" if qc_path else "")
        + (f"\n  {qc_intersection_path}" if qc_intersection_path else "")
        + (f"\n  {stress_qc_path}" if stress_qc_path else "")
        + (
            f"\n  {stress_qc_intersection_path}"
            if stress_qc_intersection_path
            else ""
        )
        + (f"\n  {qc_summary_path}" if qc_summary_path else "")
        + f"\n  {out_dir / SUMMARY_NAME}",
        flush=True,
    )


if __name__ == "__main__":
    main()
