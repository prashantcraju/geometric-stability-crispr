#!/usr/bin/env python3
"""
Magnitude-matched coherence illustration (Replogle flagship).

Two-row figure (extreme + typical), so the panel defines Sp geometrically
without overclaiming separability from the 1-in-n residual tail.

Selection (from frozen_sp_scores.csv, within Replogle 2022 (CRISPRi)):
  1. Report frac_sp_variance_remaining via rank(Sp)~rank(mag) (pathway tables).
  2. Magnitude floor: both genes must have |μ| ≥ max(dataset median, 1.5)
     so arrows are visible and Sp is well estimated.
  3. Candidate pairs: relative magnitude difference ≤ 0.05 among floor-passers.
  4. Extreme row: maximize |ΔSp|; ties → max |Δ rank-resid|, then
     lexicographic (gene_a, gene_b) with names sorted within the pair.
  5. Typical row: |ΔSp| nearest the candidate median; same ties.

Projection (displacement frame — not PC1/PC2), per perturbation:
  u = unit mean-response; v = leading PC of displacements after removing u;
  plot (d·u, d·v); color by per-cell cosine to u. Controls gray at origin.
  Each panel has its own (u, v) basis; x-range is shared across panels.

Verification: recomputed Sp vs frozen_sp_scores.csv (soft atol on Colab).

Usage:
  python magnitude_matched_coherence_illustration.py --select-only
  python magnitude_matched_coherence_illustration.py \\
      --frozen-sp shesha-crispr/frozen_sp_scores.csv
  python magnitude_matched_coherence_illustration.py --cmap brand
  python magnitude_matched_coherence_illustration.py --simple-color
  python magnitude_matched_coherence_illustration.py --no-show

Colab (inline plot — use %run, not !python):
  %matplotlib inline
  %run magnitude_matched_coherence_illustration.py \\
      --h5ad /tmp/pertpy_data/replogle_2022_k562_essential.h5ad
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_COLAB_NUMPY_FIX = (
    "Broken / mixed NumPy install (Colab: 'numpy.dtype size changed' or "
    "'_blas_supports_fpe' after a partial pip upgrade).\n"
    "Fix in a fresh cell, then Runtime → Restart session:\n"
    "  !pip install -q --force-reinstall 'numpy>=2.0,<2.3' "
    "'pandas>=2.2,<2.3' 'scipy==1.14.1' "
    "'scanpy==1.11.1' 'anndata==0.11.4' matplotlib scikit-learn\n"
    "Then re-run with %run (not !python) so the figure displays inline."
)

_COLAB_ZARR_FIX = (
    "anndata 0.11.x cannot import zarr 3.x "
    "('zarr-python major version > 2 is not supported').\n"
    "Fix in a fresh cell, then Runtime → Restart session:\n"
    "  !pip install -q 'zarr==2.18.7' 'numcodecs==0.15.1'\n"
    "Then re-run with %run (not !python). Selection already wrote "
    "magnitude_matched_coherence_pair.json; the restart only unblocks "
    "loading the Replogle h5ad."
)

try:
    import numpy as np
    import pandas as pd
except ValueError as exc:
    if "dtype size changed" in str(exc) or "binary incompatibility" in str(exc):
        raise RuntimeError(_COLAB_NUMPY_FIX) from exc
    raise

import pipeline_config as cfg
from pipeline_core import calculate_sp
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir

DATASET = "Replogle 2022 (CRISPRi)"
REL_MAG_TOL = 0.05
# Bit-identical target (same sklearn / platform as freeze run).
SP_MATCH_ATOL = 1e-6
# Colab TruncatedSVD / BLAS drift is typically ~1e-4–1e-3; still fine for
# a 3-decimal caption. Abort only above the soft ceiling unless --strict-sp-match.
SP_MATCH_ATOL_SOFT = 1e-3
CTRL_COLOR = "#C8C8C8"
ARROW_COLOR = "#1A1A1A"
# Designer palette: low cosine → red, high cosine → blue (shared colorbar).
# Midway between the pale source swatches (#b25f6e / #81b0cc) and the
# near-black ends that overpowered the control cloud.
COSINE_LOW_COLOR = "#8B3A48"
COSINE_HIGH_COLOR = "#2F6F93"
COSINE_MID_COLOR = "#B0B0B0"
SIMPLE_PERT_COLOR = COSINE_HIGH_COLOR  # single-accent fallback
DEFAULT_CMAP = "brand"
PERT_POINT_SIZE = 14
CTRL_POINT_SIZE = 9
PERT_ALPHA = 0.88
CTRL_ALPHA = 0.35

# Typography hierarchy: supertitle > row header > panel title > axis / stats
SUPTITLE_FONT = 12
ROW_FONT = 10.5
PANEL_FONT = 8
STATS_FONT = 7.5
LABEL_FONT = 10
ANNOT_FONT = 8


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _sp_rank_residuals_on_magnitude(
    mag: np.ndarray, sp: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """
    Rank-scale OLS residual of Sp | magnitude — same definition as
    pathway_analysis._residual_diagnostics / cell_quality_partial.residual_diag.

    Returns (rank_residuals, frac_sp_variance_remaining, spearman_rho).
    frac ≈ 1 − ρ_Spearman² (identical under no ties).
    """
    from scipy.stats import rankdata, spearmanr

    mag = np.asarray(mag, dtype=float)
    sp = np.asarray(sp, dtype=float)
    rsp, rmag = rankdata(sp), rankdata(mag)
    z = np.column_stack([np.ones(len(sp)), rmag])
    coef, _, _, _ = np.linalg.lstsq(z, rsp, rcond=None)
    resid = rsp - z @ coef
    ss_tot = float(np.sum((rsp - rsp.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    frac_remaining = ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho, _ = spearmanr(mag, sp)
    return resid, frac_remaining, float(rho)


TIE_BREAK_RULE = (
    "maximize |ΔSp|; then maximize mean |μ| of the pair "
    "(prefers the better-estimated, more visible arrow when Sp ties "
    "at manuscript precision); then maximize |Δ rank-residual|; then "
    "lexicographic (gene_a, gene_b) with names sorted within the pair"
)
# Well-estimated Sp + visible mean-response arrow
DEFAULT_MAG_FLOOR_ABS = 1.5
# Treat |ΔSp| as tied when equal to this absolute tolerance (covers
# WDR5/NOL6-class 3-decimal Sp ties with the same low-Sp partner).
DELTA_SP_TIE_ATOL = 1e-4


def _is_ribosomal(name: str) -> bool:
    s = str(name).upper()
    return s.startswith("RPL") or s.startswith("RPS")


def _both_ribosomal(a: str, b: str) -> bool:
    return _is_ribosomal(a) and _is_ribosomal(b)


def _pair_sort_key(
    i: int,
    j: int,
    *,
    d_sp: float,
    d_resid: float,
    mean_mag: float,
    names: np.ndarray,
    prefer_large_d_sp: bool,
    target_d: float | None = None,
    target_mean_mag: float | None = None,
) -> tuple:
    """Deterministic sort key for candidate pairs (lower = better)."""
    a, b = sorted([str(names[i]), str(names[j])])
    # Quantize |ΔSp| so near-ties at manuscript precision share a bin
    d_sp_q = round(float(d_sp) / DELTA_SP_TIE_ATOL) * DELTA_SP_TIE_ATOL
    if prefer_large_d_sp:
        # Extreme: max |ΔSp|, then max mean |μ| (pins WDR5 over NOL6)
        return (-d_sp_q, -float(mean_mag), -float(d_resid), a, b)
    # Typical: nearest median |ΔSp|; do not prefer large |μ| (that pulls
    # high-magnitude pairs and stretches the shared x-axis). Prefer |μ|
    # near the extreme pair so arrows stay comparable across rows.
    mag_ref = float(target_mean_mag) if target_mean_mag is not None else float(mean_mag)
    return (abs(d_sp_q - float(target_d)), abs(float(mean_mag) - mag_ref), a, b)


def _pair_record(
    sub: pd.DataFrame,
    sp: np.ndarray,
    resid: np.ndarray,
    i: int,
    j: int,
    *,
    dataset: str,
    role: str,
    n: int,
    rel_mag_tol: float,
    n_candidates: int,
    median_abs_d_sp: float,
    mag_floor: float,
) -> dict:
    if sp[i] > sp[j] or (
        sp[i] == sp[j]
        and str(sub.iloc[i]["perturbation"]) < str(sub.iloc[j]["perturbation"])
    ):
        hi, lo = i, j
    else:
        hi, lo = j, i

    def _row(idx: int) -> dict:
        r = sub.iloc[idx]
        return {
            "perturbation": str(r["perturbation"]),
            "stability": float(r["stability"]),
            "magnitude": float(r["magnitude"]),
            "sp_residual_rank": float(resid[idx]),
            "n_cells": int(r["n_cells"]) if "n_cells" in r and pd.notna(r["n_cells"]) else None,
            "modality": str(r["modality"]) if "modality" in r else cfg.DATASETS[dataset]["modality"],
            "cell_type": str(r["cell_type"]) if "cell_type" in r else cfg.DATASETS[dataset]["cell_type"],
            "sp_percentile_strict": float(100.0 * (sp < sp[idx]).mean()),
            "sp_percentile": float(100.0 * (sp <= sp[idx]).mean()),
            "dataset": dataset,
        }

    hi_row, lo_row = _row(hi), _row(lo)
    mean_m = 0.5 * (hi_row["magnitude"] + lo_row["magnitude"])
    abs_d_resid = abs(hi_row["sp_residual_rank"] - lo_row["sp_residual_rank"])
    return {
        "role": role,
        "dataset": dataset,
        "n_perturbations": int(n),
        "rel_mag_tol": float(rel_mag_tol),
        "mag_floor": float(mag_floor),
        "n_candidate_pairs": int(n_candidates),
        "relative_magnitude_difference": float(
            abs(hi_row["magnitude"] - lo_row["magnitude"]) / max(mean_m, 1e-12)
        ),
        "abs_sp_difference": float(abs(hi_row["stability"] - lo_row["stability"])),
        "abs_residual_sp_difference": float(abs_d_resid),
        "median_abs_sp_difference_among_candidates": float(median_abs_d_sp),
        "high_sp": hi_row,
        "low_sp": lo_row,
        "tie_break_rule": TIE_BREAK_RULE,
    }


def select_magnitude_matched_pairs(
    df: pd.DataFrame,
    *,
    dataset: str = DATASET,
    rel_mag_tol: float = REL_MAG_TOL,
    mag_floor: float | None = None,
) -> dict:
    """
    Extreme + typical magnitude-matched pairs on |ΔSp|, with a magnitude floor.

    Header residual fraction uses rank-OLS (pathway tables). Pair selection
    uses |ΔSp| among caliper-matched pairs where both genes clear the floor
    (default max(dataset median |μ|, 1.5)), so arrows stay visible and Sp is
    not noise-dominated. Fully deterministic tie-break.
    """
    sub = df[df["dataset"].astype(str).map(cfg.resolve_dataset_name) == dataset].copy()
    if sub.empty:
        raise ValueError(f"No rows for {dataset!r} in frozen Sp table")
    # Stable row order before indexing
    sub = (
        sub.dropna(subset=["magnitude", "stability"])
        .sort_values(["perturbation", "magnitude", "stability"], kind="mergesort")
        .reset_index(drop=True)
    )
    n = len(sub)
    if n < 2:
        raise ValueError(f"{dataset}: need ≥2 perturbations, got {n}")

    mag = sub["magnitude"].to_numpy(dtype=float)
    sp = sub["stability"].to_numpy(dtype=float)
    names = sub["perturbation"].astype(str).to_numpy()
    resid, frac_rem, rho_sp = _sp_rank_residuals_on_magnitude(mag, sp)

    med_mag = float(np.median(mag))
    if mag_floor is None:
        mag_floor = max(med_mag, DEFAULT_MAG_FLOOR_ABS)
    mag_floor = float(mag_floor)

    mean_mag = 0.5 * (mag[:, None] + mag[None, :])
    rel = np.abs(mag[:, None] - mag[None, :]) / np.maximum(mean_mag, 1e-12)
    d_resid = np.abs(resid[:, None] - resid[None, :])
    d_sp = np.abs(sp[:, None] - sp[None, :])

    iu, ju = np.triu_indices(n, k=1)
    ok = (
        (rel[iu, ju] <= rel_mag_tol)
        & (mag[iu] >= mag_floor)
        & (mag[ju] >= mag_floor)
    )
    if not np.any(ok):
        raise ValueError(
            f"{dataset}: no pairs with rel|Δmag| ≤ {rel_mag_tol} and both "
            f"|μ| ≥ {mag_floor:g} (median |μ| = {med_mag:g})"
        )
    i_cand, j_cand = iu[ok], ju[ok]
    d_sp_cand = d_sp[i_cand, j_cand]
    d_resid_cand = d_resid[i_cand, j_cand]
    med_d_sp = float(np.median(d_sp_cand))
    n_cand = int(ok.sum())
    n_above_floor = int(np.sum(mag >= mag_floor))

    mean_mag_cand = 0.5 * (mag[i_cand] + mag[j_cand])

    # Extreme: max |ΔSp|, then max mean |μ| (WDR5 vs NOL6-class ties)
    ex_keys = [
        (
            _pair_sort_key(
                int(i_cand[k]),
                int(j_cand[k]),
                d_sp=float(d_sp_cand[k]),
                d_resid=float(d_resid_cand[k]),
                mean_mag=float(mean_mag_cand[k]),
                names=names,
                prefer_large_d_sp=True,
            ),
            k,
        )
        for k in range(n_cand)
    ]
    ex_keys.sort()
    k_ex = ex_keys[0][1]
    i_ex, j_ex = int(i_cand[k_ex]), int(j_cand[k_ex])
    max_d_sp = float(d_sp_cand[k_ex])
    n_tied_extreme = int(
        np.sum(np.isclose(d_sp_cand, max_d_sp, rtol=0.0, atol=DELTA_SP_TIE_ATOL))
    )

    extreme_mean_mag = float(mean_mag_cand[k_ex])

    # Typical: |ΔSp| nearest median, ≠ extreme; |μ| near the extreme pair.
    # Prefer a mixed (non both-ribosomal) pair so the row does not look like
    # a deliberate RPL/RPS comparison.
    ty_keys = []
    for k in range(n_cand):
        ii, jj = int(i_cand[k]), int(j_cand[k])
        if {ii, jj} == {i_ex, j_ex}:
            continue
        ty_keys.append(
            (
                _pair_sort_key(
                    ii,
                    jj,
                    d_sp=float(d_sp_cand[k]),
                    d_resid=float(d_resid_cand[k]),
                    mean_mag=float(mean_mag_cand[k]),
                    names=names,
                    prefer_large_d_sp=False,
                    target_d=med_d_sp,
                    target_mean_mag=extreme_mean_mag,
                ),
                k,
            )
        )
    if not ty_keys:
        raise ValueError("Could not find a distinct typical pair above magnitude floor")
    ty_keys.sort()
    k_ty = None
    typical_skipped_ribosomal = False
    for _key, k in ty_keys:
        ii, jj = int(i_cand[k]), int(j_cand[k])
        if _both_ribosomal(names[ii], names[jj]):
            typical_skipped_ribosomal = True
            continue
        k_ty = k
        break
    if k_ty is None:
        k_ty = ty_keys[0][1]
        typical_skipped_ribosomal = False  # had to keep a ribosomal pair
    i_ty, j_ty = int(i_cand[k_ty]), int(j_cand[k_ty])

    extreme = _pair_record(
        sub, sp, resid, i_ex, j_ex,
        dataset=dataset, role="extreme", n=n, rel_mag_tol=rel_mag_tol,
        n_candidates=n_cand, median_abs_d_sp=med_d_sp, mag_floor=mag_floor,
    )
    typical = _pair_record(
        sub, sp, resid, i_ty, j_ty,
        dataset=dataset, role="typical", n=n, rel_mag_tol=rel_mag_tol,
        n_candidates=n_cand, median_abs_d_sp=med_d_sp, mag_floor=mag_floor,
    )
    typical["skipped_both_ribosomal_nearer_median"] = bool(
        typical_skipped_ribosomal
        and not _both_ribosomal(names[i_ty], names[j_ty])
    )

    selection_rule = (
        f"Within {dataset} (n={n}), require both genes |μ| ≥ {mag_floor:g} "
        f"(= max(median |μ|={med_mag:g}, {DEFAULT_MAG_FLOOR_ABS:g}); "
        f"n_above_floor={n_above_floor}); among pairs with "
        f"|mag_i − mag_j| / mean(mag) ≤ {rel_mag_tol} (n_candidates={n_cand:,}), "
        f"take (i) max |ΔSp| (extreme) and (ii) |ΔSp| nearest the candidate "
        f"median (typical; skip both-ribosomal pairs when a nearer mixed "
        f"pair exists). Tie-break: {TIE_BREAK_RULE}. "
        f"Header frac_sp_variance_remaining uses rank-OLS (pathway tables), "
        f"not the pair-selection statistic."
    )
    return {
        "selection_rule": selection_rule,
        "selection_statistic": "|ΔSp| among magnitude-caliper pairs above mag floor",
        "residual_definition": (
            "rank_OLS (header only): resid = rank(Sp) − OLS(rank(Sp) ~ rank(magnitude)); "
            "identical to pathway_analysis / cell_quality_partial "
            "frac_sp_variance_remaining (≈ 1 − Spearman(Sp, mag)²)"
        ),
        "tie_break_rule": TIE_BREAK_RULE,
        "mag_floor": mag_floor,
        "median_magnitude": med_mag,
        "n_perturbations_above_mag_floor": n_above_floor,
        "n_tied_at_max_abs_sp_difference": n_tied_extreme,
        "dataset": dataset,
        "n_perturbations": int(n),
        "rel_mag_tol": float(rel_mag_tol),
        "n_candidate_pairs": n_cand,
        "spearman_sp_magnitude": float(rho_sp),
        "frac_sp_variance_remaining_after_magnitude": float(frac_rem),
        "one_minus_spearman_rho_sq": float(1.0 - rho_sp**2),
        "median_abs_sp_difference_among_candidates": med_d_sp,
        "extreme": extreme,
        "typical": typical,
        "high_sp": extreme["high_sp"],
        "low_sp": extreme["low_sp"],
        "abs_sp_difference": extreme["abs_sp_difference"],
        "relative_magnitude_difference": extreme["relative_magnitude_difference"],
        "note": (
            "Extreme row = largest |ΔSp| among magnitude-matched pairs above "
            f"the |μ| floor ({mag_floor:g}), so mean-shift arrows are visible "
            "and Sp is not noise-dominated. Typical row = median |ΔSp| among "
            "the same candidates (usually near-invisible). "
            f"frac_sp_variance_remaining = {frac_rem:.3f} (rank-OLS, pathway "
            "definition) is reported in the header only."
        ),
    }


# Back-compat name used by --self-test / older call sites
def select_magnitude_matched_pair(*args, **kwargs) -> dict:
    out = select_magnitude_matched_pairs(*args, **kwargs)
    return out["extreme"]


# ---------------------------------------------------------------------------
# Displacement frame
# ---------------------------------------------------------------------------


def displacement_frame(
    X_ctrl: np.ndarray,
    X_pert: np.ndarray,
) -> dict:
    """
    Build the (u, v) displacement frame for one perturbation.

    u = unit mean-response direction (control → pert centroid).
    v = leading PC of displacements after removing the u-component.
    Coordinates: (d·u, d·v). Per-cell cosine to u is what Sp averages.
    """
    X_ctrl = np.asarray(X_ctrl, dtype=float)
    X_pert = np.asarray(X_pert, dtype=float)
    c = X_ctrl.mean(axis=0)
    d_pert = X_pert - c
    d_ctrl = X_ctrl - c

    mean_shift = d_pert.mean(axis=0)
    mag = float(np.linalg.norm(mean_shift))
    if mag < 1e-6:
        raise ValueError("Near-zero mean shift; cannot build displacement frame")
    u = mean_shift / mag

    # Residual cloud of perturbed cells (remove mean-response component)
    proj_u = d_pert @ u
    resid = d_pert - proj_u[:, None] * u
    # Leading PC of residuals via SVD (right singular vector)
    # Center residuals for PCA; the cloud is already roughly centered if mean
    # shift is exactly along u, but center anyway for a clean leading axis.
    resid_c = resid - resid.mean(axis=0)
    _, s, vt = np.linalg.svd(resid_c, full_matrices=False)
    if s.size == 0 or s[0] < 1e-12:
        # Degenerate: all mass on u — invent an orthonormal filler
        rng = np.random.default_rng(0)
        v = rng.normal(size=u.shape)
        v = v - np.dot(v, u) * u
        v = v / (np.linalg.norm(v) + 1e-12)
    else:
        v = vt[0].copy()
        # Enforce orthogonality to u (numerical cleanup)
        v = v - np.dot(v, u) * u
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            raise ValueError("Residual PC collapsed onto u")
        v = v / vn
        # Stable sign: more mass on the positive side
        if np.sum(resid @ v) < 0:
            v = -v

    xy_pert = np.column_stack([d_pert @ u, d_pert @ v])
    xy_ctrl = np.column_stack([d_ctrl @ u, d_ctrl @ v])

    norms = np.linalg.norm(d_pert, axis=1)
    valid = norms > 1e-6
    cosines = np.full(len(d_pert), np.nan, dtype=float)
    cosines[valid] = (d_pert[valid] @ u) / norms[valid]
    sp = float(np.mean(cosines[valid])) if valid.any() else float("nan")

    # Cross-check against the shared Sp definition
    metrics = calculate_sp(X_ctrl, X_pert)
    return {
        "u": u,
        "v": v,
        "magnitude": mag,
        "xy_pert": xy_pert,
        "xy_ctrl": xy_ctrl,
        "cosines": cosines,
        "sp": sp,
        "sp_calculate_sp": float(metrics["stability"]),
        "magnitude_calculate_sp": float(metrics["magnitude"]),
        "n_pert": int(X_pert.shape[0]),
        "n_ctrl": int(X_ctrl.shape[0]),
        "n_valid_cosines": int(valid.sum()),
        "frac_negative_cosine": float(np.mean(cosines[valid] < 0)) if valid.any() else 0.0,
        "median_abs_v": float(np.median(np.abs(xy_pert[:, 1]))),
        "median_abs_angle_deg": float(
            np.median(
                np.degrees(
                    np.arctan2(
                        np.abs(xy_pert[:, 1]),
                        np.maximum(xy_pert[:, 0], 1e-12),
                    )
                )
            )
        ),
        "bimodality": diagnose_partial_penetrance(xy_pert[:, 0], mag),
    }


def diagnose_partial_penetrance(proj_u: np.ndarray, magnitude: float) -> dict:
    """
    Flag classic efficiency / partial-penetrance clouds: a mass near the
    control centroid along u plus a long responding tail.
    """
    proj_u = np.asarray(proj_u, dtype=float)
    mag = max(float(magnitude), 1e-12)
    # Non-responders: little progress along the mean-response axis
    near_frac = float(np.mean(proj_u < 0.25 * mag))
    far_frac = float(np.mean(proj_u > 1.5 * mag))
    # Simple 2-bin gap score on the u-projection
    try:
        from sklearn.mixture import GaussianMixture

        x = proj_u.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
        means = np.sort(gmm.means_.ravel())
        gap = float(means[1] - means[0]) / mag
        weights = gmm.weights_[np.argsort(gmm.means_.ravel())]
        flagged = bool(near_frac >= 0.25 and gap >= 1.0 and weights[0] >= 0.2)
    except Exception:
        gap = float("nan")
        weights = [float("nan"), float("nan")]
        flagged = bool(near_frac >= 0.30 and far_frac >= 0.10)

    return {
        "frac_near_control_along_u": near_frac,
        "frac_far_tail_along_u": far_frac,
        "gmm_mean_gap_over_mag": gap,
        "partial_penetrance_flag": flagged,
        "note": (
            "Large near-control mass plus a responding tail is the classic "
            "partial-penetrance / efficiency-heterogeneity signature; "
            "the responder-filter sensitivity analysis (SI) showed global "
            "Sp-magnitude redundancy is not efficiency-driven, but individual "
            "low-Sp exemplars can still be."
            if flagged
            else "No strong partial-penetrance signature on the u-projection."
        ),
    }


def _self_test() -> None:
    """Unit checks for selection + displacement frame (no frozen CSV / h5ad)."""
    rng = np.random.default_rng(0)
    rows = []
    for i, (sp, mag) in enumerate(
        [
            (0.90, 2.00),
            (0.10, 2.05),
            (0.50, 2.02),
            (0.85, 2.01),
            (0.20, 3.00),
            (0.55, 0.40),  # below mag floor; must not win
            (0.05, 0.41),
        ]
    ):
        rows.append(
            {
                "dataset": DATASET,
                "perturbation": f"G{i}",
                "stability": sp,
                "magnitude": mag,
                "modality": "CRISPRi",
                "cell_type": "K562",
                "n_cells": 50,
            }
        )
    # Floor 1.5: G0 vs G1 is the max |ΔSp| among caliper pairs
    pair = select_magnitude_matched_pair(
        pd.DataFrame(rows), rel_mag_tol=0.05, mag_floor=1.5
    )
    assert pair["high_sp"]["perturbation"] == "G0"
    assert pair["low_sp"]["perturbation"] == "G1"
    assert abs(pair["abs_sp_difference"] - 0.8) < 1e-9

    # Coherent cloud along e0; incoherent fans in e1
    n, d = 80, 50
    Xc = rng.normal(0, 0.05, size=(n, d))
    X_hi = (
        Xc.mean(0)
        + np.array([2.0] + [0.0] * (d - 1))
        + rng.normal(0, 0.05, size=(n, d))
    )
    X_lo = Xc.mean(0) + np.column_stack(
        [
            np.full(n, 2.0),
            rng.normal(0, 1.5, size=n),
            rng.normal(0, 0.05, size=(n, d - 2)),
        ]
    )
    f_hi = displacement_frame(Xc, X_hi)
    f_lo = displacement_frame(Xc, X_lo)
    assert f_hi["sp"] > 0.9
    assert f_lo["sp"] < f_hi["sp"]
    assert abs(f_hi["sp"] - f_hi["sp_calculate_sp"]) < 1e-9
    print("self-test OK", flush=True)


# ---------------------------------------------------------------------------
# Data load (frozen pipeline, keep AnnData)
# ---------------------------------------------------------------------------


def _import_scanpy():
    try:
        import zarr
        zarr_major = int(str(zarr.__version__).split(".")[0])
    except Exception:
        zarr_major = None
    if zarr_major is not None and zarr_major > 2:
        raise RuntimeError(_COLAB_ZARR_FIX)

    try:
        import scanpy as sc
        return sc
    except ImportError as exc:
        if "zarr" in str(exc).lower():
            raise RuntimeError(_COLAB_ZARR_FIX) from exc
        raise
    except AttributeError as exc:
        if "_blas_supports_fpe" in str(exc) or "numpy" in str(exc).lower():
            raise RuntimeError(_COLAB_NUMPY_FIX) from exc
        raise


def load_dataset_pca(
    dataset_name: str = DATASET,
    *,
    h5ad_path: Path | None = None,
):
    """Full frozen preprocess → AnnData with X_pca (same path as run_frozen_main)."""
    sc = _import_scanpy()

    from pipeline_core import (
        _extract_adata,
        load_raw,
        materialize_min_cells,
        preprocess,
        setup_cache,
    )

    setup_cache()
    sc.settings.datasetdir = Path(
        os.environ.get("SCVERSE_DATADIR", str(cfg.CACHE_DIR))
    )
    dataset_name = cfg.resolve_dataset_name(dataset_name)
    print(f"\n>>> Loading {dataset_name} for displacement-frame illustration", flush=True)
    raw = load_raw(dataset_name, sc=sc, prefer_local=True, h5ad_path=h5ad_path)
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset_name, sc)
    adata, valid, counts = materialize_min_cells(
        adata, pert_col, ctrl_label,
        min_cells=cfg.MIN_CELLS,
        max_control_cells=cfg.MAX_CONTROL_CELLS,
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
    return adata, pert_col, ctrl_label, valid


def extract_pca(adata, pert_col: str, ctrl_label: str, pert: str):
    X_ctrl = np.asarray(adata.obsm["X_pca"][adata.obs[pert_col].astype(str) == ctrl_label])
    X_pert = np.asarray(adata.obsm["X_pca"][adata.obs[pert_col].astype(str) == pert])
    if X_pert.shape[0] == 0:
        raise KeyError(f"Perturbation {pert!r} not found after preprocess")
    return X_ctrl, X_pert


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _cosine_limits(frames: list[dict], *, diverging: bool) -> tuple[float, float]:
    """Fixed theoretical cosine range for brand/diverging maps (±1)."""
    if diverging:
        return -1.0, 1.0
    all_c = np.concatenate([f["cosines"][np.isfinite(f["cosines"])] for f in frames])
    if all_c.size == 0:
        return 0.0, 1.0
    if np.any(all_c < 0):
        return -1.0, 1.0
    return 0.0, 1.0


def _resolve_cmap(cmap: str):
    """Return (cmap, is_diverging). Brand = red → gray → blue centered at 0."""
    from matplotlib.colors import LinearSegmentedColormap

    name = (cmap or DEFAULT_CMAP).strip()
    if name.lower() in {"brand", "redblue", "rb", "paper"}:
        cm = LinearSegmentedColormap.from_list(
            "coherence_brand",
            [
                (0.0, COSINE_LOW_COLOR),
                (0.40, "#C46B7A"),
                (0.5, COSINE_MID_COLOR),
                (0.60, "#6AA3C2"),
                (1.0, COSINE_HIGH_COLOR),
            ],
            N=256,
        )
        return cm, True
    return name, False


def plot_extreme_and_typical(
    selection: dict,
    frames_by_role: dict[str, tuple[dict, dict]],
    *,
    out_stem: Path,
    cmap: str = DEFAULT_CMAP,
    simple_color: bool = False,
    show: bool = True,
) -> Path:
    """
    2×2 displacement-frame figure: row0 = extreme residual pair,
    row1 = typical (median) residual pair. Shared x-range + shared colorbar.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    cmap_obj, diverging = _resolve_cmap(cmap)

    roles = ["extreme", "typical"]
    row_titles = {
        "extreme": (
            f"Extreme  ·  |ΔShesha| = {selection['extreme']['abs_sp_difference']:.3f}"
        ),
        "typical": (
            f"Typical difference  ·  "
            f"|ΔShesha| = {selection['typical']['abs_sp_difference']:.3f}"
        ),
    }
    col_labels = ["Higher Shesha", "Lower Shesha"]

    all_frames = []
    for role in roles:
        all_frames.extend(frames_by_role[role])

    # Shared x across all panels so extreme spread stays legible in context
    all_xy = np.vstack(
        [f["xy_pert"] for f in all_frames] + [f["xy_ctrl"] for f in all_frames]
    )
    arrow_lens = {
        role: 0.5
        * (
            selection[role]["high_sp"]["magnitude"]
            + selection[role]["low_sp"]["magnitude"]
        )
        for role in roles
    }
    pad = 0.08 * max(
        all_xy[:, 0].max() - all_xy[:, 0].min(),
        all_xy[:, 1].max() - all_xy[:, 1].min(),
        1.0,
    )
    xlim = (
        float(all_xy[:, 0].min()) - pad,
        float(max(all_xy[:, 0].max(), max(arrow_lens.values())) + pad),
    )
    ymax = float(np.max(np.abs(all_xy[:, 1]))) + pad
    ylim = (-ymax, ymax)
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    # Axes box aspect = data height/width so equal-aspect fills (no letterbox).
    box_aspect = dy / max(dx, 1e-9)

    vmin, vmax = _cosine_limits(all_frames, diverging=diverging and not simple_color)
    if diverging and not simple_color:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cell_type = selection["extreme"]["high_sp"].get("cell_type", "")

    # Nested gridspec: thin horizontal row header + two panels (no rotated text).
    # Taller figure matches equal-aspect data so panels sit close together.
    fig = plt.figure(figsize=(10.8, 11.2))
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.10,
        left=0.09,
        right=0.88,
        top=0.94,
        bottom=0.05,
    )
    axes = np.empty((2, 2), dtype=object)
    scatters = []

    for r, role in enumerate(roles):
        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[r],
            height_ratios=[0.055, 1.0],
            hspace=0.16,
            wspace=0.06,
        )
        ax_hdr = fig.add_subplot(inner[0, :])
        ax_hdr.axis("off")
        ax_hdr.text(
            0.0,
            0.15,
            row_titles[role],
            transform=ax_hdr.transAxes,
            ha="left",
            va="center",
            fontsize=ROW_FONT,
            fontweight="bold",
            color="#222222",
        )

        pair = selection[role]
        frames = frames_by_role[role]
        rows = [pair["high_sp"], pair["low_sp"]]
        arrow_len = arrow_lens[role]
        for c, (frame, row, lab) in enumerate(zip(frames, rows, col_labels)):
            ax = fig.add_subplot(inner[1, c])
            axes[r, c] = ax
            ax.scatter(
                frame["xy_ctrl"][:, 0],
                frame["xy_ctrl"][:, 1],
                s=CTRL_POINT_SIZE,
                c=CTRL_COLOR,
                alpha=CTRL_ALPHA,
                linewidths=0,
                zorder=1,
                rasterized=True,
            )
            if simple_color:
                sc = ax.scatter(
                    frame["xy_pert"][:, 0],
                    frame["xy_pert"][:, 1],
                    s=PERT_POINT_SIZE,
                    c=SIMPLE_PERT_COLOR,
                    alpha=PERT_ALPHA,
                    linewidths=0,
                    zorder=3,
                    rasterized=True,
                )
            else:
                sc = ax.scatter(
                    frame["xy_pert"][:, 0],
                    frame["xy_pert"][:, 1],
                    s=PERT_POINT_SIZE,
                    c=frame["cosines"],
                    cmap=cmap_obj,
                    norm=norm,
                    alpha=PERT_ALPHA,
                    linewidths=0,
                    zorder=3,
                    rasterized=True,
                )
            scatters.append(sc)
            ax.annotate(
                "",
                xy=(arrow_len, 0.0),
                xytext=(0.0, 0.0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ARROW_COLOR,
                    lw=1.6,
                    mutation_scale=12,
                ),
                zorder=4,
            )
            ax.axhline(0.0, color="#DDDDDD", lw=0.6, zorder=0)
            ax.axvline(0.0, color="#DDDDDD", lw=0.6, zorder=0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_box_aspect(box_aspect)
            ax.tick_params(labelsize=ANNOT_FONT)
            if r == 1:
                ax.set_xlabel(
                    r"$\mathbf{d}\cdot\mathbf{u}$",
                    fontsize=LABEL_FONT,
                    fontweight="bold",
                )
            if c == 0:
                ax.set_ylabel(
                    r"$\mathbf{d}\cdot\mathbf{v}$",
                    fontsize=LABEL_FONT,
                    fontweight="bold",
                )
            bim = frame.get("bimodality") or {}
            bim_tag = "*" if bim.get("partial_penetrance_flag") else ""
            ax.set_title(
                f"{row['perturbation']}{bim_tag}  ·  {lab}\n"
                f"Shesha = {row['stability']:.3f}  "
                f"(≈{row['sp_percentile']:.0f}th pct)  ·  "
                f"|μ| = {row['magnitude']:.2f}",
                fontsize=PANEL_FONT,
                fontweight="normal",
                pad=2,
                linespacing=1.2,
                color="#333333",
            )
            sns.despine(ax=ax)

    legend_elems = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=CTRL_COLOR,
            markersize=6,
            label="control (same cells, panel-specific basis)",
        ),
        Line2D(
            [0], [0],
            color=ARROW_COLOR,
            lw=1.6,
            label="mean response (length matched within row)",
        ),
    ]
    axes[0, 0].legend(
        handles=legend_elems,
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    if not simple_color:
        cbar = fig.colorbar(
            scatters[0], ax=axes.ravel().tolist(), fraction=0.028, pad=0.015
        )
        cbar.set_label(
            "per-cell cosine to mean response\n(Shesha = mean; 0 = orthogonal)",
            fontsize=STATS_FONT,
        )
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=ANNOT_FONT)

    cell_bit = f", {cell_type}" if cell_type else ""
    fig.suptitle(
        f"Displacement-frame Shesha  ·  {selection['dataset']}{cell_bit}",
        fontsize=SUPTITLE_FONT,
        fontweight="bold",
        y=0.985,
    )

    pdf = out_stem.with_suffix(".pdf")
    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  saved figure → {pdf} / {svg.name} / {png.name}", flush=True)
    if show:
        _display_figure(fig, png)
    plt.close(fig)
    return pdf


# Alias kept for any external callers
def plot_pair(pair, frame_hi, frame_lo, **kwargs):
    sel = {
        "dataset": pair.get("dataset", DATASET),
        "frac_sp_variance_remaining_after_magnitude": float("nan"),
        "extreme": pair,
        "typical": pair,
    }
    return plot_extreme_and_typical(
        sel, {"extreme": (frame_hi, frame_lo), "typical": (frame_hi, frame_lo)}, **kwargs
    )


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ in {
            "ZMQInteractiveShell",  # Jupyter / Colab
            "Shell",  # some Colab builds
        } or "google.colab" in str(type(ip))
    except Exception:
        return False


def _display_figure(fig, png: Path) -> None:
    """Show the figure in Colab/Jupyter, else OS viewer / plt.show."""
    import matplotlib.pyplot as plt

    # Colab / Jupyter: inline display (works with %run; not with !python)
    if _in_notebook():
        try:
            from IPython.display import Image, display

            display(Image(filename=str(png)))
            return
        except Exception:
            try:
                plt.show()
                return
            except Exception as exc:  # pragma: no cover
                print(f"  (notebook display failed: {exc})", flush=True)

    import shutil
    import subprocess

    opener = shutil.which("open") or shutil.which("xdg-open")
    if opener:
        subprocess.run([opener, str(png)], check=False)
    try:
        plt.show(block=False)
        plt.pause(0.25)
    except Exception as exc:  # pragma: no cover
        print(f"  (plt.show unavailable: {exc})", flush=True)


def write_caption(selection: dict, verify: dict, out_path: Path) -> str:
    ex, ty = selection["extreme"], selection["typical"]
    hi_e, lo_e = ex["high_sp"], ex["low_sp"]
    hi_t, lo_t = ty["high_sp"], ty["low_sp"]
    frac = selection["frac_sp_variance_remaining_after_magnitude"]
    rho = selection.get("spearman_sp_magnitude", float("nan"))
    med = selection.get(
        "median_abs_sp_difference_among_candidates",
        ty.get("median_abs_sp_difference_among_candidates", float("nan")),
    )
    n_tie = selection.get("n_tied_at_max_abs_sp_difference", 1)
    mag_floor = selection.get("mag_floor", float("nan"))

    flagged = []
    for role, pair in ("extreme", ex), ("typical", ty):
        for side in ("high_sp", "low_sp"):
            gene = pair[side]["perturbation"]
            bim = (verify.get("bimodality") or {}).get(f"{role}:{gene}")
            if bim and bim.get("partial_penetrance_flag"):
                flagged.append(f"{gene} ({role})")

    skipped_rpl = bool(ty.get("skipped_both_ribosomal_nearer_median", False))
    text = (
        f"Displacement-frame illustration of directional coherence Shesha in "
        f"{selection['dataset']} ({hi_e['modality']}, {hi_e['cell_type']}). "
        f"frac_sp_variance_remaining = {frac:.3f} (rank-OLS; Spearman ρ = "
        f"{rho:.3f}; same definition as pathway tables). Each panel has its "
        f"own (u, v); x shared. "
        f"Selection rule (three criteria): (i) magnitude floor "
        f"|μ| ≥ {mag_floor:g} for both genes; (ii) magnitude caliper "
        f"|Δmag|/mean(mag) ≤ {selection['rel_mag_tol']} "
        f"(n_candidates = {selection['n_candidate_pairs']:,}); "
        f"(iii) extreme = max |ΔShesha|, typical = |ΔShesha| nearest the candidate "
        f"median; ties broken by maximizing mean |μ| of the pair, then "
        f"|Δ rank-residual|, then lexicographic gene names"
        + (
            f" ({n_tie} pairs within {DELTA_SP_TIE_ATOL:g} of the max |ΔShesha|)"
            if n_tie > 1
            else ""
        )
        + ". "
        f"Top row (extreme |ΔShesha| = {ex['abs_sp_difference']:.3f}): "
        f"{hi_e['perturbation']} (Shesha = {hi_e['stability']:.3f}, "
        f"≈{hi_e['sp_percentile']:.0f}th percentile within the dataset; "
        f"|μ| = {hi_e['magnitude']:.2f}) versus "
        f"{lo_e['perturbation']} (Shesha = {lo_e['stability']:.3f}, "
        f"≈{lo_e['sp_percentile']:.0f}th pct; |μ| = {lo_e['magnitude']:.2f}). "
        f"Bottom row (typical |ΔShesha| = {ty['abs_sp_difference']:.3f}, nearest "
        f"candidate median {med:.3f}): "
        f"{hi_t['perturbation']} (Shesha = {hi_t['stability']:.3f}, "
        f"≈{hi_t['sp_percentile']:.0f}th pct; |μ| = {hi_t['magnitude']:.2f}) versus "
        f"{lo_t['perturbation']} (Shesha = {lo_t['stability']:.3f}, "
        f"≈{lo_t['sp_percentile']:.0f}th pct; |μ| = {lo_t['magnitude']:.2f}). "
        f"The typical row is a typical coherence difference under the "
        f"selection rule, not a pair of typical (median-Shesha) perturbations; "
        f"the magnitude floor places both near the high-Shesha end of the "
        f"distribution because Shesha tracks magnitude (ρ = {rho:.3f}). "
        + (
            "A nearer-median pair of two ribosomal genes was skipped so the "
            "row is not read as a deliberate RPL/RPS contrast. "
            if skipped_rpl
            else ""
        )
        + "Gray points are the same control cells in every panel, each projected "
        f"into that panel's own (u, v) basis (u = mean-response direction; "
        f"v = leading residual PC), so y-axes are not comparable across panels; "
        f"x-range is shared so the extreme case remains visibly extreme. "
        f"Cells are colored by cosine to u (Shesha = mean of these; diverging scale "
        f"fixed at ±1 with midpoint 0 = orthogonal). Mean-response arrows are "
        f"length-matched within each row. The 2D view captures the u-component "
        f"exactly but compresses the remaining 49 orthogonal dimensions into "
        f"one axis, so visible angles understate the true 50-dimensional spread."
    )
    if flagged:
        text += (
            f" Asterisks mark partial-penetrance on the u-projection "
            f"({', '.join(flagged)}: near-control mass plus a responding tail). "
            f"This is a per-perturbation observation. The responder-filter "
            f"sensitivity analysis in the SI showed that responder filtering "
            f"does not remove the Shesha-magnitude relationship across datasets, "
            f"so these clouds do not overturn the global result."
        )
    if np.isfinite(verify.get("max_abs_sp_diff", float("nan"))):
        text += (
            f" Recomputed Shesha vs frozen_sp_scores.csv: max |ΔShesha| = "
            f"{verify['max_abs_sp_diff']:.2e}"
            + (
                "."
                if verify.get("exact_match", True)
                else (
                    f" (within Colab soft tolerance "
                    f"{verify.get('atol_soft', SP_MATCH_ATOL_SOFT):g}; "
                    "labels quote frozen values)."
                )
            )
        )
    else:
        text += " Sp verification against the frozen table is pending expression reload."

    out_path.write_text(text + "\n", encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frozen-sp",
        type=Path,
        default=None,
        help="Path to frozen_sp_scores.csv (default: search out-dir / standard paths)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help=f"Dataset to select within (default: {DATASET})",
    )
    parser.add_argument(
        "--rel-mag-tol",
        type=float,
        default=REL_MAG_TOL,
        help="Relative magnitude tolerance (default: 0.05)",
    )
    parser.add_argument(
        "--mag-floor",
        type=float,
        default=None,
        help=(
            "Min |μ| for both genes in a candidate pair "
            f"(default: max(dataset median, {DEFAULT_MAG_FLOOR_ABS}))"
        ),
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Report the selected pair from the CSV; skip loading expression data",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=None,
        help="Optional local Replogle h5ad (else pipeline_core cache / download)",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help=(
            "Colormap for per-cell cosine (default: brand = "
            f"{COSINE_LOW_COLOR}→{COSINE_HIGH_COLOR}; also viridis, Blues, …)"
        ),
    )
    parser.add_argument(
        "--simple-color",
        action="store_true",
        help="Fallback: one accent color for all perturbed cells (no cosine map)",
    )
    parser.add_argument(
        "--skip-digest-guard",
        action="store_true",
        help="Allow a frozen CSV without sp_digest (not for manuscript figures)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run selection + displacement-frame unit checks and exit",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display the figure after saving (default: on)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save only; do not open a figure window / OS viewer",
    )
    parser.add_argument(
        "--strict-sp-match",
        action="store_true",
        help=(
            f"Abort unless |ΔSp| ≤ {SP_MATCH_ATOL:g} (default: allow up to "
            f"{SP_MATCH_ATOL_SOFT:g} with a warning — Colab SVD drift)"
        ),
    )
    parser.add_argument(
        "--sp-match-atol",
        type=float,
        default=None,
        help="Override soft/strict Sp match absolute tolerance",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    out_dir = resolve_out_dir(args.out_dir)
    frozen_path = find_sp_csv(out_dir, args.frozen_sp)
    print(f"Frozen Sp table: {frozen_path}", flush=True)

    if not args.skip_digest_guard:
        from pipeline_core import assert_frozen_sp_compatible

        info = assert_frozen_sp_compatible(frozen_path)
        print(
            f"  guard OK: config_version={info['config_version']}  "
            f"n={info['n_rows']}  digest={info['sp_digest']}",
            flush=True,
        )
    else:
        print("  WARNING: digest guard skipped", flush=True)

    df = load_sp_table(frozen_path)
    selection = select_magnitude_matched_pairs(
        df,
        dataset=args.dataset,
        rel_mag_tol=args.rel_mag_tol,
        mag_floor=args.mag_floor,
    )

    print("\n=== Selection ===", flush=True)
    print(selection["selection_rule"], flush=True)
    frac = selection["frac_sp_variance_remaining_after_magnitude"]
    rho = selection["spearman_sp_magnitude"]
    print(
        f"  frac_sp_variance_remaining = {frac:.4f}  "
        f"(rank-OLS header; Spearman ρ = {rho:.4f}; 1−ρ² = "
        f"{selection['one_minus_spearman_rho_sq']:.4f})",
        flush=True,
    )
    print(
        f"  mag_floor = {selection['mag_floor']:.4g}  "
        f"(median |μ| = {selection['median_magnitude']:.4g}; "
        f"n_above = {selection['n_perturbations_above_mag_floor']})",
        flush=True,
    )
    print(f"  selection statistic: {selection['selection_statistic']}", flush=True)
    print(f"  tie-break: {selection['tie_break_rule']}", flush=True)
    print(
        f"  n tied at max |ΔSp| = "
        f"{selection['n_tied_at_max_abs_sp_difference']}",
        flush=True,
    )
    for role in ("extreme", "typical"):
        pair = selection[role]
        hi, lo = pair["high_sp"], pair["low_sp"]
        print(f"\n  [{role.upper()}]", flush=True)
        print(
            f"    HIGH: {hi['perturbation']}  Sp={hi['stability']:.6f}  "
            f"|μ|={hi['magnitude']:.4f}  ~{hi['sp_percentile']:.1f}th pct",
            flush=True,
        )
        print(
            f"    LOW:  {lo['perturbation']}  Sp={lo['stability']:.6f}  "
            f"|μ|={lo['magnitude']:.4f}  ~{lo['sp_percentile']:.1f}th pct",
            flush=True,
        )
        print(
            f"    |ΔSp|={pair['abs_sp_difference']:.4f}  "
            f"rel|Δmag|={pair['relative_magnitude_difference']:.4f}",
            flush=True,
        )
    print(f"\n  {selection['note']}", flush=True)

    stem = out_dir / "magnitude_matched_coherence_illustration"
    pair_path = out_dir / "magnitude_matched_coherence_pair.json"
    with open(pair_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **selection,
                "config_version": cfg.CONFIG_VERSION,
                "frozen_sp": str(frozen_path),
                "select_only": bool(args.select_only),
            },
            f,
            indent=2,
        )
    print(f"Wrote {pair_path}", flush=True)

    if args.select_only:
        caption = write_caption(
            selection,
            {"max_abs_sp_diff": float("nan")},
            out_dir / "magnitude_matched_coherence_caption.txt",
        )
        print("\n--select-only: skipping expression load / plot.", flush=True)
        print("Caption draft:\n", caption, flush=True)
        return

    adata, pert_col, ctrl_label, _valid = load_dataset_pca(
        args.dataset, h5ad_path=args.h5ad
    )

    frames_by_role: dict[str, tuple[dict, dict]] = {}
    atol = (
        args.sp_match_atol
        if args.sp_match_atol is not None
        else (SP_MATCH_ATOL if args.strict_sp_match else SP_MATCH_ATOL_SOFT)
    )
    checks = []
    bim_map: dict[str, dict] = {}

    for role in ("extreme", "typical"):
        pair = selection[role]
        hi, lo = pair["high_sp"], pair["low_sp"]
        X_ctrl_hi, X_hi = extract_pca(adata, pert_col, ctrl_label, hi["perturbation"])
        X_ctrl_lo, X_lo = extract_pca(adata, pert_col, ctrl_label, lo["perturbation"])
        frame_hi = displacement_frame(X_ctrl_hi, X_hi)
        frame_lo = displacement_frame(X_ctrl_lo, X_lo)
        frames_by_role[role] = (frame_hi, frame_lo)

        for side, row, frame in (
            ("high_sp", hi, frame_hi),
            ("low_sp", lo, frame_lo),
        ):
            d_sp = abs(frame["sp_calculate_sp"] - row["stability"])
            d_mag = abs(frame["magnitude_calculate_sp"] - row["magnitude"])
            exact = d_sp <= SP_MATCH_ATOL
            ok = d_sp <= atol
            bim = frame.get("bimodality") or {}
            bim_map[f"{role}:{row['perturbation']}"] = bim
            checks.append(
                {
                    "role": role,
                    "panel": side,
                    "perturbation": row["perturbation"],
                    "frozen_sp": row["stability"],
                    "recomputed_sp": frame["sp_calculate_sp"],
                    "abs_sp_diff": d_sp,
                    "abs_mag_diff": d_mag,
                    "match_exact": exact,
                    "match": ok,
                    "bimodality": bim,
                    "median_abs_angle_deg": frame["median_abs_angle_deg"],
                }
            )
            status = (
                "OK" if exact else ("OK (soft — Colab SVD drift)" if ok else "MISMATCH")
            )
            bim_flag = "  [partial-penetrance]" if bim.get("partial_penetrance_flag") else ""
            print(
                f"  verify [{role}] {row['perturbation']}: frozen Sp={row['stability']:.6f}  "
                f"recomputed={frame['sp_calculate_sp']:.6f}  "
                f"|Δ|={d_sp:.2e}  [{status}]{bim_flag}",
                flush=True,
            )
            if not ok:
                raise RuntimeError(
                    f"Sp mismatch for {row['perturbation']}: frozen "
                    f"{row['stability']} vs recomputed {frame['sp_calculate_sp']} "
                    f"(|Δ|={d_sp} > atol={atol}). Refuse to write the figure."
                )

    max_dsp = float(max(c["abs_sp_diff"] for c in checks))
    verify = {
        "atol": atol,
        "atol_exact": SP_MATCH_ATOL,
        "atol_soft": SP_MATCH_ATOL_SOFT,
        "strict": bool(args.strict_sp_match),
        "max_abs_sp_diff": max_dsp,
        "max_abs_mag_diff": float(max(c["abs_mag_diff"] for c in checks)),
        "exact_match": bool(all(c["match_exact"] for c in checks)),
        "panels": checks,
        "bimodality": bim_map,
    }
    if max_dsp > SP_MATCH_ATOL:
        print(
            f"  note: max|ΔSp|={max_dsp:.2e} vs freeze (soft-ok ≤ {atol:g}); "
            "panel labels / caption quote frozen Sp from the CSV.",
            flush=True,
        )

    pdf = plot_extreme_and_typical(
        selection,
        frames_by_role,
        out_stem=stem,
        cmap=args.cmap,
        simple_color=args.simple_color,
        show=args.show,
    )
    caption = write_caption(
        selection, verify, out_dir / "magnitude_matched_coherence_caption.txt"
    )

    with open(pair_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **selection,
                "config_version": cfg.CONFIG_VERSION,
                "frozen_sp": str(frozen_path),
                "verification": verify,
                "cmap": "simple_accent" if args.simple_color else args.cmap,
                "figure_pdf": str(pdf),
            },
            f,
            indent=2,
        )

    print(f"\nWrote {pdf}", flush=True)
    print(f"Wrote {stem.with_suffix('.svg')}", flush=True)
    print(f"Wrote {out_dir / 'magnitude_matched_coherence_caption.txt'}", flush=True)
    print("\nCaption:\n", caption, flush=True)


if __name__ == "__main__":
    main()
