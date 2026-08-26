"""
Shared correlation helpers.

Canonical method: rank-based partial Spearman via pingouin.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import hashlib
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

try:
    import pingouin as pg
except ImportError:  # pragma: no cover
    pg = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    # Colab often breaks statsmodels via packaging/pandas skew
    # (TypeError in deprecate_kwarg). Rank-based path does not need it.
    sm = None

import pipeline_config as cfg

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


def _as_1d(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _as_2d_covars(z: Union[ArrayLike, np.ndarray]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    return z


def bootstrap_spearman_ci(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = cfg.N_BOOTSTRAP,
    ci_level: float = cfg.CI_LEVEL,
    seed: int = cfg.SEED,
) -> dict:
    x, y = _as_1d(x), _as_1d(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rho, p = spearmanr(x, y)
    if np.isnan(rho) or len(x) < 3:
        return {
            "rho": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": int(len(x)),
            "method": "spearman",
            "n_clusters": None,
        }
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), size=len(x), replace=True)
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    alpha = 1 - ci_level
    return {
        "rho": float(rho),
        "p": float(p),
        "ci_low": float(np.percentile(valid, 100 * alpha / 2)),
        "ci_high": float(np.percentile(valid, 100 * (1 - alpha / 2))),
        "n": int(len(x)),
        "method": "spearman",
        "n_clusters": None,
        "n_bootstrap": int(len(valid)),
    }


def bootstrap_spearman_ci_clustered(
    x: ArrayLike,
    y: ArrayLike,
    cluster: ArrayLike,
    n_bootstrap: int = cfg.N_BOOTSTRAP,
    ci_level: float = cfg.CI_LEVEL,
    seed: int = cfg.SEED,
) -> dict:
    """
    Cluster bootstrap CI for Spearman rho.

    Resamples whole clusters with replacement (e.g. genes), keeping all
    nested observations (e.g. guides) from each drawn cluster. Use when
    observations are nested and ordinary row bootstrap understates uncertainty.
    """
    x, y = _as_1d(x), _as_1d(y)
    cluster = np.asarray(cluster).astype(str).ravel()
    if not (len(x) == len(y) == len(cluster)):
        raise ValueError("x, y, cluster must have the same length")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, cluster = x[mask], y[mask], cluster[mask]
    rho, p = spearmanr(x, y)
    units = np.unique(cluster)
    n_clusters = int(len(units))
    if np.isnan(rho) or len(x) < 3 or n_clusters < 3:
        return {
            "rho": float(rho) if rho is not None and np.isfinite(rho) else np.nan,
            "p": float(p) if p is not None and np.isfinite(p) else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": int(len(x)),
            "n_clusters": n_clusters,
            "method": "spearman_cluster_bootstrap",
            "n_bootstrap": 0,
        }

    # Pre-index rows per cluster for fast resampling
    rows_by_unit = {u: np.flatnonzero(cluster == u) for u in units}
    unit_list = list(units)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        drawn = rng.choice(unit_list, size=n_clusters, replace=True)
        idx = np.concatenate([rows_by_unit[u] for u in drawn])
        if len(idx) < 3:
            boot[i] = np.nan
            continue
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    alpha = 1 - ci_level
    if len(valid) < 10:
        ci_low = ci_high = np.nan
    else:
        ci_low = float(np.percentile(valid, 100 * alpha / 2))
        ci_high = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return {
        "rho": float(rho),
        "p": float(p),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(x)),
        "n_clusters": n_clusters,
        "method": "spearman_cluster_bootstrap",
        "n_bootstrap": int(len(valid)),
    }


def _partial_spearman_rank_numpy(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> dict:
    """
    Rank all variables, residualize x and y on Z via OLS, Pearson on residuals.
    Equivalent to pingouin.partial_corr(..., method='spearman') when pingouin
    is unavailable (common on Colab).
    """
    n = len(x)
    if n < 5:
        return {
            "rho_partial": np.nan,
            "p": np.nan,
            "n": int(n),
            "method": "partial_spearman_rank_numpy",
        }
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    RZ = np.column_stack([rankdata(Z[:, j]).astype(float) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(n), RZ])
    bx, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(A, ry, rcond=None)
    ex = rx - A @ bx
    ey = ry - A @ by
    if np.std(ex) < 1e-15 or np.std(ey) < 1e-15:
        return {
            "rho_partial": np.nan,
            "p": np.nan,
            "n": int(n),
            "method": "partial_spearman_rank_numpy",
        }
    rho, p = pearsonr(ex, ey)
    return {
        "rho_partial": float(rho),
        "p": float(p),
        "n": int(n),
        "method": "partial_spearman_rank_numpy",
        "n_covar": int(Z.shape[1]),
    }


def partial_spearman_rank(
    x: ArrayLike,
    y: ArrayLike,
    z: Union[ArrayLike, np.ndarray],
) -> dict:
    """
    Rank-based partial Spearman (manuscript default).

    Prefers pingouin when installed; falls back to an equivalent numpy
    rank→residualize→Pearson path so Colab runs without pingouin.
    """
    x, y = _as_1d(x), _as_1d(y)
    Z = _as_2d_covars(z)
    n = len(x)
    if Z.shape[0] != n:
        raise ValueError(f"z length {Z.shape[0]} != x length {n}")

    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[mask], y[mask], Z[mask]
    if len(x) < 5:
        return {
            "rho_partial": np.nan,
            "p": np.nan,
            "n": int(len(x)),
            "method": "partial_spearman_rank",
        }

    if pg is not None:
        data = {"x": x, "y": y}
        covar_cols = []
        for j in range(Z.shape[1]):
            col = f"z{j}"
            data[col] = Z[:, j]
            covar_cols.append(col)
        df = pd.DataFrame(data)
        res = pg.partial_corr(
            data=df,
            x="x",
            y="y",
            covar=covar_cols if len(covar_cols) > 1 else covar_cols[0],
            method="spearman",
        )
        p_col = next(c for c in res.columns if c.startswith("p"))
        return {
            "rho_partial": float(res["r"].values[0]),
            "p": float(res[p_col].values[0]),
            "n": int(len(df)),
            "method": "partial_spearman_rank_pingouin",
        }

    return _partial_spearman_rank_numpy(x, y, Z)


def icc_oneway_unbalanced(y: ArrayLike, groups: ArrayLike) -> dict:
    """
    ICC(1) one-way random effects for unbalanced group sizes
    (Shrout & Fleiss; n0 correction for unequal n_i).
    """
    y = _as_1d(y)
    groups = np.asarray(groups).astype(str)
    mask = np.isfinite(y)
    y, groups = y[mask], groups[mask]
    units, inv = np.unique(groups, return_inverse=True)
    k = len(units)
    N = len(y)
    if k < 2 or N <= k:
        return {"icc": np.nan, "n_groups": k, "n_obs": N, "n0": np.nan}
    n_i = np.bincount(inv)
    grand = float(y.mean())
    means = np.array([y[inv == i].mean() for i in range(k)])
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(sum(np.sum((y[inv == i] - means[i]) ** 2) for i in range(k)))
    df_b, df_w = k - 1, N - k
    msb, msw = ssb / df_b, ssw / df_w
    n0 = (N - float(np.sum(n_i ** 2)) / N) / df_b
    denom = msb + (n0 - 1) * msw
    icc = (msb - msw) / denom if denom != 0 else np.nan
    return {
        "icc": float(icc),
        "msb": float(msb),
        "msw": float(msw),
        "n0": float(n0),
        "n_groups": int(k),
        "n_obs": int(N),
        "method": "icc1_oneway_unbalanced",
    }


def icc_gene_clustered_bootstrap(
    y: ArrayLike,
    groups: ArrayLike,
    *,
    n_bootstrap: int = 2000,
    ci_level: float = cfg.CI_LEVEL,
    seed: int = cfg.SEED,
) -> dict:
    """ICC(1) point estimate + gene-clustered bootstrap CI."""
    y = _as_1d(y)
    groups = np.asarray(groups).astype(str)
    mask = np.isfinite(y)
    y, groups = y[mask], groups[mask]
    point = icc_oneway_unbalanced(y, groups)
    units = np.unique(groups)
    by_u = {u: y[groups == u] for u in units}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        drawn = rng.choice(units, size=len(units), replace=True)
        yb = np.concatenate([by_u[u] for u in drawn])
        gb = np.concatenate(
            [[f"{j}:{drawn[j]}"] * len(by_u[drawn[j]]) for j in range(len(drawn))]
        )
        boot[i] = icc_oneway_unbalanced(yb, gb)["icc"]
    valid = boot[np.isfinite(boot)]
    alpha = 1 - ci_level
    out = {
        "icc": point["icc"],
        "icc_details": point,
        "n_bootstrap": int(len(valid)),
        "n_groups": point.get("n_groups"),
        "n_obs": point.get("n_obs"),
    }
    if len(valid) >= 10:
        out["icc_ci_low"] = float(np.percentile(valid, 100 * alpha / 2))
        out["icc_ci_high"] = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    else:
        out["icc_ci_low"] = out["icc_ci_high"] = np.nan
    return out


def partial_spearman_raw_residuals(
    x: ArrayLike,
    y: ArrayLike,
    z: Union[ArrayLike, np.ndarray],
) -> dict:
    """
    Legacy: OLS residuals on raw scale, then Spearman.
    Not equivalent to partial Spearman — keep only for audit comparisons.
    """
    if sm is None:
        raise ImportError("statsmodels required for residual-Spearman fallback")

    x, y = _as_1d(x), _as_1d(y)
    Z = _as_2d_covars(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[mask], y[mask], Z[mask]
    if len(x) < 5:
        return {
            "rho_partial": np.nan,
            "p": np.nan,
            "n": int(len(x)),
            "method": "spearman_on_raw_residuals",
        }
    Z_aug = sm.add_constant(Z)
    x_resid = sm.OLS(x, Z_aug).fit().resid
    y_resid = sm.OLS(y, Z_aug).fit().resid
    rho, p = spearmanr(x_resid, y_resid)
    return {
        "rho_partial": float(rho),
        "p": float(p),
        "n": int(len(x)),
        "method": "spearman_on_raw_residuals",
    }


def bootstrap_partial_spearman_ci(
    x: ArrayLike,
    y: ArrayLike,
    z: Union[ArrayLike, np.ndarray],
    n_bootstrap: int = cfg.N_BOOTSTRAP,
    ci_level: float = cfg.CI_LEVEL,
    seed: int = cfg.SEED,
    method: str = "rank",
) -> dict:
    """
    Bootstrap CI for partial Spearman.

    method:
      - "rank" (default): pingouin rank-based partial Spearman
      - "raw_residuals": legacy OLS-residual Spearman
    """
    x, y = _as_1d(x), _as_1d(y)
    Z = _as_2d_covars(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[mask], y[mask], Z[mask]
    n = len(x)

    point_fn = partial_spearman_rank if method == "rank" else partial_spearman_raw_residuals
    point = point_fn(x, y, Z)
    rho0 = point["rho_partial"]
    p0 = point["p"]
    method_name = point["method"]

    if n < 5 or np.isnan(rho0):
        return {
            "rho_partial": rho0,
            "p": p0,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": n,
            "method": method_name,
            "n_bootstrap": 0,
        }

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = point_fn(x[idx], y[idx], Z[idx])["rho_partial"]
    valid = boot[~np.isnan(boot)]
    alpha = 1 - ci_level
    frac_valid = len(valid) / max(n_bootstrap, 1)
    # Degenerate resamples (small-n + collinear Z): do not invent a CI
    if len(valid) < 10 or frac_valid < 0.80:
        ci_low = ci_high = np.nan
    else:
        ci_low = float(np.percentile(valid, 100 * alpha / 2))
        ci_high = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    return {
        "rho_partial": float(rho0),
        "p": float(p0) if p0 is not None and np.isfinite(p0) else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(n),
        "method": method_name,
        "n_bootstrap": int(len(valid)),
        "bootstrap_frac_valid": float(frac_valid),
        "bootstrap_seed": int(seed),
    }


# Back-compat alias used by older scripts after they switch imports
def bootstrap_partial_correlation_ci(
    x, y, z, n_bootstrap=cfg.N_BOOTSTRAP, ci_level=cfg.CI_LEVEL, seed=cfg.SEED
):
    """Default = rank-based partial Spearman (revision canonical)."""
    return bootstrap_partial_spearman_ci(
        x, y, z, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed, method="rank"
    )


def pathway_bootstrap_seed(
    dataset_name: str,
    feature: str,
    stage: str = "partial_mag",
    n_bootstrap: int | None = None,
) -> int:
    """
    Deterministic bootstrap seed shared by pathway_analysis Approach A and
    cell_quality_partial |mag. Includes n_bootstrap so 2k vs 10k draws cannot
    silently disagree on knife-edge CIs.
    """
    nb = int(n_bootstrap if n_bootstrap is not None else cfg.N_BOOTSTRAP)
    key = f"{cfg.SEED}|{dataset_name}|{feature}|{stage}|nb{nb}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def survival_status(
    rho: float,
    ci_low: float,
    ci_high: float,
    fdr: float | None = None,
    abs_min: float | None = None,
    fdr_max: float | None = None,
    knife_edge_abs: float | None = None,
) -> dict:
    """
    Manuscript criterion (pipeline_config.SURVIVAL_CRITERION_ID):
      survives      = |rho| > abs_min AND CI excludes 0 AND FDR < fdr_max (if FDR given)
      indeterminate = knife-edge demotion of an otherwise-surviving row, OR CI↔FDR disagree

    Knife-edge uses strict < ε (margin == ε does not trigger). It only fires when
    the row would otherwise survive — null / near-null rows stay does_not_survive
    even if a CI bound lands near zero (e.g. Norman HSPA5 |QC bound = 0.020).
    """
    if abs_min is None:
        abs_min = float(getattr(cfg, "SURVIVAL_ABS_RHO_MIN", 0.1))
    if fdr_max is None:
        fdr_max = float(getattr(cfg, "SURVIVAL_FDR_MAX", 0.05))
    if knife_edge_abs is None:
        knife_edge_abs = float(getattr(cfg, "SURVIVAL_KNIFE_EDGE_ABS", 0.02))

    rho = float(rho) if rho is not None and np.isfinite(rho) else np.nan
    lo = float(ci_low) if ci_low is not None and np.isfinite(ci_low) else np.nan
    hi = float(ci_high) if ci_high is not None and np.isfinite(ci_high) else np.nan
    ci_ok = np.isfinite(lo) and np.isfinite(hi) and np.sign(lo) == np.sign(hi)
    mag_ok = np.isfinite(rho) and abs(rho) > abs_min
    # Distance of nearer CI bound to zero (works whether CI excludes or straddles)
    if np.isfinite(lo) and np.isfinite(hi):
        margin = float(min(abs(lo), abs(hi)))
    else:
        margin = np.nan
    fdr_ok = True if fdr is None or not np.isfinite(fdr) else bool(fdr < fdr_max)
    ci_survive = bool(mag_ok and ci_ok)
    otherwise_survives = bool(ci_survive and fdr_ok)
    # Strict < ε; only demote rows that clear every other bar
    near_zero = bool(np.isfinite(margin) and margin < knife_edge_abs)
    knife = bool(near_zero and otherwise_survives)
    disagree = (
        bool(ci_survive != (mag_ok and fdr_ok))
        if fdr is not None and np.isfinite(fdr)
        else False
    )
    if knife or disagree:
        label = "indeterminate"
        survives = False
    elif otherwise_survives:
        label = "survives"
        survives = True
    else:
        label = "does_not_survive"
        survives = False
    return {
        "survives": survives,
        "ci_excludes_zero": ci_ok,
        "fdr_ok": fdr_ok,
        "knife_edge": knife,
        "ci_fdr_disagree": disagree,
        "ci_margin": float(margin) if np.isfinite(margin) else np.nan,
        "status": label,
        "criterion_id": getattr(cfg, "SURVIVAL_CRITERION_ID", "ci_and_fdr.v1"),
    }
