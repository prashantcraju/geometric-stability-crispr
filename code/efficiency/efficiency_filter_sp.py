#!/usr/bin/env python3
"""
Perturbation-efficiency confound: recompute Sp on responding cells only.

A perturbation can appear incoherent because of a mix of perturbed,
unperturbed, and partially perturbed cells. Mixscape (Papalexi et al. 2021)
and PS (Song et al. 2025) address this by restricting to responding cells.

Responder calls come from one of:
  mixscape  pertpy Mixscape (Papalexi 2021). Binary KO / NP per cell.
  ps        Song et al. 2025 perturbation score, via the scMAGeCK port in
            song_ps_replication.compute_ps_python. Continuous, thresholded.
  obs       any per-cell score already in adata.obs (hook for external calls
            and for --self-test).

SENSITIVITY ANALYSIS, NOT A NEW FREEZE. This script:
  * reads frozen_sp_scores.csv, never writes it;
  * reproduces the frozen Sp for the target dataset through the manuscript
    path (pipeline_core.load_raw → materialize_min_cells → preprocess →
    calculate_sp) and ABORTS if that reproduction is not bit-identical, so a
    responder-filtered number can never be compared against a different
    preprocessing;
  * writes only efficiency_filter_* outputs.

Responder-filter knobs are versioned by EFFICIENCY_FILTER_VERSION and must
never bump CONFIG_VERSION: no frozen Sp, pathway, or QC number depends on them.

PRIMARY (cite under Song PS / Mixscape):
  1. Sp~magnitude ρ before vs after responder filtering — is the magnitude
     redundancy an efficiency artifact?
  2. ρ(Sp_all, Sp_responders) — does filtering rearrange rankings?
     (preserved ≥0.90 / partial 0.70–0.90 / change <0.70)

Song PS is direction-aligned (Y ≈ PS · β on signature DEGs), so ΔSp,
z_matched, beyond-mag, and ρ(Sp, responder fraction) are mechanical under
PS and are logged but not cited. The cell-count-matched null controls n,
not direction; a direction-independent self-test (method=obs) is the
positive control that beyond-mag stays null when selection is orthogonal.

Usage:
  python efficiency_filter_sp.py --self-test
  python efficiency_filter_sp.py --datasets "Norman 2019 (CRISPRa)" \
      --method mixscape --frozen-sp shesha-crispr/frozen_sp_scores.csv
  python efficiency_filter_sp.py --datasets norman --method ps --write-percell
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    _stable_sample_indices,
    assert_frozen_sp_compatible,
    calculate_sp,
    load_raw,
    materialize_min_cells,
    preprocess,
    score_perturbations,
    setup_cache,
)
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import bootstrap_spearman_ci, partial_spearman_rank

# ---------------------------------------------------------------------------
# Version + pre-specified constants (fixed before looking at any result)
# ---------------------------------------------------------------------------

# Separate from CONFIG_VERSION on purpose. Bump this for responder-filter
# changes; regenerate only efficiency_filter_* outputs.
#   2026-08-09.1 — first cut: pertpy Mixscape + Song PS arms, cell-count-matched
#                  null, LOESS magnitude baseline, responder-fraction partial.
#   2026-08-09.2 — monkeypatch MixscapeGaussianMixture._m_step for sklearn/pertpy
#                  xp-kwarg skew (Colab TypeError); science unchanged.
#   2026-08-09.3 — report Song PS as direction-aligned selection; primary
#                  deliverables = Sp~mag Δ + rank preservation; mark ΔSp /
#                  z_matched / beyond-mag / ρ(Sp, frac) as mechanical under PS.
#   2026-08-09.4 — --ram-lite for Replogle-scale Colab: smaller cell caps +
#                  Sp_all/mag_all from frozen table (cannot bit-reproduce the
#                  freeze at reduced n). Primary endpoints remain interpretable.
#   2026-08-09.5 — --compare-percell PS↔Mixscape agreement; Sp~mag reports
#                  frozen-full vs rescored-subset denominators (no silent 0.853).
EFFICIENCY_FILTER_VERSION = "2026-08-09.5"

# Song PS (scMAGeCK-PS) fits Y ≈ PS · β on signature DEGs: the per-cell score
# is a non-negative scalar dosage along the average perturbation-effect vector
# in gene space. Thresholding PS therefore selects cells by alignment with that
# direction — the same geometric quantity Sp summarises in PCA. Under PS,
# ΔSp / z_matched / beyond-mag / ρ(Sp, responder fraction) are expected by
# construction and are NOT manuscript endpoints. Mixscape's binary KO/NP call
# is less directly cosine-circular but still signature-driven; treat the same
# way until an orthogonal caller exists.
EFF_DIRECTION_ALIGNED_METHODS = frozenset({"ps", "mixscape"})

# Colab free OOMs Replogle at freeze caps (≤100/pert + ≤5000 controls ≈ 175k
# cells at to_memory). RAM-lite keeps the ≥min_cells membership filter but
# downsamples harder for PS + responder Sp only.
EFF_RAM_LITE_MAX_CELLS_PER_PERT = 40
EFF_RAM_LITE_MAX_CONTROL_CELLS = 500

# Minimum responder cells before a perturbation is rescored. Item #15 measured
# Sp ranking reliability down to 25 cells (ρ=0.946) and 10 cells (ρ=0.923);
# 25 is the reliable end of that curve. Perturbations below it are reported
# with a status flag instead of a number — dropping them silently would
# select against exactly the low-efficiency perturbations of interest.
EFF_MIN_RESPONDER_CELLS = 25

# Draws for the cell-count-matched random-subset null, per perturbation.
EFF_MATCHED_DRAWS = 200

# Song PS is rescaled to [0, 1] within each perturbation (divided by the
# maximum cell), so this is "at least half the inferred effect of the most
# strongly perturbed cell in the same perturbation", not an absolute efficiency.
EFF_PS_THRESHOLD = 0.5

# Mixscape global classes counted as responders. pertpy writes
# mixscape_class_global ∈ {<perturbation_type>, NP, NT} with default type "KO".
# Cells carrying neither a responder token nor NP are left unclassified rather
# than folded into either arm — mixscape declines to split target classes with
# too few DE genes, and counting those cells as responders would quietly
# inflate the efficiency of exactly the weak perturbations under test.
EFF_MIXSCAPE_RESPONDER_CLASSES = ("KO", "KD", "PERTURBED")
EFF_MIXSCAPE_NONRESPONDER_CLASSES = ("NP",)

# Span of the LOESS Sp~magnitude baseline. Matches DEFAULT_LOESS_FRAC in
# adamson_upr_magnitude_partial so the two magnitude-baseline analyses use the
# same smoothing.
EFF_LOESS_FRAC = 0.3

# Verdict bands on ρ(Sp_all, Sp_responders). Declared here, before any run.
EFF_RANK_PRESERVED_RHO = 0.90
EFF_RANK_PARTIAL_RHO = 0.70

OUT_PREFIX = "efficiency_filter"
FROZEN_TABLE_NAME = "frozen_sp_scores.csv"
# Reproduction of the frozen Sp must be exact, not close.
EFF_FROZEN_TOL = 1e-9


def _slug(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )


def _write(path: Path, writer) -> Path:
    """
    Write an output file, refusing anything that is not an efficiency_filter_*
    artifact. The frozen Sp table and the manuscript tables are inputs here.
    """
    path = Path(path)
    if not path.name.startswith(OUT_PREFIX):
        raise SystemExit(
            f"refusing to write {path.name}: this script only writes "
            f"{OUT_PREFIX}_* files (frozen and manuscript tables are read-only)"
        )
    writer(path)
    print(f"  wrote {path}")
    return path


def _eff_seed(*parts) -> int:
    """Deterministic seed for the matched-subset null (independent of run order)."""
    key = "|".join([str(cfg.SEED), EFFICIENCY_FILTER_VERSION] + [str(p) for p in parts])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def _f(x) -> Optional[float]:
    """JSON-safe float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# LOESS Sp~magnitude baseline (fit on all-cell Sp, evaluated at responder mag)
# ---------------------------------------------------------------------------


def _tricube_local_linear(
    y: np.ndarray, x: np.ndarray, x_new: np.ndarray, frac: float
) -> np.ndarray:
    """Local linear fit with tricube weights, evaluated at arbitrary x_new."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    n = len(x)
    if n == 0:
        return np.full(len(x_new), np.nan)
    k = int(np.clip(np.ceil(frac * n), 2, n))
    out = np.empty(len(x_new), dtype=float)
    for i, xt in enumerate(x_new):
        dist = np.abs(x - xt)
        nn = np.argpartition(dist, k - 1)[:k]
        dmax = float(dist[nn].max())
        if dmax < 1e-15:
            out[i] = float(y[nn].mean())
            continue
        u = dist[nn] / dmax
        w = np.clip((1 - u**3) ** 3, 0.0, None)
        sw = float(w.sum())
        if sw < 1e-15:
            out[i] = float(y[nn].mean())
            continue
        xbar = float(np.sum(w * x[nn]) / sw)
        ybar = float(np.sum(w * y[nn]) / sw)
        varx = float(np.sum(w * (x[nn] - xbar) ** 2))
        out[i] = ybar if varx < 1e-15 else ybar + (
            float(np.sum(w * (x[nn] - xbar) * (y[nn] - ybar))) / varx
        ) * (xt - xbar)
    return out


def lowess_predict(
    y: np.ndarray, x: np.ndarray, x_new: np.ndarray, frac: float = EFF_LOESS_FRAC
) -> np.ndarray:
    """
    Sp predicted at new magnitudes from the dataset's own Sp~magnitude curve.

    x_new is clipped into the observed magnitude range. Responder filtering
    pushes magnitude up, so clipping evaluates against the highest part of the
    fitted curve — the conservative direction for a "beyond magnitude" claim.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.full(len(np.atleast_1d(x_new)), np.nan)
    x_new = np.clip(np.asarray(x_new, dtype=float), float(x.min()), float(x.max()))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        pred = np.asarray(lowess(y, x, frac=frac, xvals=x_new), dtype=float)
        if np.all(np.isfinite(pred)):
            return pred
    except Exception:
        pass
    return _tricube_local_linear(y, x, x_new, frac)


# ---------------------------------------------------------------------------
# Responder callers
# ---------------------------------------------------------------------------


def _import_mixscape_class():
    """
    Import pertpy's Mixscape without executing pertpy/__init__ when avoidable.

    Same trick as pipeline_core.import_pertpy_datasets: the full pertpy import
    pulls JAX and routinely hangs on Colab. Falls back to the plain import.
    """
    spec = importlib.util.find_spec("pertpy")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "pertpy is not installed — the mixscape arm needs it. "
            "pip install pertpy==1.0.6"
        )
    pkg_path = spec.submodule_search_locations[0]
    try:
        for mod in list(sys.modules):
            if mod == "pertpy" or mod.startswith("pertpy."):
                del sys.modules[mod]
        pkg = types.ModuleType("pertpy")
        pkg.__path__ = [pkg_path]
        pkg.__spec__ = spec
        sys.modules["pertpy"] = pkg
        tools = types.ModuleType("pertpy.tools")
        tools.__path__ = [str(Path(pkg_path) / "tools")]
        sys.modules["pertpy.tools"] = tools
        mod = importlib.import_module("pertpy.tools._mixscape")
        return getattr(mod, "Mixscape")
    except Exception as e:
        print(f"    light pertpy import failed ({e}); importing full pertpy…", flush=True)
        for mod in list(sys.modules):
            if mod == "pertpy" or mod.startswith("pertpy."):
                del sys.modules[mod]
        import pertpy as pt

        return pt.tools.Mixscape


def _call_with_supported(fn, adata, pert_col, ctrl_label, **kwargs):
    """
    Call a pertpy Mixscape method, adapting to the argument names it exposes.

    pertpy renamed the label argument (labels → pert_key) between versions, and
    synthetic_benchmark.method_mixscape was left stubbed precisely because
    guessing that wrong is worse than an explicit gap. Resolve it by
    introspection and print what was resolved.
    """
    import inspect

    sig = inspect.signature(fn)
    params = sig.parameters
    label_arg = next(
        (a for a in ("pert_key", "labels", "pert_key_col") if a in params), None
    )
    ctrl_arg = next((a for a in ("control", "control_label") if a in params), None)
    if label_arg is None or ctrl_arg is None:
        raise RuntimeError(
            f"Unrecognised pertpy Mixscape API for {fn.__name__}: "
            f"parameters={list(params)}. Pin the pertpy version, confirm the "
            "signature, then re-run — do not guess."
        )
    passed = {k: v for k, v in kwargs.items() if k in params and v is not None}
    dropped = sorted(set(kwargs) - set(passed))
    print(
        f"    Mixscape.{fn.__name__}: {label_arg}={pert_col!r} {ctrl_arg}={ctrl_label!r} "
        f"kwargs={passed}" + (f" (unsupported here: {dropped})" if dropped else ""),
        flush=True,
    )
    return fn(adata, **{label_arg: pert_col, ctrl_arg: ctrl_label}, **passed)


def _patch_mixscape_sklearn_gmm() -> None:
    """
    Bridge pertpy MixscapeGaussianMixture ↔ sklearn version skew on `_m_step`.

    Colab currently ships a sklearn where GaussianMixture._m_step does not
    accept `xp=`, while pertpy's Mixscape subclass always forwards `xp=xp`
    (or the reverse under sklearn ≥1.8). Either way Mixscape dies mid-GMM with
    TypeError before writing any classification. Patch once per process; no
    effect if signatures already agree.
    """
    import inspect

    from sklearn.mixture import GaussianMixture

    try:
        from pertpy.tools._mixscape import MixscapeGaussianMixture
    except Exception:
        try:
            mod = importlib.import_module("pertpy.tools._mixscape")
            MixscapeGaussianMixture = getattr(mod, "MixscapeGaussianMixture", None)
        except Exception:
            MixscapeGaussianMixture = None
    if MixscapeGaussianMixture is None:
        return
    if getattr(MixscapeGaussianMixture, "_shesha_xp_patched", False):
        return

    parent_accepts_xp = "xp" in inspect.signature(GaussianMixture._m_step).parameters
    child = MixscapeGaussianMixture._m_step
    try:
        child_accepts_xp = "xp" in inspect.signature(child).parameters
    except (TypeError, ValueError):
        child_accepts_xp = True  # bound / C method — assume it may receive xp

    if parent_accepts_xp == child_accepts_xp:
        MixscapeGaussianMixture._shesha_xp_patched = True
        return

    def _m_step(self, X, log_resp, xp=None):
        # Always accept xp so sklearn 1.8+ fit_predict can call us; only
        # forward it when the parent signature allows.
        if parent_accepts_xp:
            GaussianMixture._m_step(self, X, log_resp, xp=xp)
        else:
            GaussianMixture._m_step(self, X, log_resp)
        if getattr(self, "fixed_mean_indices", None):
            self.means_[self.fixed_mean_indices] = self.fixed_mean_values
        if getattr(self, "fixed_cov_indices", None):
            self.covariances_[self.fixed_cov_indices] = self.fixed_cov_values
        return self

    MixscapeGaussianMixture._m_step = _m_step
    MixscapeGaussianMixture._shesha_xp_patched = True
    print(
        f"    patched MixscapeGaussianMixture._m_step "
        f"(sklearn accepts xp={parent_accepts_xp}; "
        f"pertpy child accepted xp={child_accepts_xp})",
        flush=True,
    )


def responder_calls_mixscape(
    adata,
    pert_col: str,
    ctrl_label: str,
    *,
    n_neighbors: int = 20,
    n_dims: int = 15,
    min_de_genes: int = 5,
) -> pd.DataFrame:
    """
    pertpy Mixscape → per-cell responder calls for perturbed cells.

    Runs on the same object Sp is scored on: log-normalised HVG .X and the
    frozen 50-PC X_pca, so the responder call and the geometry share one
    representation.
    """
    Mixscape = _import_mixscape_class()
    _patch_mixscape_sklearn_gmm()
    ms = Mixscape()

    print("    Mixscape perturbation signature (local NT neighbours)…", flush=True)
    _call_with_supported(
        ms.perturbation_signature,
        adata,
        pert_col,
        ctrl_label,
        use_rep="X_pca",
        n_dims=n_dims,
        n_neighbors=n_neighbors,
    )
    if "X_pert" not in getattr(adata, "layers", {}):
        raise RuntimeError(
            "Mixscape.perturbation_signature did not write layers['X_pert']; "
            f"layers={list(getattr(adata, 'layers', {}))}. Confirm the pertpy API."
        )

    print("    Mixscape GMM classification…", flush=True)
    _call_with_supported(
        ms.mixscape,
        adata,
        pert_col,
        ctrl_label,
        layer="X_pert",
        min_de_genes=min_de_genes,
        random_state=cfg.SEED,
    )

    obs = adata.obs
    class_col = next(
        (c for c in ("mixscape_class_global", "mixscape_class") if c in obs.columns),
        None,
    )
    if class_col is None:
        raise RuntimeError(
            "Mixscape wrote no classification column "
            f"(obs={list(obs.columns)}). Confirm the pertpy API."
        )
    prob_col = next(
        (c for c in obs.columns if c.startswith("mixscape_class_p_")), None
    )
    print(f"    resolved mixscape columns: class={class_col!r} prob={prob_col!r}", flush=True)

    labels = obs[pert_col].astype(str).to_numpy()
    is_pert = labels != str(ctrl_label)
    klass = obs[class_col].astype(str).to_numpy()
    token = np.array([k.split()[-1].upper() for k in klass])
    responder = np.isin(token, EFF_MIXSCAPE_RESPONDER_CLASSES)
    nonresponder = np.isin(token, EFF_MIXSCAPE_NONRESPONDER_CLASSES)
    definite = (responder | nonresponder) & is_pert
    n_unclassified = int((is_pert & ~definite).sum())
    if n_unclassified:
        print(
            f"    {n_unclassified} perturbed cells left unclassified by mixscape "
            f"(classes {sorted(set(token[is_pert & ~definite]))[:5]}) — excluded "
            "from both arms",
            flush=True,
        )
    score = (
        pd.to_numeric(obs[prob_col], errors="coerce").to_numpy()
        if prob_col is not None
        else responder.astype(float)
    )
    return pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str).to_numpy()[definite],
            "perturbation": labels[definite],
            "score": score[definite],
            "is_responder": responder[definite],
            "raw_class": klass[definite],
        }
    )


def responder_calls_ps(
    adata,
    pert_col: str,
    ctrl_label: str,
    valid_perts: list,
    *,
    threshold: float = EFF_PS_THRESHOLD,
    max_ctrl_cells: int = 1000,
) -> pd.DataFrame:
    """
    Song et al. 2025 PS per cell, via song_ps_replication.compute_ps_python.

    Controls are subsampled with the same order-invariant hash sampler used by
    the freeze. This is a PS-arm cost knob only: the Sp control centroid is
    always the full frozen control set.
    """
    try:
        from song_ps_replication import compute_ps_python
    except Exception as e:  # pertpy/scanpy import guard at that module's top
        raise ImportError(
            f"Could not import song_ps_replication.compute_ps_python ({e}). "
            "The PS arm needs that module importable (it requires pertpy)."
        ) from e

    labels = adata.obs[pert_col].astype(str).to_numpy()
    obs_names = adata.obs_names.astype(str).to_numpy()
    ctrl_idx = np.flatnonzero(labels == str(ctrl_label))
    if len(ctrl_idx) > max_ctrl_cells:
        ctrl_idx = _stable_sample_indices(ctrl_idx, obs_names, max_ctrl_cells, cfg.SEED)
        print(
            f"    PS arm: control cells subsampled to {len(ctrl_idx)} "
            "(hash-stable; Sp control set unchanged)",
            flush=True,
        )

    rows = []
    t0 = time.time()
    for i, pert in enumerate(valid_perts, start=1):
        pidx = np.flatnonzero(labels == str(pert))
        sub = adata[np.sort(np.concatenate([pidx, ctrl_idx]))].copy()
        try:
            ps = compute_ps_python(sub, pert_col, str(ctrl_label), str(pert))
        except Exception as e:
            print(f"      PS failed for {pert}: {e}", flush=True)
            ps = {}
        for cell, value in ps.items():
            rows.append(
                {
                    "cell": str(cell),
                    "perturbation": str(pert),
                    "score": float(value),
                    "is_responder": bool(float(value) >= threshold),
                    "raw_class": "",
                }
            )
        if i % 25 == 0 or i == len(valid_perts):
            print(
                f"      PS {i}/{len(valid_perts)} perturbations "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


def responder_calls_obs(
    adata, pert_col: str, ctrl_label: str, column: str, threshold: float
) -> pd.DataFrame:
    """Responder calls from a per-cell score already present in adata.obs."""
    if column not in adata.obs.columns:
        raise KeyError(f"--obs-score-column {column!r} not in adata.obs")
    labels = adata.obs[pert_col].astype(str).to_numpy()
    is_pert = labels != str(ctrl_label)
    score = pd.to_numeric(adata.obs[column], errors="coerce").to_numpy()
    return pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str).to_numpy()[is_pert],
            "perturbation": labels[is_pert],
            "score": score[is_pert],
            "is_responder": score[is_pert] >= threshold,
            "raw_class": "",
        }
    )


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------


def matched_random_null(
    X_ctrl: np.ndarray,
    X_pert: np.ndarray,
    k: int,
    *,
    n_draws: int,
    seed: int,
    min_k: int = EFF_MIN_RESPONDER_CELLS,
) -> dict:
    """
    Sp on random k-cell subsets of the same perturbation.

    Isolates responder selection from the cell-count change: E[Sp] over uniform
    subsets equals the all-cell Sp, so this null carries the sampling spread
    that the responder subset must beat.
    """
    n = X_pert.shape[0]
    if k >= n or k < min_k:
        return {"mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan, "n_draws": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        idx = rng.choice(n, size=k, replace=False)
        vals[i] = calculate_sp(X_ctrl, X_pert[idx])["stability"]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return {"mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan, "n_draws": len(vals)}
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "n_draws": int(len(vals)),
    }


def rescore_on_responders(
    adata,
    pert_col: str,
    ctrl_label: str,
    valid_perts: list,
    calls: pd.DataFrame,
    dataset_name: str,
    method: str,
    *,
    n_matched_draws: int = EFF_MATCHED_DRAWS,
    min_responder_cells: int = EFF_MIN_RESPONDER_CELLS,
) -> pd.DataFrame:
    """Per-perturbation Sp on all cells, responders, non-responders, and the null."""
    labels = adata.obs[pert_col].astype(str).to_numpy()
    obs_names = adata.obs_names.astype(str).to_numpy()
    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    X_ctrl = X[labels == str(ctrl_label)]
    if X_ctrl.shape[0] < cfg.MIN_CONTROL_CELLS:
        raise ValueError(f"Insufficient control cells: {X_ctrl.shape[0]}")

    resp_by_cell = dict(zip(calls["cell"].astype(str), calls["is_responder"].astype(bool)))
    score_by_cell = dict(zip(calls["cell"].astype(str), calls["score"].astype(float)))

    rows = []
    for pert in valid_perts:
        pmask = labels == str(pert)
        cells = obs_names[pmask]
        X_pert = X[pmask]
        n_cells = int(X_pert.shape[0])
        called = np.array([c in resp_by_cell for c in cells])
        is_resp = np.array([bool(resp_by_cell.get(c, False)) for c in cells])
        is_nonresp = called & ~is_resp
        scores = np.array([score_by_cell.get(c, np.nan) for c in cells], dtype=float)
        n_resp = int(is_resp.sum())
        n_called = int(called.sum())

        all_m = calculate_sp(X_ctrl, X_pert)
        row = {
            "dataset": dataset_name,
            "method": method,
            "perturbation": str(pert),
            "n_cells": n_cells,
            "n_called": n_called,
            "n_unclassified": int(n_cells - n_called),
            "n_responders": n_resp,
            "n_nonresponders": int(is_nonresp.sum()),
            # Denominator is the classified cells: mixscape declines to split
            # some target classes, and those cells belong in neither arm.
            "frac_responders": float(n_resp / n_called) if n_called else np.nan,
            "frac_responders_of_all": float(n_resp / n_cells) if n_cells else np.nan,
            "score_mean": float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else np.nan,
            "score_median": float(np.nanmedian(scores)) if np.any(np.isfinite(scores)) else np.nan,
            "sp_all": all_m["stability"],
            "mag_all": all_m["magnitude"],
            "spread_all": all_m["spread"],
            "snr_all": all_m["snr"],
        }

        if n_called == 0:
            row["status"] = "no_calls"
        elif n_resp == 0:
            row["status"] = "no_responders"
        elif n_resp < min_responder_cells:
            row["status"] = "too_few_responders"
        elif n_resp == n_cells:
            row["status"] = "all_responders"
        else:
            row["status"] = "ok"

        if n_resp >= min_responder_cells:
            m = calculate_sp(X_ctrl, X_pert[is_resp])
            row.update(
                sp_resp=m["stability"],
                mag_resp=m["magnitude"],
                spread_resp=m["spread"],
                snr_resp=m["snr"],
                delta_sp=m["stability"] - all_m["stability"],
            )
            null = matched_random_null(
                X_ctrl,
                X_pert,
                n_resp,
                n_draws=n_matched_draws,
                seed=_eff_seed(dataset_name, method, pert, "matched"),
                min_k=min_responder_cells,
            )
            row.update(
                sp_matched_mean=null["mean"],
                sp_matched_sd=null["sd"],
                sp_matched_lo=null["lo"],
                sp_matched_hi=null["hi"],
                n_matched_draws=null["n_draws"],
            )
            if np.isfinite(null["sd"]) and null["sd"] > 1e-12:
                row["z_matched"] = (m["stability"] - null["mean"]) / null["sd"]
            else:
                row["z_matched"] = np.nan
        else:
            for c in (
                "sp_resp",
                "mag_resp",
                "spread_resp",
                "snr_resp",
                "delta_sp",
                "sp_matched_mean",
                "sp_matched_sd",
                "sp_matched_lo",
                "sp_matched_hi",
                "z_matched",
            ):
                row[c] = np.nan
            row["n_matched_draws"] = 0

        if int(is_nonresp.sum()) >= min_responder_cells:
            mn = calculate_sp(X_ctrl, X_pert[is_nonresp])
            row.update(sp_nonresp=mn["stability"], mag_nonresp=mn["magnitude"])
        else:
            row.update(sp_nonresp=np.nan, mag_nonresp=np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    ok = df["sp_resp"].notna() & df["mag_resp"].notna()
    df["sp_pred_from_mag"] = np.nan
    if ok.sum() >= 5:
        df.loc[ok, "sp_pred_from_mag"] = lowess_predict(
            df["sp_all"].to_numpy(),
            df["mag_all"].to_numpy(),
            df.loc[ok, "mag_resp"].to_numpy(),
        )
    df["delta_beyond_mag"] = df["sp_resp"] - df["sp_pred_from_mag"]
    df["config_version"] = cfg.CONFIG_VERSION
    df["efficiency_filter_version"] = EFFICIENCY_FILTER_VERSION
    df["min_responder_cells"] = min_responder_cells
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _wilcoxon(values: np.ndarray, alternative: str = "two-sided") -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (np.abs(v) > 0)]
    out = {"n": int(len(v)), "median": _f(np.median(v)) if len(v) else None}
    if len(v) < 10:
        out["p"] = None
        out["note"] = "n<10; point estimate only"
        return out
    try:
        stat, p = wilcoxon(v, alternative=alternative)
        out["statistic"] = _f(stat)
        out["p"] = _f(p)
    except Exception as e:
        out["p"] = None
        out["note"] = str(e)
    return out


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    return float(len(a & b) / len(a | b))


def summarize(df: pd.DataFrame, dataset_name: str, method: str) -> dict:
    ok = df[df["sp_resp"].notna()].copy()
    frac = df["frac_responders"].to_numpy(dtype=float)
    frac = frac[np.isfinite(frac)]
    s = {
        "dataset": dataset_name,
        "method": method,
        "config_version": cfg.CONFIG_VERSION,
        "efficiency_filter_version": EFFICIENCY_FILTER_VERSION,
        "analysis_role": "sensitivity_only_not_a_freeze",
        "n_perturbations": int(len(df)),
        "n_rescored": int(len(ok)),
        "n_no_calls": int((df["status"] == "no_calls").sum()),
        "n_no_responders": int((df["status"] == "no_responders").sum()),
        "n_too_few_responders": int((df["status"] == "too_few_responders").sum()),
        "n_all_responders": int((df["status"] == "all_responders").sum()),
        "n_cells_unclassified": int(df.get("n_unclassified", pd.Series(dtype=int)).sum()),
        "min_responder_cells": int(
            df["min_responder_cells"].iloc[0]
            if "min_responder_cells" in df.columns and len(df)
            else EFF_MIN_RESPONDER_CELLS
        ),
        "frac_responders_median": _f(np.median(frac)) if len(frac) else None,
        "frac_responders_q25": _f(np.percentile(frac, 25)) if len(frac) else None,
        "frac_responders_q75": _f(np.percentile(frac, 75)) if len(frac) else None,
    }

    if len(ok) >= 5:
        r_all = bootstrap_spearman_ci(
            ok["sp_all"], ok["sp_resp"], seed=_eff_seed(dataset_name, method, "rank")
        )
        s["rank_agreement_sp_all_vs_responder"] = {
            "rho": _f(r_all["rho"]),
            "ci_low": _f(r_all["ci_low"]),
            "ci_high": _f(r_all["ci_high"]),
            "p": _f(r_all["p"]),
            "n": int(r_all["n"]),
        }
        if "sp_frozen" in ok.columns and ok["sp_frozen"].notna().any():
            rf, pf = spearmanr(ok["sp_frozen"], ok["sp_resp"])
            s["rank_agreement_sp_frozen_vs_responder"] = {"rho": _f(rf), "p": _f(pf)}

        # Diagnostic only under direction-aligned callers (PS / Mixscape).
        s["delta_sp_vs_all_cells"] = _wilcoxon(ok["delta_sp"].to_numpy())
        s["z_vs_cell_count_matched_null"] = _wilcoxon(ok["z_matched"].to_numpy())
        s["delta_beyond_magnitude_baseline"] = _wilcoxon(
            ok["delta_beyond_mag"].to_numpy()
        )

        rho_mag_all, _ = spearmanr(ok["sp_all"], ok["mag_all"])
        rho_mag_resp, _ = spearmanr(ok["sp_resp"], ok["mag_resp"])
        s["sp_magnitude_rho_all_cells"] = _f(rho_mag_all)
        s["sp_magnitude_rho_responders"] = _f(rho_mag_resp)
        s["sp_magnitude_rho_delta"] = (
            _f(rho_mag_resp - rho_mag_all)
            if np.isfinite(rho_mag_all) and np.isfinite(rho_mag_resp)
            else None
        )
        s["sp_magnitude_rho_all_cells_n"] = int(len(ok))
        s["sp_magnitude_rho_all_cells_note"] = (
            "Spearman on the rescored (sp_resp-available) subset only — "
            "NOT the full frozen dataset when the caller drops perts "
            "(Mixscape no-responder attrition). Prefer "
            "sp_magnitude_rho_frozen_full for the manuscript baseline."
        )
        # Frozen denominators: full dataset vs same rescored subset.
        if {"sp_frozen", "mag_frozen"}.issubset(df.columns):
            fr_all = df.dropna(subset=["sp_frozen", "mag_frozen"])
            if len(fr_all) >= 5:
                rf, _ = spearmanr(fr_all["sp_frozen"], fr_all["mag_frozen"])
                s["sp_magnitude_rho_frozen_full"] = _f(rf)
                s["sp_magnitude_rho_frozen_full_n"] = int(len(fr_all))
            fr_ok = ok.dropna(subset=["sp_frozen", "mag_frozen"])
            if len(fr_ok) >= 5:
                rs, _ = spearmanr(fr_ok["sp_frozen"], fr_ok["mag_frozen"])
                s["sp_magnitude_rho_frozen_rescored_subset"] = _f(rs)
                s["sp_magnitude_rho_frozen_rescored_subset_n"] = int(len(fr_ok))
                # Primary Δ against the matched-denominator frozen baseline.
                s["sp_magnitude_rho_delta_vs_frozen_subset"] = (
                    _f(rho_mag_resp - rs)
                    if np.isfinite(rho_mag_resp) and np.isfinite(rs)
                    else None
                )
        s["magnitude_median_all"] = _f(ok["mag_all"].median())
        s["magnitude_median_responders"] = _f(ok["mag_resp"].median())

        nonresp = ok[ok["sp_nonresp"].notna()]
        if len(nonresp) >= 5:
            u = mannwhitneyu(
                nonresp["sp_resp"], nonresp["sp_nonresp"], alternative="greater"
            )
            s["responder_vs_nonresponder_sp"] = {
                "n_pairs": int(len(nonresp)),
                "median_responder": _f(nonresp["sp_resp"].median()),
                "median_nonresponder": _f(nonresp["sp_nonresp"].median()),
                "mwu_p_responder_greater": _f(u.pvalue),
            }

        # Decile stability: are the same perturbations at the extremes?
        q = max(3, int(round(0.1 * len(ok))))
        low_before = set(ok.nsmallest(q, "sp_all")["perturbation"])
        low_after = set(ok.nsmallest(q, "sp_resp")["perturbation"])
        high_before = set(ok.nlargest(q, "sp_all")["perturbation"])
        high_after = set(ok.nlargest(q, "sp_resp")["perturbation"])
        s["extreme_decile_stability"] = {
            "decile_size": int(q),
            "jaccard_lowest_sp": _jaccard(low_before, low_after),
            "jaccard_highest_sp": _jaccard(high_before, high_after),
        }

    # Is low Sp just low perturbation efficiency?
    eff = df[df["frac_responders"].notna() & df["sp_all"].notna()]
    if len(eff) >= 10:
        r_eff = bootstrap_spearman_ci(
            eff["sp_all"],
            eff["frac_responders"],
            seed=_eff_seed(dataset_name, method, "efficiency"),
        )
        part = partial_spearman_rank(
            eff["sp_all"].to_numpy(),
            eff["frac_responders"].to_numpy(),
            eff["mag_all"].to_numpy(),
        )
        s["sp_vs_responder_fraction"] = {
            "rho": _f(r_eff["rho"]),
            "ci_low": _f(r_eff["ci_low"]),
            "ci_high": _f(r_eff["ci_high"]),
            "p": _f(r_eff["p"]),
            "n": int(r_eff["n"]),
            "partial_given_magnitude_rho": _f(part["rho_partial"]),
            "partial_given_magnitude_p": _f(part["p"]),
            "partial_method": part["method"],
        }

    # --- Endpoint roles (pinned after Norman PS circularity audit) ---
    direction_aligned = method in EFF_DIRECTION_ALIGNED_METHODS
    s["responder_selection_geometry"] = (
        "direction_aligned_signature_dosage"
        if method == "ps"
        else (
            "direction_aligned_signature_gmm"
            if method == "mixscape"
            else "caller_dependent"
        )
    )
    s["ps_threshold"] = EFF_PS_THRESHOLD if method == "ps" else None
    s["ps_control_cells_for_ps_estimate"] = (
        "subsampled_to_ps_max_ctrl_cells; Sp control centroid uses full "
        "materialized control set (MAX_CONTROL_CELLS)"
        if method == "ps"
        else None
    )
    # Primary: survive under direction-aligned selection.
    s["primary_endpoints"] = {
        "sp_magnitude_rho_all_cells": s.get("sp_magnitude_rho_all_cells"),
        "sp_magnitude_rho_responders": s.get("sp_magnitude_rho_responders"),
        "sp_magnitude_rho_delta": s.get("sp_magnitude_rho_delta"),
        "rank_agreement_sp_all_vs_responder": s.get(
            "rank_agreement_sp_all_vs_responder"
        ),
        "note": (
            "Sp~mag Δ asks whether the magnitude redundancy is an efficiency "
            "artifact; rank agreement asks whether filtering rearranges which "
            "perturbations look coherent. Both are interpretable under PS."
        ),
    }
    # Secondary: expected under PS dosage-along-β; do not cite as evidence.
    mech_keys = (
        "delta_sp_vs_all_cells",
        "z_vs_cell_count_matched_null",
        "delta_beyond_magnitude_baseline",
        "sp_vs_responder_fraction",
        "responder_vs_nonresponder_sp",
    )
    s["mechanical_under_direction_selection"] = {
        "applies": bool(direction_aligned),
        "reason": (
            "Song PS is a constrained scalar dosage along the average DEG "
            "effect β (Y≈PS·β); Mixscape classifies on a signature GMM. "
            "Selecting responders therefore enriches for cells aligned with "
            "the mean shift — the quantity Sp measures. The cell-count-matched "
            "null is orthogonal to n, not to direction. Self-test with "
            "direction-independent labels (method=obs) gave null "
            "beyond-magnitude; PS on Norman did not."
            if direction_aligned
            else "caller is not stamped direction-aligned"
        ),
        "statistics": {k: s.get(k) for k in mech_keys if k in s},
    }

    rho = (s.get("rank_agreement_sp_all_vs_responder") or {}).get("rho")
    if rho is None:
        s["verdict"] = "insufficient_perturbations_rescored"
    elif rho >= EFF_RANK_PRESERVED_RHO:
        s["verdict"] = (
            f"rankings_preserved (rho={rho:.3f} >= {EFF_RANK_PRESERVED_RHO})"
        )
    elif rho >= EFF_RANK_PARTIAL_RHO:
        s["verdict"] = f"rankings_partially_preserved (rho={rho:.3f})"
    else:
        s["verdict"] = f"rankings_change (rho={rho:.3f} < {EFF_RANK_PARTIAL_RHO})"
    return s


def methods_blurb(summaries: list[dict]) -> str:
    lines = [
        "Perturbation-efficiency sensitivity analysis.",
        "",
        "Song et al. (2025) PS was used to identify responding cells: PS is a",
        "constrained non-negative scalar dosage along the average signature-DEG",
        "effect vector β (Y ≈ PS · β), rescaled to [0, 1] within each",
        "perturbation and thresholded at PS ≥ {ps:.2f} (pinned). Sp was then",
        "recomputed on responding cells only against the unchanged control",
        "centroid from the frozen preprocess path (CONFIG_VERSION {cv}). The",
        "frozen Sp table was reproduced to float precision before filtering;",
        "PS estimation may subsample controls, but the Sp control set does not.",
        "Perturbations retaining ≥ {minc} responders were rescored.",
        "",
        "PRIMARY endpoints (interpretable under direction-aligned PS selection):",
        "(i) Spearman(Sp, magnitude) before vs after responder filtering — asks",
        "whether the Sp~magnitude redundancy is an efficiency artifact;",
        "(ii) Spearman(Sp_all, Sp_responders) — asks whether filtering rearranges",
        "perturbation rankings (preserved if ρ ≥ {rankthr}).",
        "",
        "NOT cited as evidence under PS: ΔSp, z vs cell-count-matched null,",
        "beyond-magnitude residual, and ρ(Sp, responder fraction). Because PS",
        "selects cells by alignment with the mean effect direction — the same",
        "geometry Sp summarises — those statistics are mechanical consequences",
        "of the selection rule. A direction-independent self-test (ground-truth",
        "efficiency labels) left beyond-magnitude null; PS on Norman did not.",
        "This analysis is sensitivity-only; the frozen Sp table is unchanged.",
        "",
    ]
    text = "\n".join(lines).format(
        ps=EFF_PS_THRESHOLD,
        cv=cfg.CONFIG_VERSION,
        minc=EFF_MIN_RESPONDER_CELLS,
        rankthr=EFF_RANK_PRESERVED_RHO,
    )
    for s in summaries:
        rank = s.get("rank_agreement_sp_all_vs_responder") or {}
        text += (
            f"\n[{s['dataset']} / {s['method']}] "
            f"n_rescored={s.get('n_rescored')} of {s.get('n_perturbations')}; "
            f"median responder fraction={s.get('frac_responders_median')}; "
            f"Sp~mag ρ all={s.get('sp_magnitude_rho_all_cells')} → "
            f"responders={s.get('sp_magnitude_rho_responders')} "
            f"(Δ={s.get('sp_magnitude_rho_delta')}); "
            f"rho(Sp_all, Sp_responders)={rank.get('rho')} "
            f"[{rank.get('ci_low')}, {rank.get('ci_high')}]; "
            f"verdict={s.get('verdict')}; "
            f"selection={s.get('responder_selection_geometry')}\n"
        )
    text += (
        f"\nefficiency_filter_version={EFFICIENCY_FILTER_VERSION}; "
        f"config_version={cfg.CONFIG_VERSION} (unchanged by this analysis).\n"
    )
    return text


# ---------------------------------------------------------------------------
# Frozen-baseline guard
# ---------------------------------------------------------------------------


def check_frozen_reproduction(
    baseline: pd.DataFrame, frozen_df: pd.DataFrame, dataset_name: str
) -> dict:
    """
    Refuse to run the filter unless the recomputed Sp equals the frozen Sp.

    A responder-filtered Sp is only interpretable against the frozen number if
    both come from the same preprocessing. Abort, do not warn and continue.
    """
    fr = frozen_df[frozen_df["dataset"] == dataset_name][
        ["perturbation", "stability", "magnitude"]
    ].rename(columns={"stability": "sp_frozen", "magnitude": "mag_frozen"})
    if fr.empty:
        raise SystemExit(
            f"{dataset_name} has no rows in the frozen Sp table — wrong file or "
            "wrong dataset name."
        )
    merged = baseline.merge(fr, on="perturbation", how="inner")
    n_missing = len(baseline) - len(merged)
    if merged.empty:
        raise SystemExit(
            f"{dataset_name}: no perturbation names shared with the frozen table."
        )
    d_sp = float(np.max(np.abs(merged["stability"] - merged["sp_frozen"])))
    d_mag = float(np.max(np.abs(merged["magnitude"] - merged["mag_frozen"])))
    print(
        f"  frozen reproduction: n={len(merged)} matched "
        f"({n_missing} recomputed rows unmatched)  max|dSp|={d_sp:.3g}  "
        f"max|dMag|={d_mag:.3g}",
        flush=True,
    )
    if d_sp > EFF_FROZEN_TOL:
        raise SystemExit(
            f"ABORT: {dataset_name} recomputed Sp differs from the frozen table "
            f"(max|dSp|={d_sp:.3g} > {EFF_FROZEN_TOL}). The responder filter would "
            "be measured against a different preprocessing. Fix the environment "
            "or the pin before interpreting anything."
        )
    return {
        "n_matched": int(len(merged)),
        "n_unmatched": int(n_missing),
        "max_abs_delta_sp": d_sp,
        "max_abs_delta_magnitude": d_mag,
        "frozen_reproduced": True,
    }


def apply_frozen_baseline(df: pd.DataFrame, frozen_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Replace within-run Sp_all / mag_all with frozen values.

    Used by --ram-lite: the lite cell caps cannot reproduce the freeze, so the
    primary 'before' side is taken from frozen_sp_scores.csv. Responder Sp is
    still computed on the lite cell set under the same preprocess path.
    """
    fr = frozen_df[frozen_df["dataset"] == dataset_name][
        ["perturbation", "stability", "magnitude"]
    ].rename(columns={"stability": "sp_frozen", "magnitude": "mag_frozen"})
    if fr.empty:
        raise SystemExit(f"{dataset_name}: no frozen Sp rows for ram-lite baseline")
    out = df.drop(columns=[c for c in ("sp_frozen", "mag_frozen") if c in df.columns])
    out = out.merge(fr, on="perturbation", how="left")
    n_miss = int(out["sp_frozen"].isna().sum())
    if n_miss:
        print(
            f"  WARNING: {n_miss} lite-scored perts missing from frozen table",
            flush=True,
        )
    # Keep the lite recompute for audit, then overwrite the primary 'all' side.
    out["sp_all_lite"] = out["sp_all"]
    out["mag_all_lite"] = out["mag_all"]
    out["sp_all"] = out["sp_frozen"]
    out["mag_all"] = out["mag_frozen"]
    out["delta_sp"] = out["sp_resp"] - out["sp_all"]
    ok = out["sp_resp"].notna() & out["mag_resp"].notna() & out["sp_all"].notna()
    out["sp_pred_from_mag"] = np.nan
    if int(ok.sum()) >= 5:
        out.loc[ok, "sp_pred_from_mag"] = lowess_predict(
            out.loc[ok, "sp_all"].to_numpy(),
            out.loc[ok, "mag_all"].to_numpy(),
            out.loc[ok, "mag_resp"].to_numpy(),
        )
    out["delta_beyond_mag"] = out["sp_resp"] - out["sp_pred_from_mag"]
    out["baseline_source"] = "frozen_sp_scores.csv"
    return out


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------


def run_dataset_efficiency(
    dataset_name: str,
    method: str,
    *,
    sc,
    frozen_df: Optional[pd.DataFrame],
    h5ad: Optional[Path],
    max_cells_per_pert: int,
    max_control_cells: int,
    max_perts: Optional[int],
    mixscape_kwargs: dict,
    ps_kwargs: dict,
    obs_kwargs: dict,
    n_matched_draws: int,
    min_responder_cells: int,
    ram_lite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import gc

    print(f"\n{'=' * 66}\n>>> {dataset_name}  [{method}]\n{'=' * 66}", flush=True)
    if ram_lite:
        if frozen_df is None:
            raise SystemExit("--ram-lite requires a frozen Sp table (Sp_all source)")
        print(
            f"  RAM-lite: ≤{max_cells_per_pert}/pert, ≤{max_control_cells} controls; "
            "Sp_all/mag_all from frozen table (no bit-identical reproduction)",
            flush=True,
        )

    raw = load_raw(dataset_name, prefer_local=True, h5ad_path=h5ad)
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset_name, sc)
    del raw
    gc.collect()
    adata, valid, counts = materialize_min_cells(
        adata,
        pert_col,
        ctrl_label,
        min_cells=cfg.MIN_CELLS,
        max_cells_per_pert=max_cells_per_pert,
        max_control_cells=max_control_cells,
        seed=cfg.SEED,
        max_perts=max_perts,
    )
    gc.collect()
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
    gc.collect()

    guard = {
        "frozen_reproduced": False,
        "note": "frozen check skipped",
        "ram_lite": bool(ram_lite),
        "max_cells_per_pert": int(max_cells_per_pert),
        "max_control_cells": int(max_control_cells),
    }
    if ram_lite:
        # Skip scoring the lite all-cell Sp for the guard — primary baseline
        # is the frozen table. Still score once for an optional audit column.
        baseline = score_perturbations(
            adata, pert_col, ctrl_label, valid, counts, dataset_name
        )
        guard["note"] = (
            "ram_lite: Sp_all/mag_all taken from frozen_sp_scores.csv; "
            "bit-identical freeze reproduction not attempted at reduced cell caps"
        )
        guard["n_lite_baseline_perts"] = int(len(baseline))
        print(
            f"  lite preprocess scored {len(baseline)} perts "
            f"({adata.n_obs} cells) — using frozen Sp as primary baseline",
            flush=True,
        )
    elif frozen_df is not None:
        baseline = score_perturbations(
            adata, pert_col, ctrl_label, valid, counts, dataset_name
        )
        guard = check_frozen_reproduction(baseline, frozen_df, dataset_name)
        guard["ram_lite"] = False
        guard["max_cells_per_pert"] = int(max_cells_per_pert)
        guard["max_control_cells"] = int(max_control_cells)
    else:
        print(
            "  WARNING: no frozen Sp table supplied — before/after numbers are "
            "NOT manuscript-grade.",
            flush=True,
        )

    if method == "mixscape":
        calls = responder_calls_mixscape(adata, pert_col, ctrl_label, **mixscape_kwargs)
    elif method == "ps":
        calls = responder_calls_ps(adata, pert_col, ctrl_label, valid, **ps_kwargs)
    elif method == "obs":
        calls = responder_calls_obs(adata, pert_col, ctrl_label, **obs_kwargs)
    else:
        raise ValueError(f"Unknown method {method!r}")

    n_called = len(calls)
    n_resp = int(calls["is_responder"].sum()) if n_called else 0
    print(
        f"  responder calls: {n_resp}/{n_called} perturbed cells "
        f"({100.0 * n_resp / max(n_called, 1):.1f}%)",
        flush=True,
    )

    df = rescore_on_responders(
        adata,
        pert_col,
        ctrl_label,
        valid,
        calls,
        dataset_name,
        method,
        n_matched_draws=n_matched_draws,
        min_responder_cells=min_responder_cells,
    )
    if ram_lite:
        df = apply_frozen_baseline(df, frozen_df, dataset_name)
    elif frozen_df is not None:
        fr = frozen_df[frozen_df["dataset"] == dataset_name][
            ["perturbation", "stability", "magnitude"]
        ].rename(columns={"stability": "sp_frozen", "magnitude": "mag_frozen"})
        df = df.merge(fr, on="perturbation", how="left")

    summary = summarize(df, dataset_name, method)
    summary["frozen_guard"] = guard
    summary["ram_lite"] = bool(ram_lite)
    summary["baseline_source"] = (
        "frozen_sp_scores.csv" if ram_lite else "recomputed_then_guarded"
    )
    summary["n_cells_scored"] = int(adata.n_obs)
    summary["n_control_cells"] = int((adata.obs[pert_col] == ctrl_label).sum())
    summary["modality"] = cfg.DATASETS[dataset_name]["modality"]
    summary["cell_type"] = cfg.DATASETS[dataset_name]["cell_type"]
    if method == "mixscape":
        summary["responder_definition"] = (
            f"mixscape_class_global in {list(EFF_MIXSCAPE_RESPONDER_CLASSES)}"
        )
        summary["mixscape_params"] = mixscape_kwargs
    elif method == "ps":
        summary["responder_definition"] = (
            f"Song PS >= {ps_kwargs.get('threshold', EFF_PS_THRESHOLD)} "
            "(PS is rescaled within perturbation)"
        )
        summary["ps_params"] = {k: v for k, v in ps_kwargs.items()}
    else:
        summary["responder_definition"] = (
            f"obs[{obs_kwargs.get('column')}] >= {obs_kwargs.get('threshold')}"
        )

    calls = calls.copy()
    calls["dataset"] = dataset_name
    calls["method"] = method
    del adata
    gc.collect()
    return df, calls, summary


def print_summary(s: dict) -> None:
    rank = s.get("rank_agreement_sp_all_vs_responder") or {}
    print(f"\n  --- {s['dataset']} / {s['method']} ---")
    print(
        f"  rescored {s.get('n_rescored')}/{s.get('n_perturbations')} perts; "
        f"no calls {s.get('n_no_calls')}; "
        f"no responders {s.get('n_no_responders')}; "
        f"too few {s.get('n_too_few_responders')}; "
        f"fully efficient {s.get('n_all_responders')}"
    )
    print(
        f"  responder fraction median={s.get('frac_responders_median')} "
        f"[{s.get('frac_responders_q25')}, {s.get('frac_responders_q75')}]"
    )
    print(f"  selection geometry: {s.get('responder_selection_geometry')}")
    print(
        f"  PRIMARY Sp~mag ρ (rescored n={s.get('sp_magnitude_rho_all_cells_n')}): "
        f"all={s.get('sp_magnitude_rho_all_cells')} → "
        f"responders={s.get('sp_magnitude_rho_responders')} "
        f"(Δ={s.get('sp_magnitude_rho_delta')})"
    )
    if s.get("sp_magnitude_rho_frozen_full") is not None:
        print(
            f"  frozen Sp~mag ρ: full n={s.get('sp_magnitude_rho_frozen_full_n')} "
            f"ρ={s.get('sp_magnitude_rho_frozen_full')}; "
            f"rescored-subset n={s.get('sp_magnitude_rho_frozen_rescored_subset_n')} "
            f"ρ={s.get('sp_magnitude_rho_frozen_rescored_subset')} "
            f"(Δ resp−frozen_subset="
            f"{s.get('sp_magnitude_rho_delta_vs_frozen_subset')})"
        )
    print(
        f"  PRIMARY rho(Sp_all, Sp_responders)={rank.get('rho')} "
        f"[{rank.get('ci_low')}, {rank.get('ci_high')}]"
    )
    if (s.get("mechanical_under_direction_selection") or {}).get("applies"):
        print(
            "  (ΔSp / z_matched / beyond-mag / ρ(Sp,frac) logged but NOT "
            "cited — mechanical under direction-aligned PS/Mixscape selection)"
        )
    print(f"  rank rho={rank.get('rho')} [{rank.get('ci_low')}, {rank.get('ci_high')}]")


# ---------------------------------------------------------------------------
# Self-test (no scanpy / pertpy required)
# ---------------------------------------------------------------------------


def self_test() -> int:
    """
    Validate the rescoring and comparison layer on data with known efficiency.

    Builds perturbations in PCA space whose responders share a direction and
    whose non-responders are drawn from the control distribution, then checks
    that responder filtering recovers coherence for mixtures, leaves a fully
    efficient perturbation alone, that the cell-count-matched null does not
    manufacture the effect on its own, and that the summary recovers the
    planted Sp~efficiency relationship.
    """
    print("efficiency_filter_sp self-test")
    rng = np.random.default_rng(cfg.SEED)
    n_dim, n_ctrl, n_pert_cells = 24, 400, 200
    X_ctrl = rng.normal(size=(n_ctrl, n_dim))

    efficiencies = [1.0, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
    specs = {f"EFF{int(round(100 * e)):03d}_{i}": e for i, e in enumerate(efficiencies)}
    blocks, labels, truth = [X_ctrl], ["control"] * n_ctrl, []
    for i, (name, eff) in enumerate(specs.items()):
        direction = np.zeros(n_dim)
        direction[i % n_dim] = 1.0
        n_resp = int(round(eff * n_pert_cells))
        resp = 3.0 * direction + rng.normal(scale=1.0, size=(n_resp, n_dim))
        non = rng.normal(scale=1.0, size=(n_pert_cells - n_resp, n_dim))
        blocks.append(np.vstack([resp, non]) if len(non) else resp)
        labels += [name] * n_pert_cells
        truth += [1.0] * n_resp + [0.0] * (n_pert_cells - n_resp)
    cells = [f"ctrl_{i}" for i in range(n_ctrl)] + [
        f"{p}_{i}" for p in specs for i in range(n_pert_cells)
    ]

    import anndata as ad

    X = np.vstack(blocks)
    adata = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame(
            {"condition": labels, "true_efficiency": [np.nan] * n_ctrl + truth},
            index=pd.Index(cells, name="cell"),
        ),
    )
    adata.obsm["X_pca"] = X

    calls = responder_calls_obs(
        adata, "condition", "control", "true_efficiency", 0.5
    )
    df = rescore_on_responders(
        adata,
        "condition",
        "control",
        list(specs),
        calls,
        "self_test",
        "obs",
        n_matched_draws=100,
    )
    cols = [
        "perturbation",
        "frac_responders",
        "sp_all",
        "sp_resp",
        "sp_nonresp",
        "delta_sp",
        "z_matched",
        "delta_beyond_mag",
        "status",
    ]
    print(df[cols].to_string(index=False))

    failures = []
    by_pert = df.set_index("perturbation")
    for name, eff in specs.items():
        r = by_pert.loc[name]
        if abs(r["frac_responders"] - eff) > 0.02:
            failures.append(f"{name}: responder fraction {r['frac_responders']} != {eff}")
        if eff >= 1.0:
            if abs(r["delta_sp"]) > 1e-12:
                failures.append(f"{name}: fully efficient pert changed by {r['delta_sp']}")
            continue
        if not (r["sp_resp"] > r["sp_all"]):
            failures.append(f"{name}: Sp did not rise ({r['sp_all']} -> {r['sp_resp']})")
        if not (r["z_matched"] > 3):
            failures.append(f"{name}: z vs matched null too small ({r['z_matched']})")
        if np.isfinite(r["sp_nonresp"]) and not (r["sp_resp"] > r["sp_nonresp"]):
            failures.append(f"{name}: non-responders not less coherent")
        # A random subset of the same size must not reproduce the responder gain.
        if not abs(r["sp_matched_mean"] - r["sp_all"]) < 0.05:
            failures.append(
                f"{name}: matched null mean {r['sp_matched_mean']} drifted from "
                f"all-cell Sp {r['sp_all']} — the null is not unbiased"
            )

    s = summarize(df, "self_test", "obs")
    print(json.dumps({k: v for k, v in s.items() if k != "frozen_guard"}, indent=2))

    rank = (s.get("rank_agreement_sp_all_vs_responder") or {}).get("rho")
    if rank is None:
        failures.append("summary did not compute rank agreement")
    eff_block = s.get("sp_vs_responder_fraction") or {}
    if eff_block.get("rho") is None:
        failures.append("summary did not compute the Sp~responder-fraction test")
    elif eff_block["rho"] < 0.8:
        failures.append(
            f"planted Sp~efficiency relationship not recovered (rho={eff_block['rho']})"
        )
    beyond = (s.get("delta_beyond_magnitude_baseline") or {}).get("median")
    if beyond is None:
        failures.append("LOESS magnitude baseline produced no beyond-magnitude median")

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSELF-TEST PASSED")
    return 0


# ---------------------------------------------------------------------------
# Per-cell caller agreement (PS ↔ Mixscape)
# ---------------------------------------------------------------------------


def compare_percell_calls(
    path_a: Path,
    path_b: Path,
    *,
    label_a: str = "a",
    label_b: str = "b",
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Jaccard / Cohen's κ on per-cell is_responder calls from two arms.

    Join key = (dataset, perturbation, cell). Cells present in only one file
    are counted as non-overlap attrition, not as disagreements.
    """
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    for name, df in ((label_a, a), (label_b, b)):
        need = {"cell", "perturbation", "is_responder"}
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f"{name} missing columns {sorted(miss)}: {path_a if name==label_a else path_b}")
    key = ["dataset", "perturbation", "cell"] if "dataset" in a.columns and "dataset" in b.columns else ["perturbation", "cell"]
    a = a[key + ["is_responder"]].copy()
    b = b[key + ["is_responder"]].copy()
    a["is_responder"] = a["is_responder"].astype(bool)
    b["is_responder"] = b["is_responder"].astype(bool)
    a = a.rename(columns={"is_responder": f"resp_{label_a}"})
    b = b.rename(columns={"is_responder": f"resp_{label_b}"})
    m = a.merge(b, on=key, how="inner")
    if m.empty:
        raise SystemExit("no shared (perturbation, cell) rows between the two per-cell tables")

    ra = m[f"resp_{label_a}"].to_numpy()
    rb = m[f"resp_{label_b}"].to_numpy()
    both = int((ra & rb).sum())
    only_a = int((ra & ~rb).sum())
    only_b = int((~ra & rb).sum())
    neither = int((~ra & ~rb).sum())
    union = both + only_a + only_b
    jaccard = float(both / union) if union else None
    # Cohen's κ
    n = len(m)
    p_o = (both + neither) / n
    p_yes = (ra.mean()) * (rb.mean())
    p_no = (1 - ra.mean()) * (1 - rb.mean())
    p_e = p_yes + p_no
    kappa = float((p_o - p_e) / (1 - p_e)) if abs(1 - p_e) > 1e-12 else None

    out = {
        "efficiency_filter_version": EFFICIENCY_FILTER_VERSION,
        "path_a": str(path_a),
        "path_b": str(path_b),
        "label_a": label_a,
        "label_b": label_b,
        "n_cells_a": int(len(a)),
        "n_cells_b": int(len(b)),
        "n_shared_cells": int(n),
        "frac_responders_a": _f(ra.mean()),
        "frac_responders_b": _f(rb.mean()),
        "n_both_responder": both,
        "n_only_a": only_a,
        "n_only_b": only_b,
        "n_neither": neither,
        "jaccard_responders": jaccard,
        "cohens_kappa": kappa,
        "agreement_rate": _f(p_o),
        "interpretation": (
            "poor_agreement_responder_is_method_defined"
            if (jaccard is not None and jaccard < 0.5)
            or (kappa is not None and kappa < 0.4)
            else (
                "moderate_agreement"
                if (jaccard is not None and jaccard < 0.7)
                or (kappa is not None and kappa < 0.6)
                else "high_agreement"
            )
        ),
    }
    print(json.dumps(out, indent=2))
    if out_dir is not None:
        _write(
            Path(out_dir) / f"{OUT_PREFIX}_percell_agreement_{label_a}_vs_{label_b}.json",
            lambda p: p.write_text(json.dumps(out, indent=2)),
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_datasets(requested: Optional[list[str]]) -> list[str]:
    names = cfg.main_dataset_names()
    if not requested:
        # Efficiency confound needs a powered dataset, not all six at once.
        return ["Norman 2019 (CRISPRa)"]
    out = []
    for r in requested:
        exact = cfg.resolve_dataset_name(r)
        if exact in cfg.DATASETS:
            out.append(exact)
            continue
        hits = [n for n in names if r.lower() in n.lower()]
        if len(hits) != 1:
            raise SystemExit(f"--datasets {r!r} matched {hits or 'nothing'}; be explicit")
        out.append(hits[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset display names or substrings (default: Norman 2019)",
    )
    parser.add_argument(
        "--method",
        choices=["mixscape", "ps", "obs"],
        default="mixscape",
        help="Responder caller (default: mixscape)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument(
        "--frozen-sp",
        type=Path,
        default=None,
        help=f"Path to {FROZEN_TABLE_NAME} (read-only; guards the baseline)",
    )
    parser.add_argument(
        "--no-frozen-check",
        action="store_true",
        help="Run without the frozen table (diagnostic only; not manuscript-grade)",
    )
    parser.add_argument(
        "--ram-lite",
        action="store_true",
        help=(
            "Replogle-scale Colab path: downsample to "
            f"≤{EFF_RAM_LITE_MAX_CELLS_PER_PERT}/pert and "
            f"≤{EFF_RAM_LITE_MAX_CONTROL_CELLS} controls; take Sp_all/mag_all "
            "from frozen_sp_scores.csv (skips bit-identical freeze reproduction)."
        ),
    )
    parser.add_argument("--max-cells-per-pert", type=int, default=None)
    parser.add_argument("--max-control-cells", type=int, default=None)
    parser.add_argument(
        "--max-perts",
        type=int,
        default=None,
        help="Smoke path only — breaks the frozen reproduction check",
    )
    parser.add_argument("--min-responder-cells", type=int, default=EFF_MIN_RESPONDER_CELLS)
    parser.add_argument("--matched-draws", type=int, default=EFF_MATCHED_DRAWS)
    parser.add_argument("--mixscape-n-neighbors", type=int, default=20)
    parser.add_argument("--mixscape-n-dims", type=int, default=15)
    parser.add_argument("--mixscape-min-de-genes", type=int, default=5)
    parser.add_argument("--ps-threshold", type=float, default=EFF_PS_THRESHOLD)
    parser.add_argument("--ps-max-ctrl-cells", type=int, default=1000)
    parser.add_argument("--obs-score-column", type=str, default=None)
    parser.add_argument("--obs-score-threshold", type=float, default=0.5)
    parser.add_argument("--write-percell", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--compare-percell-a",
        type=Path,
        default=None,
        help="Per-cell CSV/CSV.GZ from arm A (e.g. Norman PS)",
    )
    parser.add_argument(
        "--compare-percell-b",
        type=Path,
        default=None,
        help="Per-cell CSV/CSV.GZ from arm B (e.g. Norman Mixscape)",
    )
    parser.add_argument("--compare-label-a", type=str, default="ps")
    parser.add_argument("--compare-label-b", type=str, default="mixscape")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.compare_percell_a or args.compare_percell_b:
        if not (args.compare_percell_a and args.compare_percell_b):
            raise SystemExit("need both --compare-percell-a and --compare-percell-b")
        out_dir = resolve_out_dir(args.out_dir)
        compare_percell_calls(
            args.compare_percell_a,
            args.compare_percell_b,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
            out_dir=out_dir,
        )
        return 0

    if args.method == "obs" and not args.obs_score_column:
        raise SystemExit("--method obs requires --obs-score-column")
    if args.ram_lite and args.no_frozen_check:
        raise SystemExit("--ram-lite needs the frozen Sp table as Sp_all; drop --no-frozen-check")

    # Resolve cell caps: ram-lite defaults are lower; explicit flags win.
    if args.max_cells_per_pert is None:
        args.max_cells_per_pert = (
            EFF_RAM_LITE_MAX_CELLS_PER_PERT if args.ram_lite else cfg.MAX_CELLS_PER_PERT
        )
    if args.max_control_cells is None:
        args.max_control_cells = (
            EFF_RAM_LITE_MAX_CONTROL_CELLS if args.ram_lite else cfg.MAX_CONTROL_CELLS
        )
    # PS estimate controls should not exceed the materialized control pool.
    args.ps_max_ctrl_cells = min(int(args.ps_max_ctrl_cells), int(args.max_control_cells))

    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(
        f"CONFIG_VERSION={cfg.CONFIG_VERSION} (not modified)  "
        f"EFFICIENCY_FILTER_VERSION={EFFICIENCY_FILTER_VERSION}"
        + ("  RAM_LITE=1" if args.ram_lite else "")
    )

    frozen_df = None
    if not args.no_frozen_check:
        frozen_path = args.frozen_sp or find_sp_csv(out_dir)
        assert_frozen_sp_compatible(frozen_path)
        frozen_df = load_sp_table(Path(frozen_path))
    if args.max_perts and frozen_df is not None and not args.ram_lite:
        raise SystemExit(
            "--max-perts subsets perturbations and breaks the frozen reproduction "
            "check; add --no-frozen-check or --ram-lite to acknowledge it."
        )

    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = Path(os.environ.get("SCVERSE_DATADIR", str(cfg.CACHE_DIR)))

    datasets = resolve_datasets(args.datasets)
    print(
        f"Datasets: {datasets}  method={args.method}  "
        f"caps={args.max_cells_per_pert}/pert, {args.max_control_cells} ctrl"
    )

    all_scores, all_calls, summaries, failures = [], [], [], []
    for name in datasets:
        try:
            df, calls, summary = run_dataset_efficiency(
                name,
                args.method,
                sc=sc,
                frozen_df=frozen_df,
                h5ad=args.h5ad,
                max_cells_per_pert=args.max_cells_per_pert,
                max_control_cells=args.max_control_cells,
                max_perts=args.max_perts,
                mixscape_kwargs={
                    "n_neighbors": args.mixscape_n_neighbors,
                    "n_dims": args.mixscape_n_dims,
                    "min_de_genes": args.mixscape_min_de_genes,
                },
                ps_kwargs={
                    "threshold": args.ps_threshold,
                    "max_ctrl_cells": args.ps_max_ctrl_cells,
                },
                obs_kwargs={
                    "column": args.obs_score_column,
                    "threshold": args.obs_score_threshold,
                },
                n_matched_draws=args.matched_draws,
                min_responder_cells=args.min_responder_cells,
                ram_lite=args.ram_lite,
            )
        except SystemExit:
            raise
        except Exception as e:
            traceback.print_exc()
            failures.append({"dataset": name, "method": args.method, "error": str(e)})
            continue

        _write(
            out_dir / f"{OUT_PREFIX}_scores_{_slug(name)}_{args.method}.csv",
            lambda p: df.to_csv(p, index=False),
        )
        if args.write_percell:
            _write(
                out_dir / f"{OUT_PREFIX}_percell_{_slug(name)}_{args.method}.csv.gz",
                lambda p: calls.to_csv(p, index=False, compression="gzip"),
            )
        all_scores.append(df)
        all_calls.append(calls)
        summaries.append(summary)
        print_summary(summary)

    if not summaries:
        print("\nNo dataset completed.")
        return 1

    combined = pd.concat(all_scores, ignore_index=True)
    print(f"\ncombined: {len(combined)} rows")
    _write(
        out_dir / f"{OUT_PREFIX}_scores.csv",
        lambda p: combined.to_csv(p, index=False),
    )

    payload = {
        "efficiency_filter_version": EFFICIENCY_FILTER_VERSION,
        "config_version": cfg.CONFIG_VERSION,
        "analysis_role": "sensitivity_only_not_a_freeze",
        "frozen_table_modified": False,
        "method": args.method,
        "ram_lite": bool(args.ram_lite),
        "max_cells_per_pert": int(args.max_cells_per_pert),
        "max_control_cells": int(args.max_control_cells),
        "min_responder_cells": args.min_responder_cells,
        "matched_draws": args.matched_draws,
        "loess_frac": EFF_LOESS_FRAC,
        "datasets": summaries,
        "failures": failures,
    }
    _write(
        out_dir / f"{OUT_PREFIX}_summary_{args.method}.json",
        lambda p: p.write_text(json.dumps(payload, indent=2)),
    )
    _write(
        out_dir / f"{OUT_PREFIX}_methods_blurb_{args.method}.txt",
        lambda p: p.write_text(methods_blurb(summaries)),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
