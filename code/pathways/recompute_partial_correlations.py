#!/usr/bin/env python3
"""
Recompute key partial correlations with rank-based partial Spearman.

Reads existing result CSVs (magnitude/stability/stress columns), recomputes:
  - Sp ~ magnitude  (Spearman + CI)
  - Sp ~ stress | magnitude  (partial Spearman rank vs legacy residual method)

Writes side-by-side comparison so you can see whether conclusions change.

Usage:
  python recompute_partial_correlations.py \\
      --input shesha-crispr/shesha_crispr_results_euclidean.csv

  # also try common stress CSV layouts
  python recompute_partial_correlations.py --input shesha-crispr/*.csv --n-bootstrap 2000
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline_config as cfg
from remap_modality_labels import DATASET_RENAMES
from stats_utils import (
    bootstrap_partial_spearman_ci,
    bootstrap_spearman_ci,
)

STRESS_MARKERS = ["DDIT3", "ATF4", "XBP1", "HSPA5"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower(): c for c in df.columns}
    rename = {}
    aliases = {
        "stability": ["stability", "sp", "coherence", "directional_coherence"],
        "magnitude": ["magnitude", "mp", "effect_size"],
        "dataset": ["dataset", "dataset_name", "study"],
        "perturbation": ["perturbation", "gene", "target", "condition"],
        "n_cells": ["n_cells", "ncell", "n"],
        "spread": ["spread", "dispersion"],
    }
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}
    for canon, opts in aliases.items():
        if canon in out.columns:
            continue
        for o in opts:
            if o in lower:
                rename[lower[o]] = canon
                break
    out = out.rename(columns=rename)
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].astype(str).map(
            lambda x: DATASET_RENAMES.get(x, cfg.resolve_dataset_name(x))
        )
    return out


def _find_stress_cols(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return list of (marker, column_name)."""
    found = []
    for m in STRESS_MARKERS:
        for cand in (f"stress_{m}", m, f"{m}_expr", f"{m}_score"):
            if cand in df.columns:
                found.append((m, cand))
                break
    return found


def recompute_file(path: Path, n_bootstrap: int, seed: int) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    if "stability" not in df.columns or "magnitude" not in df.columns:
        print(f"  skip {path.name}: need stability + magnitude columns (have {list(df.columns)})")
        return pd.DataFrame()

    rows = []
    datasets = (
        sorted(df["dataset"].dropna().unique())
        if "dataset" in df.columns
        else ["ALL"]
    )

    for ds in datasets:
        sub = df if ds == "ALL" else df[df["dataset"] == ds]
        sub = sub[["stability", "magnitude"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 15:
            continue

        raw = bootstrap_spearman_ci(
            sub["stability"], sub["magnitude"], n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append(
            {
                "source_file": path.name,
                "dataset": ds,
                "test": "Sp_vs_magnitude",
                "marker": "",
                "n": raw["n"],
                "rho_spearman": raw["rho"],
                "p_spearman": raw["p"],
                "ci_low_spearman": raw["ci_low"],
                "ci_high_spearman": raw["ci_high"],
                "rho_partial_rank": np.nan,
                "p_partial_rank": np.nan,
                "ci_low_rank": np.nan,
                "ci_high_rank": np.nan,
                "rho_partial_raw_resid": np.nan,
                "p_partial_raw_resid": np.nan,
                "ci_low_raw_resid": np.nan,
                "ci_high_raw_resid": np.nan,
                "delta_rank_minus_rawresid": np.nan,
                "config_version": cfg.CONFIG_VERSION,
            }
        )

    # stress partials if columns exist
    stress_cols = _find_stress_cols(df)
    if stress_cols and "dataset" in df.columns:
        for ds in sorted(df["dataset"].dropna().unique()):
            for marker, col in stress_cols:
                sub = df[df["dataset"] == ds][["stability", "magnitude", col]].copy()
                sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub) < 15:
                    continue
                rank = bootstrap_partial_spearman_ci(
                    sub["stability"],
                    sub[col],
                    sub["magnitude"],
                    n_bootstrap=n_bootstrap,
                    seed=seed + 17,
                    method="rank",
                )
                legacy = bootstrap_partial_spearman_ci(
                    sub["stability"],
                    sub[col],
                    sub["magnitude"],
                    n_bootstrap=n_bootstrap,
                    seed=seed + 17,
                    method="raw_residuals",
                )
                delta = (
                    rank["rho_partial"] - legacy["rho_partial"]
                    if np.isfinite(rank["rho_partial"]) and np.isfinite(legacy["rho_partial"])
                    else np.nan
                )
                rows.append(
                    {
                        "source_file": path.name,
                        "dataset": ds,
                        "test": "Sp_vs_stress_partial_magnitude",
                        "marker": marker,
                        "n": rank["n"],
                        "rho_spearman": np.nan,
                        "p_spearman": np.nan,
                        "ci_low_spearman": np.nan,
                        "ci_high_spearman": np.nan,
                        "rho_partial_rank": rank["rho_partial"],
                        "p_partial_rank": rank["p"],
                        "ci_low_rank": rank["ci_low"],
                        "ci_high_rank": rank["ci_high"],
                        "rho_partial_raw_resid": legacy["rho_partial"],
                        "p_partial_raw_resid": legacy["p"],
                        "ci_low_raw_resid": legacy["ci_low"],
                        "ci_high_raw_resid": legacy["ci_high"],
                        "delta_rank_minus_rawresid": delta,
                        "config_version": cfg.CONFIG_VERSION,
                    }
                )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Result CSV(s) with stability/magnitude (and optional stress_*)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.OUTPUT_DIR / "partial_corr_rank_vs_legacy.csv",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Bootstrap iters (default 2000 for speed; use 10000 for manuscript)")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()

    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"method_default=partial_spearman_rank (pingouin)")
    print(f"n_bootstrap={args.n_bootstrap}")

    frames = []
    for path in args.input:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        print(f"\n>>> {path}")
        part = recompute_file(path, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if len(part):
            frames.append(part)
            # quick print
            for _, r in part.iterrows():
                if r["test"] == "Sp_vs_magnitude":
                    print(
                        f"  {r['dataset']}: Sp~mag rho={r['rho_spearman']:.3f} "
                        f"[{r['ci_low_spearman']:.3f},{r['ci_high_spearman']:.3f}] n={r['n']}"
                    )
                else:
                    print(
                        f"  {r['dataset']} {r['marker']}: "
                        f"rank={r['rho_partial_rank']:.3f} "
                        f"legacy={r['rho_partial_raw_resid']:.3f} "
                        f"Δ={r['delta_rank_minus_rawresid']:.3f}"
                    )

    if not frames:
        print("\nNo usable inputs. Run run_frozen_main.py first, or pass a CSV with "
              "columns: dataset, stability, magnitude.")
        return

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} rows)")
    print("Manuscript should report rho_partial_rank (not legacy raw-residual Spearman).")


if __name__ == "__main__":
    main()
