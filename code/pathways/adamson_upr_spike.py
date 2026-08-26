#!/usr/bin/env python3
"""
Adamson 2016 UPR positive-control spike.

Loads pertpy.data.adamson_2016_upr_perturb_seq (82 UPR-related gene targets),
scores Sp under the frozen pipeline_config, and writes a summary of the raw
Sp contrast (strong vs null/weak stress association).

Usage:
  python adamson_upr_spike.py
  python adamson_upr_spike.py --compare-pilot   # also score the pilot TF arm
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

# Deterministic BLAS before numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pipeline_config as cfg
from pipeline_core import run_dataset, setup_cache

# Pinned in pipeline_config — do not redefine membership here.
UPR_CORE = set(cfg.UPR_CORE_GENES)
UPR_CORE_CANONICAL = set(cfg.UPR_CORE_CANONICAL)
UPR_CORE_ALIASES = dict(cfg.UPR_CORE_ALIASES)


def _gene_token(name: str) -> str:
    """Normalize perturbation labels for membership checks."""
    s = str(name).upper().replace("-", "_")
    # strip trailing guide suffixes like _1, _P1, etc.
    parts = s.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return parts[0]


def annotate_upr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gene"] = out["perturbation"].map(_gene_token)
    out["gene_canonical"] = out["gene"].map(
        lambda g: UPR_CORE_ALIASES.get(g, g)
    )
    out["is_upr_core"] = out["gene"].isin(UPR_CORE) | out["gene_canonical"].isin(
        UPR_CORE_CANONICAL
    )
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Venue-fork summary: is Sp lower for UPR-core genes / associated with stress biology?"""
    from scipy.stats import mannwhitneyu

    df = annotate_upr(df)
    rho_mag, p_mag = spearmanr(df["magnitude"], df["stability"])
    core = df.loc[df["is_upr_core"], "stability"]
    other = df.loc[~df["is_upr_core"], "stability"]

    mwu_stat, mwu_p = (None, None)
    if len(core) >= 3 and len(other) >= 5:
        mwu_stat, mwu_p = mannwhitneyu(core, other, alternative="less")

    summary = {
        "config_version": cfg.CONFIG_VERSION,
        "upr_core_set_id": cfg.UPR_CORE_SET_ID,
        "upr_core_genes_pinned": sorted(cfg.UPR_CORE_CANONICAL),
        "upr_core_n_unique": int(cfg.UPR_CORE_N_UNIQUE),
        "upr_core_aliases": dict(cfg.UPR_CORE_ALIASES),
        "n_perturbations": int(len(df)),
        "n_upr_core_matched": int(df["is_upr_core"].sum()),
        "median_sp": float(df["stability"].median()),
        "mean_sp_upr_core": float(core.mean()) if len(core) else None,
        "mean_sp_other": float(other.mean()) if len(other) else None,
        "median_sp_upr_core": float(core.median()) if len(core) else None,
        "median_sp_other": float(other.median()) if len(other) else None,
        "mwu_upr_core_lower_stat": float(mwu_stat) if mwu_stat is not None else None,
        "mwu_upr_core_lower_p": float(mwu_p) if mwu_p is not None else None,
        "spearman_magnitude_sp": float(rho_mag),
        "spearman_magnitude_sp_p": float(p_mag),
        "preprocess_note": (
            "If scanpy/numba is broken, spike uses numpy/sklearn HVG+PCA fallback; "
            "re-run with working scanpy for manuscript numbers."
        ),
        "lowest_sp_genes": (
            df.nsmallest(10, "stability")[["perturbation", "stability", "magnitude", "n_cells"]]
            .assign(gene=lambda x: x["perturbation"].map(_gene_token))
            .to_dict(orient="records")
        ),
        "upr_core_rows": (
            df.loc[df["is_upr_core"], ["perturbation", "gene", "stability", "magnitude", "n_cells"]]
            .sort_values("stability")
            .to_dict(orient="records")
        ),
    }

    # Descriptive deltas only — the magnitude-conditioned test lives in
    # adamson_upr_magnitude_partial.py (ci_and_fdr.v1).
    if len(core) >= 3 and len(other) >= 5:
        summary["median_sp_delta_other_minus_core"] = float(
            other.median() - core.median()
        )
        summary["mean_sp_delta_other_minus_core"] = float(
            other.mean() - core.mean()
        )
        sensors_low = {"ERN1", "EIF2AK3", "ATF6"}
        sensor_rows = df[df["gene"].isin(sensors_low)]
        summary["sensor_genes_in_bottom_quartile"] = int(
            (sensor_rows["stability"] <= df["stability"].quantile(0.25)).sum()
        )
    summary["venue_signal"] = "deferred_to_magnitude_partial"
    summary["venue_note"] = (
        "Raw Sp MWU is descriptive only. Magnitude-conditioned test = "
        "adamson_upr_magnitude_partial.py (ci_and_fdr.v1 knife-edge). "
        "Do not quote this spike summary as the primary UPR claim."
    )
    # Post-hoc lowest-Sp ER trafficking list — NOT the
    # pre-specified core. Reported so we can state whether it
    # alone recovers a stronger raw MWU on this table (never the primary claim).
    trafficking = {
        "GBF1",
        "SLC35B1",
        "MANF",
        "HYOU1",
        "OST4",
        "STT3A",
        "SEC63",
    }
    tmask = df["gene"].isin(trafficking)
    summary["post_hoc_trafficking_genes"] = sorted(trafficking)
    summary["post_hoc_trafficking_n_matched"] = int(tmask.sum())
    summary["post_hoc_trafficking_matched"] = sorted(
        df.loc[tmask, "gene"].unique().tolist()
    )
    if tmask.sum() >= 3 and (~tmask).sum() >= 5:
        ta = df.loc[tmask, "stability"]
        tb = df.loc[~tmask, "stability"]
        tst, tp = mannwhitneyu(ta, tb, alternative="less")
        summary["post_hoc_trafficking_median_sp"] = float(ta.median())
        summary["post_hoc_trafficking_median_other"] = float(tb.median())
        summary["post_hoc_trafficking_mwu_p"] = float(tp)
        summary["post_hoc_trafficking_note"] = (
            "POST-HOC only — not adamson_upr_core.v2. Overlap with core = "
            "MANF, HYOU1. Does not replace the pre-specified PC definition."
        )
    else:
        summary["post_hoc_trafficking_mwu_p"] = None
        summary["post_hoc_trafficking_note"] = "insufficient matches"

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-pilot",
        action="store_true",
        help="Also score adamson_2016_pilot for contrast with the UPR arm",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=cfg.OUTPUT_DIR,
        help="Output directory (default: shesha-crispr/)",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=None,
        help="Path to adamson_2016_upr_perturb_seq.h5ad (skip download)",
    )
    parser.add_argument(
        "--via-pertpy",
        action="store_true",
        help="Load via pertpy.data.adamson_2016_upr_perturb_seq() instead of URL",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ADAMSON UPR SPIKE — frozen pipeline")
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"MIN_CELLS={cfg.MIN_CELLS}  N_PCS={cfg.N_PCS}  SEED={cfg.SEED}")
    print("=" * 72)

    # Download uses Zenodo (scverse CDN often 403s on Colab). Or pass --h5ad / --via-pertpy.
    setup_cache()
    df_upr = run_dataset(
        "Adamson 2016 UPR (CRISPRi)",
        prefer_local=not args.via_pertpy,
        h5ad_path=args.h5ad,
    )
    df_upr = annotate_upr(df_upr)

    csv_path = out_dir / "adamson_upr_sp_scores.csv"
    df_upr.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df_upr)} rows)")

    summary = summarize(df_upr)
    summary_path = out_dir / "adamson_upr_spike_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\n--- RAW Sp (descriptive; magnitude-conditioned test is separate) ---")
    print(f"signal: {summary['venue_signal']}")
    print(summary["venue_note"])
    print(
        f"n={summary['n_perturbations']}  "
        f"upr_core_matched={summary['n_upr_core_matched']}  "
        f"median Sp={summary['median_sp']:.3f}"
    )
    if summary.get("median_sp_upr_core") is not None:
        print(
            f"median Sp UPR-core={summary['median_sp_upr_core']:.3f}  "
            f"other={summary['median_sp_other']:.3f}  "
            f"MWU p={summary.get('mwu_upr_core_lower_p')}"
        )
    if summary.get("post_hoc_trafficking_mwu_p") is not None:
        print(
            f"POST-HOC trafficking list: n={summary['post_hoc_trafficking_n_matched']}  "
            f"med {summary['post_hoc_trafficking_median_sp']:.3f} vs "
            f"{summary['post_hoc_trafficking_median_other']:.3f}  "
            f"MWU p={summary['post_hoc_trafficking_mwu_p']:.5g}  "
            f"({summary['post_hoc_trafficking_note']})"
        )

    if args.compare_pilot:
        print("\n--- PILOT ARM (contrast; not the UPR positive control) ---")
        try:
            df_pilot = run_dataset("Adamson 2016 pilot (CRISPRi)", prefer_local=False)
            pilot_path = out_dir / "adamson_pilot_sp_scores.csv"
            df_pilot.to_csv(pilot_path, index=False)
            print(f"Wrote {pilot_path} ({len(df_pilot)} rows)")
        except Exception as e:
            print(f"Pilot arm skipped (needs pertpy): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
