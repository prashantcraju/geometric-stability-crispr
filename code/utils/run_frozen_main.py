#!/usr/bin/env python3
"""
Regenerate main Sp tables under the frozen pipeline.

Scores every in_main dataset from pipeline_config with one shared
MIN_CELLS / N_PCS / SEED, writes:

  shesha-crispr/frozen_sp_scores.csv
  shesha-crispr/frozen_sp_summary.json

Usage (Colab / local with scanpy + data access):
  python utils/run_frozen_main.py
  python utils/run_frozen_main.py --skip-fail
  python utils/run_frozen_main.py --datasets "Norman 2019 (CRISPRa)" "Dixit 2016 (CRISPR-KO)"
  python utils/run_frozen_main.py --adamson-h5ad /tmp/pertpy_data/adamson_2016_upr_perturb_seq.h5ad

Downloads Figshare/Zenodo h5ad files into /tmp/pertpy_data (no pertpy required).
First Norman download can take a few minutes — watch for "… XX MB" progress lines.
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

import pandas as pd
from scipy.stats import spearmanr

import pipeline_config as cfg
from pipeline_core import compute_sp_digest, run_dataset, setup_cache
from stats_utils import bootstrap_spearman_ci


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "config_version": cfg.CONFIG_VERSION,
        "min_cells": cfg.MIN_CELLS,
        "n_pcs": cfg.N_PCS,
        "seed": cfg.SEED,
        "n_rows": int(len(df)),
        "sp_digest": compute_sp_digest(df),
        "datasets": {},
    }
    for ds, sub in df.groupby("dataset"):
        rho, p = spearmanr(sub["magnitude"], sub["stability"])
        boot = bootstrap_spearman_ci(
            sub["stability"], sub["magnitude"], n_bootstrap=min(2000, cfg.N_BOOTSTRAP)
        )
        out["datasets"][ds] = {
            "n_perturbations": int(len(sub)),
            "modality": sub["modality"].iloc[0] if "modality" in sub.columns else None,
            "cell_type": sub["cell_type"].iloc[0] if "cell_type" in sub.columns else None,
            "median_sp": float(sub["stability"].median()),
            "spearman_mag_sp": float(rho),
            "spearman_mag_sp_p": float(p),
            "spearman_mag_sp_ci": [boot["ci_low"], boot["ci_high"]],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of frozen display names (default: all in_main)",
    )
    parser.add_argument(
        "--adamson-h5ad",
        type=Path,
        default=None,
        help="Local Adamson UPR h5ad (avoids download)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=cfg.OUTPUT_DIR,
    )
    parser.add_argument(
        "--skip-fail",
        action="store_true",
        help="Continue if one dataset fails (default: abort)",
    )
    parser.add_argument(
        "--max-perts",
        type=int,
        default=None,
        help="Randomly keep at most this many perturbations per dataset. "
        "Use ~200 for Colab synthetic-benchmark reference selection. "
        "Omit for manuscript-frozen tables.",
    )
    parser.add_argument(
        "--max-control-cells",
        type=int,
        default=None,
        help="Cap control cells (default: pipeline_config.MAX_CONTROL_CELLS). "
        "1000 is enough for Sp centroids on Colab.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Output CSV basename without .csv (default: frozen_sp_scores, or "
        "frozen_sp_scores_sample if --max-perts is set)",
    )
    args = parser.parse_args()

    names = args.datasets or cfg.main_dataset_names()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_ctrl = (
        args.max_control_cells
        if args.max_control_cells is not None
        else cfg.MAX_CONTROL_CELLS
    )
    if args.out_name:
        out_stem = args.out_name
    elif args.max_perts is not None:
        out_stem = "frozen_sp_scores_sample"
    else:
        out_stem = "frozen_sp_scores"

    print("=" * 72)
    print("FROZEN MAIN Sp PIPELINE")
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"MIN_CELLS={cfg.MIN_CELLS}  N_PCS={cfg.N_PCS}  SEED={cfg.SEED}")
    print(f"datasets={names}")
    if args.max_perts is not None:
        print(
            f"max_perts={args.max_perts}  max_control_cells={max_ctrl}  "
            "(SAMPLE — not manuscript-frozen)"
        )
    print("=" * 72)

    setup_cache()
    frames = []
    errors = {}

    for name in names:
        try:
            print(f"\n[{len(frames)+1}/{len(names)}] starting {name}…", flush=True)
            h5ad = args.adamson_h5ad if "UPR" in name and "Adamson" in name else None
            # Always use Figshare/Zenodo cache when configured (avoids silent pertpy hang)
            prefer_local = bool(cfg.DATASETS[name].get("local_h5ad"))
            df = run_dataset(
                name,
                prefer_local=prefer_local,
                h5ad_path=h5ad,
                max_perts=args.max_perts,
                max_control_cells=max_ctrl,
            )
            frames.append(df)
            print(f"[{len(frames)}/{len(names)}] done {name}: {len(df)} perturbations", flush=True)
        except Exception as e:
            errors[name] = str(e)
            print(f"    FAILED: {e}", flush=True)
            if not args.skip_fail:
                raise

    if not frames:
        raise RuntimeError(f"No datasets scored. Errors: {errors}")
    if errors and not args.skip_fail:
        raise RuntimeError(f"Dataset failures (refusing partial freeze): {errors}")
    # Full freeze must include every in_main dataset (Replogle truncation → abort)
    if args.max_perts is None and args.datasets is None:
        got = {cfg.resolve_dataset_name(d) for d in pd.concat(frames)["dataset"].unique()}
        need = set(cfg.main_dataset_names())
        missing = sorted(need - got)
        if missing:
            raise RuntimeError(
                f"Partial freeze — missing {missing}. "
                "Delete truncated cache under /tmp/pertpy_data and re-run."
            )

    all_df = pd.concat(frames, ignore_index=True)
    digest = compute_sp_digest(all_df)
    all_df["sp_digest"] = digest
    all_df["config_version"] = cfg.CONFIG_VERSION
    csv_path = out_dir / f"{out_stem}.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(all_df)} rows; digest={digest})")

    # also write euclidean-named alias expected by some fig scripts
    # (only when this is the full frozen table — don't overwrite with a sample)
    if args.max_perts is None:
        alias = out_dir / "shesha_crispr_results_euclidean.csv"
        all_df.to_csv(alias, index=False)
        print(f"Wrote {alias} (alias for fig scripts)")
    else:
        print(
            "NOTE: sample run — not writing shesha_crispr_results_euclidean.csv. "
            "Pass this CSV to synthetic_benchmark.py via --sp-csv."
        )

    summary = summarize(all_df)
    if errors:
        summary["errors"] = errors
    if args.max_perts is not None:
        summary["sample"] = {
            "max_perts": args.max_perts,
            "max_control_cells": max_ctrl,
            "note": "Not manuscript-frozen; for synthetic_benchmark reference selection",
        }
    summary_path = out_dir / f"{out_stem}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\n--- per-dataset Sp~magnitude ---")
    for ds, info in summary["datasets"].items():
        print(
            f"  {ds}: n={info['n_perturbations']}  "
            f"rho={info['spearman_mag_sp']:.3f}  "
            f"median_Sp={info['median_sp']:.3f}"
        )

    print("\nNext:")
    print("  python remap_modality_labels.py --apply --csv-dir shesha-crispr")
    print("    # (--out-dir is an accepted alias for --csv-dir)")
    print("  python recompute_partial_correlations.py --input shesha-crispr/frozen_sp_scores.csv")
    print("  python attach_stress_markers.py --input shesha-crispr/frozen_sp_scores.csv")
    print("  python stress_marker_tests.py --input shesha-crispr/shesha_crispr_results_euclidean.csv")


if __name__ == "__main__":
    main()
