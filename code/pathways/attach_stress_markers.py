#!/usr/bin/env python3
"""
Attach stress_* mean-expression columns to frozen Sp scores.

run_frozen_main.py writes Sp/magnitude only. stress_marker_tests.py needs
stress_DDIT3 / ATF4 / XBP1 / HSPA5. This script loads each dataset (before HVG
subset), computes mean log-normalized marker expression per perturbation, and
merges onto the Sp table.

Usage:
  python attach_stress_markers.py \\
      --input shesha-crispr/frozen_sp_scores.csv \\
      --out shesha-crispr/shesha_crispr_results_euclidean.csv

  # then
  python stress_marker_tests.py --input shesha-crispr/shesha_crispr_results_euclidean.csv
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    _log1p_inplace,
    _normalize_total_numpy,
    assert_frozen_sp_compatible,
    ensure_in_memory,
    load_raw,
    materialize_min_cells,
    resolve_matrix_is_log,
    setup_cache,
)

STRESS_MARKERS = ["DDIT3", "ATF4", "XBP1", "HSPA5"]


def _resolve_gene(adata, marker: str):
    """Case-insensitive gene match; return var name or None."""
    if marker in adata.var_names:
        return marker
    upper = {str(g).upper(): g for g in adata.var_names}
    return upper.get(marker.upper())


def extract_stress_markers(adata, pert_col: str, ctrl_label: str, markers=None) -> dict:
    """Mean expression per perturbation for each marker (pre-HVG matrix)."""
    if markers is None:
        markers = STRESS_MARKERS
    out = {}
    for marker in markers:
        gene = _resolve_gene(adata, marker)
        if gene is None:
            print(f"    {marker}: not in var_names — skip")
            continue
        expr = {}
        # vectorized-ish loop over perturbations present
        labels = adata.obs[pert_col].astype(str)
        for pert, idx in labels.groupby(labels).groups.items():
            if pert == ctrl_label:
                continue
            X = adata[idx, gene].X
            val = X.mean()
            if hasattr(val, "item"):
                val = val.item()
            expr[str(pert)] = float(val)
        out[marker] = expr
        print(f"    {marker} ({gene}): {len(expr)} perturbations")
    return out


def load_normalized_for_stress(dataset_name: str, h5ad_path: Path | None = None):
    """Load dataset, materialize (downsampled), normalize/log if needed — keep all genes."""
    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = Path(os.environ.get("SCVERSE_DATADIR", str(cfg.CACHE_DIR)))

    print(f"\n>>> stress extract: {dataset_name}", flush=True)
    raw = load_raw(dataset_name, prefer_local=True, h5ad_path=h5ad_path)
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset_name, sc)
    # Downsample like frozen Sp path so Replogle fits in RAM
    adata, _, _ = materialize_min_cells(adata, pert_col, ctrl_label)
    adata = ensure_in_memory(adata)

    already_log, log_src = resolve_matrix_is_log(
        dataset_name=dataset_name, adata=adata
    )
    if already_log:
        print(
            f"    skip normalize/log1p (matrix_is_log=True via {log_src})",
            flush=True,
        )
    else:
        print(
            f"    normalize_total + log1p (matrix_is_log=False via {log_src}; "
            "all genes retained)",
            flush=True,
        )
        _normalize_total_numpy(adata, 1e4)
        _log1p_inplace(adata)

    return adata, pert_col, ctrl_label


def attach(
    df: pd.DataFrame,
    adamson_h5ad: Path | None = None,
    skip_fail: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    for m in STRESS_MARKERS:
        col = f"stress_{m}"
        if col not in out.columns:
            out[col] = np.nan

    datasets = sorted(out["dataset"].dropna().unique())
    errors = {}
    for ds in datasets:
        try:
            h5ad = adamson_h5ad if "Adamson" in ds and "UPR" in ds else None
            # map legacy names
            name = cfg.resolve_dataset_name(ds)
            if name not in cfg.DATASETS:
                raise KeyError(f"unknown dataset key {ds!r}")
            adata, pert_col, ctrl_label = load_normalized_for_stress(name, h5ad_path=h5ad)
            stress = extract_stress_markers(adata, pert_col, ctrl_label)
            del adata

            mask = out["dataset"].astype(str) == str(ds)
            for marker, mapping in stress.items():
                col = f"stress_{marker}"
                perts = out.loc[mask, "perturbation"].astype(str)
                out.loc[mask, col] = perts.map(mapping)
            n_hit = out.loc[mask, [f"stress_{m}" for m in stress]].notna().any(axis=1).sum()
            print(f"    merged stress into {n_hit} / {mask.sum()} rows for {ds}")
        except Exception as e:
            errors[ds] = str(e)
            print(f"    FAILED {ds}: {e}")
            if not skip_fail:
                raise RuntimeError(
                    f"Stress extract failed for {ds}: {e}\n"
                    "Refusing to write a partial enriched Sp table. "
                    "Delete truncated h5ad cache if size-check failed, then retry. "
                    "Pass --skip-fail only for debugging."
                ) from e
    if errors and not skip_fail:
        raise RuntimeError(f"Stress extract failures: {errors}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Sp scores CSV (default: shesha-crispr/frozen_sp_scores.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV with stress_* columns "
        "(default: shesha-crispr/shesha_crispr_results_euclidean.csv)",
    )
    parser.add_argument(
        "--adamson-h5ad",
        type=Path,
        default=None,
        help="Local Adamson UPR h5ad if not in /tmp/pertpy_data",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset display names",
    )
    parser.add_argument(
        "--skip-fail",
        action="store_true",
        help="Continue after a dataset failure (default: abort; do not write partial)",
    )
    parser.add_argument(
        "--allow-stale-sp",
        action="store_true",
        help="Skip frozen Sp version/n_rows check (dangerous)",
    )
    args = parser.parse_args()

    in_path = args.input or cfg.OUTPUT_DIR / "frozen_sp_scores.csv"
    if not in_path.exists():
        alt = cfg.OUTPUT_DIR / "shesha_crispr_results_euclidean.csv"
        if alt.exists():
            in_path = alt
        else:
            # Colab defaults
            for p in (
                Path("/content/shesha-crispr/frozen_sp_scores.csv"),
                Path("/content/shesha-crispr/shesha_crispr_results_euclidean.csv"),
            ):
                if p.exists():
                    in_path = p
                    break
    if not in_path.exists():
        raise FileNotFoundError(f"No Sp scores CSV found (tried {in_path})")

    out_path = args.out or in_path.parent / "shesha_crispr_results_euclidean.csv"

    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"input={in_path}")
    print(f"out={out_path}")

    if not args.allow_stale_sp:
        assert_frozen_sp_compatible(in_path)

    df = pd.read_csv(in_path)
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)].copy()

    enriched = attach(df, adamson_h5ad=args.adamson_h5ad, skip_fail=args.skip_fail)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)

    available = [m for m in STRESS_MARKERS if enriched[f"stress_{m}"].notna().any()]
    print(f"\nWrote {out_path} ({len(enriched)} rows)")
    print(f"Stress markers with data: {available}")
    print("\nNext:")
    print(f"  python stress_marker_tests.py --input {out_path} --out-dir {out_path.parent}")


if __name__ == "__main__":
    main()
