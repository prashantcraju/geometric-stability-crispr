#!/usr/bin/env python3
"""
Prove bit-identical reproducibility under CONFIG_VERSION (hash-stable downsample).

Default: ALL six in_main datasets, materialize ×2 AND Sp score ×2.
Do not claim "baseline is frozen" unless Sp rescoring was included and passed.

Usage:
  python check_pipeline_reproducibility.py --out-dir shesha-crispr
  python check_pipeline_reproducibility.py --quick          # Papalexi+Adamson UPR only
  python check_pipeline_reproducibility.py --no-rescore-sp  # materialize only (not sufficient)
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
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    load_raw,
    materialize_min_cells,
    run_dataset,
    setup_cache,
)


def _digest_ids(ids) -> str:
    payload = "\n".join(sorted(map(str, ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _digest_frame(df: pd.DataFrame, cols: list[str]) -> str:
    sub = df[cols].copy()
    sub = sub.sort_values(["dataset", "perturbation"]).reset_index(drop=True)
    raw = sub.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def materialize_twice(name: str, h5ad: Path | None = None) -> dict:
    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = Path(os.environ.get("SCVERSE_DATADIR", str(cfg.CACHE_DIR)))
    name = cfg.resolve_dataset_name(name)
    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad)
    adata, pert_col, ctrl = _extract_adata(raw, name, sc)

    a1, v1, _ = materialize_min_cells(adata, pert_col, ctrl, seed=cfg.SEED)
    raw2 = load_raw(name, prefer_local=True, h5ad_path=h5ad)
    adata2, pert_col2, ctrl2 = _extract_adata(raw2, name, sc)
    a2, v2, _ = materialize_min_cells(adata2, pert_col2, ctrl2, seed=cfg.SEED)

    ids1 = set(a1.obs_names.astype(str))
    ids2 = set(a2.obs_names.astype(str))
    ok = ids1 == ids2 and sorted(v1) == sorted(v2)
    return {
        "dataset": name,
        "n_cells": len(ids1),
        "n_perts": len(v1),
        "cell_id_digest": _digest_ids(ids1),
        "match": ok,
        "n_cells_only_in_run1": len(ids1 - ids2),
        "n_cells_only_in_run2": len(ids2 - ids1),
    }


def rescore_twice(name: str, h5ad: Path | None = None) -> dict:
    df1 = run_dataset(name, h5ad_path=h5ad)
    df2 = run_dataset(name, h5ad_path=h5ad)
    cols = ["dataset", "perturbation", "stability", "magnitude"]
    for c in cols:
        if c not in df1.columns or c not in df2.columns:
            return {"dataset": name, "match": False, "error": f"missing {c}"}
    d1 = _digest_frame(df1, cols)
    d2 = _digest_frame(df2, cols)
    a = df1[cols].sort_values(["dataset", "perturbation"]).reset_index(drop=True)
    b = df2[cols].sort_values(["dataset", "perturbation"]).reset_index(drop=True)
    same = a.equals(b)
    return {
        "dataset": name,
        "digest_run1": d1,
        "digest_run2": d2,
        "match": bool(same),
        "max_abs_sp_diff": float(np.max(np.abs(a["stability"] - b["stability"])))
        if len(a) == len(b)
        else np.nan,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--adamson-h5ad", type=Path, default=None)
    p.add_argument(
        "--rescore-sp",
        dest="rescore_sp",
        action="store_true",
        default=True,
        help="Score Sp twice per dataset (default: ON — required to claim frozen)",
    )
    p.add_argument(
        "--no-rescore-sp",
        dest="rescore_sp",
        action="store_false",
        help="Materialize only (insufficient to claim baseline frozen)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only Papalexi + Adamson UPR (smoke; not sufficient for freeze claim)",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir or cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        names = [cfg.resolve_dataset_name(d) for d in args.datasets]
    elif args.quick:
        names = [
            "Papalexi 2021 (CRISPR-KO)",
            "Adamson 2016 UPR (CRISPRi)",
        ]
    else:
        names = cfg.main_dataset_names()

    print(f"config_version={cfg.CONFIG_VERSION}  SEED={cfg.SEED}")
    print(f"datasets ({len(names)})={names}")
    print(f"rescore_sp={args.rescore_sp}  quick={args.quick}")
    if args.quick or not args.rescore_sp:
        print(
            "NOTE: this run is NOT sufficient to claim 'baseline is frozen' "
            "(need all six datasets + --rescore-sp)."
        )
    print()

    mat_rows, sp_rows = [], []
    all_ok = True
    for name in names:
        h5 = args.adamson_h5ad if "UPR" in name else None
        print(f">>> materialize ×2: {name}", flush=True)
        try:
            r = materialize_twice(name, h5)
        except Exception as e:
            r = {"dataset": name, "match": False, "error": str(e)}
        mat_rows.append(r)
        status = "IDENTICAL" if r.get("match") else "MISMATCH"
        print(
            f"    {status}  n_cells={r.get('n_cells')}  "
            f"digest={r.get('cell_id_digest')}  "
            f"delta={r.get('n_cells_only_in_run1')}/{r.get('n_cells_only_in_run2')}"
        )
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
        all_ok = all_ok and bool(r.get("match"))

        if args.rescore_sp:
            print(f">>> Sp score ×2: {name}", flush=True)
            try:
                s = rescore_twice(name, h5)
            except Exception as e:
                s = {"dataset": name, "match": False, "error": str(e)}
            sp_rows.append(s)
            status = "IDENTICAL" if s.get("match") else "MISMATCH"
            print(
                f"    {status}  digests={s.get('digest_run1')}/{s.get('digest_run2')}  "
                f"max|ΔSp|={s.get('max_abs_sp_diff')}"
            )
            if s.get("error"):
                print(f"    ERROR: {s['error']}")
            all_ok = all_ok and bool(s.get("match"))

    full_claim = (
        all_ok
        and args.rescore_sp
        and not args.quick
        and len(names) >= len(cfg.main_dataset_names())
    )
    report = {
        "config_version": cfg.CONFIG_VERSION,
        "seed": cfg.SEED,
        "n_datasets": len(names),
        "rescore_sp": args.rescore_sp,
        "quick": args.quick,
        "materialize": mat_rows,
        "sp_rescore": sp_rows,
        "all_identical": all_ok,
        "baseline_frozen_claim_ok": full_claim,
    }
    out = out_dir / "reproducibility_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    if full_claim:
        print("PASS — bit-identical materialize + Sp on all six. Baseline is frozen.")
    elif all_ok:
        print(
            "PASS on tested scope only — NOT enough to claim baseline frozen "
            "(need all six + Sp rescoring)."
        )
    else:
        print("FAIL — not reproducible yet. Do not write manuscript numbers.")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
