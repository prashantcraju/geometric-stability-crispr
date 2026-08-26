#!/usr/bin/env python3
"""
Papalexi perturbation audit.

Document which perturbations are included in the frozen pipeline (gene_target
labels, cell counts, NT control) and how they were identified. The pertpy
docs list fewer gRNAs than the scored gene set.

Usage:
  python papalexi_perturbation_audit.py
  python papalexi_perturbation_audit.py --h5ad /tmp/pertpy_data/papalexi_2021.h5mu
  python papalexi_perturbation_audit.py --input shesha-crispr/frozen_sp_scores.csv
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

import pandas as pd

import pipeline_config as cfg
from pipeline_core import _extract_adata, load_raw, setup_cache
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir

DATASET = "Papalexi 2021 (CRISPR-KO)"

# Common pertpy / ECCITE-seq documentation lists a smaller guide set;
# we record what the frozen pipeline actually scores from gene_target.
PERTPY_DOC_HINT = (
    "pertpy documentation historically lists a small ECCITE-seq gRNA panel "
    "(~11 guides). The scPerturb / MuData object exposes gene_target labels "
    "that can include additional targets and NT; this audit reports the "
    "empirical label set after MIN_CELLS filtering."
)


def audit_from_mudata(h5ad: Path | None) -> dict:
    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = cfg.CACHE_DIR
    print(f"Loading {DATASET} …", flush=True)
    raw = load_raw(DATASET, prefer_local=True, h5ad_path=h5ad)
    adata, pert_col, ctrl_label = _extract_adata(raw, DATASET, sc)
    labels = adata.obs[pert_col].astype(str)
    counts = labels.value_counts().sort_values(ascending=False)

    # Guide-like columns if present on MuData or RNA
    guide_cols = [
        c for c in list(getattr(raw, "obs", pd.DataFrame()).columns)
        + list(adata.obs.columns)
        if any(k in c.lower() for k in ("guide", "sgrna", "grna", "barcode", "hto"))
    ]
    guide_cols = sorted(set(guide_cols))

    valid = [p for p in counts.index if p != ctrl_label and counts[p] >= cfg.MIN_CELLS]
    rows = []
    for p, n in counts.items():
        rows.append({
            "label": p,
            "n_cells": int(n),
            "is_control": p == ctrl_label,
            "passes_min_cells": bool(p == ctrl_label or n >= cfg.MIN_CELLS),
            "included_in_frozen_sp": bool(p != ctrl_label and n >= cfg.MIN_CELLS),
        })

    return {
        "pert_col": pert_col,
        "ctrl_label": ctrl_label,
        "n_obs": int(adata.n_obs),
        "n_unique_labels": int(labels.nunique()),
        "n_included_perturbations": int(len(valid)),
        "included_perturbations": valid,
        "all_label_counts": rows,
        "guide_like_columns": guide_cols,
        "min_cells": cfg.MIN_CELLS,
    }


def audit_from_csv(path: Path) -> dict:
    df = load_sp_table(path)
    sub = df[df["dataset"] == DATASET]
    if sub.empty:
        # try legacy name
        sub = df[df["dataset"].astype(str).str.contains("Papalexi", case=False)]
    return {
        "n_scored_in_csv": int(len(sub)),
        "perturbations_in_csv": sorted(sub["perturbation"].astype(str).tolist()) if len(sub) else [],
        "csv_path": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None,
                        help="Optional frozen Sp CSV to cross-check n")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-load", action="store_true",
                        help="Only audit from CSV (no MuData load)")
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    report = {
        "dataset": DATASET,
        "modality": cfg.DATASETS[DATASET]["modality"],
        "cell_type": cfg.DATASETS[DATASET]["cell_type"],
        "config_version": cfg.CONFIG_VERSION,
        "pertpy_doc_note": PERTPY_DOC_HINT,
        "identification_method": (
            "Load Papalexi MuData via pipeline_core.load_raw; extract RNA modality; "
            "use raw.obs['gene_target'] as perturbation labels; control label 'NT'; "
            f"score perturbations with ≥{cfg.MIN_CELLS} cells (frozen MIN_CELLS)."
        ),
    }

    if not args.skip_load:
        mud = audit_from_mudata(args.h5ad)
        report.update(mud)
        pd.DataFrame(mud["all_label_counts"]).to_csv(
            out_dir / "papalexi_label_counts.csv", index=False
        )
        print(
            f"  Labels: {mud['n_unique_labels']} unique; "
            f"{mud['n_included_perturbations']} pass MIN_CELLS={cfg.MIN_CELLS}"
        )
        print(f"  Included: {mud['included_perturbations']}")

    try:
        csv_path = find_sp_csv(out_dir, args.input)
        csv_info = audit_from_csv(csv_path)
        report["frozen_csv"] = csv_info
        print(f"  Frozen CSV scored n={csv_info['n_scored_in_csv']}")
        if report.get("included_perturbations") and csv_info["perturbations_in_csv"]:
            a = set(report["included_perturbations"])
            b = set(csv_info["perturbations_in_csv"])
            report["csv_vs_mudata"] = {
                "only_in_mudata_filter": sorted(a - b),
                "only_in_csv": sorted(b - a),
                "intersection_n": len(a & b),
            }
    except FileNotFoundError as e:
        report["frozen_csv"] = {"error": str(e)}

    n = report.get("n_included_perturbations") or report.get("frozen_csv", {}).get("n_scored_in_csv")
    blurb = (
        f"Papalexi perturbation audit. Perturbations are identified from the "
        f"MuData obs column gene_target (control: NT). After applying the frozen "
        f"MIN_CELLS={cfg.MIN_CELLS} filter, n={n} perturbations are scored. "
        f"This can exceed the smaller gRNA panel listed in some pertpy summaries "
        f"because gene_target aggregates / includes all target labels present in "
        f"the distributed object. Full label counts: papalexi_label_counts.csv."
    )
    report["methods_blurb"] = blurb

    with open(out_dir / "papalexi_perturbation_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "papalexi_perturbation_audit_blurb.txt", "w") as f:
        f.write(blurb + "\n")
    if report.get("included_perturbations"):
        pd.Series(report["included_perturbations"], name="perturbation").to_csv(
            out_dir / "papalexi_included_perturbations.csv", index=False
        )

    print(blurb)
    print(f"Wrote {out_dir}/papalexi_perturbation_audit.json")


if __name__ == "__main__":
    main()
