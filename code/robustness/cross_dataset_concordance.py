#!/usr/bin/env python3
"""
Cross-dataset Sp concordance.

Match gene names between Norman (CRISPRa) and Replogle (CRISPRi) for shared
single-gene perturbations and compare Sp. Low n / low concordance is an honest
result that supports gene × context framing (not gene-intrinsic Sp).

Usage:
  python cross_dataset_concordance.py \\
      --input shesha-crispr/frozen_sp_scores.csv
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
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import pipeline_config as cfg
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir

NORMAN = "Norman 2019 (CRISPRa)"
REPLOGLE = "Replogle 2022 (CRISPRi)"


def is_single_gene(pert: str) -> bool:
    s = str(pert)
    if "+" in s or ";" in s:
        return False
    parts = s.replace("/", "_").split("_")
    parts = [p for p in parts if p and p.upper() not in ("CTRL", "CONTROL", "NT")]
    return len(parts) == 1


def normalize_gene(pert: str) -> str:
    s = str(pert).upper().strip()
    s = s.split("+")[0].split(";")[0]
    return s.split("_")[0]


def collapse(sub: pd.DataFrame) -> pd.DataFrame:
    aggs = {
        "stability": ("stability", "mean"),
        "magnitude": ("magnitude", "mean"),
        "perturbation": ("perturbation", "first"),
    }
    if "n_cells" in sub.columns:
        aggs["n_cells"] = ("n_cells", "sum")
    return sub.groupby("gene_key", as_index=False).agg(**aggs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs=2,
        default=[NORMAN, REPLOGLE],
        help="Two dataset display names to compare",
    )
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    path = find_sp_csv(out_dir, args.input)
    df = load_sp_table(path)
    ds_a, ds_b = [cfg.resolve_dataset_name(d) for d in args.datasets]

    a = df[df["dataset"] == ds_a].copy()
    b = df[df["dataset"] == ds_b].copy()
    if a.empty or b.empty:
        raise SystemExit(
            f"Need both datasets in {path}. Have: {sorted(df['dataset'].unique())}"
        )

    a["gene_key"] = a["perturbation"].map(normalize_gene)
    b["gene_key"] = b["perturbation"].map(normalize_gene)
    a_s = a[a["perturbation"].map(is_single_gene)]
    b_s = b[b["perturbation"].map(is_single_gene)]

    ca, cb = collapse(a_s), collapse(b_s)
    merged = ca.merge(cb, on="gene_key", suffixes=("_a", "_b"))

    summary = {
        "dataset_a": ds_a,
        "dataset_b": ds_b,
        "n_a_single": int(len(ca)),
        "n_b_single": int(len(cb)),
        "n_shared_single_gene": int(len(merged)),
        "shared_genes": sorted(merged["gene_key"].tolist()),
        "config_version": cfg.CONFIG_VERSION,
        "source_csv": str(path),
    }

    if len(merged) >= 3:
        rho, p = spearmanr(merged["stability_a"], merged["stability_b"])
        summary["spearman_sp"] = float(rho)
        summary["spearman_p"] = float(p)
        rho_m, p_m = spearmanr(merged["magnitude_a"], merged["magnitude_b"])
        summary["spearman_magnitude"] = float(rho_m)
        summary["spearman_magnitude_p"] = float(p_m)
    else:
        summary["spearman_sp"] = None
        summary["spearman_p"] = None
        summary["note"] = (
            f"Only {len(merged)} shared single-gene perturbations — "
            "report n and gene list; do not over-interpret concordance."
        )

    print(f"Shared single-gene perturbations: {len(merged)}")
    print(f"  genes: {summary['shared_genes']}")
    if summary.get("spearman_sp") is not None:
        print(f"  Sp Spearman: {summary['spearman_sp']:.3f} (p={summary['spearman_p']:.3g})")
    else:
        print(f"  {summary.get('note')}")

    blurb = (
        f"Cross-dataset concordance ({ds_a} vs {ds_b}). "
        f"After matching gene symbols for single-gene perturbations, "
        f"n={len(merged)} overlapping genes"
        + (f" ({', '.join(summary['shared_genes'])})" if summary["shared_genes"] else "")
        + ". "
    )
    if summary.get("spearman_sp") is not None:
        blurb += (
            f"Sp concordance Spearman ρ={summary['spearman_sp']:.2f} "
            f"(p={summary['spearman_p']:.3g}). "
        )
    blurb += (
        "Low overlap and/or low concordance is expected under gene × context "
        "(CRISPRa vs CRISPRi; different library composition) and does not imply "
        "that Sp is a gene-intrinsic property."
    )
    summary["methods_blurb"] = blurb

    if len(merged):
        out = merged.rename(
            columns={
                "stability_a": "sp_a",
                "stability_b": "sp_b",
                "magnitude_a": "magnitude_a",
                "magnitude_b": "magnitude_b",
                "perturbation_a": "perturbation_a",
                "perturbation_b": "perturbation_b",
            }
        )
        out.insert(0, "dataset_a", ds_a)
        out.insert(1, "dataset_b", ds_b)
        out.to_csv(out_dir / "cross_dataset_concordance_pairs.csv", index=False)

    with open(out_dir / "cross_dataset_concordance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "cross_dataset_concordance_blurb.txt", "w") as f:
        f.write(blurb + "\n")

    print(f"Wrote {out_dir}/cross_dataset_concordance_summary.json")


if __name__ == "__main__":
    main()
