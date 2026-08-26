#!/usr/bin/env python3
"""
UPR gene-exclusion sensitivity.

Remove Hallmark UPR genes from the feature set, recompute PCA + Sp, then
compare Sp–stress partial correlations to the baseline (shared-feature confound).

Usage:
  python upr_gene_exclusion.py --datasets "Replogle 2022 (CRISPRi)" "Dixit 2016 (CRISPR-KO)"
  python upr_gene_exclusion.py --datasets "Adamson 2016 UPR (CRISPRi)" \\
      --adamson-h5ad /tmp/pertpy_data/adamson_2016_upr_perturb_seq.h5ad
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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    _hvg_subsampled,
    _log1p_inplace,
    _looks_log_normalized,
    _normalize_total_numpy,
    _pca_truncated_svd,
    calculate_sp,
    ensure_in_memory,
    load_raw,
    materialize_min_cells,
    setup_cache,
)
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import bootstrap_partial_spearman_ci

# Same curated UPR set as pathway_analysis.py
UPR_GENES = {
    "HSPA5", "HSP90B1", "HYOU1", "CALR", "CANX", "P4HB", "PDIA3",
    "PDIA4", "PDIA5", "PDIA6", "PPIB", "ERP29", "ERP44", "SIL1",
    "FKBP14", "DNAJB9", "DNAJB11", "DNAJC3", "DNAJC10",
    "ATF6", "ATF6B", "ERN1", "EIF2AK3", "XBP1", "DDIT3", "CREB3L2",
    "EDEM1", "DERL1", "OS9", "SEL1L", "SYVN1", "UBE2J1", "UBE2D1",
    "VIMP", "YOD1", "VCP",
    "SEC61A1", "SEC61B", "SEC11C", "SEC24D", "TRAM1", "SRPRB",
    "SPCS1", "SPCS2", "SPCS3", "SSR1", "SSR3", "SSR4",
    "LMAN1", "GOSR2", "KDELR3", "SURF4",
    "DDOST", "STT3A", "STT3B", "RPN1", "RPN2", "MOGS", "UGGT1", "SRD5A3",
    "HERPUD1", "MANF", "CRELD2", "SDF2L1", "NUCB1", "RCN1",
    "SERP1", "WIPI1", "UFM1", "BAX", "ERO1A", "MBTPS1", "MBTPS2",
    "ARCN1", "PREB", "GANAB", "TMX1", "ERLEC1",
}
STRESS_MARKERS = ["DDIT3", "ATF4", "XBP1", "HSPA5"]


def _score_sp(adata, pert_col, ctrl_label, exclude_upr: bool):
    """Normalize → (optional UPR drop) → HVG → SVD → Sp table."""
    ad = ensure_in_memory(adata)
    if not _looks_log_normalized(ad):
        _normalize_total_numpy(ad, 1e4)
        _log1p_inplace(ad)

    if exclude_upr:
        keep = [g for g in ad.var_names if str(g).upper() not in UPR_GENES]
        n_drop = ad.n_vars - len(keep)
        print(f"    excluding {n_drop} UPR genes → {len(keep)} remain", flush=True)
        ad = ensure_in_memory(ad[:, keep])

    if ad.n_vars > cfg.N_HVG:
        ad = _hvg_subsampled(ad, cfg.N_HVG, cfg.SEED)
    ad = _pca_truncated_svd(ad, cfg.N_PCS, cfg.SEED)

    ctrl_mask = ad.obs[pert_col].astype(str) == ctrl_label
    X_ctrl = np.asarray(ad.obsm["X_pca"][ctrl_mask])
    rows = []
    for pert in ad.obs[pert_col].astype(str).unique():
        if pert == ctrl_label:
            continue
        X_pert = np.asarray(ad.obsm["X_pca"][ad.obs[pert_col].astype(str) == pert])
        if X_pert.shape[0] < cfg.MIN_CELLS:
            continue
        m = calculate_sp(X_ctrl, X_pert)
        if m["magnitude"] <= 0:
            continue
        rows.append({
            "perturbation": pert,
            "gene": pert.upper().split("_")[0],
            "stability": m["stability"],
            "magnitude": m["magnitude"],
            "n_cells": X_pert.shape[0],
            "exclude_upr": exclude_upr,
        })
    return pd.DataFrame(rows), ad


def _mean_stress(adata, pert_col, ctrl_label):
    out = {m: {} for m in STRESS_MARKERS}
    labels = adata.obs[pert_col].astype(str)
    var_upper = {str(g).upper(): g for g in adata.var_names}
    for marker in STRESS_MARKERS:
        gene = var_upper.get(marker)
        if gene is None:
            continue
        for pert, idx in labels.groupby(labels).groups.items():
            if pert == ctrl_label:
                continue
            val = adata[idx, gene].X.mean()
            if hasattr(val, "item"):
                val = val.item()
            out[marker][pert] = float(val)
    return out


def run_dataset(name: str, h5ad_path: Path | None, baseline_df: pd.DataFrame | None):
    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = cfg.CACHE_DIR
    print(f"\n=== {name} ===", flush=True)
    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad_path)
    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    adata, _, _ = materialize_min_cells(adata, pert_col, ctrl_label)
    adata = ensure_in_memory(adata)

    # stress from full gene set (before UPR drop) on a log-normalized copy
    ad_stress = ensure_in_memory(adata)
    if not _looks_log_normalized(ad_stress):
        _normalize_total_numpy(ad_stress, 1e4)
        _log1p_inplace(ad_stress)
    stress = _mean_stress(ad_stress, pert_col, ctrl_label)

    df_base, _ = _score_sp(adata, pert_col, ctrl_label, exclude_upr=False)
    df_excl, _ = _score_sp(adata, pert_col, ctrl_label, exclude_upr=True)

    for d in (df_base, df_excl):
        d["dataset"] = name
        for marker, mapping in stress.items():
            d[f"stress_{marker}"] = d["perturbation"].map(mapping)

    # Sp concordance baseline vs exclusion
    both = df_base.merge(
        df_excl[["perturbation", "stability", "magnitude"]],
        on="perturbation",
        suffixes=("_base", "_excl"),
    )
    rho_sp, p_sp = spearmanr(both["stability_base"], both["stability_excl"])
    print(f"  Sp concordance base vs UPR-excl: rho={rho_sp:.3f} p={p_sp:.2e} n={len(both)}")

    partial_rows = []
    for tag, d in [("baseline", df_base), ("upr_excluded", df_excl)]:
        for marker in STRESS_MARKERS:
            col = f"stress_{marker}"
            if col not in d.columns or d[col].notna().sum() < 15:
                continue
            m = d[["stability", "magnitude", col]].dropna()
            if len(m) < 15:
                continue
            part = bootstrap_partial_spearman_ci(
                m["stability"], m[col], m["magnitude"],
                n_bootstrap=2000, seed=cfg.SEED, method="rank",
            )
            print(
                f"  [{tag}] Sp~{marker}|mag: rho={part['rho_partial']:+.3f} "
                f"[{part['ci_low']:.3f},{part['ci_high']:.3f}] n={part['n']}"
            )
            partial_rows.append({
                "dataset": name,
                "condition": tag,
                "marker": marker,
                "n": part["n"],
                "rho_partial": part["rho_partial"],
                "ci_low": part["ci_low"],
                "ci_high": part["ci_high"],
                "p": part["p"],
                "config_version": cfg.CONFIG_VERSION,
            })

    summary = {
        "dataset": name,
        "n_base": int(len(df_base)),
        "n_excl": int(len(df_excl)),
        "sp_rank_rho_base_vs_excl": float(rho_sp),
        "sp_rank_p": float(p_sp),
        "config_version": cfg.CONFIG_VERSION,
    }
    return df_base, df_excl, pd.DataFrame(partial_rows), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Replogle 2022 (CRISPRi)"],
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--adamson-h5ad", type=Path, default=None)
    parser.add_argument("--baseline-csv", type=Path, default=None,
                        help="Optional frozen Sp CSV for reference merge")
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    baseline = None
    if args.baseline_csv or (out_dir / "frozen_sp_scores.csv").exists():
        try:
            baseline = load_sp_table(find_sp_csv(out_dir, args.baseline_csv))
        except FileNotFoundError:
            pass

    all_partial = []
    summaries = []
    for name in args.datasets:
        name = cfg.resolve_dataset_name(name)
        h5ad = args.adamson_h5ad if "Adamson" in name and "UPR" in name else None
        try:
            base, excl, partial, summary = run_dataset(name, h5ad, baseline)
            base.to_csv(out_dir / f"upr_excl_sp_baseline_{_tag(name)}.csv", index=False)
            excl.to_csv(out_dir / f"upr_excl_sp_excluded_{_tag(name)}.csv", index=False)
            all_partial.append(partial)
            summaries.append(summary)
        except Exception as e:
            print(f"FAILED {name}: {e}")
            raise

    if all_partial:
        pd.concat(all_partial, ignore_index=True).to_csv(
            out_dir / "upr_exclusion_partial_correlations.csv", index=False
        )
    with open(out_dir / "upr_exclusion_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote results under {out_dir}")


def _tag(name: str) -> str:
    return name.split("(")[0].strip().lower().replace(" ", "_")


if __name__ == "__main__":
    main()
