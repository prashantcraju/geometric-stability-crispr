#!/usr/bin/env python3
"""
Cell-count downsampling for Sp reliability.

For Replogle perturbations with ≥ min_available cells (default 500):
  subsample to N ∈ {200, 100, 50, 25, 10}, recompute Sp, report Spearman of
  Sp rankings vs the full-sample Sp and pairwise ICC-style agreement.

Usage:
  python cell_count_downsampling.py \\
      --adamson-h5ad /tmp/pertpy_data/replogle_2022_k562_essential.h5ad  # optional override path
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
    setup_cache,
)
from revision_io import resolve_out_dir

LEVELS = [200, 100, 50, 25, 10]


def preprocess_embedding(adata, pert_col, ctrl_label, seed=cfg.SEED):
    ad = ensure_in_memory(adata)
    if not _looks_log_normalized(ad):
        print("    normalize + log1p…", flush=True)
        _normalize_total_numpy(ad, 1e4)
        _log1p_inplace(ad)
    # keep all perts with enough cells for the largest level
    counts = ad.obs[pert_col].astype(str).value_counts()
    valid = [p for p in counts[counts >= max(LEVELS)].index if p != ctrl_label]
    keep = valid + [ctrl_label]
    ad = ensure_in_memory(ad[ad.obs[pert_col].astype(str).isin(keep)])
    print(f"    {len(valid)} perturbations with ≥{max(LEVELS)} cells", flush=True)
    if ad.n_vars > cfg.N_HVG:
        ad = ensure_in_memory(_hvg_subsampled(ad, cfg.N_HVG, seed))
    ad = _pca_truncated_svd(ad, cfg.N_PCS, seed)
    return ad, valid


def sp_for_level(X_ctrl_full, X_pert_full, n, rng):
    if X_pert_full.shape[0] < n:
        return np.nan
    idx = rng.choice(X_pert_full.shape[0], size=n, replace=False)
    # optionally downsample controls too for fairness
    n_ctrl = min(len(X_ctrl_full), max(n, cfg.MIN_CONTROL_CELLS))
    cidx = rng.choice(len(X_ctrl_full), size=n_ctrl, replace=False)
    return calculate_sp(X_ctrl_full[cidx], X_pert_full[idx])["stability"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Replogle 2022 (CRISPRi)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--h5ad", type=Path, default=None,
                        help="Local Replogle h5ad (default: cache)")
    parser.add_argument("--min-available", type=int, default=500)
    parser.add_argument("--n-perts", type=int, default=80,
                        help="Max perturbations to include (random among eligible)")
    parser.add_argument("--n-reps", type=int, default=5,
                        help="Downsample replicates per level")
    parser.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    name = cfg.resolve_dataset_name(args.dataset)
    levels = sorted(args.levels, reverse=True)

    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = cfg.CACHE_DIR
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"dataset={name} levels={levels}")

    # Prefer local cache path for Replogle
    h5ad = args.h5ad
    if h5ad is None:
        cand = cfg.CACHE_DIR / "replogle_2022_k562_essential.h5ad"
        if cand.exists():
            h5ad = cand

    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad)
    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    # Do NOT use materialize_min_cells cap — we need high-n perts
    # Filter obs-only to control + perts with >= min_available, then materialize
    labels = adata.obs[pert_col].astype(str)
    counts = labels.value_counts()
    eligible = [p for p in counts[counts >= args.min_available].index if p != ctrl_label]
    rng = np.random.default_rng(cfg.SEED)
    if len(eligible) > args.n_perts:
        eligible = list(rng.choice(eligible, size=args.n_perts, replace=False))
    keep = set(eligible) | {ctrl_label}
    # Cap control cells for memory
    ctrl_idx = np.flatnonzero(labels.to_numpy() == ctrl_label)
    if len(ctrl_idx) > cfg.MAX_CONTROL_CELLS:
        ctrl_idx = rng.choice(ctrl_idx, size=cfg.MAX_CONTROL_CELLS, replace=False)
    pert_idx = np.flatnonzero(labels.isin(eligible).to_numpy())
    idx = np.sort(np.concatenate([ctrl_idx, pert_idx]))
    print(f"  Materializing {len(idx)} cells ({len(eligible)} perts)…", flush=True)
    adata = ensure_in_memory(adata[idx])
    try:
        if hasattr(raw, "file") and raw.file is not None:
            raw.file.close()
    except Exception:
        pass

    adata, valid = preprocess_embedding(adata, pert_col, ctrl_label)
    # intersect with eligible
    valid = [p for p in valid if p in set(eligible)]
    print(f"  Scoring {len(valid)} perturbations × {args.n_reps} reps × {levels}", flush=True)

    ctrl_mask = adata.obs[pert_col].astype(str) == ctrl_label
    X_ctrl = np.asarray(adata.obsm["X_pca"][ctrl_mask])

    # Full Sp (all available cells for each pert)
    full_sp = {}
    X_pert_cache = {}
    for pert in valid:
        Xp = np.asarray(adata.obsm["X_pca"][adata.obs[pert_col].astype(str) == pert])
        X_pert_cache[pert] = Xp
        full_sp[pert] = calculate_sp(X_ctrl, Xp)["stability"]

    rows = []
    for pert in valid:
        Xp = X_pert_cache[pert]
        rows.append({
            "perturbation": pert, "n_level": "full", "rep": 0,
            "n_cells_used": int(Xp.shape[0]), "stability": full_sp[pert],
        })
        for n in levels:
            for rep in range(args.n_reps):
                rng_i = np.random.default_rng(cfg.SEED + 17 * n + rep + hash(pert) % 10000)
                sp = sp_for_level(X_ctrl, Xp, n, rng_i)
                rows.append({
                    "perturbation": pert, "n_level": n, "rep": rep,
                    "n_cells_used": n, "stability": sp,
                })

    long = pd.DataFrame(rows)
    long["dataset"] = name
    long["config_version"] = cfg.CONFIG_VERSION
    long_path = out_dir / "downsampling_sp_long.csv"
    long.to_csv(long_path, index=False)

    # Mean Sp per pert × level; rank correlations vs full
    mean_sp = (
        long.groupby(["perturbation", "n_level"], as_index=False)["stability"].mean()
    )
    wide = mean_sp.pivot(index="perturbation", columns="n_level", values="stability")
    if "full" in wide.columns:
        full = wide["full"]
    else:
        full = pd.Series(full_sp)

    rank_rows = []
    for n in levels:
        if n not in wide.columns:
            continue
        m = pd.DataFrame({"full": full, "down": wide[n]}).dropna()
        rho, p = spearmanr(m["full"], m["down"])
        # simple ICC(consistency): corr of reps — use mean across reps already
        print(f"  n={n}: Spearman(full, down)={rho:.3f} p={p:.2e} n_perts={len(m)}")
        rank_rows.append({
            "n_level": n, "spearman_vs_full": rho, "p": p, "n_perts": len(m),
            "dataset": name, "config_version": cfg.CONFIG_VERSION,
        })

    # Pairwise Spearman between levels
    pair_rows = []
    cols = [c for c in wide.columns if c != "full"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            m = wide[[a, b]].dropna()
            rho, p = spearmanr(m[a], m[b])
            pair_rows.append({"level_a": a, "level_b": b, "spearman": rho, "p": p, "n": len(m)})

    pd.DataFrame(rank_rows).to_csv(out_dir / "downsampling_rank_vs_full.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "downsampling_pairwise_spearman.csv", index=False)
    wide.to_csv(out_dir / "downsampling_sp_wide.csv")

    summary = {
        "dataset": name,
        "n_perturbations": len(valid),
        "levels": levels,
        "n_reps": args.n_reps,
        "min_available": args.min_available,
        "rank_vs_full": rank_rows,
        "config_version": cfg.CONFIG_VERSION,
    }
    with open(out_dir / "downsampling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote downsampling_* under {out_dir}")


if __name__ == "__main__":
    main()
