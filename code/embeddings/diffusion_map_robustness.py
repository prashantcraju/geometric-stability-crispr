#!/usr/bin/env python3
"""
Diffusion-map embedding robustness for Sp.

Uses pipeline_core.preprocess (same filter_cells → normalize/log pin → HVG →
PCA as frozen Sp), then scores Sp on (1) that PCA and (2) diffusion maps.
Manuscript concordance = DiffMap Sp vs frozen_sp_scores.csv (not a re-derived
PCA). Pass --compare-frozen (default).

Usage:
  python diffusion_map_robustness.py --compare-frozen
  python diffusion_map_robustness.py --datasets "Replogle 2022 (CRISPRi)"
  python diffusion_map_robustness.py --max-cells-per-pert 50
  python diffusion_map_robustness.py --keep-dc0

If Spearman(DiffMap Sp, frozen Sp) ≥ 0.9 → SI table + Methods sentence.
If lower → report as limitation (same rule as PHATE).
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
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    _pca_truncated_svd,
    assert_frozen_sp_compatible,
    calculate_sp,
    load_raw,
    materialize_min_cells,
    preprocess,
    setup_cache,
)
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir

THRESHOLD = 0.9


def _resolve_h5ad(name: str, explicit: Path | None, adamson_h5ad: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if "adamson" in name.lower() and "upr" in name.lower() and adamson_h5ad is not None:
        return adamson_h5ad
    meta = cfg.DATASETS.get(name, {})
    local = meta.get("local_h5ad")
    if local:
        cand = cfg.CACHE_DIR / local
        if cand.exists():
            return cand
    return None


def score_sp_on_embedding(
    emb: np.ndarray,
    labels: np.ndarray,
    ctrl_label: str,
    valid: list[str],
) -> pd.DataFrame:
    ctrl_mask = labels == ctrl_label
    X_ctrl = emb[ctrl_mask]
    if X_ctrl.shape[0] < cfg.MIN_CONTROL_CELLS:
        raise ValueError(f"Too few control cells: {X_ctrl.shape[0]}")
    rows = []
    for pert in valid:
        Xp = emb[labels == pert]
        if Xp.shape[0] < cfg.MIN_CELLS:
            continue
        m = calculate_sp(X_ctrl, Xp)
        if m["magnitude"] <= 0:
            continue
        rows.append(
            {
                "perturbation": pert,
                "stability": m["stability"],
                "magnitude": m["magnitude"],
                "spread": m["spread"],
                "n_cells": int(Xp.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def run_diffmap(
    adata,
    sc,
    *,
    n_components: int,
    n_neighbors: int,
    n_pcs: int,
    seed: int,
    drop_dc0: bool,
):
    """
    Neighbors on TruncatedSVD PCs → scanpy diffusion map.

    scanpy's X_diffmap[:, 0] is the trivial DC0 eigenvalue component; by
    default it is dropped so Sp is scored on the informative DCs only.
    Request one extra component when dropping DC0 so the usable rank matches
    n_components.
    """
    n_pcs_eff = int(min(n_pcs, adata.n_obs - 1, adata.n_vars))
    n_comps_req = n_components + (1 if drop_dc0 else 0)
    # eigendecomposition of the transition matrix: at most n_obs - 1 comps
    n_comps_req = int(min(n_comps_req, max(2, adata.n_obs - 2)))

    print(
        f"  Fitting diffusion map (n_components={n_components}, "
        f"n_neighbors={n_neighbors}, n_pcs={n_pcs_eff}, drop_dc0={drop_dc0}, "
        f"n_cells={adata.n_obs}, n_features={adata.n_vars})…",
        flush=True,
    )
    t0 = time.time()

    ad = _pca_truncated_svd(adata, n_pcs_eff, seed)
    sc.pp.neighbors(
        ad,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_eff,
        random_state=seed,
    )
    sc.tl.diffmap(ad, n_comps=n_comps_req)

    emb = np.asarray(ad.obsm["X_diffmap"], dtype=np.float64)
    if drop_dc0:
        if emb.shape[1] < 2:
            raise ValueError(f"DiffMap returned {emb.shape[1]} comps; need ≥2 to drop DC0")
        emb = emb[:, 1 : 1 + n_components]
    else:
        emb = emb[:, :n_components]

    print(
        f"  DiffMap done in {(time.time() - t0) / 60:.1f} min "
        f"(emb shape={emb.shape})",
        flush=True,
    )
    return emb


def run_one_dataset(
    name: str,
    sc,
    *,
    h5ad: Path | None,
    n_components: int,
    n_neighbors: int,
    n_pcs: int,
    drop_dc0: bool,
    max_cells_per_pert: int,
    max_control_cells: int,
    compare_frozen: bool,
    out_dir: Path,
    frozen_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}", flush=True)
    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad)
    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    adata, valid, counts = materialize_min_cells(
        adata,
        pert_col,
        ctrl_label,
        min_cells=cfg.MIN_CELLS,
        max_cells_per_pert=max_cells_per_pert,
        max_control_cells=max_control_cells,
        seed=cfg.SEED,
    )
    # Same preprocess as frozen Sp (filter_cells → normalize/log per pin → HVG → PCA)
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
        dataset_name=name,
    )

    labels = adata.obs[pert_col].astype(str).to_numpy()
    print(f"  Scoring {len(valid)} perturbations ({adata.n_obs} cells)", flush=True)

    # Baseline Sp from frozen preprocess PCA (not a re-derived TruncatedSVD)
    X_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    df_pca = score_sp_on_embedding(X_pca, labels, ctrl_label, valid).rename(
        columns={
            "stability": "sp_pca",
            "magnitude": "magnitude_pca",
            "spread": "spread_pca",
        }
    )
    del X_pca

    X_dm = run_diffmap(
        adata,
        sc,
        n_components=n_components,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        seed=cfg.SEED,
        drop_dc0=drop_dc0,
    )
    del adata
    df_dm = score_sp_on_embedding(X_dm, labels, ctrl_label, valid).rename(
        columns={
            "stability": "sp_diffmap",
            "magnitude": "magnitude_diffmap",
            "spread": "spread_diffmap",
        }
    )
    del X_dm

    merged = df_pca.merge(
        df_dm[["perturbation", "sp_diffmap", "magnitude_diffmap", "spread_diffmap"]],
        on="perturbation",
        how="inner",
    )
    rho_internal, p_internal = spearmanr(merged["sp_pca"], merged["sp_diffmap"])
    rho_mag, p_mag = spearmanr(merged["magnitude_pca"], merged["magnitude_diffmap"])

    print(
        f"  internal PCA vs DiffMap Sp: n={len(merged)}  "
        f"ρ={rho_internal:.4f}  p={p_internal:.3g}  (diagnostic only)"
    )
    print(f"  magnitude ρ={rho_mag:.4f}")

    summary = {
        "dataset": name,
        "modality": cfg.DATASETS[name]["modality"],
        "cell_type": cfg.DATASETS[name]["cell_type"],
        "config_version": cfg.CONFIG_VERSION,
        "n_perturbations": int(len(merged)),
        "n_cells": int(len(labels)),
        "n_components": n_components,
        "diffmap_n_neighbors": n_neighbors,
        "diffmap_n_pcs": n_pcs,
        "diffmap_drop_dc0": drop_dc0,
        "max_cells_per_pert": max_cells_per_pert,
        "spearman_sp_pca_vs_diffmap": (
            float(rho_internal) if np.isfinite(rho_internal) else None
        ),
        "spearman_sp_p": float(p_internal) if np.isfinite(p_internal) else None,
        "spearman_magnitude_pca_vs_diffmap": (
            float(rho_mag) if np.isfinite(rho_mag) else None
        ),
        "robust_threshold": THRESHOLD,
        "status": "ok",
        "preprocess": "pipeline_core.preprocess (frozen path)",
    }

    # Manuscript concordance = DiffMap vs frozen Sp CSV (not re-derived PCA)
    rho_report = rho_internal
    if compare_frozen and frozen_df is not None:
        fr = frozen_df[frozen_df["dataset"] == name][["perturbation", "stability"]]
        fr = fr.rename(columns={"stability": "sp_frozen_pca"})
        m2 = merged.merge(fr, on="perturbation", how="inner")
        if len(m2) >= 5:
            r1, p1 = spearmanr(m2["sp_diffmap"], m2["sp_frozen_pca"])
            r2, _ = spearmanr(m2["sp_pca"], m2["sp_frozen_pca"])
            summary["spearman_diffmap_vs_frozen_pca"] = float(r1)
            summary["spearman_matched_pca_vs_frozen"] = float(r2)
            summary["n_vs_frozen"] = int(len(m2))
            rho_report = float(r1)
            print(
                f"  vs frozen CSV: DiffMap ρ={r1:.4f}; "
                f"matched-PCA ρ={r2:.4f} (n={len(m2)})"
            )
        else:
            print(
                f"  WARNING: only {len(m2)} perts overlap frozen CSV; "
                "cannot report frozen concordance"
            )
    else:
        print(
            "  WARNING: --compare-frozen not set / no frozen table; "
            "reporting internal PCA concordance only"
        )

    print(f"  ρ vs frozen Sp = {rho_report:.4f} (threshold {THRESHOLD})")
    summary["rho_vs_frozen"] = float(rho_report) if np.isfinite(rho_report) else None
    summary["rho_threshold"] = THRESHOLD
    summary["verdict"] = (
        f"rho={rho_report:.4f}" if np.isfinite(rho_report) else "undefined"
    )

    merged = merged.copy()
    merged["dataset"] = name
    merged["config_version"] = cfg.CONFIG_VERSION
    slug = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )
    merged.to_csv(out_dir / f"diffmap_vs_pca_sp_{slug}.csv", index=False)
    return merged, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset display names (default: all in_main)",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=None,
        help="Override h5ad/h5mu for a single --datasets run",
    )
    parser.add_argument(
        "--adamson-h5ad",
        type=Path,
        default=None,
        help="Local Adamson UPR h5ad path",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-components", type=int, default=cfg.N_PCS)
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="k for sc.pp.neighbors (default: 15, scanpy default)",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=100,
        help="PCs used to build the neighbor graph (default: 100)",
    )
    parser.add_argument(
        "--keep-dc0",
        action="store_true",
        help="Keep the trivial DC0 component (default: drop it)",
    )
    parser.add_argument("--max-cells-per-pert", type=int, default=cfg.MAX_CELLS_PER_PERT)
    parser.add_argument("--max-control-cells", type=int, default=cfg.MAX_CONTROL_CELLS)
    parser.add_argument(
        "--compare-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score DiffMap vs frozen_sp_scores.csv (default: on; manuscript column)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Skip failed datasets and continue (default: True)",
    )
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    names = args.datasets or cfg.main_dataset_names()
    names = [cfg.resolve_dataset_name(n) for n in names]
    if args.h5ad is not None and len(names) != 1:
        raise SystemExit("--h5ad requires exactly one dataset in --datasets")

    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = cfg.CACHE_DIR

    frozen_df = None
    if args.compare_frozen:
        frozen_path = find_sp_csv(out_dir)
        assert_frozen_sp_compatible(frozen_path)
        frozen_df = load_sp_table(frozen_path)

    drop_dc0 = not args.keep_dc0
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"datasets ({len(names)}): {names}")
    print(
        f"caps: ≤{args.max_cells_per_pert}/pert, ≤{args.max_control_cells} controls; "
        f"drop_dc0={drop_dc0}",
        flush=True,
    )

    all_rows = []
    summaries = []
    for name in names:
        if name not in cfg.DATASETS:
            print(f"SKIP unknown dataset: {name}")
            summaries.append({"dataset": name, "status": "unknown"})
            continue
        h5ad = _resolve_h5ad(name, args.h5ad, args.adamson_h5ad)
        try:
            merged, summary = run_one_dataset(
                name,
                sc,
                h5ad=h5ad,
                n_components=args.n_components,
                n_neighbors=args.n_neighbors,
                n_pcs=args.n_pcs,
                drop_dc0=drop_dc0,
                max_cells_per_pert=args.max_cells_per_pert,
                max_control_cells=args.max_control_cells,
                compare_frozen=args.compare_frozen,
                out_dir=out_dir,
                frozen_df=frozen_df,
            )
            all_rows.append(merged)
            summaries.append(summary)
        except Exception as e:
            print(f"FAILED {name}: {e}")
            traceback.print_exc()
            summaries.append(
                {"dataset": name, "status": "failed", "error": str(e)}
            )
            if not args.continue_on_error:
                raise

    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        big.to_csv(out_dir / "diffmap_vs_pca_sp_all.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "diffmap_embedding_summary.csv", index=False)
    with open(out_dir / "diffmap_embedding_summary.json", "w") as f:
        json.dump(
            {
                "config_version": cfg.CONFIG_VERSION,
                "threshold": THRESHOLD,
                "drop_dc0": drop_dc0,
                "datasets": summaries,
            },
            f,
            indent=2,
        )

    ok = [s for s in summaries if s.get("status") == "ok"]
    if ok:
        bits = [
            f"{s['dataset'].split('(')[0].strip()} ρ={s['spearman_sp_pca_vs_diffmap']:.2f} "
            f"(n={s['n_perturbations']})"
            for s in ok
            if s.get("spearman_sp_pca_vs_diffmap") is not None
        ]
        all_robust = all(
            (s.get("spearman_sp_pca_vs_diffmap") or 0) >= THRESHOLD for s in ok
        )
        blurb = (
            f"Embedding robustness (diffusion maps). Sp was recomputed in a "
            f"{args.n_components}-dimensional diffusion-map embedding "
            f"(n_neighbors={args.n_neighbors}, neighbor-graph PCs={args.n_pcs}"
            f"{', DC0 dropped' if drop_dc0 else ''}) on the same cells and HVG "
            f"features as the PCA pipeline for each main dataset. PCA vs "
            f"diffusion-map Sp ranking concordance: {'; '.join(bits)}. "
        )
        if all_robust:
            blurb += (
                "Sp rankings were robust to this nonlinear manifold embedding "
                "choice across datasets (alongside PHATE and scGPT)."
            )
        else:
            blurb += (
                "Where ρ fell below 0.9, embedding choice can reorder perturbations "
                "and is noted as a limitation for that dataset."
            )
        with open(out_dir / "diffmap_embedding_methods_blurb.txt", "w") as f:
            f.write(blurb + "\n")
        print(f"\n{blurb}")

    print(f"\nWrote {out_dir}/diffmap_embedding_summary.csv")
    if all_rows:
        print(f"Wrote {out_dir}/diffmap_vs_pca_sp_all.csv")


if __name__ == "__main__":
    main()
