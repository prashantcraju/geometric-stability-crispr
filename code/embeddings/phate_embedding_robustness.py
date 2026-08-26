#!/usr/bin/env python3
"""
PHATE embedding robustness for Sp.

For each main dataset: same cells → HVG → (1) PCA Sp  (2) PHATE Sp → Spearman
of rankings. Completes the embedding triangle with PCA (linear) and scGPT
(learned nonlinear). Default: all five in_main datasets.

  pip install phate

Usage:
  python phate_embedding_robustness.py --compare-frozen
  python phate_embedding_robustness.py --mds-solver smacof   # if SGD non-converges

Do NOT pass n_mds_iter into phate.PHATE — unknown kwargs are forwarded to
graphtools and crash at fit. For SGD non-convergence warnings, use smacof.

Manuscript concordance = PHATE Sp vs frozen_sp_scores.csv (not re-derived PCA).
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
    assert_frozen_sp_compatible,
    calculate_sp,
    load_raw,
    materialize_min_cells,
    preprocess,
    setup_cache,
)
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir

THRESHOLD = 0.9


def _dense_X(adata) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


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


def run_phate(
    X: np.ndarray,
    n_components: int,
    seed: int,
    knn: int,
    n_pca: int,
    *,
    mds_solver: str = "sgd",
    n_mds_iter: int | None = None,
):
    try:
        import phate
    except ImportError as e:
        raise SystemExit(
            "phate is not installed. Run:  pip install phate\n"
            f"Original error: {e}"
        ) from e

    # PHATE's internal PCA cannot exceed n_features or n_samples - 1
    n_pca_eff = int(min(n_pca, X.shape[0] - 1, X.shape[1]))
    # CRITICAL: PHATE(**kwargs) forwards unknown keys to graphtools.Graph.
    # n_mds_iter / n_iter are NOT PHATE parameters — passing them constructs
    # OK then crashes at fit (Base.__init__ unexpected keyword). Do not pass.
    if n_mds_iter is not None:
        print(
            f"  NOTE: --n-mds-iter={n_mds_iter} ignored — phate does not expose "
            "SGD iteration count; unknown kwargs go to graphtools and crash. "
            "For non-convergence warnings use --mds-solver smacof.",
            flush=True,
        )
    print(
        f"  Fitting PHATE (n_components={n_components}, knn={knn}, "
        f"n_pca={n_pca_eff}, mds_solver={mds_solver}, "
        f"n_cells={X.shape[0]}, n_features={X.shape[1]})…",
        flush=True,
    )
    t0 = time.time()
    op = phate.PHATE(
        n_components=n_components,
        knn=knn,
        n_pca=n_pca_eff,
        random_state=seed,
        n_jobs=1,
        verbose=1,
        mds_solver=mds_solver,
    )
    emb = op.fit_transform(X)
    print(f"  PHATE done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if mds_solver == "sgd":
        print(
            "  NOTE: if SGD-MDS printed a non-convergence %%, treat Sp "
            "concordance as partly unconverged-embedding artifact; re-run "
            "with --mds-solver smacof (slower, paper-default MDS) and state "
            "the warning in the SI.",
            flush=True,
        )
    return np.asarray(emb, dtype=np.float64)


def run_one_dataset(
    name: str,
    sc,
    *,
    h5ad: Path | None,
    n_components: int,
    knn: int,
    n_pca: int,
    mds_solver: str,
    n_mds_iter: int,
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

    X_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    df_pca = score_sp_on_embedding(X_pca, labels, ctrl_label, valid).rename(
        columns={
            "stability": "sp_pca",
            "magnitude": "magnitude_pca",
            "spread": "spread_pca",
        }
    )
    del X_pca

    X = _dense_X(adata)
    del adata
    X_phate = run_phate(
        X,
        n_components=n_components,
        seed=cfg.SEED,
        knn=knn,
        n_pca=n_pca,
        mds_solver=mds_solver,
        n_mds_iter=n_mds_iter,
    )
    del X
    df_phate = score_sp_on_embedding(X_phate, labels, ctrl_label, valid).rename(
        columns={
            "stability": "sp_phate",
            "magnitude": "magnitude_phate",
            "spread": "spread_phate",
        }
    )
    del X_phate

    merged = df_pca.merge(
        df_phate[["perturbation", "sp_phate", "magnitude_phate", "spread_phate"]],
        on="perturbation",
        how="inner",
    )
    rho_internal, p_internal = spearmanr(merged["sp_pca"], merged["sp_phate"])
    rho_mag, p_mag = spearmanr(merged["magnitude_pca"], merged["magnitude_phate"])

    print(
        f"  internal PCA vs PHATE Sp: n={len(merged)}  "
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
        "phate_knn": knn,
        "phate_n_pca": n_pca,
        "phate_mds_solver": mds_solver,
        "phate_n_mds_iter": n_mds_iter,
        "phate_sgd_convergence_caveat": (
            "If SGD-MDS prints a non-convergence %%, Sp concordance may be "
            "partly unconverged-embedding artifact; state explicitly in SI."
        ),
        "max_cells_per_pert": max_cells_per_pert,
        "spearman_sp_pca_vs_phate": (
            float(rho_internal) if np.isfinite(rho_internal) else None
        ),
        "spearman_sp_p": float(p_internal) if np.isfinite(p_internal) else None,
        "spearman_magnitude_pca_vs_phate": (
            float(rho_mag) if np.isfinite(rho_mag) else None
        ),
        "robust_threshold": THRESHOLD,
        "status": "ok",
        "preprocess": "pipeline_core.preprocess (frozen path)",
    }

    rho_report = rho_internal
    if compare_frozen and frozen_df is not None:
        fr = frozen_df[frozen_df["dataset"] == name][["perturbation", "stability"]]
        fr = fr.rename(columns={"stability": "sp_frozen_pca"})
        m2 = merged.merge(fr, on="perturbation", how="inner")
        if len(m2) >= 5:
            r1, _ = spearmanr(m2["sp_phate"], m2["sp_frozen_pca"])
            r2, _ = spearmanr(m2["sp_pca"], m2["sp_frozen_pca"])
            summary["spearman_phate_vs_frozen_pca"] = float(r1)
            summary["spearman_matched_pca_vs_frozen"] = float(r2)
            summary["n_vs_frozen"] = int(len(m2))
            rho_report = float(r1)
            print(
                f"  vs frozen CSV: PHATE ρ={r1:.4f}; "
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
    merged.to_csv(out_dir / f"phate_vs_pca_sp_{slug}.csv", index=False)
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
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--n-pca", type=int, default=100)
    parser.add_argument(
        "--mds-solver",
        default="sgd",
        choices=("sgd", "smacof"),
        help="PHATE MDS solver (default sgd; use smacof if SGD prints non-convergence)",
    )
    parser.add_argument(
        "--n-mds-iter",
        type=int,
        default=None,
        help="Ignored (phate has no such kwarg; kept so old Colab cmds do not abort)",
    )
    parser.add_argument("--max-cells-per-pert", type=int, default=cfg.MAX_CELLS_PER_PERT)
    parser.add_argument("--max-control-cells", type=int, default=cfg.MAX_CONTROL_CELLS)
    parser.add_argument(
        "--compare-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score PHATE vs frozen_sp_scores.csv (default: on; manuscript column)",
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

    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"datasets ({len(names)}): {names}")
    print(
        f"caps: ≤{args.max_cells_per_pert}/pert, ≤{args.max_control_cells} controls",
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
                knn=args.knn,
                n_pca=args.n_pca,
                mds_solver=args.mds_solver,
                n_mds_iter=args.n_mds_iter,
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
        big.to_csv(out_dir / "phate_vs_pca_sp_all.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "phate_embedding_summary.csv", index=False)
    with open(out_dir / "phate_embedding_summary.json", "w") as f:
        json.dump(
            {
                "config_version": cfg.CONFIG_VERSION,
                "threshold": THRESHOLD,
                "datasets": summaries,
            },
            f,
            indent=2,
        )

    # Methods blurb across successful runs
    ok = [s for s in summaries if s.get("status") == "ok"]
    if ok:
        bits = [
            f"{s['dataset'].split('(')[0].strip()} ρ={s['spearman_sp_pca_vs_phate']:.2f} "
            f"(n={s['n_perturbations']})"
            for s in ok
            if s.get("spearman_sp_pca_vs_phate") is not None
        ]
        all_robust = all(
            (s.get("spearman_sp_pca_vs_phate") or 0) >= THRESHOLD for s in ok
        )
        blurb = (
            f"Embedding robustness (PHATE). Sp was recomputed in a "
            f"{args.n_components}-dimensional PHATE embedding (knn={args.knn}, "
            f"internal PCA={args.n_pca}) on the same cells and HVG features as the "
            f"PCA pipeline for each main dataset. PCA vs PHATE Sp ranking "
            f"concordance: {'; '.join(bits)}. "
        )
        if all_robust:
            blurb += (
                "Sp rankings were robust to this nonlinear manifold embedding "
                "choice across datasets (alongside scGPT as a learned nonlinear check)."
            )
        else:
            blurb += (
                "Where ρ fell below 0.9, embedding choice can reorder perturbations "
                "and is noted as a limitation for that dataset."
            )
        with open(out_dir / "phate_embedding_methods_blurb.txt", "w") as f:
            f.write(blurb + "\n")
        print(f"\n{blurb}")

    print(f"\nWrote {out_dir}/phate_embedding_summary.csv")
    if all_rows:
        print(f"Wrote {out_dir}/phate_vs_pca_sp_all.csv")


if __name__ == "__main__":
    main()
