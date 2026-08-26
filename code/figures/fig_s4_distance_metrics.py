#!/usr/bin/env python3
"""
S4 — Magnitude–coherence by distance metric (Euclidean / Whitened / k-NN).

Standalone Colab script. Scores all six frozen datasets on the same PCA
embedding, then draws the original three-bar figure.

    python fig_s4_distance_metrics.py
    python fig_s4_distance_metrics.py --plot-only

Uploads needed on Colab: this file, pipeline_config.py, pipeline_core.py,
revision_io.py, stats_utils.py, fig_style.py.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline_config as cfg
from fig_style import (
    DARK, DATASETS, SEARCH_DIRS, find_csv, resolve_out_dir, despine, save_fig,
)


KNN_K = 50
WHITEN_REG = 1e-6
METHOD_ORDER = ["Euclidean", "Whitened", "k-NN"]
BAR_COLORS = {
    "Euclidean": "#888888",
    "Whitened": "#555555",
    "k-NN": "#AAAAAA",
}
DATASET_ORDER = [(full, short) for full, short, *_ in DATASETS]


def _out_dir():
    return resolve_out_dir()


def _norm_method(s):
    s = str(s).strip()
    low = s.lower().replace("_", "-")
    if low in {"euclidean", "eucl"}:
        return "Euclidean"
    if low in {"whitened", "whiten", "mahalanobis"}:
        return "Whitened"
    if low in {"k-nn", "knn", "k.nn"}:
        return "k-NN"
    return s


def load_correlation_csv():
    path = find_csv(
        "crispr_correlations_with_ci.csv",
        "papalexi_method_correlations.csv",
    )
    frames = []
    main = find_csv("crispr_correlations_with_ci.csv")
    if main is not None:
        print(f"Correlations: {main}")
        df = pd.read_csv(main)
        df["dataset"] = df["dataset"].map(cfg.resolve_dataset_name)
        df["method"] = df["method"].map(_norm_method)
        frames.append(df)
    pap = find_csv("papalexi_method_correlations.csv")
    if pap is not None:
        print(f"Papalexi methods: {pap}")
        dfp = pd.read_csv(pap)
        if "dataset" not in dfp.columns:
            dfp["dataset"] = "Papalexi 2021 (CRISPR-KO)"
        dfp["dataset"] = dfp["dataset"].map(cfg.resolve_dataset_name)
        dfp["method"] = dfp["method"].map(_norm_method)
        frames.append(dfp)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("dataset", "method", "n", "rho", "ci_low", "ci_high", "p") if c in out.columns]
    out = out[keep].drop_duplicates(subset=["dataset", "method"], keep="last")
    return out


def _whiten(control_matrix, pert_matrix, regularization=WHITEN_REG):
    from pipeline_core import calculate_sp
    cov = np.cov(control_matrix.T) + regularization * np.eye(control_matrix.shape[1])
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, regularization)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        ctrl_w = (W @ control_matrix.T).T
        pert_w = (W @ pert_matrix.T).T
        return calculate_sp(ctrl_w, pert_w)
    except np.linalg.LinAlgError:
        return calculate_sp(control_matrix, pert_matrix)


def _knn(control_matrix, pert_matrix, k=KNN_K):
    from sklearn.neighbors import NearestNeighbors
    k = min(k, control_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(control_matrix)
    _, indices = nn.kneighbors(pert_matrix)
    shift = np.empty_like(pert_matrix)
    for i, idx in enumerate(indices):
        shift[i] = pert_matrix[i] - control_matrix[idx].mean(axis=0)
    return _sp_from_shifts(shift)


def _sp_from_shifts(shift_vectors):
    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = float(np.linalg.norm(mean_shift))
    if mean_magnitude < 1e-6:
        return {"stability": 0.0, "magnitude": 0.0, "spread": 0.0, "snr": 0.0}
    norms = np.linalg.norm(shift_vectors, axis=1)
    valid = norms > 1e-6
    if np.sum(valid) < 5:
        return {"stability": 0.0, "magnitude": 0.0, "spread": 0.0, "snr": 0.0}
    unit_mean = mean_shift / mean_magnitude
    cosine = np.dot(shift_vectors[valid], unit_mean) / norms[valid]
    stability = float(np.mean(cosine))
    spread = float(np.mean(np.linalg.norm(shift_vectors - mean_shift, axis=1)))
    return {
        "stability": stability,
        "magnitude": mean_magnitude,
        "spread": spread,
        "snr": mean_magnitude / (spread + 1e-6),
    }


def score_three_methods(adata, pert_col, ctrl_label, valid, dataset_name):
    from pipeline_core import calculate_sp
    X_ctrl = np.asarray(adata.obsm["X_pca"][adata.obs[pert_col] == ctrl_label])
    if X_ctrl.shape[0] < cfg.MIN_CONTROL_CELLS:
        raise ValueError(f"Too few control cells: {X_ctrl.shape[0]}")
    rows = []
    for pert in valid:
        X_pert = np.asarray(adata.obsm["X_pca"][adata.obs[pert_col] == pert])
        n_cells = int(X_pert.shape[0])
        euc = calculate_sp(X_ctrl, X_pert)
        wht = _whiten(X_ctrl, X_pert)
        knn = _knn(X_ctrl, X_pert)
        for method, m in (("Euclidean", euc), ("Whitened", wht), ("k-NN", knn)):
            if m["magnitude"] <= 0:
                continue
            rows.append({
                "dataset": dataset_name,
                "perturbation": str(pert),
                "method": method,
                "stability": m["stability"],
                "magnitude": m["magnitude"],
                "n_cells": n_cells,
                "config_version": cfg.CONFIG_VERSION,
            })
    return pd.DataFrame(rows)


def summarize(perts, n_bootstrap, seed0=1000):
    from stats_utils import bootstrap_spearman_ci
    rows = []
    seed = seed0
    for ds, short in DATASET_ORDER:
        sub_ds = perts[perts["dataset"] == ds]
        if not len(sub_ds):
            continue
        print(f"\n--- {short} ---")
        for method in METHOD_ORDER:
            sub = sub_ds[sub_ds["method"] == method]
            if len(sub) < 3:
                print(f"  {method}: n={len(sub)}  (skip)")
                continue
            ci = bootstrap_spearman_ci(
                sub["magnitude"], sub["stability"],
                n_bootstrap=n_bootstrap, seed=seed,
            )
            seed += 1
            print(
                f"  {method:<10s}  n={len(sub):4d}  "
                f"ρ={ci['rho']:+.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            )
            rows.append({
                "dataset": ds,
                "method": method,
                "n": int(len(sub)),
                "rho": ci["rho"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "p": ci["p"],
            })
    return pd.DataFrame(rows)


def plot_s4(corr, out_dir):
    present = [
        (full, short) for full, short in DATASET_ORDER
        if full in set(corr["dataset"])
    ]
    if not present:
        raise ValueError("No datasets in the correlation table.")
    n_ds = len(present)
    n_m = len(METHOD_ORDER)
    bar_width = 0.22
    fig, ax = plt.subplots(figsize=(max(8.4, 1.7 * n_ds), 5.4))
    x_centers = np.arange(n_ds)
    offsets = np.linspace(-(n_m - 1) / 2 * bar_width, (n_m - 1) / 2 * bar_width, n_m)

    for mi, method in enumerate(METHOD_ORDER):
        rhos, lo, hi = [], [], []
        for ds, _ in present:
            hit = corr[(corr["dataset"] == ds) & (corr["method"] == method)]
            if len(hit):
                rhos.append(float(hit["rho"].iloc[0]))
                lo.append(float(hit["ci_low"].iloc[0]))
                hi.append(float(hit["ci_high"].iloc[0]))
            else:
                rhos.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
        rhos, lo, hi = map(np.asarray, (rhos, lo, hi))
        ok = np.isfinite(rhos)
        x = x_centers + offsets[mi]
        ax.bar(
            x[ok], rhos[ok], width=bar_width, color=BAR_COLORS[method],
            edgecolor="white", linewidth=0.6, label=method, zorder=3,
        )
        ax.errorbar(
            x[ok], rhos[ok],
            yerr=[rhos[ok] - lo[ok], hi[ok] - rhos[ok]],
            fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=4,
        )
        for xi, v, good in zip(x, rhos, ok):
            if good:
                ax.text(xi, min(v + 0.025, 1.05), f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [s for _, s in present], fontsize=10, fontweight="bold",
        rotation=20, ha="right",
    )
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.55, n_ds - 0.45)
    ax.yaxis.grid(True, linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Magnitude–coherence correlation by distance metric",
                 fontsize=13, fontweight="bold", pad=10)
    despine(ax)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.legend(
        loc="lower center", ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure,
        handlelength=1.2, columnspacing=1.6,
    )
    save_fig(fig, out_dir / "fig_s4_method_comparison")
    corr.to_csv(out_dir / "crispr_correlations_with_ci.csv", index=False)
    corr.to_csv(out_dir / "fig_s4_method_comparison.csv", index=False)
    (out_dir / "fig_s4_method_comparison_caption.txt").write_text(
        r"""\caption{\textbf{Magnitude--coherence is not an artifact of Euclidean PCA distance.}
Spearman $\rho$ between Shesha coherence and effect magnitude under three
control-referenced metrics, six datasets: Euclidean centroid, whitened
(Mahalanobis / $\Sigma^{-1/2}$), and $k$-NN matched local controls ($k=50$).
Bars are $95\%$ bootstrap CIs. The original five-dataset panel is recovered
and extended by the Adamson 2016 pilot.}
"""
    )


def _prepare(name, ds, sc):
    import pipeline_core as pc
    if hasattr(pc, "prepare_dataset"):
        return pc.prepare_dataset(name, ds=ds, sc=sc, prefer_local=True)
    raw = pc.load_raw(name, ds=ds, sc=sc, prefer_local=True)
    extract = getattr(pc, "_extract_adata", None) or getattr(pc, "extract_adata")
    adata, pert_col, ctrl_label = extract(raw, name, sc)
    adata, valid, counts = pc.materialize_min_cells(
        adata, pert_col, ctrl_label, min_cells=cfg.MIN_CELLS,
        max_control_cells=cfg.MAX_CONTROL_CELLS,
    )
    adata, valid, counts = pc.preprocess(
        adata, pert_col, ctrl_label, sc,
        n_pcs=cfg.N_PCS, min_cells=cfg.MIN_CELLS, seed=cfg.SEED,
        valid_perts=valid, counts=counts, dataset_name=name,
    )
    print(f"    after filter: {len(valid)} perturbations", flush=True)
    return adata, pert_col, ctrl_label, valid, counts


def score_live(names, n_bootstrap):
    from pipeline_core import import_pertpy_datasets, setup_cache
    setup_cache()
    ds, sc = import_pertpy_datasets()
    frames = []
    for name in names:
        try:
            adata, pert_col, ctrl_label, valid, _ = _prepare(name, ds, sc)
            tab = score_three_methods(adata, pert_col, ctrl_label, valid, name)
            print(f"    scored {len(tab)} rows ({tab['method'].nunique()} methods)")
            frames.append(tab)
        except Exception as e:
            print(f"    FAILED {name}: {e}")
    if not frames:
        raise RuntimeError("No dataset scored.")
    perts = pd.concat(frames, ignore_index=True)
    return perts, summarize(perts, n_bootstrap)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot-only", action="store_true",
                   help="Draw from crispr_correlations_with_ci.csv; do not rescore")
    p.add_argument("--n-bootstrap", type=int, default=cfg.N_BOOTSTRAP)
    p.add_argument("--datasets", default="",
                   help="Comma-separated frozen names; default = all six")
    args = p.parse_args(argv)

    out = _out_dir()
    print(f"OUT_DIR={out}")
    print(f"Search: {[str(d) for d in SEARCH_DIRS if d.exists()]}")

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if not names:
        names = [full for full, *_ in DATASET_ORDER]

    corr = load_correlation_csv()

    def _missing(corr_df):
        if corr_df is None or not len(corr_df):
            return list(names)
        need = []
        for ds in names:
            methods = set(corr_df.loc[corr_df["dataset"] == ds, "method"])
            if not methods.issuperset(METHOD_ORDER):
                need.append(ds)
        return need

    missing = _missing(corr)
    if args.plot_only:
        if corr is None:
            raise FileNotFoundError(
                "Need crispr_correlations_with_ci.csv for --plot-only"
            )
        print(f"Plot-only. Datasets in CSV: {sorted(set(corr['dataset']))}")
        if missing:
            print(f"  still missing 3-method rows for: {missing}")
        plot_s4(corr, out)
        return

    if not missing:
        print("All six datasets already have Euclidean / Whitened / k-NN — plotting.")
        plot_s4(corr, out)
        return

    print(f"Need to score: {missing}")
    print("Scoring Euclidean / Whitened / k-NN on the frozen pipeline embeddings.")
    perts, corr_new = score_live(missing, args.n_bootstrap)
    perts.to_csv(out / "s4_per_perturbation_three_methods.csv", index=False)
    if corr is not None:
        corr = pd.concat([corr, corr_new], ignore_index=True)
        corr = corr.drop_duplicates(subset=["dataset", "method"], keep="last")
    else:
        corr = corr_new
    plot_s4(corr, out)
    print("\nS4 three-method figure written.")


if __name__ == "__main__":
    main()
