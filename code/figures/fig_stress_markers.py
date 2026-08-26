#!/usr/bin/env python3
"""
SUPERSEDED record of the old Fig 5c / S9 / S20 family.

Do not use as manuscript art. Quadrants are deleted with S20.
Replacement forest: fig_s9_stress_forest.py (signed rho only).

Stress-marker figures from stored CSVs (no pertpy, no remount).

Reads the tables next to null_model_simulation.csv:


and writes three figures matching the existing screenshots, with
Stability → Coherence on titles and axes.

USAGE:
    python fig_stress_markers.py
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import os
from pathlib import Path
from revision_io import data_search_dirs, find_data_file, resolve_out_dir


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import pipeline_config as cfg
except ImportError:
    cfg = None

_CSV_ROOTS = data_search_dirs()

MARKERS = ["DDIT3", "ATF4", "XBP1", "HSPA5"]
HEATMAP_MARKERS = ["ATF4", "DDIT3", "HSPA5", "XBP1"]
DATASETS = ["Dixit", "Norman", "Replogle"]

HH_COLOR = "#d62728"
HL_COLOR = "#9ecae1"
LH_COLOR = "#fdae6b"
LL_COLOR = "#74c476"
BAR_COLOR = "#6baed6"


def _find_csv(*names):
    for name in names:
        for root in _CSV_ROOTS:
            p = root / name
            if p.exists():
                return p
    return None


def _out_dir():
    for p in _CSV_ROOTS:
        if p.exists():
            return p
    out = Path("./shesha-crispr")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _short_dataset(name):
    s = str(name)
    if cfg is not None:
        s = cfg.resolve_dataset_name(s)
    for key in DATASETS:
        if key.lower() in s.lower():
            return key
    return s


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _keep_main_datasets(df):
    df = df.copy()
    df["dataset_short"] = df["dataset"].map(_short_dataset)
    return df[df["dataset_short"].isin(DATASETS)].copy()


def plot_partial_bars(df, out_dir):
    """Horizontal forest: partial rho ± CI, grouped by marker."""
    rows = []
    ylabels = []
    for marker in MARKERS:
        for ds in DATASETS:
            sub = df[(df["marker"] == marker) & (df["dataset_short"] == ds)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            rows.append(r)
            ylabels.append(f"{ds} / {marker}")
    if not rows:
        print("No overlapping dataset/marker rows for the bar chart.")
        return

    plot_df = pd.DataFrame(rows).reset_index(drop=True)
    y = np.arange(len(plot_df))
    if "abs_rho_partial" in plot_df.columns and "rho_partial" not in plot_df.columns:
        raise ValueError(
            "Refuse to plot abs_rho_partial. That column is an effect-size "
            "bin only; using it as the forest bar flips every negative rho "
            "and leaves the CIs on the signed scale."
        )
    rho = plot_df["rho_partial"].to_numpy(dtype=float)
    lo = plot_df["rho_partial_ci_low"].to_numpy(dtype=float)
    hi = plot_df["rho_partial_ci_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.barh(y, rho, color=BAR_COLOR, height=0.62, zorder=2)
    ax.errorbar(
        rho, y, xerr=[rho - lo, hi - rho],
        fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=3,
    )
    ax.axvline(0, color="#888888", ls="--", lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"Spearman $\rho$ (partial)", fontsize=11, fontweight="bold")
    ax.set_title("Stress Marker Correlations with Coherence",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-0.45, 0.85)
    ax.xaxis.grid(True, ls=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    plt.tight_layout()

    stem = out_dir / "fig_stress_partial_bars"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {stem}.pdf / .png")
    plt.close(fig)


def plot_quadrants(df, out_dir):
    """2×2 grouped bars of HH/HL/LH/LL counts; * = HH depleted."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=False)
    letters = "abcd"
    x = np.arange(len(DATASETS))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    colors = [HH_COLOR, HL_COLOR, LH_COLOR, LL_COLOR]
    keys = [
        "q_high_stab_high_stress",
        "q_high_stab_low_stress",
        "q_low_stab_high_stress",
        "q_low_stab_low_stress",
    ]
    labels = ["HH", "HL", "LH", "LL"]

    for i, marker in enumerate(MARKERS):
        ax = axes.flat[i]
        sub = df[df["marker"] == marker]
        for j, (key, color) in enumerate(zip(keys, colors)):
            heights = []
            for ds in DATASETS:
                row = sub[sub["dataset_short"] == ds]
                heights.append(float(row.iloc[0][key]) if len(row) else np.nan)
            heights = np.array(heights, dtype=float)
            valid = ~np.isnan(heights)
            ax.bar(
                x[valid] + offsets[j], heights[valid],
                width=width, color=color, edgecolor="white", linewidth=0.4,
                label=labels[j] if i == 0 else None, zorder=2,
            )
        for k, ds in enumerate(DATASETS):
            row = sub[sub["dataset_short"] == ds]
            if not len(row):
                continue
            r = row.iloc[0]
            depleted = r.get("hh_depleted", False)
            if isinstance(depleted, str):
                depleted = depleted.strip().lower() in {"true", "1", "yes"}
            if depleted:
                hh = float(r["q_high_stab_high_stress"])
                ax.text(
                    x[k] + offsets[0], hh + max(hh * 0.04, 8), "*",
                    color=HH_COLOR, ha="center", va="bottom",
                    fontsize=16, fontweight="bold",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=10)
        ax.set_ylabel("Perturbation count", fontsize=10)
        ax.set_title(f"{letters[i]}. {marker}", fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, ls=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        _despine(ax)

    handles = [Patch(facecolor=c, edgecolor="white", label=lab)
               for c, lab in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Quadrant Depletion Analysis (* = HH depleted)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    stem = out_dir / "fig_stress_quadrant_depletion"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {stem}.pdf / .png")
    plt.close(fig)


def plot_heatmaps(df, out_dir):
    """Raw vs partial rho heatmaps (marker × dataset)."""
    def _mat(col):
        m = np.full((len(HEATMAP_MARKERS), len(DATASETS)), np.nan)
        for i, marker in enumerate(HEATMAP_MARKERS):
            for j, ds in enumerate(DATASETS):
                sub = df[(df["marker"] == marker) & (df["dataset_short"] == ds)]
                if len(sub):
                    m[i, j] = float(sub.iloc[0][col])
        return m

    raw = _mat("rho_raw")
    part = _mat("rho_partial")
    vmax = max(0.6, np.nanmax(np.abs(np.r_[raw.ravel(), part.ravel()])))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    titles = [
        r"Raw $\rho$ (coherence vs marker)",
        r"Partial $\rho$ (controlling for magnitude)",
    ]
    for ax, mat, title in zip(axes, (raw, part), titles):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS, fontsize=11)
        ax.set_yticks(range(len(HEATMAP_MARKERS)))
        ax.set_yticklabels(HEATMAP_MARKERS, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    continue
                ax.text(
                    j, i, f"{mat[i, j]:+.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(mat[i, j]) > 0.32 else "#222222",
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\rho$", rotation=0, labelpad=8)

    fig.suptitle("Stress Marker Correlations by Dataset and Modality",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    stem = out_dir / "fig_stress_modality_heatmap"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {stem}.pdf / .png")
    plt.close(fig)


def main():
    out_dir = _out_dir()
    print(f"Output dir: {out_dir}")

    partial_path = _find_csv("stress_partial_correlations.csv")
    quad_path = _find_csv("stress_quadrant_tests.csv")
    if partial_path is None and quad_path is None:
        raise FileNotFoundError(
            "Need stress_partial_correlations.csv and/or stress_quadrant_tests.csv "
            "(set SHESHA_OUT or place files in ./shesha-crispr)."
        )

    if partial_path is not None:
        print(f"Partials: {partial_path}")
        partial = _keep_main_datasets(pd.read_csv(partial_path))
        plot_partial_bars(partial, out_dir)
        plot_heatmaps(partial, out_dir)
    else:
        print("stress_partial_correlations.csv not found — skipping bars/heatmaps.")

    if quad_path is not None:
        print(f"Quadrants: {quad_path}")
        quad = _keep_main_datasets(pd.read_csv(quad_path))
        plot_quadrants(quad, out_dir)
    else:
        print("stress_quadrant_tests.csv not found — skipping quadrant figure.")


if __name__ == "__main__":
    main()
