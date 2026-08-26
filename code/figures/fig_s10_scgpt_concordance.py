#!/usr/bin/env python3
"""
S10 — PCA vs scGPT per-perturbation coherence concordance.

Four panels with the Figure 2 density cmaps (no colorbars). Papalexi (n=24)
and the Adamson pilot (n=8) are omitted — same thin-n scGPT rule as Figure 4.
Dashed identity line only (not a fitted trend).

USAGE:
    python fig_s10_scgpt_concordance.py
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import os
from pathlib import Path


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import pipeline_config as cfg
from revision_io import data_search_dirs, find_data_file, load_sp_table
from fig_1 import perturbation_density

SEED = cfg.SEED
np.random.seed(SEED)

SCGPT_DIRS = data_search_dirs()
PCA_DIRS = data_search_dirs()

# Same thin-n omit as Fig 4: no Papalexi, no Adamson pilot.
PANELS = [
    ("Norman 2019 (CRISPRa)",        "Norman 2019",        "CRISPRa",   "Blues"),
    ("Adamson 2016 UPR (CRISPRi)",   "Adamson 2016 UPR",   "CRISPRi",   "GnBu"),
    ("Dixit 2016 (CRISPR-KO)",       "Dixit 2016",         "CRISPR-KO", "Greens"),
    ("Replogle 2022 (CRISPRi)",      "Replogle 2022",      "CRISPRi",   "Reds"),
]


def _first_existing(*paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_scgpt():
    combined = _first_existing(*(d / "scgpt_all_datasets.csv" for d in SCGPT_DIRS))
    if combined is not None:
        df = pd.read_csv(combined)
        print(f"scGPT: {combined}  ({len(df)} rows)")
        return df

    frames = []
    for d in SCGPT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("scgpt_*.csv")):
            if p.name in {
                "scgpt_all_datasets.csv",
                "scgpt_all_datasets_zscored.csv",
                "scgpt_correlations.csv",
                "scgpt_vs_frozen_concordance.csv",
            }:
                continue
            frames.append(pd.read_csv(p))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            print(f"scGPT: concatenated {len(frames)} files under {d}")
            return df
    raise FileNotFoundError(
        "Need scgpt_all_datasets.csv in SHESHA_OUT or ./shesha-crispr"
    )


def load_pca():
    path = _first_existing(
        *(d / "frozen_sp_scores.csv" for d in PCA_DIRS),
        *(d / "shesha_crispr_results_euclidean.csv" for d in PCA_DIRS),
    )
    if path is None:
        raise FileNotFoundError("Need frozen_sp_scores.csv")
    print(f"PCA Sp: {path}")
    return load_sp_table(path)


def merge_concordance(df_scgpt, df_pca):
    sc = df_scgpt.copy()
    sc["dataset"] = sc["dataset"].map(cfg.resolve_dataset_name)
    sc["perturbation"] = sc["perturbation"].astype(str).str.strip()
    sc = sc.rename(columns={"stability": "sp_scgpt", "magnitude": "mag_scgpt"})

    pc = df_pca.copy()
    pc["dataset"] = pc["dataset"].map(cfg.resolve_dataset_name)
    pc["perturbation"] = pc["perturbation"].astype(str).str.strip()
    pc = pc.rename(columns={"stability": "sp_pca", "magnitude": "mag_pca"})

    m = sc.merge(
        pc[["dataset", "perturbation", "sp_pca", "mag_pca"]],
        on=["dataset", "perturbation"],
        how="inner",
    )
    m = m.dropna(subset=["sp_pca", "sp_scgpt"])
    print("Shared perturbations:")
    print(m.groupby("dataset").size().to_string())
    return m


def plot_panel(ax, sub, title, cmap_name, y_lo):
    n = len(sub)
    if n < 3:
        ax.text(0.5, 0.5, f"{title}\n(no data)",
                transform=ax.transAxes, ha="center", va="center", color="gray")
        _despine(ax)
        return None

    x = sub["sp_pca"].to_numpy(dtype=float)
    y = sub["sp_scgpt"].to_numpy(dtype=float)
    z = perturbation_density(x, y)
    order = np.argsort(z)
    large = n > 400
    ax.scatter(
        x[order], y[order], c=z[order], cmap=cmap_name,
        s=22 if large else 40, alpha=0.8,
        edgecolor="white", linewidth=0.35 if large else 0.5,
        rasterized=large, zorder=2,
    )
    lo = min(-0.08, float(np.nanmin(x)) - 0.03, y_lo)
    ax.plot([lo, 1.0], [lo, 1.0], "--", color="gray", linewidth=1.2,
            alpha=0.7, zorder=1)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(y_lo, 1.0)
    if y_lo < 0 or float(np.nanmin(x)) < 0:
        ax.axhline(0.0, color="#B0B0B0", linewidth=0.7, zorder=1)
        ax.axvline(0.0, color="#B0B0B0", linewidth=0.7, zorder=1)

    rho, _ = spearmanr(x, y)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(
        0.97, 0.03, f"$\\rho$ = {rho:.3f}\n$n$ = {n}",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9),
    )
    ax.set_xlabel("Shesha Coherence (PCA)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Shesha Coherence (scGPT)", fontsize=10, fontweight="bold")
    _despine(ax)
    return rho


def main():
    df_scgpt = load_scgpt()
    df_pca = load_pca()
    merged = merge_concordance(df_scgpt, df_pca)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.6))
    axes = axes.ravel()

    y_lo = 0.0
    for ds_full, *_ in PANELS:
        sub = merged[merged["dataset"] == ds_full]
        if len(sub):
            y_lo = min(y_lo, float(sub["sp_scgpt"].min()), float(sub["sp_pca"].min()))
    y_lo = min(-0.08, y_lo - 0.03)

    print("\nConcordance (scGPT vs frozen PCA Sp):")
    for ax, (ds_full, ds_short, modality, cmap_name) in zip(axes, PANELS):
        sub = merged[merged["dataset"] == ds_full]
        rho = plot_panel(ax, sub, f"{ds_short}\n({modality})", cmap_name, y_lo)
        if rho is not None:
            print(f"  {ds_short}: n={len(sub)}  ρ={rho:.3f}")

    for ax, letter in zip(axes, "abcd"):
        ax.text(-0.08, 1.08, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    fig.suptitle("PCA vs scGPT coherence concordance",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_dir = _first_existing(
        Path("./shesha-crispr"),
    ) or Path("./shesha-crispr")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_s10_scgpt_concordance"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {stem}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
