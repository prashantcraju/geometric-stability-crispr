#!/usr/bin/env python3
"""
Supplementary Figure S8 — Theoretical null model (CSV only).

Loads the stored isotropic-Gaussian simulation and plots magnitude and SNR
against Shesha Coherence in the same density style as fig_1.py.

    python fig_s8_null_model.py
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

from revision_io import find_data_file, resolve_out_dir


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress, gaussian_kde
import matplotlib.pyplot as plt

def _find_csv():
    return find_data_file("null_model_simulation.csv")


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _density(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.ones(len(x))
    try:
        return gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    except Exception:
        return np.ones(len(x))


def _density_panel(ax, x, y, xlabel, ylabel, title, cmap_name):
    z = _density(x, y)
    order = np.argsort(z)
    sc = ax.scatter(
        x[order], y[order], c=z[order], cmap=cmap_name,
        s=28, alpha=0.8, edgecolor="white", linewidth=0.4, zorder=2,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Perturbation Density", rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=8)

    slope, intercept, *_ = linregress(x, y)
    x_line = np.array([np.nanmin(x), np.nanmax(x)])
    ax.plot(x_line, slope * x_line + intercept, "--",
            color="gray", linewidth=2, alpha=0.7, zorder=1)

    rho, pval = spearmanr(x, y)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(
        0.97, 0.03, f"$\\rho$ = {rho:.3f}",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9),
    )
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    _despine(ax)
    return rho, pval


def _partial_spearman(x, y, z):
    """Rank residual Spearman of x vs y after linear adjustment for z."""
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    rz = pd.Series(z).rank().to_numpy(dtype=float)
    sx, ix, *_ = linregress(rz, rx)
    sy, iy, *_ = linregress(rz, ry)
    return spearmanr(rx - (sx * rz + ix), ry - (sy * rz + iy))


def main():
    path = _find_csv()
    if path is None:
        raise FileNotFoundError(
            "Could not find null_model_simulation.csv in SHESHA_OUT or ./shesha-crispr."
        )
    print(f"Null model CSV: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, opts in {
        "observed_magnitude": ("observed_magnitude", "magnitude", "mp"),
        "stability": ("stability", "sp", "coherence"),
        "snr": ("snr",),
        "sigma": ("sigma", "noise"),
    }.items():
        if canon not in df.columns:
            for o in opts:
                if o in lower:
                    rename[lower[o]] = canon
                    break
    if rename:
        df = df.rename(columns=rename)

    need = ["observed_magnitude", "stability"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {list(df.columns)}")

    df = df.dropna(subset=need).copy()
    print(f"Rows: {len(df)}")
    if "sigma" in df.columns:
        print("sigma levels:", sorted(df["sigma"].unique().tolist()))

    mag = df["observed_magnitude"].to_numpy(dtype=float)
    stab = df["stability"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rho_mag, p_mag = _density_panel(
        axes[0], mag, stab,
        "Observed Magnitude", "Shesha Coherence",
        "Null: Magnitude vs Coherence",
        "Greys",
    )

    if "snr" in df.columns:
        snr = df["snr"].to_numpy(dtype=float)
        rho_snr, p_snr = _density_panel(
            axes[1], snr, stab,
            "Signal-to-noise (true mag / $\\sigma$)", "Shesha Coherence",
            "Null: SNR vs Coherence",
            "Greys",
        )
        rho_part, p_part = _partial_spearman(mag, stab, snr)
    else:
        axes[1].text(
            0.5, 0.5, "No SNR column in CSV",
            transform=axes[1].transAxes, ha="center", va="center", color="gray",
        )
        _despine(axes[1])
        rho_snr = p_snr = rho_part = p_part = np.nan

    fig.suptitle(
        "Theoretical null model (isotropic Gaussian shift)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_dir = resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig_s8_null_model"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}.pdf / .png")

    caption = (
        r"\caption{\textbf{Theoretical null model under isotropic Gaussian perturbations.} "
        r"Left: magnitude versus Shesha coherence. Right: the same simulations versus SNR. "
        r"2{,}000 simulated perturbations ($d = 50$ dimensions, "
        r"$\sigma \in \{0.5, 1.0, 2.0, 3.0\}$, 500 simulations per noise level). "
        r"Under the null model, coherence is almost perfectly predicted by SNR "
        r"($\rho = 0.999$), with a partial correlation of "
        r"$\rho_{\text{partial}} = 0.292$ after controlling for SNR. "
        r"The heterogeneity observed in real data "
        r"(Norman $\rho_{\text{partial}} = -0.859$, "
        r"Dixit $\rho_{\text{partial}} = +0.627$) far exceeds this null prediction, "
        r"confirming that biological factors beyond simple SNR confounding drive "
        r"the magnitude--coherence relationship.}"
    )
    cap_path = out_dir / "fig_s8_null_model_caption.txt"
    cap_path.write_text(caption + "\n")
    print(f"Caption -> {cap_path}")

    print("\n--- Stored-null Spearman ---")
    print(f"rho(magnitude, coherence) = {rho_mag:+.3f}  p = {p_mag:.3e}")
    if np.isfinite(rho_snr):
        print(f"rho(SNR, coherence)       = {rho_snr:+.3f}  p = {p_snr:.3e}")
        print(f"partial rho(mag, coherence | SNR) = {rho_part:+.3f}  p = {p_part:.3e}")
    plt.show()


if __name__ == "__main__":
    main()
