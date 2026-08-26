#!/usr/bin/env python3
"""
Curved trajectory counterexample.

Simulate 500 cells on a curved (near-circular) trajectory in 50D space with
Gaussian noise. All cells follow the same reproducible program, but Sp is low
because the cloud mean cancels and mean-shift geometry misses the manifold.

Outputs a schematic figure + numeric summary for the Limitations section.

Usage:
  python curved_trajectory_counterexample.py
  python curved_trajectory_counterexample.py --out-dir /content/shesha-crispr
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

import matplotlib.pyplot as plt
import numpy as np

from pipeline_core import calculate_sp
from revision_io import resolve_out_dir


def simulate_arc(n_cells=500, n_dims=50, noise=0.08, seed=0, radius=3.0):
    """
    Cells on a near-full circle (shared curved program). The cloud mean sits
    near the origin, so mean-shift Sp collapses even though every cell follows
    the same reproducible trajectory.
    """
    rng = np.random.default_rng(seed)
    # Avoid exact full-circle degeneracy; leave a small gap
    t = rng.uniform(0.05 * np.pi, 1.95 * np.pi, size=n_cells)
    X = rng.normal(0, noise, size=(n_cells, n_dims))
    X[:, 0] += radius * np.cos(t)
    X[:, 1] += radius * np.sin(t)
    X_ctrl = rng.normal(0, noise, size=(n_cells, n_dims))
    return X_ctrl, X, t


def simulate_linear(n_cells=500, n_dims=50, noise=0.08, seed=1, distance=3.0):
    """Same noise budget, cells along a straight ray (high Sp expected)."""
    rng = np.random.default_rng(seed)
    # Keep all cells clearly displaced along one axis
    t = rng.uniform(0.7 * distance, 1.3 * distance, size=n_cells)
    direction = np.zeros(n_dims)
    direction[0] = 1.0
    X = rng.normal(0, noise, size=(n_cells, n_dims))
    X += t[:, None] * direction
    X_ctrl = rng.normal(0, noise, size=(n_cells, n_dims))
    return X_ctrl, X, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--n-dims", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)

    Xc_arc, Xp_arc, t_arc = simulate_arc(
        args.n_cells, args.n_dims, args.noise, args.seed
    )
    Xc_lin, Xp_lin, t_lin = simulate_linear(
        args.n_cells, args.n_dims, args.noise, args.seed + 1
    )

    sp_arc = calculate_sp(Xc_arc, Xp_arc)
    sp_lin = calculate_sp(Xc_lin, Xp_lin)

    # Within-arc: split by t and show both halves have similar structure
    # but Sp still low — the point of the counterexample
    mid = np.median(t_arc)
    half_a = Xp_arc[t_arc <= mid]
    half_b = Xp_arc[t_arc > mid]
    # Direction of each half mean vs control
    mu_c = Xc_arc.mean(axis=0)
    d_a = half_a.mean(axis=0) - mu_c
    d_b = half_b.mean(axis=0) - mu_c
    cos_halves = float(
        np.dot(d_a, d_b) / (np.linalg.norm(d_a) * np.linalg.norm(d_b) + 1e-12)
    )

    summary = {
        "n_cells": args.n_cells,
        "n_dims": args.n_dims,
        "noise": args.noise,
        "sp_curved_arc": float(sp_arc["stability"]),
        "sp_linear_ray": float(sp_lin["stability"]),
        "cosine_arc_half_means": cos_halves,
        "interpretation": (
            "Curved trajectory: cells share one reproducible program but Sp is "
            "low because opposing positions cancel in the mean shift. "
            "Linear ray: same noise budget, high Sp. Limitation: Sp is not a "
            "general measure of program reproducibility under strongly nonlinear "
            "geometry."
        ),
    }

    print(f"Sp (curved traj): {summary['sp_curved_arc']:.4f}")
    print(f"Sp (linear ray):  {summary['sp_linear_ray']:.4f}")
    print(f"cos(half means):  {cos_halves:.4f}")

    # Schematic: PC1–PC2 of arc vs linear
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
    for ax, Xp, Xc, title, sp in [
        (axes[0], Xp_arc, Xc_arc, "Curved trajectory (low Sp)", sp_arc["stability"]),
        (axes[1], Xp_lin, Xc_lin, "Linear ray (high Sp)", sp_lin["stability"]),
    ]:
        ax.scatter(Xc[:, 0], Xc[:, 1], s=6, alpha=0.35, c="#888888", label="control")
        ax.scatter(Xp[:, 0], Xp[:, 1], s=6, alpha=0.45, c="#1f4e79", label="perturbed")
        mu_c = Xc.mean(0)
        mu_p = Xp.mean(0)
        ax.annotate(
            "",
            xy=(mu_p[0], mu_p[1]),
            xytext=(mu_c[0], mu_c[1]),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
        )
        ax.set_title(f"{title}\nSp = {sp:.3f}")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(markerscale=2, fontsize=8, frameon=False)
    fig.suptitle(
        "Limitation: shared curved program ≠ high Sp\n"
        "(mean-shift geometry misses nonlinear trajectories)",
        fontsize=11,
    )
    fig_path = out_dir / "curved_trajectory_counterexample.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    with open(out_dir / "curved_trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Methods / limitations blurb
    blurb = (
        "Limitation (curved trajectories). Sp summarizes mean-shift directional "
        f"coherence. In a simulation of {args.n_cells} cells along a curved arc in "
        f"{args.n_dims}-dimensional space, Sp was {summary['sp_curved_arc']:.2f} "
        f"despite a single shared program, versus {summary['sp_linear_ray']:.2f} for "
        "a linear ray with matched noise. Sp is therefore not a general measure of "
        "program reproducibility under strongly nonlinear geometry."
    )
    with open(out_dir / "curved_trajectory_methods_blurb.txt", "w") as f:
        f.write(blurb + "\n")

    print(f"Wrote {fig_path}")
    print(f"Wrote {out_dir}/curved_trajectory_summary.json")


if __name__ == "__main__":
    main()
