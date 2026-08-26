#!/usr/bin/env python3
"""
fig_method_comparison.py

2×2 camera-ready figure comparing Linear vs LOESS discordance rankings
for Replogle 2022 (n≈1,832) and Norman 2019 (n≈236) datasets.

Layout
------
                  | Linear residuals | LOESS residuals |
  Replogle (n=…) |   Panel A        |   Panel B        |
  Norman   (n=…) |   Panel C        |   Panel D        |

Visual logic
------------
  Red   = top-5 discordant by the method shown in *that* column
  Blue  = top-3 concordant (lowest discordance by LOESS; same genes A/B and C/D)
  Dimmed= genes that were in the linear top-5 but dropped out of the LOESS top-5
  Dashed line  → linear fit   (Panels A, C)
  Solid  curve → LOESS curve  (Panels B, D)
  Same x/y axis limits within each row.

Data sources (mirrors fig_replogle.py and fig_norman.py)
---------------------------------------------------------
  Replogle: ./shesha-crispr/shesha_crispr_results_euclidean.csv
            + ./shesha-crispr/nonlinear_discordance_comparison.csv
            (merged on perturbation, filtered to Replogle rows)

  Norman:   pt.dt.norman_2019() via pertpy, processed with scanpy + shesha
            (same pipeline as fig_norman.py; uses locally cached dataset)
            disc_linear / disc_loess computed inline from magnitude & stability
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import warnings
from revision_io import data_search_dirs, find_data_file, resolve_out_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
try:
    import seaborn as sns
except ImportError:
    class _Sns:
        @staticmethod
        def despine(ax=None, **kwargs):
            if ax is not None:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
    sns = _Sns()
from scipy.stats import spearmanr, linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
import os


_CSV_ROOTS = data_search_dirs()
DATA_DIR = resolve_out_dir()
OUT_STEM = DATA_DIR / "fig_method_comparison"


def _find_csv(*names):
    for name in names:
        for root in _CSV_ROOTS:
            p = root / name
            if p.exists():
                return p
    return None

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})

RED       = "#d73027"
BLUE      = "#4575b4"
DIM_COL   = "#bbbbbb"
LOESS_FRAC = 0.3
N_TOP_DISC = 5
N_TOP_CONC = 3

SEED = 320
np.random.seed(SEED)

# ── discordance helpers ───────────────────────────────────────────────────────

def _disc_linear(mag: np.ndarray, stab: np.ndarray) -> np.ndarray:
    """z-score difference: high = high magnitude, low stability."""
    mag_z  = (mag  - mag.mean())  / mag.std()
    stab_z = (stab - stab.mean()) / stab.std()
    return mag_z - stab_z


def _disc_loess(mag: np.ndarray, stab: np.ndarray,
                frac: float = LOESS_FRAC) -> np.ndarray:
    """Sign-flipped, z-scored LOESS residual (below curve = discordant)."""
    fitted = lowess(stab, mag, frac=frac, return_sorted=False)
    d = -(stab - fitted)
    return (d - d.mean()) / d.std()


# ── data loading ──────────────────────────────────────────────────────────────

def _pert_col(df: pd.DataFrame) -> str:
    for c in ("perturbation", "gene", "pert", "perturbation_name"):
        if c in df.columns:
            return c
    raise KeyError(f"No perturbation column found. Columns: {list(df.columns)}")


def _alias_sp_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lower = {c.lower(): c for c in df.columns}
    if "stability" not in df.columns:
        for o in ("sp", "shesha", "coherence"):
            if o in lower:
                rename[lower[o]] = "stability"
                break
    if "magnitude" not in df.columns:
        for o in ("mp", "mag"):
            if o in lower:
                rename[lower[o]] = "magnitude"
                break
    return df.rename(columns=rename) if rename else df


def load_replogle() -> pd.DataFrame:
    """
    Mirror fig_replogle.py exactly:
      merge shesha_crispr_results_euclidean.csv (magnitude, stability)
      with   nonlinear_discordance_comparison.csv (disc_linear, disc_loess, ranks)
      filtered to Replogle rows.
    """
    euclid_csv = _find_csv(
        "shesha_crispr_results_euclidean.csv",
        "frozen_sp_scores.csv",
    )
    disc_csv = _find_csv("nonlinear_discordance_comparison.csv")
    if euclid_csv is None:
        raise FileNotFoundError(
            "Need frozen_sp_scores.csv or shesha_crispr_results_euclidean.csv"
        )
    print(f"Sp table: {euclid_csv}")

    df_euclid = pd.read_csv(euclid_csv)
    df_euclid.columns = df_euclid.columns.str.strip().str.lower()
    pc_e = _pert_col(df_euclid)
    df_euclid = df_euclid.rename(columns={pc_e: "perturbation"})
    if "dataset" in df_euclid.columns:
        df_euclid = df_euclid[df_euclid["dataset"].str.contains("Replogle", case=False, na=False)]

    if disc_csv is not None:
        df_disc = pd.read_csv(disc_csv)
        df_disc.columns = df_disc.columns.str.strip().str.lower()
        pc_d = _pert_col(df_disc)
        df_disc = df_disc.rename(columns={pc_d: "perturbation"})
        if "dataset" in df_disc.columns:
            df_disc = df_disc[df_disc["dataset"].str.contains("Replogle", case=False, na=False)]
        df = df_euclid.merge(df_disc, on="perturbation", how="inner",
                             suffixes=("", "_disc"))
    else:
        df = df_euclid.copy()
    df = df.set_index("perturbation")
    df = _alias_sp_cols(df)
    df = df.dropna(subset=["magnitude", "stability"])

    # Recompute inline if the disc CSV didn't include these columns
    if "disc_linear" not in df.columns:
        df["disc_linear"] = _disc_linear(df["magnitude"].values,
                                          df["stability"].values)
    if "disc_loess" not in df.columns:
        df["disc_loess"] = _disc_loess(df["magnitude"].values,
                                        df["stability"].values)

    print(f"Replogle: {len(df):,} perturbations after merge")
    return df


def load_norman() -> pd.DataFrame:
    """
    Same CSV merge as load_replogle(), filtered to Norman rows instead.
    Both shesha_crispr_results_euclidean.csv and
    nonlinear_discordance_comparison.csv contain Norman data.
    """
    euclid_csv = _find_csv(
        "shesha_crispr_results_euclidean.csv",
        "frozen_sp_scores.csv",
    )
    disc_csv = _find_csv("nonlinear_discordance_comparison.csv")
    if euclid_csv is None:
        raise FileNotFoundError(
            "Need frozen_sp_scores.csv or shesha_crispr_results_euclidean.csv"
        )

    df_euclid = pd.read_csv(euclid_csv)
    df_euclid.columns = df_euclid.columns.str.strip().str.lower()
    pc_e = _pert_col(df_euclid)
    df_euclid = df_euclid.rename(columns={pc_e: "perturbation"})
    if "dataset" in df_euclid.columns:
        df_euclid = df_euclid[df_euclid["dataset"].str.contains("Norman", case=False, na=False)]

    if disc_csv is not None:
        df_disc = pd.read_csv(disc_csv)
        df_disc.columns = df_disc.columns.str.strip().str.lower()
        pc_d = _pert_col(df_disc)
        df_disc = df_disc.rename(columns={pc_d: "perturbation"})
        if "dataset" in df_disc.columns:
            df_disc = df_disc[df_disc["dataset"].str.contains("Norman", case=False, na=False)]
        df = df_euclid.merge(df_disc, on="perturbation", how="inner",
                             suffixes=("", "_disc"))
    else:
        df = df_euclid.copy()
    df = df.set_index("perturbation")
    df = _alias_sp_cols(df)
    df = df.dropna(subset=["magnitude", "stability"])

    if "disc_linear" not in df.columns:
        df["disc_linear"] = _disc_linear(df["magnitude"].values,
                                          df["stability"].values)
    if "disc_loess" not in df.columns:
        df["disc_loess"] = _disc_loess(df["magnitude"].values,
                                        df["stability"].values)

    print(f"Norman: {len(df):,} perturbations after merge")
    return df


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_rep  = load_replogle()
    df_norm = load_norman()
    return df_rep, df_norm


# ── gene selection ────────────────────────────────────────────────────────────

def top_disc(df: pd.DataFrame, col: str, n: int) -> list[str]:
    return df[col].nlargest(n).index.tolist()


def top_conc(df: pd.DataFrame, col: str, n: int) -> list[str]:
    return df[col].nsmallest(n).index.tolist()


# ── annotation helpers ────────────────────────────────────────────────────────

# Predefined label offsets (dx, dy) in points for known key genes.
# Positive x → right, positive y → up.
_OFFSETS: dict[str, tuple[float, float]] = {
    # Replogle linear discordant
    "GATA1":  ( 35,  10),
    "CHMP3":  (-40,  12),
    "AQR":    ( 30, -15),
    "PSMD7":  ( 30,  10),
    "PSMD6":  (-35,  10),
    # Replogle LOESS discordant
    "CHMP2A": ( 38,  10),
    "SF3B3":  ( 32, -16),
    "SF3B2":  (-40,  12),
    # Replogle concordant
    "CASP8AP2": ( 42,   8),
    "CHAF1B":   (-45,   8),
    "LSG1":     ( 30, -16),
    # Norman linear discordant
    "CEBPA+JUN":   ( 50,  10),
    "CEBPA":       ( 38,  10),
    "CEBPA+CEBPB": (-60,  10),
    "CEBPB+JUN":   ( 45, -14),
    "CEBPE+SPI1":  ( 40,  10),
    # Norman LOESS discordant
    "PLK4+STIL":      ( 48,  10),
    "HES7":           ( 30, -16),
    "C3orf72+FOXL2":  (-65,  12),
    "STIL+PLK4":      ( 48,  10),
    # Norman concordant
    "KLF1": ( 30, -16),
}


def _auto_offset(x: float, y: float,
                 xlim: tuple, ylim: tuple) -> tuple[float, float]:
    """Simple quadrant-based offset when gene not in _OFFSETS."""
    x_frac = (x - xlim[0]) / (xlim[1] - xlim[0])
    y_frac = (y - ylim[0]) / (ylim[1] - ylim[0])
    dx = -38 if x_frac > 0.6 else 38
    dy =  12 if y_frac < 0.5 else -16
    return dx, dy


def _annotate(ax, df: pd.DataFrame, genes: list[str], color: str,
              xlim: tuple, ylim: tuple,
              fontstyle: str = "normal", fontsize: float = 7.5) -> None:
    for gene in genes:
        if gene not in df.index:
            continue
        x, y = df.loc[gene, ["magnitude", "stability"]]
        dx, dy = _OFFSETS.get(gene, _auto_offset(x, y, xlim, ylim))
        ax.annotate(
            gene,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold" if fontstyle == "normal" else "normal",
            fontstyle=fontstyle,
            color=color,
            ha="center",
            va="center",
            zorder=6,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.7,
                shrinkA=0,
                shrinkB=3,
            ),
        )


# ── panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(
    ax,
    df: pd.DataFrame,
    method: str,           # "linear" | "loess"
    red_genes: list[str],
    blue_genes: list[str],
    dim_genes: list[str],
    xlim: tuple,
    ylim: tuple,
    title: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:

    red_set  = set(red_genes)
    blue_set = set(blue_genes)
    dim_set  = set(dim_genes)

    is_red  = df.index.isin(red_set)
    is_blue = df.index.isin(blue_set)
    is_dim  = df.index.isin(dim_set) & ~is_red & ~is_blue
    is_bg   = ~is_red & ~is_blue & ~is_dim

    mag  = df["magnitude"].values
    stab = df["stability"].values

    # ── scatter layers (back → front) ─────────────────────────────
    ax.scatter(df.loc[is_bg,  "magnitude"], df.loc[is_bg,  "stability"],
               c="lightgray", s=10, alpha=0.30, edgecolor="none", zorder=1)

    if dim_set:
        ax.scatter(df.loc[is_dim, "magnitude"], df.loc[is_dim, "stability"],
                   c=DIM_COL, s=28, alpha=0.55, edgecolor="none", zorder=2)

    # ── fit line / curve ───────────────────────────────────────────
    if method == "linear":
        slope, intercept, *_ = linregress(mag, stab)
        x_line = np.linspace(xlim[0], xlim[1], 300)
        ax.plot(x_line, slope * x_line + intercept,
                "--", color="black", linewidth=1.4, alpha=0.85, zorder=3)
    else:
        lw = lowess(stab, mag, frac=LOESS_FRAC, return_sorted=True)
        ax.plot(lw[:, 0], lw[:, 1],
                "-", color="black", linewidth=1.8, alpha=0.90, zorder=3)

    # ── highlighted points ─────────────────────────────────────────
    ax.scatter(df.loc[is_red,  "magnitude"], df.loc[is_red,  "stability"],
               c=RED,  s=50, alpha=0.92, edgecolor="white", linewidth=0.5, zorder=4)
    ax.scatter(df.loc[is_blue, "magnitude"], df.loc[is_blue, "stability"],
               c=BLUE, s=50, alpha=0.92, edgecolor="white", linewidth=0.5, zorder=4)

    # ── annotations ───────────────────────────────────────────────
    _annotate(ax, df, red_genes,  RED,     xlim, ylim)
    _annotate(ax, df, blue_genes, BLUE,    xlim, ylim)
    _annotate(ax, df, [g for g in dim_set if g in df.index],
              DIM_COL, xlim, ylim, fontstyle="italic", fontsize=7.0)

    # ── rho label ──────────────────────────────────────────────────
    rho, _ = spearmanr(mag, stab)
    ax.text(0.03, 0.97,
            f"$\\rho$ = {rho:.3f}\n$n$ = {len(df):,}",
            transform=ax.transAxes, fontsize=8,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#CCCCCC", alpha=0.88))

    # ── formatting ─────────────────────────────────────────────────
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show_xlabel:
        ax.set_xlabel("Effect magnitude (Euclidean)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Shesha Coherence (cosine)", fontsize=9)
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_rep, df_norm = load_datasets()
    df_rep  = df_rep.dropna(subset=["magnitude", "stability"])
    df_norm = df_norm.dropna(subset=["magnitude", "stability"])

    print(f"Replogle: {len(df_rep):,} perturbations")
    print(f"Norman:   {len(df_norm):,} perturbations")

    # ── select genes ───────────────────────────────────────────────
    rep_lin_red  = top_disc(df_rep,  "disc_linear", N_TOP_DISC)
    rep_loe_red  = top_disc(df_rep,  "disc_loess",  N_TOP_DISC)
    rep_blue     = top_conc(df_rep,  "disc_loess",  N_TOP_CONC)
    # genes that were linear-flagships but disappeared from LOESS top
    rep_dim_B    = [g for g in rep_lin_red if g not in rep_loe_red]

    norm_lin_red = top_disc(df_norm, "disc_linear", N_TOP_DISC)
    norm_loe_red = top_disc(df_norm, "disc_loess",  N_TOP_DISC)
    norm_blue    = top_conc(df_norm, "disc_loess",  N_TOP_CONC)
    norm_dim_D   = [g for g in norm_lin_red if g not in norm_loe_red]

    print("\nReplogle linear  top-5:", rep_lin_red)
    print("Replogle LOESS   top-5:", rep_loe_red)
    print("Replogle concordant:   ", rep_blue)
    print("\nNorman   linear  top-5:", norm_lin_red)
    print("Norman   LOESS   top-5:", norm_loe_red)
    print("Norman   concordant:   ", norm_blue)

    # ── shared axis limits ─────────────────────────────────────────
    def _lims(df: pd.DataFrame, pad: float = 0.04):
        xlo, xhi = df["magnitude"].min(), df["magnitude"].max()
        ylo, yhi = df["stability"].min(), df["stability"].max()
        xp = (xhi - xlo) * pad
        yp = (yhi - ylo) * pad
        return (xlo - xp, xhi + xp), (ylo - yp, yhi + yp)

    rep_xlim,  rep_ylim  = _lims(df_rep)
    norm_xlim, norm_ylim = _lims(df_norm)

    # ── build figure ───────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.10, right=0.97,
        top=0.92,  bottom=0.10,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    common = dict(show_xlabel=True, show_ylabel=True)

    # Panel A — Replogle × Linear
    draw_panel(
        axes[0][0], df_rep,
        method="linear",
        red_genes=rep_lin_red, blue_genes=rep_blue, dim_genes=[],
        xlim=rep_xlim, ylim=rep_ylim,
        title="Linear residuals",
        **common,
    )

    # Panel B — Replogle × LOESS
    draw_panel(
        axes[0][1], df_rep,
        method="loess",
        red_genes=rep_loe_red, blue_genes=rep_blue, dim_genes=rep_dim_B,
        xlim=rep_xlim, ylim=rep_ylim,
        title="LOESS residuals",
        **common,
    )

    # Panel C — Norman × Linear
    draw_panel(
        axes[1][0], df_norm,
        method="linear",
        red_genes=norm_lin_red, blue_genes=norm_blue, dim_genes=[],
        xlim=norm_xlim, ylim=norm_ylim,
        title="Linear residuals",
        **common,
    )

    # Panel D — Norman × LOESS
    draw_panel(
        axes[1][1], df_norm,
        method="loess",
        red_genes=norm_loe_red, blue_genes=norm_blue, dim_genes=norm_dim_D,
        xlim=norm_xlim, ylim=norm_ylim,
        title="LOESS residuals",
        **common,
    )

    # ── panel letters ──────────────────────────────────────────────
    for (r, c), letter in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
        axes[r][c].text(
            -0.13, 1.09, letter,
            transform=axes[r][c].transAxes,
            fontsize=14, fontweight="bold", va="top",
        )

    # ── row labels (left margin) ───────────────────────────────────
    n_rep  = len(df_rep)
    n_norm = len(df_norm)

    for row_idx, label in enumerate(
        [f"Replogle\n($n$={n_rep:,})", f"Norman\n($n$={n_norm:,})"]
    ):
        # vertical centre of each row in figure coordinates
        y_top = gs.get_position(fig).y1 if hasattr(gs, "get_position") else None
        # fallback: manually placed
        y_pos = 0.71 if row_idx == 0 else 0.29
        fig.text(
            0.025, y_pos,
            label,
            va="center", ha="center",
            fontsize=11, fontweight="bold",
            rotation=90,
        )

    # ── legend ────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=RED, markersize=7,
               label="Top-5 discordant (this method)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BLUE, markersize=7,
               label="Top-3 concordant (LOESS-robust)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=DIM_COL, markersize=7,
               label="Linear-only discordant (dimmed in LOESS panel)"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.3,
               label="Linear fit"),
        Line2D([0], [0], linestyle="-",  color="black", linewidth=1.8,
               label="LOESS curve"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=5,
        fontsize=8, framealpha=0.92,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.53, 0.01),
    )

    # ── super-title ────────────────────────────────────────────────
    fig.suptitle(
        "Linear vs LOESS discordance rankings: Replogle and Norman",
        fontsize=13, fontweight="bold", y=0.975,
    )

    # ── save ───────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        out = f"{OUT_STEM}.{ext}"
        kwargs: dict = dict(dpi=300, bbox_inches="tight")
        if ext == "png":
            kwargs["facecolor"] = "white"
        plt.savefig(out, **kwargs)
        print(f"Saved → {out}")

    plt.show()


if __name__ == "__main__":
    main()
