#!/usr/bin/env python3
"""
SUPERSEDED. Old Figure 5 / S9 / S20 family. Pre-freeze numbers;
panel b applied abs() to the HSPA5 partial. Do not use as manuscript art.

Replacement: fig5_pathway_forest.py (Apoptosis, p53, DDIT3).
S9 stress forest: fig_s9_stress_forest.py. Quadrants deleted with S20.

3-panel figure (historical):
  a) HSPA5 vs stability scatter in Replogle with quadrant annotation
  b) HSPA5 vs stability scatter in Dixit with quadrant annotation
  c) Raw vs partial rho dot plot (from stress CSVs, no pertpy needed)

Panels a/b: prefer a stress-enriched Sp CSV (stress_HSPA5). Otherwise extract
HSPA5 only and merge frozen Sp — never recompute Shesha on full Replogle.
Panel c: reads from stress_partial_correlations.csv
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
from matplotlib.lines import Line2D
from pathlib import Path

import pipeline_config as cfg
from revision_io import data_search_dirs, load_sp_table

SEED = cfg.SEED
np.random.seed(SEED)

DATA_DIR = cfg.OUTPUT_DIR
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_CELLS = cfg.MIN_CELLS

# Colors
RED = '#C44E52'
BLUE = '#4C72B0'
GREEN = '#2CA02C'
DARK_GRAY = '#555555'
HSPA5_COLOR = '#8B0000'  # dark red for HSPA5 emphasis


def _first_existing(*candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    return None


def _search_csvs(*names):
    roots = data_search_dirs()
    found = []
    for name in names:
        for root in roots:
            p = root / name
            if p.exists() and p not in found:
                found.append(p)
    return found


def load_frozen_with_hspa5(dataset_full: str):
    """Use a stress-enriched Sp table if present (no expression reload)."""
    for path in _search_csvs(
        "shesha_crispr_results_euclidean.csv",
        "frozen_sp_scores.csv",
    ):
        df = load_sp_table(path)
        df["dataset"] = df["dataset"].map(cfg.resolve_dataset_name)
        sub = df[df["dataset"] == dataset_full].copy()
        if not len(sub):
            continue
        hspa5_col = next(
            (c for c in ("stress_HSPA5", "hspa5", "HSPA5", "mean_HSPA5")
             if c in sub.columns and sub[c].notna().sum() >= 3),
            None,
        )
        if hspa5_col is None:
            continue
        out = sub.rename(columns={hspa5_col: "hspa5"})
        print(f"  {dataset_full}: HSPA5+Sp from {path}  (n={len(out)})")
        return out
    return None


def _hspa5_means(adata, pert_col, ctrl_label, min_cells):
    """Mean log-normalized HSPA5 per perturbation without copying the full matrix."""
    gene = "HSPA5" if "HSPA5" in adata.var_names else None
    if gene is None:
        upper = {str(g).upper(): g for g in adata.var_names}
        gene = upper.get("HSPA5")
    if gene is None:
        print("  WARNING: HSPA5 not found!")
        return {}

    labels = adata.obs[pert_col].astype(str)
    counts = labels.value_counts()
    valid = [p for p in counts[counts >= min_cells].index if p != ctrl_label]

    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    expr = np.asarray(x, dtype=float).ravel()

    already_log = bool(adata.uns.get("log1p")) or float(np.nanmax(expr)) < 20
    if not already_log:
        lib = None
        for col in ("n_counts", "total_counts", "nUMI", "n_umi"):
            if col in adata.obs.columns:
                lib = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=float)
                break
        if lib is None:
            print("  note: no n_counts in obs; using raw HSPA5 (no library-size norm)")
        else:
            lib = np.where(lib > 0, lib, 1.0)
            expr = np.log1p(expr * (1e4 / lib))

    out = {}
    for pert in valid:
        mask = labels.to_numpy() == pert
        if mask.any():
            out[pert] = float(expr[mask].mean())
    print(f"  HSPA5 extracted for {len(out)} perturbations (single-gene, no PCA)")
    return out


def extract_hspa5_backed(dataset_full, min_cells=None):
    """HSPA5 from a backed h5ad (one gene column). Coherence from frozen Sp.

    Never calls pertpy.dt / to_memory() on the full Replogle matrix.
    """
    if min_cells is None:
        min_cells = MIN_CELLS
    from pipeline_core import _extract_adata, load_raw, setup_cache
    import scanpy as sc

    frozen_path = _first_existing(*_search_csvs(
        "frozen_sp_scores.csv",
        "shesha_crispr_results_euclidean.csv",
    ))
    if frozen_path is None:
        raise FileNotFoundError(
            "Need frozen_sp_scores.csv — fig3 does not recompute Shesha."
        )
    frozen = load_sp_table(frozen_path)
    frozen["dataset"] = frozen["dataset"].map(cfg.resolve_dataset_name)
    frozen = frozen[frozen["dataset"] == dataset_full].copy()
    print(f"  frozen Sp: {frozen_path}  (n={len(frozen)})")

    setup_cache()
    print(f"  Opening {dataset_full} backed (HSPA5 column only)…")
    raw = load_raw(dataset_full, prefer_local=True)
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset_full, sc)
    n_obs = int(adata.n_obs)
    backed = bool(getattr(adata, "isbacked", False))
    print(f"  adata: {n_obs} cells  backed={backed}")
    if not backed and n_obs > 80_000:
        raise RuntimeError(
            f"{dataset_full} was loaded fully into RAM ({n_obs} cells). "
            "That will OOM on Colab. Put the Replogle h5ad in the pertpy cache "
            "so load_raw opens it with backed='r', or run attach_stress_markers.py "
            "once and re-run fig3 from the enriched CSV."
        )
    hspa5_map = _hspa5_means(adata, pert_col, ctrl_label, min_cells)
    if getattr(adata, "file", None) is not None:
        try:
            adata.file.close()
        except Exception:
            pass
    del adata, raw

    out = frozen[["perturbation", "stability", "magnitude"]].copy()
    out["hspa5"] = out["perturbation"].astype(str).map(hspa5_map)
    return out


def panel_stress_scatter(ax, df, dataset_name, raw_rho, partial_rho,
                          hh_obs, hh_exp, hh_p_str):
    """Panels a/b: HSPA5 vs stability scatter with quadrant annotation."""
    sub = df.dropna(subset=['hspa5']).copy()
    x = sub['stability'].values
    y = sub['hspa5'].values

    # Scatter
    ax.scatter(x, y, c=BLUE, s=40, alpha=0.5, edgecolor='white', linewidth=0.3)

    # Regression + CI
    slope, intercept, _, _, _ = linregress(x, y)
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_pred + intercept
    n = len(x)
    mean_x = np.mean(x)
    se_y = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
    se_pred = se_y * np.sqrt(1/n + (x_pred - mean_x)**2 / np.sum((x - mean_x)**2))
    ci = 1.96 * se_pred
    ax.fill_between(x_pred, y_pred - ci, y_pred + ci, color='gray', alpha=0.15)
    ax.plot(x_pred, y_pred, color=RED, linewidth=2)

    # Quadrant lines
    med_x, med_y = np.median(x), np.median(y)
    ax.axvline(med_x, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(med_y, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    # HH depletion annotation (top-right quadrant)
    ax.text(0.97, 0.97,
            f'HH: {hh_obs} / {hh_exp:.0f} exp.\np = {hh_p_str}',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3F3',
                      edgecolor=RED, alpha=0.9))

    # Correlation annotations
    ax.text(0.03, 0.03,
            f'raw $\\rho$ = {raw_rho:.3f}\npartial $\\rho$ = {partial_rho:.3f}',
            transform=ax.transAxes, fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCC', alpha=0.9))

    short = dataset_name.split(' (')[0]
    ax.set_title(f'{short} (n={len(sub)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Shesha Coherence', fontweight='bold', fontsize=11)
    ax.set_ylabel('HSPA5 (BiP) Expression', fontweight='bold', fontsize=11)
    sns.despine(ax=ax)


def panel_raw_vs_partial(ax, partial_csv_path):
    """Panel c: dot plot from stress_partial_correlations.csv."""
    df_corr = pd.read_csv(partial_csv_path)

    markers = ['DDIT3', 'ATF4', 'XBP1', 'HSPA5']
    datasets_order = ['Dixit 2016 (CRISPR-KO)', 'Norman 2019 (CRISPRa)', 'Replogle 2022 (CRISPRi)']
    ds_short = {
        'Dixit 2016 (CRISPR-KO)': 'Dixit',
        'Norman 2019 (CRISPRa)': 'Norman',
        'Replogle 2022 (CRISPRi)': 'Replogle',
    }
    ds_colors = {
        'Dixit 2016 (CRISPR-KO)': BLUE,
        'Norman 2019 (CRISPRa)': RED,
        'Replogle 2022 (CRISPRi)': GREEN,
    }

    # Build y-positions
    y_positions = {}
    y = 0
    for marker in markers:
        for ds in datasets_order:
            y_positions[(marker, ds)] = y
            y += 1
        y += 0.8  # gap between markers

    for _, row in df_corr.iterrows():
        key = (row['marker'], row['dataset'])
        if key not in y_positions:
            continue
        yp = y_positions[key]
        color = ds_colors.get(row['dataset'], DARK_GRAY)

        raw = row['rho_raw']
        partial = row['rho_partial']

        # Open circle: raw
        ax.scatter(raw, yp, s=80, facecolors='none',
                   edgecolors=color, linewidth=1.5, zorder=3)
        # Filled circle: partial
        fill_alpha = 1.0 if row['survives_magnitude_control'] else 0.3
        ax.scatter(partial, yp, s=80, facecolors=color,
                   edgecolors=color, linewidth=1, alpha=fill_alpha, zorder=4)
        # Arrow
        ax.annotate('', xy=(partial, yp), xytext=(raw, yp),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4))

    # Y-axis labels
    yticks = []
    ylabels = []
    for marker in markers:
        for ds in datasets_order:
            yticks.append(y_positions[(marker, ds)])
            ylabels.append(ds_short.get(ds, ds))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)

    # Marker group labels on right
    for marker in markers:
        mid_y = np.mean([y_positions[(marker, ds)] for ds in datasets_order])
        is_headline = marker == 'HSPA5'
        ax.text(1.02, mid_y, marker, transform=ax.get_yaxis_transform(),
                fontsize=10, fontweight='bold', va='center', ha='left',
                color=HSPA5_COLOR if is_headline else DARK_GRAY)

    # Horizontal separators between marker groups
    for i, marker in enumerate(markers[1:], 1):
        first_y = y_positions[(marker, datasets_order[0])]
        ax.axhline(first_y - 0.4, color='#EEEEEE', linewidth=1, zorder=0)

    ax.axvline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.set_xlabel('Spearman $\\rho$', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Partial Correlation\n(controlling for magnitude)', fontsize=11, fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=DARK_GRAY, markeredgewidth=1.5, markersize=9,
               label='Raw $\\rho$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DARK_GRAY,
               markeredgecolor=DARK_GRAY, markersize=9,
               label='Partial $\\rho$ (survives)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DARK_GRAY,
               markeredgecolor=DARK_GRAY, markersize=9, alpha=0.3,
               label='Partial $\\rho$ (n.s.)'),
        Line2D([0], [0], marker='', color='w', label=''),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=BLUE,
               markersize=8, label='Dixit'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=RED,
               markersize=8, label='Norman'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=GREEN,
               markersize=8, label='Replogle'),
    ]
    # ax.legend(handles=legend_elements, loc='lower left', fontsize=7.5, framealpha=0.9)
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 0.25), fontsize=7.5, framealpha=0.9)
    ax.invert_yaxis()
    sns.despine(ax=ax)


def main():
    if drive_out.exists():
        global OUT_DIR
        OUT_DIR = drive_out

    print("=== REPLOGLE ===")
    df_replogle = load_frozen_with_hspa5("Replogle 2022 (CRISPRi)")
    if df_replogle is None:
        df_replogle = extract_hspa5_backed("Replogle 2022 (CRISPRi)")
    print(f"Replogle: {len(df_replogle)} perturbations, "
          f"HSPA5 available: {df_replogle['hspa5'].notna().sum()}")

    print("\n=== DIXIT ===")
    df_dixit = load_frozen_with_hspa5("Dixit 2016 (CRISPR-KO)")
    if df_dixit is None:
        df_dixit = extract_hspa5_backed("Dixit 2016 (CRISPR-KO)")
    print(f"Dixit: {len(df_dixit)} perturbations, "
          f"HSPA5 available: {df_dixit['hspa5'].notna().sum()}")

    partials = _search_csvs("stress_partial_correlations.csv")
    partial_csv = partials[0] if partials else DATA_DIR / "stress_partial_correlations.csv"
    print(f"\nLoading partial correlations from {partial_csv}")

    # =========================================================================
    # FIGURE 3
    # =========================================================================
    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

    # Panel A: Replogle HSPA5
    ax0 = fig.add_subplot(gs[0, 0])
    panel_stress_scatter(ax0, df_replogle, 'Replogle 2022 (CRISPRi)',
                          raw_rho=-0.403, partial_rho=-0.206,
                          hh_obs=301, hh_exp=458.0, hh_p_str='< 0.0001')

    # Panel B: Dixit HSPA5
    ax1 = fig.add_subplot(gs[0, 1])
    panel_stress_scatter(ax1, df_dixit, 'Dixit 2016 (CRISPR-KO)',
                          raw_rho=-0.313, partial_rho=-0.338,
                          hh_obs=29, hh_exp=38.8, hh_p_str='0.040')

    # Panel C: Raw vs partial dot plot
    ax2 = fig.add_subplot(gs[0, 2])
    panel_raw_vs_partial(ax2, str(partial_csv))

    # Panel labels
    for ax, label in zip([ax0, ax1, ax2], ['a', 'b', 'c']):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig3_stress.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'fig3_stress.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFig 3 saved to {OUT_DIR}")
    plt.show()


if __name__ == '__main__':
    main()