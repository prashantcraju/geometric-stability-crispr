#!/usr/bin/env python3
"""
Figure 1 — Magnitude–stability across six CRISPR datasets (7 panels).

Row 1 (4): Norman, Adamson 2016 UPR, Adamson 2016 pilot, Dixit
Row 2 (3): Papalexi, Replogle, pooled z-scored

Adamson UPR uses a new purple so it is distinct from the original Adamson
pilot (red). Prefers frozen Sp CSVs; overlays adamson_upr_sp_scores.csv;
falls back to pipeline_core for any missing dataset.

USAGE:
    python fig_1.py
    python fig_1.py --csv-only
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import os
import subprocess
import sys

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, linregress, gaussian_kde
import matplotlib.pyplot as plt
from pathlib import Path

import pipeline_config as cfg
from revision_io import data_search_dirs, resolve_out_dir, load_sp_table

SEED = cfg.SEED
np.random.seed(SEED)

CSV_SEARCH_DIRS = data_search_dirs()

# Sequential cmaps match the original fig2 density panels.
# Adamson UPR gets a new teal cmap so it is distinct from the original
# Adamson (Purples) which is now the pilot.
DATASETS_INFO = [
    ('Norman 2019 (CRISPRa)',        'Norman 2019',         'CRISPRa',   'Blues',   '#4C72B0'),
    ('Adamson 2016 UPR (CRISPRi)',   'Adamson 2016 UPR',    'CRISPRi',   'GnBu',    '#1B9E77'),
    ('Adamson 2016 pilot (CRISPRi)', 'Adamson 2016 pilot',  'CRISPRi',   'Purples', '#8172B2'),
    ('Dixit 2016 (CRISPR-KO)',       'Dixit 2016',          'CRISPR-KO', 'Greens',  '#2CA02C'),
    ('Papalexi 2021 (CRISPR-KO)',    'Papalexi 2021',       'CRISPR-KO', 'Oranges', '#E07B3D'),
    ('Replogle 2022 (CRISPRi)',      'Replogle 2022',       'CRISPRi',   'Reds',    '#C44E52'),
]


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def perturbation_density(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.ones(n)
    try:
        return gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    except Exception:
        return np.ones(n)


def _drop_controls(df):
    out = df.copy()
    if 'perturbation' in out.columns:
        mask = out['perturbation'].astype(str).str.lower().isin(
            {'control', 'ctrl', 'nan', 'nt'}
        )
        out = out.loc[~mask]
    return out


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_all_scores(out_dir: Path, csv_only: bool = False) -> pd.DataFrame:
    """Load frozen / Euclidean Sp table; fill gaps from other CSV dirs or live."""
    search = []
    for d in [out_dir, *CSV_SEARCH_DIRS]:
        if d is not None and d.exists() and d not in search:
            search.append(d)

    frames = []
    main_path = None
    for name in (
        "frozen_sp_scores.csv",
        "frozen_sp_scores_sample.csv",
        "shesha_crispr_results_euclidean.csv",
        "adamson_upr_sp_scores.csv",
    ):
        for d in search:
            p = d / name
            if p.exists():
                main_path = p
                break
        if main_path is not None:
            break
    if main_path is not None:
        main = _drop_controls(load_sp_table(main_path))
        print(f"Loaded {main_path}  ({len(main)} rows, "
              f"{main['dataset'].nunique()} datasets)")
        frames.append(main)
    else:
        print("No combined Sp CSV in search dirs")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df['dataset'] = df['dataset'].map(cfg.resolve_dataset_name)
        df = df.drop_duplicates(subset=['dataset', 'perturbation'], keep='last')

    have = set(df['dataset'].unique()) if len(df) else set()
    if 'Adamson 2016 UPR (CRISPRi)' not in have:
        upr_path = _first_existing(
            *(d / 'adamson_upr_sp_scores.csv' for d in search)
        )
        if upr_path is not None:
            upr = _drop_controls(load_sp_table(upr_path))
            upr['dataset'] = 'Adamson 2016 UPR (CRISPRi)'
            print(f"Added Adamson UPR from {upr_path}  (n={len(upr)})")
            df = pd.concat([df, upr], ignore_index=True) if len(df) else upr
            df['dataset'] = df['dataset'].map(cfg.resolve_dataset_name)

    have = set(df['dataset'].unique()) if len(df) else set()
    missing = [name for name, *_ in DATASETS_INFO if name not in have]
    if missing and csv_only:
        print(f"--csv-only: skipping live load for {missing}")
        return df

    if missing:
        from pipeline_core import run_dataset, setup_cache
        setup_cache()
        live = []
        for name in missing:
            try:
                print(f"Scoring missing dataset live: {name}")
                sub = run_dataset(name, prefer_local=True)
                live.append(_drop_controls(sub))
            except Exception as e:
                print(f"  ! {name} failed: {e}")
        if live:
            df = pd.concat([df, *live], ignore_index=True) if len(df) else pd.concat(live, ignore_index=True)
            df['dataset'] = df['dataset'].map(cfg.resolve_dataset_name)

    return df


def plot_panel(ax, sub, ds_short, modality, cmap_name):
    n = len(sub)
    if n < 3:
        ax.text(0.5, 0.5, f'{ds_short}\n(no data)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='gray')
        despine(ax=ax)
        return None

    x = sub['magnitude'].to_numpy(dtype=float)
    y = sub['stability'].to_numpy(dtype=float)
    z = perturbation_density(x, y)
    order = np.argsort(z)
    sc = ax.scatter(
        x[order], y[order], c=z[order], cmap=cmap_name,
        s=40, alpha=0.8, edgecolor='white', linewidth=0.5, zorder=2,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Perturbation Density', rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=8)

    slope, intercept, *_ = linregress(x, y)
    x_line = np.array([np.nanmin(x), np.nanmax(x)])
    ax.plot(x_line, slope * x_line + intercept, '--',
            color='gray', linewidth=2, alpha=0.7, zorder=1)

    rho, _ = spearmanr(x, y)
    ax.set_title(f'{ds_short}\n({modality}, n={n})',
                 fontsize=11, fontweight='bold')
    ax.text(0.97, 0.03, f'$\\rho$ = {rho:.3f}',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))
    ax.set_xlabel('Effect Magnitude', fontsize=10, fontweight='bold')
    ax.set_ylabel('Shesha Coherence', fontsize=10, fontweight='bold')
    despine(ax=ax)
    return rho


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-only', action='store_true',
                        help='Do not score missing datasets live')
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    df = load_all_scores(out_dir, csv_only=args.csv_only)
    if not len(df):
        raise SystemExit(
            f"No Sp scores found in {out_dir}. "
            "Run run_frozen_main.py or adamson_upr_spike.py first."
        )

    print("Datasets:", df['dataset'].value_counts().to_dict())

    fig, axes2d = plt.subplots(2, 4, figsize=(20, 10))
    axes2d[1, 3].axis('off')
    axes = [axes2d[0, i] for i in range(4)] + [axes2d[1, i] for i in range(3)]

    all_for_pooled = []

    for i, (ds_full, ds_short, modality, cmap_name, legend_color) in enumerate(DATASETS_INFO):
        ax = axes[i]
        sub = df[df['dataset'] == ds_full].copy()
        plot_panel(ax, sub, ds_short, modality, cmap_name)

        if len(sub) > 5:
            sub_z = sub[['magnitude', 'stability']].copy()
            sub_z['mag_z'] = (sub_z['magnitude'] - sub_z['magnitude'].mean()) / sub_z['magnitude'].std()
            sub_z['stab_z'] = (sub_z['stability'] - sub_z['stability'].mean()) / sub_z['stability'].std()
            sub_z['dataset_short'] = ds_short
            sub_z['color'] = legend_color
            all_for_pooled.append(sub_z)

    # Panel g: pooled z-scored (dataset colors, no density bar — same as original)
    ax_p = axes[6]
    if all_for_pooled:
        pooled = pd.concat(all_for_pooled, ignore_index=True)
        for ds in pooled['dataset_short'].unique():
            mask = pooled['dataset_short'] == ds
            c = pooled.loc[mask, 'color'].iloc[0]
            ax_p.scatter(pooled.loc[mask, 'mag_z'], pooled.loc[mask, 'stab_z'],
                         c=c, s=20, alpha=0.5, edgecolor='none', label=ds)

        rho_p, _ = spearmanr(pooled['mag_z'], pooled['stab_z'])
        slope_p, intercept_p, *_ = linregress(pooled['mag_z'], pooled['stab_z'])
        x_p = np.array([pooled['mag_z'].min(), pooled['mag_z'].max()])
        ax_p.plot(x_p, slope_p * x_p + intercept_p, '--',
                  color='gray', linewidth=2, alpha=0.7)

        ax_p.set_title(f'Pooled (z-scored)\n$\\rho$ = {rho_p:.3f}',
                       fontsize=11, fontweight='bold')
        ax_p.set_xlabel('Magnitude (z)', fontsize=10, fontweight='bold')
        ax_p.set_ylabel('Coherence (z)', fontsize=10, fontweight='bold')
        ax_p.legend(fontsize=7, framealpha=0.8, loc='upper left')
        despine(ax=ax_p)
    else:
        ax_p.text(0.5, 0.5, 'Pooled\n(no data)',
                  transform=ax_p.transAxes, ha='center', va='center',
                  fontsize=10, color='gray')
        despine(ax=ax_p)

    for ax, label in zip(axes, 'abcdefg'):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    out = out_dir / 'fig1_magnitude_stability'
    plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}.pdf / .png")
    plt.show()


if __name__ == '__main__':
    main()
