#!/usr/bin/env python3
"""
S1 — Magnitude–stability with linear and LOESS fits (six datasets + pooled).

Solid dataset-colored scatter (no density colorbars). Dashed linear fit and a
solid LOESS overlay restricted to the data-supported magnitude range. LOESS is
omitted when n < 12 (Adamson pilot). Spearman rho with 10,000 bootstrap 95% CIs.

Prefers frozen_sp_scores.csv. Use --live to recompute missing datasets from pertpy.

USAGE:
    python fig2_magnitude_stability_loess.py --csv-only
    python fig2_magnitude_stability_loess.py --live
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
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
except ImportError:
    _sm_lowess = None

import pipeline_config as cfg
from revision_io import resolve_out_dir
from stats_utils import bootstrap_spearman_ci
from fig_1 import (
    DATASETS_INFO,
    CSV_SEARCH_DIRS,
    load_all_scores,
    despine,
)

SEED = cfg.SEED
N_BOOTSTRAP = cfg.N_BOOTSTRAP
LOESS_FRAC = 0.65
LOESS_MIN_N = 12
np.random.seed(SEED)


def _loess_frac(n):
    if n < LOESS_MIN_N:
        return None
    if n < 30:
        return 0.85
    if n < 80:
        return 0.75
    return LOESS_FRAC


def _lowess_fallback(y, x, frac=LOESS_FRAC):
    """Tricube-weighted local linear fit when statsmodels is unavailable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    n_span = max(2, int(np.ceil(frac * n)))
    fitted = np.empty(n)
    for i in range(n):
        dist = np.abs(x - x[i])
        idx = np.argpartition(dist, n_span - 1)[:n_span]
        dmax = dist[idx].max()
        if dmax <= 0:
            fitted[i] = y[idx].mean()
            continue
        u = dist[idx] / dmax
        w = (1.0 - u ** 3) ** 3
        X = np.column_stack([np.ones(len(idx)), x[idx] - x[i]])
        try:
            beta = np.linalg.lstsq(X * w[:, None], y[idx] * w, rcond=None)[0]
            fitted[i] = beta[0]
        except Exception:
            fitted[i] = np.average(y[idx], weights=w)
    return fitted


def _loess_curve(x, y, frac=None, q_lo=0.02, q_hi=0.95, n_grid=200):
    """Smooth LOESS on a regular grid, trimmed to the supported magnitude range.

    The right-tail quantile cut stops the local-linear edge from turning over
    on a handful of sparse high-magnitude points (Norman above ~3.4). A larger
    span plus grid interpolation removes the piecewise elbows at magnitude gaps
    (Dixit ~1.2). Returns (None, None) when n is too small to support a fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    frac = _loess_frac(n) if frac is None else frac
    if frac is None:
        return None, None
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if _sm_lowess is not None:
        fitted = _sm_lowess(ys, xs, frac=frac, it=1, return_sorted=False)
    else:
        fitted = _lowess_fallback(ys, xs, frac=frac)
    x_lo, x_hi = np.quantile(xs, [q_lo, q_hi])
    if not np.isfinite(x_lo) or x_hi <= x_lo:
        return None, None
    m = (xs >= x_lo) & (xs <= x_hi)
    if int(m.sum()) < 4:
        return None, None
    grid = np.linspace(float(x_lo), float(x_hi), n_grid)
    return grid, np.interp(grid, xs[m], fitted[m])


def load_live_datasets(missing):
    """Score missing datasets live from pertpy (Colab / --live)."""
    import importlib
    import importlib.util
    import types
    import scanpy as sc
    from anndata import AnnData
    from scipy.sparse import issparse
    from shesha.bio import compute_stability, compute_magnitude

    cache = Path("/tmp/pertpy_data")
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache = Path.home() / ".cache" / "pertpy_data"
        cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SCVERSE_DATADIR", str(cache))
    os.environ.setdefault("PERTPY_CACHE_DIR", str(cache))

    for mod in list(sys.modules):
        if mod == "pertpy" or mod.startswith("pertpy."):
            del sys.modules[mod]
    spec = importlib.util.find_spec("pertpy")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("pertpy is not installed. Run: pip install pertpy==1.0.6")
    pkg = types.ModuleType("pertpy")
    pkg.__path__ = spec.submodule_search_locations
    pkg.__spec__ = spec
    sys.modules["pertpy"] = pkg
    sc.settings.datasetdir = cache
    pt = importlib.import_module("pertpy.data._datasets")
    pt.settings.datasetdir = cache

    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def _preprocess(adata, pert_col, ctrl_label):
        adata.obs[pert_col] = adata.obs[pert_col].astype(str)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        counts = adata.obs[pert_col].value_counts()
        valid = [p for p in counts[counts >= cfg.MIN_CELLS].index if p != ctrl_label]
        sub = adata[adata.obs[pert_col].isin(valid + [ctrl_label])].copy()
        sc.pp.highly_variable_genes(sub, n_top_genes=cfg.N_HVG, subset=True)
        sc.tl.pca(sub, n_comps=min(cfg.N_PCS, sub.n_vars - 1), random_state=SEED)
        adata_pca = AnnData(X=sub.obsm['X_pca'], obs=sub.obs)
        stab = compute_stability(adata_pca, perturbation_key=pert_col,
                                 control_label=ctrl_label, metric='cosine')
        mag = compute_magnitude(adata_pca, perturbation_key=pert_col,
                                control_label=ctrl_label, metric='euclidean')
        df = pd.DataFrame({'stability': pd.Series(stab), 'magnitude': pd.Series(mag)})
        if ctrl_label in df.index:
            df = df.drop(ctrl_label)
        df = df[df.index.isin(valid)].copy()
        df['perturbation'] = df.index.astype(str)
        df['n_cells'] = df.index.map(counts)
        return df.reset_index(drop=True)

    def _adamson(name, loader):
        adata = loader()
        src = next((c for c in ['perturbation_name', 'perturbation', 'gene',
                                'target', 'guide_id', 'condition']
                    if c in adata.obs.columns), None)
        if src is None:
            src = next((c for c in adata.obs.columns
                        if 'pert' in c.lower() or 'gene' in c.lower()), None)
        adata.obs[src] = adata.obs[src].astype(str)
        ctrl_kws = cfg.MANUAL_CONTROLS[name]
        adata.obs['condition'] = adata.obs[src].apply(
            lambda x: 'control' if any(kw in x.lower() for kw in ctrl_kws) else x
        )
        adata = adata[adata.obs['condition'] != 'nan'].copy()
        return _preprocess(adata, 'condition', 'control')

    loaders = {
        'Norman 2019 (CRISPRa)': lambda: _preprocess(
            pt.norman_2019(), 'perturbation_name', 'control'),
        'Adamson 2016 UPR (CRISPRi)': lambda: _adamson(
            'Adamson 2016 UPR (CRISPRi)',
            getattr(pt, 'adamson_2016_upr_perturb_seq', None) or pt.adamson_2016_pilot),
        'Adamson 2016 pilot (CRISPRi)': lambda: _adamson(
            'Adamson 2016 pilot (CRISPRi)', pt.adamson_2016_pilot),
        'Dixit 2016 (CRISPR-KO)': lambda: _preprocess(
            pt.dixit_2016(), 'perturbation_name', 'control'),
        'Papalexi 2021 (CRISPR-KO)': lambda: _preprocess_papalexi(pt),
        'Replogle 2022 (CRISPRi)': lambda: _preprocess_replogle(pt, _preprocess),
    }

    def _preprocess_papalexi(pt_mod):
        raw = pt_mod.papalexi_2021()
        adata = raw.mod['rna'].copy()
        adata.obs['gene_target'] = raw.obs['gene_target'].values
        return _preprocess(adata, 'gene_target', 'NT')

    def _preprocess_replogle(pt_mod, preprocess):
        adata = pt_mod.replogle_2022_k562_essential()
        adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)

        def _label(x):
            if 'non-targeting' in x or x.startswith('chr'):
                return 'control'
            if 'pos_control' in x:
                return 'POS_CONTROL'
            return x.split('_')[0]

        adata.obs['condition'] = adata.obs['perturbation'].apply(_label)
        adata = adata[
            (adata.obs['condition'] != 'POS_CONTROL') &
            (adata.obs['condition'] != 'nan')
        ].copy()
        return preprocess(adata, 'condition', 'control')

    frames = []
    for name in missing:
        if name not in loaders:
            continue
        print(f"Scoring live: {name}")
        sub = loaders[name]()
        sub['dataset'] = name
        frames.append(sub)
        print(f"    -> {len(sub)} perturbations")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_correlations(dfs):
    results = []
    seed_counter = SEED + 500
    print(f"\n{'Dataset':<30s}  {'n':>4s}  {'rho':>6s}  {'95% CI':>20s}  {'p':>10s}")
    print("-" * 78)
    for ds_full, ds_short, modality, *_ in DATASETS_INFO:
        if ds_full not in dfs:
            continue
        df = dfs[ds_full]
        ci = bootstrap_spearman_ci(
            df['magnitude'], df['stability'],
            n_bootstrap=N_BOOTSTRAP, seed=seed_counter,
        )
        seed_counter += 1
        ci_str = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        print(f"{ds_full:<30s}  {ci['n']:>4d}  {ci['rho']:>+.3f}  {ci_str:>20s}  {ci['p']:>10.2e}")
        results.append({
            'dataset': ds_full,
            'dataset_short': ds_short,
            'modality': modality,
            'n': ci['n'],
            'rho': ci['rho'],
            'ci_low': ci['ci_low'],
            'ci_high': ci['ci_high'],
            'p': ci['p'],
        })

    all_z = []
    for ds_full, ds_short, *_ in DATASETS_INFO:
        if ds_full not in dfs:
            continue
        sub = dfs[ds_full][['magnitude', 'stability']].copy()
        sub['mag_z'] = (sub['magnitude'] - sub['magnitude'].mean()) / sub['magnitude'].std()
        sub['stab_z'] = (sub['stability'] - sub['stability'].mean()) / sub['stability'].std()
        sub['dataset_short'] = ds_short
        all_z.append(sub)
    pooled = pd.concat(all_z, ignore_index=True)
    ci_pooled = bootstrap_spearman_ci(
        pooled['mag_z'], pooled['stab_z'],
        n_bootstrap=N_BOOTSTRAP, seed=seed_counter,
    )
    ci_str = f"[{ci_pooled['ci_low']:.3f}, {ci_pooled['ci_high']:.3f}]"
    print(f"{'Pooled (z-scored)':<30s}  {ci_pooled['n']:>4d}  {ci_pooled['rho']:>+.3f}  "
          f"{ci_str:>20s}  {ci_pooled['p']:>10.2e}")
    results.append({
        'dataset': 'Pooled (z-scored)',
        'dataset_short': 'Pooled',
        'modality': 'All',
        'n': ci_pooled['n'],
        'rho': ci_pooled['rho'],
        'ci_low': ci_pooled['ci_low'],
        'ci_high': ci_pooled['ci_high'],
        'p': ci_pooled['p'],
    })
    return results, pooled, ci_pooled


def plot_figure(dfs, corr_results, pooled, ci_pooled, out_dir):
    fig, axes2d = plt.subplots(2, 4, figsize=(22, 11))
    axes2d[1, 3].axis('off')
    axes = [axes2d[0, i] for i in range(4)] + [axes2d[1, i] for i in range(3)]
    all_for_pooled = []

    for i, (ds_full, ds_short, modality, cmap_name, legend_color) in enumerate(DATASETS_INFO):
        ax = axes[i]
        if ds_full not in dfs or len(dfs[ds_full]) < 3:
            ax.text(0.5, 0.5, f'{ds_short}\n(no data)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
            despine(ax)
            continue

        sub = dfs[ds_full].copy()
        ci = next((r for r in corr_results if r['dataset'] == ds_full), None)
        x = sub['magnitude'].to_numpy(dtype=float)
        y = sub['stability'].to_numpy(dtype=float)
        color = plt.get_cmap(cmap_name)(0.72)
        large = len(x) > 400
        ax.scatter(
            x, y, color=color, s=16 if large else 28,
            alpha=0.22 if large else 0.45,
            edgecolor='none', rasterized=large, zorder=2,
        )
        if len(x) >= LOESS_MIN_N:
            slope, intercept, *_ = linregress(x, y)
            x_line = np.array([0.0, float(np.nanmax(x))])
            ax.plot(x_line, slope * x_line + intercept, '--',
                    color='gray', linewidth=2, alpha=0.7, zorder=1, label='Linear')
            x_loess, y_loess = _loess_curve(x, y)
            if x_loess is not None:
                ax.plot(x_loess, y_loess, '-',
                        color='#333333', linewidth=2, alpha=0.85, zorder=1, label='LOESS')
            if i == 0:
                ax.legend(fontsize=7, framealpha=0.9, loc='upper left')

        if ci:
            ann = (f"$\\rho$ = {ci['rho']:.3f}\n"
                   f"95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
        else:
            ann = ''
        ax.text(0.97, 0.03, ann,
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9))
        ax.set_title(f'{ds_short}\n({modality}, n={len(sub)})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Effect Magnitude', fontsize=10, fontweight='bold')
        ax.set_ylabel('Shesha Coherence', fontsize=10, fontweight='bold')
        y_lo = min(-0.08, float(np.nanmin(y)) - 0.03)
        ax.set_xlim(0, max(float(np.nanmax(x)) * 1.06, 1.0))
        ax.set_ylim(y_lo, 1.0)
        if y_lo < 0:
            ax.axhline(0.0, color='#B0B0B0', linewidth=0.7, zorder=1)
        despine(ax)

        sub_z = sub[['magnitude', 'stability']].copy()
        sub_z['mag_z'] = (sub_z['magnitude'] - sub_z['magnitude'].mean()) / sub_z['magnitude'].std()
        sub_z['stab_z'] = (sub_z['stability'] - sub_z['stability'].mean()) / sub_z['stability'].std()
        sub_z['dataset_short'] = ds_short
        sub_z['color'] = legend_color
        all_for_pooled.append(sub_z)

    ax_p = axes[6]
    pooled_plot = pd.concat(all_for_pooled, ignore_index=True)
    for ds in pooled_plot['dataset_short'].unique():
        mask = pooled_plot['dataset_short'] == ds
        c = pooled_plot.loc[mask, 'color'].iloc[0]
        ax_p.scatter(pooled_plot.loc[mask, 'mag_z'], pooled_plot.loc[mask, 'stab_z'],
                     c=c, s=20, alpha=0.5, edgecolor='none', label=ds)

    slope_p, intercept_p, *_ = linregress(pooled_plot['mag_z'], pooled_plot['stab_z'])
    x_p = np.array([pooled_plot['mag_z'].min(), pooled_plot['mag_z'].max()])
    ax_p.plot(x_p, slope_p * x_p + intercept_p, '--',
              color='gray', linewidth=2, alpha=0.7, label='Linear')
    x_loess, y_loess = _loess_curve(pooled_plot['mag_z'], pooled_plot['stab_z'])
    ax_p.plot(x_loess, y_loess, '-',
              color='#333333', linewidth=2, alpha=0.85, label='LOESS')

    ann_p = (f"$\\rho$ = {ci_pooled['rho']:.3f}\n"
             f"95% CI [{ci_pooled['ci_low']:.3f}, {ci_pooled['ci_high']:.3f}]")
    ax_p.text(0.97, 0.03, ann_p,
              transform=ax_p.transAxes, fontsize=9, ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#CCCCCC', alpha=0.9))
    ax_p.set_title(f'Pooled (z-scored)\n(n={ci_pooled["n"]})',
                   fontsize=11, fontweight='bold')
    ax_p.set_xlabel('Magnitude (z)', fontsize=10, fontweight='bold')
    ax_p.set_ylabel('Coherence (z)', fontsize=10, fontweight='bold')
    ax_p.legend(fontsize=7, framealpha=0.8, loc='upper left')
    despine(ax_p)

    for i, label in enumerate('abcdefg'):
        axes[i].text(-0.08, 1.08, label, transform=axes[i].transAxes,
                     fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    out = out_dir / "fig2_magnitude_stability_ci"
    plt.savefig(str(out) + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved figure -> {out}.pdf / .png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-only', action='store_true',
                        help='Do not score missing datasets live')
    parser.add_argument('--live', action='store_true',
                        help='Recompute missing datasets from pertpy')
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    print("=" * 80)
    print("S1: MAGNITUDE–STABILITY WITH LINEAR + LOESS (10k bootstrap CIs)")
    print(f"Bootstrap replicates: {N_BOOTSTRAP}  |  Seed: {SEED}")
    print(f"Search dirs: {[str(d) for d in [out_dir, *CSV_SEARCH_DIRS]]}")
    print("=" * 80)

    df = load_all_scores(out_dir, csv_only=args.csv_only or not args.live)
    have = set(df['dataset'].unique()) if len(df) else set()
    missing = [name for name, *_ in DATASETS_INFO if name not in have]
    if missing and args.live and not args.csv_only:
        live = load_live_datasets(missing)
        if len(live):
            df = pd.concat([df, live], ignore_index=True) if len(df) else live
            df['dataset'] = df['dataset'].map(cfg.resolve_dataset_name)

    if not len(df):
        raise SystemExit(
            f"No Sp scores found in {out_dir}. "
            "Pass a frozen CSV via fig_1 search dirs, or rerun with --live."
        )

    print("Datasets:", df['dataset'].value_counts().to_dict())
    dfs = {name: df[df['dataset'] == name].copy()
           for name, *_ in DATASETS_INFO
           if (df['dataset'] == name).any()}

    corr_results, pooled, ci_pooled = collect_correlations(dfs)
    corr_df = pd.DataFrame(corr_results)
    corr_path = out_dir / "magnitude_stability_correlations_ci.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"\nSaved -> {corr_path}")

    plot_figure(dfs, corr_results, pooled, ci_pooled, out_dir)
    print("\nCOMPLETE")
    print(f"  - {out_dir / 'fig2_magnitude_stability_ci.pdf'}")
    print(f"  - {corr_path}")


if __name__ == '__main__':
    main()
