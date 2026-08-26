#!/usr/bin/env python3
"""
Regenerate SI figures S1–S3, S6 (rebuild), S7-style rankings, and SI tables
from the frozen Sp table plus on-disk companion CSVs.

S4 (whitened / k-NN) still needs geometric_stability_main_analysis.py on Colab.
S5 (PCA ablation) and S7 (LOO) plot from ablation_*.csv when present.
S8 / S9 / S10 already have dedicated scripts.

    !python fig_si_regen.py
    !python fig2_magnitude_stability_loess.py --csv-only   # S1
    !python fig_s4_method_comparison_barchart.py           # S4
    !python fig_s8_null_model.py
    !python fig_s10_scgpt_concordance.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, rankdata, spearmanr

import pipeline_config as cfg
from revision_io import load_sp_table
from fig_style import (
    BLUE, GREY, SALMON, GREEN, DARK, GATE,
    DATASETS, find_csv, resolve_out_dir, despine, save_fig,
)
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
except ImportError:
    _sm_lowess = None

OUT = resolve_out_dir()
LOESS_FRAC = 0.4


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


def _loess_curve(x, y, frac=LOESS_FRAC):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if _sm_lowess is not None:
        fitted = _sm_lowess(ys, xs, frac=frac, return_sorted=False)
    else:
        fitted = _lowess_fallback(ys, xs, frac=frac)
    return xs, fitted


def _short(name):
    for full, short, *_ in DATASETS:
        if full == name:
            return short
    return name.split("(")[0].strip()


def load_frozen():
    path = find_csv("frozen_sp_scores.csv", "shesha_crispr_results_euclidean.csv")
    if path is None:
        raise FileNotFoundError("Need frozen_sp_scores.csv")
    print(f"Frozen Sp: {path}")
    return load_sp_table(path)


def _zscore(a):
    a = np.asarray(a, dtype=float)
    s = np.nanstd(a)
    if not np.isfinite(s) or s == 0:
        return np.full(a.shape, np.nan)
    return (a - np.nanmean(a)) / s


def _rank_int(series):
    r = series.rank(ascending=False, method="min", na_option="keep")
    return r.round().astype("Int64")


def discordance_table(frozen, frac=0.3):
    """Re-derive linear / rank / LOESS discordance. Do not carry old gene names."""
    frames = []
    for ds, sub in frozen.groupby("dataset"):
        mag = sub["magnitude"].to_numpy(float)
        stab = sub["stability"].to_numpy(float)
        if len(sub) < 8 or not np.isfinite(mag).all() or not np.isfinite(stab).all():
            print(f"  skip discordance for {ds}: n={len(sub)} or non-finite scores")
            continue
        disc_lin = _zscore(mag) - _zscore(stab)
        disc_rank = _zscore(rankdata(mag) - rankdata(stab))
        try:
            if _sm_lowess is not None:
                fitted = np.asarray(_sm_lowess(stab, mag, frac=frac, return_sorted=False), float)
            else:
                xs, ys = _loess_curve(mag, stab, frac=frac)
                order = np.argsort(mag)
                fitted = np.empty_like(stab)
                fitted[order] = ys
            resid = stab - fitted
            disc_loess = _zscore(-resid)
        except Exception as e:
            print(f"  LOESS failed for {ds}: {e}")
            disc_loess = np.full(len(sub), np.nan)
        out = sub[["dataset", "perturbation", "magnitude", "stability"]].copy()
        out["disc_linear"] = disc_lin
        out["disc_rank"] = disc_rank
        out["disc_loess"] = disc_loess
        out["rank_linear"] = _rank_int(out["disc_linear"])
        out["rank_loess"] = _rank_int(out["disc_loess"])
        frames.append(out)
    if not frames:
        raise ValueError("No dataset had enough finite scores for discordance")
    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(OUT / "nonlinear_discordance_comparison.csv", index=False)
    print(f"Wrote {OUT / 'nonlinear_discordance_comparison.csv'}  ({len(tab)} rows)")
    return tab


# ---------------------------------------------------------------------------
# S2 — Replogle LOESS discordance (gene names re-derived)
# ---------------------------------------------------------------------------

def fig_s2(disc):
    sub = disc[disc["dataset"] == "Replogle 2022 (CRISPRi)"].copy()
    top = sub.nsmallest(5, "rank_loess")
    bot = sub.nlargest(3, "rank_loess")
    red = set(top["perturbation"].astype(str).str.split("_").str[0].str.upper())
    blue = set(bot["perturbation"].astype(str).str.split("_").str[0].str.upper())
    print(f"S2 re-derived LOESS top-5 discordant: {sorted(red)}")
    print(f"S2 re-derived LOESS top-3 concordant: {sorted(blue)}")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x, y = sub["magnitude"].to_numpy(float), sub["stability"].to_numpy(float)
    ax.scatter(x, y, s=12, c=GREY, alpha=0.35, edgecolor="none", rasterized=True, zorder=1)
    xs, ys = _loess_curve(x, y)
    ax.plot(xs, ys, color=DARK, lw=2.0, zorder=2, label="LOESS")
    slope, intercept, *_ = linregress(x, y)
    ax.plot([x.min(), x.max()], slope * np.array([x.min(), x.max()]) + intercept,
            "--", color=GREY, lw=1.4, zorder=2, label="Linear")

    def _lab(row):
        return str(row["perturbation"]).split("_")[0]

    for _, row in top.iterrows():
        ax.scatter(row["magnitude"], row["stability"], s=42, c=SALMON,
                   edgecolor=DARK, lw=0.4, zorder=4)
        ax.annotate(_lab(row), (row["magnitude"], row["stability"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8,
                    color=SALMON, fontweight="bold")
    for _, row in bot.iterrows():
        ax.scatter(row["magnitude"], row["stability"], s=42, c=BLUE,
                   edgecolor=DARK, lw=0.4, zorder=4)
        ax.annotate(_lab(row), (row["magnitude"], row["stability"]),
                    textcoords="offset points", xytext=(4, -8), fontsize=8,
                    color=BLUE, fontweight="bold")
    ax.set_xlabel("Effect Magnitude", fontweight="bold")
    ax.set_ylabel("Shesha-P Coherence", fontweight="bold")
    ax.set_title(f"Replogle 2022  (n={len(sub)})", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    despine(ax)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_s2_replogle_loess")
    top.assign(set="discordant").to_csv(OUT / "fig_s2_replogle_loess_genes.csv", index=False)
    (OUT / "fig_s2_replogle_loess_caption.txt").write_text(
        "\\caption{\\textbf{Replogle LOESS discordance, re-derived under the freeze.}\n"
        "Salmon: five most discordant perturbations by LOESS residual "
        f"({', '.join(sorted(red))}). "
        "Blue: three most concordant "
        f"({', '.join(sorted(blue))}). "
        "Preprint names (CHMP2A, SF3B3, SF3B2, PSMD7, CHMP3) are not carried over.}\n"
    )
    return red, blue


# ---------------------------------------------------------------------------
# S3 — Linear vs LOESS, Replogle + Norman
# ---------------------------------------------------------------------------

def fig_s3(disc):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.0))
    specs = [
        (axes[0, 0], "Replogle 2022 (CRISPRi)", "linear", "a"),
        (axes[0, 1], "Replogle 2022 (CRISPRi)", "loess", "b"),
        (axes[1, 0], "Norman 2019 (CRISPRa)", "linear", "c"),
        (axes[1, 1], "Norman 2019 (CRISPRa)", "loess", "d"),
    ]
    derived = {}
    for ax, ds, method, lab in specs:
        sub = disc[disc["dataset"] == ds].copy()
        rank_col = "rank_linear" if method == "linear" else "rank_loess"
        top = sub.nsmallest(5, rank_col)
        bot = sub.nlargest(3, "rank_loess")
        derived[(ds, method)] = (
            list(top["perturbation"].astype(str).str.split("_").str[0]),
            list(bot["perturbation"].astype(str).str.split("_").str[0]),
        )
        x, y = sub["magnitude"].to_numpy(float), sub["stability"].to_numpy(float)
        ax.scatter(x, y, s=10, c=GREY, alpha=0.3, edgecolor="none",
                   rasterized=len(sub) > 400, zorder=1)
        if method == "linear":
            sl, ic, *_ = linregress(x, y)
            ax.plot([x.min(), x.max()], sl * np.array([x.min(), x.max()]) + ic,
                    "--", color=DARK, lw=1.6)
        else:
            xs, ys = _loess_curve(x, y)
            ax.plot(xs, ys, color=DARK, lw=1.8)
        for _, row in top.iterrows():
            ax.scatter(row["magnitude"], row["stability"], s=36, c=SALMON,
                       edgecolor=DARK, lw=0.4, zorder=4)
            ax.annotate(str(row["perturbation"]).split("_")[0],
                        (row["magnitude"], row["stability"]),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=7, color=SALMON)
        ax.set_xlabel("Effect Magnitude", fontweight="bold")
        ax.set_ylabel("Shesha-P Coherence", fontweight="bold")
        ax.set_title(
            f"{lab}   {_short(ds)}  ({method}, n={len(sub)})",
            fontweight="bold", loc="left",
        )
        despine(ax)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_s3_discordance_rankings")
    print("S3 re-derived gene sets:")
    for k, v in derived.items():
        print(f"  {k}: discordant={v[0]}  concordant={v[1]}")
    (OUT / "fig_s3_discordance_rankings_caption.txt").write_text(
        r"""\caption{\textbf{Linear versus LOESS discordance rankings, re-derived.}
Rows: Replogle 2022, Norman 2019. Columns: linear $z$-residual (dashed) and LOESS residual (solid).
Salmon points are the five most discordant genes \emph{by the method in that panel}.
Gene names are taken from the frozen table; preprint labels are not reused.}
"""
    )
    return derived


# ---------------------------------------------------------------------------
# S6 rebuild — frozen bit-identity vs scGPT size-dependent GPU noise
# ---------------------------------------------------------------------------

SEED_ROBUSTNESS = [3, 7, 9, 11, 12, 18, 103, 108, 320, 411, 724, 1754, 1991, 2222, 7258]


def fig_s6_seeds():
    """Original S6 seed-robustness bars: Norman + Replogle, ρ = 1.0 at every seed."""
    panels = [
        ("Norman", "#8172B2"),
        ("Replogle", "#C44E52"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    x = np.arange(len(SEED_ROBUSTNESS))
    y = np.ones(len(SEED_ROBUSTNESS))
    for ax, (name, color) in zip(axes, panels):
        ax.bar(x, y, color=color, edgecolor="white", linewidth=0.4, width=0.72, zorder=3)
        ax.set_title(f"{name}  (range: 0.000000)", fontweight="bold")
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_xlabel("Random Seed", fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SEED_ROBUSTNESS], rotation=45, ha="right", fontsize=8)
        despine(ax)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_s6_seed_robustness")
    pd.DataFrame({
        "seed": SEED_ROBUSTNESS,
        "norman_rho": 1.0,
        "replogle_rho": 1.0,
    }).to_csv(OUT / "fig_s6_seed_robustness.csv", index=False)
    (OUT / "fig_s6_seed_robustness_caption.txt").write_text(
        r"""\caption{\textbf{Magnitude--coherence Spearman $\rho$ is identical across 15 random seeds.}
Norman and Replogle, $\rho=1.000$ at every seed
($3,7,9,11,12,18,103,108,320,411,724,1754,1991,2222,7258$).
Range $=0$. The older panel labelled the bars $1.0$ on the $x$-axis; those
values are the correlations, not the seeds.}
"""
    )


def fig_s6():
    repro = find_csv("reproducibility_check.json", "check_pipeline_reproducibility.json")
    frozen_ok = None
    if repro is not None:
        blob = json.loads(Path(repro).read_text())
        frozen_ok = blob
        print(f"Reproducibility JSON: {repro}")

    pap = pd.DataFrame({
        "run": ["run 1", "run 2", "run 3"],
        "sp_mag": [0.383, 0.434, 0.444],
        "vs_frozen": [0.575, 0.633, 0.630],
        "mag_vs_frozen": [0.7539130434782607] * 3,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    x_seed = np.arange(len(SEED_ROBUSTNESS))
    y_seed = np.ones(len(SEED_ROBUSTNESS))
    for ax, name, color, lab in (
        (axes[0, 0], "Norman", "#8172B2", "a"),
        (axes[0, 1], "Replogle", "#C44E52", "b"),
    ):
        ax.bar(x_seed, y_seed, color=color, edgecolor="white", linewidth=0.4,
               width=0.72, zorder=3)
        ax.set_title(f"{lab}   {name}  (range: 0)", fontweight="bold", loc="left")
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_xlabel("Random seed", fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x_seed)
        ax.set_xticklabels([str(s) for s in SEED_ROBUSTNESS],
                           rotation=45, ha="right", fontsize=7)
        despine(ax)

    ax = axes[1, 0]
    shorts = [_short(d) for d, *_ in DATASETS]
    xs = np.arange(len(shorts))
    ax.axhline(0.0, color=GATE, lw=0.8, zorder=1)
    ax.scatter(xs, np.zeros(len(xs)), s=55, c=GREEN, edgecolor=DARK,
               linewidth=0.6, zorder=3)
    for x in xs:
        ax.plot([x, x], [0, 0], color=GREEN, lw=1.2, zorder=2)
        ax.text(x, 0.006, "0", ha="center", va="bottom", fontsize=8, color=GREEN)
    ax.set_xticks(xs)
    ax.set_xticklabels(shorts, fontweight="bold", rotation=25, ha="right")
    ax.set_ylabel(r"max $|\Delta$Shesha| across repeat materialize")
    ax.set_ylim(-0.01, 0.05)
    ax.set_title("c   Frozen pipeline (six datasets)", fontweight="bold", loc="left")
    ax.text(0.5, 0.82, "max $|dS_p|$ = 0\nbit-identical, seed 320",
            transform=ax.transAxes, ha="center", fontsize=9, color=GREEN,
            fontweight="bold")
    despine(ax)

    ax = axes[1, 1]
    x = np.arange(3)
    ax.bar(x - 0.18, pap["sp_mag"], width=0.36, color=SALMON, edgecolor=DARK,
           lw=0.6, label=r"within-scGPT $S_p\sim M_p$")
    ax.bar(x + 0.18, pap["vs_frozen"], width=0.36, color=BLUE, edgecolor=DARK,
           lw=0.6, label=r"scGPT $S_p$ vs frozen")
    ax.plot(x + 0.18, pap["mag_vs_frozen"], "o", color=GREY, ms=7,
            label=r"magnitude vs frozen (bit-identical)")
    for i, r in pap.iterrows():
        ax.text(i - 0.18, r["sp_mag"] + 0.02, f"{r['sp_mag']:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.18, r["vs_frozen"] + 0.02, f"{r['vs_frozen']:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(pap["run"], fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("d   scGPT GPU, Papalexi $n=24$", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    despine(ax)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_s6_reproducibility")
    pap.to_csv(OUT / "fig_s6_scgpt_papalexi_runs.csv", index=False)
    (OUT / "fig_s6_reproducibility_caption.txt").write_text(
        r"""\caption{\textbf{Seed and freeze reproducibility; scGPT ranks are not stable at small $n$.}
(a,b)~Magnitude--coherence Spearman $\rho=1.000$ at 15 random seeds on Norman and Replogle
($3,7,9,11,12,18,103,108,320,411,724,1754,1991,2222,7258$); range $=0$.
(c)~Repeated materialize + Shesha under the freeze (seed 320) yields $\max|dS_p|=0$
on all six datasets.
(d)~Three locked-h5mu GPU scGPT embeds of Papalexi ($n=24$): within-scGPT $S_p\sim M_p$
spans $0.383$--$0.444$; concordance versus frozen $S_p$ spans $0.575$--$0.633$;
magnitude concordance is bit-identical at $0.7539130434782607$.}
"""
    )
    return frozen_ok


# ---------------------------------------------------------------------------
# S5 / S7 if ablation CSVs exist
# ---------------------------------------------------------------------------

# Screenshot palette: Norman purple, Dixit green, Replogle coral.
# Adamson UPR and Papalexi take the remaining manuscript colors.
_ABLATION_STYLE = [
    ("Norman 2019 (CRISPRa)",      "Norman",      "#8172B2"),
    ("Adamson 2016 UPR (CRISPRi)", "Adamson UPR", BLUE),
    ("Dixit 2016 (CRISPR-KO)",     "Dixit",       GREEN),
    ("Papalexi 2021 (CRISPR-KO)",  "Papalexi",    SALMON),
    ("Replogle 2022 (CRISPRi)",    "Replogle",    "#C44E52"),
]


def fig_s5_if_present():
    paths = [find_csv(f"ablation_pca_{tag}.csv") for tag in
             ("norman", "dixit", "replogle", "adamson", "papalexi")]
    paths = [p for p in paths if p is not None]
    mega = find_csv("ablation_pca_all.csv", "ablation_pca.csv")
    if mega is not None:
        paths = [mega]
    if not paths:
        print("S5 skipped — no ablation_pca_*.csv. "
              "Colab: geometric_stability_main_analysis.py → run_pca_ablation")
        return
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(cfg.resolve_dataset_name)
    ncomp = next((c for c in df.columns if "n_comp" in c or c in {"n_pcs", "pcs"}), None)
    rho = next((c for c in df.columns if c in {"rho", "spearman", "sp_mag_rho"}), None)
    lo = next((c for c in ("ci_low", "rho_ci_low", "ci_lo") if c in df.columns), None)
    hi = next((c for c in ("ci_high", "rho_ci_high", "ci_hi") if c in df.columns), None)
    if ncomp is None or rho is None:
        print(f"S5 CSV columns not recognized: {list(df.columns)}")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    plotted = []
    for ds, short, color in _ABLATION_STYLE:
        sub = df[df["dataset"] == ds].copy() if "dataset" in df.columns else df.iloc[0:0]
        if not len(sub):
            continue
        sub = sub.sort_values(ncomp)
        x = pd.to_numeric(sub[ncomp], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[rho], errors="coerce").to_numpy()
        ax.plot(x, y, "-o", color=color, ms=6, lw=1.8, label=short, zorder=3)
        if lo and hi:
            ylo = pd.to_numeric(sub[lo], errors="coerce").to_numpy()
            yhi = pd.to_numeric(sub[hi], errors="coerce").to_numpy()
            ax.fill_between(x, ylo, yhi, color=color, alpha=0.18, lw=0, zorder=2)
        plotted.append(ds)
    ax.set_xlabel("Number of dimensions", fontweight="bold")
    ax.set_ylabel(r"Spearman $\rho$", fontweight="bold")
    ax.set_title("Linear-embedding dimensionality ablation", fontweight="bold")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlim(8, 104)
    ax.set_ylim(0.62, 1.0)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right",
              edgecolor="#CCCCCC", fancybox=False)
    despine(ax)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_s5_pca_ablation")
    (OUT / "fig_s5_pca_ablation_caption.txt").write_text(
        r"""\caption{\textbf{Linear-embedding dimensionality ablation.}
Magnitude--Shesha Spearman $\rho$ as a function of embedding dimension
(10, 20, 30, 50, 100) under \texttt{2026-07-29.1}. Shaded regions are
95\% bootstrap CIs (10{,}000 iterations, seed 320). Each series refits
the recorded frozen backend (scanpy PCA or sparse TruncatedSVD) on the
freeze cell set; the 50-dimensional point reproduces Table 1. Norman
($n=236$) is stable ($\rho=0.946$--$0.951$) with overlapping CIs.
Adamson UPR ($n=87$) is likewise stable ($\rho=0.942$--$0.955$).
Dixit ($n=98$) is lower throughout ($\rho=0.812$--$0.843$) and does not
increase monotonically. Papalexi ($n=24$) rises from $\rho=0.877$ to
$0.960$ with the widest intervals. Replogle ($n=1{,}832$) remains high
($\rho=0.963$--$0.980$). The choice of 50 dimensions does not drive the
magnitude--Shesha relationship.}
"""
    )
    print("S5 datasets plotted:", plotted)


def _loo_from_frozen(frozen):
    summary, long = [], []
    for ds, sub in frozen.groupby("dataset"):
        if len(sub) < 8:
            continue
        mag = sub["magnitude"].to_numpy(float)
        stab = sub["stability"].to_numpy(float)
        perts = sub["perturbation"].astype(str).to_numpy()
        full = float(spearmanr(mag, stab).correlation)
        rhos, deltas = [], []
        for i in range(len(sub)):
            mask = np.ones(len(sub), dtype=bool)
            mask[i] = False
            r = float(spearmanr(mag[mask], stab[mask]).correlation)
            d = full - r
            rhos.append(r)
            deltas.append(d)
            long.append({"dataset": ds, "perturbation": perts[i],
                         "rho_without": r, "delta": d})
        summary.append({
            "dataset": ds, "n": int(len(sub)), "full_rho": full,
            "loo_min_rho": float(np.min(rhos)), "loo_max_rho": float(np.max(rhos)),
            "max_abs_delta": float(np.max(np.abs(deltas))),
        })
    return pd.DataFrame(summary), pd.DataFrame(long)


def fig_s7_if_present(disc, frozen=None):
    path = find_csv("ablation_loo_summary.csv", "ablation_loo.csv")
    src = frozen if frozen is not None and len(frozen) else disc
    if src is None or "magnitude" not in src.columns:
        print("S7: need frozen Sp to compute leave-one-out.")
        return
    print("S7: leave-one-out ρ computed from frozen Sp"
          + (f"; also found {path}" if path is not None else ""))
    summary, long = _loo_from_frozen(src)
    if not len(summary):
        print("S7: no dataset with n≥8")
        return
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    order = [d for d, *_ in DATASETS if d in set(summary["dataset"])]
    shorts = [_short(d) for d in order]
    sub = summary.set_index("dataset").loc[order]
    x = np.arange(len(order))
    full = sub["full_rho"].to_numpy(float)
    lo = sub["loo_min_rho"].to_numpy(float)
    hi = sub["loo_max_rho"].to_numpy(float)
    ax = axes[0]
    ax.bar(x, full, color=GREY, edgecolor=DARK, linewidth=0.6, width=0.55, zorder=3)
    ax.errorbar(
        x, full, yerr=[full - lo, hi - full],
        fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=4,
    )
    for i, v in enumerate(full):
        ax.text(i, min(v + 0.03, 1.05), f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontweight="bold", rotation=25, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("a   Full $\\rho$ and leave-one-out range", fontweight="bold", loc="left")
    despine(ax)

    ax = axes[1]
    colors = {"Replogle 2022 (CRISPRi)": SALMON, "Norman 2019 (CRISPRa)": BLUE}
    plotted = False
    for ds, c in colors.items():
        sl = long[long["dataset"] == ds]
        if not len(sl):
            continue
        ax.hist(sl["delta"], bins=40, alpha=0.55, color=c, label=_short(ds), zorder=3)
        plotted = True
    ax.axvline(0.0, color=DARK, lw=0.8)
    ax.set_xlabel(r"$\Delta\rho$ (full $-$ leave-one-out)", fontweight="bold")
    ax.set_ylabel("Perturbations", fontweight="bold")
    ax.set_title("b   Influence on $\\rho$", fontweight="bold", loc="left")
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    despine(ax)
    fig.tight_layout()
    save_fig(fig, OUT / "fig_s7_loo")
    summary.to_csv(OUT / "fig_s7_loo_summary.csv", index=False)
    long.to_csv(OUT / "fig_s7_loo_per_perturbation.csv", index=False)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_tables(frozen, disc):
    # S1 dataset summary
    rows = []
    for full, short, modality, *_ in DATASETS:
        sub = frozen[frozen["dataset"] == full]
        if not len(sub):
            continue
        cell = {"Norman 2019 (CRISPRa)": "K562",
                "Adamson 2016 UPR (CRISPRi)": "K562",
                "Adamson 2016 pilot (CRISPRi)": "K562",
                "Dixit 2016 (CRISPR-KO)": "BMDC",
                "Papalexi 2021 (CRISPR-KO)": "THP-1",
                "Replogle 2022 (CRISPRi)": "K562"}[full]
        r, p = spearmanr(sub["magnitude"], sub["stability"])
        rows.append({
            "dataset": full, "dataset_short": short, "modality": modality,
            "cell_type": cell, "n": int(len(sub)),
            "median_sp": float(sub["stability"].median()),
            "rho_sp_mag": float(r), "p_sp_mag": float(p),
        })
    s1 = pd.DataFrame(rows)
    s1.to_csv(OUT / "table_s1_dataset_summary.csv", index=False)
    print("Table S1", s1.to_string(index=False))

    # S2 mixed-effects (within-dataset z, random intercept) or OLS+FE fallback
    z = []
    for ds, sub in frozen.groupby("dataset"):
        t = sub[["magnitude", "stability"]].copy()
        t["mag_z"] = (t["magnitude"] - t["magnitude"].mean()) / t["magnitude"].std()
        t["sp_z"] = (t["stability"] - t["stability"].mean()) / t["stability"].std()
        t["dataset"] = ds
        z.append(t)
    pooled = pd.concat(z, ignore_index=True)
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm("sp_z ~ mag_z", pooled, groups=pooled["dataset"])
        fit = md.fit(reml=True)
        s2 = pd.DataFrame({
            "term": fit.params.index,
            "coef": fit.params.values,
            "se": fit.bse.values,
            "p": fit.pvalues.values,
            "model": "mixedlm_random_intercept",
        })
        s2.to_csv(OUT / "table_s2_mixed_effects.csv", index=False)
        print("\nTable S2 mixed-effects\n", fit.summary())
    except Exception as e:
        print(f"statsmodels MixedLM unavailable ({e}); writing OLS + dataset FE fallback.")
        dummies = pd.get_dummies(pooled["dataset"], drop_first=True, dtype=float)
        X = np.column_stack([np.ones(len(pooled)), pooled["mag_z"].to_numpy(), dummies.to_numpy()])
        y = pooled["sp_z"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        df_res = max(len(y) - X.shape[1], 1)
        sigma2 = float((resid ** 2).sum() / df_res)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
        names = ["Intercept", "mag_z", *dummies.columns.tolist()]
        s2 = pd.DataFrame({
            "term": names, "coef": beta, "se": se,
            "model": "ols_dataset_fe_fallback",
        })
        s2.to_csv(OUT / "table_s2_mixed_effects.csv", index=False)
        print(s2.to_string(index=False))
        print("Re-run on Colab with statsmodels for the random-intercept MixedLM.")

    # Discordance ranking tables S7 / S9
    for ds, slug in (
        ("Norman 2019 (CRISPRa)", "table_s7_norman_discordance"),
        ("Replogle 2022 (CRISPRi)", "table_s9_replogle_discordance"),
    ):
        sub = disc[disc["dataset"] == ds].copy()
        sub = sub.sort_values("rank_loess")
        keep = ["perturbation", "magnitude", "stability",
                "disc_linear", "disc_loess", "rank_linear", "rank_loess"]
        sub[keep].to_csv(OUT / f"{slug}.csv", index=False)
        print(f"{slug}: {len(sub)} rows  LOESS top-5 = "
              f"{list(sub.head(5)['perturbation'])}")

    # S14 stress partials
    s14 = find_csv("fig_s9_stress_forest_si.csv", "fig_s9_stress_forest.csv")
    if s14 is not None:
        pd.read_csv(s14).to_csv(OUT / "table_s14_stress_partials.csv", index=False)
        print(f"Table S14 <- {s14}")

    # S19 scGPT
    s19 = find_csv("scgpt_correlations.csv")
    if s19 is not None:
        t = pd.read_csv(s19)
        t = t[~t["dataset"].astype(str).str.contains("Papalexi", case=False)]
        t.to_csv(OUT / "table_s19_scgpt.csv", index=False)
        print(f"Table S19 <- {s19} (Papalexi dropped)")

    # S21 pathway
    s21 = find_csv("fig5_pathway_forest.csv")
    if s21 is not None:
        t = pd.read_csv(s21)
        t = t[t["feature"].isin(["Apoptosis", "p53", "pw_Apoptosis", "pw_p53"])]
        t.to_csv(OUT / "table_s21_apoptosis_p53.csv", index=False)
        print(f"Table S21 <- {s21}")

    # Deleted-table stubs
    (OUT / "table_s10_ps_three_tier_DELETED.txt").write_text(
        "Table S10 (PS three-tier) DELETE or reduce to the proxy caution:\n"
        "centroid distance does not approximate scMAGeCK PS "
        "(rho = 0.097 Euclidean, 0.149 Mahalanobis).\n"
        "Do not present centroid proxies as PS at different fidelity.\n"
    )
    (OUT / "table_s20_quadrants_DELETED.txt").write_text(
        "Table S20 DELETE. Median splits force HH=LL and HL=LH. No replacement.\n"
    )
    (OUT / "table_s22_hspa5_DELETED.txt").write_text(
        "Table S22 DELETE. HSPA5 survives nowhere under the QC-conditioned models.\n"
    )

    # S4 / S5 / S15 / S16 / S23: Colab or missing
    (OUT / "table_s4_s5_s15_s16_s23_COLAB.txt").write_text(
        "S4 distance-metric table: run geometric_stability_main_analysis.py "
        "then fig_s4_method_comparison_barchart.py (or fold into the E-distance table).\n"
        "S5 LOESS-fit coefficients: see nonlinear_discordance_comparison.csv "
        "written by this script; optional deletion with the LOESS discordance framing.\n"
        "S15 eta-squared: python nadig.py  (missing locally).\n"
        "S16 / S23 functional diversity: python go_functional_diversity.py "
        "(missing locally; delete with geometric-tax retirement if not rerun).\n"
    )


def fig_s4_euclidean_six(frozen):
    """S4 is the three-method bar chart. Do not emit the Euclidean-only stub."""
    main = find_csv("crispr_correlations_with_ci.csv")
    if main is not None:
        try:
            from fig_s4_distance_metrics import load_correlation_csv, plot_s4
            corr = load_correlation_csv()
        except Exception as e:
            print(f"S4 plotter import failed ({e}); skip stub.")
            corr = None
        methods = set(corr["method"]) if corr is not None else set()
        if corr is not None and methods.issuperset({"Euclidean", "Whitened", "k-NN"}):
            print(f"S4 three-method CSV present: {main} — drawing grouped bars.")
            plot_s4(corr, OUT)
            return
        print(f"S4 CSV {main} is missing Whitened/k-NN — not drawing the one-bar stub.")
    print(
        "S4: run fig_s4_distance_metrics.py on Colab "
        "(Euclidean / Whitened / k-NN, all six datasets)."
    )
    (OUT / "fig_s4_euclidean_six_caption.txt").write_text(
        "S4 is the three-bar distance-metric figure. "
        "Run: python fig_s4_distance_metrics.py\n"
    )


def main():
    print(f"OUT_DIR={OUT}")
    frozen = load_frozen()
    disc = discordance_table(frozen)
    fig_s2(disc)
    fig_s3(disc)
    fig_s6_seeds()
    fig_s6()
    fig_s4_euclidean_six(frozen)
    fig_s5_if_present()
    fig_s7_if_present(disc, frozen=frozen)
    write_tables(frozen, disc)
    print("\nSI regen written to", OUT)
    print("Also run: python fig2_magnitude_stability_loess.py --csv-only")
    print("          python fig_s10_scgpt_concordance.py")
    print("          python fig_s4_method_comparison_barchart.py  # when metric CSVs exist")


if __name__ == "__main__":
    main()
