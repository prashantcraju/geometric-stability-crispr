#!/usr/bin/env python3
"""
Revision figures: efficiency, embedding, E-distance, Approach B, methods robustness.

Looks for companion CSVs via SHESHA_OUT / ./shesha-crispr (see revision_io).

    python fig_revision_new.py
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
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import pipeline_config as cfg
from revision_io import load_sp_table
from fig_style import (
    BLUE, GREY, SALMON, GREEN, DARK, GATE,
    DATASETS, SCOREABLE, SEARCH_DIRS,
    find_csv, resolve_out_dir, despine, save_fig, grouped_x,
)

OUT = resolve_out_dir()

# scGPT omitted on the same thin-n / non-reproducible sets everywhere.
SCGPT_OMIT = {
    "Papalexi 2021 (CRISPR-KO)",
    "Adamson 2016 pilot (CRISPRi)",
}


def _short(name):
    for full, short, *_ in DATASETS:
        if full == name:
            return short
    return name.split("(")[0].strip()


def _legend_below_fig(fig, axes, ncol=2, x=0.06, bottom=None):
    """Legend immediately under the lowest x-tick label."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    handles, labels, seen = [], [], set()
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li in seen:
                continue
            seen.add(li)
            handles.append(hi)
            labels.append(li)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ymin = 1.0
    for ax in fig.axes:
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        ymin = min(ymin, inv.transform((0.0, bbox.y0))[1])
    fig.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(x, ymin - 0.006),
        bbox_transform=fig.transFigure,
        ncol=ncol, frameon=False, fontsize=8,
        handlelength=1.2, columnspacing=1.4,
    )


def load_frozen():
    path = find_csv("frozen_sp_scores.csv", "shesha_crispr_results_euclidean.csv")
    if path is None:
        raise FileNotFoundError(
            "Need frozen_sp_scores.csv (set SHESHA_OUT or place it in ./shesha-crispr)."
        )
    print(f"Frozen Sp: {path}")
    return load_sp_table(path)


# ---------------------------------------------------------------------------
# Efficiency (status §11). Locked PS/mixscape numbers if CSVs are absent.
# ---------------------------------------------------------------------------

LOCKED_EFFICIENCY = pd.DataFrame([
    {"dataset": "Norman 2019 (CRISPRa)", "method": "ps",
     "n_all": 236, "n_resp": 234, "rho_all": 0.950, "rho_resp": 0.902,
     "rank_rho": 0.918, "rank_ci_low": 0.890, "rank_ci_high": 0.937,
     "verdict": "preserved"},
    {"dataset": "Replogle 2022 (CRISPRi)", "method": "ps",
     "n_all": 1832, "n_resp": 1661, "rho_all": 0.978, "rho_resp": 0.976,
     "rank_rho": 0.916, "rank_ci_low": 0.908, "rank_ci_high": 0.923,
     "verdict": "preserved"},
    {"dataset": "Adamson 2016 UPR (CRISPRi)", "method": "ps",
     "n_all": 87, "n_resp": 80, "rho_all": 0.963, "rho_resp": 0.951,
     "rank_rho": 0.865, "rank_ci_low": 0.789, "rank_ci_high": 0.909,
     "verdict": "partial"},
    {"dataset": "Dixit 2016 (CRISPR-KO)", "method": "ps",
     "n_all": 98, "n_resp": 96, "rho_all": 0.841, "rho_resp": 0.847,
     "rank_rho": 0.642, "rank_ci_low": 0.507, "rank_ci_high": 0.748,
     "verdict": "change"},
    {"dataset": "Papalexi 2021 (CRISPR-KO)", "method": "ps",
     "n_all": 24, "n_resp": 24, "rho_all": 0.945, "rho_resp": 0.971,
     "rank_rho": 0.785, "rank_ci_low": 0.516, "rank_ci_high": 0.907,
     "verdict": "partial"},
    {"dataset": "Norman 2019 (CRISPRa)", "method": "mixscape",
     "n_all": 149, "n_resp": 149, "rho_all": 0.853, "rho_resp": 0.883,
     "rank_rho": 0.891, "rank_ci_low": 0.849, "rank_ci_high": 0.919,
     "verdict": "secondary", "note": "149/236 denominator; not a frozen baseline"},
])

LOCKED_AGREEMENT = {
    "n_shared": 23290, "ps_frac": 0.531, "mixscape_frac": 0.431,
    "both": 7521, "only_ps": 4836, "only_mix": 2512,
    "jaccard": 0.506, "kappa": 0.374, "agreement": 0.684,
}


def _first_num(row, *keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, dict):
            v = v.get("rho", v.get("value"))
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return np.nan


def _verdict_bin(raw):
    s = str(raw or "").lower()
    if "preserved" in s and "partial" not in s:
        return "preserved"
    if "partial" in s:
        return "partial"
    if "change" in s:
        return "change"
    return "partial"


def normalize_efficiency(blob, default_method=None):
    """Map efficiency_filter_summary_*.json onto the plot columns."""
    if isinstance(blob, list):
        rows, method = blob, default_method
    elif isinstance(blob, dict):
        method = blob.get("method", default_method)
        rows = blob.get("datasets", blob.get("summaries", [blob]))
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        return pd.DataFrame()

    out = []
    for raw in rows:
        if not isinstance(raw, dict) or "dataset" not in raw:
            continue
        rank = raw.get("rank_agreement_sp_all_vs_responder") or {}
        if not isinstance(rank, dict):
            rank = {}
        rho_all = _first_num(
            raw,
            "rho_all",
            "sp_magnitude_rho_frozen_full",
            "sp_magnitude_rho_all_cells",
        )
        rho_resp = _first_num(
            raw, "rho_resp", "sp_magnitude_rho_responders",
        )
        rank_rho = _first_num(raw, "rank_rho")
        if not np.isfinite(rank_rho):
            rank_rho = _first_num(rank, "rho")
        rank_lo = _first_num(raw, "rank_ci_low")
        if not np.isfinite(rank_lo):
            rank_lo = _first_num(rank, "ci_low")
        rank_hi = _first_num(raw, "rank_ci_high")
        if not np.isfinite(rank_hi):
            rank_hi = _first_num(rank, "ci_high")
        out.append({
            "dataset": cfg.resolve_dataset_name(raw["dataset"]),
            "method": str(raw.get("method", method or "ps")).lower(),
            "n_all": raw.get("n_perturbations", raw.get("n_all")),
            "n_resp": raw.get("n_rescored", raw.get("n_resp")),
            "rho_all": rho_all,
            "rho_resp": rho_resp,
            "rank_rho": rank_rho,
            "rank_ci_low": rank_lo,
            "rank_ci_high": rank_hi,
            "verdict": _verdict_bin(raw.get("verdict")),
        })
    return pd.DataFrame(out)


def load_efficiency():
    path = find_csv(
        "efficiency_filter_summary_ps.json",
        "efficiency_filter_scores.csv",
        "efficiency_filter_locked.csv",
    )
    if path is not None and path.suffix == ".json":
        print(f"Efficiency: {path}")
        blob = json.loads(path.read_text())
        df = normalize_efficiency(blob, default_method="ps")
        print("  columns:", list(df.columns))
        print(df.to_string(index=False))
        return df, False
    if path is not None and path.name == "efficiency_filter_scores.csv":
        print(f"Efficiency: {path}")
        return normalize_efficiency(pd.read_csv(path).to_dict("records")), False
    print("Efficiency CSVs not on disk — using locked status-file table (section 11).")
    return LOCKED_EFFICIENCY.copy(), True


def fig_efficiency():
    df, locked = load_efficiency()
    if "method" in df.columns and df["method"].astype(str).str.lower().eq("ps").any():
        ps = df[df["method"].astype(str).str.lower().eq("ps")].copy()
    else:
        ps = df.copy()
    if "rho_all" not in ps.columns:
        raise ValueError(
            f"efficiency table missing rho_all after normalize; columns={list(ps.columns)}"
        )
    order = [d for d, *_ in DATASETS if d in set(ps["dataset"])]
    if not order:
        order = list(ps["dataset"])
    shorts = [_short(d) for d in order]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))

    ax = axes[0]
    centers, offs = grouped_x(len(order), 2, width=0.28)
    for off, col, color, lab in (
        (offs[0], "rho_all", GREY, "All assigned"),
        (offs[1], "rho_resp", BLUE, "Responders only"),
    ):
        y = [float(ps.loc[ps["dataset"] == d, col].iloc[0]) for d in order]
        ax.bar(centers + off, y, width=0.26, color=color, edgecolor=DARK,
               linewidth=0.6, label=lab, zorder=3)
        for x, v in zip(centers + off, y):
            ax.text(
                x, v - 0.03, f"{v:.3f}",
                ha="center", va="top", fontsize=6.5, rotation=90, color="white",
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(shorts, fontweight="bold", rotation=25, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ (Shesha $\sim$ effect magnitude)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("a   Responder restriction (Song PS)", fontweight="bold", loc="left")
    despine(ax)

    ax = axes[1]
    y = [float(ps.loc[ps["dataset"] == d, "rank_rho"].iloc[0]) for d in order]
    lo = [float(ps.loc[ps["dataset"] == d, "rank_ci_low"].iloc[0]) for d in order]
    hi = [float(ps.loc[ps["dataset"] == d, "rank_ci_high"].iloc[0]) for d in order]
    colors = [GREEN if str(ps.loc[ps["dataset"] == d, "verdict"].iloc[0]) == "preserved"
              else (SALMON if str(ps.loc[ps["dataset"] == d, "verdict"].iloc[0]) == "change"
                    else GREY) for d in order]
    ax.bar(centers, y, width=0.55, color=colors, edgecolor=DARK, linewidth=0.6, zorder=3)
    ax.errorbar(centers, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=4)
    for x, v in zip(centers, y):
        ax.text(x, min(v + 0.05, 1.02), f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.9, color=GATE, ls="--", lw=0.8, zorder=1)
    ax.set_xticks(centers)
    ax.set_xticklabels(shorts, fontweight="bold", rotation=25, ha="right")
    ax.set_ylabel(r"$\rho$ (Shesha all, Shesha resp.)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("b   Ranking preservation", fontweight="bold", loc="left")
    despine(ax)

    ax = axes[2]
    ag = LOCKED_AGREEMENT
    agree_path = find_csv(
        "efficiency_filter_percell_agreement_ps_vs_mixscape.json",
        "efficiency_filter_percell_agreement_norman.json",
        "efficiency_filter_percell_agreement.json",
    )
    if agree_path is not None:
        raw_ag = json.loads(agree_path.read_text())
        print(f"Agreement: {agree_path}")
        ag = {
            **ag,
            "n_shared": raw_ag.get("n_shared", raw_ag.get("n_shared_cells", ag["n_shared"])),
            "ps_frac": raw_ag.get("ps_frac", raw_ag.get("frac_responders_a", ag["ps_frac"])),
            "mixscape_frac": raw_ag.get("mixscape_frac", raw_ag.get("frac_responders_b", ag["mixscape_frac"])),
            "both": raw_ag.get("both", raw_ag.get("n_both_responder", ag["both"])),
            "only_ps": raw_ag.get("only_ps", raw_ag.get("n_only_a", ag["only_ps"])),
            "only_mix": raw_ag.get("only_mix", raw_ag.get("n_only_b", ag["only_mix"])),
            "jaccard": raw_ag.get("jaccard", raw_ag.get("jaccard_responders", ag["jaccard"])),
            "kappa": raw_ag.get("kappa", raw_ag.get("cohens_kappa", ag["kappa"])),
            "agreement": raw_ag.get("agreement", raw_ag.get("agreement_rate", ag["agreement"])),
        }
    labels = ["Both", "PS only", "Mixscape only"]
    counts = [ag["both"], ag["only_ps"], ag["only_mix"]]
    cols = [BLUE, GREY, SALMON]
    ax.bar(np.arange(3), counts, color=cols, edgecolor=DARK, linewidth=0.6, zorder=3)
    for i, v in enumerate(counts):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Norman cells (n = 23,290 shared)")
    ax.set_title("c   PS vs Mixscape calls", fontweight="bold", loc="left")
    ax.text(
        0.97, 0.97,
        f"$\\kappa$ = {ag['kappa']:.3f}\nJaccard = {ag['jaccard']:.3f}\n"
        f"agree = {ag['agreement']:.1%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC"),
    )
    despine(ax)

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _legend_under(ax, handles=None, ncol=2, fontsize=8):
        inv = ax.transAxes.inverted()
        y_min = 0.0
        for lab in ax.get_xticklabels():
            if not str(lab.get_text()).strip():
                continue
            y_min = min(y_min, inv.transform((0.0, lab.get_window_extent(renderer).y0))[1])
        kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.0, y_min - 0.01),
            bbox_transform=ax.transAxes,
            frameon=False, fontsize=fontsize, ncol=ncol,
            handlelength=1.2, columnspacing=1.4,
            borderaxespad=0.0,
        )
        if handles is not None:
            ax.legend(handles=handles, **kw)
        else:
            ax.legend(**kw)

    _legend_under(axes[0], ncol=2, fontsize=8)
    _legend_under(
        axes[1],
        handles=[
            Patch(facecolor=GREEN, edgecolor=DARK, label=r"Preserved ($\rho>0.9$)"),
            Patch(facecolor=GREY, edgecolor=DARK, label="Partial"),
            Patch(facecolor=SALMON, edgecolor=DARK, label="Change"),
        ],
        ncol=3,
        fontsize=7,
    )
    save_fig(fig, OUT / "fig_efficiency_filter")
    out_csv = OUT / "fig_efficiency_filter.csv"
    ps.to_csv(out_csv, index=False)
    print(f"  table -> {out_csv}  (locked={locked})")
    caption = OUT / "fig_efficiency_filter_caption.txt"
    caption.write_text(
        r"""\caption{\textbf{Responder-only restriction does not create the magnitude--coherence relationship.}
(a)~Shesha $\sim$ effect-magnitude Spearman $\rho$ on all assigned cells versus Song-PS responders (threshold $\ge 0.5$).
Norman $0.950\to 0.902$; Adamson UPR $0.963\to 0.951$; Dixit $0.841\to 0.847$;
Papalexi $0.945\to 0.971$ ($24/24$ rescored); Replogle $0.978\to 0.976$.
The Adamson pilot ($n=8$) is omitted by the thin-$n$ floor used elsewhere.
(b)~Rank correlation of Shesha before vs after filtering, 95\% bootstrap CI.
Green = preserved ($\rho>0.9$); grey = partial; salmon = change.
Dixit is the coherent outlier ($\rho=0.642$); Papalexi is partial ($\rho=0.785$, wide CI).
(c)~Per-cell PS vs Mixscape agreement on Norman ($n=23{,}290$ shared cells): $\kappa=0.374$, Jaccard $=0.506$,
68.4\% agreement. ``Responder'' is method-defined, so the filter is a sensitivity analysis, not a correction.
Mixscape Norman uses the $149/236$ survivor set and is not a frozen baseline.}
"""
    )


# ---------------------------------------------------------------------------
# Embedding comparison (status §8) — Fig 4 proper
# ---------------------------------------------------------------------------

def _scgpt_vs_frozen(frozen):
    path = find_csv("scgpt_all_datasets.csv", "scgpt_vs_frozen_concordance.csv")
    if path is None:
        print("scGPT CSV not found.")
        return pd.DataFrame()
    print(f"scGPT: {path}")
    if path.name == "scgpt_vs_frozen_concordance.csv":
        return pd.read_csv(path)
    sc = pd.read_csv(path)
    sc["dataset"] = sc["dataset"].map(cfg.resolve_dataset_name)
    sc = sc.rename(columns={"stability": "sp_scgpt", "magnitude": "mag_scgpt"})
    pc = frozen.rename(columns={"stability": "sp_pca", "magnitude": "mag_pca"})
    m = sc.merge(
        pc[["dataset", "perturbation", "sp_pca", "mag_pca"]],
        on=["dataset", "perturbation"], how="inner",
    )
    rows = []
    for ds, sub in m.groupby("dataset"):
        if ds in SCGPT_OMIT:
            continue
        rsp = spearmanr(sub["sp_pca"], sub["sp_scgpt"]).correlation
        rmg = spearmanr(sub["mag_pca"], sub["mag_scgpt"]).correlation
        rows.append({"dataset": ds, "n": len(sub), "arm": "scGPT",
                     "rho_sp": float(rsp), "rho_mag": float(rmg)})
    return pd.DataFrame(rows)


def _diffmap_rows():
    path = find_csv("diffmap_embedding_summary.csv", "diffmap_vs_pca_sp_all.csv")
    if path is None:
        print("DiffMap CSV not found — leaving arm empty (run diffusion_map_robustness.py).")
        return pd.DataFrame()
    print(f"DiffMap: {path}")
    if path.name == "diffmap_embedding_summary.csv":
        d = pd.read_csv(path)
        d["dataset"] = d["dataset"].map(cfg.resolve_dataset_name)
        return pd.DataFrame({
            "dataset": d["dataset"],
            "n": d.get("n_perturbations", d.get("n")),
            "arm": "DiffMap",
            "rho_sp": d["spearman_sp_pca_vs_diffmap"],
            "rho_mag": d["spearman_magnitude_pca_vs_diffmap"],
            "config_version": d.get("config_version", ""),
        })
    d = pd.read_csv(path)
    d["dataset"] = d["dataset"].map(cfg.resolve_dataset_name)
    rows = []
    for ds, sub in d.groupby("dataset"):
        rows.append({
            "dataset": ds, "n": len(sub), "arm": "DiffMap",
            "rho_sp": float(spearmanr(sub["sp_pca"], sub["sp_diffmap"]).correlation),
            "rho_mag": float(spearmanr(sub["magnitude_pca"], sub["magnitude_diffmap"]).correlation),
        })
    return pd.DataFrame(rows)


def _phate_rows():
    path = find_csv(
        "phate_embedding_summary.csv",
        "phate_vs_pca_sp_all.csv",
        "phate_embedding_summary.json",
    )
    if path is None:
        print("PHATE CSV not found — HOLD slots only.")
        print("  Colab: !pip install -q phate")
        print("         !python phate_embedding_robustness.py --compare-frozen --mds-solver smacof")
        return pd.DataFrame()
    print(f"PHATE: {path}")
    if path.suffix == ".json":
        blob = json.loads(path.read_text())
        recs = blob.get("datasets", blob if isinstance(blob, list) else [])
        d = pd.DataFrame(recs)
    else:
        d = pd.read_csv(path)
    if "dataset" not in d.columns:
        return pd.DataFrame()
    d["dataset"] = d["dataset"].map(cfg.resolve_dataset_name)
    if "spearman_phate_vs_frozen_pca" in d.columns or "spearman_sp_pca_vs_phate" in d.columns:
        rho_sp = d["spearman_phate_vs_frozen_pca"] if "spearman_phate_vs_frozen_pca" in d.columns else d["spearman_sp_pca_vs_phate"]
        return pd.DataFrame({
            "dataset": d["dataset"],
            "n": d.get("n_perturbations", d.get("n")),
            "arm": "PHATE",
            "rho_sp": rho_sp,
            "rho_mag": d.get("spearman_magnitude_pca_vs_phate", np.nan),
        })
    if "sp_phate" not in d.columns:
        return pd.DataFrame()
    rows = []
    for ds, sub in d.groupby("dataset"):
        rho_sp = spearmanr(sub["sp_pca"], sub["sp_phate"]).correlation
        if "sp_frozen_pca" in sub.columns:
            rho_sp = spearmanr(sub["sp_phate"], sub["sp_frozen_pca"]).correlation
        rows.append({
            "dataset": ds, "n": len(sub), "arm": "PHATE",
            "rho_sp": float(rho_sp),
            "rho_mag": float(spearmanr(sub["magnitude_pca"], sub["magnitude_phate"]).correlation),
        })
    return pd.DataFrame(rows)


def fig_embedding(frozen):
    parts = [_diffmap_rows(), _phate_rows(), _scgpt_vs_frozen(frozen)]
    tab = pd.concat([p for p in parts if len(p)], ignore_index=True)
    arms = ["DiffMap", "PHATE", "scGPT"]
    colors = {"DiffMap": GREY, "PHATE": SALMON, "scGPT": BLUE}
    order = [d for d, *_ in DATASETS]
    shorts = [_short(d) for d in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, col, title in (
        (axes[0], "rho_sp", "a   Shesha vs frozen PCA"),
        (axes[1], "rho_mag", "b   Magnitude vs frozen PCA"),
    ):
        centers, offs = grouped_x(len(order), 3, width=0.24)
        for off, arm in zip(offs, arms):
            y = []
            for ds in order:
                hit = tab[(tab["dataset"] == ds) & (tab["arm"] == arm)]
                if arm == "scGPT" and ds in SCGPT_OMIT:
                    y.append(np.nan)
                elif len(hit):
                    y.append(float(hit[col].iloc[0]))
                else:
                    y.append(np.nan)
            y = np.asarray(y, float)
            present = np.isfinite(y)
            phate_hold = arm == "PHATE"
            labeled = False
            if present.any():
                ax.bar(
                    centers[present] + off, y[present],
                    width=0.22, color=colors[arm], edgecolor=DARK, linewidth=0.6,
                    label=arm, zorder=3,
                )
                labeled = True
            for x, v, ok in zip(centers + off, y, present):
                if ok:
                    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
                elif phate_hold:
                    lab = None if labeled else "PHATE (HOLD)"
                    ax.bar(
                        [x], [0.18], width=0.22, color=colors[arm],
                        edgecolor=DARK, linewidth=0.6, hatch="///", alpha=0.55,
                        label=lab, zorder=3,
                    )
                    ax.text(
                        x, 0.20, "HOLD", ha="center", va="bottom",
                        fontsize=6.5, color=SALMON, fontweight="bold", rotation=90,
                    )
                    labeled = True
            if arm != "PHATE" and not labeled:
                ax.bar([], [], width=0.22, color=colors[arm], edgecolor=DARK,
                       linewidth=0.6, label=arm)
        ax.set_xticks(centers)
        ax.set_xticklabels(shorts, fontweight="bold", rotation=20, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_title(title, fontweight="bold", loc="left")
        despine(ax)
    _legend_below_fig(fig, axes[0], ncol=3, x=0.18)
    save_fig(fig, OUT / "fig_embedding_comparison")
    tab.to_csv(OUT / "fig_embedding_comparison.csv", index=False)
    (OUT / "fig_embedding_comparison_caption.txt").write_text(
        r"""\caption{\textbf{Representation concordance against frozen PCA Shesha.}
Manuscript column is each embedding's Shesha (and, separately, its magnitude) versus frozen PCA scores,
not a within-embedding magnitude relationship (that check sits with Figure 2).
(a)~Shesha concordance. (b)~Magnitude concordance.
scGPT is omitted for Papalexi ($n=24$; GPU ranks not reproducible) and the Adamson pilot ($n=8$),
the same thin-$n$ floor used elsewhere. No $0.9$ reference line: this is not a binary
robust / limitation threshold.
Hatched PHATE bars marked HOLD have no converged embedding on disk; regenerate with
\texttt{phate\_embedding\_robustness.py --compare-frozen --mds-solver smacof}.
DiffMap values use the on-disk embedding table if present (re-run \texttt{diffusion\_map\_robustness.py}
under the current freeze if the CSV stamp is older than 2026-07-29.1).}
"""
    )


# ---------------------------------------------------------------------------
# E-distance competitor (status §3)
# ---------------------------------------------------------------------------

def fig_edistance():
    path = find_csv("edistance_dataset_correlations.csv")
    if path is None:
        raise FileNotFoundError(
            "Need edistance_dataset_correlations.csv. On Colab: "
            "python edistance_competitor_analysis.py --correlations-only"
        )
    print(f"E-distance: {path}")
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].map(cfg.resolve_dataset_name)
    order = [d for d, *_ in DATASETS if d in set(df["dataset"])]
    shorts = [_short(d) for d in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    centers, offs = grouped_x(len(order), 2, width=0.30)
    for off, col, color, lab in (
        (offs[0], "frac_Sp_var_left_after_centroid_magnitude", GREY, "After centroid"),
        (offs[1], "frac_Sp_var_left_after_edistance", SALMON, "After E-distance"),
    ):
        y = [float(df.loc[df["dataset"] == d, col].iloc[0]) for d in order]
        ax.bar(centers + off, y, width=0.28, color=color, edgecolor=DARK,
               linewidth=0.6, label=lab, zorder=3)
        for x, v in zip(centers + off, y):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(centers)
    ax.set_xticklabels(shorts, fontweight="bold", rotation=20, ha="right")
    ax.set_ylabel("Residual Shesha variance (rank-OLS)")
    ax.set_ylim(0, 0.40)
    ax.set_title("Variance left after effect magnitude", fontweight="bold")
    despine(ax)
    _legend_below_fig(fig, ax, ncol=2, x=0.28)
    save_fig(fig, OUT / "fig_edistance_competitor")
    df.to_csv(OUT / "fig_edistance_competitor.csv", index=False)
    (OUT / "fig_edistance_competitor_caption.txt").write_text(
        r"""\caption{\textbf{E-distance absorbs Dixit's extra residual variance.}
Residual Shesha variance after rank-OLS on centroid magnitude (grey) versus scPerturb
E-distance (salmon), six datasets. Dixit drops from $0.289$ to $0.108$; every other
dataset also leaves less residual under E-distance. The three-method magnitude--coherence
correlations (PCA / E-distance / scGPT) are in the methods-robustness figure.
E-distance is never labelled magnitude; it is the competitor used in the QC-conditioned
pathway models.}
"""
    )


# ---------------------------------------------------------------------------
# Approach B balance (status §3) — pre-specified gate, failed on cell quality
# ---------------------------------------------------------------------------

def _rank_resid(stab, mag):
    from scipy.stats import rankdata
    rsp, rmg = rankdata(stab), rankdata(mag)
    z = np.column_stack([np.ones(len(rsp)), rmg])
    coef = np.linalg.lstsq(z, rsp, rcond=None)[0]
    return rsp - z @ coef


def _smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / max(len(a) + len(b) - 2, 1))
    return 0.0 if pooled < 1e-12 else float((a.mean() - b.mean()) / pooled)


def _caliper_pairs(df, caliper=None):
    caliper = float(caliper or cfg.APPROACH_B_CALIPER_MAG_SD)
    mag_sd = float(df["magnitude"].std(ddof=1))
    cal = caliper * mag_sd if mag_sd > 0 else 0.0
    q1 = df.index[df["resid_q"] == "Q1"].tolist()
    q4 = (df.loc[df["resid_q"] == "Q4"]
          .assign(_abs=lambda d: d["sp_resid"].abs())
          .sort_values("_abs", ascending=False).index.tolist())
    used, pairs = set(), []
    for p4 in q4:
        m4 = float(df.loc[p4, "magnitude"])
        best, best_d = None, np.inf
        for p1 in q1:
            if p1 in used:
                continue
            d = abs(float(df.loc[p1, "magnitude"]) - m4)
            if d <= cal and d < best_d:
                best, best_d = p1, d
        if best is None:
            continue
        used.add(best)
        pairs.append((p4, best))
    return pairs


def compute_approach_b_balance(frozen):
    qc_path = find_csv("cell_quality_per_perturbation.csv")
    existing = list(filter(None, [
        find_csv(f"pathway_approach_b_balance_{tag}.csv")
        for tag in ("norman_2019", "replogle_2022")
    ]))
    if existing and qc_path is None:
        frames = []
        for p in existing:
            t = pd.read_csv(p)
            t["source"] = p.name
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    if qc_path is None:
        print("No cell_quality_per_perturbation.csv or Approach B balance CSVs.")
        return pd.DataFrame()
    print(f"QC table: {qc_path}")
    qc = pd.read_csv(qc_path)
    qc["dataset"] = qc["dataset"].map(cfg.resolve_dataset_name)
    cov_map = {
        "magnitude": "magnitude",
        "n_cells": "n_cells_qc",
        "n_counts": "qc_n_counts",
        "n_genes": "qc_n_genes",
        "percent_mito": "qc_percent_mito",
    }
    rows = []
    for ds in cfg.APPROACH_B_DATASETS:
        sub = qc[qc["dataset"] == ds].copy()
        if len(sub) < 40:
            fr = frozen[frozen["dataset"] == ds][["perturbation", "stability", "magnitude"]].copy()
            sub = fr.merge(sub, on="perturbation", how="inner", suffixes=("", "_qc"))
        sub = sub.dropna(subset=["stability", "magnitude"]).set_index(
            sub["perturbation"].astype(str)
        )
        sub.index = sub.index.astype(str)
        sub["sp_resid"] = _rank_resid(sub["stability"].to_numpy(), sub["magnitude"].to_numpy())
        try:
            sub["resid_q"] = pd.qcut(
                sub["sp_resid"], 4, labels=["Q4", "Q3", "Q2", "Q1"], duplicates="drop"
            )
        except ValueError:
            continue
        q1 = sub[sub["resid_q"] == "Q1"]
        q4 = sub[sub["resid_q"] == "Q4"]
        pairs = _caliper_pairs(sub)
        print(f"  {ds}: Q1={len(q1)} Q4={len(q4)} matched={len(pairs)}")
        for stage, a_idx, b_idx in (
            ("before", q4.index, q1.index),
            ("after", [p[0] for p in pairs], [p[1] for p in pairs]),
        ):
            for cov, col in cov_map.items():
                if col not in sub.columns:
                    continue
                smd = _smd(sub.loc[list(a_idx), col], sub.loc[list(b_idx), col])
                rows.append({
                    "dataset": ds, "stage": stage, "covariate": cov,
                    "n_q4": len(a_idx), "n_q1": len(b_idx),
                    "smd": smd,
                    "gate_fail": bool(np.isfinite(smd) and abs(smd) > cfg.APPROACH_B_SMD_MAX),
                })
    return pd.DataFrame(rows)


def fig_approach_b(frozen):
    bal = compute_approach_b_balance(frozen)
    if bal.empty:
        print("Approach B balance skipped (need QC CSV or Colab pathway_analysis.py).")
        return
    bal.to_csv(OUT / "fig_approach_b_balance.csv", index=False)
    covs = [c for c in ("magnitude", "n_cells", "n_counts", "n_genes", "percent_mito")
            if c in set(bal["covariate"])]
    dsets = [d for d in cfg.APPROACH_B_DATASETS if d in set(bal["dataset"])]
    gate = float(cfg.APPROACH_B_SMD_MAX)
    n_ax = max(len(dsets), 1)
    fig, axes = plt.subplots(1, n_ax, figsize=(5.6 * n_ax, 4.8),
                             squeeze=False, sharey=True)
    pad = 0.18
    ymin = min(float(bal["smd"].min()) - pad, -gate - pad)
    ymax = max(float(bal["smd"].max()) + 0.38, gate + 0.38)
    for ax, ds in zip(axes[0], dsets):
        sub = bal[bal["dataset"] == ds]
        centers, offs = grouped_x(len(covs), 2, width=0.32)
        plotted = {}
        for off, stage, color, lab in (
            (offs[0], "before", GREY, "Before matching"),
            (offs[1], "after", BLUE, "After caliper"),
        ):
            y, fail = [], []
            for c in covs:
                hit = sub[(sub["covariate"] == c) & (sub["stage"] == stage)]
                y.append(float(hit["smd"].iloc[0]) if len(hit) else np.nan)
                fail.append(bool(hit["gate_fail"].iloc[0]) if len(hit) else False)
            cols = [SALMON if f and stage == "after" else color for f in fail]
            ax.bar(centers + off, y, width=0.30, color=cols, edgecolor=DARK,
                   linewidth=0.6, label=lab, zorder=3)
            plotted[stage] = (centers + off, np.asarray(y, float), fail)
        # Labels after both series so near-gate / colliding values can be staggered.
        xb, yb, _ = plotted["before"]
        xa, ya, fa = plotted["after"]
        for i in range(len(covs)):
            extra_b = extra_a = 0.0
            if np.isfinite(yb[i]) and np.isfinite(ya[i]) and yb[i] * ya[i] >= 0:
                if abs(yb[i] - ya[i]) < 0.08:
                    if abs(ya[i]) >= abs(yb[i]):
                        extra_a = 0.16
                    else:
                        extra_b = 0.16
            for x, v, extra, f, stage in (
                (xb[i], yb[i], extra_b, False, "before"),
                (xa[i], ya[i], extra_a, fa[i], "after"),
            ):
                if not np.isfinite(v):
                    continue
                step = (0.05 + extra) if v >= 0 else -(0.05 + extra)
                fmt = f"{v:+.3f}" if abs(abs(v) - gate) < 0.02 else f"{v:+.2f}"
                ax.text(
                    x, v + step, fmt, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7,
                    color=SALMON if f and stage == "after" else DARK,
                )
        ax.axhline(0, color=DARK, lw=0.9, zorder=1)
        ax.axhline(gate, color=GATE, ls="--", lw=0.8, zorder=1)
        ax.axhline(-gate, color=GATE, ls="--", lw=0.8, zorder=1)
        ax.set_xticks(centers)
        ax.set_xticklabels(
            ["Magnitude", r"$n$ cells", "UMIs", "Genes", "% mito"][:len(covs)],
            fontweight="bold", rotation=15, ha="right",
        )
        ax.set_ylabel("SMD (Q4 − Q1)")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{_short(ds)}  (gate $|$SMD$|$ $\\leq$ {gate:g})",
                     fontweight="bold")
        despine(ax)
    axes[0][0].legend(
        handles=[
            Patch(facecolor=GREY, edgecolor=DARK, label="Before matching"),
            Patch(facecolor=BLUE, edgecolor=DARK, label=r"After caliper, $|$SMD$|$ $\leq$ 0.25"),
            Patch(facecolor=SALMON, edgecolor=DARK, label=r"After caliper, $|$SMD$|$ $> 0.25$"),
        ],
        frameon=False, fontsize=7, loc="lower left",
    )
    fig.tight_layout()
    save_fig(fig, OUT / "fig_approach_b_balance")
    (OUT / "fig_approach_b_balance_caption.txt").write_text(
        r"""\caption{\textbf{Approach B's pre-specified balance gate failed on cell quality.}
Residual-Shesha quartiles were caliper-matched on magnitude ($|\Delta M|\le 0.25$ SD).
The gate fails when $|$SMD$|>0.25$ (equality passes); it was set before the analysis ran.
Both panels share a $y$-axis. Magnitude balances after matching; cell-quality covariates
do not. Replogle fails genes ($-0.46$) and sits on the $n$-cell boundary at $+0.249$ (pass);
Norman fails $n$ cells, UMIs, and genes (to $-1.34$).
Salmon bars mark post-match failures. The arm is dropped; Approach A carries the pathway result.
This failed gate is the concrete answer to version-drift: the criterion was not relaxed after seeing the outcome.
The synthetic benchmark is not shown here --- its tolerance was chosen after seeing a result.}
"""
    )


# ---------------------------------------------------------------------------
# Robustness methods figure — all six datasets
# PCA centroid / E-distance / scGPT  (the three arms we can fill)
# ---------------------------------------------------------------------------

def fig_robustness_methods(frozen):
    ed_path = find_csv("edistance_dataset_correlations.csv")
    sc_path = find_csv("scgpt_all_datasets.csv")
    ed = pd.read_csv(ed_path) if ed_path is not None else pd.DataFrame()
    if len(ed):
        ed["dataset"] = ed["dataset"].map(cfg.resolve_dataset_name)
    sc = pd.read_csv(sc_path) if sc_path is not None else pd.DataFrame()
    if len(sc):
        sc["dataset"] = sc["dataset"].map(cfg.resolve_dataset_name)

    order = [d for d, *_ in DATASETS]
    shorts = [_short(d) for d in order]
    rows = []
    for ds in order:
        fr = frozen[frozen["dataset"] == ds]
        r_pca = spearmanr(fr["magnitude"], fr["stability"]).correlation if len(fr) > 3 else np.nan
        r_ed = (float(ed.loc[ed["dataset"] == ds, "rho_Sp_edistance"].iloc[0])
                if len(ed) and ds in set(ed["dataset"]) else np.nan)
        r_sc = np.nan
        if len(sc) and ds in set(sc["dataset"]) and ds not in SCGPT_OMIT:
            sub = sc[sc["dataset"] == ds]
            if len(sub) > 3:
                r_sc = spearmanr(sub["magnitude"], sub["stability"]).correlation
        rows.append({"dataset": ds, "PCA": r_pca, "E-distance": r_ed, "scGPT": r_sc})
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    centers, offs = grouped_x(len(order), 3, width=0.26)
    series = [("PCA", GREY), ("E-distance", SALMON), ("scGPT", BLUE)]
    for off, (name, color) in zip(offs, series):
        y = tab[name].to_numpy(float)
        present = np.isfinite(y)
        ax.bar(centers[present] + off, y[present], width=0.24,
               color=color, edgecolor=DARK, linewidth=0.6, label=name, zorder=3)
        for x, v, ok in zip(centers + off, y, present):
            if ok:
                ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(centers)
    ax.set_xticklabels(shorts, fontweight="bold")
    ax.set_ylabel(r"Spearman $\rho$ (Shesha $\sim$ effect magnitude)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Magnitude–coherence correlation by method", fontweight="bold")
    despine(ax)
    _legend_below_fig(fig, ax, ncol=3, x=0.28)
    save_fig(fig, OUT / "fig_robustness_methods")
    tab.to_csv(OUT / "fig_robustness_methods.csv", index=False)
    (OUT / "fig_robustness_methods_caption.txt").write_text(
        r"""\caption{\textbf{The magnitude--coherence relationship is not a PCA-centroid artifact.}
Spearman $\rho$ between Shesha and effect magnitude in the frozen PCA centroid (grey),
scPerturb E-distance (salmon), and scGPT embeddings (blue), six datasets.
scGPT is omitted for Papalexi ($n=24$; GPU ranks not reproducible) and the Adamson
pilot ($n=8$), the same thin-$n$ floor used elsewhere.
Whitened / $k$-NN distance-metric bars remain a Colab regeneration
(\texttt{geometric\_stability\_main\_analysis.py} $\to$ \texttt{fig\_s4\_method\_comparison\_barchart.py}).}
"""
    )


def main():
    print(f"OUT_DIR={OUT}")
    print(f"Search: {[str(p) for p in SEARCH_DIRS if p.exists()]}")
    frozen = load_frozen()
    fig_efficiency()
    fig_embedding(frozen)
    fig_edistance()
    fig_approach_b(frozen)
    fig_robustness_methods(frozen)
    print("\nNew revision figures written to", OUT)


if __name__ == "__main__":
    main()
