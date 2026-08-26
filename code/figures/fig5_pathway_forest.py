#!/usr/bin/env python3
"""
Figure 5 — apoptosis, p53, and DDIT3 forests (replaces old HSPA5 / quadrant Fig 5).

Panel a: Apoptosis, five datasets, raw + three QC models.
Panel b: p53, same construction.
Panel c: DDIT3, same construction; Replogle raw vs partial annotated.

Never plots abs_rho_partial. No quadrant tests. HSPA5 is not featured.

USAGE:
    python fig5_pathway_forest.py
"""

from __future__ import annotations

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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pipeline_config as cfg
from revision_io import data_search_dirs
import stats_utils as _stats_utils
from stats_utils import (
    bootstrap_partial_spearman_ci,
    partial_spearman_rank,
    pathway_bootstrap_seed,
    survival_status,
)

if _stats_utils.pg is not None:
    _stats_utils.pg = None

FEATURES = [
    ("Apoptosis", "pw_Apoptosis", "pathway"),
    ("p53", "pw_p53", "pathway"),
    ("DDIT3", "stress_DDIT3", "stress"),
]
QC_MODELS = ("centroid+QC", "edistance+QC", "centroid+edistance+QC")
FDR_FAMILY = getattr(cfg, "SURVIVAL_FDR_FAMILY_ID", "bh_dataset_model_9features")

DATASETS = [
    ("Norman 2019 (CRISPRa)", "Norman 2019"),
    ("Adamson 2016 UPR (CRISPRi)", "Adamson 2016 UPR"),
    ("Dixit 2016 (CRISPR-KO)", "Dixit 2016"),
    ("Papalexi 2021 (CRISPR-KO)", "Papalexi 2021"),
    ("Replogle 2022 (CRISPRi)", "Replogle 2022"),
]

_CSV_ROOTS = data_search_dirs()

PLOT_MODELS = [
    ("raw", "#888888", "o", "raw"),
    ("centroid+QC", "#4C72B0", "s", r"| centroid+QC"),
    ("edistance+QC", "#2CA02C", "D", r"| E-distance+QC"),
    ("centroid+edistance+QC", "#E5A84B", "^", r"| centroid+E-distance+QC"),
]

EDIST_MODELS = {
    "edistance+QC": (["edistance"], "partial_edistance_qc"),
    "centroid+edistance+QC": (
        ["magnitude", "edistance"],
        "partial_mag_edistance_qc",
    ),
}


def _find(*names):
    for name in names:
        for root in _CSV_ROOTS:
            p = (root / name).resolve()
            if p.exists():
                return p
    return None


def _short(ds):
    return next((s for f, s in DATASETS if f == ds), ds)


def _from_centroid(partials):
    rows = []
    want = {col: name for name, col, _ in FEATURES}
    for _, r in partials.iterrows():
        feat = str(r["feature"])
        if feat not in want:
            continue
        name = want[feat]
        ds = r["dataset"]
        n = int(r["n"])
        descriptive = bool(r.get("qc_descriptive_only", False))
        n_boot = int(r["n_bootstrap"]) if pd.notna(r.get("n_bootstrap", np.nan)) else cfg.N_BOOTSTRAP
        rows.append({
            "dataset": ds, "dataset_short": _short(ds), "feature": name,
            "n": n, "model": "raw", "rho": float(r["rho_raw"]),
            "ci_low": np.nan, "ci_high": np.nan,
            "p": float(r["p_raw"]) if pd.notna(r["p_raw"]) else np.nan,
            "status": "raw", "survives": False, "descriptive": False,
            "n_bootstrap": 0, "bootstrap_seed": np.nan,
            "source": "cell_quality_partials",
            "bootstrap_procedure": "none (raw, uncorrected)",
        })
        rho = float(r["rho_partial_mag_qc"])
        status = str(r["survival_status_qc"])
        seed = float(r["bootstrap_seed_qc"]) if pd.notna(r.get("bootstrap_seed_qc", np.nan)) else np.nan
        rows.append({
            "dataset": ds, "dataset_short": _short(ds), "feature": name,
            "n": n, "model": "centroid+QC", "rho": rho,
            "ci_low": float(r["rho_partial_mag_qc_ci_low"]) if pd.notna(r["rho_partial_mag_qc_ci_low"]) else np.nan,
            "ci_high": float(r["rho_partial_mag_qc_ci_high"]) if pd.notna(r["rho_partial_mag_qc_ci_high"]) else np.nan,
            "p": float(r["p_partial_mag_qc"]) if pd.notna(r["p_partial_mag_qc"]) else np.nan,
            "status": status, "survives": status == "survives",
            "descriptive": descriptive,
            "n_bootstrap": 0 if descriptive else n_boot,
            "bootstrap_seed": seed,
            "source": "cell_quality_partials",
            "bootstrap_procedure": "bootstrap_partial_spearman_ci/rank",
        })
    return pd.DataFrame(rows)


def _ddit3_from_s9(s9):
    keep = [
        "dataset", "dataset_short", "n", "model", "rho", "ci_low", "ci_high",
        "p", "descriptive", "n_bootstrap", "bootstrap_seed",
        "status", "survives", "fdr", "partial_r2",
    ]
    src = "marker" if "marker" in s9.columns else "feature"
    sub = s9[s9[src] == "DDIT3"].copy()
    sub = sub[sub["model"].isin(("edistance+QC", "centroid+edistance+QC"))]
    if src == "marker":
        sub = sub.rename(columns={"marker": "feature"})
    extra = [c for c in keep if c not in sub.columns]
    for c in extra:
        sub[c] = np.nan
    cols = ["feature"] + [c for c in keep if c in sub.columns]
    sub = sub[cols]
    sub["feature"] = "DDIT3"
    sub["source"] = "fig_s9_edistance_partials"
    sub["bootstrap_procedure"] = "bootstrap_partial_spearman_ci/rank"
    return sub


def _ddit3_edist_from_disk():
    """E-distance / joint DDIT3 rows. Prefer S9; else the assembled forest cache."""
    s9_path = _find("fig_s9_edistance_partials.csv")
    if s9_path is not None:
        print(f"DDIT3 E-distance: {s9_path}")
        return _ddit3_from_s9(pd.read_csv(s9_path))
    forest = _find("fig5_pathway_forest.csv")
    if forest is not None:
        t = pd.read_csv(forest)
        sub = t[
            (t["feature"] == "DDIT3")
            & (t["model"].isin(tuple(EDIST_MODELS)))
        ].copy()
        if len(sub):
            print(f"DDIT3 E-distance: {forest} ({len(sub)} cached rows)")
            return sub
    return pd.DataFrame()


def _compute_pathway_edist(merged, cache_path, n_bootstrap, already=None):
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    have = set()
    if not cached.empty:
        have = set(zip(cached["dataset"], cached["feature"], cached["model"]))
        print(f"Pathway E-distance cache: {len(have)} rows")
    if already:
        have |= set(already)

    qc_cols = ["qc_percent_mito", "qc_n_genes", "qc_n_counts"]
    mito_only_n = int(getattr(cfg, "SURVIVAL_QC_MITO_ONLY_MAX_N", 40))
    descriptive_n = int(getattr(cfg, "SURVIVAL_QC_DESCRIPTIVE_MAX_N", 30))
    rows = cached.to_dict("records") if not cached.empty else []
    all_feats = [(n, c) for n, c, _ in FEATURES]

    for ds_full, _ in DATASETS:
        sub0 = merged[merged["dataset"] == ds_full]
        if sub0.empty:
            continue
        for name, col in all_feats:
            if col not in sub0.columns:
                continue
            need = ["stability", "magnitude", "edistance", col]
            need += [c for c in qc_cols if c in sub0.columns]
            sub = sub0.dropna(subset=need)
            if len(sub) < 8:
                continue
            selected_qc = ["qc_percent_mito"] if len(sub) < mito_only_n else qc_cols
            descriptive = len(sub) < descriptive_n
            sp = sub["stability"].to_numpy(float)
            y = sub[col].to_numpy(float)
            for model, (effect_cols, stage) in EDIST_MODELS.items():
                if (ds_full, name, model) in have:
                    continue
                z = sub[[*effect_cols, *selected_qc]].to_numpy(float)
                seed = pathway_bootstrap_seed(
                    ds_full, col.replace("pw_", ""), stage, n_bootstrap=n_bootstrap,
                )
                print(
                    f"  {_short(ds_full):20s} {name:10s} {model:24s} "
                    f"n={len(sub)} "
                    + ("descriptive" if descriptive else f"boot={n_bootstrap:,}"),
                    flush=True,
                )
                if descriptive:
                    pt = partial_spearman_rank(sp, y, z)
                    part = {
                        "rho_partial": pt["rho_partial"], "p": pt["p"],
                        "ci_low": np.nan, "ci_high": np.nan,
                    }
                else:
                    part = bootstrap_partial_spearman_ci(
                        sp, y, z, n_bootstrap=n_bootstrap, seed=seed, method="rank",
                    )
                rho = part["rho_partial"]
                rows.append({
                    "dataset": ds_full, "dataset_short": _short(ds_full),
                    "feature": name, "n": len(sub), "model": model,
                    "rho": float(rho) if rho is not None else np.nan,
                    "ci_low": part.get("ci_low", np.nan),
                    "ci_high": part.get("ci_high", np.nan),
                    "p": part.get("p", np.nan),
                    "descriptive": descriptive,
                    "n_bootstrap": 0 if descriptive else n_bootstrap,
                    "source": "edistance_scores_per_pert",
                    "bootstrap_seed": seed,
                    "bootstrap_procedure": "bootstrap_partial_spearman_ci/rank",
                })
                pd.DataFrame(rows).to_csv(cache_path, index=False)
    return pd.DataFrame(rows)


def _attach_fdr_and_status(tab, family):
    fam = family.rename(columns={"fdr": "fdr_fam"})
    tab = tab.merge(
        fam[["dataset", "feature", "model", "fdr_fam", "fdr_rank", "family_size"]],
        on=["dataset", "feature", "model"],
        how="left",
    )
    tab["fdr"] = tab["fdr_fam"]
    tab = tab.drop(columns=["fdr_fam"])
    statuses, survives = [], []
    for _, r in tab.iterrows():
        if r["model"] == "raw":
            statuses.append("raw")
            survives.append(False)
            continue
        if bool(r.get("descriptive", False)):
            statuses.append("descriptive_small_n")
            survives.append(False)
            continue
        st = survival_status(r["rho"], r["ci_low"], r["ci_high"], fdr=r["fdr"])
        statuses.append(st["status"])
        survives.append(st["survives"])
    tab["status"] = statuses
    tab["survives"] = survives
    tab["partial_r2"] = [
        np.nan if model == "raw" or not np.isfinite(rho) else float(rho ** 2)
        for rho, model in zip(tab["rho"], tab["model"])
    ]
    tab["fdr_family"] = FDR_FAMILY
    qc_mask = tab["model"].isin(QC_MODELS)
    tab.loc[qc_mask, "bootstrap_procedure"] = "bootstrap_partial_spearman_ci/rank"
    tab.loc[tab["model"] == "raw", "bootstrap_procedure"] = "none (raw, uncorrected)"
    return tab


def _intersection(tab, feature):
    hits = set()
    sub = tab[(tab["feature"] == feature) & (tab["model"].isin(QC_MODELS))]
    for ds, g in sub.groupby("dataset_short"):
        if len(g) == 3 and bool(g["survives"].all()):
            hits.add(ds)
    return hits


def _draw_panel(ax, tab, feature, *, annotate_replogle=False):
    ds_order = [s for _, s in DATASETS]
    inter = _intersection(tab, feature)
    ylabels = [f"{ds} ★" if ds in inter else ds for ds in ds_order]
    colors = {m: c for m, c, _, _ in PLOT_MODELS}
    markers = {m: mk for m, _, mk, _ in PLOT_MODELS}
    model_names = [m for m, _, _, _ in PLOT_MODELS]
    offsets = np.linspace(-0.28, 0.28, len(model_names))
    sub = tab[tab["feature"] == feature]
    ymap = {ds: i for i, ds in enumerate(ds_order)}
    for j, model in enumerate(model_names):
        for _, r in sub[sub["model"] == model].iterrows():
            y = ymap[r["dataset_short"]] + offsets[j]
            faded = bool(r.get("descriptive", False)) or r["status"] == "descriptive_small_n"
            ax.scatter(
                r["rho"], y, s=68, c=colors[model], marker=markers[model],
                zorder=3, edgecolor="white", linewidth=0.5,
                alpha=0.45 if faded else 1.0,
            )
            if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
                ax.plot(
                    [r["ci_low"], r["ci_high"]], [y, y],
                    color=colors[model], lw=1.25, zorder=2,
                    alpha=0.45 if faded else 1.0,
                )
            if (
                model == "centroid+edistance+QC"
                and r["dataset_short"] in inter
                and np.isfinite(r["partial_r2"])
            ):
                ax.text(
                    r["ci_high"] + 0.03 if np.isfinite(r["ci_high"]) else r["rho"] + 0.03,
                    y, f"R²={r['partial_r2']:.2f}",
                    fontsize=7.5, va="center", color=colors[model],
                )
    ax.axvline(0, color="#888888", ls="--", lw=1, zorder=1)
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_title(feature, fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Spearman $\rho$ (signed)", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.85, 0.80)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    if annotate_replogle:
        raw = sub[(sub["dataset_short"] == "Replogle 2022") & (sub["model"] == "raw")]
        qc = sub[(sub["dataset_short"] == "Replogle 2022") & (sub["model"] == "centroid+QC")]
        if len(raw) and len(qc):
            y_qc = ymap["Replogle 2022"] + offsets[1]
            rho_qc = float(qc.iloc[0]["rho"])
            rho_raw = float(raw.iloc[0]["rho"])
            ax.annotate(
                rf"raw ${rho_raw:+.3f}$ $\rightarrow$ ${rho_qc:+.3f}$",
                xy=(rho_qc, y_qc),
                xycoords="data",
                xytext=(0.58, 0.10),
                textcoords="axes fraction",
                fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
            )


def _plot_forest(tab, out_dir):
    print("\n=== Intersection (all three QC models) ===")
    for feat, _, _ in FEATURES:
        hits = sorted(_intersection(tab, feat))
        print(f"  {feat}: {hits or 'none'}")
        for _, r in tab[(tab["feature"] == feat) & (tab["model"] != "raw")].iterrows():
            ci = (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                  if np.isfinite(r.get("ci_low", np.nan)) else "[n/a]")
            print(f"    {r['dataset_short']:20s} {r['model']:24s} "
                  f"ρ={r['rho']:+.3f}  {ci}  {r['status']}")

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.8), sharex=True)
    _draw_panel(axes[0], tab, "Apoptosis")
    _draw_panel(axes[1], tab, "p53")
    _draw_panel(axes[2], tab, "DDIT3", annotate_replogle=True)
    for i, letter in enumerate("abc"):
        axes[i].text(
            -0.08, 1.06, letter, transform=axes[i].transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right",
        )
    handles = [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor=c, markersize=8, label=lab)
        for _, c, mk, lab in PLOT_MODELS
    ]
    handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#222222",
               markersize=12, label="intersection survivor (all three QC models)")
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               fontsize=8.2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Pathway and DDIT3 coherence associations (signed)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    stem = out_dir / "fig5_pathway_forest"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {stem}.pdf / .png")


def main():
    try:
        from fig_style import resolve_out_dir
        out_dir = resolve_out_dir()
    except Exception:
        out_dir = Path("./shesha-crispr")
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = {
        "cell_quality_partials.csv": _find("cell_quality_partials.csv"),
        "fig_s9_fdr_family.csv": _find("fig_s9_fdr_family.csv"),
        "edistance_scores_per_pert.csv": _find("edistance_scores_per_pert.csv"),
        "cell_quality_per_perturbation.csv": _find("cell_quality_per_perturbation.csv"),
    }
    for name, path in needed.items():
        print(f"  {name}: {path or 'MISSING'}")
    missing = [k for k, v in needed.items() if v is None]

    cached = _find("fig5_pathway_forest.csv")
    if missing:
        if cached is not None:
            print(f"Inputs missing ({missing}); plotting cached table {cached}")
            tab = pd.read_csv(cached)
            extra = _ddit3_edist_from_disk()
            if len(extra):
                have = set(zip(tab["dataset"], tab["feature"], tab["model"]))
                add = extra[~extra.apply(
                    lambda r: (r["dataset"], r["feature"], r["model"]) in have, axis=1
                )]
                if len(add):
                    tab = pd.concat([tab, add], ignore_index=True)
            _plot_forest(tab, out_dir)
            return
        raise FileNotFoundError(
            "Need " + ", ".join(missing) +
            "or upload shesha-crispr/fig5_pathway_forest.csv and rerun."
        )

    partials_path = needed["cell_quality_partials.csv"]
    family_path = needed["fig_s9_fdr_family.csv"]
    ed_path = needed["edistance_scores_per_pert.csv"]
    qc_path = needed["cell_quality_per_perturbation.csv"]
    s9_path = _find("fig_s9_edistance_partials.csv")
    print(f"Partials: {partials_path}")
    print(f"FDR family: {family_path}")

    partials = pd.read_csv(partials_path)
    if "abs_rho_partial" in partials.columns and "rho_partial_mag_qc" not in partials.columns:
        raise ValueError("Refuse to plot abs_rho_partial as a signed forest.")
    family = pd.read_csv(family_path)
    tab = _from_centroid(partials)

    extra = _ddit3_edist_from_disk()
    if len(extra):
        tab = pd.concat([tab, extra], ignore_index=True)
    elif s9_path is not None:
        tab = pd.concat([tab, _ddit3_from_s9(pd.read_csv(s9_path))], ignore_index=True)

    ed = pd.read_csv(ed_path)
    qc = pd.read_csv(qc_path)
    ed["dataset"] = ed["dataset"].map(cfg.resolve_dataset_name)
    qc["dataset"] = qc["dataset"].map(cfg.resolve_dataset_name)
    merged = qc.merge(
        ed[["dataset", "perturbation", "edistance"]],
        on=["dataset", "perturbation"], how="inner",
    )
    already = set(zip(tab["dataset"], tab["feature"], tab["model"]))
    pw_ed = _compute_pathway_edist(
        merged, out_dir / "fig5_edistance_pathway_partials.csv", cfg.N_BOOTSTRAP,
        already=already,
    )
    tab = pd.concat([tab, pw_ed], ignore_index=True)
    tab = _attach_fdr_and_status(tab, family)
    tab.to_csv(out_dir / "fig5_pathway_forest.csv", index=False)
    qc_rows = tab[tab["model"].isin(QC_MODELS) & ~tab["descriptive"].fillna(False)]
    n_seed = int(qc_rows["bootstrap_seed"].notna().sum())
    print(
        f"Seed log: {n_seed}/{len(qc_rows)} non-descriptive QC rows have "
        f"bootstrap_seed; procedure is bootstrap_partial_spearman_ci/rank "
        f"via pathway_bootstrap_seed for all three inputs."
    )
    _plot_forest(tab, out_dir)


if __name__ == "__main__":
    main()
