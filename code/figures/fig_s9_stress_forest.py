#!/usr/bin/env python3
"""
S9 — signed stress-marker forest (replaces old Fig 5c / S9).

Featured: DDIT3, XBP1. Five datasets. Models:
  raw, |centroid+QC, |E-distance+QC, |centroid+E-distance+QC (joint).

Centroid+QC comes from frozen cell_quality_partials.csv.
E-distance models are computed from edistance_scores_per_pert.csv joined
to cell_quality_per_perturbation.csv, then cached.

Never plots abs_rho_partial.

USAGE:
    python fig_s9_stress_forest.py
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

# pingouin rebuilds a DataFrame on every draw; NumPy path is equivalent.
if _stats_utils.pg is not None:
    _stats_utils.pg = None

FEATURED = ["DDIT3", "XBP1"]
TABLE_ONLY = ["ATF4", "HSPA5"]
ALL_MARKERS = FEATURED + TABLE_ONLY
PATHWAY_COLS = ["pw_UPR", "pw_mTORC1", "pw_p53", "pw_Apoptosis", "pw_ROS"]
FDR_FAMILY = getattr(cfg, "SURVIVAL_FDR_FAMILY_ID", "bh_dataset_model_9features")
QC_MODELS = ("centroid+QC", "edistance+QC", "centroid+edistance+QC")

DATASETS = [
    ("Norman 2019 (CRISPRa)", "Norman 2019", "CRISPRa"),
    ("Adamson 2016 UPR (CRISPRi)", "Adamson 2016 UPR", "CRISPRi"),
    ("Dixit 2016 (CRISPR-KO)", "Dixit 2016", "CRISPR-KO"),
    ("Papalexi 2021 (CRISPR-KO)", "Papalexi 2021", "CRISPR-KO"),
    ("Replogle 2022 (CRISPRi)", "Replogle 2022", "CRISPRi"),
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


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1], start=0):
        rank = n - i
        val = min(prev, p[idx] * n / max(rank, 1))
        ranked[idx] = val
        prev = val
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    return np.clip(ranked, 0, 1), ranks, n


def _short(ds_full):
    return next((s for f, s, _ in DATASETS if f == ds_full), ds_full)


def _modality(ds_full):
    return next((m for f, _, m in DATASETS if f == ds_full), "")


def _from_centroid_partials(df):
    rows = []
    for _, r in df.iterrows():
        marker = str(r["feature"]).replace("stress_", "")
        ds = r["dataset"]
        n = int(r["n"])
        descriptive = bool(r.get("qc_descriptive_only", False))
        n_boot = int(r["n_bootstrap"]) if "n_bootstrap" in r.index and pd.notna(r["n_bootstrap"]) else cfg.N_BOOTSTRAP
        rows.append({
            "dataset": ds, "dataset_short": _short(ds),
            "modality": _modality(ds), "marker": marker, "n": n,
            "model": "raw", "rho": float(r["rho_raw"]),
            "ci_low": np.nan, "ci_high": np.nan,
            "p": float(r["p_raw"]) if pd.notna(r["p_raw"]) else np.nan,
            "fdr": np.nan, "partial_r2": np.nan,
            "status": "raw", "survives": False, "descriptive": False,
            "source": "cell_quality_partials", "n_bootstrap": 0,
            "fdr_family": FDR_FAMILY,
        })
        rho_m = float(r["rho_partial_mag"])
        rows.append({
            "dataset": ds, "dataset_short": _short(ds),
            "modality": _modality(ds), "marker": marker, "n": n,
            "model": "magnitude", "rho": rho_m,
            "ci_low": float(r["rho_partial_mag_ci_low"]) if pd.notna(r["rho_partial_mag_ci_low"]) else np.nan,
            "ci_high": float(r["rho_partial_mag_ci_high"]) if pd.notna(r["rho_partial_mag_ci_high"]) else np.nan,
            "p": float(r["p_partial_mag"]) if pd.notna(r["p_partial_mag"]) else np.nan,
            "fdr": float(r["p_partial_mag_fdr_bh"]) if pd.notna(r["p_partial_mag_fdr_bh"]) else np.nan,
            "partial_r2": float(rho_m ** 2) if np.isfinite(rho_m) else np.nan,
            "status": str(r["survival_status_mag"]),
            "survives": bool(r["survives_magnitude"]) if "survives_magnitude" in r.index else str(r["survival_status_mag"]) == "survives",
            "descriptive": False, "source": "cell_quality_partials",
            "n_bootstrap": n_boot, "fdr_family": FDR_FAMILY,
        })
        rho = float(r["rho_partial_mag_qc"])
        status = str(r["survival_status_qc"])
        rows.append({
            "dataset": ds, "dataset_short": _short(ds),
            "modality": _modality(ds), "marker": marker, "n": n,
            "model": "centroid+QC", "rho": rho,
            "ci_low": float(r["rho_partial_mag_qc_ci_low"]) if pd.notna(r["rho_partial_mag_qc_ci_low"]) else np.nan,
            "ci_high": float(r["rho_partial_mag_qc_ci_high"]) if pd.notna(r["rho_partial_mag_qc_ci_high"]) else np.nan,
            "p": float(r["p_partial_mag_qc"]) if pd.notna(r["p_partial_mag_qc"]) else np.nan,
            "fdr": float(r["p_partial_mag_qc_fdr_bh"]) if pd.notna(r["p_partial_mag_qc_fdr_bh"]) else np.nan,
            "partial_r2": float(rho ** 2) if np.isfinite(rho) else np.nan,
            "status": status, "survives": status == "survives",
            "descriptive": descriptive, "source": "cell_quality_partials",
            "n_bootstrap": 0 if descriptive else n_boot, "fdr_family": FDR_FAMILY,
        })
    return pd.DataFrame(rows)


def _expected_edist_keys():
    return {
        (ds, marker, model)
        for ds, _, _ in DATASETS
        for marker in ALL_MARKERS
        for model in EDIST_MODELS
    }


def _pathway_point_p(merged):
    """Point estimates for the five Hallmark scores under E-distance QC models."""
    qc_cols = ["qc_percent_mito", "qc_n_genes", "qc_n_counts"]
    mito_only_n = int(getattr(cfg, "SURVIVAL_QC_MITO_ONLY_MAX_N", 40))
    rows = []
    for ds_full, _, _ in DATASETS:
        sub0 = merged[merged["dataset"] == ds_full]
        if sub0.empty:
            continue
        for feat in PATHWAY_COLS:
            if feat not in sub0.columns:
                continue
            need = ["stability", "magnitude", "edistance", feat, *qc_cols]
            sub = sub0.dropna(subset=need)
            if len(sub) < 15:
                continue
            selected_qc = ["qc_percent_mito"] if len(sub) < mito_only_n else qc_cols
            sp = sub["stability"].to_numpy(float)
            y = sub[feat].to_numpy(float)
            for model, (effect_cols, _) in EDIST_MODELS.items():
                z = sub[[*effect_cols, *selected_qc]].to_numpy(float)
                pt = partial_spearman_rank(sp, y, z)
                rows.append({
                    "dataset": ds_full,
                    "model": model,
                    "feature": feat,
                    "feature_type": "pathway",
                    "n": int(pt["n"]),
                    "rho": pt["rho_partial"],
                    "p": pt["p"],
                })
    return pd.DataFrame(rows)


def _fdr_from_edistance_qc_table(edist):
    """Use pathway_qc_partials_edistance.csv if it already carries the 9-feature family."""
    path = _find("pathway_qc_partials_edistance.csv")
    if path is None:
        return None
    src = pd.read_csv(path)
    print(f"FDR source: {path}")
    feat_col = next((c for c in ("pathway", "outcome", "feature", "marker") if c in src.columns), None)
    model_col = next((c for c in ("covariate_model", "model") if c in src.columns), None)
    fdr_col = next((c for c in ("p_partial_fdr_bh", "fdr") if c in src.columns), None)
    if feat_col is None or model_col is None or fdr_col is None:
        return None
    src["marker"] = src[feat_col].astype(str).str.replace(r"^(stress_|pw_)", "", regex=True)
    src["model"] = src[model_col].map({
        "edistance+QC": "edistance+QC",
        "centroid_magnitude+edistance+QC": "centroid+edistance+QC",
        "centroid+edistance+QC": "centroid+edistance+QC",
    })
    src = src[src["model"].notna() & src["marker"].isin(ALL_MARKERS)]
    key = ["dataset", "marker", "model"]
    return src[key + [fdr_col]].rename(columns={fdr_col: "fdr"})


def _apply_family(df):
    """BH within (dataset × model). Returns the same rows with fdr, fdr_rank, family_size."""
    out = df.copy()
    out["fdr"] = np.nan
    out["fdr_rank"] = np.nan
    out["family_size"] = np.nan
    for (_, _), idx in out.groupby(["dataset", "model"]).groups.items():
        p = out.loc[idx, "p"].to_numpy(float)
        p_fill = np.where(np.isfinite(p), p, 1.0)
        fdr, ranks, n = _bh_fdr(p_fill)
        out.loc[idx, "fdr"] = fdr
        out.loc[idx, "fdr_rank"] = ranks
        out.loc[idx, "family_size"] = n
    out["fdr_family"] = FDR_FAMILY
    return out


def _audit_from_centroid(partials):
    rows = []
    for _, r in partials.iterrows():
        feat = str(r["feature"])
        ftype = "stress" if feat.startswith("stress_") else "pathway"
        name = feat.replace("stress_", "").replace("pw_", "")
        for model, rho_c, p_c in (
            ("magnitude", "rho_partial_mag", "p_partial_mag"),
            ("centroid+QC", "rho_partial_mag_qc", "p_partial_mag_qc"),
        ):
            rows.append({
                "dataset": r["dataset"],
                "model": model,
                "feature": name,
                "feature_col": feat,
                "feature_type": ftype,
                "n": int(r["n"]),
                "rho": float(r[rho_c]) if pd.notna(r[rho_c]) else np.nan,
                "p": float(r[p_c]) if pd.notna(r[p_c]) else np.nan,
            })
    return _apply_family(pd.DataFrame(rows))


def _finalize_edist(out, merged):
    """BH within (dataset × model) across 9 features — same family as centroid+QC."""
    out = out.copy()
    pw = _pathway_point_p(merged)
    print(
        f"FDR family {FDR_FAMILY}: BH within dataset×model "
        f"across {len(PATHWAY_COLS)} pathways + {len(ALL_MARKERS)} markers"
    )
    family_rows = []
    for _, r in out.iterrows():
        family_rows.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "feature": r["marker"],
            "feature_col": f"stress_{r['marker']}",
            "feature_type": "stress",
            "n": int(r["n"]),
            "rho": r["rho"],
            "p": r["p"],
        })
    for _, r in pw.iterrows():
        family_rows.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "feature": str(r["feature"]).replace("pw_", ""),
            "feature_col": r["feature"],
            "feature_type": "pathway",
            "n": int(r["n"]),
            "rho": r["rho"],
            "p": r["p"],
        })
    audit = _apply_family(pd.DataFrame(family_rows))
    stress_fdr = audit[audit["feature_type"] == "stress"][
        ["dataset", "feature", "model", "fdr", "fdr_rank", "family_size"]
    ].rename(columns={"feature": "marker"})
    out = out.drop(columns=["fdr", "fdr_rank", "family_size"], errors="ignore")
    out = out.merge(stress_fdr, on=["dataset", "marker", "model"], how="left")
    statuses, survives = [], []
    for _, r in out.iterrows():
        if bool(r["descriptive"]):
            statuses.append("descriptive_small_n")
            survives.append(False)
            continue
        st = survival_status(r["rho"], r["ci_low"], r["ci_high"], fdr=r["fdr"])
        statuses.append(st["status"])
        survives.append(st["survives"])
    out["status"] = statuses
    out["survives"] = survives
    out["fdr_family"] = FDR_FAMILY
    return out, audit


def _compute_edistance_models(merged, cache_path, n_bootstrap):
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    have = set()
    if not cached.empty and {"dataset", "marker", "model", "rho"}.issubset(cached.columns):
        have = set(zip(cached["dataset"], cached["marker"], cached["model"]))
        print(f"E-distance cache: {len(have)} / {len(_expected_edist_keys())} rows")

    qc_cols = ["qc_percent_mito", "qc_n_genes", "qc_n_counts"]
    mito_only_n = int(getattr(cfg, "SURVIVAL_QC_MITO_ONLY_MAX_N", 40))
    descriptive_n = int(getattr(cfg, "SURVIVAL_QC_DESCRIPTIVE_MAX_N", 30))
    rows = cached.to_dict("records") if not cached.empty else []

    for ds_full, _, _ in DATASETS:
        sub0 = merged[merged["dataset"] == ds_full]
        if sub0.empty:
            continue
        for marker in ALL_MARKERS:
            feat = f"stress_{marker}"
            need = ["stability", "magnitude", "edistance", feat, *qc_cols]
            sub = sub0.dropna(subset=need)
            if len(sub) < 15:
                print(f"  skip {ds_full} {marker}: n={len(sub)}")
                continue
            selected_qc = ["qc_percent_mito"] if len(sub) < mito_only_n else qc_cols
            descriptive = len(sub) < descriptive_n
            sp = sub["stability"].to_numpy(float)
            y = sub[feat].to_numpy(float)
            for model, (effect_cols, stage) in EDIST_MODELS.items():
                if (ds_full, marker, model) in have:
                    continue
                covar_cols = [*effect_cols, *selected_qc]
                z = sub[covar_cols].to_numpy(float)
                seed = pathway_bootstrap_seed(
                    ds_full, feat, stage, n_bootstrap=n_bootstrap,
                )
                print(
                    f"  {_short(ds_full):20s} {marker:5s} {model:24s} "
                    f"n={len(sub)} "
                    + ("descriptive" if descriptive else f"boot={n_bootstrap:,}"),
                    flush=True,
                )
                if descriptive:
                    pt = partial_spearman_rank(sp, y, z)
                    part = {
                        "rho_partial": pt["rho_partial"],
                        "p": pt["p"],
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                else:
                    part = bootstrap_partial_spearman_ci(
                        sp, y, z, n_bootstrap=n_bootstrap, seed=seed, method="rank",
                    )
                rho = part["rho_partial"]
                rows.append({
                    "dataset": ds_full,
                    "dataset_short": _short(ds_full),
                    "modality": _modality(ds_full),
                    "marker": marker,
                    "n": len(sub),
                    "model": model,
                    "rho": float(rho) if rho is not None else np.nan,
                    "ci_low": part.get("ci_low", np.nan),
                    "ci_high": part.get("ci_high", np.nan),
                    "p": part.get("p", np.nan),
                    "partial_r2": float(rho ** 2) if np.isfinite(rho) else np.nan,
                    "descriptive": descriptive,
                    "source": "edistance_scores_per_pert",
                    "n_bootstrap": 0 if descriptive else n_bootstrap,
                    "bootstrap_seed": seed,
                })
                pd.DataFrame(rows).to_csv(cache_path, index=False)

    out, audit = _finalize_edist(pd.DataFrame(rows), merged)
    out.to_csv(cache_path, index=False)
    return out, audit


SI_MODEL_PREFIX = {
    "raw": "raw",
    "magnitude": "magnitude",
    "centroid+QC": "centroid_qc",
    "edistance+QC": "edistance_qc",
    "centroid+edistance+QC": "joint",
}


def _wide_si(tab):
    rows = []
    for (ds, marker), g in tab.groupby(["dataset", "marker"], sort=False):
        row = {
            "dataset": ds,
            "dataset_short": g["dataset_short"].iloc[0],
            "modality": g["modality"].iloc[0],
            "marker": marker,
            "n": int(g["n"].iloc[0]),
            "fdr_family": FDR_FAMILY,
        }
        qc = g[g["model"].isin(QC_MODELS)]
        row["intersection_survives"] = bool(len(qc) == 3 and qc["survives"].all())
        for model, prefix in SI_MODEL_PREFIX.items():
            sub = g[g["model"] == model]
            if sub.empty:
                continue
            r = sub.iloc[0]
            row[f"rho_{prefix}"] = r["rho"]
            row[f"ci_low_{prefix}"] = r["ci_low"]
            row[f"ci_high_{prefix}"] = r["ci_high"]
            row[f"p_{prefix}"] = r["p"]
            if prefix != "raw":
                row[f"fdr_{prefix}"] = r["fdr"]
                row[f"fdr_rank_{prefix}"] = r.get("fdr_rank", np.nan)
                row[f"family_size_{prefix}"] = r.get("family_size", np.nan)
            row[f"status_{prefix}"] = r["status"]
            row[f"n_bootstrap_{prefix}"] = r["n_bootstrap"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    partials_path = _find("cell_quality_partials.csv")
    ed_path = _find("edistance_scores_per_pert.csv")
    qc_path = _find("cell_quality_per_perturbation.csv")
    if partials_path is None:
        raise FileNotFoundError("Need cell_quality_partials.csv")
    if ed_path is None:
        raise FileNotFoundError("Need edistance_scores_per_pert.csv")
    if qc_path is None:
        raise FileNotFoundError("Need cell_quality_per_perturbation.csv")
    print(f"Centroid+QC: {partials_path}")
    print(f"E-distance:  {ed_path}")
    print(f"QC table:    {qc_path}")

    raw = pd.read_csv(partials_path)
    if "abs_rho_partial" in raw.columns and "rho_partial_mag_qc" not in raw.columns:
        raise ValueError("Refuse to plot abs_rho_partial as a signed forest.")
    stress = raw[raw["feature"].isin([f"stress_{m}" for m in ALL_MARKERS])].copy()
    tab = _from_centroid_partials(stress)

    ed = pd.read_csv(ed_path)
    qc = pd.read_csv(qc_path)
    ed["dataset"] = ed["dataset"].map(cfg.resolve_dataset_name)
    qc["dataset"] = qc["dataset"].map(cfg.resolve_dataset_name)
    merged = qc.merge(
        ed[["dataset", "perturbation", "edistance"]],
        on=["dataset", "perturbation"],
        how="inner",
    )
    print(f"Joined n={len(merged)}  {merged.dataset.value_counts().to_dict()}")

    out_dir = Path("./shesha-crispr")
    out_dir.mkdir(parents=True, exist_ok=True)
    edist, edist_audit = _compute_edistance_models(
        merged,
        out_dir / "fig_s9_edistance_partials.csv",
        n_bootstrap=cfg.N_BOOTSTRAP,
    )
    centroid_audit = _audit_from_centroid(raw)
    audit = pd.concat([centroid_audit, edist_audit], ignore_index=True)
    audit.to_csv(out_dir / "fig_s9_fdr_family.csv", index=False)

    ranks = centroid_audit[centroid_audit["feature_type"] == "stress"][
        ["dataset", "feature", "model", "fdr", "fdr_rank", "family_size"]
    ].rename(columns={"feature": "marker"})
    tab = tab.drop(columns=["fdr_rank", "family_size"], errors="ignore")
    tab = tab.merge(ranks, on=["dataset", "marker", "model"], how="left", suffixes=("", "_fam"))
    if "fdr_fam" in tab.columns:
        # Prefer the reconstructed 9-feature FDR; keep stored value only if merge missed.
        tab["fdr"] = tab["fdr_fam"].where(tab["fdr_fam"].notna(), tab["fdr"])
        tab = tab.drop(columns=["fdr_fam"])

    keep = [
        "dataset", "dataset_short", "modality", "marker", "n", "model",
        "rho", "ci_low", "ci_high", "p", "fdr", "fdr_rank", "family_size",
        "partial_r2", "status", "survives", "descriptive", "source",
        "n_bootstrap", "fdr_family",
    ]
    for frame in (tab, edist):
        for col in keep:
            if col not in frame.columns:
                frame[col] = np.nan
    tab = pd.concat([tab[keep], edist[keep]], ignore_index=True)
    tab.to_csv(out_dir / "fig_s9_stress_forest.csv", index=False)
    wide = _wide_si(tab)
    wide.to_csv(out_dir / "fig_s9_stress_forest_si.csv", index=False)

    check = audit[
        (audit["dataset"].str.startswith("Adamson"))
        & (audit["feature"] == "ATF4")
        & (audit["model"] == "centroid+QC")
    ]
    if len(check):
        r = check.iloc[0]
        print(
            f"Audit check Adamson ATF4 | centroid+QC: "
            f"p={r['p']:.4g}  FDR={r['fdr']:.3f}  rank={int(r['fdr_rank'])}/{int(r['family_size'])}"
        )

    print("\n=== Featured (DDIT3, XBP1) ===")
    feat = tab[tab["marker"].isin(FEATURED)]
    for _, r in feat.sort_values(["marker", "dataset_short", "model"]).iterrows():
        ci = (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              if np.isfinite(r["ci_low"]) else "[n/a]")
        r2 = f"R²={r['partial_r2']:.3f}" if np.isfinite(r["partial_r2"]) else ""
        fdr = f"FDR={r['fdr']:.3f}" if pd.notna(r["fdr"]) else ""
        print(f"  {r['dataset_short']:20s} {r['marker']:5s} {r['model']:24s} "
              f"ρ={r['rho']:+.3f}  {ci}  {r2}  {fdr}  {r['status']}")

    print("\n=== Table-only ===")
    for marker in TABLE_ONLY:
        print(f"  {marker}:")
        for _, r in tab[(tab["marker"] == marker) & (tab["model"] != "raw")].iterrows():
            print(f"    {r['dataset_short']:20s} {r['model']:24s} "
                  f"ρ={r['rho']:+.3f}  {r['status']}")

    ds_order = [s for _, s, _ in DATASETS]
    colors = {m: c for m, c, _, _ in PLOT_MODELS}
    markers_pt = {m: mk for m, _, mk, _ in PLOT_MODELS}
    labels = {m: lab for m, _, _, lab in PLOT_MODELS}
    model_names = [m for m, _, _, _ in PLOT_MODELS]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharex=True)
    for ax, marker in zip(axes, FEATURED):
        sub = tab[tab["marker"] == marker]
        ymap = {ds: i for i, ds in enumerate(ds_order)}
        offsets = np.linspace(-0.28, 0.28, len(model_names))
        for j, model in enumerate(model_names):
            for _, r in sub[sub["model"] == model].iterrows():
                y = ymap[r["dataset_short"]] + offsets[j]
                faded = bool(r["descriptive"]) or r["status"] == "descriptive_small_n"
                ax.scatter(
                    r["rho"], y, s=68, c=colors[model], marker=markers_pt[model],
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
                    and r["survives"]
                    and np.isfinite(r["partial_r2"])
                ):
                    ax.text(
                        r["ci_high"] + 0.03 if np.isfinite(r["ci_high"]) else r["rho"] + 0.03,
                        y, f"R²={r['partial_r2']:.2f}",
                        fontsize=7.5, va="center", color=colors[model],
                    )
        ax.axvline(0, color="#888888", ls="--", lw=1, zorder=1)
        ax.set_yticks(range(len(ds_order)))
        ax.set_yticklabels(ds_order, fontsize=10)
        ax.set_title(marker, fontsize=13, fontweight="bold")
        ax.set_xlabel(r"Spearman $\rho$ (signed)", fontsize=10, fontweight="bold")
        ax.set_xlim(-0.85, 0.80)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, ls=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.invert_yaxis()

    handles = [
        Line2D([0], [0], marker=markers_pt[m], color="w",
               markerfacecolor=colors[m], markersize=8, label=labels[m])
        for m in model_names
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Stress-marker coherence associations (signed)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    stem = out_dir / "fig_s9_stress_forest"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {stem}.pdf / .png")
    print(f"Long table -> {out_dir / 'fig_s9_stress_forest.csv'}")
    print(f"SI table   -> {out_dir / 'fig_s9_stress_forest_si.csv'}")
    print(f"FDR family -> {out_dir / 'fig_s9_fdr_family.csv'}")


if __name__ == "__main__":
    main()
