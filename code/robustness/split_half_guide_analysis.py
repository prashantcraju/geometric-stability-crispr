#!/usr/bin/env python3
"""
Split-half circularity fix — Replogle flagship.

1. Inspect Replogle metadata for guide-level annotations.
2. Predictive endpoint: gene-level Sp → cosine(guide_A mean shift, guide_B
   mean shift), with rank-based partial|magnitude. This is the reported number.
3. Metric concordance: ICC(1) of guide-level Sp (gene-clustered bootstrap) —
   NOT unordered pair-Spearman (ill-defined).
4. Document gene attrition (inventory multi-guide → scored) and coverage
   (~6% of screen has ≥2 guides passing cutoff).
5. If no guides: relabel split-half → "split-half directional agreement".

Papalexi GEO is a smaller, discordant secondary check of the same endpoint.

Usage:
  python split_half_guide_analysis.py
  python split_half_guide_analysis.py --h5ad /tmp/pertpy_data/replogle_2022_k562_essential.h5ad
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
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    _hvg_subsampled,
    _log1p_inplace,
    _looks_log_normalized,
    _normalize_total_numpy,
    _pca_truncated_svd,
    calculate_sp,
    ensure_in_memory,
    load_raw,
    setup_cache,
)
from revision_io import resolve_out_dir
from stats_utils import (
    bootstrap_partial_spearman_ci,
    bootstrap_spearman_ci,
    partial_spearman_rank,
)

try:
    from stats_utils import icc_gene_clustered_bootstrap, icc_oneway_unbalanced
except ImportError:  # pragma: no cover

    def icc_oneway_unbalanced(y, groups):
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups).astype(str)
        mask = np.isfinite(y)
        y, groups = y[mask], groups[mask]
        units, inv = np.unique(groups, return_inverse=True)
        k, N = len(units), len(y)
        if k < 2 or N <= k:
            return {"icc": np.nan, "n_groups": k, "n_obs": N, "n0": np.nan}
        n_i = np.bincount(inv)
        grand = float(y.mean())
        means = np.array([y[inv == i].mean() for i in range(k)])
        ssb = float(np.sum(n_i * (means - grand) ** 2))
        ssw = float(sum(np.sum((y[inv == i] - means[i]) ** 2) for i in range(k)))
        msb, msw = ssb / (k - 1), ssw / (N - k)
        n0 = (N - float(np.sum(n_i ** 2)) / N) / (k - 1)
        denom = msb + (n0 - 1) * msw
        return {
            "icc": float((msb - msw) / denom) if denom else np.nan,
            "msb": float(msb),
            "msw": float(msw),
            "n0": float(n0),
            "n_groups": int(k),
            "n_obs": int(N),
            "method": "icc1_oneway_unbalanced_local_fallback",
        }

    def icc_gene_clustered_bootstrap(
        y, groups, *, n_bootstrap=2000, ci_level=cfg.CI_LEVEL, seed=cfg.SEED
    ):
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups).astype(str)
        mask = np.isfinite(y)
        y, groups = y[mask], groups[mask]
        point = icc_oneway_unbalanced(y, groups)
        units = np.unique(groups)
        by_u = {u: y[groups == u] for u in units}
        rng = np.random.default_rng(seed)
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            drawn = rng.choice(units, size=len(units), replace=True)
            yb = np.concatenate([by_u[u] for u in drawn])
            gb = np.concatenate(
                [[f"{j}:{drawn[j]}"] * len(by_u[drawn[j]]) for j in range(len(drawn))]
            )
            boot[i] = icc_oneway_unbalanced(yb, gb)["icc"]
        valid = boot[np.isfinite(boot)]
        alpha = 1 - ci_level
        out = {
            "icc": point["icc"],
            "icc_details": point,
            "n_bootstrap": int(len(valid)),
            "n_groups": point.get("n_groups"),
            "n_obs": point.get("n_obs"),
        }
        if len(valid) >= 10:
            out["icc_ci_low"] = float(np.percentile(valid, 100 * alpha / 2))
            out["icc_ci_high"] = float(np.percentile(valid, 100 * (1 - alpha / 2)))
        else:
            out["icc_ci_low"] = out["icc_ci_high"] = np.nan
        print(
            "  NOTE: using local ICC fallback — re-upload stats_utils.py "
            "from the repo for the canonical helpers + numpy partial.",
            flush=True,
        )
        return out

GUIDE_COL_CANDIDATES = [
    "guide_id", "guide", "sgRNA", "sgrna", "gRNA", "grna",
    "barcode", "guide_identity", "feature_call", "gene_guide",
]


def find_guide_columns(obs: pd.DataFrame) -> list[str]:
    hits = []
    for c in obs.columns:
        cl = c.lower()
        if any(k.lower() in cl for k in GUIDE_COL_CANDIDATES):
            hits.append(c)
        elif "guide" in cl or "sgrna" in cl or "grna" in cl:
            hits.append(c)
    return hits


def _unit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def between_guide_shift_prediction(
    adata,
    gene_col,
    guide_col,
    ctrl_label,
    min_per_guide=30,
    n_bootstrap=2000,
    inventory_multi_genes: list[str] | None = None,
):
    """
    Item 16 predictive endpoint (Replogle flagship): gene Sp → median pairwise
    cosine of guide mean-shift vectors, plus rank-based partial|magnitude.
    """
    obs = adata.obs
    genes = obs[gene_col].astype(str).to_numpy()
    guides = obs[guide_col].astype(str).to_numpy()
    X = np.asarray(adata.obsm["X_pca"])
    ctrl_mask = genes == ctrl_label
    X_ctrl = X[ctrl_mask]
    if X_ctrl.shape[0] < 20:
        raise ValueError("Too few control cells")
    ctrl_centroid = X_ctrl.mean(axis=0)

    attrition: dict[str, list] = {
        "dropped_gene_too_few_cells": [],
        "dropped_gene_zero_magnitude": [],
        "dropped_fewer_than_2_guides_at_min_cells": [],
        "dropped_no_finite_cosine": [],
    }
    guide_sp_rows = []
    gene_rows = []
    pair_rows = []

    candidates = sorted(set(genes) - {ctrl_label})
    if inventory_multi_genes is not None:
        inv_set = set(inventory_multi_genes)
        # Genes in inventory multi-guide list but absent after materialize
        for g in sorted(inv_set - set(candidates)):
            attrition.setdefault("dropped_absent_after_materialize", []).append(g)

    for gene in candidates:
        gene_idx = np.flatnonzero(genes == gene)
        if len(gene_idx) < min_per_guide:
            attrition["dropped_gene_too_few_cells"].append(
                {"gene": gene, "n_cells": int(len(gene_idx))}
            )
            continue
        gene_metrics = calculate_sp(X_ctrl, X[gene_idx])
        if gene_metrics["magnitude"] <= 0:
            attrition["dropped_gene_zero_magnitude"].append(gene)
            continue

        guide_ids = sorted(set(guides[gene_idx]))
        guide_info = []
        for g in guide_ids:
            idx = np.flatnonzero((genes == gene) & (guides == g))
            n_g = int(len(idx))
            if n_g < min_per_guide:
                continue
            Xp = X[idx]
            m = calculate_sp(X_ctrl, Xp)
            if m["magnitude"] <= 0:
                continue
            guide_info.append(
                {
                    "guide": g,
                    "mean_shift": Xp.mean(axis=0) - ctrl_centroid,
                    "n": n_g,
                    "sp": float(m["stability"]),
                    "magnitude": float(m["magnitude"]),
                }
            )
            guide_sp_rows.append(
                {
                    "gene": gene,
                    "guide": g,
                    "stability": float(m["stability"]),
                    "magnitude": float(m["magnitude"]),
                    "n_cells": n_g,
                }
            )
        if len(guide_info) < 2:
            attrition["dropped_fewer_than_2_guides_at_min_cells"].append(
                {
                    "gene": gene,
                    "n_guides_passing_min_cells": int(len(guide_info)),
                    "n_guide_labels_present": int(len(guide_ids)),
                }
            )
            continue

        cosines = []
        for i in range(len(guide_info)):
            for j in range(i + 1, len(guide_info)):
                cos = _unit_cosine(
                    guide_info[i]["mean_shift"], guide_info[j]["mean_shift"]
                )
                if not np.isfinite(cos):
                    continue
                cosines.append(cos)
                pair_rows.append(
                    {
                        "gene": gene,
                        "guide_a": guide_info[i]["guide"],
                        "guide_b": guide_info[j]["guide"],
                        "cosine_mean_shift": cos,
                        "n_a": guide_info[i]["n"],
                        "n_b": guide_info[j]["n"],
                        "sp_a": guide_info[i]["sp"],
                        "sp_b": guide_info[j]["sp"],
                        "gene_sp": float(gene_metrics["stability"]),
                        "gene_magnitude": float(gene_metrics["magnitude"]),
                    }
                )
        if not cosines:
            attrition["dropped_no_finite_cosine"].append(gene)
            continue
        gene_rows.append(
            {
                "gene": gene,
                "n_guides": int(len(guide_info)),
                "n_pairs": int(len(cosines)),
                "median_cosine_mean_shift": float(np.median(cosines)),
                "gene_sp": float(gene_metrics["stability"]),
                "gene_magnitude": float(gene_metrics["magnitude"]),
            }
        )

    pairs = pd.DataFrame(pair_rows)
    gene_df = pd.DataFrame(gene_rows)
    guide_df = pd.DataFrame(guide_sp_rows)
    n_inv = len(inventory_multi_genes) if inventory_multi_genes is not None else None
    mean_pairs = float(len(pairs) / len(gene_df)) if len(gene_df) else np.nan

    dropped_counts = {k: len(v) for k, v in attrition.items()}
    summary: dict = {
        "endpoint": "gene_sp → median pairwise cosine(guide mean shifts)",
        "flagship": True,
        "n_genes": int(len(gene_df)),
        "n_pairs": int(len(pairs)),
        "mean_pairs_per_gene": mean_pairs,
        "n_genes_inventory_multi_guide": n_inv,
        "n_genes_scored": int(len(gene_df)),
        "attrition_counts": dropped_counts,
        "attrition_detail": {
            k: (v if len(v) <= 40 else v[:40] + [f"... +{len(v)-40} more"])
            for k, v in attrition.items()
            if v
        },
        "claim_evaluable": False,
        "caveat": (
            "Partly mechanical; independent reagents/cells — step down from "
            "split-half circularity, not a new experiment. Uncontrolled "
            "Spearman is not manuscript-ready until partial|magnitude is reported. "
            "Replogle is mostly ~1 pair/gene (noisy outcome)."
        ),
    }
    if n_inv is not None and n_inv != len(gene_df):
        note = (
            f"Inventory multi-guide genes={n_inv} → scored={len(gene_df)} "
            f"(dropped {n_inv - len(gene_df)}). Reasons: {dropped_counts}."
        )
        summary["attrition_note"] = note
        print(f"  ATTRITION: {note}", flush=True)
        for k, v in attrition.items():
            if not v:
                continue
            preview = v[:8]
            print(f"    {k} (n={len(v)}): {preview}", flush=True)

    if len(gene_df) >= 5:
        rho, p = spearmanr(
            gene_df["gene_sp"], gene_df["median_cosine_mean_shift"]
        )
        summary["spearman_gene_sp_vs_median_cosine"] = float(rho)
        summary["spearman_p"] = float(p)
        summary["spearman_boot"] = bootstrap_spearman_ci(
            gene_df["gene_sp"],
            gene_df["median_cosine_mean_shift"],
            n_bootstrap=n_bootstrap,
            seed=cfg.SEED,
        )
        part = partial_spearman_rank(
            gene_df["gene_sp"].to_numpy(),
            gene_df["median_cosine_mean_shift"].to_numpy(),
            gene_df["gene_magnitude"].to_numpy(),
        )
        summary["partial_spearman_given_magnitude"] = part
        summary["partial_boot"] = bootstrap_partial_spearman_ci(
            gene_df["gene_sp"].to_numpy(),
            gene_df["median_cosine_mean_shift"].to_numpy(),
            gene_df["gene_magnitude"].to_numpy(),
            n_bootstrap=min(n_bootstrap, 2000),
            seed=cfg.SEED,
        )
        summary["claim_evaluable"] = bool(
            np.isfinite(part.get("rho_partial", np.nan))
        )
        print(
            f"  Item 16 predictive (Replogle FLAGSHIP): n_genes={summary['n_genes']} "
            f"pairs={summary['n_pairs']} (mean pairs/gene={mean_pairs:.2f})  "
            f"Spearman={rho:.3f} "
            f"[{summary['spearman_boot'].get('ci_low')}, "
            f"{summary['spearman_boot'].get('ci_high')}]  "
            f"partial|mag={part.get('rho_partial')} "
            f"[{summary['partial_boot'].get('ci_low')}, "
            f"{summary['partial_boot'].get('ci_high')}] "
            f"({part.get('method')})",
            flush=True,
        )
        if not summary["claim_evaluable"]:
            print(
                "  *** partial|mag failed — predictive claim NOT evaluable ***",
                flush=True,
            )
    return pairs, gene_df, guide_df, summary


def between_guide_sp_icc(
    guide_df: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = cfg.SEED,
) -> tuple[pd.DataFrame, dict]:
    """
    Metric concordance: ICC(1) of guide Sp nested in genes.
    Replaces ill-defined Spearman on unordered (sp_a, sp_b) pairs.
    """
    sub = guide_df.copy()
    sizes = sub.groupby("gene").size()
    keep = sizes[sizes >= 2].index
    sub = sub[sub["gene"].isin(keep)]
    detail = sub.copy()
    summary: dict = {
        "analysis": "between_guide_sp_icc",
        "replication_label": "independent-reagent, shared-control replication",
        "n_genes_multi_guide": int(sub["gene"].nunique()),
        "n_guides": int(len(sub)),
        "note": (
            "ICC(1) with gene-clustered bootstrap. Unordered pair-Spearman is "
            "not used (arbitrary x/y; near coin-flip when mostly 1 pair/gene). "
            "ICC may be near zero or negative when within-gene variance exceeds "
            "between-gene variance."
        ),
    }
    if sub["gene"].nunique() < 3:
        summary["icc"] = np.nan
        return detail, summary

    sp_icc = icc_gene_clustered_bootstrap(
        sub["stability"].to_numpy(),
        sub["gene"].to_numpy(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary["icc"] = sp_icc["icc"]
    summary["icc_ci_low"] = sp_icc["icc_ci_low"]
    summary["icc_ci_high"] = sp_icc["icc_ci_high"]
    summary["icc_details"] = sp_icc["icc_details"]
    summary["n_bootstrap"] = sp_icc["n_bootstrap"]

    if "magnitude" in sub.columns:
        mag_icc = icc_gene_clustered_bootstrap(
            sub["magnitude"].to_numpy(),
            sub["gene"].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        summary["icc_magnitude"] = mag_icc["icc"]
        summary["icc_magnitude_ci_low"] = mag_icc["icc_ci_low"]
        summary["icc_magnitude_ci_high"] = mag_icc["icc_ci_high"]
        # Paired Δ
        genes_arr = sub["gene"].to_numpy()
        sp_vals = sub["stability"].to_numpy(dtype=float)
        mag_vals = sub["magnitude"].to_numpy(dtype=float)
        units = np.unique(genes_arr)
        by_sp = {u: sp_vals[genes_arr == u] for u in units}
        by_mag = {u: mag_vals[genes_arr == u] for u in units}
        rng = np.random.default_rng(seed)
        boot_d = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            drawn = rng.choice(units, size=len(units), replace=True)
            y_sp = np.concatenate([by_sp[u] for u in drawn])
            y_mag = np.concatenate([by_mag[u] for u in drawn])
            g = np.concatenate(
                [[f"{j}:{drawn[j]}"] * len(by_sp[drawn[j]]) for j in range(len(drawn))]
            )
            boot_d[i] = (
                icc_oneway_unbalanced(y_sp, g)["icc"]
                - icc_oneway_unbalanced(y_mag, g)["icc"]
            )
        valid = boot_d[np.isfinite(boot_d)]
        summary["icc_sp_minus_icc_magnitude"] = float(
            sp_icc["icc"] - mag_icc["icc"]
        )
        if len(valid) >= 10:
            alpha = 1 - cfg.CI_LEVEL
            summary["icc_sp_minus_icc_magnitude_ci_low"] = float(
                np.percentile(valid, 100 * alpha / 2)
            )
            summary["icc_sp_minus_icc_magnitude_ci_high"] = float(
                np.percentile(valid, 100 * (1 - alpha / 2))
            )

    print(
        f"  Between-guide Sp ICC (metric concordance): "
        f"n_genes={summary['n_genes_multi_guide']} n_guides={summary['n_guides']}  "
        f"ICC(1)={summary.get('icc')} "
        f"[{summary.get('icc_ci_low')}, {summary.get('icc_ci_high')}]  "
        f"ICC(mag)={summary.get('icc_magnitude')}  "
        f"Δ={summary.get('icc_sp_minus_icc_magnitude')} "
        f"[{summary.get('icc_sp_minus_icc_magnitude_ci_low')}, "
        f"{summary.get('icc_sp_minus_icc_magnitude_ci_high')}]",
        flush=True,
    )
    return detail, summary


METHODS_BLURB_NO_GUIDE = """
Split-half analysis limitation (revision):
The split-half procedure partitions cells from the same perturbation and the same
control reference. A perturbation with tightly concentrated shift directions will,
by construction, yield more similar half-sample means. We therefore refer to this
endpoint as "split-half directional agreement" (sampling precision), not as
independent experimental reproducibility. Between-guide or between-replicate
analyses were not available for Replogle in the distributed metadata
(guide-level columns: {guide_cols_found}).
""".strip()

METHODS_BLURB_WITH_GUIDE = """
Guide-level reproducibility (revision; Replogle FLAGSHIP for item 16):
Only {n_multi}/{n_screen} perturbations ({pct_multi:.1f}%) carry ≥2 guides
passing min_cells={min_per_guide}, so reagent-level analysis is not representative
of the full screen. Inventory multi-guide n={n_inv} → scored n={n_genes}
({attrition}). Predictive endpoint: Spearman(gene Sp, median pairwise cosine of
guide mean shifts)={spearman_pred:.3f} [{spearman_lo:.3f}, {spearman_hi:.3f}]
(n_genes={n_genes}, n_pairs={n_pairs}, mean pairs/gene={mean_pairs:.2f});
rank-based partial|magnitude={partial:.3f} [{partial_lo:.3f}, {partial_hi:.3f}]
({partial_method}). This uncontrolled association is modest; the partial is
required before any predictive claim. Metric concordance: ICC(1) of guide Sp=
{icc:.3f} [{icc_lo:.3f}, {icc_hi:.3f}] (not unordered pair-Spearman). Papalexi
GEO is a smaller, discordant secondary check — do not lead with Papalexi rho.
Split-half remains within-sample directional agreement.
""".strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Replogle 2022 (CRISPRi)")
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-per-guide", type=int, default=30)
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Optional cap on multi-guide genes (default: no cap)",
    )
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    name = cfg.resolve_dataset_name(args.dataset)

    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = cfg.CACHE_DIR
    h5ad = args.h5ad
    if h5ad is None:
        cand = cfg.CACHE_DIR / "replogle_2022_k562_essential.h5ad"
        if cand.exists():
            h5ad = cand

    print(f"Opening {name} …", flush=True)
    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad)
    if type(raw).__name__ == "MuData":
        obs = raw.mod["rna"].obs if "rna" in raw.mod else list(raw.mod.values())[0].obs
    else:
        obs = raw.obs

    guide_cols = find_guide_columns(obs)
    print(f"  obs columns ({len(obs.columns)}): {list(obs.columns)[:40]}…")
    print(f"  guide-like columns: {guide_cols}")

    report = {
        "dataset": name,
        "n_obs": int(len(obs)),
        "obs_columns": list(map(str, obs.columns)),
        "guide_columns_found": guide_cols,
        "config_version": cfg.CONFIG_VERSION,
    }

    if not guide_cols:
        report["recommendation"] = "relabel_split_half_directional_agreement"
        report["methods_blurb"] = METHODS_BLURB_NO_GUIDE.format(
            guide_cols_found="none"
        )
        print("\n*** No guide-level columns found.")
        print("*** Relabel split-half as 'split-half directional agreement' in the manuscript.")
        with open(out_dir / "split_half_guide_report.json", "w") as f:
            json.dump(report, f, indent=2)
        with open(out_dir / "split_half_methods_blurb.txt", "w") as f:
            f.write(report["methods_blurb"] + "\n")
        print(f"Wrote {out_dir}/split_half_guide_report.json")
        return

    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    guide_col = guide_cols[0]
    if guide_col not in adata.obs.columns and guide_col in obs.columns:
        adata.obs[guide_col] = obs.loc[adata.obs_names, guide_col].astype(str).values

    meta = adata.obs[[pert_col, guide_col]].copy()
    meta[pert_col] = meta[pert_col].astype(str)
    meta[guide_col] = meta[guide_col].astype(str)

    # Full-screen inventory (before any gene cap)
    all_genes = [
        str(g)
        for g, sub in meta.groupby(pert_col, observed=True)
        if g != ctrl_label
    ]
    multi_all = []
    for gene, sub in meta.groupby(pert_col, observed=True):
        if gene == ctrl_label:
            continue
        # guides with ≥min_per_guide cells
        vc = sub[guide_col].value_counts()
        if int((vc >= args.min_per_guide).sum()) >= 2:
            multi_all.append(str(gene))
    n_screen = len(all_genes)
    n_multi = len(multi_all)
    pct_multi = 100.0 * n_multi / n_screen if n_screen else 0.0
    report["coverage"] = {
        "n_perturbations_screen": n_screen,
        "n_genes_multi_guide_inventory": n_multi,
        "fraction_multi_guide": float(n_multi / n_screen) if n_screen else np.nan,
        "min_per_guide": args.min_per_guide,
        "note": (
            f"Only {n_multi}/{n_screen} ({pct_multi:.1f}%) perturbations carry "
            f"≥2 guides at min_cells={args.min_per_guide}; reagent-level "
            f"analysis is not representative of the full screen."
        ),
    }
    print(
        f"  Coverage: {n_multi}/{n_screen} perturbations "
        f"({pct_multi:.1f}%) have ≥2 guides at min_cells={args.min_per_guide}",
        flush=True,
    )

    multi = list(multi_all)
    rng = np.random.default_rng(cfg.SEED)
    if args.max_genes is not None and len(multi) > args.max_genes:
        multi = list(rng.choice(multi, size=args.max_genes, replace=False))
        print(
            f"  Capped multi-guide genes to {len(multi)} (--max-genes)",
            flush=True,
        )
    keep = set(multi) | {ctrl_label}
    print(f"  Genes with ≥2 guides (materialized): {len(multi)}", flush=True)

    label_arr = meta[pert_col].to_numpy()
    keep_idx = np.flatnonzero(np.isin(label_arr, list(keep)))
    max_per_gene = max(cfg.MAX_CELLS_PER_PERT * 3, 300)
    max_ctrl = cfg.MAX_CONTROL_CELLS
    sampled = []
    for lab in list(keep):
        idx = keep_idx[label_arr[keep_idx] == lab]
        cap = max_ctrl if lab == ctrl_label else max_per_gene
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        sampled.append(idx)
    keep_idx = np.sort(np.concatenate(sampled)) if sampled else np.array([], dtype=int)
    print(
        f"  Materializing {len(keep_idx)} cells "
        f"(backed={getattr(adata, 'isbacked', False)})…",
        flush=True,
    )
    adata = ensure_in_memory(adata[keep_idx])

    if not _looks_log_normalized(adata):
        _normalize_total_numpy(adata, 1e4)
        _log1p_inplace(adata)
    if adata.n_vars > cfg.N_HVG:
        adata = _hvg_subsampled(adata, cfg.N_HVG, cfg.SEED)
        adata = ensure_in_memory(adata) if getattr(adata, "isbacked", False) else adata
    adata = _pca_truncated_svd(adata, cfg.N_PCS, cfg.SEED)

    n_boot = min(2000, cfg.N_BOOTSTRAP)
    pred_pairs, pred_genes, guide_df, pred_sum = between_guide_shift_prediction(
        adata,
        pert_col,
        guide_col,
        ctrl_label,
        min_per_guide=args.min_per_guide,
        n_bootstrap=n_boot,
        inventory_multi_genes=multi,
    )
    report["between_guide_shift_cosine"] = pred_sum
    if len(pred_pairs):
        pred_pairs.to_csv(
            out_dir / "between_guide_shift_cosine_pairs.csv", index=False
        )
    if len(pred_genes):
        pred_genes.to_csv(
            out_dir / "between_guide_shift_cosine_genes.csv", index=False
        )

    detail, icc_sum = between_guide_sp_icc(
        guide_df, n_bootstrap=n_boot, seed=cfg.SEED
    )
    report["between_guide_sp_icc"] = icc_sum
    # Keep old key but mark deprecated
    report["between_guide_sp_concordance"] = {
        "deprecated": True,
        "note": (
            "Unordered pair-Spearman is ill-defined; use between_guide_sp_icc."
        ),
        "replaced_by": "between_guide_sp_icc",
    }
    if len(detail):
        detail.to_csv(out_dir / "between_guide_sp_guides.csv", index=False)

    report["recommendation"] = (
        "report_replogle_flagship_partial_and_relabel_split_half; "
        "papalexi_secondary_only"
    )
    part = pred_sum.get("partial_spearman_given_magnitude") or {}
    part_ci = pred_sum.get("partial_boot") or {}
    sp_boot = pred_sum.get("spearman_boot") or {}
    attrition_note = pred_sum.get("attrition_note") or "none"
    report["methods_blurb"] = METHODS_BLURB_WITH_GUIDE.format(
        n_multi=n_multi,
        n_screen=n_screen,
        pct_multi=pct_multi,
        min_per_guide=args.min_per_guide,
        n_inv=pred_sum.get("n_genes_inventory_multi_guide", n_multi),
        n_genes=pred_sum.get("n_genes", 0),
        attrition=attrition_note,
        spearman_pred=pred_sum.get(
            "spearman_gene_sp_vs_median_cosine", float("nan")
        ),
        spearman_lo=sp_boot.get("ci_low", float("nan")),
        spearman_hi=sp_boot.get("ci_high", float("nan")),
        n_pairs=pred_sum.get("n_pairs", 0),
        mean_pairs=pred_sum.get("mean_pairs_per_gene", float("nan")),
        partial=part.get("rho_partial", float("nan")),
        partial_lo=part_ci.get("ci_low", float("nan")),
        partial_hi=part_ci.get("ci_high", float("nan")),
        partial_method=part.get("method", "NA"),
        icc=icc_sum.get("icc", float("nan")),
        icc_lo=icc_sum.get("icc_ci_low", float("nan")),
        icc_hi=icc_sum.get("icc_ci_high", float("nan")),
    )
    with open(out_dir / "split_half_guide_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out_dir / "split_half_methods_blurb.txt", "w") as f:
        f.write(report["methods_blurb"] + "\n")
    print(report["methods_blurb"])
    print(f"Wrote report under {out_dir}")


if __name__ == "__main__":
    main()
