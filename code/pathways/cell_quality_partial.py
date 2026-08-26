#!/usr/bin/env python3
"""
Cell-quality confound gate for magnitude-conditioned pathway / stress associations.

Apoptosis and p53 partials are sign-consistent across five scoreable datasets.
Dying / low-quality cells also lower directional coherence (mito↑, ngenes↓,
technical noise↑). Replogle Q4 DE already shows MT-CO1/2 among top hits.

This script asks the decisive question before any manuscript text:

  Does Sp ~ pathway (or stress marker) survive after conditioning on
  magnitude AND per-perturbation mean(percent_mito, n_genes, n_counts)?

If yes  → magnitude-independent biology survives the quality confound.
If no   → residual incoherence tracks cell quality; do not lead with p53/apoptosis.

Usage:
  python cell_quality_partial.py --verify-bootstrap
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

import ast

import pipeline_config as cfg
from pipeline_core import (
    _extract_adata,
    assert_frozen_sp_compatible,
    ensure_in_memory,
    load_raw,
    materialize_min_cells,
    resolve_matrix_is_log,
    setup_cache,
    _log1p_inplace,
    _normalize_total_numpy,
)
from scipy.stats import spearmanr

from stats_utils import (
    bootstrap_partial_spearman_ci,
    bootstrap_spearman_ci,
    partial_spearman_rank,
    pathway_bootstrap_seed,
    survival_status,
)

# Decisive pathways; gene sets loaded from pathway_analysis.py without importing
# it (that module has heavy Colab side effects at import time).
FOCUS_PATHWAYS = ["Apoptosis", "p53", "UPR", "ROS", "mTORC1"]
STRESS_MARKERS = ["DDIT3", "ATF4", "XBP1", "HSPA5"]
QC_COLS = ["qc_percent_mito", "qc_n_genes", "qc_n_counts"]
MIN_N = 15
MIN_GENE_OVERLAP = 5


def _load_hallmark_gene_sets() -> dict:
    path = Path(__file__).resolve().parent / "pathway_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if getattr(t, "id", None) == "HALLMARK_GENE_SETS":
                return ast.literal_eval(node.value)
    raise RuntimeError("HALLMARK_GENE_SETS not found in pathway_analysis.py")


HALLMARK_GENE_SETS = _load_hallmark_gene_sets()


def apoptosis_p53_redundancy(
    big: pd.DataFrame,
    out_dir: Path,
    gate_res: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Independence check for the dual-pathway claim.

    Verdict uses only gate-relevant datasets (exclude descriptive_small_n).
    High score ρ on Papalexi alone must not reframe the claim — it is outside
    the gate. Moderate ρ with divergent partials (e.g. Dixit) supports independence.
    """
    apo_genes = set(HALLMARK_GENE_SETS.get("Apoptosis", []))
    p53_genes = set(HALLMARK_GENE_SETS.get("p53", []))
    inter = apo_genes & p53_genes
    union = apo_genes | p53_genes
    jacc_full = len(inter) / len(union) if union else np.nan

    # Dataset → gate labels for Apoptosis / p53
    gate_apo, gate_p53 = {}, {}
    if gate_res is not None and not gate_res.empty:
        for _, r in gate_res.iterrows():
            if r["feature"] == "pw_Apoptosis":
                gate_apo[r["dataset"]] = r["gate"]
            elif r["feature"] == "pw_p53":
                gate_p53[r["dataset"]] = r["gate"]

    rows = []
    cov_path = out_dir / "pathway_gene_coverage.csv"
    cov = pd.read_csv(cov_path) if cov_path.exists() else None
    for ds, sub in big.groupby("dataset"):
        if "pw_Apoptosis" not in sub.columns or "pw_p53" not in sub.columns:
            continue
        a = sub["pw_Apoptosis"].astype(float)
        b = sub["pw_p53"].astype(float)
        m = a.notna() & b.notna()
        if m.sum() < 5:
            continue
        rho, p = spearmanr(a[m], b[m])
        jacc_ds = np.nan
        n_inter_ds = n_union_ds = np.nan
        if cov is not None:
            hit = cov[
                (cov["dataset"] == ds) & (cov["pathway"] == "Apoptosis∩p53")
            ]
            if len(hit):
                jacc_ds = float(hit.iloc[0].get("jaccard", np.nan))
                n_inter_ds = hit.iloc[0].get("n_intersection", np.nan)
                n_union_ds = hit.iloc[0].get("n_union", np.nan)
        g_a = gate_apo.get(ds, "")
        g_p = gate_p53.get(ds, "")
        # Gate-relevant = at least one pathway has an estimable QC status
        descriptive = (
            g_a == "descriptive_small_n" and g_p == "descriptive_small_n"
        ) if (g_a or g_p) else (
            int(m.sum()) < int(getattr(cfg, "SURVIVAL_QC_DESCRIPTIVE_MAX_N", 30))
        )
        rows.append({
            "dataset": ds,
            "n": int(m.sum()),
            "spearman_apoptosis_p53": float(rho),
            "p_spearman": float(p),
            "jaccard_hallmark_full": float(jacc_full),
            "n_shared_hallmark": len(inter),
            "n_union_hallmark": len(union),
            "jaccard_genes_in_dataset": jacc_ds,
            "n_shared_in_dataset": n_inter_ds,
            "n_union_in_dataset": n_union_ds,
            "gate_apoptosis": g_a or "n/a",
            "gate_p53": g_p or "n/a",
            "gate_relevant": (not descriptive),
            "survives_qc_apoptosis": g_a == "survives_qc",
            "survives_qc_p53": g_p == "survives_qc",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    path = out_dir / "apoptosis_p53_redundancy.csv"
    out.to_csv(path, index=False)

    print("\n" + "=" * 72)
    print("APOPTOSIS ↔ p53 INDEPENDENCE (verdict = gate-relevant datasets only)")
    print("=" * 72)
    print(
        f"Hallmark gene-set Jaccard (full lists): {jacc_full:.3f} "
        f"(|∩|={len(inter)} of |∪|={len(union)}; mostly disjoint)"
    )
    for _, r in out.iterrows():
        j = r["jaccard_genes_in_dataset"]
        j_s = f"{j:.3f}" if np.isfinite(j) else "n/a"
        scope = "gate" if r["gate_relevant"] else "excluded (descriptive)"
        surv = []
        if r["survives_qc_apoptosis"]:
            surv.append("apo✓")
        if r["survives_qc_p53"]:
            surv.append("p53✓")
        surv_s = ",".join(surv) if surv else "—"
        note = ""
        if (not r["gate_relevant"]) and abs(r["spearman_apoptosis_p53"]) >= 0.7:
            note = "  ← high ρ but outside gate; ignored for verdict"
        print(
            f"  {r['dataset']} [{scope}]: score ρ={r['spearman_apoptosis_p53']:+.3f} "
            f"(n={int(r['n'])}); gene Jaccard={j_s}; "
            f"QC survivors: {surv_s}{note}"
        )

    relevant = out[out["gate_relevant"]]
    if relevant.empty:
        print("  → No gate-relevant datasets; independence claim N/A.")
    else:
        max_abs = float(relevant["spearman_apoptosis_p53"].abs().max())
        dual = relevant[
            relevant["survives_qc_apoptosis"] & relevant["survives_qc_p53"]
        ]
        apo_surv = relevant[relevant["survives_qc_apoptosis"]]
        p53_surv = relevant[relevant["survives_qc_p53"]]
        if max_abs >= 0.7:
            print(
                "  → High score correlation among gate-relevant datasets: "
                "report as one stress-response axis; do not count dual-pathway "
                "survival as independent confirmation."
            )
        else:
            print(
                f"  → INDEPENDENT ENOUGH FOR THE CLAIM: gate-relevant |score ρ| "
                f"max={max_abs:.3f} (<0.7); gene-set Jaccard={jacc_full:.3f}. "
                "Partially different survivor sets strengthen two-pathway wording."
            )
            if len(apo_surv):
                print(
                    "    Apoptosis QC survivors ρ: "
                    + ", ".join(
                        f"{r['dataset']}={r['spearman_apoptosis_p53']:+.3f}"
                        for _, r in apo_surv.iterrows()
                    )
                )
            if len(p53_surv):
                print(
                    "    p53 QC survivors ρ: "
                    + ", ".join(
                        f"{r['dataset']}={r['spearman_apoptosis_p53']:+.3f}"
                        for _, r in p53_surv.iterrows()
                    )
                )
            if len(dual):
                print(
                    "    Both pathways survive only in: "
                    + ", ".join(
                        f"{r['dataset']} (ρ={r['spearman_apoptosis_p53']:+.3f})"
                        for _, r in dual.iterrows()
                    )
                )
            else:
                print("    No gate-relevant dataset survives QC for both pathways.")
            print(
                "    State explicitly in manuscript; put per-dataset ρ in SI "
                "(state overlap vs Hallmark gene-set overlap)."
            )
    print(f"Wrote {path}")
    return out


def _mito_mask(var_names) -> np.ndarray:
    names = pd.Index(var_names).astype(str)
    return names.str.upper().str.startswith("MT-").to_numpy()


def ensure_cell_qc(adata) -> None:
    """Write percent_mito / n_genes / n_counts on obs if missing (from .X)."""
    X = adata.X
    if hasattr(X, "getnnz"):
        n_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(float)
    else:
        Xa = np.asarray(X)
        n_counts = Xa.sum(axis=1).astype(float)
        n_genes = (Xa > 0).sum(axis=1).astype(float)

    # prefer existing columns when present and finite
    for cand in ("n_counts", "ncounts", "total_counts"):
        if cand in adata.obs.columns and adata.obs[cand].notna().any():
            n_counts = pd.to_numeric(adata.obs[cand], errors="coerce").to_numpy()
            break
    for cand in ("n_genes", "ngenes", "n_genes_by_counts"):
        if cand in adata.obs.columns and adata.obs[cand].notna().any():
            n_genes = pd.to_numeric(adata.obs[cand], errors="coerce").to_numpy()
            break

    mito = None
    for cand in ("percent_mito", "pct_counts_mt", "percent.mito", "mito_frac"):
        if cand in adata.obs.columns and adata.obs[cand].notna().any():
            mito = pd.to_numeric(adata.obs[cand], errors="coerce").to_numpy()
            # convert fraction → percent if needed
            if np.nanmax(mito) <= 1.5:
                mito = mito * 100.0
            break
    if mito is None:
        mt = _mito_mask(adata.var_names)
        if mt.any():
            if hasattr(X, "getnnz"):
                mt_counts = np.asarray(X[:, mt].sum(axis=1)).ravel().astype(float)
            else:
                mt_counts = np.asarray(X)[:, mt].sum(axis=1).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mito = np.where(n_counts > 0, 100.0 * mt_counts / n_counts, np.nan)
        else:
            mito = np.full(adata.n_obs, np.nan)
            print("    WARNING: no MT- genes and no percent_mito in obs")

    adata.obs["qc_percent_mito"] = mito
    adata.obs["qc_n_genes"] = n_genes
    adata.obs["qc_n_counts"] = n_counts


def per_pert_means(adata, pert_col: str, ctrl_label: str, cols: list[str]) -> pd.DataFrame:
    labels = adata.obs[pert_col].astype(str)
    rows = []
    for pert, idx in labels.groupby(labels).groups.items():
        if pert == ctrl_label:
            continue
        sub = adata.obs.loc[idx, cols]
        row = {"perturbation": str(pert), "n_cells_qc": int(len(idx))}
        for c in cols:
            row[c] = float(pd.to_numeric(sub[c], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("perturbation")


def score_pathways(adata_norm, pert_col: str, ctrl_label: str, min_cells: int) -> pd.DataFrame:
    import scanpy as sc

    counts = adata_norm.obs[pert_col].astype(str).value_counts()
    valid = [v for v in counts[counts >= min_cells].index if v != ctrl_label]
    out = {}
    for pw in FOCUS_PATHWAYS:
        genes = HALLMARK_GENE_SETS.get(pw, [])
        if not genes:
            continue
        overlap = sorted(g for g in genes if g in adata_norm.var_names)
        if len(overlap) < MIN_GENE_OVERLAP:
            print(f"    {pw}: skip (overlap={len(overlap)})")
            continue
        col = f"score_{pw}"
        np.random.seed(cfg.SEED)
        sc.tl.score_genes(
            adata_norm, gene_list=overlap, score_name=col,
            ctrl_size=50, random_state=cfg.SEED,
        )
        means = {}
        for pert in valid:
            mask = adata_norm.obs[pert_col].astype(str) == pert
            means[pert] = float(adata_norm.obs.loc[mask, col].mean())
        out[f"pw_{pw}"] = pd.Series(means)
        print(f"    {pw}: {len(overlap)}/{len(genes)} genes, {len(means)} perts")
    return pd.DataFrame(out)


def residual_diag(sp, mag, y):
    sp = np.asarray(sp, float)
    mag = np.asarray(mag, float)
    y = np.asarray(y, float)
    mask = np.isfinite(sp) & np.isfinite(mag) & np.isfinite(y)
    sp, mag, y = sp[mask], mag[mask], y[mask]
    n = len(sp)
    if n < 5:
        return {"r2_sp_on_magnitude": np.nan, "frac_sp_variance_remaining": np.nan,
                "partial_r2": np.nan}
    rsp, rmag, ry = rankdata(sp), rankdata(mag), rankdata(y)
    Z = np.column_stack([np.ones(n), rmag])
    e_sp = rsp - Z @ np.linalg.lstsq(Z, rsp, rcond=None)[0]
    e_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    ss_tot = float(np.sum((rsp - rsp.mean()) ** 2))
    ss_res = float(np.sum(e_sp ** 2))
    frac = ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_mag = 1.0 - frac if np.isfinite(frac) else np.nan
    if np.std(e_sp) < 1e-15 or np.std(e_y) < 1e-15:
        pr2 = np.nan
    else:
        pr2 = float(np.corrcoef(e_sp, e_y)[0, 1] ** 2)
    return {
        "r2_sp_on_magnitude": float(r2_mag),
        "frac_sp_variance_remaining": float(frac),
        "partial_r2": pr2,
    }


def analyze_table(df: pd.DataFrame, feature_cols: list[str], n_bootstrap: int) -> pd.DataFrame:
    rows = []
    # Stable dataset order so logging is readable; seeds are content-addressed
    for ds in sorted(df["dataset"].dropna().unique()):
        sub0 = df[df["dataset"] == ds]
        for feat in sorted(feature_cols):
            if feat not in sub0.columns:
                continue
            need = ["stability", "magnitude", feat] + QC_COLS
            sub = sub0.dropna(subset=need).copy()
            if len(sub) < MIN_N:
                print(f"  {ds} | {feat}: skip n={len(sub)} < {MIN_N}")
                continue

            # Strip pw_ for seed parity with pathway_analysis feature names
            feat_key = feat[3:] if feat.startswith("pw_") else feat
            raw_seed = pathway_bootstrap_seed(
                ds, feat_key, "raw", n_bootstrap=n_bootstrap
            )
            mag_seed = pathway_bootstrap_seed(
                ds, feat_key, "partial_mag", n_bootstrap=n_bootstrap
            )
            qc_seed = pathway_bootstrap_seed(
                ds, feat_key, "partial_mag_qc", n_bootstrap=n_bootstrap
            )

            sp = sub["stability"].values
            mag = sub["magnitude"].values
            y = sub[feat].values
            Z_mag = mag.reshape(-1, 1)
            # Small-n / near-collinear QC: mito only (Papalexi n=24 rank-deficient
            # under mag + mito + ngenes + ncounts).
            mito_only_n = int(getattr(cfg, "SURVIVAL_QC_MITO_ONLY_MAX_N", 40))
            if len(sub) < mito_only_n:
                Z_full = np.column_stack([mag, sub["qc_percent_mito"].to_numpy(dtype=float)])
                covar_label = "magnitude,qc_percent_mito"
                print(
                    f"  {ds} | {feat}: QC mito-only (n={len(sub)} < {mito_only_n}; "
                    "drop n_genes/n_counts — rank-deficient otherwise)"
                )
            else:
                Z_qc = sub[QC_COLS].to_numpy(dtype=float)
                Z_full = np.column_stack([mag, Z_qc])
                covar_label = "magnitude,qc_percent_mito,qc_n_genes,qc_n_counts"

            descriptive_n = int(getattr(cfg, "SURVIVAL_QC_DESCRIPTIVE_MAX_N", 30))
            qc_descriptive = len(sub) < descriptive_n

            raw = bootstrap_spearman_ci(sp, y, n_bootstrap=n_bootstrap, seed=raw_seed)
            p_mag = bootstrap_partial_spearman_ci(
                sp, y, Z_mag, n_bootstrap=n_bootstrap, seed=mag_seed, method="rank"
            )
            if qc_descriptive:
                # Point estimate only — bootstrap CIs degenerate at n=24 (Papalexi)
                pt = partial_spearman_rank(sp, y, Z_full)
                p_qc = {
                    "rho_partial": pt["rho_partial"],
                    "p": pt["p"],
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": pt["n"],
                    "method": pt.get("method", "partial_spearman_rank"),
                    "n_bootstrap": 0,
                    "bootstrap_frac_valid": np.nan,
                }
                print(
                    f"  {ds} | {feat}: QC descriptive only "
                    f"(n={len(sub)} < {descriptive_n}; no bootstrap CI)"
                )
            else:
                p_qc = bootstrap_partial_spearman_ci(
                    sp, y, Z_full, n_bootstrap=n_bootstrap, seed=qc_seed, method="rank"
                )
            resid = residual_diag(sp, mag, y)

            # FDR filled after the loop; provisional CI-only status here
            st_mag = survival_status(
                p_mag["rho_partial"], p_mag["ci_low"], p_mag["ci_high"], fdr=None
            )
            if qc_descriptive:
                st_qc = {
                    "status": "descriptive_small_n",
                    "survives": False,
                    "knife_edge": False,
                    "ci_margin": np.nan,
                }
            else:
                st_qc = survival_status(
                    p_qc["rho_partial"], p_qc["ci_low"], p_qc["ci_high"], fdr=None
                )
            # Suppression: appears only after QC conditioning
            suppression = bool(
                st_mag["status"] != "survives"
                and st_qc["status"] == "survives"
            )
            # Collapse = |mag survives AND |QC fails (not indeterminate/descriptive)
            collapses = bool(
                st_mag["status"] == "survives"
                and st_qc["status"] == "does_not_survive"
            )
            attenuated = bool(
                st_mag["status"] == "survives"
                and st_qc["status"] == "survives"
                and np.isfinite(p_mag["rho_partial"])
                and np.isfinite(p_qc["rho_partial"])
                and abs(p_qc["rho_partial"]) < 0.5 * abs(p_mag["rho_partial"])
            )
            if qc_descriptive:
                gate = "descriptive_small_n"
            elif st_qc["status"] == "indeterminate" or st_mag["status"] == "indeterminate":
                gate = "indeterminate"
            elif st_qc["status"] == "survives":
                gate = "qc_conditional" if suppression else "survives_qc"
            elif st_mag["status"] == "survives" and st_qc["status"] == "does_not_survive":
                gate = "collapses_under_qc"
            elif st_mag["status"] == "survives":
                gate = "indeterminate"  # |mag ok but |QC not a clean fail
            else:
                gate = "no_mag_partial"

            def _ci(lo, hi):
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    return "[n/a]"
                return f"[{lo:.3f},{hi:.3f}]"

            print(
                f"  {ds} | {feat}: raw={raw['rho']:+.3f}  "
                f"|mag={p_mag['rho_partial']:+.3f} "
                f"{_ci(p_mag['ci_low'], p_mag['ci_high'])}  "
                f"|mag+QC={p_qc['rho_partial']:+.3f} "
                f"{_ci(p_qc['ci_low'], p_qc['ci_high'])}  "
                f"frac_Sp_left={resid['frac_sp_variance_remaining']:.3f}  "
                f"partial_R²={resid['partial_r2']:.3f}  "
                f"→ {gate}"
                + (" (attenuated ≥50%)" if attenuated else "")
            )

            rows.append({
                "dataset": ds,
                "feature": feat,
                "feature_type": "pathway" if feat.startswith("pw_") else "stress",
                "n": len(sub),
                "rho_raw": raw["rho"],
                "p_raw": raw["p"],
                "rho_partial_mag": p_mag["rho_partial"],
                "rho_partial_mag_ci_low": p_mag["ci_low"],
                "rho_partial_mag_ci_high": p_mag["ci_high"],
                "p_partial_mag": p_mag["p"],
                "survival_status_mag_ci": st_mag["status"],
                "rho_partial_mag_qc": p_qc["rho_partial"],
                "rho_partial_mag_qc_ci_low": p_qc["ci_low"],
                "rho_partial_mag_qc_ci_high": p_qc["ci_high"],
                "p_partial_mag_qc": p_qc["p"],
                "survival_status_qc_ci": st_qc["status"],
                "gate": gate,
                "collapses_under_qc": collapses,
                "attenuated_under_qc": attenuated,
                "appears_only_after_qc": suppression,
                "qc_descriptive_only": qc_descriptive,
                "r2_sp_on_magnitude": resid["r2_sp_on_magnitude"],
                "frac_sp_variance_remaining": resid["frac_sp_variance_remaining"],
                "partial_r2_mag": resid["partial_r2"],
                "mean_qc_percent_mito": float(sub["qc_percent_mito"].mean()),
                "mean_qc_n_genes": float(sub["qc_n_genes"].mean()),
                "mean_qc_n_counts": float(sub["qc_n_counts"].mean()),
                "config_version": cfg.CONFIG_VERSION,
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_seed_mag": int(mag_seed),
                "bootstrap_seed_qc": int(qc_seed),
                "knife_edge_abs": float(getattr(cfg, "SURVIVAL_KNIFE_EDGE_ABS", 0.02)),
                "covariates_mag_qc": covar_label,
                "qc_note": (
                    "QC covariates partly downstream of apoptotic biology; "
                    "|mag+QC is a lower bound on the biological association"
                ),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH within dataset across features, then finalize survive flags
    for pcol, fdr_col in (
        ("p_partial_mag", "p_partial_mag_fdr_bh"),
        ("p_partial_mag_qc", "p_partial_mag_qc_fdr_bh"),
    ):
        out[fdr_col] = np.nan
        for ds, idx in out.groupby("dataset").groups.items():
            pvals = out.loc[idx, pcol].astype(float).values
            p_fill = np.where(np.isfinite(pvals), pvals, 1.0)
            out.loc[idx, fdr_col] = multipletests(p_fill, method="fdr_bh")[1]

    surv_mag, surv_qc, st_mag_f, st_qc_f = [], [], [], []
    for _, r in out.iterrows():
        sm = survival_status(
            r["rho_partial_mag"], r["rho_partial_mag_ci_low"],
            r["rho_partial_mag_ci_high"], fdr=r["p_partial_mag_fdr_bh"],
        )
        if bool(r.get("qc_descriptive_only", False)):
            sq = {
                "survives": False,
                "status": "descriptive_small_n",
                "knife_edge": False,
                "ci_margin": np.nan,
            }
        else:
            sq = survival_status(
                r["rho_partial_mag_qc"], r["rho_partial_mag_qc_ci_low"],
                r["rho_partial_mag_qc_ci_high"], fdr=r["p_partial_mag_qc_fdr_bh"],
            )
        surv_mag.append(sm["survives"])
        surv_qc.append(sq["survives"])
        st_mag_f.append(sm["status"])
        st_qc_f.append(sq["status"])
    out["survives_magnitude"] = surv_mag
    out["survives_mag_qc"] = surv_qc
    out["survival_status_mag"] = st_mag_f
    out["survival_status_qc"] = st_qc_f
    # Recompute suppression / collapse / gate with FDR-aware status.
    # Collapse only when |QC cleanly fails — not when indeterminate/descriptive.
    supp, coll, gates = [], [], []
    for _, r in out.iterrows():
        if bool(r.get("qc_descriptive_only", False)):
            gates.append("descriptive_small_n")
            supp.append(False)
            coll.append(False)
            continue
        is_supp = (not r["survives_magnitude"]) and bool(r["survives_mag_qc"])
        is_coll = (
            bool(r["survives_magnitude"])
            and r["survival_status_qc"] == "does_not_survive"
        )
        supp.append(is_supp)
        coll.append(is_coll)
        if r["survival_status_qc"] == "indeterminate" or r["survival_status_mag"] == "indeterminate":
            gates.append("indeterminate")
        elif r["survives_mag_qc"]:
            # Null under |mag, significant only after QC — not a primary survivor
            gates.append("qc_conditional" if is_supp else "survives_qc")
        elif is_coll:
            gates.append("collapses_under_qc")
        else:
            gates.append("no_mag_partial")
    out["appears_only_after_qc"] = supp
    out["collapses_under_qc"] = coll
    out["gate"] = gates
    return out


def build_dataset_table(
    frozen: pd.DataFrame,
    dataset_name: str,
    h5ad_path: Path | None,
    include_stress: bool,
    pathway_scores: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    import scanpy as sc

    name = cfg.resolve_dataset_name(dataset_name)
    sub = frozen[frozen["dataset"].astype(str).map(cfg.resolve_dataset_name) == name]
    if sub.empty:
        sub = frozen[frozen["dataset"].astype(str) == dataset_name]
    if len(sub) < MIN_N:
        print(f"\n>>> {name}: skip (frozen n={len(sub)} < {MIN_N})")
        return None

    print(f"\n>>> {name} (frozen n={len(sub)})", flush=True)
    setup_cache()
    sc.settings.datasetdir = Path(os.environ.get("SCVERSE_DATADIR", str(cfg.CACHE_DIR)))

    h5 = h5ad_path if (h5ad_path and "UPR" in name) else None
    raw = load_raw(name, prefer_local=True, h5ad_path=h5)
    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    # QC on the same cells as Sp / pathway (stable hash materialize)
    adata, _, _ = materialize_min_cells(
        adata, pert_col, ctrl_label, seed=cfg.SEED
    )
    adata = ensure_in_memory(adata)
    ensure_cell_qc(adata)
    qc = per_pert_means(adata, pert_col, ctrl_label, QC_COLS)
    print(
        f"    QC means: mito={qc['qc_percent_mito'].mean():.2f}%  "
        f"ngenes={qc['qc_n_genes'].mean():.0f}  "
        f"ncounts={qc['qc_n_counts'].mean():.0f}"
    )
    if "Replogle" in name and qc["qc_percent_mito"].mean() > 8:
        print(
            "    NOTE: Replogle mean mito is elevated vs other datasets; "
            "strongest pathway hits land here — report explicitly."
        )

    # Prefer frozen pathway scores (one number); else recompute on same cells
    pw_cols = []
    if pathway_scores is not None:
        ps = pathway_scores[
            pathway_scores["dataset"].astype(str).map(cfg.resolve_dataset_name) == name
        ].copy()
        if ps.empty:
            ps = pathway_scores[pathway_scores["dataset"].astype(str) == name].copy()
        pw_cols = [c for c in ps.columns if c.startswith("pw_")]
        if pw_cols:
            pw = ps.set_index(ps["perturbation"].astype(str))[pw_cols]
            print(f"    joined pathway scores ({len(pw)} rows, {len(pw_cols)} pathways)")
        else:
            pw = None
    else:
        pw = None

    if pw is None or (isinstance(pw, pd.DataFrame) and pw.empty):
        already_log, log_src = resolve_matrix_is_log(dataset_name=name, adata=adata)
        if already_log:
            print(f"    skip normalize/log1p (matrix_is_log=True via {log_src})")
        else:
            print(f"    normalize_total + log1p (matrix_is_log=False via {log_src})")
            try:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            except Exception:
                _normalize_total_numpy(adata, 1e4)
                _log1p_inplace(adata)
        pw = score_pathways(adata, pert_col, ctrl_label, cfg.MIN_CELLS)

    del adata

    stab_col = "stability" if "stability" in sub.columns else "Sp"
    base = sub.set_index(sub["perturbation"].astype(str))[
        [stab_col, "magnitude"]
    ].rename(columns={stab_col: "stability"})
    out = base.join(qc, how="inner").join(pw, how="left")

    if include_stress:
        for m in STRESS_MARKERS:
            col = f"stress_{m}"
            if col in sub.columns:
                out[col] = sub.set_index(sub["perturbation"].astype(str))[col]

    out["dataset"] = name
    out = out.reset_index().rename(columns={"index": "perturbation"})
    if "perturbation" not in out.columns:
        out = out.rename(columns={out.columns[0]: "perturbation"})
    print(f"    joined table n={len(out)}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-sp", type=Path, default=None)
    p.add_argument(
        "--pathway-scores",
        type=Path,
        default=None,
        help="pathway_scores_per_pert.csv from pathway_analysis (preferred; "
             "guarantees |mag matches Approach A)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--adamson-h5ad", type=Path, default=None)
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=None,
        help=f"Bootstrap draws (default: pipeline_config.N_BOOTSTRAP={cfg.N_BOOTSTRAP}; "
             "MUST match pathway_analysis or knife-edge CIs disagree)",
    )
    # Stress markers (DDIT3 etc.) are part of the QC gate by default.
    # Omitting them silently dropped the panel from 45→25 rows under 2026-07-29.1.
    p.add_argument(
        "--include-stress",
        dest="include_stress",
        action="store_true",
        default=True,
        help="Join stress_* columns (default on)",
    )
    p.add_argument(
        "--no-include-stress",
        dest="include_stress",
        action="store_false",
        help="Skip stress markers (pathways only)",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset display names (default: all in frozen CSV with n≥15)",
    )
    p.add_argument(
        "--verify-bootstrap",
        action="store_true",
        help="Run analyze_table twice and assert bit-identical CIs "
             "(proves pathway_bootstrap_seed is effective).",
    )
    p.add_argument(
        "--allow-stale-sp",
        action="store_true",
        help="Skip frozen Sp version/n_rows check (dangerous)",
    )
    p.add_argument(
        "--skip-fail",
        action="store_true",
        help="Continue if a dataset table fails to build (default: abort)",
    )
    args = p.parse_args()

    out_dir = args.out_dir or cfg.OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = args.frozen_sp
    if frozen_path is None:
        for cand in (
            out_dir / "shesha_crispr_results_euclidean.csv",
            out_dir / "frozen_sp_scores.csv",
            Path("/content/shesha-crispr/shesha_crispr_results_euclidean.csv"),
            Path("/content/shesha-crispr/frozen_sp_scores.csv"),
        ):
            if cand.exists():
                frozen_path = cand
                break
    if frozen_path is None or not Path(frozen_path).exists():
        raise FileNotFoundError("Need --frozen-sp (frozen_sp_scores.csv)")

    pw_path = args.pathway_scores
    if pw_path is not None and not Path(pw_path).exists():
        raise FileNotFoundError(
            f"--pathway-scores not found: {pw_path}\n"
            "Run pathway_analysis.py first (writes pathway_scores_per_pert.csv), "
            "or omit --pathway-scores to recompute (discouraged)."
        )
    if pw_path is None:
        cand = out_dir / "pathway_scores_per_pert.csv"
        pw_path = cand if cand.exists() else None
    pathway_scores = pd.read_csv(pw_path) if pw_path else None

    n_boot = int(args.n_bootstrap if args.n_bootstrap is not None else cfg.N_BOOTSTRAP)
    args.n_bootstrap = n_boot
    knife = float(getattr(cfg, "SURVIVAL_KNIFE_EDGE_ABS", 0.02))
    descriptive_n = int(getattr(cfg, "SURVIVAL_QC_DESCRIPTIVE_MAX_N", 30))
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"survival_criterion={getattr(cfg, 'SURVIVAL_CRITERION_ID', '?')}")
    print(
        f"survive = CI excludes 0 ∧ |ρ|>{getattr(cfg, 'SURVIVAL_ABS_RHO_MIN', 0.1)} "
        f"∧ FDR<{getattr(cfg, 'SURVIVAL_FDR_MAX', 0.05)}; "
        f"knife-edge ε={knife:g} (strict <; only demotes otherwise-surviving rows)"
    )
    print(
        f"QC descriptive (no CI / excluded from gate) when n < {descriptive_n}; "
        f"mito-only when n < {getattr(cfg, 'SURVIVAL_QC_MITO_ONLY_MAX_N', 40)} "
        "(post hoc conservative; declare in methods)"
    )
    print(
        "Denominator: six datasets in frozen Sp (n=2,285); "
        "five scoreable for pathway/QC (n≥15; Adamson pilot n=8 dropped)."
    )
    print(f"frozen_sp={frozen_path}")
    print(f"pathway_scores={pw_path or 'RECOMPUTE (discouraged)'}")
    print(f"n_bootstrap={n_boot}  (must equal pathway_analysis N_BOOTSTRAP)")
    print(
        "covariates: magnitude + percent_mito + n_genes + n_counts "
        f"(mito-only when n < {getattr(cfg, 'SURVIVAL_QC_MITO_ONLY_MAX_N', 40)})"
    )
    print(
        "NOTE: mito/ngenes partly downstream of apoptosis — |mag+QC is a "
        "conservative lower bound, not an unbiased estimate."
    )
    print("NOTE: n_genes ≈ n_counts near-collinear; widens CIs, does not bias.")

    if not args.allow_stale_sp:
        # Enriched Sp must still carry the frozen version stamp / full n
        assert_frozen_sp_compatible(
            frozen_path,
            expect_n_rows=(
                None if args.datasets else getattr(cfg, "FROZEN_SP_EXPECTED_N_ROWS", 2285)
            ),
        )

    frozen = pd.read_csv(frozen_path)
    stress_in_csv = [m for m in STRESS_MARKERS if f"stress_{m}" in frozen.columns
                     and frozen[f"stress_{m}"].notna().any()]
    if args.include_stress and not stress_in_csv:
        print(
            "WARNING: --include-stress set but no stress_* columns in Sp CSV.\n"
            "  Run: python attach_stress_markers.py "
            "--input …/frozen_sp_scores.csv "
            "--out …/shesha_crispr_results_euclidean.csv\n"
            "  then pass that file as --frozen-sp. Otherwise DDIT3 panel is absent."
        )
    datasets = args.datasets or sorted(frozen["dataset"].dropna().unique())

    tables = []
    errors = {}
    for ds in datasets:
        try:
            t = build_dataset_table(
                frozen, ds, args.adamson_h5ad, args.include_stress,
                pathway_scores=pathway_scores,
            )
            if t is not None:
                tables.append(t)
        except Exception as e:
            errors[ds] = str(e)
            print(f"FAILED {ds}: {e}")
            if not args.skip_fail:
                raise RuntimeError(
                    f"QC table build failed for {ds}: {e}\n"
                    "Refusing partial gate output. Pass --skip-fail only to debug."
                ) from e

    if not tables:
        raise RuntimeError("No dataset tables built")

    big = pd.concat(tables, ignore_index=True)
    # Gate must see all required scoreable datasets (Replogle absence → abort)
    if not args.datasets and not args.skip_fail:
        got = {cfg.resolve_dataset_name(str(x)) for x in big["dataset"].unique()}
        required = list(getattr(cfg, "PATHWAY_REQUIRED_DATASETS", []))
        missing = [d for d in required if d not in got]
        if missing:
            raise RuntimeError(
                f"Partial QC table — missing {missing}. "
                f"n={len(big)} (five-scoreable expectation ≈ "
                f"{getattr(cfg, 'FROZEN_SP_EXPECTED_SCOREABLE_N', 2277)}). "
                "Not writing gate CSVs."
            )
        if pathway_scores is not None:
            pw_got = {
                cfg.resolve_dataset_name(str(x))
                for x in pathway_scores["dataset"].unique()
            }
            pw_missing = [d for d in required if d not in pw_got]
            if pw_missing:
                raise RuntimeError(
                    f"pathway_scores_per_pert.csv missing {pw_missing} — "
                    "re-run pathway_analysis after fixing Replogle download."
                )

    per_path = out_dir / "cell_quality_per_perturbation.csv"
    big.to_csv(per_path, index=False)
    expect_sc = getattr(cfg, "FROZEN_SP_EXPECTED_SCOREABLE_N", 2277)
    print(
        f"\nWrote {per_path} ({len(big)} rows; "
        f"five-scoreable expectation = {expect_sc})"
    )
    if len(big) < int(expect_sc) * 0.9 and not args.datasets:
        raise RuntimeError(
            f"QC per-perturbation table looks partial (n={len(big)} << {expect_sc}). "
            "Refusing to continue to gate verdict."
        )

    feat_cols = [c for c in big.columns if c.startswith("pw_")]
    if args.include_stress:
        feat_cols += [f"stress_{m}" for m in STRESS_MARKERS if f"stress_{m}" in big.columns]
        print(f"stress features in gate: {[c for c in feat_cols if c.startswith('stress_')]}")

    print("\n" + "=" * 72)
    print("PARTIALS: |magnitude  vs  |magnitude+QC  (both CIs + residual R²)")
    print("=" * 72)
    res = analyze_table(big, feat_cols, args.n_bootstrap)
    if args.verify_bootstrap:
        print("\n--verify-bootstrap: second pass …")
        res2 = analyze_table(big, feat_cols, args.n_bootstrap)
        cmp_cols = [
            c for c in res.columns
            if c.startswith("rho_") or c.startswith("p_") or c.endswith("_ci_low")
            or c.endswith("_ci_high") or c in ("gate", "survives_magnitude", "survives_mag_qc")
        ]
        # Align on dataset×feature
        a = res.set_index(["dataset", "feature"])[cmp_cols].sort_index()
        b = res2.set_index(["dataset", "feature"])[cmp_cols].sort_index()
        if not a.equals(b):
            # Numeric float compare with nan equality
            mismatch = []
            for col in cmp_cols:
                if col in ("gate",):
                    if not (a[col] == b[col]).all():
                        mismatch.append(col)
                    continue
                va, vb = a[col].astype(float), b[col].astype(float)
                same = ((va == vb) | (va.isna() & vb.isna())).all()
                if not same:
                    mismatch.append(col)
            if mismatch:
                raise RuntimeError(
                    f"Bootstrap not bit-identical across runs; mismatch cols={mismatch}. "
                    "Check pathway_bootstrap_seed / RNG."
                )
        print("  PASS: bit-identical CIs / gates across two gate runs.")

    res_path = out_dir / "cell_quality_partials.csv"
    res.to_csv(res_path, index=False)
    print(f"\nWrote {res_path} ({len(res)} rows)")

    apoptosis_p53_redundancy(big, out_dir, gate_res=res)

    # Gene-set coverage caveat (Replogle carries both pathway survivors)
    cov_path = out_dir / "pathway_gene_coverage.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        focus_cov = cov[cov["pathway"].isin(["p53", "Apoptosis"])]
        thin = focus_cov[focus_cov["pct_overlap"] < 80]
        if not thin.empty:
            print("\n" + "=" * 72)
            print("GENE-SET COVERAGE CAVEAT (state in methods)")
            print("=" * 72)
            for _, r in thin.iterrows():
                print(
                    f"  {r['dataset']} {r['pathway']}: "
                    f"{int(r['n_overlap'])}/{int(r['n_hallmark'])} "
                    f"({r['pct_overlap']:.0f}%) on {int(r['n_genes_dataset'])} genes"
                )

    focus = res[res["feature"].isin(["pw_Apoptosis", "pw_p53"])].copy()
    if not focus.empty:
        print("\n" + "=" * 72)
        print("GATE VERDICT — Apoptosis / p53")
        print("=" * 72)
        print(
            f"Criterion ({getattr(cfg, 'SURVIVAL_CRITERION_ID', 'ci_and_fdr.v1')}): "
            f"survive = CI excludes 0 ∧ |ρ|>{getattr(cfg, 'SURVIVAL_ABS_RHO_MIN', 0.1)} "
            f"∧ BH-FDR<{getattr(cfg, 'SURVIVAL_FDR_MAX', 0.05)}. "
            f"Knife-edge ε={knife:g} (strict <; demotes otherwise-surviving only; "
            "e.g. Norman p53 |QC high ≈ −0.015). "
            f"n < {descriptive_n} → descriptive_small_n (excluded from intervals). "
            "ε / n<30 / n<40 are post hoc conservative — declare in methods."
        )
        print(
            "Denominator: five scoreable of six frozen datasets "
            "(Adamson pilot n=8 dropped). "
            "Sign− counts: point estimates in 5/5 scoreable; "
            "QC intervals estimable only where gate ≠ descriptive_small_n. "
            "Check apoptosis_p53_redundancy.csv before claiming two pathways."
        )
        for feat, sub in focus.groupby("feature"):
            # Count from gate labels only — never mix collapses_under_qc bool
            # with indeterminate (Norman p53 was the false collapse=1).
            estimable = sub[sub["gate"] != "descriptive_small_n"]
            n_neg_mag = int((sub["rho_partial_mag"] < 0).sum())
            n_neg_qc = int((sub["rho_partial_mag_qc"] < 0).sum())
            n_neg_qc_est = int((estimable["rho_partial_mag_qc"] < 0).sum())
            n_surv_mag = int(sub["survives_magnitude"].sum())
            n_surv_qc = int((sub["gate"] == "survives_qc").sum())
            n_ind = int((sub["gate"] == "indeterminate").sum())
            n_desc = int((sub["gate"] == "descriptive_small_n").sum())
            n_supp = int((sub["gate"] == "qc_conditional").sum())
            n_collapse = int((sub["gate"] == "collapses_under_qc").sum())
            surv_ds = ", ".join(
                sub.loc[sub["gate"] == "survives_qc", "dataset"]
            ) or "(none)"
            print(
                f"  {feat}: sign− |mag {n_neg_mag}/{len(sub)} scoreable; "
                f"survive |mag {n_surv_mag}; "
                f"sign− |mag+QC {n_neg_qc}/{len(sub)} point est "
                f"({n_neg_qc_est}/{len(estimable)} with estimable CI); "
                f"survive |mag+QC {n_surv_qc}; "
                f"indeterminate {n_ind}; collapse {n_collapse}; "
                f"qc_conditional {n_supp}; descriptive {n_desc}"
            )
            print(f"    primary QC survivors: {surv_ds}")
            for _, r in sub.iterrows():
                def _fmt_ci(lo, hi):
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        return "[n/a]"
                    return f"[{lo:.3f},{hi:.3f}]"
                print(
                    f"    {r['dataset']}: |mag={r['rho_partial_mag']:+.3f} "
                    f"{_fmt_ci(r['rho_partial_mag_ci_low'], r['rho_partial_mag_ci_high'])} "
                    f"FDR={r['p_partial_mag_fdr_bh']:.3f}  "
                    f"|QC={r['rho_partial_mag_qc']:+.3f} "
                    f"{_fmt_ci(r['rho_partial_mag_qc_ci_low'], r['rho_partial_mag_qc_ci_high'])} "
                    f"FDR={r['p_partial_mag_qc_fdr_bh']:.3f}  "
                    f"frac_left={r['frac_sp_variance_remaining']:.3f}  "
                    f"R²={r['partial_r2_mag']:.3f}  {r['gate']}"
                )
        n_qc_surv = int((focus["gate"] == "survives_qc").sum())
        if n_qc_surv >= 2:
            print(
                "\n  → PRIMARY QC SURVIVORS ≥2 (conservative lower bound). "
                "Report residual R²; check redundancy CSV; "
                "indeterminate ≠ survive; descriptive ≠ survive."
            )
        elif (focus["gate"] == "collapses_under_qc").any():
            print(
                "\n  → SIGNAL COLLAPSES under cell-quality covariates. "
                "Do not lead with p53/apoptosis as response biology."
            )
        else:
            print(
                "\n  → INCONCLUSIVE. Do not write the headline until this clears."
            )

    print("\nNext: inspect cell_quality_partials.csv before any manuscript text.")


if __name__ == "__main__":
    main()
