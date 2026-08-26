#!/usr/bin/env python3
"""
Figure 7 gate: Norman combinatorial vs single-gene, magnitude-conditioned.

Pre-specified before looking at partials:
  y = Shesha coherence (Sp)
  x = binary combinatorial indicator (1 = combo, 0 = single)
  Six rank-partial models, ci_and_fdr.v1, knife-edge demote-only.
  Verdict = intersection of the six.

  keep as main  — survives all six
  SI            — survives some
  DELETE        — survives none (report in negatives)

Also reports magnitude / n_cells standardized differences.
E-distance models run only if edistance_scores_per_pert.csv is found.
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
from scipy.stats import mannwhitneyu, spearmanr


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1], start=0):
        rank = n - i
        val = min(prev, p[idx] * n / rank)
        ranked[idx] = val
        prev = val
    return np.clip(ranked, 0, 1)

import pipeline_config as cfg
from revision_io import data_search_dirs, find_data_file, load_sp_table, resolve_out_dir
from stats_utils import bootstrap_partial_spearman_ci, pathway_bootstrap_seed, survival_status

DATASET = "Norman 2019 (CRISPRa)"

_CSV_ROOTS = data_search_dirs()


def _find(*names):
    for name in names:
        for root in _CSV_ROOTS:
            p = root / name
            if p.exists():
                return p
    return None


def _smd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def _overlap_iqr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = max(np.percentile(a, 25), np.percentile(b, 25))
    hi = min(np.percentile(a, 75), np.percentile(b, 75))
    return float(max(0.0, hi - lo))


def main():
    sp_path = _find("frozen_sp_scores.csv", "shesha_crispr_results_euclidean.csv")
    qc_path = _find("cell_quality_per_perturbation.csv")
    ed_path = _find("edistance_scores_per_pert.csv")
    if sp_path is None:
        raise FileNotFoundError("Need frozen_sp_scores.csv")
    print(f"Sp: {sp_path}")
    print(f"QC: {qc_path}")
    print(f"E-distance: {ed_path}")

    df = load_sp_table(sp_path)
    df["dataset"] = df["dataset"].map(cfg.resolve_dataset_name)
    sub = df[df["dataset"] == DATASET].copy()
    sub["perturbation"] = sub["perturbation"].astype(str).str.strip()
    sub["is_combo"] = sub["perturbation"].str.contains(r"\+", regex=True).astype(int)
    n_combo = int(sub["is_combo"].sum())
    n_single = int((sub["is_combo"] == 0).sum())
    print(f"Norman: n={len(sub)}  single={n_single}  combo={n_combo}")

    if qc_path is not None:
        qc = pd.read_csv(qc_path)
        qc["dataset"] = qc["dataset"].map(cfg.resolve_dataset_name)
        qc = qc[qc["dataset"] == DATASET]
        qc["perturbation"] = qc["perturbation"].astype(str).str.strip()
        keep = ["perturbation", "qc_percent_mito", "qc_n_genes"]
        if "qc_n_counts" in qc.columns:
            keep.append("qc_n_counts")
        sub = sub.merge(qc[keep], on="perturbation", how="left")

    if ed_path is not None:
        ed = pd.read_csv(ed_path)
        if "dataset" in ed.columns:
            ed["dataset"] = ed["dataset"].map(cfg.resolve_dataset_name)
            ed = ed[ed["dataset"] == DATASET]
        ed["perturbation"] = ed["perturbation"].astype(str).str.strip()
        ecol = next(
            (c for c in ("edistance", "e_distance", "E_distance") if c in ed.columns),
            None,
        )
        if ecol is not None:
            sub = sub.merge(
                ed[["perturbation", ecol]].rename(columns={ecol: "edistance"}),
                on="perturbation", how="left",
            )

    combo = sub[sub["is_combo"] == 1]
    single = sub[sub["is_combo"] == 0]

    mag_smd = _smd(combo["magnitude"], single["magnitude"])
    ncell_smd = _smd(combo["n_cells"], single["n_cells"]) if "n_cells" in sub.columns else np.nan
    rho_sp_mag, p_sp_mag = spearmanr(sub["stability"], sub["magnitude"])
    u, p_mw = mannwhitneyu(combo["stability"], single["stability"], alternative="two-sided")
    rho_raw, p_raw = spearmanr(sub["stability"], sub["is_combo"])

    print("\n=== Unconditioned / confound checks (not the gate) ===")
    print(f"  mean Sp  combo={combo['stability'].mean():.3f}  single={single['stability'].mean():.3f}")
    print(f"  MW U={u:.1f}  p={p_mw:.3e}")
    print(f"  raw Spearman(Sp, combo) = {rho_raw:+.3f}  p={p_raw:.3e}")
    print(f"  Spearman(Sp, magnitude) = {rho_sp_mag:+.3f}  p={p_sp_mag:.3e}")
    print(f"  mean magnitude combo={combo['magnitude'].mean():.3f}  "
          f"single={single['magnitude'].mean():.3f}  SMD={mag_smd:+.3f}")
    print(f"  magnitude IQR overlap = {_overlap_iqr(combo['magnitude'], single['magnitude']):.3f}")
    if np.isfinite(ncell_smd):
        print(f"  mean n_cells combo={combo['n_cells'].mean():.2f}  "
              f"single={single['n_cells'].mean():.2f}  SMD={ncell_smd:+.3f}")

    models = [("magnitude", ["magnitude"])]
    if "edistance" in sub.columns and sub["edistance"].notna().sum() >= 50:
        models.append(("edistance", ["edistance"]))
        models.append(("magnitude+edistance", ["magnitude", "edistance"]))
    else:
        print("\nE-distance table not found — models 2/3/5/6 skipped.")

    qc_cols = [c for c in ("qc_percent_mito", "qc_n_genes") if c in sub.columns]
    extra = qc_cols + (["n_cells"] if "n_cells" in sub.columns else [])
    if extra:
        models.append(("magnitude+QC+n_cells", ["magnitude"] + extra))
        if "edistance" in sub.columns and sub["edistance"].notna().sum() >= 50:
            models.append(("edistance+QC+n_cells", ["edistance"] + extra))
            models.append(("magnitude+edistance+QC+n_cells",
                           ["magnitude", "edistance"] + extra))

    print(f"\n=== Gate models ({cfg.SURVIVAL_CRITERION_ID}, n_boot={cfg.N_BOOTSTRAP}) ===")
    rows = []
    y = sub["stability"].to_numpy(dtype=float)
    x = sub["is_combo"].to_numpy(dtype=float)
    for name, cols in models:
        Z = sub[cols].to_numpy(dtype=float)
        seed = pathway_bootstrap_seed(DATASET, f"combo|{name}", stage="fig7")
        part = bootstrap_partial_spearman_ci(
            x, y, Z,
            n_bootstrap=cfg.N_BOOTSTRAP,
            ci_level=cfg.CI_LEVEL,
            seed=seed,
            method="rank",
        )
        rows.append({
            "model": name,
            "covariates": ",".join(cols),
            "n": part["n"],
            "rho_partial": part["rho_partial"],
            "partial_r2": float(part["rho_partial"] ** 2) if np.isfinite(part["rho_partial"]) else np.nan,
            "p": part["p"],
            "ci_low": part["ci_low"],
            "ci_high": part["ci_high"],
            "bootstrap_seed": seed,
        })
        print(
            f"  {name:32s}  ρ={part['rho_partial']:+.3f}  "
            f"R²={part['rho_partial']**2:.3f}  "
            f"[{part['ci_low']:.3f}, {part['ci_high']:.3f}]  p={part['p']:.3e}"
        )

    tab = pd.DataFrame(rows)
    if len(tab) and tab["p"].notna().any():
        tab["fdr"] = _bh_fdr(tab["p"].fillna(1.0).to_numpy())
    else:
        tab["fdr"] = np.nan

    statuses = []
    for i, r in tab.iterrows():
        st = survival_status(r["rho_partial"], r["ci_low"], r["ci_high"], fdr=r["fdr"])
        for k, v in st.items():
            tab.loc[i, k] = v
        statuses.append(st["status"])
        print(f"    -> {r['model']}: {st['status']}")

    n_surv = int((tab["status"] == "survives").sum()) if len(tab) else 0
    n_mod = len(tab)
    planned = 6
    if n_surv == planned:
        verdict = "KEEP_MAIN"
        reason = "Survives all six pre-specified models."
    elif n_surv > 0:
        verdict = "SI_ONLY"
        reason = f"Survives {n_surv}/{n_mod} fitted models (planned {planned})."
    else:
        verdict = "DELETE"
        reason = (
            f"Survives 0/{n_mod} fitted models. "
            "Report in the negatives subsection."
        )
    if n_mod < planned and verdict == "KEEP_MAIN":
        verdict = "SI_ONLY"
        reason = f"Cannot claim all six; only {n_mod} models were fitted."

    print(f"survives {n_surv}/{n_mod} models (planned {planned})")
    print(reason)

    out_dir = resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / "fig7_combinatorial_partials.csv", index=False)
    summary = {
        "dataset": DATASET,
        "n_single": n_single,
        "n_combinatorial": n_combo,
        "rho_sp_magnitude": float(rho_sp_mag),
        "magnitude_smd_combo_minus_single": mag_smd,
        "n_cells_smd_combo_minus_single": ncell_smd,
        "unconditioned_mw_p": float(p_mw),
        "unconditioned_spearman_sp_combo": float(rho_raw),
        "n_models_fitted": n_mod,
        "n_models_planned": planned,
        "n_survive": n_surv,
        "verdict": verdict,
        "reason": reason,
        "selection_caveat": (
            "Norman chose gene pairs expected to interact; "
            "even a surviving effect is about a curated set."
        ),
        "single_dataset": True,
        "config_version": cfg.CONFIG_VERSION,
        "survival_criterion": cfg.SURVIVAL_CRITERION_ID,
        "edistance_available": bool(ed_path is not None and "edistance" in sub.columns),
    }
    (out_dir / "fig7_combinatorial_verdict.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Wrote {out_dir}/fig7_combinatorial_partials.csv")
    print(f"Wrote {out_dir}/fig7_combinatorial_verdict.json")


if __name__ == "__main__":
    main()
