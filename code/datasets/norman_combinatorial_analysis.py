#!/usr/bin/env python3
"""
Norman combinatorial perturbation analysis.

List all combinatorial perturbations and test whether pairs containing known
erythroid factors (KLF1, GATA1, …) drive higher Sp. Supports or refutes the
developmental-trajectory claim; if null, remove the claim from the manuscript.

Usage:
  python norman_combinatorial_analysis.py \\
      --input shesha-crispr/frozen_sp_scores.csv
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
import re
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

import pipeline_config as cfg
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir

DATASET = "Norman 2019 (CRISPRa)"

# Erythroid / lineage factors discussed in the manuscript
ERYTHROID_FACTORS = {
    "KLF1", "GATA1", "GATA2", "NFE2", "TAL1", "LMO2", "RUNX1",
    "SPI1", "CEBPA", "CEBPB", "IKZF1", "MYB", "FOG1", "ZFPM1",
}


def parse_partners(pert: str) -> list[str]:
    """Norman labels are typically GeneA+GeneB or GeneA_GeneB."""
    s = str(pert).strip()
    if "+" in s:
        parts = s.split("+")
    elif ";" in s:
        parts = s.split(";")
    else:
        # underscore: treat as combo only if exactly two gene-like tokens
        parts = s.split("_")
        if len(parts) != 2:
            return [s.upper()]
        # avoid treating single genes with weird suffixes
        if any(len(p) < 2 for p in parts):
            return [s.upper()]
    return [p.strip().upper() for p in parts if p.strip()]


def is_combinatorial(pert: str) -> bool:
    partners = parse_partners(pert)
    if "+" in str(pert) or ";" in str(pert):
        return len(partners) >= 2
    # underscore doubles
    parts = str(pert).split("_")
    return len(parts) == 2 and all(re.match(r"^[A-Za-z0-9-]+$", p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--factors",
        nargs="*",
        default=sorted(ERYTHROID_FACTORS),
        help="Gene symbols counted as erythroid/lineage factors",
    )
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    path = find_sp_csv(out_dir, args.input)
    df = load_sp_table(path)
    ds = cfg.resolve_dataset_name(DATASET)
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        raise SystemExit(f"No {ds} rows in {path}")

    factors = {f.upper() for f in args.factors}
    sub["partners"] = sub["perturbation"].map(parse_partners)
    sub["n_partners"] = sub["partners"].map(len)
    sub["is_combo"] = sub["perturbation"].map(is_combinatorial)
    # Prefer explicit + / ; ; fall back to n_partners>=2 from parser
    sub.loc[sub["n_partners"] >= 2, "is_combo"] = True
    sub["has_erythroid"] = sub["partners"].map(
        lambda ps: bool(set(ps) & factors)
    )
    sub["erythroid_partners"] = sub["partners"].map(
        lambda ps: ",".join(sorted(set(ps) & factors))
    )
    sub["is_single"] = ~sub["is_combo"]

    combos = sub[sub["is_combo"]].copy()
    singles = sub[sub["is_single"]].copy()

    # Full combo list
    combo_list = combos[
        ["perturbation", "stability", "magnitude", "n_partners",
         "has_erythroid", "erythroid_partners"]
        + (["n_cells"] if "n_cells" in combos.columns else [])
    ].sort_values("stability", ascending=False)

    combo_list.to_csv(out_dir / "norman_combinatorial_list.csv", index=False)

    results = {
        "dataset": ds,
        "n_total": int(len(sub)),
        "n_single": int(len(singles)),
        "n_combinatorial": int(len(combos)),
        "erythroid_factors": sorted(factors),
        "config_version": cfg.CONFIG_VERSION,
        "source_csv": str(path),
    }

    print(f"Norman: {results['n_total']} perts "
          f"({results['n_single']} single, {results['n_combinatorial']} combo)")

    # Test 1: combo vs single Sp
    if len(combos) >= 10 and len(singles) >= 10:
        u, p = mannwhitneyu(
            combos["stability"], singles["stability"], alternative="two-sided"
        )
        results["combo_vs_single"] = {
            "median_sp_combo": float(combos["stability"].median()),
            "median_sp_single": float(singles["stability"].median()),
            "mean_sp_combo": float(combos["stability"].mean()),
            "mean_sp_single": float(singles["stability"].mean()),
            "mannwhitney_U": float(u),
            "mannwhitney_p": float(p),
        }
        print(
            f"  Combo vs single Sp: median "
            f"{results['combo_vs_single']['median_sp_combo']:.3f} vs "
            f"{results['combo_vs_single']['median_sp_single']:.3f} "
            f"(MW p={p:.3g})"
        )

    # Test 2: among combos, erythroid-containing vs not
    if len(combos) >= 10:
        ery = combos[combos["has_erythroid"]]
        other = combos[~combos["has_erythroid"]]
        results["n_combo_with_erythroid"] = int(len(ery))
        results["n_combo_without_erythroid"] = int(len(other))
        if len(ery) >= 5 and len(other) >= 5:
            u, p = mannwhitneyu(
                ery["stability"], other["stability"], alternative="greater"
            )
            results["erythroid_combo_vs_other"] = {
                "median_sp_erythroid": float(ery["stability"].median()),
                "median_sp_other": float(other["stability"].median()),
                "mean_sp_erythroid": float(ery["stability"].mean()),
                "mean_sp_other": float(other["stability"].mean()),
                "mannwhitney_U": float(u),
                "mannwhitney_p_greater": float(p),
                "hypothesis": (
                    "pairs containing erythroid factors have higher Sp "
                    "(one-sided greater)"
                ),
            }
            print(
                f"  Erythroid-containing combos vs other: median Sp "
                f"{ery['stability'].median():.3f} vs {other['stability'].median():.3f} "
                f"(one-sided MW p={p:.3g}; n={len(ery)} vs {len(other)})"
            )
        else:
            results["erythroid_combo_vs_other"] = {
                "note": f"Insufficient n (ery={len(ery)}, other={len(other)})"
            }

    # Test 3: magnitude-matched — residual Sp after regressing on magnitude
    if len(combos) >= 15:
        from sklearn.linear_model import LinearRegression

        X = combos[["magnitude"]].values
        y = combos["stability"].values
        resid = y - LinearRegression().fit(X, y).predict(X)
        combos = combos.copy()
        combos["sp_resid_mag"] = resid
        ery = combos[combos["has_erythroid"]]
        other = combos[~combos["has_erythroid"]]
        if len(ery) >= 5 and len(other) >= 5:
            u, p = mannwhitneyu(
                ery["sp_resid_mag"], other["sp_resid_mag"], alternative="greater"
            )
            results["erythroid_combo_vs_other_mag_residual"] = {
                "median_resid_erythroid": float(ery["sp_resid_mag"].median()),
                "median_resid_other": float(other["sp_resid_mag"].median()),
                "mannwhitney_p_greater": float(p),
            }
            print(
                f"  Mag-residual Sp: erythroid vs other p={p:.3g}"
            )

    # Recommendation for manuscript
    ery_test = results.get("erythroid_combo_vs_other", {})
    p_ery = ery_test.get("mannwhitney_p_greater")
    if p_ery is not None and p_ery < 0.05:
        rec = (
            "SUPPORT: erythroid-factor combinations show higher Sp — "
            "keep a cautious gene × context claim with this test cited."
        )
    elif p_ery is not None:
        rec = (
            "REMOVE or soften developmental-trajectory claim: "
            "erythroid-containing combinations do not show significantly "
            "higher Sp in this frozen analysis. Keep the full combo list as SI."
        )
    else:
        rec = (
            "INCONCLUSIVE: report the full combinatorial list (SI) and avoid "
            "strong trajectory claims without a clearer test."
        )
    results["manuscript_recommendation"] = rec
    print(f"\n*** {rec}")

    blurb = (
        f"Norman combinatorial audit. Of {results['n_total']} scored perturbations, "
        f"{results['n_combinatorial']} are combinatorial (full list: "
        f"norman_combinatorial_list.csv). "
    )
    if "combo_vs_single" in results:
        c = results["combo_vs_single"]
        blurb += (
            f"Median Sp combo vs single: {c['median_sp_combo']:.3f} vs "
            f"{c['median_sp_single']:.3f} (MW p={c['mannwhitney_p']:.3g}). "
        )
    if p_ery is not None:
        blurb += (
            f"Combinations containing erythroid/lineage factors "
            f"({', '.join(sorted(factors)[:6])}, …; n={results.get('n_combo_with_erythroid')}) "
            f"vs other combinations: median Sp "
            f"{ery_test['median_sp_erythroid']:.3f} vs "
            f"{ery_test['median_sp_other']:.3f} "
            f"(one-sided MW p={p_ery:.3g}). "
        )
    blurb += rec
    results["methods_blurb"] = blurb

    with open(out_dir / "norman_combinatorial_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "norman_combinatorial_blurb.txt", "w") as f:
        f.write(blurb + "\n")

    # Annotated table for all Norman rows
    export_cols = [
        "perturbation", "stability", "magnitude", "is_combo", "is_single",
        "n_partners", "has_erythroid", "erythroid_partners",
    ]
    if "n_cells" in sub.columns:
        export_cols.insert(3, "n_cells")
    sub[export_cols].to_csv(out_dir / "norman_all_perturbations_annotated.csv", index=False)
    print(f"Wrote {out_dir}/norman_combinatorial_summary.json")
    print(f"Wrote {out_dir}/norman_combinatorial_list.csv ({len(combo_list)} combos)")


if __name__ == "__main__":
    main()
