#!/usr/bin/env python3
"""
CORUM / TF benchmarking on LOESS discordance (magnitude-adjusted Sp).

Raw Sp favors complex subunits (larger effects → higher Sp). The informative
test is whether complex genes have *higher LOESS discordance* than expected
for their magnitude (below the Sp~magnitude curve).

Reports both:
  1. Raw Sp: complex vs other (often higher Sp — magnitude-driven)
  2. LOESS discordance: complex vs other (H1: complex more discordant)

Also correlates discordance (and Sp) with max complex size and TRRUST TF
target set size.

Usage:
  python corum_loess_discordance.py \\
      --input shesha-crispr/shesha_crispr_results_euclidean.csv
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.nonparametric.smoothers_lowess import lowess

import pipeline_config as cfg
from corum_systematic_benchmark import load_corum, load_trrust
from revision_io import ANNOT_DIR, find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import bootstrap_spearman_ci

# Match fig_method_comparison / robustness_tests LOESS discordance
LOESS_FRAC = 0.3


def disc_loess(mag: np.ndarray, stab: np.ndarray, frac: float = LOESS_FRAC) -> np.ndarray:
    """Sign-flipped, z-scored LOESS residual (below curve = high discordance)."""
    mag = np.asarray(mag, dtype=float)
    stab = np.asarray(stab, dtype=float)
    fitted = lowess(stab, mag, frac=frac, return_sorted=False)
    d = -(stab - fitted)
    sd = d.std()
    if sd < 1e-12:
        return np.zeros_like(d)
    return (d - d.mean()) / sd


def disc_linear(mag: np.ndarray, stab: np.ndarray) -> np.ndarray:
    mag_z = (mag - mag.mean()) / mag.std()
    stab_z = (stab - stab.mean()) / stab.std()
    return mag_z - stab_z


def _mwu_report(a: pd.Series, b: pd.Series, alternative: str) -> dict:
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return {
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "median_a": float(a.median()) if len(a) else np.nan,
            "median_b": float(b.median()) if len(b) else np.nan,
            "mean_a": float(a.mean()) if len(a) else np.nan,
            "mean_b": float(b.mean()) if len(b) else np.nan,
            "U": np.nan,
            "p": np.nan,
            "alternative": alternative,
        }
    u, p = mannwhitneyu(a, b, alternative=alternative)
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(a.median()),
        "median_b": float(b.median()),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "U": float(u),
        "p": float(p),
        "alternative": alternative,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="Replogle 2022 (CRISPRi)")
    parser.add_argument("--corum-zip", type=Path, default=None)
    parser.add_argument("--loess-frac", type=float, default=LOESS_FRAC)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    sp_path = find_sp_csv(out_dir, args.input)
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"Sp table: {sp_path}")

    df = load_sp_table(sp_path)
    ds = cfg.resolve_dataset_name(args.dataset)
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        sub = df[df["dataset"].str.contains("Replogle", case=False, na=False)].copy()
        ds = sub["dataset"].iloc[0] if len(sub) else ds
    if sub.empty:
        raise SystemExit(f"No rows for {args.dataset!r}. Have: {list(df['dataset'].unique())}")

    sub = sub.dropna(subset=["stability", "magnitude"]).copy()
    print(f"Using {ds}: n={len(sub)} perturbations")
    print(f"LOESS frac={args.loess_frac}")

    sub["disc_loess"] = disc_loess(
        sub["magnitude"].values, sub["stability"].values, frac=args.loess_frac
    )
    sub["disc_linear"] = disc_linear(sub["magnitude"].values, sub["stability"].values)

    corum = load_corum(ANNOT_DIR, local_path=args.corum_zip)
    trrust = load_trrust(ANNOT_DIR)

    merged = sub.merge(corum, on="gene", how="left")
    merged["in_corum"] = merged["in_corum"].fillna(False).astype(bool)
    merged["max_complex_size"] = merged["max_complex_size"].fillna(0)
    merged["n_complexes"] = merged["n_complexes"].fillna(0)
    merged = merged.merge(trrust, on="gene", how="left")

    complex_m = merged["in_corum"]
    print(f"  CORUM members in dataset: {int(complex_m.sum())} / {len(merged)}")

    # --- Group tests ---
    tests = {}

    # Raw Sp: historical H1 complex lower; also report two-sided / greater for honesty
    sp_c = merged.loc[complex_m, "stability"]
    sp_o = merged.loc[~complex_m, "stability"]
    tests["raw_Sp_complex_lower"] = _mwu_report(sp_c, sp_o, "less")
    tests["raw_Sp_complex_higher"] = _mwu_report(sp_c, sp_o, "greater")
    print("\n=== Raw Sp: complex vs non-complex ===")
    print(
        f"  complex median Sp={tests['raw_Sp_complex_higher']['median_a']:.3f} "
        f"(n={tests['raw_Sp_complex_higher']['n_a']})"
    )
    print(
        f"  other   median Sp={tests['raw_Sp_complex_higher']['median_b']:.3f} "
        f"(n={tests['raw_Sp_complex_higher']['n_b']})"
    )
    print(
        f"  MWU H1 complex higher: p={tests['raw_Sp_complex_higher']['p']:.3e}  "
        f"| H1 complex lower: p={tests['raw_Sp_complex_lower']['p']:.3e}"
    )

    # LOESS discordance: H1 complex more discordant than expected for magnitude
    d_c = merged.loc[complex_m, "disc_loess"]
    d_o = merged.loc[~complex_m, "disc_loess"]
    tests["disc_loess_complex_greater"] = _mwu_report(d_c, d_o, "greater")
    tests["disc_loess_complex_less"] = _mwu_report(d_c, d_o, "less")
    print("\n=== LOESS discordance: complex vs non-complex ===")
    print(
        f"  complex median disc={tests['disc_loess_complex_greater']['median_a']:.3f}"
    )
    print(
        f"  other   median disc={tests['disc_loess_complex_greater']['median_b']:.3f}"
    )
    print(
        f"  MWU H1 complex MORE discordant: "
        f"p={tests['disc_loess_complex_greater']['p']:.3e}"
    )

    # Linear discordance for comparison
    tests["disc_linear_complex_greater"] = _mwu_report(
        merged.loc[complex_m, "disc_linear"],
        merged.loc[~complex_m, "disc_linear"],
        "greater",
    )
    print(
        f"  (linear discordance H1 greater: "
        f"p={tests['disc_linear_complex_greater']['p']:.3e})"
    )

    # --- Correlations vs discordance and vs Sp ---
    corr_rows = []
    for y_label, y_col, mask_fn in [
        ("max_complex_size", "max_complex_size", lambda m: m["in_corum"]),
        ("n_complexes", "n_complexes", lambda m: m["in_corum"]),
        ("n_tf_targets", "n_tf_targets", lambda m: m["n_tf_targets"].notna()),
        ("magnitude", "magnitude", lambda m: np.ones(len(m), dtype=bool)),
    ]:
        if y_col not in merged.columns:
            continue
        for x_name in ("disc_loess", "stability"):
            m = merged.loc[mask_fn(merged), [x_name, y_col]].dropna()
            if len(m) < 15:
                print(f"  skip {x_name} ~ {y_label}: n={len(m)}")
                continue
            boot = bootstrap_spearman_ci(
                m[x_name], m[y_col], n_bootstrap=args.n_bootstrap, seed=cfg.SEED
            )
            print(
                f"  {x_name} ~ {y_label}: rho={boot['rho']:+.3f} "
                f"[{boot['ci_low']:.3f},{boot['ci_high']:.3f}] "
                f"n={boot['n']} p={boot['p']:.2e}"
            )
            corr_rows.append(
                {
                    "dataset": ds,
                    "x": x_name,
                    "y": y_label,
                    "n": boot["n"],
                    "rho": boot["rho"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "p": boot["p"],
                    "loess_frac": args.loess_frac,
                    "config_version": cfg.CONFIG_VERSION,
                }
            )

    # Top discordant complex members (for transparency vs SF3B2/CHMP2A examples)
    top_complex = (
        merged.loc[complex_m]
        .nlargest(15, "disc_loess")[
            ["perturbation", "gene", "stability", "magnitude", "disc_loess",
             "max_complex_size", "example_complex"]
        ]
    )
    print("\n=== Top 15 LOESS-discordant CORUM members ===")
    print(top_complex.to_string(index=False))

    # Manuscript recommendation
    p_disc = tests["disc_loess_complex_greater"]["p"]
    p_sp_hi = tests["raw_Sp_complex_higher"]["p"]
    rho_tf_disc = next(
        (r["rho"] for r in corr_rows if r["x"] == "disc_loess" and r["y"] == "n_tf_targets"),
        None,
    )
    rho_size_disc = next(
        (r["rho"] for r in corr_rows if r["x"] == "disc_loess" and r["y"] == "max_complex_size"),
        None,
    )

    if p_disc == p_disc and p_disc < 0.05:
        disc_verdict = (
            "SUPPORT (magnitude-adjusted): complex subunits are more LOESS-discordant "
            "than non-complex genes — keep a cautious complex-incoherence claim."
        )
    else:
        disc_verdict = (
            "NULL (magnitude-adjusted): complex subunits are not systematically more "
            "discordant than expected for their magnitude. The multi-subunit "
            "incoherence narrative was driven by illustrative examples "
            "(e.g. SF3B2, CHMP2A), not a genome-wide pattern. Prefer gene × context "
            "framing over broad complex generalizations."
        )

    blurb = (
        f"CORUM benchmarking ({ds}). Raw Sp is higher for CORUM complex-subunit "
        f"perturbations than others (median "
        f"{tests['raw_Sp_complex_higher']['median_a']:.3f} vs "
        f"{tests['raw_Sp_complex_higher']['median_b']:.3f}; "
        f"MWU H1 higher p={p_sp_hi:.2e}), consistent with larger effect sizes among "
        f"essential complex genes. The magnitude-adjusted test uses LOESS "
        f"discordance (frac={args.loess_frac}; below Sp~magnitude curve). "
        f"Complex vs other discordance: median "
        f"{tests['disc_loess_complex_greater']['median_a']:.3f} vs "
        f"{tests['disc_loess_complex_greater']['median_b']:.3f} "
        f"(MWU H1 complex more discordant p={p_disc:.2e}). "
    )
    if rho_size_disc is not None:
        blurb += f"Discordance ~ max complex size: ρ={rho_size_disc:+.3f}. "
    if rho_tf_disc is not None:
        blurb += (
            f"Discordance ~ TRRUST TF target-set size: ρ={rho_tf_disc:+.3f} "
            f"(raw Sp~targets remains a separate null). "
        )
    blurb += disc_verdict

    print(blurb)

    # Outputs
    merged_path = out_dir / "corum_loess_discordance_merged.csv"
    merged.to_csv(merged_path, index=False)
    top_complex.to_csv(out_dir / "corum_loess_top_discordant_complex.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(
        out_dir / "corum_loess_discordance_correlations.csv", index=False
    )

    # Flatten tests for CSV
    test_rows = []
    for name, t in tests.items():
        row = {"test": name, "dataset": ds, "config_version": cfg.CONFIG_VERSION}
        row.update(t)
        test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(
        out_dir / "corum_loess_discordance_group_tests.csv", index=False
    )

    summary = {
        "dataset": ds,
        "config_version": cfg.CONFIG_VERSION,
        "loess_frac": args.loess_frac,
        "n": int(len(merged)),
        "n_corum": int(complex_m.sum()),
        "tests": tests,
        "correlations": corr_rows,
        "manuscript_recommendation": disc_verdict,
        "methods_blurb": blurb,
        "source_csv": str(sp_path),
    }
    with open(out_dir / "corum_loess_discordance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "corum_loess_discordance_blurb.txt", "w") as f:
        f.write(blurb + "\n")

    print(
        f"\nWrote:\n  {merged_path}\n  corum_loess_discordance_group_tests.csv\n"
        f"  corum_loess_discordance_correlations.csv\n"
        f"  corum_loess_discordance_summary.json\n"
        f"  corum_loess_discordance_blurb.txt"
    )


if __name__ == "__main__":
    main()
