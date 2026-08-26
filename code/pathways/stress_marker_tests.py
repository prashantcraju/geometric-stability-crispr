#!/usr/bin/env python3
"""
Stress Marker Deep-Dive Analysis

1. PARTIAL CORRELATIONS (primary): Sp vs stress markers | magnitude.
   With frozen Sp + pinned preprocess, DDIT3 is negative across scoreable
   datasets; report residual Sp variance alongside rho.

2. QUADRANT TEST — DROPPED by default. Median splits force HH=LL / HL=LH
   (one free parameter); the binomial is a lower-powered restatement of the
   correlation sign, and it runs on raw Sp so it can contradict the partial.
   Pass --quadrant-on-residuals to recompute on magnitude-residual Sp if needed.

3. MODALITY SUMMARY: per-modality tables for all markers. DDIT3 is no longer
   framed as a CRISPRa-vs-CRISPRi sign flip (partials are 5/5 negative).
   ATF4/XBP1 Norman↔Replogle flips belong in gene × context, not modality.

INPUT:  shesha_crispr_results_euclidean.csv (from attach_stress_markers.py)
OUTPUT: stress_partial_correlations.csv
        stress_modality_summary.csv / stress_modality_detail.csv
        (optional) stress_quadrant_tests.csv
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import subprocess
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def _detect_colab() -> bool:
    """True only inside an interactive Colab notebook kernel (not `python script.py`)."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


# =============================================================================
# CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
try:
    from scipy.stats import binomtest
    def binom_test_compat(k, n, p, alternative='less'):
        return binomtest(k, n, p, alternative=alternative).pvalue
except ImportError:
    from scipy.stats import binom_test
    def binom_test_compat(k, n, p, alternative='less'):
        return binom_test(k, n, p, alternative=alternative)
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    import pipeline_config as _pcfg
    SEED = _pcfg.SEED
    N_BOOTSTRAP = _pcfg.N_BOOTSTRAP
    CI_LEVEL = _pcfg.CI_LEVEL
    _DEFAULT_OUT = _pcfg.OUTPUT_DIR
except ImportError:
    SEED = 320
    N_BOOTSTRAP = 10000
    CI_LEVEL = 0.95
    _DEFAULT_OUT = Path("./shesha-crispr")

# Search OUTPUT_DIR then SHESHA_OUT.
# NOTE: Path("") == cwd and .exists() is True — never use empty env strings.
def _valid_dir(p: Path | None) -> bool:
    return p is not None and str(p).strip() not in {"", "."} and p.exists() and p.is_dir()


_env = os.environ.get("SHESHA_OUT", "").strip()
_CANDIDATE_OUTS = [
    Path(_env) if _env else None,
    Path("/content/shesha-crispr"),
    Path("./shesha-crispr"),
    _DEFAULT_OUT if _DEFAULT_OUT != Path(".") else None,
]
OUTPUT_DIR = next((p for p in _CANDIDATE_OUTS if _valid_dir(p)), Path("./shesha-crispr"))
print(f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}")

STRESS_MARKERS = ['DDIT3', 'ATF4', 'XBP1', 'HSPA5']

# Frozen modality / context (see pipeline_config.py). Dixit and Papalexi are CRISPR-KO.
try:
    from pipeline_config import DATASET_CONTEXT, LEGACY_NAME_MAP, MODALITY_MAP
except ImportError:
    MODALITY_MAP = {
        'Norman 2019 (CRISPRa)': 'CRISPRa',
        'Adamson 2016 (CRISPRi)': 'CRISPRi',
        'Adamson 2016 pilot (CRISPRi)': 'CRISPRi',
        'Adamson 2016 UPR (CRISPRi)': 'CRISPRi',
        'Dixit 2016 (CRISPR-KO)': 'CRISPR-KO',
        'Dixit 2016 (CRISPRi)': 'CRISPR-KO',  # legacy CSV key
        'Papalexi 2021 (CRISPR-KO)': 'CRISPR-KO',
        'Papalexi 2021 (CRISPR)': 'CRISPR-KO',  # legacy CSV key
        'Replogle 2022 (CRISPRi)': 'CRISPRi',
    }
    DATASET_CONTEXT = {
        'Norman 2019 (CRISPRa)': {'cell_type': 'K562', 'design': 'CRISPRa activation, paired combinatorial'},
        'Adamson 2016 (CRISPRi)': {'cell_type': 'K562', 'design': 'CRISPRi pilot (legacy name)'},
        'Dixit 2016 (CRISPR-KO)': {'cell_type': 'BMDC', 'design': 'CRISPR-Cas9 KO Perturb-seq (NOT CRISPRi)'},
        'Papalexi 2021 (CRISPR-KO)': {'cell_type': 'THP-1', 'design': 'ECCITE-seq; includes KO'},
        'Replogle 2022 (CRISPRi)': {'cell_type': 'K562', 'design': 'CRISPRi, genome-scale essential genes'},
    }
    LEGACY_NAME_MAP = {}


# =============================================================================
# BOOTSTRAP PARTIAL CORRELATION
# =============================================================================

def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=10000, ci_level=0.95, seed=42):
    """Rank-based partial Spearman (canonical); see stats_utils.py."""
    from stats_utils import bootstrap_partial_spearman_ci

    return bootstrap_partial_spearman_ci(
        x, y, z, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed, method="rank"
    )


def bootstrap_spearman_ci(x, y, n_bootstrap=10000, ci_level=0.95, seed=42):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed=seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    alpha = 1 - ci_level
    return {
        'rho': rho, 'p': p,
        'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
    }


# =============================================================================
# TEST 1: PARTIAL CORRELATIONS (stability ~ stress | magnitude)
# =============================================================================

def run_partial_correlations(df):
    print("=" * 80)
    print("TEST 1: PARTIAL CORRELATIONS  --  stability ~ stress | magnitude")
    print("=" * 80)
    print("If rho_partial remains significant after controlling for magnitude,")
    print("the stress relationship is not an artifact of effect size.\n")

    results = []
    seed_counter = 8000

    for marker in STRESS_MARKERS:
        col = f'stress_{marker}'
        if col not in df.columns:
            continue
        for ds_name in sorted(df['dataset'].unique()):
            subset = df[(df['dataset'] == ds_name) & df[col].notna()].copy()
            if len(subset) < 15:
                continue

            # Raw correlation for comparison
            raw = bootstrap_spearman_ci(
                subset['stability'].values, subset[col].values,
                n_bootstrap=N_BOOTSTRAP, seed=seed_counter,
            )
            seed_counter += 1

            # Partial correlation controlling for magnitude
            partial = bootstrap_partial_correlation_ci(
                subset['stability'].values,
                subset[col].values,
                subset['magnitude'].values,
                n_bootstrap=N_BOOTSTRAP, seed=seed_counter,
            )
            seed_counter += 1

            modality = MODALITY_MAP.get(ds_name, '?')
            n = len(subset)
            sig_raw = '*' if raw['p'] < 0.05 else ''
            sig_par = '*' if partial['p'] < 0.05 else ''

            # Effect-size bin only. NEVER plot abs_rho_partial: the old
            # Fig 5c / S9 forest did, which flipped every negative rho
            # and left the CIs signed (Dixit HSPA5 bar +0.33, CI [−0.51, 0.84]).
            abs_rho = abs(partial['rho_partial'])
            if abs_rho >= 0.3:
                effect_size = 'medium-large'
            elif abs_rho >= 0.2:
                effect_size = 'small-medium'
            elif abs_rho >= 0.1:
                effect_size = 'small'
            else:
                effect_size = 'negligible'

            ci_excludes_zero = (
                not np.isnan(partial['ci_low'])
                and np.sign(partial['ci_low']) == np.sign(partial['ci_high'])
            )

            # "Survives" requires both meaningful effect size AND CI excluding zero.
            # For large-n datasets (e.g. Replogle n>1000), CI alone is insufficient
            # because even rho=0.05 will be "significant".
            survives = abs_rho > 0.1 and ci_excludes_zero

            print(f"{ds_name} ({modality}) | {marker}  [n={n}]:")
            print(f"  Raw:     rho = {raw['rho']:+.3f} [{raw['ci_low']:.3f}, {raw['ci_high']:.3f}] p={raw['p']:.2e} {sig_raw}")
            print(f"  Partial: rho = {partial['rho_partial']:+.3f} [{partial['ci_low']:.3f}, {partial['ci_high']:.3f}] p={partial['p']:.2e} {sig_par}")
            print(f"  Effect size: {effect_size} (|rho|={abs_rho:.3f}), CI excludes zero: {ci_excludes_zero}")
            if n > 500 and abs_rho < 0.1 and ci_excludes_zero:
                print(f"  NOTE: Large n ({n}) inflates significance; effect is negligible")
            print(f"  Survives magnitude control: {'YES' if survives else 'no'}\n")

            results.append({
                'dataset': ds_name,
                'modality': modality,
                'marker': marker,
                'n': n,
                'rho_raw': raw['rho'],
                'rho_raw_ci_low': raw['ci_low'],
                'rho_raw_ci_high': raw['ci_high'],
                'p_raw': raw['p'],
                'rho_partial': partial['rho_partial'],
                'rho_partial_ci_low': partial['ci_low'],
                'rho_partial_ci_high': partial['ci_high'],
                'p_partial': partial['p'],
                'abs_rho_partial': abs_rho,
                'effect_size': effect_size,
                'ci_excludes_zero': ci_excludes_zero,
                'survives_magnitude_control': survives,
            })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "stress_partial_correlations.csv", index=False)
    print(f"Saved -> stress_partial_correlations.csv  ({len(out)} rows)\n")
    return out


# =============================================================================
# TEST 2: QUADRANT TEST (Fisher's exact)
# =============================================================================

def _magnitude_residual_sp(subset: pd.DataFrame) -> pd.Series:
    """Rank-OLS residual of Sp on magnitude (same scale as partial Spearman path)."""
    from scipy.stats import rankdata

    sp = subset["stability"].to_numpy(dtype=float)
    mag = subset["magnitude"].to_numpy(dtype=float)
    mask = np.isfinite(sp) & np.isfinite(mag)
    resid = np.full(len(subset), np.nan)
    if mask.sum() < 5:
        return pd.Series(resid, index=subset.index)
    rsp, rmag = rankdata(sp[mask]), rankdata(mag[mask])
    Z = np.column_stack([np.ones(mask.sum()), rmag])
    e = rsp - Z @ np.linalg.lstsq(Z, rsp, rcond=None)[0]
    resid[np.where(mask)[0]] = e
    return pd.Series(resid, index=subset.index)


def run_quadrant_tests(df, on_residuals: bool = True):
    """
    Optional / SI only. Median–median tables are degenerate on raw Sp.
    Default uses magnitude-residual Sp so the quadrant cannot contradict the partial.
    """
    print("=" * 80)
    print("TEST 2: QUADRANT TEST (optional; default = magnitude-residual Sp)")
    print("=" * 80)
    if on_residuals:
        print("Using Sp residual after rank-regression on magnitude.\n")
    else:
        print("WARNING: raw Sp quadrants are degenerate and can contradict partials.\n")

    results = []

    for marker in STRESS_MARKERS:
        col = f'stress_{marker}'
        if col not in df.columns:
            continue
        for ds_name in sorted(df['dataset'].unique()):
            subset = df[(df['dataset'] == ds_name) & df[col].notna()].copy()
            if len(subset) < 20:
                continue

            if on_residuals:
                if "magnitude" not in subset.columns:
                    print(f"{ds_name} | {marker}: no magnitude — skip residual quadrant")
                    continue
                stab = _magnitude_residual_sp(subset)
            else:
                stab = subset["stability"]

            med_stab = stab.median()
            med_stress = subset[col].median()

            hi_stab = stab >= med_stab
            hi_stress = subset[col] >= med_stress

            q_hh = int((hi_stab & hi_stress).sum())
            q_hl = int((hi_stab & ~hi_stress).sum())
            q_lh = int((~hi_stab & hi_stress).sum())
            q_ll = int((~hi_stab & ~hi_stress).sum())
            n_total = len(subset)

            p_hi_stab = hi_stab.sum() / n_total
            p_hi_stress = hi_stress.sum() / n_total
            p_expected = p_hi_stab * p_hi_stress
            expected_hh = p_expected * n_total

            binom_p = binom_test_compat(q_hh, n_total, p_expected, alternative='less')

            table = np.array([[q_hh, q_hl],
                              [q_lh, q_ll]])
            odds_ratio, fisher_p = fisher_exact(table, alternative='two-sided')

            frac_hh = q_hh / n_total
            modality = MODALITY_MAP.get(ds_name, '?')
            depleted = q_hh < expected_hh and binom_p < 0.10

            print(f"{ds_name} ({modality}) | {marker}:")
            print(f"  Quadrants: HH={q_hh}  HL={q_hl}  LH={q_lh}  LL={q_ll}  (total={n_total})")
            print(f"  Binomial (HH < expected): p={binom_p:.4f}  Fisher OR={odds_ratio:.2f}")
            print(f"  HH depleted: {'YES' if depleted else 'no'}\n")

            results.append({
                'dataset': ds_name,
                'modality': modality,
                'marker': marker,
                'n': n_total,
                'stability_axis': 'mag_residual' if on_residuals else 'raw',
                'median_stability': float(med_stab) if np.isfinite(med_stab) else np.nan,
                'median_stress': med_stress,
                'q_high_stab_high_stress': q_hh,
                'q_high_stab_low_stress': q_hl,
                'q_low_stab_high_stress': q_lh,
                'q_low_stab_low_stress': q_ll,
                'p_hi_stab': p_hi_stab,
                'p_hi_stress': p_hi_stress,
                'expected_hh': expected_hh,
                'frac_high_high': frac_hh,
                'binom_p_less': binom_p,
                'fisher_odds_ratio': odds_ratio,
                'fisher_p_twosided': fisher_p,
                'hh_depleted': depleted,
            })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "stress_quadrant_tests.csv", index=False)
    print(f"Saved -> stress_quadrant_tests.csv  ({len(out)} rows)\n")
    return out


# =============================================================================
# TEST 3: MODALITY SUMMARY (not a CRISPRa-vs-i explanation)
# =============================================================================

def run_modality_analysis(df, partial_df):
    print("=" * 80)
    print("TEST 3: MODALITY SUMMARY")
    print("=" * 80)
    print("DDIT3 partials are negative across scoreable datasets (CRISPRi / a / KO).")
    print("Do NOT frame DDIT3 as a CRISPRa-vs-CRISPRi sign flip — that was a raw-")
    print("correlation story. ATF4/XBP1 Norman↔Replogle flips → gene × context.\n")
    print("Denominator: six frozen datasets (n=2,285); five scoreable with n≥15")
    print("(n=2,277). Adamson pilot (n=8) excluded from partials.\n")

    if partial_df is None or partial_df.empty:
        print("No partial correlation results to analyze.")
        return None

    ddit3 = partial_df[partial_df['marker'] == 'DDIT3'].copy()
    if ddit3.empty:
        print("No DDIT3 results found.")
        return None

    n_neg = int((ddit3["rho_partial"] < 0).sum())
    n_surv = int(ddit3["survives_magnitude_control"].sum()) if "survives_magnitude_control" in ddit3 else 0
    print(f"DDIT3 overall: {n_neg}/{len(ddit3)} negative partial; {n_surv} survive |magnitude\n")

    # --- Per-modality summary ---
    rows = []
    for modality in ['CRISPRa', 'CRISPRi', 'CRISPR-KO', 'CRISPR']:
        mod_rows = ddit3[ddit3['modality'] == modality]
        if mod_rows.empty:
            continue

        mean_raw = mod_rows['rho_raw'].mean()
        mean_partial = mod_rows['rho_partial'].mean()
        datasets = ', '.join(mod_rows['dataset'].tolist())
        n_datasets = len(mod_rows)
        # Consistent sign on PARTIALS (not raw); N/A if only one dataset
        if n_datasets < 2:
            sign_label = "N/A (single dataset)"
            all_same_sign = None
        else:
            all_same_sign = len(mod_rows['rho_partial'].apply(np.sign).unique()) == 1
            sign_label = "YES" if all_same_sign else "NO"

        n_crispri_expected = (
            " (3 CRISPRi in freeze; pilot n=8 excluded)" if modality == "CRISPRi" else ""
        )
        print(f"  {modality} ({n_datasets} scoreable dataset{'s' if n_datasets > 1 else ''})"
              f"{n_crispri_expected}:")
        print(f"    Mean raw rho:     {mean_raw:+.3f}")
        print(f"    Mean partial rho: {mean_partial:+.3f}")
        print(f"    Consistent PARTIAL sign: {sign_label}")

        for _, row in mod_rows.iterrows():
            ctx = DATASET_CONTEXT.get(row['dataset'], {})
            cell = ctx.get('cell_type', '?')
            design = ctx.get('design', '?')
            print(f"      {row['dataset']}  [{cell}]")
            print(f"        raw={row['rho_raw']:+.3f}, partial={row['rho_partial']:+.3f}")
            print(f"        design: {design}")

        if all_same_sign is False and n_datasets > 1:
            cell_types = [DATASET_CONTEXT.get(d, {}).get('cell_type', '?')
                          for d in mod_rows['dataset']]
            print(f"\n    ** WITHIN-{modality} PARTIAL DISCORDANCE **")
            print(f"    Cell types: {', '.join(sorted(set(cell_types)))}")
        print()

        rows.append({
            'modality': modality,
            'n_datasets': n_datasets,
            'datasets': datasets,
            'mean_rho_raw': mean_raw,
            'mean_rho_partial': mean_partial,
            'consistent_partial_sign': sign_label,
        })

    # --- All markers by modality with context ---
    print("\n--- All markers by modality ---")
    detail_rows = []
    for modality in ['CRISPRa', 'CRISPRi', 'CRISPR-KO', 'CRISPR']:
        mod_rows = partial_df[partial_df['modality'] == modality]
        if mod_rows.empty:
            continue
        print(f"\n  {modality}:")
        for marker in STRESS_MARKERS:
            m_rows = mod_rows[mod_rows['marker'] == marker]
            if m_rows.empty:
                continue
            for _, r in m_rows.iterrows():
                ctx = DATASET_CONTEXT.get(r['dataset'], {})
                cell = ctx.get('cell_type', '?')
                tag = '*' if r['survives_magnitude_control'] else ''
                eff = r.get('effect_size', '')
                print(f"    {r['dataset']} [{cell}] | {marker}: "
                      f"raw={r['rho_raw']:+.3f}, partial={r['rho_partial']:+.3f} "
                      f"({eff}) {tag}")
                detail_rows.append({
                    'modality': modality,
                    'dataset': r['dataset'],
                    'cell_type': cell,
                    'marker': marker,
                    'rho_raw': r['rho_raw'],
                    'rho_partial': r['rho_partial'],
                    'effect_size': eff,
                    'survives': r['survives_magnitude_control'],
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "stress_modality_summary.csv", index=False)

    detail_out = pd.DataFrame(detail_rows)
    detail_out.to_csv(OUTPUT_DIR / "stress_modality_detail.csv", index=False)

    print(f"\nSaved -> stress_modality_summary.csv  ({len(out)} rows)")
    print(f"Saved -> stress_modality_detail.csv   ({len(detail_out)} rows)\n")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Results CSV with stability, magnitude, and stress_* columns",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-detected shesha-crispr/)",
    )
    parser.add_argument(
        "--run-quadrant",
        action="store_true",
        help="Run optional quadrant test on magnitude-residual Sp (off by default)",
    )
    parser.add_argument(
        "--quadrant-raw",
        action="store_true",
        help="With --run-quadrant, use raw Sp (discouraged; degenerate)",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.out_dir is not None:
        OUTPUT_DIR = Path(args.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = []
    if args.input is not None:
        candidates.append(Path(args.input))
    candidates.extend(
        [
            OUTPUT_DIR / "shesha_crispr_results_euclidean.csv",
            OUTPUT_DIR / "frozen_sp_scores.csv",
            Path("/content/shesha-crispr/shesha_crispr_results_euclidean.csv"),
            Path("/content/shesha-crispr/frozen_sp_scores.csv"),
        ]
    )

    input_path = next((p for p in candidates if p.exists()), None)
    if input_path is None:
        print("ERROR: no results CSV found. Tried:")
        for p in candidates:
            print(f"  - {p}")
        print("Run attach_stress_markers.py on frozen_sp_scores.csv first.")
        return

    print(f"Loading results from: {input_path}\n")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {df['dataset'].nunique()} datasets\n")

    # Check which stress columns exist
    available = [m for m in STRESS_MARKERS if f'stress_{m}' in df.columns]
    print(f"Stress markers available: {available}\n")

    if not available:
        print(
            "No stress_* columns in this CSV.\n"
            "frozen_sp_scores.csv from run_frozen_main.py only has Sp/magnitude.\n"
            "Run: python attach_stress_markers.py --input shesha-crispr/frozen_sp_scores.csv"
        )
        return

    partial_df = run_partial_correlations(df)
    if args.run_quadrant:
        run_quadrant_tests(df, on_residuals=not args.quadrant_raw)
    else:
        print("TEST 2 (quadrant): skipped (default). Pass --run-quadrant for residual Sp.\n")
    run_modality_analysis(df, partial_df)

    print("=" * 80)
    print("ALL STRESS MARKER TESTS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  - stress_partial_correlations.csv")
    if args.run_quadrant:
        print("  - stress_quadrant_tests.csv")
    print("  - stress_modality_summary.csv")
    print("  - stress_modality_detail.csv")
    print("\nBefore writing DDIT3/p53/apoptosis claims, run cell_quality_partial.py.")


if __name__ == "__main__":
    main()
