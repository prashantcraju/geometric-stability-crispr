#!/usr/bin/env python3
"""
Stress Marker Deep-Dive Analysis

Runs three targeted tests on the stress marker results from the main analysis:

1. PARTIAL CORRELATIONS: stability vs stress markers controlling for magnitude.
   If the correlation survives after partialling out magnitude, the relationship
   is not an artifact of larger perturbations driving both stability and expression.

2. QUADRANT TEST (Fisher's exact): Split perturbations by median stability and
   median stress into four quadrants. Test whether the high-stability / high-stress
   quadrant is systematically depleted. This formalizes the qualitative observation
   from the perspective paper that no perturbations occupy that corner.

3. CRISPRa vs CRISPRi GROUPING: Annotate each dataset by modality and test whether
   the DDIT3 sign flip (positive in CRISPRa, negative in CRISPRi) is consistent
   with the activation-vs-interference mechanism.

INPUT:  shesha_crispr_results_euclidean.csv (from main analysis)
OUTPUT: stress_partial_correlations.csv
        stress_quadrant_tests.csv
        stress_modality_summary.csv
        (all saved to same OUTPUT_DIR as main analysis)
"""

import os

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
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 320
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95

OUTPUT_DIR = Path("./shesha-crispr")

STRESS_MARKERS = ['DDIT3', 'ATF4', 'XBP1', 'HSPA5']

MODALITY_MAP = {
    'Norman 2019 (CRISPRa)':   'CRISPRa',
    'Adamson 2016 (CRISPRi)':  'CRISPRi',
    'Dixit 2016 (CRISPRi)':    'CRISPRi',
    'Papalexi 2021 (CRISPR)':  'CRISPR',
    'Replogle 2022 (CRISPRi)': 'CRISPRi',
}

# Cell type and experimental design context for interpreting within-modality discordance
DATASET_CONTEXT = {
    'Norman 2019 (CRISPRa)':   {'cell_type': 'K562 (CML)',   'design': 'CRISPRa activation, paired combinatorial'},
    'Adamson 2016 (CRISPRi)':  {'cell_type': 'K562 (CML)',   'design': 'CRISPRi, UPR-focused panel'},
    'Dixit 2016 (CRISPRi)':    {'cell_type': 'BMDCs',        'design': 'CRISPRi Perturb-seq, immune TFs'},
    'Papalexi 2021 (CRISPR)':  {'cell_type': 'THP-1',        'design': 'ECCITE-seq, multi-modal'},
    'Replogle 2022 (CRISPRi)': {'cell_type': 'K562 (CML)',   'design': 'CRISPRi, genome-scale essential genes'},
}


# =============================================================================
# BOOTSTRAP PARTIAL CORRELATION
# =============================================================================

def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=10000, ci_level=0.95, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = len(x)

    def _partial(x, y, z):
        Z_aug = sm.add_constant(z)
        x_resid = sm.OLS(x, Z_aug).fit().resid
        y_resid = sm.OLS(y, Z_aug).fit().resid
        return spearmanr(x_resid, y_resid)

    rho_partial, p = _partial(x, y, z)

    rng = np.random.default_rng(seed=seed)
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = _partial(x[idx], y[idx], z[idx])
        boot[i] = r

    valid = boot[~np.isnan(boot)]
    if len(valid) < 10:
        return {'rho_partial': rho_partial, 'ci_low': np.nan, 'ci_high': np.nan, 'p': p}

    alpha = 1 - ci_level
    return {
        'rho_partial': rho_partial,
        'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
        'p': p,
    }


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

            # Effect size classification
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

def run_quadrant_tests(df):
    print("=" * 80)
    print("TEST 2: QUADRANT DEPLETION TEST  --  high-stability / high-stress")
    print("=" * 80)
    print("Split perturbations at median stability & median stress.")
    print("Primary test: one-sided binomial -- is the observed HH count less")
    print("than expected under independence?  Fisher's exact included for reference.\n")

    results = []

    for marker in STRESS_MARKERS:
        col = f'stress_{marker}'
        if col not in df.columns:
            continue
        for ds_name in sorted(df['dataset'].unique()):
            subset = df[(df['dataset'] == ds_name) & df[col].notna()].copy()
            if len(subset) < 20:
                continue

            med_stab = subset['stability'].median()
            med_stress = subset[col].median()

            hi_stab = subset['stability'] >= med_stab
            hi_stress = subset[col] >= med_stress

            q_hh = int((hi_stab & hi_stress).sum())
            q_hl = int((hi_stab & ~hi_stress).sum())
            q_lh = int((~hi_stab & hi_stress).sum())
            q_ll = int((~hi_stab & ~hi_stress).sum())
            n_total = len(subset)

            # Expected HH count under independence:
            # P(hi_stab) * P(hi_stress) * n
            p_hi_stab = hi_stab.sum() / n_total
            p_hi_stress = hi_stress.sum() / n_total
            p_expected = p_hi_stab * p_hi_stress
            expected_hh = p_expected * n_total

            # One-sided binomial: is HH count significantly *less* than expected?
            binom_p = binom_test_compat(q_hh, n_total, p_expected, alternative='less')

            # Fisher's exact (two-sided) for reference
            table = np.array([[q_hh, q_hl],
                              [q_lh, q_ll]])
            odds_ratio, fisher_p = fisher_exact(table, alternative='two-sided')

            frac_hh = q_hh / n_total
            modality = MODALITY_MAP.get(ds_name, '?')
            depleted = q_hh < expected_hh and binom_p < 0.10

            print(f"{ds_name} ({modality}) | {marker}:")
            print(f"  Quadrants: HH={q_hh}  HL={q_hl}  LH={q_lh}  LL={q_ll}  (total={n_total})")
            print(f"  P(hi_stab)={p_hi_stab:.2f}, P(hi_stress)={p_hi_stress:.2f}")
            print(f"  Expected HH under independence: {expected_hh:.1f}, Observed: {q_hh}")
            print(f"  Binomial (one-sided, HH < expected): p={binom_p:.4f}")
            print(f"  Fisher OR={odds_ratio:.2f}, p={fisher_p:.3f} (two-sided, for reference)")
            print(f"  HH quadrant depleted: {'YES' if depleted else 'no'}\n")

            results.append({
                'dataset': ds_name,
                'modality': modality,
                'marker': marker,
                'n': n_total,
                'median_stability': med_stab,
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
# TEST 3: CRISPRa vs CRISPRi MODALITY ANALYSIS
# =============================================================================

def run_modality_analysis(df, partial_df):
    print("=" * 80)
    print("TEST 3: CRISPRa vs CRISPRi MODALITY ANALYSIS")
    print("=" * 80)
    print("The DDIT3 sign flip may partly reflect CRISPRa-vs-CRISPRi biology:")
    print("activation coherently upregulates stress genes, while interference")
    print("inverts the relationship.  However, within CRISPRi the sign also flips")
    print("(Dixit vs Replogle), so cell type and experimental design also matter.\n")

    if partial_df is None or partial_df.empty:
        print("No partial correlation results to analyze.")
        return None

    ddit3 = partial_df[partial_df['marker'] == 'DDIT3'].copy()
    if ddit3.empty:
        print("No DDIT3 results found.")
        return None

    # --- Per-modality summary ---
    rows = []
    for modality in ['CRISPRa', 'CRISPRi', 'CRISPR']:
        mod_rows = ddit3[ddit3['modality'] == modality]
        if mod_rows.empty:
            continue

        mean_raw = mod_rows['rho_raw'].mean()
        mean_partial = mod_rows['rho_partial'].mean()
        datasets = ', '.join(mod_rows['dataset'].tolist())
        n_datasets = len(mod_rows)
        all_same_sign = len(mod_rows['rho_raw'].apply(np.sign).unique()) == 1

        print(f"  {modality} ({n_datasets} dataset{'s' if n_datasets > 1 else ''}):")
        print(f"    Mean raw rho:     {mean_raw:+.3f}")
        print(f"    Mean partial rho: {mean_partial:+.3f}")
        print(f"    Consistent sign across datasets: {'YES' if all_same_sign else 'NO'}")

        for _, row in mod_rows.iterrows():
            ctx = DATASET_CONTEXT.get(row['dataset'], {})
            cell = ctx.get('cell_type', '?')
            design = ctx.get('design', '?')
            print(f"      {row['dataset']}  [{cell}]")
            print(f"        raw={row['rho_raw']:+.3f}, partial={row['rho_partial']:+.3f}")
            print(f"        design: {design}")

        # Flag within-modality discordance
        if not all_same_sign and n_datasets > 1:
            cell_types = [DATASET_CONTEXT.get(d, {}).get('cell_type', '?')
                          for d in mod_rows['dataset']]
            same_cell = len(set(cell_types)) == 1
            print(f"\n    ** WITHIN-{modality} DISCORDANCE **")
            if not same_cell:
                print(f"    Cell types differ ({', '.join(set(cell_types))}), which likely")
                print(f"    explains the sign flip within this modality.")
            else:
                print(f"    Same cell type ({cell_types[0]}) -- sign flip may reflect")
                print(f"    differences in perturbation panel or guide library design.")
        print()

        rows.append({
            'modality': modality,
            'n_datasets': n_datasets,
            'datasets': datasets,
            'mean_rho_raw': mean_raw,
            'mean_rho_partial': mean_partial,
            'consistent_sign': all_same_sign,
        })

    # --- All markers by modality with context ---
    print("\n--- All markers by modality ---")
    detail_rows = []
    for modality in ['CRISPRa', 'CRISPRi', 'CRISPR']:
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
    input_path = OUTPUT_DIR / "shesha_crispr_results_euclidean.csv"
    print(f"Loading results from: {input_path}\n")

    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print("Run the main analysis first to generate this file.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {df['dataset'].nunique()} datasets\n")

    # Check which stress columns exist
    available = [m for m in STRESS_MARKERS if f'stress_{m}' in df.columns]
    print(f"Stress markers available: {available}\n")

    if not available:
        print("No stress marker columns found. Nothing to analyze.")
        return

    partial_df = run_partial_correlations(df)
    quadrant_df = run_quadrant_tests(df)
    modality_df = run_modality_analysis(df, partial_df)

    print("=" * 80)
    print("ALL STRESS MARKER TESTS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  - stress_partial_correlations.csv")
    print("  - stress_quadrant_tests.csv")
    print("  - stress_modality_summary.csv")
    print("  - stress_modality_detail.csv")


if __name__ == "__main__":
    main()
