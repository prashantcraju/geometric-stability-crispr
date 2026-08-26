#!/usr/bin/env python3
"""
Adamson UPR positive-control magnitude partial.

Raw claim: UPR-core genes have lower Sp than other
Adamson UPR perturbations (median 0.135 vs 0.263, MWU p=0.024). That is a
structural, non-correlational finding.

This script asks whether that claim is magnitude-independent:
  does the UPR-core Sp deficit survive conditioning on mean-shift magnitude?

If yes  → one magnitude-independent result; paper is a descriptive statistic
          with a validated positive control (publishable and honest).
If no   → honest conclusion is that Sp is largely redundant with effect size
          in CRISPR data (still publishable as a negative methods result,
          but a different and smaller paper).

Do this before retitling or restructuring Results.

Primary tests (all reported; fork uses the conditioned tests):
  1. Raw Sp:            MWU core < other
  2. Magnitude:         MWU core vs other (direction reported both ways)
  3. Partial Spearman:  Sp ~ is_upr_core | magnitude  (rank-based; manuscript)
  4. Mag-residual Sp:   MWU on Sp after linear residualization on magnitude
  5. LOESS discordance: MWU core more discordant than expected for magnitude

Usage:
  python adamson_upr_magnitude_partial.py
  python adamson_upr_magnitude_partial.py --input shesha-crispr/adamson_upr_sp_scores.csv
  python adamson_upr_magnitude_partial.py --rescore   # re-run frozen pipeline
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
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LinearRegression

import pipeline_config as cfg
from revision_io import resolve_out_dir
from stats_utils import (
    bootstrap_partial_spearman_ci,
    partial_spearman_rank,
    survival_status,
)

# Pinned in pipeline_config — do not require adamson_upr_spike.py on Colab.
UPR_CORE = set(cfg.UPR_CORE_GENES)
UPR_CORE_CANONICAL = set(cfg.UPR_CORE_CANONICAL)
UPR_CORE_ALIASES = dict(cfg.UPR_CORE_ALIASES)


def _gene_token(name: str) -> str:
    s = str(name).upper().replace("-", "_")
    parts = s.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return parts[0]


def _canonical_upr_gene(token: str) -> str:
    return UPR_CORE_ALIASES.get(token, token)


def annotate_upr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gene"] = out["perturbation"].map(_gene_token)
    out["gene_canonical"] = out["gene"].map(_canonical_upr_gene)
    out["is_upr_core"] = out["gene"].isin(UPR_CORE) | out["gene_canonical"].isin(
        UPR_CORE_CANONICAL
    )
    return out

DEFAULT_LOESS_FRAC = 0.3
ALPHA = 0.05

# Previously cited medians (for mismatch warning)
PLAN_MEDIAN_CORE = 0.135
PLAN_MEDIAN_OTHER = 0.263
PLAN_MWU_P = 0.024


def _mwu(a: pd.Series, b: pd.Series, alternative: str) -> dict:
    a, b = a.dropna(), b.dropna()
    out = {
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
    if len(a) < 3 or len(b) < 5:
        out["note"] = "insufficient n"
        return out
    u, p = mannwhitneyu(a, b, alternative=alternative)
    out["U"] = float(u)
    out["p"] = float(p)
    return out


def _lowess_fitted(y: np.ndarray, x: np.ndarray, frac: float) -> np.ndarray:
    """Prefer statsmodels LOWESS; fall back to tricube local linear."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        return np.asarray(lowess(y, x, frac=frac, return_sorted=False), dtype=float)
    except Exception:
        pass

    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    k = max(2, int(np.ceil(frac * n)))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    fitted_sorted = np.empty(n, dtype=float)
    for i in range(n):
        dist = np.abs(xs - xs[i])
        nn = np.argpartition(dist, k - 1)[:k]
        dmax = dist[nn].max()
        if dmax < 1e-15:
            fitted_sorted[i] = ys[nn].mean()
            continue
        u = dist[nn] / dmax
        w = (1 - u**3) ** 3
        w = np.clip(w, 0.0, None)
        sw = w.sum()
        if sw < 1e-15:
            fitted_sorted[i] = ys[nn].mean()
            continue
        xbar = np.sum(w * xs[nn]) / sw
        ybar = np.sum(w * ys[nn]) / sw
        varx = np.sum(w * (xs[nn] - xbar) ** 2)
        if varx < 1e-15:
            fitted_sorted[i] = ybar
        else:
            slope = np.sum(w * (xs[nn] - xbar) * (ys[nn] - ybar)) / varx
            fitted_sorted[i] = ybar + slope * (xs[i] - xbar)
    out = np.empty(n, dtype=float)
    out[order] = fitted_sorted
    return out


def disc_loess(
    mag: np.ndarray, stab: np.ndarray, frac: float = DEFAULT_LOESS_FRAC
) -> np.ndarray:
    """Sign-flipped, z-scored LOESS residual (below Sp~mag curve = high discordance)."""
    mag = np.asarray(mag, dtype=float)
    stab = np.asarray(stab, dtype=float)
    fitted = _lowess_fitted(stab, mag, frac=frac)
    d = -(stab - fitted)
    sd = float(d.std())
    if sd < 1e-12:
        return np.zeros_like(d)
    return (d - d.mean()) / sd


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "stability" not in df.columns or "magnitude" not in df.columns:
        raise SystemExit(f"{path} missing stability/magnitude columns")
    if "perturbation" not in df.columns:
        raise SystemExit(f"{path} missing perturbation column")
    df = annotate_upr(df)
    return df.dropna(subset=["stability", "magnitude"]).copy()


def decide_fork(results: dict) -> dict:
    """
    Fork on magnitude-conditioned evidence, not the raw MWU alone.

    Same manuscript gate as pathway rows (`ci_and_fdr.v1`): CI excludes 0,
    |ρ| > SURVIVAL_ABS_RHO_MIN, and knife-edge demotion when the nearer |CI
    bound| is strictly < SURVIVAL_KNIFE_EDGE_ABS. FDR is N/A for this single
    pre-specified contrast (passed as None). Residual/LOESS are supporting
    only — they must not promote past an indeterminate / null primary.
    """
    raw = results["raw_sp_core_lower"]
    part = results["partial_spearman_sp_vs_core_given_magnitude"]
    boot = results.get("partial_boot") or {}
    resid = results["mag_residual_sp_core_lower"]
    loess = results["loess_discordance_core_greater"]

    raw_dir = (
        np.isfinite(raw.get("median_a", np.nan))
        and np.isfinite(raw.get("median_b", np.nan))
        and (
            raw["median_a"] < raw["median_b"]
            or (
                np.isfinite(raw.get("mean_a", np.nan))
                and raw["mean_a"] < raw["mean_b"]
            )
        )
    )
    raw_sig = bool(np.isfinite(raw.get("p", np.nan)) and raw["p"] < ALPHA)

    part_rho = part.get("rho_partial", np.nan)
    part_p = part.get("p", np.nan)
    ci_lo = boot.get("ci_low", np.nan)
    ci_hi = boot.get("ci_high", np.nan)
    ci_finite = np.isfinite(ci_lo) and np.isfinite(ci_hi)
    ci_excludes_zero_neg = bool(ci_finite and ci_hi < 0)
    ci_includes_zero = bool(ci_finite and ci_lo <= 0 <= ci_hi)

    part_point_ok = bool(
        np.isfinite(part_rho) and np.isfinite(part_p) and part_rho < 0 and part_p < ALPHA
    )
    # Gate-identical primary (no multi-test FDR on this single contrast)
    gate = survival_status(
        part_rho if np.isfinite(part_rho) else np.nan,
        ci_lo if np.isfinite(ci_lo) else np.nan,
        ci_hi if np.isfinite(ci_hi) else np.nan,
        fdr=None,
    )
    # Require negative direction for a positive-control claim
    part_ok = bool(gate["survives"] and part_rho < 0 and ci_excludes_zero_neg)
    part_indeterminate = bool(
        gate["status"] == "indeterminate" and part_rho < 0 and ci_excludes_zero_neg
    )
    part_suggestive = bool(part_point_ok and ci_includes_zero)
    resid_ok = bool(np.isfinite(resid.get("p", np.nan)) and resid["p"] < ALPHA)
    loess_ok = bool(np.isfinite(loess.get("p", np.nan)) and loess["p"] < ALPHA)
    support_ok = resid_ok or loess_ok
    eps = float(cfg.SURVIVAL_KNIFE_EDGE_ABS)

    if part_ok and (raw_sig or raw_dir):
        signal = "survives_magnitude"
        note = (
            "Primary partial Sp~core|magnitude clears ci_and_fdr.v1 "
            f"(CI excludes 0, |ρ|>{cfg.SURVIVAL_ABS_RHO_MIN}, nearer |CI bound| "
            f"≥ {eps}). Magnitude-independent positive control is reportable "
            "only with matrix_is_log settled on .X."
        )
    elif part_indeterminate and (raw_sig or raw_dir):
        signal = "indeterminate_knife_edge"
        note = (
            "Partial Sp~core|magnitude CI excludes zero but the nearer |CI "
            f"bound| ({gate.get('ci_margin')}) is < ε={eps} — demoted to "
            "indeterminate under the same ci_and_fdr.v1 knife-edge used for "
            "pathway rows (cf. Norman p53). Do not claim the positive control "
            "survives magnitude conditioning"
            + (
                f" (supporting residual/LOESS fire in the same direction: "
                f"resid_p={resid.get('p')}, loess_p={loess.get('p')})"
                if support_ok
                else ""
            )
            + "."
        )
    elif part_suggestive and (raw_sig or raw_dir):
        signal = "suggestive_ci_includes_zero"
        note = (
            "Partial Sp~core|magnitude is negative with p<α but the bootstrap "
            f"CI includes zero ([{ci_lo}, {ci_hi}]). Suggestive only — do not "
            "claim the positive control survives magnitude conditioning"
            + (
                f" (supporting residual/LOESS also fire: resid_p={resid.get('p')}, "
                f"loess_p={loess.get('p')})"
                if support_ok
                else ""
            )
            + "."
        )
    elif support_ok and (raw_sig or raw_dir) and not part_point_ok:
        signal = "support_tests_only"
        note = (
            "Residual/LOESS tests are significant but the primary rank partial "
            "is not. Do not lead with a magnitude-independent claim."
        )
    elif raw_sig:
        signal = "fails_magnitude"
        note = (
            "Raw UPR-core Sp deficit is significant but does not survive "
            "magnitude control on the primary partial. Honest conclusion: Sp "
            "is largely redundant with effect size on this test."
        )
    else:
        signal = "no_raw_signal"
        note = (
            "No significant raw UPR-core Sp deficit, and no clear "
            "magnitude-conditioned signal. Do not lead with Adamson as a "
            "validated positive control."
        )

    return {
        "fork_signal": signal,
        "fork_note": note,
        "raw_significant": raw_sig,
        "partial_point_significant_negative": part_point_ok,
        "partial_ci_excludes_zero_negative": ci_excludes_zero_neg,
        "partial_ci_includes_zero": ci_includes_zero,
        "partial_significant_negative": part_ok,
        "partial_indeterminate_knife_edge": part_indeterminate,
        "partial_suggestive_ci_includes_zero": part_suggestive,
        "partial_survival_status": gate,
        "survival_criterion_id": cfg.SURVIVAL_CRITERION_ID,
        "survival_knife_edge_abs": eps,
        "mag_residual_significant": resid_ok,
        "loess_discordance_significant": loess_ok,
        "any_conditioned_significant": part_ok,
        "support_tests_significant": support_ok,
        "alpha": ALPHA,
    }


def run_tests(
    df: pd.DataFrame, *, n_bootstrap: int, loess_frac: float = DEFAULT_LOESS_FRAC
) -> dict:
    core_m = df["is_upr_core"].astype(bool)
    core = df.loc[core_m]
    other = df.loc[~core_m]

    rho_mag, p_mag = spearmanr(df["magnitude"], df["stability"])

    results: dict = {
        "config_version": cfg.CONFIG_VERSION,
        "upr_core_set_id": cfg.UPR_CORE_SET_ID,
        "n_perturbations": int(len(df)),
        "n_upr_core": int(core_m.sum()),
        "upr_core_genes": sorted(core["gene"].unique().tolist()),
        "upr_core_genes_canonical_pinned": sorted(cfg.UPR_CORE_CANONICAL),
        "upr_core_aliases": dict(cfg.UPR_CORE_ALIASES),
        "upr_core_genes_pinned": sorted(cfg.UPR_CORE_CANONICAL),  # unique only
        "upr_core_set_size": int(cfg.UPR_CORE_N_UNIQUE),
        "upr_core_match_set_size_with_aliases": len(UPR_CORE),
        "spearman_magnitude_sp": float(rho_mag),
        "spearman_magnitude_sp_p": float(p_mag),
        "loess_frac": loess_frac,
    }

    # 1. Raw Sp
    results["raw_sp_core_lower"] = _mwu(core["stability"], other["stability"], "less")
    results["raw_sp_core_lower"]["label"] = "H1: UPR-core Sp < other"

    # 2. Magnitude both directions
    results["magnitude_core_greater"] = _mwu(
        core["magnitude"], other["magnitude"], "greater"
    )
    results["magnitude_core_greater"]["label"] = "H1: UPR-core magnitude > other"
    results["magnitude_core_less"] = _mwu(
        core["magnitude"], other["magnitude"], "less"
    )
    results["magnitude_core_less"]["label"] = "H1: UPR-core magnitude < other"

    # 3. Rank-based partial: Sp ~ binary core | magnitude
    y_core = core_m.astype(float).to_numpy()
    part = partial_spearman_rank(
        df["stability"].to_numpy(),
        y_core,
        df["magnitude"].to_numpy(),
    )
    results["partial_spearman_sp_vs_core_given_magnitude"] = part
    results["partial_boot"] = bootstrap_partial_spearman_ci(
        df["stability"].to_numpy(),
        y_core,
        df["magnitude"].to_numpy(),
        n_bootstrap=min(n_bootstrap, 2000),
        seed=cfg.SEED,
    )

    # 4. Linear mag-residual Sp (Norman combinatorial pattern)
    X = df[["magnitude"]].to_numpy()
    y = df["stability"].to_numpy()
    resid = y - LinearRegression().fit(X, y).predict(X)
    df = df.copy()
    df["sp_resid_mag"] = resid
    results["mag_residual_sp_core_lower"] = _mwu(
        df.loc[core_m, "sp_resid_mag"],
        df.loc[~core_m, "sp_resid_mag"],
        "less",
    )
    results["mag_residual_sp_core_lower"]["label"] = (
        "H1: UPR-core mag-residual Sp < other"
    )

    # 5. LOESS discordance (CORUM pattern)
    df["disc_loess"] = disc_loess(
        df["magnitude"].to_numpy(), df["stability"].to_numpy(), frac=loess_frac
    )
    results["loess_discordance_core_greater"] = _mwu(
        df.loc[core_m, "disc_loess"],
        df.loc[~core_m, "disc_loess"],
        "greater",
    )
    results["loess_discordance_core_greater"]["label"] = (
        "H1: UPR-core more LOESS-discordant than expected for magnitude"
    )

    # Plan-number mismatch warning
    med_c = results["raw_sp_core_lower"]["median_a"]
    med_o = results["raw_sp_core_lower"]["median_b"]
    p_raw = results["raw_sp_core_lower"]["p"]
    results["plan_claim"] = {
        "median_sp_upr_core": PLAN_MEDIAN_CORE,
        "median_sp_other": PLAN_MEDIAN_OTHER,
        "mwu_p": PLAN_MWU_P,
        "this_table_median_sp_upr_core": med_c,
        "this_table_median_sp_other": med_o,
        "this_table_mwu_p": p_raw,
        "matches_plan": bool(
            np.isfinite(med_c)
            and np.isfinite(med_o)
            and abs(med_c - PLAN_MEDIAN_CORE) < 0.02
            and abs(med_o - PLAN_MEDIAN_OTHER) < 0.02
            and np.isfinite(p_raw)
            and abs(p_raw - PLAN_MWU_P) < 0.02
        ),
    }

    results["upr_core_rows"] = (
        df.loc[
            core_m,
            ["perturbation", "gene", "stability", "magnitude", "sp_resid_mag", "disc_loess", "n_cells"]
            if "n_cells" in df.columns
            else ["perturbation", "gene", "stability", "magnitude", "sp_resid_mag", "disc_loess"],
        ]
        .sort_values("stability")
        .to_dict(orient="records")
    )

    results.update(decide_fork(results))
    results["annotated"] = df
    return results


def _blurb(results: dict) -> str:
    raw = results["raw_sp_core_lower"]
    part = results["partial_spearman_sp_vs_core_given_magnitude"]
    resid = results["mag_residual_sp_core_lower"]
    loess = results["loess_discordance_core_greater"]
    mag_g = results["magnitude_core_greater"]
    plan = results["plan_claim"]
    boot = results.get("partial_boot") or {}

    gate = results.get("partial_survival_status") or {}
    lines = [
        (
            f"Adamson UPR magnitude partial (config {results['config_version']}, "
            f"n={results['n_perturbations']}, UPR-core matched={results['n_upr_core']})."
        ),
        (
            f"UPR_CORE pinned ({results['upr_core_set_id']}, "
            f"n_unique={results['upr_core_set_size']}): "
            f"{', '.join(results['upr_core_genes_canonical_pinned'])}. "
            f"Aliases (label-match only): {results['upr_core_aliases']}."
        ),
        (
            f"Matched in this table (n={results['n_upr_core']}): "
            f"{', '.join(results['upr_core_genes'])}."
        ),
        (
            f"Raw Sp: core median={raw['median_a']:.3f} vs other={raw['median_b']:.3f} "
            f"(MWU less p={raw['p']:.3g})."
        ),
        (
            f"Magnitude: core median={mag_g['median_a']:.3f} vs other={mag_g['median_b']:.3f} "
            f"(MWU greater p={mag_g['p']:.3g}; less p={results['magnitude_core_less']['p']:.3g})."
        ),
        (
            f"Partial Spearman Sp~core|magnitude: rho={part.get('rho_partial')} "
            f"p={part.get('p')} "
            f"boot CI=[{boot.get('ci_low')}, {boot.get('ci_high')}] "
            f"({part.get('method')}); gate={gate.get('status')} "
            f"(criterion={results.get('survival_criterion_id')}, "
            f"ci_margin={gate.get('ci_margin')}, "
            f"ε={results.get('survival_knife_edge_abs')})."
        ),
        (
            f"Mag-residual Sp MWU less p={resid['p']:.3g}; "
            f"LOESS discordance MWU greater p={loess['p']:.3g} "
            f"(frac={results['loess_frac']})."
        ),
        f"Fork: {results['fork_signal']}. {results['fork_note']}",
    ]
    if not plan["matches_plan"]:
        lines.append(
            f"NOTE: this table does not match the plan claim "
            f"({PLAN_MEDIAN_CORE} vs {PLAN_MEDIAN_OTHER}, p={PLAN_MWU_P}); "
            f"got {plan['this_table_median_sp_upr_core']:.3f} vs "
            f"{plan['this_table_median_sp_other']:.3f}, "
            f"p={plan['this_table_mwu_p']}."
        )
    return " ".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Adamson Sp scores CSV (default: shesha-crispr/adamson_upr_sp_scores.csv)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-run frozen pipeline on Adamson UPR before testing",
    )
    parser.add_argument(
        "--force-normalize-log1p",
        action="store_true",
        help=(
            "With --rescore: force normalize_total + log1p (matrix_is_log=False). "
            "Diagnostic for plan p=0.024; writes *_forcenorm* outputs so the "
            "pinned skip-log path is not overwritten."
        ),
    )
    parser.add_argument(
        "--assume-log",
        action="store_true",
        help="With --rescore: skip normalize/log1p (matrix_is_log=True).",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=None,
        help="Optional local Adamson UPR h5ad for --rescore",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--loess-frac", type=float, default=DEFAULT_LOESS_FRAC)
    args = parser.parse_args()

    if args.force_normalize_log1p and args.assume_log:
        raise SystemExit("Pass only one of --force-normalize-log1p / --assume-log")

    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = ""
    if args.force_normalize_log1p:
        tag = "_forcenorm"
    elif args.assume_log:
        tag = "_assumelog"

    if args.rescore:
        from pipeline_core import run_dataset, setup_cache

        matrix_is_log = None
        if args.force_normalize_log1p:
            matrix_is_log = False
            print(
                "Re-scoring Adamson UPR with FORCE normalize+log1p "
                "(matrix_is_log=False)…",
                flush=True,
            )
        elif args.assume_log:
            matrix_is_log = True
            print(
                "Re-scoring Adamson UPR with assume-log "
                "(matrix_is_log=True)…",
                flush=True,
            )
        else:
            print("Re-scoring Adamson UPR under frozen pipeline pin…", flush=True)

        setup_cache()
        df = run_dataset(
            "Adamson 2016 UPR (CRISPRi)",
            prefer_local=True,
            h5ad_path=args.h5ad,
            matrix_is_log=matrix_is_log,
        )
        df = annotate_upr(df)
        csv_path = out_dir / f"adamson_upr_sp_scores{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df)} rows)", flush=True)
    else:
        csv_path = Path(args.input) if args.input else out_dir / "adamson_upr_sp_scores.csv"
        if not csv_path.exists():
            raise SystemExit(
                f"No scores at {csv_path}. Run adamson_upr_spike.py or pass --rescore."
            )
        df = load_scores(csv_path)
        print(f"Loaded {csv_path} ({len(df)} rows)", flush=True)

    print("=" * 72)
    print("ADAMSON UPR — MAGNITUDE PARTIAL (paper fork)")
    print(f"config_version={cfg.CONFIG_VERSION}  SEED={cfg.SEED}")
    if args.force_normalize_log1p:
        print("preprocess diagnostic: FORCE normalize+log1p")
    print("=" * 72)

    results = run_tests(
        df, n_bootstrap=args.n_bootstrap, loess_frac=float(args.loess_frac)
    )
    annotated = results.pop("annotated")
    results["preprocess_tag"] = tag or "pinned"
    results["matrix_is_log_forced"] = (
        False if args.force_normalize_log1p else (True if args.assume_log else None)
    )

    # Persist annotated table
    ann_path = out_dir / f"adamson_upr_magnitude_partial_scores{tag}.csv"
    keep = [
        c
        for c in [
            "dataset",
            "perturbation",
            "gene",
            "is_upr_core",
            "stability",
            "magnitude",
            "sp_resid_mag",
            "disc_loess",
            "n_cells",
            "config_version",
        ]
        if c in annotated.columns
    ]
    annotated[keep].to_csv(ann_path, index=False)

    blurb = _blurb(results)
    results["methods_blurb"] = blurb

    summary_path = out_dir / f"adamson_upr_magnitude_partial_summary{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    blurb_path = out_dir / f"adamson_upr_magnitude_partial_blurb{tag}.txt"
    blurb_path.write_text(blurb + "\n")

    # Console report
    raw = results["raw_sp_core_lower"]
    mag_g = results["magnitude_core_greater"]
    part = results["partial_spearman_sp_vs_core_given_magnitude"]
    resid = results["mag_residual_sp_core_lower"]
    loess = results["loess_discordance_core_greater"]
    boot = results["partial_boot"]

    print("\n--- 1. Raw Sp (plan claim) ---")
    print(
        f"  UPR-core median Sp={raw['median_a']:.3f} (n={raw['n_a']})  "
        f"other={raw['median_b']:.3f} (n={raw['n_b']})"
    )
    print(f"  MWU H1 core lower: p={raw['p']:.3g}")
    if not results["plan_claim"]["matches_plan"]:
        print(
            f"  *** mismatch vs plan {PLAN_MEDIAN_CORE} vs {PLAN_MEDIAN_OTHER}, "
            f"p={PLAN_MWU_P} ***"
        )

    print("\n--- 2. Magnitude (confound check) ---")
    print(
        f"  UPR-core median mag={mag_g['median_a']:.3f}  "
        f"other={mag_g['median_b']:.3f}"
    )
    print(
        f"  MWU H1 core greater: p={mag_g['p']:.3g}  |  "
        f"H1 core less: p={results['magnitude_core_less']['p']:.3g}"
    )
    print(
        f"  Sp–magnitude Spearman={results['spearman_magnitude_sp']:.3f} "
        f"(p={results['spearman_magnitude_sp_p']:.2e})"
    )

    print("\n--- 3. Partial Spearman Sp ~ core | magnitude ---")
    print(
        f"  UPR_CORE ({results['upr_core_set_id']}): "
        f"pinned={results['upr_core_genes_pinned']}; "
        f"matched n={results['n_upr_core']} → {results['upr_core_genes']}"
    )
    print(
        f"  rho_partial={part.get('rho_partial')}  p={part.get('p')}  "
        f"n={part.get('n')}  ({part.get('method')})"
    )
    print(
        f"  bootstrap CI=[{boot.get('ci_low')}, {boot.get('ci_high')}] "
        f"(n_boot={boot.get('n_bootstrap')})"
    )
    gate = results.get("partial_survival_status") or {}
    print(
        f"  gate={gate.get('status')}  criterion={results.get('survival_criterion_id')}  "
        f"ci_margin={gate.get('ci_margin')}  knife_edge={gate.get('knife_edge')}  "
        f"ε={results.get('survival_knife_edge_abs')}"
    )

    print("\n--- 4. Mag-residual Sp MWU ---")
    print(
        f"  median resid core={resid['median_a']:.3f}  other={resid['median_b']:.3f}  "
        f"p={resid['p']:.3g}"
    )

    print("\n--- 5. LOESS discordance MWU ---")
    print(
        f"  median disc core={loess['median_a']:.3f}  other={loess['median_b']:.3f}  "
        f"p={loess['p']:.3g}  (frac={results['loess_frac']})"
    )

    print("\n--- FORK ---")
    print(f"signal: {results['fork_signal']}")
    print(results["fork_note"])
    print(f"\nWrote {summary_path}")
    print(f"Wrote {ann_path}")
    print(f"Wrote {blurb_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
