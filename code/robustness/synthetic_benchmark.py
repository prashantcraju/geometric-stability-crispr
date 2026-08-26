#!/usr/bin/env python3
"""
Synthetic benchmarking with controlled perturbation efficiency.

Song-style synthetic datasets varying cell state, perturbation efficiency,
and multi-program structure.

DESIGN (pre-registered; grid lives in pipeline_config, fixed before any result)

Two independent factors both lower Sp, and separating them is the whole point:

  efficiency  a mixture of unperturbed / partial / full responders moving along
              ONE shared axis (technical confound).
  programs    cells at full efficiency moving along SEVERAL distinct axes.
              The biology Sp claims to detect.

The benchmark therefore scores three distinct tasks, not one:

  1. Efficiency recovery (Song's task).  Per-cell |psi_true - psi_pred| <= 0.1.
     Sp is a population-level directional statistic, NOT a per-cell efficiency
     estimator, so PS is expected to win.  Sp is reported here as a SENSITIVITY
     CURVE (how far Sp falls as efficiency drops) — characterising a confound,
     not competing.  Running this also validates our pure-Python PS port
     against published numbers.

  2. Coherence recovery (the novelty claim).  AUROC for discriminating
     1-program from K-program responses at MATCHED magnitude and efficiency.
     PS scores per-cell displacement along a fitted signature and is
     direction-blind by construction, so Sp should win.  If it does not, there
     is no novelty argument and we need to know that before writing.

  3. Identifiability.  Requires the programs arm to hold population mean-shift
     magnitude fixed while varying direction diversity
     (SYNTH_MATCH_PROGRAM_MAGNITUDE).  Without that, Sp tracks magnitude by
     construction and the test is about the generator, not Sp.  Then:
     magnitude-matched efficiency-mixture vs multi-program settings — does Sp
     separate them?  Mixscape/PS correction is a related sensitivity analysis.

GENERATIVE MODEL

scDesign3 (Song, Wang, Yan et al. Nat Biotechnol 42:247, 2024), driven from R
exactly as Song et al. 2025 did, with their four-step Perturb-seq modification:

  Step 1  ZINB marginals; downstream (DEG) genes' mean depends on the target
          gene's own expression C, non-downstream genes' does not.  Implemented
          by fitting fit_marginal() twice with mu_formula "C" and "1" and
          splicing the two marginal lists.
  Step 2  Gaussian copula fit separately for KO and WT cells (corr_by = "K").
  Step 3  Efficiency enters by REPLACING C in perturbed cells with
          C* = (1 - psi_j) * C_sampled_from_WT.  psi = 1 gives C* = 0 (full
          knockout), psi = 0 gives WT-level C (no response).  We do this by
          handing extract_para() a modified covariate frame rather than editing
          alpha/beta by hand — mathematically identical, far more robust.
          Dispersion and zero-inflation are left unmodified, as in Song.
  Step 4  simu_new() draws the synthetic count matrix.

Our extensions all reduce to one abstraction: a per-cell, per-gene choice
between the psi=0 mean and the psi=psi_j mean.  Multi-program responses give a
cell the perturbed mean only on its own program's DEG set; non-responsive cell
states give a cell the psi=0 mean everywhere.

DEVIATIONS FROM SONG, stated for the record:
  - Song ranked downstream genes from bulk RNA-seq of Nelfb KO vs WT.  We have
    no matched bulk, so DEGs are ranked from the reference dataset's own real
    KO-vs-WT single-cell contrast.  See rank_reference_degs().
  - Song used one reference gene (Nelfb, mouse T cells).  We use several real
    perturbations spanning the observed Sp range, from the frozen datasets, so
    conclusions are not hostage to one signature's geometry.

Synthetic counts are scored by running pipeline_core.preprocess() and
calculate_sp() — the SAME code path as the manuscript.  Injecting shifts
directly into PC space would make Sp recover them trivially and the benchmark
vacuous (the same issue as an isotropic PC-space simulation).

REQUIREMENTS
  Python: numpy, pandas, scipy, scanpy, anndata  (+ pingouin for partials)
  R:      scDesign3, SingleCellExperiment, Matrix, gamlss, mgcv
          install.packages("devtools")
          devtools::install_github("SONGDONGYUAN1994/scDesign3")

USAGE
  # no R needed: emit the R script + ground-truth tables and check the grid
  python synthetic_benchmark.py --dry-run
  python synthetic_benchmark.py --self-test

  # Status (.9): attempted; SYNTH_MANUSCRIPT_USABLE=False. Code kept for audit.
  # Does NOT call materialize / pathway / QC. Do not bump CONFIG_VERSION here.
  # Do not re-tune effect_scale to a new auto-selected reference.

  # full run (needs R + scDesign3)
  python synthetic_benchmark.py --arm song_replication
  python synthetic_benchmark.py --arm all --reference-gene GATA1
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
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import pipeline_config as cfg
from revision_io import resolve_out_dir

WT = "WT"
KO = "KO"


# ===========================================================================
# GRID
# ===========================================================================


@dataclass(frozen=True)
class SimSetting:
    """One simulated dataset. `arm` records which sweep produced it."""

    arm: str
    reference: str
    n_deg: int
    efficiency: float
    efficiency_dist: str
    n_programs: int
    program_overlap: float
    state_mix: float
    n_pert_cells: int
    n_ctrl_cells: int
    seed: int

    @property
    def setting_id(self) -> str:
        return (
            f"{self.arm}__ref-{self.reference}__deg{self.n_deg}"
            f"__eff{self.efficiency:.2f}__{self.efficiency_dist}"
            f"__prog{self.n_programs}__ov{self.program_overlap:+.2f}"
            f"__state{self.state_mix:.2f}__n{self.n_pert_cells}"
            f"__seed{self.seed}"
        )


def _base_kwargs() -> dict:
    return dict(cfg.SYNTH_REFERENCE_CELL)


def build_grid(
    reference: str,
    arms: Optional[list[str]] = None,
    n_seeds: int = cfg.SYNTH_N_SEEDS,
) -> list[SimSetting]:
    """
    Song's D x psi factorial, then one-factor-at-a-time sweeps around the
    reference cell.  Deliberately not a full factorial: the full cross would
    explode and produce an unreadable figure without answering anything the
    marginal sweeps do not.
    """
    all_arms = [
        "song_replication",
        "efficiency_dist",
        "programs",
        "state",
        "cell_count",
    ]
    arms = arms or all_arms
    unknown = set(arms) - set(all_arms)
    if unknown:
        raise ValueError(f"Unknown arm(s) {sorted(unknown)}; choose from {all_arms}")

    settings: list[SimSetting] = []
    seeds = [cfg.SEED + 1000 * i for i in range(n_seeds)]

    def add(arm: str, **overrides):
        kw = _base_kwargs()
        kw.update(overrides)
        for s in seeds:
            settings.append(
                SimSetting(arm=arm, reference=reference, seed=s, **kw)
            )

    if "song_replication" in arms:
        for d in cfg.SONG_N_DEG:
            for eff in cfg.SONG_EFFICIENCY:
                add("song_replication", n_deg=d, efficiency=eff)

    if "efficiency_dist" in arms:
        for dist in cfg.SYNTH_EFFICIENCY_DISTS:
            for eff in cfg.SONG_EFFICIENCY:
                add("efficiency_dist", efficiency_dist=dist, efficiency=eff)

    if "programs" in arms:
        for k in cfg.SYNTH_N_PROGRAMS:
            for ov in cfg.SYNTH_PROGRAM_OVERLAP:
                if k == 1 and ov != 1.0:
                    continue  # overlap is undefined for a single program
                # Antagonistic (ov=-1) is two opposite axes on one gene set.
                # With K>2 the old impl put (K-1) programs on − and one on +,
                # so the mean was nonzero and Sp could exceed ov=0 — a geometry
                # bug, not biology. Only K=2 is well-defined for ov=-1.
                if ov < 0 and k != 2:
                    continue
                # full efficiency isolates program structure from efficiency
                add("programs", n_programs=k, program_overlap=ov, efficiency=1.0)

    if "state" in arms:
        for mix in cfg.SYNTH_STATE_MIX:
            add("state", state_mix=mix, efficiency=1.0)

    if "cell_count" in arms:
        for n in cfg.SYNTH_N_PERT_CELLS:
            add("cell_count", n_pert_cells=n)

    # de-duplicate: the reference cell recurs across arms
    seen, unique = set(), []
    for s in settings:
        if s.setting_id in seen:
            continue
        seen.add(s.setting_id)
        unique.append(s)
    return unique


# ===========================================================================
# GROUND TRUTH: per-cell efficiency, program, state
# ===========================================================================
# Generated in Python so that all randomness is controlled by the frozen
# config seed rather than by R's RNG.


def draw_efficiencies(
    setting: SimSetting,
    rng: np.random.Generator,
    n: Optional[int] = None,
) -> np.ndarray:
    """
    Per-cell true efficiency psi_i in [0, 1] for the perturbed cells.

    All four distributions have the same MEAN efficiency, so any difference in
    Sp between them is attributable to heterogeneity alone: a mix of
    perturbed, unperturbed, and partially perturbed cells, which the
    homogeneous arm (all Song used) cannot produce.
    """
    n = setting.n_pert_cells if n is None else n
    eff, dist = setting.efficiency, setting.efficiency_dist

    if dist == "homogeneous":
        return np.full(n, eff, dtype=float)

    if dist == "bimodal":
        # fraction `eff` of cells fully perturbed, remainder untouched
        psi = np.zeros(n, dtype=float)
        n_resp = int(round(eff * n))
        psi[rng.permutation(n)[:n_resp]] = 1.0
        return psi

    if dist in cfg.SYNTH_BETA_CONCENTRATION:
        conc = cfg.SYNTH_BETA_CONCENTRATION[dist]
        eps = 1e-3
        m = float(np.clip(eff, eps, 1 - eps))
        return rng.beta(m * conc, (1 - m) * conc, size=n).astype(float)

    raise ValueError(f"Unknown efficiency_dist {dist!r}")


def assign_programs(
    setting: SimSetting,
    rng: np.random.Generator,
    n: Optional[int] = None,
) -> np.ndarray:
    """Program index per perturbed cell, equal mixing weights."""
    n = setting.n_pert_cells if n is None else n
    return rng.integers(0, setting.n_programs, size=n)


def assign_states(
    setting: SimSetting,
    rng: np.random.Generator,
    n: Optional[int] = None,
) -> np.ndarray:
    """
    1 = responsive control substate, 0 = non-responsive.

    Non-responsive cells carry the perturbation but do not respond to it —
    coherence can depend on the intersection of the perturbed factor and its
    context, with ground truth attached.
    """
    n = setting.n_pert_cells if n is None else n
    return (rng.random(n) < setting.state_mix).astype(int)


def build_program_gene_sets(
    setting: SimSetting,
    deg_ranked: list[str],
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    """
    Program 0 is the real signature: the top n_deg ranked DEGs, sign +1.

    Additional programs share `program_overlap` of program 0's genes and draw
    the remainder from lower-ranked genes.  Parameterising the angle between
    programs by DEG-set OVERLAP rather than by rotating a signature in gene
    space keeps every program a plausible signature; an arbitrary rotation
    would produce a biologically meaningless DEG set.

    program_overlap = -1.0 encodes an antagonistic program: same gene set,
    sign flipped.  Returns {program_id: {gene: sign}}.
    """
    d = setting.n_deg
    if d > len(deg_ranked):
        raise ValueError(
            f"n_deg={d} exceeds {len(deg_ranked)} ranked reference DEGs"
        )
    primary = list(deg_ranked[:d])
    programs: dict[int, dict[str, float]] = {0: {g: 1.0 for g in primary}}

    if setting.n_programs == 1:
        return programs

    if setting.program_overlap < 0:
        if setting.n_programs != 2:
            raise ValueError(
                "program_overlap=-1 (antagonistic) requires n_programs=2 "
                f"(got {setting.n_programs}); more than two signed copies of "
                "the same axis unbalances the mixture and inflates Sp"
            )
        programs[1] = {g: -1.0 for g in primary}
        return programs

    pool = [g for g in deg_ranked[d:] if g not in set(primary)]
    n_shared = int(round(setting.program_overlap * d))
    n_new = d - n_shared
    if n_new * (setting.n_programs - 1) > len(pool):
        raise ValueError(
            f"Not enough reference genes: {setting.n_programs - 1} extra programs "
            f"need {n_new} fresh genes each, pool has {len(pool)}"
        )
    for k in range(1, setting.n_programs):
        # Overlap is defined against program 0, so the shared part may recur
        # across programs; the fresh part must not, or overlap=0 would still
        # leave programs 1..K-1 partially aligned with each other.
        shared = list(rng.choice(primary, size=n_shared, replace=False)) if n_shared else []
        fresh = list(rng.choice(pool, size=n_new, replace=False)) if n_new else []
        pool = [g for g in pool if g not in set(fresh)]
        programs[k] = {g: 1.0 for g in list(shared) + list(fresh)}
    return programs


# Every setting is simulated at the largest cell count and subsampled at
# scoring time. scDesign3's expensive steps (fit_marginal over thousands of
# genes, fit_copula) are then amortised over the whole grid instead of being
# repeated per cell-count setting.
SIM_N_PERT = max(cfg.SYNTH_N_PERT_CELLS)
SIM_N_CTRL = int(cfg.SYNTH_REFERENCE_CELL["n_ctrl_cells"])


def build_ground_truth(
    setting: SimSetting,
    deg_ranked: list[str],
    wt_C: np.ndarray,
    n_pert: int = SIM_N_PERT,
    n_ctrl: int = SIM_N_CTRL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (cells, programs) tables handed to R and kept for scoring.

    cells:    cell_id, group, psi_true, psi_effective, program, state, C_star
    programs: program, gene, sign

    psi_effective is the efficiency actually applied: a cell in the
    non-responsive state has psi_effective 0 regardless of psi_true.  Metrics
    must score against psi_effective, not psi_true, or the cell-state arm will
    read as method failure when it is correct by construction.

    C_star implements Song's Step 3.  For a perturbed cell,
        C_star = (1 - psi_effective) * C_sampled_from_WT
    so psi=1 gives C_star=0 (full knockout of the target's expression) and
    psi=0 gives a WT-level C (no downstream response).  WT cells keep their
    observed C.  Sampling happens here, in Python, so the frozen seed controls
    it rather than R's RNG.
    """
    rng = np.random.default_rng(setting.seed)

    psi = draw_efficiencies(setting, rng, n=n_pert)
    program = assign_programs(setting, rng, n=n_pert)
    state = assign_states(setting, rng, n=n_pert)
    psi_eff = psi * state

    c_sampled = rng.choice(np.asarray(wt_C, dtype=float), size=n_pert, replace=True)

    pert = pd.DataFrame(
        {
            "cell_id": [f"ko_{i:05d}" for i in range(n_pert)],
            "group": KO,
            "psi_true": psi,
            "psi_effective": psi_eff,
            "program": program,
            "state": state,
            "C_star": (1.0 - psi_eff) * c_sampled,
        }
    )
    ctrl = pd.DataFrame(
        {
            "cell_id": [f"wt_{i:05d}" for i in range(n_ctrl)],
            "group": WT,
            "psi_true": 0.0,
            "psi_effective": 0.0,
            "program": -1,
            "state": 1,
            # placeholder; R overwrites WT rows with their observed C
            "C_star": np.nan,
        }
    )
    cells = pd.concat([pert, ctrl], ignore_index=True)

    prog_sets = build_program_gene_sets(setting, deg_ranked, rng)
    programs = pd.DataFrame(
        [
            {"program": k, "gene": g, "sign": s}
            for k, genes in prog_sets.items()
            for g, s in genes.items()
        ]
    )
    return cells, programs


# ===========================================================================
# SELF-TEST (runs without R, scanpy, or any data)
# ===========================================================================


def self_test() -> None:
    """Validate grid construction and ground-truth assembly logic."""
    print("=" * 72)
    print("SELF-TEST")
    print(f"synthetic_config_version = {cfg.SYNTHETIC_CONFIG_VERSION}")
    print("=" * 72)

    grid = build_grid("mid_sp", n_seeds=2)
    by_arm = pd.Series([s.arm for s in grid]).value_counts().sort_index()
    print("\nGrid sizes by arm (n_seeds=2):")
    for arm, n in by_arm.items():
        print(f"  {arm:20s} {n:4d}")
    print(f"  {'TOTAL':20s} {len(grid):4d}")

    full = build_grid("mid_sp")
    print(f"\nAt n_seeds={cfg.SYNTH_N_SEEDS}: {len(full)} datasets per reference")

    deg_ranked = [f"GENE{i:04d}" for i in range(cfg.SYNTH_N_HVG_REFERENCE)]
    wt_C = np.random.default_rng(0).poisson(12.0, size=SIM_N_CTRL).astype(float)

    # mean efficiency must match across distributions
    print("\nEfficiency distributions (target mean 0.50, n=200):")
    for dist in cfg.SYNTH_EFFICIENCY_DISTS:
        s = SimSetting(
            arm="t", reference="mid_sp", n_deg=50, efficiency=0.50,
            efficiency_dist=dist, n_programs=1, program_overlap=1.0,
            state_mix=1.0, n_pert_cells=200, n_ctrl_cells=500, seed=cfg.SEED,
        )
        psi = draw_efficiencies(s, np.random.default_rng(s.seed))
        frac_zero = float(np.mean(psi < 0.05))
        print(
            f"  {dist:12s} mean={psi.mean():.3f} sd={psi.std():.3f} "
            f"frac_unperturbed={frac_zero:.2f}"
        )
        assert abs(psi.mean() - 0.50) < 0.12, f"{dist} mean drifted"
        assert psi.min() >= 0.0 and psi.max() <= 1.0

    # program overlap must translate into the intended gene-set overlap
    print("\nProgram gene-set overlap (n_deg=50, 2 programs):")
    for ov in cfg.SYNTH_PROGRAM_OVERLAP:
        s = SimSetting(
            arm="t", reference="mid_sp", n_deg=50, efficiency=1.0,
            efficiency_dist="homogeneous", n_programs=2, program_overlap=ov,
            state_mix=1.0, n_pert_cells=100, n_ctrl_cells=500, seed=cfg.SEED,
        )
        _, programs = build_ground_truth(s, deg_ranked, wt_C)
        g0 = set(programs.loc[programs["program"] == 0, "gene"])
        g1 = set(programs.loc[programs["program"] == 1, "gene"])
        signs = programs.loc[programs["program"] == 1, "sign"].unique()
        realized = len(g0 & g1) / len(g0)
        print(
            f"  overlap={ov:+.2f} -> realized={realized:.2f} "
            f"signs={sorted(float(x) for x in signs)}"
        )
        if ov >= 0:
            assert abs(realized - ov) < 0.02, f"overlap {ov} not realized"
    try:
        build_program_gene_sets(
            SimSetting(
                arm="t", reference="mid_sp", n_deg=50, efficiency=1.0,
                efficiency_dist="homogeneous", n_programs=3, program_overlap=-1.0,
                state_mix=1.0, n_pert_cells=100, n_ctrl_cells=500, seed=cfg.SEED,
            ),
            deg_ranked,
            np.random.default_rng(0),
        )
        raise AssertionError("ov=-1 with n_programs=3 should raise")
    except ValueError as e:
        print(f"  ov=-1 / K=3 correctly rejected: {e}")
    prog_grid = [s for s in build_grid("mid_sp", arms=["programs"], n_seeds=1)]
    assert not any(s.program_overlap < 0 and s.n_programs != 2 for s in prog_grid)
    print(f"  programs arm (1 seed): {len(prog_grid)} settings (no ov=-1 for K≠2)")

    # state mix zeroes out effective efficiency
    print("\nState mix -> effective efficiency (psi_true=1.0):")
    for mix in cfg.SYNTH_STATE_MIX:
        s = SimSetting(
            arm="t", reference="mid_sp", n_deg=50, efficiency=1.0,
            efficiency_dist="homogeneous", n_programs=1, program_overlap=1.0,
            state_mix=mix, n_pert_cells=400, n_ctrl_cells=500, seed=cfg.SEED,
        )
        cells, _ = build_ground_truth(s, deg_ranked, wt_C)
        ko = cells[cells["group"] == KO]
        print(
            f"  state_mix={mix:.2f} -> mean psi_effective="
            f"{ko['psi_effective'].mean():.3f}  "
            f"mean C_star={ko['C_star'].mean():.2f}"
        )
        assert abs(ko["psi_effective"].mean() - mix) < 0.10
        # C_star must fall as effective efficiency rises (Song's Step 3)
        assert ko.loc[ko["psi_effective"] > 0.9, "C_star"].max() < 1e-9

    # Identifiability: Sp separates when magnitude is matched (synthetic toy)
    print("\nIdentifiability (magnitude-matched Sp separation, toy table):")
    toy = pd.DataFrame(
        [
            # efficiency mixtures: mid magnitude, mid Sp
            {"setting_id": "mix1", "arm": "efficiency_dist", "n_programs": 1,
             "program_overlap": 1.0, "efficiency": 0.5, "efficiency_dist": "bimodal",
             "sp": 0.50, "magnitude": 20.0},
            {"setting_id": "mix2", "arm": "efficiency_dist", "n_programs": 1,
             "program_overlap": 1.0, "efficiency": 0.25, "efficiency_dist": "beta_broad",
             "sp": 0.40, "magnitude": 14.0},
            # multi-program: same magnitudes, lower Sp
            {"setting_id": "mp1", "arm": "programs", "n_programs": 2,
             "program_overlap": 0.0, "efficiency": 1.0, "efficiency_dist": "homogeneous",
             "sp": 0.30, "magnitude": 20.5},
            {"setting_id": "mp2", "arm": "programs", "n_programs": 3,
             "program_overlap": 0.0, "efficiency": 1.0, "efficiency_dist": "homogeneous",
             "sp": 0.25, "magnitude": 13.5},
        ]
    )
    ident = metric_identifiability(toy, mag_tol_rel=0.15)
    print(f"  {ident.iloc[0].to_dict()}")
    assert int(ident.iloc[0]["n_matched_pairs"]) == 2
    # AUROC is oriented as P(multi Sp > mix Sp); here multi is lower → ~0.0
    assert abs(float(ident.iloc[0]["auroc_sp"]) - 0.5) >= 0.4
    assert abs(float(ident.iloc[0]["auroc_magnitude"]) - 0.5) < 0.25
    assert bool(ident.iloc[0]["sp_separates_regimes"])

    print("\nSELF-TEST PASSED")


def check_env(out_dir: Path, r_executable: str = cfg.SYNTH_R_EXECUTABLE) -> bool:
    """
    Report what each stage needs and whether it is available.

    Stages degrade independently, so a partial environment still gets you
    something useful:
      --self-test  numpy + pandas only
      --dry-run    + scanpy/anndata + a frozen Sp table + dataset access
      full run     + R with scDesign3
    """
    print("=" * 72)
    print("ENVIRONMENT CHECK")
    print("=" * 72)

    def probe(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    stages = {
        "--self-test (grid + ground-truth logic)": ["numpy", "pandas"],
        "--dry-run (build reference, emit R script)": [
            "numpy", "pandas", "scipy", "scanpy", "anndata",
        ],
        "full run (simulate + score)": [
            "numpy", "pandas", "scipy", "scanpy", "anndata", "sklearn",
        ],
    }
    ok_all = True
    for stage, mods in stages.items():
        missing = [m for m in mods if not probe(m)]
        status = "READY" if not missing else f"MISSING {', '.join(missing)}"
        print(f"  {'OK ' if not missing else '-- '} {stage:44s} {status}")
        ok_all &= not missing

    print("\n  optional:")
    for mod, why in [
        ("pingouin", "rank-based partial correlations (stats_utils)"),
        ("pertpy", "method_mixscape once implemented"),
        ("statsmodels", "PS port helpers in song_ps_replication"),
    ]:
        print(f"  {'OK ' if probe(mod) else '-- '} {mod:12s} {why}")

    print("\n  R (required for scDesign3):")
    try:
        proc = subprocess.run(
            [r_executable, "-e", 'cat(as.character(packageVersion("scDesign3")))'],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            print(f"  OK  scDesign3 {proc.stdout.strip().splitlines()[-1]}")
        else:
            print(f"  --  {r_executable} found but scDesign3 not installed:")
            print('      R -e \'devtools::install_github("SONGDONGYUAN1994/scDesign3")\'')
            ok_all = False
    except FileNotFoundError:
        print(f"  --  {r_executable} not on PATH; install R, or pass --r-executable")
        ok_all = False
    except subprocess.TimeoutExpired:
        print("  --  R probe timed out")
        ok_all = False

    print(f"\n  reference Sp table (searched in {out_dir}):")
    try:
        from revision_io import find_sp_csv, load_sp_table

        p = find_sp_csv(out_dir)
        df = load_sp_table(p)
        usable = df[df["dataset"].isin(cfg.SYNTH_REFERENCE_DATASETS)]
        print(f"  OK  {p.name}: {len(df)} rows, datasets {sorted(df['dataset'].unique())}")
        if usable.empty:
            print(
                f"  --  none are in SYNTH_REFERENCE_DATASETS "
                f"{cfg.SYNTH_REFERENCE_DATASETS}; run run_frozen_main.py, or use "
                "--reference-gene to bypass selection"
            )
            ok_all = False
        else:
            at_cap = int((usable["n_cells"] >= cfg.MAX_CELLS_PER_PERT).sum())
            print(f"  OK  {len(usable)} usable rows, {at_cap} at the downsample cap")
    except Exception as e:
        print(f"  --  {e}")
        ok_all = False

    print("\n" + ("READY for a full run." if ok_all else "NOT ready for a full run; see above."))
    return ok_all


# ===========================================================================
# REFERENCE DATA: real KO + WT cells to fit scDesign3 against
# ===========================================================================


def select_reference_perturbations(
    out_dir: Path,
    sp_csv: Optional[Path] = None,
) -> dict[str, dict]:
    """
    Pick real perturbations spanning the observed Sp range from the frozen
    table, rather than hard-coding gene names that may not survive filtering.

    Song used one reference gene; using low/mid/high-Sp references means the
    simulation is not hostage to one signature's geometry.
    """
    from revision_io import find_sp_csv, load_sp_table

    path = sp_csv or find_sp_csv(out_dir)
    df = load_sp_table(path)
    have = sorted(df["dataset"].unique())
    df = df[df["dataset"].isin(cfg.SYNTH_REFERENCE_DATASETS)]
    if df.empty:
        raise ValueError(
            f"{path} contains no rows for {cfg.SYNTH_REFERENCE_DATASETS}.\n"
            f"  It has: {have}\n"
            "  Run run_frozen_main.py to produce frozen_sp_scores.csv, or pass "
            "--reference-gene/--reference-dataset to skip selection."
        )
    # n_cells in the frozen table is post-downsample and capped at
    # MAX_CELLS_PER_PERT, so this only screens out perturbations that never
    # reached the cap. The real KO count is validated in
    # build_reference_matrices() against the raw data.
    df = df[df["n_cells"] >= cfg.MAX_CELLS_PER_PERT]
    # Combinatorial / intergenic labels are bad scDesign3 references (often
    # few cells, messy signatures). Prefer single-gene Replogle targets.
    gene_col = "perturbation" if "perturbation" in df.columns else "gene"
    df = df[~df[gene_col].astype(str).str.contains(r"[+&]", regex=True)]
    prefer = "Replogle 2022 (CRISPRi)"
    if (df["dataset"] == prefer).any():
        df = df[df["dataset"] == prefer]
        print(f"  (preferring {prefer} for scDesign3 references)")
    if df.empty:
        raise ValueError(
            f"No perturbation in {path} reached the {cfg.MAX_CELLS_PER_PERT}-cell "
            "downsample cap, so none can supply enough real cells to be a "
            "scDesign3 reference."
        )

    refs: dict[str, dict] = {}
    for label, q in cfg.SYNTH_REFERENCE_SP_QUANTILES.items():
        target = df["stability"].quantile(q)
        # nearest by Sp; keep a few alternates in case raw cell count is short
        order = (df["stability"] - target).abs().argsort()
        row = df.iloc[order.iloc[0]]
        alternates = [
            {
                "dataset": str(df.iloc[i]["dataset"]),
                "gene": str(df.iloc[i][gene_col]),
                "observed_sp": float(df.iloc[i]["stability"]),
            }
            for i in order.iloc[1:8]
        ]
        refs[label] = {
            "label": label,
            "dataset": str(row["dataset"]),
            "gene": str(row[gene_col]),
            "observed_sp": float(row["stability"]),
            "observed_magnitude": float(row["magnitude"]),
            "observed_n_cells": int(row["n_cells"]),
            "sp_quantile": q,
            "alternates": alternates,
        }
        print(
            f"  reference {label:8s}: {row[gene_col]} "
            f"({row['dataset']}) Sp={row['stability']:.3f} n={row['n_cells']}"
        )
    return refs


def rank_reference_degs(
    counts: np.ndarray,
    genes: list[str],
    is_ko: np.ndarray,
) -> list[str]:
    """
    Rank downstream genes by the reference dataset's own real KO-vs-WT contrast.

    DEVIATION FROM SONG: they ranked downstream genes from matched bulk RNA-seq
    of Nelfb KO vs WT.  No matched bulk exists for the frozen Perturb-seq
    datasets, so we substitute the single-cell contrast.  Ranking is by
    Wilcoxon p-value with a log-fold-change tiebreak, mirroring how Song's
    bulk list was ordered.  Must be reported as a deviation in Methods.
    """
    from scipy.stats import mannwhitneyu

    cpm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e4
    lg = np.log1p(cpm)
    ko, wt = lg[is_ko.astype(bool)], lg[~is_ko.astype(bool)]
    lfc = ko.mean(axis=0) - wt.mean(axis=0)

    pvals = np.ones(len(genes))
    for j in range(len(genes)):
        if ko[:, j].std() == 0 and wt[:, j].std() == 0:
            continue
        try:
            pvals[j] = mannwhitneyu(ko[:, j], wt[:, j], alternative="two-sided")[1]
        except ValueError:
            pvals[j] = 1.0

    order = np.lexsort((-np.abs(lfc), pvals))
    return [genes[j] for j in order]


def build_reference_matrices(
    ref: dict,
    n_ko: int,
    n_wt: int,
    seed: int = cfg.SEED,
) -> dict:
    """
    Assemble the reference for scDesign3: raw counts, the K (KO/WT) covariate,
    the C covariate (the target gene's own counts), and the ranked DEG list.

    C is the mechanism by which efficiency enters the model: Song's downstream
    genes regress on the target's expression, so setting C to (1-psi)*WT-level
    is what makes a cell partially perturbed.
    """
    import scanpy as sc
    from scipy import sparse

    from pipeline_core import _extract_adata, ensure_in_memory, load_raw, setup_cache

    setup_cache()
    dataset, gene = ref["dataset"], ref["gene"]
    print(f"\n  Loading reference {gene} from {dataset}…", flush=True)

    raw = load_raw(dataset, sc=sc, prefer_local=True)
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset, sc)
    labels = adata.obs[pert_col].astype(str).to_numpy()

    rng = np.random.default_rng(seed)
    ko_idx = np.flatnonzero(labels == gene)
    wt_idx = np.flatnonzero(labels == ctrl_label)
    # Ground-truth tables are built at n_ko/n_wt. Prefer the requested size,
    # but accept fewer KO cells down to MAX_CELLS_PER_PERT so a high-Sp gene
    # with 112 raw cells can still run the programs arm (n=100). The cell_count
    # arm's n=200 settings will simply subsample to available.
    if len(ko_idx) < n_ko:
        if len(ko_idx) < cfg.MAX_CELLS_PER_PERT:
            raise ValueError(
                f"{gene} in {dataset}: {len(ko_idx)} real KO cells, need ≥"
                f"{cfg.MAX_CELLS_PER_PERT}. Pick another --reference-gene."
            )
        print(
            f"    WARNING: {gene} has {len(ko_idx)} KO cells < requested {n_ko}; "
            f"using {len(ko_idx)}",
            flush=True,
        )
        n_ko = int(len(ko_idx))
    if len(wt_idx) < n_wt:
        if len(wt_idx) < 100:
            raise ValueError(
                f"{dataset}: {len(wt_idx)} real control cells, need ≥100."
            )
        print(
            f"    WARNING: {len(wt_idx)} control cells < requested {n_wt}; "
            f"using {len(wt_idx)}",
            flush=True,
        )
        n_wt = int(len(wt_idx))

    if len(ko_idx) > n_ko:
        ko_idx = rng.choice(ko_idx, size=n_ko, replace=False)
    if len(wt_idx) > n_wt:
        wt_idx = rng.choice(wt_idx, size=n_wt, replace=False)
    keep = np.sort(np.concatenate([ko_idx, wt_idx]))

    sub = ensure_in_memory(adata[keep])
    X = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    X = np.rint(np.asarray(X, dtype=float))
    X[X < 0] = 0.0
    # Song Methods: amplify WT counts ×10 so perturbation effects have range.
    # Applied to the whole reference matrix (KO+WT) after integer rounding.
    amp = float(cfg.SYNTH_COUNT_AMPLIFICATION)
    if amp != 1.0:
        X = np.rint(X * amp)
        print(f"    Song amplification: counts ×{amp:g}", flush=True)

    all_genes = [str(g) for g in sub.var_names]
    is_ko = (sub.obs[pert_col].astype(str).to_numpy() == gene).astype(int)

    # Song reduces to 3000 HVGs (scran) plus the refined DEG list, and drops
    # genes expressed in too few cells before fitting.
    detected = (X > 0).sum(axis=0)
    keep_g = detected >= 3
    X, all_genes = X[:, keep_g], [g for g, k in zip(all_genes, keep_g) if k]

    mean = X.mean(axis=0)
    var = X.var(axis=0)
    disp = var / np.maximum(mean, 1e-8)
    top = np.argsort(-disp)[: cfg.SYNTH_N_HVG_REFERENCE]

    target_col = next((i for i, g in enumerate(all_genes) if g.upper() == gene.upper()), None)
    # Target may be missing or all-zero after HVG filter; we still need a
    # dosage covariate for Song's Step 3.
    C_raw = (
        X[:, target_col].astype(float)
        if target_col is not None
        else np.zeros(X.shape[0], dtype=float)
    )
    wt_med = float(np.median(C_raw[is_ko == 0])) if (is_ko == 0).any() else 0.0

    # If the target is not expressed in WT (common for Replogle CRISPRi
    # essentials — SLU7 had WT median 0), regressing DEGs on C injects
    # nothing and Sp sticks at the noise floor. Fall back to a synthetic
    # dosage: WT = amp, KO = 0. Efficiency then sets C* = (1-psi)*amp.
    if wt_med < cfg.SYNTH_MIN_TARGET_WT_MEDIAN:
        C = np.where(is_ko == 0, amp, 0.0).astype(float)
        c_mode = f"synthetic dosage (WT={amp:g}, KO=0; {gene} WT median was {wt_med:.2f})"
    else:
        C = C_raw
        c_mode = f"{gene} counts (WT median {wt_med:.1f})"

    sel = sorted(set(top.tolist()) | ({target_col} if target_col is not None else set()))
    X_sel = X[:, sel]
    genes_sel = [all_genes[i] for i in sel]
    # Keep C aligned to the (possibly reordered) cell axis of X_sel
    # (same rows as X / is_ko).

    deg_ranked = rank_reference_degs(X_sel, genes_sel, is_ko)
    if cfg.SYNTH_EXCLUDE_TARGET_GENE:
        # Song excludes the perturbed gene from evaluation; it must also not be
        # a "downstream" gene of itself.
        deg_ranked = [g for g in deg_ranked if g.upper() != gene.upper()]

    print(
        f"    reference matrix: {X_sel.shape[0]} cells x {X_sel.shape[1]} genes "
        f"({is_ko.sum()} KO / {(1 - is_ko).sum()} WT); "
        f"C = {c_mode}",
        flush=True,
    )
    return {
        "counts": X_sel,
        "genes": genes_sel,
        "is_ko": is_ko,
        "C": C,
        "deg_ranked": deg_ranked,
        "target_gene": gene,
        "dataset": dataset,
        "label": ref["label"],
        "c_mode": c_mode,
        # Actual counts after QC; ground truth must be built at these sizes,
        # not at the config constants, or R skips every setting.
        "n_ko": int(is_ko.sum()),
        "n_wt": int((1 - is_ko).sum()),
    }


# ===========================================================================
# R BRIDGE
# ===========================================================================


def export_reference_for_r(refdata: dict, work: Path) -> None:
    """Write the reference matrix + covariates that the R script reads."""
    from scipy import io as spio
    from scipy import sparse

    work.mkdir(parents=True, exist_ok=True)
    # scDesign3 wants genes x cells
    spio.mmwrite(str(work / "reference_counts.mtx"), sparse.csc_matrix(refdata["counts"].T))
    (work / "reference_genes.txt").write_text("\n".join(refdata["genes"]) + "\n")
    pd.DataFrame(
        {
            "cell_id": [
                f"ref_{i:05d}_{KO if k else WT}"
                for i, k in enumerate(refdata["is_ko"])
            ],
            "K": [KO if k else WT for k in refdata["is_ko"]],
            "C": refdata["C"],
        }
    ).to_csv(work / "reference_covariates.csv", index=False)
    (work / "deg_ranked.txt").write_text("\n".join(refdata["deg_ranked"]) + "\n")


def export_setting_for_r(
    setting: SimSetting,
    cells: pd.DataFrame,
    programs: pd.DataFrame,
    work: Path,
) -> None:
    d = work / "settings" / setting.setting_id
    d.mkdir(parents=True, exist_ok=True)
    cells.to_csv(d / "cells.csv", index=False)
    programs.to_csv(d / "programs.csv", index=False)
    with open(d / "setting.json", "w") as f:
        json.dump(asdict(setting), f, indent=2)


R_TEMPLATE = r'''
# =====================================================================
# AUTO-GENERATED by synthetic_benchmark.py -- do not edit by hand.
# Song et al. 2025 (Nat Cell Biol) Perturb-seq simulation, 4 steps,
# via scDesign3 (Song, Wang, Yan et al. Nat Biotechnol 42:247, 2024).
# =====================================================================

# Colab: forked mclapply workers die when BLAS/OMP is multi-threaded.
Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1",
           OPENBLAS_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")

suppressPackageStartupMessages({
  if (!requireNamespace("scDesign3", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager", repos = "https://cloud.r-project.org")
    BiocManager::install("scDesign3", ask = FALSE, update = FALSE)
  }
  library(scDesign3)
  library(SingleCellExperiment)
  library(Matrix)
})

work      <- "__WORK__"
family    <- "__FAMILY__"
copula    <- "__COPULA__"
n_cores   <- __NCORES__
seed0     <- __SEED__
setting_ids <- c(__SETTING_IDS__)

# Force serial fitting on Colab-like hosts. n_cores>1 → mclapply returns NULL
# → "attempt to set an attribute on NULL" inside fit_marginal.
if (n_cores > 1) {
  cat(sprintf("NOTE: clamping n_cores %d → 1 (Colab-safe)\n", n_cores))
  n_cores <- 1L
}

cat(sprintf("scDesign3 %s  family=%s  n_cores=%d\n",
            as.character(packageVersion("scDesign3")), family, n_cores))

# ---------------------------------------------------------------------
# Reference SCE
# ---------------------------------------------------------------------
counts <- readMM(file.path(work, "reference_counts.mtx"))   # genes x cells
genes  <- readLines(file.path(work, "reference_genes.txt"))
cov    <- read.csv(file.path(work, "reference_covariates.csv"),
                   stringsAsFactors = FALSE)
deg_ranked <- readLines(file.path(work, "deg_ranked.txt"))

counts <- as(counts, "CsparseMatrix")
# gamlss/ZINB needs non-negative integers
counts@x <- pmax(round(counts@x), 0)
rownames(counts) <- genes
colnames(counts) <- cov$cell_id

cov$K <- factor(cov$K, levels = c("WT", "KO"))
rownames(cov) <- cov$cell_id

sce <- SingleCellExperiment(list(counts = counts), colData = cov)
cat(sprintf("reference: %d genes x %d cells (%d KO)\n",
            nrow(sce), ncol(sce), sum(cov$K == "KO")))

# ---------------------------------------------------------------------
# Step 1/2: marginals (per Song, DEG genes regress on C, others do not)
#           and a Gaussian copula fit separately for KO and WT cells.
# Cached: these dominate runtime and do not depend on the setting.
# ---------------------------------------------------------------------
fit_path <- file.path(work, "scdesign3_fit.rds")

fit_marginals_song <- function(family_use) {
  set.seed(seed0)
  dat <- construct_data(
    sce = sce, assay_use = "counts", celltype = "K",
    pseudotime = NULL, spatial = NULL, other_covariates = "C",
    corr_by = "K"
  )

  max_deg <- __MAX_DEG__
  deg_use <- intersect(deg_ranked[seq_len(min(max_deg, length(deg_ranked)))],
                       rownames(sce))
  cat(sprintf("marginals (%s): %d C-dependent DEG / %d C-independent\n",
              family_use, length(deg_use), nrow(sce) - length(deg_use)))

  # Fit ALL genes with intercept-only mean (stable; one construct_data).
  cat("  fit_marginal mu_formula=1 on all genes…\n")
  marg_all <- fit_marginal(
    data = dat, predictor = "gene",
    mu_formula = "1", sigma_formula = "1",
    family_use = family_use, n_cores = n_cores, usebam = FALSE
  )

  # Re-fit DEG genes with mean ~ C on a proper subset SCE (do NOT surgically
  # subset dat$count_mat — that leaves other construct_data fields inconsistent
  # and is what crashed mclapply with NULL results).
  if (length(deg_use) > 0) {
    cat(sprintf("  fit_marginal mu_formula=C on %d DEG genes…\n", length(deg_use)))
    sce_deg <- sce[deg_use, ]
    dat_deg <- construct_data(
      sce = sce_deg, assay_use = "counts", celltype = "K",
      pseudotime = NULL, spatial = NULL, other_covariates = "C",
      corr_by = "K"
    )
    marg_deg <- fit_marginal(
      data = dat_deg, predictor = "gene",
      mu_formula = "C", sigma_formula = "1",
      family_use = family_use, n_cores = n_cores, usebam = FALSE
    )
    for (g in names(marg_deg)) {
      marg_all[[g]] <- marg_deg[[g]]
    }
  }

  # Drop genes filtered out by construct_data / failed fits
  keep <- names(marg_all)
  keep <- keep[!vapply(marg_all, is.null, logical(1))]
  marg_all <- marg_all[keep]
  cat(sprintf("  kept %d / %d gene marginals\n", length(marg_all), nrow(sce)))

  set.seed(seed0)
  cat("  fit_copula…\n")
  # scDesign3 ≥1.10: important_feature is "all", a numeric zero-fraction
  # cutoff, or a logical vector — NOT "auto" (removed; that caused our crash).
  # 0.8 matches the old "auto" behaviour (drop genes with >80% zeros).
  important_feature <- 0.8
  cop <- fit_copula(
    sce = sce, assay_use = "counts", marginal_list = marg_all,
    family_use = family_use, copula = copula, n_cores = n_cores,
    input_data = dat$dat, important_feature = important_feature
  )
  list(dat = dat, marginal_list = marg_all, cop = cop, family = family_use,
       important_feature = important_feature)
}

# Do NOT name this `fit` — some loaded packages lock a binding of that name,
# and `fit <<- …` then errors with "cannot change value of locked binding".
if (file.exists(fit_path)) {
  cat("loading cached marginal+copula fit\n")
  scd_fit <- readRDS(fit_path)
} else {
  families_try <- unique(c(family, "nb", "zinb"))
  scd_fit <- NULL
  last_err <- NULL
  for (fam in families_try) {
    cat(sprintf("\n=== trying family_use=%s ===\n", fam))
    trial <- tryCatch(
      fit_marginals_song(fam),
      error = function(e) {
        last_err <<- conditionMessage(e)
        cat(sprintf("family %s FAILED: %s\n", fam, last_err))
        NULL
      }
    )
    if (!is.null(trial)) {
      scd_fit <- trial
      break
    }
  }
  if (is.null(scd_fit))
    stop(sprintf("All family attempts failed. Last error: %s", last_err))
  saveRDS(scd_fit, fit_path)
  cat(sprintf("cached marginal+copula fit (family=%s)\n", scd_fit$family))
}
# Prefer the family that actually fitted (may have fallen back).
family <- scd_fit$family

dat           <- scd_fit$dat
marginal_list <- scd_fit$marginal_list
cop           <- scd_fit$cop
base_cov      <- dat$dat

stopifnot("C" %in% colnames(base_cov))

# extract_para requires a corr_group column in new_covariate
if (!"corr_group" %in% colnames(base_cov)) {
  base_cov$corr_group <- as.character(base_cov$K)
}

extract_with_C <- function(Cvec) {
  nc <- base_cov
  nc$C <- Cvec
  extract_para(
    sce = sce, assay_use = "counts", marginal_list = marginal_list,
    n_cores = n_cores, family_use = family,
    new_covariate = nc, data = base_cov
  )
}

# ---------------------------------------------------------------------
# psi = 0 baseline. Cells/genes that do not respond take these means, and
# sigma/zero come from here throughout: Song does not modify dispersion or
# zero-inflation with efficiency.
# ---------------------------------------------------------------------
is_ko_ref <- base_cov$K == "KO"
wt_C      <- base_cov$C[!is_ko_ref]

para_zero_path <- file.path(work, "para_zero.rds")
if (file.exists(para_zero_path)) {
  para_zero <- readRDS(para_zero_path)
} else {
  set.seed(seed0)
  C_zero <- base_cov$C
  C_zero[is_ko_ref] <- sample(wt_C, sum(is_ko_ref), replace = TRUE)
  para_zero <- extract_with_C(C_zero)
  saveRDS(para_zero, para_zero_path)
}

sim_genes <- colnames(para_zero$mean_mat)
if (is.null(sim_genes)) sim_genes <- rownames(sce)

# ---------------------------------------------------------------------
# Steps 3/4 per setting
# ---------------------------------------------------------------------
for (sid in setting_ids) {
  sdir <- file.path(work, "settings", sid)
  out_mtx <- file.path(sdir, "synthetic_counts.mtx")
  if (file.exists(out_mtx)) { cat(sprintf("skip (exists): %s\n", sid)); next }

  cells <- read.csv(file.path(sdir, "cells.csv"), stringsAsFactors = FALSE)
  progs <- read.csv(file.path(sdir, "programs.csv"), stringsAsFactors = FALSE)

  # cells.csv is ordered KO-then-WT; align it to the reference cell order
  ko_rows <- which(cells$group == "KO")
  wt_rows <- which(cells$group == "WT")
  if (length(ko_rows) != sum(is_ko_ref) || length(wt_rows) != sum(!is_ko_ref)) {
    cat(sprintf("SKIP %s: cell counts (%d KO/%d WT) != reference (%d/%d)\n",
                sid, length(ko_rows), length(wt_rows),
                sum(is_ko_ref), sum(!is_ko_ref)))
    next
  }
  aligned <- integer(nrow(cells))
  aligned[which(is_ko_ref)]  <- ko_rows
  aligned[which(!is_ko_ref)] <- wt_rows
  cells <- cells[aligned, , drop = FALSE]

  # Step 3: efficiency enters as C* for perturbed cells
  C_eff <- base_cov$C
  C_eff[is_ko_ref] <- cells$C_star[is_ko_ref]
  set.seed(seed0)
  para_eff <- extract_with_C(C_eff)

  # Responder mask: a perturbed cell gets the perturbed mean only on the DEG
  # set of ITS OWN program. Everything else stays at the psi = 0 baseline.
  # effect_scale amplifies (para_eff - para_zero) so Sp has dynamic range in
  # PCA space (D genes in ~3000 is otherwise noise-dominated).
  effect_scale <- __EFFECT_SCALE__
  match_program_mag <- __MATCH_PROGRAM_MAG__
  mean_mat <- para_zero$mean_mat
  n_prog <- length(unique(progs$program))
  for (k in sort(unique(progs$program))) {
    cells_k <- which(is_ko_ref & cells$program == k)
    if (!length(cells_k)) next
    genes_k <- match(intersect(progs$gene[progs$program == k], sim_genes), sim_genes)
    genes_k <- genes_k[!is.na(genes_k)]
    if (!length(genes_k)) next
    signs_k <- progs$sign[progs$program == k][1]
    delta <- para_eff$mean_mat[cells_k, genes_k, drop = FALSE] -
             para_zero$mean_mat[cells_k, genes_k, drop = FALSE]
    if (signs_k < 0) delta <- -delta  # antagonistic program
    mean_mat[cells_k, genes_k] <- pmax(
      para_zero$mean_mat[cells_k, genes_k, drop = FALSE] + effect_scale * delta,
      0
    )
  }

  # Magnitude match (programs arm): rescale every KO cell's displacement from
  # its psi=0 baseline so ||mu_KO - mu_WT|| equals the 1-program target at this
  # efficiency. Ill-posed when the mixture mean is near zero (antagonistic
  # 50/50 +/-): skip rather than apply a huge scale_m. Note: match is in
  # gene-mean space; scored PCA magnitude can still differ across DEG sets.
  if (match_program_mag && n_prog > 1) {
    genes_0 <- match(intersect(progs$gene[progs$program == 0], sim_genes), sim_genes)
    genes_0 <- genes_0[!is.na(genes_0)]
    if (length(genes_0)) {
      mean_1prog <- para_zero$mean_mat
      cells_ko <- which(is_ko_ref)
      d0 <- para_eff$mean_mat[cells_ko, genes_0, drop = FALSE] -
            para_zero$mean_mat[cells_ko, genes_0, drop = FALSE]
      mean_1prog[cells_ko, genes_0] <- pmax(
        para_zero$mean_mat[cells_ko, genes_0, drop = FALSE] + effect_scale * d0,
        0
      )
      mu_wt <- colMeans(para_zero$mean_mat[!is_ko_ref, , drop = FALSE])
      mu_cur <- colMeans(mean_mat[is_ko_ref, , drop = FALSE])
      mu_tgt <- colMeans(mean_1prog[is_ko_ref, , drop = FALSE])
      cur_mag <- sqrt(sum((mu_cur - mu_wt)^2))
      tgt_mag <- sqrt(sum((mu_tgt - mu_wt)^2))
      ratio <- if (tgt_mag > 1e-12) cur_mag / tgt_mag else NA_real_
      if (!is.finite(cur_mag) || !is.finite(tgt_mag) || cur_mag <= 1e-12 ||
          tgt_mag <= 1e-12 || ratio < 0.25) {
        cat(sprintf(
          "  mag-match SKIP %s: cur=%.4g tgt=%.4g ratio=%s (ill-posed / near-cancel)\n",
          sid, cur_mag, tgt_mag, format(ratio)
        ))
      } else {
        scale_m <- min(tgt_mag / cur_mag, 5.0)
        for (i in cells_ko) {
          disp <- mean_mat[i, ] - para_zero$mean_mat[i, ]
          mean_mat[i, ] <- pmax(para_zero$mean_mat[i, ] + scale_m * disp, 0)
        }
        mu_after <- colMeans(mean_mat[is_ko_ref, , drop = FALSE])
        after_mag <- sqrt(sum((mu_after - mu_wt)^2))
        cat(sprintf(
          "  mag-match %s: cur=%.4g -> after=%.4g tgt=%.4g scale_m=%.3f\n",
          sid, cur_mag, after_mag, tgt_mag, scale_m
        ))
      }
    }
  }

  # Step 4
  set.seed(seed0)
  imp_feat <- if (!is.null(scd_fit$important_feature)) scd_fit$important_feature else 0.8
  newcount <- simu_new(
    sce = sce, assay_use = "counts",
    mean_mat = mean_mat,
    sigma_mat = para_zero$sigma_mat,
    zero_mat  = para_zero$zero_mat,
    quantile_mat = NULL,
    copula_list = cop$copula_list,
    n_cores = n_cores, family_use = family,
    nonnegative = TRUE, nonzerovar = TRUE,
    input_data = base_cov, new_covariate = base_cov,
    important_feature = imp_feat,
    filtered_gene = dat$filtered_gene
  )

  writeMM(as(as.matrix(newcount), "CsparseMatrix"), out_mtx)
  writeLines(rownames(newcount), file.path(sdir, "synthetic_genes.txt"))
  write.csv(cells, file.path(sdir, "cells_aligned.csv"), row.names = FALSE)
  cat(sprintf("wrote %s (%d genes x %d cells)\n",
              sid, nrow(newcount), ncol(newcount)))
}

cat("R DONE\n")
'''


def render_r_script(
    work: Path,
    settings: list[SimSetting],
    seed: int = cfg.SEED,
    family: Optional[str] = None,
    n_cores: Optional[int] = None,
) -> Path:
    ids = ", ".join(f'"{s.setting_id}"' for s in settings)
    script = (
        R_TEMPLATE.replace("__WORK__", str(work))
        .replace("__FAMILY__", family or cfg.SYNTH_FAMILY)
        .replace("__COPULA__", cfg.SYNTH_COPULA)
        .replace("__NCORES__", str(n_cores if n_cores is not None else cfg.SYNTH_R_N_CORES))
        .replace("__SEED__", str(seed))
        .replace("__MAX_DEG__", str(max(cfg.SONG_N_DEG)))
        .replace("__EFFECT_SCALE__", str(cfg.SYNTH_EFFECT_SCALE))
        .replace(
            "__MATCH_PROGRAM_MAG__",
            "TRUE" if cfg.SYNTH_MATCH_PROGRAM_MAGNITUDE else "FALSE",
        )
        .replace("__SETTING_IDS__", ids)
    )
    path = work / "run_scdesign3.R"
    path.write_text(script)
    return path


def run_r(script: Path, r_executable: str = cfg.SYNTH_R_EXECUTABLE) -> bool:
    """Run the generated R script; returns True on success."""
    try:
        proc = subprocess.run(
            [r_executable, str(script)],
            capture_output=True,
            text=True,
            timeout=cfg.SYNTH_R_TIMEOUT_S,
        )
    except FileNotFoundError:
        print(
            f"\n  {r_executable!r} not found. scDesign3 requires R.\n"
            "  Install R + scDesign3, or pass --r-executable /path/to/Rscript.\n"
            "  Use --dry-run to emit the R script and ground-truth tables without R.",
            flush=True,
        )
        return False

    for line in (proc.stdout or "").splitlines():
        if line.strip():
            print(f"    [R] {line.rstrip()}", flush=True)
    if proc.returncode != 0:
        print(f"    R FAILED (exit {proc.returncode}):", flush=True)
        print((proc.stderr or "")[-4000:], flush=True)
        return False
    return "R DONE" in (proc.stdout or "")


def read_synthetic(setting: SimSetting, work: Path) -> Optional[tuple]:
    """Read one simulated dataset back: (counts cells x genes, genes, cells)."""
    from scipy import io as spio

    d = work / "settings" / setting.setting_id
    mtx = d / "synthetic_counts.mtx"
    if not mtx.exists():
        return None
    X = spio.mmread(str(mtx))
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float).T
    genes = (d / "synthetic_genes.txt").read_text().split()
    aligned = d / "cells_aligned.csv"
    cells = pd.read_csv(aligned if aligned.exists() else d / "cells.csv")
    if X.shape[0] != len(cells):
        raise ValueError(
            f"{setting.setting_id}: {X.shape[0]} simulated cells vs {len(cells)} ground-truth rows"
        )
    return X, genes, cells


# ===========================================================================
# SCORING -- through the SAME code path as the manuscript
# ===========================================================================

PERT_LABEL = "SYNTH_PERT"
CTRL_LABEL = "control"


def build_synthetic_adata(
    setting: SimSetting,
    X: np.ndarray,
    genes: list[str],
    cells: pd.DataFrame,
    target_gene: str,
):
    """
    Assemble an AnnData of raw synthetic counts, subsampled to the setting's
    cell count, with the target gene dropped (Song excludes the perturbed gene
    from evaluation, so Sp cannot be driven by the target's own knockdown).
    """
    import anndata as ad

    rng = np.random.default_rng(setting.seed + 7)
    ko = np.flatnonzero((cells["group"] == KO).to_numpy())
    wt = np.flatnonzero((cells["group"] == WT).to_numpy())
    if len(ko) > setting.n_pert_cells:
        ko = rng.choice(ko, size=setting.n_pert_cells, replace=False)
    keep = np.sort(np.concatenate([ko, wt]))

    obs = cells.iloc[keep].reset_index(drop=True)
    obs["condition"] = np.where(obs["group"] == KO, PERT_LABEL, CTRL_LABEL)

    gene_mask = np.ones(len(genes), dtype=bool)
    if cfg.SYNTH_EXCLUDE_TARGET_GENE:
        gene_mask = np.array([g.upper() != target_gene.upper() for g in genes])

    adata = ad.AnnData(
        X=X[np.ix_(keep, np.flatnonzero(gene_mask))],
        obs=obs,
        var=pd.DataFrame(index=[g for g, m in zip(genes, gene_mask) if m]),
    )
    adata.obs_names = [str(c) for c in obs["cell_id"]]
    return adata


def score_synthetic(setting: SimSetting, adata) -> dict:
    """
    Sp and mean-shift magnitude via pipeline_core, i.e. the manuscript path:
    QC -> normalize -> log1p -> HVG -> PCA -> calculate_sp.

    Scoring in PC space obtained any other way would make Sp recover the
    injected shift trivially. min_cells is relaxed to 5 because the cell-count
    arm goes
    deliberately below the frozen MIN_CELLS of 50.

    Isolation from real-data freeze (CONFIG_VERSION):
      - Does NOT call materialize_min_cells (hash-stable downsample is N/A —
        the simulator already emits the exact cell counts requested).
      - Passes matrix_is_log=False explicitly (synthetic counts are raw);
        DATASETS[*] pins are never consulted (dataset_name=None).
      - Shares only preprocess/calculate_sp + N_PCS/N_HVG. Generator knobs
        live under SYNTHETIC_CONFIG_VERSION and must not be edited when
        bumping the real-data freeze.
    """
    import scanpy as sc

    from pipeline_core import calculate_sp, preprocess

    n_before = adata.n_obs
    # Synthetic X is raw counts — pin False; never fall through to the
    # unpinned heuristic (wrong/unstable on real Adamson; unsafe here too).
    adata, valid, _ = preprocess(
        adata,
        "condition",
        CTRL_LABEL,
        sc,
        n_pcs=cfg.N_PCS,
        min_cells=5,
        seed=setting.seed,
        valid_perts=[PERT_LABEL],
        counts=adata.obs["condition"].value_counts(),
        dataset_name=None,
        matrix_is_log=False,
    )
    if PERT_LABEL not in valid:
        return {"error": "perturbation dropped by QC"}
    if adata.n_obs < n_before:
        print(
            f"      note: QC dropped {n_before - adata.n_obs} synthetic cells",
            flush=True,
        )

    labels = adata.obs["condition"].astype(str).to_numpy()
    Xp = np.asarray(adata.obsm["X_pca"])
    metrics = calculate_sp(Xp[labels == CTRL_LABEL], Xp[labels == PERT_LABEL])
    return {
        "sp": metrics["stability"],
        "magnitude": metrics["magnitude"],
        "spread": metrics["spread"],
        "snr": metrics["snr"],
        "n_pert_scored": int((labels == PERT_LABEL).sum()),
        "n_ctrl_scored": int((labels == CTRL_LABEL).sum()),
        "adata": adata,
    }


# ===========================================================================
# METHODS UNDER COMPARISON
# ===========================================================================
# Only the REAL PS is used here. Euclidean and Mahalanobis "PS estimators"
# are not in Song et al. and would misrepresent the method.


def method_ps(adata) -> Optional[tuple[np.ndarray, list[str]]]:
    """Per-cell Song et al. PS via the pure-Python scMAGeCK port."""
    try:
        from song_ps_replication import compute_ps_python
    except Exception as e:  # pragma: no cover
        print(f"      PS unavailable ({e})", flush=True)
        return None

    cell_ps = compute_ps_python(adata, "condition", CTRL_LABEL, PERT_LABEL)
    if not cell_ps:
        return None
    order = [c for c in adata.obs_names if c in cell_ps]
    return np.array([cell_ps[c] for c in order]), order


def method_mixscape(adata):
    """
    STUB -- pertpy mixscape posterior probability per cell.

    Intended:
        import pertpy as pt
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "condition", CTRL_LABEL)
        ms.mixscape(adata, labels="condition", control=CTRL_LABEL)
    then read adata.obs["mixscape_class_p_ko"].

    Left stubbed on purpose: the pertpy Mixscape API has moved between
    versions and guessing it wrong is worse than an explicit gap. Pin the
    pertpy version first, confirm the obs column name, then implement.
    Mixscape is a named comparator for efficiency filtering, not optional.
    """
    raise NotImplementedError("method_mixscape: pin pertpy, confirm API, then implement")


# ===========================================================================
# METRICS (STUBS)
# ===========================================================================
# Each consumes the per-setting table written by run_grid() and/or the
# per-cell table. Contracts are fixed here so filling them in is mechanical.


def metric_efficiency_recovery(percell: pd.DataFrame) -> pd.DataFrame:
    """
    Song's Fig 2 metric. Per-cell accuracy at |psi_true - psi_pred| <= 0.1,
    per (n_deg, efficiency) cell, per method.

    Score against psi_effective, not psi_true: a cell in the non-responsive
    state genuinely has zero effective perturbation, and scoring it against
    psi_true would penalise a correct estimate.

    Implemented because it is unambiguous and validates our PS port against
    Song's published numbers.
    """
    required = {"setting_id", "arm", "n_deg", "efficiency", "method", "psi_effective", "psi_pred"}
    missing = required - set(percell.columns)
    if missing:
        raise ValueError(f"metric_efficiency_recovery: missing columns {sorted(missing)}")

    df = percell.dropna(subset=["psi_pred", "psi_effective"]).copy()
    df["abs_err"] = (df["psi_pred"] - df["psi_effective"]).abs()
    df["correct"] = df["abs_err"] <= 0.1
    return (
        df.groupby(["arm", "n_deg", "efficiency", "method"], as_index=False)
        .agg(
            pct_correct=("correct", lambda s: 100.0 * float(np.mean(s))),
            mean_abs_err=("abs_err", "mean"),
            n_cells=("correct", "size"),
        )
        .sort_values(["arm", "n_deg", "efficiency", "method"])
    )


def _mannwhitney_auroc(scores: np.ndarray, is_pos: np.ndarray) -> float:
    """AUROC = Mann–Whitney U / (n_pos * n_neg). NaN if either class empty."""
    from scipy.stats import mannwhitneyu

    pos = np.asarray(scores, dtype=float)[np.asarray(is_pos, dtype=bool)]
    neg = np.asarray(scores, dtype=float)[~np.asarray(is_pos, dtype=bool)]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 1 or len(neg) < 1:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def _magnitude_matched_pairs(
    mix: pd.DataFrame,
    multi: pd.DataFrame,
    mag_tol_rel: float = 0.15,
) -> pd.DataFrame:
    """
    Greedy 1:1 matches: each multi-program setting paired to the closest
    efficiency-mixture setting within relative magnitude tolerance.
    """
    if mix.empty or multi.empty:
        return pd.DataFrame()
    mix = mix.reset_index(drop=True).copy()
    multi = multi.reset_index(drop=True).copy()
    used_mix: set[int] = set()
    rows = []
    # Match hardest (extreme magnitude) first so mid-range doesn't steal all partners
    order = multi["magnitude"].sub(mix["magnitude"].median()).abs().sort_values(
        ascending=False
    ).index
    for mi in order:
        mrow = multi.loc[mi]
        mag_m = float(mrow["magnitude"])
        denom = max(abs(mag_m), 1e-9)
        best_j, best_d = None, float("inf")
        for j, xrow in mix.iterrows():
            if j in used_mix:
                continue
            d = abs(float(xrow["magnitude"]) - mag_m) / denom
            if d <= mag_tol_rel and d < best_d:
                best_j, best_d = j, d
        if best_j is None:
            continue
        used_mix.add(best_j)
        xrow = mix.loc[best_j]
        rows.append(
            {
                "multi_setting_id": mrow["setting_id"],
                "mix_setting_id": xrow["setting_id"],
                "multi_sp": float(mrow["sp"]),
                "mix_sp": float(xrow["sp"]),
                "multi_magnitude": float(mrow["magnitude"]),
                "mix_magnitude": float(xrow["magnitude"]),
                "mag_rel_diff": best_d,
                "multi_n_programs": int(mrow["n_programs"]),
                "multi_program_overlap": float(mrow["program_overlap"]),
                "mix_efficiency": float(xrow["efficiency"]),
                "mix_efficiency_dist": xrow["efficiency_dist"],
                "sp_delta_multi_minus_mix": float(mrow["sp"]) - float(xrow["sp"]),
            }
        )
    return pd.DataFrame(rows)


def metric_identifiability(
    per_setting: pd.DataFrame,
    mag_tol_rel: float = 0.15,
) -> pd.DataFrame:
    """
    Magnitude-matched regime separation (the manuscript-critical test).

    Regime mix  : efficiency_dist in {bimodal, beta_broad}, n_programs==1,
                  efficiency < 1 (mixtures that actually dilute coherence).
    Regime multi: arm==programs, n_programs>1, program_overlap<=0.5,
                  efficiency==1.0 (directional diversity at full response).

    Hold magnitude matched (relative tol), then ask whether Sp separates the
    two regimes. Magnitude-alone AUROC is the control (~0.5 if matching worked).

    Returns a one-row summary plus writes pair detail via the caller if needed.
    Pair table is attached as attrs['pairs'] when present.
    """
    required = {
        "setting_id", "arm", "n_programs", "program_overlap",
        "efficiency", "efficiency_dist", "sp", "magnitude",
    }
    missing = required - set(per_setting.columns)
    if missing:
        raise ValueError(f"metric_identifiability: missing columns {sorted(missing)}")

    df = per_setting.dropna(subset=["sp", "magnitude"]).copy()
    mix = df[
        (df["arm"] == "efficiency_dist")
        & (df["n_programs"] == 1)
        & (df["efficiency_dist"].isin(["bimodal", "beta_broad"]))
        & (df["efficiency"] < 1.0)
    ]
    multi = df[
        (df["arm"] == "programs")
        & (df["n_programs"] > 1)
        & (df["program_overlap"] <= 0.5)
        & (df["efficiency"] == 1.0)
    ]
    pairs = _magnitude_matched_pairs(mix, multi, mag_tol_rel=mag_tol_rel)
    if pairs.empty:
        out = pd.DataFrame(
            [
                {
                    "n_mix_candidates": int(len(mix)),
                    "n_multi_candidates": int(len(multi)),
                    "n_matched_pairs": 0,
                    "mag_tol_rel": mag_tol_rel,
                    "auroc_sp": float("nan"),
                    "auroc_magnitude": float("nan"),
                    "median_sp_multi": float("nan"),
                    "median_sp_mix": float("nan"),
                    "median_sp_delta": float("nan"),
                    "median_mag_rel_diff": float("nan"),
                    "sp_separates_regimes": False,
                    "note": "no magnitude-matched pairs; need both arms in one table",
                }
            ]
        )
        out.attrs["pairs"] = pairs
        return out

    # Stack matched settings: label multi=1, mix=0
    sp_scores = np.concatenate(
        [pairs["multi_sp"].to_numpy(), pairs["mix_sp"].to_numpy()]
    )
    mag_scores = np.concatenate(
        [pairs["multi_magnitude"].to_numpy(), pairs["mix_magnitude"].to_numpy()]
    )
    is_multi = np.array([True] * len(pairs) + [False] * len(pairs))

    auroc_sp = _mannwhitney_auroc(sp_scores, is_multi)
    auroc_mag = _mannwhitney_auroc(mag_scores, is_multi)
    # Sp "separates" if AUROC is away from 0.5 more than magnitude control
    # (either direction: multi may have lower Sp). Use |AUROC-0.5|.
    sp_sep = abs(auroc_sp - 0.5) > abs(auroc_mag - 0.5) + 0.05 and abs(auroc_sp - 0.5) >= 0.15

    out = pd.DataFrame(
        [
            {
                "n_mix_candidates": int(len(mix)),
                "n_multi_candidates": int(len(multi)),
                "n_matched_pairs": int(len(pairs)),
                "mag_tol_rel": mag_tol_rel,
                "auroc_sp": auroc_sp,
                "auroc_magnitude": auroc_mag,
                "median_sp_multi": float(pairs["multi_sp"].median()),
                "median_sp_mix": float(pairs["mix_sp"].median()),
                "median_sp_delta": float(pairs["sp_delta_multi_minus_mix"].median()),
                "median_mag_rel_diff": float(pairs["mag_rel_diff"].median()),
                "sp_separates_regimes": bool(sp_sep),
                "note": (
                    "Sp separates magnitude-matched regimes"
                    if sp_sep
                    else "Sp does not beat magnitude at separating regimes"
                ),
            }
        ]
    )
    out.attrs["pairs"] = pairs
    return out


def metric_coherence_auroc(per_setting: pd.DataFrame) -> pd.DataFrame:
    """
    STUB -- within-programs-arm novelty claim (secondary to identifiability).

    AUROC for discriminating n_programs == 1 from n_programs > 1 using each
    method's per-setting score, WITHIN strata matched on magnitude and
    efficiency (otherwise the AUROC just re-measures effect size).

    Prefer metric_identifiability first: that compares efficiency mixtures to
    multi-program biology directly.
    """
    raise NotImplementedError("metric_coherence_auroc: see docstring contract")


def metric_realism_calibration(
    per_setting: pd.DataFrame,
    observed_sp: Optional[float] = None,
    observed_label: str = "reference",
) -> pd.DataFrame:
    """
    Gate: does 1-program full-efficiency simulated Sp sit near the real
    reference Sp? Full distributional checks (library size, dropout, DEG
    count) can extend this later.

    Pass rule: |sim_median - observed| <= SYNTH_REALISM_MAX_ABS_GAP
    AND sim_median <= SYNTH_REALISM_MAX_SIM_SP. Thresholds are returned in the
    row so the gate is not a silent magic number.
    """
    max_gap = float(cfg.SYNTH_REALISM_MAX_ABS_GAP)
    max_sim = float(cfg.SYNTH_REALISM_MAX_SIM_SP)
    df = per_setting.dropna(subset=["sp"]).copy()
    mask = (df["n_programs"] == 1) & (df["efficiency"] == 1.0)
    if "program_overlap" in df.columns:
        mask = mask & (df["program_overlap"] == 1.0)
    one = df.loc[mask]
    # Prefer homogeneous / programs 1-prog cells
    if "efficiency_dist" in one.columns:
        pref = one[one["efficiency_dist"] == "homogeneous"]
        if len(pref):
            one = pref
    sim_med = float(one["sp"].median()) if len(one) else float("nan")
    obs = float(observed_sp) if observed_sp is not None else float("nan")
    gap = abs(sim_med - obs) if np.isfinite(sim_med) and np.isfinite(obs) else float("nan")
    covers = bool(
        np.isfinite(sim_med)
        and sim_med <= max_sim
        and (not np.isfinite(gap) or gap <= max_gap)
    )
    return pd.DataFrame(
        [
            {
                "quantity": "sp_one_program_full_eff",
                "observed_label": observed_label,
                "observed": obs,
                "simulated_median": sim_med,
                "simulated_n": int(len(one)),
                "abs_gap": gap,
                "max_abs_gap": max_gap,
                "max_sim_sp": max_sim,
                "covers_real_range": covers,
            }
        ]
    )


# Filenames this script must never overwrite (real-data freeze / pathway / QC).
_FORBIDDEN_WRITE_NAMES = frozenset(
    {
        "frozen_sp_scores.csv",
        "frozen_sp_scores_sample.csv",
        "frozen_sp_scores_summary.json",
        "shesha_crispr_results_euclidean.csv",
        "pathway_signature_correlations.csv",
        "pathway_scores_per_pert.csv",
        "cell_quality_partials.csv",
        "cell_quality_per_perturbation.csv",
    }
)


def _assert_synth_write_ok(path: Path) -> None:
    name = Path(path).name
    if name in _FORBIDDEN_WRITE_NAMES or (
        name.startswith("frozen_sp") and not name.startswith("synthetic_")
    ):
        raise RuntimeError(
            f"synthetic_benchmark refuses to write {path} — real-data / pathway "
            "artifact. Synth outputs must be synthetic_benchmark_*.csv only."
        )


def upsert_by_setting_id(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new rows so arm runs do not erase prior arms.

    Key is (setting_id, synthetic_config_version) when version is present, so
    effect_scale / generator changes do not silently overwrite or mix with
    older scored rows under the same setting_id.
    """
    _assert_synth_write_ok(path)
    if new_df is None or not len(new_df):
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    if path.exists():
        old = pd.read_csv(path)
        if "setting_id" in old.columns and "setting_id" in new_df.columns:
            if (
                "synthetic_config_version" in old.columns
                and "synthetic_config_version" in new_df.columns
            ):
                new_keys = set(
                    zip(new_df["setting_id"], new_df["synthetic_config_version"])
                )
                keep = [
                    (sid, ver) not in new_keys
                    for sid, ver in zip(
                        old["setting_id"], old["synthetic_config_version"]
                    )
                ]
                old = old.loc[keep]
            else:
                old = old[~old["setting_id"].isin(set(new_df["setting_id"]))]
            out = pd.concat([old, new_df], ignore_index=True)
        else:
            out = new_df.copy()
    else:
        out = new_df.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _arm_slug(df: pd.DataFrame) -> str:
    arms = sorted(df["arm"].astype(str).unique()) if "arm" in df.columns else ["na"]
    return "+".join(arms) if arms else "na"


# ===========================================================================
# RUNNER
# ===========================================================================


def run_grid(
    refdata: dict,
    settings: list[SimSetting],
    work: Path,
    out_dir: Path,
    dry_run: bool = False,
    r_executable: str = cfg.SYNTH_R_EXECUTABLE,
    with_ps: bool = True,
    family: Optional[str] = None,
    r_n_cores: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export -> simulate in R -> score through the frozen pipeline."""
    n_ko, n_wt = refdata["n_ko"], refdata["n_wt"]
    print(
        f"\n  Exporting reference ({n_ko} KO / {n_wt} WT cells) + "
        f"{len(settings)} settings to {work}",
        flush=True,
    )
    export_reference_for_r(refdata, work)
    wt_C = refdata["C"][refdata["is_ko"] == 0]
    for s in settings:
        cells, programs = build_ground_truth(
            s, refdata["deg_ranked"], wt_C, n_pert=n_ko, n_ctrl=n_wt
        )
        export_setting_for_r(s, cells, programs, work)

    script = render_r_script(
        work, settings, family=family, n_cores=r_n_cores
    )
    print(f"  Wrote {script}", flush=True)

    if dry_run:
        print(
            "\n  --dry-run: stopping before R. Inspect the script and the\n"
            f"  ground-truth tables under {work / 'settings'}.",
            flush=True,
        )
        return pd.DataFrame(), pd.DataFrame()

    if not run_r(script, r_executable=r_executable):
        raise RuntimeError("scDesign3 simulation failed; see R output above")

    rows, percell_rows = [], []
    for i, s in enumerate(settings, 1):
        got = read_synthetic(s, work)
        if got is None:
            print(f"    [{i}/{len(settings)}] MISSING output: {s.setting_id}", flush=True)
            continue
        X, genes, cells = got
        adata = build_synthetic_adata(s, X, genes, cells, refdata["target_gene"])
        scored = score_synthetic(s, adata)
        if "error" in scored:
            print(f"    [{i}/{len(settings)}] {scored['error']}: {s.setting_id}", flush=True)
            continue
        scored_adata = scored.pop("adata")

        row = asdict(s)
        row.update(
            {
                "setting_id": s.setting_id,
                "synthetic_config_version": cfg.SYNTHETIC_CONFIG_VERSION,
                "config_version": cfg.CONFIG_VERSION,
                "effect_scale": float(cfg.SYNTH_EFFECT_SCALE),
                "match_program_magnitude": bool(cfg.SYNTH_MATCH_PROGRAM_MAGNITUDE),
                "reference_dataset": refdata["dataset"],
                "reference_gene": refdata["target_gene"],
                "mean_psi_effective": float(
                    cells.loc[cells["group"] == KO, "psi_effective"].mean()
                ),
                **scored,
            }
        )

        if with_ps:
            ps = method_ps(scored_adata)
            if ps is not None:
                ps_vals, ps_cells = ps
                row["ps_mean"] = float(np.mean(ps_vals))
                truth = cells.set_index("cell_id")["psi_effective"]
                for cid, v in zip(ps_cells, ps_vals):
                    percell_rows.append(
                        {
                            "setting_id": s.setting_id,
                            "arm": s.arm,
                            "n_deg": s.n_deg,
                            "efficiency": s.efficiency,
                            "efficiency_dist": s.efficiency_dist,
                            "method": "PS",
                            "cell_id": cid,
                            "psi_effective": float(truth.get(cid, np.nan)),
                            "psi_pred": float(v),
                        }
                    )
            else:
                row["ps_mean"] = np.nan

        rows.append(row)
        if i % 10 == 0 or i == len(settings):
            print(f"    [{i}/{len(settings)}] scored", flush=True)

    per_setting = pd.DataFrame(rows)
    percell = pd.DataFrame(percell_rows)

    tag = refdata["label"]
    ver = cfg.SYNTHETIC_CONFIG_VERSION.replace(".", "")
    if len(per_setting):
        slug = _arm_slug(per_setting)
        # Immutable per-run snapshot (arm + config) — never overwrite prior arms
        snap = out_dir / f"synthetic_benchmark_settings_{tag}__{slug}__{ver}.csv"
        per_setting.to_csv(snap, index=False)
        print(f"  Wrote snapshot {snap} ({len(per_setting)} rows)")
        # Cumulative per-reference table (upsert by setting_id)
        cum = out_dir / f"synthetic_benchmark_settings_{tag}.csv"
        merged = upsert_by_setting_id(cum, per_setting)
        print(f"  Upserted {cum} (now {len(merged)} rows)")
    if len(percell):
        slug = _arm_slug(percell) if len(percell) else "na"
        snap = out_dir / f"synthetic_benchmark_percell_{tag}__{slug}__{ver}.csv"
        percell.to_csv(snap, index=False)
        cum = out_dir / f"synthetic_benchmark_percell_{tag}.csv"
        upsert_by_setting_id(cum, percell)
        print(f"  Wrote per-cell snapshot {snap} ({len(percell)} rows)")
    return per_setting, percell


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        nargs="*",
        default=None,
        help="Sweeps to run: song_replication efficiency_dist programs state "
        "cell_count (default: all)",
    )
    parser.add_argument(
        "--reference",
        nargs="*",
        default=None,
        help="Reference labels to use (default: all of low_sp mid_sp high_sp)",
    )
    parser.add_argument(
        "--reference-gene",
        default=None,
        help="Override reference selection with an explicit gene",
    )
    parser.add_argument(
        "--reference-dataset",
        default=cfg.SYNTH_REFERENCE_DATASETS[0],
        help="Dataset for --reference-gene",
    )
    parser.add_argument("--sp-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=cfg.CACHE_DIR / "synthetic_benchmark")
    parser.add_argument("--n-seeds", type=int, default=cfg.SYNTH_N_SEEDS)
    parser.add_argument("--r-executable", default=cfg.SYNTH_R_EXECUTABLE)
    parser.add_argument("--no-ps", action="store_true", help="Skip PS (much faster)")
    parser.add_argument(
        "--family",
        default=cfg.SYNTH_FAMILY,
        choices=["nb", "zinb", "poisson", "zip"],
        help="scDesign3 marginal family (default nb; Song used zinb)",
    )
    parser.add_argument(
        "--r-n-cores",
        type=int,
        default=cfg.SYNTH_R_N_CORES,
        help="R fit_marginal cores (default 1; Colab breaks with >1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit the R script and ground-truth tables, then stop. No R needed.",
    )
    parser.add_argument("--self-test", action="store_true", help="Validate grid logic and exit")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Report which stages are runnable in this environment, then exit",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.check_env:
        ok = check_env(resolve_out_dir(args.out_dir), args.r_executable)
        sys.exit(0 if ok else 1)

    out_dir = resolve_out_dir(args.out_dir)
    work_root = Path(args.work_dir)

    print("=" * 72)
    print("SYNTHETIC BENCHMARK")
    print(f"synthetic_config_version = {cfg.SYNTHETIC_CONFIG_VERSION}")
    print(f"config_version           = {cfg.CONFIG_VERSION}")
    print(f"manuscript_usable        = {cfg.SYNTH_MANUSCRIPT_USABLE}")
    print(f"effect_scale             = {cfg.SYNTH_EFFECT_SCALE}")
    print(f"match_program_magnitude  = {cfg.SYNTH_MATCH_PROGRAM_MAGNITUDE}")
    print(f"simulator                = scDesign3 ({cfg.SYNTH_FAMILY}, {cfg.SYNTH_COPULA} copula)")
    print(f"out_dir                  = {out_dir}")
    print(f"work_dir                 = {work_root}")
    print(
        "version contract: SYNTHETIC_* owns generator; CONFIG_VERSION owns "
        "preprocess/Sp only. Scoring uses matrix_is_log=False, no materialize. "
        "Writes only synthetic_benchmark_*.csv — will not touch frozen_sp / pathway / QC."
    )
    if not cfg.SYNTH_MANUSCRIPT_USABLE:
        print(
            "POLICY: arm closed for manuscript citation. Realism is "
            "reference-dependent under a fixed effect_scale; do not re-dial "
            "scale to chase the current freeze's auto-selected gene."
        )
    print("=" * 72)
    if (
        cfg.SYNTHETIC_CONFIG_VERSION < "2026-07-25.9"
        or abs(float(cfg.SYNTH_EFFECT_SCALE) - 1.5) > 1e-9
        or bool(cfg.SYNTH_MATCH_PROGRAM_MAGNITUDE)
        or bool(getattr(cfg, "SYNTH_MANUSCRIPT_USABLE", True))
    ):
        print(
            "WARNING: unexpected generator policy (need synthetic_config_version>=2026-07-25.9, "
            "effect_scale=1.5, match_program_magnitude=False, manuscript_usable=False). "
            "Re-upload pipeline_config.py.",
            flush=True,
        )
    # Anchor check: realism compares to observed Sp in frozen_sp_scores.csv.
    # After a real-data CONFIG_VERSION regen, that CSV's Sp values may move;
    # never keep a hardcoded SLU7=0.514 from an older freeze.
    try:
        from revision_io import find_sp_csv, load_sp_table

        _sp_path = args.sp_csv or find_sp_csv(out_dir)
        _sp_df = load_sp_table(_sp_path)
        _sum = out_dir / "frozen_sp_scores_summary.json"
        _csv_ver = None
        if _sum.exists():
            import json as _json

            _csv_ver = _json.loads(_sum.read_text()).get("config_version")
        print(
            f"observed-Sp anchor: {_sp_path.name}  rows={len(_sp_df)}  "
            f"csv_config_version={_csv_ver or 'unknown'}  "
            f"runtime_config_version={cfg.CONFIG_VERSION}",
            flush=True,
        )
        if _csv_ver and _csv_ver != cfg.CONFIG_VERSION:
            print(
                f"WARNING: frozen Sp table stamped {_csv_ver} but runtime "
                f"CONFIG_VERSION={cfg.CONFIG_VERSION}. Realism gate will use "
                "whatever Sp is in the CSV — regen Sp or pass the matching "
                "--sp-csv. Do not bump SYNTHETIC_CONFIG_VERSION for this.",
                flush=True,
            )
    except Exception as _e:
        print(f"observed-Sp anchor: not resolved yet ({_e})", flush=True)

    if args.reference_gene:
        refs = {
            "custom": {
                "label": "custom",
                "dataset": cfg.resolve_dataset_name(args.reference_dataset),
                "gene": args.reference_gene,
            }
        }
    else:
        print("\nSelecting reference perturbations spanning the observed Sp range:")
        refs = select_reference_perturbations(out_dir, args.sp_csv)
        if args.reference:
            refs = {k: v for k, v in refs.items() if k in set(args.reference)}
            if not refs:
                raise ValueError(f"No reference matched {args.reference}")

    all_settings, all_percell = [], []
    for label, ref in refs.items():
        print(f"\n{'-' * 72}\nREFERENCE {label}: {ref['gene']} ({ref['dataset']})\n{'-' * 72}")
        candidates = [ref] + [
            {**ref, **alt, "label": label} for alt in ref.get("alternates", [])
        ]
        refdata = None
        last_err: Optional[Exception] = None
        for cand in candidates:
            try:
                if cand is not ref:
                    print(
                        f"  trying alternate: {cand['gene']} "
                        f"(Sp={cand.get('observed_sp', float('nan')):.3f})",
                        flush=True,
                    )
                refdata = build_reference_matrices(
                    cand, n_ko=SIM_N_PERT, n_wt=SIM_N_CTRL
                )
                ref = cand
                break
            except ValueError as e:
                last_err = e
                print(f"  skip: {e}", flush=True)
        if refdata is None:
            raise RuntimeError(
                f"No usable reference for {label}. Last error: {last_err}"
            )
        settings = build_grid(label, arms=args.arm, n_seeds=args.n_seeds)
        print(f"  grid: {len(settings)} simulated datasets")
        per_setting, percell = run_grid(
            refdata,
            settings,
            work=work_root / label,
            out_dir=out_dir,
            dry_run=args.dry_run,
            r_executable=args.r_executable,
            with_ps=not args.no_ps,
            family=args.family,
            r_n_cores=args.r_n_cores,
        )
        if len(per_setting):
            all_settings.append(per_setting)
        if len(percell):
            all_percell.append(percell)

    if not all_settings:
        print("\nNo settings scored (dry run, or simulation produced no output).")
        return

    settings_df = pd.concat(all_settings, ignore_index=True)
    ver = cfg.SYNTHETIC_CONFIG_VERSION.replace(".", "")
    slug = _arm_slug(settings_df)
    snap = out_dir / f"synthetic_benchmark_settings__{slug}__{ver}.csv"
    n_snap = len(settings_df)
    settings_df.to_csv(snap, index=False)
    settings_path = out_dir / "synthetic_benchmark_settings.csv"
    upsert_by_setting_id(settings_path, settings_df)
    # Metrics run on THIS run only (current config), not a mixed cumulative table
    print(f"\nWrote snapshot {snap} ({n_snap} rows)")
    print(f"Upserted {settings_path}")
    print(
        "  NOTE: cumulative CSVs mix synthetic_config_version rows; "
        "filter on version/effect_scale or use the snapshot above.",
        flush=True,
    )

    if all_percell:
        percell_df = pd.concat(all_percell, ignore_index=True)
        percell_path = out_dir / "synthetic_benchmark_percell.csv"
        upsert_by_setting_id(percell_path, percell_df)
        print(f"Upserted {percell_path}")

        try:
            eff = metric_efficiency_recovery(percell_df)
            eff_path = out_dir / "synthetic_benchmark_efficiency_recovery.csv"
            eff.to_csv(eff_path, index=False)
            print(f"Wrote {eff_path}")
        except ValueError as e:
            print(f"  skip efficiency_recovery: {e}", flush=True)

    def _pivot2(df, index, columns, values):
        return (
            df.groupby([index, columns], as_index=False)[values]
            .median()
            .pivot(index=index, columns=columns, values=values)
            .round(3)
        )

    song = settings_df[settings_df["arm"] == "song_replication"]
    if len(song):
        print("\nSong replication: Sp by n_deg × efficiency (homogeneous):")
        print(_pivot2(song, "n_deg", "efficiency", "sp").to_string())
        print("Song replication: magnitude by n_deg × efficiency:")
        print(_pivot2(song, "n_deg", "efficiency", "magnitude").to_string())

    edist = settings_df[settings_df["arm"] == "efficiency_dist"]
    if len(edist):
        print("\nSp by efficiency distribution (D=ref, efficiency confound):")
        print(_pivot2(edist, "efficiency_dist", "efficiency", "sp").to_string())
        print("Magnitude by efficiency distribution:")
        print(_pivot2(edist, "efficiency_dist", "efficiency", "magnitude").to_string())

    prog = settings_df[settings_df["arm"] == "programs"]
    if len(prog):
        print("\nSp vs multi-program structure (programs arm):")
        print(
            prog.groupby(["n_programs", "program_overlap"], as_index=False)[
                ["sp", "magnitude"]
            ]
            .median()
            .round(4)
            .to_string(index=False)
        )
        print(
            "  NOTE: ov=1.0 rows are code-path identities with 1-program "
            "(same DEG set) — not independent cells. "
            "n_programs=3 × ov=-1 was removed in .6 (unbalanced +/- bug).",
            flush=True,
        )
        one = prog.loc[prog["n_programs"] == 1]
        # Genuinely multi: exclude ov=1 identity; keep ov<=0.5 and ov=-1
        multi = prog[
            (prog["n_programs"] > 1) & (prog["program_overlap"] < 1.0)
        ]
        if len(one) and len(multi):
            print(
                f"  all non-identity multi: 1-prog Sp={one['sp'].median():.3f}; "
                f"multi Sp={multi['sp'].median():.3f}; "
                f"delta={one['sp'].median() - multi['sp'].median():+.3f} "
                f"(n_multi={len(multi)}; includes mag-mismatched cells)",
                flush=True,
            )
            tgt_mag = float(one["magnitude"].median())
            tol = float(cfg.SYNTH_MAG_MATCH_REL_TOL)
            # Scored-magnitude-matched subset (the number worth having)
            rel = (multi["magnitude"] - tgt_mag).abs() / max(tgt_mag, 1e-9)
            matched = multi.loc[rel <= tol]
            print(
                f"  scored-mag-matched (rel_err<={tol}, exclude ov=1): "
                f"n={len(matched)}/{len(multi)}",
                flush=True,
            )
            if len(matched):
                print(
                    f"    1-prog Sp={one['sp'].median():.3f}; "
                    f"matched multi Sp={matched['sp'].median():.3f}; "
                    f"delta={one['sp'].median() - matched['sp'].median():+.3f}",
                    flush=True,
                )
                for _, r in matched.sort_values(
                    ["n_programs", "program_overlap"]
                ).iterrows():
                    print(
                        f"    K={int(r['n_programs'])} ov={r['program_overlap']:+.1f}: "
                        f"Sp={r['sp']:.3f} mag={r['magnitude']:.2f}",
                        flush=True,
                    )
            else:
                print(
                    "    no cells within tol — cannot claim a magnitude-held "
                    "multi-program Sp delta yet.",
                    flush=True,
                )

    # Realism gate + identifiability (need programs + efficiency_dist in one table)
    obs_sp, obs_label = None, "reference"
    if refs:
        first_ref = next(iter(refs.values()))
        obs_sp = first_ref.get("observed_sp")
        obs_label = f"{first_ref.get('gene', 'ref')} ({first_ref.get('label', '')})"

    realism = metric_realism_calibration(
        settings_df, observed_sp=obs_sp, observed_label=obs_label
    )
    realism["manuscript_usable"] = bool(cfg.SYNTH_MANUSCRIPT_USABLE)
    realism_path = out_dir / "synthetic_benchmark_realism.csv"
    _assert_synth_write_ok(realism_path)
    realism.to_csv(realism_path, index=False)
    print(f"\nRealism gate ({realism_path}):")
    print(realism.to_string(index=False))
    print(
        f"  pass rule: abs_gap <= {cfg.SYNTH_REALISM_MAX_ABS_GAP} "
        f"AND simulated_median <= {cfg.SYNTH_REALISM_MAX_SIM_SP} "
        f"(no portable scale→Sp curve — calibration is reference-dependent)",
        flush=True,
    )
    gate_ok = bool(realism.iloc[0]["covers_real_range"])
    if not gate_ok:
        print(
            "  FAIL: simulated 1-program Sp not in real range — "
            "do not treat arm tables as manuscript numbers.",
            flush=True,
        )
    elif int(realism.iloc[0]["simulated_n"]) < 3:
        print(
            "  NOTE: simulated_n is small (duplicate 1-prog cells across arms); "
            "gate rests on few rows.",
            flush=True,
        )
    if not cfg.SYNTH_MANUSCRIPT_USABLE:
        print(
            "\nVERDICT (SYNTH_MANUSCRIPT_USABLE=False): controlled scDesign3 "
            "benchmark attempted; not usable for manuscript numbers. "
            "effect_scale that hits realism is reference-dependent (auto-selected "
            "gene + dosage regime change when frozen Sp is regenerated), and "
            "holding programs-arm magnitude fixed is not available without "
            "confounding Sp. Do not re-tune effect_scale to the current reference.",
            flush=True,
        )
    elif not gate_ok:
        print(
            "\nVERDICT: realism failed; arm tables blocked even though "
            "SYNTH_MANUSCRIPT_USABLE is True — investigate before citing.",
            flush=True,
        )

    if not cfg.SYNTH_MATCH_PROGRAM_MAGNITUDE:
        print(
            "\nIdentifiability: deferred — generator does not hold program-arm "
            "magnitude fixed (Sp tracks magnitude by construction).",
            flush=True,
        )
    else:
        print(
            "\nIdentifiability: parked until scored-mag-matched multi-program "
            "grid is dense (not reporting AUROC on 1–few pairs).",
            flush=True,
        )

    print("\nClosed / stubbed (not manuscript deliverables):")
    print("  efficiency_dist / programs tables  — realism + reference-dependence")
    print("  metric_coherence_auroc             — stub")
    print("  method_mixscape                    — stub (#24)")
    print("  mid_sp / low_sp refs               — would repeat reference-dependence")


if __name__ == "__main__":
    main()
