"""
Frozen analysis configuration.

All manuscript-facing scripts should import parameters from here so that
tables, figures, and ablations share one preprocessing/filtering definition.
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

# ---------------------------------------------------------------------------
# Version / paths
# ---------------------------------------------------------------------------

# Canonical freeze for ALL real-data manuscript tables (Sp, pathway, stress,
# partials, CORUM, …). One label only — quote this in CSVs.
CONFIG_VERSION = "2026-07-29.1"

# Supersession (do not mix numbers across these):
#   2026-07-17.5  — heuristic-era Adamson skip-normalize; RETIRED.
#   2026-07-25.1  — matrix_is_log pinned; positional downsample (order-dependent).
#                   RETIRED after pathway |mag moved between pathway_analysis and
#                   cell_quality_partial at the same stamp (Dixit-class).
#   2026-07-29.1  — CURRENT. Order-invariant cell downsample
#                   (hash(seed|obs_name)); pathway + QC share pipeline_core load
#                   + materialize; regenerate frozen Sp + pathway + QC together.
# Synthetic generator versions live in SYNTHETIC_CONFIG_VERSION below and do
# NOT bump CONFIG_VERSION (simulation knobs ≠ real-data preprocess freeze).
#
# Calendar rename: early Drive/local CSVs were stamped with August calendar
# dates that were later retagged to the July labels above (same freeze content,
# label only). resolve_config_version() maps those aliases before the guard
# compares. Prefer rewriting the CSV column to the July stamp when regenerating.
CONFIG_VERSION_ALIASES = {
    "2026-08-03.1": "2026-07-25.1",  # retired positional-downsample era
    "2026-08-04.1": "2026-07-29.1",  # current hash-stable freeze
}

# Full six-dataset frozen Sp table under CONFIG_VERSION (refuse join if wrong).
FROZEN_SP_EXPECTED_N_ROWS = 2285
# Five scoreable (n≥15) after dropping Adamson pilot n=8
FROZEN_SP_EXPECTED_SCOREABLE_N = 2277

# code/ (parent of utils/). Default outputs go to code/shesha-crispr/.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ["SHESHA_OUT"]).expanduser() if os.environ.get("SHESHA_OUT") else ROOT / "shesha-crispr"
CACHE_DIR = Path(
    os.environ.get("PERTPY_CACHE_DIR")
    or os.environ.get("SCVERSE_DATADIR")
    or "/tmp/pertpy_data"
)

# Pin for figure scripts that import shesha.bio. Manuscript Sp uses
# pipeline_core.calculate_sp. Bumping the pin does not bump CONFIG_VERSION.
SHESHA_GEOMETRY_PIN = "shesha-geometry==0.2.27"
SHESHA_GEOMETRY_GIT = "https://github.com/prashantcraju/shesha"
SHESHA_GEOMETRY_COMMIT = "dc0b4b1ea80e7b6ab76b61d4553dcd9757fd6aa1"  # version bump → 0.2.27
SHESHA_GEOMETRY_SP_PATH = "pipeline_core.calculate_sp"  # freeze path (not shesha.bio)

# ---------------------------------------------------------------------------
# Randomness / bootstrap
# ---------------------------------------------------------------------------

SEED = 320
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
BOOTSTRAP_NAN_WARN_THRESHOLD = 0.05

# Survival gate for pathway / QC partials (quote in methods).
# Thresholds are conservative; SI reports sensitivity to CI-only and to
# ε ∈ {0.01, 0.02, 0.03}.
SURVIVAL_CRITERION_ID = "ci_and_fdr.v1"
SURVIVAL_ABS_RHO_MIN = 0.1
SURVIVAL_FDR_MAX = 0.05
# BH family (one family, applied once): within each (dataset × covariate
# model) across the nine scored features — five Hallmark pathway scores
# (UPR, mTORC1, p53, Apoptosis, ROS) plus four stress markers
# (DDIT3, ATF4, XBP1, HSPA5). Do not BH across models or across a
# stress-only subset; that produces a second FDR for the same row.
SURVIVAL_FDR_FAMILY_ID = "bh_dataset_model_9features"
# Knife-edge: nearer |CI bound| < ε demotes an *otherwise-surviving* row to
# indeterminate. Strict < (bound == ε does not trigger). Null rows with a
# near-zero bound stay does_not_survive. Norman p53 |QC high ≈ −0.015 → catch.
SURVIVAL_KNIFE_EDGE_ABS = 0.02
# Small-n QC: drop collinear n_genes/n_counts; mito only (+ magnitude)
SURVIVAL_QC_MITO_ONLY_MAX_N = 40
# Below this n, QC partials are point-estimate only (bootstrap CIs unreliable;
# Papalexi n=24 still rank-deficient in resamples even with mito-only).
SURVIVAL_QC_DESCRIPTIVE_MAX_N = 30

# Approach B (unbiased discordance GSEA) — redesigned 2026-08-05.
# Pre-specified question: do Hallmark apoptosis / p53 appear among discordant
# terms after residual-Sp quartiles + caliper matching + pseudobulk DE?
# If the balance gate fails, drop the arm (Approach A carries the result).
APPROACH_B_CALIPER_MAG_SD = 0.25       # |mag_Q4 - mag_Q1| ≤ this × SD(mag)
APPROACH_B_SMD_MAX = 0.25              # |standardized mean diff| gate
APPROACH_B_MIN_MATCHED_PAIRS = 15
APPROACH_B_EQUALIZE_CELLS = 50         # cells/pert before pseudobulk
APPROACH_B_TARGET_TERMS = ("APOPTOSIS", "P53")  # substring match in Hallmark terms
# Power floor (pre-specified): need ≥ MIN_MATCHED_PAIRS per residual quartile
# before caliper. Dixit n≈98 → ~25/Q is borderline; Papalexi n=24 → ~6 fails;
# Adamson UPR n=87 → ~22/Q borderline. Only Replogle + Norman are powered for
# the matched design. Do not expand post hoc after seeing balance failures.
APPROACH_B_DATASETS = (
    "Replogle 2022 (CRISPRi)",
    "Norman 2019 (CRISPRa)",
)

# ---------------------------------------------------------------------------
# Preprocessing (canonical paper pipeline)
# ---------------------------------------------------------------------------

N_PCS = 50
N_HVG = 2000
MIN_GENES_PER_CELL = 100
NORMALIZE_TARGET_SUM = None  # exact 2026-07-29.1 freeze: median library size
LOG1P = True

# Per-dataset matrix scale pin (see DATASETS[*]["matrix_is_log"]).
# True  → skip normalize/log1p (matrix already log-scale)
# False → always normalize_total + log1p
# unset/None → LEGACY heuristic (_looks_log_normalized); emits a warning.
#
# AUDIT 2026-08-03: heuristic is WRONG and UNSTABLE on Adamson UPR —
# full-matrix inspect → raw (max≈295, integer-like); filtered 13k-cell
# subset → heuristic≈True. Never trust the heuristic; pin from
# inspect_matrix_scale.py on .X before preprocess. Bump CONFIG_VERSION
# when any pin changes.
#
# Synthetic counts are always raw → pass matrix_is_log=False to preprocess
# (see synthetic_benchmark.score_synthetic). Do not rely on the heuristic.

# Minimum perturbed cells required to score a perturbation.
# Historical drift: main_analysis used 10; fig2/S1/S4 used 50 (Dixit n≈153 vs ≈98).
# Freeze at 50 for the revision — more defensible for reliability claims;
# regenerate every table/figure from this cutoff.
MIN_CELLS = 50
MIN_CONTROL_CELLS = 5

# Cap cells loaded into RAM / used for Sp (obs-only downsample before to_memory).
# Sp is a mean cosine over cells — 100 cells/pert is enough and keeps Replogle tractable.
MAX_CELLS_PER_PERT = 100
MAX_CONTROL_CELLS = 5_000

# Above this n_obs after downsample, use sparse fast-path (skip scanpy normalize/PCA)
LARGE_DATASET_N_OBS = 40_000

# Replogle is filtered at load time with the same cutoff
REPLOGLE_MIN_CELLS = MIN_CELLS

# Ablation grids (not used by the spike; kept centralized)
PCA_DIMS = [10, 20, 30, 50, 100]
RANDOM_SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

# ---------------------------------------------------------------------------
# Synthetic benchmarking
# ---------------------------------------------------------------------------
# Song-style scDesign3 simulations with controlled perturbation efficiency.
#
# Versioned separately from CONFIG_VERSION on purpose: adding simulation
# settings does not change any preprocessing/filtering decision behind the
# frozen real-data tables, so CONFIG_VERSION must NOT bump for these. Bump
# SYNTHETIC_CONFIG_VERSION instead and regenerate only synthetic outputs.
#
# CONTRACT with real-data freeze (do not break from either chat):
#   SHARED (owned by CONFIG_VERSION): N_PCS, N_HVG, SEED default, MIN_GENES,
#     LOG1P, NORMALIZE_TARGET_SUM, calculate_sp, preprocess() internals.
#     Synthetic scoring calls preprocess(..., matrix_is_log=False) explicitly —
#     it does NOT use materialize_min_cells / DATASETS pins / pathway scores.
#   SEPARATE (owned by SYNTHETIC_CONFIG_VERSION): effect_scale, program grid,
#     efficiency dists, mag-match, realism thresholds, R/scDesign3 knobs.
#   ANCHOR: observed Sp for the realism gate comes from frozen_sp_scores.csv.
#     After a CONFIG_VERSION bump + frozen regen, re-read that CSV (do not
#     hardcode SLU7=0.514 from an older freeze). Do NOT bump SYNTHETIC_* just
#     because CONFIG_VERSION moved.
#
# This grid is pre-registered: it is fixed before looking at any result, so
# the benchmark cannot be tuned to favour Sp. Do not edit it in response to
# an outcome — add a new version and report both.

# Synthetic-only version lineage (independent of CONFIG_VERSION):
#   2026-07-25.3 — effect_scale=4.0; hot; dist ranking A
#   2026-07-25.4 — effect_scale=2.0; hot; dist ranking flipped
#   2026-07-25.5 — effect_scale=1.0; cold (Sp=0.236); dists null
#   2026-07-25.6 — effect_scale=1.5; R mag-match; ov=-1 K=2 blew up (mag 26)
#   2026-07-25.7 — same scale; skip ill-posed R mag-match; PCA mag still unmatched
#   2026-07-25.8 — R-space mag-match OFF; post-hoc PCA mag filter only
#   2026-07-25.9 — CURRENT; arm closed for manuscript use (see SYNTH_MANUSCRIPT_USABLE)
SYNTHETIC_CONFIG_VERSION = "2026-07-25.9"

# Manuscript policy for this lane (synthetic only — never bumps CONFIG_VERSION).
# Attempted under .8 with auto-selected high_sp GRPEL1 (frozen Sp ~0.397):
#   sim Sp ~0.144 at effect_scale=1.5 (gap 0.253 > 0.2); efficiency arm flat
#   near noise floor. The scale that previously hit SLU7 realism under synthetic
#   dosage does not transfer when the freeze picks a different reference / dosage
#   regime. Re-tuning effect_scale to the new target would be outcome-fitting.
# Status: attempted, not usable. Keep code for audit; do not cite arm tables.
SYNTH_MANUSCRIPT_USABLE = False

# --- Song et al. 2025 (Nat Cell Biol) Fig 2 grid, reproduced verbatim ---
# 20 settings = 5 DEG counts x 4 efficiencies.
SONG_N_DEG = [10, 50, 100, 200, 500]
SONG_EFFICIENCY = [0.25, 0.50, 0.75, 1.00]

# --- Reference setting for the one-factor-at-a-time extensions ---
# D=50 (Song Fig 2b-e) left Sp stuck at the noise floor (~0.15) in Replogle
# PCA space: 50 shifted genes in 3000 is too weak for directional coherence.
# Extensions use D=200; song_replication still sweeps the full Song D grid.
SYNTH_REFERENCE_CELL = {
    "n_deg": 200,
    "efficiency": 0.50,
    "efficiency_dist": "homogeneous",
    "n_programs": 1,
    "program_overlap": 1.0,
    "state_mix": 1.0,
    "n_pert_cells": MAX_CELLS_PER_PERT,
    "n_ctrl_cells": 500,
}

# --- Extension axes, each swept around the reference ---

# Efficiency heterogeneity. Song applied one scalar efficiency per dataset;
# a mixture of perturbed, unperturbed, and partially perturbed cells is
# produced only by the non-homogeneous arms.
#   homogeneous : every perturbed cell at psi
#   bimodal     : fraction psi of cells at 1.0, remainder at 0.0
#   beta_tight  : Beta with mean psi, high concentration
#   beta_broad  : Beta with mean psi, low concentration
SYNTH_EFFICIENCY_DISTS = ["homogeneous", "bimodal", "beta_tight", "beta_broad"]
SYNTH_BETA_CONCENTRATION = {"beta_tight": 20.0, "beta_broad": 2.0}

# Genuine multi-program responses (the biology Sp claims to detect).
# program_overlap = fraction of DEGs shared between programs;
# -1.0 encodes an antagonistic (sign-flipped) second program.
# NOTE: ov=-1 is only well-defined for n_programs=2 (equal +/- on one axis).
# n_programs=3 × ov=-1 was REMOVED from the grid in 2026-07-25.6 because the
# old impl put (K-1) programs on − and one on +, unbalancing the mixture and
# inflating Sp (the 0.191 > 0.130 inversion vs ov=0). Documented, not silent.
SYNTH_N_PROGRAMS = [1, 2, 3]
SYNTH_PROGRAM_OVERLAP = [1.0, 0.5, 0.0, -1.0]

# Cell state. Fraction of perturbed cells drawn from the responsive control
# substate; the remainder carry the perturbation but do not respond to it.
SYNTH_STATE_MIX = [1.0, 0.75, 0.50, 0.25]

# Cell count, linking to the downsampling analysis.
SYNTH_N_PERT_CELLS = [25, 50, 100, 200]

SYNTH_N_SEEDS = 20

# --- scDesign3 / R settings (Song's Methods) ---
SYNTH_FAMILY = "nb"            # NB is Colab-stable; Song used zinb — override via --family zinb
SYNTH_COPULA = "gaussian"      # Song: Gaussian copula, fit per KO/WT group
SYNTH_N_HVG_REFERENCE = 3000   # Song: scran-selected 3000 HVGs in the reference
SYNTH_R_EXECUTABLE = "Rscript"
SYNTH_R_N_CORES = 1            # Colab: mclapply with n_cores>1 returns NULL → fit_marginal crash
SYNTH_R_TIMEOUT_S = 14_400

# Song excludes the perturbed gene itself from evaluation; we do the same so
# Sp cannot be driven by the target's own knockdown.
SYNTH_EXCLUDE_TARGET_GENE = True

# Song amplified the WT count matrix ×10 before fitting so downstream effects
# have usable dynamic range. Replogle CRISPRi targets are often near-zero in
# the expression matrix (WT median C = 0 for SLU7), which makes Song's
# "regress DEGs on target counts C" mechanism inject nothing — Sp then sticks
# at the noise floor (~0.15) for every program setting.
SYNTH_COUNT_AMPLIFICATION = 10.0
SYNTH_MIN_TARGET_WT_MEDIAN = 1.0  # below this, use synthetic dosage instead of C

# Scale the KO−WT mean difference when building simulated KO cells:
#   mean = para_zero + scale * (para_eff - para_zero)
# Historical note (SLU7 + synthetic dosage only; NOT portable): under an older
# freeze, 1→0.236 / 1.5→~0.59 / 4→0.921 vs SLU7~0.514. Under GRPEL1 + real
# dosage (freeze 2026-07-29.1), the same 1.5 yields ~0.144. Do not print that
# SLU7 curve as if it calibrates the current reference. Do not re-dial this
# constant to chase a new observed Sp — see SYNTH_MANUSCRIPT_USABLE.
SYNTH_EFFECT_SCALE = 1.5

# Generative R-space magnitude rescale (OFF from .8). Matching ||μ|| in count
# mean space does not control scored PCA magnitude (HVG + 50-PC intervene);
# R-match quality was anti-correlated with scored match, and large scale_m
# inflated Sp via SNR. Matching is post-hoc on scored mag only (see tol).
SYNTH_MATCH_PROGRAM_MAGNITUDE = False
SYNTH_MAG_MATCH_REL_TOL = 0.15  # |mag - mag_1prog| / mag_1prog for matched subset

# Realism gate thresholds (printed with the metric). Absolute Sp gap vs the
# auto-selected reference's observed Sp. Failures are expected to be
# reference-dependent under a fixed effect_scale; that is why the arm is
# closed rather than re-tuned (SYNTH_MANUSCRIPT_USABLE=False).
SYNTH_REALISM_MAX_ABS_GAP = 0.20
SYNTH_REALISM_MAX_SIM_SP = 0.70

# Reference perturbations for the simulator. Song used a single gene (Nelfb);
# we span the observed Sp range so conclusions are not hostage to one
# signature's geometry. Genes are selected at runtime from the frozen Sp
# table by quantile, unless overridden on the command line.
SYNTH_REFERENCE_DATASETS = ["Replogle 2022 (CRISPRi)", "Dixit 2016 (CRISPR-KO)"]
SYNTH_REFERENCE_SP_QUANTILES = {"low_sp": 0.10, "mid_sp": 0.50, "high_sp": 0.90}

# ---------------------------------------------------------------------------
# Adamson UPR positive-control gene set (PINNED — do not edit quietly)
# ---------------------------------------------------------------------------
# Pre-specified canonical PERK/IRE1/ATF6 signaling + select ER-QC genes.
# Used by adamson_upr_spike / adamson_upr_magnitude_partial since first commit
# of adamson_upr_spike.py (2026-07-17) — same biology, now deduplicated.
#
# Report n_unique=13 + aliases; do not say "18 pinned genes."
# Aliases are for label matching only (BIP/GRP78→HSPA5, etc.); Adamson gene
# grain typically matches ~9 canonical symbols.
#
# NOT the positive-control definition: a lowest-Sp ER trafficking
# list (GBF1, SLC35B1, MANF, HYOU1, OST4, STT3A, SEC63) was a post-hoc
# description of the Sp ranking bottom. Only MANF/HYOU1 overlap this core.
# Plan claim 0.135/0.263/p=0.024 has no logged artifact of either set;
# frozen contrast is 0.158/0.268/p=0.0131. Discrepancy causes: (1) pre-freeze
# cell selection; (2) possible conflation of post-hoc lowest-Sp list with
# this pre-specified core. Never retune the set to recover a p-value.
# Bump UPR_CORE_SET_ID (not CONFIG_VERSION) for membership/reporting changes
# that leave Sp tables unchanged.
UPR_CORE_CANONICAL = frozenset(
    {
        "ATF4",
        "ATF6",
        "DDIT3",  # alias CHOP
        "EIF2AK3",  # alias PERK
        "ERN1",  # alias IRE1
        "HSPA5",  # aliases BIP, GRP78
        "XBP1",
        "HYOU1",
        "MANF",
        "SYVN1",
        "SEL1L",
        "DNAJC3",
        "PPP1R15A",
    }
)
UPR_CORE_ALIASES = {
    "IRE1": "ERN1",
    "PERK": "EIF2AK3",
    "CHOP": "DDIT3",
    "BIP": "HSPA5",
    "GRP78": "HSPA5",
}
# Match set = canonical ∪ alias keys (behavior unchanged vs v1 18-item bag)
UPR_CORE_GENES = UPR_CORE_CANONICAL | frozenset(UPR_CORE_ALIASES)
UPR_CORE_SET_ID = "adamson_upr_core.v2"
UPR_CORE_N_UNIQUE = len(UPR_CORE_CANONICAL)

# Minimum real KO / WT cells required for a perturbation to serve as a
# scDesign3 reference. Every setting is simulated at the largest cell count
# and subsampled, so the reference must supply max(SYNTH_N_PERT_CELLS) real
# KO cells or the cell-count sweep has nothing to subsample from.
#
# NOTE: this cannot be checked against the frozen Sp table. n_cells there is
# recorded AFTER the MAX_CELLS_PER_PERT downsample, so it never exceeds 100
# and cannot distinguish a 100-cell perturbation from a 5000-cell one. The
# real count must be validated against the raw data at reference-build time.
SYNTH_MIN_REFERENCE_KO_CELLS = max(SYNTH_N_PERT_CELLS)
SYNTH_MIN_REFERENCE_WT_CELLS = 300

# ---------------------------------------------------------------------------
# Dataset registry (display name → metadata)
# ---------------------------------------------------------------------------
# Display names are the canonical keys used in CSVs and figures.
# modality / cell_type / design are required for gene × context framing.

DATASETS = {
    # matrix_is_log pins: set ONLY from inspect_matrix_scale.py on .X.
    # matrix_scale_verified: True once a logged inspect matches the pin.
    "Norman 2019 (CRISPRa)": {
        "loader": "norman_2019",
        "modality": "CRISPRa",
        "cell_type": "K562",
        "design": "CRISPRa activation; includes combinatorial perturbations",
        "control_keywords": ["control", "ctrl", "non-targeting", "scrambled", "nt"],
        # Prior inspect: max≈6.7, frac_near_int=0 → already log. Re-confirm under
        # --all-main gate before regenerating manuscript tables.
        "matrix_is_log": True,
        "matrix_scale_verified": True,  # inspect 2026-08-03: max≈6.3, frac_near_int=0
        "local_h5ad": "norman_2019.h5ad",
        "download_urls": [
            "https://figshare.com/ndownloader/files/34027562",
        ],
        "in_main": True,
    },
    "Adamson 2016 pilot (CRISPRi)": {
        "loader": "adamson_2016_pilot",
        "modality": "CRISPRi",
        "cell_type": "K562",
        "design": "CRISPRi pilot (TF knockdowns; NOT the UPR arm)",
        "control_keywords": ["gal4", "gfp", "neg", "scramble", "unperturbed", "nan", "control"],
        # inspect 2026-08-05: max=47, frac_near_int≈1 → raw counts; pin matches.
        "matrix_is_log": False,
        "matrix_scale_verified": True,
        "local_h5ad": "adamson_2016_pilot.h5ad",
        "download_urls": [
            "https://zenodo.org/records/10044268/files/AdamsonWeissman2016_GSM2406675_10X001.h5ad?download=1",
            "https://zenodo.org/record/10044268/files/AdamsonWeissman2016_GSM2406675_10X001.h5ad?download=1",
        ],
        "in_main": True,  # sixth frozen dataset (UPR arm remains the positive control)
    },
    "Adamson 2016 UPR (CRISPRi)": {
        "loader": "adamson_2016_upr_perturb_seq",
        "modality": "CRISPRi",
        "cell_type": "K562",
        "design": "CRISPRi UPR Perturb-seq; 91 sgRNAs / 82 genes",
        # scPerturb labels use Gal4 / 63(mod) / 62(mod) as non-targeting controls
        "control_keywords": [
            "gal4", "gfp", "neg", "scramble", "unperturbed", "nan", "control", "nt",
            "63(mod)", "62(mod)", "(mod)",
        ],
        "aggregate_to_gene": True,  # collapse GUIDE_pDSxxx → gene before scoring
        # VERIFIED 2026-08-03: raw counts (max≈295, frac_near_int≈1). Heuristic
        # skipped normalize on the filtered subset — wrong. Force normalize+log1p.
        "matrix_is_log": False,
        "matrix_scale_verified": True,
        "local_h5ad": "adamson_2016_upr_perturb_seq.h5ad",
        # scverse CDN often returns 403 on Colab/cloud; Zenodo is the reliable fallback
        "download_urls": [
            "https://zenodo.org/records/10044268/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1",
            "https://zenodo.org/record/10044268/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1",
            "https://exampledata.scverse.org/pertpy/adamson_2016_upr_perturb_seq.h5ad",
        ],
        "in_main": True,
    },
    "Dixit 2016 (CRISPR-KO)": {
        "loader": "dixit_2016",
        "modality": "CRISPR-KO",
        "cell_type": "BMDC",
        "design": "Pooled CRISPR-Cas9 knockout Perturb-seq (NOT CRISPRi)",
        "control_keywords": ["nan", "control", "neg", "intergenic", "ctrl"],
        # Prior inspect: max≈6.0, frac_near_int=0 → already log. Re-confirm gate.
        "matrix_is_log": True,
        "matrix_scale_verified": True,  # inspect 2026-08-03: max≈6.2, frac_near_int=0
        "local_h5ad": "dixit_2016.h5ad",
        "download_urls": [
            "https://figshare.com/ndownloader/files/34014608",
        ],
        "in_main": True,
    },
    "Papalexi 2021 (CRISPR-KO)": {
        "loader": "papalexi_2021",
        "modality": "CRISPR-KO",
        "cell_type": "THP-1",
        "design": "ECCITE-seq; includes knockout perturbations (MuData / gene_target)",
        "control_keywords": ["nt", "non-targeting", "control"],
        # inspect 2026-08-03: max≈127, frac_near_int=1 → raw counts.
        "matrix_is_log": False,
        "matrix_scale_verified": True,
        "local_h5ad": "papalexi_2021.h5mu",  # MuData
        # Figshare ndownloader is often WAF-challenged (0-byte). Current pertpy
        # uses the scverse mirror of the same MuData — try that first.
        "download_urls": [
            "https://exampledata.scverse.org/pertpy/papalexi_2021.h5mu",
            "https://figshare.com/ndownloader/files/36509460",
        ],
        "in_main": True,
    },
    "Replogle 2022 (CRISPRi)": {
        "loader": "replogle_2022_k562_essential",
        "modality": "CRISPRi",
        "cell_type": "K562",
        "design": "CRISPRi genome-scale essential-gene screen",
        "control_keywords": ["control"],
        # inspect 2026-08-03: max≈184, frac_near_int=1 → raw counts.
        "matrix_is_log": False,
        "matrix_scale_verified": True,
        "local_h5ad": "replogle_2022_k562_essential.h5ad",
        # Zenodo Content-Length ≈ 1546.7 MB. A 72 MB truncated cache was accepted
        # previously and silently gutted every downstream table — assert size.
        "expected_bytes": 1_546_700_000,
        "download_urls": [
            "https://zenodo.org/records/10044268/files/ReplogleWeissman2022_K562_essential.h5ad?download=1",
            "https://zenodo.org/record/10044268/files/ReplogleWeissman2022_K562_essential.h5ad?download=1",
        ],
        "in_main": True,
    },
}

# Datasets that must appear in pathway / QC outputs for a non-partial run
PATHWAY_REQUIRED_DATASETS = [
    "Adamson 2016 UPR (CRISPRi)",
    "Papalexi 2021 (CRISPR-KO)",
    "Replogle 2022 (CRISPRi)",
    "Norman 2019 (CRISPRa)",
    "Dixit 2016 (CRISPR-KO)",
]

# Convenience maps for scripts that previously hard-coded these
MODALITY_MAP = {name: meta["modality"] for name, meta in DATASETS.items()}
DATASET_CONTEXT = {
    name: {
        "cell_type": meta["cell_type"],
        "design": meta["design"],
        "modality": meta["modality"],
    }
    for name, meta in DATASETS.items()
}
MANUAL_CONTROLS = {
    name: meta["control_keywords"] for name, meta in DATASETS.items()
}

# Legacy display-name aliases → frozen names (for reading old CSVs)
LEGACY_NAME_MAP = {
    "Adamson 2016 (CRISPRi)": "Adamson 2016 pilot (CRISPRi)",
    "Dixit 2016 (CRISPRi)": "Dixit 2016 (CRISPR-KO)",
    "Papalexi 2021 (CRISPR)": "Papalexi 2021 (CRISPR-KO)",
}


def main_dataset_names() -> list[str]:
    return [n for n, m in DATASETS.items() if m.get("in_main", True)]


def resolve_dataset_name(name: str) -> str:
    """Map a legacy or current display name to the frozen key."""
    if name in DATASETS:
        return name
    return LEGACY_NAME_MAP.get(name, name)


def resolve_config_version(version: str) -> str:
    """Map a legacy August calendar stamp to its July CONFIG_VERSION label."""
    v = str(version).strip()
    return CONFIG_VERSION_ALIASES.get(v, v)
