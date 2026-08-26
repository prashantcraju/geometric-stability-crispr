#!/usr/bin/env python3
"""
Pathway-Level Analysis of Geometric Instability

Approach A (Main Text): Perturbation-level gene set activity scores
  For each perturbation, compute mean expression of MSigDB Hallmark gene sets.
  Run Spearman + partial correlation (controlling magnitude) between Sp and
  each pathway score.  Directly extends the Fig 3c stress marker framework.

Approach B (Supplementary): Differential pathway enrichment, discordance Q1 vs Q4
  Within a magnitude-matched band (middle two magnitude quartiles), split
  perturbations by discordance into Q1 (most concordant) vs Q4 (most discordant).
  Run Wilcoxon DE between Q4 and Q1 cells, then preranked GSEA against MSigDB
  Hallmark, KEGG, and Reactome collections.

Datasets (Approach A):
  Powered: Replogle, Norman, Dixit, Adamson UPR, Papalexi (n≥15).
  Logged skip: Adamson pilot (n=8 < 15) — SI Sp table only.
  Approach B (exploratory / confounded until rematched): Replogle, Norman.

Uses DATASETS[*].matrix_is_log pins (do NOT always normalize+log1p).
Prefer joining Sp/magnitude from frozen_sp_scores.csv so pathway partials
sit on the same Sp table as the rest of the paper (CONFIG_VERSION).

INPUT:  pipeline_core / Zenodo-Figshare cache (/tmp/pertpy_data) or pertpy fallback
OUTPUT: pathway_signature_correlations.csv        (Approach A; BH-FDR per dataset)
        pathway_gsea_Q4_vs_Q1_*.csv               (Approach B)
        pathway_de_Q4_vs_Q1_*.csv                 (Approach B)

Adamson UPR h5ad (same as adamson_upr_spike / run_frozen_main):
  /tmp/pertpy_data/adamson_2016_upr_perturb_seq.h5ad
  Override with --adamson-h5ad if needed.
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import subprocess
import sys
import os
import types
import importlib.util
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def _detect_colab() -> bool:
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


IN_DEEPNOTE = os.environ.get("DEEPNOTE_PROJECT_ID") is not None

def _ensure_pkgs(pkgs):
    """Install missing packages (works under `python script.py` on Colab too)."""
    missing = []
    for pkg, mod in pkgs:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    print(f"Installing missing packages: {missing}", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing],
    )


# Always try to ensure gseapy when missing (not only interactive Colab kernels)
try:
    _ensure_pkgs([
        ("gseapy", "gseapy"),
        ("pingouin", "pingouin"),
        ("statsmodels", "statsmodels"),
        ("scikit-learn", "sklearn"),
    ])
except Exception as e:
    print(f"WARNING: package install failed ({e}). For Approach B run: pip install gseapy")

_cache = Path("/tmp/pertpy_data")
try:
    _cache.mkdir(parents=True, exist_ok=True)
except OSError:
    _cache = Path.home() / ".cache" / "pertpy_data"
    _cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SCVERSE_DATADIR", str(_cache))
os.environ.setdefault("PERTPY_CACHE_DIR", str(_cache))

import numpy as np
import pandas as pd
import scanpy as sc
sc.settings.datasetdir = _cache

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

try:
    import pipeline_config as cfg
    SEED = cfg.SEED
    N_BOOTSTRAP = cfg.N_BOOTSTRAP
    CI_LEVEL = cfg.CI_LEVEL
    MIN_CELLS = cfg.MIN_CELLS
    _DEFAULT_OUT = cfg.OUTPUT_DIR
    CONFIG_VERSION = cfg.CONFIG_VERSION
except ImportError:
    SEED = 320
    N_BOOTSTRAP = 10_000
    CI_LEVEL = 0.95
    MIN_CELLS = 50
    _DEFAULT_OUT = Path("./shesha-crispr")
    CONFIG_VERSION = "unknown"

np.random.seed(SEED)

# Optional pertpy for Norman/Dixit/Replogle (Adamson UPR uses pipeline_core + local h5ad)
pt = None
try:
    for _mod in list(sys.modules):
        if _mod == "pertpy" or _mod.startswith("pertpy."):
            del sys.modules[_mod]
    _pertpy_spec = importlib.util.find_spec("pertpy")
    if _pertpy_spec is not None and _pertpy_spec.submodule_search_locations:
        _pertpy_path = _pertpy_spec.submodule_search_locations[0]
        _pertpy_pkg = types.ModuleType("pertpy")
        _pertpy_pkg.__path__ = [_pertpy_path]
        _pertpy_pkg.__spec__ = _pertpy_spec
        sys.modules["pertpy"] = _pertpy_pkg
        _pt_datasets = importlib.import_module("pertpy.data._datasets")
        _pt_datasets.settings.datasetdir = _cache
        pt = types.SimpleNamespace(
            dt=types.SimpleNamespace(
                norman_2019=_pt_datasets.norman_2019,
                dixit_2016=_pt_datasets.dixit_2016,
                replogle_2022_k562_essential=_pt_datasets.replogle_2022_k562_essential,
            )
        )
except Exception as e:
    print(f"WARNING: pertpy unavailable ({e}); use pipeline_core local h5ads where possible")

try:
    import gseapy as gp
    HAS_GSEAPY = True
    print(f"gseapy {getattr(gp, '__version__', '?')} available — Approach B enabled")
except ImportError:
    HAS_GSEAPY = False
    print("WARNING: gseapy not installed — Approach B (GSEA) will be skipped.")
    print("  Fix:  pip install gseapy")
    print("  Or:   python pathway_analysis.py --skip-approach-b")

# =============================================================================
# CONFIGURATION
# =============================================================================

_env = os.environ.get("SHESHA_OUT", "").strip()
_CANDIDATE_OUTS = [
    Path(_env) if _env else None,
    Path("/content/shesha-crispr"),
    Path("./shesha-crispr"),
    _DEFAULT_OUT,
]
OUTPUT_DIR = next(
    (p for p in _CANDIDATE_OUTS if p is not None and str(p).strip() and p.exists()),
    Path("./shesha-crispr"),
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
print(f"DATA_CACHE = {_cache}  config_version={CONFIG_VERSION}")

REPLOGLE_MIN_CELLS = MIN_CELLS
NORMAN_MIN_CELLS = MIN_CELLS
DIXIT_MIN_CELLS = MIN_CELLS  # frozen at 50 (was 10)
ADAMSON_MIN_CELLS = MIN_CELLS

MIN_GENE_OVERLAP = 5
GSEA_PERMUTATIONS = 1000

# Enrichr / gseapy library names for Approach B
GSEA_LIBRARY_HALLMARK = "MSigDB_Hallmark_2020"
GSEA_LIBRARIES_SI = [
    "KEGG_2021_Human",
    "Reactome_2022",
]
GSEA_LIBRARIES = [GSEA_LIBRARY_HALLMARK] + GSEA_LIBRARIES_SI

# =============================================================================
# MSigDB HALLMARK GENE SETS  (canonical v2023.2, curated core members)
#
# Each list contains the well-characterised members most commonly detected in
# scRNA-seq.  The analysis reports overlap with each dataset's var_names so the
# effective gene-set size is transparent.
# =============================================================================

HALLMARK_GENE_SETS = {
    'UPR': [
        # ER chaperones / foldases
        'HSPA5', 'HSP90B1', 'HYOU1', 'CALR', 'CANX', 'P4HB', 'PDIA3',
        'PDIA4', 'PDIA5', 'PDIA6', 'PPIB', 'ERP29', 'ERP44', 'SIL1',
        'FKBP14', 'DNAJB9', 'DNAJB11', 'DNAJC3', 'DNAJC10',
        # UPR sensors / effectors
        'ATF6', 'ATF6B', 'ERN1', 'EIF2AK3', 'XBP1', 'DDIT3', 'CREB3L2',
        # ERAD
        'EDEM1', 'DERL1', 'OS9', 'SEL1L', 'SYVN1', 'UBE2J1', 'UBE2D1',
        'VIMP', 'YOD1', 'VCP',
        # translocon / signal peptidase
        'SEC61A1', 'SEC61B', 'SEC11C', 'SEC24D', 'TRAM1', 'SRPRB',
        'SPCS1', 'SPCS2', 'SPCS3', 'SSR1', 'SSR3', 'SSR4',
        # ER-to-Golgi / vesicle
        'LMAN1', 'GOSR2', 'KDELR3', 'SURF4',
        # glycosylation / lipid
        'DDOST', 'STT3A', 'STT3B', 'RPN1', 'RPN2', 'MOGS', 'UGGT1',
        'SRD5A3',
        # other UPR-associated
        'HERPUD1', 'MANF', 'CRELD2', 'SDF2L1', 'NUCB1', 'RCN1',
        'SERP1', 'WIPI1', 'UFM1', 'BAX', 'ERO1A', 'MBTPS1', 'MBTPS2',
        'ARCN1', 'PREB', 'GANAB', 'TMX1', 'ERLEC1',
    ],
    'mTORC1': [
        # lipid / cholesterol synthesis
        'ACLY', 'ACSS2', 'SCD', 'FASN', 'ELOVL5', 'ELOVL6', 'FADS1',
        'FADS2', 'HMGCR', 'HMGCS1', 'MVK', 'MVD', 'IDI1', 'FDPS',
        'FDFT1', 'SQLE', 'LSS', 'DHCR7', 'DHCR24', 'SC5D', 'NSDHL',
        'TM7SF2', 'HSD17B7', 'INSIG1', 'INSIG2', 'LDLR', 'STARD4',
        'VLDLR',
        # nucleotide synthesis
        'IMPDH2', 'RRM1', 'RRM2', 'PRPS1', 'UNG',
        # amino acid / one-carbon
        'PHGDH', 'PSPH', 'SHMT2', 'MTHFD2', 'MTHFD1L', 'SLC7A5',
        'SLC7A11', 'SLC1A4', 'BCAT1', 'GOT1', 'IARS1', 'AARS1',
        # glycolysis
        'HK2', 'GPI', 'PGK1', 'PGM1', 'TPI1', 'PKM', 'SLC2A1',
        # proteasome
        'PSMA3', 'PSMA4', 'PSMB2', 'PSMB4', 'PSMC2', 'PSMC4',
        'PSMC6', 'PSMD1', 'PSMD12', 'PSMD13', 'PSMD14', 'PSME3',
        # ribosome biogenesis / translation
        'NOP14', 'NOP56', 'PNO1', 'GNL3',
        # redox / chaperones
        'PRDX1', 'PRDX6', 'GLRX', 'G6PD', 'PPIA', 'PPIB', 'CALR',
        'CANX', 'HSPA5', 'HSPA9', 'SERP1', 'SSR1',
        # signalling
        'MYC', 'DDIT4', 'TRIB3', 'CDKN1A', 'CCNF', 'BHLHE40',
        'NAMPT', 'AK4', 'ME1', 'PC', 'IDH1',
    ],
    'p53': [
        'CDKN1A', 'MDM2', 'MDM4', 'BAX', 'BBC3', 'PMAIP1', 'PERP',
        'FAS', 'TNFRSF10B', 'PIDD1', 'APAF1', 'CASP1',
        'GADD45A', 'GADD45B', 'GADD45G', 'DDIT4', 'SESN1', 'SESN2',
        'DDB2', 'RRM2B', 'PCNA', 'POLK', 'RPA1',
        'TP53I3', 'TP53INP1', 'ZMAT3', 'EI24', 'TRIAP1', 'AEN',
        'BTG2', 'CCNG1', 'CCNG2', 'PLK2', 'PLK3', 'SFN',
        'GDF15', 'ATF3', 'FDXR', 'SCO2', 'TIGAR', 'GLS2',
        'SAT1', 'STEAP3', 'RPS27L', 'PTEN', 'TSC2',
        'IGFBP4', 'HDAC1', 'ISG15', 'GPX1', 'ING1',
    ],
    'Apoptosis': [
        # BCL-2 family
        'BAX', 'BAK1', 'BCL2', 'BCL2L1', 'BCL2L11', 'BCL2L2', 'MCL1',
        'BID', 'BAD', 'BIK', 'HRK', 'BNIP3', 'BNIP3L', 'PMAIP1',
        # caspases
        'CASP1', 'CASP2', 'CASP3', 'CASP4', 'CASP6', 'CASP7', 'CASP8',
        'CASP9',
        # death receptors / ligands
        'FAS', 'FADD', 'TNFRSF10A', 'TNFRSF10B', 'TNFRSF1A', 'TNFSF10',
        'RIPK1', 'RIPK2', 'CFLAR', 'TRAF1', 'TRAF2',
        # mitochondrial
        'CYCS', 'DIABLO', 'APAF1', 'AIFM1', 'ENDOG', 'HTRA2',
        # inhibitors
        'BIRC2', 'BIRC3', 'XIAP',
        # DNA fragmentation / structural
        'DFFA', 'DFFB', 'LMNA', 'LMNB1', 'LMNB2', 'SPTAN1',
        # other pro-apoptotic / regulators
        'CDKN1A', 'JUN', 'MYC', 'NFKBIA', 'SQSTM1', 'CLU', 'TXNIP',
        'ATF3', 'FDXR', 'SAT1', 'GPX1', 'GPX3', 'GPX4', 'GSR',
        'SOD1', 'SOD2', 'ETF1', 'ADD1', 'APP', 'ANXA1', 'DCN',
        'CD44', 'CCND1', 'CCND2', 'RHOB',
    ],
    'ROS': [
        # superoxide dismutases
        'SOD1', 'SOD2',
        # catalase
        'CAT',
        # glutathione peroxidases
        'GPX1', 'GPX3', 'GPX4',
        # thioredoxin system
        'TXN', 'TXNRD1', 'TXNIP',
        # peroxiredoxins
        'PRDX1', 'PRDX2', 'PRDX3', 'PRDX4', 'PRDX5', 'PRDX6',
        # glutathione synthesis / conjugation
        'GCLC', 'GCLM', 'GSR', 'GSS', 'GLRX', 'GLRX2',
        'GSTP1', 'GSTM1', 'GSTM2', 'GSTM4', 'GSTO1', 'MGST1',
        # Nrf2 / KEAP1
        'NFE2L2', 'KEAP1', 'NQO1', 'HMOX1', 'HMOX2', 'SRXN1',
        # iron / ferritin
        'FTH1', 'FTL',
        # other redox
        'G6PD', 'MSRA', 'PARK7', 'SQSTM1', 'EPHX1',
        'ALDH1A1', 'AKR1B1', 'OXSR1', 'STK25', 'JUNB',
        'ABCC1', 'PRNP', 'PDLIM1', 'CLU', 'CYBA', 'NOX4',
        # mitochondrial complex I (ROS source)
        'NDUFA6', 'NDUFB4', 'NDUFC2', 'NDUFS2',
    ],
}

PATHWAY_FULL_NAMES = {
    'UPR':       'HALLMARK_UNFOLDED_PROTEIN_RESPONSE',
    'mTORC1':    'HALLMARK_MTORC1_SIGNALING',
    'p53':       'HALLMARK_P53_PATHWAY',
    'Apoptosis': 'HALLMARK_APOPTOSIS',
    'ROS':       'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY',
}


# =============================================================================
# BOOTSTRAP HELPERS  (same framework as stress_marker_tests.py)
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=42):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return {'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p': np.nan}
    rng = np.random.default_rng(seed=seed)
    boot = np.empty(n_bootstrap)
    n = len(x)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot[i] = spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    if len(valid) < 10:
        return {'rho': rho, 'ci_low': np.nan, 'ci_high': np.nan, 'p': p}
    alpha = 1 - ci_level
    return {
        'rho': rho, 'p': p,
        'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
    }


def bootstrap_partial_correlation_ci(x, y, z, n_bootstrap=N_BOOTSTRAP,
                                     ci_level=CI_LEVEL, seed=42):
    """Rank-based partial Spearman (canonical); see stats_utils.py."""
    try:
        from stats_utils import bootstrap_partial_spearman_ci
        return bootstrap_partial_spearman_ci(
            x, y, z, n_bootstrap=n_bootstrap, ci_level=ci_level,
            seed=seed, method="rank",
        )
    except ImportError:
        # Fallback: legacy residual Spearman (label as such in outputs if used)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float).reshape(-1, 1) if np.asarray(z).ndim == 1 \
            else np.asarray(z, dtype=float)
        n = len(x)

        def _partial(x, y, z):
            Z_aug = sm.add_constant(z)
            x_r = sm.OLS(x, Z_aug).fit().resid
            y_r = sm.OLS(y, Z_aug).fit().resid
            return spearmanr(x_r, y_r)

        rho_partial, p = _partial(x, y, z)
        rng = np.random.default_rng(seed=seed)
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            boot[i], _ = _partial(x[idx], y[idx], z[idx])
        valid = boot[~np.isnan(boot)]
        if len(valid) < 10:
            return {'rho_partial': rho_partial, 'ci_low': np.nan,
                    'ci_high': np.nan, 'p': p}
        alpha = 1 - ci_level
        return {
            'rho_partial': rho_partial, 'p': p,
            'ci_low': float(np.percentile(valid, 100 * alpha / 2)),
            'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
        }


# =============================================================================
# DATA LOADING  (mirrors fig3.py / fig_replogle.py patterns)
# =============================================================================

def clean_replogle(adata):
    """Label-clean Replogle 2022: merge non-targeting / chr → control."""
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    def _label(x):
        if 'non-targeting' in x or x.startswith('chr'):
            return 'control'
        if 'pos_control' in x:
            return 'POS_CONTROL'
        return x.split('_')[0]
    adata.obs['condition'] = adata.obs['perturbation'].apply(_label)
    return adata[
        (adata.obs['condition'] != 'POS_CONTROL') &
        (adata.obs['condition'] != 'nan')
    ].copy()


def load_adamson_upr(h5ad_path=None):
    """
    Load Adamson 2016 UPR via pipeline_core (Zenodo / /tmp/pertpy_data cache).
    Returns (adata_in_memory, pert_col, ctrl_label) with gene-level conditions.
    """
    from pipeline_core import (
        ensure_in_memory,
        load_raw,
        materialize_min_cells,
        setup_cache,
        _extract_adata,
    )

    setup_cache(_cache)
    sc.settings.datasetdir = _cache
    name = "Adamson 2016 UPR (CRISPRi)"
    print(f"\n>>> Loading {name} via pipeline_core …", flush=True)
    if h5ad_path is None:
        default = _cache / "adamson_2016_upr_perturb_seq.h5ad"
        if default.exists():
            h5ad_path = default
            print(f"    using cached {h5ad_path}", flush=True)
    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad_path)
    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    adata, _, _ = materialize_min_cells(
        adata, pert_col, ctrl_label, min_cells=ADAMSON_MIN_CELLS
    )
    adata = ensure_in_memory(adata)
    print(
        f"    Adamson UPR ready: {adata.n_obs} cells, "
        f"pert_col={pert_col!r}, ctrl={ctrl_label!r}",
        flush=True,
    )
    return adata, pert_col, ctrl_label


def _load_frozen_sp(frozen_csv, dataset_name):
    """Return Series stability / magnitude indexed by perturbation, or None."""
    if frozen_csv is None:
        return None, None
    path = Path(frozen_csv)
    if not path.exists():
        print(f"    frozen Sp CSV not found ({path}) — will recompute Sp")
        return None, None
    fdf = pd.read_csv(path)
    if "dataset" not in fdf.columns or "perturbation" not in fdf.columns:
        print("    frozen CSV missing dataset/perturbation columns — recompute Sp")
        return None, None
    # File-level version already checked in main(); still surface per-row stamp
    file_ver = None
    if "config_version" in fdf.columns and fdf["config_version"].notna().any():
        file_ver = str(fdf["config_version"].dropna().iloc[0])
    # resolve legacy names
    try:
        name = cfg.resolve_dataset_name(dataset_name)
    except Exception:
        name = dataset_name
    sub = fdf[fdf["dataset"].astype(str).map(
        lambda x: cfg.resolve_dataset_name(x) if hasattr(cfg, "resolve_dataset_name") else x
    ) == name].copy()
    if sub.empty:
        # also try exact match on display string
        sub = fdf[fdf["dataset"].astype(str) == str(dataset_name)].copy()
    if sub.empty:
        print(f"    no frozen rows for {dataset_name!r} — will recompute Sp")
        return None, None
    stab_col = "stability" if "stability" in sub.columns else "Sp"
    mag_col = "magnitude" if "magnitude" in sub.columns else None
    if mag_col is None:
        print("    frozen CSV missing magnitude — will recompute Sp")
        return None, None
    stab = sub.set_index(sub["perturbation"].astype(str))[stab_col]
    mag = sub.set_index(sub["perturbation"].astype(str))[mag_col]
    print(
        f"    joined frozen Sp/mag for {len(stab)} perturbations "
        f"(file config_version={file_ver!r}; code={CONFIG_VERSION})"
    )
    return stab, mag


def load_and_process(name, loader_func=None, pert_col=None, ctrl_label=None,
                     clean_func=None, min_cells=50, adata=None,
                     frozen_csv=None, recompute_sp=False,
                     already_materialized=False):
    """
    Load → materialize (stable hash downsample) → normalize per matrix_is_log
    → pathway scores. Sp/magnitude prefer frozen_sp_scores.csv.

    Always materialize unless `already_materialized=True` (caller used
    load_via_pipeline_core / load_adamson_upr which already did). Never score
    on a full pertpy load without the ≤100/pert pin — that caused |mag drift
    vs cell_quality_partial at the same CONFIG_VERSION.
    """
    from pipeline_core import (
        _log1p_inplace,
        _normalize_total_numpy,
        materialize_min_cells,
        resolve_matrix_is_log,
    )

    print(f"\n>>> Loading {name} ...")
    if adata is None:
        if loader_func is None:
            raise ValueError("Need loader_func or adata")
        adata = loader_func()
        if clean_func:
            adata = clean_func(adata)
        already_materialized = False
    if pert_col is None or ctrl_label is None:
        raise ValueError("pert_col and ctrl_label are required")
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    if not already_materialized:
        adata, _, _ = materialize_min_cells(
            adata, pert_col, ctrl_label, min_cells=min_cells, seed=SEED
        )

    # --- Normalise using pinned matrix_is_log (never always-log1p) ---
    already_log, log_src = resolve_matrix_is_log(dataset_name=name, adata=adata)
    if already_log:
        print(f"    skip normalize/log1p (matrix_is_log=True via {log_src})")
    else:
        print(f"    normalize_total + log1p (matrix_is_log=False via {log_src})")
        try:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        except Exception as e:
            print(f"    scanpy normalize failed ({e}); using numpy path")
            _normalize_total_numpy(adata, 1e4)
            _log1p_inplace(adata)
    adata_norm = adata  # alias; all genes retained

    # --- Identify valid perturbations ---
    counts = adata_norm.obs[pert_col].value_counts()
    valid = sorted(
        v for v in counts[counts >= min_cells].index if v != ctrl_label
    )
    print(f"    {len(valid)} perturbations with ≥{min_cells} cells")

    # --- Compute pathway scores per perturbation ---
    pw_scores = {}
    coverage_rows = []
    overlap_sets = {}  # pathway → set of genes present in this dataset
    n_genes_total = int(adata_norm.n_vars)
    for pw_short, pw_genes in HALLMARK_GENE_SETS.items():
        # sorted overlap → deterministic score_genes reference draw
        overlap = sorted(g for g in pw_genes if g in adata_norm.var_names)
        overlap_sets[pw_short] = set(overlap)
        pct = 100 * len(overlap) / len(pw_genes) if pw_genes else 0
        print(f"    {pw_short}: {len(overlap)}/{len(pw_genes)} genes "
              f"({pct:.0f}% overlap)")
        coverage_rows.append({
            "dataset": name,
            "pathway": pw_short,
            "n_genes_dataset": n_genes_total,
            "n_overlap": len(overlap),
            "n_hallmark": len(pw_genes),
            "pct_overlap": float(pct),
            "scored": len(overlap) >= MIN_GENE_OVERLAP,
        })
        if len(overlap) < MIN_GENE_OVERLAP:
            print(f"      ↳ skipped (< {MIN_GENE_OVERLAP} genes)")
            continue

        score_col = f'score_{pw_short}'
        np.random.seed(SEED)
        sc.tl.score_genes(adata_norm, gene_list=overlap,
                          score_name=score_col,
                          ctrl_size=50, random_state=SEED)

        scores = {}
        for pert in valid:
            mask = adata_norm.obs[pert_col] == pert
            scores[pert] = float(adata_norm[mask].obs[score_col].mean())
        pw_scores[pw_short] = scores

    # Apoptosis ∩ p53: if high Jaccard + high score ρ, not two independent findings
    if "Apoptosis" in overlap_sets and "p53" in overlap_sets:
        a, b = overlap_sets["Apoptosis"], overlap_sets["p53"]
        inter, union = a & b, a | b
        jacc = len(inter) / len(union) if union else np.nan
        coverage_rows.append({
            "dataset": name,
            "pathway": "Apoptosis∩p53",
            "n_genes_dataset": n_genes_total,
            "n_overlap": len(inter),
            "n_hallmark": len(union),
            "pct_overlap": float(100 * jacc) if np.isfinite(jacc) else np.nan,
            "scored": False,
            "jaccard": float(jacc) if np.isfinite(jacc) else np.nan,
            "n_intersection": len(inter),
            "n_union": len(union),
        })
        print(
            f"    Apoptosis∩p53: |∩|={len(inter)}, Jaccard={jacc:.3f} "
            f"(shared genes on this dataset's var_names)"
        )

    # --- Sp / magnitude: frozen table preferred ---
    frozen_stab, frozen_mag = (None, None)
    if not recompute_sp:
        frozen_stab, frozen_mag = _load_frozen_sp(frozen_csv, name)

    if frozen_stab is not None:
        df = pd.DataFrame({
            "stability": frozen_stab,
            "magnitude": frozen_mag,
        })
        df = df[df.index.isin([str(v) for v in valid])].copy()
        n_miss = len(valid) - len(df)
        if n_miss > 0:
            print(f"    note: {n_miss} valid perts absent from frozen Sp table")
    else:
        # Fallback only — manuscript path joins frozen_sp_scores.csv.
        # Uses pipeline_core.calculate_sp (no shesha package required).
        print("    recomputing Sp via HVG+PCA (not joined to frozen table)")
        from pipeline_core import calculate_sp

        adata_proc = adata_norm[
            adata_norm.obs[pert_col].isin(valid + [ctrl_label])
        ].copy()
        sc.pp.highly_variable_genes(adata_proc, n_top_genes=2000, subset=True)
        sc.tl.pca(adata_proc, n_comps=50, random_state=SEED)
        Xp = np.asarray(adata_proc.obsm["X_pca"])
        labels = adata_proc.obs[pert_col].astype(str).to_numpy()
        X_ctrl = Xp[labels == ctrl_label]
        stab, mag = {}, {}
        for pert in valid:
            m = calculate_sp(X_ctrl, Xp[labels == pert])
            stab[pert] = m["stability"]
            mag[pert] = m["magnitude"]
        df = pd.DataFrame({"stability": pd.Series(stab), "magnitude": pd.Series(mag)})
        df = df[df.index.isin(valid)].copy()

    for pw_short, scores in pw_scores.items():
        df[f'pw_{pw_short}'] = df.index.map(scores)

    df["dataset"] = name
    df["perturbation"] = df.index.astype(str)
    print(f"    → {len(df)} perturbations in final table")
    return df, adata_norm, coverage_rows


def load_via_pipeline_core(name, h5ad_path=None, min_cells=None):
    """Load any DATASETS entry via pipeline_core (Papalexi / pilot / Adamson)."""
    from pipeline_core import (
        ensure_in_memory,
        load_raw,
        materialize_min_cells,
        setup_cache,
        _extract_adata,
    )
    setup_cache(_cache)
    sc.settings.datasetdir = _cache
    name = cfg.resolve_dataset_name(name)
    print(f"\n>>> Loading {name} via pipeline_core …", flush=True)
    raw = load_raw(name, prefer_local=True, h5ad_path=h5ad_path)
    adata, pert_col, ctrl_label = _extract_adata(raw, name, sc)
    mc = min_cells if min_cells is not None else MIN_CELLS
    adata, _, _ = materialize_min_cells(adata, pert_col, ctrl_label, min_cells=mc)
    adata = ensure_in_memory(adata)
    print(
        f"    ready: {adata.n_obs} cells, pert_col={pert_col!r}, ctrl={ctrl_label!r}",
        flush=True,
    )
    return adata, pert_col, ctrl_label


# =============================================================================
# APPROACH A: Perturbation-level pathway-score ↔ stability correlations
# =============================================================================

def _residual_diagnostics(sp, mag, pathway):
    """
    With Sp~mag ρ≈0.95, partials live on a thin residual. Report how much
    Sp variance magnitude removes and the residual pathway association R².
    Rank-scale OLS matches the rank partial Spearman path.
    """
    from scipy.stats import rankdata

    sp = np.asarray(sp, dtype=float)
    mag = np.asarray(mag, dtype=float)
    pw = np.asarray(pathway, dtype=float)
    mask = np.isfinite(sp) & np.isfinite(mag) & np.isfinite(pw)
    sp, mag, pw = sp[mask], mag[mask], pw[mask]
    n = len(sp)
    if n < 5:
        return {
            "r2_sp_on_magnitude": np.nan,
            "frac_sp_variance_remaining": np.nan,
            "sp_residual_sd": np.nan,
            "sp_sd": np.nan,
            "partial_r2": np.nan,
        }

    rsp, rmag, rpw = rankdata(sp), rankdata(mag), rankdata(pw)
    Z = np.column_stack([np.ones(n), rmag])
    # Sp residual after magnitude
    b_sp, _, _, _ = np.linalg.lstsq(Z, rsp, rcond=None)
    e_sp = rsp - Z @ b_sp
    # Pathway residual after magnitude (for partial R²)
    b_pw, _, _, _ = np.linalg.lstsq(Z, rpw, rcond=None)
    e_pw = rpw - Z @ b_pw

    ss_tot = float(np.sum((rsp - rsp.mean()) ** 2))
    ss_res = float(np.sum(e_sp ** 2))
    r2_mag = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    frac_rem = ss_res / ss_tot if ss_tot > 0 else np.nan
    # partial R² = squared correlation of residuals (= rho_partial² for Pearson-on-ranks)
    if np.std(e_sp) < 1e-15 or np.std(e_pw) < 1e-15:
        partial_r2 = np.nan
    else:
        partial_r2 = float(np.corrcoef(e_sp, e_pw)[0, 1] ** 2)

    return {
        "r2_sp_on_magnitude": float(r2_mag),
        "frac_sp_variance_remaining": float(frac_rem),
        "sp_residual_sd": float(np.std(e_sp, ddof=1)) if n > 1 else np.nan,
        "sp_sd": float(np.std(rsp, ddof=1)) if n > 1 else np.nan,
        "partial_r2": partial_r2,
    }


def approach_a(df, dataset_name, results_list):
    """
    For each pathway score, compute raw Spearman and partial Spearman
    (controlling for magnitude) against stability.  Appends rows to
    *results_list* (mutated in place).
    """
    print(f"\n{'=' * 70}")
    print(f"APPROACH A — Pathway Signature Correlations: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"  n = {len(df)} perturbations\n")

    pw_cols = sorted(c for c in df.columns if c.startswith('pw_'))
    # Deterministic per-dataset seed derived from global SEED
    seed_ctr = SEED + sum(ord(c) for c in dataset_name)

    for col in pw_cols:
        pw_name = col[3:]  # strip 'pw_'
        sub = df.dropna(subset=[col]).copy()
        if len(sub) < 15:
            print(f"  {pw_name}: skipped (n = {len(sub)} < 15)")
            continue

        try:
            from stats_utils import pathway_bootstrap_seed
            raw_seed = pathway_bootstrap_seed(
                dataset_name, pw_name, "raw", n_bootstrap=N_BOOTSTRAP
            )
            mag_seed = pathway_bootstrap_seed(
                dataset_name, pw_name, "partial_mag", n_bootstrap=N_BOOTSTRAP
            )
        except Exception:
            raw_seed, mag_seed = seed_ctr, seed_ctr + 1
            seed_ctr += 2

        # Always pass n_bootstrap=N_BOOTSTRAP (must match cell_quality_partial)
        raw = bootstrap_spearman_ci(
            sub['stability'].values, sub[col].values,
            n_bootstrap=N_BOOTSTRAP, seed=raw_seed)

        partial = bootstrap_partial_correlation_ci(
            sub['stability'].values, sub[col].values,
            sub['magnitude'].values,
            n_bootstrap=N_BOOTSTRAP, seed=mag_seed)

        resid = _residual_diagnostics(
            sub["stability"].values,
            sub["magnitude"].values,
            sub[col].values,
        )

        # Effect-size bin only. NEVER plot abs_rho_partial as a forest
        # bar: the old Fig 5c / S9 did, which flipped every negative rho
        # and left the CIs signed.
        abs_rho = abs(partial['rho_partial'])
        ci_excludes_zero = (
            not np.isnan(partial['ci_low'])
            and np.sign(partial['ci_low']) == np.sign(partial['ci_high'])
        )

        if abs_rho >= 0.3:
            effect = 'medium–large'
        elif abs_rho >= 0.2:
            effect = 'small–medium'
        elif abs_rho >= 0.1:
            effect = 'small'
        else:
            effect = 'negligible'

        # Provisional until BH is applied in _save_approach_a; final flag =
        # CI ∧ |ρ|>0.1 ∧ FDR<0.05 (see survival_status).
        survives_ci = abs_rho > 0.1 and ci_excludes_zero

        print(f"  {pw_name} (n={len(sub)}):")
        print(f"    Raw:     ρ = {raw['rho']:+.3f}  "
              f"[{raw['ci_low']:.3f}, {raw['ci_high']:.3f}]  "
              f"p = {raw['p']:.2e}")
        print(f"    Partial: ρ = {partial['rho_partial']:+.3f}  "
              f"[{partial['ci_low']:.3f}, {partial['ci_high']:.3f}]  "
              f"p = {partial['p']:.2e}")
        print(
            f"    Residual: R²(Sp|mag)={resid['r2_sp_on_magnitude']:.3f}  "
            f"frac_Sp_var_left={resid['frac_sp_variance_remaining']:.3f}  "
            f"partial_R²={resid['partial_r2']:.3f}"
        )
        print(f"    Effect: {effect}  |  "
              f"CI survives (|ρ|>0.1): {'YES' if survives_ci else 'no'}")

        results_list.append({
            'dataset':               dataset_name,
            'pathway':               pw_name,
            'pathway_full':          PATHWAY_FULL_NAMES.get(pw_name, ''),
            'n':                     len(sub),
            'rho_raw':               raw['rho'],
            'rho_raw_ci_low':        raw['ci_low'],
            'rho_raw_ci_high':       raw['ci_high'],
            'p_raw':                 raw['p'],
            'rho_partial':           partial['rho_partial'],
            'rho_partial_ci_low':    partial['ci_low'],
            'rho_partial_ci_high':   partial['ci_high'],
            'p_partial':             partial['p'],
            'abs_rho_partial':       abs_rho,
            'r2_sp_on_magnitude':    resid['r2_sp_on_magnitude'],
            'frac_sp_variance_remaining': resid['frac_sp_variance_remaining'],
            'sp_residual_sd':        resid['sp_residual_sd'],
            'sp_sd':                 resid['sp_sd'],
            'partial_r2':            resid['partial_r2'],
            'effect_size':           effect,
            'ci_excludes_zero':      ci_excludes_zero,
            'survives_magnitude_control': survives_ci,  # updated with FDR in save
        })


# =============================================================================
# APPROACH B: residual-Sp quartiles → caliper match → pseudobulk DE → GSEA
# =============================================================================
# Pre-specified question: after fixing pseudoreplication / mag selection /
# depth, does unbiased Hallmark enrichment recover apoptosis and p53 among
# discordant-associated terms? If the balance gate fails → drop the arm.


def _rank_residual_sp(stability: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Sp residual after magnitude — rank-scale OLS (matches Approach A path)."""
    from scipy.stats import rankdata

    sp = np.asarray(stability, dtype=float)
    mag = np.asarray(magnitude, dtype=float)
    rsp, rmag = rankdata(sp), rankdata(mag)
    Z = np.column_stack([np.ones(len(sp)), rmag])
    coef, _, _, _ = np.linalg.lstsq(Z, rsp, rcond=None)
    return rsp - Z @ coef


def _pert_qc_table(adata, pert_col: str, perts) -> pd.DataFrame:
    """Per-perturbation mean QC covariates from the normalized AnnData."""
    from scipy import sparse

    X = adata.X
    labels = adata.obs[pert_col].astype(str)
    # Index.str.startswith can return a ndarray (no .to_numpy) depending on pandas.
    gene_names = pd.Index(adata.var_names.astype(str))
    mito = np.asarray(gene_names.str.upper().str.startswith("MT-"), dtype=bool)
    rows = []
    for p in perts:
        idx = np.flatnonzero(labels.to_numpy() == str(p))
        if len(idx) == 0:
            continue
        Xs = X[idx]
        if sparse.issparse(Xs):
            n_counts = np.asarray(Xs.sum(axis=1)).ravel()
            n_genes = np.asarray((Xs > 0).sum(axis=1)).ravel()
            if mito.any():
                pct_mito = (
                    np.asarray(Xs[:, mito].sum(axis=1)).ravel()
                    / np.maximum(n_counts, 1.0)
                    * 100.0
                )
            else:
                pct_mito = np.zeros(len(idx))
        else:
            Xs = np.asarray(Xs)
            n_counts = Xs.sum(axis=1)
            n_genes = (Xs > 0).sum(axis=1)
            pct_mito = (
                Xs[:, mito].sum(axis=1) / np.maximum(n_counts, 1.0) * 100.0
                if mito.any()
                else np.zeros(len(idx))
            )
        rows.append({
            "perturbation": str(p),
            "n_cells": int(len(idx)),
            "mean_n_counts": float(np.mean(n_counts)),
            "mean_n_genes": float(np.mean(n_genes)),
            "mean_percent_mito": float(np.mean(pct_mito)),
        })
    return pd.DataFrame(rows).set_index("perturbation")


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (Cohen's d pooling)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _caliper_match_q4_to_q1(
    df: pd.DataFrame,
    q1_idx,
    q4_idx,
    caliper_sd: float,
) -> pd.DataFrame:
    """
    Greedy 1:1 match: each Q4 perturbation → nearest unused Q1 by magnitude
    within caliper_sd × SD(magnitude). Returns matched-pair table.
    """
    mag_sd = float(df["magnitude"].std(ddof=1))
    cal = caliper_sd * mag_sd if mag_sd > 0 else 0.0
    q1_pool = list(q1_idx)
    pairs = []
    # Match largest |residual| discordant first (most extreme Q4)
    q4_ordered = (
        df.loc[list(q4_idx)]
        .assign(_abs=lambda d: d["sp_resid"].abs())
        .sort_values("_abs", ascending=False)
        .index
    )
    used_q1 = set()
    for p4 in q4_ordered:
        m4 = float(df.loc[p4, "magnitude"])
        best, best_d = None, np.inf
        for p1 in q1_pool:
            if p1 in used_q1:
                continue
            d = abs(float(df.loc[p1, "magnitude"]) - m4)
            if d <= cal and d < best_d:
                best, best_d = p1, d
        if best is None:
            continue
        used_q1.add(best)
        pairs.append({
            "q4_pert": str(p4),
            "q1_pert": str(best),
            "mag_q4": m4,
            "mag_q1": float(df.loc[best, "magnitude"]),
            "mag_abs_diff": float(best_d),
            "sp_resid_q4": float(df.loc[p4, "sp_resid"]),
            "sp_resid_q1": float(df.loc[best, "sp_resid"]),
        })
    return pd.DataFrame(pairs)


def _pseudobulk_means(adata, pert_col: str, perts, n_cells: int, seed: int):
    """Equalize cells/pert then mean expression → (n_perts × n_genes)."""
    from scipy import sparse

    rng = np.random.default_rng(seed)
    labels = adata.obs[pert_col].astype(str).to_numpy()
    mats = []
    kept = []
    for p in perts:
        idx = np.flatnonzero(labels == str(p))
        if len(idx) < max(5, n_cells // 5):
            continue
        if len(idx) > n_cells:
            idx = rng.choice(idx, size=n_cells, replace=False)
        Xs = adata.X[idx]
        if sparse.issparse(Xs):
            mu = np.asarray(Xs.mean(axis=0)).ravel()
        else:
            mu = np.asarray(Xs).mean(axis=0).ravel()
        mats.append(mu)
        kept.append(str(p))
    if not mats:
        return None, []
    return np.vstack(mats), kept


def _pseudobulk_mw_de(X_q4, X_q1, gene_names) -> pd.DataFrame:
    """Mann–Whitney U across perturbations (not cells). Rank by effect z."""
    from scipy.stats import mannwhitneyu, norm

    n_genes = X_q4.shape[1]
    rows = []
    for j in range(n_genes):
        a, b = X_q4[:, j], X_q1[:, j]
        if np.allclose(a, a[0]) and np.allclose(b, b[0]) and a[0] == b[0]:
            continue
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        # Direction: positive score = higher in Q4 (discordant)
        n1, n2 = len(a), len(b)
        mu_u = n1 * n2 / 2.0
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u - mu_u) / sigma if sigma > 0 else 0.0
        # flip so positive z = Q4 > Q1
        # mannwhitneyu(a,b): large U means a tends > b
        rows.append({
            "names": gene_names[j],
            "scores": float(z),
            "logfoldchanges": float(np.mean(a) - np.mean(b)),
            "pvals": float(p),
            "n_q4": int(n1),
            "n_q1": int(n2),
        })
    de = pd.DataFrame(rows)
    if de.empty:
        return de
    from statsmodels.stats.multitest import multipletests

    de["pvals_adj"] = multipletests(de["pvals"].values, method="fdr_bh")[1]
    return de.sort_values("scores", ascending=False).reset_index(drop=True)


def approach_b(adata_norm, pert_col, ctrl_label, shesha_df,
               dataset_name, min_cells=50, force_confounded=False):
    """
    Redesigned Approach B (2026-08-05):

      1. Rank-residualize Sp on magnitude (same residual family as Approach A).
      2. Quartile on residual Sp → Q1 concordant / Q4 discordant.
      3. Caliper-match Q4→Q1 on magnitude; gate on SMD for mag / n_cells /
         percent_mito / n_genes.
      4. Equalize cells/pert → pseudobulk means → Mann–Whitney across perts.
      5. Pre-ranked GSEA: Hallmark (main question); KEGG/Reactome (SI).

    Pre-specified question: do Hallmark apoptosis / p53 appear among
    discordant-associated terms? If the balance gate fails → drop (return None).
    """
    if not HAS_GSEAPY:
        print(f"\n  Approach B skipped for {dataset_name}: gseapy not installed.")
        return None, None

    caliper = float(getattr(cfg, "APPROACH_B_CALIPER_MAG_SD", 0.25))
    smd_max = float(getattr(cfg, "APPROACH_B_SMD_MAX", 0.25))
    min_pairs = int(getattr(cfg, "APPROACH_B_MIN_MATCHED_PAIRS", 15))
    n_eq = int(getattr(cfg, "APPROACH_B_EQUALIZE_CELLS", 50))
    targets = tuple(getattr(cfg, "APPROACH_B_TARGET_TERMS", ("APOPTOSIS", "P53")))

    print(f"\n{'=' * 70}")
    print(f"APPROACH B v2 — residual-Sp / caliper / pseudobulk: {dataset_name}")
    print(f"{'=' * 70}")
    print(
        "  Pre-specified question: do Hallmark apoptosis / p53 enrich among "
        "discordant (Q4) terms after matching?"
    )
    print(
        f"  Power floor: datasets={list(getattr(cfg, 'APPROACH_B_DATASETS', ()))}; "
        f"need ≥{min_pairs} per residual quartile before caliper. "
        "If balance gate fails → Approach A carries the result "
        "(do not relax SMD after seeing failures).",
        flush=True,
    )

    df = shesha_df.copy()
    if "perturbation" in df.columns and df.index.name != "perturbation":
        df = df.set_index(df["perturbation"].astype(str))
    df.index = df.index.astype(str)
    df["sp_resid"] = _rank_residual_sp(
        df["stability"].to_numpy(), df["magnitude"].to_numpy()
    )
    # High residual Sp = more coherent than mag predicts (concordant)
    # Low residual Sp = discordant
    try:
        df["resid_q"] = pd.qcut(
            df["sp_resid"], q=4, labels=["Q4", "Q3", "Q2", "Q1"],
            duplicates="drop",
        )
    except ValueError as e:
        print(f"  Cannot form residual quartiles ({e}) — dropping Approach B.")
        return None, None

    # qcut labels: lowest resid → Q4 discordant, highest → Q1 concordant
    q1_idx = df.index[df["resid_q"] == "Q1"]
    q4_idx = df.index[df["resid_q"] == "Q4"]
    print(
        f"  Perturbations: {len(df)}; "
        f"Q1 (concordant/high resid Sp)={len(q1_idx)}; "
        f"Q4 (discordant/low resid Sp)={len(q4_idx)}"
    )
    if len(q1_idx) < min_pairs or len(q4_idx) < min_pairs:
        print(
            f"  Too few quartile members for matching (need ≥{min_pairs}) — "
            "dropping Approach B."
        )
        return None, None

    pairs = _caliper_match_q4_to_q1(df, q1_idx, q4_idx, caliper_sd=caliper)
    print(
        f"  Caliper match (|Δmag| ≤ {caliper}·SD): "
        f"{len(pairs)} pairs (of {len(q4_idx)} Q4)"
    )
    if len(pairs) < min_pairs and not force_confounded:
        print(
            f"  Balance gate FAIL: matched pairs {len(pairs)} < {min_pairs}. "
            "Approach B dropped (Approach A carries the result)."
        )
        return None, None

    q4_m = pairs["q4_pert"].tolist()
    q1_m = pairs["q1_pert"].tolist()
    qc = _pert_qc_table(adata_norm, pert_col, q4_m + q1_m)
    bal_rows = []
    for cov in ("magnitude", "n_cells", "mean_n_counts", "mean_n_genes", "mean_percent_mito"):
        if cov == "magnitude":
            a = pairs["mag_q4"].to_numpy()
            b = pairs["mag_q1"].to_numpy()
        else:
            if cov not in qc.columns:
                continue
            a = qc.loc[q4_m, cov].to_numpy()
            b = qc.loc[q1_m, cov].to_numpy()
        s = _smd(a, b)
        bal_rows.append({
            "covariate": cov,
            "mean_q4": float(np.mean(a)),
            "mean_q1": float(np.mean(b)),
            "smd_q4_minus_q1": s,
            "gate_fail": bool(np.isfinite(s) and abs(s) > smd_max),
        })
    bal = pd.DataFrame(bal_rows)
    tag = dataset_name.split("(")[0].strip().lower().replace(" ", "_")
    bal_path = OUTPUT_DIR / f"pathway_approach_b_balance_{tag}.csv"
    pairs_path = OUTPUT_DIR / f"pathway_approach_b_matched_pairs_{tag}.csv"
    bal.to_csv(bal_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    print("  Balance table (SMD gate):")
    print(bal.to_string(index=False))
    print(f"  Wrote {bal_path.name}; {pairs_path.name}")

    failed = bal.loc[bal["gate_fail"], "covariate"].tolist()
    if failed and not force_confounded:
        print(
            f"  Balance gate FAIL on {failed} (|SMD| > {smd_max}). "
            "Approach B dropped — attempted but could not be balanced. "
            "Approach A carries the pathway result."
        )
        return None, None
    if failed and force_confounded:
        print("  WARNING: --force-confounded-gsea; writing despite balance FAIL.")

    # Pseudobulk DE (unit = perturbation)
    print(
        f"  Pseudobulk DE: equalize to {n_eq} cells/pert, "
        f"Mann–Whitney across {len(q4_m)} vs {len(q1_m)} perturbations…"
    )
    X4, kept4 = _pseudobulk_means(adata_norm, pert_col, q4_m, n_eq, SEED)
    X1, kept1 = _pseudobulk_means(adata_norm, pert_col, q1_m, n_eq, SEED + 1)
    if X4 is None or X1 is None or len(kept4) < min_pairs or len(kept1) < min_pairs:
        print("  Pseudobulk failed (too few perts with cells) — dropping Approach B.")
        return None, None
    # Align to shared gene space
    genes = list(adata_norm.var_names.astype(str))
    de_df = _pseudobulk_mw_de(X4, X1, genes)
    if de_df.empty:
        print("  Pseudobulk DE produced no genes — dropping.")
        return None, None
    de_path = OUTPUT_DIR / f"pathway_de_Q4_vs_Q1_pseudobulk_{tag}.csv"
    de_df.to_csv(de_path, index=False)
    print(f"  Saved pseudobulk DE ({len(de_df)} genes) → {de_path.name}")
    print("  Top 10 UP in Q4 (discordant):")
    print(de_df.head(10)[["names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))

    rnk = (
        de_df[["names", "scores"]]
        .drop_duplicates(subset="names")
        .rename(columns={"names": "Gene", "scores": "Score"})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    all_gsea = []
    hallmark_hit = {t: False for t in targets}
    for lib in GSEA_LIBRARIES:
        role = "MAIN" if lib == GSEA_LIBRARY_HALLMARK else "SI"
        print(f"\n  GSEA [{role}]: {lib} ...")
        try:
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=lib,
                min_size=5,
                max_size=500,
                permutation_num=GSEA_PERMUTATIONS,
                outdir=None,
                seed=SEED,
                verbose=False,
            )
            res = pre_res.res2d.copy()
            res["library"] = lib
            res["role"] = role
            fdr_col = "FDR q-val" if "FDR q-val" in res.columns else "FDR q-val"
            if fdr_col not in res.columns:
                # gseapy version variance
                for c in res.columns:
                    if "fdr" in c.lower():
                        fdr_col = c
                        break
            sig = res[res[fdr_col].astype(float) < 0.25].copy()
            if len(sig):
                sig = sig.sort_values("NES" if "NES" in sig.columns else sig.columns[1])
                show = sig.head(15) if role == "MAIN" else sig.head(5)
                print(f"    {len(sig)} terms FDR<0.25 (showing ≤{len(show)}):")
                for _, r in show.iterrows():
                    term = str(r.get("Term", r.get("term", "")))
                    nes = float(r["NES"]) if "NES" in r else np.nan
                    fdr = float(r[fdr_col])
                    print(f"      NES={nes:+.2f}  FDR={fdr:.3f}  {term}")
                    if role == "MAIN":
                        up = term.upper()
                        for t in targets:
                            if t in up:
                                hallmark_hit[t] = True
            else:
                print("    No terms with FDR < 0.25")
            all_gsea.append(res)
        except Exception as e:
            print(f"    Error: {e}")

    gsea_df = None
    if all_gsea:
        gsea_df = pd.concat(all_gsea, ignore_index=True)
        gsea_df["dataset"] = dataset_name
        gsea_df["design"] = "residual_sp_caliper_pseudobulk_v2"
        gsea_path = OUTPUT_DIR / f"pathway_gsea_Q4_vs_Q1_{tag}.csv"
        gsea_df.to_csv(gsea_path, index=False)
        print(f"\n  Saved GSEA → {gsea_path.name}")

    print("\n  PRE-SPECIFIED Hallmark recovery (apoptosis / p53):")
    for t, hit in hallmark_hit.items():
        print(f"    {t}: {'YES — supports Approach A' if hit else 'NO — limitation (signal in scores only)'}")
    verdict_path = OUTPUT_DIR / f"pathway_approach_b_verdict_{tag}.json"
    import json as _json
    verdict_path.write_text(_json.dumps({
        "dataset": dataset_name,
        "n_matched_pairs": int(len(pairs)),
        "balance_gate_failed_covariates": failed,
        "force_confounded": bool(force_confounded),
        "hallmark_target_hits": hallmark_hit,
        "design": "residual_sp_caliper_pseudobulk_v2",
        "config_version": CONFIG_VERSION,
    }, indent=2))
    print(f"  Wrote {verdict_path.name}")
    return de_df, gsea_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adamson-h5ad",
        type=Path,
        default=None,
        help="Path to adamson_2016_upr_perturb_seq.h5ad "
             "(default: /tmp/pertpy_data/adamson_2016_upr_perturb_seq.h5ad)",
    )
    parser.add_argument(
        "--adamson-only",
        action="store_true",
        help="Run only Adamson UPR Approach A (skip Norman/Dixit/Replogle)",
    )
    parser.add_argument(
        "--skip-approach-b",
        action="store_true",
        default=True,
        help="Skip Approach B (DEFAULT). Opt in with --run-approach-b.",
    )
    parser.add_argument(
        "--run-approach-b",
        dest="skip_approach_b",
        action="store_false",
        help="Run Approach B v2: residual-Sp quartiles, caliper match, "
             "pseudobulk DE, Hallmark GSEA. Drops cleanly if balance gate fails.",
    )
    parser.add_argument(
        "--force-confounded-gsea",
        action="store_true",
        help="Write Approach B even if SMD balance gate fails (SI debug only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-detected shesha-crispr/)",
    )
    parser.add_argument(
        "--frozen-sp",
        type=Path,
        default=None,
        help="Join Sp/mag from this CSV (default: <out>/frozen_sp_scores.csv)",
    )
    parser.add_argument(
        "--recompute-sp",
        action="store_true",
        help="Ignore frozen Sp table and recompute Sp inside this script",
    )
    parser.add_argument(
        "--skip-fail",
        action="store_true",
        help="Continue after a dataset failure (default: abort; do not write partial CSVs)",
    )
    parser.add_argument(
        "--allow-stale-sp",
        action="store_true",
        help="Skip frozen Sp version/n_rows/Replogle check (dangerous)",
    )
    parser.add_argument(
        "--skip-papalexi",
        action="store_true",
        help="Skip Papalexi Approach A",
    )
    parser.add_argument(
        "--skip-pilot",
        action="store_true",
        help="Skip Adamson pilot (default: attempt; n=8 usually skips Approach A)",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.out_dir is not None:
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen_csv = args.frozen_sp
    if frozen_csv is None:
        cand = OUTPUT_DIR / "frozen_sp_scores.csv"
        frozen_csv = cand if cand.exists() else None
    print(f"frozen_sp={'recompute' if args.recompute_sp else frozen_csv}")

    if frozen_csv is not None and not args.recompute_sp and not args.allow_stale_sp:
        from pipeline_core import assert_frozen_sp_compatible
        assert_frozen_sp_compatible(frozen_csv)

    def _proc(name, **kwargs):
        return load_and_process(
            name,
            frozen_csv=frozen_csv,
            recompute_sp=args.recompute_sp,
            **kwargs,
        )

    all_corr = []  # Approach A rows
    all_per_pert = []  # joined Sp + pw_* for QC gate / one-number freeze
    all_coverage = []  # Hallmark gene overlap per dataset (Replogle caveat)
    failures = {}

    def _record(df, coverage=None):
        if df is not None and len(df):
            all_per_pert.append(df.reset_index(drop=True))
        if coverage:
            all_coverage.extend(coverage)

    def _fail(name, err):
        failures[name] = str(err)
        print(f"{name} FAILED: {err}")
        if not args.skip_fail:
            raise RuntimeError(
                f"{name} failed: {err}\n"
                "Refusing to write partial pathway outputs. "
                "Fix the failure (often truncated Replogle cache) or pass --skip-fail."
            ) from err

    # =================================================================
    # ADAMSON 2016 UPR (CRISPRi) — Approach A (positive-control stress panel)
    # =================================================================
    print("\n" + "=" * 80)
    print("ADAMSON 2016 UPR (CRISPRi) — Approach A")
    print("=" * 80)
    try:
        adata_adam, pert_a, ctrl_a = load_adamson_upr(args.adamson_h5ad)
        df_a, _, cov_a = _proc(
            "Adamson 2016 UPR (CRISPRi)",
            adata=adata_adam,
            pert_col=pert_a,
            ctrl_label=ctrl_a,
            min_cells=ADAMSON_MIN_CELLS,
            already_materialized=True,
        )
        approach_a(df_a, "Adamson 2016 UPR (CRISPRi)", all_corr)
        _record(df_a, cov_a)
        del adata_adam
    except Exception as e:
        _fail("Adamson 2016 UPR (CRISPRi)", e)

    if args.adamson_only:
        _save_approach_a(all_corr)
        _save_per_pert(all_per_pert)
        _save_coverage(all_coverage)
        return

    # =================================================================
    # PAPALEXI 2021 (CRISPR-KO) — Approach A (n=24; thin but ≥15)
    # =================================================================
    if not args.skip_papalexi:
        print("\n" + "=" * 80)
        print("PAPALEXI 2021 (CRISPR-KO) — Approach A")
        print("=" * 80)
        try:
            adata_p, pert_p, ctrl_p = load_via_pipeline_core(
                "Papalexi 2021 (CRISPR-KO)"
            )
            df_p, _, cov_p = _proc(
                "Papalexi 2021 (CRISPR-KO)",
                adata=adata_p,
                pert_col=pert_p,
                ctrl_label=ctrl_p,
                min_cells=MIN_CELLS,
                already_materialized=True,
            )
            approach_a(df_p, "Papalexi 2021 (CRISPR-KO)", all_corr)
            _record(df_p, cov_p)
            del adata_p
        except Exception as e:
            _fail("Papalexi 2021 (CRISPR-KO)", e)

    # =================================================================
    # ADAMSON 2016 pilot — logged attempt; Approach A skips if n < 15
    # =================================================================
    if not args.skip_pilot:
        print("\n" + "=" * 80)
        print("ADAMSON 2016 pilot (CRISPRi) — Approach A (expect n=8 skip)")
        print("=" * 80)
        try:
            adata_pi, pert_pi, ctrl_pi = load_via_pipeline_core(
                "Adamson 2016 pilot (CRISPRi)"
            )
            df_pi, _, cov_pi = _proc(
                "Adamson 2016 pilot (CRISPRi)",
                adata=adata_pi,
                pert_col=pert_pi,
                ctrl_label=ctrl_pi,
                min_cells=MIN_CELLS,
                already_materialized=True,
            )
            approach_a(df_pi, "Adamson 2016 pilot (CRISPRi)", all_corr)
            _record(df_pi, cov_pi)
            del adata_pi
        except Exception as e:
            print(f"Adamson pilot FAILED (non-fatal): {e}")
            failures["Adamson 2016 pilot (CRISPRi)"] = str(e)

    # Always load Norman / Dixit / Replogle via pipeline_core (local h5ad +
    # stable materialize). Do NOT score on a raw pertpy object without the
    # ≤100/pert pin — that was the |mag drift vs cell_quality_partial.
    print("\nLoading Replogle/Norman/Dixit via pipeline_core (stable downsample).")
    for ds_name in (
        "Replogle 2022 (CRISPRi)",
        "Norman 2019 (CRISPRa)",
        "Dixit 2016 (CRISPR-KO)",
    ):
        print("\n" + "=" * 80)
        print(ds_name)
        print("=" * 80)
        try:
            adata_x, pert_x, ctrl_x = load_via_pipeline_core(ds_name)
            df_x, adata_kept, cov_x = _proc(
                ds_name,
                adata=adata_x,
                pert_col=pert_x,
                ctrl_label=ctrl_x,
                min_cells=MIN_CELLS,
                already_materialized=True,
            )
            approach_a(df_x, ds_name, all_corr)
            _record(df_x, cov_x)
            b_datasets = tuple(
                getattr(
                    cfg,
                    "APPROACH_B_DATASETS",
                    ("Replogle 2022 (CRISPRi)", "Norman 2019 (CRISPRa)"),
                )
            )
            if not args.skip_approach_b and ds_name in b_datasets:
                approach_b(
                    adata_kept, pert_x, ctrl_x, df_x, ds_name, MIN_CELLS,
                    force_confounded=args.force_confounded_gsea,
                )
            elif not args.skip_approach_b and ds_name not in b_datasets:
                print(
                    f"  Approach B skipped for {ds_name}: below power floor "
                    f"(pre-specified APPROACH_B_DATASETS="
                    f"{list(b_datasets)}; need ≥"
                    f"{getattr(cfg, 'APPROACH_B_MIN_MATCHED_PAIRS', 15)} "
                    "per residual quartile).",
                    flush=True,
                )
            del adata_x, adata_kept
        except Exception as e:
            _fail(ds_name, e)

    if all_per_pert:
        got = set()
        for frame in all_per_pert:
            got.update(
                cfg.resolve_dataset_name(str(x)) for x in frame["dataset"].unique()
            )
        required = list(getattr(cfg, "PATHWAY_REQUIRED_DATASETS", []))
        if args.skip_papalexi:
            required = [d for d in required if "Papalexi" not in d]
        missing = [d for d in required if d not in got]
        if missing and not args.skip_fail:
            raise RuntimeError(
                f"Partial pathway run — missing {missing}. "
                f"Got n={sum(len(f) for f in all_per_pert)} rows "
                f"(five-scoreable expectation ≈ "
                f"{getattr(cfg, 'FROZEN_SP_EXPECTED_SCOREABLE_N', 2277)}). "
                "Not writing CSVs. Fix load failures first."
            )

    _save_approach_a(all_corr)
    _save_coverage(all_coverage)
    _save_per_pert(all_per_pert)

    print(f"\n{'=' * 80}")
    print("PATHWAY ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("  Approach A: pathway_signature_correlations.csv")
    print("  Per-pert:   pathway_scores_per_pert.csv  (join into QC gate)")
    if HAS_GSEAPY and not args.skip_approach_b:
        print("  Approach B: pathway_de_Q4_vs_Q1_*.csv / pathway_gsea_Q4_vs_Q1_*.csv")


def _save_approach_a(all_corr):
    if not all_corr:
        print("No Approach A results to save.")
        return
    corr_df = pd.DataFrame(all_corr)
    # BH-FDR within each dataset across pathways (exploratory Hallmark set)
    corr_df["p_partial_fdr_bh"] = np.nan
    for ds, idx in corr_df.groupby("dataset").groups.items():
        pvals = corr_df.loc[idx, "p_partial"].astype(float).values
        if np.all(np.isnan(pvals)):
            continue
        # multipletests needs finite p; fill nan with 1
        p_fill = np.where(np.isfinite(pvals), pvals, 1.0)
        corr_df.loc[idx, "p_partial_fdr_bh"] = multipletests(p_fill, method="fdr_bh")[1]
    corr_df["config_version"] = CONFIG_VERSION
    corr_df["partial_method"] = "partial_spearman_rank"
    # Manuscript rule: CI ∧ |ρ|>0.1 ∧ BH-FDR<0.05; knife-edge / CI↔FDR clash → indeterminate
    try:
        from stats_utils import survival_status
        statuses, knife, disagree = [], [], []
        final = []
        for _, r in corr_df.iterrows():
            st = survival_status(
                r["rho_partial"], r["rho_partial_ci_low"], r["rho_partial_ci_high"],
                fdr=r["p_partial_fdr_bh"],
            )
            statuses.append(st["status"])
            knife.append(st["knife_edge"])
            disagree.append(st["ci_fdr_disagree"])
            final.append(st["survives"])
        corr_df["survival_status"] = statuses
        corr_df["knife_edge_ci"] = knife
        corr_df["ci_fdr_disagree"] = disagree
        corr_df["survives_magnitude_control"] = final
    except Exception as e:
        print(f"WARNING: survival_status failed ({e}); keeping CI-only flag")

    out_path = OUTPUT_DIR / "pathway_signature_correlations.csv"
    corr_df.to_csv(out_path, index=False)
    print(f"\n{'=' * 80}")
    crit = getattr(cfg, "SURVIVAL_CRITERION_ID", "ci_and_fdr.v1") if 'cfg' in dir() else "ci_and_fdr.v1"
    try:
        import pipeline_config as _cfg
        crit = _cfg.SURVIVAL_CRITERION_ID
        nb = _cfg.N_BOOTSTRAP
    except Exception:
        nb = N_BOOTSTRAP
    print(f"APPROACH A — SUMMARY (criterion={crit}; n_bootstrap={nb})")
    print(f"{'=' * 80}")
    cols_show = [
        "dataset", "pathway", "n", "rho_raw", "rho_partial", "partial_r2",
        "frac_sp_variance_remaining", "p_partial_fdr_bh", "survival_status",
        "survives_magnitude_control",
    ]
    print(corr_df[cols_show].to_string(index=False))
    n_ind = int((corr_df.get("survival_status") == "indeterminate").sum()) if "survival_status" in corr_df else 0
    if n_ind:
        print(f"\n  {n_ind} indeterminate (knife-edge CI or CI/FDR disagree) — do not count as survivors.")
    print(f"\nSaved → {out_path}")


def _save_per_pert(all_per_pert):
    if not all_per_pert:
        return
    df = pd.concat(all_per_pert, ignore_index=True)
    df["config_version"] = CONFIG_VERSION
    out_path = OUTPUT_DIR / "pathway_scores_per_pert.csv"
    # Exclude pilot from the scoreable count check
    scoreable = df[
        ~df["dataset"].astype(str).str.contains("pilot", case=False, na=False)
    ]
    n_score = len(scoreable)
    expect = int(getattr(cfg, "FROZEN_SP_EXPECTED_SCOREABLE_N", 2277))
    if n_score < expect * 0.9:
        raise RuntimeError(
            f"pathway_scores_per_pert looks partial (scoreable n={n_score} << {expect}). "
            "Not writing CSV — fix load failures (often truncated Replogle cache)."
        )
    df.to_csv(out_path, index=False)
    print(
        f"Saved → {out_path} ({len(df)} rows total; "
        f"scoreable (no pilot)={n_score}; expect {expect})"
    )


def _save_coverage(all_coverage):
    if not all_coverage:
        return
    df = pd.DataFrame(all_coverage)
    df["config_version"] = CONFIG_VERSION
    out_path = OUTPUT_DIR / "pathway_gene_coverage.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    # Flag the only dual-pathway QC survivor's incomplete Hallmark coverage
    focus = df[df["pathway"].isin(["p53", "Apoptosis"])]
    for _, r in focus.iterrows():
        if r["pct_overlap"] < 80:
            print(
                f"  COVERAGE CAVEAT: {r['dataset']} {r['pathway']} "
                f"{r['n_overlap']}/{r['n_hallmark']} ({r['pct_overlap']:.0f}%) "
                f"on {r['n_genes_dataset']} genes — state in methods if this "
                f"dataset carries a primary claim."
            )


if __name__ == "__main__":
    main()
