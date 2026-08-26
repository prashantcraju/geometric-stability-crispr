#!/usr/bin/env python3
"""
scGPT Geometric Stability Analysis -- All Datasets

Computes geometric stability and magnitude metrics using scGPT embeddings
on all 6 CRISPR perturbation datasets in pipeline_config.DATASETS
(frozen display names):
    - Norman 2019 (CRISPRa)
    - Adamson 2016 UPR (CRISPRi)     # same arm as adamson_upr_spike.py
    - Adamson 2016 pilot (CRISPRi)   # TF pilot arm (contrast)
    - Dixit 2016 (CRISPR-KO)
    - Papalexi 2021 (CRISPR-KO)
    - Replogle 2022 (CRISPRi)

Saves one CSV per dataset (scgpt_<name>.csv), then merges to
scgpt_all_datasets.csv + scgpt_correlations.csv under --out-dir.

Run one dataset at a time with --datasets norman. Prefer GPU + --batch-size 256.
Optional --downsample applies frozen PCA caps (off by default).

REQUIRES:
    - Pre-downloaded scGPT pretrained model (https://github.com/bowang-lab/scGPT)
    - GPU strongly recommended (CPU is ~40s/batch)
"""

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import json
import subprocess
import sys
import os
from pathlib import Path

# CUBLAS workspace MUST be set before the first CUDA/cuBLAS call. Setting it
# later in main() is a no-op for determinism (flag-looks-set-but-didn't-apply).
if "--deterministic" in sys.argv:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(
        "PRE-TORCH: CUBLAS_WORKSPACE_CONFIG=:4096:8 (--deterministic in argv)",
        flush=True,
    )


def _maybe_check_identical_early() -> None:
    """
    Compare two CSVs without Colab bootstrap / torch install.
    Usage: python scgpt_analysis.py --check-identical A.csv B.csv
    """
    if "--check-identical" not in sys.argv:
        return
    idx = sys.argv.index("--check-identical")
    paths = sys.argv[idx + 1 : idx + 3]
    if len(paths) < 2 or paths[0].startswith("-") or paths[1].startswith("-"):
        raise SystemExit(
            "Usage: python scgpt_analysis.py --check-identical CSV_A CSV_B\n"
            "Snapshot run1 first, e.g.:\n"
            "  cp $OUT/scgpt_Papalexi_2021__CRISPR_KO_.csv /tmp/pap_run1.csv"
        )
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    path_a, path_b = Path(paths[0]), Path(paths[1])
    for p in (path_a, path_b):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Snapshot the first embed before the second run:\n"
                f"  cp <out>/scgpt_Papalexi_2021__CRISPR_KO_.csv {path_a}"
            )
    cols = ["perturbation", "stability", "magnitude", "n_cells"]
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    for c in cols:
        if c not in a.columns or c not in b.columns:
            raise SystemExit(f"missing column {c}")
    a = a[cols].sort_values("perturbation").reset_index(drop=True)
    b = b[cols].sort_values("perturbation").reset_index(drop=True)
    same = len(a) == len(b) and a.equals(b)
    sp_max = float(np.nanmax(np.abs(a["stability"] - b["stability"]))) if len(a) else float("nan")
    mag_max = float(np.nanmax(np.abs(a["magnitude"] - b["magnitude"]))) if len(a) else float("nan")
    sp_rho = (
        float(spearmanr(a["stability"], b["stability"])[0]) if len(a) >= 3 else float("nan")
    )
    report = {
        "identical": bool(same),
        "n": int(len(a)),
        "path_a": str(path_a),
        "path_b": str(path_b),
        "max_abs_sp_diff": sp_max,
        "max_abs_mag_diff": mag_max,
        "spearman_sp_run1_vs_run2": sp_rho,
        "reason": (
            "bit-identical"
            if same
            else (
                "values differ — Sp ranks fragile to GPU float noise at small n "
                f"(max|ΔSp|={sp_max:.3g}, max|Δmag|={mag_max:.3g})"
            )
        ),
    }
    print(report)
    raise SystemExit(0 if same else 1)


_maybe_check_identical_early()


def _purge_modules(prefix: str) -> None:
    for mod in list(sys.modules):
        if mod == prefix or mod.startswith(prefix + "."):
            del sys.modules[mod]


def _zarr_status():
    """Return (major_version_or_None, error_or_None)."""
    try:
        import zarr
        return int(str(zarr.__version__).split(".")[0]), None
    except Exception as e:
        return None, e


# zarr 2.x imports numcodecs.blosc.cbuffer_sizes — removed in numcodecs>=0.16
ZARR_PIN = "zarr==2.18.7"
NUMCODECS_PIN = "numcodecs==0.15.1"


def _install_zarr_stack() -> None:
    print(f"Installing {NUMCODECS_PIN} + {ZARR_PIN}…", flush=True)
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "zarr", "numcodecs"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "-q",
            NUMCODECS_PIN, ZARR_PIN,
        ],
    )
    _purge_modules("zarr")
    _purge_modules("numcodecs")


def _ensure_zarr_v2(*, hard: bool = True) -> None:
    """anndata 0.11.x needs zarr 2.x + numcodecs<0.16."""
    major, err = _zarr_status()
    if major == 2:
        print("zarr OK: 2.x", flush=True)
        return
    print(
        f"Fixing zarr/numcodecs (current major={major}, err={err!r})…",
        flush=True,
    )
    _install_zarr_stack()
    major, err = _zarr_status()
    if major == 2:
        print("zarr OK: 2.x", flush=True)
        return
    msg = (
        f"zarr still not importable (major={major}, err={err!r}).\n"
        "Binary packages were already loaded — Colab needs a restart.\n\n"
        "1) Runtime → Restart session\n"
        "2) Run this cell once:\n"
        f"     !pip install -q '{NUMCODECS_PIN}' '{ZARR_PIN}' 'numpy>=2.0,<2.5'\n"
        "3) Re-run scgpt_analysis.py (do NOT set SHESHA_SKIP_BOOTSTRAP yet)\n"
        "4) Quick check: import zarr; print(zarr.__version__)"
    )
    if hard:
        raise SystemExit(msg)
    print("WARNING: " + msg, flush=True)


try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def _torch_stack_ok() -> bool:
    try:
        import torch as _torch  # noqa: F401
        import torchtext  # noqa: F401
        return True
    except Exception:
        return False


def _install_torch_stack() -> None:
    print("Installing torch==2.3.1 + torchtext==0.18.0 (cu121)…", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1",
        "torchtext==0.18.0",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    _purge_modules("torch")
    _purge_modules("torchvision")
    _purge_modules("torchaudio")
    _purge_modules("torchtext")


def _colab_bootstrap() -> None:
    """Install pins needed for scGPT + anndata on Colab."""
    print("Colab bootstrap: installing deps…", flush=True)
    # Install zarr stack first with pinned numcodecs<0.16 (unpinned pulls 0.16+ and breaks)
    _install_zarr_stack()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy>=2.0,<2.5",
        "scanpy==1.11.1", "pertpy==1.0.6", "mudata==0.3.7",
        "anndata==0.11.4", "statsmodels", "tqdm",
        "scikit-learn", "shesha-geometry", "scgpt", "transformers",
        NUMCODECS_PIN, ZARR_PIN,  # re-assert after other packages
    ])
    _install_torch_stack()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy>=2.0,<2.5", NUMCODECS_PIN, ZARR_PIN,
    ])
    _ensure_zarr_v2(hard=True)


def _zarr_ok() -> bool:
    major, _ = _zarr_status()
    return major == 2


# Skip bootstrap ONLY if torch+torchtext AND zarr already import
_want_skip = os.environ.get("SHESHA_SKIP_BOOTSTRAP", "").strip().lower() in {
    "1", "true", "yes",
}
if IN_COLAB:
    if _want_skip and _torch_stack_ok() and _zarr_ok():
        print("SHESHA_SKIP_BOOTSTRAP: deps OK, skipping pip", flush=True)
    else:
        if _want_skip:
            print(
                "SHESHA_SKIP_BOOTSTRAP set but deps incomplete — bootstrapping anyway",
                flush=True,
            )
        _colab_bootstrap()
elif not IN_COLAB:
    if not _zarr_ok():
        _ensure_zarr_v2(hard=False)


def _ensure_torchtext() -> None:
    """Ensure torch + torchtext import; install if missing (do not rely on skip env)."""
    if _torch_stack_ok():
        import torch as _torch
        print(f"torch={_torch.__version__}  torchtext OK", flush=True)
        return

    print("torch/torchtext not importable — installing…", flush=True)
    try:
        _install_torch_stack()
    except Exception as e:
        raise SystemExit(
            f"Failed to pip-install torch/torchtext: {e}\n"
            "In a Colab cell (then Runtime → Restart session):\n"
            "  !pip install -q torch==2.3.1 torchvision==0.18.1 "
            "torchaudio==2.3.1 torchtext==0.18.0 "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "Do NOT set SHESHA_SKIP_BOOTSTRAP until torch imports."
        ) from e

    if _torch_stack_ok():
        import torch as _torch
        print(f"torch={_torch.__version__}  torchtext OK", flush=True)
        return

    raise SystemExit(
        "torch/torchtext still not importable after pip.\n"
        "Runtime → Restart session, then re-run WITHOUT "
        "SHESHA_SKIP_BOOTSTRAP (or unset it).\n"
        "Quick check after restart:\n"
        "  import torch, torchtext; print(torch.__version__)"
    )


_ensure_torchtext()

import numpy as np
import pandas as pd
import torch
import random
import warnings
import re
from pathlib import Path
from scipy.stats import spearmanr
from anndata import AnnData
import scanpy as sc
# NOTE: do NOT `import pertpy` — pertpy.tools breaks with newer scanpy.
# All data loading goes through pipeline_core (lazy pertpy.data._datasets).
from shesha.bio import compute_stability, compute_magnitude

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

import pipeline_config as cfg
from pipeline_core import (
    _EXCLUDED_PERT_LABELS,
    _extract_adata,
    assert_frozen_sp_compatible,
    ensure_in_memory,
    ensure_local_h5ad,
    load_raw,
    materialize_min_cells,
    setup_cache,
)

# Papalexi manuscript scGPT numbers must come from the canonical MuData file
# (papalexi_2021.h5mu via scverse mirror / figshare / --h5ad), loaded with
# pull_obs-only (no pull_var), not via pertpy's loader. GEO rebuild ρ=1.000 is
# a different verification and does not lock this embed.
PAPALEXI_NAME = "Papalexi 2021 (CRISPR-KO)"
PAPALEXI_LOAD_SOURCE = "locked_h5mu"
PAPALEXI_LOAD_SOURCE_ALIASES = frozenset(
    {"locked_h5mu", "figshare_h5mu", "scverse_h5mu"}
)
# Bit-identical 2× FAILED under locked h5mu + same gene_digest (GPU forward-pass
# noise). Observed Sp~mag 0.383 / 0.434 / 0.444; vs-frozen Sp 0.575 / 0.633 /
# 0.683 / 0.630; mag ρ bit-identical at 0.7539130434782607 across runs.
# Manuscript policy: EXCLUDE Papalexi from the scGPT concordance column
# (rank-fragile at n=24). Not a load/preprocess bug. GEO ρ=1.0 is separate.
SCGPT_PAPALEXI_POLICY = "exclude"
from revision_io import find_sp_csv, load_sp_table

SEED = cfg.SEED
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # overridden by configure_torch_determinism
else:
    torch.backends.cudnn.deterministic = True


def configure_torch_determinism(enabled: bool) -> dict:
    """
    Attempt GPU-deterministic inference. CUBLAS_WORKSPACE_CONFIG should already
    be set at process start when --deterministic is in argv. Some ops have no
    deterministic kernel and will raise at embed time.
    Returns an audit dict that is printed and written into the gene_digest sidecar.
    """
    audit = {
        "requested": bool(enabled),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "are_deterministic_algorithms_enabled": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "applied": False,
        "note": "",
    }
    if not enabled:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        audit["note"] = "OFF — pass --deterministic to enable"
        audit["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        audit["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            audit["are_deterministic_algorithms_enabled"] = bool(
                torch.are_deterministic_algorithms_enabled()
            )
        print(f"DETERMINISM AUDIT: {audit}", flush=True)
        return audit

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    audit["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
        audit["applied"] = True
        audit["note"] = "ON"
    except Exception as e:
        audit["note"] = f"partial: {e}"
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        audit["are_deterministic_algorithms_enabled"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
    audit["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    audit["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    # Fail loud if the flag was requested but did not stick (prior revision pattern).
    if not audit.get("are_deterministic_algorithms_enabled"):
        print(
            "WARNING: --deterministic requested but "
            "torch.are_deterministic_algorithms_enabled() is False",
            flush=True,
        )
    if audit.get("cudnn_benchmark"):
        print(
            "WARNING: --deterministic requested but cudnn.benchmark is still True",
            flush=True,
        )
    print(f"DETERMINISM AUDIT: {audit}", flush=True)
    return audit

OUTPUT_DIR = cfg.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADAMSON_UPR_NAME = "Adamson 2016 UPR (CRISPRi)"
ADAMSON_PILOT_NAME = "Adamson 2016 pilot (CRISPRi)"

REPLOGLE_MIN_CELLS = cfg.MIN_CELLS
MIN_CELLS_PER_PERT = cfg.MIN_CELLS
# Same floor as before the pilot exception — do not relax after seeing exclusions.
# Adamson pilot (n=8) stays out of the Sp~mag redundancy column (Approach A
# already skips n<15; QC descriptive at n<30). Reverting n≥5 (2026-08-06).
MIN_CORR_N = 10

setup_cache()
sc.settings.datasetdir = cfg.CACHE_DIR


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x, ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / sd


def add_within_dataset_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score Sp and magnitude within each dataset (for pooled plots)."""
    out = df.copy()
    if "dataset" not in out.columns:
        out["stability_z"] = _zscore(out["stability"].to_numpy())
        out["magnitude_z"] = _zscore(out["magnitude"].to_numpy())
        return out
    parts = []
    for _, g in out.groupby("dataset", sort=False):
        g = g.copy()
        g["stability_z"] = _zscore(g["stability"].to_numpy())
        g["magnitude_z"] = _zscore(g["magnitude"].to_numpy())
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def within_scgpt_sp_mag_corr(df: pd.DataFrame, dataset_name: str) -> dict | None:
    if len(df) < MIN_CORR_N or "stability" not in df.columns:
        return None
    ci = bootstrap_spearman_ci(df["magnitude"], df["stability"], seed=SEED)
    # Spearman is rank-invariant to z-score; report both for plot parity.
    z = add_within_dataset_zscores(df.assign(dataset=dataset_name))
    ci_z = bootstrap_spearman_ci(z["magnitude_z"], z["stability_z"], seed=SEED)
    return {
        "dataset": dataset_name,
        "metric": "sp_vs_magnitude_within_scgpt",
        **ci,
        "rho_zscored": float(ci_z["rho"]),
        "ci_low_zscored": float(ci_z["ci_low"]),
        "ci_high_zscored": float(ci_z["ci_high"]),
        "thin_n": bool(len(df) < 30),
    }

# All 6 frozen dataset keys (same as pipeline_config.DATASETS)
DATASETS = {name: {} for name in cfg.DATASETS}

# Short CLI aliases → frozen display names
DATASET_ALIASES = {
    "norman": "Norman 2019 (CRISPRa)",
    "adamson": ADAMSON_UPR_NAME,
    "adamson_upr": ADAMSON_UPR_NAME,
    "adamson_pilot": ADAMSON_PILOT_NAME,
    "dixit": "Dixit 2016 (CRISPR-KO)",
    "papalexi": "Papalexi 2021 (CRISPR-KO)",
    "replogle": "Replogle 2022 (CRISPRi)",
}


def resolve_cli_dataset(name: str) -> str:
    key = name.strip()
    if key.lower() in DATASET_ALIASES:
        return DATASET_ALIASES[key.lower()]
    return cfg.resolve_dataset_name(key)


def dataset_csv_path(out_dir: Path, dataset_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_name)
    return out_dir / f"scgpt_{safe}.csv"


# =============================================================================
# DATASET LOADING (via pipeline_core — no top-level pertpy import)
# =============================================================================

def _papalexi_sidecar_locked(meta_path: Path) -> bool:
    """True iff prior Papalexi CSV was scored from locked .h5mu (not pertpy)."""
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    return meta.get("load_source") in PAPALEXI_LOAD_SOURCE_ALIASES


def load_dataset_raw(
    dataset_name,
    *,
    downsample: bool = False,
    max_cells_per_pert: int = cfg.MAX_CELLS_PER_PERT,
    max_control_cells: int = cfg.MAX_CONTROL_CELLS,
    h5ad_path: Path | str | None = None,
):
    """
    Load raw AnnData. Downsampling is off by default (full cell set).

    Returns (adata, pert_col, ctrl, load_meta). load_meta records provenance
    for the gene_digest sidecar (required for Papalexi manuscript lock).
    """
    print(f"\n>>> Loading {dataset_name}...")
    load_meta = {
        "dataset": dataset_name,
        "load_source": "default",
        "source_path": None,
        "source_bytes": None,
        "source_name": None,
    }
    try:
        # Papalexi: refuse pertpy fallback. The open item was embed provenance
        # after figshare failed once; GEO ρ=1.0 does not close this.
        if dataset_name == PAPALEXI_NAME:
            if h5ad_path is not None:
                path = Path(h5ad_path)
                if not path.exists():
                    raise FileNotFoundError(f"--h5ad not found: {path}")
            else:
                path = ensure_local_h5ad(dataset_name)
            raw = load_raw(dataset_name, prefer_local=True, h5ad_path=path)
            load_meta.update(
                {
                    "load_source": PAPALEXI_LOAD_SOURCE,
                    "source_path": str(path.resolve()),
                    "source_bytes": int(path.stat().st_size),
                    "source_name": path.name,
                }
            )
            print(
                f"    LOCKED Papalexi load: {path.name} "
                f"({load_meta['source_bytes'] / 1e6:.1f} MB; "
                f"source={PAPALEXI_LOAD_SOURCE}; no pertpy)",
                flush=True,
            )
        elif h5ad_path is not None:
            path = Path(h5ad_path)
            raw = load_raw(dataset_name, prefer_local=True, h5ad_path=path)
            load_meta.update(
                {
                    "load_source": "cli_h5ad",
                    "source_path": str(path.resolve()),
                    "source_bytes": int(path.stat().st_size),
                    "source_name": path.name,
                }
            )
        else:
            raw = load_raw(dataset_name, prefer_local=True)
            load_meta["load_source"] = "prefer_local_or_pertpy"

        adata, pert_col, ctrl = _extract_adata(raw, dataset_name, sc)
        n_before = int(adata.n_obs)
        if downsample:
            adata, _valid, _counts = materialize_min_cells(
                adata,
                pert_col,
                ctrl,
                min_cells=cfg.MIN_CELLS,
                max_cells_per_pert=max_cells_per_pert,
                max_control_cells=max_control_cells,
                seed=cfg.SEED,
            )
        if getattr(adata, "isbacked", False) or getattr(adata, "filename", None):
            print("    to_memory()…", flush=True)
            adata = ensure_in_memory(adata)
    except Exception as e:
        print(f"    ! Load failed: {e}")
        return None, None, None, load_meta

    if downsample:
        print(
            f"    pert_col={pert_col!r}, ctrl={ctrl!r}, "
            f"n_obs={adata.n_obs} (was {n_before} before downsample), "
            f"n_vars={adata.n_vars}, "
            f"caps ≤{max_cells_per_pert}/pert ≤{max_control_cells} ctrl"
        )
    else:
        print(
            f"    pert_col={pert_col!r}, ctrl={ctrl!r}, "
            f"n_obs={adata.n_obs}, n_vars={adata.n_vars} (no downsample)"
        )
    return adata, pert_col, ctrl, load_meta


# =============================================================================
# scGPT EMBEDDING
# =============================================================================

def _gene_list_digest(adata) -> tuple[str, int]:
    """Stable digest of var_names fed to scGPT (catch MuData/join drift)."""
    import hashlib

    genes = [str(g) for g in adata.var_names]
    h = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()[:16]
    return h, len(genes)


def embed_with_scgpt(adata, model_dir, device, batch_size: int = 64):
    """
    Prepare raw counts and generate scGPT cell embeddings.
    Returns a new AnnData whose .X contains the embeddings.
    """
    try:
        from scgpt.tasks import embed_data
    except Exception as e:
        raise RuntimeError(
            "Failed to import scgpt.tasks.embed_data (often a torch/torchtext mismatch). "
            "See _ensure_torchtext() install instructions."
        ) from e

    if str(device) == "cpu":
        print(
            "    WARNING: embedding on CPU — expect ~30–60s/batch. "
            "Use a GPU runtime (Runtime → Change runtime type → T4/A100).",
            flush=True,
        )

    # Use raw counts if available
    if "counts" in adata.layers:
        adata_raw = adata.copy()
        adata_raw.X = adata_raw.layers["counts"].copy()
    else:
        print("    WARNING: no 'counts' layer found -- using .X as-is (should be raw counts)")
        adata_raw = adata.copy()

    gene_digest, n_genes = _gene_list_digest(adata_raw)
    n_batches = int(np.ceil(adata_raw.n_obs / max(batch_size, 1)))
    print(
        f"    scGPT embed: n_cells={adata_raw.n_obs}, n_genes={n_genes}, "
        f"gene_digest={gene_digest}, batch_size={batch_size}, "
        f"~{n_batches} batches, device={device}",
        flush=True,
    )

    embedded = embed_data(
        adata_raw,
        model_dir=model_dir,
        gene_col="index",
        batch_size=batch_size,
        device=device,
        use_fast_transformer=False,
    )
    # stash for callers / CSV sidecars
    embedded.uns["scgpt_gene_digest"] = gene_digest
    embedded.uns["scgpt_n_genes"] = n_genes
    return embedded


# =============================================================================
# GEOMETRIC STABILITY METRICS (via shesha)
# =============================================================================

def compute_metrics_from_embeddings(embedded_adata, obs_source, pert_col, ctrl_label):
    """
    Given an AnnData whose .X are scGPT embeddings, compute per-perturbation
    stability and magnitude using shesha.bio.

    Returns a DataFrame with columns:
        perturbation, stability, magnitude, n_cells
    """
    proxy = AnnData(X=embedded_adata.X, obs=obs_source.obs[[pert_col]].copy())

    stability_scores = compute_stability(
        proxy,
        perturbation_key=pert_col,
        control_label=ctrl_label,
        metric='cosine',
    )

    magnitude_scores = compute_magnitude(
        proxy,
        perturbation_key=pert_col,
        control_label=ctrl_label,
        metric='euclidean',
    )

    counts = proxy.obs[pert_col].value_counts()

    results = []
    ctrl = str(ctrl_label)
    for pert in stability_scores:
        p = str(pert)
        # Match frozen pipeline: never score control or excluded labels.
        # Adamson UPR n=88 vs frozen 87 was scgpt_only=['DROP'] — collapsed
        # unparseable/guide-junk labels marked DROP in pipeline_core and
        # dropped in materialize_min_cells via _EXCLUDED_PERT_LABELS.
        if p == ctrl or p.lower() == ctrl.lower():
            continue
        if p in _EXCLUDED_PERT_LABELS:
            continue
        n_cells = int(counts.get(pert, 0))
        if n_cells < MIN_CELLS_PER_PERT:
            continue
        results.append({
            'perturbation': p,
            'stability': stability_scores[pert],
            'magnitude': magnitude_scores.get(pert, np.nan),
            'n_cells': n_cells,
        })

    return pd.DataFrame(results)


def compare_to_frozen(
    df: pd.DataFrame,
    dataset_name: str,
    frozen_df: pd.DataFrame,
) -> dict:
    """Manuscript concordance: scGPT Sp/mag vs frozen PCA Sp/mag."""
    fr = frozen_df[frozen_df["dataset"] == dataset_name][
        ["perturbation", "stability", "magnitude"]
    ].rename(
        columns={
            "stability": "sp_frozen_pca",
            "magnitude": "magnitude_frozen_pca",
        }
    )
    m = df.merge(fr, on="perturbation", how="inner")
    n_scgpt = int(len(df))
    n_frozen = int(len(fr))
    out = {
        "dataset": dataset_name,
        "n_scgpt": n_scgpt,
        "n_frozen": n_frozen,
        "n_shared": int(len(m)),
        "n_mismatch": n_scgpt != n_frozen,
        "concordance_vs": "frozen_sp_scores.csv (PCA Sp)",
    }
    if n_scgpt != n_frozen:
        scgpt_only = sorted(set(df["perturbation"]) - set(fr["perturbation"]))
        frozen_only = sorted(set(fr["perturbation"]) - set(df["perturbation"]))
        out["scgpt_only_perts"] = scgpt_only[:20]
        out["frozen_only_perts"] = frozen_only[:20]
        print(
            f"    WARNING: scGPT n={n_scgpt} vs frozen n={n_frozen} "
            f"(shared={len(m)}). scgpt_only={scgpt_only[:5]}… "
            f"frozen_only={frozen_only[:5]}…",
            flush=True,
        )
    if len(m) >= 5:
        r_sp, p_sp = spearmanr(m["stability"], m["sp_frozen_pca"])
        r_mag, p_mag = spearmanr(m["magnitude"], m["magnitude_frozen_pca"])
        out["spearman_scgpt_vs_frozen_sp"] = float(r_sp)
        out["spearman_scgpt_vs_frozen_sp_p"] = float(p_sp)
        out["spearman_scgpt_vs_frozen_magnitude"] = float(r_mag)
        out["spearman_scgpt_vs_frozen_magnitude_p"] = float(p_mag)
        print(
            f"    vs frozen PCA Sp: ρ={r_sp:.4f} (n={len(m)})  "
            f"← manuscript concordance column",
            flush=True,
        )
        print(
            f"    vs frozen magnitude: ρ={r_mag:.4f} (n={len(m)})",
            flush=True,
        )
    else:
        print("    WARNING: <5 shared perts with frozen CSV; no concordance", flush=True)
    return out


# =============================================================================
# BOOTSTRAP SPEARMAN CI
# =============================================================================

def bootstrap_spearman_ci(x, y, n_bootstrap=10000, ci_level=0.95, seed=320):
    x, y = np.asarray(x), np.asarray(y)
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
        'ci_low':  float(np.percentile(valid, 100 * alpha / 2)),
        'ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))),
        'n': len(x),
    }


# =============================================================================
# MAIN
# =============================================================================

def _per_dataset_scgpt_paths(out_dir: Path) -> list[Path]:
    skip = {
        "scgpt_all_datasets.csv",
        "scgpt_all_datasets_zscored.csv",
        "scgpt_correlations.csv",
        "scgpt_vs_frozen_concordance.csv",
    }
    return sorted(
        p
        for p in out_dir.glob("scgpt_*.csv")
        if p.name not in skip and ".stale." not in p.name
    )


def _dataset_label_for_csv(path: Path, df: pd.DataFrame) -> str:
    if "dataset" in df.columns and len(df):
        return str(df["dataset"].iloc[0])
    return path.stem


def _merge_preference_key(path: Path, dataset_name: str) -> tuple:
    """
    Higher is better when resolving duplicate CSVs for one dataset.
    Prefer locked Papalexi sidecar; then canonical filename from dataset_csv_path.
    """
    meta = path.with_suffix(".gene_digest.json")
    locked = 1 if (
        dataset_name == PAPALEXI_NAME and _papalexi_sidecar_locked(meta)
    ) else 0
    canonical = 1 if path == dataset_csv_path(path.parent, dataset_name) else 0
    return (locked, canonical, path.stat().st_mtime)


def resolve_per_dataset_csvs(out_dir: Path) -> list[Path]:
    """
    One CSV per dataset for merge. Quarantines losers (rename *.stale.csv) so
    a later glob cannot silently reintroduce pertpy-era Papalexi rows.
    """
    out_dir = Path(out_dir)
    by_ds: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for p in _per_dataset_scgpt_paths(out_dir):
        df = pd.read_csv(p)
        if df.empty or "magnitude" not in df.columns:
            continue
        ds = _dataset_label_for_csv(p, df)
        by_ds.setdefault(ds, []).append((p, df))

    chosen: list[Path] = []
    for ds, items in sorted(by_ds.items()):
        if len(items) == 1:
            chosen.append(items[0][0])
            continue
        ranked = sorted(
            items,
            key=lambda it: _merge_preference_key(it[0], ds),
            reverse=True,
        )
        winner_path, _ = ranked[0]
        print(
            f"WARNING: {len(items)} CSVs for {ds!r}; keeping {winner_path.name}",
            flush=True,
        )
        for loser_path, _ in ranked[1:]:
            stale = loser_path.with_name(
                loser_path.stem + ".stale" + loser_path.suffix
            )
            if stale.exists():
                stale.unlink()
            loser_path.rename(stale)
            meta = loser_path.with_suffix(".gene_digest.json")
            if meta.exists():
                meta.rename(stale.with_suffix(".gene_digest.json"))
            print(f"  quarantined -> {stale.name}", flush=True)
        if ds == PAPALEXI_NAME and not _papalexi_sidecar_locked(
            winner_path.with_suffix(".gene_digest.json")
        ):
            print(
                "  WARNING: winning Papalexi CSV is not load_source=locked_h5mu — "
                "do not cite; re-embed with locked path.",
                flush=True,
            )
        chosen.append(winner_path)
    return chosen


def csvs_bit_identical(path_a: Path, path_b: Path) -> dict:
    """Close criterion for Papalexi lock: two locked embeds must match exactly."""
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    cols = ["perturbation", "stability", "magnitude", "n_cells"]
    for c in cols:
        if c not in a.columns or c not in b.columns:
            return {"identical": False, "reason": f"missing column {c}"}
    a = a[cols].sort_values("perturbation").reset_index(drop=True)
    b = b[cols].sort_values("perturbation").reset_index(drop=True)
    if len(a) != len(b):
        return {"identical": False, "reason": f"nrow {len(a)} != {len(b)}"}
    same = a.equals(b)
    sp_max = float(np.nanmax(np.abs(a["stability"] - b["stability"])))
    mag_max = float(np.nanmax(np.abs(a["magnitude"] - b["magnitude"])))
    sp_rho = float(spearmanr(a["stability"], b["stability"])[0]) if len(a) >= 3 else float("nan")
    return {
        "identical": bool(same),
        "n": int(len(a)),
        "path_a": str(path_a),
        "path_b": str(path_b),
        "max_abs_sp_diff": sp_max,
        "max_abs_mag_diff": mag_max,
        "spearman_sp_run1_vs_run2": sp_rho,
        "reason": "bit-identical on perturbation/stability/magnitude/n_cells"
        if same
        else (
            "values differ — at small n Sp ranks are fragile to GPU float noise "
            f"(max|ΔSp|={sp_max:.3g}, max|Δmag|={mag_max:.3g})"
        ),
    }


def merge_existing_csvs(out_dir: Path):
    """Rebuild combined + correlation CSVs — one file per dataset after quarantine."""
    out_dir = Path(out_dir)
    paths = resolve_per_dataset_csvs(out_dir)
    if not paths:
        return None, pd.DataFrame()

    all_dfs = []
    corr_results = []
    for p in paths:
        df = pd.read_csv(p)
        all_dfs.append(df)
        ds = _dataset_label_for_csv(p, df)
        row = within_scgpt_sp_mag_corr(df, ds)
        if row is not None:
            corr_results.append(row)

    combined = add_within_dataset_zscores(pd.concat(all_dfs, ignore_index=True))
    n_ds = int(combined["dataset"].nunique()) if "dataset" in combined.columns else 0
    if n_ds != len(paths):
        raise RuntimeError(
            f"Merge produced {n_ds} datasets from {len(paths)} CSVs — "
            "duplicate dataset labels remain; inspect out_dir scgpt_*.csv"
        )
    combined.to_csv(out_dir / "scgpt_all_datasets.csv", index=False)
    combined.to_csv(out_dir / "scgpt_all_datasets_zscored.csv", index=False)
    corr_df = pd.DataFrame(corr_results)
    if not corr_df.empty:
        # One Sp~mag row per dataset (duplicates were a silent citation hazard).
        corr_df = corr_df.drop_duplicates(subset=["dataset"], keep="first")
        corr_df.to_csv(out_dir / "scgpt_correlations.csv", index=False)
        print("\n=== scGPT within-embedding Sp~mag (n≥10; pilot excluded) ===")
        for _, row in corr_df.iterrows():
            print(
                f"  {row['dataset']}: rho={row['rho']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                f"p={row['p']:.2e}, n={int(row['n'])}"
            )
    print(
        f"Merged {len(paths)} dataset CSV(s) ({n_ds} unique datasets) "
        "-> scgpt_all_datasets.csv "
        "(+ stability_z / magnitude_z; also scgpt_all_datasets_zscored.csv)"
    )
    return combined, corr_df


def run_all(
    model_dir: str,
    out_dir=None,
    *,
    batch_size: int = 64,
    skip_existing: bool = True,
    downsample: bool = False,
    max_cells_per_pert: int = cfg.MAX_CELLS_PER_PERT,
    max_control_cells: int = cfg.MAX_CONTROL_CELLS,
    compare_frozen: bool = True,
    frozen_df=None,
    h5ad_path: Path | str | None = None,
    deterministic: bool = False,
    exclude_from_concordance: frozenset | set | None = None,
    snapshot_path: Path | str | None = None,
):
    out_dir = Path(out_dir) if out_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_from_concordance = frozenset(exclude_from_concordance or ())

    det_audit = configure_torch_determinism(deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output dir: {out_dir}")
    print(f"config_version={cfg.CONFIG_VERSION}  MIN_CELLS={MIN_CELLS_PER_PERT}")
    print(f"batch_size={batch_size}  skip_existing={skip_existing}")
    print(f"papalexi_policy={SCGPT_PAPALEXI_POLICY}")
    if exclude_from_concordance:
        print(f"exclude_from_concordance={sorted(exclude_from_concordance)}")
    if snapshot_path:
        print(f"snapshot_after_embed -> {snapshot_path}")
    print(
        "NOTE: Sp~mag below is within-scGPT only. Manuscript embedding "
        "concordance = scGPT Sp vs frozen PCA Sp (--compare-frozen).",
        flush=True,
    )
    if downsample:
        print(
            f"downsample: ON ≤{max_cells_per_pert}/pert ≤{max_control_cells} ctrl"
        )
    else:
        print("downsample: OFF (full cell set; still excludes control label)")
    print(f"datasets: {list(DATASETS)}")

    all_dfs = []
    corr_results = []
    frozen_rows = []

    for dataset_name in DATASETS:
        csv_path = dataset_csv_path(out_dir, dataset_name)
        meta_path = csv_path.with_suffix(".gene_digest.json")
        if skip_existing and csv_path.exists():
            # Stale Papalexi CSVs from pertpy-fallback embeds are not skippable.
            if dataset_name == PAPALEXI_NAME and not _papalexi_sidecar_locked(meta_path):
                print(
                    f"\n>>> {dataset_name}: existing {csv_path.name} lacks "
                    f"load_source={PAPALEXI_LOAD_SOURCE} lock — re-embedding "
                    "(prior vs-frozen ~0.691 / Sp~mag ~0.530 are not final).",
                    flush=True,
                )
            else:
                print(f"\n>>> Skipping {dataset_name} (exists: {csv_path.name})")
                df = pd.read_csv(csv_path)
                if not df.empty:
                    all_dfs.append(df)
                    row = within_scgpt_sp_mag_corr(df, dataset_name)
                    if row is not None:
                        thin = "  [thin n]" if row.get("thin_n") else ""
                        print(
                            f"    within-scGPT Sp~mag ρ={row['rho']:.3f} "
                            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                            f"n={row['n']}{thin}"
                        )
                        corr_results.append(row)
                    if (
                        compare_frozen
                        and frozen_df is not None
                        and dataset_name not in exclude_from_concordance
                    ):
                        frozen_rows.append(
                            compare_to_frozen(df, dataset_name, frozen_df)
                        )
                    elif dataset_name in exclude_from_concordance:
                        print(
                            f"    concordance: EXCLUDED from manuscript scGPT "
                            f"column ({dataset_name})",
                            flush=True,
                        )
                continue

        adata, pert_col, ctrl_label, load_meta = load_dataset_raw(
            dataset_name,
            downsample=downsample,
            max_cells_per_pert=max_cells_per_pert,
            max_control_cells=max_control_cells,
            h5ad_path=h5ad_path,
        )
        if adata is None:
            continue

        print(f"    Embedding with scGPT...")
        try:
            embedded = embed_with_scgpt(
                adata, model_dir, device, batch_size=batch_size
            )
        except RuntimeError as e:
            msg = str(e)
            if deterministic and "deterministic" in msg.lower():
                print(
                    f"    ! Deterministic kernels unavailable: {e}\n"
                    "      Fall back: exclude Papalexi from manuscript scGPT "
                    "column (rank-fragile at n=24 under GPU inference).",
                    flush=True,
                )
            print(f"    ! Embedding failed: {e}")
            continue
        except Exception as e:
            print(f"    ! Embedding failed: {e}")
            continue

        print(f"    Computing stability & magnitude...")
        df = compute_metrics_from_embeddings(embedded, adata, pert_col, ctrl_label)
        if df.empty:
            print(f"    ! No results for {dataset_name}")
            continue

        df["dataset"] = dataset_name
        df = add_within_dataset_zscores(df)
        df.to_csv(csv_path, index=False)
        sidecar = {
            "dataset": dataset_name,
            "gene_digest": embedded.uns.get("scgpt_gene_digest"),
            "n_genes": embedded.uns.get("scgpt_n_genes"),
            "n_perturbations": int(len(df)),
            "config_version": cfg.CONFIG_VERSION,
            "load_source": load_meta.get("load_source"),
            "source_path": load_meta.get("source_path"),
            "source_bytes": load_meta.get("source_bytes"),
            "source_name": load_meta.get("source_name"),
            "torch_deterministic": bool(deterministic),
            "determinism_audit": det_audit,
        }
        with open(meta_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        print(
            f"    Saved {len(df)} perturbations -> {csv_path} "
            f"(gene_digest={embedded.uns.get('scgpt_gene_digest')}, "
            f"load_source={sidecar['load_source']})"
        )
        if snapshot_path is not None:
            snap = Path(snapshot_path)
            snap.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(snap, index=False)
            print(f"    snapshot -> {snap}", flush=True)

        all_dfs.append(df)

        row = within_scgpt_sp_mag_corr(df, dataset_name)
        if row is not None:
            thin = "  [thin n]" if row.get("thin_n") else ""
            print(
                f"    within-scGPT Sp~mag ρ={row['rho']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                f"p={row['p']:.2e}, n={row['n']}{thin}  (diagnostic only)"
            )
            corr_results.append(row)

        if (
            compare_frozen
            and frozen_df is not None
            and dataset_name not in exclude_from_concordance
        ):
            frozen_rows.append(compare_to_frozen(df, dataset_name, frozen_df))
        elif dataset_name in exclude_from_concordance:
            print(
                f"    concordance: EXCLUDED from manuscript scGPT "
                f"column ({dataset_name})",
                flush=True,
            )

        # free GPU memory between datasets
        del adata, embedded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Always rebuild combined / Sp~mag tables from *all* per-dataset CSVs on
    # disk — a single-dataset re-run must not clobber the six-dataset merge
    # (Papalexi-only run previously wrote n=24 over n=2285).
    combined, corr_df = merge_existing_csvs(out_dir)

    if frozen_rows:
        fr_path = out_dir / "scgpt_vs_frozen_concordance.csv"
        # Merge with any existing concordance rows for other datasets
        prev = pd.read_csv(fr_path) if fr_path.exists() else pd.DataFrame()
        fr_df = pd.DataFrame(frozen_rows)
        if not prev.empty and "dataset" in prev.columns:
            keep = prev[~prev["dataset"].isin(fr_df["dataset"])]
            fr_df = pd.concat([keep, fr_df], ignore_index=True)
        fr_df.to_csv(fr_path, index=False)
        print("\n=== scGPT vs frozen PCA Sp (manuscript column; this run) ===")
        for _, row in fr_df.iterrows():
            if row["dataset"] not in {r["dataset"] for r in frozen_rows}:
                continue
            rho = row.get("spearman_scgpt_vs_frozen_sp")
            rmag = row.get("spearman_scgpt_vs_frozen_magnitude")
            flag = " ⚠ n mismatch" if row.get("n_mismatch") else ""
            print(
                f"  {row['dataset']}: Sp ρ={rho}  mag ρ={rmag}  "
                f"n_scgpt={row.get('n_scgpt')} n_frozen={row.get('n_frozen')} "
                f"shared={row.get('n_shared')}{flag}"
            )
        print(f"\nSaved -> {fr_path.name} ({len(fr_df)} dataset rows)")

    return combined, corr_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Colab — one dataset at a time):
      --out-dir /content/shesha-crispr --datasets norman --batch-size 256

      --out-dir /content/shesha-crispr --datasets dixit --batch-size 256

  # After all done, merge only:
      --out-dir /content/shesha-crispr --merge-only

Aliases: norman, dixit, papalexi, replogle, adamson, adamson_pilot

Papalexi lock (closes embed-provenance item; GEO ρ=1.0 is separate):
  # Figshare is often WAF-blocked; download uses scverse mirror of same .h5mu.
  rm -f $OUT/scgpt_Papalexi_2021__CRISPR_KO_.csv \\
        $OUT/scgpt_Papalexi_2021__CRISPR_KO_.gene_digest.json
      --out-dir $OUT --datasets papalexi --batch-size 256 \\
      --compare-frozen --no-skip-existing
  # Or pass a pre-downloaded file (~589 MB):
  #   wget -O /content/papalexi_2021.h5mu \\
  #     https://exampledata.scverse.org/pertpy/papalexi_2021.h5mu
  #   python scgpt_analysis.py ... --datasets papalexi --h5ad /content/papalexi_2021.h5mu
  # Papalexi scGPT column: EXCLUDED (Sp non-reproducible at n=24). Close the
  # finding with Replogle 2× (must see DETERMINISM AUDIT applied=True):
  #   python scgpt_analysis.py --model-dir ... --out-dir $OUT --datasets replogle \\
  #     --batch-size 256 --no-skip-existing --deterministic \\
  #     --snapshot /tmp/replogle_scgpt_run1.csv
  #   python scgpt_analysis.py ... --datasets replogle --no-skip-existing \\
  #     --deterministic
  #   python scgpt_analysis.py --check-identical \\
  #     /tmp/replogle_scgpt_run1.csv $OUT/scgpt_Replogle_2022__CRISPRi_.csv
""",
    )
    parser.add_argument(
        "--h5ad",
        type=str,
        default=None,
        help=(
            "Explicit path to dataset file (Papalexi: papalexi_2021.h5mu). "
            "Skips download when set."
        ),
    )
    parser.add_argument(
        "--check-identical",
        nargs=2,
        metavar=("CSV_A", "CSV_B"),
        default=None,
        help=(
            "Exit 0 iff two per-dataset scGPT CSVs match on "
            "perturbation/stability/magnitude/n_cells (Papalexi lock close check)."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to pretrained scGPT model directory (required unless --merge-only)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write CSVs (default: pipeline_config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="One or more names/aliases (default: all 6). Prefer one at a time.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="scGPT embed batch size (default: 256 on GPU, 32 on CPU)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even if scgpt_<dataset>.csv already exists",
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Optional: apply frozen PCA cell caps before embed (off by default)",
    )
    parser.add_argument(
        "--max-cells-per-pert",
        type=int,
        default=cfg.MAX_CELLS_PER_PERT,
        help=f"With --downsample: max cells/pert (default: {cfg.MAX_CELLS_PER_PERT})",
    )
    parser.add_argument(
        "--max-control-cells",
        type=int,
        default=cfg.MAX_CONTROL_CELLS,
        help=f"With --downsample: max controls (default: {cfg.MAX_CONTROL_CELLS})",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only rebuild scgpt_all_datasets.csv from existing per-dataset CSVs",
    )
    parser.add_argument(
        "--compare-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concordance vs frozen_sp_scores.csv (default: on; manuscript column)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Force torch deterministic algorithms + cudnn.deterministic. "
            "Sets CUBLAS_WORKSPACE_CONFIG before torch import when present in argv. "
            "Log must show DETERMINISM AUDIT applied=True."
        ),
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help=(
            "After embed, also write this CSV path (use as run1 for --check-identical). "
            "Requires a single --datasets entry."
        ),
    )
    parser.add_argument(
        "--exclude-from-concordance",
        nargs="*",
        default=None,
        help=(
            "Dataset aliases to omit from scgpt_vs_frozen_concordance.csv "
            "(e.g. papalexi). Still embeds / writes per-dataset CSV."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check_identical:
        report = csvs_bit_identical(Path(args.check_identical[0]), Path(args.check_identical[1]))
        print(report)
        raise SystemExit(0 if report.get("identical") else 1)

    if args.merge_only:
        merge_existing_csvs(out_dir)
        exclude = set()
        if args.exclude_from_concordance:
            exclude = {resolve_cli_dataset(d) for d in args.exclude_from_concordance}
        if SCGPT_PAPALEXI_POLICY == "exclude":
            exclude.add(PAPALEXI_NAME)
        if exclude and args.compare_frozen:
            frozen_path = find_sp_csv(out_dir)
            assert_frozen_sp_compatible(frozen_path)
            frozen_df = load_sp_table(frozen_path)
            rows = []
            for p in resolve_per_dataset_csvs(out_dir):
                df = pd.read_csv(p)
                ds = _dataset_label_for_csv(p, df)
                if ds in exclude:
                    print(f"concordance merge: excluding {ds}", flush=True)
                    continue
                rows.append(compare_to_frozen(df, ds, frozen_df))
            fr = pd.DataFrame(rows)
            fr_path = out_dir / "scgpt_vs_frozen_concordance.csv"
            fr.to_csv(fr_path, index=False)
            print(
                f"Rewrote {fr_path.name} with {len(fr)} dataset rows "
                f"(excluded {sorted(exclude)}; papalexi_policy={SCGPT_PAPALEXI_POLICY})"
            )
        raise SystemExit(0)

    if not args.model_dir:
        raise SystemExit(
            "--model-dir is required unless --merge-only or --check-identical"
        )

    if args.datasets:
        wanted = {resolve_cli_dataset(d) for d in args.datasets}
        missing = wanted - set(DATASETS)
        if missing:
            raise SystemExit(
                f"Unknown datasets: {sorted(missing)}. "
                f"Have: {list(DATASETS)} or aliases: {list(DATASET_ALIASES)}"
            )
        keep = {k: v for k, v in DATASETS.items() if k in wanted}
        DATASETS.clear()
        DATASETS.update(keep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 256 if device.type == "cuda" else 32

    frozen_df = None
    if args.compare_frozen:
        frozen_path = find_sp_csv(out_dir)
        assert_frozen_sp_compatible(frozen_path)
        frozen_df = load_sp_table(frozen_path)

    print(f"Running scGPT on {len(DATASETS)} datasets: {list(DATASETS)}")
    print(f"model_dir={args.model_dir}")
    print(f"out_dir={out_dir}")
    if args.h5ad and len(DATASETS) != 1:
        raise SystemExit("--h5ad requires exactly one --datasets entry")

    exclude = set()
    if args.exclude_from_concordance:
        exclude = {resolve_cli_dataset(d) for d in args.exclude_from_concordance}
    if SCGPT_PAPALEXI_POLICY == "exclude":
        exclude.add(PAPALEXI_NAME)

    if args.snapshot and len(DATASETS) != 1:
        raise SystemExit("--snapshot requires exactly one --datasets entry")

    run_all(
        model_dir=args.model_dir,
        out_dir=out_dir,
        batch_size=batch_size,
        skip_existing=not args.no_skip_existing,
        downsample=args.downsample,
        max_cells_per_pert=args.max_cells_per_pert,
        max_control_cells=args.max_control_cells,
        compare_frozen=args.compare_frozen,
        frozen_df=frozen_df,
        h5ad_path=args.h5ad,
        deterministic=args.deterministic,
        exclude_from_concordance=exclude,
        snapshot_path=args.snapshot,
    )
