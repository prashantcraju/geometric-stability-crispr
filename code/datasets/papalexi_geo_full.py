#!/usr/bin/env python3
"""
Papalexi 2021 from GEO — independent of pertpy.

The ECCITE-seq screen is deposited at GEO (GSE153056) with the full
~111-guide library. The frozen pipeline's pertpy / scPerturb MuData scores
gene_target labels (~24 genes after MIN_CELLS). This script rebuilds the RNA
matrix + author assignments from GEO and scores Sp under the frozen pipeline at:

  gene   — parity check vs pertpy gene_target (expect ~24)
  guide  — full guide_ID library (up to ~111; ~83 pass MIN_CELLS=50)

Default --level both:
  1) gene-grain own HVG/PCA → manuscript/parity Sp (n=24 at MIN_CELLS=50)
  2) shared HVG/PCA on guide-materialized cells → guide Sp + paired contrast
     (shared-basis gene Sp is paired-only; often n=23 after MYC drops)

Usage:
  python papalexi_geo_full.py
  python papalexi_geo_full.py --compare-pertpy
  python papalexi_geo_full.py --build-only

Manuscript number: frozen MIN_CELLS=50 → ~24 genes / ~83 guides.
Reporting ladder: 112 deposited → 96 guides@10 (inventory) → 83@50 → 24 genes@50.
Item 16 predictive endpoint: gene-level Sp → cosine(guide_A mean shift,
guide_B mean shift) on the shared embedding (independent reagents; still
partly mechanical, a step down from split-half circularity). ICC(1) is
metric concordance across reagents, not that endpoint. Paired gene-vs-guide
median contrast is a size-confound footnote only (claim_allowed=false).
Papalexi is thin (n≈23 multi-guide genes); Replogle is the flagship check.

  python papalexi_geo_full.py --sensitivity --min-cells 10

Outputs (under shesha-crispr/ or --out-dir):
  papalexi_geo_sp_gene.csv              # manuscript gene-grain
  papalexi_geo_sp_gene_shared_basis.csv # paired-contrast only
  papalexi_geo_sp_guide.csv
  papalexi_geo_summary.json / methods_blurb.txt
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
import re
import tarfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, wilcoxon

import pipeline_config as cfg
from pipeline_core import (
    calculate_sp,
    materialize_min_cells,
    preprocess,
    score_perturbations,
    setup_cache,
)
from revision_io import download, find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import (
    bootstrap_partial_spearman_ci,
    bootstrap_spearman_ci,
    bootstrap_spearman_ci_clustered,
    partial_spearman_rank,
)

# ICC helpers live in stats_utils (revision); keep local fallbacks so a stale
# Colab upload of stats_utils.py does not hard-crash at import time.
try:
    from stats_utils import icc_gene_clustered_bootstrap, icc_oneway_unbalanced
except ImportError:  # pragma: no cover

    def icc_oneway_unbalanced(y, groups):
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups).astype(str)
        mask = np.isfinite(y)
        y, groups = y[mask], groups[mask]
        units, inv = np.unique(groups, return_inverse=True)
        k, N = len(units), len(y)
        if k < 2 or N <= k:
            return {"icc": np.nan, "n_groups": k, "n_obs": N, "n0": np.nan}
        n_i = np.bincount(inv)
        grand = float(y.mean())
        means = np.array([y[inv == i].mean() for i in range(k)])
        ssb = float(np.sum(n_i * (means - grand) ** 2))
        ssw = float(sum(np.sum((y[inv == i] - means[i]) ** 2) for i in range(k)))
        msb, msw = ssb / (k - 1), ssw / (N - k)
        n0 = (N - float(np.sum(n_i ** 2)) / N) / (k - 1)
        denom = msb + (n0 - 1) * msw
        return {
            "icc": float((msb - msw) / denom) if denom else np.nan,
            "msb": float(msb),
            "msw": float(msw),
            "n0": float(n0),
            "n_groups": int(k),
            "n_obs": int(N),
            "method": "icc1_oneway_unbalanced_local_fallback",
        }

    def icc_gene_clustered_bootstrap(
        y, groups, *, n_bootstrap=2000, ci_level=cfg.CI_LEVEL, seed=cfg.SEED
    ):
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups).astype(str)
        mask = np.isfinite(y)
        y, groups = y[mask], groups[mask]
        point = icc_oneway_unbalanced(y, groups)
        units = np.unique(groups)
        by_u = {u: y[groups == u] for u in units}
        rng = np.random.default_rng(seed)
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            drawn = rng.choice(units, size=len(units), replace=True)
            yb = np.concatenate([by_u[u] for u in drawn])
            gb = np.concatenate(
                [[f"{j}:{drawn[j]}"] * len(by_u[drawn[j]]) for j in range(len(drawn))]
            )
            boot[i] = icc_oneway_unbalanced(yb, gb)["icc"]
        valid = boot[np.isfinite(boot)]
        alpha = 1 - ci_level
        out = {
            "icc": point["icc"],
            "icc_details": point,
            "n_bootstrap": int(len(valid)),
            "n_groups": point.get("n_groups"),
            "n_obs": point.get("n_obs"),
        }
        if len(valid) >= 10:
            out["icc_ci_low"] = float(np.percentile(valid, 100 * alpha / 2))
            out["icc_ci_high"] = float(np.percentile(valid, 100 * (1 - alpha / 2)))
        else:
            out["icc_ci_low"] = out["icc_ci_high"] = np.nan
        print(
            "  NOTE: using local ICC fallback — re-upload stats_utils.py "
            "from the repo for the canonical helpers.",
            flush=True,
        )
        return out

# Inventory cutoff used only for the n=25 diagnostic / sensitivity ladder.
# Never the manuscript number (that remains cfg.MIN_CELLS).
SENSITIVITY_INVENTORY_MIN_CELLS = 10

# ---------------------------------------------------------------------------
# GEO GSE153056 — at-scale ECCITE-seq (Papalexi et al. 2021, Nat Genet)
# ---------------------------------------------------------------------------

GEO_ACCESSION = "GSE153056"
DATASET = "Papalexi 2021 (CRISPR-KO)"

# Prefer sample-level FTP (smaller than the 122 MB series RAW.tar).
GEO_FILES = {
    "metadata": {
        "fname": "GSE153056_ECCITE_metadata.tsv.gz",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153056/suppl/GSE153056_ECCITE_metadata.tsv.gz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&format=file&file=GSE153056%5FECCITE%5Fmetadata.tsv.gz",
        ],
        "min_bytes": 100_000,
    },
    "rna": {
        "fname": "GSM4633614_ECCITE_cDNA_counts.tsv.gz",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4633nnn/GSM4633614/suppl/GSM4633614_ECCITE_cDNA_counts.tsv.gz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4633614&format=file&file=GSM4633614%5FECCITE%5FcDNA%5Fcounts.tsv.gz",
        ],
        "min_bytes": 10_000_000,
    },
    "gdo_barcodes": {
        "fname": "GSM4633618_ECCITE_GDO_Barcodes.csv.gz",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4633nnn/GSM4633618/suppl/GSM4633618_ECCITE_GDO_Barcodes.csv.gz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4633618&format=file&file=GSM4633618%5FECCITE%5FGDO%5FBarcodes.csv.gz",
        ],
        "min_bytes": 500,
    },
}

# Optional: full series archive (contains the same RNA + GDO files).
GEO_RAW_TAR = {
    "fname": "GSE153056_RAW.tar",
    "urls": [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153056/suppl/GSE153056_RAW.tar",
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&format=file&file=GSE153056%5FRAW.tar",
    ],
    "min_bytes": 50_000_000,
    "members": {
        "rna": "GSM4633614_ECCITE_cDNA_counts.tsv.gz",
        "gdo_barcodes": "GSM4633618_ECCITE_GDO_Barcodes.csv.gz",
    },
}

NT_RE = re.compile(r"^NT", re.IGNORECASE)
GUIDE_GENE_RE = re.compile(r"^(.+?)g\d+$", re.IGNORECASE)


def geo_cache_dir(explicit: Path | None = None, out_dir: Path | None = None) -> Path:
    """
    Prefer a durable cache under out_dir (survives Colab /tmp wipes).
    Fall back to /tmp/pertpy_data/geo/... only if out_dir is unavailable.
    """
    if explicit is not None:
        p = Path(explicit)
        p.mkdir(parents=True, exist_ok=True)
        return p
    if out_dir is not None:
        cache = Path(out_dir) / "geo_cache" / GEO_ACCESSION
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    cache = setup_cache() / "geo" / GEO_ACCESSION
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def ensure_archived_h5ad(h5ad_path: Path, out_dir: Path) -> Path:
    """Copy built AnnData into out_dir so it is not /tmp-only."""
    import shutil

    dest = Path(out_dir) / "papalexi_2021_geo.h5ad"
    src = Path(h5ad_path)
    if not src.exists():
        return src
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists() and dest.stat().st_size >= src.stat().st_size * 0.9:
        print(f"  Archive h5ad already present: {dest}", flush=True)
        return dest
    print(f"  Archiving h5ad → {dest} ({src.stat().st_size / 1e6:.1f} MB) …", flush=True)
    shutil.copy2(src, dest)
    return dest


def _legacy_geo_dirs(out_dir: Path | None = None) -> list[Path]:
    """Places prior runs may have left GEO files / h5ad (Colab + local)."""
    dirs: list[Path] = []
    if out_dir is not None:
        dirs.append(Path(out_dir) / "geo_cache" / GEO_ACCESSION)
        # sibling out dirs from earlier notebook cells
        parent = Path(out_dir).parent
        if parent.exists():
            for child in sorted(parent.glob("shesha-crispr*")):
                dirs.append(child / "geo_cache" / GEO_ACCESSION)
                dirs.append(child)  # archived h5ad at out_dir root
    dirs.extend(
        [
            setup_cache() / "geo" / GEO_ACCESSION,
            Path("/tmp/pertpy_data/geo") / GEO_ACCESSION,
            Path.home() / ".cache" / "pertpy_data" / "geo" / GEO_ACCESSION,
            Path("/content/shesha-crispr/geo_cache") / GEO_ACCESSION,
            Path("/content/shesha-crispr"),
        ]
    )
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def find_existing_h5ad(
    preferred: Path,
    out_dir: Path,
    explicit: Path | None = None,
) -> Path | None:
    """Locate a built papalexi_2021_geo.h5ad across preferred + legacy paths."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(preferred),
            Path(out_dir) / "papalexi_2021_geo.h5ad",
        ]
    )
    for d in _legacy_geo_dirs(out_dir):
        candidates.append(d / "papalexi_2021_geo.h5ad")
        if d.name.endswith(".h5ad"):
            candidates.append(d)
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c
    return None


def find_existing_geo_dir(preferred: Path, out_dir: Path) -> Path | None:
    """Find a directory that already has the GEO metadata (+ preferably RNA)."""
    meta_name = GEO_FILES["metadata"]["fname"]
    for d in [preferred, *_legacy_geo_dirs(out_dir)]:
        if (d / meta_name).exists():
            return d
    return None


def resolve_geo_paths(
    geo_dir: Path,
    out_dir: Path,
    skip_download: bool,
    prefer_tar: bool = False,
) -> dict[str, Path]:
    """
    Map logical GEO keys → local files. Searches preferred geo_dir then legacy
    caches. Downloads only what is still missing (into geo_dir) unless
    skip_download.
    """
    search_dirs = [geo_dir, *[d for d in _legacy_geo_dirs(out_dir) if d != geo_dir]]
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for key, spec in GEO_FILES.items():
        hit = None
        for d in search_dirs:
            cand = d / spec["fname"]
            if cand.exists() and cand.stat().st_size >= spec["min_bytes"]:
                hit = cand
                if d != geo_dir:
                    print(f"  Reusing {key} from {cand}", flush=True)
                break
        if hit is None:
            missing.append(key)
        else:
            paths[key] = hit
    if not missing:
        return paths
    if skip_download:
        raise FileNotFoundError(
            "--skip-download but GEO files not found.\n"
            f"  looked under: {geo_dir}\n"
            f"  also tried: /tmp/pertpy_data/geo/{GEO_ACCESSION}\n"
            "  Fix (pick one):\n"
            "    1) Drop --skip-download to re-download (~65 MB RNA + metadata)\n"
            f"    2) --geo-dir /tmp/pertpy_data/geo/{GEO_ACCESSION}\n"
            "    3) --h5ad /path/to/papalexi_2021_geo.h5ad\n"
            f"  missing: {missing}"
        )
    # Download missing into preferred geo_dir; keep already-found paths.
    downloaded = ensure_geo_files(geo_dir, prefer_tar=prefer_tar)
    for key in missing:
        paths[key] = downloaded[key]
    return paths


def _download_first(urls: list[str], dest: Path, min_bytes: int) -> Path:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  Reusing {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
        return dest
    last_err: Exception | None = None
    for url in urls:
        try:
            return download(url, dest, min_bytes=min_bytes)
        except Exception as e:
            last_err = e
            print(f"    failed: {e}", flush=True)
            dest.unlink(missing_ok=True)
    raise FileNotFoundError(f"Could not download {dest.name}: {last_err}")


def ensure_geo_files(geo_dir: Path, prefer_tar: bool = False) -> dict[str, Path]:
    """Download metadata + RNA (+ GDO barcodes). Returns local paths."""
    paths: dict[str, Path] = {}

    meta_spec = GEO_FILES["metadata"]
    paths["metadata"] = _download_first(
        meta_spec["urls"], geo_dir / meta_spec["fname"], meta_spec["min_bytes"]
    )

    need = ["rna", "gdo_barcodes"]
    missing = [
        k
        for k in need
        if not (geo_dir / GEO_FILES[k]["fname"]).exists()
        or (geo_dir / GEO_FILES[k]["fname"]).stat().st_size < GEO_FILES[k]["min_bytes"]
    ]

    def _extract_from_tar(keys: list[str]) -> None:
        tar_path = _download_first(
            GEO_RAW_TAR["urls"],
            geo_dir / GEO_RAW_TAR["fname"],
            GEO_RAW_TAR["min_bytes"],
        )
        print(f"  Extracting from {tar_path.name} …", flush=True)
        with tarfile.open(tar_path, "r") as tar:
            names = tar.getnames()
            for key in keys:
                member = GEO_RAW_TAR["members"][key]
                hit = next((n for n in names if n.endswith(member)), None)
                if hit is None:
                    raise FileNotFoundError(f"{member} not in {tar_path.name}")
                dest = geo_dir / GEO_FILES[key]["fname"]
                src = tar.extractfile(hit)
                if src is None:
                    raise IOError(f"Could not extract {hit}")
                with open(dest, "wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                paths[key] = dest
                print(f"    extracted {dest.name}", flush=True)

    if missing and prefer_tar:
        _extract_from_tar(missing)
        missing = []
    elif missing:
        for key in list(missing):
            spec = GEO_FILES[key]
            try:
                paths[key] = _download_first(
                    spec["urls"], geo_dir / spec["fname"], spec["min_bytes"]
                )
                missing.remove(key)
            except FileNotFoundError as e:
                print(f"  Direct download of {key} failed ({e}); will try RAW.tar", flush=True)
        if missing:
            _extract_from_tar(missing)

    for key in need:
        if key not in paths:
            spec = GEO_FILES[key]
            paths[key] = _download_first(
                spec["urls"], geo_dir / spec["fname"], spec["min_bytes"]
            )

    return paths


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", index_col=0)
    # Strip residual quotes if any
    meta.index = meta.index.astype(str).str.strip('"')
    for c in meta.columns:
        if meta[c].dtype == object:
            meta[c] = meta[c].astype(str).str.strip('"')
    required = {"guide_ID", "gene", "crispr"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"GEO metadata missing columns {missing}; have {list(meta.columns)}")
    return meta


def load_gdo_library(path: Path) -> pd.DataFrame:
    """Parse GDO barcode CSV: sequence,guide_name (no header)."""
    df = pd.read_csv(path, header=None, names=["barcode", "guide"])
    df["guide"] = df["guide"].astype(str).str.strip()
    df["is_nt"] = df["guide"].str.upper().str.startswith("NT")
    df["gene"] = df["guide"].map(_gene_from_guide)
    return df


def _gene_from_guide(guide: str) -> str:
    g = str(guide).strip()
    if NT_RE.match(g) or g.upper() in {"NT", "NON-TARGETING", "NONTARGETING"}:
        return "NT"
    # Library uses PDL1 / PDL2 for CD274 / PDCD1LG2
    alias = {"PDL1": "CD274", "PDL2": "PDCD1LG2", "EGFP": "EGFP"}
    m = GUIDE_GENE_RE.match(g)
    gene = m.group(1) if m else g
    return alias.get(gene.upper(), gene)


def collapse_nt_guides(guides: pd.Series) -> pd.Series:
    """Map NTg1…NTg10 → NT; leave targeting guides unchanged."""
    out = guides.astype(str)
    return out.where(~out.map(lambda x: bool(NT_RE.match(x))), other="NT")


def load_rna_counts(path: Path, cell_ids: pd.Index) -> "anndata.AnnData":
    """
    Read GEO gene × cell TSV (gzipped) into cells × genes sparse AnnData.

    Aligns columns to metadata cell barcodes; drops cells absent from either side.
    """
    import anndata as ad

    print(f"  Reading RNA counts ({path.name}) — may take 1–3 min / ~2–4 GB peak RAM …", flush=True)
    # Seurat-style: rows = genes, cols = cells
    mat = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    mat.index = mat.index.astype(str).str.strip('"')
    mat.columns = mat.columns.astype(str).str.strip('"')
    print(f"    raw matrix: {mat.shape[0]} genes × {mat.shape[1]} cells", flush=True)

    shared = cell_ids.intersection(mat.columns)
    if len(shared) == 0:
        raise ValueError(
            f"No overlapping cell barcodes between metadata ({len(cell_ids)}) "
            f"and RNA matrix ({mat.shape[1]}). Example meta: {list(cell_ids[:3])}; "
            f"example RNA: {list(mat.columns[:3])}"
        )
    if len(shared) < len(cell_ids):
        print(
            f"    WARNING: {len(cell_ids) - len(shared)} metadata cells missing from RNA matrix",
            flush=True,
        )
    mat = mat.loc[:, shared]
    # cells × genes sparse
    X = sparse.csr_matrix(mat.to_numpy().T)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(shared, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(mat.index.astype(str), name="gene")),
    )
    print(f"    AnnData: {adata.n_obs} cells × {adata.n_vars} genes (sparse)", flush=True)
    return adata


def build_adata(paths: dict[str, Path], h5ad_out: Path) -> "anndata.AnnData":
    meta = load_metadata(paths["metadata"])
    gdo = load_gdo_library(paths["gdo_barcodes"])
    adata = load_rna_counts(paths["rna"], meta.index)

    # Attach metadata for cells present in RNA
    meta_aln = meta.loc[adata.obs_names].copy()
    adata.obs["guide_ID"] = meta_aln["guide_ID"].astype(str).values
    adata.obs["gene"] = meta_aln["gene"].astype(str).values
    adata.obs["gene_target"] = adata.obs["gene"]  # pertpy-compatible name
    adata.obs["guide_collapsed"] = collapse_nt_guides(adata.obs["guide_ID"])
    adata.obs["crispr"] = meta_aln["crispr"].astype(str).values
    for col in ("replicate", "Phase", "con", "MULTI_ID"):
        if col in meta_aln.columns:
            adata.obs[col] = meta_aln[col].astype(str).values
    if "percent.mito" in meta_aln.columns:
        adata.obs["percent_mito"] = pd.to_numeric(meta_aln["percent.mito"], errors="coerce").values

    adata.uns["geo_accession"] = GEO_ACCESSION
    adata.uns["source"] = "GEO GSE153056 ECCITE-seq (no pertpy)"
    adata.uns["gdo_library_n"] = int(len(gdo))
    adata.uns["gdo_library_guides"] = sorted(gdo["guide"].tolist())
    adata.uns["config_version"] = cfg.CONFIG_VERSION

    h5ad_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {h5ad_out} …", flush=True)
    adata.write_h5ad(h5ad_out)
    print(f"  Saved {h5ad_out} ({h5ad_out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return adata


def label_inventory(adata, gdo: pd.DataFrame | None, min_cells: int) -> pd.DataFrame:
    rows = []
    for level, col, ctrl in (
        ("gene", "gene_target", "NT"),
        ("guide", "guide_collapsed", "NT"),
    ):
        counts = adata.obs[col].astype(str).value_counts()
        for lab, n in counts.items():
            rows.append(
                {
                    "level": level,
                    "label": lab,
                    "n_cells": int(n),
                    "is_control": lab == ctrl,
                    "passes_min_cells": bool(lab == ctrl or n >= min_cells),
                    "gene": lab if level == "gene" else _gene_from_guide(lab),
                }
            )
    inv = pd.DataFrame(rows)
    if gdo is not None:
        assigned = set(adata.obs["guide_ID"].astype(str))
        lib = []
        for _, r in gdo.iterrows():
            lib.append(
                {
                    "level": "library",
                    "label": r["guide"],
                    "n_cells": int((adata.obs["guide_ID"] == r["guide"]).sum()),
                    "is_control": bool(r["is_nt"]),
                    "passes_min_cells": False,
                    "gene": r["gene"],
                    "in_assigned_metadata": r["guide"] in assigned,
                }
            )
        inv = pd.concat([inv, pd.DataFrame(lib)], ignore_index=True)
    return inv


def _annotate_sp_df(df: pd.DataFrame, level: str, pert_col: str, shared_basis: bool) -> pd.DataFrame:
    if not len(df):
        return df
    out = df.copy()
    out["label_level"] = level
    out["pert_col"] = pert_col
    out["source"] = "GEO"
    out["geo_accession"] = GEO_ACCESSION
    out["shared_basis"] = shared_basis
    if level == "guide":
        out["gene"] = out["perturbation"].map(_gene_from_guide)
    else:
        out["gene"] = out["perturbation"].astype(str)
    return out


def score_level(
    adata,
    *,
    pert_col: str,
    ctrl_label: str,
    level: str,
    min_cells: int,
    sc,
) -> pd.DataFrame:
    """Single-grain scoring (own HVG/PCA). Used only when --level is gene or guide alone."""
    print(f"\n>>> Scoring level={level!r}  pert_col={pert_col!r}  ctrl={ctrl_label!r}", flush=True)
    ad = adata.copy()
    ad.obs[pert_col] = ad.obs[pert_col].astype(str)

    ad, valid, counts = materialize_min_cells(
        ad,
        pert_col,
        ctrl_label,
        min_cells=min_cells,
        max_cells_per_pert=cfg.MAX_CELLS_PER_PERT,
        max_control_cells=cfg.MAX_CONTROL_CELLS,
        seed=cfg.SEED,
    )
    ad, valid, counts = preprocess(
        ad,
        pert_col,
        ctrl_label,
        sc,
        n_pcs=cfg.N_PCS,
        min_cells=min_cells,
        seed=cfg.SEED,
        valid_perts=valid,
        counts=counts,
        dataset_name=DATASET,
        matrix_is_log=cfg.DATASETS[DATASET]["matrix_is_log"],
    )
    df = score_perturbations(ad, pert_col, ctrl_label, valid, counts, DATASET)
    df = _annotate_sp_df(df, level, pert_col, shared_basis=False)
    print(f"    scored {len(df)} perturbations at {level} level (separate embedding)", flush=True)
    return df


def _genes_passing_gene_grain(adata, min_cells: int) -> set[str]:
    """Genes with ≥min_cells on the full object (gene_target grain; frozen parity set)."""
    counts = adata.obs["gene_target"].astype(str).value_counts()
    return {g for g, n in counts.items() if g != "NT" and n >= min_cells}


def score_shared_basis(
    adata,
    *,
    min_cells: int,
    sc,
) -> tuple[pd.DataFrame, pd.DataFrame, object, dict]:
    """
    One cell set → one HVG/PCA → score both grains.

    Materialize NT + guides with ≥min_cells (finest grain), fit embedding once,
    then score guide_collapsed and gene_target on the same X_pca.

    Returns (gene_shared_df, guide_df, embedded_adata, meta).
    gene_shared_df is for the paired contrast ONLY — never the manuscript
    gene-level headline (that comes from a separate gene-grain score, n=24).
    """
    print(
        "\n>>> Shared-basis scoring (guide-selected cells → one HVG/PCA → "
        "gene + guide Sp; paired-contrast numbers only)",
        flush=True,
    )
    frozen_gene_set = _genes_passing_gene_grain(adata, min_cells)

    ad = adata.copy()
    ad.obs["guide_collapsed"] = ad.obs["guide_collapsed"].astype(str)
    ad.obs["gene_target"] = ad.obs["gene_target"].astype(str)

    ad, valid_guides, guide_counts = materialize_min_cells(
        ad,
        "guide_collapsed",
        "NT",
        min_cells=min_cells,
        max_cells_per_pert=cfg.MAX_CELLS_PER_PERT,
        max_control_cells=cfg.MAX_CONTROL_CELLS,
        seed=cfg.SEED,
    )
    ad, valid_guides, guide_counts = preprocess(
        ad,
        "guide_collapsed",
        "NT",
        sc,
        n_pcs=cfg.N_PCS,
        min_cells=min_cells,
        seed=cfg.SEED,
        valid_perts=valid_guides,
        counts=guide_counts,
        dataset_name=DATASET,
        matrix_is_log=cfg.DATASETS[DATASET]["matrix_is_log"],
    )

    guide_df = score_perturbations(
        ad, "guide_collapsed", "NT", valid_guides, guide_counts, DATASET
    )
    guide_df = _annotate_sp_df(guide_df, "guide", "guide_collapsed", shared_basis=True)

    gene_counts = ad.obs["gene_target"].astype(str).value_counts()
    valid_genes = [
        g for g, n in gene_counts.items() if g != "NT" and n >= min_cells
    ]
    gene_df = score_perturbations(
        ad, "gene_target", "NT", valid_genes, gene_counts, DATASET
    )
    gene_df = _annotate_sp_df(gene_df, "gene", "gene_target", shared_basis=True)

    genes_guide = set(guide_df["gene"].astype(str)) if len(guide_df) else set()
    genes_shared_gene = set(gene_df["gene"].astype(str)) if len(gene_df) else set()
    # Genes in the frozen gene-grain set that never enter the shared embedding
    # because no individual guide reaches min_cells (typically MYC).
    dropped_by_guide_materialize = sorted(frozen_gene_set - genes_guide)
    myc_counts = {}
    for g in adata.obs["guide_collapsed"].astype(str).unique():
        if _gene_from_guide(g) == "MYC" and g != "NT":
            myc_counts[g] = int((adata.obs["guide_collapsed"].astype(str) == g).sum())
    myc_note = None
    if dropped_by_guide_materialize:
        myc_note = (
            f"Guide-grain materialization drops {dropped_by_guide_materialize} "
            f"relative to the frozen gene-grain set (n={len(frozen_gene_set)}). "
            f"Those genes have pooled gene-level n≥{min_cells} but no individual "
            f"guide ≥{min_cells}"
            + (f" (MYC guide counts: {myc_counts})" if myc_counts else "")
            + f". Shared-basis gene n={len(genes_shared_gene)}; "
            f"manuscript/parity gene n remains {len(frozen_gene_set)} from the "
            f"separate gene-grain score — never call shared-basis n parity."
        )

    meta = {
        "shared_basis": True,
        "role": "paired_contrast_only",
        "n_cells_embedded": int(ad.n_obs),
        "n_hvg": int(ad.n_vars),
        "n_pcs": int(ad.obsm["X_pca"].shape[1]),
        "n_control": int((ad.obs["guide_collapsed"].astype(str) == "NT").sum()),
        "materialize_grain": "guide_collapsed",
        "n_genes_frozen_gene_grain": int(len(frozen_gene_set)),
        "n_genes_shared_basis": int(len(genes_shared_gene)),
        "n_genes_with_passing_guides": int(len(genes_guide)),
        "genes_dropped_by_guide_materialize": dropped_by_guide_materialize,
        "genes_only_at_gene_level": dropped_by_guide_materialize,  # vs frozen set
        "genes_only_at_guide_level": sorted(genes_guide - genes_shared_gene),
        "myc_exclusion_note": myc_note,
        "note": (
            "HVG/PCA fit once on guide-materialized cells. Shared-basis gene Sp "
            "supports the paired contrast only. Manuscript gene-level n and "
            "mag–Sp come from the separate gene-grain score (frozen parity)."
        ),
    }
    print(
        f"    shared basis: {meta['n_cells_embedded']} cells × {meta['n_hvg']} HVG × "
        f"{meta['n_pcs']} PCs; shared-gene n={len(gene_df)}, guide n={len(guide_df)}; "
        f"frozen gene-grain n={len(frozen_gene_set)}",
        flush=True,
    )
    if myc_note:
        print(f"    {myc_note}", flush=True)
    return gene_df, guide_df, ad, meta


def _sp_on_indices(X_pca: np.ndarray, ctrl_idx: np.ndarray, pert_idx: np.ndarray) -> float:
    if len(pert_idx) < 5 or len(ctrl_idx) < cfg.MIN_CONTROL_CELLS:
        return np.nan
    m = calculate_sp(X_pca[ctrl_idx], X_pca[pert_idx])
    return float(m["stability"]) if m["magnitude"] > 0 else np.nan


def pseudo_guide_null(
    ad,
    guide_df: pd.DataFrame,
    gene_shared_df: pd.DataFrame,
    *,
    n_perm: int = 2000,
    seed: int = cfg.SEED,
) -> dict:
    """
    Group-size confound diagnostic for guide-median Sp > gene Sp.

    Randomly partition each gene's shared-basis cells into fake guides matching
    real guide sizes. The null median Δ is the size/self-consistency artifact;
    compare it to the observed gap. A small p only says a residual excess
    survives the null — it does NOT mean size fails to explain the gap.
    Typical outcome: size accounts for most of the guide>gene difference.
    """
    X = np.asarray(ad.obsm["X_pca"])
    labels_guide = ad.obs["guide_collapsed"].astype(str).to_numpy()
    labels_gene = ad.obs["gene_target"].astype(str).to_numpy()
    ctrl_idx = np.flatnonzero(labels_guide == "NT")

    gene_sp = {
        r["perturbation"]: float(r["stability"])
        for _, r in gene_shared_df.iterrows()
    }
    sizes_by_gene: dict[str, list[int]] = {}
    for gene, sub in guide_df.groupby("gene"):
        if gene == "NT":
            continue
        sizes_by_gene[str(gene)] = sorted(
            int(n) for n in sub["n_cells"].tolist()
        )

    genes = sorted(set(gene_sp) & set(sizes_by_gene))
    cells_by_gene = {g: np.flatnonzero(labels_gene == g) for g in genes}

    obs_deltas = []
    for g in genes:
        gmed = float(
            guide_df.loc[guide_df["gene"] == g, "stability"].median()
        )
        obs_deltas.append(gmed - gene_sp[g])
    obs_median_delta = float(np.median(obs_deltas)) if obs_deltas else np.nan
    gene_median_sp = (
        float(np.median([gene_sp[g] for g in genes])) if genes else np.nan
    )

    rng = np.random.default_rng(seed)
    null_medians = np.empty(n_perm)
    for p in range(n_perm):
        deltas = []
        for g in genes:
            idx = cells_by_gene[g].copy()
            sizes = sizes_by_gene[g]
            need = sum(sizes)
            if len(idx) < need:
                continue
            rng.shuffle(idx)
            pos = 0
            fake_sps = []
            for sz in sizes:
                fake_sps.append(_sp_on_indices(X, ctrl_idx, idx[pos : pos + sz]))
                pos += sz
            fake_sps = [s for s in fake_sps if np.isfinite(s)]
            if not fake_sps:
                continue
            deltas.append(float(np.median(fake_sps)) - gene_sp[g])
        null_medians[p] = float(np.median(deltas)) if deltas else np.nan

    valid = null_medians[np.isfinite(null_medians)]
    n_valid = int(len(valid))
    null_med = float(np.median(valid)) if n_valid else np.nan
    if n_valid and np.isfinite(obs_median_delta):
        n_ge = int(np.sum(valid >= obs_median_delta - 1e-15))
        # Conservative +1; at floor when n_ge==0 → p = 1/(n+1)
        p_null = float((n_ge + 1) / (n_valid + 1))
        p_at_floor = bool(n_ge == 0)
    else:
        n_ge = 0
        p_null = np.nan
        p_at_floor = False

    residual = (
        float(obs_median_delta - null_med)
        if np.isfinite(obs_median_delta) and np.isfinite(null_med)
        else np.nan
    )
    size_frac = (
        float(null_med / obs_median_delta)
        if np.isfinite(obs_median_delta)
        and np.isfinite(null_med)
        and abs(obs_median_delta) > 1e-15
        else np.nan
    )

    return {
        "n_genes": int(len(genes)),
        "n_perm": n_valid,
        "n_perm_requested": int(n_perm),
        "observed_median_delta_guide_minus_gene": obs_median_delta,
        "null_median_delta_mean": float(np.mean(valid)) if n_valid else np.nan,
        "null_median_delta_median": null_med,
        "null_median_delta_ci": (
            [
                float(np.percentile(valid, 2.5)),
                float(np.percentile(valid, 97.5)),
            ]
            if n_valid >= 20
            else None
        ),
        "fraction_of_gap_attributable_to_size": size_frac,
        "residual_median_delta": residual,
        "gene_median_sp_base": gene_median_sp,
        "residual_vs_gene_median_sp": (
            float(residual / gene_median_sp)
            if np.isfinite(residual)
            and np.isfinite(gene_median_sp)
            and abs(gene_median_sp) > 1e-15
            else np.nan
        ),
        "p_excess_over_null": p_null,
        "p_at_resolution_floor": p_at_floor,
        "n_null_ge_observed": n_ge,
        "interpretation": (
            "Most of the guide-over-gene median Sp gap is attributable to "
            "group-size / Sp self-consistency bias (null median Δ / observed). "
            "A small residual excess may survive the null; that is footnote-scale "
            "and consistent with any guide-correlated structure (efficiency, "
            "batch, cell cycle, etc.) — do not name reagent efficiency as the "
            "cause. claim_allowed stays false; do not headline the raw contrast."
        ),
        "note": (
            "Pseudo-guide null diagnoses the size confound. Small p means a "
            "residual exceeds the size-matched null, not that size fails to "
            "explain most of the gap. Residual = guide-correlated structure, "
            "not a causal efficiency claim."
        ),
    }


def paired_gene_vs_guide_median(
    gene_df: pd.DataFrame,
    guide_df: pd.DataFrame,
    *,
    shared_basis: bool,
    pseudo_null: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Per shared gene: gene-level Sp vs median guide-level Sp (Wilcoxon).

    claim_allowed is always False for manuscript text: the pseudo-guide null
    typically shows most of the guide>gene gap is group-size artifact. Report
    the size fraction and residual as a footnote diagnostic only.
    """
    gmed = (
        guide_df.groupby("gene", as_index=False)
        .agg(
            stability_guide_median=("stability", "median"),
            magnitude_guide_median=("magnitude", "median"),
            n_guides=("perturbation", "count"),
            median_n_cells_guide=("n_cells", "median"),
        )
    )
    gene_side = gene_df[["perturbation", "stability", "magnitude", "n_cells"]].rename(
        columns={
            "perturbation": "gene",
            "stability": "stability_gene",
            "magnitude": "magnitude_gene",
            "n_cells": "n_cells_gene",
        }
    )
    cross = gene_side.merge(gmed, on="gene", how="inner")
    cross["delta_guide_minus_gene"] = (
        cross["stability_guide_median"] - cross["stability_gene"]
    )
    n = int(len(cross))
    out: dict = {
        "n_shared_genes": n,
        "shared_basis": shared_basis,
        "claim_allowed": False,
        "median_sp_gene_n_shared": float(cross["stability_gene"].median()) if n else None,
        "median_sp_guide_median_n_shared": (
            float(cross["stability_guide_median"].median()) if n else None
        ),
        "median_delta_guide_minus_gene": (
            float(cross["delta_guide_minus_gene"].median()) if n else None
        ),
        "n_genes_guide_median_higher": (
            int((cross["delta_guide_minus_gene"] > 0).sum()) if n else 0
        ),
        "mean_n_cells_gene": float(cross["n_cells_gene"].mean()) if n else None,
        "mean_n_cells_guide": float(cross["median_n_cells_guide"].mean()) if n else None,
        "denominator_note": (
            f"Paired medians are over n={n} shared genes on the shared basis only — "
            f"not the frozen gene-grain manuscript n. Do not headline the raw "
            f"guide vs gene median contrast (size-confounded)."
        ),
    }
    if not shared_basis:
        out["caveat"] = (
            "NOT INTERPRETABLE: separate HVG/PCA fits. claim_allowed=false."
        )
        return cross, out

    if n >= 5:
        try:
            stat, p = wilcoxon(
                cross["stability_guide_median"].to_numpy(),
                cross["stability_gene"].to_numpy(),
                alternative="greater",
                zero_method="wilcox",
            )
            out["wilcoxon_guide_gt_gene_stat"] = float(stat)
            out["wilcoxon_guide_gt_gene_p"] = float(p)
            out["wilcoxon_alternative"] = "greater"
            out["wilcoxon_saturated"] = bool(
                out["n_genes_guide_median_higher"] == n and p <= 2.0 ** (-n) * 1.01
            )
        except ValueError as e:
            out["wilcoxon_error"] = str(e)
        rho, p_rho = spearmanr(cross["stability_gene"], cross["stability_guide_median"])
        out["spearman_gene_vs_guide_median"] = float(rho)
        out["spearman_gene_vs_guide_median_p"] = float(p_rho)

    # Never unlock a labeling-grain claim from this contrast.
    out["claim_allowed"] = False
    if pseudo_null is not None:
        out["pseudo_guide_null"] = pseudo_null
        sf = pseudo_null.get("fraction_of_gap_attributable_to_size")
        res = pseudo_null.get("residual_median_delta")
        out["caveat"] = (
            "Pseudo-guide null: most of the guide>gene median Sp gap is "
            f"group-size / self-consistency (size fraction≈{_fmt_rho(sf)}; "
            f"residual Δ≈{_fmt_rho(res)}). Footnote diagnostic only — "
            "claim_allowed=false; do not use the near-2× median contrast in text."
        )
    else:
        out["caveat"] = (
            "Pseudo-guide null not run. claim_allowed=false: the guide>gene "
            "gap is expected under a pure size null."
        )
    return cross, out


def _unit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def between_guide_shift_cosine(
    ad,
    gene_df: pd.DataFrame,
    guide_df: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = cfg.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Item 16 predictive endpoint: does gene-level Sp predict that a
    perturbation's effect direction replicates across independent reagents?

    For each gene with ≥2 guides on the shared embedding: cosine between
    guide_A and guide_B mean-shift vectors (same NT control). Per gene, take
    the median pairwise cosine; Spearman(gene Sp, median cosine), plus partial
    vs magnitude.

    Still partly mechanical (tightly concentrated directions agree by
    construction), but guides are independent reagents in independent cells —
    a real step down from split-half circularity. Papalexi is thin; Replogle
    is the flagship for this claim.
    """
    X = np.asarray(ad.obsm["X_pca"])
    labels_guide = ad.obs["guide_collapsed"].astype(str).to_numpy()
    ctrl_idx = np.flatnonzero(labels_guide == "NT")
    if len(ctrl_idx) < cfg.MIN_CONTROL_CELLS:
        return pd.DataFrame(), pd.DataFrame(), {
            "error": f"Insufficient NT cells: {len(ctrl_idx)}"
        }
    ctrl_centroid = X[ctrl_idx].mean(axis=0)

    gene_sp_map = {
        str(r["perturbation"]): float(r["stability"])
        for _, r in gene_df.iterrows()
    }
    gene_mag_map = {
        str(r["perturbation"]): float(r["magnitude"])
        for _, r in gene_df.iterrows()
    }

    # Mean shift per guide
    shifts: dict[str, dict] = {}
    for _, r in guide_df.iterrows():
        g = str(r["gene"])
        if g == "NT":
            continue
        guide = str(r["perturbation"])
        idx = np.flatnonzero(labels_guide == guide)
        if len(idx) < 5:
            continue
        mean_shift = X[idx].mean(axis=0) - ctrl_centroid
        shifts[guide] = {
            "gene": g,
            "guide": guide,
            "mean_shift": mean_shift,
            "n_cells": int(len(idx)),
            "sp_guide": float(r["stability"]),
            "mag_guide": float(r["magnitude"]),
        }

    by_gene: dict[str, list] = {}
    for info in shifts.values():
        by_gene.setdefault(info["gene"], []).append(info)

    pair_rows = []
    gene_rows = []
    for gene, items in sorted(by_gene.items()):
        if len(items) < 2 or gene not in gene_sp_map:
            continue
        cosines = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                cos = _unit_cosine(items[i]["mean_shift"], items[j]["mean_shift"])
                if not np.isfinite(cos):
                    continue
                cosines.append(cos)
                pair_rows.append(
                    {
                        "gene": gene,
                        "guide_a": items[i]["guide"],
                        "guide_b": items[j]["guide"],
                        "cosine_mean_shift": cos,
                        "n_a": items[i]["n_cells"],
                        "n_b": items[j]["n_cells"],
                        "sp_a": items[i]["sp_guide"],
                        "sp_b": items[j]["sp_guide"],
                        "gene_sp": gene_sp_map[gene],
                        "gene_magnitude": gene_mag_map[gene],
                    }
                )
        if not cosines:
            continue
        gene_rows.append(
            {
                "gene": gene,
                "n_guides": int(len(items)),
                "n_pairs": int(len(cosines)),
                "median_cosine_mean_shift": float(np.median(cosines)),
                "mean_cosine_mean_shift": float(np.mean(cosines)),
                "gene_sp": gene_sp_map[gene],
                "gene_magnitude": gene_mag_map[gene],
            }
        )

    pairs = pd.DataFrame(pair_rows)
    genes = pd.DataFrame(gene_rows)
    mean_pairs_per_gene = (
        float(len(pairs) / len(genes)) if len(genes) else np.nan
    )
    summary: dict = {
        "analysis": "between_guide_shift_cosine",
        "endpoint": (
            "gene_sp → median pairwise cosine(guide mean shifts); "
            "independent-reagent, shared-control"
        ),
        "n_genes": int(len(genes)),
        "n_pairs": int(len(pairs)),
        "mean_pairs_per_gene": mean_pairs_per_gene,
        "role": (
            "Item 16 predictive endpoint — Papalexi is the smaller, secondary "
            "case. Replogle (flagship) must lead any manuscript sentence; "
            "do not lead with Papalexi's uncontrolled rho."
        ),
        "claim_evaluable": False,  # flipped True only after partial runs
        "caveat": (
            "Partly mechanical: tightly concentrated directions agree by "
            "construction. Uncontrolled Spearman is not evaluable until "
            "rank-based partial|magnitude is reported. Papalexi has many "
            "pairs/gene (measurement averaging); Replogle is mostly one "
            "pair/gene (noisier outcome)."
        ),
    }
    if len(genes) < 5:
        summary["spearman_gene_sp_vs_median_cosine"] = np.nan
        return pairs, genes, summary

    rho, p = spearmanr(genes["gene_sp"], genes["median_cosine_mean_shift"])
    summary["spearman_gene_sp_vs_median_cosine"] = float(rho)
    summary["spearman_gene_sp_vs_median_cosine_p"] = float(p)
    boot = bootstrap_spearman_ci(
        genes["gene_sp"],
        genes["median_cosine_mean_shift"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary["spearman_gene_sp_vs_median_cosine_boot"] = boot

    part = partial_spearman_rank(
        genes["gene_sp"].to_numpy(),
        genes["median_cosine_mean_shift"].to_numpy(),
        genes["gene_magnitude"].to_numpy(),
    )
    summary["partial_spearman_sp_vs_cosine_given_magnitude"] = part
    part_ci = bootstrap_partial_spearman_ci(
        genes["gene_sp"].to_numpy(),
        genes["median_cosine_mean_shift"].to_numpy(),
        genes["gene_magnitude"].to_numpy(),
        n_bootstrap=min(n_bootstrap, 2000),
        seed=seed,
    )
    summary["partial_spearman_sp_vs_cosine_given_magnitude_boot"] = part_ci
    summary["claim_evaluable"] = bool(
        part.get("rho_partial") is not None and np.isfinite(part.get("rho_partial", np.nan))
    )

    # Attenuation check: one random pair/gene (matches Replogle's ~1 pair/gene)
    if len(pairs) and mean_pairs_per_gene > 1.05:
        summary["one_pair_per_gene_attenuation"] = _one_pair_per_gene_attenuation(
            pairs, n_seeds=50, seed=seed, n_bootstrap=min(500, n_bootstrap)
        )

    summary["median_of_gene_median_cosines"] = float(
        genes["median_cosine_mean_shift"].median()
    )
    att = summary.get("one_pair_per_gene_attenuation") or {}
    print(
        f"  Between-guide shift cosine (Papalexi secondary): "
        f"n_genes={summary['n_genes']} n_pairs={summary['n_pairs']} "
        f"(mean pairs/gene={mean_pairs_per_gene:.2f})  "
        f"Spearman={rho:.3f} [{boot.get('ci_low')}, {boot.get('ci_high')}]  "
        f"partial|mag={part.get('rho_partial')} "
        f"[{part_ci.get('ci_low')}, {part_ci.get('ci_high')}] "
        f"({part.get('method')})",
        flush=True,
    )
    if att:
        print(
            f"  One-pair/gene attenuation ({att.get('n_seeds')} seeds): "
            f"Spearman median={att.get('spearman_median')} "
            f"[{att.get('spearman_ci_low')}, {att.get('spearman_ci_high')}]  "
            f"partial|mag median={att.get('partial_median')} "
            f"— if this falls toward Replogle (~0.36), gap is measurement noise",
            flush=True,
        )
    return pairs, genes, summary


def _one_pair_per_gene_attenuation(
    pairs: pd.DataFrame,
    *,
    n_seeds: int = 50,
    seed: int = cfg.SEED,
    n_bootstrap: int = 500,
) -> dict:
    """
    Restrict to one randomly chosen guide pair per gene (Replogle-like noise),
    repeat over seeds. If Papalexi ~0.83 falls toward ~0.4, the gap vs Replogle
    is mostly outcome measurement noise, not biology.
    """
    rhos = []
    partials = []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed + s)
        rows = []
        for _, sub in pairs.groupby("gene"):
            rows.append(sub.iloc[int(rng.integers(0, len(sub)))])
        samp = pd.DataFrame(rows)
        if len(samp) < 5:
            continue
        rho, _ = spearmanr(samp["gene_sp"], samp["cosine_mean_shift"])
        if np.isfinite(rho):
            rhos.append(float(rho))
        part = partial_spearman_rank(
            samp["gene_sp"].to_numpy(),
            samp["cosine_mean_shift"].to_numpy(),
            samp["gene_magnitude"].to_numpy(),
        )
        if np.isfinite(part.get("rho_partial", np.nan)):
            partials.append(float(part["rho_partial"]))
    rhos_a = np.asarray(rhos, dtype=float)
    parts_a = np.asarray(partials, dtype=float)
    out = {
        "n_seeds": int(len(rhos_a)),
        "n_seeds_requested": int(n_seeds),
        "spearman_median": float(np.median(rhos_a)) if len(rhos_a) else np.nan,
        "spearman_mean": float(np.mean(rhos_a)) if len(rhos_a) else np.nan,
        "spearman_ci_low": (
            float(np.percentile(rhos_a, 2.5)) if len(rhos_a) >= 10 else np.nan
        ),
        "spearman_ci_high": (
            float(np.percentile(rhos_a, 97.5)) if len(rhos_a) >= 10 else np.nan
        ),
        "partial_median": float(np.median(parts_a)) if len(parts_a) else np.nan,
        "partial_mean": float(np.mean(parts_a)) if len(parts_a) else np.nan,
        "partial_ci_low": (
            float(np.percentile(parts_a, 2.5)) if len(parts_a) >= 10 else np.nan
        ),
        "partial_ci_high": (
            float(np.percentile(parts_a, 97.5)) if len(parts_a) >= 10 else np.nan
        ),
        "note": (
            "One random guide-pair cosine per gene, repeated over seeds, to "
            "match Replogle's ~1 pair/gene measurement noise. Compare median "
            "Spearman to the full multi-pair Papalexi estimate and to Replogle."
        ),
    }
    return out


def between_guide_sp_icc(
    guide_df: pd.DataFrame,
    *,
    min_guides: int = 2,
    n_bootstrap: int = 2000,
    seed: int = cfg.SEED,
) -> tuple[pd.DataFrame, dict]:
    """
    Between-guide Sp metric concordance via ICC(1), not the item-16 predictive
    endpoint (see between_guide_shift_cosine).

    Asks whether guides targeting the same gene produce similar Sp values —
    Sp as a stable property of gene×context across reagents. Does NOT ask
    whether Sp predicts that effect directions replicate across guides.
    """
    sub = guide_df[guide_df["gene"].astype(str) != "NT"].copy()
    sizes = sub.groupby("gene").size()
    keep = sizes[sizes >= min_guides].index
    sub = sub[sub["gene"].isin(keep)]
    detail_cols = ["gene", "perturbation", "stability", "n_cells"]
    if "magnitude" in sub.columns:
        detail_cols.insert(3, "magnitude")
    detail = sub[detail_cols].copy()

    summary: dict = {
        "analysis": "between_guide_sp_icc",
        "replication_label": "independent-reagent, shared-control replication",
        "n_genes_multi_guide": int(sub["gene"].nunique()),
        "n_guides": int(len(sub)),
        "min_guides": min_guides,
        "note": (
            "ICC(1) = metric concordance: guides targeting one gene produce "
            "similar Sp (gene×context across reagents; same NT control). "
            "Not the item-16 predictive endpoint (gene Sp → cosine of guide "
            "mean shifts). Paired bootstrap for ICC(Sp), ICC(magnitude), and Δ. "
            "If Δ≈0, Sp concordance is not independent of effect-size agreement."
        ),
    }
    if sub["gene"].nunique() < 3:
        summary["icc"] = np.nan
        return detail, summary

    genes_arr = sub["gene"].to_numpy()
    sp_vals = sub["stability"].to_numpy(dtype=float)
    point_sp = icc_oneway_unbalanced(sp_vals, genes_arr)
    summary["icc"] = point_sp["icc"]
    summary["icc_details"] = point_sp

    has_mag = "magnitude" in sub.columns
    mag_vals = sub["magnitude"].to_numpy(dtype=float) if has_mag else None
    point_mag = (
        icc_oneway_unbalanced(mag_vals, genes_arr) if has_mag else None
    )
    if point_mag is not None:
        summary["icc_magnitude"] = point_mag["icc"]
        summary["icc_sp_minus_icc_magnitude"] = float(
            point_sp["icc"] - point_mag["icc"]
        )

    # One gene resample → both ICCs → Δ. Separate CIs do not give a CI on Δ.
    rng = np.random.default_rng(seed)
    units = np.unique(genes_arr)
    by_sp = {u: sp_vals[genes_arr == u] for u in units}
    by_mag = (
        {u: mag_vals[genes_arr == u] for u in units} if has_mag else None
    )
    boot_sp = np.empty(n_bootstrap)
    boot_mag = np.empty(n_bootstrap) if has_mag else None
    boot_delta = np.empty(n_bootstrap) if has_mag else None
    for i in range(n_bootstrap):
        drawn = rng.choice(units, size=len(units), replace=True)
        y_sp = np.concatenate([by_sp[u] for u in drawn])
        g = np.concatenate(
            [[f"{j}:{drawn[j]}"] * len(by_sp[drawn[j]]) for j in range(len(drawn))]
        )
        isp = icc_oneway_unbalanced(y_sp, g)["icc"]
        boot_sp[i] = isp
        if has_mag:
            y_mag = np.concatenate([by_mag[u] for u in drawn])
            imag = icc_oneway_unbalanced(y_mag, g)["icc"]
            boot_mag[i] = imag
            boot_delta[i] = isp - imag

    alpha = 1 - cfg.CI_LEVEL
    valid_sp = boot_sp[np.isfinite(boot_sp)]
    summary["n_bootstrap"] = int(len(valid_sp))
    if len(valid_sp) >= 10:
        summary["icc_ci_low"] = float(np.percentile(valid_sp, 100 * alpha / 2))
        summary["icc_ci_high"] = float(np.percentile(valid_sp, 100 * (1 - alpha / 2)))
    else:
        summary["icc_ci_low"] = summary["icc_ci_high"] = np.nan

    if has_mag and boot_mag is not None:
        valid_mag = boot_mag[np.isfinite(boot_mag)]
        if len(valid_mag) >= 10:
            summary["icc_magnitude_ci_low"] = float(
                np.percentile(valid_mag, 100 * alpha / 2)
            )
            summary["icc_magnitude_ci_high"] = float(
                np.percentile(valid_mag, 100 * (1 - alpha / 2))
            )
        else:
            summary["icc_magnitude_ci_low"] = summary["icc_magnitude_ci_high"] = np.nan
        valid_d = boot_delta[np.isfinite(boot_delta)]
        summary["n_bootstrap_delta"] = int(len(valid_d))
        if len(valid_d) >= 10:
            summary["icc_sp_minus_icc_magnitude_ci_low"] = float(
                np.percentile(valid_d, 100 * alpha / 2)
            )
            summary["icc_sp_minus_icc_magnitude_ci_high"] = float(
                np.percentile(valid_d, 100 * (1 - alpha / 2))
            )
        else:
            summary["icc_sp_minus_icc_magnitude_ci_low"] = np.nan
            summary["icc_sp_minus_icc_magnitude_ci_high"] = np.nan
        # Negative / null: Sp agreement ≈ magnitude agreement.
        d = summary.get("icc_sp_minus_icc_magnitude")
        d_lo = summary.get("icc_sp_minus_icc_magnitude_ci_low")
        d_hi = summary.get("icc_sp_minus_icc_magnitude_ci_high")
        summary["icc_sp_vs_magnitude_note"] = (
            "Paired bootstrap Δ=ICC(Sp)−ICC(magnitude). If Δ≈0 (CI covers 0), "
            "between-guide Sp concordance is not independent evidence beyond "
            "effect-size agreement — expected when guide mag–Sp ρ is high. "
            "Report ICC(Sp) for reagent-level reproducibility; do not claim Sp "
            "captures something magnitude misses."
            + (
                f" Observed Δ={d:.4f} [{d_lo:.4f}, {d_hi:.4f}]."
                if d is not None
                and d_lo is not None
                and d_hi is not None
                and np.isfinite(d)
                and np.isfinite(d_lo)
                and np.isfinite(d_hi)
                else ""
            )
        )

    abs_diffs = []
    for gene, gdf in sub.groupby("gene"):
        sps = gdf["stability"].to_numpy()
        if len(sps) < 2:
            continue
        for i in range(len(sps)):
            for j in range(i + 1, len(sps)):
                abs_diffs.append(abs(sps[i] - sps[j]))
    if abs_diffs:
        summary["median_abs_diff_descriptive"] = float(np.median(abs_diffs))
        summary["n_unordered_pairs_descriptive"] = int(len(abs_diffs))
    return detail, summary


def compare_pertpy(gene_df: pd.DataFrame, out_dir: Path, frozen_csv: Path | None) -> dict:
    """Correlate GEO gene-level Sp with frozen / pertpy Sp table if available."""
    report: dict = {"available": False}
    try:
        path = find_sp_csv(out_dir, frozen_csv)
    except FileNotFoundError as e:
        report["error"] = str(e)
        return report

    frozen = load_sp_table(path)
    sub = frozen[frozen["dataset"] == DATASET].copy()
    if sub.empty:
        report["error"] = f"No {DATASET} rows in {path}"
        return report

    merged = gene_df.merge(
        sub[["perturbation", "stability", "magnitude", "n_cells"]].rename(
            columns={
                "stability": "stability_frozen",
                "magnitude": "magnitude_frozen",
                "n_cells": "n_cells_frozen",
            }
        ),
        on="perturbation",
        how="outer",
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"]
    report["available"] = True
    report["frozen_csv"] = str(path)
    report["n_geo"] = int(len(gene_df))
    report["n_frozen"] = int(len(sub))
    report["n_shared"] = int(len(both))
    report["only_geo"] = sorted(
        merged.loc[merged["_merge"] == "left_only", "perturbation"].astype(str)
    )
    report["only_frozen"] = sorted(
        merged.loc[merged["_merge"] == "right_only", "perturbation"].astype(str)
    )
    if len(both) >= 5:
        rho_s, p_s = spearmanr(both["stability"], both["stability_frozen"])
        rho_m, p_m = spearmanr(both["magnitude"], both["magnitude_frozen"])
        report["spearman_sp"] = float(rho_s)
        report["spearman_sp_p"] = float(p_s)
        report["spearman_magnitude"] = float(rho_m)
        report["spearman_magnitude_p"] = float(p_m)
    merged.to_csv(out_dir / "papalexi_geo_vs_frozen.csv", index=False)
    return report


def _fmt_rho(x) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{float(x):.3f}"


def _fmt_p(p, *, n_perm: int | None = None, at_floor: bool = False) -> str:
    """Format a p-value; at permutation floor report p<1/(n+1)."""
    if p is None or not np.isfinite(p):
        return "NA"
    if at_floor and n_perm and n_perm > 0:
        return f"<{1.0 / (n_perm + 1):.3g}"
    if n_perm and n_perm > 0 and p <= (1.0 / (n_perm + 1)) * 1.01:
        return f"<{1.0 / (n_perm + 1):.3g}"
    return f"{float(p):.3g}"


def _fmt_ci(boot: dict | None, lo_key: str = "ci_low", hi_key: str = "ci_high") -> str:
    if not boot:
        return ""
    lo, hi = boot.get(lo_key), boot.get(hi_key)
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f" [{float(lo):.3f}, {float(hi):.3f}]"


def _n_pass(inv: pd.DataFrame, level: str, min_cells: int) -> int:
    sub = inv[(inv["level"] == level) & (~inv["is_control"])]
    return int((sub["n_cells"] >= min_cells).sum())


def _summarize_sp_df(
    df: pd.DataFrame,
    *,
    level: str,
    out_csv: Path,
    shared_basis: bool,
    n_boot: int,
    role: str | None = None,
) -> dict:
    """Build a level summary; every median/rho carries its denominator."""
    level_sum: dict = {
        "n_scored": int(len(df)),
        "csv": str(out_csv),
        "shared_basis": shared_basis,
        "role": role,
        "denominator": (
            f"n={len(df)} genes" if level == "gene" else f"n={len(df)} guides"
        ),
    }
    if role:
        level_sum["role_note"] = role
    if len(df) < 5:
        return level_sum
    rho, p = spearmanr(df["magnitude"], df["stability"])
    level_sum["spearman_mag_sp"] = float(rho)
    level_sum["spearman_mag_sp_p"] = float(p)
    level_sum["median_sp"] = float(df["stability"].median())
    level_sum["median_sp_denominator"] = level_sum["denominator"]
    level_sum["perturbations"] = sorted(df["perturbation"].astype(str).tolist())
    if level == "guide":
        gci = bootstrap_spearman_ci_clustered(
            df["magnitude"],
            df["stability"],
            df["gene"],
            n_bootstrap=n_boot,
        )
        level_sum["spearman_mag_sp_gene_clustered"] = gci
        level_sum["n_genes"] = gci.get("n_clusters")
        print(
            f"    guide mag–Sp rho={rho:.3f}  "
            f"gene-clustered 95% CI [{gci['ci_low']:.3f}, {gci['ci_high']:.3f}] "
            f"(n_guides={len(df)}, n_genes={gci.get('n_clusters')}; "
            f"median Sp={level_sum['median_sp']:.3f})",
            flush=True,
        )
    else:
        boot = bootstrap_spearman_ci(
            df["magnitude"], df["stability"], n_bootstrap=n_boot
        )
        level_sum["spearman_mag_sp_boot"] = boot
        tag = "shared-basis gene" if shared_basis else "manuscript gene-grain"
        print(
            f"    {tag} mag–Sp rho={rho:.3f}  "
            f"95% CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] "
            f"(n_genes={len(df)}; median Sp={level_sum['median_sp']:.3f})",
            flush=True,
        )
    return level_sum


def methods_blurb(summary: dict) -> str:
    """
    Manuscript-facing text. Never call a non-50 cutoff 'frozen pipeline'.
    Always attach the cutoff when saying 'parity'.
    """
    g = summary.get("gene") or {}
    gd = summary.get("guide") or {}
    ladder = summary.get("reporting_ladder") or {}
    lib_n = ladder.get("n_guides_deposited", summary.get("gdo_library_n"))
    n_guide_frozen = ladder.get("n_guides_frozen_min_cells")
    n_guide_run = ladder.get("n_guides_this_cutoff", gd.get("n_scored"))
    n_gene_frozen = ladder.get("n_genes_frozen_min_cells")
    n_gene_run = ladder.get("n_genes_this_cutoff", g.get("n_scored"))
    sens = bool(summary.get("is_sensitivity", False))
    cutoff = int(summary.get("min_cells"))
    frozen = int(summary.get("frozen_min_cells", cfg.MIN_CELLS))

    if sens:
        settings = (
            f"Under frozen pipeline settings (N_PCS={cfg.N_PCS}, seed={cfg.SEED}) "
            f"with MIN_CELLS relaxed to {cutoff} as a sensitivity analysis "
            f"(manuscript number remains MIN_CELLS={frozen})"
        )
    else:
        settings = (
            f"Under frozen pipeline settings "
            f"(MIN_CELLS={frozen}, N_PCS={cfg.N_PCS}, seed={cfg.SEED})"
        )

    # Headline gene numbers: frozen gene-grain only (never shared-basis n=23).
    gene_head = summary.get("gene") or g
    gene_n = gene_head.get("n_scored")
    if gene_n is None:
        gene_n = n_gene_frozen if not sens else n_gene_run
    guide_n = gd.get("n_scored")
    if guide_n is None:
        guide_n = n_guide_run

    parity = (
        f"parity with pertpy/scPerturb gene_target at matched "
        f"MIN_CELLS={cutoff}"
        if (not sens and gene_n == n_gene_frozen)
        else f"at MIN_CELLS={cutoff}"
    )

    shared_meta = summary.get("shared_basis_meta") or {}
    shared = bool(shared_meta.get("shared_basis"))
    basis_txt = ""
    if shared:
        n_shared_g = shared_meta.get("n_genes_shared_basis")
        basis_txt = (
            f" Guide Sp and the paired gene contrast use one shared HVG/PCA "
            f"(guide-materialized; shared-basis gene n={n_shared_g}, "
            f"not the manuscript gene n={gene_n})."
        )

    rho_bits = []
    if gene_head.get("spearman_mag_sp") is not None:
        rho_bits.append(
            f"manuscript gene-level magnitude–Sp Spearman rho="
            f"{_fmt_rho(gene_head['spearman_mag_sp'])}"
            f"{_fmt_ci(gene_head.get('spearman_mag_sp_boot'))} "
            f"(n={gene_head.get('n_scored')} genes, gene-grain embedding; "
            f"median Sp={_fmt_rho(gene_head.get('median_sp'))})"
        )
    if gd.get("spearman_mag_sp") is not None:
        gci = gd.get("spearman_mag_sp_gene_clustered") or {}
        n_cl = gci.get("n_clusters")
        emb = (
            "shared-basis embedding"
            if gd.get("shared_basis")
            else "guide-grain embedding"
        )
        rho_bits.append(
            f"guide-level rho={_fmt_rho(gd['spearman_mag_sp'])}"
            f"{_fmt_ci(gci)} "
            f"(n={gd.get('n_scored')} guides over n_genes={n_cl}; "
            f"median Sp={_fmt_rho(gd.get('median_sp'))}; gene-clustered bootstrap; "
            f"{emb})"
        )
    # Never put shared-basis gene mag–Sp in the headline sentence.
    g_shared = summary.get("gene_shared_basis") or {}
    if g_shared.get("spearman_mag_sp") is not None:
        rho_bits.append(
            f"shared-basis gene mag–Sp rho={_fmt_rho(g_shared['spearman_mag_sp'])} "
            f"(n={g_shared.get('n_scored')} genes; paired-contrast only — "
            f"not the Papalexi headline)"
        )
    rho_txt = ("; ".join(rho_bits) + ".") if rho_bits else ""

    sens_inv = (summary.get("reporting_ladder") or {}).get("sensitivity_only") or {}
    n_guide_10 = sens_inv.get("n_guides")
    n_gene_10 = sens_inv.get("n_genes")
    if sens:
        ladder_txt = (
            f"Reporting ladder: {lib_n} guides deposited; "
            f"{n_guide_run} guides at this sensitivity cutoff (MIN_CELLS={cutoff}); "
            f"{n_guide_frozen} guides at frozen MIN_CELLS={frozen}; "
            f"{n_gene_frozen} genes at frozen MIN_CELLS={frozen}. "
            f"Manuscript reports {n_guide_frozen} guides / {n_gene_frozen} genes; "
            f"MIN_CELLS={cutoff} guide/gene counts are sensitivity-only."
        )
    else:
        ladder_txt = (
            f"Reporting ladder: {lib_n} guides deposited; "
            f"{n_guide_10} guides / {n_gene_10} genes at inventory "
            f"MIN_CELLS={SENSITIVITY_INVENTORY_MIN_CELLS} (sensitivity-only); "
            f"{n_guide_frozen} guides at frozen MIN_CELLS={frozen}; "
            f"{n_gene_frozen} genes at frozen MIN_CELLS={frozen}. "
            f"Manuscript reports {n_guide_frozen} guides / {n_gene_frozen} genes."
        )

    n25_note = summary.get("inventory_n25_note") or ""
    if n25_note:
        n25_note = " " + n25_note

    myc = shared_meta.get("myc_exclusion_note")
    myc_txt = f" {myc}" if myc else ""

    paired = summary.get("paired_gene_vs_guide_median") or {}
    paired_txt = ""
    if paired.get("shared_basis") and paired.get("n_shared_genes"):
        n_sh = paired["n_shared_genes"]
        pn = paired.get("pseudo_guide_null") or {}
        null_txt = ""
        if pn:
            p_str = _fmt_p(
                pn.get("p_excess_over_null"),
                n_perm=pn.get("n_perm"),
                at_floor=bool(pn.get("p_at_resolution_floor")),
            )
            sf = pn.get("fraction_of_gap_attributable_to_size")
            sf_pct = (
                f"{100 * float(sf):.0f}%"
                if sf is not None and np.isfinite(sf)
                else "NA"
            )
            null_txt = (
                f" Pseudo-guide null (footnote): observed median Δ="
                f"{_fmt_rho(pn.get('observed_median_delta_guide_minus_gene'))}, "
                f"size-matched null median Δ="
                f"{_fmt_rho(pn.get('null_median_delta_median'))} "
                f"(~{sf_pct} of gap attributable to group size); "
                f"residual Δ={_fmt_rho(pn.get('residual_median_delta'))} "
                f"(p_excess={p_str}, n_perm={pn.get('n_perm')}; "
                f"residual = guide-correlated structure, not named efficiency). "
                f"Most of the guide>gene difference is size artifact; "
                f"do not headline the raw median contrast."
            )
        paired_txt = (
            f" Paired gene-vs-guide contrast on the shared embedding "
            f"(n={n_sh} shared genes; not manuscript n={gene_n}) is a "
            f"size-confound diagnostic only (claim_allowed=false).{null_txt}"
        )

    # Item 16 predictive — Papalexi secondary; Replogle must lead in text
    sh = summary.get("between_guide_shift_cosine") or {}
    sh_txt = ""
    if sh.get("spearman_gene_sp_vs_median_cosine") is not None and np.isfinite(
        sh.get("spearman_gene_sp_vs_median_cosine", np.nan)
    ):
        part = sh.get("partial_spearman_sp_vs_cosine_given_magnitude") or {}
        part_ci = sh.get("partial_spearman_sp_vs_cosine_given_magnitude_boot") or {}
        att = sh.get("one_pair_per_gene_attenuation") or {}
        att_txt = ""
        if att.get("spearman_median") is not None and np.isfinite(
            att.get("spearman_median", np.nan)
        ):
            att_txt = (
                f" One-pair/gene attenuation (Replogle-like noise): Spearman "
                f"median={_fmt_rho(att.get('spearman_median'))}"
                f"{_fmt_ci({'ci_low': att.get('spearman_ci_low'), 'ci_high': att.get('spearman_ci_high')})} "
                f"partial|mag median={_fmt_rho(att.get('partial_median'))}."
            )
        sh_txt = (
            f" Item 16 predictive endpoint on Papalexi (secondary to Replogle): "
            f"Spearman(gene Sp, median pairwise cosine of guide mean shifts)="
            f"{_fmt_rho(sh.get('spearman_gene_sp_vs_median_cosine'))}"
            f"{_fmt_ci(sh.get('spearman_gene_sp_vs_median_cosine_boot'))} "
            f"(n_genes={sh.get('n_genes')}, n_pairs={sh.get('n_pairs')}, "
            f"mean pairs/gene={_fmt_rho(sh.get('mean_pairs_per_gene'))}; "
            f"partial|magnitude={_fmt_rho(part.get('rho_partial'))}"
            f"{_fmt_ci(part_ci)}; method={part.get('method')}). "
            f"Do not lead manuscript text with this rho — Replogle is flagship."
            f"{att_txt}"
        )

    bg = summary.get("between_guide_sp_icc") or {}
    bg_txt = ""
    if bg.get("icc") is not None and np.isfinite(bg.get("icc", np.nan)):
        mag_bit = ""
        if bg.get("icc_magnitude") is not None and np.isfinite(
            bg.get("icc_magnitude", np.nan)
        ):
            d_ci = _fmt_ci(
                {
                    "ci_low": bg.get("icc_sp_minus_icc_magnitude_ci_low"),
                    "ci_high": bg.get("icc_sp_minus_icc_magnitude_ci_high"),
                }
            )
            mag_bit = (
                f" ICC(magnitude)={_fmt_rho(bg.get('icc_magnitude'))}"
                f"{_fmt_ci({'ci_low': bg.get('icc_magnitude_ci_low'), 'ci_high': bg.get('icc_magnitude_ci_high')})}; "
                f"paired-bootstrap Δ ICC_Sp−ICC_mag="
                f"{_fmt_rho(bg.get('icc_sp_minus_icc_magnitude'))}{d_ci}. "
                f"Sp and magnitude between-guide agreement are indistinguishable; "
                f"this does not show Sp captures something magnitude misses."
            )
        bg_txt = (
            f" Between-guide Sp metric concordance (secondary; "
            f"{bg.get('replication_label', 'independent-reagent, shared-control replication')}): "
            f"ICC(1)={_fmt_rho(bg.get('icc'))}"
            f"{_fmt_ci(bg, 'icc_ci_low', 'icc_ci_high')} "
            f"(n_genes={bg.get('n_genes_multi_guide')}, n_guides={bg.get('n_guides')}) "
            f"— Sp as a stable property of gene×context across reagents, not "
            f"the predictive replication endpoint above."
            f"{mag_bit}"
        )

    prefix = "SENSITIVITY ANALYSIS — not for main tables. " if sens else ""
    return (
        f"{prefix}"
        f"Papalexi 2021 ECCITE-seq was reprocessed from GEO {GEO_ACCESSION} "
        f"(GSM4633614 RNA counts + GSE153056_ECCITE_metadata), without pertpy. "
        f"Author singlet assignments yield {summary.get('n_cells')} cells. "
        f"{settings}, "
        f"gene-level scoring (gene / gene_target, control NT) gives n={gene_n} "
        f"({parity}); guide-level scoring (guide_ID with NTg* collapsed to NT) "
        f"gives n={guide_n}.{basis_txt} {ladder_txt}{n25_note}{myc_txt}"
        f"{paired_txt}{sh_txt}{bg_txt} {rho_txt}"
    ).rstrip()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--level",
        choices=["gene", "guide", "both"],
        default="both",
        help="Which label grain to score (default: both)",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=cfg.MIN_CELLS,
        help=(
            f"Min cells per perturbation (default: frozen {cfg.MIN_CELLS}). "
            f"Any other value requires --sensitivity."
        ),
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help=(
            "Required if --min-cells != frozen MIN_CELLS. Writes "
            "papalexi_geo_*_sensitivity_mincellsN.* and never overwrites main tables."
        ),
    )
    parser.add_argument(
        "--geo-dir",
        type=Path,
        default=None,
        help="Cache directory for GEO downloads",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=None,
        help="Reuse / write path for built AnnData (default: geo-dir/papalexi_2021_geo.h5ad)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs / JSON",
    )
    parser.add_argument(
        "--prefer-tar",
        action="store_true",
        help="Download GSE153056_RAW.tar instead of individual sample files",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Download + build h5ad + label inventory; skip Sp scoring",
    )
    parser.add_argument(
        "--compare-pertpy",
        action="store_true",
        help="Correlate gene-level GEO Sp with frozen_sp_scores / results CSV",
    )
    parser.add_argument(
        "--frozen-csv",
        type=Path,
        default=None,
        help="Explicit frozen Sp CSV for --compare-pertpy",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require GEO files already present in --geo-dir",
    )
    parser.add_argument(
        "--no-copy-h5ad",
        action="store_true",
        help="Skip archiving h5ad into --out-dir (default archives ~0.8 GB copy)",
    )
    parser.add_argument(
        "--n-pseudo-perm",
        type=int,
        default=2000,
        help=(
            "Pseudo-guide null permutations for paired gene-vs-guide "
            "(default: 2000, matched to Cap bootstrap convention)"
        ),
    )
    args = parser.parse_args()

    is_sensitivity = args.min_cells != cfg.MIN_CELLS
    if is_sensitivity and not args.sensitivity:
        raise SystemExit(
            f"--min-cells={args.min_cells} differs from frozen MIN_CELLS={cfg.MIN_CELLS}. "
            "Pass --sensitivity to write separately labeled outputs. "
            "Do not mix non-50 cutoffs into main tables (version drift)."
        )
    if args.sensitivity and not is_sensitivity:
        print(
            f"  NOTE: --sensitivity with min_cells={cfg.MIN_CELLS} "
            "(same as frozen); treating as main run.",
            flush=True,
        )
        is_sensitivity = False

    out_dir = resolve_out_dir(args.out_dir)
    # Durable cache under out_dir by default (Colab /tmp is wiped).
    geo_dir = geo_cache_dir(args.geo_dir, out_dir=out_dir)
    preferred_h5ad = args.h5ad or (geo_dir / "papalexi_2021_geo.h5ad")
    found_h5ad = find_existing_h5ad(preferred_h5ad, out_dir, explicit=args.h5ad)
    h5ad_path = found_h5ad or Path(preferred_h5ad)
    if found_h5ad is not None and found_h5ad.resolve() != Path(preferred_h5ad).resolve():
        print(f"  Found existing h5ad: {found_h5ad}", flush=True)

    # Sensitivity runs get a filename tag so they cannot overwrite main CSVs.
    file_tag = f"_sensitivity_mincells{args.min_cells}" if is_sensitivity else ""

    print(f"GEO {GEO_ACCESSION} → {DATASET}", flush=True)
    print(f"  geo_dir={geo_dir}", flush=True)
    print(f"  out_dir={out_dir}", flush=True)
    print(
        f"  config={cfg.CONFIG_VERSION}  min_cells={args.min_cells}  "
        f"n_pcs={cfg.N_PCS}  sensitivity={is_sensitivity}",
        flush=True,
    )
    if is_sensitivity:
        print(
            "  *** SENSITIVITY RUN — outputs tagged; frozen MIN_CELLS=50 "
            "remains the manuscript number ***",
            flush=True,
        )

    import scanpy as sc

    setup_cache()
    sc.settings.seed = cfg.SEED

    # --- acquire / build ---
    # Prefer h5ad if present (no GEO raw files needed).
    if h5ad_path.exists():
        print(f"  Loading existing {h5ad_path}", flush=True)
        adata = sc.read_h5ad(h5ad_path)
        if "guide_collapsed" not in adata.obs.columns and "guide_ID" in adata.obs.columns:
            adata.obs["guide_collapsed"] = collapse_nt_guides(adata.obs["guide_ID"])
        if "gene_target" not in adata.obs.columns and "gene" in adata.obs.columns:
            adata.obs["gene_target"] = adata.obs["gene"].astype(str)
        gdo = None
        # optional GDO library for inventory "library" rows
        for d in [geo_dir, *_legacy_geo_dirs(out_dir)]:
            gdo_path = d / GEO_FILES["gdo_barcodes"]["fname"]
            if gdo_path.exists():
                gdo = load_gdo_library(gdo_path)
                break
    else:
        paths = resolve_geo_paths(
            geo_dir,
            out_dir,
            skip_download=args.skip_download,
            prefer_tar=args.prefer_tar,
        )
        gdo = load_gdo_library(paths["gdo_barcodes"])
        # Always write new builds into the durable geo_dir (not a legacy /tmp path)
        build_h5ad = geo_dir / "papalexi_2021_geo.h5ad"
        print(f"  Building AnnData from GEO files → {build_h5ad}", flush=True)
        adata = build_adata(paths, build_h5ad)
        h5ad_path = build_h5ad

    if not args.no_copy_h5ad:
        try:
            h5ad_path = ensure_archived_h5ad(h5ad_path, out_dir)
        except OSError as e:
            print(f"  (could not archive h5ad: {e})", flush=True)

    # Inventory at the run cutoff, frozen 50, and sensitivity 10 (always recorded).
    inv = label_inventory(adata, gdo, args.min_cells)
    inv_path = out_dir / f"papalexi_geo_label_counts{file_tag}.csv"
    inv.to_csv(inv_path, index=False)

    gdo_n = int(adata.uns.get("gdo_library_n") or (len(gdo) if gdo is not None else 0))
    n_gene_run = _n_pass(inv, "gene", args.min_cells)
    n_guide_run = _n_pass(inv, "guide", args.min_cells)
    n_gene_frozen = _n_pass(inv, "gene", cfg.MIN_CELLS)
    n_guide_frozen = _n_pass(inv, "guide", cfg.MIN_CELLS)
    n_gene_sens10 = _n_pass(inv, "gene", SENSITIVITY_INVENTORY_MIN_CELLS)
    n_guide_sens10 = _n_pass(inv, "guide", SENSITIVITY_INVENTORY_MIN_CELLS)
    print(
        f"  Cells={adata.n_obs}; GDO library={gdo_n} guides; "
        f"gene≥{args.min_cells}: {n_gene_run}; guide≥{args.min_cells}: {n_guide_run}; "
        f"frozen gene≥{cfg.MIN_CELLS}: {n_gene_frozen}; "
        f"frozen guide≥{cfg.MIN_CELLS}: {n_guide_frozen}; "
        f"sens10 gene≥{SENSITIVITY_INVENTORY_MIN_CELLS}: {n_gene_sens10}; "
        f"sens10 guide≥{SENSITIVITY_INVENTORY_MIN_CELLS}: {n_guide_sens10}",
        flush=True,
    )

    reporting_ladder = {
        "n_guides_deposited": gdo_n,
        "n_guides_this_cutoff": n_guide_run,
        "n_guides_frozen_min_cells": n_guide_frozen,
        "n_genes_this_cutoff": n_gene_run,
        "n_genes_frozen_min_cells": n_gene_frozen,
        "this_cutoff": args.min_cells,
        "frozen_min_cells": cfg.MIN_CELLS,
        "manuscript_reports": {
            "n_guides": n_guide_frozen,
            "n_genes": n_gene_frozen,
            "min_cells": cfg.MIN_CELLS,
        },
        # Always populated from inventory (no Sp re-run required).
        "sensitivity_only": {
            "min_cells": SENSITIVITY_INVENTORY_MIN_CELLS,
            "n_guides": n_guide_sens10,
            "n_genes": n_gene_sens10,
            "kind": "inventory",
            "note": (
                "Label counts at MIN_CELLS=10 for the n=25 diagnostic. "
                "Not a manuscript number; Sp at this cutoff is sensitivity-only."
            ),
        },
    }

    # Always record when inventory@10 reproduces n=25 (even on frozen@50 runs).
    inventory_n25_note = None
    if n_gene_sens10 == 25:
        inventory_n25_note = (
            "Gene-level inventory n=25 at "
            f"MIN_CELLS={SENSITIVITY_INVENTORY_MIN_CELLS} matches a prior "
            "Papalexi count; frozen "
            f"MIN_CELLS={cfg.MIN_CELLS} yields n={n_gene_frozen}. "
            "The pre-freeze n=25 vs post-freeze n=24 discrepancy is explained by "
            "that cutoff change (same shape as the Dixit 10-vs-50 finding)."
        )

    summary: dict = {
        "dataset": DATASET,
        "geo_accession": GEO_ACCESSION,
        "source": "GEO (no pertpy)",
        "config_version": cfg.CONFIG_VERSION,
        "frozen_min_cells": cfg.MIN_CELLS,
        "min_cells": args.min_cells,
        "is_sensitivity": is_sensitivity,
        "n_pcs": cfg.N_PCS,
        "seed": cfg.SEED,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "gdo_library_n": gdo_n,
        # Always present (inventory), even if a level was not scored this run.
        "n_gene_labels_pass_min_cells": n_gene_run,
        "n_guide_labels_pass_min_cells": n_guide_run,
        "n_gene_labels_pass_frozen_min_cells": n_gene_frozen,
        "n_guide_labels_pass_frozen_min_cells": n_guide_frozen,
        "reporting_ladder": reporting_ladder,
        "h5ad": str(h5ad_path),
        "h5ad_archived": str(out_dir / "papalexi_2021_geo.h5ad"),
        "inventory_n25_note": inventory_n25_note,
        "note": (
            "Honest ladder: deposited GDO library (~112) → guides at "
            f"MIN_CELLS={SENSITIVITY_INVENTORY_MIN_CELLS} (sensitivity inventory) "
            "→ guides at frozen MIN_CELLS=50 (manuscript) → genes at frozen "
            "MIN_CELLS=50. Manuscript gene Sp is gene-grain embedding only. "
            "Shared-basis gene Sp is paired-contrast only (often drops MYC). "
            "Paired guide>gene is a size-confound diagnostic: claim_allowed "
            "stays false; residual over null = guide-correlated structure. "
            "Item 16 predictive = gene Sp → between-guide mean-shift cosine "
            "(Papalexi secondary; Replogle flagship). ICC(Sp) is metric "
            "concordance only; Δ vs ICC(magnitude) usually ≈0."
        ),
    }

    summary_json = out_dir / f"papalexi_geo_summary{file_tag}.json"
    summary_blurb = out_dir / f"papalexi_geo_methods_blurb{file_tag}.txt"

    if args.build_only:
        # Inventory-only: no Sp scored, but n's come from reporting_ladder.
        summary["methods_blurb"] = methods_blurb(summary)
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        with open(summary_blurb, "w") as f:
            f.write(summary["methods_blurb"] + "\n")
        print(summary["methods_blurb"])
        print(f"Wrote {summary_json} (build-only)")
        return

    # Cap bootstrap reps for Colab; full N_BOOTSTRAP is fine but slower.
    n_boot = min(2000, cfg.N_BOOTSTRAP)
    gene_manuscript_df: pd.DataFrame | None = None
    gene_shared_df: pd.DataFrame | None = None
    guide_df: pd.DataFrame | None = None
    ad_shared = None

    if args.level in ("both", "gene"):
        # Manuscript / parity gene Sp: own gene-grain HVG/PCA (n=24 at frozen 50).
        gene_manuscript_df = score_level(
            adata,
            pert_col="gene_target",
            ctrl_label="NT",
            level="gene",
            min_cells=args.min_cells,
            sc=sc,
        )
        gene_csv = out_dir / f"papalexi_geo_sp_gene{file_tag}.csv"
        gene_manuscript_df.to_csv(gene_csv, index=False)
        print(f"  Wrote {gene_csv} (manuscript gene-grain)", flush=True)
        summary["gene"] = _summarize_sp_df(
            gene_manuscript_df,
            level="gene",
            out_csv=gene_csv,
            shared_basis=False,
            n_boot=n_boot,
            role="manuscript_gene_grain",
        )

    if args.level == "both":
        gene_shared_df, guide_df, ad_shared, shared_meta = score_shared_basis(
            adata, min_cells=args.min_cells, sc=sc
        )
        summary["shared_basis_meta"] = shared_meta

        shared_csv = out_dir / f"papalexi_geo_sp_gene_shared_basis{file_tag}.csv"
        gene_shared_df.to_csv(shared_csv, index=False)
        print(
            f"  Wrote {shared_csv} (paired-contrast only; not headline)",
            flush=True,
        )
        summary["gene_shared_basis"] = _summarize_sp_df(
            gene_shared_df,
            level="gene",
            out_csv=shared_csv,
            shared_basis=True,
            n_boot=n_boot,
            role="paired_contrast_only",
        )

        guide_csv = out_dir / f"papalexi_geo_sp_guide{file_tag}.csv"
        guide_df.to_csv(guide_csv, index=False)
        print(f"  Wrote {guide_csv}", flush=True)
        summary["guide"] = _summarize_sp_df(
            guide_df,
            level="guide",
            out_csv=guide_csv,
            shared_basis=True,
            n_boot=n_boot,
            role="shared_basis_guide",
        )

        print(
            f"\n>>> Pseudo-guide null ({args.n_pseudo_perm} perms)…",
            flush=True,
        )
        pn = pseudo_guide_null(
            ad_shared,
            guide_df,
            gene_shared_df,
            n_perm=args.n_pseudo_perm,
            seed=cfg.SEED,
        )
        cross, paired = paired_gene_vs_guide_median(
            gene_shared_df,
            guide_df,
            shared_basis=True,
            pseudo_null=pn,
        )
        summary["paired_gene_vs_guide_median"] = paired
        cross.to_csv(
            out_dir / f"papalexi_geo_gene_vs_guide{file_tag}.csv", index=False
        )
        print(
            f"  Paired gene vs guide-median Sp (shared_basis=True; "
            f"claim_allowed={paired.get('claim_allowed')}): "
            f"n_shared={paired.get('n_shared_genes')}  "
            f"obs Δ={pn.get('observed_median_delta_guide_minus_gene')}  "
            f"null Δ={pn.get('null_median_delta_median')}  "
            f"size_fraction={pn.get('fraction_of_gap_attributable_to_size')}  "
            f"residual Δ={pn.get('residual_median_delta')}  "
            f"p_excess={_fmt_p(pn.get('p_excess_over_null'), n_perm=pn.get('n_perm'), at_floor=bool(pn.get('p_at_resolution_floor')))}  "
            f"(n_perm={pn.get('n_perm')})",
            flush=True,
        )

        # Item 16 predictive: gene Sp → between-guide mean-shift cosine
        pairs_sh, genes_sh, sh_sum = between_guide_shift_cosine(
            ad_shared,
            gene_shared_df,
            guide_df,
            n_bootstrap=n_boot,
            seed=cfg.SEED,
        )
        summary["between_guide_shift_cosine"] = sh_sum
        if len(pairs_sh):
            pairs_sh.to_csv(
                out_dir / f"papalexi_geo_between_guide_shift_pairs{file_tag}.csv",
                index=False,
            )
        if len(genes_sh):
            genes_sh.to_csv(
                out_dir / f"papalexi_geo_between_guide_shift_genes{file_tag}.csv",
                index=False,
            )

    elif args.level == "guide":
        guide_df = score_level(
            adata,
            pert_col="guide_collapsed",
            ctrl_label="NT",
            level="guide",
            min_cells=args.min_cells,
            sc=sc,
        )
        guide_csv = out_dir / f"papalexi_geo_sp_guide{file_tag}.csv"
        guide_df.to_csv(guide_csv, index=False)
        print(f"  Wrote {guide_csv}", flush=True)
        summary["guide"] = _summarize_sp_df(
            guide_df,
            level="guide",
            out_csv=guide_csv,
            shared_basis=False,
            n_boot=n_boot,
            role="guide_grain_alone",
        )
        summary["shared_basis_meta"] = {
            "shared_basis": False,
            "note": "Guide-only run; no paired gene-vs-guide claim.",
        }
    else:
        # gene-only: manuscript block already written above
        summary["shared_basis_meta"] = {
            "shared_basis": False,
            "note": "Gene-only run; no paired gene-vs-guide claim.",
        }

    if guide_df is not None and len(guide_df):
        detail, bg = between_guide_sp_icc(
            guide_df, n_bootstrap=n_boot, seed=cfg.SEED
        )
        summary["between_guide_sp_icc"] = bg
        if len(detail):
            detail.to_csv(
                out_dir / f"papalexi_geo_between_guide_sp{file_tag}.csv",
                index=False,
            )
        print(
            f"  Between-guide Sp ICC (item 16): "
            f"n_genes={bg.get('n_genes_multi_guide')}  "
            f"n_guides={bg.get('n_guides')}  "
            f"ICC(1)={bg.get('icc')}  "
            f"95% CI [{bg.get('icc_ci_low')}, {bg.get('icc_ci_high')}]  "
            f"ICC(mag)={bg.get('icc_magnitude')}  "
            f"95% CI [{bg.get('icc_magnitude_ci_low')}, {bg.get('icc_magnitude_ci_high')}]  "
            f"paired Δ={bg.get('icc_sp_minus_icc_magnitude')}  "
            f"95% CI [{bg.get('icc_sp_minus_icc_magnitude_ci_low')}, "
            f"{bg.get('icc_sp_minus_icc_magnitude_ci_high')}]  "
            f"({bg.get('replication_label')})",
            flush=True,
        )

    if args.compare_pertpy and gene_manuscript_df is not None:
        summary["pertpy_comparison"] = compare_pertpy(
            gene_manuscript_df, out_dir, args.frozen_csv
        )
        pc = summary["pertpy_comparison"]
        if pc.get("available"):
            print(
                f"  GEO vs frozen gene Sp: n_shared={pc['n_shared']}  "
                f"rho={pc.get('spearman_sp')}",
                flush=True,
            )
        else:
            print(f"  pertpy comparison skipped: {pc.get('error')}", flush=True)

    summary["methods_blurb"] = methods_blurb(summary)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_blurb, "w") as f:
        f.write(summary["methods_blurb"] + "\n")

    print()
    print(summary["methods_blurb"])
    print(f"\nWrote {summary_json}")
    # Footer never says n/a — use scored n if present, else inventory at this cutoff.
    gene_footer = (
        summary.get("gene", {}).get("n_scored")
        if summary.get("gene")
        else n_gene_run
    )
    if gene_footer is None:
        gene_footer = n_gene_run
    guide_footer = (
        summary.get("guide", {}).get("n_scored")
        if summary.get("guide")
        else n_guide_run
    )
    if guide_footer is None:
        guide_footer = n_guide_run
    gci = (summary.get("guide") or {}).get("spearman_mag_sp_gene_clustered") or {}
    n_shared_gene = (summary.get("gene_shared_basis") or {}).get("n_scored")
    claim = (summary.get("paired_gene_vs_guide_median") or {}).get("claim_allowed")
    print(
        f"Ladder: deposited={gdo_n}; "
        f"guides@min_cells={args.min_cells}: {guide_footer}; "
        f"guides@frozen={cfg.MIN_CELLS}: {n_guide_frozen}; "
        f"genes@frozen={cfg.MIN_CELLS}: {n_gene_frozen} "
        f"(manuscript gene-grain n={gene_footer}"
        + (
            f"; shared-basis gene n={n_shared_gene}"
            if n_shared_gene is not None
            else ""
        )
        + (
            f"; guide gene-clustered CI n_genes={gci.get('n_clusters')}"
            if gci
            else ""
        )
        + (
            f"; paired claim_allowed={claim}"
            if claim is not None
            else ""
        )
        + ")"
    )
    if is_sensitivity:
        print(
            f"*** Manuscript number remains guides={n_guide_frozen} / "
            f"genes={n_gene_frozen} at MIN_CELLS={cfg.MIN_CELLS} ***"
        )


if __name__ == "__main__":
    main()
