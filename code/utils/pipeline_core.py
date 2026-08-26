"""
Shared load / preprocess / Sp scoring for the frozen pipeline.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import hashlib
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import pipeline_config as cfg


def setup_cache(cache_dir: Optional[Path] = None) -> Path:
    """Writable pertpy/scanpy cache; prefer /tmp then ~/.cache."""
    cache = Path(cache_dir or cfg.CACHE_DIR)
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache = Path.home() / ".cache" / "pertpy_data"
        cache.mkdir(parents=True, exist_ok=True)
    os.environ["SCVERSE_DATADIR"] = str(cache)
    os.environ["PERTPY_CACHE_DIR"] = str(cache)
    return cache


def import_pertpy_datasets(cache_dir: Optional[Path] = None):
    """
    Load pertpy.data._datasets without importing full pertpy (avoids JAX).
    Returns the datasets module.
    """
    cache = setup_cache(cache_dir)

    for mod in list(sys.modules):
        if mod == "pertpy" or mod.startswith("pertpy."):
            del sys.modules[mod]

    spec = importlib.util.find_spec("pertpy")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("pertpy is not installed. Run: pip install pertpy==1.0.6")

    path = spec.submodule_search_locations[0]
    pkg = types.ModuleType("pertpy")
    pkg.__path__ = [path]
    pkg.__spec__ = spec
    sys.modules["pertpy"] = pkg

    import scanpy as sc

    sc.settings.datasetdir = cache
    ds = importlib.import_module("pertpy.data._datasets")
    ds.settings.datasetdir = cache
    return ds, sc


def get_loader(ds, loader_name: str) -> Callable:
    if not hasattr(ds, loader_name):
        raise AttributeError(
            f"pertpy.data has no loader '{loader_name}'. "
            f"Available adamson*: {[n for n in dir(ds) if 'adamson' in n.lower()]}"
        )
    return getattr(ds, loader_name)


def _download_urls_for(meta: dict) -> list[str]:
    urls = list(meta.get("download_urls") or [])
    if meta.get("download_url"):
        urls.append(meta["download_url"])
    # de-dupe, preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _file_size_ok(path: Path, meta: dict, content_length: int | None = None) -> tuple[bool, str]:
    """Return (ok, reason). Truncated caches must fail (Replogle 72 MB vs 1546 MB)."""
    size = path.stat().st_size
    expected = meta.get("expected_bytes")
    min_bytes = meta.get("min_bytes")
    if min_bytes is None:
        # .partial downloads keep suffix ".h5mu.partial" — suffix alone is wrong.
        name = path.name.lower()
        is_h5mu = name.endswith(".h5mu") or ".h5mu." in name
        min_bytes = 100_000 if is_h5mu else 1_000_000
    if size < min_bytes:
        return False, f"too small ({size / 1e6:.1f} MB < min {min_bytes / 1e6:.1f} MB)"
    if expected is not None:
        # Allow 1% slack for CDN variance; reject clear truncations
        if size < int(0.99 * expected):
            return (
                False,
                f"truncated/incomplete ({size / 1e6:.1f} MB; expected ≥ "
                f"{0.99 * expected / 1e6:.1f} MB / ~{expected / 1e6:.1f} MB)",
            )
    if content_length is not None and content_length > 0:
        if size < int(0.99 * content_length):
            return (
                False,
                f"incomplete vs Content-Length ({size} < {content_length})",
            )
    return True, f"{size / 1e6:.1f} MB"


def ensure_local_h5ad(dataset_name: str, cache_dir: Optional[Path] = None) -> Path:
    """Download dataset h5ad into cache; rejects truncated files and re-downloads."""
    import urllib.error
    import urllib.request

    meta = cfg.DATASETS[dataset_name]
    cache = setup_cache(cache_dir)
    fname = meta.get("local_h5ad")
    if not fname:
        raise KeyError(f"No local_h5ad configured for {dataset_name}")
    path = cache / fname
    if path.exists() and path.stat().st_size > 0:
        ok, reason = _file_size_ok(path, meta)
        if ok:
            print(f"    cache hit {path.name} ({reason})", flush=True)
            return path
        print(
            f"    WARNING: cached {path.name} failed size check ({reason}); "
            "deleting and re-downloading",
            flush=True,
        )
        path.unlink(missing_ok=True)

    urls = _download_urls_for(meta)
    if not urls:
        raise FileNotFoundError(f"Missing {path} and no download_urls in config")

    last_err: Optional[Exception] = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; shesha-crispr-pipeline/1.0; "
            "+https://github.com/)"
        )
    }
    for url in urls:
        tmp = path.with_suffix(path.suffix + ".partial")
        try:
            print(f"    Downloading {fname}\n      from {url}", flush=True)
            req = urllib.request.Request(url, headers=headers)
            content_length = None
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
                total = resp.headers.get("Content-Length")
                if total and str(total).isdigit():
                    content_length = int(total)
                total_mb = (content_length / 1e6) if content_length else None
                done = 0
                last_report = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    # progress every ~25 MB so Colab doesn't look hung
                    if done - last_report >= 25 * 1024 * 1024:
                        last_report = done
                        if total_mb:
                            print(
                                f"      … {done / 1e6:.0f} / {total_mb:.0f} MB",
                                flush=True,
                            )
                        else:
                            print(f"      … {done / 1e6:.0f} MB", flush=True)
            ok, reason = _file_size_ok(tmp, meta, content_length=content_length)
            if not ok:
                raise IOError(f"Download size check failed ({reason}): {url}")
            if content_length is not None and done != content_length:
                # Exact match preferred when server sent Content-Length
                if done < content_length:
                    raise IOError(
                        f"Incomplete download: got {done} of {content_length} bytes"
                    )
            tmp.replace(path)
            print(f"    Saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            return path
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            print(f"    download failed: {e}", flush=True)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    raise FileNotFoundError(
        f"Could not download {fname}. Last error: {last_err}. "
        f"Place {fname} in the cache dir or pass --h5ad /path/to/{fname}."
    )


def compute_sp_digest(df: pd.DataFrame) -> str:
    """Stable digest of Sp table identity (dataset, pert, stability, magnitude)."""
    need = ["dataset", "perturbation", "stability", "magnitude"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Cannot digest Sp table: missing column {c!r}")
    sub = (
        df[need]
        .assign(
            dataset=lambda d: d["dataset"].astype(str),
            perturbation=lambda d: d["perturbation"].astype(str),
            stability=lambda d: d["stability"].astype(float).round(12),
            magnitude=lambda d: d["magnitude"].astype(float).round(12),
        )
        .sort_values(["dataset", "perturbation"], kind="mergesort")
        .reset_index(drop=True)
    )
    lines = [
        f"{r.dataset}\t{r.perturbation}\t{r.stability:.12g}\t{r.magnitude:.12g}"
        for r in sub.itertuples(index=False)
    ]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def assert_frozen_sp_compatible(
    frozen_csv: Path | str,
    *,
    expect_n_rows: int | None = None,
    require_digest: bool = True,
) -> dict:
    """
    Refuse to join a stale / wrong-version / tampered frozen Sp table.

    ABORTS (raises) — never warns and continues. The 2026-07-25.1 Drive copy
    joined against 2026-07-29.1 pathway scores produced a false INCONCLUSIVE;
    require_digest=True is what catches Sp values rewritten under a matching
    config_version stamp.

    Pass require_digest=False only for pre-digest legacy probes.
    """
    path = Path(frozen_csv)
    if not path.exists():
        raise FileNotFoundError(f"frozen Sp CSV not found: {path}")
    df = pd.read_csv(path)
    if "config_version" not in df.columns:
        raise ValueError(
            f"{path} has no config_version column — refusing join. "
            f"Need {cfg.CONFIG_VERSION} from run_frozen_main.py "
            "(Drive may still hold the 2026-07-25.1 pre-hash file)."
        )
    raw_versions = sorted({str(v) for v in df["config_version"].dropna().unique()})
    versions = sorted({cfg.resolve_config_version(v) for v in raw_versions})
    if versions != [cfg.CONFIG_VERSION]:
        raise ValueError(
            f"{path} config_version={raw_versions} "
            f"(resolved={versions}) ≠ {cfg.CONFIG_VERSION!r}. "
            "Copy the regenerated frozen_sp_scores.csv to the output dir; "
            "do not join pathway scores to a stale Sp table."
        )
    if raw_versions != versions:
        print(
            f"  note: remapped config_version {raw_versions} → {versions} "
            f"(August→July calendar rename; same freeze)",
            flush=True,
        )
    n = len(df)
    expect_n = (
        expect_n_rows
        if expect_n_rows is not None
        else getattr(cfg, "FROZEN_SP_EXPECTED_N_ROWS", None)
    )
    if expect_n is not None and n != int(expect_n):
        raise ValueError(
            f"{path} has n={n} rows; expected {expect_n} for full six-dataset "
            f"freeze under {cfg.CONFIG_VERSION}. Partial/wrong file."
        )
    required = list(getattr(cfg, "PATHWAY_REQUIRED_DATASETS", []))
    # Pilot is in the full freeze but not required for pathway gate
    present = set(df["dataset"].astype(str).unique())
    # resolve legacy names
    present_res = {cfg.resolve_dataset_name(x) for x in present}
    missing = [d for d in required if d not in present_res and d not in present]
    if missing:
        raise ValueError(
            f"{path} missing required datasets {missing}. "
            "A truncated Replogle download previously caused this."
        )
    digest = compute_sp_digest(df)
    if "sp_digest" not in df.columns:
        if require_digest:
            raise ValueError(
                f"{path} missing sp_digest column — refusing join. "
                "Re-run run_frozen_main.py under the current freeze "
                f"({cfg.CONFIG_VERSION}) so the digest stamp is written."
            )
    else:
        stored = {str(x) for x in df["sp_digest"].dropna().unique()}
        if not stored:
            raise ValueError(f"{path} has empty sp_digest column — refusing join.")
        if stored != {digest}:
            raise ValueError(
                f"{path} sp_digest MISMATCH — refusing join. "
                f"stored={stored} recomputed={digest!r}. "
                "Sp/magnitude values do not match the stamp (stale or tampered)."
            )
    info = {
        "path": str(path),
        "config_version": cfg.CONFIG_VERSION,
        "n_rows": n,
        "sp_digest": digest,
        "datasets": sorted(present_res),
    }
    print(
        f"frozen Sp OK: {path.name}  config={cfg.CONFIG_VERSION}  "
        f"n={n}  digest={digest}",
        flush=True,
    )
    return info


# Files larger than this open in backed mode (X stays on disk until filtered)
BACKED_READ_MB = 200


def _needs_to_memory(adata) -> bool:
    """True if .copy() would fail / X is still on disk."""
    if getattr(adata, "isbacked", False):
        return True
    # Views into a backed file sometimes report isbacked=False but still can't .copy()
    filename = getattr(adata, "filename", None)
    if filename is not None:
        return True
    return False


def ensure_in_memory(adata):
    """Return an in-memory AnnData; never call .copy() on backed objects."""
    if _needs_to_memory(adata):
        print("    to_memory()…", flush=True)
        mem = adata.to_memory()
        # Materialize X before closing the backed handle. Some anndata builds
        # leave mem.X=None (or still file-backed) if the parent file is closed
        # too early — Norman Colab failure after scanpy/anndata reinstalls.
        try:
            if getattr(mem, "X", None) is None:
                source_x = getattr(adata, "X", None)
                # Backed AnnData views can report view.X=None even though the
                # parent file has X. Read the selected rows from the still-open
                # parent before closing its file handle.
                if source_x is None:
                    parent = getattr(adata, "_adata_ref", None)
                    oidx = getattr(adata, "_oidx", None)
                    parent_x = getattr(parent, "X", None) if parent is not None else None
                    if parent_x is not None and oidx is not None:
                        parent_rows = np.arange(parent.n_obs)[oidx]
                        source_x = parent_x[parent_rows]
                if source_x is not None:
                    mem.X = source_x.copy() if hasattr(source_x, "copy") else source_x
            elif getattr(mem, "X", None) is not None and hasattr(mem.X, "copy"):
                # Touch / detach from any remaining file-backed buffer
                _ = mem.X.shape
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: could not detach X after to_memory ({exc})", flush=True)
        for obj in (adata, getattr(adata, "_adata", None), getattr(adata, "_adata_ref", None)):
            if obj is None:
                continue
            try:
                if hasattr(obj, "file") and obj.file is not None:
                    obj.file.close()
            except Exception:
                pass
        return mem
    try:
        return adata.copy()
    except ValueError as e:
        if "backed" in str(e).lower() or "to_memory" in str(e).lower():
            print("    .copy() failed on backed view; using to_memory()…", flush=True)
            return adata.to_memory()
        raise


def load_raw(
    dataset_name: str,
    ds=None,
    sc=None,
    prefer_local: bool = True,
    h5ad_path: Optional[Path] = None,
):
    """Load raw AnnData/MuData via explicit path, local cache, or pertpy loader."""
    meta = cfg.DATASETS[dataset_name]
    if sc is None:
        import scanpy as sc

    def _read_cached(path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        size_mb = path.stat().st_size / 1e6
        print(f"    Opening {path.name} ({size_mb:.1f} MB)…", flush=True)
        if path.suffix == ".h5mu":
            import mudata as md

            mdata = md.read_h5mu(path)
            # pull_obs: need global gene_target on mdata.obs for Papalexi extract.
            # Do NOT pull_var: intersecting modality var_names emit
            # "Cannot join columns with the same name" and can resolve the
            # joined var table differently across mudata versions — a plausible
            # cause of Papalexi-only scGPT Sp~mag drift at fixed n=24. RNA
            # gene set for scoring/embed comes from mod["rna"] only.
            if hasattr(mdata, "pull_obs"):
                mdata.pull_obs()
            print(
                f"    MuData ready: mods={list(mdata.mod.keys())} "
                f"(pull_obs only; skip pull_var)",
                flush=True,
            )
            return mdata

        # Large files: backed mode so Colab doesn't thrash loading 1–2GB into RAM
        if size_mb >= BACKED_READ_MB:
            print(
                "    Using backed='r' (expression matrix stays on disk until cell filter).",
                flush=True,
            )
            adata = sc.read_h5ad(path, backed="r")
            print(
                f"    Opened backed: {adata.n_obs} cells × {adata.n_vars} genes",
                flush=True,
            )
            return adata

        print("    Reading into memory…", flush=True)
        adata = sc.read_h5ad(path)
        print(f"    Loaded: {adata.n_obs} cells × {adata.n_vars} genes", flush=True)
        return adata

    if h5ad_path is not None:
        return _read_cached(h5ad_path)

    # Prefer direct Figshare/Zenodo download (pertpy import often hangs on Colab)
    if prefer_local and meta.get("local_h5ad"):
        try:
            path = ensure_local_h5ad(dataset_name)
            return _read_cached(path)
        except FileNotFoundError as e:
            print(f"    local download failed ({e}); trying pertpy loader", flush=True)

    print("    Loading via pertpy (can be slow / hang if pertpy pulls JAX)…", flush=True)
    if ds is None:
        ds, sc = import_pertpy_datasets()
    loader = get_loader(ds, meta["loader"])
    return loader()


def _collapse_adamson_upr_labels(adata, pert_col: str = "perturbation"):
    """
    Map Adamson UPR scPerturb labels to gene-level conditions.
    Controls: Gal4* / * (mod)* → control. Guides: GENE_pDSxxx → GENE.
    Obs-only; safe for backed AnnData (no full X copy).
    """
    # Arrow/string or float-NaN columns: force plain python strings first
    labels = adata.obs[pert_col].astype("string").fillna("nan").astype(str)

    def map_label(x) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "DROP"
        s = str(x).strip()
        xl = s.lower()
        if xl in {"nan", "none", "<na>", "nat", "*", ""}:
            return "DROP"
        if "gal4" in xl or "(mod)" in xl:
            return "control"
        gene = s.split("_")[0]
        if gene.lower() in {"nan", "*", "", "<na>"}:
            return "DROP"
        return gene

    # Do NOT subset here when backed: AnnData forbids view-of-view, and
    # materialize_min_cells will index again. Mark DROP; materialize excludes it.
    adata.obs["condition"] = [map_label(x) for x in labels.to_numpy()]
    return adata, "condition", "control"


# Labels written by extractors that must never enter Sp scoring.
_EXCLUDED_PERT_LABELS = frozenset({"DROP", "POS_CONTROL", "nan", "NaN_Control"})


def _stable_sample_indices(
    idx: np.ndarray,
    obs_names: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    """
    Pick `cap` rows from `idx` by ranking blake2b(seed|obs_name).

    Independent of AnnData row order, so pertpy vs local-h5ad loaders that
    permute cells still yield the same downsample at a fixed seed.
    """
    import hashlib

    if len(idx) <= cap:
        return idx
    keys = []
    for i in idx:
        h = hashlib.blake2b(
            f"{seed}|{obs_names[i]}".encode("utf-8"), digest_size=8
        ).digest()
        keys.append((h, int(i)))
    keys.sort()
    return np.array([i for _, i in keys[:cap]], dtype=idx.dtype)


def materialize_min_cells(
    adata,
    pert_col: str,
    ctrl_label: str,
    min_cells: int = cfg.MIN_CELLS,
    max_cells_per_pert: int = cfg.MAX_CELLS_PER_PERT,
    max_control_cells: int = cfg.MAX_CONTROL_CELLS,
    seed: int = cfg.SEED,
    max_perts: Optional[int] = None,
):
    """
    Filter to control + perturbations with >= min_cells using obs only,
    downsample large groups, then load X into memory.

    Downsampling before to_memory() is what makes Replogle (~300k cells) runnable
    on Colab — Sp only needs ~50–100 cells per perturbation.

    max_perts: if set, randomly keep only this many perturbations (plus control).
    Use for Colab reference-selection / smoke runs — NOT for manuscript tables.
    Full Replogle is ~1832 × 100 + 5000 controls ≈ 175k cells and to_memory()
    routinely stalls or OOMs on Colab.

    Backed AnnData forbids view-of-view. If `adata` is already a view, resolve
    indices onto the parent and slice the parent once.
    """
    labels = adata.obs[pert_col].astype(str)
    counts = labels.value_counts()
    valid = [
        p
        for p in counts[counts >= min_cells].index
        if p != ctrl_label and p not in _EXCLUDED_PERT_LABELS
    ]
    # Stable order for valid list (value_counts order can depend on appearance)
    valid = sorted(valid)

    rng = np.random.default_rng(seed)
    if max_perts is not None and len(valid) > max_perts:
        # smoke path only — still sort chosen names for determinism
        chosen = list(rng.choice(valid, size=max_perts, replace=False))
        valid = sorted(chosen)
        print(
            f"    Subsampling to {max_perts} perturbations (Colab / smoke path; "
            "not manuscript-frozen)",
            flush=True,
        )

    label_arr = labels.to_numpy()
    # Order-invariant downsample: select by hash(seed|obs_name), not by
    # positional RNG on whatever row order the loader left. Positional choice
    # was the Dixit-class failure mode (pertpy vs local h5ad reordering →
    # different ≤100/pert cells at the same seed → moving pathway |mag).
    obs_names = adata.obs_names.astype(str).to_numpy()
    keep_idx = []
    for lab in list(valid) + [ctrl_label]:
        idx = np.flatnonzero(label_arr == lab)
        cap = max_control_cells if lab == ctrl_label else max_cells_per_pert
        if len(idx) > cap:
            idx = _stable_sample_indices(idx, obs_names, cap, seed)
        keep_idx.append(idx)
    keep_idx = np.sort(np.concatenate(keep_idx)) if keep_idx else np.array([], dtype=int)

    obs_digest = hashlib.sha256(
        "\n".join(sorted(obs_names[keep_idx].astype(str))).encode()
    ).hexdigest()[:12]
    print(
        f"    Cell filter (obs only): {len(valid)} perts (≥{min_cells}); "
        f"downsample ≤{max_cells_per_pert}/pert, ≤{max_control_cells} controls "
        f"(stable hash, seed={seed}) → {len(keep_idx)}/{adata.n_obs} cells "
        f"obs_digest={obs_digest}",
        flush=True,
    )

    # One slice only — never view-of-view on backed AnnData (Replogle Colab crash).
    if getattr(adata, "is_view", False):
        parent = getattr(adata, "_adata_ref", None)
        oidx = getattr(adata, "_oidx", None)
        if parent is None or oidx is None:
            raise ValueError(
                "Backed AnnData view without parent/_oidx; cannot materialize "
                "without creating a forbidden view-of-view. Reload without a "
                "prior subset, or call to_memory() earlier."
            )
        parent_idx = np.arange(parent.n_obs)[oidx][keep_idx]
        view = parent[parent_idx]
    else:
        view = adata[keep_idx]

    print("    Materializing filtered matrix into RAM…", flush=True)
    adata_mem = ensure_in_memory(view)
    print(
        f"    In memory: {adata_mem.n_obs} cells × {adata_mem.n_vars} genes  "
        f"backed={getattr(adata_mem, 'isbacked', False)}",
        flush=True,
    )
    return adata_mem, valid, counts


def _extract_adata(raw, dataset_name: str, sc):
    """Handle MuData (Papalexi) and AnnData; return (adata, pert_col, ctrl_label)."""
    meta = cfg.DATASETS[dataset_name]

    if "papalexi" in dataset_name.lower():
        if type(raw).__name__ != "MuData":
            raise TypeError(f"Expected MuData for Papalexi, got {type(raw)}")
        if "rna" not in raw.mod:
            raise KeyError("No 'rna' modality in Papalexi MuData")
        adata = raw.mod["rna"].copy()
        if "gene_target" not in raw.obs.columns:
            raise KeyError("'gene_target' not in Papalexi MuData.obs")
        adata.obs["gene_target"] = raw.obs["gene_target"].values
        return adata, "gene_target", "NT"

    adata = raw
    if type(adata).__name__ == "MuData":
        if "rna" in adata.mod:
            adata = adata.mod["rna"]
        elif "gex" in adata.mod:
            adata = adata.mod["gex"]
        else:
            adata = adata.mod[list(adata.mod.keys())[0]]

    if not isinstance(adata, sc.AnnData):
        raise TypeError(f"Expected AnnData after extract, got {type(adata)}")

    # Adamson UPR: gene-level aggregation + mod/Gal4 controls
    if meta.get("aggregate_to_gene") and "adamson" in dataset_name.lower() and "upr" in dataset_name.lower():
        return _collapse_adamson_upr_labels(adata)

    # Replogle label cleaning (obs-only; do not .copy() backed objects)
    if "replogle" in dataset_name.lower():
        adata.obs["perturbation"] = adata.obs["perturbation"].astype(str)

        def clean(x: str) -> str:
            if "non-targeting" in x or x.startswith("chr"):
                return "control"
            if "pos_control" in x:
                return "POS_CONTROL"
            return x.split("_")[0]

        # Annotate only — do NOT subset. Returning adata[mask] on a backed
        # file makes a view; materialize_min_cells then indexes again and
        # AnnData raises "cannot make a view of a view".
        adata.obs["condition"] = adata.obs["perturbation"].apply(clean)
        return adata, "condition", "control"

    possible = [
        "condition",
        "perturbation_name",
        "perturbation",
        "gene",
        "target",
        "guide_id",
        "sgRNA",
        "gene_target",
    ]
    pert_col = next((c for c in possible if c in adata.obs.columns), None)
    if pert_col is None:
        pert_col = next(
            (
                c
                for c in adata.obs.columns
                if any(k in c.lower() for k in ("pert", "guide", "gene", "target"))
            ),
            None,
        )
    if pert_col is None:
        raise KeyError(f"No perturbation column in {dataset_name}: {list(adata.obs.columns)}")

    # Obs-only edits (safe for backed). Avoid adata.copy() — that loads full X.
    # Adamson pilot (and some pertpy tables) store missing labels as float NaN;
    # coerce before any .lower() call.
    def _as_label(x) -> str:
        if x is None:
            return "nan"
        if isinstance(x, float) and np.isnan(x):
            return "nan"
        s = str(x).strip()
        if s.lower() in {"nan", "none", "<na>", "nat", ""}:
            return "nan"
        return s

    adata.obs[pert_col] = (
        pd.Series(adata.obs[pert_col]).map(_as_label).replace("nan", "NaN_Control")
    )
    labels = [str(x) for x in adata.obs[pert_col].unique()]

    keywords = list(meta["control_keywords"])
    exact = [c.lower() for c in keywords]
    ctrl_label = next((x for x in labels if x.lower() in exact), None)
    if ctrl_label is None:
        substr = [c for c in keywords if len(c) >= 3]
        ctrl_label = next(
            (x for x in labels if any(c in x.lower() for c in substr)),
            None,
        )
    if ctrl_label is None:
        # Collapse common control-like labels into a single 'control' bucket
        ctrl_kws = meta["control_keywords"]
        mapped = adata.obs[pert_col].apply(
            lambda x: "control" if any(k in str(x).lower() for k in ctrl_kws) else str(x)
        )
        if (mapped == "control").any():
            adata.obs[pert_col] = mapped
            ctrl_label = "control"
        else:
            ctrl_label = str(adata.obs[pert_col].value_counts().idxmax())
            print(f"    WARNING: fell back to most frequent label as control: {ctrl_label}")

    return adata, pert_col, ctrl_label


def _normalize_total_numpy(adata, target_sum: Optional[float] = None):
    """Library-size normalize without scanpy/numba (workaround for broken numba installs)."""
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        counts = np.asarray(X.sum(axis=1)).ravel()
    else:
        counts = np.asarray(X).sum(axis=1)
    counts = np.maximum(counts, 1e-8)
    if target_sum is None:
        target_sum = float(np.median(counts))
    scale = target_sum / counts
    if sparse.issparse(X):
        adata.X = sparse.diags(scale) @ X
    else:
        adata.X = np.asarray(X) * scale[:, None]


def _log1p_inplace(adata):
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.log1p(X.data)
        adata.X = X
    else:
        adata.X = np.log1p(np.asarray(X))


def _filter_cells_min_genes(adata, min_genes: int):
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        n_genes = (np.asarray(X) > 0).sum(axis=1)
    return adata[n_genes >= min_genes].copy()


def _hvg_seurat_v3_simple(adata, n_top: int, seed: int):
    """Variance-based HVG fallback (not identical to seurat_v3, but stable)."""
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = np.mean(X ** 2, axis=0)
    var = np.maximum(mean_sq - mean ** 2, 0)
    # dispersion-like score
    score = var / (mean + 1e-8)
    top = np.argsort(-score)[:n_top]
    return adata[:, top].copy()


def _expression_matrix(adata, *, allow_counts: bool = False):
    """
    Return the primary expression matrix, recovering from common backed /
    anndata edge cases where ``adata.X`` is None after ``to_memory()``.
    """
    X = getattr(adata, "X", None)
    if X is not None:
        return X

    layers = getattr(adata, "layers", None)
    if layers is not None:
        # Prefer matrices whose scale is compatible with a pinned
        # matrix_is_log=True. Never silently substitute raw counts for a
        # processed .X: doing so while skipping normalize/log1p changes PCA.
        for key in ("X", "logcounts", "normalized", "log1p"):
            if key in layers and layers[key] is not None:
                print(
                    f"    WARNING: adata.X is None; using layers[{key!r}]",
                    flush=True,
                )
                adata.X = layers[key]
                return adata.X
    raw = getattr(adata, "raw", None)
    if raw is not None and getattr(raw, "X", None) is not None:
        print(
            "    WARNING: adata.X is None; copying from adata.raw.X",
            flush=True,
        )
        # Align genes if raw has a wider gene set
        try:
            if raw.n_vars == adata.n_vars:
                adata.X = raw.X.copy() if hasattr(raw.X, "copy") else raw.X
            else:
                adata.X = raw[:, adata.var_names].X
                if hasattr(adata.X, "copy"):
                    adata.X = adata.X.copy()
        except Exception:
            adata.X = raw.X
        if adata.X is not None:
            return adata.X

    if allow_counts and layers is not None:
        for key in ("counts", "raw_counts"):
            if key in layers and layers[key] is not None:
                print(
                    f"    WARNING: adata.X is None; using raw layers[{key!r}]",
                    flush=True,
                )
                adata.X = layers[key]
                return adata.X

    raise ValueError(
        "AnnData has no compatible expression matrix. adata.X is None and no "
        "log-scale layer/raw matrix is available; refusing to substitute raw "
        "counts while matrix_is_log=True. Re-download the h5ad, use a compatible "
        "anndata version, or pass --h5ad to a known-good local file."
    )


def _looks_log_normalized(adata) -> bool:
    """Heuristic: scPerturb Replogle/Norman often ship already log1p-normalized."""
    from scipy import sparse

    try:
        X = _expression_matrix(adata, allow_counts=True)
    except ValueError:
        return False
    if sparse.issparse(X):
        if X.nnz == 0:
            return False
        # sample up to ~50k nonzero entries
        data = X.data[: min(len(X.data), 50_000)]
    else:
        flat = np.asarray(X[: min(500, X.shape[0]), : min(500, X.shape[1])]).ravel()
        data = flat[np.isfinite(flat)]
        if data.size == 0:
            return False
    mx = float(np.max(data))
    mean = float(np.mean(data))
    # raw UMI counts are typically >> 30; log1p expression sits lower
    return mx < 40.0 and mean < 8.0


def _pca_truncated_svd(adata, n_pcs: int, seed: int):
    """Sparse-friendly PCA substitute — never densifies the full matrix."""
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X))
    else:
        X = X.tocsr()
    n_comps = min(n_pcs, X.shape[0] - 1, X.shape[1] - 1, 100)
    print(f"    TruncatedSVD n_comps={n_comps} on {X.shape} sparse…", flush=True)
    svd = TruncatedSVD(n_components=n_comps, random_state=seed, n_iter=5)
    adata.obsm["X_pca"] = svd.fit_transform(X)
    return adata


def _hvg_subsampled(adata, n_top: int, seed: int, max_cells_for_hvg: int = 20_000):
    """Variance HVG using a cell subsample when n_obs is huge."""
    from scipy import sparse

    n = adata.n_obs
    if n > max_cells_for_hvg:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_cells_for_hvg, replace=False)
        sub = adata[idx]
        print(f"    HVG on {max_cells_for_hvg} / {n} cell subsample…", flush=True)
    else:
        sub = adata
        print(f"    HVG on all {n} cells…", flush=True)

    X = sub.X
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = np.mean(X ** 2, axis=0)
    var = np.maximum(mean_sq - mean ** 2, 0)
    score = var / (mean + 1e-8)
    n_top = min(n_top, adata.n_vars)
    top = np.argsort(-score)[:n_top]
    return adata[:, top].copy()


def resolve_matrix_is_log(
    dataset_name: Optional[str] = None,
    matrix_is_log: Optional[bool] = None,
    adata=None,
) -> tuple[bool, str]:
    """
    Resolve whether to skip normalize/log1p.

    Precedence: explicit override → DATASETS[name]["matrix_is_log"] → heuristic.
    Returns (is_log, source) where source is 'override'|'config'|'heuristic'.
    """
    if matrix_is_log is not None:
        return bool(matrix_is_log), "override"
    if dataset_name is not None:
        meta = cfg.DATASETS.get(cfg.resolve_dataset_name(dataset_name), {})
        if "matrix_is_log" in meta and meta["matrix_is_log"] is not None:
            return bool(meta["matrix_is_log"]), "config"
    if adata is None:
        raise ValueError("adata required when matrix_is_log is not pinned")
    guessed = _looks_log_normalized(adata)
    print(
        "    WARNING: matrix_is_log not pinned for this dataset; "
        f"using heuristic already_log≈{guessed}. Pin DATASETS[*]['matrix_is_log'] "
        "from inspect_matrix_scale.py — the heuristic is wrong/unstable on Adamson "
        "(full matrix raw vs filtered subset → True).",
        flush=True,
    )
    return guessed, "heuristic"


def resolve_normalize_target_sum(
    dataset_name: Optional[str] = None,
    target_sum: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """Library-size target. Precedence: override → DATASETS pin → config default."""
    if target_sum is not None:
        return float(target_sum), "override"
    if dataset_name is not None:
        meta = cfg.DATASETS.get(cfg.resolve_dataset_name(dataset_name), {})
        if meta.get("normalize_target_sum") is not None:
            return float(meta["normalize_target_sum"]), "config"
    if cfg.NORMALIZE_TARGET_SUM is None:
        return None, "median"
    return float(cfg.NORMALIZE_TARGET_SUM), "global"


def _mark_log1p(adata):
    """Tell scanpy HVG the matrix is already log — _log1p_inplace does not set this."""
    if not hasattr(adata, "uns") or adata.uns is None:
        adata.uns = {}
    adata.uns.setdefault("log1p", {"base": None})


def preprocess(
    adata,
    pert_col: str,
    ctrl_label: str,
    sc,
    n_pcs: int = cfg.N_PCS,
    min_cells: int = cfg.MIN_CELLS,
    seed: int = cfg.SEED,
    valid_perts: Optional[list] = None,
    counts: Optional[pd.Series] = None,
    dataset_name: Optional[str] = None,
    matrix_is_log: Optional[bool] = None,
    force_scanpy: bool = False,
):
    """Exact 2026-07-29.1 frozen preprocessing.

    The Aug 4 generation log records the Replogle path unambiguously:
    sparse normalize/log1p at ``target_sum=None``; variance HVGs on a
    seed-320 sample of 20,000 cells; TruncatedSVD with ``n_iter=5``.
    ``force_scanpy`` is retained only for call compatibility and does not
    change this frozen backend.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    if hasattr(sc, "settings"):
        sc.settings.seed = seed

    # Defensive: never .copy() a backed AnnData (Adamson/Norman Colab failure mode)
    adata = ensure_in_memory(adata)
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    large = adata.n_obs >= cfg.LARGE_DATASET_N_OBS
    already_log, log_src = resolve_matrix_is_log(
        dataset_name=dataset_name, matrix_is_log=matrix_is_log, adata=adata
    )
    # The freeze used the global target for every dataset. In particular,
    # Replogle used None (median library size), not the later fig2 value 1e4.
    target_sum = cfg.NORMALIZE_TARGET_SUM
    target_src = "frozen_global"
    # Some anndata/backed to_memory paths leave .X=None. Recover only from a
    # scale-compatible source: raw counts are allowed solely when the pipeline
    # is going to normalize/log them.
    _expression_matrix(adata, allow_counts=not already_log)
    # Heuristic is only diagnostic when the pin is unset; never let it abort a
    # pinned preprocess (Norman/Dixit Colab: to_memory can leave X briefly unset).
    if log_src == "heuristic":
        heuristic_guess = already_log
    else:
        try:
            heuristic_guess = _looks_log_normalized(adata)
        except Exception as exc:  # noqa: BLE001
            heuristic_guess = f"unavailable ({type(exc).__name__})"
    print(
        f"    Preprocess: n_obs={adata.n_obs} n_vars={adata.n_vars}  "
        f"large={large} "
        f"matrix_is_log={already_log} (source={log_src}; "
        f"heuristic≈{heuristic_guess}) "
        f"target_sum={target_sum} ({target_src})",
        flush=True,
    )
    if force_scanpy:
        print(
            "    note: force_scanpy is ignored; using the recorded "
            "2026-07-29.1 frozen embedding backend",
            flush=True,
        )

    # This branching is copied from the code that wrote frozen_sp_scores.csv.
    # Already-log matrices and n_obs>=40k take the sparse/SVD route.
    use_scanpy_pp = not (large or already_log)
    if large or already_log:
        if not already_log:
            print("    sparse normalize + log1p…", flush=True)
            if adata.n_vars > 500:
                print("    filter_cells (min_genes)…", flush=True)
                adata = _filter_cells_min_genes(adata, cfg.MIN_GENES_PER_CELL)
            _normalize_total_numpy(adata, target_sum)
            if cfg.LOG1P:
                _log1p_inplace(adata)
                _mark_log1p(adata)
        else:
            print(
                f"    skipping normalize/log1p (matrix_is_log=True via {log_src})",
                flush=True,
            )
            _mark_log1p(adata)
    else:
        print("    Preprocess: filter_cells → normalize → log1p…", flush=True)
        try:
            sc.pp.filter_cells(adata, min_genes=cfg.MIN_GENES_PER_CELL)
            if target_sum is None:
                sc.pp.normalize_total(adata)
            else:
                sc.pp.normalize_total(adata, target_sum=target_sum)
            if cfg.LOG1P:
                sc.pp.log1p(adata)
        except (AttributeError, ImportError, MemoryError) as e:
            print(
                f"    scanpy.pp failed ({e}); using frozen numpy/SVD fallback",
                flush=True,
            )
            use_scanpy_pp = False
            adata = _filter_cells_min_genes(adata, cfg.MIN_GENES_PER_CELL)
            _normalize_total_numpy(adata, target_sum)
            if cfg.LOG1P:
                _log1p_inplace(adata)
                _mark_log1p(adata)

    if counts is None or valid_perts is None:
        counts = adata.obs[pert_col].value_counts()
        valid_perts = [p for p in counts[counts >= min_cells].index if p != ctrl_label]
        keep = valid_perts + [ctrl_label]
        adata = adata[adata.obs[pert_col].isin(keep)].copy()
    else:
        # Recompute counts after QC filter; drop perts that fell below min_cells
        counts = adata.obs[pert_col].value_counts()
        valid_perts = [p for p in valid_perts if counts.get(p, 0) >= min_cells]
        keep = valid_perts + [ctrl_label]
        adata = adata[adata.obs[pert_col].isin(keep)].copy()

    print(f"    HVG ({cfg.N_HVG}) → embedding ({n_pcs})…", flush=True)
    backend = None
    if use_scanpy_pp and not large:
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=cfg.N_HVG, subset=True
            )
            sc.pp.pca(
                adata,
                n_comps=min(n_pcs, adata.n_vars - 1),
                random_state=seed,
            )
            backend = "scanpy.pp.pca"
        except (AttributeError, ImportError, MemoryError) as e:
            print(
                f"    scanpy HVG/PCA failed ({e}); using frozen SVD fallback",
                flush=True,
            )
            use_scanpy_pp = False

    if not use_scanpy_pp or large:
        if adata.n_vars > cfg.N_HVG:
            adata = _hvg_subsampled(adata, cfg.N_HVG, seed)
        adata = _pca_truncated_svd(adata, n_pcs, seed)
        backend = "frozen_truncated_svd"

    hvg = [str(x) for x in adata.var_names]
    hvg_digest = hashlib.sha256(
        "\n".join(sorted(hvg)).encode()
    ).hexdigest()[:12]
    print(
        f"    HVG n={len(hvg)} digest={hvg_digest} first={hvg[:8]}",
        flush=True,
    )

    if not hasattr(adata, "uns") or adata.uns is None:
        adata.uns = {}
    adata.uns["shesha_matrix_is_log"] = bool(already_log)
    adata.uns["shesha_matrix_is_log_source"] = log_src
    adata.uns["shesha_embed_backend"] = backend
    adata.uns["shesha_normalize_target_sum"] = target_sum
    adata.uns["shesha_normalize_target_sum_source"] = target_src
    print(
        f"    Preprocess done: {adata.n_obs} × {adata.n_vars}  "
        f"embed_backend={backend} target_sum={target_sum}",
        flush=True,
    )
    return adata, valid_perts, counts


def calculate_sp(control_matrix: np.ndarray, pert_matrix: np.ndarray) -> dict:
    """Directional coherence (Sp) + mean-shift magnitude + spread."""
    control_centroid = np.mean(control_matrix, axis=0)
    shift_vectors = pert_matrix - control_centroid
    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = float(np.linalg.norm(mean_shift))

    if mean_magnitude < 1e-6:
        return {"stability": 0.0, "magnitude": 0.0, "spread": 0.0, "snr": 0.0}

    norms = np.linalg.norm(shift_vectors, axis=1)
    valid_idx = norms > 1e-6
    if np.sum(valid_idx) < 5:
        return {"stability": 0.0, "magnitude": 0.0, "spread": 0.0, "snr": 0.0}

    unit_mean = mean_shift / mean_magnitude
    cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
    stability = float(np.mean(cosine_sims))

    pert_centroid = np.mean(shift_vectors, axis=0)
    internal_spread = float(np.mean(np.linalg.norm(shift_vectors - pert_centroid, axis=1)))
    snr = mean_magnitude / (internal_spread + 1e-6)

    return {
        "stability": stability,
        "magnitude": mean_magnitude,
        "spread": internal_spread,
        "snr": snr,
    }


def score_perturbations(
    adata,
    pert_col: str,
    ctrl_label: str,
    valid_perts: list[str],
    counts: pd.Series,
    dataset_name: str,
) -> pd.DataFrame:
    ctrl_mask = adata.obs[pert_col] == ctrl_label
    X_ctrl = np.asarray(adata.obsm["X_pca"][ctrl_mask])
    if X_ctrl.shape[0] < cfg.MIN_CONTROL_CELLS:
        raise ValueError(
            f"Insufficient control cells for {dataset_name}: {X_ctrl.shape[0]}"
        )

    rows = []
    for pert in valid_perts:
        X_pert = np.asarray(adata.obsm["X_pca"][adata.obs[pert_col] == pert])
        n_cells = X_pert.shape[0]
        metrics = calculate_sp(X_ctrl, X_pert)
        if metrics["magnitude"] <= 0:
            continue
        meta = cfg.DATASETS[dataset_name]
        rows.append(
            {
                "dataset": dataset_name,
                "perturbation": str(pert),
                "stability": metrics["stability"],
                "magnitude": metrics["magnitude"],
                "spread": metrics["spread"],
                "snr": metrics["snr"],
                "n_cells": n_cells,
                "n_control": int(X_ctrl.shape[0]),
                "modality": meta["modality"],
                "cell_type": meta["cell_type"],
                "config_version": cfg.CONFIG_VERSION,
                "min_cells": cfg.MIN_CELLS,
                "n_pcs": cfg.N_PCS,
                "seed": cfg.SEED,
                "matrix_is_log": bool(adata.uns.get("shesha_matrix_is_log"))
                if hasattr(adata, "uns")
                else meta.get("matrix_is_log"),
                "matrix_is_log_source": adata.uns.get("shesha_matrix_is_log_source")
                if hasattr(adata, "uns")
                else None,
            }
        )
    return pd.DataFrame(rows)


def prepare_dataset(
    dataset_name: str,
    ds=None,
    sc=None,
    n_pcs: int = cfg.N_PCS,
    min_cells: int = cfg.MIN_CELLS,
    seed: int = cfg.SEED,
    prefer_local: bool = True,
    h5ad_path: Optional[Path] = None,
    max_perts: Optional[int] = None,
    max_control_cells: int = cfg.MAX_CONTROL_CELLS,
    matrix_is_log: Optional[bool] = None,
    force_scanpy: bool = False,
):
    """Load + preprocess one frozen dataset. Returns adata with X_pca."""
    dataset_name = cfg.resolve_dataset_name(dataset_name)
    if dataset_name not in cfg.DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    if sc is None:
        import scanpy as sc
        setup_cache()
        sc.settings.datasetdir = Path(
            os.environ.get("SCVERSE_DATADIR", str(cfg.CACHE_DIR))
        )

    meta = cfg.DATASETS[dataset_name]
    print(f"\n>>> {dataset_name}")
    print(f"    modality={meta['modality']}  cell_type={meta['cell_type']}")
    print(f"    loader={meta['loader']}  min_cells={min_cells}  n_pcs={n_pcs}")
    if matrix_is_log is not None:
        print(f"    matrix_is_log override={matrix_is_log}")
    elif "matrix_is_log" in meta:
        print(f"    matrix_is_log pin={meta['matrix_is_log']}")
    if max_perts is not None:
        print(f"    max_perts={max_perts}  max_control_cells={max_control_cells}")

    raw = load_raw(
        dataset_name,
        ds=ds,
        sc=sc,
        prefer_local=prefer_local,
        h5ad_path=h5ad_path,
    )
    adata, pert_col, ctrl_label = _extract_adata(raw, dataset_name, sc)
    print(
        f"    pert_col={pert_col!r}  ctrl={ctrl_label!r}  n_obs={adata.n_obs}  "
        f"backed={getattr(adata, 'isbacked', False)}",
        flush=True,
    )

    adata, valid, counts = materialize_min_cells(
        adata,
        pert_col,
        ctrl_label,
        min_cells=min_cells,
        max_control_cells=max_control_cells,
        max_perts=max_perts,
    )

    adata, valid, counts = preprocess(
        adata,
        pert_col,
        ctrl_label,
        sc,
        n_pcs=n_pcs,
        min_cells=min_cells,
        seed=seed,
        valid_perts=valid,
        counts=counts,
        dataset_name=dataset_name,
        matrix_is_log=matrix_is_log,
        force_scanpy=force_scanpy,
    )
    n_ctrl = int((adata.obs[pert_col] == ctrl_label).sum())
    print(f"    after filter: {len(valid)} perturbations, {n_ctrl} control cells", flush=True)
    return adata, pert_col, ctrl_label, valid, counts


def run_dataset(
    dataset_name: str,
    ds=None,
    sc=None,
    n_pcs: int = cfg.N_PCS,
    min_cells: int = cfg.MIN_CELLS,
    seed: int = cfg.SEED,
    prefer_local: bool = True,
    h5ad_path: Optional[Path] = None,
    max_perts: Optional[int] = None,
    max_control_cells: int = cfg.MAX_CONTROL_CELLS,
    matrix_is_log: Optional[bool] = None,
    force_scanpy: bool = False,
) -> pd.DataFrame:
    """End-to-end: load → preprocess → Sp for one frozen dataset name.

    max_perts / max_control_cells: Colab escape hatches. Leave None / default
    for manuscript-frozen tables. For synthetic-benchmark reference selection,
    max_perts=200 and max_control_cells=1000 is enough and avoids the 175k-cell
    to_memory stall on Replogle.

    matrix_is_log: override DATASETS pin / heuristic. True skips normalize+log1p;
    False forces them. None uses the pinned config value (or heuristic if unset).
    """
    adata, pert_col, ctrl_label, valid, counts = prepare_dataset(
        dataset_name,
        ds=ds,
        sc=sc,
        n_pcs=n_pcs,
        min_cells=min_cells,
        seed=seed,
        prefer_local=prefer_local,
        h5ad_path=h5ad_path,
        max_perts=max_perts,
        max_control_cells=max_control_cells,
        matrix_is_log=matrix_is_log,
        force_scanpy=force_scanpy,
    )
    df = score_perturbations(adata, pert_col, ctrl_label, valid, counts, dataset_name)
    print(f"    scored: {len(df)} perturbations", flush=True)
    return df
