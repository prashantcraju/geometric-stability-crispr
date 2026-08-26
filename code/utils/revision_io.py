"""Shared I/O helpers: output directory, frozen Sp tables, downloads."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import os
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

import pipeline_config as cfg


def data_search_dirs() -> list[Path]:
    """Directories searched for companion CSVs.

    Order: ``SHESHA_OUT`` / ``CRISPR_DATA``, ``pipeline_config.OUTPUT_DIR``,
    ``./shesha-crispr``, the current directory, then ``/content/shesha-crispr``
    (generic Colab drop folder). Set ``SHESHA_OUT`` to a Drive path on Colab.
    """
    dirs: list[Path] = []
    seen: set[str] = set()
    for key in ("SHESHA_OUT", "CRISPR_DATA"):
        raw = os.environ.get(key, "").strip()
        if raw:
            dirs.append(Path(raw).expanduser())
    dirs.extend(
        [
            cfg.OUTPUT_DIR,
            Path("shesha-crispr"),
            Path("."),
            Path("/content/shesha-crispr"),
        ]
    )
    out: list[Path] = []
    for p in dirs:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def find_data_file(*names: str) -> Optional[Path]:
    """Return the first existing file among *names* in ``data_search_dirs()``."""
    for name in names:
        for root in data_search_dirs():
            p = Path(root) / name
            if p.exists():
                return p
    return None


def resolve_out_dir(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for p in data_search_dirs():
        if p.exists() and p.is_dir() and p != Path("."):
            return p
        if p == cfg.OUTPUT_DIR:
            p.mkdir(parents=True, exist_ok=True)
            return p
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.OUTPUT_DIR


def find_sp_csv(out_dir: Path, explicit: Optional[Path] = None) -> Path:
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)
    # Prefer stamped freeze (config_version / sp_digest) over the euclidean
    # alias, which attach_stress_markers may overwrite with stress columns.
    for name in (
        "frozen_sp_scores.csv",
        "frozen_sp_scores_sample.csv",
        "shesha_crispr_results_euclidean.csv",
        "adamson_upr_sp_scores.csv",
    ):
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No Sp scores CSV in {out_dir}. Run run_frozen_main.py first "
        "(or --max-perts sample → frozen_sp_scores_sample.csv)."
    )


def load_sp_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    rename = {}
    lower = {c.lower(): c for c in df.columns}
    for canon, opts in {
        "stability": ["stability", "sp"],
        "magnitude": ["magnitude", "mp"],
        "dataset": ["dataset", "dataset_name"],
        "perturbation": ["perturbation", "gene", "target"],
        "n_cells": ["n_cells", "ncell"],
    }.items():
        if canon not in df.columns:
            for o in opts:
                if o in lower:
                    rename[lower[o]] = canon
                    break
    df = df.rename(columns=rename)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).map(cfg.resolve_dataset_name)
    if "perturbation" in df.columns:
        df["gene"] = df["perturbation"].astype(str).str.upper().str.split("_").str[0]
    return df


def _ssl_contexts():
    """Yield (context_or_None, label) — verified first, then Colab fallback."""
    yield None, "default"
    try:
        import certifi

        yield ssl.create_default_context(cafile=certifi.where()), "certifi"
    except Exception:
        pass
    # Helmholtz / some Colab images fail CA verification
    yield ssl._create_unverified_context(), "unverified"


def download(url: str, dest: Path, min_bytes: int = 1000) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    print(f"  Downloading {dest.name}\n    from {url}", flush=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (shesha-crispr-revision)"},
    )
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err: Exception | None = None
    for ctx, label in _ssl_contexts():
        try:
            if label == "unverified":
                print("    (SSL verify disabled for this attempt)", flush=True)
            kwargs = {"timeout": 600}
            if ctx is not None:
                kwargs["context"] = ctx
            with urllib.request.urlopen(req, **kwargs) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            last_err = None
            break
        except Exception as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            continue
    if last_err is not None:
        raise last_err
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Download too small ({size} B): {url}")
    # Reject HTML error/login pages that some hosts return with 200
    with open(tmp, "rb") as fh:
        head = fh.read(200).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        tmp.unlink(missing_ok=True)
        raise IOError(f"Download looks like HTML, not data: {url}")
    tmp.replace(dest)
    print(f"  Saved {dest} ({size / 1e6:.2f} MB)", flush=True)
    return dest


ANNOT_DIR = cfg.CACHE_DIR / "annotations"
