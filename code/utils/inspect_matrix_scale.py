#!/usr/bin/env python3
"""
Inspect AnnData .X BEFORE preprocessing to decide matrix_is_log pin.

Settles the Adamson force-normalize question on the matrix, not the p-value:
  - integer-valued / large max / count-like  → raw → matrix_is_log=False
  - non-integer floats, max ~5–10           → already log → matrix_is_log=True

Usage:
  python inspect_matrix_scale.py --dataset "Adamson 2016 UPR (CRISPRi)"
  python inspect_matrix_scale.py --h5ad /content/adamson_2016_upr_perturb_seq.h5ad
  python inspect_matrix_scale.py --all-main
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

import pipeline_config as cfg
from pipeline_core import load_raw, setup_cache
from revision_io import resolve_out_dir


def _as_1d_float(a) -> np.ndarray:
    """Coerce sparse .data / h5py samples to a flat float1d (backed-safe)."""
    a = np.asarray(a)
    if a.dtype == object:
        # Backed/h5py sometimes yields object arrays of row chunks
        a = np.concatenate([np.asarray(x, dtype=float).ravel() for x in a.flat])
    return np.asarray(a, dtype=float).ravel()


def _sample_values(X, n_max: int = 200_000) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(X):
        if getattr(X, "nnz", 0) == 0:
            return np.asarray([], dtype=float)
        raw = getattr(X, "data", None)
        if raw is None:
            # densify a tiny corner as last resort
            arr = X[: min(500, X.shape[0]), : min(500, X.shape[1])]
            data = _as_1d_float(arr.toarray() if sparse.issparse(arr) else arr)
        else:
            n = min(int(getattr(raw, "shape", [len(raw)])[0]), n_max)
            data = _as_1d_float(raw[:n])
        return data[np.isfinite(data)]

    # Dense / array-like (including some backed dense views)
    try:
        arr = X[: min(2000, X.shape[0]), : min(2000, X.shape[1])]
        flat = _as_1d_float(arr)
    except Exception:
        # Last resort: iterate a few rows
        rows = []
        for i in range(min(200, X.shape[0])):
            rows.append(_as_1d_float(X[i, : min(2000, X.shape[1])]))
        flat = np.concatenate(rows) if rows else np.asarray([], dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size > n_max:
        rng = np.random.default_rng(cfg.SEED)
        flat = rng.choice(flat, size=n_max, replace=False)
    return flat


def _materialize_sample(adata, n_cells: int = 2000):
    """
    Pull a small in-memory cell subset for scale inspection.

    Backed AnnData forbids naive .X.data access ("setting an array element
    with a sequence"); sampling cells then to_memory() is the safe path.
    """
    n = int(adata.n_obs)
    n_take = min(n_cells, n)
    rng = np.random.default_rng(cfg.SEED)
    idx = np.sort(rng.choice(n, size=n_take, replace=False))
    print(f"  sampling {n_take}/{n} cells into memory for .X inspect…", flush=True)
    try:
        sub = adata[idx].to_memory()
    except Exception:
        # Some views already in memory
        sub = adata[idx].copy()
    return sub


def inspect_X(X, *, label: str, full_shape=None) -> dict:
    from scipy import sparse

    is_sparse = bool(sparse.issparse(X))
    data = _sample_values(X)
    if data.size == 0:
        return {
            "label": label,
            "empty": True,
            "recommendation": None,
            "note": "empty matrix sample",
        }

    mx = float(np.max(data))
    mn = float(np.min(data))
    mean = float(np.mean(data))
    med = float(np.median(data))
    # integer-like: nearly all values close to integers
    frac_near_int = float(np.mean(np.abs(data - np.round(data)) < 1e-6))
    # negative values almost never appear in raw UMI; log can be zero-inflated non-neg
    frac_neg = float(np.mean(data < 0))
    shape = list(full_shape) if full_shape is not None else list(X.shape)
    # fraction of exact zeros in sample (sparse data already nonzero-only for .data)
    if is_sparse:
        try:
            nnz = int(X.nnz)
            n_tot = int(X.shape[0] * X.shape[1])
            frac_zero = 1.0 - (nnz / n_tot) if n_tot else np.nan
        except Exception:
            frac_zero = float(np.mean(data == 0))
        dtype = str(getattr(X, "dtype", "unknown"))
    else:
        frac_zero = float(np.mean(data == 0))
        dtype = str(getattr(X, "dtype", type(X)))

    heuristic = bool(mx < 40.0 and mean < 8.0)

    # Decision rule (documented in module docstring)
    if frac_near_int >= 0.98 and mx >= 20:
        rec = False
        reason = "integer-like with large max → raw counts; force normalize+log1p"
    elif frac_near_int >= 0.98 and mx < 20:
        rec = False
        reason = "integer-like but small max → likely small-integer counts; normalize+log1p"
    elif mx <= 15 and mean < 5 and frac_near_int < 0.5:
        rec = True
        reason = "non-integer floats, max≲15 → already log-normalized; skip normalize+log1p"
    elif mx < 40 and mean < 8 and frac_near_int < 0.8:
        rec = True
        reason = "log-like floats (matches heuristic band); prefer skip unless provenance says counts"
    else:
        rec = False
        reason = "count-like / large dynamic range; normalize+log1p"

    return {
        "label": label,
        "shape": shape,
        "sparse": is_sparse,
        "dtype": dtype,
        "n_sampled": int(data.size),
        "min": mn,
        "max": mx,
        "mean": mean,
        "median": med,
        "frac_near_integer": frac_near_int,
        "frac_negative": frac_neg,
        "frac_zero_approx": frac_zero,
        "heuristic_already_log": heuristic,
        "recommended_matrix_is_log": rec,
        "reason": reason,
    }


def _resolve_h5ad(name: str, explicit: Path | None) -> Path | None:
    """Prefer an existing explicit path; else search cache /cwd for local_h5ad."""
    if explicit is not None:
        p = Path(explicit)
        if p.exists() and p.stat().st_size > 0:
            return p
        print(f"  note: --h5ad not found or empty: {p}", flush=True)

    meta = cfg.DATASETS.get(name, {})
    fname = meta.get("local_h5ad")
    if not fname:
        return None
    cache = setup_cache()
    candidates = [
        cache / fname,
        Path("/content") / fname,
        Path("/content/shesha-crispr") / fname,
        Path.cwd() / fname,
        Path.cwd() / "shesha-crispr" / fname,
        cfg.OUTPUT_DIR / fname,
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            print(f"  using h5ad: {c}", flush=True)
            return c
    return None


def inspect_dataset(
    name: str,
    *,
    h5ad: Path | None = None,
    prefer_local: bool = True,
) -> dict:
    import scanpy as sc

    setup_cache()
    sc.settings.datasetdir = cfg.CACHE_DIR
    resolved = _resolve_h5ad(name, h5ad)
    raw = load_raw(name, sc=sc, prefer_local=prefer_local, h5ad_path=resolved)
    # load_raw may return AnnData or MuData-ish; prefer .X
    adata = raw
    if hasattr(raw, "mod"):  # MuData
        # Papalexi: RNA modality
        adata = raw.mod.get("rna", next(iter(raw.mod.values())))

    full_shape = [int(adata.n_obs), int(adata.n_vars)]
    backed = bool(getattr(adata, "isbacked", False))
    if backed or full_shape[0] > 50_000:
        # Never touch full backed .X — materialize a cell subsample first
        sub = _materialize_sample(adata, n_cells=2000)
        X = sub.X
        out = inspect_X(X, label=name, full_shape=full_shape)
        out["sampled_cells"] = int(sub.n_obs)
        out["backed"] = backed
        # close backed handle if we still hold it
        try:
            if backed and hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
    else:
        out = inspect_X(adata.X, label=name, full_shape=full_shape)
        out["backed"] = backed

    pinned = cfg.DATASETS.get(name, {}).get("matrix_is_log")
    out["pinned_matrix_is_log"] = pinned
    out["pin_matches_recommendation"] = (
        pinned == out["recommended_matrix_is_log"] if pinned is not None else None
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="Adamson 2016 UPR (CRISPRi)")
    p.add_argument("--h5ad", type=Path, default=None)
    p.add_argument(
        "--adamson-h5ad",
        type=Path,
        default=None,
        help="Local Adamson UPR h5ad (used for that dataset even under --all-main)",
    )
    p.add_argument("--all-main", action="store_true", help="Inspect every in_main dataset")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    names = (
        [n for n, m in cfg.DATASETS.items() if m.get("in_main")]
        if args.all_main
        else [cfg.resolve_dataset_name(args.dataset)]
    )

    reports = []
    for name in names:
        print("=" * 72)
        print(name)
        try:
            h5 = None
            if "Adamson" in name and "UPR" in name:
                h5 = args.adamson_h5ad or args.h5ad
            elif not args.all_main:
                h5 = args.h5ad
            rep = inspect_dataset(name, h5ad=h5)
        except Exception as e:
            import traceback

            rep = {
                "label": name,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            print(f"  ERROR: {type(e).__name__}: {e}")
            # short stack for Colab debugging
            print("".join(traceback.format_exception_only(type(e), e)), end="")
        reports.append(rep)
        if "error" not in rep:
            print(
                f"  shape={rep['shape']} sparse={rep['sparse']} dtype={rep['dtype']}\n"
                f"  max={rep['max']:.4g} mean={rep['mean']:.4g} "
                f"frac_near_int={rep['frac_near_integer']:.3f}\n"
                f"  heuristic_already_log={rep['heuristic_already_log']}\n"
                f"  RECOMMEND matrix_is_log={rep['recommended_matrix_is_log']}  "
                f"({rep['reason']})\n"
                f"  pinned={rep['pinned_matrix_is_log']}  "
                f"matches={rep['pin_matches_recommendation']}"
            )

    if args.all_main:
        out_path = out_dir / "matrix_scale_all_main.json"
    else:
        slug = (
            cfg.resolve_dataset_name(args.dataset)
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        out_path = out_dir / f"matrix_scale_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(reports if args.all_main else reports[0], f, indent=2)
    print(f"\nWrote {out_path}")

    # Pin table for methods (six-dataset)
    rows = []
    for rep in reports:
        if "error" in rep:
            rows.append({
                "dataset": rep.get("label"),
                "error": rep["error"],
                "pinned_matrix_is_log": cfg.DATASETS.get(
                    rep.get("label", ""), {}
                ).get("matrix_is_log"),
                "matrix_scale_verified": cfg.DATASETS.get(
                    rep.get("label", ""), {}
                ).get("matrix_scale_verified"),
            })
            continue
        rows.append({
            "dataset": rep["label"],
            "max": rep["max"],
            "mean": rep["mean"],
            "frac_near_integer": rep["frac_near_integer"],
            "heuristic_already_log": rep["heuristic_already_log"],
            "recommended_matrix_is_log": rep["recommended_matrix_is_log"],
            "pinned_matrix_is_log": rep["pinned_matrix_is_log"],
            "pin_matches_recommendation": rep["pin_matches_recommendation"],
            "matrix_scale_verified_in_config": cfg.DATASETS.get(
                rep["label"], {}
            ).get("matrix_scale_verified"),
            "reason": rep["reason"],
        })
    import pandas as pd

    pin_csv = out_dir / "matrix_scale_pin_table.csv"
    pd.DataFrame(rows).to_csv(pin_csv, index=False)
    print(f"Wrote {pin_csv}")
    pilot = [r for r in rows if r.get("dataset") and "pilot" in str(r["dataset"]).lower()]
    if pilot and not pilot[0].get("pin_matches_recommendation"):
        print(
            "\n⚠ Adamson pilot: pin does not match .X recommendation — "
            "update pipeline_config before claiming verified."
        )
    elif pilot and pilot[0].get("pin_matches_recommendation"):
        print(
            "\nAdamson pilot: .X inspect agrees with pin "
            f"(matrix_is_log={pilot[0].get('pinned_matrix_is_log')}). "
            "Set matrix_scale_verified=True in pipeline_config.py."
        )


if __name__ == "__main__":
    main()
