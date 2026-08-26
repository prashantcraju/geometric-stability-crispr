#!/usr/bin/env python3
"""
Prove assert_frozen_sp_compatible ABORTS on mismatch (does not warn-and-continue).

Usage:
  python check_frozen_sp_guard.py --frozen-sp shesha-crispr/frozen_sp_scores.csv

Exits 0 only if: matching file PASSes and three synthetic mismatches each raise.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import pipeline_config as cfg
from pipeline_core import assert_frozen_sp_compatible, compute_sp_digest


def _must_raise(label: str, path: Path) -> None:
    try:
        assert_frozen_sp_compatible(path)
    except (ValueError, FileNotFoundError) as e:
        print(f"  ABORT OK [{label}]: {type(e).__name__}: {e}")
        return
    raise SystemExit(f"FAIL: expected abort for {label}, but guard passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-sp", type=Path, required=True)
    args = p.parse_args()
    src = Path(args.frozen_sp)
    if not src.exists():
        raise SystemExit(f"missing {src}")

    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"probe file={src}")

    info = assert_frozen_sp_compatible(src)
    print(f"  PASS on matching file: digest={info['sp_digest']}")

    df = pd.read_csv(src)
    tmpdir = Path(tempfile.mkdtemp(prefix="sp_guard_"))
    try:
        # 1) wrong config_version, intact Sp
        bad_ver = tmpdir / "bad_version.csv"
        d1 = df.copy()
        d1["config_version"] = "2026-07-25.1"
        d1.to_csv(bad_ver, index=False)
        _must_raise("stale config_version", bad_ver)

        # 2) matching stamp but Sp values tampered (digest must catch)
        bad_sp = tmpdir / "bad_sp.csv"
        d2 = df.copy()
        d2["stability"] = d2["stability"].astype(float) + 0.123456
        # keep old digest column → mismatch vs recomputed
        if "sp_digest" not in d2.columns:
            d2["sp_digest"] = compute_sp_digest(df)  # stamp from untampered
        d2.to_csv(bad_sp, index=False)
        _must_raise("tampered Sp / digest mismatch", bad_sp)

        # 3) matching version+n but missing digest column
        bad_dig = tmpdir / "no_digest.csv"
        d3 = df.copy()
        if "sp_digest" in d3.columns:
            d3 = d3.drop(columns=["sp_digest"])
        d3.to_csv(bad_dig, index=False)
        _must_raise("missing sp_digest", bad_dig)

        # 4) wrong n_rows
        bad_n = tmpdir / "partial.csv"
        d4 = df.head(max(10, len(df) // 2)).copy()
        d4.to_csv(bad_n, index=False)
        _must_raise("partial n_rows", bad_n)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\nGUARD SELF-TEST PASSED — mismatches abort; matching file accepted.")


if __name__ == "__main__":
    main()
