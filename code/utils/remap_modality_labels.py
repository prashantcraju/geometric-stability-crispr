#!/usr/bin/env python3
"""
Remap legacy dataset display names → frozen CRISPR-KO / Adamson UPR names.

Dixit is CRISPR-KO (not CRISPRi); Papalexi includes KO;
Adamson pilot vs UPR should be distinguished.

Usage:
  # rewrite CSVs under shesha-crispr/ (creates *.bak)
  python remap_modality_labels.py --apply

  # dry-run
  python remap_modality_labels.py

  # also print which .py files still contain legacy strings
  python remap_modality_labels.py --scan-py

  # --out-dir is an accepted alias for --csv-dir (same flag used elsewhere)
  python remap_modality_labels.py --apply --out-dir shesha-crispr
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

import pipeline_config as cfg

# Exact dataset-name remaps for CSV `dataset` columns / figure keys
DATASET_RENAMES = {
    "Dixit 2016 (CRISPRi)": "Dixit 2016 (CRISPR-KO)",
    "Papalexi 2021 (CRISPR)": "Papalexi 2021 (CRISPR-KO)",
    # Pilot was previously the only Adamson key in most tables
    "Adamson 2016 (CRISPRi)": "Adamson 2016 pilot (CRISPRi)",
}

# Substring replacements safe for source comments / print strings
# (order matters — longer patterns first)
SOURCE_REPLACEMENTS = [
    ("Dixit 2016 (CRISPRi)", "Dixit 2016 (CRISPR-KO)"),
    ("Papalexi 2021 (CRISPR)", "Papalexi 2021 (CRISPR-KO)"),
    # leave Adamson pilot string as-is in source unless it's clearly the UPR arm
]


def remap_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: DATASET_RENAMES.get(x, cfg.resolve_dataset_name(str(x)) if pd.notna(x) else x))


def remap_csv(path: Path, apply: bool) -> dict:
    df = pd.read_csv(path)
    changed_cols = []
    for col in df.columns:
        if col.lower() in {"dataset", "dataset_name", "study"} or col == "dataset":
            before = df[col].astype(str)
            after = before.map(lambda x: DATASET_RENAMES.get(x, x))
            if (before != after).any():
                changed_cols.append(col)
                if apply:
                    df[col] = after
        # modality column corrections when tied to Dixit/Papalexi
        if col.lower() in {"modality", "perturbation_type", "design_modality"}:
            # only fix rows whose dataset (if present) is Dixit/Papalexi
            if "dataset" in df.columns:
                mask = df["dataset"].astype(str).str.contains("Dixit|Papalexi", regex=True)
                if mask.any() and apply:
                    df.loc[mask, col] = df.loc[mask, col].replace(
                        {"CRISPRi": "CRISPR-KO", "CRISPR": "CRISPR-KO", "Pooled": "CRISPR-KO"}
                    )
                    changed_cols.append(col)

    if apply and changed_cols:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        df.to_csv(path, index=False)
    return {"path": str(path), "changed_cols": changed_cols, "n_rows": len(df)}


def scan_py(root: Path) -> list[tuple[str, list[str]]]:
    hits = []
    patterns = [
        r"Dixit 2016 \(CRISPRi\)",
        r"Papalexi 2021 \(CRISPR\)",
    ]
    skip_dirs = {".venv-rev", ".venv-test", "__pycache__", ".git"}
    for py in root.rglob("*.py"):
        if any(part in skip_dirs for part in py.parts):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        found = [p for p in patterns if re.search(p, text)]
        if found:
            hits.append((str(py.relative_to(root)), found))
    return hits


def patch_py_files(root: Path, apply: bool) -> list[str]:
    """Apply SOURCE_REPLACEMENTS to project .py files (not venvs)."""
    changed = []
    skip_dirs = {".venv-rev", ".venv-test", "__pycache__", ".git"}
    for py in root.rglob("*.py"):
        if any(part in skip_dirs for part in py.parts):
            continue
        # do not rewrite this script's documentation of legacy names in DATASET_RENAMES
        if py.name == "remap_modality_labels.py":
            continue
        if py.name in {"pipeline_config.py"}:
            continue  # already has LEGACY_NAME_MAP
        text = py.read_text(encoding="utf-8", errors="ignore")
        new = text
        for old, repl in SOURCE_REPLACEMENTS:
            new = new.replace(old, repl)
        if new != text:
            changed.append(str(py.relative_to(root)))
            if apply:
                py.write_text(new, encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory of result CSVs (default: shesha-crispr/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Alias for --csv-dir (accepted so batch scripts can use one flag)",
    )
    parser.add_argument("--scan-py", action="store_true", help="List .py files with legacy names")
    parser.add_argument(
        "--patch-py",
        action="store_true",
        help="Replace legacy Dixit/Papalexi strings in project .py files",
    )
    args = parser.parse_args()

    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print("\nDataset renames:")
    for k, v in DATASET_RENAMES.items():
        print(f"  {k!r} → {v!r}")

    if args.csv_dir is not None and args.out_dir is not None and args.csv_dir != args.out_dir:
        raise SystemExit("Pass only one of --csv-dir / --out-dir (they must agree).")
    csv_dir = Path(args.csv_dir or args.out_dir or cfg.OUTPUT_DIR)
    if csv_dir.exists():
        print(f"\n--- CSVs in {csv_dir} ---")
        for csv in sorted(csv_dir.glob("*.csv")):
            info = remap_csv(csv, apply=args.apply)
            status = "CHANGED" if info["changed_cols"] else "ok"
            print(f"  [{status}] {csv.name}  cols={info['changed_cols']}")
    else:
        print(f"\n(no csv dir yet: {csv_dir})")

    if args.scan_py or args.patch_py:
        root = cfg.ROOT
        print("\n--- .py legacy-name scan ---")
        for rel, pats in scan_py(root):
            print(f"  {rel}: {pats}")

    if args.patch_py:
        print("\n--- .py patch ---")
        changed = patch_py_files(cfg.ROOT, apply=args.apply)
        if not changed:
            print("  (no files needed patching)")
        for rel in changed:
            print(f"  {'wrote' if args.apply else 'would write'}: {rel}")

    if not args.apply:
        print("\nRe-run with --apply to write. Example:")
        print("  python remap_modality_labels.py --apply --patch-py")


if __name__ == "__main__":
    main()
