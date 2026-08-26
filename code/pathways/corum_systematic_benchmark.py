#!/usr/bin/env python3
"""
CORUM / TF-target / DepMap systematic benchmarking.

For Replogle Sp scores:
  1. CORUM: complex-subunit perturbations vs others; Sp vs complex membership /
     max complex size
  2. TRRUST: Sp vs TF regulon size (number of annotated targets)
  3. DepMap (optional): Sp vs CRISPR gene-effect fitness in K562 if available
  4. DEG count (optional): Sp vs n_DEGs if column present, or lightweight
     mean-expression z-count when --compute-degs

Usage:
  python corum_systematic_benchmark.py \\
      --input shesha-crispr/frozen_sp_scores.csv \\
      --out-dir shesha-crispr
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import pipeline_config as cfg
from revision_io import ANNOT_DIR, download, find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import bootstrap_spearman_ci

# Helmholtz often returns tiny HTML stubs from Colab; Zenodo mirror is reliable.
CORUM_URLS = [
    # Official Helmholtz dump mirrored on Zenodo (CORUM 5.1; plain TSV, ~6 MB)
    (
        "https://zenodo.org/api/records/17419058/files/corum_humanComplexes.txt/content",
        "corum_humanComplexes.txt",
        100_000,
    ),
    (
        "https://zenodo.org/records/17419058/files/corum_humanComplexes.txt?download=1",
        "corum_humanComplexes.txt",
        100_000,
    ),
    (
        "https://zenodo.org/api/records/17419058/files/corum_allComplexes.txt/content",
        "corum_allComplexes.txt",
        100_000,
    ),
    # Official site (often blocked / HTML stub on Colab)
    (
        "https://mips.helmholtz-muenchen.de/corum/download/humanComplexes.txt.zip",
        "corum_humanComplexes.txt.zip",
        50_000,
    ),
    (
        "https://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip",
        "corum_allComplexes.txt.zip",
        50_000,
    ),
]
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
# DepMap 23Q4 CRISPR gene effect (large). Optional.
DEPMAP_URL = (
    "https://figshare.com/ndownloader/files/42472731"  # may change; script tolerates failure
)


def _read_corum_table(path: Path) -> str:
    """Read CORUM TSV text from .txt or .zip."""
    path = Path(path)
    if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            inner = next(n for n in names if n.endswith(".txt"))
            return zf.read(inner).decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def load_corum(cache: Path, local_path: Path | None = None) -> pd.DataFrame:
    """Return tidy table: gene, complex_id, complex_name, n_subunits."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)

    if local_path is not None:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"--corum-zip not found: {path}")
        print(f"  Using local CORUM file: {path}")
    else:
        path = None
        # Reuse any previously cached good file
        for cand in (
            cache / "corum_humanComplexes.txt",
            cache / "corum_allComplexes.txt",
            cache / "corum_humanComplexes.txt.zip",
            cache / "corum_allComplexes.txt.zip",
        ):
            if cand.exists() and cand.stat().st_size >= 50_000:
                path = cand
                print(f"  Using cached CORUM: {path}")
                break

        if path is None:
            last = None
            for url, fname, min_b in CORUM_URLS:
                dest = cache / fname
                try:
                    download(url, dest, min_bytes=min_b)
                    path = dest
                    break
                except Exception as e:
                    last = e
                    print(f"  CORUM URL failed: {e}")
                    dest.unlink(missing_ok=True)
            if path is None:
                raise RuntimeError(
                    f"Could not download CORUM ({last}). "
                    "Download corum_humanComplexes.txt from "
                    "https://zenodo.org/records/17419058 "
                    "or allComplexes.txt.zip from the CORUM site, then pass "
                    "--corum-zip /path/to/file"
                )

    raw = _read_corum_table(path)
    if "ComplexID" not in raw[:2000] and "complex" not in raw[:500].lower():
        raise ValueError(
            f"CORUM file does not look like a complexes table: {path} "
            f"(head={raw[:120]!r})"
        )

    # CORUM uses tab-separated; gene names column varies by release
    df = pd.read_csv(io.StringIO(raw), sep="\t")
    # Find columns
    cols = {c.lower(): c for c in df.columns}
    if "organism" in cols:
        org = cols["organism"]
        human = df[org].astype(str).str.lower().str.contains("human", na=False)
        if human.any():
            df = df.loc[human].copy()
            print(f"  CORUM human complexes: {len(df)} rows")
    id_col = cols.get("complexid") or cols.get("complex_id") or df.columns[0]
    name_col = cols.get("complexname") or cols.get("complex_name") or df.columns[1]
    gene_col = None
    for key in (
        "subunits(gene name)",
        "subunits (gene name)",
        "subunits_gene_name",
        "genes",
        "gene names",
        "subunits gene name",
    ):
        if key in cols:
            gene_col = cols[key]
            break
    if gene_col is None:
        # heuristic: column containing ';' separated gene symbols
        for c in df.columns:
            sample = str(df[c].iloc[0]) if len(df) else ""
            if ";" in sample and sample.split(";")[0].isupper():
                gene_col = c
                break
    if gene_col is None:
        raise KeyError(f"Cannot find gene-name column in CORUM. Columns={list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        genes = str(r[gene_col]).replace(",", ";").split(";")
        genes = [g.strip().upper() for g in genes if g.strip() and g.strip().lower() != "nan"]
        if not genes:
            continue
        cid = r[id_col]
        cname = r[name_col]
        n = len(genes)
        for g in genes:
            rows.append(
                {"gene": g, "complex_id": cid, "complex_name": cname, "n_subunits": n}
            )
    out = pd.DataFrame(rows)
    # per gene: in any complex? max complex size
    agg = out.groupby("gene").agg(
        n_complexes=("complex_id", "nunique"),
        max_complex_size=("n_subunits", "max"),
        example_complex=("complex_name", "first"),
    ).reset_index()
    agg["in_corum"] = True
    print(f"  CORUM: {agg['gene'].nunique()} unique genes in complexes")
    return agg


def load_trrust(cache: Path) -> pd.DataFrame:
    """TF → n_targets from TRRUST human."""
    path = download(TRRUST_URL, cache / "trrust_rawdata.human.tsv", min_bytes=1000)
    df = pd.read_csv(path, sep="\t", header=None)
    # columns: TF, target, mode, PMID (no header in classic file)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["tf", "target"]
    else:
        raise ValueError("Unexpected TRRUST format")
    df["tf"] = df["tf"].astype(str).str.upper()
    df["target"] = df["target"].astype(str).str.upper()
    n_targets = df.groupby("tf")["target"].nunique().rename("n_tf_targets").reset_index()
    n_targets = n_targets.rename(columns={"tf": "gene"})
    print(f"  TRRUST: {len(n_targets)} TFs with annotated targets")
    return n_targets


def try_load_depmap_k562(cache: Path) -> pd.DataFrame | None:
    """Best-effort DepMap gene effect for K562. Returns gene, depmap_gene_effect."""
    dest = cache / "CRISPRGeneEffect.csv"
    try:
        download(DEPMAP_URL, dest, min_bytes=100_000)
    except Exception as e:
        print(f"  DepMap download skipped ({e})")
        print("  Optional: place CRISPRGeneEffect.csv in", cache)
        if not dest.exists():
            return None
    # File can be huge — read only header first to find K562 column
    header = pd.read_csv(dest, nrows=0)
    cols = list(header.columns)
    k562_cols = [c for c in cols if "K562" in str(c).upper()]
    if not k562_cols:
        print(f"  No K562 column in DepMap file (ncols={len(cols)})")
        return None
    col = k562_cols[0]
    gene_col = cols[0]
    print(f"  Reading DepMap column {col!r} …")
    df = pd.read_csv(dest, usecols=[gene_col, col])
    df = df.rename(columns={gene_col: "gene", col: "depmap_gene_effect"})
    df["gene"] = df["gene"].astype(str).str.upper().str.split(" ").str[0].str.split("(").str[0]
    print(f"  DepMap K562: {df['gene'].nunique()} genes")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="Replogle 2022 (CRISPRi)",
        help="Dataset key in Sp CSV (default: Replogle)",
    )
    parser.add_argument(
        "--corum-zip",
        type=Path,
        default=None,
        help="Optional local CORUM .txt or .zip (Zenodo/humanComplexes / allComplexes)",
    )
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    sp_path = find_sp_csv(out_dir, args.input)
    print(f"config_version={cfg.CONFIG_VERSION}")
    print(f"Sp table: {sp_path}")

    df = load_sp_table(sp_path)
    ds = cfg.resolve_dataset_name(args.dataset)
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        # try substring
        sub = df[df["dataset"].str.contains("Replogle", case=False, na=False)].copy()
        ds = sub["dataset"].iloc[0] if len(sub) else ds
    if sub.empty:
        raise SystemExit(f"No rows for dataset {args.dataset!r}. Have: {df['dataset'].unique()}")

    print(f"Using {ds}: n={len(sub)} perturbations")

    corum = load_corum(ANNOT_DIR, local_path=args.corum_zip)
    trrust = load_trrust(ANNOT_DIR)
    depmap = try_load_depmap_k562(ANNOT_DIR)

    merged = sub.merge(corum, on="gene", how="left")
    merged["in_corum"] = merged["in_corum"].fillna(False).astype(bool)
    merged["max_complex_size"] = merged["max_complex_size"].fillna(0)
    merged["n_complexes"] = merged["n_complexes"].fillna(0)
    merged = merged.merge(trrust, on="gene", how="left")
    if depmap is not None:
        merged = merged.merge(depmap, on="gene", how="left")

    # --- Test 1: complex vs non-complex Sp ---
    a = merged.loc[merged["in_corum"], "stability"].dropna()
    b = merged.loc[~merged["in_corum"], "stability"].dropna()
    if len(a) >= 5 and len(b) >= 5:
        u, p = mannwhitneyu(a, b, alternative="less")  # complex lower Sp?
        print("\n=== CORUM complex vs non-complex Sp (H1: complex lower) ===")
        print(f"  complex: n={len(a)} median Sp={a.median():.3f}")
        print(f"  other:   n={len(b)} median Sp={b.median():.3f}")
        print(f"  Mann-Whitney U={u:.1f}  p={p:.3e}")
    else:
        p = np.nan
        print("Insufficient groups for CORUM MWU")

    results = []
    results.append({
        "test": "corum_complex_vs_other_Sp",
        "n_complex": int(len(a)),
        "n_other": int(len(b)),
        "median_Sp_complex": float(a.median()) if len(a) else np.nan,
        "median_Sp_other": float(b.median()) if len(b) else np.nan,
        "mwu_p_complex_lower": float(p) if p == p else np.nan,
        "config_version": cfg.CONFIG_VERSION,
        "dataset": ds,
    })

    # --- Correlations ---
    corr_rows = []
    for label, col in [
        ("max_complex_size", "max_complex_size"),
        ("n_complexes", "n_complexes"),
        ("n_tf_targets", "n_tf_targets"),
        ("depmap_gene_effect", "depmap_gene_effect"),
        ("magnitude", "magnitude"),
        ("n_cells", "n_cells") if "n_cells" in merged.columns else (None, None),
    ]:
        if label is None or col not in merged.columns:
            continue
        m = merged[["stability", col]].dropna()
        if col in ("max_complex_size", "n_complexes", "n_tf_targets"):
            # for TF targets, only TFs; for complex size, only complex members
            if col == "n_tf_targets":
                m = merged.loc[merged["n_tf_targets"].notna(), ["stability", col]].dropna()
            elif col in ("max_complex_size", "n_complexes"):
                m = merged.loc[merged["in_corum"], ["stability", col]].dropna()
        if len(m) < 15:
            print(f"  skip corr {label}: n={len(m)}")
            continue
        boot = bootstrap_spearman_ci(m["stability"], m[col], n_bootstrap=2000, seed=cfg.SEED)
        print(
            f"  Sp ~ {label}: rho={boot['rho']:+.3f} "
            f"[{boot['ci_low']:.3f},{boot['ci_high']:.3f}] n={boot['n']} p={boot['p']:.2e}"
        )
        corr_rows.append({
            "dataset": ds,
            "x": "stability",
            "y": label,
            "n": boot["n"],
            "rho": boot["rho"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "p": boot["p"],
            "config_version": cfg.CONFIG_VERSION,
        })

    # If stress / n_degs columns exist, correlate too
    for col in merged.columns:
        if col.startswith("stress_") or col in ("n_degs", "n_DEG", "n_deg"):
            m = merged[["stability", col]].dropna()
            if len(m) < 15:
                continue
            boot = bootstrap_spearman_ci(m["stability"], m[col], n_bootstrap=2000, seed=cfg.SEED)
            corr_rows.append({
                "dataset": ds, "x": "stability", "y": col, "n": boot["n"],
                "rho": boot["rho"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "p": boot["p"], "config_version": cfg.CONFIG_VERSION,
            })

    merged_path = out_dir / "corum_benchmark_replogle_merged.csv"
    merged.to_csv(merged_path, index=False)
    pd.DataFrame(results).to_csv(out_dir / "corum_benchmark_group_tests.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out_dir / "corum_benchmark_correlations.csv", index=False)
    print(f"\nWrote:\n  {merged_path}\n  corum_benchmark_group_tests.csv\n  corum_benchmark_correlations.csv")


if __name__ == "__main__":
    main()
