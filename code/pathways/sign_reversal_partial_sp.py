#!/usr/bin/env python3
"""
Sign reversal in partial Sp–stress / Sp–magnitude.

Norman ≈ −0.859 vs Dixit ≈ +0.627 for the same
partial correlation. With Dixit correctly labeled CRISPR-KO, this is a
modality / context difference (CRISPRa vs KO), not a within-CRISPRi discrepancy.

Recomputes from frozen Sp (+ stress columns if present) with rank-based
partial Spearman and writes a short addressing blurb.

Usage:
  python sign_reversal_partial_sp.py \\
      --input shesha-crispr/shesha_crispr_results_euclidean.csv
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
import pandas as pd

import pipeline_config as cfg
from revision_io import find_sp_csv, load_sp_table, resolve_out_dir
from stats_utils import bootstrap_partial_spearman_ci, bootstrap_spearman_ci

FOCUS = [
    "Norman 2019 (CRISPRa)",
    "Dixit 2016 (CRISPR-KO)",
    "Replogle 2022 (CRISPRi)",
    "Adamson 2016 UPR (CRISPRi)",
    "Papalexi 2021 (CRISPR-KO)",
]

STRESS_MARKERS = ["DDIT3", "ATF4", "XBP1", "HSPA5"]


def _stress_cols(df: pd.DataFrame) -> list[tuple[str, str]]:
    found = []
    for m in STRESS_MARKERS:
        for cand in (f"stress_{m}", m, f"{m}_expr"):
            if cand in df.columns:
                found.append((m, cand))
                break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    path = find_sp_csv(out_dir, args.input)
    df = load_sp_table(path)
    stress = _stress_cols(df)

    rows = []
    for ds in sorted(df["dataset"].dropna().unique()):
        sub = df[df["dataset"] == ds]
        modal = cfg.DATASETS.get(ds, {}).get("modality", "?")
        cell = cfg.DATASETS.get(ds, {}).get("cell_type", "?")
        sm = sub[["stability", "magnitude"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sm) < 15:
            continue
        raw = bootstrap_spearman_ci(
            sm["stability"], sm["magnitude"],
            n_bootstrap=args.n_bootstrap, seed=cfg.SEED,
        )
        rows.append({
            "dataset": ds,
            "modality": modal,
            "cell_type": cell,
            "test": "Sp_vs_magnitude",
            "marker": "",
            "n": raw["n"],
            "rho": raw["rho"],
            "p": raw["p"],
            "ci_low": raw["ci_low"],
            "ci_high": raw["ci_high"],
        })
        for marker, col in stress:
            m = sub[["stability", "magnitude", col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(m) < 15:
                continue
            part = bootstrap_partial_spearman_ci(
                m["stability"], m[col], m["magnitude"],
                n_bootstrap=args.n_bootstrap, seed=cfg.SEED,
            )
            rows.append({
                "dataset": ds,
                "modality": modal,
                "cell_type": cell,
                "test": "Sp_vs_stress_partial_magnitude",
                "marker": marker,
                "n": part["n"],
                "rho": part["rho_partial"],
                "p": part["p"],
                "ci_low": part["ci_low"],
                "ci_high": part["ci_high"],
            })

    out = pd.DataFrame(rows)
    out_path = out_dir / "sign_reversal_partial_sp.csv"
    out.to_csv(out_path, index=False)

    # Highlight Norman vs Dixit for DDIT3 (or first available marker) and Sp~mag
    focus_ds = [d for d in FOCUS if d in set(out["dataset"])]
    highlight = out[out["dataset"].isin(focus_ds)].copy()

    norman = highlight[highlight["dataset"].str.contains("Norman", case=False)]
    dixit = highlight[highlight["dataset"].str.contains("Dixit", case=False)]

    def _pick(frame, test, marker=""):
        m = frame[(frame["test"] == test) & (frame["marker"] == marker)]
        if m.empty and marker:
            m = frame[frame["test"] == test]
        return m.iloc[0].to_dict() if len(m) else None

    pairs = []
    for test, marker in [
        ("Sp_vs_magnitude", ""),
        ("Sp_vs_stress_partial_magnitude", "DDIT3"),
        ("Sp_vs_stress_partial_magnitude", "ATF4"),
    ]:
        n = _pick(norman, test, marker)
        d = _pick(dixit, test, marker)
        if n and d:
            pairs.append({
                "test": test,
                "marker": marker or None,
                "norman_rho": n["rho"],
                "dixit_rho": d["rho"],
                "sign_flip": bool(np.sign(n["rho"]) != np.sign(d["rho"])),
                "norman_modality": n["modality"],
                "dixit_modality": d["modality"],
            })

    blurb = (
        "Sign reversal (revision). Partial correlations involving Sp are not "
        "sign-consistent across modalities. Norman 2019 is CRISPRa in K562; "
        "Dixit 2016 is CRISPR-KO in BMDCs (not CRISPRi). Opposite signs "
        "(historically Table S14: Norman ≈ −0.86 vs Dixit ≈ +0.63) therefore "
        "reflect gene × modality × cell-type context, not a within-CRISPRi "
        "discrepancy. We do not advance a single 'geometric tax' interpretation "
        "across activation and knockout."
    )
    if pairs:
        bits = []
        for p in pairs:
            lab = p["test"] + (f"/{p['marker']}" if p["marker"] else "")
            bits.append(
                f"{lab}: Norman ρ={p['norman_rho']:.3f} vs Dixit ρ={p['dixit_rho']:.3f}"
                f"{' (sign flip)' if p['sign_flip'] else ''}"
            )
        blurb += " Frozen recomputation: " + "; ".join(bits) + "."

    if not stress:
        blurb += (
            " Note: stress_* columns were absent from the input CSV; "
            "Sp~magnitude was recomputed. Run attach_stress_markers.py for "
            "partial Sp~stress tests."
        )

    summary = {
        "source_csv": str(path),
        "config_version": cfg.CONFIG_VERSION,
        "n_stress_markers_found": len(stress),
        "stress_markers": [m for m, _ in stress],
        "norman_vs_dixit": pairs,
        "methods_blurb": blurb,
    }

    print(blurb)
    print(f"\nWrote {out_path}")
    with open(out_dir / "sign_reversal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "sign_reversal_blurb.txt", "w") as f:
        f.write(blurb + "\n")


if __name__ == "__main__":
    main()
