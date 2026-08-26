"""Shared manuscript style and CSV search paths."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CODE_ROOT))
import paths as _code_paths  # noqa: F401

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from revision_io import data_search_dirs, find_data_file, resolve_out_dir

# Bar-chart theme (blue / grey / salmon / green).
BLUE = "#4C72B0"
GREY = "#AAAAAA"
SALMON = "#D67A6B"
GREEN = "#55A868"
DARK = "#333333"
GATE = "#888888"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

SEARCH_DIRS = data_search_dirs()

DATASETS = [
    ("Norman 2019 (CRISPRa)",        "Norman",        "CRISPRa",   BLUE),
    ("Adamson 2016 UPR (CRISPRi)",   "Adamson UPR",   "CRISPRi",   GREEN),
    ("Adamson 2016 pilot (CRISPRi)", "Adamson pilot", "CRISPRi",   "#8172B2"),
    ("Dixit 2016 (CRISPR-KO)",       "Dixit",         "CRISPR-KO", GREEN),
    ("Papalexi 2021 (CRISPR-KO)",    "Papalexi",      "CRISPR-KO", SALMON),
    ("Replogle 2022 (CRISPRi)",      "Replogle",      "CRISPRi",   "#C44E52"),
]

SCOREABLE = [
    "Norman 2019 (CRISPRa)",
    "Adamson 2016 UPR (CRISPRi)",
    "Dixit 2016 (CRISPR-KO)",
    "Papalexi 2021 (CRISPR-KO)",
    "Replogle 2022 (CRISPRi)",
]


def find_csv(*names):
    return find_data_file(*names)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, stem: Path):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {stem}.pdf / .png")
    plt.close(fig)


def grouped_x(n_groups, n_series, width=0.22):
    centers = np.arange(n_groups)
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width
    return centers, offsets
