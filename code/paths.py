"""Put every ``code/`` subdirectory on ``sys.path``.

Scripts live one level below ``code/`` (``figures/``, ``utils/``, …) and still
import each other as a flat module set:

    python figures/fig2_magnitude_stability_loess.py
    python utils/run_frozen_main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent

# Prefer shared modules when names collide.
_PATH_ORDER = ("utils", "figures", "competitors", "pathways", "embeddings", "efficiency", "robustness", "datasets")


def add_code_paths() -> Path:
    ordered = [CODE_ROOT]
    for name in _PATH_ORDER:
        child = CODE_ROOT / name
        if child.is_dir():
            ordered.append(child)
    for child in sorted(CODE_ROOT.iterdir()):
        if child.is_dir() and child not in ordered and not child.name.startswith(".") and child.name != "__pycache__":
            ordered.append(child)
    for path in reversed(ordered):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return CODE_ROOT


add_code_paths()
