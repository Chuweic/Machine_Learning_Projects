from __future__ import annotations
import json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

SEED = 42

@dataclass
class Paths:
    root: Path
    reports: Path
    figures: Path
    results: Path

    @staticmethod
    def from_root(root: str | os.PathLike = ".") -> "Paths":
        root = Path(root)
        reports = root / "reports"
        figures = reports / "figures"
        results = reports / "results"
        for p in (reports, figures, results):
            p.mkdir(parents=True, exist_ok=True)
        return Paths(root, reports, figures, results)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: str | os.PathLike, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_bars(values: Iterable[float],
              labels: Iterable[str],
              out_path: str,
              title: Optional[str] = None) -> None:
    """Generic bar plot utility."""
    values = np.asarray(list(values))
    labels = np.asarray(list(labels))
    idx = np.argsort(values)[::-1]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(values)), values[idx])
    plt.xticks(range(len(values)), labels[idx], rotation=45, ha="right")
    if title:
        plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
