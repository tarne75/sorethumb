"""Helpers that produce a completed, persisted pipeline run.

Collapses the common "write a planted-anomaly CSV, build a Config, run
detection" sequence that recurs across the end-to-end and score-forward
suites into one call, returning a named result object rather than a
positional tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sorethumb._pipeline import run_detection

from .configs import make_config
from .frames import write_planted_csv

if TYPE_CHECKING:
    from sorethumb._pipeline import RunResult


@dataclass
class PlantedRun:
    """The result of running detection over a freshly written planted-anomaly CSV."""

    result: RunResult
    csv_path: Path
    workdir: Path
    anomaly_indices: list[int]


def run_planted_detection(
    tmp_path: Path,
    *,
    n_normal: int = 280,
    n_anomaly: int = 5,
    seed: int = 0,
    csv_name: str = "data.csv",
    workdir_name: str = "ws",
    no_report: bool = True,
    **config_kwargs: object,
) -> PlantedRun:
    """Write a planted-anomaly CSV under ``tmp_path``, build a ``Config`` for
    it, and run detection. Extra keyword arguments pass through to
    ``make_config`` (e.g. ``contamination``, ``combination``, ``detectors``).
    """
    csv_path = tmp_path / csv_name
    anomaly_indices = write_planted_csv(csv_path, n_normal=n_normal, n_anomaly=n_anomaly, seed=seed)
    workdir = tmp_path / workdir_name
    cfg = make_config(csv_path, workdir, **config_kwargs)  # type: ignore[arg-type]
    result = run_detection(cfg, no_report=no_report)
    return PlantedRun(result=result, csv_path=csv_path, workdir=workdir, anomaly_indices=anomaly_indices)
