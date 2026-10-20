"""The uninformative-boundary guard through a real run and a score-forward.

On a frame of uniform low-cardinality categoricals, Isolation Forest's own boundary
flags every row and One-Class SVM's sits on every row. The fit-time share is
recorded in each model's manifest, so a score-forward run applies the same verdict
instead of judging the boundary on new data (where a majority outside it would be
drift, which the boundary should report).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml import score_forward
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import DetectorConfig
from sorethumb_ml.errors import UninformativeBoundaryWarning
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration

_DEFAULT = ("isolation_forest", "kmeans_distance", "one_class_svm")


def _categorical_csv(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    pl.DataFrame(
        {"colour": rng.choice(["red", "green", "blue"], 400), "size": rng.choice(["s", "m", "l", "xl"], 400)}
    ).write_csv(path)
    return path


def _config(csv: Path, ws: Path, names: tuple[str, ...] = _DEFAULT):
    return make_config(csv, ws, id_column=None, detectors=[DetectorConfig(name=n) for n in names])


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UninformativeBoundaryWarning)
        return run_detection(cfg, no_report=True)


def test_isolation_forest_alone_no_longer_flags_every_categorical_row(tmp_path: Path) -> None:
    csv = _categorical_csv(tmp_path / "c.csv", seed=1)
    result = _run(_config(csv, tmp_path / "ws", ("isolation_forest",)))
    (group,) = result.groups

    assert group.detector_flag_rates["isolation_forest"] == 1.0  # its boundary, as it was
    assert group.abstaining_detectors == ["isolation_forest"]
    assert group.n_anomalies == 0  # was 400: every row
    codes = {r["code"] for r in group.warning_records}
    assert "scoring.boundary_uninformative" in codes
    assert len(group.top_ranked) > 0  # still ranked


def test_the_default_detectors_flag_by_the_boundaries_that_can_vote(tmp_path: Path) -> None:
    csv = _categorical_csv(tmp_path / "c.csv", seed=1)
    result = _run(_config(csv, tmp_path / "ws"))
    (group,) = result.groups
    rates = group.detector_flag_rates

    assert set(group.abstaining_detectors) == {d for d, r in rates.items() if r > 0.5}
    assert "isolation_forest" in group.abstaining_detectors
    assert "kmeans_distance" not in group.abstaining_detectors
    # Only KMeans votes here, so the run flags exactly its rows.
    assert group.n_anomalies == round(rates["kmeans_distance"] * group.n_records)


def test_the_fit_time_rate_is_recorded(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    fitted = _run(_config(_categorical_csv(tmp_path / "fit.csv", seed=1), ws))
    (group,) = fitted.groups
    manifest = json.loads(
        (ws / "models" / fitted.run_id / group.group_key / "isolation_forest.manifest.json").read_text()
    )
    assert manifest["boundary_flag_rate"] == 1.0


def test_score_forward_takes_the_verdict_from_the_fit_not_from_the_new_rows(tmp_path: Path) -> None:
    # Ordinary data: Isolation Forest's boundary flags a minority at fit and when scoring.
    ws = tmp_path / "ws"
    csv = tmp_path / "d.csv"
    write_planted_csv(csv)
    fitted = _run(_config(csv, ws))
    (group,) = fitted.groups
    assert fitted.groups[0].abstaining_detectors == []
    # Record the fit as uninformative: the score-forward must follow the manifest,
    # although the boundary flags a minority of the rows it now scores.
    path = ws / "models" / fitted.run_id / group.group_key / "isolation_forest.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["boundary_flag_rate"] = 0.9
    path.write_text(json.dumps(manifest))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UninformativeBoundaryWarning)
        scored = score_forward(_config(csv, ws), fitted.run_id, no_report=True)
    (sgroup,) = scored.groups
    assert sgroup.detector_flag_rates["isolation_forest"] < 0.5
    assert sgroup.abstaining_detectors == ["isolation_forest"]


def test_a_manifest_from_before_the_guard_keeps_its_vote(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    fitted = _run(_config(_categorical_csv(tmp_path / "fit.csv", seed=1), ws, ("isolation_forest",)))
    (group,) = fitted.groups
    path = ws / "models" / fitted.run_id / group.group_key / "isolation_forest.manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["boundary_flag_rate"]
    path.write_text(json.dumps(manifest))

    scored = score_forward(
        _config(tmp_path / "fit.csv", ws, ("isolation_forest",)), fitted.run_id, no_report=True
    )
    (sgroup,) = scored.groups
    assert sgroup.abstaining_detectors == []
    assert sgroup.n_anomalies == 400  # the old behaviour, unchanged for old models


def test_ordinary_data_is_unaffected(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    write_planted_csv(csv)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UninformativeBoundaryWarning)
        result = run_detection(_config(csv, tmp_path / "ws"), no_report=True)
    (group,) = result.groups
    assert group.abstaining_detectors == []
    assert group.n_anomalies > 0
