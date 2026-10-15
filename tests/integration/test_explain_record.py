"""``attributions.parquet``: the long-format record behind every reason, end to end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import app
from sorethumb_ml.config import (
    ColumnsConfig,
    DetectorConfig,
    ExplainConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb_ml.explain.record import EXPLAINER_VERSION
from sorethumb_ml.store.results import attributions_path, read_attributions, remove_explanation_outputs
from sorethumb_ml.store.workspace import Workspace
from tests.factories.frames import write_planted_csv
from tests.factories.results import joined_results

pytestmark = pytest.mark.integration

runner = CliRunner()


def _cfg(tmp_path: Path, detectors: list[str], **explain: object) -> Config:
    csv = tmp_path / "data.csv"
    if not csv.exists():
        write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=7)
    return Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name=n) for n in detectors],
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
        explain=ExplainConfig(top_n=2, **explain),  # type: ignore[arg-type]
    )


def _records(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame, object]:
    result = run_detection(cfg, no_report=True)
    group = result.groups[0]
    assert group.results_path is not None
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        records = read_attributions(ws, result.run_id, group.group_key)
    assert records is not None
    return joined_results(group.results_path), records, group


def test_every_reason_has_a_record_that_reproduces_its_label(tmp_path: Path) -> None:
    rows, records, _ = _records(_cfg(tmp_path, ["isolation_forest", "ecod"]))
    flagged = rows.filter(pl.col("flagged"))
    explained = flagged.filter(pl.col("explain_status") != "unavailable")
    assert explained.height > 0
    assert records.height == explained.height * 2  # top_n=2 reasons each
    assert set(explained["explainer_version"].to_list()) == {EXPLAINER_VERSION}

    by_key = {(r["row_id"], r["reason_rank"]): r for r in records.iter_rows(named=True)}
    for row in explained.iter_rows(named=True):
        for k in (1, 2):
            rec = by_key[(row["row_id"], k)]
            assert row[f"reason_{k}"] == f"{rec['original_field']}={rec['raw_value']}"

    for rec in records.iter_rows(named=True):
        assert rec["explainer_version"] == EXPLAINER_VERSION
        assert rec["reference_id"] == "unflagged_rows"
        assert {m["detector"] for m in rec["methods"]} == {"isolation_forest", "ecod"}
        assert {m["method"] for m in rec["methods"]} <= {
            "tree_shap",
            "finite_difference_gradient",
            "ecod_decomposition",
        }
        total = sum(m["weight"] * m["signed_contribution"] for m in rec["methods"])
        assert rec["signed_contribution"] == pytest.approx(total, abs=1e-9)
        assert rec["transformed_features"]

    # The planted anomalies (num_a=999) are explained by num_a, above the reference median.
    planted = records.filter((pl.col("row_id") >= 180) & (pl.col("reason_rank") == 1))
    assert planted.height > 0
    assert set(planted["original_field"].to_list()) == {"num_a"}
    assert set(planted["direction"].to_list()) == {"above"}
    assert all(float(v) < 5 for v in planted["reference_value"].to_list())
    assert (planted["signed_contribution"] > 0).all()

    ranked = records.sort(["row_id", "reason_rank"]).group_by("row_id", maintain_order=True)
    for _, grp in ranked:
        mags = grp["contribution_magnitude"].to_list()
        assert mags == sorted(mags, reverse=True)


def test_tree_shap_and_centroid_caveats_are_flagged(tmp_path: Path) -> None:
    _, records, _ = _records(_cfg(tmp_path, ["isolation_forest", "kmeans_distance"]))
    flags = {f for fl in records["approximation_flags"].to_list() for f in fl}
    methods = {m["method"] for ms in records["methods"].to_list() for m in ms}
    assert "unsigned_method:kmeans_distance" in flags
    if "tree_shap" in methods:
        assert {"treeshap_additivity_unchecked", "treeshap_path_dependent"} <= flags
    else:  # shap not installed in this environment
        assert "fallback_used:isolation_forest" in flags


def test_a_row_beyond_one_detectors_max_rows_is_partial_not_ok(tmp_path: Path) -> None:
    rows, records, group = _records(_cfg(tmp_path, ["ecod", "one_class_svm"], max_rows=1))
    flagged = rows.filter(pl.col("flagged")).sort("rank")
    assert flagged.height > 1
    beyond = flagged.slice(1)
    assert set(beyond["explain_status"].to_list()) == {"partial"}
    assert set(beyond["explain_issue"].to_list()) == {"detectors_unexplained:one_class_svm"}
    beyond_records = records.filter(pl.col("row_id").is_in(beyond["row_id"].to_list()))
    assert all("not_covered_by:one_class_svm" in f for f in beyond_records["approximation_flags"].to_list())
    assert group.explain_status == "partial"  # type: ignore[attr-defined]


def test_rows_no_source_says_anything_about_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sorethumb_ml.explain.gradient as gradient

    def _flat(_det: object, x: np.ndarray, **_k: object) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(x.shape, dtype=np.float64), np.zeros(x.shape[0], dtype=bool)

    monkeypatch.setattr(gradient, "gradient_attributions_detailed", _flat)
    cfg = _cfg(tmp_path, ["one_class_svm"])
    result = run_detection(cfg, no_report=True)
    group = result.groups[0]
    assert group.results_path is not None
    rows = joined_results(group.results_path).filter(pl.col("flagged"))
    assert set(rows["explain_status"].to_list()) == {"unavailable"}
    assert set(rows["explain_issue"].to_list()) == {"no_attribution_signal"}
    assert rows["reason_1"].null_count() == rows.height
    assert group.explain_status == "failed"
    assert "explain.no_signal" in {w["code"] for w in group.warning_records}
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        records = read_attributions(ws, result.run_id, group.group_key)
    assert records is not None
    assert records.height == 0


def test_a_failed_rerun_leaves_no_stale_explanation_files(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, ["ecod"])
    result = run_detection(cfg, no_report=True)
    key = result.groups[0].group_key
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        path = attributions_path(ws, result.run_id, key)
        assert path.exists()
        assert path.with_name("explanations.parquet").exists()
        remove_explanation_outputs(ws, result.run_id, key)
        assert not path.exists()
        assert not path.with_name("explanations.parquet").exists()
        assert read_attributions(ws, result.run_id, key) is None
        remove_explanation_outputs(ws, result.run_id, key)  # idempotent


def test_anomalies_detail_shows_the_record(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, ["isolation_forest", "ecod"])
    run_detection(cfg, no_report=True)
    ws = str(tmp_path / "ws")

    res = runner.invoke(
        app, ["anomalies", "--workdir", ws, "--json", "--detail", "--top", "3", "--reasons", "1"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert len(payload) == 3
    for item in payload:
        detail = item["reason_detail"]
        if item["explain_status"] == "unavailable":
            assert detail == []
            continue
        assert len(detail) == 1
        assert item["reason_1"] == f"{detail[0]['original_field']}={detail[0]['raw_value']}"
        assert detail[0]["methods"]

    plain = runner.invoke(app, ["anomalies", "--workdir", ws, "--json", "--top", "1"])
    assert "reason_detail" not in json.loads(plain.output)[0]

    table = runner.invoke(
        app, ["anomalies", "--workdir", ws, "--detail", "--top", "2"], env={"COLUMNS": "250"}
    )
    assert table.exit_code == 0, table.output
    assert "Reason detail" in table.output
    assert "num_a" in table.output


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (pl.DataFrame({"reason_rank": [1]}), "missing required column 'row_id'"),
        (pl.DataFrame({"row_id": [1, None], "reason_rank": [1, 1]}), "null 'row_id'"),
        (pl.DataFrame({"row_id": [1, 1], "reason_rank": [1, 1]}), "repeated"),
    ],
)
def test_write_attributions_rejects_malformed_frames(tmp_path: Path, frame: pl.DataFrame, match: str) -> None:
    from sorethumb_ml.errors import StoreError
    from sorethumb_ml.store.results import write_attributions

    with Workspace.init(tmp_path / "ws") as ws, pytest.raises(StoreError, match=match):
        write_attributions(ws, "run_x", "g", frame)
