"""Equal composite scores are ordered by earliest *source* row, end to end.

The group frame is time-sorted before scoring (descending timestamps in the file make
position order the exact reverse of source order), and twelve identical anomalous rows
tie exactly. Which of them are flagged, how they rank, and which ones ``explain.max_rows``
explains must all follow source order, not array position or a sort accident.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import (
    ColumnsConfig,
    DetectorConfig,
    ExplainConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb_ml.scoring.combine import ScoreEnsemble

pytestmark = pytest.mark.integration

_N = 200


def _write(path: Path) -> None:
    """Integer-valued features: every (num_a, num_b) pair is a tie group of identical rows."""
    rng = np.random.default_rng(5)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    pl.DataFrame(
        {
            "id": list(range(_N)),
            # Descending in file order: the pipeline's time sort reverses source order.
            "ts": pl.Series([base + timedelta(minutes=_N - 1 - i) for i in range(_N)]).dt.cast_time_unit(
                "us"
            ),
            "num_a": rng.integers(0, 5, _N).astype(float).tolist(),
            "num_b": rng.integers(0, 3, _N).astype(float).tolist(),
        }
    ).write_parquet(str(path))


def _cfg(tmp_path: Path, *, contamination: float, max_rows: int = 5000) -> Config:
    return Config(
        source=SourceConfig(uri=str(tmp_path / "d.parquet"), format="parquet"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=7),
        columns=ColumnsConfig(id_column="id", time_column="ts"),
        detectors=[DetectorConfig(name="one_class_svm")],
        scoring=ScoringConfig(
            combination="composite", contamination=contamination, weighting="equal", min_records=5
        ),
        explain=ExplainConfig(max_rows=max_rows, top_n=2),
    )


def _results(cfg: Config) -> pl.DataFrame:
    result = run_detection(cfg, no_report=True, period_label_override="2024-01-01")
    assert result.n_failed == 0
    path = result.groups[0].results_path
    assert path is not None
    return pl.read_parquet(path)


def _capture_combine(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Record the pipeline's real ensemble inputs/outputs (the full score vector is not persisted)."""
    captured: dict[str, np.ndarray] = {}
    real = ScoreEnsemble.combine

    def spy(self: ScoreEnsemble, scores: dict, flags: dict, source_row: np.ndarray | None = None) -> dict:
        out = real(self, scores, flags, source_row=source_row)
        assert source_row is not None, "the pipeline must pass the source row ids for tie-breaking"
        captured.update(
            source_row=np.asarray(source_row), score=out["combined_score"], flag=out["anomaly_flag"]
        )
        return out

    monkeypatch.setattr(ScoreEnsemble, "combine", spy)
    return captured


def test_flag_boundary_inside_a_tie_takes_the_earliest_source_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "d.parquet")
    captured = _capture_combine(monkeypatch)
    _results(_cfg(tmp_path, contamination=0.1))

    source, score, flag = captured["source_row"], captured["score"], captured["flag"]
    boundary = score[flag].min()
    tied = score == boundary
    flagged_rows = source[tied & flag]
    unflagged_rows = source[tied & ~flag]
    assert len(flagged_rows) >= 1
    assert len(unflagged_rows) >= 1, "fixture: the flag boundary must fall inside a tie"
    # The group is time-sorted (descending timestamps), so array position runs opposite to source order:
    # a position-based tie-break would flag the *latest* source rows here.
    assert source[0] > source[-1]
    assert flagged_rows.max() < unflagged_rows.min(), (sorted(flagged_rows), sorted(unflagged_rows))


def test_rank_within_every_tie_group_follows_source_row(tmp_path: Path) -> None:
    _write(tmp_path / "d.parquet")
    df = _results(_cfg(tmp_path, contamination=0.15))

    flagged = df.filter(pl.col("flagged")).sort("rank")
    assert flagged["rank"].to_list() == list(range(1, flagged.height + 1))
    # Scores never increase down the ranking, and within equal scores source rows ascend.
    pairs = list(zip(flagged["composite_score"].to_list(), flagged["row_id"].to_list(), strict=True))
    for (s1, r1), (s2, r2) in pairwise(pairs):
        assert s1 >= s2
        if s1 == s2:
            assert r1 < r2, f"tied rows {r1} and {r2} ranked out of source order"
    assert any(s1 == s2 for (s1, _), (s2, _) in pairwise(pairs)), "fixture must contain ties"


def test_explain_max_rows_keeps_the_earliest_source_rows_of_a_tie(tmp_path: Path) -> None:
    _write(tmp_path / "d.parquet")
    df = _results(_cfg(tmp_path, contamination=0.15, max_rows=8))

    flagged = df.filter(pl.col("flagged")).sort("rank")
    kinds = flagged["attribution_kind"].to_list()
    assert all(k != "unavailable" for k in kinds[:8])
    assert all(k == "unavailable" for k in kinds[8:])
    # The cut falls inside a tie group here, and the explained members of that group are the earliest rows.
    scores = flagged["composite_score"].to_list()
    ids = flagged["row_id"].to_list()
    assert scores[7] == scores[8], "fixture: max_rows must cut through a tie"
    cut_group = [i for sc, i in zip(scores, ids, strict=True) if sc == scores[7]]
    explained = set(ids[:8])
    assert [i for i in cut_group if i in explained] == sorted(cut_group)[
        : len([i for i in cut_group if i in explained])
    ]
