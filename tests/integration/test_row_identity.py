"""Integration tests for P0-4: stable global source-row identity.

Without a configured ``id_column``, the fallback ``row_id`` used to be a
plain positional ``arange(len(df_group))`` recomputed independently inside
*each* group's feature space -- every group's rows started back at 0, so two
different groups' flagged rows could carry the same ``row_id`` and joining
results back to the source frame was ambiguous. The fix stamps one stable
global row index on the raw source frame before any period filter, group
filter, or time sort, and carries it through as the fallback ``row_id``.

These tests use no ``id_column`` (except where noted) so they exercise that
fallback path directly, complementary to the existing id_column-based tests
in ``test_end_to_end.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb._pipeline import run_detection
from sorethumb.config import (
    ColumnsConfig,
    Config,
    DetectorConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb.store.workspace import make_group_key

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grouped_csv(path: Path, *, groups: list[str], per_group: int = 60, seed: int = 0) -> None:
    """CSV with a ``group_by`` column and one planted num_a=999 anomaly per group.

    ``marker`` equals each row's 0-based position in the file -- the same
    order the pipeline sees before any filter or sort -- so a correct fallback
    row_id must always equal its own ``marker`` value.
    """
    rng = np.random.default_rng(seed)
    n = per_group * len(groups)
    cat: list[str] = []
    num_a: list[float] = []
    for g in groups:
        cat.extend([g] * per_group)
        vals = rng.normal(0.0, 1.0, per_group).tolist()
        vals[-1] = 999.0  # plant one obvious anomaly at the end of each group
        num_a.extend(vals)

    df = pl.DataFrame(
        {
            "marker": list(range(n)),
            "cat": cat,
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    )
    df.write_csv(str(path))


def _base_config(source: Path, workdir: Path, *, group_by: list[str] | None = None) -> Config:
    return Config(
        source=SourceConfig(uri=str(source), format="csv" if source.suffix == ".csv" else "parquet"),
        run=RunConfig(workdir=str(workdir), seed=42),
        columns=ColumnsConfig(group_by=group_by or []),
        detectors=[DetectorConfig(name="isolation_forest")],
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
    )


# ---------------------------------------------------------------------------
# Fallback row_id is globally unique and joins back to the source frame
# ---------------------------------------------------------------------------


def test_fallback_row_id_unique_and_joinable_across_groups(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    groups = ["A", "B", "C"]
    _make_grouped_csv(csv, groups=groups, per_group=60)
    cfg = _base_config(csv, tmp_path / "ws", group_by=["cat"])

    result = run_detection(cfg, no_report=True)

    assert result.n_succeeded == len(groups)

    all_row_ids: list[int] = []
    for g in result.groups:
        assert g.results_path is not None
        df_g = pl.read_parquet(g.results_path)
        all_row_ids.extend(df_g["row_id"].to_list())

    assert len(all_row_ids) > 0
    assert len(all_row_ids) == len(set(all_row_ids)), (
        f"row_id collided across groups: {sorted(x for x in set(all_row_ids) if all_row_ids.count(x) > 1)}"
    )

    # row_id must address the exact source row it came from.
    src = pl.read_csv(csv)
    for rid in all_row_ids:
        assert src["marker"][rid] == rid


def test_fallback_row_id_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _make_grouped_csv(csv, groups=["A", "B"], per_group=60)
    cfg = _base_config(csv, tmp_path / "ws", group_by=["cat"])

    first = run_detection(cfg, no_report=True)
    second = run_detection(cfg, force=True, no_report=True)

    def _flagged_row_ids(result) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for g in result.groups:
            assert g.results_path is not None
            out[g.group_key] = set(pl.read_parquet(g.results_path)["row_id"].to_list())
        return out

    assert _flagged_row_ids(first) == _flagged_row_ids(second)


# ---------------------------------------------------------------------------
# Fallback row_id survives an internal time-sort
# ---------------------------------------------------------------------------


def _make_time_sorted_parquet(path: Path, *, n: int = 20, anomaly_file_pos: int = 0, seed: int = 7) -> None:
    """Parquet whose ``ts`` column is in descending file order.

    Ascending time-sort (what the pipeline applies) moves file row 0 to the
    end -- a fallback row_id computed fresh from the post-sort position would
    misreport it. ``marker`` records each row's 0-based file position.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [base + timedelta(days=n - 1 - i) for i in range(n)]
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[anomaly_file_pos] = 999.0

    pl.DataFrame(
        {
            "marker": list(range(n)),
            "ts": pl.Series(timestamps).dt.cast_time_unit("us"),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(path))


def test_fallback_row_id_survives_internal_time_sort(tmp_path: Path) -> None:
    parquet = tmp_path / "data.parquet"
    n = 30
    anomaly_file_pos = 0  # latest timestamp in file order -> last row after ascending sort
    _make_time_sorted_parquet(parquet, n=n, anomaly_file_pos=anomaly_file_pos)

    cfg = Config(
        source=SourceConfig(uri=str(parquet), format="parquet"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(),  # no id_column, no declared time_column -> exercises fallback
        detectors=[DetectorConfig(name="isolation_forest")],
        scoring=ScoringConfig(combination="composite", contamination=0.07, weighting="equal", min_records=5),
    )
    result = run_detection(cfg, no_report=True)

    assert result.n_anomalies > 0
    group = result.groups[0]
    assert group.results_path is not None
    df_anomalies = pl.read_parquet(group.results_path)

    # The planted anomaly's row_id must be its original file position, not its
    # position after the internal ascending time-sort moved it to the end.
    assert anomaly_file_pos in df_anomalies["row_id"].to_list()


# ---------------------------------------------------------------------------
# Null groups: canonical identity distinct from "" and the literal "None"
# ---------------------------------------------------------------------------


def test_null_group_processed_distinctly_from_empty_and_literal_none(tmp_path: Path) -> None:
    per_group = 60
    n = per_group * 3
    rng = np.random.default_rng(3)

    cat: list[str | None] = [None] * per_group + [""] * per_group + ["None"] * per_group
    num_a = rng.normal(0.0, 1.0, n).tolist()

    csv = tmp_path / "data.csv"
    pl.DataFrame(
        {
            "marker": list(range(n)),
            "cat": cat,
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_csv(str(csv))

    cfg = _base_config(csv, tmp_path / "ws", group_by=["cat"])
    result = run_detection(cfg, no_report=True)

    assert result.n_succeeded == 3, f"expected 3 distinct groups, got {result.groups}"

    by_key = {g.group_key: g for g in result.groups}
    expected_keys = {
        make_group_key({"cat": None}): "null",
        make_group_key({"cat": ""}): "empty string",
        make_group_key({"cat": "None"}): "literal 'None'",
    }
    assert set(by_key) == set(expected_keys), (
        f"group keys did not match the expected typed identities: got {set(by_key)}, "
        f"expected {set(expected_keys)}"
    )
    for key, label in expected_keys.items():
        assert by_key[key].n_records == per_group, f"{label} group had wrong record count"
        assert by_key[key].status == "success"
