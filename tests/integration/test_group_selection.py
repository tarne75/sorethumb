"""Group-selection inputs (only_groups, group_filter_regex,
limit_groups) are validated, and a selector that matches zero of the
groups actually discovered in the data is a distinct, explicit failure --
not a silent no-op recorded as a complete run.

See tests/integration/test_cli.py for the CLI-surfaced versions of these
same scenarios (exit code, --json output, stderr message).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import DetectorConfig
from sorethumb_ml.errors import ConfigError
from sorethumb_ml.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.frames import write_grouped_csv, write_two_period_parquet

pytestmark = pytest.mark.integration


def _grouped_cfg(csv: Path, workdir: Path):
    return make_config(csv, workdir, group_by=["group"], detectors=[DetectorConfig(name="isolation_forest")])


# ---------------------------------------------------------------------------
# A selector matching zero groups is a distinct, explicit failure
# ---------------------------------------------------------------------------


def test_only_groups_matching_nothing_marks_run_failed_not_complete(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    write_grouped_csv(csv, n_rows=300, n_groups=2)  # group labels "G0", "G1"
    cfg = _grouped_cfg(csv, tmp_path / "ws")

    result = run_detection(cfg, only_groups=["does-not-exist"], no_report=True)

    assert result.groups == []
    assert result.group_selection_error is not None
    assert "does-not-exist" in result.group_selection_error
    assert "2 group(s)" in result.group_selection_error

    with Workspace.open(tmp_path / "ws") as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        assert ws.store.run_status(run_id) == "failed"


def test_group_filter_regex_matching_nothing_marks_run_failed(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    write_grouped_csv(csv, n_rows=300, n_groups=2)
    cfg = _grouped_cfg(csv, tmp_path / "ws")

    result = run_detection(cfg, group_filter_regex="^ZZZ_no_match$", no_report=True)

    assert result.groups == []
    assert result.group_selection_error is not None
    assert "ZZZ_no_match" in result.group_selection_error

    with Workspace.open(tmp_path / "ws") as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        assert ws.store.run_status(run_id) == "failed"


def test_only_groups_and_group_filter_regex_both_named_when_both_configured(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    write_grouped_csv(csv, n_rows=300, n_groups=2)
    cfg = _grouped_cfg(csv, tmp_path / "ws")

    result = run_detection(cfg, only_groups=["G0"], group_filter_regex="^ZZZ_no_match$", no_report=True)
    assert result.group_selection_error is not None
    assert "G0" in result.group_selection_error
    assert "ZZZ_no_match" in result.group_selection_error


# ---------------------------------------------------------------------------
# limit_groups must be >= 1 when supplied
# ---------------------------------------------------------------------------


def test_limit_groups_zero_raises_config_error(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    write_grouped_csv(csv, n_rows=300, n_groups=2)
    cfg = _grouped_cfg(csv, tmp_path / "ws")

    with pytest.raises(ConfigError, match="at least 1"):
        run_detection(cfg, limit_groups=0, no_report=True)


def test_limit_groups_negative_raises_config_error(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    write_grouped_csv(csv, n_rows=300, n_groups=2)
    cfg = _grouped_cfg(csv, tmp_path / "ws")

    with pytest.raises(ConfigError, match="at least 1"):
        run_detection(cfg, limit_groups=-1, no_report=True)


# ---------------------------------------------------------------------------
# A genuinely empty source period stays a distinct, non-error outcome
# ---------------------------------------------------------------------------


def test_valid_sparse_period_is_not_a_group_selection_error(tmp_path: Path) -> None:
    """A period with real data but fewer rows than scoring.min_records (no
    selector involved at all) still attempts exactly one group (status
    "too_few_records") and completes normally -- must never be confused
    with a selector matching nothing."""
    rng = np.random.default_rng(0)
    ts_dense = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(100)]
    ts_sparse = [datetime(2024, 1, 16, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(2)]
    n = len(ts_dense) + len(ts_sparse)
    parquet = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "id": list(range(n)),
            "ts": pl.Series(ts_dense + ts_sparse).dt.cast_time_unit("us"),
            "num_a": rng.normal(0.0, 1.0, n).tolist(),
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(parquet))

    cfg = make_config(
        parquet,
        tmp_path / "ws",
        source_format="parquet",
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )
    # 2024-01-16 has only 2 rows, below make_config's min_records=5 default.
    result = run_detection(cfg, no_report=True, period_label_override="2024-01-16")

    assert result.group_selection_error is None
    assert len(result.groups) == 1
    assert result.groups[0].status == "too_few_records"

    with Workspace.open(tmp_path / "ws") as ws:
        assert ws.store.run_status(result.run_id) == "complete"


def test_period_with_truly_zero_rows_raises_a_different_error_than_a_selector_mismatch(
    tmp_path: Path,
) -> None:
    """A period matching *zero* rows outright hits fit_features' own
    "encoded feature matrix has zero columns" guard (profiling a genuinely
    empty frame classifies every column as droppable) -- a pre-existing,
    unrelated limitation these tests do not address. The only thing they
    care about here: this must never be misreported as -- or silently
    swallowed into -- a group_selection_error; it is a different exception
    entirely, raised well before group discovery even runs."""
    parquet = tmp_path / "data.parquet"
    write_two_period_parquet(parquet)  # only 2024-01-15 and 2024-01-16 have data

    cfg = make_config(
        parquet,
        tmp_path / "ws",
        source_format="parquet",
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )
    from sorethumb_ml.errors import PlanError

    with pytest.raises(PlanError):
        run_detection(cfg, no_report=True, period_label_override="2024-01-17")


def test_group_selection_error_writes_no_period_history_marker(tmp_path: Path) -> None:
    """A selector matching zero groups must not leave the period ledger
    thinking this (dataset_fp, period_label, config_hash) is done -- that
    would silently hide a real gap from a later `sorethumb backfill`."""
    n = 60
    rng = np.random.default_rng(0)
    ts = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    pl.DataFrame(
        {
            "id": list(range(n)),
            "group": ["A"] * (n // 2) + ["B"] * (n // 2),
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "num_a": rng.normal(0.0, 1.0, n).tolist(),
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(tmp_path / "data.parquet"))

    cfg = make_config(
        tmp_path / "data.parquet",
        tmp_path / "ws",
        source_format="parquet",
        time_column="ts",
        group_by=["group"],
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )
    result = run_detection(
        cfg, only_groups=["does-not-exist"], no_report=True, period_label_override="2024-01-15"
    )
    assert result.group_selection_error is not None

    with Workspace.open(tmp_path / "ws") as ws:
        dataset_fp = ws.store.get_run(result.run_id)["dataset_fp"]
        assert not ws.store.period_is_complete(dataset_fp, "2024-01-15", cfg.config_hash())
        assert ws.store.completed_group_keys(dataset_fp, "2024-01-15", cfg.config_hash()) == []
