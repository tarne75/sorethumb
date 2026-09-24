"""Cross-layer contract: `Config.config_hash()` is the ONE canonical value

meant by every "config_hash" column/field across the store and the public
API -- the `run` table row, `totals`, `period_execution`, and `RunResult`.

Before P0-6, `Store.insert_run` independently hashed the full (redacted)
`config_json` string and stored *that* under the `run.config_hash` column,
while `totals`/`period_execution`/`RunResult`/`run_id` itself all used
`Config.config_hash()` (which deliberately excludes purely cosmetic or
execution-only fields, e.g. `run.workdir`, `run.log_level`, the entire
`report` section). Same column name, two different values for the same
execution -- breaking any join or comparison across those tables. There is
no dedicated "did the bug come back" regression test possible other than
this: assert every layer agrees, for real runs written through the real
pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb._pipeline import run_detection
from sorethumb.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.contract


def _period_parquet(path: Path, *, per_day: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(per_day)]
    pl.DataFrame(
        {
            "id": list(range(per_day)),
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "num_a": rng.normal(0.0, 1.0, per_day).tolist(),
            "num_b": rng.normal(5.0, 2.0, per_day).tolist(),
        }
    ).write_parquet(str(path))


def test_config_hash_agrees_across_run_row_totals_and_period_execution(tmp_path: Path) -> None:
    ws_path = tmp_path / "ws"
    parquet = tmp_path / "data.parquet"
    _period_parquet(parquet, per_day=100)

    cfg = make_config(
        parquet,
        ws_path,
        source_format="parquet",
        combination="composite",
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )
    result = run_detection(cfg, no_report=True, period_label_override="2024-01-15")
    assert result.n_succeeded >= 1
    assert result.period_label is not None

    canonical = cfg.config_hash()
    assert result.config_hash == canonical

    with Workspace.open(ws_path) as ws:
        run_row = ws.store.get_run(result.run_id)
        assert run_row is not None
        assert run_row["config_hash"] == canonical

        dataset_fp = run_row["dataset_fp"]
        assert ws.store.period_is_complete(dataset_fp, result.period_label, canonical)

        group_keys = ws.store.completed_group_keys(dataset_fp, result.period_label, canonical)
        assert group_keys, "no totals row found under the canonical hash -- config_hash values disagree"

        totals = ws.store.totals_for_periods(dataset_fp, [result.period_label], canonical)
        assert totals, "no totals row found under the canonical hash -- config_hash values disagree"


def test_config_hash_and_run_id_are_unaffected_by_execution_only_or_report_settings(tmp_path: Path) -> None:
    """run.workdir/log_level/slow_stage_seconds/reuse_models and the entire
    report section are excluded from Config.config_hash() by design (purely
    cosmetic or execution-only, never result-affecting) -- two configs that
    differ only there must hash identically and therefore resolve to the
    very same run_id (the ordinary "no-op resume" path), not two supposedly
    distinct runs of the "same" configuration."""
    ws_path = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=6, seed=0)

    cfg_a = make_config(
        csv,
        ws_path,
        run_kwargs={"log_level": "DEBUG"},
        report_kwargs={"formats": ["html"]},
    )
    cfg_b = make_config(
        csv,
        ws_path,
        run_kwargs={"log_level": "WARNING", "slow_stage_seconds": 999.0},
        report_kwargs={"formats": ["json", "csv"]},
    )
    assert cfg_a.config_hash() == cfg_b.config_hash()

    result_a = run_detection(cfg_a, no_report=True)
    result_b = run_detection(cfg_b, no_report=True)
    assert result_b.run_id == result_a.run_id
    assert result_b.config_hash == result_a.config_hash

    with Workspace.open(ws_path) as ws:
        assert ws.store.get_run(result_a.run_id)["config_hash"] == cfg_a.config_hash()
