"""columns.include_only: documented since the first release, honoured from this one on."""

from __future__ import annotations

import polars as pl
import pytest

from sorethumb_ml.config import ColumnsConfig, Config, RunConfig, SourceConfig
from sorethumb_ml.errors import ConfigError
from sorethumb_ml.profiling.classify import ColumnClass
from sorethumb_ml.profiling.plan import build_feature_plan
from sorethumb_ml.profiling.temporal import detect_temporal_parses

pytestmark = pytest.mark.unit


def _cfg(tmp_path: object, **columns_kw: object) -> Config:
    return Config(
        source=SourceConfig(uri="/dev/null"),
        run=RunConfig(workdir=str(tmp_path)),
        columns=ColumnsConfig(**columns_kw),  # type: ignore[arg-type]
    )


def _frame() -> pl.DataFrame:
    n = 60
    return pl.DataFrame(
        {
            "a": [float(i % 7) for i in range(n)],
            "b": [float(i % 5) for i in range(n)],
            "c": [float(i % 3) for i in range(n)],
            "site": ["x", "y"] * (n // 2),
        }
    )


def test_only_listed_columns_become_features(tmp_path: object) -> None:
    plan = build_feature_plan(_frame(), _cfg(tmp_path, include_only=["a"]))
    by_col = {d.column: d for d in plan.decisions}
    assert by_col["a"].col_class == ColumnClass.numeric
    for col in ("b", "c", "site"):
        assert by_col[col].col_class == ColumnClass.ignored
        assert by_col[col].reason == "not listed in columns.include_only"
    assert plan.output_features == ["a"]


def test_protected_columns_are_kept_even_when_not_listed(tmp_path: object) -> None:
    plan = build_feature_plan(_frame(), _cfg(tmp_path, include_only=["a"], group_by=["site"]))
    site = next(d for d in plan.decisions if d.column == "site")
    assert site.reason != "not listed in columns.include_only"


def test_unset_include_only_admits_every_column(tmp_path: object) -> None:
    plan = build_feature_plan(_frame(), _cfg(tmp_path))
    assert not any(d.reason == "not listed in columns.include_only" for d in plan.decisions)


def test_a_name_not_in_the_dataset_is_a_config_error(tmp_path: object) -> None:
    """A typo would otherwise silently drop every feature."""
    with pytest.raises(ConfigError, match=r"include_only.*\['typo'\]"):
        build_feature_plan(_frame(), _cfg(tmp_path, include_only=["a", "typo"]))


def test_a_column_include_only_leaves_out_is_not_parsed_as_a_date() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "when": ["2026-01-01", "2026-01-02"]})
    assert detect_temporal_parses(df, ColumnsConfig(include_only=["a"])) == {}
    assert "when" in detect_temporal_parses(df, ColumnsConfig(include_only=["a", "when"]))
