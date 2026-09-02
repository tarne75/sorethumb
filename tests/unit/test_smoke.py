"""M0 smoke tests — verify the package installs, imports, and CLI works."""

import importlib
import warnings

import polars as pl
import pytest
from typer.testing import CliRunner

import sorethumb
from sorethumb.cli import app
from sorethumb.errors import (
    CalibrationModeWarning,
    ColumnDroppedWarning,
    ConfigError,
    DetectorError,
    ExplainError,
    FallbackAttributionWarning,
    FeatureWidthWarning,
    LowVarianceWarning,
    MemoryBudgetError,
    ModelSchemaDriftError,
    ModelSchemaDriftWarning,
    NonFiniteWarning,
    PlanError,
    PopulationMismatchWarning,
    SampleTruncatedWarning,
    SchemaError,
    SlowStageWarning,
    SorethumbError,
    SorethumbWarning,
    SourceError,
    StoreError,
)
from tests.synth import make_frame


def test_version_attribute() -> None:
    assert sorethumb.__version__ == "0.1.0"


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "sorethumb 0.1.0" in result.output


def test_error_hierarchy() -> None:
    for cls in (
        ConfigError,
        SourceError,
        SchemaError,
        PlanError,
        DetectorError,
        ExplainError,
        StoreError,
        MemoryBudgetError,
        ModelSchemaDriftError,
    ):
        assert issubclass(cls, SorethumbError)
        assert issubclass(cls, Exception)


def test_warning_hierarchy() -> None:
    for cls in (
        ColumnDroppedWarning,
        FeatureWidthWarning,
        NonFiniteWarning,
        LowVarianceWarning,
        SampleTruncatedWarning,
        FallbackAttributionWarning,
        ModelSchemaDriftWarning,
        PopulationMismatchWarning,
        CalibrationModeWarning,
        SlowStageWarning,
    ):
        assert issubclass(cls, SorethumbWarning)
        assert issubclass(cls, UserWarning)


def test_strict_mode_warning_becomes_error() -> None:
    """SorethumbWarning subclasses are errors in the test suite (filterwarnings config)."""
    with pytest.raises(ColumnDroppedWarning):
        warnings.warn("test", ColumnDroppedWarning, stacklevel=1)


def test_synth_make_frame_basic() -> None:
    df, anomaly_indices = make_frame(n_rows=100, seed=42, n_anomalies=5)
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 100
    assert len(anomaly_indices) == 5
    assert "num_a" in df.columns
    assert "num_b" in df.columns


def test_synth_make_frame_all_flags() -> None:
    df, _ = make_frame(
        n_rows=50,
        seed=0,
        null_ratio=0.1,
        with_constant=True,
        with_all_null=True,
        with_high_cardinality_string=True,
        with_low_cardinality_string=True,
        with_free_text=True,
        with_guid=True,
        with_int_identifier=True,
        with_boolean=True,
        with_timestamp=True,
        with_array=True,
        with_struct=True,
        with_correlated=True,
    )
    assert "const_col" in df.columns
    assert "all_null_col" in df.columns
    assert "guid_col" in df.columns
    assert "arr_col" in df.columns
    assert "struct_col" in df.columns
    assert "corr_a" in df.columns
    assert "corr_b" in df.columns
    assert df.shape[0] == 50


def test_synth_anomaly_indices_are_valid() -> None:
    df, indices = make_frame(n_rows=200, seed=7, n_anomalies=10)
    assert all(0 <= i < 200 for i in indices)
    assert len(set(indices)) == 10
    for i in indices:
        assert df["num_a"][i] == 999.0


def test_synth_deterministic() -> None:
    df1, idx1 = make_frame(n_rows=100, seed=42, n_anomalies=3)
    df2, idx2 = make_frame(n_rows=100, seed=42, n_anomalies=3)
    assert df1.equals(df2)
    assert idx1 == idx2


def test_package_importable() -> None:
    mod = importlib.import_module("sorethumb")
    assert hasattr(mod, "__version__")
