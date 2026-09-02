"""Unit tests for the IO layer (M1a)."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from sorethumb.errors import ColumnDroppedWarning, SchemaError, SourceError
from sorethumb.io.fingerprint import content_fingerprint, schema_fingerprint
from sorethumb.io.nested import derive_array_features, unnest_all
from sorethumb.io.readers import read_frame
from sorethumb.io.source import resolve_source
from tests.synth import make_frame

# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_schema_fingerprint_stable() -> None:
    df, _ = make_frame(n_rows=10, seed=0)
    assert schema_fingerprint(df) == schema_fingerprint(df)


def test_schema_fingerprint_changes_on_column_add() -> None:
    df, _ = make_frame(n_rows=10, seed=0)
    df2 = df.with_columns(pl.lit(1).alias("extra"))
    assert schema_fingerprint(df) != schema_fingerprint(df2)


def test_schema_fingerprint_same_for_lazy_and_eager() -> None:
    df, _ = make_frame(n_rows=10, seed=0)
    assert schema_fingerprint(df) == schema_fingerprint(df.lazy())


def test_content_fingerprint_bytes_consistency() -> None:
    data = b"hello world"
    fp1 = content_fingerprint(data)
    fp2 = content_fingerprint(data)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_content_fingerprint_file_matches_bytes(tmp_path: Path) -> None:
    data = b"test content 123"
    p = tmp_path / "data.bin"
    p.write_bytes(data)
    assert content_fingerprint(p) == content_fingerprint(data)


def test_content_fingerprint_different_content(tmp_path: Path) -> None:
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"aaa")
    p2.write_bytes(b"bbb")
    assert content_fingerprint(p1) != content_fingerprint(p2)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_read_csv(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=20, seed=1)
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(csv_path))
    lf = read_frame(csv_path, cfg)
    result = lf.collect()
    assert result.shape[0] == 20
    assert set(result.columns) == set(df.columns)


def test_read_parquet(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=30, seed=2)
    pq_path = tmp_path / "data.parquet"
    df.write_parquet(pq_path)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(pq_path))
    lf = read_frame(pq_path, cfg)
    result = lf.collect()
    assert result.shape == df.shape


def test_read_jsonl(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=10, seed=3)
    jsonl_path = tmp_path / "data.jsonl"
    df.write_ndjson(jsonl_path)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(jsonl_path))
    lf = read_frame(jsonl_path, cfg)
    result = lf.collect()
    assert result.shape[0] == 10


def test_read_single_column_csv_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("only_col\n1\n2\n3\n")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SchemaError, match="only one column"):
        read_frame(p, cfg).collect()


def test_read_auto_format_unknown_ext_raises(tmp_path: Path) -> None:
    p = tmp_path / "data.xyz"
    p.write_bytes(b"whatever")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SourceError, match="auto-detect"):
        read_frame(p, cfg)


def test_read_explicit_format_overrides_extension(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=5, seed=0)
    p = tmp_path / "data.noext"
    df.write_csv(p)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p), format="csv")
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 5


# ---------------------------------------------------------------------------
# Local source resolution
# ---------------------------------------------------------------------------


def test_resolve_source_local(tmp_path: Path) -> None:
    p = tmp_path / "src.csv"
    p.write_text("a,b\n1,2\n")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p))
    resolved = resolve_source(cfg, tmp_path / "cache")
    assert resolved == p.resolve()


def test_resolve_source_missing_file_raises(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(tmp_path / "no_such_file.csv"))
    with pytest.raises(SourceError, match="not found"):
        resolve_source(cfg, tmp_path / "cache")


def test_resolve_source_unsupported_scheme_raises(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri="s3://bucket/key.parquet")
    with pytest.raises(SourceError, match="Unsupported URI scheme"):
        resolve_source(cfg, tmp_path / "cache")


# ---------------------------------------------------------------------------
# Auth token not persisted in config JSON
# ---------------------------------------------------------------------------


def test_auth_token_not_in_config_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET_TOKEN", "super-secret-value-xyz")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(
        uri="https://example.com/data.csv",
        auth="bearer",
        auth_env_var="MY_SECRET_TOKEN",
    )
    serialised = cfg.model_dump_json()
    assert "super-secret-value-xyz" not in serialised
    assert "MY_SECRET_TOKEN" in serialised  # only the variable NAME is stored


# ---------------------------------------------------------------------------
# Struct unnesting
# ---------------------------------------------------------------------------


def test_unnest_simple_struct() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "point": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}, {"x": 5.0, "y": 6.0}],
        }
    )
    result = unnest_all(df, max_depth=5)
    assert "point" not in result.columns
    assert "point_x" in result.columns
    assert "point_y" in result.columns
    assert result.shape == (3, 3)  # id, point_x, point_y


def test_unnest_preserves_values() -> None:
    df = pl.DataFrame(
        {
            "nested": [{"a": 10, "b": 20}, {"a": 30, "b": 40}],
        }
    )
    result = unnest_all(df, max_depth=5)
    assert result["nested_a"].to_list() == [10, 30]
    assert result["nested_b"].to_list() == [20, 40]


def test_unnest_two_levels() -> None:
    df = pl.DataFrame(
        {
            "outer": [{"inner": {"z": float(i)}} for i in range(3)],
        }
    )
    result = unnest_all(df, max_depth=5)
    # After depth 0: outer_inner (a Struct with z)
    # After depth 1: outer_inner_z
    assert "outer_inner_z" in result.columns


def test_unnest_name_collision_raises() -> None:
    df = pl.DataFrame(
        {
            "col_x": [1, 2],
            "col": [{"x": 10, "y": 20}, {"x": 30, "y": 40}],
        }
    )
    with pytest.raises(SchemaError, match="duplicate"):
        unnest_all(df, max_depth=5)


def test_unnest_no_structs_noop() -> None:
    df, _ = make_frame(n_rows=5, seed=0)
    result = unnest_all(df, max_depth=5)
    assert result.equals(df)


def test_unnest_depth_zero_leaves_structs() -> None:
    df = pl.DataFrame(
        {
            "s": [{"a": 1}, {"a": 2}],
        }
    )
    result = unnest_all(df, max_depth=0)
    assert "s" in result.columns
    assert "s_a" not in result.columns


# ---------------------------------------------------------------------------
# Array feature derivation
# ---------------------------------------------------------------------------


def test_derive_array_numeric_features() -> None:
    df = pl.DataFrame(
        {
            "nums": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        result = derive_array_features(df)
    assert "nums" not in result.columns
    assert "nums__len" in result.columns
    assert "nums__mean" in result.columns
    assert "nums__min" in result.columns
    assert "nums__max" in result.columns
    assert result["nums__len"].to_list() == [3, 3, 3]


def test_derive_array_string_no_numeric_stats() -> None:
    df = pl.DataFrame(
        {
            "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        result = derive_array_features(df)
    assert "tags__len" in result.columns
    assert "tags__mean" not in result.columns


def test_derive_array_zero_length_warns() -> None:
    df = pl.DataFrame(
        {
            "empty_arr": [[], [], []],
        },
        schema={"empty_arr": pl.List(pl.Float64)},
    )
    with pytest.raises(ColumnDroppedWarning):
        derive_array_features(df)


def test_derive_array_non_array_cols_preserved() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "vals": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        result = derive_array_features(df)
    assert "id" in result.columns
    assert result["id"].to_list() == [1, 2, 3]
