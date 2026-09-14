"""Unit tests for the IO layer (M1a)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl
import pytest

from sorethumb.errors import ColumnDroppedWarning, SchemaError, SourceError
from sorethumb.io.fingerprint import (
    content_fingerprint,
    logical_dataset_id,
    schema_fingerprint,
    snapshot_fingerprint,
)
from sorethumb.io.nested import derive_array_features, unnest_all
from sorethumb.io.readers import read_frame
from sorethumb.io.source import resolve_source
from tests.synth import make_frame

pytestmark = pytest.mark.unit

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
# Logical dataset identity vs. snapshot identity
# ---------------------------------------------------------------------------


def test_logical_dataset_id_prefers_configured_value() -> None:
    assert logical_dataset_id("sales-eu", "/data/whatever.parquet") == "sales-eu"


def test_logical_dataset_id_derived_is_stable_and_readable() -> None:
    a = logical_dataset_id(None, "/srv/data/sales.parquet")
    b = logical_dataset_id(None, "/srv/data/sales.parquet")
    assert a == b
    assert a.startswith("sales-")


def test_logical_dataset_id_derived_distinguishes_same_basename() -> None:
    eu = logical_dataset_id(None, "/srv/eu/sales.parquet")
    us = logical_dataset_id(None, "/srv/us/sales.parquet")
    assert eu != us
    assert eu.startswith("sales-")
    assert us.startswith("sales-")


def test_logical_dataset_id_ignores_query_string() -> None:
    plain = logical_dataset_id(None, "https://host/data/sales.parquet")
    signed = logical_dataset_id(None, "https://host/data/sales.parquet?sig=abc123&exp=999")
    assert plain == signed


def test_snapshot_fingerprint_tracks_content_and_schema() -> None:
    base = snapshot_fingerprint("c" * 64, "s" * 64)
    assert base == snapshot_fingerprint("c" * 64, "s" * 64)
    assert base != snapshot_fingerprint("d" * 64, "s" * 64)  # content changed
    assert base != snapshot_fingerprint("c" * 64, "t" * 64)  # schema changed


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


def test_read_tsv_injects_tab_separator(tmp_path: Path) -> None:
    """TSV format injects separator='\\t' when the caller didn't set one."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 3)
    assert "col_a" in result.columns


def test_read_tsv_explicit_separator_not_overridden(tmp_path: Path) -> None:
    """TSV does NOT inject a separator if the caller already provided one."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")

    cfg = SourceConfig(uri=str(p), read_options={"separator": "\t"})
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 2


def test_read_json(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    p = tmp_path / "data.json"
    rows = [{"x": float(i), "y": float(i * 2), "z": float(i + 1)} for i in range(5)]
    p.write_text(json.dumps(rows))

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (5, 3)
    assert "x" in result.columns


def test_read_all_string_schema_raises(tmp_path: Path) -> None:
    """A schema where every column is String (no numeric signal at all) raises."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "strings.csv"
    p.write_text("col_a,col_b,col_c\nfoo,bar,baz\nhello,world,test\n")

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SchemaError, match="String"):
        read_frame(p, cfg).collect()


# ---------------------------------------------------------------------------
# TSF reader
# ---------------------------------------------------------------------------


def test_read_tsf_basic(tmp_path: Path) -> None:
    """Basic TSF file: one series, no attributes, single data row."""
    from sorethumb.config import SourceConfig

    content = "@data\n1.0,2.0,3.0\n"
    p = tmp_path / "basic.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (1, 3)
    assert result["value_0"][0] == pytest.approx(1.0)
    assert result["value_2"][0] == pytest.approx(3.0)


def test_read_tsf_with_numeric_attribute(tmp_path: Path) -> None:
    """A numeric @attribute column is parsed as int/float."""
    from sorethumb.config import SourceConfig

    content = "@attribute series_id numeric\n@data\n42:10.0,20.0,30.0\n7:1.5,2.5,3.5\n"
    p = tmp_path / "attrs.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 4)
    assert result["series_id"].to_list() == [42, 7]
    assert result["value_0"].to_list() == pytest.approx([10.0, 1.5])


def test_read_tsf_with_float_numeric_attribute(tmp_path: Path) -> None:
    """A numeric attribute containing a decimal point parses as float."""
    from sorethumb.config import SourceConfig

    content = "@attribute score numeric\n@data\n1.5:10.0,20.0,30.0\n"
    p = tmp_path / "float_attr.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["score"][0] == pytest.approx(1.5)


def test_read_tsf_with_string_attribute(tmp_path: Path) -> None:
    """A string @attribute keeps its value as-is."""
    from sorethumb.config import SourceConfig

    content = "@attribute category string\n@data\ntrain:1.0,2.0,3.0\ntest:4.0,5.0,6.0\n"
    p = tmp_path / "str_attr.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 4)
    assert result["category"].to_list() == ["train", "test"]


def test_read_tsf_with_date_attribute(tmp_path: Path) -> None:
    """A date @attribute is treated as a plain string (kept as-is)."""
    from sorethumb.config import SourceConfig

    content = "@attribute start_timestamp date\n@data\n2020-01-01:1.0,2.0\n2020-01-02:3.0,4.0\n"
    p = tmp_path / "date_attr.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["start_timestamp"].to_list() == ["2020-01-01", "2020-01-02"]


def test_read_tsf_missing_values(tmp_path: Path) -> None:
    """'?' tokens and empty tokens both become None."""
    from sorethumb.config import SourceConfig

    content = "@data\n1.0,?,3.0\n"
    p = tmp_path / "missing.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["value_0"][0] == pytest.approx(1.0)
    assert result["value_1"][0] is None
    assert result["value_2"][0] == pytest.approx(3.0)


def test_read_tsf_variable_length_series_padded(tmp_path: Path) -> None:
    """Shorter series are padded to the max length with None."""
    from sorethumb.config import SourceConfig

    content = "@data\n1.0,2.0,3.0\n4.0,5.0\n"
    p = tmp_path / "varlen.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 3)
    assert result["value_2"][0] == pytest.approx(3.0)
    assert result["value_2"][1] is None


def test_read_tsf_comment_lines_skipped(tmp_path: Path) -> None:
    """Lines starting with # are silently skipped."""
    from sorethumb.config import SourceConfig

    content = "# this is a comment\n@data\n# another comment\n1.0,2.0,3.0\n"
    p = tmp_path / "comments.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_unknown_at_directives_skipped(tmp_path: Path) -> None:
    """Unknown @ directives (not @attribute / @data) are skipped."""
    from sorethumb.config import SourceConfig

    content = "@frequency yearly\n@horizon 10\n@data\n1.0,2.0,3.0\n"
    p = tmp_path / "directives.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_lines_before_data_skipped(tmp_path: Path) -> None:
    """Non-@ lines before @data are ignored (data hasn't started yet)."""
    from sorethumb.config import SourceConfig

    content = "stray line\n@data\n1.0,2.0,3.0\n"
    p = tmp_path / "stray.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_malformed_row_too_few_fields_skipped(tmp_path: Path) -> None:
    """Malformed rows (too few colon-separated fields) are silently skipped."""
    from sorethumb.config import SourceConfig

    content = (
        "@attribute id numeric\n"
        "@data\n"
        "1:10.0,20.0\n"
        "BADROW\n"  # no colon → fewer fields than attributes + 1
        "2:30.0,40.0\n"
    )
    p = tmp_path / "malformed.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 2  # only 2 valid rows


def test_read_tsf_numeric_attribute_bad_value_gives_none(tmp_path: Path) -> None:
    """A non-numeric value in a numeric @attribute field becomes None."""
    from sorethumb.config import SourceConfig

    content = "@attribute id numeric\n@data\nnotanumber:1.0,2.0,3.0\n"
    p = tmp_path / "badnum.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["id"][0] is None


def test_read_tsf_empty_no_data_rows(tmp_path: Path) -> None:
    """A file with @attribute but no data rows returns an empty LazyFrame
    with the declared schema."""
    from sorethumb.io.readers import _read_tsf

    content = "@attribute id numeric\n@attribute label string\n@data\n"
    p = tmp_path / "empty.tsf"
    p.write_text(content)

    result = _read_tsf(p).collect()
    assert result.shape[0] == 0
    assert set(result.columns) == {"id", "label"}


def test_read_tsf_multiple_rows_multiple_attrs(tmp_path: Path) -> None:
    """Multi-row, multi-attribute TSF file integrates end-to-end correctly."""
    from sorethumb.config import SourceConfig

    content = (
        "@attribute series_id numeric\n"
        "@attribute split string\n"
        "@data\n"
        "1:train:10.0,20.0,30.0\n"
        "2:test:40.0,50.0\n"
    )
    p = tmp_path / "multi.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 5)  # series_id, split, value_0, value_1, value_2
    assert result["split"].to_list() == ["train", "test"]
    assert result["value_2"][1] is None  # second row is shorter


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
