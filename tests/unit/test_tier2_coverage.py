"""Tier 2 coverage tests — import-time code and reader paths.

Targets five modules that still have low coverage:
  - io/readers.py      (TSV, JSON, TSF, all-string SchemaError, import-time)
  - evaluate/metrics.py (import-time only — function bodies covered by test_evaluate.py)
  - explain/centroid.py (import-time only — function bodies covered by test_explain.py)
  - explain/shap_tree.py (import-time only — function bodies covered by test_explain.py)
  - store/results.py    (import-time + write_results + read_results)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# evaluate/metrics.py — import-time lines
# ---------------------------------------------------------------------------


def test_metrics_reexecuted_under_coverage() -> None:
    import sorethumb.evaluate.metrics as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# explain/centroid.py — import-time lines
# ---------------------------------------------------------------------------


def test_centroid_reexecuted_under_coverage() -> None:
    import sorethumb.explain.centroid as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# explain/shap_tree.py — import-time lines
# ---------------------------------------------------------------------------


def test_shap_tree_reexecuted_under_coverage() -> None:
    import sorethumb.explain.shap_tree as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# io/readers.py — import-time + format branches
# ---------------------------------------------------------------------------


def test_readers_reexecuted_under_coverage() -> None:
    import sorethumb.io.readers as m

    importlib.reload(m)


def test_read_tsv_injects_tab_separator(tmp_path: Path) -> None:
    """Line 51: TSV branch injects separator=\\t when not already set."""
    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 3)
    assert "col_a" in result.columns


def test_read_tsv_explicit_separator_not_overridden(tmp_path: Path) -> None:
    """TSV branch does NOT inject separator if caller already provided one."""
    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")

    cfg = SourceConfig(uri=str(p), read_options={"separator": "\t"})
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 2


def test_read_json(tmp_path: Path) -> None:
    """Line 54: JSON branch uses pl.read_json(...).lazy()."""
    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    p = tmp_path / "data.json"
    rows = [{"x": float(i), "y": float(i * 2), "z": float(i + 1)} for i in range(5)]
    p.write_text(json.dumps(rows))

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (5, 3)
    assert "x" in result.columns


def test_read_tsf_basic(tmp_path: Path) -> None:
    """Basic TSF file: one series, no attributes, single data row."""
    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    content = "@data\n1.0,2.0,3.0\n"
    p = tmp_path / "basic.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (1, 3)
    assert result["value_0"][0] == pytest.approx(1.0)
    assert result["value_2"][0] == pytest.approx(3.0)


def test_read_tsf_with_numeric_attribute(tmp_path: Path) -> None:
    """TSF with a numeric @attribute column: value should be parsed as int/float."""
    content = "@attribute series_id numeric\n@data\n42:10.0,20.0,30.0\n7:1.5,2.5,3.5\n"
    p = tmp_path / "attrs.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 4)
    assert result["series_id"].to_list() == [42, 7]
    assert result["value_0"].to_list() == pytest.approx([10.0, 1.5])


def test_read_tsf_with_float_numeric_attribute(tmp_path: Path) -> None:
    """Numeric attribute containing a decimal point → float."""
    content = "@attribute score numeric\n@data\n1.5:10.0,20.0,30.0\n"
    p = tmp_path / "float_attr.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["score"][0] == pytest.approx(1.5)


def test_read_tsf_with_string_attribute(tmp_path: Path) -> None:
    """TSF with a string @attribute: value kept as-is string."""
    content = "@attribute category string\n@data\ntrain:1.0,2.0,3.0\ntest:4.0,5.0,6.0\n"
    p = tmp_path / "str_attr.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 4)
    assert result["category"].to_list() == ["train", "test"]


def test_read_tsf_with_date_attribute(tmp_path: Path) -> None:
    """TSF date @attribute is treated as string (kept as-is)."""
    content = "@attribute start_timestamp date\n@data\n2020-01-01:1.0,2.0\n2020-01-02:3.0,4.0\n"
    p = tmp_path / "date_attr.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["start_timestamp"].to_list() == ["2020-01-01", "2020-01-02"]


def test_read_tsf_missing_values(tmp_path: Path) -> None:
    """? tokens → None; empty tokens → None."""
    content = "@data\n1.0,?,3.0\n"
    p = tmp_path / "missing.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["value_0"][0] == pytest.approx(1.0)
    assert result["value_1"][0] is None
    assert result["value_2"][0] == pytest.approx(3.0)


def test_read_tsf_variable_length_series_padded(tmp_path: Path) -> None:
    """Shorter series are padded to max length with None."""
    content = "@data\n1.0,2.0,3.0\n4.0,5.0\n"
    p = tmp_path / "varlen.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 3)
    assert result["value_2"][0] == pytest.approx(3.0)
    assert result["value_2"][1] is None


def test_read_tsf_comment_lines_skipped(tmp_path: Path) -> None:
    """Lines starting with # are silently skipped."""
    content = "# this is a comment\n@data\n# another comment\n1.0,2.0,3.0\n"
    p = tmp_path / "comments.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_unknown_at_directives_skipped(tmp_path: Path) -> None:
    """Unknown @ directives (not @attribute / @data) are skipped."""
    content = "@frequency yearly\n@horizon 10\n@data\n1.0,2.0,3.0\n"
    p = tmp_path / "directives.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_lines_before_data_skipped(tmp_path: Path) -> None:
    """Non-@ lines before @data are ignored (not data_started yet)."""
    content = "stray line\n@data\n1.0,2.0,3.0\n"
    p = tmp_path / "stray.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_malformed_row_too_few_fields_skipped(tmp_path: Path) -> None:
    """Malformed rows (too few colon-separated fields) are silently skipped."""
    content = (
        "@attribute id numeric\n"
        "@data\n"
        "1:10.0,20.0\n"
        "BADROW\n"  # no colon → fewer fields than attributes + 1
        "2:30.0,40.0\n"
    )
    p = tmp_path / "malformed.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 2  # only 2 valid rows


def test_read_tsf_numeric_attribute_bad_value_gives_none(tmp_path: Path) -> None:
    """Non-numeric value in a numeric @attribute field → None."""
    content = "@attribute id numeric\n@data\nnotanumber:1.0,2.0,3.0\n"
    p = tmp_path / "badnum.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["id"][0] is None


def test_read_tsf_empty_no_data_rows(tmp_path: Path) -> None:
    """File with @attribute but no data rows returns empty LazyFrame with declared schema."""
    content = "@attribute id numeric\n@attribute label string\n@data\n"
    p = tmp_path / "empty.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import _read_tsf

    result = _read_tsf(p).collect()
    assert result.shape[0] == 0
    assert set(result.columns) == {"id", "label"}


def test_read_tsf_multiple_rows_multiple_attrs(tmp_path: Path) -> None:
    """Multi-row, multi-attribute TSF file integrates end-to-end correctly."""
    content = (
        "@attribute series_id numeric\n"
        "@attribute split string\n"
        "@data\n"
        "1:train:10.0,20.0,30.0\n"
        "2:test:40.0,50.0\n"
    )
    p = tmp_path / "multi.tsf"
    p.write_text(content)

    from sorethumb.config import SourceConfig
    from sorethumb.io.readers import read_frame

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 5)  # series_id, split, value_0, value_1, value_2
    assert result["split"].to_list() == ["train", "test"]
    assert result["value_2"][1] is None  # second row is shorter


def test_read_all_string_schema_raises(tmp_path: Path) -> None:
    """Line 176: all-string schema raises SchemaError."""
    from sorethumb.config import SourceConfig
    from sorethumb.errors import SchemaError
    from sorethumb.io.readers import read_frame

    p = tmp_path / "strings.csv"
    p.write_text("col_a,col_b,col_c\nfoo,bar,baz\nhello,world,test\n")

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SchemaError, match="String"):
        read_frame(p, cfg).collect()


# ---------------------------------------------------------------------------
# store/results.py — import-time + write/read round-trip
# ---------------------------------------------------------------------------


def test_results_reexecuted_under_coverage() -> None:
    import sorethumb.store.results as m

    importlib.reload(m)


def test_read_results_returns_none_when_absent(tmp_path: Path) -> None:
    """read_results returns None when the Parquet file does not exist."""
    from sorethumb.store.results import read_results
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    result = read_results(ws, run_id="run_abc", group_key="ALL")
    assert result is None


def test_write_then_read_results(tmp_path: Path) -> None:
    """write_results persists a DataFrame; read_results reloads it correctly."""
    from sorethumb.store.results import read_results, write_results
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    run_id = "run_001"
    group_key = "ALL"

    df = pl.DataFrame(
        {
            "row_id": [0, 1, 2],
            "composite_score": [0.9, 0.7, 0.5],
            "is_anomaly": [True, True, False],
        }
    )
    out_path = write_results(ws, run_id=run_id, group_key=group_key, df=df)
    assert out_path.exists()

    loaded = read_results(ws, run_id=run_id, group_key=group_key)
    assert loaded is not None
    assert loaded.shape == df.shape
    assert loaded["row_id"].to_list() == [0, 1, 2]


def test_write_results_returns_correct_path(tmp_path: Path) -> None:
    """write_results path ends with anomalies.parquet."""
    from sorethumb.store.results import write_results
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    df = pl.DataFrame({"row_id": [0], "composite_score": [0.99], "flag": [1]})
    out_path = write_results(ws, run_id="r1", group_key="g1", df=df)
    assert out_path.name == "anomalies.parquet"
