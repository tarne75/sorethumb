"""report.csv_bom: an optional UTF-8 byte-order mark for Excel on Windows.

Excel on Windows opens a UTF-8 CSV without a BOM as ANSI (cp1252), so a column
named "温度" or a value "naïve" shows as mojibake. Off by default: every other
tool reads plain UTF-8, and some treat a BOM as part of the first header.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sorethumb_ml.config import ReportConfig
from sorethumb_ml.report.csv import write_group_csv

pytestmark = pytest.mark.unit

_BOM = b"\xef\xbb\xbf"


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"row_id": [0, 1], "温度": [1.5, 2.5], "cat": ["naïve", "b"]})


def test_off_by_default() -> None:
    assert ReportConfig().csv_bom is False


def test_without_a_bom_the_file_is_plain_utf8(tmp_path: Path) -> None:
    raw = write_group_csv(_frame(), tmp_path, "g" * 32).read_bytes()
    assert not raw.startswith(_BOM)
    assert raw.decode("utf-8").startswith("row_id,温度,cat")


def test_with_a_bom_excel_sees_utf8(tmp_path: Path) -> None:
    raw = write_group_csv(_frame(), tmp_path, "g" * 32, include_bom=True).read_bytes()
    assert raw.startswith(_BOM)
    assert raw[len(_BOM) :].decode("utf-8").startswith("row_id,温度,cat")
    assert pl.read_csv(tmp_path / f"{'g' * 32}.csv").columns == ["row_id", "温度", "cat"]
