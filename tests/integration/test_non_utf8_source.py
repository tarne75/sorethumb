"""A cp1252 CSV (Excel's plain "CSV" on Windows) fails with an actionable message.

Polars reads text sources as UTF-8 only and reports another encoding as a bare
"ComputeError: invalid utf-8 sequence" when rows are read. It now becomes a
SourceError that says what is wrong and how to fix it; utf8-lossy still works.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sorethumb_ml.config import SourceConfig
from sorethumb_ml.errors import SourceError
from sorethumb_ml.io.readers import collect_frame, read_frame

pytestmark = pytest.mark.integration


@pytest.fixture
def cp1252_csv(tmp_path: Path) -> Path:
    path = tmp_path / "excel.csv"
    rows = ["name,amount,city"] + [f"café{i},{i},Zürich €{i}" for i in range(50)]
    path.write_bytes("\r\n".join(rows).encode("cp1252"))
    return path


def test_a_cp1252_csv_is_a_clear_source_error(cp1252_csv: Path) -> None:
    cfg = SourceConfig(uri=str(cp1252_csv))
    with pytest.raises(SourceError, match="not valid UTF-8") as info:
        collect_frame(read_frame(cp1252_csv, cfg), cp1252_csv)
    assert "CSV UTF-8" in str(info.value)
    assert "utf8-lossy" in str(info.value)


def test_utf8_lossy_reads_it(cp1252_csv: Path) -> None:
    cfg = SourceConfig(uri=str(cp1252_csv), read_options={"encoding": "utf8-lossy"})
    df = collect_frame(read_frame(cp1252_csv, cfg), cp1252_csv)
    assert df.height == 50
    assert df.columns == ["name", "amount", "city"]


def test_other_compute_errors_are_not_reworded(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"a": ["x"]}).select(pl.col("a").cast(pl.Int64, strict=True))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        collect_frame(lf, tmp_path / "unused.csv")
