"""P1-3: columns.id_column's identity contract is validated over the full
source population, before any period/group/anomaly filtering -- not just
among the rows that happen to be flagged (which could hide a duplicate
whose two instances land on opposite sides of the flag/normal split).

See tests/integration/test_row_identity.py for the *fallback* row_id
(no id_column configured) and its own global-uniqueness guarantee.
"""

from __future__ import annotations

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
from sorethumb.errors import SchemaError

pytestmark = pytest.mark.integration


def _cfg(
    source: Path,
    workdir: Path,
    *,
    id_column: str,
    id_scope: str = "auto",
    group_by: list[str] | None = None,
    contamination: float = 0.1,
) -> Config:
    return Config(
        source=SourceConfig(uri=str(source), format="csv"),
        run=RunConfig(workdir=str(workdir), seed=42),
        columns=ColumnsConfig(id_column=id_column, id_scope=id_scope, group_by=group_by or []),
        detectors=[DetectorConfig(name="isolation_forest")],
        scoring=ScoringConfig(
            combination="composite", contamination=contamination, weighting="equal", min_records=5
        ),
    )


# ---------------------------------------------------------------------------
# Duplicate id_column, only one instance ever flagged
# ---------------------------------------------------------------------------


def test_duplicate_id_raises_even_when_only_one_duplicate_row_is_flagged(tmp_path: Path) -> None:
    """The exact scenario a post-filter (or absent) check would miss: two
    rows share the same id value, but only one of them is anomalous. A check
    that only ever looked at the flagged subset would see a single, unique
    id there and never notice its unflagged duplicate exists -- so this must
    be caught before any anomaly filtering, over the whole population."""
    n = 20
    rng = np.random.default_rng(0)
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[-1] = 999.0  # the only planted anomaly
    ids = list(range(n))
    ids[5] = ids[10]  # row 10 (the anomaly's neighbour) duplicates row 5's id; row 10 itself stays normal

    csv = tmp_path / "data.csv"
    pl.DataFrame({"id": ids, "num_a": num_a, "num_b": rng.normal(5.0, 2.0, n).tolist()}).write_csv(str(csv))

    cfg = _cfg(csv, tmp_path / "ws", id_column="id", contamination=1 / n)
    with pytest.raises(SchemaError, match="not unique"):
        run_detection(cfg, no_report=True)


# ---------------------------------------------------------------------------
# Group-scoped identity: repeats across groups are fine, repeats within a
# group are not
# ---------------------------------------------------------------------------


def _grouped_csv(path: Path, *, per_group: int = 20, seed: int = 0) -> None:
    """Two groups ("A", "B"), each with its own id sequence 0..per_group-1 --
    a legitimate group-scoped identifier (e.g. an order_id that resets per
    store_id)."""
    rng = np.random.default_rng(seed)
    cat: list[str] = []
    ids: list[int] = []
    num_a: list[float] = []
    for g in ("A", "B"):
        cat.extend([g] * per_group)
        ids.extend(range(per_group))
        vals = rng.normal(0.0, 1.0, per_group).tolist()
        vals[-1] = 999.0  # one planted anomaly per group
        num_a.extend(vals)
    pl.DataFrame(
        {
            "cat": cat,
            "id": ids,
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, len(cat)).tolist(),
        }
    ).write_csv(str(path))


def test_id_repeating_across_groups_is_allowed_under_group_scope(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _grouped_csv(csv, per_group=20)
    cfg = _cfg(csv, tmp_path / "ws", id_column="id", group_by=["cat"], contamination=1 / 20)

    result = run_detection(cfg, no_report=True)
    assert result.n_succeeded == 2
    assert result.id_identity_scope == "group"


def test_id_duplicated_within_one_group_still_raises_under_group_scope(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _grouped_csv(csv, per_group=20)
    df = pl.read_csv(csv)
    # Introduce a real duplicate inside group "A" only: row 1's id now
    # collides with row 0's, both still within cat="A".
    ids = df["id"].to_list()
    ids[1] = ids[0]
    df = df.with_columns(pl.Series("id", ids))
    df.write_csv(str(csv))

    cfg = _cfg(csv, tmp_path / "ws", id_column="id", group_by=["cat"], contamination=1 / 20)
    with pytest.raises(SchemaError, match="group_by"):
        run_detection(cfg, no_report=True)


def test_id_scope_dataset_override_catches_cross_group_duplicate_group_scope_would_allow(
    tmp_path: Path,
) -> None:
    """id_scope='dataset' forces whole-population uniqueness even with
    group_by configured -- the exact cross-group repeat that 'group' (or
    'auto') scope legitimately allows in the previous test must now raise."""
    csv = tmp_path / "data.csv"
    _grouped_csv(csv, per_group=20)  # ids 0..19 repeat identically in both "A" and "B"

    cfg = _cfg(
        csv, tmp_path / "ws", id_column="id", id_scope="dataset", group_by=["cat"], contamination=1 / 20
    )
    with pytest.raises(SchemaError, match="across the dataset"):
        run_detection(cfg, no_report=True)


# ---------------------------------------------------------------------------
# Null ids
# ---------------------------------------------------------------------------


def test_null_id_raises(tmp_path: Path) -> None:
    n = 20
    rng = np.random.default_rng(0)
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[-1] = 999.0
    ids: list[int | None] = list(range(n))
    ids[3] = None

    csv = tmp_path / "data.csv"
    pl.DataFrame({"id": ids, "num_a": num_a, "num_b": rng.normal(5.0, 2.0, n).tolist()}).write_csv(str(csv))

    cfg = _cfg(csv, tmp_path / "ws", id_column="id", contamination=1 / n)
    with pytest.raises(SchemaError, match="null"):
        run_detection(cfg, no_report=True)


def test_missing_id_column_raises(tmp_path: Path) -> None:
    n = 20
    rng = np.random.default_rng(0)
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[-1] = 999.0
    csv = tmp_path / "data.csv"
    pl.DataFrame({"num_a": num_a, "num_b": rng.normal(5.0, 2.0, n).tolist()}).write_csv(str(csv))

    cfg = _cfg(csv, tmp_path / "ws", id_column="does_not_exist", contamination=1 / n)
    with pytest.raises(SchemaError, match="is not in the dataset"):
        run_detection(cfg, no_report=True)


# ---------------------------------------------------------------------------
# Mixed id dtypes -- validation is dtype-agnostic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("id_values", [list(range(20)), [f"txn_{i:03d}" for i in range(20)]])
def test_valid_id_column_passes_regardless_of_dtype(tmp_path: Path, id_values: list[object]) -> None:
    n = 20
    rng = np.random.default_rng(0)
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[-1] = 999.0
    csv = tmp_path / "data.csv"
    pl.DataFrame({"id": id_values, "num_a": num_a, "num_b": rng.normal(5.0, 2.0, n).tolist()}).write_csv(
        str(csv)
    )

    cfg = _cfg(csv, tmp_path / "ws", id_column="id", contamination=1 / n)
    result = run_detection(cfg, no_report=True)
    assert result.n_succeeded == 1
    assert result.id_identity_scope == "dataset"


# ---------------------------------------------------------------------------
# Successful join back to the original source frame
# ---------------------------------------------------------------------------


def test_valid_id_column_joins_back_to_original_frame(tmp_path: Path) -> None:
    n = 30
    rng = np.random.default_rng(0)
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[-1] = 999.0
    ids = [f"txn_{i:03d}" for i in range(n)]
    num_b = rng.normal(5.0, 2.0, n).tolist()

    csv = tmp_path / "data.csv"
    src_df = pl.DataFrame({"id": ids, "num_a": num_a, "num_b": num_b})
    src_df.write_csv(str(csv))

    cfg = _cfg(csv, tmp_path / "ws", id_column="id", contamination=1 / n)
    result = run_detection(cfg, no_report=True)
    assert result.n_succeeded == 1
    assert result.id_identity_scope == "dataset"

    results_path = result.groups[0].results_path
    assert results_path is not None
    df_anomalies = pl.read_parquet(results_path)
    assert len(df_anomalies) >= 1

    # row_id must be exactly the configured id_column's values (not the
    # internal positional fallback), each joining back to exactly one source
    # row -- and that row must be the actual planted anomaly (num_a=999.0),
    # not an arbitrary/wrong row a broken join would silently accept.
    joined = df_anomalies.join(src_df, left_on="row_id", right_on="id", how="inner")
    assert len(joined) == len(df_anomalies), "every flagged row_id must join back to exactly one source row"
    assert all(v == pytest.approx(999.0) for v in joined["num_a"].to_list())
