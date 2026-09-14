"""On-disk (CSV/Parquet) dataset builders shared across CLI, config-wiring,
end-to-end and score-forward tests.

``tests/synth.make_frame`` builds in-memory frames for profiling/feature
tests; these builders write files because their callers exercise the CLI or
the full pipeline, which read from a source URI on disk.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def write_planted_csv(path: Path, *, n_normal: int = 280, n_anomaly: int = 5, seed: int = 0) -> list[int]:
    """Write a CSV with ``n_anomaly`` planted anomalies in its trailing rows.

    Anomalies have ``num_a=999`` (far outside the normal ``N(0, 1)`` range).
    Columns: ``id`` (integer, joinable back to source), ``num_a``, ``num_b``
    (numeric), ``cat`` (low-cardinality string). Returns the 0-based row
    indices of the planted anomalies.
    """
    rng = np.random.default_rng(seed)
    n_total = n_normal + n_anomaly
    path.parent.mkdir(parents=True, exist_ok=True)

    num_a = rng.normal(0.0, 1.0, n_total).tolist()
    anomaly_indices = list(range(n_normal, n_total))  # last rows are anomalies
    for i in anomaly_indices:
        num_a[i] = 999.0

    df = pl.DataFrame(
        {
            "id": list(range(n_total)),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n_total).tolist(),
            "cat": ["A" if i % 3 else "B" for i in range(n_total)],
        }
    )
    df.write_csv(str(path))
    return anomaly_indices


def write_leading_anomaly_csv(path: Path, *, n_rows: int = 120, n_anomaly: int = 3, seed: int = 0) -> Path:
    """Write a CSV whose *first* ``n_anomaly`` rows are planted anomalies.

    Columns: ``id``, ``num_a``, ``num_b`` — no ``cat`` column. Used where a
    report/explain section needs a few guaranteed-real flagged rows without
    caring about their exact position.
    """
    rng = np.random.default_rng(seed)
    num_a = rng.normal(0.0, 1.0, n_rows)
    num_a[:n_anomaly] = 999.0
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "id": list(range(n_rows)),
            "num_a": num_a.tolist(),
            "num_b": rng.normal(5.0, 2.0, n_rows).tolist(),
        }
    ).write_csv(str(path))
    return path


def write_grouped_csv(path: Path, *, n_rows: int = 300, n_groups: int = 2, seed: int = 0) -> Path:
    """Write a synthetic multi-group CSV with no planted anomalies.

    Columns: ``id``, ``group`` (``G0``..``G{n_groups-1}``), ``value_a``,
    ``value_b`` (numeric), ``cat`` (two-valued string).
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = [f"G{i}" for i in range(n_groups)]
    df = pl.DataFrame(
        {
            "id": list(range(n_rows)),
            "group": [groups[i % n_groups] for i in range(n_rows)],
            "value_a": rng.normal(0, 1, n_rows).tolist(),
            "value_b": rng.normal(5, 2, n_rows).tolist(),
            "cat": (["A"] * (n_rows // 2) + ["B"] * (n_rows - n_rows // 2)),
        }
    )
    df.write_csv(str(path))
    return path


def write_time_sorted_parquet(path: Path, *, n: int = 20, anomaly_orig_idx: int = 0, seed: int = 0) -> None:
    """Write a Parquet file where sorting by ``ts`` reorders rows.

    Row at ``anomaly_orig_idx`` has ``num_a=999``. The timestamp is
    DESCENDING in file order, so an ascending time-sort moves row 0
    (latest ``ts``) to the end — this surfaces bugs where a pipeline looks
    up a raw value at the wrong *original*-frame position after sorting.

    Written as Parquet (not CSV) so ``ts`` round-trips as a proper Datetime
    type; Polars' CSV type inference may not parse date strings.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    base = datetime(2024, 1, 1, tzinfo=UTC)
    # Timestamps in descending order so ascending sort moves row 0 to the end
    timestamps = [base + timedelta(days=n - 1 - i) for i in range(n)]
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[anomaly_orig_idx] = 999.0

    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "ts": pl.Series(timestamps).dt.cast_time_unit("us"),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    )
    df.write_parquet(str(path))


def write_two_period_parquet(path: Path, *, per_day: int = 100, seed: int = 0) -> dict[str, list[int]]:
    """Write two calendar days of data with distinct planted anomalies per day.

    Day 2024-01-15: some ids get ``num_a = +999``.
    Day 2024-01-16: a *different* set of ids get ``num_a = -999``.
    Returns ``{"2024-01-15": [ids...], "2024-01-16": [ids...]}``.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = per_day * 2
    ids = list(range(n))
    ts = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(per_day)]
    ts += [datetime(2024, 1, 16, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(per_day)]
    num_a = rng.normal(0.0, 1.0, n).tolist()

    planted = {
        "2024-01-15": [3, 17, 42],
        "2024-01-16": [per_day + 5, per_day + 8, per_day + 60, per_day + 91],
    }
    for i in planted["2024-01-15"]:
        num_a[i] = 999.0
    for i in planted["2024-01-16"]:
        num_a[i] = -999.0

    pl.DataFrame(
        {
            "id": ids,
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(path))
    return planted
