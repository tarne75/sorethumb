"""Synthetic dataset generators used by all unit and property tests.

Every generator is seeded and deterministic. Pass ``seed`` explicitly to get
reproducible frames. Generators must be able to produce on demand:

- nulls at a chosen ratio
- constant and all-null columns
- high/low cardinality strings, free text, GUIDs, integer identifiers
- booleans, timestamps, arrays (List[Float64]), nested structs
- perfectly correlated column pairs
- point anomalies with known row indices
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl


def make_frame(
    n_rows: int = 200,
    *,
    seed: int = 42,
    null_ratio: float = 0.0,
    with_constant: bool = False,
    with_all_null: bool = False,
    with_high_cardinality_string: bool = False,
    with_low_cardinality_string: bool = False,
    with_free_text: bool = False,
    with_guid: bool = False,
    with_int_identifier: bool = False,
    with_boolean: bool = False,
    with_timestamp: bool = False,
    with_array: bool = False,
    with_struct: bool = False,
    with_correlated: bool = False,
    n_anomalies: int = 0,
) -> tuple[pl.DataFrame, list[int]]:
    """Return ``(frame, anomaly_row_indices)``.

    All columns are seeded from ``seed``. Anomaly indices are a random sample
    of rows where ``num_a`` is set to 999.0 — far outside the normal range.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, list[object]] = {}

    # Base numeric columns — always present
    data["num_a"] = rng.normal(0.0, 1.0, n_rows).tolist()
    data["num_b"] = rng.exponential(1.0, n_rows).tolist()

    if null_ratio > 0.0:
        mask = rng.random(n_rows) < null_ratio
        vals: list[float | None] = [
            None if m else float(v) for m, v in zip(mask, rng.normal(0.0, 1.0, n_rows), strict=False)
        ]
        data["num_nullable"] = vals

    if with_constant:
        data["const_col"] = [42.0] * n_rows

    if with_all_null:
        data["all_null_col"] = [None] * n_rows

    if with_high_cardinality_string:
        data["high_card_str"] = [f"item_{i}" for i in range(n_rows)]

    if with_low_cardinality_string:
        cats = ["cat_a", "cat_b", "cat_c"]
        data["low_card_str"] = [cats[i % 3] for i in range(n_rows)]

    if with_free_text:
        sentence = "this is a long free text sentence used for testing the free text classifier"
        data["free_text_col"] = [sentence] * n_rows

    if with_guid:
        data["guid_col"] = [
            str(uuid.UUID(bytes=bytes(rng.integers(0, 256, 16, dtype=np.uint8).tolist())))
            for _ in range(n_rows)
        ]

    if with_int_identifier:
        data["int_id_col"] = list(range(n_rows))

    if with_boolean:
        data["bool_col"] = [bool(v) for v in rng.integers(0, 2, n_rows).tolist()]

    if with_timestamp:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        data["ts_col"] = [base + timedelta(hours=int(h)) for h in range(n_rows)]

    if with_array:
        data["arr_col"] = [[float(x) for x in rng.normal(0.0, 1.0, 3).tolist()] for _ in range(n_rows)]

    if with_struct:
        data["struct_col"] = [{"x": float(rng.normal()), "y": float(rng.normal())} for _ in range(n_rows)]

    if with_correlated:
        base_vals = rng.normal(0.0, 1.0, n_rows)
        data["corr_a"] = base_vals.tolist()
        data["corr_b"] = (base_vals + rng.normal(0.0, 0.01, n_rows)).tolist()

    # Inject point anomalies: set num_a far outside the normal range
    anomaly_indices: list[int] = []
    if n_anomalies > 0:
        raw_idxs = rng.choice(n_rows, size=n_anomalies, replace=False)
        anomaly_indices = [int(i) for i in raw_idxs.tolist()]
        num_a = data["num_a"]
        assert isinstance(num_a, list)
        for i in anomaly_indices:
            num_a[i] = 999.0

    return pl.DataFrame(data), anomaly_indices
