"""Schema and content fingerprinting for change detection and caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl


def schema_fingerprint(df: pl.DataFrame | pl.LazyFrame) -> str:
    """Stable hash of the column names and dtypes in declaration order.

    Two frames with identical schemas (regardless of data) produce the same
    fingerprint. Used to detect schema drift between runs.
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    parts = "|".join(f"{name}:{dtype}" for name, dtype in schema.items())
    return hashlib.sha256(parts.encode()).hexdigest()[:32]


def content_fingerprint(source: Path | bytes) -> str:
    """SHA-256 fingerprint of raw file bytes.

    Pass a ``Path`` to hash the file without loading it into memory.
    Pass ``bytes`` to hash already-loaded content.
    Both routes produce identical digests for the same content.
    """
    hasher = hashlib.sha256()
    if isinstance(source, bytes):
        hasher.update(source)
    else:
        with source.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hasher.update(chunk)
    return hasher.hexdigest()
