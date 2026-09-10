"""Schema and content fingerprinting for change detection and caching.

Two distinct identities live here:

* **Logical dataset identity** (:func:`logical_dataset_id`) -- stable across
  snapshots. Appending a day of rows, correcting a value, or moving the file to
  a new path must NOT change it, or all prior period / totals / run history is
  orphaned. It is a configured label (``source.dataset_id``) or, failing that,
  derived from the source URI.
* **Snapshot identity** (:func:`snapshot_fingerprint`) -- the content+schema
  fingerprint of one observed version of the data. Recorded per snapshot so a
  run can be traced to exactly the bytes it saw, and so "has the data changed?"
  stays answerable.
"""

from __future__ import annotations

import hashlib
import re
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


def snapshot_fingerprint(content_fp: str, schema_fp: str) -> str:
    """Identity of one observed *snapshot* of a dataset: its content + schema.

    This is the value ``dataset_fp`` used to hold before logical and snapshot
    identity were separated; it now versions a single dataset rather than
    identifying it.
    """
    return f"{content_fp[:32]}_{schema_fp[:16]}"


def _uri_derived_id(uri: str) -> str:
    """Stable, readable id for a source URI when no ``dataset_id`` is configured.

    ``<sanitised file stem>-<12 hex of sha256(uri)>`` -- the stem aids humans
    scanning ``sorethumb runs``; the digest keeps two different sources with the
    same file name apart. Query strings and trailing slashes are ignored so a
    signed-URL refresh does not change the id.
    """
    base = uri.strip().split("?", 1)[0].split("#", 1)[0].rstrip("/")
    stem = Path(base).stem or Path(base).name or "dataset"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "dataset"
    digest = hashlib.sha256(base.encode()).hexdigest()[:12]
    return f"{stem[:64]}-{digest}"


def logical_dataset_id(dataset_id: str | None, uri: str) -> str:
    """Stable logical identity for a dataset, constant across snapshots.

    Returns the configured ``dataset_id`` verbatim when given, otherwise an id
    derived from ``uri`` (see :func:`_uri_derived_id`).
    """
    if dataset_id:
        return dataset_id
    return _uri_derived_id(uri)
