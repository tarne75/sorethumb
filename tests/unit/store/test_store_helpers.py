"""Unit tests for pure store helper functions: make_group_key's hashing and
validate_identifier's SQL-identifier guard. No Workspace, no SQLite -- see
tests/contract/test_database_schema.py and tests/integration/test_store_workflows.py
for anything that touches a real store.
"""

from __future__ import annotations

import pytest

from sorethumb.errors import StoreError
from sorethumb.store.identifiers import validate_identifier
from sorethumb.store.workspace import make_group_key

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# make_group_key
# ---------------------------------------------------------------------------


def test_group_key_is_hex():
    k = make_group_key({"country": "US", "cat": "A"})
    assert all(c in "0123456789abcdef" for c in k)


def test_group_key_length():
    k = make_group_key({"a": "1"})
    assert len(k) == 32


def test_group_key_stable():
    a = make_group_key({"country": "US", "cat": "A"})
    b = make_group_key({"cat": "A", "country": "US"})  # key order doesn't matter
    assert a == b


def test_group_key_different_for_different_values():
    a = make_group_key({"country": "US"})
    b = make_group_key({"country": "AU"})
    assert a != b


def test_group_key_special_chars_round_trip():
    # Special chars: quotes, semicolon, slash, newline
    group = {"val": 'it\'s a "test"; /path\nnewline'}
    k = make_group_key(group)
    assert len(k) == 32
    # Same key reconstructed from the same values
    assert k == make_group_key(group)


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def test_valid_identifier():
    assert validate_identifier("my_table") == "my_table"
    assert validate_identifier("Column1") == "Column1"
    assert validate_identifier("_private") == "_private"


def test_invalid_identifier_raises():
    with pytest.raises(StoreError, match="Invalid SQL"):
        validate_identifier("1bad")


def test_identifier_rejects_spaces():
    with pytest.raises(StoreError):
        validate_identifier("my column")


def test_identifier_rejects_dash():
    with pytest.raises(StoreError):
        validate_identifier("my-col")


def test_identifier_rejects_dot():
    with pytest.raises(StoreError):
        validate_identifier("schema.table")
