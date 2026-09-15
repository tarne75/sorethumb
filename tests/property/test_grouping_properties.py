"""Hypothesis property tests for make_group_key: invariant to the order keys
were supplied in, and distinguishes null from falsy-looking stand-ins and
distinct types that would otherwise stringify the same way.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sorethumb.store.workspace import make_group_key
from tests.factories.hypothesis_profiles import scaled_examples

pytestmark = pytest.mark.property

_JSON_SAFE_VALUE = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(max_size=20),
)
_GROUP_VALUES = st.dictionaries(st.text(min_size=1, max_size=10), _JSON_SAFE_VALUE, min_size=1, max_size=6)


@given(values=_GROUP_VALUES, shuffle_seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=scaled_examples(200))
def test_make_group_key_is_mapping_order_invariant(values: dict[str, object], shuffle_seed: int) -> None:
    items = list(values.items())
    shuffled = items.copy()
    random.Random(shuffle_seed).shuffle(shuffled)
    assert make_group_key(dict(items)) == make_group_key(dict(shuffled))


def test_make_group_key_distinguishes_null_empty_and_literal_none_string() -> None:
    keys = {
        make_group_key({"g": None}),
        make_group_key({"g": ""}),
        make_group_key({"g": "None"}),
    }
    assert len(keys) == 3, "None, empty string, and the literal string 'None' must hash distinctly"


def test_make_group_key_distinguishes_types_with_the_same_string_form() -> None:
    assert make_group_key({"g": 1}) != make_group_key({"g": "1"})
    assert make_group_key({"g": True}) != make_group_key({"g": "True"})
