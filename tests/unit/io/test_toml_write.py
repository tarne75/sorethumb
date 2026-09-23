"""Unit tests for the TOML literal serialiser (P0-4)."""

from __future__ import annotations

import math
import tomllib

import pytest

from sorethumb.io.toml_write import render_toml_key, render_toml_string, render_toml_value

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# render_toml_string / render_toml_value(str)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\alice\data\events.csv",
        r"\\server\share\events.csv",
        'a "quoted" value',
        "a\ttab\tand\nnewline",
        "control\x01char\x7f",
        "unicode café 日本語",
        "",
    ],
)
def test_render_toml_string_round_trips(value: str):
    rendered = render_toml_string(value)
    assert tomllib.loads(f"x = {rendered}")["x"] == value


def test_render_toml_string_escapes_backslash_and_quote():
    assert render_toml_string(r"C:\a") == r'"C:\\a"'
    assert render_toml_string('a"b') == r'"a\"b"'


# ---------------------------------------------------------------------------
# render_toml_key
# ---------------------------------------------------------------------------


def test_render_toml_key_bare_when_safe():
    assert render_toml_key("n_estimators") == "n_estimators"
    assert render_toml_key("some-key") == "some-key"


def test_render_toml_key_quotes_when_unsafe():
    rendered = render_toml_key("has space")
    assert rendered == '"has space"'
    assert tomllib.loads(f"{rendered} = 1") == {"has space": 1}


# ---------------------------------------------------------------------------
# render_toml_value
# ---------------------------------------------------------------------------


def test_render_toml_value_bool():
    assert render_toml_value(True) == "true"
    assert render_toml_value(False) == "false"


def test_render_toml_value_int():
    assert render_toml_value(42) == "42"
    assert render_toml_value(-7) == "-7"


def test_render_toml_value_float_finite():
    rendered = render_toml_value(3.5)
    assert tomllib.loads(f"x = {rendered}")["x"] == 3.5


def test_render_toml_value_float_nan_and_inf():
    assert math.isnan(tomllib.loads(f"x = {render_toml_value(float('nan'))}")["x"])
    assert tomllib.loads(f"x = {render_toml_value(float('inf'))}")["x"] == float("inf")
    assert tomllib.loads(f"x = {render_toml_value(float('-inf'))}")["x"] == float("-inf")


def test_render_toml_value_list():
    rendered = render_toml_value([1, "two", True])
    assert tomllib.loads(f"x = {rendered}")["x"] == [1, "two", True]


def test_render_toml_value_empty_dict():
    assert render_toml_value({}) == "{}"


def test_render_toml_value_nested_dict_round_trips():
    value = {"n_jobs": 2, "bootstrap": True, "nested": {"a": 1, "b": [1, 2]}}
    rendered = render_toml_value(value)
    assert tomllib.loads(f"x = {rendered}")["x"] == value


def test_render_toml_value_dict_with_key_needing_quotes():
    value = {"has space": 1}
    rendered = render_toml_value(value)
    assert tomllib.loads(f"x = {rendered}")["x"] == value


def test_render_toml_value_rejects_none():
    with pytest.raises(TypeError):
        render_toml_value(None)


def test_render_toml_value_rejects_unrepresentable_object():
    with pytest.raises(TypeError):
        render_toml_value(object())
