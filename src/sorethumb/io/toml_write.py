r"""Correct TOML literal serialisation for generated config files.

Python's stdlib has no TOML *writer* (only ``tomllib``, read-only), and
CLI-generated config content (the ``sorethumb init``/``sorethumb run``
starter file, detector params written back from a live ``Config``)
previously interpolated values into f-strings by hand -- broken for
anything containing a backslash (a Windows path, ``C:\Users\...``), an
embedded quote, a control character, or a nested dict (rendered via
``json.dumps``, which produces JSON object syntax -- quoted keys, ``:``
separators -- not valid TOML inline-table syntax).

This module is the single place that turns a Python value into a valid
TOML literal; nothing else in the CLI layer should hand-format one.
"""

from __future__ import annotations

import math
import re

# A TOML "bare key": one or more letters, digits, underscores, or hyphens.
# Anything else (spaces, dots, unicode, empty string) needs a quoted key.
_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# TOML basic-string escapes that have a short form; every other control
# character falls back to \\uXXXX in render_toml_string.
_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def render_toml_string(s: str) -> str:
    """Render *s* as a TOML basic string literal.

    Correct for a Windows path (backslashes), an embedded quote, tabs and
    newlines, and arbitrary Unicode (left as-is -- TOML basic strings are
    UTF-8 and need no escaping for ordinary printable text; only control
    characters and the two syntactically significant characters,
    backslash and double-quote, are escaped).
    """
    out = ['"']
    for ch in s:
        escape = _STRING_ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def render_toml_key(key: str) -> str:
    """Render *key* as a TOML key: bare if valid, quoted otherwise."""
    return key if _BARE_KEY_RE.match(key) else render_toml_string(key)


def render_toml_value(value: object) -> str:
    """Render *value* as a valid TOML value literal.

    Handles every value type the Config layer can actually produce: bool,
    int, float (including NaN/±inf, which TOML has literals for), str,
    list/tuple (as a TOML array), and dict (as a TOML inline table, with
    each key rendered via :func:`render_toml_key` and each value
    recursively via this function -- this is what makes a nested
    ``extra_params`` dict inside a detector's ``params`` render correctly).

    TOML has no null. A caller holding an optional/``None`` value must
    omit the key entirely (or render a commented placeholder) rather than
    call this with ``None`` -- it raises ``TypeError`` for that and any
    other type with no TOML literal form, so a caller that wants a
    graceful fallback for such values should catch that explicitly rather
    than have one silently render as something wrong.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return render_toml_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        pairs = ", ".join(f"{render_toml_key(str(k))} = {render_toml_value(v)}" for k, v in value.items())
        return "{ " + pairs + " }"
    raise TypeError(f"No TOML literal representation for {type(value).__name__}: {value!r}")
