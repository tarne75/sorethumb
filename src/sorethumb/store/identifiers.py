"""SQL identifier validation.

Column and table names cannot be parameterised in SQLite — only values can.
Validate any identifier that comes from outside the library before interpolating
it into a query string. The pattern is deliberately narrow: alphanumeric plus
underscore, starting with a letter or underscore. No spaces, hyphens, dots or
non-ASCII.

This module is the only place in the library that does identifier-level string
safety. Everything else uses bound parameters.
"""

from __future__ import annotations

import re

from sorethumb.errors import StoreError

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, context: str = "identifier") -> str:
    """Return *name* unchanged if it is a valid SQL identifier, else raise StoreError."""
    if not _IDENT_RE.match(name):
        msg = (
            f"Invalid SQL {context} {name!r}: must match ^[A-Za-z_][A-Za-z0-9_]*$. "
            "Do not put data values in identifier positions — use a parameterised column."
        )
        raise StoreError(msg)
    return name
