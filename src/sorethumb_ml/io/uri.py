r"""Canonical form of a source URI, for identity only.

``source.uri`` feeds two identities: the logical dataset id (when
``source.dataset_id`` is unset), which every period, total and history row is
filed under, and ``Config.config_hash()``, which keys run ids and model reuse.
Both hash the URI string. On Windows one file has many spellings --
``C:\Data\sales.csv``, ``c:\data\sales.csv``, ``C:/Data/sales.csv``,
``file:///C:/Data/sales.csv`` -- because NTFS is case-insensitive and accepts
both separators, and a literal hash would split one dataset's history (and
miss every cached model) across them.

:func:`canonical_source_key` maps every spelling of a Windows path to one key,
and returns **anything else exactly as given**: POSIX paths, relative paths
(their meaning depends on the working directory, so no spelling can safely be
merged), and http(s) URLs. That keeps every identity derived on Linux or macOS
unchanged. The key is used only for hashing; the user's own spelling is what
gets persisted and displayed (redacted, see ``io.source.redact_source_uri``).
"""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from urllib.parse import unquote, urlparse

# "C:\..." or "C:/..." (an absolute drive path).
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
# "\\server\share..." -- a UNC path. Backslashes only: "//x" is a valid POSIX path.
_UNC_PATH_RE = re.compile(r"^\\\\[^\\/]+[\\/][^\\/]+")
# "/C:/..." -- the path part of "file:///C:/...".
_URI_DRIVE_PATH_RE = re.compile(r"^/[A-Za-z]:/")
# A drive letter where a file:// URI's host would be ("file://C:/data/x.csv").
_DRIVE_NETLOC_RE = re.compile(r"^[A-Za-z]:$")


def _windows_path_from(uri: str) -> str | None:
    """Return *uri* as a Windows path string if it names one, else None."""
    if _DRIVE_PATH_RE.match(uri) or _UNC_PATH_RE.match(uri):
        return uri
    if uri[:7].lower() != "file://":
        return None
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if _DRIVE_NETLOC_RE.match(parsed.netloc):
        return parsed.netloc + path
    if parsed.netloc in ("", "localhost"):
        return path[1:] if _URI_DRIVE_PATH_RE.match(path) else None
    # file://server/share/... is a UNC path.
    return f"\\\\{parsed.netloc}{path}"


def canonical_source_key(uri: str) -> str:
    r"""Return the identity key for *uri*: one key per Windows file, else *uri* unchanged.

    For a Windows drive path, UNC path, or a ``file://`` URI naming one: forward
    slashes, case-folded (NTFS is case-insensitive), no trailing separator, and
    percent-escapes decoded for URIs. For example ``C:\Data\Sales.csv``,
    ``c:/data/sales.csv`` and ``file:///C:/Data/Sales.csv`` all give
    ``c:/data/sales.csv``.
    """
    windows = _windows_path_from(uri.strip())
    if windows is None:
        return uri
    key = PureWindowsPath(windows).as_posix().casefold()
    if len(key) > 3 and key.endswith("/"):  # keep "c:/" itself
        key = key.rstrip("/")
    return key
