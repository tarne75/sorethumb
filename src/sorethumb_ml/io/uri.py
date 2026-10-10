r"""Canonical form and safe display form of a source URI.

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
gets displayed (redacted, see :func:`display_source_uri`).

Provenance: what may be stored, logged or reported
--------------------------------------------------
The full URI -- userinfo, every query value, any fragment -- exists only in the
in-memory config and in the HTTP request it describes. Everything at rest or on
screen uses one of two derived forms instead:

* :func:`display_source_uri` -- scheme, host, non-default port, path and the
  query **key names**, with every query value replaced by ``REDACTED``. No list of
  "sensitive" names is consulted, so an unlisted name (``client_secret``,
  ``jwt``, ``accessKey``, ...) is protected exactly like a listed one.
* :func:`source_digest` -- a one-way SHA-256 of the canonical full URI, so two
  sources that differ only in a query value stay distinguishable without
  recording either value.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PureWindowsPath
from urllib.parse import SplitResult, unquote, urlparse, urlsplit, urlunsplit

REDACTED = "REDACTED"

_DEFAULT_PORTS = {"http": 80, "https": 443}
_DIGEST_DOMAIN = b"sorethumb.source-uri.v1\0"

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


def _redact_query(query: str) -> str:
    """Keep each query key name, replace every value (blank ones too) with REDACTED.

    Order and duplicate keys are preserved. A segment with no ``=`` could be a flag
    *or* a bare secret (``?abc123``), and there is no telling which, so the whole
    segment is replaced.
    """
    parts = []
    for segment in query.split("&"):
        if not segment:
            continue
        key, sep, _ = segment.partition("=")
        parts.append(f"{key}={REDACTED}" if sep else REDACTED)
    return "&".join(parts)


def display_source_uri(uri: str) -> str:
    """Return *uri* in the only form that may be stored, logged or reported.

    Scheme, host (lower-cased), a non-default port, the path and the query key
    names are kept; userinfo becomes ``***@``; every query value becomes
    ``REDACTED``; the fragment is dropped. Local paths (no URL scheme) are returned
    unchanged. Idempotent: ``display(display(x)) == display(x)``.

    Path segments are kept verbatim, so a secret embedded in the *path* is not
    hidden -- see SECURITY.md.
    """
    stripped = uri.strip()
    try:
        parts = urlsplit(stripped)
        hostname = parts.hostname
    except ValueError:
        return "<unparseable source URI>"
    if len(parts.scheme) < 2:  # no scheme, or a Windows drive letter ("C:\\...")
        return uri

    netloc = ""
    if parts.netloc:
        host = hostname or ""
        if ":" in host:
            host = f"[{host}]"
        try:
            port = parts.port
        except ValueError:
            port = None
        if port is not None and port != _DEFAULT_PORTS.get(parts.scheme):
            host = f"{host}:{port}"
        netloc = f"***@{host}" if "@" in parts.netloc else host
    return urlunsplit(SplitResult(parts.scheme, netloc, parts.path, _redact_query(parts.query), ""))


def query_key_names(uri: str) -> frozenset[str]:
    """Return the set of query key names in *uri* (never values), read from its display form."""
    query = urlsplit(display_source_uri(uri)).query
    return frozenset(segment.partition("=")[0] for segment in query.split("&") if segment)


def _canonical_full_uri(uri: str) -> str:
    stripped = uri.strip()
    try:
        parts = urlsplit(stripped)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return stripped
    if parts.scheme not in _DEFAULT_PORTS or not parts.netloc:
        return canonical_source_key(stripped)
    host = hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != _DEFAULT_PORTS[parts.scheme]:
        host = f"{host}:{port}"
    userinfo = parts.netloc.rpartition("@")[0]
    netloc = f"{userinfo}@{host}" if "@" in parts.netloc else host
    return urlunsplit(SplitResult(parts.scheme, netloc, parts.path, parts.query, ""))


def source_digest(uri: str) -> str:
    """One-way SHA-256 of the canonical full *uri*, query values included.

    Stable across spelling (host case, default port, fragment, Windows path
    spelling) and different for any two sources whose URIs differ in a query value,
    so it can be persisted next to :func:`display_source_uri` to tell them apart.
    It is a plain hash, not a keyed one: a low-entropy secret could be confirmed by
    guessing it -- see SECURITY.md.
    """
    return hashlib.sha256(_DIGEST_DOMAIN + _canonical_full_uri(uri).encode("utf-8")).hexdigest()
