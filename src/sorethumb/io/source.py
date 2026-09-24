"""Source resolution: turn a SourceConfig URI into a local file path.

Local paths (including Windows drive-letter and UNC paths, and ``file://``
URIs) are returned immediately. HTTP(S) URIs are downloaded, cached by
content fingerprint, and returned as a local path. Auth credentials are read
from the environment at call time and are never logged or persisted.

Download hardening
-------------------
- Every download writes to a unique per-attempt temp file (never a fixed
  shared name), fsyncs it, and promotes it with ``os.replace`` -- several
  concurrent downloads into the same cache_dir can never clobber or
  misattribute each other's bytes.
- ``source.max_download_bytes`` bounds both the declared ``Content-Length``
  (checked before any body is read) and the actual streamed size.
- Redirects are followed manually, up to ``_MAX_REDIRECTS`` hops. Any hop
  whose target host differs from the original request's host must resolve
  to a public, non-reserved address -- see ``_assert_host_is_safe`` --
  refusing an obvious pivot to a cloud metadata endpoint (169.254.169.254)
  or another internal/link-local target a malicious or compromised remote
  server redirects to. A same-host redirect (http -> https, a path change)
  is never blocked: the user already asked for that host explicitly.
- An ``Authorization`` header (``source.auth``/``auth_env_var``) is only
  ever sent to the exact origin (scheme, host, effective port) the
  original request targeted -- any redirect to a *different* origin, even
  a same-host scheme change, gets the request without it, so a bearer or
  basic credential can never leak to another site a compromised or
  malicious remote server redirects to. See ``_is_same_origin`` /
  ``_strip_authorization``. Separately, an HTTPS -> HTTP downgrade at any
  hop is refused outright (not just stripped of credentials), since it
  silently drops transport security for the response body too.

Redaction
---------
:func:`redact_source_uri` strips URI userinfo (``user:pass@``) and known
sensitive query parameters (signed-URL tokens, API keys, ...) before a
source URI is logged, persisted (``dataset.source_uri``, ``run.config_json``),
or rendered into a report. Called at every one of those call sites; nothing
downstream needs to remember to do it.
"""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import logging
import os
import re
import socket
import tempfile
import time
from pathlib import Path
from urllib.parse import ParseResult, parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import url2pathname

import httpx

from sorethumb._atomic import _fsync_path
from sorethumb.config import SourceConfig
from sorethumb.errors import SourceError
from sorethumb.io.fingerprint import content_fingerprint
from sorethumb.io.readers import _FORMAT_EXTENSIONS

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 2.0
_MAX_REDIRECTS = 5

# "C:\..." or "C:/..." -- urlparse would otherwise read the drive letter as a
# URL scheme ("c"), rejecting every Windows absolute path as an unsupported
# scheme.
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

# Query parameter names (case-insensitive) that commonly carry a signed-URL
# token, API key, or other bearer secret -- stripped by redact_source_uri.
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "sig",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-goog-signature",
        "x-ms-signature",
        "token",
        "access_token",
        "api_key",
        "apikey",
        "password",
        "secret",
        "auth",
        "key",
    }
)


def redact_source_uri(uri: str) -> str:
    """Strip URI userinfo and known sensitive query parameters from *uri*.

    Applied before a source URI is logged, persisted (dataset.source_uri,
    run.config_json), or rendered into a report -- a signed download URL or
    one with embedded ``user:pass@`` credentials must never end up at rest
    in a database, a log file, or an HTML report handed to someone else.
    Local paths (no recognisable userinfo or query string) pass through
    unchanged.
    """
    parsed = urlparse(uri)
    if not parsed.scheme or not parsed.netloc:
        return uri  # a local path, not a URL -- nothing to redact

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]

    query = urlencode(
        [
            (k, "REDACTED" if k.lower() in _SENSITIVE_QUERY_KEYS else v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunparse(parsed._replace(netloc=netloc, query=query))


def resolve_source(config: SourceConfig, cache_dir: Path) -> Path:
    """Return a local ``Path`` for the source described by *config*.

    For local paths (including Windows drive-letter/UNC paths and
    ``file://`` URIs): expand ``~`` and resolve relative paths.
    For HTTP(S) URIs: download to *cache_dir*, using cached copy when content
    has not changed (content-fingerprint match).

    Raises:
        SourceError: URI scheme is not supported, or the download fails.

    """
    uri = config.uri

    if _WINDOWS_DRIVE_RE.match(uri):
        return _resolve_local(uri)

    parsed = urlparse(uri)

    if parsed.scheme == "file":
        return _resolve_local(_file_uri_to_path(parsed))

    if parsed.scheme == "":
        return _resolve_local(uri)

    if parsed.scheme in ("http", "https"):
        return _resolve_http(config, cache_dir)

    raise SourceError(f"Unsupported URI scheme '{parsed.scheme}' in '{redact_source_uri(config.uri)}'")


def _file_uri_to_path(parsed: ParseResult) -> str:
    """Convert a parsed ``file://`` URI to a local path string.

    ``url2pathname`` is ``urllib.request``'s platform dispatch (posixpath vs.
    ``nturl2path`` on Windows) for exactly this: percent-decoding with the
    right separator conventions, including a Windows drive letter that
    followed the third slash (``file:///C:/...``).
    """
    netloc = parsed.netloc
    if netloc and netloc != "localhost":
        # file://server/share/path -> a UNC path.
        return f"\\\\{netloc}{url2pathname(parsed.path)}"
    return url2pathname(parsed.path)


def _resolve_local(uri: str) -> Path:
    path = Path(uri).expanduser().resolve()
    if not path.exists():
        raise SourceError(f"Local source file not found: {path}")
    return path


def _resolve_http(config: SourceConfig, cache_dir: Path) -> Path:
    """Download the URI, cache by content fingerprint, return cached path."""
    url = config.uri
    headers = _build_auth_headers(config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # A unique per-attempt scratch file -- never a fixed shared name.
    # Several concurrent downloads into the same cache_dir (different URLs,
    # or the same URL raced by two callers) must never share one temp path:
    # the final cache location is only known *after* the download completes
    # and its content fingerprint is computed, so this file has to survive
    # on its own, unclobbered, for the whole download+fingerprint step.
    fd, tmp_name = tempfile.mkstemp(dir=str(cache_dir), prefix=".download-", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        logger.debug("Downloading source: %s", redact_source_uri(url))
        _download_to(url, headers, tmp_path, max_bytes=config.max_download_bytes)

        ext = _extension_from_url(url, config)

        if not config.cache:
            # source.cache = False: keep only a single transient copy, overwritten
            # every call, and never a fingerprint-keyed cache dir.
            uncached_file = cache_dir / f"uncached_data{ext}"
            _promote(tmp_path, uncached_file)
            logger.info("Source not cached (source.cache=False): %s", uncached_file)
            return uncached_file

        fp = content_fingerprint(tmp_path)
        cached_dir = cache_dir / fp
        cached_file = cached_dir / f"data{ext}"

        if cached_file.exists():
            logger.info("Source cache hit (fp=%s): skipping download", fp[:8])
            tmp_path.unlink(missing_ok=True)
            return cached_file

        cached_dir.mkdir(parents=True, exist_ok=True)
        _promote(tmp_path, cached_file)
        logger.info("Source cached (fp=%s): %s", fp[:8], cached_file)
        return cached_file
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _promote(tmp_path: Path, dest: Path) -> None:
    """Fsync *tmp_path* then atomically rename it to *dest* (same filesystem)."""
    _fsync_path(tmp_path)
    os.replace(tmp_path, dest)  # noqa: PTH105 — os.replace IS the atomic-rename primitive


def _build_auth_headers(config: SourceConfig) -> dict[str, str]:
    if config.auth == "none" or not config.auth_env_var:
        return {}
    # .strip() -- a token/credential read from an env var populated via
    # `export X=$(cat file)` or a CI secrets manager commonly carries a
    # trailing newline or padding space; sent verbatim that breaks the
    # header (and, for "bearer", is silently wrong rather than rejected).
    # A whitespace-only value is therefore treated the same as unset.
    token = os.environ.get(config.auth_env_var, "").strip()
    if not token:
        raise SourceError(
            f"Auth env var '{config.auth_env_var}' is not set or empty. Set it before calling resolve_source."
        )
    if config.auth == "bearer":
        return {"Authorization": f"Bearer {token}"}
    if config.auth == "basic":
        # auth_env_var holds "user:password" in plain text (see SourceConfig
        # docs) -- this is the one place it gets base64-encoded, not the
        # caller's responsibility.
        encoded = base64.b64encode(token.encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}
    return {}  # unreachable given Literal type


def _assert_host_is_safe(url: httpx.URL) -> None:
    """Refuse a request/redirect whose host resolves to an obvious internal target.

    Link-local (including the 169.254.169.254 cloud metadata address),
    loopback, private, reserved, or multicast.

    Not a defence against DNS rebinding (the resolved address isn't pinned
    for the actual connection): it catches the obvious, common case the plan
    calls for -- a remote server redirecting the fetcher somewhere internal
    -- not a sophisticated network-level attack.
    """
    host = url.host
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise SourceError(f"Cannot resolve host '{host}': {exc}") from exc

    for _family, _type, _proto, _canon, sockaddr in infos:
        addr = str(sockaddr[0]).split("%", 1)[0]  # strip an IPv6 zone id, if present
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            # Can't parse it -- fail closed rather than assume it's safe.
            raise SourceError(
                f"Refusing to fetch '{url}': host {host!r} resolved to unparseable {addr!r}."
            ) from None
        if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved or ip.is_multicast:
            raise SourceError(
                f"Refusing to fetch '{url}': host {host!r} resolves to {addr} "
                "(loopback/link-local/private/reserved/multicast), which looks like "
                "an internal or cloud-metadata target rather than a public dataset host."
            )


class _RetryableStatusError(Exception):
    """Internal signal: a retryable HTTP status was returned; loop again."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(str(status_code))


def _download_to(
    url: str,
    headers: dict[str, str],
    dest: Path,
    *,
    max_bytes: int,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Download *url* to *dest*.

    *transport* is an injection point for tests (e.g. ``httpx.MockTransport``)
    to simulate redirects, oversized responses, or unsafe-host targets
    without a real network call; production callers never pass it.
    """
    if transport is None:
        transport = httpx.HTTPTransport(retries=1)
    with httpx.Client(transport=transport, timeout=120.0) as client:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                _download_once(client, url, headers, dest, max_bytes=max_bytes)
                return
            except httpx.TransportError as exc:
                if attempt < _MAX_ATTEMPTS - 1:
                    wait = _BACKOFF_BASE**attempt
                    logger.warning("Network error (%s); retrying in %.0fs", exc, wait)
                    time.sleep(wait)
                    continue
                raise SourceError(
                    f"Failed to download '{redact_source_uri(url)}' after {_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
            except _RetryableStatusError as retry:
                if attempt < _MAX_ATTEMPTS - 1:
                    wait = _BACKOFF_BASE**attempt
                    logger.warning(
                        "HTTP %s from %s; retrying in %.0fs", retry.status_code, redact_source_uri(url), wait
                    )
                    time.sleep(wait)
                    continue
                raise SourceError(
                    f"HTTP {retry.status_code} downloading '{redact_source_uri(url)}'"
                ) from None


def _effective_port(url: httpx.URL) -> int:
    """Return the port a connection to *url* actually uses.

    httpx already normalises an explicit default port (e.g. ``:443`` on
    ``https://``) away to ``None``, so an explicit-vs-implicit default port
    is never mistaken for a real difference.
    """
    if url.port is not None:
        return url.port
    return 443 if url.scheme == "https" else 80


def _is_same_origin(a: httpx.URL, b: httpx.URL) -> bool:
    """Return True if scheme, host, and effective port all match.

    httpx already lower-cases both ``.scheme`` and ``.host``.
    """
    return a.scheme == b.scheme and a.host == b.host and _effective_port(a) == _effective_port(b)


def _strip_authorization(headers: dict[str, str]) -> dict[str, str]:
    """Drop any Authorization header, case-insensitively.

    Used whenever a redirect moves to a different origin than the one the
    caller configured, so a bearer/basic credential is never sent anywhere
    but the exact (scheme, host, port) `source.uri` named.
    """
    return {k: v for k, v in headers.items() if k.lower() != "authorization"}


def _download_once(
    client: httpx.Client, url: str, headers: dict[str, str], dest: Path, *, max_bytes: int
) -> None:
    """One download attempt.

    Follows redirects manually (host-checked, capped), then streams the
    final response to *dest* with a hard size ceiling.

    Authorization is only ever sent to the exact origin (scheme, host,
    effective port) the original request targeted -- a redirect to any
    other origin gets the request without it, regardless of how "safe"
    that other host's resolved address looks. An HTTPS -> HTTP downgrade
    at any hop is refused outright, independent of whether credentials are
    even configured, since it silently drops transport security for the
    response body too, not just for an auth header.
    """
    request = client.build_request("GET", url, headers=headers)
    original_host = request.url.host
    original_url = request.url

    for _hop in range(_MAX_REDIRECTS + 1):
        if request.url.host != original_host:
            _assert_host_is_safe(request.url)

        resp = client.send(request, stream=True)
        try:
            if resp.is_redirect:
                next_url = resp.headers.get("location")
                if not next_url:
                    raise SourceError(
                        f"HTTP {resp.status_code} redirect from '{url}' had no Location header."
                    )
                target = request.url.join(next_url)
                if request.url.scheme == "https" and target.scheme == "http":
                    raise SourceError(
                        f"Refusing HTTPS -> HTTP downgrade redirect while fetching "
                        f"'{redact_source_uri(url)}' (from {request.url.scheme}://{request.url.host} "
                        f"to {target.scheme}://{target.host})."
                    )
                next_headers = (
                    headers if _is_same_origin(original_url, target) else _strip_authorization(headers)
                )
                request = client.build_request("GET", target, headers=next_headers)
                continue

            if resp.status_code in _RETRYABLE_STATUS:
                raise _RetryableStatusError(resp.status_code)
            if resp.status_code >= 400:
                raise SourceError(f"HTTP {resp.status_code} downloading '{redact_source_uri(url)}'")

            declared = resp.headers.get("content-length")
            if declared is not None and int(declared) > max_bytes:
                raise SourceError(
                    f"Refusing to download '{redact_source_uri(url)}': declared size "
                    f"{int(declared):,} bytes exceeds source.max_download_bytes={max_bytes:,}."
                )

            written = 0
            with dest.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    written += len(chunk)
                    if written > max_bytes:
                        raise SourceError(
                            f"Refusing to download '{redact_source_uri(url)}': streamed size "
                            f"exceeded source.max_download_bytes={max_bytes:,}."
                        )
                    fh.write(chunk)
            return
        finally:
            resp.close()

    raise SourceError(f"Too many redirects (> {_MAX_REDIRECTS}) fetching '{redact_source_uri(url)}'.")


def _extension_from_url(url: str, config: SourceConfig) -> str:
    fmt = config.format
    if fmt != "auto":
        return f".{fmt}"
    path_part = urlparse(url).path.lower()
    # Longest first, so "data.csv.gz" matches ".csv.gz" (needed for readers.py
    # to auto-detect it as a gzipped CSV) rather than the unrelated, shorter
    # ".gz" a naive shortest/first match would settle for.
    all_exts = sorted((e for exts in _FORMAT_EXTENSIONS.values() for e in exts), key=len, reverse=True)
    for ext in all_exts:
        if path_part.endswith(ext):
            return ext
    return ".bin"
