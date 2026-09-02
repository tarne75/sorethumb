"""Source resolution: turn a SourceConfig URI into a local file path.

Local paths are returned immediately. HTTP(S) URIs are downloaded, cached by
content fingerprint, and returned as a local path. Auth credentials are read
from the environment at call time and are never logged or persisted.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from sorethumb.config import SourceConfig
from sorethumb.errors import SourceError
from sorethumb.io.fingerprint import content_fingerprint

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 2.0


def resolve_source(config: SourceConfig, cache_dir: Path) -> Path:
    """Return a local ``Path`` for the source described by *config*.

    For local URIs: expand ``~`` and resolve relative paths.
    For HTTP(S) URIs: download to *cache_dir*, using cached copy when content
    has not changed (content-fingerprint match).

    Raises:
        SourceError: URI scheme is not supported, or the download fails.

    """
    parsed = urlparse(config.uri)

    if parsed.scheme in ("", "file"):
        return _resolve_local(config.uri)

    if parsed.scheme in ("http", "https"):
        return _resolve_http(config, cache_dir)

    raise SourceError(f"Unsupported URI scheme '{parsed.scheme}' in '{config.uri}'")


def _resolve_local(uri: str) -> Path:
    path = Path(uri).expanduser().resolve()
    if not path.exists():
        raise SourceError(f"Local source file not found: {path}")
    return path


def _resolve_http(config: SourceConfig, cache_dir: Path) -> Path:
    """Download the URI, cache by content fingerprint, return cached path."""
    url = config.uri
    headers = _build_auth_headers(config)

    tmp_path = cache_dir / "_download_tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Downloading source: %s", url)
    _download_to(url, headers, tmp_path)

    fp = content_fingerprint(tmp_path)
    cached_dir = cache_dir / fp
    ext = _extension_from_url(url, config)
    cached_file = cached_dir / f"data{ext}"

    if cached_file.exists():
        logger.info("Source cache hit (fp=%s): skipping download", fp[:8])
        tmp_path.unlink(missing_ok=True)
        return cached_file

    cached_dir.mkdir(parents=True, exist_ok=True)
    tmp_path.rename(cached_file)
    logger.info("Source cached (fp=%s): %s", fp[:8], cached_file)
    return cached_file


def _build_auth_headers(config: SourceConfig) -> dict[str, str]:
    if config.auth == "none" or not config.auth_env_var:
        return {}
    token = os.environ.get(config.auth_env_var, "")
    if not token:
        raise SourceError(
            f"Auth env var '{config.auth_env_var}' is not set or empty. Set it before calling resolve_source."
        )
    if config.auth == "bearer":
        return {"Authorization": f"Bearer {token}"}
    if config.auth == "basic":
        return {"Authorization": f"Basic {token}"}
    return {}  # unreachable given Literal type


def _download_to(url: str, headers: dict[str, str], dest: Path) -> None:
    transport = httpx.HTTPTransport(retries=1)
    with httpx.Client(transport=transport, follow_redirects=True, timeout=120.0) as client:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                        wait = _BACKOFF_BASE**attempt
                        logger.warning("HTTP %s from %s; retrying in %.0fs", resp.status_code, url, wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code >= 400:
                        raise SourceError(f"HTTP {resp.status_code} downloading '{url}'")
                    with dest.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
                return
            except httpx.TransportError as exc:
                if attempt < _MAX_ATTEMPTS - 1:
                    wait = _BACKOFF_BASE**attempt
                    logger.warning("Network error (%s); retrying in %.0fs", exc, wait)
                    time.sleep(wait)
                    continue
                raise SourceError(
                    f"Failed to download '{url}' after {_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc


def _extension_from_url(url: str, config: SourceConfig) -> str:
    fmt = config.format
    if fmt != "auto":
        return f".{fmt}"
    path_part = urlparse(url).path
    for ext in (".parquet", ".csv", ".tsv", ".jsonl", ".ndjson", ".json", ".gz"):
        if path_part.endswith(ext):
            return ext
    return ".bin"
