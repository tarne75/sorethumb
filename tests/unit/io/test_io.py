"""Unit tests for the IO layer (M1a)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl
import pytest

from sorethumb.errors import ColumnDroppedWarning, SchemaError, SourceError
from sorethumb.io.fingerprint import (
    content_fingerprint,
    logical_dataset_id,
    schema_fingerprint,
    snapshot_fingerprint,
)
from sorethumb.io.nested import derive_array_features, unnest_all
from sorethumb.io.readers import read_frame
from sorethumb.io.source import resolve_source
from tests.synth import make_frame

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_schema_fingerprint_stable() -> None:
    df, _ = make_frame(n_rows=10, seed=0)
    assert schema_fingerprint(df) == schema_fingerprint(df)


def test_schema_fingerprint_changes_on_column_add() -> None:
    df, _ = make_frame(n_rows=10, seed=0)
    df2 = df.with_columns(pl.lit(1).alias("extra"))
    assert schema_fingerprint(df) != schema_fingerprint(df2)


def test_schema_fingerprint_same_for_lazy_and_eager() -> None:
    df, _ = make_frame(n_rows=10, seed=0)
    assert schema_fingerprint(df) == schema_fingerprint(df.lazy())


def test_content_fingerprint_bytes_consistency() -> None:
    data = b"hello world"
    fp1 = content_fingerprint(data)
    fp2 = content_fingerprint(data)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_content_fingerprint_file_matches_bytes(tmp_path: Path) -> None:
    data = b"test content 123"
    p = tmp_path / "data.bin"
    p.write_bytes(data)
    assert content_fingerprint(p) == content_fingerprint(data)


def test_content_fingerprint_different_content(tmp_path: Path) -> None:
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"aaa")
    p2.write_bytes(b"bbb")
    assert content_fingerprint(p1) != content_fingerprint(p2)


# ---------------------------------------------------------------------------
# Logical dataset identity vs. snapshot identity
# ---------------------------------------------------------------------------


def test_logical_dataset_id_prefers_configured_value() -> None:
    assert logical_dataset_id("sales-eu", "/data/whatever.parquet") == "sales-eu"


def test_logical_dataset_id_derived_is_stable_and_readable() -> None:
    a = logical_dataset_id(None, "/srv/data/sales.parquet")
    b = logical_dataset_id(None, "/srv/data/sales.parquet")
    assert a == b
    assert a.startswith("sales-")


def test_logical_dataset_id_derived_distinguishes_same_basename() -> None:
    eu = logical_dataset_id(None, "/srv/eu/sales.parquet")
    us = logical_dataset_id(None, "/srv/us/sales.parquet")
    assert eu != us
    assert eu.startswith("sales-")
    assert us.startswith("sales-")


def test_logical_dataset_id_ignores_query_string() -> None:
    plain = logical_dataset_id(None, "https://host/data/sales.parquet")
    signed = logical_dataset_id(None, "https://host/data/sales.parquet?sig=abc123&exp=999")
    assert plain == signed


def test_snapshot_fingerprint_tracks_content_and_schema() -> None:
    base = snapshot_fingerprint("c" * 64, "s" * 64)
    assert base == snapshot_fingerprint("c" * 64, "s" * 64)
    assert base != snapshot_fingerprint("d" * 64, "s" * 64)  # content changed
    assert base != snapshot_fingerprint("c" * 64, "t" * 64)  # schema changed


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_read_csv(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=20, seed=1)
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(csv_path))
    lf = read_frame(csv_path, cfg)
    result = lf.collect()
    assert result.shape[0] == 20
    assert set(result.columns) == set(df.columns)


def test_read_parquet(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=30, seed=2)
    pq_path = tmp_path / "data.parquet"
    df.write_parquet(pq_path)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(pq_path))
    lf = read_frame(pq_path, cfg)
    result = lf.collect()
    assert result.shape == df.shape


def test_read_jsonl(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=10, seed=3)
    jsonl_path = tmp_path / "data.jsonl"
    df.write_ndjson(jsonl_path)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(jsonl_path))
    lf = read_frame(jsonl_path, cfg)
    result = lf.collect()
    assert result.shape[0] == 10


def test_read_single_column_csv_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("only_col\n1\n2\n3\n")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SchemaError, match="only one column"):
        read_frame(p, cfg).collect()


def test_read_auto_format_unknown_ext_raises(tmp_path: Path) -> None:
    p = tmp_path / "data.xyz"
    p.write_bytes(b"whatever")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SourceError, match="auto-detect"):
        read_frame(p, cfg)


def test_read_explicit_format_overrides_extension(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=5, seed=0)
    p = tmp_path / "data.noext"
    df.write_csv(p)

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p), format="csv")
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 5


def test_read_tsv_injects_tab_separator(tmp_path: Path) -> None:
    """TSV format injects separator='\\t' when the caller didn't set one."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 3)
    assert "col_a" in result.columns


def test_read_tsv_explicit_separator_not_overridden(tmp_path: Path) -> None:
    """TSV does NOT inject a separator if the caller already provided one."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")

    cfg = SourceConfig(uri=str(p), read_options={"separator": "\t"})
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 2


def test_read_json(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    p = tmp_path / "data.json"
    rows = [{"x": float(i), "y": float(i * 2), "z": float(i + 1)} for i in range(5)]
    p.write_text(json.dumps(rows))

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (5, 3)
    assert "x" in result.columns


def test_read_all_string_schema_raises(tmp_path: Path) -> None:
    """A schema where every column is String (no numeric signal at all) raises."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "strings.csv"
    p.write_text("col_a,col_b,col_c\nfoo,bar,baz\nhello,world,test\n")

    cfg = SourceConfig(uri=str(p))
    with pytest.raises(SchemaError, match="String"):
        read_frame(p, cfg).collect()


# ---------------------------------------------------------------------------
# TSF reader
# ---------------------------------------------------------------------------


def test_read_tsf_basic(tmp_path: Path) -> None:
    """Basic TSF file: one series, no attributes, single data row."""
    from sorethumb.config import SourceConfig

    content = "@data\n1.0,2.0,3.0\n"
    p = tmp_path / "basic.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (1, 3)
    assert result["value_0"][0] == pytest.approx(1.0)
    assert result["value_2"][0] == pytest.approx(3.0)


def test_read_tsf_with_numeric_attribute(tmp_path: Path) -> None:
    """A numeric @attribute column is parsed as int/float."""
    from sorethumb.config import SourceConfig

    content = "@attribute series_id numeric\n@data\n42:10.0,20.0,30.0\n7:1.5,2.5,3.5\n"
    p = tmp_path / "attrs.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 4)
    assert result["series_id"].to_list() == [42, 7]
    assert result["value_0"].to_list() == pytest.approx([10.0, 1.5])


def test_read_tsf_with_float_numeric_attribute(tmp_path: Path) -> None:
    """A numeric attribute containing a decimal point parses as float."""
    from sorethumb.config import SourceConfig

    content = "@attribute score numeric\n@data\n1.5:10.0,20.0,30.0\n"
    p = tmp_path / "float_attr.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["score"][0] == pytest.approx(1.5)


def test_read_tsf_with_string_attribute(tmp_path: Path) -> None:
    """A string @attribute keeps its value as-is."""
    from sorethumb.config import SourceConfig

    content = "@attribute category string\n@data\ntrain:1.0,2.0,3.0\ntest:4.0,5.0,6.0\n"
    p = tmp_path / "str_attr.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 4)
    assert result["category"].to_list() == ["train", "test"]


def test_read_tsf_with_date_attribute(tmp_path: Path) -> None:
    """A date @attribute is treated as a plain string (kept as-is)."""
    from sorethumb.config import SourceConfig

    content = "@attribute start_timestamp date\n@data\n2020-01-01:1.0,2.0\n2020-01-02:3.0,4.0\n"
    p = tmp_path / "date_attr.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["start_timestamp"].to_list() == ["2020-01-01", "2020-01-02"]


def test_read_tsf_missing_values(tmp_path: Path) -> None:
    """'?' tokens and empty tokens both become None."""
    from sorethumb.config import SourceConfig

    content = "@data\n1.0,?,3.0\n"
    p = tmp_path / "missing.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["value_0"][0] == pytest.approx(1.0)
    assert result["value_1"][0] is None
    assert result["value_2"][0] == pytest.approx(3.0)


def test_read_tsf_variable_length_series_padded(tmp_path: Path) -> None:
    """Shorter series are padded to the max length with None."""
    from sorethumb.config import SourceConfig

    content = "@data\n1.0,2.0,3.0\n4.0,5.0\n"
    p = tmp_path / "varlen.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 3)
    assert result["value_2"][0] == pytest.approx(3.0)
    assert result["value_2"][1] is None


def test_read_tsf_comment_lines_skipped(tmp_path: Path) -> None:
    """Lines starting with # are silently skipped."""
    from sorethumb.config import SourceConfig

    content = "# this is a comment\n@data\n# another comment\n1.0,2.0,3.0\n"
    p = tmp_path / "comments.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_unknown_at_directives_skipped(tmp_path: Path) -> None:
    """Unknown @ directives (not @attribute / @data) are skipped."""
    from sorethumb.config import SourceConfig

    content = "@frequency yearly\n@horizon 10\n@data\n1.0,2.0,3.0\n"
    p = tmp_path / "directives.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_lines_before_data_skipped(tmp_path: Path) -> None:
    """Non-@ lines before @data are ignored (data hasn't started yet)."""
    from sorethumb.config import SourceConfig

    content = "stray line\n@data\n1.0,2.0,3.0\n"
    p = tmp_path / "stray.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 1


def test_read_tsf_malformed_row_too_few_fields_skipped(tmp_path: Path) -> None:
    """Malformed rows (too few colon-separated fields) are silently skipped."""
    from sorethumb.config import SourceConfig

    content = (
        "@attribute id numeric\n"
        "@data\n"
        "1:10.0,20.0\n"
        "BADROW\n"  # no colon → fewer fields than attributes + 1
        "2:30.0,40.0\n"
    )
    p = tmp_path / "malformed.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape[0] == 2  # only 2 valid rows


def test_read_tsf_numeric_attribute_bad_value_gives_none(tmp_path: Path) -> None:
    """A non-numeric value in a numeric @attribute field becomes None."""
    from sorethumb.config import SourceConfig

    content = "@attribute id numeric\n@data\nnotanumber:1.0,2.0,3.0\n"
    p = tmp_path / "badnum.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result["id"][0] is None


def test_read_tsf_empty_no_data_rows(tmp_path: Path) -> None:
    """A file with @attribute but no data rows returns an empty LazyFrame
    with the declared schema."""
    from sorethumb.io.readers import _read_tsf

    content = "@attribute id numeric\n@attribute label string\n@data\n"
    p = tmp_path / "empty.tsf"
    p.write_text(content)

    result = _read_tsf(p).collect()
    assert result.shape[0] == 0
    assert set(result.columns) == {"id", "label"}


def test_read_tsf_multiple_rows_multiple_attrs(tmp_path: Path) -> None:
    """Multi-row, multi-attribute TSF file integrates end-to-end correctly."""
    from sorethumb.config import SourceConfig

    content = (
        "@attribute series_id numeric\n"
        "@attribute split string\n"
        "@data\n"
        "1:train:10.0,20.0,30.0\n"
        "2:test:40.0,50.0\n"
    )
    p = tmp_path / "multi.tsf"
    p.write_text(content)

    cfg = SourceConfig(uri=str(p))
    result = read_frame(p, cfg).collect()
    assert result.shape == (2, 5)  # series_id, split, value_0, value_1, value_2
    assert result["split"].to_list() == ["train", "test"]
    assert result["value_2"][1] is None  # second row is shorter


# ---------------------------------------------------------------------------
# Local source resolution
# ---------------------------------------------------------------------------


def test_resolve_source_local(tmp_path: Path) -> None:
    p = tmp_path / "src.csv"
    p.write_text("a,b\n1,2\n")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(p))
    resolved = resolve_source(cfg, tmp_path / "cache")
    assert resolved == p.resolve()


def test_resolve_source_missing_file_raises(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=str(tmp_path / "no_such_file.csv"))
    with pytest.raises(SourceError, match="not found"):
        resolve_source(cfg, tmp_path / "cache")


def test_resolve_source_unsupported_scheme_raises(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri="s3://bucket/key.parquet")
    with pytest.raises(SourceError, match="Unsupported URI scheme"):
        resolve_source(cfg, tmp_path / "cache")


def test_resolve_source_windows_drive_path_is_not_an_unsupported_scheme(tmp_path: Path) -> None:
    """A Windows absolute path ('C:\\...') must be treated as local, not
    misread by urlparse as scheme='c' and rejected. This environment isn't
    Windows, so the path genuinely doesn't exist -- the point is that it
    fails with "not found", never "Unsupported URI scheme 'c'"."""
    from sorethumb.config import SourceConfig

    cfg = SourceConfig(uri=r"C:\Users\someone\data.csv")
    with pytest.raises(SourceError, match="not found") as exc_info:
        resolve_source(cfg, tmp_path / "cache")
    assert "Unsupported URI scheme" not in str(exc_info.value)


def test_resolve_source_file_uri_resolves_to_the_real_path(tmp_path: Path) -> None:
    from sorethumb.config import SourceConfig

    p = tmp_path / "src.csv"
    p.write_text("a,b\n1,2\n")
    cfg = SourceConfig(uri=p.as_uri())  # e.g. "file:///tmp/.../src.csv"
    resolved = resolve_source(cfg, tmp_path / "cache")
    assert resolved == p.resolve()


def test_resolve_source_file_uri_decodes_percent_escapes(tmp_path: Path) -> None:
    """A file:// URI percent-encodes reserved characters (a space becomes
    %20); resolving it must decode them back, not treat "%20" literally."""
    from sorethumb.config import SourceConfig

    p = tmp_path / "has space.csv"
    p.write_text("a,b\n1,2\n")
    uri = "file://" + str(p.resolve()).replace(" ", "%20")
    cfg = SourceConfig(uri=uri)
    resolved = resolve_source(cfg, tmp_path / "cache")
    assert resolved == p.resolve()


# ---------------------------------------------------------------------------
# Extension detection: gzip-compound extensions (P2-5)
# ---------------------------------------------------------------------------


def test_extension_from_url_matches_compound_gzip_extension() -> None:
    """data.csv.gz must resolve to .csv.gz, not the unrelated, shorter .gz --
    readers.py's auto-format detection only recognises the compound form, so
    losing the .csv part here broke every gzipped auto-format download."""
    from sorethumb.config import SourceConfig
    from sorethumb.io.source import _extension_from_url

    cfg = SourceConfig(uri="https://example.com/data.csv.gz")
    assert _extension_from_url("https://example.com/data.csv.gz", cfg) == ".csv.gz"


def test_extension_from_url_plain_gzip_variants() -> None:
    from sorethumb.config import SourceConfig
    from sorethumb.io.source import _extension_from_url

    cfg = SourceConfig(uri="https://example.com/x")
    assert _extension_from_url("https://example.com/data.jsonl.gz", cfg) == ".jsonl.gz"
    assert _extension_from_url("https://example.com/data.parquet", cfg) == ".parquet"


def test_extension_from_url_explicit_format_overrides_url() -> None:
    from sorethumb.config import SourceConfig
    from sorethumb.io.source import _extension_from_url

    cfg = SourceConfig(uri="https://example.com/x", format="csv")
    assert _extension_from_url("https://example.com/data.whatever", cfg) == ".csv"


# ---------------------------------------------------------------------------
# Basic auth header construction (P2-5)
# ---------------------------------------------------------------------------


def test_build_auth_headers_basic_base64_encodes_user_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """auth_env_var holds 'user:password' in plain text; basic auth must
    base64-encode it to build the header, not require the caller to
    pre-encode it."""
    import base64

    from sorethumb.config import SourceConfig
    from sorethumb.io.source import _build_auth_headers

    monkeypatch.setenv("BASIC_CREDS", "alice:s3cret")
    cfg = SourceConfig(uri="https://x/data.csv", auth="basic", auth_env_var="BASIC_CREDS")
    headers = _build_auth_headers(cfg)
    expected = "Basic " + base64.b64encode(b"alice:s3cret").decode("ascii")
    assert headers["Authorization"] == expected


def test_build_auth_headers_bearer_passes_token_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb.config import SourceConfig
    from sorethumb.io.source import _build_auth_headers

    monkeypatch.setenv("BEARER_TOKEN", "abc123")
    cfg = SourceConfig(uri="https://x/data.csv", auth="bearer", auth_env_var="BEARER_TOKEN")
    assert _build_auth_headers(cfg) == {"Authorization": "Bearer abc123"}


# ---------------------------------------------------------------------------
# Redaction (P2-5)
# ---------------------------------------------------------------------------


def test_redact_source_uri_strips_userinfo() -> None:
    from sorethumb.io.source import redact_source_uri

    redacted = redact_source_uri("https://user:pass@example.com/data.csv")
    assert "user" not in redacted
    assert "pass" not in redacted
    assert redacted == "https://***@example.com/data.csv"


def test_redact_source_uri_strips_sensitive_query_params_keeps_others() -> None:
    from sorethumb.io.source import redact_source_uri

    redacted = redact_source_uri("https://example.com/data.csv?sig=abc123&format=csv")
    assert "abc123" not in redacted
    assert "sig=REDACTED" in redacted
    assert "format=csv" in redacted


def test_redact_source_uri_local_path_passthrough() -> None:
    from sorethumb.io.source import redact_source_uri

    assert redact_source_uri("/data/local/file.csv") == "/data/local/file.csv"


def test_redact_source_uri_case_insensitive_query_key() -> None:
    from sorethumb.io.source import redact_source_uri

    redacted = redact_source_uri("https://example.com/data.csv?X-Amz-Signature=deadbeef")
    assert "deadbeef" not in redacted


# ---------------------------------------------------------------------------
# Download hardening: redirects, size limits, concurrency (P2-5)
# ---------------------------------------------------------------------------


def test_download_follows_a_redirect_to_a_safe_host(tmp_path: Path) -> None:
    import httpx

    from sorethumb.io.source import _download_to

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "198.51.100.1":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/final.csv"})
        return httpx.Response(200, content=b"a,b\n1,2\n")

    dest = tmp_path / "out.csv"
    _download_to(
        "http://198.51.100.1/data.csv",
        {},
        dest,
        max_bytes=10_000,
        transport=httpx.MockTransport(handler),
    )
    assert dest.read_bytes() == b"a,b\n1,2\n"


def test_download_refuses_redirect_to_cloud_metadata_host(tmp_path: Path) -> None:
    """The classic SSRF-via-redirect pattern: an external host redirects the
    fetcher to the cloud metadata address. Must be refused, not followed."""
    import httpx

    from sorethumb.errors import SourceError
    from sorethumb.io.source import _download_to

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "198.51.100.1":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, content=b"should not be reached")

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match=r"169\.254\.169\.254|link-local|internal"):
        _download_to(
            "http://198.51.100.1/data.csv",
            {},
            dest,
            max_bytes=10_000,
            transport=httpx.MockTransport(handler),
        )


def test_download_refuses_redirect_to_loopback_host(tmp_path: Path) -> None:
    import httpx

    from sorethumb.errors import SourceError
    from sorethumb.io.source import _download_to

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "198.51.100.1":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8080/internal"})
        return httpx.Response(200, content=b"should not be reached")

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match=r"loopback|internal"):
        _download_to(
            "http://198.51.100.1/data.csv",
            {},
            dest,
            max_bytes=10_000,
            transport=httpx.MockTransport(handler),
        )


def test_download_refuses_too_many_redirects(tmp_path: Path) -> None:
    import httpx

    from sorethumb.errors import SourceError
    from sorethumb.io.source import _download_to

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://8.8.8.8/loop"})

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match=r"[Tt]oo many redirects"):
        _download_to(
            "http://8.8.8.8/data.csv",
            {},
            dest,
            max_bytes=10_000,
            transport=httpx.MockTransport(handler),
        )


# ---------------------------------------------------------------------------
# Authorization is origin-scoped across redirects (P0-3)
# ---------------------------------------------------------------------------

_SECRET_TOKEN = "super-secret-do-not-leak-xyz"


def test_download_preserves_authorization_on_same_origin_redirect(tmp_path: Path) -> None:
    """A same-origin redirect (identical scheme/host/port, path change only)
    must still carry the Authorization header -- this is exactly the
    ordinary case (e.g. a CDN redirecting one path to another on itself)
    the P0-3 origin check must not break."""
    import httpx

    from sorethumb.io.source import _download_to

    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        if request.url.path == "/data.csv":
            return httpx.Response(302, headers={"location": "http://198.51.100.1/final.csv"})
        return httpx.Response(200, content=b"a,b\n1,2\n")

    dest = tmp_path / "out.csv"
    _download_to(
        "http://198.51.100.1/data.csv",
        {"Authorization": f"Bearer {_SECRET_TOKEN}"},
        dest,
        max_bytes=10_000,
        transport=httpx.MockTransport(handler),
    )
    assert seen_auth == [f"Bearer {_SECRET_TOKEN}", f"Bearer {_SECRET_TOKEN}"]


def test_download_strips_authorization_on_cross_host_redirect(tmp_path: Path) -> None:
    """A redirect to a *different* host, even a public/safe one, must never
    carry the original Authorization header -- the credential is scoped to
    the host the caller explicitly configured, not to "wherever that host's
    server later points us"."""
    import httpx

    from sorethumb.io.source import _download_to

    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        if request.url.host == "198.51.100.1":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/final.csv"})
        return httpx.Response(200, content=b"a,b\n1,2\n")

    dest = tmp_path / "out.csv"
    _download_to(
        "http://198.51.100.1/data.csv",
        {"Authorization": f"Bearer {_SECRET_TOKEN}"},
        dest,
        max_bytes=10_000,
        transport=httpx.MockTransport(handler),
    )
    assert seen_auth == [f"Bearer {_SECRET_TOKEN}", None]


def test_download_strips_authorization_on_port_change_redirect(tmp_path: Path) -> None:
    """Same host, different explicit port: a different origin, so the
    credential must still be stripped even though the hostname string is
    identical."""
    import httpx

    from sorethumb.io.source import _download_to

    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        if request.url.port is None:
            return httpx.Response(302, headers={"location": "http://198.51.100.1:8080/final.csv"})
        return httpx.Response(200, content=b"a,b\n1,2\n")

    dest = tmp_path / "out.csv"
    _download_to(
        "http://198.51.100.1/data.csv",
        {"Authorization": f"Bearer {_SECRET_TOKEN}"},
        dest,
        max_bytes=10_000,
        transport=httpx.MockTransport(handler),
    )
    assert seen_auth == [f"Bearer {_SECRET_TOKEN}", None]


def test_download_refuses_https_to_http_downgrade_redirect(tmp_path: Path) -> None:
    """An HTTPS -> HTTP downgrade must be refused outright -- not merely
    stripped of credentials -- since it also drops transport security for
    the response body. The secret must not leak into the raised error."""
    import httpx

    from sorethumb.io.source import _download_to

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "https":
            return httpx.Response(302, headers={"location": "http://198.51.100.1/final.csv"})
        return httpx.Response(200, content=b"should not be reached")

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match=r"[Dd]owngrade") as exc_info:
        _download_to(
            "https://198.51.100.1/data.csv",
            {"Authorization": f"Bearer {_SECRET_TOKEN}"},
            dest,
            max_bytes=10_000,
            transport=httpx.MockTransport(handler),
        )
    assert _SECRET_TOKEN not in str(exc_info.value)


def test_download_refuses_https_to_http_downgrade_even_without_auth_configured(
    tmp_path: Path,
) -> None:
    """The downgrade refusal is unconditional -- it protects the response
    body's transport security too, not just a credential that may not even
    be configured."""
    import httpx

    from sorethumb.io.source import _download_to

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "https":
            return httpx.Response(302, headers={"location": "http://198.51.100.1/final.csv"})
        return httpx.Response(200, content=b"should not be reached")

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match=r"[Dd]owngrade"):
        _download_to(
            "https://198.51.100.1/data.csv",
            {},
            dest,
            max_bytes=10_000,
            transport=httpx.MockTransport(handler),
        )


def test_download_cross_origin_redirect_does_not_leak_secret_via_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole point of stripping Authorization cross-origin is that the
    credential must never reach anywhere it could be recorded -- confirm it
    never shows up in the log output this download produces, or in the
    final downloaded file (the mock final host echoes nothing back, but
    this guards against a future change accidentally logging headers)."""
    import logging

    import httpx

    from sorethumb.io.source import _download_to

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "198.51.100.1":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/final.csv"})
        return httpx.Response(200, content=b"a,b\n1,2\n")

    dest = tmp_path / "out.csv"
    with caplog.at_level(logging.DEBUG):
        _download_to(
            "http://198.51.100.1/data.csv",
            {"Authorization": f"Bearer {_SECRET_TOKEN}"},
            dest,
            max_bytes=10_000,
            transport=httpx.MockTransport(handler),
        )
    assert _SECRET_TOKEN not in caplog.text


def test_download_rejects_oversized_declared_content_length(tmp_path: Path) -> None:
    """A Content-Length above the limit must be rejected before any body
    is read."""
    import httpx

    from sorethumb.errors import SourceError
    from sorethumb.io.source import _download_to

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "1000000"}, content=b"x" * 10)

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match="max_download_bytes"):
        _download_to(
            "http://8.8.8.8/data.csv",
            {},
            dest,
            max_bytes=100,
            transport=httpx.MockTransport(handler),
        )


def test_download_rejects_oversized_streamed_body_without_content_length(tmp_path: Path) -> None:
    """No Content-Length declared (e.g. chunked transfer) -- the streamed
    byte count itself must still be capped."""
    import httpx

    from sorethumb.errors import SourceError
    from sorethumb.io.source import _download_to

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1000)  # no content-length header set

    dest = tmp_path / "out.csv"
    with pytest.raises(SourceError, match="max_download_bytes"):
        _download_to(
            "http://8.8.8.8/data.csv",
            {},
            dest,
            max_bytes=100,
            transport=httpx.MockTransport(handler),
        )


def test_concurrent_downloads_do_not_corrupt_or_cross_contaminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different URLs downloaded concurrently into the same cache_dir
    must each end up cached under their own content fingerprint, with
    correct content -- never sharing or clobbering the other's temp file
    (the old fixed "_download_tmp" name let exactly that happen)."""
    import threading

    from sorethumb.config import SourceConfig
    from sorethumb.io import source as src

    barrier = threading.Barrier(2)

    def _slow_download(url: str, _headers: dict, dest: Path, **_kwargs: object) -> None:
        content = b"content-A\n" if "a.csv" in url else b"content-B\n"
        barrier.wait()  # maximise the window where both are mid-download at once
        dest.write_bytes(content)

    monkeypatch.setattr(src, "_download_to", _slow_download)

    results: dict[str, Path] = {}
    errors: list[Exception] = []

    def _run(name: str, url: str) -> None:
        try:
            results[name] = src.resolve_source(SourceConfig(uri=url), tmp_path / "cache")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_run, args=("a", "https://x/a.csv")),
        threading.Thread(target=_run, args=("b", "https://x/b.csv")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent downloads raised: {errors}"
    assert results["a"].read_bytes() == b"content-A\n"
    assert results["b"].read_bytes() == b"content-B\n"
    # No stray per-attempt temp files left behind.
    assert list((tmp_path / "cache").glob(".download-*")) == []


# ---------------------------------------------------------------------------
# Auth token not persisted in config JSON
# ---------------------------------------------------------------------------


def test_auth_token_not_in_config_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET_TOKEN", "super-secret-value-xyz")

    from sorethumb.config import SourceConfig

    cfg = SourceConfig(
        uri="https://example.com/data.csv",
        auth="bearer",
        auth_env_var="MY_SECRET_TOKEN",
    )
    serialised = cfg.model_dump_json()
    assert "super-secret-value-xyz" not in serialised
    assert "MY_SECRET_TOKEN" in serialised  # only the variable NAME is stored


# ---------------------------------------------------------------------------
# Struct unnesting
# ---------------------------------------------------------------------------


def test_unnest_simple_struct() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "point": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}, {"x": 5.0, "y": 6.0}],
        }
    )
    result = unnest_all(df, max_depth=5)
    assert "point" not in result.columns
    assert "point_x" in result.columns
    assert "point_y" in result.columns
    assert result.shape == (3, 3)  # id, point_x, point_y


def test_unnest_preserves_values() -> None:
    df = pl.DataFrame(
        {
            "nested": [{"a": 10, "b": 20}, {"a": 30, "b": 40}],
        }
    )
    result = unnest_all(df, max_depth=5)
    assert result["nested_a"].to_list() == [10, 30]
    assert result["nested_b"].to_list() == [20, 40]


def test_unnest_two_levels() -> None:
    df = pl.DataFrame(
        {
            "outer": [{"inner": {"z": float(i)}} for i in range(3)],
        }
    )
    result = unnest_all(df, max_depth=5)
    # After depth 0: outer_inner (a Struct with z)
    # After depth 1: outer_inner_z
    assert "outer_inner_z" in result.columns


def test_unnest_name_collision_raises() -> None:
    df = pl.DataFrame(
        {
            "col_x": [1, 2],
            "col": [{"x": 10, "y": 20}, {"x": 30, "y": 40}],
        }
    )
    with pytest.raises(SchemaError, match="duplicate"):
        unnest_all(df, max_depth=5)


def test_unnest_no_structs_noop() -> None:
    df, _ = make_frame(n_rows=5, seed=0)
    result = unnest_all(df, max_depth=5)
    assert result.equals(df)


def test_unnest_depth_zero_leaves_structs() -> None:
    df = pl.DataFrame(
        {
            "s": [{"a": 1}, {"a": 2}],
        }
    )
    result = unnest_all(df, max_depth=0)
    assert "s" in result.columns
    assert "s_a" not in result.columns


# ---------------------------------------------------------------------------
# Array feature derivation
# ---------------------------------------------------------------------------


def test_derive_array_numeric_features() -> None:
    df = pl.DataFrame(
        {
            "nums": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        result = derive_array_features(df)
    assert "nums" not in result.columns
    assert "nums__len" in result.columns
    assert "nums__mean" in result.columns
    assert "nums__min" in result.columns
    assert "nums__max" in result.columns
    assert result["nums__len"].to_list() == [3, 3, 3]


def test_derive_array_string_no_numeric_stats() -> None:
    df = pl.DataFrame(
        {
            "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        result = derive_array_features(df)
    assert "tags__len" in result.columns
    assert "tags__mean" not in result.columns


def test_derive_array_zero_length_warns() -> None:
    df = pl.DataFrame(
        {
            "empty_arr": [[], [], []],
        },
        schema={"empty_arr": pl.List(pl.Float64)},
    )
    with pytest.raises(ColumnDroppedWarning):
        derive_array_features(df)


def test_derive_array_non_array_cols_preserved() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "vals": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        result = derive_array_features(df)
    assert "id" in result.columns
    assert result["id"].to_list() == [1, 2, 3]
