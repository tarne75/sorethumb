"""display_source_uri / source_digest: what may be stored, logged or reported about a source.

The guarantee under test: no query value, userinfo or fragment survives
``display_source_uri`` -- for *any* parameter name, not just a known-sensitive list --
while ``source_digest`` still tells two sources that differ only in a query value apart.
"""

from __future__ import annotations

import pytest

from sorethumb_ml.io.uri import display_source_uri, query_key_names, source_digest
from tests.factories.source_uris import SIGNED_URI_CASES

CASE_IDS = list(SIGNED_URI_CASES)


@pytest.mark.parametrize("name", CASE_IDS)
def test_no_secret_value_survives_display(name: str) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    shown = display_source_uri(uri)
    for secret in secrets:
        assert secret not in shown, f"{secret!r} leaked into {shown!r}"
    assert "#" not in shown


@pytest.mark.parametrize("name", CASE_IDS)
def test_display_is_idempotent(name: str) -> None:
    shown = display_source_uri(SIGNED_URI_CASES[name][0])
    assert display_source_uri(shown) == shown


def test_keeps_scheme_host_path_and_key_names_replacing_every_value() -> None:
    shown = display_source_uri("HTTPS://Data.Example.COM/a/b/t.csv?client_secret=x&Credential=y&jwt=z")
    assert (
        shown == "https://data.example.com/a/b/t.csv?client_secret=REDACTED&Credential=REDACTED&jwt=REDACTED"
    )


def test_arbitrary_mixed_case_names_are_redacted_like_known_ones() -> None:
    shown = display_source_uri("https://h/p?AcCeSsKeY=v1&SoMeThInG_Else=v2&sig=v3")
    assert shown == "https://h/p?AcCeSsKeY=REDACTED&SoMeThInG_Else=REDACTED&sig=REDACTED"


def test_duplicate_keys_keep_order_and_count() -> None:
    shown = display_source_uri("https://h/p?k=1&k=2&K=3&k=4")
    assert shown == "https://h/p?k=REDACTED&k=REDACTED&K=REDACTED&k=REDACTED"


def test_blank_values_are_redacted_so_blankness_is_not_revealed() -> None:
    assert (
        display_source_uri("https://h/p?token=&flag=&x=1")
        == "https://h/p?token=REDACTED&flag=REDACTED&x=REDACTED"
    )


def test_bare_segment_is_redacted_because_it_could_be_the_secret() -> None:
    assert display_source_uri("https://h/p?abc123secret&a=b") == "https://h/p?REDACTED&a=REDACTED"


def test_separator_inside_a_value_cannot_smuggle_text_through() -> None:
    shown = display_source_uri("https://h/p?a=1;b=leaky&c=2")
    assert "leaky" not in shown


def test_userinfo_and_fragment_are_removed() -> None:
    assert display_source_uri("https://user:pw@h/p?x=1#sec") == "https://***@h/p?x=REDACTED"
    assert display_source_uri("https://@h/p") == "https://***@h/p"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://h:443/p?x=1", "https://h/p?x=REDACTED"),
        ("http://h:80/p", "http://h/p"),
        ("https://h:8443/p", "https://h:8443/p"),
        ("http://h:443/p", "http://h:443/p"),
        ("https://[::1]:8443/p", "https://[::1]:8443/p"),
        ("https://[2001:db8::1]/p", "https://[2001:db8::1]/p"),
    ],
)
def test_port_shown_only_when_it_is_not_the_scheme_default(uri: str, expected: str) -> None:
    assert display_source_uri(uri) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/data/local/file.csv",
        "relative/file.csv",
        "C:\\data\\file.csv",
        "c:/data/file.csv",
        "\\\\srv\\share\\f.csv",
    ],
)
def test_local_paths_pass_through_unchanged(path: str) -> None:
    assert display_source_uri(path) == path


def test_file_uri_is_unchanged() -> None:
    assert display_source_uri("file:///C:/data/file.csv") == "file:///C:/data/file.csv"


def test_unparseable_uri_does_not_echo_the_input() -> None:
    shown = display_source_uri("https://[::1/p?secret=leak")
    assert "leak" not in shown


def test_query_without_values_is_dropped_cleanly() -> None:
    assert display_source_uri("https://h/p?") == "https://h/p"


@pytest.mark.parametrize("name", CASE_IDS)
def test_query_key_names_never_contain_a_value(name: str) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    for key in query_key_names(uri):
        assert not any(secret in key for secret in secrets)


def test_query_key_names_are_stable_across_a_signature_refresh() -> None:
    old = "https://h/p.csv?X-Amz-Date=20260101&X-Amz-Signature=aaaa"
    new = "https://h/p.csv?X-Amz-Signature=bbbb&X-Amz-Date=20260202"
    assert query_key_names(old) == query_key_names(new) == {"X-Amz-Date", "X-Amz-Signature"}
    assert query_key_names("https://h/p.csv") == frozenset()


# ── source_digest ─────────────────────────────────────────────────────────────


def test_digest_is_deterministic_and_hex_sha256() -> None:
    uri = "https://h/p.csv?sig=abc"
    assert source_digest(uri) == source_digest(uri)
    assert len(source_digest(uri)) == 64
    int(source_digest(uri), 16)


def test_digest_separates_sources_that_differ_only_in_a_query_value() -> None:
    assert source_digest("https://h/p.csv?sig=abc") != source_digest("https://h/p.csv?sig=abd")
    assert source_digest("https://h/p.csv?t=sales") != source_digest("https://h/p.csv?t=costs")
    assert source_digest("https://h/p.csv?a=1&b=2") != source_digest("https://h/p.csv?b=2&a=1")


def test_digest_separates_userinfo() -> None:
    assert source_digest("https://a:1@h/p.csv") != source_digest("https://a:2@h/p.csv")


def test_digest_ignores_spelling_that_does_not_change_the_request() -> None:
    base = source_digest("https://h.example.com/p.csv?sig=abc")
    assert source_digest("HTTPS://H.Example.COM:443/p.csv?sig=abc") == base
    assert source_digest("https://h.example.com/p.csv?sig=abc#fragment") == base
    assert source_digest("  https://h.example.com/p.csv?sig=abc  ") == base


def test_digest_treats_windows_path_spellings_as_one_source() -> None:
    assert source_digest("C:\\Data\\Sales.csv") == source_digest("c:/data/sales.csv")


@pytest.mark.parametrize("name", CASE_IDS)
def test_digest_contains_no_secret_text(name: str) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    digest = source_digest(uri)
    assert not any(secret in digest for secret in secrets)
