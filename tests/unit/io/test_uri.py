"""Every spelling of one Windows path is one identity; nothing else changes.

``source.uri`` feeds the logical dataset id (history is filed under it) and
``Config.config_hash()`` (run ids, model reuse). On Windows ``C:\\Data\\x.csv``,
``c:/data/x.csv`` and ``file:///C:/Data/x.csv`` are one file, so they must be
one identity. POSIX paths, relative paths and URLs are hashed exactly as given,
so no identity created on Linux or macOS moves.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sorethumb_ml.config import Config
from sorethumb_ml.errors import SourceError
from sorethumb_ml.io.fingerprint import logical_dataset_id
from sorethumb_ml.io.uri import canonical_source_key

pytestmark = pytest.mark.unit


def _hash(uri: str) -> str:
    return Config.model_validate({"source": {"uri": uri}, "run": {"workdir": "ws"}}).config_hash()


# Taken from the code before canonicalisation existed: these must never move.
_PINNED = [
    ("/data/sales/2026 Q3.csv", "2026-Q3-f8f9aa16390c", "b7d633fcc63f2a5163b85014361c3894"),
    ("data/relative.parquet", "relative-fdeb11238495", "1b3550cf47d9dc164a6ab02d32ba4fb8"),
    (
        "https://example.com/exports/daily.csv?sig=abc",
        "daily-e81dab578786",
        "50ddc50d817b7fa0ac847496a399fe4f",
    ),
    ("file:///tmp/x%20y.csv", "x-20y-3ce4812025b2", "9d123ab29469fce0b68efc20d09604cd"),
]


@pytest.mark.parametrize(("uri", "dataset_id", "config_hash"), _PINNED)
def test_posix_and_url_identities_are_unchanged(uri: str, dataset_id: str, config_hash: str) -> None:
    assert logical_dataset_id(None, uri) == dataset_id
    assert _hash(uri) == config_hash


_POSIX_SEGMENT = st.text(
    st.characters(blacklist_characters="/\\\x00", blacklist_categories=("Cs",)), min_size=1
)


@given(st.lists(_POSIX_SEGMENT, min_size=1, max_size=5), st.booleans())
def test_posix_paths_are_their_own_key(segments: list[str], absolute: bool) -> None:
    path = ("/" if absolute else "") + "/".join(segments)
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return  # "C:..." as a first segment is a Windows drive spelling, covered below
    assert canonical_source_key(path) == path


@given(st.sampled_from(["http", "https", "s3"]), st.text(min_size=0, max_size=30))
def test_remote_urls_are_their_own_key(scheme: str, rest: str) -> None:
    uri = f"{scheme}://example.com/{rest}"
    assert canonical_source_key(uri) == uri


@given(st.lists(_POSIX_SEGMENT, min_size=1, max_size=4))
def test_posix_file_uris_are_their_own_key(segments: list[str]) -> None:
    # file:///... and file://localhost/... naming a POSIX path are left alone; a
    # file URI with any other host is a UNC path by definition (canonicalised).
    for prefix in ("file:///", "file://localhost/"):
        uri = prefix + "/".join(segments)
        if len(segments[0]) >= 2 and segments[0][1] == ":" and segments[0][0].isalpha():
            continue
        assert canonical_source_key(uri) == uri


_SAME_FILE = [
    "C:\\Data\\Sales.csv",
    "c:\\data\\sales.csv",
    "C:/Data/Sales.csv",
    "c:/DATA/SALES.CSV",
    "file:///C:/Data/Sales.csv",
    "file:///c:/data/sales.csv",
    "file://localhost/C:/Data/Sales.csv",
    "file://C:/Data/Sales.csv",
    "  C:\\Data\\Sales.csv  ",
]


def test_every_spelling_of_one_windows_file_is_one_identity() -> None:
    keys = {canonical_source_key(u) for u in _SAME_FILE}
    assert keys == {"c:/data/sales.csv"}
    assert len({logical_dataset_id(None, u) for u in _SAME_FILE}) == 1
    assert len({_hash(u) for u in _SAME_FILE}) == 1
    assert logical_dataset_id(None, _SAME_FILE[0]).startswith("sales-")  # same stem on every OS


def test_unc_paths_and_their_file_uris_agree() -> None:
    assert canonical_source_key("\\\\Server\\Share\\x.csv") == canonical_source_key(
        "file://server/share/x.csv"
    )
    assert canonical_source_key("file://server/share/x%20y.csv") == "//server/share/x y.csv"


def test_different_windows_files_stay_different() -> None:
    assert _hash("C:\\Data\\a.csv") != _hash("C:\\Data\\b.csv")
    assert _hash("C:\\Data\\a.csv") != _hash("D:\\Data\\a.csv")


def test_a_drive_relative_path_is_a_clear_error(tmp_path) -> None:
    from sorethumb_ml.config import SourceConfig
    from sorethumb_ml.io.source import resolve_source

    with pytest.raises(SourceError, match="relative to drive C:"):
        resolve_source(SourceConfig(uri="C:data.csv"), tmp_path)


def test_a_unc_share_root_is_one_identity_with_or_without_a_trailing_separator() -> None:
    keys = {
        canonical_source_key(u) for u in ["\\\\Server\\Share\\", "\\\\server\\share", "file://server/share/"]
    }
    assert keys == {"//server/share"}
    assert canonical_source_key("C:\\") == "c:/"
