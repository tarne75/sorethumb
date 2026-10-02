"""Windows MAX_PATH preflight: fail before fitting, not at the first deep write.

Without long-path support Windows caps a path at 259 characters, and the
deepest file a run creates sits about 112 characters below the workspace root.
These tests simulate Windows (sys.platform and the registry lookup), so they run
on every platform; the real-Windows check is the Windows CI lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml import _atomic
from sorethumb_ml.cli import app
from sorethumb_ml.errors import PathTooLongError
from sorethumb_ml.store import workspace as ws_mod
from sorethumb_ml.store.workspace import WINDOWS_MAX_PATH, check_path_length, deepest_path_suffix_length

pytestmark = pytest.mark.unit

_DETECTORS = ("isolation_forest", "kmeans_distance", "one_class_svm")


@pytest.fixture
def windows_without_long_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_mod.sys, "platform", "win32")
    monkeypatch.setattr(ws_mod, "_long_paths_enabled", lambda: False)


def _root_of_length(tmp_path: Path, length: int) -> Path:
    base = str(tmp_path.resolve())
    assert len(base) + 2 <= length, "tmp_path is already too long for this test"
    return Path(base) / ("x" * (length - len(base) - 1))


def test_the_suffix_covers_the_real_deepest_file() -> None:
    model_file = f"/models/score_{'0' * 32}/{'0' * 32}/isolation_forest.calibrator.json"
    assert deepest_path_suffix_length(_DETECTORS) >= len(model_file)
    assert deepest_path_suffix_length(
        ["a_much_longer_third_party_detector_name"]
    ) > deepest_path_suffix_length(_DETECTORS)


def test_temp_names_are_short_and_counted() -> None:
    assert len(_atomic.TEMP_PREFIX) + 8 + len(".tmp") == _atomic.TEMP_NAME_MAX_LEN
    assert _atomic.TEMP_NAME_MAX_LEN <= 16


@pytest.mark.usefixtures("windows_without_long_paths")
def test_a_root_at_the_limit_passes_and_one_character_more_fails(tmp_path: Path) -> None:
    headroom = WINDOWS_MAX_PATH - deepest_path_suffix_length(_DETECTORS)
    check_path_length(_root_of_length(tmp_path, headroom), _DETECTORS)
    with pytest.raises(PathTooLongError, match=r"1 over the 259-character limit"):
        check_path_length(_root_of_length(tmp_path, headroom + 1), _DETECTORS)


def test_enabled_long_paths_skip_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_mod.sys, "platform", "win32")
    monkeypatch.setattr(ws_mod, "_long_paths_enabled", lambda: True)
    check_path_length(_root_of_length(tmp_path, 250), _DETECTORS)


def test_other_platforms_never_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_mod.sys, "platform", "linux")
    monkeypatch.setattr(ws_mod, "_long_paths_enabled", lambda: False)
    check_path_length(_root_of_length(tmp_path, 250), _DETECTORS)


@pytest.mark.usefixtures("windows_without_long_paths")
def test_init_fails_as_preflight_and_writes_nothing(tmp_path: Path) -> None:
    target = _root_of_length(tmp_path, 200)
    result = CliRunner().invoke(app, ["init", str(target)])
    assert result.exit_code == 2, result.output
    flat = " ".join(result.output.split())
    assert "too long for Windows" in flat
    assert "Nothing was written" in flat
    assert not target.exists()
