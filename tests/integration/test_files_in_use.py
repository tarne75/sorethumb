"""Download cache and prune keep working when a file is held open (Windows).

- Two processes downloading identical content can both find the cache entry
  missing; the loser's replace onto the winner's (open, on Windows) file used
  to fail. Content-addressed, so the winner's copy is used after a check.
- source.cache = false replaced one shared "uncached_data" file every call;
  each call now gets its own file, and stale ones are cleaned up later.
- A prune that hit one undeletable file stopped there with the index and disk
  half-pruned; it now prunes everything else and reports what it couldn't.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml.config import SourceConfig
from sorethumb_ml.errors import FileInUseError, StoreError
from sorethumb_ml.io import source as src
from sorethumb_ml.store import workspace as ws_mod
from sorethumb_ml.store.workspace import Workspace

pytestmark = pytest.mark.integration

_BODY = b"a,b\n1,2\n"


def _fake_download(body: bytes = _BODY):
    def _download(_url: str, _headers: dict, dest: Path, **_kwargs: object) -> None:
        dest.write_bytes(body)

    return _download


def _lose_the_race(monkeypatch: pytest.MonkeyPatch, winner_bytes: bytes) -> None:
    """Make promotion behave as if another process cached the file first and holds it open."""
    real_promote = src._promote

    def _promote(tmp: Path, dest: Path) -> None:
        if dest.name.startswith("data"):
            dest.write_bytes(winner_bytes)
            raise FileInUseError(f"Could not replace {dest}: the file is open in another program (simulated)")
        real_promote(tmp, dest)

    monkeypatch.setattr(src, "_promote", _promote)


def test_a_lost_cache_race_with_identical_bytes_uses_the_winners_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(src, "_download_to", _fake_download())
    _lose_the_race(monkeypatch, _BODY)
    path = src.resolve_source(SourceConfig(uri="https://example.com/data.csv"), tmp_path)
    assert path.read_bytes() == _BODY
    assert not list(tmp_path.glob(".download-*")), "our temp download was left behind"


def test_a_lost_cache_race_with_different_bytes_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(src, "_download_to", _fake_download())
    _lose_the_race(monkeypatch, b"x,y\n9,9\n")
    with pytest.raises(FileInUseError):
        src.resolve_source(SourceConfig(uri="https://example.com/data.csv"), tmp_path)
    assert not list(tmp_path.glob(".download-*"))


def test_uncached_downloads_get_their_own_file_each_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(src, "_download_to", _fake_download())
    cfg = SourceConfig(uri="https://example.com/data.csv", cache=False)
    first = src.resolve_source(cfg, tmp_path)
    second = src.resolve_source(cfg, tmp_path)
    assert first != second
    assert first.read_bytes() == second.read_bytes() == _BODY


def test_stale_uncached_downloads_are_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(src, "_download_to", _fake_download())
    stale = tmp_path / "uncached_data-old.csv"
    recent = tmp_path / "uncached_data-recent.csv"
    for f in (stale, recent):
        f.write_bytes(_BODY)
    two_hours_ago = time.time() - 7200
    os.utime(stale, (two_hours_ago, two_hours_ago))
    src.resolve_source(SourceConfig(uri="https://example.com/data.csv", cache=False), tmp_path)
    assert not stale.exists()
    assert recent.exists()


def _old_artifact(ws: Workspace, artifact_id: str, path: Path) -> None:
    path.write_bytes(b"x")
    ws.store._conn.execute(
        "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
        "VALUES (?, ?, 'cache', 1, 1, datetime('now', '-400 days'))",
        (artifact_id, str(path)),
    )
    ws.store._conn.commit()


def test_prune_continues_past_a_locked_file_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Workspace.init(tmp_path / "ws") as ws:
        free, locked = ws.root / "cache" / "free.parquet", ws.root / "cache" / "locked.parquet"
        _old_artifact(ws, "free", free)
        _old_artifact(ws, "locked", locked)
        real_unlink = ws_mod.unlink_with_retry

        def _unlink(path: Path, **kwargs: bool) -> None:
            if path.name == "locked.parquet":
                raise FileInUseError(
                    f"Could not delete {path}: the file is open in another program (simulated)"
                )
            real_unlink(path, **kwargs)

        monkeypatch.setattr(ws_mod, "unlink_with_retry", _unlink)
        with pytest.raises(StoreError, match=r"could not delete 1.*locked\.parquet"):
            ws.prune(retention_days=1)
        assert not free.exists()
        assert locked.exists()
        ids = {r[0] for r in ws.store._conn.execute("SELECT artifact_id FROM artifact")}
        assert ids == {"locked"}, "the locked file must stay indexed so the next prune retries it"


def test_prune_command_reports_a_locked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb_ml.cli import app

    with Workspace.init(tmp_path / "ws") as ws:
        _old_artifact(ws, "locked", ws.root / "cache" / "locked.parquet")

    def _locked(path: Path, **_kwargs: bool) -> None:
        raise FileInUseError(f"Could not delete {path}: the file is open in another program (simulated)")

    monkeypatch.setattr(ws_mod, "unlink_with_retry", _locked)
    result = CliRunner().invoke(app, ["workspace", "prune", "--workdir", str(tmp_path / "ws"), "--days", "1"])
    assert result.exit_code == 1, result.output
    assert "locked.parquet" in "".join(result.output.split())  # Rich may wrap a long path mid-name


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing semantics")
def test_a_real_open_handle_gives_file_in_use_on_windows(tmp_path: Path) -> None:
    """Python opens files without FILE_SHARE_DELETE, like most Windows programs,
    so while this handle is open the file can be neither replaced nor deleted.
    Checks the error codes the retry recognises against real Windows behaviour."""
    from sorethumb_ml._atomic import atomic_write_text, unlink_with_retry

    target = tmp_path / "report.csv"
    target.write_text("old", encoding="utf-8")
    with target.open(encoding="utf-8"):
        with pytest.raises(FileInUseError, match=r"report\.csv"):
            atomic_write_text(target, "new")
        with pytest.raises(FileInUseError, match=r"report\.csv"):
            unlink_with_retry(target)
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["report.csv"]  # temp file cleaned up
    atomic_write_text(target, "new")  # works once the handle is closed
    assert target.read_text(encoding="utf-8") == "new"
