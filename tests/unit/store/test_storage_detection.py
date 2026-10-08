"""Workspaces on network or cloud-synced storage are flagged, never fatally.

SQLite in WAL mode and atomic renames need a local filesystem. On Windows the
Documents folder is often OneDrive-backed, and mapped network drives are
common, so a workspace there gets one clear warning. Detection is advisory:
every probe is faked here, and any failure inside it must return None.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sorethumb_ml.store import storage
from sorethumb_ml.store.storage import detect_risky_storage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_real_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(storage, "_mount_fs_type", lambda _path: "ext4")


def test_a_local_folder_is_not_flagged(tmp_path: Path) -> None:
    assert detect_risky_storage(tmp_path) is None


def test_a_folder_under_the_onedrive_root_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "Business Docs"
    (root / "proj").mkdir(parents=True)
    monkeypatch.setenv("OneDriveCommercial", str(root))
    assert detect_risky_storage(root / "proj") == "a OneDrive-synced folder"


@pytest.mark.parametrize(
    ("folder", "reason"),
    [
        ("OneDrive - Contoso", "a OneDrive-synced folder"),
        ("Dropbox", "a Dropbox-synced folder"),
        ("Google Drive", "a Google Drive-synced folder"),
        ("iCloudDrive", "an iCloud Drive-synced folder"),
        ("CloudStorage", "a cloud-synced folder"),
    ],
)
def test_sync_client_folders_are_flagged(tmp_path: Path, folder: str, reason: str) -> None:
    target = tmp_path / folder / "analysis"
    target.mkdir(parents=True)
    assert detect_risky_storage(target) == reason


@pytest.mark.parametrize("fs_type", ["nfs4", "cifs", "smbfs", "fuse.sshfs"])
def test_network_filesystems_are_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fs_type: str
) -> None:
    monkeypatch.setattr(storage.sys, "platform", "linux")
    monkeypatch.setattr(storage, "_mount_fs_type", lambda _path: fs_type)
    assert detect_risky_storage(tmp_path) == f"a network filesystem ({fs_type})"


def test_detection_errors_are_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_path: Path) -> str:
        raise OSError("mount table unreadable")

    monkeypatch.setattr(storage, "_mount_fs_type", _boom)
    assert detect_risky_storage(tmp_path) is None


@pytest.mark.skipif(storage.sys.platform == "win32", reason="POSIX mount table")
def test_the_longest_matching_mount_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # use the real _mount_fs_type
    table = tmp_path / "mounts"
    table.write_text("/dev/sda1 / ext4 rw 0 0\n//nas/share /mnt/nas cifs rw 0 0\n", encoding="utf-8")
    real_path = Path
    monkeypatch.setattr(storage, "Path", lambda p: table if p == "/proc/mounts" else real_path(p))
    assert storage._mount_fs_type(real_path("/mnt/nas/projects/ws")) == "cifs"
    assert storage._mount_fs_type(real_path("/mnt/nasty")) == "ext4"
    assert storage._mount_fs_type(real_path("/home/me")) == "ext4"
