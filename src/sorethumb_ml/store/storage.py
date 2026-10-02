"""Detect workspace locations that SQLite and atomic renames can't rely on.

A workspace is a SQLite database in WAL mode plus files replaced by atomic
rename. Both need a local filesystem: WAL's shared-memory index is not reliable
over SMB/CIFS or NFS, and file-sync clients (OneDrive, Dropbox, Google Drive,
iCloud) lock, upload and sometimes restore files mid-write. On Windows the
default Documents folder is often OneDrive-backed and mapped network drives are
common, so this is the most likely source of "database is locked" and
"file is open in another program" reports there.

:func:`detect_risky_storage` returns a short reason or None. It never raises:
detection is best-effort, and failing to detect must never stop a run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Filesystem types that are network or FUSE-backed (Linux /proc/mounts, macOS `mount`).
_NETWORK_FS_TYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb",
        "smb3",
        "smbfs",
        "afpfs",
        "webdav",
        "davfs",
        "fuse.sshfs",
        "sshfs",
        "9p",
        "fuse.rclone",
        "fuse.gcsfuse",
        "fuse.s3fs",
    }
)
# Folder names that sync clients create (Windows and macOS).
_SYNCED_FOLDER_MARKERS = {
    "dropbox": "a Dropbox-synced folder",
    "google drive": "a Google Drive-synced folder",
    "googledrive": "a Google Drive-synced folder",
    "icloud drive": "an iCloud Drive-synced folder",
    "iclouddrive": "an iCloud Drive-synced folder",
    "mobile documents": "an iCloud Drive-synced folder",
    "cloudstorage": "a cloud-synced folder",
}
_ONEDRIVE_ENV_VARS = ("OneDrive", "OneDriveCommercial", "OneDriveConsumer")
_DRIVE_REMOTE = 4  # GetDriveTypeW


def detect_risky_storage(path: Path) -> str | None:
    """Return why *path* is a poor place for a workspace, or None if it looks local."""
    try:
        resolved = path.expanduser().resolve()
        return _synced_folder(resolved) or _network_location(resolved)
    except Exception:  # noqa: BLE001 -- detection is advisory; never fail a run over it
        return None


def _synced_folder(path: Path) -> str | None:
    for var in _ONEDRIVE_ENV_VARS:
        root = os.environ.get(var)
        if root and path.is_relative_to(Path(root).expanduser().resolve()):
            return "a OneDrive-synced folder"
    for part in path.parts:
        reason = _SYNCED_FOLDER_MARKERS.get(part.casefold())
        if reason is None and part.casefold().startswith("onedrive"):
            reason = "a OneDrive-synced folder"
        if reason is not None:
            return reason
    return None


def _network_location(path: Path) -> str | None:
    if sys.platform == "win32":
        return _windows_network_location(path)
    fs_type = _mount_fs_type(path)
    if fs_type is not None and fs_type.casefold() in _NETWORK_FS_TYPES:
        return f"a network filesystem ({fs_type})"
    return None


def _windows_network_location(path: Path) -> str | None:
    if path.drive.startswith("\\\\"):
        return "a network share"
    if sys.platform == "win32":  # narrows for mypy; the caller already checked
        import ctypes  # noqa: PLC0415

        drive_type = ctypes.windll.kernel32.GetDriveTypeW(path.anchor)
        if drive_type == _DRIVE_REMOTE:
            return "a network drive"
    return None


def _mount_fs_type(path: Path) -> str | None:
    """Return the filesystem type of the mount holding *path* (Linux and macOS)."""
    mounts: list[tuple[str, str]] = []  # (mount point, fs type)
    proc_mounts = Path("/proc/mounts")
    if proc_mounts.is_file():
        for line in proc_mounts.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 3:
                mounts.append((fields[1].replace("\\040", " "), fields[2]))
    elif sys.platform == "darwin":
        out = subprocess.run(
            ["/sbin/mount"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        ).stdout
        # "//user@host/share on /Volumes/share (smbfs, nodev, nosuid, mounted by user)"
        for line in out.splitlines():
            if " on " in line and " (" in line:
                rest = line.split(" on ", 1)[1]
                point, _, opts = rest.rpartition(" (")
                mounts.append((point, opts.split(",", 1)[0].strip(") ")))
    best: tuple[int, str] | None = None
    text = str(path)
    for point, fs_type in mounts:
        if text == point or text.startswith(point.rstrip("/") + "/") or point == "/":
            if best is None or len(point) > best[0]:
                best = (len(point), fs_type)
    return best[1] if best else None
