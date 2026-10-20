# Supported platforms

| OS | Python | Tested in CI on every change |
| --- | --- | --- |
| Linux (x86-64) | 3.11–3.14 | unit, contract, integration, property, packaging (`ubuntu-latest`) |
| macOS (Apple silicon) | 3.11–3.14 | unit, contract, integration (`macos-latest`) |
| Windows 10/11 (x64) | 3.11–3.14 | unit, contract, integration, and installing the built wheel and sdist and running them from PowerShell (`windows-latest`) |

Unit and contract tests run on every listed Python version on all three. The
integration suite, which exercises a real workspace, SQLite, the CLI as a subprocess
and report rendering, runs on Python 3.12 on all three
(`.github/workflows/release-validation.yml`), and the daily newest-dependencies job runs
it on 3.11 (Linux) and 3.14 (all three). `requires-python` stops below 3.15: a runtime
is added when it is tested, not before. On Windows, the CLI works the
same in PowerShell, Windows PowerShell 5.1 and cmd.exe.

Not supported:

- **A workspace on a network share or in a synced folder** (SMB/NFS, a mapped
  drive, OneDrive, Dropbox, Google Drive, iCloud) on any OS. Its SQLite
  database and atomic file replacement need a local drive; sorethumb warns
  when it detects one. Your data can be anywhere.
- **Very deep workspace paths on Windows** without long-path support enabled:
  sorethumb stops before running and says how much shorter the path must be.

Untested: **native Windows on ARM.** Every core dependency publishes a Windows
ARM64 wheel, but no CI job runs there; an x64 Python under emulation is the
tested route. The `benchmark` extra needs pyarrow, which publishes no Windows
ARM64 wheel.
