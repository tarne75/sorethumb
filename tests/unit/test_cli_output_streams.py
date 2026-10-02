"""The CLI's output streams can't abort a command over an unencodable character.

On Windows a pipe or file gets the ANSI code page (cp1252), so a column named
"温度" in a reason, or an arrow in our own text, raised UnicodeEncodeError from
``sorethumb anomalies`` whenever output was redirected. These tests drive
``_configure_output_streams`` with in-memory text streams standing in for
sys.stdout/sys.stderr, so they run identically on every platform.
"""

from __future__ import annotations

import io

import pytest

from sorethumb_ml.cli import _configure_output_streams, _SorethumbTyper, app

pytestmark = pytest.mark.unit


class _Stream(io.TextIOWrapper):
    def __init__(self, encoding: str, *, tty: bool, errors: str = "strict") -> None:
        super().__init__(io.BytesIO(), encoding=encoding, errors=errors)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _no_pythonioencoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)


def test_a_redirected_cp1252_stream_becomes_utf8_and_never_raises() -> None:
    stream = _Stream("cp1252", tty=False)
    _configure_output_streams((stream,))
    assert stream.encoding == "utf-8"
    assert stream.errors == "replace"
    stream.write("温度 → naïve")
    stream.flush()
    assert stream.buffer.getvalue().decode("utf-8") == "温度 → naïve"  # type: ignore[attr-defined]


def test_a_console_keeps_its_encoding_but_stops_raising() -> None:
    stream = _Stream("cp437", tty=True)
    _configure_output_streams((stream,))
    assert stream.encoding == "cp437"
    assert stream.errors == "replace"
    stream.write("温度")  # would raise UnicodeEncodeError under "strict"


def test_an_explicit_pythonioencoding_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    stream = _Stream("cp1252", tty=False)
    _configure_output_streams((stream,))
    assert stream.encoding == "cp1252"
    assert stream.errors == "replace"
    stream.write("温度")


def test_an_existing_non_strict_handler_is_kept() -> None:
    stream = _Stream("cp1252", tty=False, errors="backslashreplace")  # what sys.stderr uses
    _configure_output_streams((stream,))
    assert stream.encoding == "utf-8"
    assert stream.errors == "backslashreplace"


def test_utf8_streams_are_left_alone() -> None:
    stream = _Stream("UTF-8", tty=False, errors="surrogateescape")
    _configure_output_streams((stream,))
    assert stream.encoding == "UTF-8"
    assert stream.errors == "surrogateescape"


def test_streams_without_reconfigure_are_skipped() -> None:
    _configure_output_streams((io.StringIO(),))  # no reconfigure(): must not raise


def test_the_console_script_target_configures_streams_on_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """``app`` itself is the console-script entry point; calling it configures streams first."""
    assert isinstance(app, _SorethumbTyper)
    called: list[bool] = []
    monkeypatch.setattr("sorethumb_ml.cli._configure_output_streams", lambda: called.append(True))
    with pytest.raises(SystemExit):
        app(["--version"])
    assert called == [True]
