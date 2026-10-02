"""The JUnit summariser groups failures by cause and can emit GitHub annotations.

Fixture XML mirrors what pytest's --junitxml writes for --tb=short failures,
including a Windows-style absolute path in a frame line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import summarise_junit as sj

pytestmark = pytest.mark.unit

_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="2" skipped="1" tests="5">
<testcase classname="tests.unit.test_a" name="test_one">
<failure message="OSError: [Errno 9] Bad file descriptor">tests/unit/test_a.py:10: in test_one
    atomic_write_bytes(p, b"x")
D:\\a\\sorethumb\\sorethumb\\src\\sorethumb_ml\\_atomic.py:95: in _fsync_path
    os.fsync(fd)
E   OSError: [Errno 9] Bad file descriptor</failure></testcase>
<testcase classname="tests.unit.test_a" name="test_two">
<failure message="OSError: [Errno 9] Bad file descriptor">tests/unit/test_a.py:20: in test_two
    run()
src/sorethumb_ml/_atomic.py:95: in _fsync_path
    os.fsync(fd)
E   OSError: [Errno 9] Bad file descriptor</failure></testcase>
<testcase classname="tests.unit.test_b" name="test_three">
<error message="failed on setup with &quot;AttributeError: module 'os' has no attribute 'geteuid'&quot;">tests/unit/test_b.py:5: in fixture
    os.geteuid()
E   AttributeError: module 'os' has no attribute 'geteuid'</error></testcase>
<testcase classname="tests.unit.test_b" name="test_ok"/>
<testcase classname="tests.unit.test_b" name="test_skipped"><skipped message="nope"/></testcase>
</testsuite></testsuites>
"""


@pytest.fixture
def junit(tmp_path: Path) -> Path:
    path = tmp_path / "results.xml"
    path.write_text(_XML, encoding="utf-8")
    return path


def test_failures_are_grouped_by_exception_and_deepest_src_frame(junit: Path) -> None:
    groups = sj.collect([junit])
    assert set(groups) == {
        ("OSError", "src/sorethumb_ml/_atomic.py:95"),
        ("AttributeError", "tests/unit/test_b.py:5"),
    }
    assert len(groups[("OSError", "src/sorethumb_ml/_atomic.py:95")].tests) == 2
    assert groups[("AttributeError", "tests/unit/test_b.py:5")].tests == [
        "tests.unit.test_b::test_three [error]"
    ]


def test_plain_output_reports_counts_and_largest_group_first(
    junit: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sj.main([str(junit), "--label", "win"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[win] tests=5 failures=2 errors=1 skipped=1 groups=2"
    assert out[1] == "2 x OSError at src/sorethumb_ml/_atomic.py:95"


def test_annotations_are_single_line_workflow_commands(
    junit: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sj.main([str(junit), "--github-annotations", "--label", "win"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("::notice title=win summary::")
    errors = [line for line in lines if line.startswith("::error ")]
    assert len(errors) == 2
    assert all("\n" not in e and "%0A" in e for e in errors)


def test_groups_beyond_the_annotation_limit_are_folded(
    junit: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sj.main([str(junit), "--github-annotations", "--max-annotations", "1"])
    errors = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::error ")]
    assert len(errors) == 2
    assert "remaining 1 groups" in errors[1]


def test_a_missing_file_is_reported_not_fatal(
    junit: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sj.main([str(junit), str(tmp_path / "absent.xml")]) == 0
    assert "missing-files=" in capsys.readouterr().out
