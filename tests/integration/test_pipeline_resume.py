"""Run-identity and resume semantics (P0-1, Phase 3): run_id is deterministic
for identical inputs, a resumed run skips already-complete groups, and a
group that fails (no detector produced scores, a fit() exception, a
score_samples() exception) is persisted as failed -- never complete -- and
is genuinely retried rather than skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sorethumb import Config
from sorethumb._pipeline import run_detection
from sorethumb.config import DetectorConfig
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration


def _minimal_config(
    csv_path: Path,
    workdir: Path,
    *,
    contamination: str | float = "auto",
    combination: str = "composite",
    detectors: list[DetectorConfig] | None = None,
) -> Config:
    return make_config(
        csv_path,
        workdir,
        contamination=contamination,
        combination=combination,
        detectors=detectors,
    )


def test_run_id_is_deterministic(tmp_path: Path) -> None:
    """Two calls with identical config and data must return the same run_id."""
    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"

    cfg = _minimal_config(csv, workdir, contamination=0.02)

    r1 = run_detection(cfg, no_report=True)
    r2 = run_detection(cfg, no_report=True)

    assert r1.run_id == r2.run_id, (
        f"Expected identical run_ids for identical inputs; got {r1.run_id!r} vs {r2.run_id!r}"
    )


def test_resume_skips_completed_group(tmp_path: Path) -> None:
    """A second run with identical inputs must skip already-complete groups."""
    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"

    cfg = _minimal_config(csv, workdir, contamination=0.02)

    r1 = run_detection(cfg, no_report=True)
    assert r1.n_succeeded >= 1

    r2 = run_detection(cfg, no_report=True)
    assert r2.n_skipped >= 1, (
        f"Second identical run should skip completed groups; "
        f"got n_skipped={r2.n_skipped}, n_succeeded={r2.n_succeeded}"
    )


def test_group_with_no_scoring_detectors_is_failed_not_complete(tmp_path: Path) -> None:
    """A group where no configured detector produces scores must be recorded
    as failed in both the RunResult and the ledger, and must be retried (not
    silently skipped as already complete) on the next call with the same
    inputs.
    """
    from sorethumb import Workspace

    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.02, detectors=[DetectorConfig(name="does_not_exist")])

    r1 = run_detection(cfg, no_report=True)
    assert r1.n_failed == 1
    assert r1.n_succeeded == 0
    g1 = r1.groups[0]
    assert g1.status == "failed"
    assert g1.error is not None
    assert "No detectors produced scores" in g1.error

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r1.run_id) == "failed"
        assert ws.store.group_status(r1.run_id, g1.group_key) == "failed"

    # Same inputs -> same deterministic run_id. The group must be re-executed,
    # not treated as already complete, and the run must still report failure.
    r2 = run_detection(cfg, no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_failed == 1
    assert r2.n_succeeded == 0
    assert r2.groups[0].status == "failed", "a failed group must be retried, not skipped"

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r2.run_id) == "failed"


def test_group_ledger_status_survives_detector_fit_failure_and_retry(tmp_path: Path) -> None:
    """A detector.fit() exception must fail the group and the run; once the
    underlying problem is gone, retrying the same run_id must actually
    re-execute the group (not report false success from a stale 'complete'
    ledger row, and not report false failure once it truly works).
    """
    from sorethumb import Workspace

    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.02)

    def _boom(*_a: object, **_k: object) -> None:
        raise ValueError("synthetic fit failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IsolationForestDetector, "fit", _boom)
        r1 = run_detection(cfg, no_report=True)

    assert r1.n_failed == 1
    assert r1.n_succeeded == 0
    g1 = r1.groups[0]
    assert g1.status == "failed"
    assert "synthetic fit failure" in (g1.error or "")

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r1.run_id) == "failed"
        assert ws.store.group_status(r1.run_id, g1.group_key) == "failed"

    # fit() is no longer patched -- the retry must actually re-run the group
    # (the ledger must not have recorded it as already complete) and succeed.
    r2 = run_detection(cfg, no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_failed == 0
    assert r2.n_succeeded == 1

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r2.run_id) == "complete"
        assert ws.store.group_status(r2.run_id, g1.group_key) == "complete"


def test_group_ledger_status_survives_score_samples_failure_and_retry(tmp_path: Path) -> None:
    """Same as the fit-failure case, but the exception comes from
    score_samples (after a successful fit) instead of fit itself.
    """
    from sorethumb import Workspace

    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.02)

    def _boom(*_a: object, **_k: object) -> None:
        raise ValueError("synthetic scoring failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IsolationForestDetector, "score_samples", _boom)
        r1 = run_detection(cfg, no_report=True)

    assert r1.n_failed == 1
    g1 = r1.groups[0]
    assert g1.status == "failed"

    with Workspace.open(workdir) as ws:
        assert ws.store.group_status(r1.run_id, g1.group_key) == "failed"

    r2 = run_detection(cfg, no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_succeeded == 1
    assert r2.n_failed == 0
