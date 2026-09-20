"""Unit tests for scripts.validation.schema."""

from __future__ import annotations

import pytest
from scripts.validation.schema import CaseKey, CaseResult, RunIdentity

pytestmark = pytest.mark.unit


def _identity(**overrides: object) -> RunIdentity:
    defaults: dict[str, object] = {
        "schema_version": 1,
        "code_revision": "abc123",
        "data_fingerprint": "deadbeef",
        "config_hash": "cafef00d",
        "seed": 0,
        "dependency_versions": {"numpy": "1.2.3"},
    }
    defaults.update(overrides)
    return RunIdentity(**defaults)  # type: ignore[arg-type]


def _result(**overrides: object) -> CaseResult:
    defaults: dict[str, object] = {
        "dataset": "kddcup99_sa",
        "pca": False,
        "combo": "baseline",
        "detectors": ["isolation_forest"],
        "seed": 0,
        "identity": _identity(),
        "status": "success",
        "elapsed_seconds": 1.23,
    }
    defaults.update(overrides)
    return CaseResult(**defaults)  # type: ignore[arg-type]


def test_case_result_key_matches_its_own_fields():
    result = _result()
    key = result.key()
    assert key == CaseKey(dataset="kddcup99_sa", pca=False, combo="baseline", seed=0)


def test_case_result_round_trips_through_dict():
    result = _result(
        roc_auc=0.91,
        average_precision=0.5,
        n_holdout_anomalies_true=12,
        warnings=["a warning"],
        run_id="run_1",
        score_run_id="run_2",
    )
    restored = CaseResult.from_dict(result.as_dict())
    assert restored == result


def test_case_result_round_trips_error_status_with_none_metrics():
    result = _result(status="error", error="boom", roc_auc=None)
    restored = CaseResult.from_dict(result.as_dict())
    assert restored == result
    assert restored.roc_auc is None


def test_case_key_is_hashable_and_usable_as_dict_key():
    k1 = CaseKey(dataset="d", pca=True, combo="c", seed=0)
    k2 = CaseKey(dataset="d", pca=True, combo="c", seed=0)
    assert k1 == k2
    assert hash(k1) == hash(k2)
    d = {k1: "value"}
    assert d[k2] == "value"


def test_run_identity_equality_is_field_wise():
    assert _identity() == _identity()
    assert _identity(seed=1) != _identity(seed=0)
    assert _identity(code_revision="different") != _identity()
    assert _identity(dependency_versions={"numpy": "9.9.9"}) != _identity()
