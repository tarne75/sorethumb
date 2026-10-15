"""Every project warning is a structured record with a stable code."""

from __future__ import annotations

import json
import warnings

import pytest

import sorethumb_ml.errors as errors_mod
from sorethumb_ml.errors import (
    AttributionBackendWarning,
    ExplanationCoverageWarning,
    SorethumbWarning,
    warning_record,
    warning_records_from_json,
)

pytestmark = pytest.mark.unit


def _warning_classes() -> list[type[SorethumbWarning]]:
    return [
        obj
        for obj in vars(errors_mod).values()
        if isinstance(obj, type) and issubclass(obj, SorethumbWarning) and obj is not SorethumbWarning
    ]


def test_every_warning_class_has_its_own_dotted_code() -> None:
    codes = [cls.code for cls in _warning_classes()]
    assert all("." in c for c in codes), codes
    assert len(codes) == len(set(codes)), codes


def test_a_record_carries_code_stage_detector_and_rows() -> None:
    w = AttributionBackendWarning("backend broke", detector="one_class_svm", rows_affected=7)
    assert warning_record(w, group="g1") == {
        "code": "explain.backend_failed",
        "stage": "explain",
        "message": "backend broke",
        "group": "g1",
        "detector": "one_class_svm",
        "rows_affected": 7,
    }
    specific = ExplanationCoverageWarning("x", code="explain.rows_beyond_max_rows", rows_affected=3)
    assert warning_record(specific)["code"] == "explain.rows_beyond_max_rows"
    assert ExplanationCoverageWarning.code == "explain.coverage"  # the instance override does not leak
    assert warning_record(RuntimeWarning("theirs"))["code"] == "external"


def test_a_structured_warning_survives_the_warnings_machinery() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(AttributionBackendWarning("m", detector="d"), stacklevel=1)
    (w,) = caught
    assert issubclass(w.category, AttributionBackendWarning)
    assert warning_record(w.message)["detector"] == "d"


def test_plain_string_warnings_from_older_runs_read_back_as_legacy_records() -> None:
    raw = json.dumps(["an old warning", {"code": "explain.backend_failed", "message": "new"}])
    old, new = warning_records_from_json(raw)
    assert (old["code"], old["message"]) == ("legacy", "an old warning")
    assert new["code"] == "explain.backend_failed"
    assert warning_records_from_json(None) == []


def test_the_anomalies_table_says_why_a_row_has_no_full_explanation() -> None:
    from sorethumb_ml.cli import _explain_kind_cell

    assert _explain_kind_cell({"explain_status": "unavailable", "explain_issue": "beyond_max_rows"}) == (
        "unavailable: beyond_max_rows"
    )
    assert (
        _explain_kind_cell({"explain_status": "partial", "attribution_kind": "heuristic"})
        == "heuristic (partial)"
    )
    assert _explain_kind_cell({"explain_status": "ok", "attribution_kind": "exact"}) == "exact"
    assert _explain_kind_cell({"attribution_kind": None, "explain_status": None}) == "unexplained"
    assert _explain_kind_cell({"rank": 1}) == "—"
