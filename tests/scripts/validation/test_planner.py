"""Unit tests for scripts.validation.planner (pure matrix planning + result-file I/O)."""

from __future__ import annotations

import pytest
from scripts.validation.planner import build_cases, plan_run, read_results_file, write_results_atomic
from scripts.validation.schema import (
    CASE_SCHEMA_VERSION,
    CaseKey,
    CaseResult,
    ComboSpec,
    DatasetSpec,
    RunIdentity,
)

pytestmark = pytest.mark.unit


def _dataset(name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name, file=f"{name}.parquet", ignore=[], source="s", description="d", rows=1, cols=1
    )


def _combo(name: str) -> ComboSpec:
    return ComboSpec(name=name, detectors=("isolation_forest",))


def _identity(**overrides: object) -> RunIdentity:
    defaults: dict[str, object] = {
        "schema_version": CASE_SCHEMA_VERSION,
        "code_revision": "abc123",
        "data_fingerprint": "deadbeef",
        "config_hash": "cafef00d",
        "seed": 0,
        "dependency_versions": {"numpy": "1.2.3"},
    }
    defaults.update(overrides)
    return RunIdentity(**defaults)  # type: ignore[arg-type]


def _result(
    dataset: str, pca: bool, combo: str, seed: int = 0, status: str = "success", **overrides: object
) -> CaseResult:
    defaults: dict[str, object] = {
        "dataset": dataset,
        "pca": pca,
        "combo": combo,
        "detectors": ["isolation_forest"],
        "seed": seed,
        "identity": _identity(),
        "status": status,
        "elapsed_seconds": 1.0,
    }
    defaults.update(overrides)
    return CaseResult(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_cases
# ---------------------------------------------------------------------------


def test_build_cases_full_cartesian_product():
    datasets = [_dataset("a"), _dataset("b")]
    combos = [_combo("x"), _combo("y")]
    cases = build_cases(datasets, combos, [False, True], [0])
    assert len(cases) == 2 * 2 * 2  # datasets x combos x pca


def test_build_cases_is_deterministically_ordered():
    datasets = [_dataset("a"), _dataset("b")]
    combos = [_combo("x"), _combo("y")]
    c1 = build_cases(datasets, combos, [False, True], [0])
    c2 = build_cases(datasets, combos, [False, True], [0])
    assert c1 == c2


def test_build_cases_dataset_filter():
    datasets = [_dataset("a"), _dataset("b")]
    combos = [_combo("x")]
    cases = build_cases(datasets, combos, [False], [0], dataset_filter="b")
    assert {c.dataset for c in cases} == {"b"}


def test_build_cases_combo_filter():
    datasets = [_dataset("a")]
    combos = [_combo("x"), _combo("y")]
    cases = build_cases(datasets, combos, [False], [0], combo_filter="y")
    assert {c.combo for c in cases} == {"y"}


def test_build_cases_multiple_seeds():
    cases = build_cases([_dataset("a")], [_combo("x")], [False], [0, 1, 2])
    assert {c.seed for c in cases} == {0, 1, 2}
    assert len(cases) == 3


# ---------------------------------------------------------------------------
# plan_run
# ---------------------------------------------------------------------------


def test_plan_run_no_prior_results_runs_everything():
    cases = [CaseKey("a", False, "x", 0), CaseKey("b", False, "x", 0)]
    to_run, reused = plan_run(cases, {}, lambda _c: _identity())
    assert to_run == cases
    assert reused == []


def test_plan_run_reuses_matching_successful_result():
    case = CaseKey("a", False, "x", 0)
    prior = _result("a", False, "x", identity=_identity())
    to_run, reused = plan_run([case], {case: prior}, lambda _c: _identity())
    assert to_run == []
    assert reused == [prior]


def test_plan_run_reruns_when_identity_differs():
    case = CaseKey("a", False, "x", 0)
    prior = _result("a", False, "x", identity=_identity(code_revision="old"))
    to_run, reused = plan_run([case], {case: prior}, lambda _c: _identity(code_revision="new"))
    assert to_run == [case]
    assert reused == []


def test_plan_run_reruns_when_data_fingerprint_differs():
    case = CaseKey("a", False, "x", 0)
    prior = _result("a", False, "x", identity=_identity(data_fingerprint="old-fp"))
    to_run, reused = plan_run([case], {case: prior}, lambda _c: _identity(data_fingerprint="new-fp"))
    assert to_run == [case]


def test_plan_run_reruns_when_config_hash_differs():
    case = CaseKey("a", False, "x", 0)
    prior = _result("a", False, "x", identity=_identity(config_hash="old-cfg"))
    to_run, reused = plan_run([case], {case: prior}, lambda _c: _identity(config_hash="new-cfg"))
    assert to_run == [case]


def test_plan_run_reruns_when_dependency_versions_differ():
    case = CaseKey("a", False, "x", 0)
    prior = _result("a", False, "x", identity=_identity(dependency_versions={"numpy": "1.0.0"}))
    to_run, reused = plan_run(
        [case], {case: prior}, lambda _c: _identity(dependency_versions={"numpy": "2.0.0"})
    )
    assert to_run == [case]


def test_plan_run_always_reruns_error_status_even_with_matching_identity():
    case = CaseKey("a", False, "x", 0)
    prior = _result("a", False, "x", status="error", identity=_identity())
    to_run, reused = plan_run([case], {case: prior}, lambda _c: _identity())
    assert to_run == [case]
    assert reused == []


def test_plan_run_partial_reuse():
    cases = [CaseKey("a", False, "x", 0), CaseKey("b", False, "x", 0)]
    prior_a = _result("a", False, "x", identity=_identity())
    to_run, reused = plan_run(cases, {cases[0]: prior_a}, lambda _c: _identity())
    assert to_run == [cases[1]]
    assert reused == [prior_a]


# ---------------------------------------------------------------------------
# results.json I/O
# ---------------------------------------------------------------------------


def test_write_then_read_results_round_trips(tmp_path):
    path = tmp_path / "results.json"
    results = [_result("a", False, "x"), _result("b", True, "y", status="error", error="boom")]
    write_results_atomic(path, results)
    loaded = read_results_file(path)
    assert loaded == results


def test_read_results_file_missing_returns_empty(tmp_path):
    assert read_results_file(tmp_path / "does_not_exist.json") == []


def test_read_results_file_garbage_json_returns_empty(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("not json{{{")
    assert read_results_file(path) == []


def test_read_results_file_wrong_shape_returns_empty(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('[{"dataset": "a"}]')  # old flat-list shape, not {schema_version, results}
    assert read_results_file(path) == []


def test_read_results_file_skips_unparsable_rows_but_keeps_the_rest(tmp_path):
    path = tmp_path / "results.json"
    good = _result("a", False, "x")
    import json

    payload = {
        "schema_version": CASE_SCHEMA_VERSION,
        "results": [good.as_dict(), {"totally": "unparsable"}],
    }
    path.write_text(json.dumps(payload))
    loaded = read_results_file(path)
    assert loaded == [good]


def test_write_results_atomic_leaves_no_temp_files_behind(tmp_path):
    path = tmp_path / "results.json"
    write_results_atomic(path, [_result("a", False, "x")])
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []
