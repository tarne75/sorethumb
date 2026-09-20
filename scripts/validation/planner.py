"""Pure matrix planning and result-file I/O for the validation sweep.

``build_cases``/``plan_run`` do no I/O and are fully unit-tested with
in-memory data. ``read_results_file``/``write_results_atomic`` do file I/O
but no dataset loading or model fitting -- the split from ``runner.py`` the
plan asks for.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from scripts.validation.schema import CASE_SCHEMA_VERSION, CaseKey, CaseResult, ComboSpec, DatasetSpec

logger = logging.getLogger(__name__)


def build_cases(
    datasets: list[DatasetSpec],
    combos: list[ComboSpec],
    pca_settings: list[bool],
    seeds: list[int],
    *,
    dataset_filter: str | None = None,
    combo_filter: str | None = None,
    pca_filter: bool | None = None,
) -> list[CaseKey]:
    """Deterministic Cartesian product of dataset × pca × combo × seed.

    Iteration order is dataset, then pca, then combo, then seed -- stable
    across runs so progress output and results.json ordering don't churn.
    Any of the three ``*_filter`` args narrows the product to a single value;
    ``None`` keeps every value.
    """
    cases: list[CaseKey] = []
    for ds in datasets:
        if dataset_filter is not None and ds.name != dataset_filter:
            continue
        for pca in pca_settings:
            if pca_filter is not None and pca != pca_filter:
                continue
            for combo in combos:
                if combo_filter is not None and combo.name != combo_filter:
                    continue
                for seed in seeds:
                    cases.append(CaseKey(dataset=ds.name, pca=pca, combo=combo.name, seed=seed))
    return cases


def plan_run(
    cases: list[CaseKey],
    existing: dict[CaseKey, CaseResult],
    identity_for_case: Callable[[CaseKey], object],
) -> tuple[list[CaseKey], list[CaseResult]]:
    """Split *cases* into (to_run, reused) given already-stored results.

    A case is reused only when a stored result exists for its key, that
    result's status is "success" or "too_few_records" -- both are genuine,
    stable outcomes of running the case (an "error" result is stale by
    definition and always reruns) -- and its full identity -- schema
    version, code revision, data fingerprint, config hash, seed, dependency
    versions -- matches exactly what ``identity_for_case`` computes for that
    case right now. Any mismatch (code changed, dataset file changed, config
    changed, a dependency was upgraded) forces a rerun rather than silently
    trusting a result that may no longer describe the current code/data/config.
    """
    to_run: list[CaseKey] = []
    reused: list[CaseResult] = []
    for case in cases:
        prior = existing.get(case)
        if (
            prior is not None
            and prior.status in ("success", "too_few_records")
            and prior.identity == identity_for_case(case)
        ):
            reused.append(prior)
            continue
        to_run.append(case)
    return to_run, reused


def read_results_file(path: Path) -> list[CaseResult]:
    """Load previously stored results, tolerating a missing or foreign-schema file.

    ``validation/`` is gitignored scratch output with no backward-compatibility
    contract with older schema versions of this script -- a file that doesn't
    parse as the current wrapped ``{"schema_version", "results"}`` shape is
    treated as "nothing stored yet" (a fresh start) rather than crashing the
    whole run. Individual rows that fail to parse are skipped and logged
    rather than aborting the entire load.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s (%s); starting with no prior results.", path, exc)
        return []

    if not isinstance(raw, dict) or "results" not in raw:
        logger.warning("%s is not in the expected {schema_version, results} shape; ignoring.", path)
        return []

    results: list[CaseResult] = []
    for row in raw["results"]:
        try:
            results.append(CaseResult.from_dict(row))
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping unparsable result row in %s: %s", path, exc)
    return results


def write_results_atomic(path: Path, results: list[CaseResult]) -> None:
    """Write *results* to *path* atomically (temp file + fsync + rename).

    A crash or interrupt mid-write leaves the previous complete file intact
    rather than a truncated/corrupt one -- this file is read back on every
    resume, so a partial write would otherwise cost the whole run's progress.
    """
    from sorethumb._atomic import atomic_write_text  # noqa: PLC0415

    payload = {
        "schema_version": CASE_SCHEMA_VERSION,
        "results": [r.as_dict() for r in results],
    }
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
