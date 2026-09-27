"""P1-1 regression guard: the full-dataset FeatureSpace fit_features() returns
must not be kept reachable once run_detection moves on to per-group
processing.

Verified via a weakref rather than absolute memory numbers -- deterministic
and immune to noise from unrelated memory consumers (parquet/CSV decoding,
polars' own buffers, per-process baseline overhead) that make an
RSS-threshold assertion fragile. See tests/benchmark/test_memory_budget.py
for the complementary real-subprocess peak-RSS angle.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest

import sorethumb._pipeline as pipeline_mod
from sorethumb._pipeline import run_detection
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration


def test_full_dataset_feature_space_is_not_kept_alive_after_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit_features() is called once, on the full dataset, purely for its
    side effect of mutating *plan* -- its returned FeatureSpace (a
    whole-dataset-sized matrix) is never used afterwards. Before P1-1 it was
    bound to a local (``full_space``) captured by the per-group closure and
    so stayed reachable for run_detection's entire per-group loop; a weakref
    to it must now die once run_detection returns (in practice, as soon as
    fit_features' caller stops referencing it -- refcounting collects it
    immediately, well before the group loop even starts)."""
    captured: list[weakref.ReferenceType] = []
    real_fit_features = pipeline_mod.fit_features

    def spy_fit_features(df, plan, config):
        space = real_fit_features(df, plan, config)
        captured.append(weakref.ref(space))
        return space

    monkeypatch.setattr(pipeline_mod, "fit_features", spy_fit_features)

    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=6, seed=0)
    cfg = make_config(csv, ws)

    result = run_detection(cfg, no_report=True)
    assert result.n_succeeded >= 1

    assert captured, "fit_features was never called"
    gc.collect()
    assert captured[0]() is None, (
        "fit_features' returned FeatureSpace is still reachable after "
        "run_detection finished -- the dead full_space reference (P1-1) may "
        "have been reintroduced."
    )
