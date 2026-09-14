"""Unit tests for config.py behavioural contracts: validation, config_hash,
and the shipped default detector ensemble.

Pure field-default/assignment echoes were deliberately not ported here from
the old coverage-tier files (P1-3): docs/generate_config_docs.py already
derives docs/configuration.md from the same Pydantic Field defaults and
Literal enum members, and its drift check (tests/unit/test_docs_checks.py)
already catches a silently-changed default or narrowed/widened Literal --
duplicating that as a hand-written assertion here defended nothing an
independent mechanism didn't already defend.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_detectors_factory() -> None:
    """The shipped default ensemble is exactly these three detectors."""
    from sorethumb.config import _default_detectors

    dets = _default_detectors()
    assert len(dets) == 3
    names = {d.name for d in dets}
    assert names == {"isolation_forest", "kmeans_distance", "one_class_svm"}


def test_scoring_config_contamination_invalid_string() -> None:
    from pydantic import ValidationError

    from sorethumb.config import ScoringConfig

    with pytest.raises(ValidationError):
        ScoringConfig(contamination="bad")


def test_scoring_config_contamination_out_of_range() -> None:
    from pydantic import ValidationError

    from sorethumb.config import ScoringConfig

    with pytest.raises(ValidationError):
        ScoringConfig(contamination=0.6)


def test_config_hash_is_stable() -> None:
    from sorethumb.config import Config, RunConfig, SourceConfig

    cfg = Config(
        source=SourceConfig(uri="/tmp/data.csv"),
        run=RunConfig(workdir="/tmp/ws"),
    )
    h1 = cfg.config_hash()
    h2 = cfg.config_hash()
    assert h1 == h2
    assert len(h1) == 32


def test_config_hash_excludes_cosmetic_fields() -> None:
    """workdir, log_level and slow_stage_seconds are execution-only knobs --
    changing them must not bust artefact caches keyed on config_hash()."""
    from sorethumb.config import Config, RunConfig, SourceConfig

    cfg1 = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws"))
    cfg2 = Config(
        source=SourceConfig(uri="/tmp/data.csv"),
        run=RunConfig(workdir="/different/path", log_level="DEBUG", slow_stage_seconds=60),
    )
    assert cfg1.config_hash() == cfg2.config_hash()


def test_config_hash_changes_with_result_affecting_fields() -> None:
    from sorethumb.config import Config, RunConfig, SourceConfig

    cfg1 = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws"))
    cfg2 = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws", seed=999))
    assert cfg1.config_hash() != cfg2.config_hash()
