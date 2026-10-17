"""The report's explanation detail: the record behind each reason, in index.html and index.json."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import DetectorConfig
from sorethumb_ml.explain.record import RECORD_SCHEMA, records_frame
from sorethumb_ml.report.html import GroupSection, RunMeta, _reason_detail, render_report
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv
from tests.factories.golden import assert_matches_golden

pytestmark = pytest.mark.integration

_META = RunMeta(
    run_id="run_detail_01",
    dataset_uri="file:///data/t.csv",
    dataset_fp="fp",
    config_hash="ch",
    seed=1,
    library_version="0.1.0",
    python_version="3.12.0",
    config_json="{}",
)

_ROWS = pl.DataFrame(
    {
        "row_id": [7, 3, 9],
        "rank": [1, 2, 3],
        "composite_score": [0.99, 0.95, 0.90],
        "flagged": [True, True, True],
        "explain_status": ["ok", "partial", "unavailable"],
        "explain_issue": [None, "detectors_unexplained:lof", "beyond_max_rows"],
        "reason_1": ["amount=900.0", "<b>cat</b>=x", None],
    }
)


def _record(row_id: int, rank: int, field: str, raw: str | None, **kw: object) -> dict[str, object]:
    rec: dict[str, object] = {
        "row_id": row_id,
        "reason_rank": rank,
        "original_field": field,
        "raw_value": raw,
        "reference_value": "50.0",
        "direction": "above",
        "signed_contribution": 0.8,
        "contribution_magnitude": 0.9,
        "attribution_kind": "model_specific",
        "methods": [
            {
                "detector": "isolation_forest",
                "method": "tree_shap",
                "attribution_kind": "model_specific",
                "weight": 1.0,
                "signed_contribution": 0.8,
            }
        ],
        "transformed_features": [{"feature": field, "signed_contribution": 0.8}],
        "approximation_flags": ["treeshap_path_length"],
        "explainer_version": "explain@2",
        "reference_id": "consensus_normal",
    }
    rec.update(kw)
    return rec


def _attributions() -> pl.DataFrame:
    return records_frame(
        [
            _record(7, 1, "amount", "900.0"),
            _record(7, 2, "lat", "1.5", signed_contribution=-0.1, direction="below"),
            _record(
                3,
                1,
                "<b>cat</b>",
                "<script>alert(1)</script>",
                reference_value="A",
                direction="different",
                approximation_flags=["not_covered_by:lof", "treeshap_path_length"],
            ),
        ],
        row_id_dtype=pl.Int64,
    )


def _group(attributions: pl.DataFrame | None) -> GroupSection:
    return GroupSection(group_key="g1", group_label="all", records=_ROWS, attributions=attributions)


def test_detail_block_matches_golden() -> None:
    assert_matches_golden(_reason_detail(_group(_attributions())), "report_reason_detail.html")


def test_detail_lists_rows_by_rank_with_reasons_flags_and_legend() -> None:
    out = _reason_detail(_group(_attributions()))
    assert out.index("#1 · row 7") < out.index("#2 · row 3")
    assert "Explainer explain@2, reference consensus_normal." in out
    assert "isolation_forest: tree_shap (model_specific) w=1.00" in out
    assert '<span class="pos">+0.8</span>' in out
    assert '<span class="neg">-0.1</span>' in out
    assert "#3 · row 9: no explanation (beyond_max_rows)." in out
    assert "What the flags mean" in out
    assert "<dt>not_covered_by:&lt;det&gt;</dt>" in out
    assert "<dt>treeshap_path_length</dt>" in out


def test_detail_escapes_field_names_and_values() -> None:
    out = _reason_detail(_group(_attributions()))
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "<b>cat</b>" not in out
    assert "&lt;b&gt;cat&lt;/b&gt;" in out


@pytest.mark.parametrize("attributions", [None, records_frame([], row_id_dtype=pl.Int64)])
def test_no_records_renders_no_detail(tmp_path: Path, attributions: pl.DataFrame | None) -> None:
    assert _reason_detail(_group(attributions)) == ""
    page = render_report(_META, [_group(attributions)], tmp_path, formats=["html", "json"])
    assert "Explanation detail" not in page.read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    group = payload["groups"][0]
    if attributions is None:
        assert "attributions" not in group
    else:
        assert group["attributions"] == []


def test_records_without_a_row_id_column_render_no_detail() -> None:
    grp = GroupSection(
        group_key="g1", group_label="all", records=pl.DataFrame({"x": [1]}), attributions=_attributions()
    )
    assert _reason_detail(grp) == ""


def test_json_report_carries_the_records(tmp_path: Path) -> None:
    render_report(_META, [_group(_attributions())], tmp_path, formats=["json"])
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    recs = payload["groups"][0]["attributions"]
    assert len(recs) == 3
    assert set(recs[0]) == {"row_id", *RECORD_SCHEMA}


def test_a_real_run_report_shows_the_explanation_detail(tmp_path: Path) -> None:
    csv, ws = tmp_path / "d.csv", tmp_path / "ws"
    write_planted_csv(csv, n_normal=200, n_anomaly=6, seed=0)
    cfg = make_config(
        csv,
        ws,
        combination="composite",
        contamination=0.05,
        detectors=[DetectorConfig(name="isolation_forest"), DetectorConfig(name="ecod")],
        report_kwargs={"formats": ["html", "json"]},
    )
    result = run_detection(cfg)
    page = (ws / "reports" / result.run_id / "index.html").read_text(encoding="utf-8")
    assert "Explanation detail" in page
    assert "reference consensus_normal" in page
    assert "ecod: ecod_decomposition (exact)" in page
    assert "num_a" in page
    payload = json.loads((ws / "reports" / result.run_id / "index.json").read_text(encoding="utf-8"))
    assert payload["groups"][0]["attributions"]
