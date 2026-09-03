"""Unit tests for M7: charts, CSV, HTML report, and contrast analysis."""

from __future__ import annotations

import base64
import html as html_mod
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb.analysis.contrast import compute_contrast
from sorethumb.report.charts import render_trend_chart
from sorethumb.report.csv import write_group_csv
from sorethumb.report.html import GroupSection, RunMeta, render_report

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_RUN_META = RunMeta(
    run_id="run_test_01",
    dataset_uri="file:///data/test.parquet",
    dataset_fp="abcd1234",
    config_hash="cafebabe",
    seed=42,
    library_version="0.1.0",
    python_version="3.12.0",
    config_json='{"scoring": {"contamination": "auto"}}',
    started_at="2026-09-03T08:00:00Z",
)

_RECORDS_DF = pl.DataFrame(
    {
        "row_id": [0, 1, 2],
        "score": [0.95, 0.88, 0.75],
        "rank": [1, 2, 3],
        "reason_1": ["high_value", "low_value", "outlier"],
        "attribution_kind": ["exact", "heuristic", "heuristic"],
    }
)


def _group(key: str = "abc123def456abcd", label: str = "US", **kw: object) -> GroupSection:
    return GroupSection(
        group_key=key,
        group_label=label,
        records=_RECORDS_DF,
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# charts.py
# ---------------------------------------------------------------------------


class TestRenderTrendChart:
    def test_returns_valid_base64_png(self):
        png_b64 = render_trend_chart(
            period_labels=["2026-09-01", "2026-09-02", "2026-09-03"],
            group_anomaly_counts={"gk1": [5, 3, 7]},
            period_population=[1000, 1000, 1000],
            windows=[1, 7],
            reference_label="2026-09-03",
        )
        raw = base64.b64decode(png_b64)
        # PNG magic bytes
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_empty_groups_does_not_raise(self):
        png_b64 = render_trend_chart(
            period_labels=["2026-09-01"],
            group_anomaly_counts={},
            period_population=[500],
            windows=[1],
            reference_label="2026-09-01",
        )
        assert len(png_b64) > 0

    def test_cal_break_labels_accepted(self):
        png_b64 = render_trend_chart(
            period_labels=["2026-09-01", "2026-09-02"],
            group_anomaly_counts={"gk": [1, 2]},
            period_population=[100, 100],
            windows=[1],
            reference_label="2026-09-02",
            cal_break_labels={"2026-09-02"},
        )
        assert len(png_b64) > 0

    def test_non_business_labels_accepted(self):
        png_b64 = render_trend_chart(
            period_labels=["2026-09-05", "2026-09-06"],
            group_anomaly_counts={},
            period_population=[0, 0],
            windows=[1],
            reference_label="2026-09-06",
            non_business_labels={"2026-09-05", "2026-09-06"},
        )
        assert len(png_b64) > 0

    def test_multiple_groups_rendered(self):
        png_b64 = render_trend_chart(
            period_labels=["2026-09-01", "2026-09-02"],
            group_anomaly_counts={"gk1": [2, 3], "gk2": [1, 4], "gk3": [0, 1]},
            period_population=[200, 200],
            windows=[1, 2],
            reference_label="2026-09-02",
        )
        assert len(base64.b64decode(png_b64)) > 1000


# ---------------------------------------------------------------------------
# csv.py
# ---------------------------------------------------------------------------


class TestWriteGroupCsv:
    def test_writes_csv_file(self, tmp_path: Path):
        path = write_group_csv(_RECORDS_DF, tmp_path, "abc123def456abcd")
        assert path.exists()
        assert path.name == "abc123def456abcd.csv"

    def test_csv_has_correct_columns(self, tmp_path: Path):
        path = write_group_csv(_RECORDS_DF, tmp_path, "testkey12345678a")
        df = pl.read_csv(str(path))
        assert set(df.columns) == set(_RECORDS_DF.columns)

    def test_csv_row_count_matches(self, tmp_path: Path):
        path = write_group_csv(_RECORDS_DF, tmp_path, "rowcountkey12345")
        df = pl.read_csv(str(path))
        assert len(df) == len(_RECORDS_DF)

    def test_creates_out_dir_if_missing(self, tmp_path: Path):
        subdir = tmp_path / "new" / "nested"
        write_group_csv(_RECORDS_DF, subdir, "key1234567890123")
        assert subdir.is_dir()


# ---------------------------------------------------------------------------
# html.py — offline, escaping, CSV links, provenance
# ---------------------------------------------------------------------------


class TestRenderReport:
    def test_writes_index_html(self, tmp_path: Path):
        grp = _group()
        path = render_report(_RUN_META, [grp], tmp_path)
        assert path.name == "index.html"
        assert path.exists()

    def test_html_contains_run_id(self, tmp_path: Path):
        grp = _group()
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "run_test_01" in content

    def test_html_contains_dataset_uri(self, tmp_path: Path):
        grp = _group()
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "file:///data/test.parquet" in content

    def test_html_is_self_contained_no_external_refs(self, tmp_path: Path):
        """No CDN URLs, no external stylesheets, no external script src."""
        grp = _group()
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "://cdn" not in content
        assert '<link rel="stylesheet"' not in content
        assert "<script src=" not in content

    def test_html_escaping_script_tag(self, tmp_path: Path):
        """Values containing <script> must be escaped, not injected."""
        malicious_df = pl.DataFrame(
            {
                "row_id": [0],
                "reason_1": ["<script>alert('xss')</script>"],
                "score": [0.9],
                "rank": [1],
                "attribution_kind": ["exact"],
            }
        )
        grp = GroupSection(
            group_key="xsstest1234567ab",
            group_label="XSS test",
            records=malicious_df,
        )
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        # The raw <script>alert(...) must NOT appear verbatim
        assert "<script>alert('xss')</script>" not in content
        # The escaped version must appear
        assert html_mod.escape("<script>alert('xss')</script>") in content

    def test_csv_link_is_relative(self, tmp_path: Path):
        """CSV link uses ./<group_key>.csv — not an absolute path or data: URI."""
        key = "relativekey123ab"
        grp = _group(key=key)
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert f"./{key}.csv" in content
        assert "data:text/csv" not in content

    def test_sibling_csv_written(self, tmp_path: Path):
        """render_report must write the CSV alongside the HTML."""
        key = "siblingcsvkey12a"
        grp = _group(key=key)
        render_report(_RUN_META, [grp], tmp_path)
        assert (tmp_path / f"{key}.csv").exists()

    def test_multiple_groups_all_present(self, tmp_path: Path):
        groups = [_group(key=f"group{i}1234567890ab"[:16], label=f"G{i}") for i in range(3)]
        path = render_report(_RUN_META, groups, tmp_path)
        content = path.read_text(encoding="utf-8")
        for grp in groups:
            assert grp.group_key in content

    def test_chart_png_embedded(self, tmp_path: Path):
        from sorethumb.report.charts import render_trend_chart

        png_b64 = render_trend_chart(
            period_labels=["2026-09-01"],
            group_anomaly_counts={},
            period_population=[100],
            windows=[1],
            reference_label="2026-09-01",
        )
        grp = _group(chart_png_b64=png_b64)
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_plan_dropped_rendered(self, tmp_path: Path):
        grp = _group(plan_dropped=[{"column": "id_col", "reason": "identifier"}])
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "id_col" in content
        assert "identifier" in content

    def test_empty_groups_list_does_not_raise(self, tmp_path: Path):
        path = render_report(_RUN_META, [], tmp_path)
        assert path.exists()

    def test_no_anomalies_message(self, tmp_path: Path):
        grp = GroupSection(
            group_key="noanom12345678ab",
            group_label="Empty",
            records=pl.DataFrame(schema={"row_id": pl.Int64, "score": pl.Float64}),
        )
        path = render_report(_RUN_META, [grp], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "No anomalies flagged" in content

    def test_config_json_in_provenance(self, tmp_path: Path):
        path = render_report(_RUN_META, [], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "contamination" in content  # from config_json


# ---------------------------------------------------------------------------
# analysis/contrast.py
# ---------------------------------------------------------------------------


class TestComputeContrast:
    def _data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        rng = np.random.default_rng(42)
        flagged = pl.DataFrame(
            {
                "value": rng.normal(10, 1, 50).tolist(),
                "category": (["A"] * 40 + ["B"] * 10),
            }
        )
        unflagged = pl.DataFrame(
            {
                "value": rng.normal(0, 1, 200).tolist(),
                "category": (["A"] * 100 + ["B"] * 100),
            }
        )
        return flagged, unflagged

    def test_returns_dataframe(self):
        flagged, unflagged = self._data()
        df = compute_contrast(flagged, unflagged, ["value"], ["category"])
        assert isinstance(df, pl.DataFrame)

    def test_numeric_columns_scored(self):
        flagged, unflagged = self._data()
        df = compute_contrast(flagged, unflagged, ["value"], [])
        assert "value" in df["feature"].to_list()

    def test_categorical_columns_scored(self):
        flagged, unflagged = self._data()
        df = compute_contrast(flagged, unflagged, [], ["category"])
        assert "category" in df["feature"].to_list()

    def test_sorted_by_contrast_descending(self):
        flagged, unflagged = self._data()
        df = compute_contrast(flagged, unflagged, ["value"], ["category"])
        scores = df["contrast_score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_top_n_respected(self):
        rng = np.random.default_rng(0)
        flagged = pl.DataFrame({f"c{i}": rng.normal(i, 1, 20).tolist() for i in range(10)})
        unflagged = pl.DataFrame({f"c{i}": rng.normal(0, 1, 100).tolist() for i in range(10)})
        df = compute_contrast(flagged, unflagged, [f"c{i}" for i in range(10)], [], top_n=3)
        assert len(df) <= 3

    def test_empty_when_too_few_samples(self):
        flagged = pl.DataFrame({"value": [1.0]})  # only 1 sample
        unflagged = pl.DataFrame({"value": [0.0, 0.1, 0.2]})
        df = compute_contrast(flagged, unflagged, ["value"], [])
        assert len(df) == 0

    def test_missing_column_skipped_gracefully(self):
        flagged = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        unflagged = pl.DataFrame({"b": [0.0, 0.5, 0.3]})
        df = compute_contrast(flagged, unflagged, ["a"], [])
        # "a" is in flagged but not unflagged — skipped silently
        assert len(df) == 0

    def test_returns_correct_schema(self):
        flagged, unflagged = self._data()
        df = compute_contrast(flagged, unflagged, ["value"], [])
        assert "feature" in df.columns
        assert "kind" in df.columns
        assert "stat_name" in df.columns
        assert "stat_value" in df.columns
        assert "contrast_score" in df.columns

    def test_constant_column_handled(self):
        flagged = pl.DataFrame({"val": [5.0] * 20})
        unflagged = pl.DataFrame({"val": [5.0] * 100})
        # All same value — cohens_d = 0
        df = compute_contrast(flagged, unflagged, ["val"], [])
        if len(df) > 0:
            assert df.filter(pl.col("feature") == "val")["contrast_score"][0] == 0.0
