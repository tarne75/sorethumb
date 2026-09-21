"""Unit tests for report/charts.py's lazy matplotlib import (P3-3)."""

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def test_import_charts_module_does_not_require_matplotlib(monkeypatch):
    """matplotlib lives in the optional `report` extra (P3-3), not a core
    dependency -- importing sorethumb.report.charts must never need it, only
    actually calling render_trend_chart does."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` raise ImportError
    sys.modules.pop("sorethumb.report.charts", None)

    try:
        module = importlib.import_module("sorethumb.report.charts")
        assert hasattr(module, "render_trend_chart")
    finally:
        sys.modules.pop("sorethumb.report.charts", None)
        importlib.import_module("sorethumb.report.charts")


def test_render_trend_chart_raises_clear_error_without_matplotlib(monkeypatch):
    """render_trend_chart has no fallback (unlike shap) -- render_trend_chart
    is not wired into the pipeline, so without matplotlib installed it must
    raise a clear ImportError rather than fail silently."""
    from sorethumb.report.charts import render_trend_chart

    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(ImportError):
        render_trend_chart(
            period_labels=["2026-01-01", "2026-01-02"],
            group_anomaly_counts={"g1": [1, 2]},
            period_population=[10, 10],
            windows=[1, 7],
            reference_label="2026-01-02",
        )
