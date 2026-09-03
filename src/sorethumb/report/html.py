"""Self-contained HTML report renderer.

IMPORTANT: html.escape() is the only escaping helper used in this module. It is
never rebound or aliased. Rewriting ``escape = html.escape`` and then calling
``escape(...)`` works on a clean top-to-bottom execution but fails if the module is
reloaded in the same process, which is the worst possible failure profile.

The generated file must work from a file:// URL with no internet connection. All
CSS and images are inline. External CSS, CDN links, and <script src="..."> are
forbidden.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunMeta:
    """Provenance block written to the top of every report."""

    run_id: str
    dataset_uri: str
    dataset_fp: str
    config_hash: str
    seed: int
    library_version: str
    python_version: str
    config_json: str
    started_at: str = ""


@dataclass
class GroupSection:
    """All data needed to render one group section of the report."""

    group_key: str
    group_label: str
    records: pl.DataFrame
    plan_dropped: list[dict[str, str]] = field(default_factory=list)
    contrast: pl.DataFrame | None = None
    chart_png_b64: str | None = None
    window_results: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_report(
    run_meta: RunMeta,
    groups: list[GroupSection],
    out_dir: Path,
) -> Path:
    """Write a self-contained ``index.html`` to *out_dir* and return its path.

    Also writes one sibling CSV per group (``<group_key>.csv``).
    Moving the HTML without its CSV siblings breaks the relative links.
    """
    from sorethumb.report.csv import write_group_csv  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSVs first (so links in HTML are valid as soon as the file appears)
    for grp in groups:
        write_group_csv(grp.records, out_dir, grp.group_key)

    body_parts: list[str] = [_provenance_block(run_meta)]

    # Tab nav
    body_parts.append(_tab_nav(groups))

    for i, grp in enumerate(groups):
        body_parts.append(_group_section(grp, i))

    html_content = _page(run_meta.run_id, "\n".join(body_parts))
    out_path = out_dir / "index.html"
    out_path.write_text(html_content, encoding="utf-8")
    logger.info("HTML report written: %s.", out_path)
    return out_path


# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 1rem 2rem; color: #222; }
h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.1rem; margin-top: 1.5rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem; margin: 0.5rem 0; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
th { background: #f5f5f5; }
tr:nth-child(even) { background: #fafafa; }
.tabs { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0 0; }
.tab-btn { cursor: pointer; padding: 0.3rem 0.8rem; border: 1px solid #aaa;
           border-radius: 4px 4px 0 0; background: #eee; font-size: 0.85rem; }
.tab-btn.active { background: #fff; border-bottom-color: #fff; font-weight: bold; }
.tab-panel { border: 1px solid #aaa; padding: 1rem; display: none; }
.tab-panel.active { display: block; }
.provenance { background: #f8f9fa; border: 1px solid #dee2e6; padding: 0.75rem 1rem;
              font-size: 0.8rem; margin: 0.5rem 0 1rem; }
.provenance pre { margin: 0; white-space: pre-wrap; }
.dropped-col { color: #888; font-size: 0.78rem; }
.chart-img { max-width: 100%; margin: 0.5rem 0; }
.csv-link { font-size: 0.82rem; }
"""

_JS = """
function showTab(groupIdx, tabName) {
  var panels = document.querySelectorAll('.tab-panel[data-group="' + groupIdx + '"]');
  panels.forEach(function(p) { p.classList.remove('active'); });
  var btns = document.querySelectorAll('.tab-btn[data-group="' + groupIdx + '"]');
  btns.forEach(function(b) { b.classList.remove('active'); });
  var panel = document.querySelector('.tab-panel[data-group="' + groupIdx + '"][data-tab="' + tabName + '"]');
  if (panel) panel.classList.add('active');
  var btn = document.querySelector('.tab-btn[data-group="' + groupIdx + '"][data-tab="' + tabName + '"]');
  if (btn) btn.classList.add('active');
}
"""


def _page(run_id: str, body: str) -> str:
    safe_run_id = html.escape(run_id)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        f"<title>sorethumb report — {safe_run_id}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>sorethumb report — {safe_run_id}</h1>\n"
        f"{body}\n"
        f"<script>{_JS}</script>\n"
        "</body>\n"
        "</html>"
    )


def _provenance_block(meta: RunMeta) -> str:
    config_pretty = json.dumps(json.loads(meta.config_json), indent=2) if meta.config_json else "{}"
    return (
        '<div class="provenance">'
        f"<strong>Run ID:</strong> {html.escape(meta.run_id)}&nbsp;&nbsp;"
        f"<strong>Dataset URI:</strong> {html.escape(meta.dataset_uri)}&nbsp;&nbsp;"
        f"<strong>Dataset FP:</strong> {html.escape(meta.dataset_fp)}&nbsp;&nbsp;"
        f"<strong>Config hash:</strong> {html.escape(meta.config_hash)}&nbsp;&nbsp;"
        f"<strong>Seed:</strong> {html.escape(str(meta.seed))}&nbsp;&nbsp;"
        f"<strong>sorethumb:</strong> {html.escape(meta.library_version)}&nbsp;&nbsp;"
        f"<strong>Python:</strong> {html.escape(meta.python_version)}"
        f"{'&nbsp;&nbsp;<strong>Started:</strong> ' + html.escape(meta.started_at) if meta.started_at else ''}"
        "<br><details><summary>Resolved config</summary>"
        f"<pre>{html.escape(config_pretty)}</pre>"
        "</details>"
        "</div>"
    )


def _tab_nav(groups: list[GroupSection]) -> str:
    if not groups:
        return ""
    parts = ["<div>"]
    for i, grp in enumerate(groups):
        safe_label = html.escape(grp.group_label or grp.group_key)
        parts.append(
            f"<h2>Group: {safe_label} "
            f'<span style="font-size:0.75rem;color:#888">({html.escape(grp.group_key)})</span></h2>'
            f'<div class="tabs">'
            f'<button class="tab-btn active" data-group="{i}" data-tab="records" '
            f"onclick=\"showTab({i}, 'records')\">Records</button>"
            f'<button class="tab-btn" data-group="{i}" data-tab="chart" '
            f"onclick=\"showTab({i}, 'chart')\">Chart</button>"
            f'<button class="tab-btn" data-group="{i}" data-tab="contrast" '
            f"onclick=\"showTab({i}, 'contrast')\">Contrast</button>"
            f'<button class="tab-btn" data-group="{i}" data-tab="plan" '
            f"onclick=\"showTab({i}, 'plan')\">Feature plan</button>"
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _group_section(grp: GroupSection, idx: int) -> str:
    parts: list[str] = []

    # Records tab
    csv_link = f'<p class="csv-link"><a href="./{html.escape(grp.group_key)}.csv">Download CSV</a> (moving this HTML without its sibling CSVs breaks this link)</p>'
    records_html = _df_to_table(grp.records) if len(grp.records) > 0 else "<p>No anomalies flagged.</p>"
    parts.append(
        f'<div class="tab-panel active" data-group="{idx}" data-tab="records">{csv_link}{records_html}</div>'
    )

    # Windows table
    if grp.window_results:
        parts[-1] = parts[-1].replace("</div>", _windows_table(grp.window_results) + "</div>")

    # Chart tab
    chart_html = ""
    if grp.chart_png_b64:
        chart_html = f'<img class="chart-img" src="data:image/png;base64,{html.escape(grp.chart_png_b64)}" alt="Trend chart">'
    parts.append(
        f'<div class="tab-panel" data-group="{idx}" data-tab="chart">'
        f"{chart_html or '<p>No chart available.</p>'}</div>"
    )

    # Contrast tab
    contrast_html = (
        _df_to_table(grp.contrast)
        if grp.contrast is not None and len(grp.contrast) > 0
        else "<p>No contrast data.</p>"
    )
    parts.append(f'<div class="tab-panel" data-group="{idx}" data-tab="contrast">{contrast_html}</div>')

    # Feature plan tab
    plan_html = _plan_table(grp.plan_dropped)
    parts.append(f'<div class="tab-panel" data-group="{idx}" data-tab="plan">{plan_html}</div>')

    return "\n".join(parts)


def _df_to_table(df: pl.DataFrame | None) -> str:
    if df is None or len(df) == 0:
        return "<p>No data.</p>"
    rows: list[str] = []
    header = "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
    for row in df.iter_rows():
        cells = "".join(f"<td>{html.escape(str(v) if v is not None else '')}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table><thead>" + header + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def _plan_table(dropped: list[dict[str, str]]) -> str:
    if not dropped:
        return "<p>No columns were dropped.</p>"
    rows = "".join(
        f'<tr><td class="dropped-col">{html.escape(str(d.get("column", "")))}</td>'
        f"<td>{html.escape(str(d.get('reason', '')))}</td></tr>"
        for d in dropped
    )
    return f"<table><thead><tr><th>Column</th><th>Drop reason</th></tr></thead><tbody>{rows}</tbody></table>"


def _windows_table(window_results: list[Any]) -> str:
    headers = [
        "Window",
        "Cur count",
        "Cur pop",
        "Cur rate %",
        "Prior rate %",
        "Δ abs",
        "Δ %",
        "Low vol",
        "Cal break",
    ]
    header_html = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"
    rows: list[str] = []
    for wr in window_results:

        def _fmt_pct(v: float | None) -> str:
            return f"{v * 100:.2f}" if v is not None else "—"

        def _fmt_f(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "—"

        cells = [
            str(wr.window_size),
            str(wr.current_anomaly_count),
            str(wr.current_population),
            _fmt_pct(wr.current_rate),
            _fmt_pct(wr.prior_rate),
            _fmt_f(wr.absolute_change),
            _fmt_pct(wr.pct_change),
            "yes" if wr.low_volume else "no",
            "⚡ yes" if wr.calibration_break else "no",
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    return (
        "<h2>Rolling windows</h2>"
        "<table><thead>" + header_html + "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
