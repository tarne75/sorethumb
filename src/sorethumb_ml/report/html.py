"""Self-contained HTML report renderer.

IMPORTANT: html.escape() is the only escaping helper used in this module. It is
never rebound or aliased. Rewriting ``escape = html.escape`` and then calling
``escape(...)`` works on a clean top-to-bottom execution but fails if the module is
reloaded in the same process, which is the worst possible failure profile.

The generated file must work from a file:// URL with no internet connection. All
CSS and images are inline. External CSS, CDN links, and <script src="..."> are
forbidden.

index.html / index.json are written atomically (sibling temp file + rename,
see sorethumb_ml._atomic) so a browser or another process never opens a
half-written report.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from sorethumb_ml._atomic import atomic_write_text

logger = logging.getLogger(__name__)

# Shown under every rolling-window trend table. Each period is fitted and self-calibrated on
# its own, so the trend is a relative diagnostic; reading it as an absolute level would be a
# misreading.
_TREND_NOTE = (
    "Each period is fitted and self-calibrated independently, so this trend shows relative "
    "movement, not an absolute anomaly level. For one fixed scale, score later data against an "
    "accepted reference run with <code>sorethumb score --from-run</code>."
)

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
    # The fitted run this run reused persisted models/plan from (score-forward
    # only) -- None for an ordinary run that fit its own detectors.
    source_run_id: str | None = None
    # What "unique" meant for columns.id_column when this run's identity
    # contract was validated ("dataset" or "group" -- see
    # _pipeline._resolve_id_identity_scope) -- None if no id_column is
    # configured.
    id_identity_scope: str | None = None


@dataclass
class GroupSection:
    """All data needed to render one group section of the report."""

    group_key: str
    group_label: str
    records: pl.DataFrame
    plan_dropped: list[dict[str, str]] = field(default_factory=list)
    contrast: pl.DataFrame | None = None
    window_results: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # attributions.parquet: the record behind each reason (None when the run
    # has no explanations, or predates them).
    attributions: pl.DataFrame | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_report(
    run_meta: RunMeta,
    groups: list[GroupSection],
    out_dir: Path,
    formats: list[str] | tuple[str, ...] = ("html", "csv"),
    *,
    csv_bom: bool = False,
) -> Path:
    """Write the report artefacts named in *formats* to *out_dir*.

    ``report.formats`` selects any of ``html`` (``index.html``), ``csv`` (one
    ``<group_key>.csv`` per group, which the HTML links to relatively) and
    ``json`` (``index.json`` — provenance + per-group records). Returns
    ``index.html`` when it was written, otherwise ``index.json``, otherwise
    *out_dir*.
    """
    from sorethumb_ml.report.csv import write_group_csv  # noqa: PLC0415

    fmts = {f.lower() for f in formats}
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSVs first (so links in HTML are valid as soon as the file appears)
    if "csv" in fmts:
        for grp in groups:
            write_group_csv(grp.records, out_dir, grp.group_key, include_bom=csv_bom)

    json_path: Path | None = None
    if "json" in fmts:
        json_path = _write_json_report(run_meta, groups, out_dir)

    if "html" not in fmts:
        logger.info("Report written (formats=%s): %s.", sorted(fmts), out_dir)
        return json_path or out_dir

    body_parts: list[str] = [_provenance_block(run_meta)]

    # Tab nav
    body_parts.append(_tab_nav(groups))

    csv_written = "csv" in fmts
    for i, grp in enumerate(groups):
        body_parts.append(_group_section(grp, i, csv_written=csv_written))

    html_content = _page(run_meta.run_id, "\n".join(body_parts))
    out_path = out_dir / "index.html"
    atomic_write_text(out_path, html_content)
    logger.info("HTML report written: %s.", out_path)
    return out_path


def _write_json_report(run_meta: RunMeta, groups: list[GroupSection], out_dir: Path) -> Path:
    """Write ``index.json``: the provenance block plus each group's flagged records."""
    payload = {
        "run_id": run_meta.run_id,
        "dataset_uri": run_meta.dataset_uri,
        "dataset_fp": run_meta.dataset_fp,
        "config_hash": run_meta.config_hash,
        "seed": run_meta.seed,
        "library_version": run_meta.library_version,
        "python_version": run_meta.python_version,
        "started_at": run_meta.started_at,
        "source_run_id": run_meta.source_run_id,
        "id_identity_scope": run_meta.id_identity_scope,
        "groups": [
            {
                "group_key": g.group_key,
                "group_label": g.group_label,
                "n_records": len(g.records),
                "records": g.records.to_dicts(),
                "warnings": g.warnings,
                **({"attributions": g.attributions.to_dicts()} if g.attributions is not None else {}),
            }
            for g in groups
        ],
    }
    path = out_dir / "index.json"
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))
    logger.info("JSON report written: %s.", path)
    return path


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
.csv-link { font-size: 0.82rem; }
.reason-detail details { margin: 0.25rem 0; }
.reason-detail summary { cursor: pointer; font-size: 0.85rem; }
.reason-detail .meta { color: #666; font-size: 0.78rem; }
.reason-detail .pos { color: #a33; }
.reason-detail .neg { color: #276; }
.reason-detail dt { font-family: monospace; font-size: 0.8rem; margin-top: 0.3rem; }
.reason-detail dd { font-size: 0.8rem; margin-left: 1rem; }
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
        f"{'&nbsp;&nbsp;<strong>Scored forward from:</strong> ' + html.escape(meta.source_run_id) if meta.source_run_id else ''}"
        f"{'&nbsp;&nbsp;<strong>id_column unique per:</strong> ' + html.escape(meta.id_identity_scope) if meta.id_identity_scope else ''}"
        "<br><details><summary>Resolved config</summary>"
        f"<pre>{html.escape(config_pretty)}</pre>"
        "</details>"
        "</div>"
    )


def _warnings_callout(warnings: list[str]) -> str:
    """Render a group's persisted warnings as a visible callout above its tabs.

    E.g. ZeroAnomalyWarning for an empty intersection -- otherwise easy to
    miss in a report viewed long after the run finished.
    """
    if not warnings:
        return ""
    items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    return (
        '<div class="warnings-callout" '
        'style="border-left:4px solid #c77c00;background:#fff8ec;padding:0.5rem 1rem;margin:0.5rem 0;">'
        '<strong style="color:#c77c00;">Warnings</strong>'
        f'<ul style="margin:0.25rem 0 0 0;">{items}</ul>'
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
            f"{_warnings_callout(grp.warnings)}"
            f'<div class="tabs">'
            f'<button class="tab-btn active" data-group="{i}" data-tab="records" '
            f"onclick=\"showTab({i}, 'records')\">Records</button>"
            f'<button class="tab-btn" data-group="{i}" data-tab="contrast" '
            f"onclick=\"showTab({i}, 'contrast')\">Contrast</button>"
            f'<button class="tab-btn" data-group="{i}" data-tab="plan" '
            f"onclick=\"showTab({i}, 'plan')\">Feature plan</button>"
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _group_section(grp: GroupSection, idx: int, *, csv_written: bool) -> str:
    parts: list[str] = []

    # Records tab. The CSV download link is only rendered when a CSV was
    # actually written for this report (report.formats included "csv") --
    # otherwise it links to a sibling file that was never written.
    csv_link = (
        f'<p class="csv-link"><a href="./{html.escape(grp.group_key)}.csv">Download CSV</a> '
        "(moving this HTML without its sibling CSVs breaks this link)</p>"
        if csv_written
        else ""
    )
    records_html = _df_to_table(grp.records) if len(grp.records) > 0 else "<p>No anomalies flagged.</p>"
    records_html += _reason_detail(grp)
    parts.append(
        f'<div class="tab-panel active" data-group="{idx}" data-tab="records">{csv_link}{records_html}</div>'
    )

    # Windows table
    if grp.window_results:
        windows_html = _windows_table(grp.window_results) + f"<p><small>{_TREND_NOTE}</small></p>"
        parts[-1] = parts[-1].replace("</div>", windows_html + "</div>")

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


def _fmt_signed(v: Any) -> str:
    if v is None:
        return ""
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{html.escape(f"{v:+.4g}")}</span>'


def _cell(v: Any) -> str:
    return html.escape(str(v)) if v is not None else "—"


def _reason_detail(grp: GroupSection) -> str:
    """Render the record behind each reason, one collapsed block per flagged row, most anomalous first.

    Empty when the group has no attribution records, so a run without
    explanations renders exactly as before.
    """
    from sorethumb_ml.explain.record import APPROXIMATION_FLAGS  # noqa: PLC0415

    att = grp.attributions
    if att is None or len(att) == 0 or "row_id" not in grp.records.columns:
        return ""
    by_row: dict[Any, list[dict[str, Any]]] = {}
    for rec in att.sort(["row_id", "reason_rank"]).iter_rows(named=True):
        by_row.setdefault(rec["row_id"], []).append(rec)
    rows = grp.records.sort("rank") if "rank" in grp.records.columns else grp.records
    first = att.row(0, named=True)
    parts = [
        '<div class="reason-detail"><h2>Explanation detail</h2>',
        '<p class="meta">What produced each reason: the field\'s value against the reference '
        "population's typical value, its signed contribution (positive pushes toward anomaly), "
        "each detector's method and share, and every caveat that applies. "
        f"Explainer {html.escape(str(first['explainer_version']))}, "
        f"reference {html.escape(str(first['reference_id']))}.</p>",
    ]
    used_flags: set[str] = set()
    for row in rows.iter_rows(named=True):
        rid = row.get("row_id")
        status = row.get("explain_status")
        label = f"#{row.get('rank', '')} · row {rid}"
        if rid not in by_row:
            if status in (None, "ok", "partial"):
                continue
            issue = row.get("explain_issue")
            parts.append(
                f'<p class="meta">{html.escape(label)}: no explanation'
                f"{' (' + html.escape(str(issue)) + ')' if issue else ''}.</p>"
            )
            continue
        recs = by_row[rid]
        summary = " · ".join(
            [html.escape(label), html.escape(str(row.get("reason_1") or "")), html.escape(str(status or ""))]
        )
        body = [
            "<table><thead><tr>"
            + "".join(
                f"<th>{h}</th>"
                for h in (
                    "#",
                    "Field",
                    "Value",
                    "Reference",
                    "Direction",
                    "Signed",
                    "Magnitude",
                    "Methods (weight, share)",
                    "Transformed features",
                    "Flags",
                )
            )
            + "</tr></thead><tbody>"
        ]
        for rec in recs:
            methods = "<br>".join(
                f"{html.escape(str(m['detector']))}: {html.escape(str(m['method']))} "
                f"({html.escape(str(m['attribution_kind']))}) w={m['weight']:.2f}, {_fmt_signed(m['signed_contribution'])}"
                for m in rec.get("methods") or []
            )
            feats = "<br>".join(
                f"{html.escape(str(f['feature']))} {_fmt_signed(f['signed_contribution'])}"
                for f in rec.get("transformed_features") or []
            )
            flags = rec.get("approximation_flags") or []
            used_flags.update(flags)
            body.append(
                "<tr>"
                f"<td>{rec['reason_rank']}</td>"
                f"<td>{_cell(rec['original_field'])}</td>"
                f"<td>{_cell(rec['raw_value'])}</td>"
                f"<td>{_cell(rec['reference_value'])}</td>"
                f"<td>{_cell(rec['direction'])}</td>"
                f"<td>{_fmt_signed(rec['signed_contribution'])}</td>"
                f"<td>{html.escape(format(rec['contribution_magnitude'], '.4g'))}</td>"
                f"<td>{methods or '—'}</td>"
                f"<td>{feats or '—'}</td>"
                f"<td>{'<br>'.join(html.escape(str(f)) for f in flags) or '—'}</td>"
                "</tr>"
            )
        body.append("</tbody></table>")
        parts.append(f"<details><summary>{summary}</summary>{''.join(body)}</details>")
    if used_flags:
        keys = sorted({f.split(":", 1)[0] for f in used_flags})
        items = []
        for key in keys:
            doc_key = next((k for k in APPROXIMATION_FLAGS if k.split(":", 1)[0] == key), None)
            if doc_key is not None:
                items.append(
                    f"<dt>{html.escape(doc_key)}</dt><dd>{html.escape(APPROXIMATION_FLAGS[doc_key])}</dd>"
                )
        if items:
            parts.append(
                f"<details><summary>What the flags mean</summary><dl>{''.join(items)}</dl></details>"
            )
    parts.append("</div>")
    return "".join(parts)


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
