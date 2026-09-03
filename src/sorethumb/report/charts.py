"""Trend chart rendering.

matplotlib is forced to the non-interactive Agg backend at import time.
All rendering is headless; nothing is displayed to a screen.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_WINDOW_ALPHAS = {1: 0.10, 7: 0.13, 14: 0.16, 28: 0.20}
_TOTAL_COLOR = "#1a1a2e"
_CAL_BREAK_COLOR = "#e63946"
_NON_BUSINESS_COLOR = "#f0f0f0"


def render_trend_chart(
    period_labels: list[str],
    group_anomaly_counts: dict[str, list[int]],
    period_population: list[int],
    windows: list[int],
    reference_label: str,
    cal_break_labels: set[str] | None = None,
    non_business_labels: set[str] | None = None,
) -> str:
    """Render a trend chart and return a base64-encoded PNG string.

    Parameters
    ----------
    period_labels:
        Ordered period labels for the x-axis.
    group_anomaly_counts:
        Mapping of group_key → list of anomaly counts, one per period.
    period_population:
        Total population per period (across all groups), used for the rate axis.
    windows:
        Window sizes to shade (e.g. [1, 7, 14, 28]).
    reference_label:
        The reference period; gets a vertical marker.
    cal_break_labels:
        Period labels where calibration mode changed; shown as red vertical lines.
    non_business_labels:
        Period labels that are non-business days; shaded grey.

    Returns
    -------
    Base64-encoded PNG string (no ``data:image/png;base64,`` prefix).

    """
    cal_breaks = cal_break_labels or set()
    non_business = non_business_labels or set()

    n = len(period_labels)
    x = list(range(n))
    label_to_x = {lbl: i for i, lbl in enumerate(period_labels)}

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # Shade non-business periods
    for lbl in non_business:
        xi = label_to_x.get(lbl)
        if xi is not None:
            ax1.axvspan(xi - 0.5, xi + 0.5, color=_NON_BUSINESS_COLOR, zorder=0)

    # Shade rolling-window bands (one shade per window centred on reference)
    ref_x = label_to_x.get(reference_label)
    if ref_x is not None:
        for w in sorted(windows, reverse=True):
            alpha = _WINDOW_ALPHAS.get(w, 0.10)
            ax1.axvspan(ref_x - w + 0.5, ref_x + 0.5, color="#4cc9f0", alpha=alpha, zorder=0)

    # Per-group thin lines
    total_counts = [0] * n
    for group_key, counts in group_anomaly_counts.items():
        padded = (counts + [0] * n)[:n]
        for i, v in enumerate(padded):
            total_counts[i] += v
        ax1.plot(x, padded, linewidth=0.8, alpha=0.5, label=group_key[:12])

    # Bold total line
    ax1.plot(x, total_counts, color=_TOTAL_COLOR, linewidth=2.0, label="Total", zorder=3)

    # Rate on secondary axis
    rates = []
    for i, pop in enumerate(period_population):
        rates.append(total_counts[i] / pop if pop > 0 else 0.0)
    ax2.plot(x, rates, color="#e07b39", linewidth=1.5, linestyle="--", label="Rate", zorder=2)

    # Reference marker
    if ref_x is not None:
        ax1.axvline(ref_x, color="#2d6a4f", linewidth=1.5, linestyle=":", zorder=4, label="Reference")

    # Calibration breaks — explicit red lines
    for lbl in cal_breaks:
        xi = label_to_x.get(lbl)
        if xi is not None:
            ax1.axvline(xi, color=_CAL_BREAK_COLOR, linewidth=1.5, zorder=4)
            ax1.text(xi + 0.1, ax1.get_ylim()[1] * 0.95, "⚡", fontsize=8, color=_CAL_BREAK_COLOR)

    # Axes labels and ticks
    step = max(1, n // 10)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(period_labels[::step], rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("Anomaly count")
    ax2.set_ylabel("Anomaly rate")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_title("Anomaly trend")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
