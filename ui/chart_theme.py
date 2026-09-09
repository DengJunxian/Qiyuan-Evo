"""Shared Plotly theme helpers for the Streamlit workbench."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


PLOTLY_DARK_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e2e8f0", "family": "Microsoft YaHei, SimHei, sans-serif"},
    "hoverlabel": {"font": {"family": "Microsoft YaHei, SimHei, sans-serif"}},
}


CHINESE_RANGE_SELECTOR = [
    {"count": 7, "label": "近 1 周", "step": "day", "stepmode": "backward"},
    {"count": 1, "label": "近 1 月", "step": "month", "stepmode": "backward"},
    {"step": "all", "label": "全部"},
]


def apply_dark_theme(fig: go.Figure, *, title: str | None = None, height: int | None = None, **layout: Any) -> go.Figure:
    """Apply the shared dark layout without mutating caller-owned defaults."""

    merged = dict(PLOTLY_DARK_LAYOUT)
    if title is not None:
        merged["title"] = title
    if height is not None:
        merged["height"] = height
    merged.update(layout)
    fig.update_layout(**merged)
    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(148,163,184,0.22)",
        color="#cbd5e1",
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.14)",
        linecolor="rgba(148,163,184,0.22)",
        color="#cbd5e1",
        automargin=True,
    )
    return fig


def chinese_range_selector_axis(*, rangeslider_visible: bool = False) -> dict[str, Any]:
    return {
        "rangeslider": {"visible": rangeslider_visible},
        "rangeselector": {
            "buttons": CHINESE_RANGE_SELECTOR,
            "bgcolor": "#0a1931",
            "activecolor": "#1890ff",
            "font": {"color": "#e2e8f0"},
        },
    }


__all__ = ["CHINESE_RANGE_SELECTOR", "PLOTLY_DARK_LAYOUT", "apply_dark_theme", "chinese_range_selector_axis"]
