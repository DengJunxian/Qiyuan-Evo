"""Shared Plotly theme helpers for the Streamlit workbench."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import plotly.graph_objects as go
from ui.appearance import is_dark


def _plotly_layout():
    dark = is_dark()
    foreground = "#f5f5f7" if dark else "#1d1d1f"
    return {
        "template": "plotly_dark" if dark else "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": foreground, "family": "-apple-system, BlinkMacSystemFont, PingFang SC, sans-serif"},
        "hoverlabel": {"bgcolor": "#182439" if dark else "#f5f5f7", "font": {"color": foreground}},
    }


class _SessionPlotlyLayout(Mapping):
    """Resolve legacy imports per session, without sharing mutable theme state."""
    def __getitem__(self, key):
        return _plotly_layout()[key]

    def __iter__(self):
        return iter(_plotly_layout())

    def __len__(self):
        return len(_plotly_layout())


PLOTLY_DARK_LAYOUT: Mapping[str, Any] = _SessionPlotlyLayout()


CHINESE_RANGE_SELECTOR = [
    {"count": 7, "label": "近 1 周", "step": "day", "stepmode": "backward"},
    {"count": 1, "label": "近 1 月", "step": "month", "stepmode": "backward"},
    {"step": "all", "label": "全部"},
]


def apply_dark_theme(fig: go.Figure, *, title: str | None = None, height: int | None = None, **layout: Any) -> go.Figure:
    """Apply the shared dark layout without mutating caller-owned defaults."""

    dark = is_dark()
    muted = "#a4b1c5" if dark else "#6e6e73"
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
        color=muted,
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.14)",
        linecolor="rgba(148,163,184,0.22)",
        color=muted,
        automargin=True,
    )
    return fig


def chinese_range_selector_axis(*, rangeslider_visible: bool = False) -> dict[str, Any]:
    return {
        "rangeslider": {"visible": rangeslider_visible},
        "rangeselector": {
            "buttons": CHINESE_RANGE_SELECTOR,
            "bgcolor": "#182439" if is_dark() else "#f0f0f2",
            "activecolor": "#436484" if is_dark() else "#cce4ff",
            "font": {"color": "#f5f5f7" if is_dark() else "#1d1d1f"},
        },
    }


__all__ = ["CHINESE_RANGE_SELECTOR", "PLOTLY_DARK_LAYOUT", "apply_dark_theme", "chinese_range_selector_axis"]
