"""Event marker helpers for Plotly market charts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ui_text import localize_dataframe_columns


EVENT_KIND_ALIASES = {
    "policy": "policy",
    "policy_event": "policy",
    "government_policy": "policy",
    "major_news": "news",
    "news": "news",
    "rumor": "news",
    "refute": "news",
    "macro_shock": "news",
    "regulator": "regulator",
    "regulatory_action": "regulator",
    "regulation": "regulator",
    "intervention": "regulator",
}

EVENT_STYLES = {
    "policy": {"label": "政策事件", "color": "#38bdf8", "symbol": "diamond"},
    "news": {"label": "新闻事件", "color": "#f59e0b", "symbol": "circle"},
    "regulator": {"label": "监管动作", "color": "#a78bfa", "symbol": "triangle-up"},
}


def _event_kind(item: Mapping[str, Any]) -> str:
    for key in ("kind", "event_type", "type", "category", "source_type"):
        raw = str(item.get(key, "") or "").strip().lower()
        if raw in EVENT_KIND_ALIASES:
            return EVENT_KIND_ALIASES[raw]
    title = str(item.get("title", item.get("event", "")) or "").lower()
    if any(token in title for token in ("监管", "干预", "intervention", "regulator")):
        return "regulator"
    if any(token in title for token in ("新闻", "传闻", "谣言", "news", "rumor")):
        return "news"
    return "policy"


def _event_x_value(item: Mapping[str, Any], time_values: Optional[Sequence[Any]] = None) -> Any:
    for key in ("x", "time", "date", "timestamp", "trading_day", "datetime"):
        value = item.get(key)
        if value not in (None, ""):
            return value

    for key in ("effective_day", "step", "tick", "day"):
        if key not in item:
            continue
        try:
            numeric = int(float(item[key]))
        except Exception:
            continue
        if time_values:
            index = min(max(numeric - 1, 0), len(time_values) - 1)
            return time_values[index]
        return numeric
    return None


def normalize_event_markers(
    events: Optional[Iterable[Mapping[str, Any]]],
    *,
    time_values: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Normalize policy/news/regulator events into chart-ready rows."""

    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(events or []):
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        kind = _event_kind(item)
        x_value = _event_x_value(item, time_values=time_values)
        if x_value is None:
            continue
        rows.append(
            {
                "event_id": str(item.get("event_id", item.get("id", f"event_{index}"))),
                "kind": kind,
                "kind_label": EVENT_STYLES[kind]["label"],
                "x": x_value,
                "label": str(item.get("title", item.get("event", item.get("policy_name", EVENT_STYLES[kind]["label"]))) or ""),
                "description": str(item.get("raw_text", item.get("description", item.get("policy_text", ""))) or ""),
                "strength": float(item.get("strength", item.get("current_strength", item.get("intensity", 1.0))) or 0.0),
                "source": str(item.get("source", "")),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "step": item.get("step", item.get("effective_day", item.get("tick", ""))),
            }
        )
    return pd.DataFrame(rows)


def event_marker_counts(events: Optional[Iterable[Mapping[str, Any]]]) -> Dict[str, int]:
    frame = normalize_event_markers(events)
    if frame.empty:
        return {"policy": 0, "news": 0, "regulator": 0}
    counts = frame["kind"].value_counts().to_dict()
    return {kind: int(counts.get(kind, 0)) for kind in ("policy", "news", "regulator")}


def add_event_marker_layer(
    fig: go.Figure,
    events: Optional[Iterable[Mapping[str, Any]]],
    *,
    time_values: Optional[Sequence[Any]] = None,
    y_lookup: Optional[Mapping[Any, float]] = None,
    y_default: Optional[float] = None,
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    """Add policy, news, and regulator markers to an existing Plotly figure."""

    marker_frame = normalize_event_markers(events, time_values=time_values)
    if marker_frame.empty:
        return fig

    fallback_y = float(y_default if y_default is not None else 0.0)
    for kind, style in EVENT_STYLES.items():
        subset = marker_frame[marker_frame["kind"] == kind]
        if subset.empty:
            continue
        y_values = [
            float(y_lookup.get(x_value, fallback_y) if y_lookup is not None else fallback_y)
            for x_value in subset["x"].tolist()
        ]
        hover_text = [
            "<br>".join(
                part
                for part in [
                    f"{style['label']}: {row_item['label']}",
                    f"强度: {float(row_item['strength']):.2f}",
                    str(row_item.get("description", ""))[:140],
                ]
                if str(part).strip()
            )
            for _, row_item in subset.iterrows()
        ]
        trace = go.Scatter(
            x=subset["x"],
            y=y_values,
            mode="markers",
            name=style["label"],
            marker=dict(
                color=style["color"],
                symbol=style["symbol"],
                size=11,
                line=dict(color="#0f172a", width=1),
            ),
            text=subset["label"],
            hovertext=hover_text,
            hoverinfo="text",
        )
        try:
            fig.add_trace(trace, row=row, col=col)
        except Exception:
            fig.add_trace(trace)
    return fig


def render_event_marker_legend(
    events: Optional[Iterable[Mapping[str, Any]]],
    *,
    key_prefix: str = "events",
) -> pd.DataFrame:
    marker_frame = normalize_event_markers(events)
    counts = event_marker_counts(events)
    cols = st.columns(3)
    for idx, kind in enumerate(("policy", "news", "regulator")):
        cols[idx].metric(EVENT_STYLES[kind]["label"], counts[kind])
    if marker_frame.empty:
        st.info("暂无可标记事件。")
        return marker_frame
    display_cols = ["kind_label", "x", "label", "strength", "source", "confidence"]
    st.dataframe(localize_dataframe_columns(marker_frame[display_cols]), use_container_width=True, hide_index=True)
    return marker_frame


__all__ = [
    "EVENT_KIND_ALIASES",
    "EVENT_STYLES",
    "add_event_marker_layer",
    "event_marker_counts",
    "normalize_event_markers",
    "render_event_marker_legend",
]
