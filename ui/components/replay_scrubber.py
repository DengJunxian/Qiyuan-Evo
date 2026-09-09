"""Replay scrubber for policy experiments and history windows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from core.ui_text import localize_dataframe_columns
from ui.components.event_marker_layer import normalize_event_markers


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _active_events_for_step(events: pd.DataFrame, step: int, time_value: Any) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    numeric_step = pd.to_numeric(out.get("step"), errors="coerce")
    by_step = numeric_step.notna() & (numeric_step.astype(float) <= float(step))
    by_x = out["x"].astype(str) == str(time_value)
    return out[by_step | by_x].tail(8)


def build_replay_timeline(
    market_frame: pd.DataFrame,
    *,
    events: Optional[Iterable[Mapping[str, Any]]] = None,
    trade_tape: Optional[Sequence[Mapping[str, Any]]] = None,
    reports: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build per-step replay snapshots with policy/news/order-flow context."""

    if market_frame is None or market_frame.empty:
        return []
    frame = market_frame.copy().reset_index(drop=True)
    if "step" not in frame.columns:
        frame["step"] = np.arange(1, len(frame) + 1)
    if "time" not in frame.columns:
        frame["time"] = frame.get("timestamp", frame["step"]).astype(str)
    marker_frame = normalize_event_markers(events, time_values=frame["time"].tolist())
    tape_rows = [dict(item) for item in list(trade_tape or []) if isinstance(item, Mapping)]
    report_rows = [dict(item) for item in list(reports or []) if isinstance(item, Mapping)]

    snapshots: List[Dict[str, Any]] = []
    for idx, row in frame.iterrows():
        step = int(_safe_float(row.get("step", idx + 1), idx + 1))
        time_value = row.get("time", row.get("timestamp", step))
        row_events = _active_events_for_step(marker_frame, step, time_value)
        close = _safe_float(row.get("close", row.get("price", 0.0)))
        open_price = _safe_float(row.get("open", close), close)
        volume = _safe_float(row.get("volume", 0.0))
        buy_volume = _safe_float(row.get("买入量", row.get("buy_volume", volume * 0.52)))
        sell_volume = _safe_float(row.get("卖出量", row.get("sell_volume", max(volume - buy_volume, 0.0))))
        spread = _safe_float(row.get("spread", abs(close - open_price) / max(close, 1.0)))
        depth_imbalance = _safe_float(
            row.get("depth_imbalance", (buy_volume - sell_volume) / max(buy_volume + sell_volume, 1.0))
        )
        latest_report = report_rows[min(idx, len(report_rows) - 1)] if report_rows else {}
        matching = dict(latest_report.get("matching_result", {}) or {})
        belief = {
            "panic_level": _safe_float(row.get("panic_level", row.get("panic", 0.0))),
            "csad": _safe_float(row.get("csad", 0.0)),
            "sentiment_index": _safe_float(row.get("sentiment_index", 1.0 - row.get("panic_level", 0.0))),
            "agent_comment": str(latest_report.get("agent_comment", latest_report.get("phase", "")) or ""),
        }
        matching_trade_count = int(_safe_float(matching.get("trade_count", row.get("trade_count", max(volume / 1000.0, 0.0)))))
        current_tape = [
            item
            for item in tape_rows
            if int(_safe_float(item.get("tick", item.get("step", step)), step)) <= step
        ][-12:]
        snapshots.append(
            {
                "cursor": idx,
                "step": step,
                "time": str(time_value),
                "policy": row_events[row_events["kind"] == "policy"].to_dict(orient="records"),
                "news": row_events[row_events["kind"] == "news"].to_dict(orient="records"),
                "regulator": row_events[row_events["kind"] == "regulator"].to_dict(orient="records"),
                "agent_belief": belief,
                "order_flow": {
                    "buy_volume": float(buy_volume),
                    "sell_volume": float(sell_volume),
                    "imbalance": float(buy_volume - sell_volume),
                },
                "order_book": {
                    "best_bid": round(close * (1.0 - max(spread, 0.0001) * 0.5), 2),
                    "best_ask": round(close * (1.0 + max(spread, 0.0001) * 0.5), 2),
                    "spread": float(spread),
                    "depth_imbalance": float(depth_imbalance),
                },
                "trades": {
                    "trade_count": matching_trade_count,
                    "recent_trade_tape": current_tape,
                },
                "kline": row.to_dict(),
            }
        )
    return snapshots


def get_replay_snapshot(timeline: Sequence[Mapping[str, Any]], cursor: int) -> Dict[str, Any]:
    if not timeline:
        return {}
    idx = min(max(int(cursor), 0), len(timeline) - 1)
    return dict(timeline[idx])


def render_replay_scrubber(
    market_frame: pd.DataFrame,
    *,
    events: Optional[Iterable[Mapping[str, Any]]] = None,
    trade_tape: Optional[Sequence[Mapping[str, Any]]] = None,
    reports: Optional[Sequence[Mapping[str, Any]]] = None,
    key_prefix: str = "replay",
) -> Dict[str, Any]:
    timeline = build_replay_timeline(market_frame, events=events, trade_tape=trade_tape, reports=reports)
    if not timeline:
        st.info("暂无可回放数据。")
        return {}

    cursor_key = f"{key_prefix}_cursor"
    playing_key = f"{key_prefix}_playing"
    st.session_state.setdefault(cursor_key, 0)
    st.session_state.setdefault(playing_key, False)

    st.markdown("### 回放定位器")
    controls = st.columns([0.7, 0.7, 0.7, 3.2])
    if controls[0].button("暂停" if st.session_state[playing_key] else "播放", key=f"{key_prefix}_play", use_container_width=True):
        st.session_state[playing_key] = not bool(st.session_state[playing_key])
    if controls[1].button("后退", key=f"{key_prefix}_back", use_container_width=True):
        st.session_state[cursor_key] = max(0, int(st.session_state[cursor_key]) - 1)
        st.session_state[playing_key] = False
    if controls[2].button("前进", key=f"{key_prefix}_forward", use_container_width=True):
        st.session_state[cursor_key] = min(len(timeline) - 1, int(st.session_state[cursor_key]) + 1)
        st.session_state[playing_key] = False

    mode = controls[3].radio("回放粒度", ["交易日", "逐笔"], horizontal=True, key=f"{key_prefix}_mode")
    if st.session_state[playing_key]:
        st.session_state[cursor_key] = min(len(timeline) - 1, int(st.session_state[cursor_key]) + 1)
        if int(st.session_state[cursor_key]) >= len(timeline) - 1:
            st.session_state[playing_key] = False

    labels = [f"{item['step']} | {item['time']}" for item in timeline]
    selected = st.slider(
        f"按{mode}定位",
        min_value=0,
        max_value=len(timeline) - 1,
        value=min(int(st.session_state[cursor_key]), len(timeline) - 1),
        format="%d",
        key=f"{key_prefix}_slider",
    )
    st.session_state[cursor_key] = int(selected)
    snapshot = get_replay_snapshot(timeline, selected)
    st.caption(f"当前位置：{labels[selected]}")

    top = st.columns(4)
    top[0].metric("政策事件", len(snapshot.get("policy", []) or []))
    top[1].metric("新闻事件", len(snapshot.get("news", []) or []))
    top[2].metric("监管动作", len(snapshot.get("regulator", []) or []))
    top[3].metric("成交笔数", int(snapshot.get("trades", {}).get("trade_count", 0)))

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("#### 当时事件与信念")
        event_rows = [*snapshot.get("policy", []), *snapshot.get("news", []), *snapshot.get("regulator", [])]
        if event_rows:
            st.dataframe(localize_dataframe_columns(pd.DataFrame(event_rows)[["kind_label", "label", "strength"]]), use_container_width=True, hide_index=True)
        else:
            st.info("当前时点未命中新事件。")
        with st.expander("技术细节（可展开）", expanded=False):
            st.json(snapshot.get("agent_belief", {}), expanded=False)
    with detail_cols[1]:
        st.markdown("#### 订单流、盘口与成交")
        with st.expander("结构化数据载荷", expanded=False):
            st.json(
                {
                    "order_flow": snapshot.get("order_flow", {}),
                    "order_book": snapshot.get("order_book", {}),
                    "kline": snapshot.get("kline", {}),
                },
                expanded=False,
            )
    return snapshot


__all__ = ["build_replay_timeline", "get_replay_snapshot", "render_replay_scrubber"]
