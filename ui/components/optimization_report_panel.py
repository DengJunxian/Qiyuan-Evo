"""Optimization report panel for regulator outputs."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pandas as pd
import plotly.express as px
import streamlit as st

from core.ui_text import localize_dataframe_columns, zh_metric_name
from ui.chart_theme import PLOTLY_DARK_LAYOUT


def normalize_optimization_report(result: Mapping[str, Any] | None) -> Dict[str, pd.DataFrame]:
    payload = dict(result or {})
    blackbox = dict(payload.get("blackbox_optimization", {}) or {})
    opt_report = dict(payload.get("optimization_report", {}) or {})
    recommendation = dict(payload.get("recommendation", {}) or {})
    validation = dict(blackbox.get("validation", opt_report.get("validation", {})) or {})
    nsga = dict(blackbox.get("nsga_ii", {}) or {})
    pareto_rows = []
    for item in list(nsga.get("pareto_frontier", payload.get("pareto_frontier", [])) or []):
        if not isinstance(item, Mapping):
            continue
        metrics = dict(item.get("metrics", {}) or {})
        objectives = dict(item.get("objectives", {}) or {})
        pareto_rows.append({**dict(item), **metrics, **objectives})
    best = dict(opt_report.get("best_solution", {}) or recommendation.get("scorecard", {}) or {})
    evidence = list(recommendation.get("evidence_chain", []) or [])
    windows = list(validation.get("windows", []) or [])
    return {
        "best_solution": pd.DataFrame([best]) if best else pd.DataFrame(),
        "pareto": pd.DataFrame(pareto_rows),
        "validation_windows": pd.DataFrame([dict(item) for item in windows if isinstance(item, Mapping)]),
        "evidence": pd.DataFrame([dict(item) for item in evidence if isinstance(item, Mapping)]),
    }


def render_optimization_report_panel(
    result: Mapping[str, Any] | None,
    *,
    key_prefix: str = "optimization_report",
) -> Dict[str, pd.DataFrame]:
    frames = normalize_optimization_report(result)
    st.markdown("### 优化报告")
    if frames["best_solution"].empty and frames["pareto"].empty:
        st.info("暂无优化报告数据。")
        return frames
    if not frames["best_solution"].empty:
        st.markdown("#### 最优方案")
        st.dataframe(localize_dataframe_columns(frames["best_solution"]), use_container_width=True, hide_index=True)
    pareto = frames["pareto"]
    if not pareto.empty:
        metric_cols = st.columns(3)
        metric_cols[0].metric("帕累托点数", len(pareto))
        score_series = pd.to_numeric(pareto["score"], errors="coerce") if "score" in pareto.columns else pd.Series(dtype=float)
        metric_cols[1].metric("最高得分", f"{float(score_series.max() if not score_series.empty else 0.0):.4f}")
        metric_cols[2].metric("候选指标数", len(pareto.columns))
        x_col = "intervention_cost" if "intervention_cost" in pareto.columns else pareto.columns[0]
        y_col = "macro_stability" if "macro_stability" in pareto.columns else pareto.columns[min(1, len(pareto.columns) - 1)]
        fig = px.scatter(
            pareto,
            x=x_col,
            y=y_col,
            color="score" if "score" in pareto.columns else None,
            hover_data=[col for col in ("action_description", "action_signature", "avg_reward") if col in pareto.columns],
            title="优化帕累托 / 候选解分布",
        )
        fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            height=360,
            margin=dict(l=18, r=18, t=42, b=20),
            xaxis_title=zh_metric_name(x_col),
            yaxis_title=zh_metric_name(y_col),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_pareto")
        with st.expander("原始指标表（可展开）", expanded=False):
            st.dataframe(localize_dataframe_columns(pareto), use_container_width=True, hide_index=True)
    if not frames["validation_windows"].empty:
        st.markdown("#### 固定窗口验证")
        st.dataframe(localize_dataframe_columns(frames["validation_windows"]), use_container_width=True, hide_index=True)
    if not frames["evidence"].empty:
        st.markdown("#### 推荐证据链")
        st.dataframe(localize_dataframe_columns(frames["evidence"]), use_container_width=True, hide_index=True)
    return frames


__all__ = ["normalize_optimization_report", "render_optimization_report_panel"]
