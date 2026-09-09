"""Unified scorecard panel for replay and policy experiments."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import pandas as pd
import streamlit as st

from core.ui_text import localize_dataframe_columns


def mock_scorecard() -> Dict[str, Any]:
    return {
        "experiment_id": "mock_replay_scorecard",
        "benchmark_symbol": "sh000001",
        "seed": 42,
        "config_hash": "mock_config_hash",
        "data_snapshot_hash": "mock_snapshot_hash",
        "path_fit_metrics": {
            "direction_hit_rate": 0.62,
            "tracking_rmse": 0.041,
            "price_correlation": 0.73,
            "return_correlation": 0.48,
        },
        "risk_metrics": {
            "sim_volatility": 0.018,
            "real_volatility": 0.016,
            "volatility_gap": 0.002,
            "sim_max_drawdown": 0.082,
            "real_max_drawdown": 0.074,
            "max_drawdown_gap": 0.008,
        },
        "microstructure_metrics": {
            "spread": 0.0016,
            "depth_imbalance": 0.12,
            "trade_count": 320,
        },
        "behavioral_metrics": {
            "csad_mean": 0.082,
            "herd_intensity": 0.34,
            "sentiment_index": 0.58,
        },
        "regulatory_metrics": {
            "intervention_cost": 0.021,
            "fairness_compliance": 0.86,
            "market_confidence": 0.67,
        },
        "pass_fail_flags": {
            "direction_beats_random": True,
            "tracking_rmse_acceptable": True,
            "has_hashes": True,
        },
        "schema_version": "replay_scorecard_v2",
    }


def scorecard_summary_frames(scorecard: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    card = dict(scorecard or mock_scorecard())
    sections = []
    for section_name in (
        "path_fit_metrics",
        "risk_metrics",
        "microstructure_metrics",
        "behavioral_metrics",
        "regulatory_metrics",
    ):
        values = dict(card.get(section_name, {}) or {})
        for key, value in values.items():
            sections.append({"section": section_name, "metric": key, "value": value})
    flags = pd.DataFrame(
        [{"flag": key, "passed": bool(value)} for key, value in dict(card.get("pass_fail_flags", {}) or {}).items()]
    )
    return pd.DataFrame(sections), flags


def render_scorecard_panel(scorecard: Mapping[str, Any] | None = None, *, key_prefix: str = "scorecard") -> Dict[str, Any]:
    card = dict(scorecard or mock_scorecard())
    metrics_frame, flags_frame = scorecard_summary_frames(card)
    st.markdown("### 评估卡")
    path = dict(card.get("path_fit_metrics", {}) or {})
    risk = dict(card.get("risk_metrics", {}) or {})
    cols = st.columns(4)
    cols[0].metric("方向命中率", f"{float(path.get('direction_hit_rate', 0.0)):.2%}")
    cols[1].metric("路径跟踪误差", f"{float(path.get('tracking_rmse', path.get('normalized_rmse', 0.0))):.4f}")
    cols[2].metric("波动差", f"{float(risk.get('volatility_gap', 0.0)):.4f}")
    cols[3].metric("回撤差", f"{float(risk.get('max_drawdown_gap', 0.0)):.2%}")
    with st.expander("原始指标表（可展开）", expanded=False):
        st.dataframe(localize_dataframe_columns(metrics_frame), use_container_width=True, hide_index=True)
    if not flags_frame.empty:
        st.dataframe(localize_dataframe_columns(flags_frame), use_container_width=True, hide_index=True)
    return card


__all__ = ["mock_scorecard", "render_scorecard_panel", "scorecard_summary_frames"]
