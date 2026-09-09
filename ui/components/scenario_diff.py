"""Scenario comparison panel for research workbench."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ui_text import localize_dataframe_columns, zh_metric_name
from ui.chart_theme import PLOTLY_DARK_LAYOUT


SCENARIO_LABELS = {
    "baseline": "不介入基线",
    "policy_a": "提前介入",
    "policy_b": "延后介入",
    "optimized_policy": "推荐方案",
}


def _ensure_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
    if out.empty:
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        close = 3000.0 + np.cumsum(np.sin(np.arange(20) / 3.0) * 3.0 + 1.5)
        out = pd.DataFrame({"step": np.arange(1, 21), "time": dates.strftime("%Y-%m-%d"), "close": close})
    if "step" not in out.columns:
        out["step"] = np.arange(1, len(out) + 1)
    if "time" not in out.columns:
        out["time"] = out["step"].astype(str)
    if "close" not in out.columns:
        out["close"] = pd.to_numeric(out.get("price", 3000.0), errors="coerce").fillna(3000.0)
    for col in ("panic_level", "csad", "spread", "depth_imbalance", "sentiment_index"):
        if col not in out.columns:
            out[col] = 0.0
    return out.reset_index(drop=True)


def build_default_scenarios(base_frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base = _ensure_market_frame(base_frame)
    close = pd.to_numeric(base["close"], errors="coerce").ffill().bfill().fillna(3000.0)
    returns = close.pct_change().fillna(0.0)
    idx = np.arange(len(base), dtype=float)

    def variant(name: str, drift: float, shock: float, smooth: float) -> pd.DataFrame:
        out = base.copy()
        adjusted_returns = returns + drift + shock * np.exp(-idx / max(len(base) / 3.0, 1.0))
        if smooth > 0:
            adjusted_returns = adjusted_returns.ewm(alpha=smooth, adjust=False).mean()
        path = [float(close.iloc[0])]
        for value in adjusted_returns.iloc[1:]:
            path.append(max(1.0, path[-1] * (1.0 + float(value))))
        out["close"] = np.round(path, 2)
        out["scenario"] = name
        out["panic_level"] = np.clip(pd.to_numeric(out["panic_level"], errors="coerce").fillna(0.0) + abs(shock) * 12.0, 0.0, 1.0)
        out["csad"] = np.clip(pd.to_numeric(out["csad"], errors="coerce").fillna(0.0) + abs(shock) * 4.0, 0.0, 1.0)
        return out

    baseline = base.copy()
    baseline["scenario"] = "baseline"
    return {
        "baseline": baseline,
        "policy_a": variant("policy_a", 0.0005, 0.0008, 0.0),
        "policy_b": variant("policy_b", -0.0002, -0.0012, 0.0),
        "optimized_policy": variant("optimized_policy", 0.00035, 0.0003, 0.45),
    }


def _risk_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().bfill().fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    drawdown = close / close.cummax().replace(0.0, np.nan) - 1.0
    return {
        "return_pct": float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0) if len(close) else 0.0,
        "volatility": float(returns.std()) if len(returns) else 0.0,
        "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "panic_contribution": float(pd.to_numeric(frame.get("panic_level"), errors="coerce").fillna(0.0).mean()),
        "csad_contribution": float(pd.to_numeric(frame.get("csad"), errors="coerce").fillna(0.0).mean()),
        "spread_contribution": float(pd.to_numeric(frame.get("spread"), errors="coerce").fillna(0.0).mean()),
        "depth_contribution": float(abs(pd.to_numeric(frame.get("depth_imbalance"), errors="coerce").fillna(0.0)).mean()),
    }


def build_scenario_diff(
    scenarios: Optional[Mapping[str, pd.DataFrame]],
    *,
    base_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    scenario_frames = dict(scenarios or {})
    if not scenario_frames:
        scenario_frames = build_default_scenarios(base_frame if base_frame is not None else pd.DataFrame())

    normalized: Dict[str, pd.DataFrame] = {}
    for key in ("baseline", "policy_a", "policy_b", "optimized_policy"):
        frame = scenario_frames.get(key)
        if frame is not None:
            normalized[key] = _ensure_market_frame(frame)
    if "baseline" not in normalized:
        normalized.update(build_default_scenarios(base_frame if base_frame is not None else next(iter(normalized.values()), pd.DataFrame())))

    combined_rows = []
    metric_rows = []
    for key, frame in normalized.items():
        label = SCENARIO_LABELS.get(key, key)
        local = frame.copy()
        local["scenario_key"] = key
        local["scenario_label"] = label
        combined_rows.append(local)
        metric_rows.append({"scenario_key": key, "scenario_label": label, **_risk_metrics(local)})

    combined = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    baseline_metrics = metrics[metrics["scenario_key"] == "baseline"].head(1)
    deltas = metrics.copy()
    if not baseline_metrics.empty:
        base = baseline_metrics.iloc[0].to_dict()
        for col in [c for c in metrics.columns if c not in {"scenario_key", "scenario_label"}]:
            deltas[f"delta_{col}"] = pd.to_numeric(deltas[col], errors="coerce") - float(base.get(col, 0.0) or 0.0)
    risk_cols = ["panic_contribution", "csad_contribution", "spread_contribution", "depth_contribution"]
    risk = metrics[["scenario_label", *risk_cols]].copy() if set(risk_cols).issubset(metrics.columns) else pd.DataFrame()
    return {"combined": combined, "metrics": metrics, "deltas": deltas, "risk_contribution": risk}


def render_scenario_diff(
    scenarios: Optional[Mapping[str, pd.DataFrame]] = None,
    *,
    base_frame: Optional[pd.DataFrame] = None,
    key_prefix: str = "scenario_diff",
) -> Dict[str, pd.DataFrame]:
    data = build_scenario_diff(scenarios, base_frame=base_frame)
    combined = data["combined"]
    if combined.empty:
        st.info("暂无场景对比数据。")
        return data

    st.markdown("### 场景对比")
    fig = go.Figure()
    colors = {
        "baseline": "#94a3b8",
        "policy_a": "#22c55e",
        "policy_b": "#f97316",
        "optimized_policy": "#38bdf8",
    }
    for key, label in SCENARIO_LABELS.items():
        subset = combined[combined["scenario_key"] == key]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["time"],
                y=subset["close"],
                mode="lines",
                name=label,
                line=dict(color=colors.get(key, "#cbd5e1"), width=2.2 if key == "optimized_policy" else 1.8),
            )
        )
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        height=380,
        margin=dict(l=18, r=18, t=42, b=20),
        title="场景对比：不介入、提前介入、延后介入与推荐方案",
        yaxis_title="同刻度指数点位",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_line")
    with st.expander("原始指标表（可展开）", expanded=False):
        st.dataframe(localize_dataframe_columns(data["deltas"]), use_container_width=True, hide_index=True)
    if not data["risk_contribution"].empty:
        risk_long = data["risk_contribution"].melt(id_vars=["scenario_label"], var_name="risk_factor", value_name="value")
        risk_fig = go.Figure()
        for label, subset in risk_long.groupby("scenario_label"):
            risk_fig.add_trace(go.Bar(x=subset["risk_factor"].map(lambda value: zh_metric_name(str(value))), y=subset["value"], name=str(label)))
        risk_fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            barmode="group",
            height=320,
            margin=dict(l=18, r=18, t=36, b=20),
            title="风险贡献差异",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(risk_fig, use_container_width=True, key=f"{key_prefix}_risk")
    return data


__all__ = ["SCENARIO_LABELS", "build_default_scenarios", "build_scenario_diff", "render_scenario_diff"]
