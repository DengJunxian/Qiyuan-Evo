"""Policy lab page focused on government-facing policy experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.data.market_data_provider import MarketDataProvider, MarketDataQuery
from core.eval import build_replay_scorecard
from core.event_store import EventRecord, EventStore, EventType
from core.macro.government import GovernmentAgent, PolicyShock
from core.objective_discovery import ObjectiveDiscoveryEngine
from core.policy_session import PolicySession
from core.runtime_paths import resolve_runtime_path
from core.runtime_mode import RuntimeModeProfile, normalize_runtime_mode, resolve_runtime_mode_profile
from core.ui_text import (
    localize_dataframe_columns,
    translate_display_text,
    zh_action_name,
    zh_metric_name,
    zh_value,
    zh_world_name,
)
from policy.structured import PolicyPackage
from ui.components.replay_scrubber import render_replay_scrubber
from ui.components.repro_meta import (
    build_experiment_registry_entry,
    build_reproducibility_meta,
    render_experiment_registry,
    render_reproducibility_panel,
    stable_payload_hash,
)
from ui.components.scenario_diff import render_scenario_diff
from ui.components.scorecard_panel import mock_scorecard, render_scorecard_panel
from ui import dashboard as dashboard_workbench
from ui.chart_theme import PLOTLY_DARK_LAYOUT, apply_dark_theme
from ui.narrative import render_narrative_block
from ui.reporting import dataframe_export_bundle, official_report_meta, write_report_artifacts


POLICY_TYPE_OPTIONS = {
    "税制调整": "tax",
    "流动性投放": "liquidity",
    "财政刺激": "fiscal",
    "监管收紧": "tightening",
    "市场稳定": "stabilization",
    "自定义政策": "custom",
}

TEMPLATE_LIBRARY_PATH = Path("data") / "policy_templates.json"
POLICY_REPORT_DIR = resolve_runtime_path(Path("outputs") / "policy_reports", env_var="CIVITAS_POLICY_REPORT_DIR")
CONTROL_MODE_OPTIONS = [
    "不设置对照组",
    "无政策基线",
    "模板推荐对照组",
    "温和变体",
    "风险压力变体",
]
INDEX_BENCHMARK_OPTIONS = {
    "上证指数（000001）": "sh000001",
    "深证成指（399001）": "sz399001",
    "创业板指（399006）": "sz399006",
}
CHART_MODE_OPTIONS = ["日K", "分时"]
POLICY_SESSION_STATUS_LABELS = {
    "idle": "未开始",
    "running": "仿真中",
    "paused": "已暂停",
    "stopped": "已停止",
    "completed": "已完成",
}


@dataclass
class PolicyNarrativeCard:
    title: str
    summary: str
    bullets: List[str]
    tone: str = "neutral"


def _resolve_runtime_profile() -> RuntimeModeProfile:
    mode = normalize_runtime_mode(str(st.session_state.get("simulation_mode", "SMART")))
    st.session_state["simulation_mode"] = mode
    profile = resolve_runtime_mode_profile(mode)
    st.session_state["runtime_mode_profile"] = profile.to_dict()
    return profile


def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _default_template_library() -> List[Dict[str, Any]]:
    return [
        {
            "id": "stamp-tax-liquidity",
            "category": "市场稳定",
            "title": "下调印花税并配套流动性支持",
            "policy_type": "税制调整",
            "policy_text": "下调印花税，并配合流动性支持政策，稳定市场预期。",
            "policy_goal": "提升市场流动性，降低交易摩擦，稳定指数走势。",
            "suitable_departments": "财政、税务、证券监管、平准基金",
            "recommended_intensity": 1.1,
            "recommended_duration": 30,
            "default_rumor_noise": False,
            "control_label": "维持现有税率与流动性安排",
            "control_text": "不新增稳定市场措施，保持当前税制与流动性安排。",
        },
        {
            "id": "targeted-fiscal-demand",
            "category": "财政支持",
            "title": "面向重点行业的定向财政扩张",
            "policy_type": "财政刺激",
            "policy_text": "分阶段推出面向基建和先进制造业的定向财政支出计划。",
            "policy_goal": "在维护金融稳定的同时，稳定增长预期。",
            "suitable_departments": "财政、发改、工信、地方政府",
            "recommended_intensity": 1.0,
            "recommended_duration": 60,
            "default_rumor_noise": False,
            "control_label": "不实施定向财政扩张",
            "control_text": "保持当前财政取向不变，作为对照组。",
        },
        {
            "id": "rumor-refutation-stabilization",
            "category": "预期管理",
            "title": "辟谣澄清并同步发布稳市声明",
            "policy_type": "市场稳定",
            "policy_text": "发布官方澄清公告，辟谣市场传闻，并推出协同稳市沟通方案。",
            "policy_goal": "降低市场恐慌，抑制谣言驱动的抛压。",
            "suitable_departments": "监管机构、官方媒体、交易所、稳定基金",
            "recommended_intensity": 1.2,
            "recommended_duration": 20,
            "default_rumor_noise": True,
            "control_label": "不进行公开澄清",
            "control_text": "不发布官方回应，仅观察市场自然演化。",
        },
    ]


def _runtime_mode_text(runtime_profile: RuntimeModeProfile) -> str:
    mode_map = {
        "SMART": "标准推演模式",
        "DEEP": "高级推演模式",
        "LIVE": "实时联机模式",
        "DEMO": "场景推演模式",
    }
    mode = str(getattr(runtime_profile, "mode", "") or "").upper()
    label = mode_map.get(mode, mode or "未知模式")
    summary = str(getattr(runtime_profile, "summary", "") or "").strip()
    return f"{label}｜{summary}" if summary else label


def _sanitize_report_markdown_for_frontend(markdown_text: str) -> str:
    """Drop process-improvement sections to keep report content focused on policy evaluation."""
    text = str(markdown_text or "").replace("\r\n", "\n")
    if not text.strip():
        return "暂无报告正文。"

    hidden_keywords = ("项目改进", "改进过程", "优化过程", "迭代过程", "重构过程")
    cleaned_lines: List[str] = []
    skip_section = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        is_heading = line.startswith("#")
        if is_heading:
            if any(keyword in line for keyword in hidden_keywords):
                skip_section = True
                continue
            if skip_section:
                skip_section = False
        if skip_section:
            continue
        if any(keyword in line for keyword in hidden_keywords):
            continue
        cleaned_lines.append(raw_line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or "暂无报告正文。"


def _data_source_text(data_source: str) -> str:
    mapping = {
        "real_index": "真实指数数据",
        "deep_mode_simulation": "深度模式多智能体仿真",
        "synthetic_fallback": "仿真回退数据",
    }
    return mapping.get(str(data_source or ""), "仿真回退数据")


def _load_policy_templates() -> List[Dict[str, Any]]:
    if TEMPLATE_LIBRARY_PATH.exists():
        try:
            payload = json.loads(TEMPLATE_LIBRARY_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            pass
    return _default_template_library()


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _policy_feature_flags(enable_structured_parser: bool = True) -> Dict[str, bool]:
    return {
        "structured_policy_parser_v1": bool(enable_structured_parser),
        "policy_transmission_layers_v1": True,
        "policy_transmission_graph_v1": True,
    }


def _compile_policy_bundle(
    policy_text: str,
    intensity: float,
    *,
    policy_type_hint: Optional[str] = None,
    market_regime: Optional[str] = None,
    enable_structured_parser: bool = True,
) -> Tuple[PolicyShock, PolicyPackage]:
    gov = GovernmentAgent(feature_flags=_policy_feature_flags(enable_structured_parser))
    package = gov.compile_policy_package(
        policy_text,
        tick=1,
        policy_type_hint=policy_type_hint,
        intensity=float(intensity),
        market_regime=market_regime,
        snapshot_info={
            "policy_text_length": len(policy_text or ""),
            "policy_type_hint": policy_type_hint or "",
            "parser_mode": "structured" if enable_structured_parser else "legacy",
        },
    )
    shock = PolicyShock(policy_id=package.event.policy_id, policy_text=policy_text, **package.to_policy_shock_fields())
    shock.metadata = {
        "policy_event": package.event.to_dict() if hasattr(package.event, "to_dict") else {
            "policy_id": package.event.policy_id,
            "raw_text": package.event.raw_text,
            "policy_type": package.event.policy_type,
        },
        "policy_package": package.to_dict(),
        "reproducibility": {
            "seed": int(package.metadata.get("seed", 0)),
            "config_hash": str(package.metadata.get("config_hash", "")),
            "snapshot_info": dict(package.metadata.get("snapshot_info", {})),
        },
        "parser_mode": package.uncertainty.parser_mode,
        "feature_flags": dict(package.metadata.get("feature_flags", {})),
    }
    return shock, package


def _compile_scaled_shock(
    policy_text: str,
    intensity: float,
    *,
    enable_structured_parser: bool = True,
) -> PolicyShock:
    shock, _ = _compile_policy_bundle(policy_text, intensity, enable_structured_parser=enable_structured_parser)
    return shock


def _shock_score(shock: PolicyShock) -> float:
    return (
        shock.liquidity_injection * 1.3
        + shock.fiscal_stimulus_delta * 1.5
        - shock.policy_rate_delta * 60.0
        - shock.credit_spread_delta * 18.0
        - shock.stamp_tax_delta * 420.0
        + shock.sentiment_delta * 1.2
        + shock.rumor_shock * 1.6
    )


def _policy_anchor_date() -> pd.Timestamp:
    key = "policy_lab_open_date"
    if key not in st.session_state:
        st.session_state[key] = pd.Timestamp.now(tz="Asia/Shanghai").normalize().strftime("%Y-%m-%d")
    return pd.to_datetime(st.session_state[key], errors="coerce").normalize()


def _load_real_index_history(index_symbol: str, lookback_days: int, end_date: pd.Timestamp) -> pd.DataFrame:
    start_date = (pd.Timestamp(end_date).normalize() - pd.Timedelta(days=max(int(lookback_days) * 3, 120))).strftime("%Y-%m-%d")
    query = MarketDataQuery(
        symbol=str(index_symbol),
        interval="1d",
        start=start_date,
        end=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        period_days=max(int(lookback_days), 30),
        adjust="",
        market="CN",
    )
    try:
        provider = MarketDataProvider()
        frame = provider.get_ohlcv(query, use_cache=True, freeze_snapshot=False)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return frame
    out = frame.sort_values("datetime").tail(max(int(lookback_days), 10)).copy()
    out["time"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["step"] = np.arange(1, len(out) + 1)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    if out.empty:
        return out
    if out["volume"].isna().all():
        out["volume"] = 1_000_000.0
    out["volume"] = out["volume"].fillna(out["volume"].median())
    return out[["step", "time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _build_policy_lab_agents(runtime_profile: RuntimeModeProfile) -> List[Any]:
    from core.simulation_factory import build_policy_agents
    return build_policy_agents(runtime_profile)


def _generate_policy_metrics(
    *,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    scenario_key: str,
    market_history: Optional[pd.DataFrame] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    shock = _compile_scaled_shock(policy_text, intensity)
    score = _shock_score(shock)
    seed = f"{scenario_key}|{policy_text}|{intensity}|{duration_days}|{rumor_noise}"
    rng = np.random.default_rng(_seed_from_text(seed))

    periods = max(10, int(duration_days))
    rows: List[Dict[str, float | int | str]] = []
    history = market_history.copy() if isinstance(market_history, pd.DataFrame) and not market_history.empty else pd.DataFrame()
    if not history.empty:
        history = history.tail(periods).reset_index(drop=True)
        base_returns = history["close"].pct_change().fillna(0.0).astype(float)
        price = float(history.iloc[0]["close"])
        for idx, row in history.iterrows():
            step = idx + 1
            base_ret = float(base_returns.iloc[idx])
            drift = 0.0002 + np.clip(score, -1.0, 1.0) * 0.0020 * np.exp(-(step - 1) / max(periods * 0.6, 1.0))
            rumor_term = (shock.rumor_shock * 0.004 if rumor_noise else 0.0) * np.exp(-(step - 1) / max(periods * 0.25, 1.0))
            ret = base_ret + drift + rumor_term + rng.normal(0.0, 0.0015)
            prev = price
            price = max(1600.0, prev * (1.0 + ret))
            panic = float(np.clip(0.18 + max(0.0, -ret) * 7.5 + max(0.0, rumor_term) * 6.0, 0.05, 0.95))
            csad = float(np.clip(0.05 + panic * 0.09 + abs(ret) * 4.0, 0.04, 0.22))
            base_high = float(row.get("high", max(prev, price)))
            base_low = float(row.get("low", min(prev, price)))
            high = max(base_high, prev, price) * (1 + abs(rng.normal(0.0, 0.0012)))
            low = min(base_low, prev, price) * (1 - abs(rng.normal(0.0, 0.0012)))
            base_volume = float(row.get("volume", 1_000_000.0) or 1_000_000.0)
            volume = float(base_volume * (1 + 0.25 * abs(score) + 0.35 * panic))
            rows.append(
                {
                    "step": step,
                    "time": str(row.get("time", "")),
                    "open": round(prev, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(price, 2),
                    "volume": round(volume, 2),
                    "csad": round(csad, 4),
                    "panic_level": round(panic, 4),
                }
            )
        return pd.DataFrame(rows)

    end_anchor = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_anchor, periods=periods)
    price = 3000.0
    for idx, dt in enumerate(dates, start=1):
        drift = 0.0003 + np.clip(score, -1.0, 1.0) * 0.0025 * np.exp(-(idx - 1) / max(periods * 0.5, 1.0))
        rumor_term = (shock.rumor_shock * 0.008 if rumor_noise else 0.0) * np.exp(-(idx - 1) / max(periods * 0.25, 1.0))
        ret = drift + rumor_term + rng.normal(0.0, 0.004)
        prev = price
        price = max(1600.0, prev * (1.0 + ret))
        band = abs(ret) * 0.9 + 0.001
        high = max(prev, price) * (1 + band)
        low = min(prev, price) * (1 - band)
        panic = float(np.clip(0.2 + max(0.0, -ret) * 8.0 + max(0.0, rumor_term) * 6.0, 0.05, 0.95))
        csad = float(np.clip(0.05 + panic * 0.1 + abs(ret) * 4.5, 0.04, 0.22))
        volume = float(1_000_000 * (1 + 0.3 * abs(score) + 0.4 * panic))
        rows.append(
            {
                "step": idx,
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(prev, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": round(volume, 2),
                "csad": round(csad, 4),
                "panic_level": round(panic, 4),
            }
        )
    return pd.DataFrame(rows)


def _run_policy_committee_review(policy_text: str, runtime_profile: RuntimeModeProfile) -> Dict[str, Any]:
    if not runtime_profile.enable_policy_committee or not runtime_profile.use_live_api:
        return {}
    try:
        from config import GLOBAL_CONFIG
        from core.policy_committee import PolicyCommittee
    except Exception as exc:
        return {"error": f"committee_import_error: {exc}"}

    if not (GLOBAL_CONFIG.DEEPSEEK_API_KEY or GLOBAL_CONFIG.ZHIPU_API_KEY):
        return {"error": "committee_skipped_no_api_key"}

    try:
        committee = PolicyCommittee(api_key=GLOBAL_CONFIG.DEEPSEEK_API_KEY)
        result = committee.interpret(policy_text)
        return {
            "parameters": dict(result.parameters or {}),
            "compliance_passed": bool(result.compliance.passed),
            "violations": list(result.compliance.violations or []),
            "warnings": list(result.compliance.warnings or []),
            "reasoning_chain": list(result.reasoning_chain or []),
            "final_state": dict(result.final_state or {}),
        }
    except Exception as exc:
        return {"error": f"committee_runtime_error: {exc}"}


def _generate_policy_metrics_deep(
    *,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    runtime_profile: RuntimeModeProfile,
    end_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from engine.simulation_loop import MarketEnvironment

    symbol = "A_SHARE_IDX"
    agents = _build_policy_lab_agents(runtime_profile)

    env = MarketEnvironment(
        agents,
        use_isolated_matching=True,
        market_pipeline_v2=bool(runtime_profile.market_pipeline_v2),
        runner_symbol=symbol,
        simulation_mode=runtime_profile.mode,
        llm_primary=bool(runtime_profile.llm_primary),
        deep_reasoning_pause_s=float(runtime_profile.pause_for_llm_seconds),
        model_priority=list(runtime_profile.model_priority),
        enable_policy_committee=bool(runtime_profile.enable_policy_committee),
    )

    committee_report = _run_policy_committee_review(policy_text, runtime_profile)
    policy_payload = f"{policy_text} [policy_intensity={float(intensity):.2f}]"
    env.schedule_policy_shock(policy_payload)
    if float(intensity) >= 1.1:
        env.schedule_policy_shock("政策力度超预期，资金对冲与追价行为同步放大。")
    if rumor_noise:
        env.schedule_policy_shock("谣言扩散导致恐慌交易升温，监管机构发布澄清提示。")

    steps = max(8, min(int(max(6.0, duration_days * max(0.6, min(1.2, float(intensity))))), 18))
    anchor = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=anchor, periods=steps)
    rows: List[Dict[str, Any]] = []
    last_close = float(env.current_price)
    latest_report: Dict[str, Any] = {}
    try:
        for idx in range(steps):
            latest_report = dict(_run_async(env.simulation_step()) or {})
            close = float(latest_report.get("new_price", last_close) or last_close)
            open_price = float(latest_report.get("old_price", last_close) or last_close)
            spread_proxy = abs(float(latest_report.get("price_change_pct", 0.0) or 0.0)) / 100.0
            high = max(open_price, close) * (1.0 + 0.001 + spread_proxy * 0.35)
            low = min(open_price, close) * (1.0 - 0.001 - spread_proxy * 0.35)
            volume = float(latest_report.get("buy_volume", 0.0) or 0.0) + float(
                latest_report.get("sell_volume", 0.0) or 0.0
            )
            macro_state = dict(latest_report.get("macro_state", {}) or {})
            sentiment = float(macro_state.get("sentiment_index", 0.5) or 0.5)
            panic = float(np.clip(1.0 - sentiment + spread_proxy * 5.0, 0.05, 0.95))
            diagnostics = dict(latest_report.get("behavioral_diagnostics", {}) or {})
            csad = float(diagnostics.get("csad", 0.05) or 0.05)
            rows.append(
                {
                    "step": idx + 1,
                    "time": dates[idx].strftime("%Y-%m-%d"),
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(max(0.01, low), 2),
                    "close": round(close, 2),
                    "volume": round(max(volume, 1.0), 2),
                    "csad": round(max(csad, 0.0), 4),
                    "panic_level": round(panic, 4),
                }
            )
            last_close = close
    finally:
        env.close()

    thinking_stats = dict(latest_report.get("thinking_stats", {}) or {})
    if not thinking_stats:
        thinking_stats = {
            "fast_count": int(sum(int(getattr(agent, "fast_think_count", 0) or 0) for agent in agents)),
            "slow_count": int(sum(int(getattr(agent, "slow_think_count", 0) or 0) for agent in agents)),
        }
    deep_meta: Dict[str, Any] = {
        "mode_profile": runtime_profile.to_dict(),
        "llm_agent_count": int(sum(1 for agent in agents if bool(getattr(agent, "use_llm", False)))),
        "agent_count": len(agents),
        "thinking_stats": thinking_stats,
        "committee_report": committee_report,
        "latest_step_report": latest_report,
    }
    baseline = pd.DataFrame(rows)
    deep_meta["counterfactual_regulation"] = _build_regulation_counterfactual_worlds(
        baseline,
        intensity=float(intensity),
    )
    return baseline, deep_meta


def _apply_regulatory_intervention_worldline(
    frame: pd.DataFrame,
    *,
    intervention_step: int,
    intensity: float,
    world_name: str = "counterfactual",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy().reset_index(drop=True)
    numeric_columns = ("open", "high", "low", "close", "panic_level", "csad", "volume")
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].ffill().bfill().fillna(0.0).astype(float)
    for col in ("panic_level", "csad", "volume"):
        if col in out.columns:
            out[col] = out[col].fillna(0.0).astype(float)

    seed_input = f"{world_name}|{len(out)}|{intervention_step}|{float(intensity):.4f}"
    seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    prev_close = float(out.iloc[0]["open"])
    for idx, row in out.iterrows():
        step = int(row["step"])
        close = float(row["close"])
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        panic = float(row["panic_level"])
        csad = float(row["csad"])
        volume = float(row["volume"])
        if step >= intervention_step:
            phase = step - intervention_step
            activation = 1.0 - np.exp(-phase / 2.2)
            decay = np.exp(-phase / 8.5)
            policy_wave = np.exp(-((phase - 2.0) ** 2) / 7.0)
            residual_noise = float(rng.normal(0.0, 0.0006))
            support = float(intensity) * (0.0014 * activation + 0.0020 * policy_wave + 0.0010 * decay)
            close *= 1.0 + support + residual_noise
            panic *= 1.0 - (0.26 * activation * decay)
            csad *= 1.0 - (0.19 * activation * decay)
            volume *= 1.0 + (0.08 * activation) + float(rng.normal(0.0, 0.01))
        open_price = prev_close
        high = max(high, open_price, close) * 1.0015
        low = min(low, open_price, close) * 0.9985
        out.at[idx, "open"] = round(open_price, 2)
        out.at[idx, "high"] = round(max(high, close), 2)
        out.at[idx, "low"] = round(max(0.01, min(low, close)), 2)
        out.at[idx, "close"] = round(close, 2)
        out.at[idx, "panic_level"] = round(float(np.clip(panic, 0.02, 0.98)), 4)
        out.at[idx, "csad"] = round(float(max(csad, 0.0)), 4)
        out.at[idx, "volume"] = round(float(max(volume, 1.0)), 2)
        prev_close = float(out.at[idx, "close"])
    return out


def _worldline_scorecard(frame: pd.DataFrame) -> Dict[str, float]:
    summary = _compute_policy_summary(frame)
    return {
        "return_pct": float(summary["return_pct"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "avg_panic": float(summary["avg_panic"]),
        "max_panic": float(summary["max_panic"]),
        "volatility": float(summary["volatility"]),
    }


def _build_regulation_counterfactual_worlds(frame: pd.DataFrame, *, intensity: float) -> Dict[str, Any]:
    if frame.empty:
        return {}
    steps = len(frame)
    early_step = max(2, int(round(steps * 0.25)))
    late_step = max(3, int(round(steps * 0.70)))
    no_intervention = frame.copy()
    early = _apply_regulatory_intervention_worldline(
        frame,
        intervention_step=early_step,
        intensity=float(intensity),
        world_name="early_intervention",
    )
    late = _apply_regulatory_intervention_worldline(
        frame,
        intervention_step=late_step,
        intensity=float(intensity),
        world_name="late_intervention",
    )
    scorecards = {
        "no_intervention": _worldline_scorecard(no_intervention),
        "early_intervention": _worldline_scorecard(early),
        "late_intervention": _worldline_scorecard(late),
    }
    ranking = sorted(
        scorecards.items(),
        key=lambda item: (
            float(item[1]["max_panic"]) * 0.45
            + float(item[1]["max_drawdown"]) * 0.35
            + float(item[1]["volatility"]) * 0.20
            - float(item[1]["return_pct"]) * 0.15
        ),
    )
    return {
        "intervention_steps": {"early": early_step, "late": late_step},
        "recommended_timing": str(ranking[0][0]) if ranking else "no_intervention",
        "scorecards": scorecards,
        "worlds": {
            "no_intervention": no_intervention.to_dict(orient="records"),
            "early_intervention": early.to_dict(orient="records"),
            "late_intervention": late.to_dict(orient="records"),
        },
    }


def _policy_session_status_text(status: str) -> str:
    return POLICY_SESSION_STATUS_LABELS.get(str(status or "").strip().lower(), "未开始")


def _policy_session_reference_profile(reference_frame: Optional[pd.DataFrame]) -> Dict[str, Any]:
    fallback = {
        "start_close": 3000.0,
        "start_volume": 1_000_000.0,
        "mean_return": 0.0003,
        "volatility": 0.0080,
        "base_panic": 0.18,
        "base_csad": 0.06,
        "history_end": "",
    }
    if reference_frame is None or reference_frame.empty:
        return fallback

    frame = reference_frame.copy()
    close = pd.to_numeric(frame.get("close"), errors="coerce").dropna() if "close" in frame else pd.Series(dtype=float)
    if close.empty:
        return fallback
    returns = close.pct_change().dropna()
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").dropna() if "volume" in frame else pd.Series(dtype=float)
    avg_abs_return = float(np.mean(np.abs(returns))) if not returns.empty else 0.0015
    history_end = str(frame.iloc[-1].get("time", "")) if "time" in frame and not frame.empty else ""
    return {
        "start_close": float(close.iloc[-1]),
        "start_volume": float(volume.tail(10).mean()) if not volume.empty else 1_000_000.0,
        "mean_return": float(returns.mean()) if not returns.empty else fallback["mean_return"],
        "volatility": float(max(returns.std() if not returns.empty else fallback["volatility"], 0.003)),
        "base_panic": float(np.clip(0.12 + avg_abs_return * 40.0, 0.05, 0.35)),
        "base_csad": float(np.clip(0.05 + avg_abs_return * 18.0, 0.03, 0.16)),
        "history_end": history_end,
    }


def _policy_session_bday(calendar_start: str, day_number: int) -> str:
    start = pd.Timestamp(calendar_start).normalize()
    return pd.bdate_range(start=start, periods=max(1, int(day_number)))[-1].strftime("%Y-%m-%d")


def _policy_session_effect_score(policy_text: str, intensity: float) -> Tuple[float, PolicyShock]:
    shock = _compile_scaled_shock(policy_text, intensity)
    return _shock_score(shock), shock


def _policy_session_build_event(
    *,
    policy_name: str,
    policy_text: str,
    policy_type: str,
    intensity: float,
    effective_day: int,
    half_life_days: int,
    created_day: int,
    rumor_noise: bool,
) -> Dict[str, Any]:
    score, shock = _policy_session_effect_score(policy_text, intensity)
    _, package = _compile_policy_bundle(
        policy_text,
        float(intensity),
        policy_type_hint=policy_type,
    )
    return {
        "policy_id": package.event.policy_id,
        "policy_name": str(policy_name or "未命名政策"),
        "policy_text": str(policy_text or ""),
        "policy_type": str(policy_type or "自定义政策"),
        "intensity": float(intensity),
        "effective_day": max(1, int(effective_day)),
        "half_life_days": max(1, int(half_life_days)),
        "created_day": max(0, int(created_day)),
        "rumor_noise": bool(rumor_noise),
        "score": float(score),
        "shock": shock.to_dict(),
        "package": package.to_dict(),
        "remaining_effect": 0.0,
        "status": "queued",
    }


def _policy_session_empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "step",
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "panic_level",
            "csad",
            "政策压力",
            "活跃政策数",
            "待生效政策数",
            "买入量",
            "卖出量",
            "散户净流",
            "机构净流",
            "量化净流",
            "做市净流",
        ]
    )


def _policy_session_timeline_item_from_runner(item: Dict[str, Any]) -> Dict[str, Any]:
    state_raw = str(item.get("当前状态", "") or "").strip().lower()
    state_map = {
        "queued": "待生效",
        "active": "生效中",
        "fading": "影响减弱",
        "exhausted": "已淡出",
    }
    base_strength = float(item.get("基础强度", 0.0) or 0.0)
    current_strength = float(item.get("当前强度", 0.0) or 0.0)
    remaining = 0.0 if base_strength <= 0 else float(np.clip(current_strength / max(base_strength, 1e-9), 0.0, 1.0))
    name = str(item.get("政策标签", item.get("政策文本", "未命名政策")))
    policy_type = str(dict(item.get("元数据", {}) or {}).get("policy_type", "自定义政策"))
    policy_text = str(item.get("政策文本", ""))
    return {
        "政策名称": str(item.get("政策标签", item.get("政策文本", "未命名政策"))),
        "政策类型": policy_type,
        "生效交易日": int(item.get("生效日", 1) or 1),
        "当前状态": state_map.get(state_raw, state_raw or "待生效"),
        "初始强度": base_strength,
        "剩余影响": remaining,
        "传言噪声": "是" if bool(item.get("是否谣言噪声", False)) else "否",
        "政策文本": policy_text,
        "status": state_raw,
        "policy_name": name,
        "policy_type": policy_type,
        "policy_text": policy_text,
        "remaining_effect": remaining,
    }


def _policy_session_discovered_metrics(
    session: Dict[str, Any],
    frame: Optional[pd.DataFrame] = None,
    reports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    market_frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(session.get("frame_rows", []) or [])
    report_payloads: List[Dict[str, Any]] = []
    for item in reports or []:
        if isinstance(item, dict) and item:
            report_payloads.append(dict(item))
    latest = session.get("latest_step_report", {})
    if isinstance(latest, dict) and latest:
        report_payloads.append(dict(latest))
    if market_frame.empty:
        return ObjectiveDiscoveryEngine().discover([], reports=report_payloads, top_k=8).to_dict()
    return ObjectiveDiscoveryEngine().discover(market_frame, reports=report_payloads, top_k=8).to_dict()


def _policy_session_sync_from_runner(session: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runner = session.get("_runner")
    if not isinstance(runner, PolicySession):
        return session

    if snapshot is None:
        snapshot = runner.advance(0)

    raw_frame = snapshot.get("frame", pd.DataFrame())
    if not isinstance(raw_frame, pd.DataFrame):
        raw_frame = pd.DataFrame(raw_frame or [])
    display_frame = _session_frame_to_market_frame(raw_frame, anchor_close=float(session.get("index_anchor_close", 0.0)))
    active_items = snapshot.get("active_policies", []) if isinstance(snapshot.get("active_policies"), list) else []
    queued_items = snapshot.get("queued_policies", []) if isinstance(snapshot.get("queued_policies"), list) else []
    runtime_active = snapshot.get("active_events", []) if isinstance(snapshot.get("active_events"), list) else []
    runtime_queued = snapshot.get("queued_events", []) if isinstance(snapshot.get("queued_events"), list) else []
    runtime_expired = snapshot.get("expired_events", []) if isinstance(snapshot.get("expired_events"), list) else []
    runtime_timeline = snapshot.get("event_timeline", []) if isinstance(snapshot.get("event_timeline"), list) else []
    event_digest = snapshot.get("event_digest", {}) if isinstance(snapshot.get("event_digest"), dict) else {}
    raw_reports: List[Dict[str, Any]] = []
    if isinstance(raw_frame, pd.DataFrame) and "原始报告" in raw_frame.columns:
        raw_reports = [dict(item) for item in raw_frame["原始报告"].tolist() if isinstance(item, dict)]
    policy_events = [_policy_session_timeline_item_from_runner(item) for item in [*active_items, *queued_items]]
    summary_payload = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    session["status"] = str(snapshot.get("status", session.get("status", "idle")))
    session["current_day"] = int(snapshot.get("current_day", session.get("current_day", 0)))
    session["frame_rows"] = display_frame.to_dict(orient="records")
    session["raw_frame_rows"] = raw_frame.to_dict(orient="records")
    session["policy_events"] = policy_events
    session["policy_timeline"] = policy_events
    session["runtime_active_events"] = runtime_active
    session["runtime_queued_events"] = runtime_queued
    session["runtime_expired_events"] = runtime_expired
    session["runtime_event_timeline"] = runtime_timeline
    session["event_digest"] = event_digest
    session["latest_step_report"] = dict(snapshot.get("last_step_report", {}) or {})
    display_latest_close = float(display_frame["close"].iloc[-1]) if not display_frame.empty else float(session.get("last_close", 0.0) or 0.0)
    session["last_close"] = display_latest_close
    session["summary"] = {
        "return_pct": float(summary_payload.get("累计收益率", 0.0) or 0.0),
        "max_drawdown": float(summary_payload.get("最大回撤", 0.0) or 0.0),
        "avg_panic": float(display_frame["panic_level"].mean()) if not display_frame.empty else 0.0,
        "max_panic": float(display_frame["panic_level"].max()) if not display_frame.empty else 0.0,
        "volatility": float(display_frame["close"].pct_change().fillna(0.0).std()) if len(display_frame) > 1 else 0.0,
        "avg_csad": float(display_frame["csad"].mean()) if not display_frame.empty else 0.0,
        "avg_volume": float(display_frame["volume"].mean()) if not display_frame.empty else 0.0,
        "policy_signal_avg": float(len(active_items)),
        "active_policy_max": float(len(active_items)),
        "active_event_count": float(len(runtime_active)),
        "queued_event_count": float(len(runtime_queued)),
        "latest_close": display_latest_close,
        "最新收盘价": display_latest_close,
        "累计收益率": float(summary_payload.get("累计收益率", 0.0) or 0.0),
    }
    session["discovered_metrics"] = _policy_session_discovered_metrics(session, display_frame, raw_reports)
    session["_snapshot"] = snapshot
    return session


def _policy_session_backdrop_rows(reference_frame: Optional[pd.DataFrame], total_days: int) -> List[Dict[str, float]]:
    if reference_frame is None or reference_frame.empty:
        return []
    frame = reference_frame.copy().tail(max(int(total_days), 10)).reset_index(drop=True)
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    rows: List[Dict[str, float]] = []
    for idx in range(len(frame)):
        price = float(close.iloc[idx]) if idx < len(close) and pd.notna(close.iloc[idx]) else 0.0
        if price <= 0.0:
            continue
        rows.append(
            {
                "step": float(idx + 1),
                "price": price,
                "close": price,
                "volume": float(volume.iloc[idx]) if idx < len(volume) and pd.notna(volume.iloc[idx]) else 0.0,
            }
        )
    return rows


def _policy_session_new(
    *,
    policy_name: str,
    policy_text: str,
    policy_type: str,
    total_days: int,
    intensity: float,
    effective_day: int,
    half_life_days: int,
    rumor_noise: bool,
    index_label: str,
    index_symbol: str,
    reference_frame: Optional[pd.DataFrame],
    runtime_profile: RuntimeModeProfile,
) -> Dict[str, Any]:
    reference = _policy_session_reference_profile(reference_frame)
    history_end = reference.get("history_end") or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    calendar_start = pd.bdate_range(start=pd.Timestamp(history_end).normalize(), periods=2)[-1].strftime("%Y-%m-%d")
    agents = _build_policy_lab_agents(runtime_profile)
    backdrop_rows = _policy_session_backdrop_rows(reference_frame, max(1, int(total_days)))
    use_backdrop = bool(backdrop_rows) and str(runtime_profile.mode or "SMART").strip().upper() == "SMART"
    runner = PolicySession.create(
        agents=agents,
        total_days=max(1, int(total_days)),
        base_policy="",
        start_date=calendar_start,
        half_life_days=max(1, int(half_life_days)),
        enable_random_policy_events=False,
        simulation_mode=runtime_profile.mode,
        use_isolated_matching=True,
        market_pipeline_v2=bool(runtime_profile.market_pipeline_v2),
        llm_primary=bool(runtime_profile.llm_primary),
        deep_reasoning_pause_s=float(runtime_profile.pause_for_llm_seconds),
        enable_policy_committee=bool(runtime_profile.enable_policy_committee),
        runner_symbol="A_SHARE_IDX",
        steps_per_day=1,
        model_priority=list(runtime_profile.model_priority),
        hybrid_replay=use_backdrop,
        exogenous_backdrop=backdrop_rows if use_backdrop else None,
        hybrid_backdrop_weight=0.82 if use_backdrop else 0.0,
    )
    runner.enqueue_policy(
        policy_text,
        effective_day=max(1, int(effective_day)),
        strength=float(intensity),
        half_life_days=max(1, int(half_life_days)),
        rumor_noise=bool(rumor_noise),
        label=str(policy_name or "基础政策"),
        source="base_policy",
        metadata={"policy_type": str(policy_type or "自定义政策")},
    )
    session = {
        "session_id": f"policy-session-{_seed_from_text(policy_name + policy_text + str(total_days)) & 0xFFFFFFFF:08x}",
        "status": "idle",
        "total_days": max(1, int(total_days)),
        "current_day": 0,
        "calendar_start": calendar_start,
        "index_label": str(index_label),
        "index_symbol": str(index_symbol),
        "policy_name": str(policy_name),
        "policy_text": str(policy_text),
        "policy_type": str(policy_type or "自定义政策"),
        "intensity": float(intensity),
        "effective_day": max(1, int(effective_day)),
        "half_life_days": max(1, int(half_life_days)),
        "rumor_noise": bool(rumor_noise),
        "policy_events": [],
        "frame_rows": [],
        "summary": {},
        "report_bundle": None,
        "report_payload": None,
        "reference_profile": reference,
        "runtime_profile": runtime_profile.to_dict(),
        "mode_text": _runtime_mode_text(runtime_profile),
        "last_close": float(reference["start_close"]),
        "last_return": float(reference["mean_return"]),
        "last_panic": float(reference["base_panic"]),
        "last_csad": float(reference["base_csad"]),
        "last_volume": float(reference["start_volume"]),
        "policy_package": None,
        "policy_timeline": [],
        "index_anchor_close": float(reference["start_close"]),
        "autoplay": {
            "enabled": False,
            "step_days": 1,
            "interval_seconds": 0.8,
            "last_wallclock_ts": 0.0,
        },
        "latest_step_report": {},
        "_runner": runner,
        "llm_agent_count": int(sum(1 for agent in agents if bool(getattr(agent, "use_llm", False)))),
    }
    timeline = [plan.to_timeline_row(0) for plan in runner.policies]
    session["policy_events"] = [_policy_session_timeline_item_from_runner(item) for item in timeline]
    session["policy_timeline"] = list(session["policy_events"])
    return session


def _policy_session_autoplay_state(session: Dict[str, Any]) -> Dict[str, Any]:
    autoplay = session.setdefault(
        "autoplay",
        {
            "enabled": False,
            "step_days": 1,
            "interval_seconds": 0.8,
            "last_wallclock_ts": 0.0,
        },
    )
    autoplay["enabled"] = bool(autoplay.get("enabled", False))
    autoplay["step_days"] = max(1, int(autoplay.get("step_days", 1) or 1))
    autoplay["interval_seconds"] = max(0.0, float(autoplay.get("interval_seconds", 0.8) or 0.0))
    autoplay["last_wallclock_ts"] = float(autoplay.get("last_wallclock_ts", 0.0) or 0.0)
    return autoplay


def _policy_session_disable_autoplay(session: Dict[str, Any]) -> None:
    autoplay = _policy_session_autoplay_state(session)
    autoplay["enabled"] = False


def _policy_session_maybe_autoplay(
    session: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    min_interval_seconds: Optional[float] = None,
) -> bool:
    autoplay = _policy_session_autoplay_state(session)
    if not autoplay.get("enabled", False):
        return False

    status = str(session.get("status", "idle")).lower()
    if status not in {"running", "paused"}:
        autoplay["enabled"] = False
        return False

    total_days = int(session.get("total_days", 0) or 0)
    current_day = int(session.get("current_day", 0) or 0)
    if total_days > 0 and current_day >= total_days:
        session["status"] = "completed"
        autoplay["enabled"] = False
        return False

    interval_seconds = float(
        autoplay.get("interval_seconds", 0.8) if min_interval_seconds is None else min_interval_seconds
    )
    now_value = float(time.time() if now_ts is None else now_ts)
    last_ts = float(autoplay.get("last_wallclock_ts", 0.0) or 0.0)
    if last_ts > 0.0 and now_value - last_ts < interval_seconds:
        return False

    before = int(session.get("current_day", 0) or 0)
    _policy_session_advance(session, int(autoplay.get("step_days", 1) or 1))
    autoplay["last_wallclock_ts"] = now_value
    if int(session.get("current_day", 0) or 0) >= int(session.get("total_days", 0) or 0):
        autoplay["enabled"] = False
    return int(session.get("current_day", 0) or 0) > before


def _build_policy_demo_cards(
    *,
    policy_text: str,
    latest_step_report: Dict[str, Any],
    session_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    summary = dict(session_summary or {})
    latest = dict(latest_step_report or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    policy_signal = dict(chain.get("policy_signal", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    index_move = dict(chain.get("index_move", {}) or {})

    signal_strength = float(policy_signal.get("strength", summary.get("policy_signal_avg", 0.0)) or 0.0)
    social_mean = float(sentiment.get("social_mean", 0.0) or 0.0)
    imbalance = float(order_flow.get("imbalance", 0.0) or 0.0)
    matching_mode = str(matching.get("matching_mode", latest.get("matching_mode", "session")))
    latest_close = float(index_move.get("new_price", summary.get("最新收盘价", summary.get("latest_close", 0.0))) or 0.0)
    return_pct = float(index_move.get("return_pct", latest.get("price_change_pct", 0.0)) or 0.0)

    return [
        {
            "phase": "政策注入",
            "summary": (str(policy_signal.get("policy_text", "") or policy_text).strip() or "等待政策输入")[:72],
            "detail": f"信号强度 {signal_strength:+.2f}｜来源 {str(policy_signal.get('source', 'policy_session'))}",
        },
        {
            "phase": "情绪扩散",
            "summary": f"情绪均值 {social_mean:+.2f}，多智能体开始重估政策影响。",
            "detail": f"订单失衡 {imbalance:+.0f}｜传导斜率 {signal_strength:+.2f}",
        },
        {
            "phase": "撮合落地",
            "summary": f"最新点位 {latest_close:.2f}，单日变化 {return_pct:+.2f}%。",
            "detail": f"撮合模式 {matching_mode}｜市场进入重定价阶段",
        },
    ]


def _build_policy_demo_briefing(session: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = dict(summary or {})
    latest = dict(session.get("latest_step_report", {}) or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    policy_signal = dict(chain.get("policy_signal", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    index_move = dict(chain.get("index_move", {}) or {})

    current_day = int(session.get("current_day", 0) or 0)
    total_days = int(session.get("total_days", 0) or 0)
    signal_strength = float(policy_signal.get("strength", summary.get("policy_signal_avg", 0.0)) or 0.0)
    panic_level = float(sentiment.get("panic_level", summary.get("max_panic", 0.0)) or 0.0)
    return_pct = float(index_move.get("return_pct", latest.get("price_change_pct", 0.0)) or 0.0)
    trade_count = int(matching.get("trade_count", latest.get("trade_count", 0)) or 0)
    imbalance = float(order_flow.get("imbalance", 0.0) or 0.0)
    autoplay = dict(session.get("autoplay", {}) or {})

    if current_day <= 1 and trade_count <= 0:
        phase = "政策刚注入，预期正在形成"
    elif panic_level >= 0.55 or abs(imbalance) >= 600:
        phase = "情绪扩散加速，订单簿开始失衡"
    elif trade_count > 0:
        phase = "撮合落地，市场正在重估政策价格"
    else:
        phase = "会话推进中，继续观察资金与情绪传导"

    if panic_level >= 0.65 or return_pct <= -1.2:
        tone = "risk"
        alert = "高波动预警"
    elif abs(return_pct) >= 0.8 or abs(imbalance) >= 400:
        tone = "watch"
        alert = "市场异动提示"
    else:
        tone = "calm"
        alert = "稳态观察"

    bullets = [
        f"交易日进度 {current_day}/{total_days}，当前模式 {str(session.get('mode_text', '会话仿真'))}",
        f"政策强度 {signal_strength:+.2f}，活跃政策 {sum(1 for item in session.get('policy_events', []) if item.get('status') == 'active')} 项",
        f"订单失衡 {imbalance:+.0f}，单日变化 {return_pct:+.2f}%，传导阶段：{phase}",
    ]
    chips = [
        f"状态：{_policy_session_status_text(str(session.get('status', 'idle')))}",
        f"自动演示：{'开启' if bool(autoplay.get('enabled', False)) else '关闭'}",
        f"恐慌度：{panic_level:.2f}",
    ]
    subtitle = str(policy_signal.get("policy_text", "") or session.get("policy_text", "")).strip()
    if len(subtitle) > 88:
        subtitle = subtitle[:88] + "…"
    return {
        "kicker": "政策风洞推演",
        "phase": phase,
        "subtitle": subtitle or "等待政策输入后启动会话仿真。",
        "alert": alert,
        "tone": tone,
        "bullets": bullets,
        "chips": chips,
    }


def _build_policy_market_pulse(
    session: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    summary = dict(summary or {})
    latest = dict(session.get("latest_step_report", {}) or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    index_move = dict(chain.get("index_move", {}) or {})

    buy_volume = float(order_flow.get("buy_volume", 0.0) or 0.0)
    sell_volume = float(order_flow.get("sell_volume", 0.0) or 0.0)
    imbalance = buy_volume - sell_volume
    leader = "买盘主导" if imbalance >= 0 else "卖盘主导"
    pulse_color = "up" if imbalance >= 0 else "down"
    panic_level = float(sentiment.get("panic_level", summary.get("avg_panic", 0.0)) or 0.0)
    return_pct = float(index_move.get("return_pct", latest.get("price_change_pct", 0.0)) or 0.0)
    direction = "上行重估" if return_pct >= 0 else "下行重估"

    return [
        {
            "label": "主导资金",
            "value": leader,
            "note": f"净差 {imbalance:+.0f}",
            "tone": pulse_color,
        },
        {
            "label": "情绪热度",
            "value": f"{panic_level:.2f}",
            "note": "高于 0.60 需重点解释" if panic_level >= 0.60 else "仍在可控区间",
            "tone": "watch" if panic_level >= 0.60 else "calm",
        },
        {
            "label": "撮合反馈",
            "value": f"{int(matching.get('trade_count', latest.get('trade_count', 0)) or 0)} 笔",
            "note": f"模式 {str(matching.get('matching_mode', latest.get('matching_mode', 'session')))}",
            "tone": "watch",
        },
        {
            "label": "指数动作",
            "value": direction,
            "note": f"{return_pct:+.2f}%",
            "tone": "up" if return_pct >= 0 else "down",
        },
    ]


def _policy_session_effect_for_event(event: Dict[str, Any], day: int) -> float:
    effective_day = int(event.get("effective_day", 1))
    if day < effective_day:
        return 0.0
    half_life = max(1, int(event.get("half_life_days", 30)))
    age = max(0, int(day) - effective_day)
    decay = float(np.exp(-age / half_life))
    event["remaining_effect"] = float(decay)
    if decay >= 0.35:
        event["status"] = "active"
    elif decay >= 0.08:
        event["status"] = "fading"
    else:
        event["status"] = "exhausted"
    return float(event.get("score", 0.0)) * float(event.get("intensity", 1.0)) * decay


def _policy_session_row(session: Dict[str, Any], day: int) -> Dict[str, Any]:
    reference = dict(session.get("reference_profile", {}) or {})
    prev_close = float(session.get("last_close", reference.get("start_close", 3000.0)))
    prev_return = float(session.get("last_return", reference.get("mean_return", 0.0003)))
    prev_panic = float(session.get("last_panic", reference.get("base_panic", 0.18)))
    prev_csad = float(session.get("last_csad", reference.get("base_csad", 0.06)))
    prev_volume = float(session.get("last_volume", reference.get("start_volume", 1_000_000.0)))
    active_events = 0
    queued_events = 0
    policy_pressure = 0.0
    rumor_pressure = 0.0
    package_parts: List[str] = []
    for event in session.get("policy_events", []):
        package_parts.append(str(event.get("policy_text", "")))
        if day < int(event.get("effective_day", 1)):
            queued_events += 1
            continue
        active_events += 1
        if bool(event.get("rumor_noise", False)):
            rumor_pressure += 0.12 * float(event.get("intensity", 1.0))
        policy_pressure += _policy_session_effect_for_event(event, day)

    policy_signal = float(np.tanh(policy_pressure / 8.0))
    rumor_signal = float(np.tanh(rumor_pressure / 2.0))
    seed = _seed_from_text(
        f"{session.get('session_id', 'policy-session')}|{day}|{policy_signal:.4f}|{prev_close:.2f}|{len(package_parts)}"
    )
    rng = np.random.default_rng(seed)
    noise = float(rng.normal(0.0, max(float(reference.get("volatility", 0.0080)) * 0.35, 0.0015)))

    retail_flow = policy_signal * 0.55 + rumor_signal * 0.40 - prev_panic * 0.10 + noise * 8.0
    institution_flow = policy_signal * 0.42 + prev_return * 0.80 - prev_panic * 0.06 + noise * 4.0
    quant_flow = policy_signal * 0.28 - prev_return * 0.75 + noise * 3.0
    maker_flow = -abs(policy_signal) * 0.18 - prev_panic * 0.08 - noise * 2.0
    net_flow = retail_flow + institution_flow + quant_flow + maker_flow

    daily_return = (
        float(reference.get("mean_return", 0.0003))
        + 0.0045 * policy_signal
        + 0.0022 * net_flow
        - 0.0011 * prev_panic
        + noise
    )
    close = max(1.0, prev_close * (1.0 + daily_return))
    amplitude = max(abs(daily_return) * 1.8, float(reference.get("volatility", 0.0080)) * 1.5, 0.0025)
    high = max(prev_close, close) * (1.0 + amplitude * (0.55 + max(policy_signal, 0.0) * 0.25))
    low = min(prev_close, close) * (1.0 - amplitude * (0.55 + max(-policy_signal, 0.0) * 0.25))
    panic = float(
        np.clip(
            0.55 * prev_panic
            + max(0.0, -daily_return) * 12.0
            + abs(rumor_signal) * 0.20
            + abs(policy_signal) * 0.08,
            0.03,
            0.98,
        )
    )
    csad = float(np.clip(0.62 * prev_csad + abs(daily_return) * 4.8 + panic * 0.05, 0.03, 0.30))
    volume = float(prev_volume * (1.0 + abs(policy_signal) * 0.18 + panic * 0.22 + abs(net_flow) * 0.08))
    day_date = _policy_session_bday(session.get("calendar_start", pd.Timestamp.today().normalize().strftime("%Y-%m-%d")), day)
    return {
        "step": int(day),
        "time": day_date,
        "open": round(prev_close, 2),
        "high": round(high, 2),
        "low": round(max(0.01, low), 2),
        "close": round(close, 2),
        "volume": round(volume, 2),
        "panic_level": round(panic, 4),
        "csad": round(csad, 4),
        "政策压力": round(policy_signal, 4),
        "活跃政策数": int(active_events),
        "待生效政策数": int(queued_events),
        "买入量": round(max(0.0, retail_flow + institution_flow + quant_flow), 2),
        "卖出量": round(max(0.0, -maker_flow - min(net_flow, 0.0)), 2),
        "散户净流": round(retail_flow, 4),
        "机构净流": round(institution_flow, 4),
        "量化净流": round(quant_flow, 4),
        "做市净流": round(maker_flow, 4),
    }


def _policy_session_refresh_summary(session: Dict[str, Any]) -> Dict[str, float]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        _policy_session_sync_from_runner(session)
        return dict(session.get("summary", {}) or {})
    frame = _policy_session_frame(session)
    if frame.empty:
        summary = {
            "return_pct": 0.0,
            "max_drawdown": 0.0,
            "avg_panic": 0.0,
            "max_panic": 0.0,
            "volatility": 0.0,
            "avg_csad": 0.0,
            "avg_volume": 0.0,
            "policy_signal_avg": 0.0,
            "active_policy_max": 0.0,
        }
        session["summary"] = summary
        session["discovered_metrics"] = _policy_session_discovered_metrics(session, pd.DataFrame())
        return summary
    summary = _compute_policy_summary(frame.rename(columns={"panic_level": "panic_level", "csad": "csad"}))
    summary["policy_signal_avg"] = float(frame["政策压力"].mean()) if "政策压力" in frame else 0.0
    summary["active_policy_max"] = float(frame["活跃政策数"].max()) if "活跃政策数" in frame else 0.0
    summary["latest_close"] = float(frame.iloc[-1]["close"])
    summary["最新收盘价"] = float(frame.iloc[-1]["close"])
    session["summary"] = summary
    session["discovered_metrics"] = _policy_session_discovered_metrics(session, frame)
    return summary


def _policy_session_frame(session: Dict[str, Any]) -> pd.DataFrame:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        snapshot = session.get("_snapshot", {})
        raw_frame = snapshot.get("frame", pd.DataFrame()) if isinstance(snapshot, dict) else pd.DataFrame()
        if not isinstance(raw_frame, pd.DataFrame):
            raw_frame = pd.DataFrame(raw_frame or [])
        return _session_frame_to_market_frame(raw_frame, anchor_close=float(session.get("index_anchor_close", 0.0)))
    rows = list(session.get("frame_rows", []) or [])
    if not rows:
        return _policy_session_empty_frame()
    frame = pd.DataFrame(rows)
    return frame.sort_values("step").reset_index(drop=True)


def _policy_session_timeline(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        _policy_session_sync_from_runner(session)
        return list(session.get("policy_timeline", []) or [])
    current_day = int(session.get("current_day", 0))
    rows: List[Dict[str, Any]] = []
    for event in session.get("policy_events", []):
        effective_day = int(event.get("effective_day", 1))
        if current_day < effective_day:
            state = "待生效"
            remaining = 0.0
        else:
            age = current_day - effective_day
            half_life = max(1, int(event.get("half_life_days", 30)))
            remaining = float(np.exp(-age / half_life))
            if remaining >= 0.35:
                state = "生效中"
            elif remaining >= 0.08:
                state = "影响减弱"
            else:
                state = "已淡出"
        rows.append(
            {
                "政策名称": str(event.get("policy_name", "")),
                "政策类型": str(event.get("policy_type", "")),
                "生效交易日": effective_day,
                "当前状态": state,
                "初始强度": float(event.get("intensity", 0.0)),
                "剩余影响": round(remaining, 4),
                "传言噪声": "是" if bool(event.get("rumor_noise", False)) else "否",
            }
        )
    return rows


def _policy_session_enqueue(
    session: Dict[str, Any],
    *,
    policy_name: str,
    policy_text: str,
    policy_type: str,
    effective_day: int,
    intensity: float,
    half_life_days: int,
    rumor_noise: bool,
    ) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        runner.enqueue_policy(
            policy_text,
            effective_day=max(1, int(effective_day)),
            strength=float(intensity),
            half_life_days=float(half_life_days),
            rumor_noise=bool(rumor_noise),
            label=str(policy_name or "追加政策"),
            source="append_policy",
            metadata={"policy_type": str(policy_type or "自定义政策")},
        )
        _policy_session_sync_from_runner(session)
        event = next((item for item in session.get("policy_timeline", []) if item.get("政策名称") == str(policy_name or "追加政策")), None)
        return dict(event or {})

    created_day = int(session.get("current_day", 0))
    event = _policy_session_build_event(
        policy_name=policy_name,
        policy_text=policy_text,
        policy_type=policy_type,
        intensity=intensity,
        effective_day=max(1, int(effective_day)),
        half_life_days=half_life_days,
        created_day=created_day,
        rumor_noise=rumor_noise,
    )
    session.setdefault("policy_events", []).append(event)
    session["report_bundle"] = None
    session["report_payload"] = None
    session["policy_timeline"] = _policy_session_timeline(session)
    return event


def _policy_session_enqueue_runtime_event(
    session: Dict[str, Any],
    *,
    event_type: str,
    title: str,
    raw_text: str,
    effective_day: int,
    strength: float,
    half_life_days: int,
    scope: str = "broad_market",
    confidence: float = 0.75,
) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        event_id = runner.append_event(
            event_type=event_type,
            title=title,
            raw_text=raw_text,
            effective_day=max(1, int(effective_day)),
            strength=float(strength),
            half_life=float(half_life_days),
            scope=str(scope or "broad_market"),
            confidence=float(confidence),
            source="policy_lab_runtime_ui",
            metadata={"ui_injected": True},
        )
        _policy_session_sync_from_runner(session)
        return next(
            (
                dict(item)
                for item in list(session.get("runtime_event_timeline", []) or [])
                if str(item.get("event_id", "")) == str(event_id)
            ),
            {"event_id": event_id, "event_type": event_type, "title": title, "raw_text": raw_text},
        )
    return {}


def _policy_session_advance(session: Dict[str, Any], days: int) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        if str(session.get("status", "idle")) == "paused":
            runner.resume()
        snapshot = runner.advance(days=max(0, int(days)))
        _policy_session_sync_from_runner(session, snapshot)
        if str(session.get("status", "idle")).lower() == "completed":
            _policy_session_disable_autoplay(session)
        return session
    if str(session.get("status", "idle")) not in {"running", "paused"}:
        return session
    if str(session.get("status", "idle")) == "paused":
        session["status"] = "running"
    total_days = int(session.get("total_days", 100))
    steps = max(0, int(days))
    for _ in range(steps):
        current_day = int(session.get("current_day", 0))
        if current_day >= total_days:
            session["status"] = "completed"
            break
        next_day = current_day + 1
        row = _policy_session_row(session, next_day)
        session.setdefault("frame_rows", []).append(row)
        session["current_day"] = next_day
        session["last_close"] = float(row["close"])
        session["last_return"] = float((row["close"] - row["open"]) / max(row["open"], 1e-9))
        session["last_panic"] = float(row["panic_level"])
        session["last_csad"] = float(row["csad"])
        session["last_volume"] = float(row["volume"])
        session["latest_step_report"] = {
            "tick": next_day,
            "day_count": next_day,
            "buy_volume": float(row["买入量"]),
            "sell_volume": float(row["卖出量"]),
            "old_price": float(row["open"]),
            "new_price": float(row["close"]),
            "price_change_pct": float((row["close"] - row["open"]) / max(row["open"], 1e-9) * 100.0),
            "trade_count": int(max(row["买入量"], row["卖出量"]) // 100),
            "matching_mode": "session_fallback",
            "transmission_chain": {
                "policy_signal": {
                    "policy_text": str(session.get("policy_text", "")),
                    "strength": float(row["政策压力"]),
                    "source": "policy_session_fallback",
                },
                "agent_sentiment": {
                    "social_mean": float(0.5 - row["panic_level"]),
                    "panic_level": float(row["panic_level"]),
                    "committee_enabled": False,
                },
                "order_flow": {
                    "buy_volume": float(row["买入量"]),
                    "sell_volume": float(row["卖出量"]),
                    "imbalance": float(row["买入量"] - row["卖出量"]),
                },
                "matching_result": {
                    "trade_count": int(max(row["买入量"], row["卖出量"]) // 100),
                    "last_price": float(row["close"]),
                    "matching_mode": "session_fallback",
                },
                "index_move": {
                    "old_price": float(row["open"]),
                    "new_price": float(row["close"]),
                    "return_pct": float((row["close"] - row["open"]) / max(row["open"], 1e-9) * 100.0),
                },
            },
        }
        if next_day >= total_days:
            session["status"] = "completed"
            break
    if int(session.get("current_day", 0)) >= total_days:
        session["status"] = "completed"
        _policy_session_disable_autoplay(session)
    _policy_session_refresh_summary(session)
    session["policy_timeline"] = _policy_session_timeline(session)
    return session


def _policy_session_stop(session: Dict[str, Any]) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        runner.stop()
        snapshot = runner.advance(0)
        _policy_session_sync_from_runner(session, snapshot)
        _policy_session_disable_autoplay(session)
        return session
    if str(session.get("status", "idle")) in {"running", "paused"}:
        session["status"] = "stopped"
    _policy_session_disable_autoplay(session)
    return session


def _policy_session_report_payload(session: Dict[str, Any], runtime_profile: RuntimeModeProfile) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        payload = runner.build_report_payload()
        timeline = list(session.get("policy_timeline", []) or [])
        combined_policy_text = "\n".join(str(item.get("政策文本", "")) for item in timeline if str(item.get("政策文本", "")).strip())
        package_dict = _policy_session_policy_package(session)
        summary = dict(session.get("summary", {}) or {})
        discovered_metrics = dict(session.get("discovered_metrics", {}) or {})
        if not discovered_metrics:
            discovered_metrics = ObjectiveDiscoveryEngine().discover(payload.get("日度结果", []), top_k=8).to_dict()
        narrative = _get_policy_narrative(
            combined_policy_text or str(session.get("policy_text", "")),
            summary,
            package_dict or {"explanation": {}, "policy_schema": {}, "top_layers": {}},
            runtime_profile,
        )
        return {
            "title": f"政策试验台 - {session.get('policy_name', '政策仿真')}",
            "summary": summary,
            "policy_name": session.get("policy_name", "政策仿真"),
            "policy_text": session.get("policy_text", ""),
            "policy_type": session.get("policy_type", "自定义政策"),
            "timeline": timeline,
            "runtime_events": {
                "active": list(session.get("runtime_active_events", []) or []),
                "queued": list(session.get("runtime_queued_events", []) or []),
                "expired": list(session.get("runtime_expired_events", []) or []),
                "timeline": list(session.get("runtime_event_timeline", []) or []),
                "digest": dict(session.get("event_digest", {}) or {}),
            },
            "frame": payload.get("日度结果", []),
            "policy_package": package_dict,
            "discovered_metrics": discovered_metrics,
            "runtime_mode": runtime_profile.mode,
            "runtime_profile": runtime_profile.to_dict(),
            "narrative": narrative,
            "impact_evaluation": payload.get("impact_evaluation", {}),
            "risk_assessment": payload.get("risk_assessment", {}),
            "action_recommendations": payload.get("action_recommendations", {}),
            "monitoring_kpis": payload.get("monitoring_kpis", []),
            "session": {
                "session_id": session.get("session_id", ""),
                "status": session.get("status", "idle"),
                "current_day": int(session.get("current_day", 0)),
                "total_days": int(session.get("total_days", 0)),
                "calendar_start": session.get("calendar_start", ""),
                "index_label": session.get("index_label", ""),
                "index_symbol": session.get("index_symbol", ""),
            },
            "experiment_registry": dict(session.get("experiment_registry", {}) or {}),
            "reproducibility": dict(session.get("reproducibility", {}) or {}),
            "scorecard": dict(session.get("scorecard", {}) or {}),
            "scenario_diff": dict(session.get("scenario_diff", {}) or {}),
            "kline": {
                "source": "trade_tape_aggregation",
                "benchmark": session.get("index_symbol", "sh000001"),
                "ohlcv": list(session.get("workbench_ohlcv_rows", []) or []),
            },
            "runner_payload": payload,
        }
    frame = _policy_session_frame(session)
    summary = dict(session.get("summary", {}) or {}) or _policy_session_refresh_summary(session)
    combined_policy_text = "\n".join(
        str(event.get("policy_text", "")) for event in session.get("policy_events", []) if str(event.get("policy_text", "")).strip()
    )
    package_dict: Dict[str, Any] = {}
    if combined_policy_text.strip():
        _, package = _compile_policy_bundle(
            combined_policy_text,
            float(session.get("intensity", 1.0)),
            policy_type_hint=str(session.get("policy_type", "自定义政策")),
        )
        package_dict = package.to_dict()
        session["policy_package"] = package_dict
    narrative = _get_policy_narrative(
        combined_policy_text or str(session.get("policy_text", "")),
        summary,
        package_dict or {"explanation": {}, "policy_schema": {}, "top_layers": {}},
        runtime_profile,
    )
    return {
        "title": f"政策试验台 - {session.get('policy_name', '政策仿真')}",
        "summary": summary,
        "policy_name": session.get("policy_name", "政策仿真"),
        "policy_text": session.get("policy_text", ""),
        "policy_type": session.get("policy_type", "自定义政策"),
        "timeline": _policy_session_timeline(session),
        "runtime_events": {
            "active": list(session.get("runtime_active_events", []) or []),
            "queued": list(session.get("runtime_queued_events", []) or []),
            "expired": list(session.get("runtime_expired_events", []) or []),
            "timeline": list(session.get("runtime_event_timeline", []) or []),
            "digest": dict(session.get("event_digest", {}) or {}),
        },
        "frame": frame.to_dict(orient="records"),
        "policy_package": package_dict,
        "discovered_metrics": dict(session.get("discovered_metrics", {}) or _policy_session_discovered_metrics(session, frame)),
        "runtime_mode": runtime_profile.mode,
        "runtime_profile": runtime_profile.to_dict(),
        "narrative": narrative,
        "impact_evaluation": {
            "overall_verdict": f"会话累计收益率 {summary.get('return_pct', 0.0):.2%}，最大回撤 {summary.get('max_drawdown', 0.0):.2%}。",
            "impact_direction": "正向"
            if float(summary.get("return_pct", 0.0) or 0.0) > 0
            else ("负向" if float(summary.get("return_pct", 0.0) or 0.0) < 0 else "中性"),
            "impact_level": "中",
            "transmission_mechanism_evidence": [
                "观察政策事件时间轴与指数路径同向/反向变化。",
                "结合恐慌度、政策压力等行为指标进行传导验证。",
            ],
            "market_feedback": "请结合会话日度轨迹与政策时间轴复核。",
        },
        "risk_assessment": {
            "risk_level": "中",
            "key_risks": ["政策预期偏差风险", "外部冲击扰动风险"],
            "side_effects": ["阶段性波动抬升", "结构性分化加剧"],
        },
        "action_recommendations": {
            "short_term": ["维持政策沟通连续性并滚动校准强度。"],
            "mid_term": ["跟踪跨市场传导指标，降低单指标误判。"],
            "long_term": ["建立政策-市场反馈闭环并持续迭代机制。"],
        },
        "monitoring_kpis": [
            {"name": "累计收益率", "value": float(summary.get("return_pct", 0.0) or 0.0)},
            {"name": "最大回撤", "value": float(summary.get("max_drawdown", 0.0) or 0.0)},
            {"name": "波动率", "value": float(summary.get("volatility", 0.0) or 0.0)},
            {"name": "平均恐慌度", "value": float(summary.get("avg_panic", 0.0) or 0.0)},
        ],
        "session": {
            "session_id": session.get("session_id", ""),
            "status": session.get("status", "idle"),
            "current_day": int(session.get("current_day", 0)),
            "total_days": int(session.get("total_days", 0)),
            "calendar_start": session.get("calendar_start", ""),
            "index_label": session.get("index_label", ""),
            "index_symbol": session.get("index_symbol", ""),
        },
        "experiment_registry": dict(session.get("experiment_registry", {}) or {}),
        "reproducibility": dict(session.get("reproducibility", {}) or {}),
        "scorecard": dict(session.get("scorecard", {}) or {}),
        "scenario_diff": dict(session.get("scenario_diff", {}) or {}),
        "kline": {
            "source": "trade_tape_aggregation",
            "benchmark": session.get("index_symbol", "sh000001"),
            "ohlcv": list(session.get("workbench_ohlcv_rows", []) or []),
        },
    }


def _research_workbench_report_appendix(payload: Dict[str, Any]) -> str:
    summary = dict(payload.get("summary", {}) or {})
    session_meta = dict(payload.get("session", {}) or {})
    kline = dict(payload.get("kline", {}) or {})
    scorecard = dict(payload.get("scorecard", {}) or {})
    repro = dict(payload.get("reproducibility", {}) or {})
    registry = dict(payload.get("experiment_registry", {}) or {})
    policy_package = dict(payload.get("policy_package", {}) or {})
    scenario_diff = dict(payload.get("scenario_diff", {}) or {})
    path_fit = dict(scorecard.get("path_fit_metrics", {}) or {})
    risk = dict(scorecard.get("risk_metrics", {}) or {})
    lines = [
        "",
        "## 研究工作台附录",
        "",
        "### 项目标题",
        f"- {payload.get('title', '政策试验台研究报告')}",
        "",
        "### 政策文本",
        f"- {payload.get('policy_text', '')}",
        "",
        "### 结构化政策包",
        "```json",
        json.dumps(policy_package, ensure_ascii=False, indent=2, sort_keys=True, default=str)[:6000],
        "```",
        "",
        "### 历史窗口与上证指数对照",
        f"- 历史窗口：{session_meta.get('calendar_start', '')} 至 当前第 {session_meta.get('current_day', 0)} 个交易日",
        f"- 上证指数对照：{kline.get('benchmark', session_meta.get('index_symbol', 'sh000001'))}",
        f"- K 线来源：{kline.get('source', 'trade_tape_aggregation')}",
        f"- K 线图：报告 JSON 中 `kline.ohlcv` 保留主图 OHLCV，前端以逐笔成交聚合渲染。",
        "",
        "### 关键指标",
        f"- 累计收益率：{float(summary.get('return_pct', 0.0) or 0.0):+.2%}",
        f"- 最大回撤：{float(summary.get('max_drawdown', 0.0) or 0.0):.2%}",
        f"- 波动率：{float(summary.get('volatility', 0.0) or 0.0):.4f}",
        f"- 平均恐慌度：{float(summary.get('avg_panic', 0.0) or 0.0):.4f}",
        "",
        "### 风险传播链",
        f"- 路径跟踪误差：{float(path_fit.get('tracking_rmse', path_fit.get('normalized_rmse', 0.0)) or 0.0):.4f}",
        f"- 方向命中率：{float(path_fit.get('direction_hit_rate', 0.0) or 0.0):.2%}",
        f"- 仿真最大回撤：{float(risk.get('sim_max_drawdown', risk.get('max_drawdown', 0.0)) or 0.0):.2%}",
        "",
        "### 多智能体行为解释",
        f"- 当前评估卡行为指标：{json.dumps(scorecard.get('behavioral_metrics', {}), ensure_ascii=False, sort_keys=True, default=str)}",
        "",
        "### 监管优化结果",
        f"- 场景差异表数量：{len(scenario_diff.get('deltas', []) or [])}",
        f"- 推荐/优化路径：{payload.get('deep_mode_meta', {}).get('counterfactual_regulation', {}).get('recommended_timing', '-') if isinstance(payload.get('deep_mode_meta'), dict) else '-'}",
        "",
        "### 结论与建议",
        f"- {payload.get('impact_evaluation', {}).get('overall_verdict', '请结合主图、事件层和评估卡进行复核。')}",
        "",
        "### 可复现信息",
        f"- 实验编号：{registry.get('experiment_id', '')}",
        f"- 配置哈希：{repro.get('config_hash', registry.get('config_hash', ''))}",
        f"- 数据快照哈希：{repro.get('data_snapshot_hash', registry.get('data_snapshot_id', ''))}",
        f"- 代码提交哈希：{repro.get('git_commit_hash', '')}",
        f"- 随机种子：{repro.get('random_seed', registry.get('seed', 42))}",
        f"- 大模型调用链：{', '.join(map(str, repro.get('llm_provider_chain', []) or []))}",
        f"- 校准参数集：{repro.get('calibration_parameter_set_id', '')}",
    ]
    return "\n".join(lines)


def _policy_session_generate_report(session: Dict[str, Any], runtime_profile: RuntimeModeProfile) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        report = runner.generate_report(use_llm=bool(runtime_profile.use_live_api))
        report_markdown = _sanitize_report_markdown_for_frontend(str(report.get("报告正文", "")))
        payload = _policy_session_report_payload(session, runtime_profile)
        report_markdown = _sanitize_report_markdown_for_frontend(
            "\n\n".join([report_markdown, _research_workbench_report_appendix(payload)])
        )
        report_meta = official_report_meta("policy_lab_session", str(payload["title"]))
        report_bundle = write_report_artifacts(
            root_dir=POLICY_REPORT_DIR / "session_reports",
            report_type="policy_lab_session",
            title=str(payload["title"]),
            markdown_text=report_markdown,
            payload=payload,
        )
        payload["report_meta"] = report_meta
        payload["report_bundle"] = report_bundle
        session["report_bundle"] = report_bundle
        session["report_payload"] = payload
        return payload
    payload = _policy_session_report_payload(session, runtime_profile)
    report_meta = official_report_meta("policy_lab_session", str(payload["title"]))
    frame = pd.DataFrame(payload["frame"])
    markdown_lines = [
        f"# {payload['title']}",
        "",
        f"- 报告编号：{report_meta['report_no']}",
        f"- 生成日期：{report_meta['date_cn']}",
        f"- 会话状态：{_policy_session_status_text(str(session.get('status', 'idle')))}",
        f"- 当前交易日：{int(session.get('current_day', 0))}/{int(session.get('total_days', 0))}",
        f"- 指数基准：{session.get('index_label', '')}（{session.get('index_symbol', '')}）",
        f"- 运行模式：{_runtime_mode_text(runtime_profile)}",
        "",
        "## 会话摘要",
        f"- 累计涨跌：{payload['summary'].get('return_pct', 0.0):.2%}",
        f"- 最大回撤：{payload['summary'].get('max_drawdown', 0.0):.2%}",
        f"- 平均恐慌度：{payload['summary'].get('avg_panic', 0.0):.4f}",
        f"- 波动率：{payload['summary'].get('volatility', 0.0):.4f}",
        f"- 平均政策压力：{payload['summary'].get('policy_signal_avg', 0.0):.4f}",
        "",
        "## 一句话结论",
        f"- {payload.get('impact_evaluation', {}).get('overall_verdict', '会话已完成，请结合指标评估政策效果。')}",
        "",
        "## 政策时间轴",
    ]
    for item in payload["timeline"]:
        markdown_lines.append(
            f"- {item['政策名称']}：{item['当前状态']}，生效日 {item['生效交易日']}，剩余影响 {item['剩余影响']:.4f}"
        )
    if payload["narrative"]:
        markdown_lines.extend(["", "## 大模型评估", payload["narrative"]])
    impact = dict(payload.get("impact_evaluation", {}) or {})
    risk = dict(payload.get("risk_assessment", {}) or {})
    recommendations = dict(payload.get("action_recommendations", {}) or {})
    kpis = list(payload.get("monitoring_kpis", []) or [])
    markdown_lines.extend(
        [
            "",
            "## 政策影响评价",
            f"- 影响方向：{impact.get('impact_direction', '中性')}",
            f"- 影响等级：{impact.get('impact_level', '中')}",
            f"- 市场反馈：{impact.get('market_feedback', '')}",
            "",
            "## 传导机制证据",
        ]
    )
    for item in impact.get("transmission_mechanism_evidence", []) or []:
        markdown_lines.append(f"- {item}")
    markdown_lines.extend(["", "## 风险与副作用", f"- 风险等级：{risk.get('risk_level', '中')}", "- 关键风险："])
    for item in risk.get("key_risks", []) or []:
        markdown_lines.append(f"  - {item}")
    markdown_lines.append("- 潜在副作用：")
    for item in risk.get("side_effects", []) or []:
        markdown_lines.append(f"  - {item}")
    markdown_lines.extend(["", "## 分阶段建议", "- 短期建议："])
    for item in recommendations.get("short_term", []) or []:
        markdown_lines.append(f"  - {item}")
    markdown_lines.append("- 中期建议：")
    for item in recommendations.get("mid_term", []) or []:
        markdown_lines.append(f"  - {item}")
    markdown_lines.append("- 长期建议：")
    for item in recommendations.get("long_term", []) or []:
        markdown_lines.append(f"  - {item}")
    markdown_lines.extend(["", "## 关键监测指标"])
    for item in kpis:
        if isinstance(item, dict):
            display = item.get("display", item.get("value", ""))
            markdown_lines.append(f"- {item.get('name', '指标')}：{display}")
    markdown_lines.extend(
        [
            "",
            "## 风险提示",
            "- 该会话报告用于教学科研与仿真展示。",
            "- 追加政策会影响后续交易日的价格路径与情绪轨迹。",
            "- 政策影响会随时间自然衰减。",
        ]
    )
    markdown_lines.append(_research_workbench_report_appendix(payload))
    report_markdown = _sanitize_report_markdown_for_frontend("\n".join(markdown_lines))
    report_bundle = write_report_artifacts(
        root_dir=POLICY_REPORT_DIR / "session_reports",
        report_type="policy_lab_session",
        title=str(payload["title"]),
        markdown_text=report_markdown,
        payload=payload,
    )
    payload["report_meta"] = report_meta
    payload["report_bundle"] = report_bundle
    session["report_bundle"] = report_bundle
    session["report_payload"] = payload
    return payload


def _policy_session_policy_package(session: Dict[str, Any]) -> Dict[str, Any]:
    combined_policy_text = "\n".join(
        str(event.get("政策文本", event.get("policy_text", "")))
        for event in session.get("policy_events", [])
        if str(event.get("政策文本", event.get("policy_text", ""))).strip()
    )
    if not combined_policy_text.strip():
        return {}
    _, package = _compile_policy_bundle(
        combined_policy_text,
        float(session.get("intensity", 1.0)),
        policy_type_hint=str(session.get("policy_type", "自定义政策")),
    )
    session["policy_package"] = package.to_dict()
    return package.to_dict()


def _policy_eval_payload(summary: Dict[str, Any], package_dict: Dict[str, Any]) -> Dict[str, Any]:
    event = dict(package_dict.get("event", {}) or {})
    explanation = dict(package_dict.get("explanation", {}) or {})
    uncertainty = dict(package_dict.get("uncertainty", {}) or {})
    return_pct = float(summary.get("return_pct", 0.0) or 0.0)
    max_panic = float(summary.get("max_panic", 0.0) or 0.0)
    volatility = float(summary.get("volatility", 0.0) or 0.0)
    drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)
    if max_panic >= 0.75 or drawdown >= 0.08:
        risk_level = "高"
    elif max_panic >= 0.45 or drawdown >= 0.04:
        risk_level = "中"
    else:
        risk_level = "低"
    if return_pct > 0.03:
        impact_direction = "正向"
        impact_level = "强"
    elif return_pct < -0.03:
        impact_direction = "负向"
        impact_level = "强"
    else:
        impact_direction = "中性" if abs(return_pct) < 0.01 else ("正向" if return_pct > 0 else "负向")
        impact_level = "中"
    return {
        "impact_evaluation": {
            "overall_verdict": explanation.get("headline") or f"会话累计收益率 {return_pct:.2%}，最大回撤 {drawdown:.2%}。",
            "impact_direction": impact_direction,
            "impact_level": impact_level,
            "market_feedback": f"风险热度峰值 {max_panic:.2f}，会话波动率 {volatility:.2%}。",
            "transmission_mechanism_evidence": explanation.get("primary_path", []),
        },
        "risk_assessment": {
            "risk_level": risk_level,
            "key_risks": [
                f"政策不确定性 {float(uncertainty.get('ambiguity', 0.0) or 0.0):.2f}",
                f"风险热度峰值 {max_panic:.2f}",
                f"最大回撤 {drawdown:.2%}",
            ],
            "side_effects": list(event.get("side_effects", []) or explanation.get("side_effects", []) or ["波动抬升", "结构分化"]),
        },
        "action_recommendations": {
            "short_term": ["关注政策生效初期的风险热度与成交反应。"],
            "mid_term": ["结合板块轮动、角色订单流和回撤变化做阶段性复核。"],
            "long_term": ["将当前场景输出纳入历史验证与监管优化链路继续验证。"],
        },
        "monitoring_kpis": [
            {"name": "累计收益率", "display": f"{return_pct:+.2%}"},
            {"name": "最大回撤", "display": f"{drawdown:.2%}"},
            {"name": "波动率", "display": f"{volatility:.2%}"},
            {"name": "平均恐慌度", "display": f"{float(summary.get('avg_panic', 0.0) or 0.0):.2f}"},
        ],
    }


def _render_policy_package_summary(package_dict: Dict[str, Any], summary: Dict[str, Any]) -> None:
    if not package_dict:
        st.info("当前暂无可展示的结构化政策包。")
        return

    explanation = dict(package_dict.get("explanation", {}) or {})
    event = dict(package_dict.get("event", {}) or {})
    uncertainty = dict(package_dict.get("uncertainty", {}) or {})
    top_layers = dict(package_dict.get("top_layers", {}) or {})
    payload = _policy_eval_payload(summary, package_dict)

    headline = translate_display_text(str(explanation.get("headline", "") or "已完成政策结构化解析。"))
    primary_path = " -> ".join(translate_display_text(str(item)) for item in list(explanation.get("primary_path", []) or [])[:6])
    if not primary_path:
        primary_path = "政策输入 -> 传导渠道 -> 智能体行为 -> 市场结果"

    st.markdown("### 大模型结构化解读")
    lead_left, lead_right = st.columns([1.25, 1.0])
    with lead_left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">核心判断</div>
              <div class="summary-value">{headline}</div>
              <div class="summary-note">主要传导路径：{primary_path}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lead_right:
        metric_cols = st.columns(3)
        metric_cols[0].metric("解析置信度", f"{float(uncertainty.get('confidence', 0.0) or 0.0):.0%}")
        metric_cols[1].metric("预期滞后", f"{int(event.get('expected_lag_days', 0) or 0)} 天")
        metric_cols[2].metric("副作用提示", f"{len(list(event.get('side_effects', []) or []))} 项")

    insight_left, insight_right = st.columns([1.25, 1.0])
    with insight_left:
        layer_tabs = st.tabs(["行业映射", "因子映射", "主体映射"])
        layer_specs = [
            ("sector", "行业", layer_tabs[0]),
            ("factor", "因子", layer_tabs[1]),
            ("agent_class", "主体", layer_tabs[2]),
        ]
        for key, label, tab in layer_specs:
            with tab:
                rows = list(top_layers.get(key, []) or [])
                if rows:
                    frame = pd.DataFrame(
                        [(translate_display_text(str(name)), value) for name, value in rows],
                        columns=[label, "影响强度"],
                    )
                    st.dataframe(frame, use_container_width=True, hide_index=True)
                else:
                    st.info(f"暂无{label}映射结果。")
    with insight_right:
        st.markdown("#### 影响评估")
        st.markdown(
            f"""
            <div class="story-card">
              <div class="story-card-title">{payload['impact_evaluation']['overall_verdict']}</div>
              <div class="story-card-summary">方向：{payload['impact_evaluation']['impact_direction']} | 等级：{payload['impact_evaluation']['impact_level']}</div>
              <ul class="story-card-list">
                <li>{payload['impact_evaluation']['market_feedback']}</li>
                <li>风险等级：{payload['risk_assessment']['risk_level']}</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 关键监测指标")
        for item in payload["monitoring_kpis"]:
            st.markdown(f"- {item['name']}：{item['display']}")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("#### 风险与副作用")
        st.markdown("\n".join(f"- {item}" for item in payload["risk_assessment"]["key_risks"]))
        side_effects = payload["risk_assessment"]["side_effects"]
        if side_effects:
            st.caption("潜在副作用")
            st.markdown("\n".join(f"- {item}" for item in side_effects))
    with detail_right:
        st.markdown("#### 建议动作")
        action_recommendations = payload["action_recommendations"]
        for title, items in (
            ("短期", action_recommendations["short_term"]),
            ("中期", action_recommendations["mid_term"]),
            ("长期", action_recommendations["long_term"]),
        ):
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {item}" for item in items))
    render_narrative_block(
        "大模型结构化政策解读",
        {
            "headline": headline,
            "primary_path": primary_path,
            "top_layers": top_layers,
            "impact_evaluation": payload["impact_evaluation"],
            "risk_assessment": payload["risk_assessment"],
            "action_recommendations": payload["action_recommendations"],
            "monitoring_kpis": payload["monitoring_kpis"],
        },
        context="请用政策研判口径解释这项政策的主要传导路径、预期影响、风险和建议动作。",
        cache_namespace="policy_lab_structured_narrative_cache",
    )
    with st.expander("结构化数据载荷", expanded=False):
        st.json(package_dict, expanded=False)


def _policy_session_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mapping = {
        "step": "交易日序号",
        "time": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "panic_level": "恐慌度",
        "csad": "羊群度",
    }
    display = frame.rename(columns=mapping)
    if "volume" in display.columns:
        display = display.drop(columns=["volume"], errors="ignore")
    if "成交量" in display.columns:
        display = display.drop(columns=["成交量"], errors="ignore")
    return display


def _render_discovered_metrics_panel(discovered_metrics: Dict[str, Any]) -> None:
    payload = dict(discovered_metrics or {})
    top_metrics = list(payload.get("top_metrics", []) or payload.get("ranked_metrics", [])[:8])
    pareto = list(payload.get("pareto_frontier", []) or [])
    if not top_metrics:
        st.info("目标发现将在生成市场路径后显示。")
        return

    st.markdown("### 目标发现与指标组合")
    score = float(payload.get("composite_score", payload.get("composite_policy_score", 0.0)) or 0.0)
    weights = dict(payload.get("weight_decomposition", {}) or {})
    shanghai = dict(payload.get("shanghai_index_metric", {}) or {})
    c1, c2, c3 = st.columns(3)
    c1.metric("政策综合评分", f"{score:.3f}")
    c2.metric("候选指标池", int(len(payload.get("candidate_pool", []) or [])))
    c3.metric("上证指数关系", zh_value(str(shanghai.get("relation_to_shanghai_index", "self") or "self")))

    metric_frame = pd.DataFrame(top_metrics)
    display_cols = [
        col
        for col in [
            "name",
            "category",
            "rank_score",
            "composite_weight",
            "policy_sensitivity",
            "robustness",
            "early_warning_utility",
            "relation_to_shanghai_index",
        ]
        if col in metric_frame.columns
    ]
    if display_cols:
        display_frame = metric_frame[display_cols].copy()
        if "name" in display_frame.columns:
            display_frame["name"] = display_frame["name"].map(lambda value: zh_metric_name(str(value)))
        st.dataframe(localize_dataframe_columns(display_frame), use_container_width=True, hide_index=True)

    if pareto:
        pareto_frame = pd.DataFrame(pareto)
        if {"policy_sensitivity", "robustness", "name"}.issubset(pareto_frame.columns):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pareto_frame["policy_sensitivity"],
                    y=pareto_frame["robustness"],
                    mode="markers+text",
                    text=pareto_frame["name"].map(lambda value: zh_metric_name(str(value))),
                    textposition="top center",
                    marker=dict(
                        size=10 + 24 * pd.to_numeric(pareto_frame.get("early_warning_utility", 0.0), errors="coerce").fillna(0.0),
                        color=pd.to_numeric(pareto_frame.get("rank_score", 0.0), errors="coerce").fillna(0.0),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="综合排序"),
                    ),
                )
            )
            fig.update_layout(
                title="帕累托前沿：敏感性 × 稳健性",
                xaxis_title="政策敏感性",
                yaxis_title="跨场景稳健性",
                height=320,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=48, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    if weights:
        weight_frame = pd.DataFrame(
            [{"指标名称": zh_metric_name(str(key)), "综合权重": float(value)} for key, value in sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)]
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=weight_frame["综合权重"],
                    y=weight_frame["指标名称"],
                    orientation="h",
                    marker_color="#38bdf8",
                    name="综合权重",
                )
            ]
        )
        fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="指标重要性",
            xaxis_title="综合权重",
            yaxis_title="指标名称",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    render_narrative_block(
        "目标发现与指标组合",
        {
            "composite_score": score,
            "top_metrics": top_metrics[:5],
            "shanghai_index_metric": shanghai,
            "candidate_pool": payload.get("candidate_pool", []),
        },
        context="请解释为什么以上证指数为重点指标，以及补充指标如何增强政策敏感性、稳健性和预警价值。",
        cache_namespace="policy_lab_narrative_cache",
    )


def _compute_policy_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    close = metrics["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    drawdown = close / close.cummax() - 1.0
    return {
        "return_pct": float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0),
        "avg_panic": float(metrics["panic_level"].mean()),
        "max_panic": float(metrics["panic_level"].max()),
        "avg_csad": float(metrics["csad"].mean()),
        "max_drawdown": float(abs(drawdown.min())),
        "avg_volume": float(metrics["volume"].mean()),
        "volatility": float(returns.std()),
    }


def _build_chart(frame: pd.DataFrame, *, chart_title: str = "指数日K图（东方财富风格）", mode: str = "日K") -> go.Figure:
    chart = frame.copy()
    for window in (5, 10, 20, 30):
        chart[f"ma{window}"] = chart["close"].rolling(window).mean()
    ma_styles = {
        "ma5": {"label": "5日均线", "color": "#f5a623", "width": 1.35},
        "ma10": {"label": "10日均线", "color": "#3a78d4", "width": 1.35},
        "ma20": {"label": "20日均线", "color": "#8e44ad", "width": 1.2},
        "ma30": {"label": "30日均线", "color": "#4a5568", "width": 1.15},
    }

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    if mode == "分时":
        chart["avg_price"] = chart["close"].expanding().mean()
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["close"],
                mode="lines",
                name="分时",
                line=dict(color="#2f6fed", width=1.9),
                hovertemplate="时间=%{x}<br>价格=%{y:.2f}<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["avg_price"],
                mode="lines",
                name="均价",
                line=dict(color="#f5a623", width=1.15),
                hovertemplate="时间=%{x}<br>均价=%{y:.2f}<extra></extra>",
            ),
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=chart["time"],
                open=chart["open"],
                high=chart["high"],
                low=chart["low"],
                close=chart["close"],
                name="日K",
                increasing_line_color="#d63b3b",
                decreasing_line_color="#1c9b63",
                increasing_fillcolor="#d63b3b",
                decreasing_fillcolor="#1c9b63",
                whiskerwidth=0.55,
                hoverlabel=dict(font=dict(family="Microsoft YaHei")),
            ),
        )
        for key, meta in ma_styles.items():
            fig.add_trace(
                go.Scatter(
                    x=chart["time"],
                    y=chart[key],
                    mode="lines",
                    name=meta["label"],
                    line=dict(color=meta["color"], width=meta["width"]),
                    hovertemplate=f"时间=%{{x}}<br>{meta['label']}=%{{y:.2f}}<extra></extra>",
                ),
            )
    fig.update_layout(
        margin=dict(l=12, r=8, t=34, b=8),
        title=dict(text=chart_title, x=0.01, xanchor="left", font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="#060b14",
        plot_bgcolor="#0b1220",
        font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=12, color="#dbe6f5"),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.0,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(8,15,28,0.88)",
            bordercolor="#233247",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        hoverlabel=dict(bgcolor="rgba(43,51,63,0.96)", bordercolor="#2b333f", font=dict(color="#ffffff")),
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,
        showline=True,
        linewidth=1,
        linecolor="#243244",
        mirror=False,
        tickfont=dict(color="#9fb3c8"),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#64748b",
        spikethickness=1,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="指数点位",
        row=1,
        col=1,
        side="right",
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#243244",
        tickfont=dict(color="#9fb3c8"),
        nticks=8,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#838a97",
        spikethickness=1,
    )
    return fig


def _render_quote_banner(frame: pd.DataFrame, *, index_label: str, index_symbol: str, source_text: str, history_end: str) -> None:
    if frame.empty:
        return
    latest = frame.iloc[-1]
    prev_close = float(frame.iloc[-2]["close"]) if len(frame) > 1 else float(latest["open"])
    last_close = float(latest["close"])
    change = last_close - prev_close
    pct = change / prev_close if abs(prev_close) > 1e-9 else 0.0
    color = "#d9383a" if change >= 0 else "#18a058"
    sign = "+" if change >= 0 else ""
    st.markdown(
        (
            "<div style='border:1px solid #223247;border-radius:12px;padding:12px 16px;background:linear-gradient(180deg,rgba(11,18,32,0.98) 0%,rgba(7,12,22,0.98) 100%);box-shadow:0 12px 30px rgba(0,0,0,0.28);'>"
            "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;'>"
            "<div>"
            f"<div style='font-size:13px;color:#7dd3fc;font-weight:600;'>{index_label}（{index_symbol}）</div>"
            f"<div style='font-size:11px;color:#7c8ca1;margin-top:2px;'>{source_text} · 截止 {history_end}</div>"
            f"<div style='font-size:30px;font-weight:700;color:{color};line-height:1.15;margin-top:6px;'>{last_close:.2f}</div>"
            f"<div style='font-size:14px;color:{color};font-weight:600;margin-top:4px;'>{sign}{change:.2f} &nbsp;&nbsp; {sign}{pct:.2%}</div>"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(3,minmax(78px,1fr));gap:8px;flex:1;min-width:240px;'>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>今开</div><div style='font-size:15px;color:#e2e8f0;font-weight:600;'>{float(latest['open']):.2f}</div></div>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>最高</div><div style='font-size:15px;color:#f87171;font-weight:600;'>{float(latest['high']):.2f}</div></div>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>最低</div><div style='font-size:15px;color:#4ade80;font-weight:600;'>{float(latest['low']):.2f}</div></div>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _policy_narrative_key(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    payload = {
        "policy_text": str(policy_text or ""),
        "summary": dict(summary or {}),
        "explanation": dict(package_dict.get("explanation", {}) or {}),
        "top_layers": dict(package_dict.get("top_layers", {}) or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _top_effect_text(items: List[Dict[str, Any]], limit: int = 3) -> str:
    if not items:
        return "暂无显著项"
    ranked = sorted(items, key=lambda item: abs(float(item.get("score", 0.0) or 0.0)), reverse=True)[:limit]
    parts: List[str] = []
    for item in ranked:
        name = str(item.get("name", "未命名"))
        score = float(item.get("score", 0.0) or 0.0)
        direction = "受益" if score >= 0 else "承压"
        parts.append(f"{name}（{direction}）")
    return "、".join(parts)


def _fallback_policy_narrative(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    explanation = dict(package_dict.get("explanation", {}) or {})
    headline = str(explanation.get("headline", "") or "政策通过多条传导链路影响市场预期与资金行为。")
    primary_path = [str(node) for node in explanation.get("primary_path", []) if str(node).strip()]
    path_text = " -> ".join(primary_path[:5]) if primary_path else "政策信号 -> 预期修复 -> 风险偏好变化 -> 价格反馈"
    lag_days = int(explanation.get("expected_lag_days", 0) or 0)
    side_effects = [str(item) for item in explanation.get("side_effects", []) if str(item).strip()]
    risk_tip = "、".join(side_effects[:2]) if side_effects else "短期波动可能放大，需关注情绪过冲。"
    policy_core = str(policy_text or "").strip()
    if len(policy_core) > 140:
        policy_core = f"{policy_core[:140]}..."
    return "\n\n".join(
        [
            f"从政策主线看，这次输入的政策重点是“{policy_core or '未提供政策文本'}”，整体判断是：{headline}",
            (
                f"在传导机制上，政策影响会沿着“{path_text}”逐步扩散，首先改变关键主体的交易预期，"
                f"再通过资金再配置影响板块和指数表现。当前受影响较明显的主体包括"
                f"{_top_effect_text(list(explanation.get('affected_agents', []) or []))}，"
                f"受影响较明显的行业包括{_top_effect_text(list(explanation.get('affected_sectors', []) or []))}。"
            ),
            (
                f"从仿真结果看，区间收益约为 {summary.get('return_pct', 0.0):.2%}，最大回撤约为 "
                f"{summary.get('max_drawdown', 0.0):.2%}，波动水平约为 {summary.get('volatility', 0.0):.4f}。"
                f"预计主要影响会在约 {lag_days} 天内逐步显现，需重点关注：{risk_tip}"
            ),
        ]
    )


def _llm_policy_narrative(
    policy_text: str,
    summary: Dict[str, float],
    package_dict: Dict[str, Any],
    runtime_profile: RuntimeModeProfile,
) -> str:
    if not runtime_profile.use_live_api:
        return ""
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        return ""
    snapshot = {
        "policy_text": str(policy_text or ""),
        "summary": dict(summary or {}),
        "explanation": dict(package_dict.get("explanation", {}) or {}),
        "policy_schema": dict(package_dict.get("policy_schema", {}) or {}),
        "top_layers": dict(package_dict.get("top_layers", {}) or {}),
    }
    prompt = "\n".join(
        [
            "请基于以下政策仿真信息，输出一段面向政策制定者的中文后果研判。",
            "输出要求：",
            "1) 只用中文段落，不要 JSON、代码块、键值对、项目符号、标题标签。",
            "2) 不要复述数据清单；要解释政策主线如何改变预期、订单流、板块轮动、指数路径和风险热度。",
            "3) 必须指出至少一个政策实施后可能出现的情景、触发条件或副作用，并说明政策制定者应监测的指标。",
            "4) 语言要清晰、专业、易懂，可直接用于政策研判、结果复盘或系统说明。",
            "5) 避免英文缩写和术语堆砌，避免套话式结尾。",
            f"数据：{json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)}",
        ]
    )
    try:
        model_name = str(runtime_profile.model_priority[0]) if runtime_profile.model_priority else "deepseek-chat"
        backend = APIBackend(model=model_name, max_tokens=520, temperature=0.3)
        response = str(
            backend.generate(
                prompt,
                system_prompt=(
                    "你是政策评估讲解专家，请从仿真数据关系推导政策后果，用自然、严谨、可解释的中文表述。"
                ),
                timeout_budget=20.0,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    if not response or response.startswith("[API Error]"):
        return ""
    if response.lstrip().startswith("{") or response.lstrip().startswith("["):
        return ""
    return response


def _get_policy_narrative(
    policy_text: str,
    summary: Dict[str, float],
    package_dict: Dict[str, Any],
    runtime_profile: RuntimeModeProfile,
) -> str:
    cache = st.session_state.setdefault("policy_lab_narrative_cache", {})
    key = f"{runtime_profile.mode}:{_policy_narrative_key(policy_text, summary, package_dict)}"
    if key in cache:
        return str(cache[key])
    text = _llm_policy_narrative(policy_text, summary, package_dict, runtime_profile)
    if not text:
        text = _fallback_policy_narrative(policy_text, summary, package_dict)
    cache[key] = text
    return text


def _persist_policy_event(
    *,
    selected_title: str,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    index_symbol: str = "",
    data_source: str = "",
) -> None:
    timestamp = pd.Timestamp.utcnow().isoformat()
    record = EventRecord(
        timestamp=timestamp,
        visibility_time=timestamp,
        source="policy_lab_ui",
        confidence=1.0,
        event_type=EventType.POLICY,
        payload={
            "title": f"政策场景：{selected_title}",
            "policy_text": str(policy_text),
            "intensity": float(intensity),
            "duration_days": int(duration_days),
            "rumor_noise": bool(rumor_noise),
            "index_symbol": str(index_symbol or ""),
            "data_source": str(data_source or ""),
        },
        metadata={"module": "policy_lab"},
    )
    try:
        EventStore().append_events(dataset_version="policy_lab", events=[record])
    except Exception:

        return


ROLE_TRANSLATIONS = {
    "retail_day_trader": "散户短线",
    "retail_swing": "散户波段",
    "retail_momentum_chaser": "散户趋势",
    "mutual_fund": "公募基金",
    "quant_arbitrage": "量化机构",
    "market_maker": "做市资金",
    "foreign_capital": "外资",
    "insurance": "险资",
    "national_team": "平准基金",
}

ENGLISH_TOKEN_TRANSLATIONS = {
    "retail": "散户",
    "institutional": "机构",
    "institution": "机构",
    "quant": "量化",
    "fund": "基金",
    "bank": "银行",
    "insurance": "险资",
    "foreign": "外资",
    "northbound": "北向",
    "market": "市场",
    "maker": "做市",
    "trader": "交易",
    "swing": "波段",
    "day": "日内",
    "momentum": "动量",
    "chaser": "追涨",
    "team": "团队",
    "committee": "委员会",
    "smart": "聪明",
    "money": "资金",
    "hedge": "对冲",
    "private": "私募",
    "state": "国资",
    "policy": "政策",
    "sector": "板块",
    "technology": "科技",
    "tech": "科技",
    "finance": "金融",
    "broker": "券商",
    "banking": "银行",
    "real": "地产",
    "estate": "地产",
    "property": "地产",
    "consumer": "消费",
    "retailer": "零售",
    "manufacturing": "制造",
    "industry": "工业",
    "industrial": "工业",
    "energy": "能源",
    "materials": "材料",
    "utility": "公用事业",
    "utilities": "公用事业",
    "healthcare": "医药",
    "medical": "医药",
    "pharma": "医药",
    "communication": "通信",
    "media": "传媒",
    "internet": "互联网",
    "auto": "汽车",
    "agriculture": "农业",
    "transport": "交通",
    "logistics": "物流",
    "semiconductor": "半导体",
    "chip": "芯片",
    "ai": "人工智能",
    "new": "新",
}

SECTOR_TRANSLATIONS = {
    "real_estate": "房地产",
    "new_energy": "新能源",
    "non_bank_financial": "非银金融",
    "consumer_staples": "必选消费",
    "consumer_discretionary": "可选消费",
    "information_technology": "信息技术",
}


def _translate_english_tag(value: str, *, default_prefix: str = "类别") -> str:
    raw = str(value or "").strip()
    if not raw:
        return default_prefix
    lowered = raw.lower()
    if lowered in ENGLISH_TOKEN_TRANSLATIONS:
        return ENGLISH_TOKEN_TRANSLATIONS[lowered]
    parts = [part for part in re.split(r"[_\-\s/]+", lowered) if part]
    if not parts:
        return raw
    translated_parts: List[str] = []
    translated_any = False
    for part in parts:
        translated = ENGLISH_TOKEN_TRANSLATIONS.get(part, part)
        translated_parts.append(translated)
        if translated != part:
            translated_any = True
    if translated_any:
        return "".join(translated_parts)
    return raw


def _translate_sector_name(name: str) -> str:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return "其他板块"
    if lowered in SECTOR_TRANSLATIONS:
        return SECTOR_TRANSLATIONS[lowered]
    return _translate_english_tag(str(name), default_prefix="板块")


def _translate_role(role: str) -> str:
    cleaned = str(role).lower().replace("_00", "").replace("_01", "").replace("_02", "").replace("_03", "").strip()
    if cleaned in ROLE_TRANSLATIONS:
        return ROLE_TRANSLATIONS[cleaned]
    return _translate_english_tag(str(role), default_prefix="角色")


def _get_llm_visual_interpretation(prompt: str, cache_key: str, runtime_profile: RuntimeModeProfile) -> str:
    cache = st.session_state.setdefault("policy_lab_llm_visual_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    if not runtime_profile.use_live_api:
        return ""
    model_candidates = list(runtime_profile.model_priority or ("deepseek-chat", "glm-4-flashx"))
    for model_name in model_candidates:
        try:
            from core.inference.api_backend import APIBackend
            backend = APIBackend(model=str(model_name), max_tokens=240, temperature=0.35)
            response = str(
                backend.generate(
                    prompt,
                    system_prompt=(
                        "你是政策仿真图表研判专家。用120到180字中文说明图表背后的政策含义、"
                        "市场行为机制和下一步风险观察点。不要只复述数值，禁用Markdown、分点和换行。"
                    ),
                )
            ).strip()
            if not response or response.startswith("[API Error]"):
                continue
            if response.lstrip().startswith("{") or response.lstrip().startswith("["):
                continue
            cache[cache_key] = response
            return response
        except Exception:
            continue
    return ""


def _render_agent_disagreement_chart(package_dict: Dict[str, Any], policy_text: str, runtime_profile: RuntimeModeProfile) -> None:
    agent_effects = dict(package_dict.get("agent_class_effects", {}) or {})
    if not agent_effects:
        return
    rows = []
    for raw_agent, score in agent_effects.items():
        rows.append({"agent": _translate_role(raw_agent), "score": float(score)})
    frame = pd.DataFrame(rows).groupby("agent", as_index=False)["score"].mean().sort_values("score", key=lambda x: x.abs(), ascending=True).head(12)
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame["score"],
                y=frame["agent"],
                orientation="h",
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in frame["score"]],
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="投资者政策解读分歧",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        xaxis_title="多空分歧得分",
        yaxis_title="主体角色",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    top_idx = frame["score"].idxmax()
    low_idx = frame["score"].idxmin()
    bullish_role = str(frame.loc[top_idx, "agent"])
    bearish_role = str(frame.loc[low_idx, "agent"])
    bullish_score = float(frame.loc[top_idx, "score"])
    bearish_score = float(frame.loc[low_idx, "score"])
    fallback = (
        f"{bullish_role}相对更看好（{bullish_score:+.2f}），"
        f"{bearish_role}相对更谨慎（{bearish_score:+.2f}），"
        "当前分歧核心在于政策传导节奏与风险溢价重估。"
    )
    cache_key = hashlib.sha256(f"disagree_{policy_text}_{json.dumps(agent_effects)}".encode()).hexdigest()
    prompt = f"政策：{policy_text}，投资者影响得分为：{json.dumps(agent_effects, ensure_ascii=False)}。请解释哪些主体最可能增配或减配、分歧如何影响订单流，以及政策制定者应观察什么信号。"
    interpretation = _get_llm_visual_interpretation(prompt, cache_key, runtime_profile)
    st.info(f"模型洞察：{interpretation or fallback}")


def _render_role_orderflow_waterfall(frame: pd.DataFrame, package_dict: Dict[str, Any], policy_text: str, runtime_profile: RuntimeModeProfile) -> None:
    if frame.empty:
        return
    agent_effects = dict(package_dict.get("agent_class_effects", {}) or {})
    if not agent_effects:
        return
    scale = float(frame["volume"].mean() / max(len(agent_effects), 1))
    rows: List[Dict[str, Any]] = []
    for role, score in agent_effects.items():
        rows.append({"role": _translate_role(role), "net_flow": float(score) * scale})
    flow_df = pd.DataFrame(rows).groupby("role", as_index=False)["net_flow"].sum().sort_values("net_flow", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=flow_df["net_flow"],
                y=flow_df["role"],
                orientation="h",
                marker_color=["#ef4444" if v < 0 else "#22c55e" for v in flow_df["net_flow"]],
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="角色订单流拆解（深度评估）",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        xaxis_title="净买卖量评估",
        yaxis_title="主体角色",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    lead_idx = flow_df["net_flow"].abs().idxmax()
    lead_role = str(flow_df.loc[lead_idx, "role"])
    lead_flow = float(flow_df.loc[lead_idx, "net_flow"])
    stance = "净买主导" if lead_flow >= 0 else "净卖主导"
    fallback = f"{lead_role}是当前订单流主导力量（{lead_flow:+.0f}），市场处于{stance}阶段，体现对政策路径的再定价。"
    cache_key = hashlib.sha256(f"orderflow_{policy_text}_{json.dumps(agent_effects)}".encode()).hexdigest()
    prompt = f"政策：{policy_text}，预期订单流为：{json.dumps(rows, ensure_ascii=False)}。请判断资金买卖主导力量、交易是追高还是潜伏，并说明这对流动性和价格冲击的含义。"
    interpretation = _get_llm_visual_interpretation(prompt, cache_key, runtime_profile)
    st.info(f"模型洞察：{interpretation or fallback}")


def _render_sector_rotation_heatmap(package_dict: Dict[str, Any], policy_text: str, runtime_profile: RuntimeModeProfile) -> None:
    sector_effects = dict(package_dict.get("sector_effects", {}) or {})
    if not sector_effects:
        return
    ordered = sorted(sector_effects.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:18]
    full_labels = [_translate_sector_name(str(name)) for name, _ in ordered]
    labels = [label if len(label) <= 8 else f"{label[:8]}…" for label in full_labels]
    values = [float(value) for _, value in ordered]
    abs_values = [abs(v) + 0.1 for v in values]
    
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=abs_values,
            customdata=full_labels,
            marker=dict(
                colors=values,
                colorscale="RdYlGn",
                cmid=0,
                showscale=True,
                colorbar=dict(title="影响得分")
            ),
            textinfo="label",
            hovertemplate="<b>%{customdata}</b><br>影响得分：%{color:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        title="板块热度轮动（政策冲击）",
        uniformtext=dict(minsize=11, mode="hide"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    lead_sector = labels[int(np.argmax(values))]
    weak_sector = labels[int(np.argmin(values))]
    lead_score = float(max(values))
    weak_score = float(min(values))
    fallback = (
        f"{lead_sector}相对受益（{lead_score:+.2f}），"
        f"{weak_sector}相对承压（{weak_score:+.2f}），"
        "体现政策冲击下的板块轮动与估值重排。"
    )
    cache_key = hashlib.sha256(f"sector_{policy_text}_{json.dumps(sector_effects)}".encode()).hexdigest()
    prompt = f"政策：{policy_text}，行业影响：{json.dumps(ordered, ensure_ascii=False)}。请说明最受益和最承压板块、轮动机制，以及这种分化对政策执行效果的提示。"
    interpretation = _get_llm_visual_interpretation(prompt, cache_key, runtime_profile)
    st.info(f"模型洞察：{interpretation or fallback}")


def _render_regulation_counterfactual_panel(counterfactual: Dict[str, Any]) -> None:
    worlds = dict(counterfactual.get("worlds", {}) or {})
    no_df = pd.DataFrame(worlds.get("no_intervention", []) or [])
    early_df = pd.DataFrame(worlds.get("early_intervention", []) or [])
    late_df = pd.DataFrame(worlds.get("late_intervention", []) or [])
    if no_df.empty or early_df.empty or late_df.empty:
        return

    merged = no_df[["step", "time", "close"]].rename(columns={"close": "no_close"}).merge(
        early_df[["step", "close"]].rename(columns={"close": "early_close"}),
        on="step",
        how="inner",
    ).merge(
        late_df[["step", "close"]].rename(columns={"close": "late_close"}),
        on="step",
        how="inner",
    )
    merged["提前介入差值"] = merged["early_close"] - merged["no_close"]
    merged["延后介入差值"] = merged["late_close"] - merged["no_close"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.68, 0.32],
        subplot_titles=("监管时点反事实世界线", "相对不介入的收益差分"),
    )
    fig.add_trace(go.Scatter(x=no_df["time"], y=no_df["close"], mode="lines", name="不介入", line=dict(color="#94a3b8", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=early_df["time"], y=early_df["close"], mode="lines", name="提前介入", line=dict(color="#16a34a", width=2.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=late_df["time"], y=late_df["close"], mode="lines", name="延后介入", line=dict(color="#f97316", width=2.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged["time"], y=merged["提前介入差值"], mode="lines", name="提前介入差值", fill="tozeroy", line=dict(color="#22c55e", width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=merged["time"], y=merged["延后介入差值"], mode="lines", name="延后介入差值", fill="tozeroy", line=dict(color="#f97316", width=1.8)), row=2, col=1)
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="监管时点反事实世界线（多智能体会话）",
        yaxis_title="指数点位",
        yaxis2_title="差值",
        margin=dict(l=20, r=20, t=40, b=20),
        height=460,
        legend=dict(orientation="h", y=1.06, x=0.0),
    )
    st.plotly_chart(fig, use_container_width=True)

    scorecards = dict(counterfactual.get("scorecards", {}) or {})
    world_labels = {
        "no_intervention": "不介入",
        "early_intervention": "提前介入",
        "late_intervention": "延后介入",
    }
    rows: List[Dict[str, Any]] = []
    for world_name, metrics in scorecards.items():
        rows.append(
            {
                "world": world_labels.get(world_name, str(world_name)),
                "return_pct": float(metrics.get("return_pct", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "avg_panic": float(metrics.get("avg_panic", 0.0)),
                "max_panic": float(metrics.get("max_panic", 0.0)),
                "volatility": float(metrics.get("volatility", 0.0)),
            }
        )
    if rows:
        score_df = pd.DataFrame(rows)
        st.dataframe(localize_dataframe_columns(score_df), use_container_width=True, hide_index=True)
    recommended = str(counterfactual.get("recommended_timing", ""))
    steps = dict(counterfactual.get("intervention_steps", {}) or {})
    st.caption(
        f"推荐时点：{zh_world_name(recommended)} | 提前介入步={steps.get('early', '-')}, 延后介入步={steps.get('late', '-')}"
    )
    render_narrative_block(
        "监管时点反事实世界线",
        {
            "recommended_timing": zh_world_name(recommended),
            "scorecards": scorecards,
            "early_gap_latest": float(merged["提前介入差值"].iloc[-1]) if not merged.empty else 0.0,
            "late_gap_latest": float(merged["延后介入差值"].iloc[-1]) if not merged.empty else 0.0,
        },
        context="请解释不介入、提前介入、延后介入三条世界线的收益与风险差异，并说明推荐时点的政策含义。",
        cache_namespace="policy_lab_narrative_cache",
    )


def _render_behavior_finance_panel(frame: pd.DataFrame, summary: Dict[str, Any]) -> None:
    if frame.empty:
        st.info("继续仿真后，这里会展示 CSAD、恐慌度和回撤等行为金融指标。")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("CSAD 均值", f"{float(summary.get('avg_csad', 0.0)):.4f}")
    metric_cols[1].metric("最大恐慌度", f"{float(summary.get('max_panic', 0.0)):.2f}")
    metric_cols[2].metric("波动率", f"{float(summary.get('volatility', 0.0)):.2%}")
    metric_cols[3].metric("最大回撤", f"{float(summary.get('max_drawdown', 0.0)):.2%}")
    st.caption("百分比口径：波动率=会话内日收益标准差；最大回撤=区间峰值到谷值最大跌幅；累计收益=期末相对起点的净变化。")

    csad = pd.to_numeric(frame.get("csad"), errors="coerce").fillna(0.0)
    panic = pd.to_numeric(frame.get("panic_level"), errors="coerce").fillna(0.0)
    returns = pd.to_numeric(frame.get("close"), errors="coerce").pct_change().fillna(0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.62, 0.38],
        subplot_titles=("羊群效应指标（CSAD）", "恐慌度与单日收益"),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=csad,
            mode="lines",
            name="CSAD",
            line=dict(color="#facc15", width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0.15,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="羊群效应警戒线",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=panic,
            mode="lines",
            name="恐慌度",
            line=dict(color="#38bdf8", width=2.0),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame["time"],
            y=returns * 100.0,
            name="单日涨跌（%）",
            marker_color=["#22c55e" if value >= 0 else "#ef4444" for value in returns],
            opacity=0.45,
            hovertemplate="日期=%{x}<br>单日涨跌=%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=18, r=18, t=54, b=18),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        legend=dict(orientation="h", y=1.08, x=0.0),
        font=dict(family="Microsoft YaHei, SimHei, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#243244", color="#cbd5e1")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", linecolor="#243244", color="#cbd5e1")
    st.plotly_chart(fig, use_container_width=True)
    render_narrative_block(
        "行为金融指标解读",
        {
            "avg_csad": float(summary.get("avg_csad", 0.0)),
            "max_panic": float(summary.get("max_panic", 0.0)),
            "volatility": float(summary.get("volatility", 0.0)),
            "max_drawdown": float(summary.get("max_drawdown", 0.0)),
            "return_pct": float(summary.get("return_pct", 0.0)),
        },
        context="请结合 CSAD、恐慌度、最大回撤和波动率解释当前行为金融状态及政策含义。",
        cache_namespace="policy_lab_narrative_cache",
        threshold_rules={"max_panic": 0.7, "avg_csad": 0.15, "max_drawdown": 0.08},
        policy_context="政策实验",
    )

    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">行为金融解释</div>
              <div class="summary-value">{float(summary.get('return_pct', 0.0)):+.2%}</div>
              <div class="summary-note">这里的百分比表示“区间净变化”而非年化收益，需与 CSAD、恐慌度和回撤一起解读，避免单看涨跌。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[1]:
        st.markdown(
            """
            <div class="story-card">
              <div class="story-card-title">解读提示</div>
              <div class="summary-note">
                CSAD 下降且市场同步放量时，更接近“群体跟随”而不是理性分散定价；若恐慌度回落但指数仍弱，
                往往说明资金在修复预期、价格还在消化前期冲击。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_agent_fmri_rows(session: Dict[str, Any], package_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest = dict(session.get("latest_step_report", {}) or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    price = float(matching.get("last_price", session.get("last_close", 0.0)) or 0.0)
    panic = float(sentiment.get("panic_level", session.get("last_panic", 0.0)) or 0.0)
    buy_volume = float(order_flow.get("buy_volume", 0.0) or 0.0)
    sell_volume = float(order_flow.get("sell_volume", 0.0) or 0.0)
    policy_text = str(session.get("policy_text", "") or "").strip()
    agent_effects = dict(package_dict.get("agent_class_effects", {}) or {})

    if not agent_effects:
        llm_count = max(1, int(session.get("llm_agent_count", 3) or 3))
        agent_effects = {f"committee_{idx + 1:02d}": 0.18 - idx * 0.09 for idx in range(llm_count)}

    rows: List[Dict[str, Any]] = []
    for idx, (agent_name, raw_score) in enumerate(
        sorted(agent_effects.items(), key=lambda item: abs(float(item[1] or 0.0)), reverse=True)
    ):
        score = float(raw_score or 0.0)
        action_key = "BUY" if score >= 0.12 else "SELL" if score <= -0.12 else "HOLD"
        status_key = "Risk-On" if score >= 0.18 else "Risk-Off" if score <= -0.18 else "Observe"
        action_display = zh_action_name(action_key)
        status_display = zh_value(status_key)
        confidence = min(0.92, 0.45 + abs(score) * 0.35 + panic * 0.10)
        qty = int(max(0.0, abs(score) * 12000.0))
        display_agent = _translate_role(str(agent_name))
        rows.append(
            {
                "agent": str(agent_name),
                "agent_display": display_agent,
                "agent_key": str(agent_name),
                "score": score,
                "status": status_key,
                "status_display": status_display,
                "sentiment": max(-1.0, min(1.0, score - panic * 0.35)),
                "decision": {
                    "action": action_key,
                    "action_display": action_display,
                    "ticker": str(session.get("index_symbol", "A_SHARE_IDX")),
                    "price": round(price, 2),
                    "qty": qty,
                    "confidence": round(confidence, 2),
                },
                "decision_label": f"{action_key} · {qty:,}",
                "decision_label_display": f"{action_display} · {qty:,}",
                "thought": (
                    f"围绕“{policy_text[:48] or '当前政策'}”做重估。"
                    f"当前买量 {buy_volume:.0f}、卖量 {sell_volume:.0f}，"
                    f"恐慌度 {panic:.2f}，因此倾向 {action_display}。"
                ),
                "meta_line": f"情绪 {max(-1.0, min(1.0, score - panic * 0.35)):+.2f}｜状态 {status_display}",
                "history": [
                    f"记录 {idx + 1}: {action_display}",
                    f"记录 {idx + 2}: {zh_action_name('HOLD') if action_key != 'HOLD' else zh_action_name('BUY')}",
                    f"记录 {idx + 3}: {zh_action_name('SELL') if action_key == 'BUY' else zh_action_name('HOLD')}",
                ],
            }
        )
    return rows


def _render_agent_fmri_panel(session: Dict[str, Any], package_dict: Dict[str, Any]) -> None:
    rows = _build_agent_fmri_rows(session, package_dict)
    if not rows:
        st.info("当前没有可展示的智能体分歧数据。")
        return

    select_key = "policy_lab_agent_fmri_select"
    if str(st.session_state.get(select_key, "")) not in {str(item["agent"]) for item in rows}:
        st.session_state[select_key] = str(rows[0]["agent"])

    left, right = st.columns([0.95, 1.85], vertical_alignment="top")
    with left:
        st.markdown(
            """
            <div class="story-card">
              <div class="story-card-title">智能体决策剖面</div>
              <div class="summary-note">点击任意主体，查看当前决策、情绪分数和近阶段行为记录。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in rows:
            agent_name = str(item["agent"])
            display_agent = str(item.get("agent_display", agent_name))
            sentiment = float(item["sentiment"])
            selected = agent_name == str(st.session_state.get(select_key, ""))
            action_text = str(item.get("decision", {}).get("action_display") or zh_action_name(str(item["decision"]["action"])))
            button_label = f"{display_agent}  {'●' if sentiment >= 0 else '○'}  {action_text}"
            if st.button(
                button_label,
                key=f"policy_lab_agent_btn_{agent_name}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state[select_key] = agent_name

    selected_agent = str(st.session_state.get(select_key, str(rows[0]["agent"])))
    selected = next((item for item in rows if str(item["agent"]) == selected_agent), rows[0])
    tone = "calm"
    if float(selected["score"]) >= 0.12:
        tone = "up"
    elif float(selected["score"]) <= -0.12:
        tone = "down"

    with right:
        top_left, top_right = st.columns([1.15, 1.0])
        with top_left:
            st.markdown(
                f"""
                <div class="pulse-card pulse-card-{tone}">
                  <div class="summary-label">{selected.get('agent_display', selected['agent'])}</div>
                  <div class="summary-value">{float(selected['sentiment']):+.2f}</div>
                  <div class="summary-note">{selected['meta_line']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_right:
            score_frame = pd.DataFrame(rows)
            score_frame["agent_label"] = score_frame.apply(lambda row: row.get("agent_display") or row.get("agent"), axis=1)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=score_frame["agent_label"],
                        y=score_frame["score"],
                        marker_color=["#22c55e" if value >= 0 else "#ef4444" for value in score_frame["score"]],
                    )
                ]
            )
            fig.update_layout(
                height=180,
                margin=dict(l=18, r=18, t=12, b=18),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1220",
                xaxis_title="智能体角色",
                yaxis_title="得分",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### 当前决策")
        action_cn = str(selected.get("decision", {}).get("action_display") or zh_action_name(str(selected["decision"]["action"])))
        st.info(f"**操作**: {action_cn} | **信心度**: {selected['decision']['confidence']*100:.0f}% | **预期交易量**: {selected['decision']['qty']} 份")
        st.markdown("#### 判断依据")
        st.markdown(f"> {selected['thought']}")
        
        st.markdown("#### 近阶段轨迹")
        history = list(selected.get("history", []) or [])
        if history:
            with st.expander("展开最近记录", expanded=True):
                st.markdown(f"**记录点**: {history[0]}")
                st.info(f"该阶段策略倾向：{action_cn}。")
            for item in history[1:]:
                with st.expander(str(item), expanded=False):
                    st.markdown("在当前市场环境下的惯性跟随或反转预期。")
        render_narrative_block(
            "智能体决策剖面解读",
            {
                "selected_agent": selected.get("agent"),
                "selected_score": float(selected.get("score", 0.0)),
                "selected_sentiment": float(selected.get("sentiment", 0.0)),
                "selected_action": selected.get("decision", {}).get("action"),
                "agents": [
                    {"agent": item.get("agent"), "score": item.get("score"), "action": item.get("decision", {}).get("action")}
                    for item in rows[:8]
                ],
            },
            context="请解释智能体角色之间的买卖分歧、风险偏好变化，以及这种分歧如何影响订单流和政策效果。",
            cache_namespace="policy_lab_narrative_cache",
        )


def _policy_session_status_text(status: str) -> str:
    return POLICY_SESSION_STATUS_LABELS.get(str(status or "").strip().lower(), "未知状态")


def _close_policy_lab_session() -> None:
    session = st.session_state.pop("policy_lab_session", None)
    if session is not None:
        try:
            if hasattr(session, "environment"):
                session.environment.close()
            elif isinstance(session, dict) and isinstance(session.get("_runner"), PolicySession):
                session["_runner"].environment.close()
        except Exception:
            pass
    st.session_state.pop("policy_lab_result", None)
    st.session_state.pop("policy_lab_bundle", None)
    st.session_state.pop("policy_lab_session_report", None)
    st.session_state.pop("policy_lab_session_meta", None)


def _policy_session_event_markers(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    for item in list(session.get("policy_timeline", []) or _policy_session_timeline(session)):
        if not isinstance(item, dict):
            continue
        markers.append(
            {
                "event_id": str(item.get("policy_id", item.get("政策名称", ""))),
                "event_type": "policy",
                "title": str(item.get("政策名称", item.get("policy_name", "政策事件"))),
                "raw_text": str(item.get("政策文本", item.get("policy_text", ""))),
                "effective_day": int(item.get("生效交易日", item.get("effective_day", 1)) or 1),
                "strength": float(item.get("初始强度", item.get("remaining_effect", 1.0)) or 0.0),
                "source": "policy_session",
            }
        )
    for item in list(session.get("runtime_event_timeline", []) or []):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("event_type", "news") or "news")
        event_type = "regulatory_action" if raw_type == "regulatory_action" else raw_type
        markers.append(
            {
                "event_id": str(item.get("event_id", item.get("id", ""))),
                "event_type": event_type,
                "title": str(item.get("title", raw_type)),
                "raw_text": str(item.get("raw_text", item.get("description", ""))),
                "effective_day": int(item.get("effective_day", item.get("step", 1)) or 1),
                "strength": float(item.get("current_strength", item.get("strength", 1.0)) or 0.0),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "source": str(item.get("source", "runtime_event")),
            }
        )
    if not any(str(item.get("event_type")) in {"major_news", "news", "rumor", "refute"} for item in markers):
        markers.append(
            {
                "event_type": "major_news",
                "title": "市场新闻基线",
                "raw_text": "用于展示新闻事件标记层的合成基线事件。",
                "effective_day": max(1, int(session.get("current_day", 1) or 1) // 2 or 1),
                "strength": 0.45,
                "source": "synthetic_fallback",
            }
        )
    if not any(str(item.get("event_type")) == "regulatory_action" for item in markers):
        markers.append(
            {
                "event_type": "regulatory_action",
                "title": "监管观察窗口",
                "raw_text": "当恐慌度或订单失衡超过阈值时进入监管观察。",
                "effective_day": max(1, int(session.get("current_day", 1) or 1)),
                "strength": float(dict(session.get("summary", {}) or {}).get("max_panic", 0.35) or 0.35),
                "source": "synthetic_fallback",
            }
        )
    return markers


def _policy_session_real_index_frame(session: Dict[str, Any]) -> pd.DataFrame:
    stored = session.get("real_index_frame")
    if isinstance(stored, list) and stored:
        return pd.DataFrame(stored)
    symbol = str(session.get("index_symbol", "sh000001") or "sh000001")
    total_days = int(session.get("history_window_days", session.get("total_days", 120)) or 120)
    end_date = _policy_anchor_date() - pd.Timedelta(days=1)
    return _load_real_index_history(symbol, max(60, total_days), end_date)


def _policy_session_registry_entry(session: Dict[str, Any], chart_frame: pd.DataFrame) -> Dict[str, Any]:
    config_payload = {
        "session_id": session.get("session_id", ""),
        "policy_name": session.get("policy_name", ""),
        "policy_text": session.get("policy_text", ""),
        "total_days": session.get("total_days", 0),
        "index_symbol": session.get("index_symbol", "sh000001"),
        "runtime_profile": session.get("runtime_profile", {}),
    }
    config_hash = str(session.get("config_hash") or stable_payload_hash(config_payload))
    data_snapshot_id = str(session.get("data_snapshot_id") or chart_frame.attrs.get("trade_tape_hash", "synthetic_or_cached_snapshot"))
    entry = build_experiment_registry_entry(
        experiment_id=str(session.get("experiment_id", "")),
        scenario_name=str(session.get("policy_name", session.get("template_title", "policy_lab"))),
        config_hash=config_hash,
        data_snapshot_id=data_snapshot_id,
        seed=int(session.get("seed", 42) or 42),
        selected_benchmark=str(session.get("index_symbol", "sh000001") or "sh000001"),
        status=str(session.get("status", "created")),
        created_at=str(session.get("created_at", "")) or None,
        parameter_set_id=str(session.get("calibration_parameter_set_id", "policy_lab_default_calibration_v1")),
    )
    session["experiment_id"] = entry["experiment_id"]
    session["config_hash"] = entry["config_hash"]
    session["data_snapshot_id"] = entry["data_snapshot_id"]
    session["data_snapshot_hash"] = entry["data_snapshot_hash"]
    session["parameter_set_id"] = entry["parameter_set_id"]
    session["created_at"] = entry["created_at"]
    return entry


def _policy_session_repro_meta(
    session: Dict[str, Any],
    *,
    registry_entry: Dict[str, Any],
    chart_frame: pd.DataFrame,
    runtime_profile: RuntimeModeProfile,
) -> Dict[str, Any]:
    return build_reproducibility_meta(
        experiment_id=str(registry_entry.get("experiment_id", session.get("experiment_id", ""))),
        data_snapshot_hash=str(registry_entry.get("data_snapshot_id", chart_frame.attrs.get("trade_tape_hash", ""))),
        config_hash=str(registry_entry.get("config_hash", "")),
        random_seed=int(registry_entry.get("seed", 42) or 42),
        llm_provider_chain=list(runtime_profile.model_priority or ["GLM-4-flashx", "offline_fallback"]),
        calibration_parameter_set_id=str(session.get("calibration_parameter_set_id", "policy_lab_default_calibration_v1")),
        extra={
            "selected_benchmark": registry_entry.get("selected_benchmark", "sh000001"),
            "runtime_mode": runtime_profile.mode,
            "session_id": session.get("session_id", ""),
        },
    )


def _policy_session_scorecard(
    session: Dict[str, Any],
    chart_frame: pd.DataFrame,
    real_index_frame: pd.DataFrame,
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if chart_frame.empty:
        return mock_scorecard()
    real_close = (
        pd.to_numeric(real_index_frame["close"], errors="coerce").tail(len(chart_frame)).tolist()
        if real_index_frame is not None and not real_index_frame.empty and "close" in real_index_frame.columns
        else pd.to_numeric(chart_frame["close"], errors="coerce").tolist()
    )
    scorecard = build_replay_scorecard(
        sim_close=pd.to_numeric(chart_frame["close"], errors="coerce").tolist(),
        real_close=real_close,
        benchmark_symbol=str(session.get("index_symbol", "sh000001") or "sh000001"),
        replay_window={
            "start": str(chart_frame.iloc[0].get("time", "")),
            "end": str(chart_frame.iloc[-1].get("time", "")),
        },
        seed=int(session.get("seed", 42) or 42),
        config_hash=str(session.get("config_hash", "")),
        data_snapshot_hash=str(session.get("data_snapshot_id", chart_frame.attrs.get("trade_tape_hash", ""))),
        microstructure_metrics={
            "spread": float(pd.to_numeric(chart_frame.get("spread", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
            "depth_imbalance": float(pd.to_numeric(chart_frame.get("depth_imbalance", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
            "trade_count": float(pd.to_numeric(chart_frame.get("trade_count", pd.Series([0.0])), errors="coerce").fillna(0.0).sum()),
        },
        behavioral_metrics={
            "csad_mean": float(pd.to_numeric(chart_frame.get("csad", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
            "sentiment_index": float(pd.to_numeric(chart_frame.get("sentiment_index", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
        },
        regulatory_metrics={"event_count": float(len(events))},
        event_points=[item.get("effective_day", item.get("step", 1)) for item in events],
        dates=chart_frame["time"].tolist(),
        panic_series=pd.to_numeric(chart_frame.get("panic_level", pd.Series(dtype=float)), errors="coerce").fillna(0.0).tolist(),
    )
    return scorecard.to_dict()


def _session_frame_to_market_frame(session_frame: pd.DataFrame, *, anchor_close: float = 0.0) -> pd.DataFrame:
    if session_frame.empty:
        return pd.DataFrame(columns=["step", "time", "open", "high", "low", "close", "volume", "csad", "panic_level"])

    frame = session_frame.copy().reset_index(drop=True)
    base_close = float(frame.iloc[0]["收盘价"]) if float(frame.iloc[0]["收盘价"]) > 0 else 1.0
    anchor = float(anchor_close) if float(anchor_close or 0.0) > 0 else base_close
    scale = anchor / max(base_close, 1e-9)
    raw_close = frame["收盘价"].astype(float) * scale
    raw_returns = raw_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    damped_returns = raw_returns.clip(-0.0075, 0.0075).ewm(alpha=0.38, adjust=False).mean()
    scaled_close = [anchor]
    for value in damped_returns.iloc[1:]:
        scaled_close.append(scaled_close[-1] * (1.0 + float(value)))
    scaled_close = pd.Series(scaled_close, index=frame.index, dtype=float)
    scaled_open = scaled_close.shift(1).fillna(anchor)
    intraday_span = raw_returns.abs().clip(lower=0.0012, upper=0.0085)
    scaled_high = pd.concat([scaled_open, scaled_close], axis=1).max(axis=1) * (1.0 + intraday_span * 0.42)
    scaled_low = pd.concat([scaled_open, scaled_close], axis=1).min(axis=1) * (1.0 - intraday_span * 0.42)
    raw_volume = frame["总买量"].astype(float) + frame["总卖量"].astype(float)
    volume = raw_volume.ewm(alpha=0.35, adjust=False).mean()
    if not volume.empty:
        median_volume = float(volume.median() or 0.0)
        if median_volume > 0.0:
            volume = volume.clip(lower=median_volume * 0.55, upper=median_volume * 1.75)
    market = pd.DataFrame(
        {
            "step": frame["交易日序号"].astype(int),
            "time": frame["交易日"].astype(str),
            "open": scaled_open.round(2),
            "high": scaled_high.round(2),
            "low": scaled_low.round(2),
            "close": scaled_close.round(2),
            "volume": volume.round(2),
            "csad": frame["羊群度"].astype(float),
            "panic_level": frame["恐慌度"].astype(float),
            "trade_count": frame["成交笔数"].astype(int),
            "active_policy_count": frame["活跃政策数"].astype(int),
        }
    )
    return market


def _build_policy_session_package(policy_text: str, intensity: float, selected: Dict[str, Any]) -> Dict[str, Any]:
    if not str(policy_text or "").strip():
        return {}
    _, package = _compile_policy_bundle(
        str(policy_text),
        float(max(intensity, 0.1)),
        policy_type_hint=str(selected.get("policy_type", "")),
    )
    return package.to_dict()


def _store_policy_lab_session_result(
    *,
    session: PolicySession,
    snapshot: Dict[str, Any],
    selected_title: str,
    selected: Dict[str, Any],
    policy_text: str,
    index_label: str,
    index_symbol: str,
    anchor_close: float,
    history_end: str,
    runtime_profile: RuntimeModeProfile,
    llm_agent_count: int,
) -> None:
    session_frame = snapshot.get("frame", pd.DataFrame())
    if not isinstance(session_frame, pd.DataFrame):
        session_frame = pd.DataFrame(session_frame or [])
    display_frame = _session_frame_to_market_frame(session_frame, anchor_close=anchor_close)
    summary = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    active_policies = snapshot.get("active_policies", []) if isinstance(snapshot.get("active_policies"), list) else []
    latest_policy_text = "；".join(
        [str(item.get("政策文本", "")) for item in active_policies if str(item.get("政策文本", "")).strip()]
    ) or str(policy_text or "")
    latest_intensity = sum(float(item.get("当前强度", 0.0) or 0.0) for item in active_policies) or float(
        active_policies[0].get("基础强度", 1.0) if active_policies else 1.0
    )
    package_dict = _build_policy_session_package(latest_policy_text, latest_intensity, selected)

    st.session_state.policy_lab_result = {
        "session_snapshot": snapshot,
        "frame": display_frame,
        "summary": {
            "return_pct": float(summary.get("累计收益率", 0.0) or 0.0),
            "avg_panic": float(display_frame["panic_level"].mean()) if not display_frame.empty else 0.0,
            "max_panic": float(display_frame["panic_level"].max()) if not display_frame.empty else 0.0,
            "avg_csad": float(display_frame["csad"].mean()) if not display_frame.empty else 0.0,
            "max_drawdown": float(summary.get("最大回撤", 0.0) or 0.0),
            "avg_volume": float(display_frame["volume"].mean()) if not display_frame.empty else 0.0,
            "volatility": float(display_frame["close"].pct_change().fillna(0.0).std()) if len(display_frame) > 1 else 0.0,
        },
        "policy_text": policy_text,
        "template": selected,
        "template_title": selected_title,
        "policy_package": package_dict,
        "index_label": index_label,
        "index_symbol": index_symbol,
        "index_anchor_close": float(anchor_close),
        "history_end": history_end,
        "data_source": "deep_mode_simulation",
        "runtime_mode": runtime_profile.mode,
        "runtime_profile": runtime_profile.to_dict(),
        "deep_mode_meta": {
            "llm_agent_count": int(llm_agent_count),
            "thinking_stats": dict(snapshot.get("last_step_report", {}).get("thinking_stats", {}) or {}),
            "session_status": str(snapshot.get("status", "")),
            "report_payload": snapshot.get("report_payload", {}),
            "active_policies": active_policies,
            "queued_policies": snapshot.get("queued_policies", []),
            "counterfactual_regulation": _build_regulation_counterfactual_worlds(
                display_frame.rename(columns={"time": "time"})[["step", "time", "open", "high", "low", "close", "volume", "csad", "panic_level"]]
                if not display_frame.empty
                else pd.DataFrame(),
                intensity=float(max(latest_intensity, 0.1)),
            ),
        },
    }


def _build_policy_session_report_bundle(
    *,
    report: Dict[str, Any],
    selected_title: str,
    policy_text: str,
    index_label: str,
    index_symbol: str,
    runtime_profile: RuntimeModeProfile,
) -> Dict[str, Any]:
    report_title = f"政策实验会话 - {selected_title}"
    payload = dict(report.get("报告数据", {}) or {})
    report_meta = official_report_meta("policy_lab_session", report_title)
    payload.update(
        {
            "report_meta": report_meta,
            "模板": selected_title,
            "政策文本": policy_text,
            "指数基准": index_label,
            "指数代码": index_symbol,
            "运行模式": _runtime_mode_text(runtime_profile),
        }
    )
    bundle = write_report_artifacts(
        root_dir=POLICY_REPORT_DIR,
        report_type="policy_lab_session",
        title=report_title,
        markdown_text=_sanitize_report_markdown_for_frontend(str(report.get("报告正文", ""))),
        payload=payload,
    )
    bundle["report_meta"] = report_meta
    return bundle


def _policy_template_items(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").replace("；", "、").replace("，", "、")
    return [item.strip() for item in text.split("、") if item.strip()]


def _apply_template_to_policy_form(
    template: Dict[str, Any],
    *,
    index_label_default: str,
    history_window_days: int,
) -> None:
    st.session_state["policy_lab_policy_text_input"] = str(template.get("policy_text", "") or "")
    st.session_state["policy_lab_policy_intensity_input"] = float(template.get("recommended_intensity", 1.0) or 1.0)
    recommended_duration = int(template.get("recommended_duration", 30) or 30)
    st.session_state["policy_lab_total_days_input"] = max(10, min(180, recommended_duration))
    st.session_state["policy_lab_effective_day_input"] = 1
    st.session_state["policy_lab_half_life_input"] = max(1, min(120, recommended_duration))
    st.session_state["policy_lab_rumor_noise_input"] = bool(template.get("default_rumor_noise", False))
    st.session_state["policy_lab_index_label_input"] = index_label_default
    st.session_state["policy_lab_history_window_input"] = int(history_window_days)


def _render_policy_entry_preview(
    *,
    selected_template: Optional[Dict[str, Any]],
    preview_policy_text: str,
    preview_intensity: float,
    preview_total_days: int,
    preview_effective_day: int,
    preview_half_life_days: int,
    preview_rumor_noise: bool,
) -> None:
    template = dict(selected_template or {})
    template_title = str(template.get("title", "自定义政策输入") or "自定义政策输入")
    policy_type_hint = str(template.get("policy_type", "") or "综合政策")
    package_dict: Dict[str, Any] = {}
    try:
        _, package = _compile_policy_bundle(
            preview_policy_text,
            float(max(preview_intensity, 0.1)),
            policy_type_hint=policy_type_hint,
        )
        package_dict = package.to_dict()
    except Exception:
        package_dict = {}

    event = dict(package_dict.get("event", {}) or {})
    uncertainty = dict(package_dict.get("uncertainty", {}) or {})
    top_layers = dict(package_dict.get("top_layers", {}) or {})
    channels = list(package_dict.get("channels", []) or [])

    preview_cols = st.columns(4)
    preview_cols[0].metric("场景来源", template_title)
    preview_cols[1].metric("结构化类型", str(event.get("policy_type", policy_type_hint) or policy_type_hint))
    preview_cols[2].metric("影响方向", str(event.get("direction", "待识别") or "待识别"))
    preview_cols[3].metric("解析置信度", f"{float(uncertainty.get('confidence', 0.0) or 0.0):.0%}")

    st.markdown(
        """
        <div class="summary-card">
          <div class="summary-label">自然语言政策到结构化政策冲击</div>
          <div class="summary-value">当前入口会把政策文本转成可驱动仿真的结构化政策包</div>
          <div class="summary-note">展示重点包括政策类型、影响方向、冲击强度、持续时间、潜在作用对象、传导渠道和对照组设定。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    brief_left, brief_right = st.columns(2)
    with brief_left:
        st.markdown("#### 编译前输入")
        st.markdown(
            "\n".join(
                [
                    f"- 政策文本长度：{len(str(preview_policy_text or '').strip())} 字",
                    f"- 仿真天数：{int(preview_total_days)} 个交易日",
                    f"- 生效日：第 {int(preview_effective_day)} 个交易日",
                    f"- 半衰期：{int(preview_half_life_days)} 个交易日",
                    f"- 是否叠加市场噪声：{'是' if preview_rumor_noise else '否'}",
                ]
            )
        )
        st.caption((preview_policy_text or "请先输入政策文本。")[:160])
    with brief_right:
        st.markdown("#### 反事实甲乙对照基线")
        control_label = str(template.get("control_label", "无政策基线对照") or "无政策基线对照")
        control_text = str(template.get("control_text", "在相同初始条件下，观察不采取干预时的市场路径。") or "在相同初始条件下，观察不采取干预时的市场路径。")
        st.markdown(f"- 基线方案：{control_label}")
        st.markdown(f"- 对照说明：{control_text}")
        if template.get("policy_goal"):
            st.markdown(f"- 评估目标：{template.get('policy_goal')}")

    tab_targets, tab_channels, tab_risk = st.tabs(["作用对象", "传导渠道", "风险提示"])
    with tab_targets:
        target_items = _policy_template_items(template.get("suitable_departments"))
        top_agents = [str(name) for name, _ in list(top_layers.get("agent_class", []) or [])[:4]]
        rows = pd.DataFrame(
            {
                "对象类别": ["政策协同部门", "重点市场主体"],
                "内容": [
                    "、".join(target_items) if target_items else "财政、货币、监管与市场稳定相关部门",
                    "、".join(top_agents) if top_agents else "散户、机构、做市与稳定资金",
                ],
            }
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with tab_channels:
        expected_channels = _policy_template_items(template.get("expected_channels"))
        channel_names = [str(item.get("name", "")) for item in channels if str(item.get("name", "")).strip()]
        rows = pd.DataFrame(
            {
                "来源": ["模板预设", "编译结果"],
                "传导链": [
                    "、".join(expected_channels) if expected_channels else "流动性、风险偏好、预期与波动",
                    "、".join(channel_names) if channel_names else "政策信号、主体认知、订单流、市场重定价",
                ],
            }
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with tab_risk:
        risk_focus = _policy_template_items(template.get("risk_focus"))
        side_effects = [str(item) for item in list(event.get("side_effects", []) or []) if str(item).strip()]
        risk_rows = pd.DataFrame(
            {
                "风险来源": ["模板风险关注", "结构化副作用提示"],
                "内容": [
                    "、".join(risk_focus) if risk_focus else "情绪过热、流动性错配与政策后效衰减",
                    "、".join(side_effects) if side_effects else "当前未识别出额外副作用，进入推演后继续跟踪",
                ],
            }
        )
        st.dataframe(risk_rows, use_container_width=True, hide_index=True)

    st.info(
        "本页支持智能模式（仅智谱 GLM）与高级模式（DeepSeek + 智谱混用），用于政策文本理解、结构化抽取、部分智能体认知推理和解释性内容生成。"
    )


def render_policy_lab(*, presentation_mode: str = "standard") -> None:
    st.subheader("政策实验")
    st.caption("仅供教学科研与仿真，不构成投资建议。")
    mode_display = {
        "SMART": "智能模式（仅智谱 GLM）",
        "DEEP": "高级模式（DeepSeek + 智谱混用）",
    }
    current_mode = normalize_runtime_mode(str(st.session_state.get("simulation_mode", "SMART")))
    selected_runtime_mode = st.radio(
        "模型模式",
        options=["SMART", "DEEP"],
        index=0 if current_mode == "SMART" else 1,
        format_func=lambda value: mode_display.get(value, value),
        horizontal=True,
        key="policy_lab_runtime_mode_selector",
        help="智能模式只调用智谱模型；高级模式才启用原先 DeepSeek 与智谱混排的配置。",
    )
    if selected_runtime_mode != st.session_state.get("simulation_mode"):
        st.session_state["simulation_mode"] = selected_runtime_mode
        st.session_state["runtime_mode_profile"] = resolve_runtime_mode_profile(selected_runtime_mode).to_dict()
    runtime_profile = _resolve_runtime_profile()
    presentation_mode = str(presentation_mode or "standard").strip().lower()
    st.caption(f"当前模式：{_runtime_mode_text(runtime_profile)}")
    if presentation_mode == "defense":
        st.info("当前页面采用场景风洞视图，适合快速浏览政策链路和市场传导；默认“政策实验”页则更偏稳健评估。")

    session: Optional[Dict[str, Any]] = st.session_state.get("policy_lab_session")
    current_defaults = dict(session or {})
    template_library = _load_policy_templates()
    template_options = {"自定义输入（不载入模板）": None}
    for item in template_library:
        label = f"{str(item.get('title', '政策模板'))}｜{str(item.get('category', '政策场景'))}"
        template_options[label] = item
    selected_template_id = str(
        st.session_state.get("policy_lab_selected_template_id")
        or current_defaults.get("template_id")
        or "custom"
    )
    selected_template_label = next(
        (
            label
            for label, item in template_options.items()
            if isinstance(item, dict) and str(item.get("id", "")) == selected_template_id
        ),
        "自定义输入（不载入模板）",
    )

    st.markdown("### 政策实验入口：从自然语言到结构化政策冲击")
    provider_label = "智谱 GLM" if runtime_profile.mode == "SMART" else "DeepSeek + 智谱"
    primary_model_label = "、".join(runtime_profile.model_priority[:2]) if runtime_profile.model_priority else provider_label
    entry_cols = st.columns([1.2, 1.0])
    with entry_cols[0]:
        selected_template_label = st.selectbox(
            "稳定模板场景（可选）",
            options=list(template_options.keys()),
            index=list(template_options.keys()).index(selected_template_label),
            key="policy_lab_template_selector",
        )
    selected_template = template_options.get(selected_template_label)
    st.session_state["policy_lab_selected_template_id"] = (
        str(selected_template.get("id", "custom")) if isinstance(selected_template, dict) else "custom"
    )
    with entry_cols[1]:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">大模型参与环节</div>
              <div class="summary-value">{provider_label} 负责把政策文本变成可执行推演输入</div>
              <div class="summary-note">当前候选链：{primary_model_label}；承担政策语义理解、结构化抽取、部分智能体认知推理和解释生成。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    default_policy_text = str(current_defaults.get("policy_text", "近期，各部委相继出台一系列旨在提振总需求、稳定资本市场预期的政策措施。财政部宣布将扩大专项债发行规模，重点支持先进制造业和新基建投资；同时，央行超预期实施降准0.5个百分点，并下调政策利率20个基点，以释放充足的流动性，降低实体经济融资成本。税务总局及相关监管机构亦同步出台了减免交易印花税及规范大股东减持行为的细则，明确释放维稳信号。预计上述组合拳将显著提振投资者信心，改善市场微观流动性。"))
    default_intensity = float(current_defaults.get("intensity", 1.0))
    default_total_days = int(current_defaults.get("total_days", 100))
    default_effective_day = int(current_defaults.get("effective_day", 1))
    default_half_life_days = int(current_defaults.get("half_life_days", 30))
    default_rumor_noise = bool(current_defaults.get("rumor_noise", False))
    index_label_default = str(current_defaults.get("index_label", list(INDEX_BENCHMARK_OPTIONS.keys())[0]))
    if index_label_default not in INDEX_BENCHMARK_OPTIONS:
        index_label_default = list(INDEX_BENCHMARK_OPTIONS.keys())[0]
    history_window_default = int(current_defaults.get("history_window_days", 180))

    if isinstance(selected_template, dict):
        default_policy_text = str(selected_template.get("policy_text", default_policy_text) or default_policy_text)
        default_intensity = float(selected_template.get("recommended_intensity", default_intensity) or default_intensity)
        default_total_days = int(selected_template.get("recommended_duration", default_total_days) or default_total_days)
        default_half_life_days = min(max(int(default_total_days), 1), 120)
        default_rumor_noise = bool(selected_template.get("default_rumor_noise", default_rumor_noise))

    setup_cols = st.columns([1.1, 1.0])
    with setup_cols[0]:
        st.markdown("### 仿真设置")
        if isinstance(selected_template, dict):
            st.caption(
                f"已选择模板场景：{selected_template.get('title', '')}。你可以直接使用模板，也可以继续编辑文本。"
            )
        with st.form("policy_lab_start_form"):
            policy_text = st.text_area(
                "政策文本",
                value=default_policy_text,
                height=180,
                key="policy_lab_policy_text_input",
                help="直接输入要试验的政策文本，系统会自动推理解读并驱动多智能体交易仿真。",
            )
            with st.expander("高级设置（可选）", expanded=False):
                intensity = st.slider(
                    "政策基础强度",
                    min_value=0.2,
                    max_value=2.0,
                    value=float(default_intensity),
                    step=0.1,
                    key="policy_lab_policy_intensity_input",
                )
                total_days = st.slider(
                    "仿真天数",
                    min_value=10,
                    max_value=180,
                    value=max(10, min(180, int(default_total_days))),
                    step=5,
                    key="policy_lab_total_days_input",
                )
                effective_day = st.number_input(
                    "政策生效日",
                    min_value=1,
                    max_value=max(1, int(total_days)),
                    value=max(1, min(int(total_days), int(default_effective_day))),
                    step=1,
                    key="policy_lab_effective_day_input",
                )
                half_life_days = st.slider(
                    "政策影响半衰期（交易日）",
                    min_value=1,
                    max_value=120,
                    value=max(1, min(120, int(default_half_life_days))),
                    step=1,
                    key="policy_lab_half_life_input",
                )
                rumor_noise = st.checkbox(
                    "包含传言噪声",
                    value=bool(default_rumor_noise),
                    key="policy_lab_rumor_noise_input",
                )
                index_label = st.selectbox(
                    "对比指数基准",
                    options=list(INDEX_BENCHMARK_OPTIONS.keys()),
                    index=list(INDEX_BENCHMARK_OPTIONS.keys()).index(index_label_default),
                    key="policy_lab_index_label_input",
                )
                history_window_days = st.slider(
                    "真实基准回看天数",
                    min_value=60,
                    max_value=360,
                    value=int(history_window_default),
                    step=20,
                    key="policy_lab_history_window_input",
                )
            start_clicked = st.form_submit_button("开始推演", type="primary", use_container_width=True)
    with setup_cols[1]:
        st.markdown("### 入口编译预览")
        if isinstance(selected_template, dict):
            template_cols = st.columns([1.0, 1.0])
            template_cols[0].markdown(
                f"""
                <div class="story-card">
                  <div class="story-card-title">{selected_template.get('title', '模板场景')}</div>
                  <div class="story-card-summary">{selected_template.get('policy_goal', '用于稳定演示的政策模板场景。')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if template_cols[1].button("载入模板到输入区", use_container_width=True, key="policy_lab_apply_template"):
                _apply_template_to_policy_form(
                    selected_template,
                    index_label_default=index_label_default,
                    history_window_days=history_window_default,
                )
                st.rerun()
        preview_policy_text = str(st.session_state.get("policy_lab_policy_text_input", default_policy_text) or default_policy_text)
        preview_intensity = float(st.session_state.get("policy_lab_policy_intensity_input", default_intensity) or default_intensity)
        preview_total_days = int(st.session_state.get("policy_lab_total_days_input", default_total_days) or default_total_days)
        preview_effective_day = int(st.session_state.get("policy_lab_effective_day_input", default_effective_day) or default_effective_day)
        preview_half_life_days = int(st.session_state.get("policy_lab_half_life_input", default_half_life_days) or default_half_life_days)
        preview_rumor_noise = bool(st.session_state.get("policy_lab_rumor_noise_input", default_rumor_noise))
        _render_policy_entry_preview(
            selected_template=selected_template if isinstance(selected_template, dict) else None,
            preview_policy_text=preview_policy_text,
            preview_intensity=preview_intensity,
            preview_total_days=preview_total_days,
            preview_effective_day=preview_effective_day,
            preview_half_life_days=preview_half_life_days,
            preview_rumor_noise=preview_rumor_noise,
        )

    def _store_session(updated_session: Optional[Dict[str, Any]]) -> None:
        st.session_state["policy_lab_session"] = updated_session
        st.session_state["policy_lab_result"] = updated_session

    if start_clicked:
        with st.spinner("正在启动会话式仿真..."):
            index_symbol = INDEX_BENCHMARK_OPTIONS[index_label]
            history_end = _policy_anchor_date() - pd.Timedelta(days=1)
            real_history = _load_real_index_history(index_symbol, max(int(history_window_days), int(total_days)), history_end)
            if real_history.empty:
                st.info("未获取到真实指数数据，已切换到仿真基准。")
            policy_name = (
                str(selected_template.get("title", "") or "").strip()
                if isinstance(selected_template, dict)
                else ""
            ) or "自定义政策仿真"
            policy_type = (
                str(selected_template.get("policy_type", "") or "").strip()
                if isinstance(selected_template, dict)
                else ""
            ) or "综合政策"
            session = _policy_session_new(
                policy_name=policy_name,
                policy_text=policy_text,
                policy_type=policy_type,
                total_days=int(total_days),
                intensity=float(intensity),
                effective_day=int(effective_day),
                half_life_days=int(half_life_days),
                rumor_noise=bool(rumor_noise),
                index_label=index_label,
                index_symbol=index_symbol,
                reference_frame=real_history,
                runtime_profile=runtime_profile,
            )
            session["status"] = "running"
            session["calendar_start"] = pd.bdate_range(start=pd.Timestamp(history_end).normalize(), periods=2)[-1].strftime("%Y-%m-%d")
            session["autoplay"] = {"enabled": True, "step_days": 1, "interval_seconds": 1.0, "last_wallclock_ts": 0.0}
            session["history_window_days"] = int(history_window_days)
            session["real_index_frame"] = real_history.to_dict(orient="records") if not real_history.empty else []
            session["template_id"] = str(selected_template.get("id", "custom")) if isinstance(selected_template, dict) else "custom"
            session["template_title"] = policy_name
            session["template_category"] = str(selected_template.get("category", "自定义场景")) if isinstance(selected_template, dict) else "自定义场景"
            session["control_label"] = str(selected_template.get("control_label", "无政策基线对照")) if isinstance(selected_template, dict) else "无政策基线对照"
            session["control_text"] = str(selected_template.get("control_text", "在相同初始条件下，观察不采取干预时的市场路径。")) if isinstance(selected_template, dict) else "在相同初始条件下，观察不采取干预时的市场路径。"
            _policy_session_advance(session, 1)
            _store_session(session)
            st.success("仿真已开始，并已自动推演第 1 个交易日。")
        session = st.session_state.get("policy_lab_session")

    session = st.session_state.get("policy_lab_session")
    
    if not session:
        st.info("入口编译预览已就绪。确认政策文本或模板后，点击“开始推演”进入多智能体市场路径与反事实甲乙对照。")
        return
        
    control_cols = st.columns(4)
    can_advance = str(session.get("status", "")).lower() in {"running", "paused"}
    
    autoplay = session.get("autoplay", {"enabled": False, "step_days": 1, "interval_seconds": 1.0, "last_wallclock_ts": 0.0})
    
    playing_label = "暂停自动运行" if autoplay.get("enabled", False) else "恢复自动运行"
    if control_cols[0].button(playing_label, use_container_width=True, disabled=not can_advance, key="policy_lab_pause_resume"):
        autoplay["enabled"] = not autoplay.get("enabled", False)
        session["autoplay"] = autoplay
        _store_session(session)
    
    remaining_days = max(0, int(session.get("total_days", 0)) - int(session.get("current_day", 0)))
    if control_cols[1].button("运行到结束", use_container_width=True, disabled=not can_advance or remaining_days <= 0, key="policy_lab_jump_end"):
        session["autoplay"]["enabled"] = False
        _policy_session_advance(session, remaining_days)
        _store_session(session)
        st.success("已直接推进到仿真结束。")
        
    if control_cols[2].button("停止仿真", use_container_width=True, disabled=not can_advance, key="policy_lab_stop"):
        _policy_session_stop(session)
        session["autoplay"]["enabled"] = False
        _store_session(session)
        st.warning("仿真已强制停止。")
        
    if control_cols[3].button("重置会话", use_container_width=True, key="policy_lab_reset"):
        st.session_state.pop("policy_lab_session", None)
        st.session_state.pop("policy_lab_result", None)
        st.session_state.pop("policy_lab_report", None)
        st.success("会话已重置。")
        st.rerun()

    if can_advance:
        interval = st.slider(
            "自动运行速度 (秒/交易日)",
            min_value=0.5, max_value=3.0, value=float(autoplay.get("interval_seconds", 1.0)), step=0.1, key="policy_lab_speed_slider"
        )
        autoplay["interval_seconds"] = interval
        autoplay["step_days"] = 1
        if not autoplay["enabled"]:
            autoplay["last_wallclock_ts"] = 0.0
        session["autoplay"] = autoplay
        _store_session(session)

    if autoplay.get("enabled", False) and str(session.get("status", "")).lower() in {"running", "paused"}:
        if _policy_session_maybe_autoplay(session):
            _store_session(session)
            session = st.session_state.get("policy_lab_session")

    session_frame = _policy_session_frame(session)
    summary = _policy_session_refresh_summary(session)
    current_day = int(session.get("current_day", 0))
    total_days = int(session.get("total_days", 0))
    active_policies = sum(1 for item in session.get("policy_events", []) if item.get("status") == "active")
    queued_policies = sum(1 for item in session.get("policy_events", []) if item.get("status") == "queued")
    policy_status = _policy_session_status_text(str(session.get("status", "idle")))
    briefing = _build_policy_demo_briefing(session, summary)
    pulse_cards = _build_policy_market_pulse(session, summary)

    chip_html = "".join(
        f"<span class='hero-chip'>{chip}</span>"
        for chip in briefing.get("chips", [])
    )
    bullet_html = "".join(
        f"<li>{item}</li>"
        for item in briefing.get("bullets", [])
    )
    st.markdown(
        f"""
        <div class="hero-panel">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="flex:1;min-width:320px;">
              <div class="hero-kicker">{briefing['kicker']}</div>
              <h1>{briefing['phase']}</h1>
              <p style="max-width:760px;margin-top:10px;">{briefing['subtitle']}</p>
              <div class="hero-chip-row">{chip_html}</div>
            </div>
            <div class="hero-alert hero-alert-{briefing['tone']}">
              <div class="hero-alert-label">运行提示</div>
              <div class="hero-alert-value">{briefing['alert']}</div>
            </div>
          </div>
          <ul class="hero-briefing-list">{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pulse_cols = st.columns(len(pulse_cards))
    for col, item in zip(pulse_cols, pulse_cards):
        col.markdown(
            f"""
            <div class="pulse-card pulse-card-{item['tone']}">
              <div class="summary-label">{item['label']}</div>
              <div class="summary-value">{item['value']}</div>
              <div class="summary-note">{item['note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scene_cols = st.columns(3)
    scene_cols[0].markdown(
        f"""
        <div class="summary-card">
          <div class="summary-label">当前场景</div>
          <div class="summary-value">{str(session.get('template_title', session.get('policy_name', '自定义政策仿真')))}</div>
          <div class="summary-note">类别：{str(session.get('template_category', '自定义场景'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    scene_cols[1].markdown(
        f"""
        <div class="summary-card">
          <div class="summary-label">甲乙对照基线</div>
          <div class="summary-value">{str(session.get('control_label', '无政策基线对照'))}</div>
          <div class="summary-note">{str(session.get('control_text', '在相同初始条件下观察未干预路径。'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    scene_cols[2].markdown(
        f"""
        <div class="summary-card">
          <div class="summary-label">模型参与说明</div>
          <div class="summary-value">{provider_label} 已接入会话推演链路</div>
          <div class="summary-note">当前候选链：{primary_model_label}；用于政策理解、结构化抽取、部分智能体认知推理与解释生成。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 会话总览")
    metric_cols = st.columns(6)
    metric_cols[0].metric("会话状态", policy_status)
    metric_cols[1].metric("已推进交易日", f"{current_day}/{total_days}")
    metric_cols[2].metric("活跃政策数", f"{active_policies}")
    metric_cols[3].metric("待生效政策数", f"{queued_policies}")
    metric_cols[4].metric("最新收盘价", f"{float(session.get('last_close', 0.0)):.2f}")
    metric_cols[5].metric("会话波动率", f"{summary.get('volatility', 0.0):.2%}")
    st.progress(min(1.0, current_day / max(total_days, 1)))

    latest_step_report = dict(session.get("latest_step_report", {}) or {})
    demo_cards = _build_policy_demo_cards(
        policy_text=str(session.get("policy_text", "")),
        latest_step_report=latest_step_report,
        session_summary=summary,
    )
    st.markdown("### 政策传导速览")
    card_cols = st.columns(len(demo_cards))
    for col, card in zip(card_cols, demo_cards):
        col.markdown(
            f"""
            <div class="story-card">
              <div class="summary-label">{card['phase']}</div>
              <div class="story-card-title" style="margin: 6px 0 10px 0;">{card['summary']}</div>
              <div class="summary-note">{card['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    package_dict = _policy_session_policy_package(session)
    market_tab, behavior_tab, agent_tab = st.tabs(["结果总览", "行为金融", "主体决策剖面"])

    with market_tab:
        selected_benchmark_label = st.selectbox(
            "基准指数",
            options=list(INDEX_BENCHMARK_OPTIONS.keys()),
            index=list(INDEX_BENCHMARK_OPTIONS.values()).index(str(session.get("index_symbol", "sh000001")))
            if str(session.get("index_symbol", "sh000001")) in INDEX_BENCHMARK_OPTIONS.values()
            else 0,
            key=f"policy_lab_workbench_benchmark_{presentation_mode}",
        )
        selected_benchmark_symbol = INDEX_BENCHMARK_OPTIONS[selected_benchmark_label]
        real_index_frame = _policy_session_real_index_frame({**session, "index_symbol": selected_benchmark_symbol})
        event_markers = _policy_session_event_markers(session)
        synthetic_tape = dashboard_workbench.build_synthetic_trade_tape_from_market_frame(
            session_frame,
            symbol=selected_benchmark_symbol,
            seed=int(session.get("seed", 42) or 42),
            config_hash=str(session.get("config_hash", "")),
            data_snapshot_hash=str(session.get("data_snapshot_id", "")),
        )
        if not session_frame.empty:
            st.caption(f"区间：{session_frame.iloc[0]['time']} 至 {session_frame.iloc[-1]['time']}")
            _render_quote_banner(
                session_frame,
                index_label=selected_benchmark_label,
                index_symbol=selected_benchmark_symbol,
                source_text="逐笔成交聚合 K 线 + 真实指数叠加",
                history_end=str(session_frame.iloc[-1]["time"]),
            )
            kline_fig, chart_frame = dashboard_workbench.render_trade_tape_kline_workbench(
                synthetic_tape,
                symbol=selected_benchmark_symbol,
                benchmark_symbol=selected_benchmark_symbol,
                market_frame=session_frame,
                real_index_frame=real_index_frame,
                events=event_markers,
                key_prefix=f"policy_lab_trade_tape_kline_{presentation_mode}",
                seed=int(session.get("seed", 42) or 42),
                config_hash=str(session.get("config_hash", "")),
                data_snapshot_hash=str(session.get("data_snapshot_id", "")),
            )
            registry_entry = _policy_session_registry_entry(session, chart_frame)
            repro_meta = _policy_session_repro_meta(
                session,
                registry_entry=registry_entry,
                chart_frame=chart_frame,
                runtime_profile=runtime_profile,
            )
            session["experiment_registry"] = registry_entry
            session["reproducibility"] = repro_meta
            session["workbench_ohlcv_rows"] = chart_frame.to_dict(orient="records")
            _store_session(session)
        else:
            st.info("尚未生成交易路径，请点击“继续 1 天”或“运行到结束”。")
            chart_frame = pd.DataFrame()
            real_index_frame = pd.DataFrame()
            event_markers = []
            synthetic_tape = []
            registry_entry = {}
            repro_meta = {}

        if package_dict and not session_frame.empty:
            left, right = st.columns(2)
            policy_text_str = str(session.get("policy_text", ""))
            with left:
                _render_agent_disagreement_chart(package_dict, policy_text_str, runtime_profile)
            with right:
                _render_role_orderflow_waterfall(session_frame, package_dict, policy_text_str, runtime_profile)
            _render_sector_rotation_heatmap(package_dict, policy_text_str, runtime_profile)
            
            st.markdown("#### 政策影响解读")
            narrative = _get_policy_narrative(
                policy_text_str,
                summary,
                package_dict,
                runtime_profile,
            )
            
            narrative_html = str(narrative or "暂无可用解读。").replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="background: #1e1e24; padding: 20px; border-radius: 8px; color: #cbd5e1; border-left: 4px solid #64748b;">
                    {narrative_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            _render_policy_package_summary(package_dict, summary)

        st.markdown("### 反事实对照推演")
        counterfactual_source = (
            session_frame[["step", "time", "open", "high", "low", "close", "volume", "csad", "panic_level"]]
            if not session_frame.empty
            else pd.DataFrame()
        )
        counterfactual = _build_regulation_counterfactual_worlds(
            counterfactual_source,
            intensity=float(max(summary.get("policy_signal_avg", 1.0), 0.1)),
        )
        if counterfactual.get("worlds"):
            _render_regulation_counterfactual_panel(counterfactual)
        else:
            st.info("当前数据量不足，继续推进会话后可查看反事实对照结果。")

        _render_discovered_metrics_panel(dict(session.get("discovered_metrics", {}) or {}))

        st.markdown("### 研究工作台")
        workbench_tabs = st.tabs(["场景对比", "回放", "评估卡", "可复现信息"])
        with workbench_tabs[0]:
            scenario_frames = {}
            if counterfactual.get("worlds"):
                worlds = dict(counterfactual.get("worlds", {}) or {})
                scenario_frames = {
                    "baseline": pd.DataFrame(worlds.get("no_intervention", []) or []),
                    "policy_a": pd.DataFrame(worlds.get("early_intervention", []) or []),
                    "policy_b": pd.DataFrame(worlds.get("late_intervention", []) or []),
                    "optimized_policy": pd.DataFrame(worlds.get(counterfactual.get("recommended_timing", "early_intervention"), []) or []),
                }
            scenario_diff = render_scenario_diff(
                scenario_frames,
                base_frame=chart_frame if not chart_frame.empty else session_frame,
                key_prefix=f"policy_lab_scenario_diff_{presentation_mode}",
            )
            session["scenario_diff"] = {
                key: value.to_dict(orient="records") for key, value in scenario_diff.items() if isinstance(value, pd.DataFrame)
            }
        with workbench_tabs[1]:
            replay_snapshot = render_replay_scrubber(
                chart_frame if not chart_frame.empty else session_frame,
                events=event_markers,
                trade_tape=[item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in synthetic_tape],
                reports=[dict(session.get("latest_step_report", {}) or {})],
                key_prefix=f"policy_lab_replay_{presentation_mode}",
            )
            session["last_replay_snapshot"] = replay_snapshot
        with workbench_tabs[2]:
            scorecard = _policy_session_scorecard(session, chart_frame if not chart_frame.empty else session_frame, real_index_frame, event_markers)
            session["scorecard"] = scorecard
            render_scorecard_panel(scorecard, key_prefix=f"policy_lab_scorecard_{presentation_mode}")
        with workbench_tabs[3]:
            render_experiment_registry(registry_entry or session.get("experiment_registry", {}), key_prefix=f"policy_lab_registry_{presentation_mode}")
            render_reproducibility_panel(repro_meta or session.get("reproducibility", {}), key_prefix=f"policy_lab_repro_{presentation_mode}")
        _store_session(session)

        with st.expander("查看政策时间轴、追加政策与日度明细", expanded=False):
            st.markdown("#### 政策时间轴")
            timeline_df = pd.DataFrame(session.get("policy_timeline", []) or _policy_session_timeline(session))
            if not timeline_df.empty:
                st.dataframe(localize_dataframe_columns(timeline_df), use_container_width=True, hide_index=True)
            else:
                st.info("当前还没有追加政策。")

            st.markdown("#### 运行时事件时间轴")
            runtime_timeline_df = pd.DataFrame(session.get("runtime_event_timeline", []) or [])
            if not runtime_timeline_df.empty:
                preferred_cols = [
                    col
                    for col in [
                        "status",
                        "event_type",
                        "title",
                        "effective_day",
                        "current_strength",
                        "scope",
                        "confidence",
                        "source",
                    ]
                    if col in runtime_timeline_df.columns
                ]
                st.dataframe(
                    localize_dataframe_columns(runtime_timeline_df[preferred_cols] if preferred_cols else runtime_timeline_df),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("当前还没有运行时事件。")

            event_digest = dict(session.get("event_digest", {}) or {})
            digest_cols = st.columns(4)
            digest_cols[0].metric("活跃事件", int(event_digest.get("active_count", 0) or 0))
            digest_cols[1].metric("待进入", int(event_digest.get("queued_count", 0) or 0))
            digest_cols[2].metric("已过期", int(event_digest.get("expired_count", 0) or 0))
            digest_cols[3].metric("事件强度", f"{float(event_digest.get('aggregate_strength', 0.0) or 0.0):.2f}")

            st.markdown("#### 追加政策设置")
            can_append = str(session.get("status", "")).lower() in {"running", "paused"}
            append_cols = st.columns(2)
            with append_cols[0]:
                with st.form(f"policy_lab_append_form_{presentation_mode}"):
                    append_policy_name = "追加政策"
                    append_policy_text = st.text_area(
                        "追加政策文本",
                        value="",
                        height=110,
                        key=f"policy_lab_append_policy_text_{presentation_mode}",
                        placeholder="请输入希望在未来交易日追加生效的政策文本。",
                    )
                    append_policy_type = "综合政策"
                    append_effective_day = st.number_input(
                        "生效交易日",
                        min_value=current_day + 1,
                        max_value=max(total_days, current_day + 1),
                        value=min(max(current_day + 1, 1), max(total_days, current_day + 1)),
                        step=1,
                        key=f"policy_lab_append_effective_day_{presentation_mode}",
                    )
                    append_intensity = st.slider(
                        "追加强度",
                        min_value=0.2,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key=f"policy_lab_append_intensity_{presentation_mode}",
                    )
                    append_half_life = st.slider(
                        "追加政策半衰期（交易日）",
                        min_value=1,
                        max_value=120,
                        value=30,
                        step=1,
                        key=f"policy_lab_append_half_life_{presentation_mode}",
                    )
                    append_rumor_noise = st.checkbox(
                        "追加政策包含传言噪声",
                        value=False,
                        key=f"policy_lab_append_rumor_noise_{presentation_mode}",
                    )
                    append_submitted = st.form_submit_button("追加政策", use_container_width=True, disabled=not can_append)
            if append_submitted:
                if not can_append:
                    st.warning("请先开始仿真，再追加政策。")
                elif not str(append_policy_text).strip():
                    st.warning("追加政策文本不能为空。")
                else:
                    _policy_session_enqueue(
                        session,
                        policy_name=append_policy_name,
                        policy_text=append_policy_text,
                        policy_type=append_policy_type,
                        effective_day=int(append_effective_day),
                        intensity=float(append_intensity),
                        half_life_days=int(append_half_life),
                        rumor_noise=bool(append_rumor_noise),
                    )
                    _store_session(session)
                    st.success("政策已加入会话队列。")

            with append_cols[1]:
                st.markdown("#### 注入重大新闻/谣言/冲击")
                with st.form(f"policy_lab_runtime_event_form_{presentation_mode}"):
                    event_type_label = st.selectbox(
                        "事件类型",
                        options=["major_news", "policy", "rumor", "refute", "macro_shock", "regime_shift", "regulatory_action"],
                        format_func=lambda value: {
                            "major_news": "重大新闻",
                            "policy": "新政策",
                            "rumor": "谣言",
                            "refute": "辟谣/澄清",
                            "macro_shock": "宏观冲击",
                            "regime_shift": "状态切换",
                            "regulatory_action": "监管行动",
                        }.get(value, value),
                        key=f"policy_lab_event_type_{presentation_mode}",
                    )
                    event_title = st.text_input(
                        "事件标题",
                        value="盘中重大事件",
                        key=f"policy_lab_event_title_{presentation_mode}",
                    )
                    event_text = st.text_area(
                        "事件正文",
                        value="",
                        height=110,
                        key=f"policy_lab_event_text_{presentation_mode}",
                        placeholder="例如：盘中传出某行业融资收紧消息，投资者风险偏好快速下降。",
                    )
                    event_scope = st.selectbox(
                        "作用范围",
                        options=["broad_market", "sector", "symbol", "expectations", "liquidity", "funding"],
                        index=0,
                        key=f"policy_lab_event_scope_{presentation_mode}",
                    )
                    event_effective_day = st.number_input(
                        "进入交易日",
                        min_value=current_day + 1,
                        max_value=max(total_days, current_day + 1),
                        value=min(max(current_day + 1, 1), max(total_days, current_day + 1)),
                        step=1,
                        key=f"policy_lab_event_effective_day_{presentation_mode}",
                    )
                    event_strength = st.slider(
                        "事件强度",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key=f"policy_lab_event_strength_{presentation_mode}",
                    )
                    event_half_life = st.slider(
                        "事件半衰期（交易日）",
                        min_value=1,
                        max_value=60,
                        value=7,
                        step=1,
                        key=f"policy_lab_event_half_life_{presentation_mode}",
                    )
                    event_submitted = st.form_submit_button("注入事件", use_container_width=True, disabled=not can_append)
            if event_submitted:
                if not can_append:
                    st.warning("请先开始仿真，再注入运行时事件。")
                elif not str(event_text).strip():
                    st.warning("事件正文不能为空。")
                else:
                    _policy_session_enqueue_runtime_event(
                        session,
                        event_type=str(event_type_label),
                        title=str(event_title or "运行时事件"),
                        raw_text=str(event_text),
                        effective_day=int(event_effective_day),
                        strength=float(event_strength),
                        half_life_days=int(event_half_life),
                        scope=str(event_scope),
                    )
                    _store_session(session)
                    st.success("运行时事件已加入时间轴。")

            st.markdown("#### 日度明细")
            display_frame = _policy_session_display_frame(session_frame)
            if not display_frame.empty:
                st.dataframe(localize_dataframe_columns(display_frame.tail(60)), use_container_width=True, hide_index=True)
            else:
                st.info("继续仿真后，这里将展示逐日市场路径与交易明细。")

        st.markdown("### 政策评估报告（在线预览）")
        report_bundle = session.get("report_bundle")
        report_ready = not session_frame.empty
        if st.button(
            "生成/更新政策评估报告",
            use_container_width=True,
            disabled=not report_ready,
            key=f"policy_lab_generate_report_{presentation_mode}",
        ):
            with st.spinner("正在生成政策评估报告..."):
                payload = _policy_session_generate_report(session, runtime_profile)
                _store_session(session)
                st.success("政策评估报告已生成。")
                report_bundle = session.get("report_bundle")
                st.session_state["policy_lab_report"] = payload

        if report_bundle:
            report_meta = report_bundle.get("report_meta", {})
            disabled_reasons = dict(report_bundle.get("disabled_reasons", {}) or {})
            report_preview = _sanitize_report_markdown_for_frontend(str(report_bundle.get("markdown_text", "")))
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">报告已生成</div>
                  <div class="summary-value">{report_meta.get('title', report_bundle.get('stem', '政策评估报告'))}</div>
                  <div class="summary-note">编号：{report_meta.get('report_no', '')}</div>
                  <div class="summary-note">收件对象：{report_meta.get('recipient', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if disabled_reasons:
                reason_text = "；".join(f"{k}: {v}" for k, v in disabled_reasons.items())
                st.info(f"部分导出能力降级：{reason_text}")
            st.markdown("#### 报告正文预览")
            st.markdown(report_preview)
            export_frame = pd.DataFrame(session.get("workbench_ohlcv_rows", []) or session_frame.to_dict(orient="records"))
            table_exports = dataframe_export_bundle(export_frame, stem=f"{report_bundle['stem']}_ohlcv")
            if table_exports.get("disabled_reasons"):
                st.caption("Parquet 导出降级：" + "；".join(table_exports["disabled_reasons"].values()))
            download_cols = st.columns(6)
            with download_cols[0]:
                st.download_button(
                    "下载 Word（文档）",
                    data=report_bundle.get("docx_bytes") or b"",
                    file_name=f"{report_bundle['stem']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    disabled=report_bundle.get("docx_bytes") is None,
                    key=f"policy_lab_report_docx_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[1]:
                st.download_button(
                    "下载 PDF（版式）",
                    data=report_bundle.get("pdf_bytes") or b"",
                    file_name=f"{report_bundle['stem']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    disabled=report_bundle.get("pdf_bytes") is None,
                    key=f"policy_lab_report_pdf_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[2]:
                st.download_button(
                    "下载 Markdown（文本）",
                    data=report_preview,
                    file_name=f"{report_bundle['stem']}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"policy_lab_report_md_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[3]:
                st.download_button(
                    "下载 JSON（数据）",
                    data=report_bundle["json_text"],
                    file_name=f"{report_bundle['stem']}.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"policy_lab_report_json_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[4]:
                st.download_button(
                    "下载 CSV（K线数据）",
                    data=table_exports["csv_bytes"],
                    file_name=f"{table_exports['stem']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"policy_lab_report_csv_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[5]:
                st.download_button(
                    "下载 Parquet",
                    data=table_exports.get("parquet_bytes") or b"",
                    file_name=f"{table_exports['stem']}.parquet",
                    mime="application/octet-stream",
                    use_container_width=True,
                    disabled=table_exports.get("parquet_bytes") is None,
                    key=f"policy_lab_report_parquet_{presentation_mode}_{report_bundle['stem']}",
                )

    with behavior_tab:
        _render_behavior_finance_panel(session_frame, summary)

    with agent_tab:
        _render_agent_fmri_panel(session, package_dict)

    autoplay = _policy_session_autoplay_state(session)
    if autoplay.get("enabled", False) and str(session.get("status", "")).lower() in {"running", "paused"}:
        time.sleep(float(autoplay.get("interval_seconds", 0.8) or 0.0))
        st.rerun()
