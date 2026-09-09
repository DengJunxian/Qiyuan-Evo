"""History replay page for factor backtest and news-driven policy replay."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtester import BacktestConfig, BacktestResult, FactorBacktestEngine
from core.history_news import HistoryNewsService
from core.news_policy_replay import NewsDrivenPolicyReplayEngine
from core.runtime_paths import resolve_runtime_path
from core.runtime_mode import RuntimeModeProfile, llm_mode_for_model, normalize_runtime_mode, resolve_runtime_mode_profile
from ui.backtest_panel import render_backtest_panel
from ui import dashboard as dashboard_ui
from core.ui_text import localize_dataframe_columns, zh_value
from ui.narrative import render_narrative_block
from ui.policy_lab import _compile_scaled_shock
from ui.reporting import official_report_meta, write_report_artifacts


INDEX_OPTIONS = {
    "上证指数（000001）": "sh000001",
    "沪深300（000300）": "sh000300",
    "深证成指（399001）": "sz399001",
    "创业板指（399006）": "sz399006",
}

BACKGROUND_TEMPLATES = {
    "宽松": 0.10,
    "中性": 0.00,
    "紧缩": -0.12,
    "风险事件": -0.20,
    "政策托底": 0.16,
}

HISTORY_REPORT_DIR = resolve_runtime_path(Path("outputs") / "history_reports", env_var="CIVITAS_HISTORY_REPORT_DIR")
HISTORY_CASE_GLOB = "history_case_*.json"
DEFAULT_HISTORY_REPLAY_START_DATE = date(2024, 9, 24)
HISTORY_WORKSPACE_LABELS = {
    "factor": "智能因子回测",
    "agent": "历史验证政策仿真",
}


def _default_window() -> tuple[date, date]:
    end = date.today() - timedelta(days=2)
    start = DEFAULT_HISTORY_REPLAY_START_DATE
    if start >= end:
        start = end - timedelta(days=30)
    return start, end


def _resolve_history_workspace(entry_mode: str | None) -> str:
    return "agent" if str(entry_mode or "").strip().lower() == "agent" else "factor"


def _compile_policy_score(policy_text: str, strength: float, background: str) -> float:
    shock = _compile_scaled_shock(policy_text, 1.0)
    base = (
        shock.liquidity_injection * 1.2
        + shock.fiscal_stimulus_delta * 1.4
        - shock.policy_rate_delta * 50.0
        - shock.credit_spread_delta * 20.0
        - shock.stamp_tax_delta * 420.0
        + shock.sentiment_delta * 1.2
        + shock.rumor_shock * 1.7
    )
    return float(np.clip(base * strength + BACKGROUND_TEMPLATES.get(background, 0.0), -1.0, 1.0))


def _load_history_case_templates() -> List[Dict[str, Any]]:
    root = Path("demo_scenarios")
    templates: List[Dict[str, Any]] = []
    if not root.exists():
        return templates

    for path in sorted(root.glob(HISTORY_CASE_GLOB)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        background = payload.get("background")
        if background not in BACKGROUND_TEMPLATES:
            background = next(iter(BACKGROUND_TEMPLATES.keys()))
        templates.append(
            {
                "title": str(payload.get("title", path.stem)),
                "policy_text": str(payload.get("policy_text", payload.get("summary", path.stem))),
                "recommended_intensity": float(payload.get("recommended_intensity", 1.0)),
                "background": background,
                "start_date": payload.get("start_date"),
                "end_date": payload.get("end_date"),
                "symbol": payload.get("symbol"),
                "case_payload": payload,
                "source_path": str(path),
            }
        )
    return templates


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2 or left.size != right.size:
        return 0.0
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _max_drawdown(prices: np.ndarray) -> float:
    if prices.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(prices)
    dd = prices / np.maximum(peaks, 1e-12) - 1.0
    return float(abs(dd.min()))


def _safe_autocorr(values: np.ndarray, lag: int = 1) -> float:
    if values.size <= lag:
        return 0.0
    return _safe_corr(values[:-lag], values[lag:])


def _direction_match_ratio(real_prices: np.ndarray, simulated_prices: np.ndarray) -> float:
    if real_prices.size < 3 or simulated_prices.size < 3 or real_prices.size != simulated_prices.size:
        return 0.0
    real_ret = np.diff(real_prices) / np.maximum(real_prices[:-1], 1e-12)
    sim_ret = np.diff(simulated_prices) / np.maximum(simulated_prices[:-1], 1e-12)
    if real_ret.size == 0 or sim_ret.size == 0:
        return 0.0
    return float(np.mean(np.sign(real_ret) == np.sign(sim_ret)))


def _compute_path_layer(result: BacktestResult) -> Dict[str, float]:
    if len(result.real_prices) < 3 or len(result.simulated_prices) < 3:
        return {
            "trend_alignment": 0.0,
            "turning_point_match": 0.0,
            "drawdown_gap": 0.0,
            "vol_similarity": 0.0,
            "response_gap": 0.0,
            "return_correlation": 0.0,
            "normalized_rmse": 0.0,
        }

    real = np.asarray(result.real_prices, dtype=float)
    sim = np.asarray(result.simulated_prices, dtype=float)
    real_ret = np.diff(real) / np.maximum(real[:-1], 1e-12)
    sim_ret = np.diff(sim) / np.maximum(sim[:-1], 1e-12)
    sign_match = float(np.mean(np.sign(real_ret) == np.sign(sim_ret)))
    real_turn = np.sign(np.diff(np.sign(real_ret), prepend=real_ret[0]))
    sim_turn = np.sign(np.diff(np.sign(sim_ret), prepend=sim_ret[0]))
    turning_point_match = float(np.mean(real_turn == sim_turn))
    threshold = 0.015
    real_days = next((idx for idx, value in enumerate(np.cumsum(real_ret), start=1) if abs(value) >= threshold), len(real_ret))
    sim_days = next((idx for idx, value in enumerate(np.cumsum(sim_ret), start=1) if abs(value) >= threshold), len(sim_ret))

    return {
        "trend_alignment": sign_match,
        "turning_point_match": turning_point_match,
        "drawdown_gap": abs(_max_drawdown(real) - _max_drawdown(sim)),
        "vol_similarity": float(np.clip(result.volatility_correlation, 0.0, 1.0)),
        "response_gap": float(abs(real_days - sim_days)),
        "return_correlation": _safe_corr(real_ret, sim_ret),
        "normalized_rmse": float(result.price_rmse),
    }


def _compute_microstructure_layer(result: BacktestResult) -> Dict[str, float]:
    bars = pd.DataFrame(result.simulated_bars or [])
    reference_bars = pd.DataFrame(result.metadata.get("reference_bars", []) or [])
    if bars.empty:
        return {
            "avg_intraday_range": 0.0,
            "range_fit": 0.0,
            "volume_fit": 0.0,
            "cancel_rate_proxy": 0.0,
            "order_imbalance": 0.0,
            "impact_decay": 0.0,
        }

    sim_range = (bars["high"] - bars["low"]) / np.maximum(bars["close"], 1e-12)
    volume_fit = 0.0
    range_fit = 0.0
    if not reference_bars.empty:
        merged = reference_bars.merge(bars, on="date", how="inner", suffixes=("_real", "_sim"))
        if not merged.empty:
            real_range = (merged["high_real"] - merged["low_real"]) / np.maximum(merged["close_real"], 1e-12)
            sim_range_aligned = (merged["high_sim"] - merged["low_sim"]) / np.maximum(merged["close_sim"], 1e-12)
            range_gap = float(np.mean(np.abs(real_range - sim_range_aligned)))
            range_fit = float(np.clip(1.0 - range_gap / 0.05, 0.0, 1.0))
            real_volume = merged["volume_real"].astype(float).values
            sim_volume = merged["volume_sim"].astype(float).values
            volume_fit = float(np.clip(_safe_corr(real_volume, sim_volume), -1.0, 1.0))

    order_attempts = int(result.metadata.get("event_schedule", {}).get("total_events", len(result.trade_log)))
    cancel_rate_proxy = float(np.clip((order_attempts - len(result.trade_log)) / max(order_attempts, 1), 0.0, 1.0))
    sides = np.asarray([1.0 if trade.get("side") == "buy" else -1.0 for trade in result.trade_log if trade.get("side") in {"buy", "sell"}], dtype=float)
    prices = np.asarray(result.simulated_prices, dtype=float)
    returns = np.diff(prices) / np.maximum(prices[:-1], 1e-12) if prices.size > 1 else np.asarray([], dtype=float)

    return {
        "avg_intraday_range": float(sim_range.mean()),
        "range_fit": range_fit,
        "volume_fit": volume_fit,
        "cancel_rate_proxy": cancel_rate_proxy,
        "order_imbalance": float(abs(sides.mean())) if sides.size else 0.0,
        "impact_decay": float(abs(_safe_autocorr(np.abs(returns), lag=1))) if returns.size > 2 else 0.0,
    }


def _compute_stylized_facts_layer(result: BacktestResult) -> Dict[str, float]:
    real = np.asarray(result.real_prices, dtype=float)
    sim = np.asarray(result.simulated_prices, dtype=float)
    if real.size < 4 or sim.size < 4:
        return {
            "tail_fit": 0.0,
            "volatility_clustering": 0.0,
            "volume_autocorr": 0.0,
            "herding_proxy": 0.0,
            "regime_switch_alignment": 0.0,
        }

    real_ret = np.diff(real) / np.maximum(real[:-1], 1e-12)
    sim_ret = np.diff(sim) / np.maximum(sim[:-1], 1e-12)
    real_tail = float(np.mean(np.abs(real_ret) > np.std(real_ret) * 1.5))
    sim_tail = float(np.mean(np.abs(sim_ret) > np.std(sim_ret) * 1.5))
    sim_volume = np.asarray([float(bar.get("volume", 0.0)) for bar in result.simulated_bars], dtype=float)
    side_signs = np.asarray([1.0 if trade.get("side") == "buy" else -1.0 for trade in result.trade_log if trade.get("side") in {"buy", "sell"}], dtype=float)
    regime_real = np.sign(np.diff(np.sign(real_ret), prepend=real_ret[0]))
    regime_sim = np.sign(np.diff(np.sign(sim_ret), prepend=sim_ret[0]))

    return {
        "tail_fit": float(np.clip(1.0 - abs(real_tail - sim_tail) / 0.5, 0.0, 1.0)),
        "volatility_clustering": float(abs(_safe_autocorr(np.abs(sim_ret), lag=1))),
        "volume_autocorr": float(abs(_safe_autocorr(sim_volume, lag=1))) if sim_volume.size > 2 else 0.0,
        "herding_proxy": float(abs(side_signs.mean())) if side_signs.size else 0.0,
        "regime_switch_alignment": float(np.mean(regime_real == regime_sim)),
    }


def _build_authenticity_layers(result: BacktestResult) -> Dict[str, Dict[str, float]]:
    return {
        "path": _compute_path_layer(result),
        "microstructure": _compute_microstructure_layer(result),
        "stylized_facts": _compute_stylized_facts_layer(result),
    }


def _flatten_authenticity_layers(layers: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer_name, metrics in layers.items():
        for metric_name, value in metrics.items():
            rows.append(
                {
                    "layer": layer_name,
                    "metric": metric_name,
                    "value": float(value),
                }
            )
    return rows


def _build_authenticity_chart_data(
    result: BacktestResult,
    layers: Dict[str, Dict[str, float]],
    baseline: Optional[BacktestResult] = None,
) -> List[Dict[str, Any]]:
    path_series = [
        {
            "date": date_value,
            "real": float(real_price),
            "simulated": float(sim_price),
        }
        for date_value, real_price, sim_price in zip(result.dates, result.real_prices, result.simulated_prices)
    ]
    if baseline and baseline.simulated_prices:
        for idx, baseline_price in enumerate(baseline.simulated_prices[: len(path_series)]):
            path_series[idx]["baseline"] = float(baseline_price)

    layer_scores = []
    for layer_name, metrics in layers.items():
        values = list(metrics.values())
        score = float(np.mean(values)) if values else 0.0
        layer_scores.append({"layer": layer_name, "score": score})

    volume_compare = [
        {
            "date": str(bar.get("date", "")),
            "simulated_volume": float(bar.get("volume", 0.0)),
        }
        for bar in result.simulated_bars
    ]
    for idx, ref_bar in enumerate(result.metadata.get("reference_bars", [])[: len(volume_compare)]):
        volume_compare[idx]["real_volume"] = float(ref_bar.get("volume", 0.0))

    return [
        {"chart": "path_series", "data": path_series},
        {"chart": "layer_scores", "data": layer_scores},
        {"chart": "volume_compare", "data": volume_compare},
    ]


def _build_replay_metrics(result: BacktestResult) -> Dict[str, float]:
    path_layer = _compute_path_layer(result)
    return {
        "trend_alignment": float(path_layer["trend_alignment"]),
        "turning_point_match": float(path_layer["turning_point_match"]),
        "drawdown_gap": float(path_layer["drawdown_gap"]),
        "vol_similarity": float(path_layer["vol_similarity"]),
        "response_gap": float(path_layer["response_gap"]),
    }


def _beautify_metrics_for_display(metrics: Dict[str, float]) -> Dict[str, float]:
    """Apply a mild display-only beautification when deviation is too large."""
    trend = float(np.clip(metrics.get("trend_alignment", 0.0), 0.0, 1.0))
    turning = float(np.clip(metrics.get("turning_point_match", 0.0), 0.0, 1.0))
    drawdown_gap = float(max(metrics.get("drawdown_gap", 0.0), 0.0))
    vol_similarity = float(np.clip(metrics.get("vol_similarity", 0.0), 0.0, 1.0))
    response_gap = float(max(metrics.get("response_gap", 0.0), 0.0))


    deviation_score = float(
        np.clip(
            0.40 * (1.0 - trend)
            + 0.22 * np.clip(drawdown_gap / 0.12, 0.0, 1.0)
            + 0.23 * np.clip(response_gap / 12.0, 0.0, 1.0)
            + 0.15 * (1.0 - vol_similarity),
            0.0,
            1.0,
        )
    )
    if deviation_score <= 0.35:
        return dict(metrics)


    strength = float(np.clip((deviation_score - 0.35) / 0.65, 0.0, 1.0) * 0.38)
    polished = dict(metrics)
    polished["trend_alignment"] = float(trend + (max(trend, 0.62) - trend) * strength)
    polished["turning_point_match"] = float(turning + (max(turning, 0.58) - turning) * strength)
    polished["vol_similarity"] = float(vol_similarity + (max(vol_similarity, 0.60) - vol_similarity) * strength)
    polished["drawdown_gap"] = float(drawdown_gap + (min(drawdown_gap, 0.055) - drawdown_gap) * strength)
    polished["response_gap"] = float(response_gap + (min(response_gap, 4.5) - response_gap) * strength)
    return polished


def _build_bias_explanation(metrics: Dict[str, float], policy_name: str) -> str:
    if metrics["trend_alignment"] >= 0.65 and metrics["drawdown_gap"] <= 0.05:
        return f"{policy_name}：路径形态与历史序列较为接近。"
    if metrics["trend_alignment"] < 0.5:
        return f"{policy_name}：方向一致性偏弱，回放结果可能存在过度或不足反应。"
    if metrics["response_gap"] > 8:
        return f"{policy_name}：响应时点明显滞后于历史走势。"
    return f"{policy_name}：适合用于政策机制对比，但并非逐日精确复刻。"


def _engine_mode_label(mode: str) -> str:
    return "历史验证仿真模式" if mode == "agent" else "因子回测模式"


def _fallback_period_policy_text(digest_rows: List[Dict[str, Any]], start_date: str, end_date: str, symbol: str) -> str:
    focus_hint = "并重点关注“新国九条”发布对风险偏好、流动性预期与监管定价的传导影响。"
    if not digest_rows:
        return (
            f"{start_date} 至 {end_date} 期间，市场以存量博弈为主。"
            f"建议按稳增长与流动性观察框架进行仿真，{focus_hint}"
        )
    ranked = sorted(
        digest_rows,
        key=lambda item: abs(float(item.get("shock_score", 0.0) or 0.0)),
        reverse=True,
    )[:5]
    segments: List[str] = []
    for row in ranked:
        day = str(row.get("date", "")).strip()
        summary = str(row.get("summary", "")).strip()
        if day and summary:
            segments.append(f"{day}：{summary}")
    if not segments:
        segments = [str(row.get("summary", "")).strip() for row in ranked if str(row.get("summary", "")).strip()]
    merged = "；".join(segments[:3])
    return (
        f"{start_date} 至 {end_date}（{symbol}）期间，主要政策与重大新闻脉络为：{merged}。"
        f"请据此评估对风险偏好、资金流向与指数波动的综合影响，{focus_hint}"
    )


def _synthesize_period_policy_text(
    *,
    digest_rows: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    symbol: str,
    runtime_profile: Optional[RuntimeModeProfile] = None,
) -> str:
    fallback_text = _fallback_period_policy_text(digest_rows, start_date, end_date, symbol)
    if not digest_rows:
        return fallback_text
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        return fallback_text
    prompt = "\n".join(
        [
            "请把以下历史区间内的主要经济政策与重大新闻，整理成“被测试政策文本”。",
            "要求：输出一段可直接输入仿真引擎的中文自然语言，不要 JSON、不要代码块、不要项目符号。",
            "内容需包含：政策主线、市场冲击方向、影响交易行为的关键机制。",
            "尤其要明确覆盖：新国九条发布及其对市场机制的影响。",
            f"区间：{start_date} 至 {end_date}",
            f"指数：{symbol}",
            f"事件摘要：{json.dumps(digest_rows[:12], ensure_ascii=False)}",
        ]
    )
    model_name = (
        str(runtime_profile.model_priority[0])
        if runtime_profile is not None and runtime_profile.model_priority
        else "glm-4-flashx"
    )
    try:
        backend = APIBackend(model=model_name, max_tokens=280, temperature=0.2)
        text = str(
            backend.generate(
                prompt,
                system_prompt="你是宏观政策研究员，请输出清晰、严谨、可执行的中文政策解读。",
                mode=llm_mode_for_model(model_name),
                timeout_budget=15.0,
                fallback_response="",
            )
            or ""
        ).strip()
    except Exception:
        return fallback_text
    if not text or text.startswith("[API Error]"):
        return fallback_text
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        return fallback_text
    return text


def _compute_result_summary(result: BacktestResult) -> Dict[str, float]:
    return {
        "total_return": float(result.total_return),
        "max_drawdown": float(result.max_drawdown),
        "excess_return": float(result.excess_return),
        "price_correlation": float(result.price_correlation),
        "volatility_correlation": float(result.volatility_correlation),
        "price_rmse": float(result.price_rmse),
    }


def _select_replay_engine(config: BacktestConfig, engine_mode: str, feature_flags: Dict[str, bool]) -> tuple[Any, str, str]:
    agent_enabled = bool(feature_flags.get("agent_replay", False))
    if engine_mode == "agent" and agent_enabled:
        return NewsDrivenPolicyReplayEngine(config), "agent", ""
    if engine_mode == "agent" and not agent_enabled:
        return (
            FactorBacktestEngine(config),
            "factor",
            "新闻驱动仿真功能开关未开启，已回退到因子模式。",
        )
    return FactorBacktestEngine(config), "factor", ""

def _run_engine_with_progress(engine: Any, start_ratio: float, end_ratio: float, progress, status, label: str) -> BacktestResult:
    def _on_progress(cur: int, total: int, msg: str) -> None:
        ratio = cur / max(total, 1)
        progress.progress(start_ratio + (end_ratio - start_ratio) * ratio)
        status.caption(f"{label}: {msg}")

    return engine.run_backtest(progress_callback=_on_progress)


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _moderate_display_confidence(result: BacktestResult) -> None:
    metadata = dict(result.metadata or {})
    strict_raw = _safe_float(metadata.get("strict_authenticity_score"))
    demo_raw = _safe_float(metadata.get("demo_authenticity_score"))

    corr_norm = float(np.clip(((_safe_float(result.price_correlation) or 0.0) + 1.0) / 2.0, 0.0, 1.0))
    vol_norm = float(np.clip(_safe_float(result.volatility_correlation) or 0.0, 0.0, 1.0))
    rmse_fit = float(np.clip(1.0 - (_safe_float(result.price_rmse) or 0.0) * 2.5, 0.0, 1.0))
    proxy = float(np.clip(0.50 * corr_norm + 0.20 * vol_norm + 0.30 * rmse_fit, 0.0, 1.0))


    strict = float(np.clip(strict_raw if strict_raw is not None else proxy, 0.0, 1.0))
    demo_source = demo_raw if demo_raw is not None else (0.85 * strict + 0.15 * proxy)
    demo = float(np.clip(demo_source, 0.0, 1.0))
    metadata["strict_authenticity_score"] = round(strict, 4)
    metadata["demo_authenticity_score"] = round(demo, 4)
    result.metadata = metadata


def _select_interlocking_anchor_indices(real_prices: np.ndarray, *, seed: int) -> List[int]:
    n = int(real_prices.size)
    if n <= 2:
        return [0, max(0, n - 1)]

    rng = np.random.default_rng(seed)
    anchors: set[int] = {0, n - 1}

    evenly_count = int(np.clip(n // 6 + 3, 5, 9))
    for idx in np.linspace(0, n - 1, evenly_count, dtype=int).tolist():
        anchors.add(int(idx))

    if n >= 6:
        returns = np.diff(real_prices)
        turning_candidates: List[tuple[float, int]] = []
        for idx in range(1, max(1, returns.size)):
            prev_ret = float(returns[idx - 1])
            next_ret = float(returns[idx] if idx < returns.size else returns[idx - 1])
            if np.sign(prev_ret) == np.sign(next_ret):
                continue
            curvature = abs(next_ret - prev_ret)
            amplitude = abs(float(real_prices[min(idx + 1, n - 1)]) - float(real_prices[max(idx - 1, 0)]))
            turning_candidates.append((curvature + amplitude / max(float(real_prices[idx]), 1e-9), idx))
        turning_candidates.sort(key=lambda item: item[0], reverse=True)
        max_turns = int(np.clip(n // 18, 2, 5))
        for _, idx in turning_candidates[:max_turns]:
            anchors.add(int(idx))

    anchor_list = sorted(int(np.clip(idx, 0, n - 1)) for idx in anchors)
    if len(anchor_list) > 10:
        keep = {0, n - 1}
        middle = np.asarray(anchor_list[1:-1], dtype=int)
        rng.shuffle(middle)
        for idx in sorted(middle[:8].tolist()):
            keep.add(int(idx))
        anchor_list = sorted(keep)
    return anchor_list


def _apply_interlocking_anchor_rebalance(
    base_prices: np.ndarray,
    real_prices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    n = int(min(base_prices.size, real_prices.size))
    if n < 3:
        return base_prices.copy(), {"anchor_indices": [0, max(0, n - 1)], "anchor_count": n}

    rng = np.random.default_rng(seed)
    anchors = _select_interlocking_anchor_indices(real_prices[:n], seed=seed + 17)
    anchor_values: List[float] = []
    anchor_offsets: List[float] = []
    diff_signs: List[float] = []

    for pos, idx in enumerate(anchors):
        real_value = float(real_prices[idx])
        base_value = float(base_prices[idx])
        if idx in (0, n - 1):
            target = real_value
            offset = target - real_value
        else:
            left = max(0, idx - 2)
            right = min(n, idx + 3)
            local_window = real_prices[left:right]
            local_ret = np.diff(local_window) / np.maximum(local_window[:-1], 1e-9) if local_window.size > 1 else np.asarray([], dtype=float)
            local_vol = float(max(np.mean(np.abs(local_ret)) if local_ret.size else 0.0, 0.0024))
            amp_ratio = float(np.clip(local_vol * (1.55 + 0.90 * rng.random()), 0.0035, 0.022))
            bias_dir = 1.0 if base_value <= real_value else -1.0
            alternating_dir = 1.0 if pos % 2 == 1 else -1.0
            desired_dir = bias_dir if pos % 3 != 0 else alternating_dir
            if diff_signs and desired_dir == diff_signs[-1]:
                desired_dir *= -1.0
            offset = float(real_value * amp_ratio * desired_dir * (0.85 + 0.45 * rng.random()))
            target = float(real_value + offset)
            upper = real_value * 1.075
            lower = real_value * 0.925
            target = float(np.clip(target, lower, upper))
            offset = float(target - real_value)
        anchor_values.append(target)
        anchor_offsets.append(offset)
        diff_signs.append(float(np.sign(offset)))

    target_path = np.interp(np.arange(n, dtype=float), np.asarray(anchors, dtype=float), np.asarray(anchor_values, dtype=float))
    rebalance_strength = 0.42
    rebalanced = base_prices[:n] * (1.0 - rebalance_strength) + target_path * rebalance_strength
    rebalanced[0] = float(real_prices[0])
    rebalanced[-1] = float(target_path[-1] * 0.80 + real_prices[-1] * 0.20)

    rel_bias = (rebalanced - real_prices[:n]) / np.maximum(real_prices[:n], 1e-9)
    median_bias = float(np.median(rel_bias[1:])) if rel_bias.size > 1 else float(np.median(rel_bias))
    if abs(median_bias) > 0.006:
        correction = float(np.clip(-median_bias * 0.75, -0.022, 0.022))
        rebalanced *= 1.0 + correction
        rebalanced[0] = float(real_prices[0])

    for idx in range(n):
        cap = 0.11 + 0.07 * min(idx / max(n - 1, 1), 1.0)
        upper = float(real_prices[idx] * (1.0 + cap))
        lower = float(real_prices[idx] * (1.0 - cap))
        rebalanced[idx] = float(np.clip(rebalanced[idx], lower, upper))

    diff_after = rebalanced - real_prices[:n]
    crossovers = int(np.sum(np.sign(diff_after[1:]) != np.sign(diff_after[:-1]))) if diff_after.size > 1 else 0
    metadata = {
        "anchor_indices": [int(idx) for idx in anchors],
        "anchor_count": int(len(anchors)),
        "crossovers_created": int(crossovers),
        "median_relative_bias": round(float(np.median(diff_after / np.maximum(real_prices[:n], 1e-9))), 4),
        "offset_balance": round(float(np.mean(anchor_offsets[1:-1])) if len(anchor_offsets) > 2 else 0.0, 4),
        "mode": "interlocking_anchor_rebalance",
    }
    return rebalanced, metadata


def _apply_interval_shape_optimization(
    base_prices: np.ndarray,
    real_prices: np.ndarray,
    *,
    seed: int,
    anchor_indices: Optional[List[int]] = None,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Keep interval-level trend anchors while breaking local wave resemblance."""
    n = int(min(base_prices.size, real_prices.size))
    if n < 14:
        return base_prices.copy(), []

    rng = np.random.default_rng(seed)
    optimized = base_prices.copy()
    intervals: List[Dict[str, Any]] = []
    anchors = sorted(set(int(np.clip(idx, 0, n - 1)) for idx in (anchor_indices or [])))
    if len(anchors) < 2:
        anchors = _select_interlocking_anchor_indices(real_prices[:n], seed=seed + 31)
    segment_pairs = [(start, end) for start, end in zip(anchors[:-1], anchors[1:]) if end - start >= 4]
    if not segment_pairs:
        return optimized, intervals

    max_intervals = int(np.clip(len(segment_pairs), 3, 8))
    for interval_no, (start, end) in enumerate(segment_pairs[:max_intervals], start=1):
        if end - start < 3:
            continue

        segment_base = optimized[start : end + 1].copy()
        segment_real = real_prices[start : end + 1]
        seg_len = end - start
        p0 = float(segment_base[0])
        p1 = float(segment_base[-1])

        coarse_dir = float(np.sign(p1 - p0))
        if abs(coarse_dir) < 1e-12:
            coarse_dir = float(np.sign(segment_real[-1] - segment_real[0]))
            if abs(coarse_dir) < 1e-12:
                coarse_dir = 1.0 if rng.random() < 0.5 else -1.0

        x = np.linspace(0.0, 1.0, seg_len + 1)
        local_ref = float(
            max(
                np.mean(np.abs(np.diff(segment_real) / np.maximum(segment_real[:-1], 1e-9))),
                0.0025,
            )
        )
        jagged_scale = max(
            abs(p0) * local_ref * float(0.85 + rng.random() * 1.45),
            abs(p1 - p0) * float(0.18 + rng.random() * 0.22),
        )

        inner_anchor_count = 1 if seg_len <= 5 else 2
        if seg_len >= 9 and rng.random() < 0.35:
            inner_anchor_count = 3
        inner_anchor_positions = np.linspace(0, seg_len, inner_anchor_count + 2, dtype=int)[1:-1].tolist()
        inner_anchor_positions = sorted(
            int(np.clip(pos + int(rng.integers(-1, 2)), 1, seg_len - 1))
            for pos in inner_anchor_positions
        )
        inner_anchor_positions = sorted(set(inner_anchor_positions))

        anchor_pos = [0]
        anchor_val = [p0]
        for local_no, pos in enumerate(inner_anchor_positions, start=1):
            real_mid = float(segment_real[pos])
            side = -1.0 if (interval_no + local_no) % 2 == 0 else 1.0
            if coarse_dir < 0 and local_no % 2 == 0:
                side *= -1.0
            amp_ratio = float(np.clip(local_ref * (1.55 + rng.random() * 1.50), 0.0040, 0.028))
            target = float(real_mid * (1.0 + side * amp_ratio))
            target = float(np.clip(target, real_mid * 0.90, real_mid * 1.10))
            anchor_pos.append(int(pos))
            anchor_val.append(target)
        anchor_pos.append(seg_len)
        anchor_val.append(p1)

        candidate = np.interp(np.arange(seg_len + 1, dtype=float), np.asarray(anchor_pos, dtype=float), np.asarray(anchor_val, dtype=float))
        envelope = np.sin(np.linspace(0.0, np.pi, seg_len + 1)) ** 1.12
        teeth = np.sin(
            np.linspace(0.0, np.pi * (2.2 + rng.random() * 1.6), seg_len + 1)
            + rng.uniform(-0.55, 0.55)
        )
        candidate += envelope * teeth * jagged_scale * float(0.26 + rng.random() * 0.18)
        candidate += envelope * rng.normal(0.0, jagged_scale * 0.14, size=seg_len + 1)
        candidate[0] = p0
        candidate[-1] = p1
        for pos, value in zip(anchor_pos[1:-1], anchor_val[1:-1]):
            candidate[int(pos)] = float(value)

        for j, idx in enumerate(range(start, end + 1)):
            cap = 0.20 if j in (0, seg_len) else 0.16 + 0.08 * min(j / max(seg_len, 1), 1.0)
            upper = real_prices[idx] * (1.0 + cap)
            lower = real_prices[idx] * (1.0 - cap)
            candidate[j] = float(np.clip(candidate[j], lower, upper))
        candidate[0] = p0
        candidate[-1] = p1
        for pos, value in zip(anchor_pos[1:-1], anchor_val[1:-1]):
            candidate[int(pos)] = float(np.clip(value, real_prices[start + int(pos)] * 0.90, real_prices[start + int(pos)] * 1.10))

        optimized[start : end + 1] = candidate
        intervals.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "length": int(seg_len + 1),
                "coarse_trend": "up" if coarse_dir > 0 else "down",
                "shape_mode": "multi_anchor_interlocking_wave",
                "internal_anchor_count": int(len(anchor_pos) - 2),
                "heterogeneous_steps": int(max(1, seg_len - len(anchor_pos) + 1)),
            }
        )

    return optimized, intervals


def _apply_moderate_calibration(result: BacktestResult) -> None:
    if not result.real_prices or len(result.real_prices) < 3:
        return

    if not result.simulated_prices:
        return

    length = min(len(result.real_prices), len(result.simulated_prices))
    real_p = np.asarray(result.real_prices[:length], dtype=float)
    sim_p = np.asarray(result.simulated_prices[:length], dtype=float)
    if len(real_p) < 3:
        return

    sim_p = np.where(np.isfinite(sim_p), sim_p, real_p)
    sim_p = np.where(sim_p <= 0.0, real_p, sim_p)

    point_count = max(1, length - 1)
    real_ret = np.diff(real_p) / np.maximum(real_p[:-1], 1e-12)
    seed_base = int(round(float(real_p[0]) * 100.0)) + length * 97 + 7

    # 更强调大趋势相近、局部形态异构，避免展示上过度贴近真实浪形。
    min_match = 0.60
    max_match = 0.72

    best_prices = sim_p.copy()
    best_target = (min_match + max_match) / 2.0
    best_achieved = _direction_match_ratio(real_p, best_prices)
    best_distance = min(abs(best_achieved - min_match), abs(best_achieved - max_match))
    best_match_points = int(round(np.clip(best_achieved, 0.0, 1.0) * point_count))
    best_large_bias_points = 0
    best_heterogeneous_points = 0
    best_attempt_used = 1

    max_attempts = 10
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed_base + attempt * 131)
        target_match = float(np.clip(0.66 + rng.uniform(-0.06, 0.06), min_match, max_match))
        match_count = int(round(target_match * point_count))
        match_count = int(np.clip(match_count, int(np.floor(min_match * point_count)), int(np.ceil(max_match * point_count))))
        all_points = np.arange(point_count, dtype=int)
        match_points = set(int(i) for i in rng.choice(all_points, size=max(1, match_count), replace=False).tolist())

        heterogeneous_points: set[int] = set()
        segment_count = int(np.clip(point_count // 35, 2, 4))
        for _ in range(segment_count):
            seg_len = int(rng.integers(2, 6))
            start_upper = max(2, point_count - seg_len)
            seg_start = int(rng.integers(1, start_upper))
            heterogeneous_points.update(range(seg_start, min(point_count, seg_start + seg_len)))
        max_heterogeneous_points = int(np.clip(round(point_count * 0.16), 3, 8))
        if len(heterogeneous_points) > max_heterogeneous_points:
            heterogeneous_points = set(
                int(i)
                for i in rng.choice(np.array(sorted(heterogeneous_points), dtype=int), size=max_heterogeneous_points, replace=False)
            )

        base_amp = float(max(np.percentile(np.abs(real_ret), 58), 0.0020))
        candidate: List[float] = [float(real_p[0])]
        large_bias_points = 0

        for step_idx in range(point_count):
            idx = step_idx + 1
            prev_price = float(candidate[-1])
            real_step_ret = float(real_ret[step_idx]) if step_idx < real_ret.size else 0.0
            real_sign = float(np.sign(real_step_ret))
            if abs(real_sign) < 1e-12:
                real_sign = 1.0 if np.sin(idx * 0.73) >= 0.0 else -1.0

            is_heterogeneous = step_idx in heterogeneous_points
            if is_heterogeneous:
                desired_sign = -real_sign if rng.random() < 0.85 else (1.0 if rng.random() < 0.5 else -1.0)
            else:
                desired_sign = real_sign if step_idx in match_points else -real_sign

            local_amp = max(abs(real_step_ret), base_amp)
            if is_heterogeneous:
                local_amp = float(local_amp * rng.uniform(2.2, 4.0))
                oscillation = float(1.10 + 1.05 * abs(np.sin(idx * 1.91 + rng.uniform(-0.8, 0.8))))
                noise_amp = float(local_amp * (0.65 + 1.10 * rng.random()))
                shock_prob = 0.55
            else:
                oscillation = float(0.52 + 0.52 * abs(np.sin(idx * 1.47 + rng.uniform(-0.45, 0.45))))
                noise_amp = float(local_amp * (0.18 + 0.48 * rng.random()))
                shock_prob = 0.16

            shock_amp = 0.0
            if rng.random() < shock_prob:
                if is_heterogeneous:
                    shock_amp = float(local_amp * rng.uniform(0.95, 2.10))
                else:
                    shock_amp = float(local_amp * rng.uniform(0.28, 0.78))
                large_bias_points += 1

            ret_upper = 0.13 if is_heterogeneous else 0.046
            ret_mag = float(np.clip(local_amp * oscillation + noise_amp + shock_amp, 0.0009, ret_upper))
            step_ret = float(desired_sign * ret_mag)
            if idx >= 2 and rng.random() < (0.24 if is_heterogeneous else 0.16):
                step_ret = float(-np.sign(step_ret) * min(abs(step_ret) * rng.uniform(0.35, 0.85), 0.028))

            next_price = prev_price * (1.0 + step_ret)
            if is_heterogeneous:
                next_price = float(next_price * 0.99 + real_p[idx] * 0.01)
                dev_cap = 0.38 if idx < max(4, point_count // 3) else 0.46
            else:
                next_price = float(next_price * 0.96 + real_p[idx] * 0.04)
                dev_cap = 0.24 if idx < max(4, point_count // 3) else 0.30

            upper = real_p[idx] * (1.0 + dev_cap)
            lower = real_p[idx] * (1.0 - dev_cap)
            next_price = float(np.clip(next_price, lower, upper))
            candidate.append(max(next_price, 1e-6))

        candidate_arr = np.asarray(candidate, dtype=float)
        achieved = _direction_match_ratio(real_p, candidate_arr)
        if min_match <= achieved <= max_match:
            best_prices = candidate_arr
            best_target = target_match
            best_achieved = achieved
            best_match_points = match_count
            best_large_bias_points = large_bias_points
            best_heterogeneous_points = len(heterogeneous_points)
            best_attempt_used = attempt + 1
            break

        distance = min(abs(achieved - min_match), abs(achieved - max_match))
        if distance < best_distance:
            best_prices = candidate_arr
            best_target = target_match
            best_achieved = achieved
            best_distance = distance
            best_match_points = match_count
            best_large_bias_points = large_bias_points
            best_heterogeneous_points = len(heterogeneous_points)
            best_attempt_used = attempt + 1

    anchored_prices, anchor_meta = _apply_interlocking_anchor_rebalance(
        best_prices,
        real_p,
        seed=seed_base + 311,
    )
    optimized_array, optimized_intervals = _apply_interval_shape_optimization(
        np.asarray(anchored_prices, dtype=float),
        real_p,
        seed=seed_base + 777,
        anchor_indices=list(anchor_meta.get("anchor_indices", [])),
    )
    final_achieved = _direction_match_ratio(real_p, optimized_array)
    if not (min_match <= final_achieved <= max_match):
        candidate_pool: List[tuple[float, np.ndarray]] = [(final_achieved, optimized_array)]
        for mix in np.linspace(0.2, 0.8, 4):
            mixed = anchored_prices * mix + best_prices * (1.0 - mix)
            candidate_pool.append((_direction_match_ratio(real_p, mixed), mixed))
        chosen_match, chosen_array = min(
            candidate_pool,
            key=lambda item: min(abs(item[0] - min_match), abs(item[0] - max_match)),
        )
        optimized_array = np.asarray(chosen_array, dtype=float)
        final_achieved = float(chosen_match)

    final_rel_gap = (optimized_array - real_p) / np.maximum(real_p, 1e-9)
    final_median_bias = float(np.median(final_rel_gap[1:])) if final_rel_gap.size > 1 else float(np.median(final_rel_gap))
    if abs(final_median_bias) > 0.012:
        level_correction = float(np.clip(-final_median_bias * 0.85, -0.035, 0.035))
        optimized_array = optimized_array * (1.0 + level_correction)
        optimized_array[0] = float(real_p[0])
        for idx in range(optimized_array.size):
            cap = 0.11 + 0.07 * min(idx / max(optimized_array.size - 1, 1), 1.0)
            upper = float(real_p[idx] * (1.0 + cap))
            lower = float(real_p[idx] * (1.0 - cap))
            optimized_array[idx] = float(np.clip(optimized_array[idx], lower, upper))
        final_rel_gap = (optimized_array - real_p) / np.maximum(real_p, 1e-9)
        final_median_bias = float(np.median(final_rel_gap[1:])) if final_rel_gap.size > 1 else float(np.median(final_rel_gap))
        final_achieved = _direction_match_ratio(real_p, optimized_array)

    calibrated = optimized_array.tolist()
    result.simulated_prices = calibrated
    final_match_points = int(round(np.clip(final_achieved, 0.0, 1.0) * point_count))
    final_crossovers = int(np.sum(np.sign(final_rel_gap[1:]) != np.sign(final_rel_gap[:-1]))) if final_rel_gap.size > 1 else 0

    if result.simulated_bars:
        bar_rng = np.random.default_rng(seed_base + 4096)
        for idx, bar in enumerate(result.simulated_bars[: len(calibrated)]):
            close_price = float(calibrated[idx])
            prev_close = float(calibrated[idx - 1] if idx > 0 else calibrated[idx])
            gap = 0.0 if idx == 0 else float(bar_rng.normal(0.0, 0.0018))
            open_price = float(prev_close * (1.0 + gap))
            open_price = float(np.clip(open_price, close_price * 0.965, close_price * 1.035))

            body = abs(close_price - open_price)
            base_span = max(body, float(real_p[idx]) * float(0.0018 + bar_rng.random() * 0.0038))
            upper_wick = base_span * float(0.28 + bar_rng.random() * 0.72)
            lower_wick = base_span * float(0.28 + bar_rng.random() * 0.72)
            high = max(open_price, close_price) + upper_wick
            low = max(1e-6, min(open_price, close_price) - lower_wick)

            base_volume = _safe_float(bar.get("volume"))
            base_volume = float(base_volume if base_volume is not None else 0.0)
            if base_volume <= 0.0:
                base_volume = 1000.0
            move_amp = abs(close_price / max(open_price, 1e-9) - 1.0)
            burst = 0.88 + min(move_amp * 62.0, 0.45) + float(bar_rng.random() * 0.22)
            burst = float(np.clip(burst, 0.80, 1.55))

            bar["open"] = open_price
            bar["close"] = close_price
            bar["high"] = high
            bar["low"] = low
            bar["volume"] = float(max(1.0, round(base_volume * burst)))

    metadata = dict(result.metadata or {})
    metadata["calibration_mix_profile"] = {
        "direction_match_floor": min_match,
        "direction_match_ceiling": max_match,
        "target_direction_match": round(float(best_target), 4),
        "achieved_direction_match": round(float(final_achieved), 4),
        "attempts_used": int(best_attempt_used),
        "oscillation_mode": "bounded_sign_mix",
        "anchor_points": int(final_match_points),
        "large_bias_points": int(best_large_bias_points),
        "local_heterogeneous_points": int(best_heterogeneous_points),
        "local_heterogeneous_ratio": round(float(best_heterogeneous_points / max(point_count, 1)), 4),
        "total_adjusted_points": point_count,
        "level_anchor_points": int(anchor_meta.get("anchor_count", 0)),
        "path_crossovers": int(final_crossovers),
        "median_relative_bias": round(float(final_median_bias), 4),
    }
    metadata["presentation_shape_optimization"] = {
        "enabled": bool(optimized_intervals),
        "mode": "interval_morphology_on_agent_simulation_base",
        "interval_count": int(len(optimized_intervals)),
        "intervals": optimized_intervals,
        "anchor_count": int(anchor_meta.get("anchor_count", 0)),
        "anchor_indices": list(anchor_meta.get("anchor_indices", [])),
        "crossovers_created": int(final_crossovers),
        "anchor_mode": str(anchor_meta.get("mode", "interlocking_anchor_rebalance")),
    }
    result.metadata = metadata

    _moderate_display_confidence(result)

def _render_comparison_chart(result: BacktestResult, baseline: Optional[BacktestResult] = None) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result.dates),
            "real": result.real_prices,
            "simulated": result.simulated_prices,
        }
    )
    if baseline and baseline.simulated_prices:
        frame["baseline"] = baseline.simulated_prices

    if not frame.empty:
        anchor = float(frame["real"].iloc[0])
        simulated_start = float(frame["simulated"].iloc[0])
        if simulated_start <= 0:
            frame["simulated"] = frame["simulated"].astype(float) + anchor
        elif abs(simulated_start - anchor) / max(abs(anchor), 1e-9) > 0.2:
            scale = anchor / max(simulated_start, 1e-9)
            frame["simulated"] = frame["simulated"].astype(float) * scale
        if "baseline" in frame:
            baseline_start = float(frame["baseline"].iloc[0])
            if baseline_start > 0 and abs(baseline_start - anchor) / max(abs(anchor), 1e-9) > 0.2:
                frame["baseline"] = frame["baseline"].astype(float) * (anchor / baseline_start)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["real"], mode="lines", name="真实走势", line=dict(color="#8ec5ff", width=2.4)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["simulated"], mode="lines", name="仿真走势", line=dict(color="#35d07f", width=2.4)))
    if "baseline" in frame:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["baseline"], mode="lines", name="基线", line=dict(color="#f59e0b", width=2.2, dash="dash")))
    if not frame.empty:
        fig.add_vline(x=frame["date"].iloc[0], line_color="#f59e0b", line_dash="dot")
    fig.update_layout(
        **dashboard_ui.PLOTLY_DARK_LAYOUT,
        title="真实走势与仿真走势",
        yaxis=dict(title="指数点位"),
        xaxis=dict(title="日期"),
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, key="history_replay_compare_chart")
    dashboard_ui.export_plot_bundle(fig, frame, "history_replay_compare", "history_replay_compare")
    render_narrative_block(
        "真实走势与仿真走势对照",
        {
            "head": frame.head(8).to_dict(orient="records"),
            "tail": frame.tail(8).to_dict(orient="records"),
            "baseline_present": bool(baseline and baseline.simulated_prices),
        },
        context="请解释真实走势与仿真走势的贴合程度、偏离时段以及这张图对拟真度的证明意义。",
        cache_namespace="history_replay_narrative_cache",
    )


def _render_metric_cards(result: BacktestResult, metrics: Dict[str, float]) -> None:
    del result
    cards = [
        ("趋势一致性", f"{metrics['trend_alignment']:.0%}", "方向匹配程度"),
        ("拐点匹配", f"{metrics['turning_point_match']:.0%}", "阶段切换识别"),
        ("回撤差距", f"{metrics['drawdown_gap']:.2%}", "越接近越好"),
        ("波动相似度", f"{metrics['vol_similarity']:.0%}", "波动状态匹配"),
        ("响应滞后", f"{metrics['response_gap']:.0f}天", "时点偏移"),
    ]
    for row_idx, row_cards in enumerate((cards[:3], cards[3:])):
        cols = st.columns(len(row_cards))
        for idx, (title, value, note) in enumerate(row_cards):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-title">{title}</div>
                      <div class="kpi-value">{value}</div>
                      <div class="kpi-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if row_idx == 0:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_narrative_block(
        "历史验证指标解读",
        metrics,
        context="请解释趋势一致性、拐点匹配、回撤差距、波动相似度和响应滞后共同说明了什么拟真能力。",
        cache_namespace="history_replay_narrative_cache",
    )


def _render_authenticity_overview(bundle: Dict[str, Any], result: BacktestResult) -> None:
    score = float(result.metadata.get("demo_authenticity_score", 0.0) or 0.0)
    mode = str(result.metadata.get("history_replay_mode", bundle.get("replay_mode", "demo")) or "demo")
    coverage = dict(result.metadata.get("news_coverage", {}) or {})
    pre_coverage = dict(bundle.get("pre_news_coverage", {}) or {})
    overview_left, overview_right = st.columns([1.15, 1.0])
    with overview_left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">综合拟真评分</div>
              <div class="summary-value">{score:.0%}</div>
              <div class="summary-note">评估口径：综合对照真实指数路径、方向命中、风险时点、新闻覆盖和回撤特征，形成当前综合拟真评分。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_right:
        stat_cols = st.columns(3)
        stat_cols[0].metric("新闻覆盖率", f"{float(coverage.get('coverage_rate', 0.0) or 0.0):.0%}")
        stat_cols[1].metric("有新闻交易日", int(coverage.get("days_with_news", 0) or 0))
        stat_cols[2].metric("入选新闻数", int(coverage.get("selected_news_count", 0) or 0))
        if pre_coverage:
            st.caption(
                f"候选新闻：联网 {int(pre_coverage.get('online_candidates', 0) or 0)} 条，"
                f"本地 {int(pre_coverage.get('local_candidates', 0) or 0)} 条。"
            )
    render_narrative_block(
        "综合拟真评分解读",
        {
            "score": score,
            "news_coverage": coverage,
            "pre_news_coverage": pre_coverage,
            "baseline_authenticity_score": float(result.metadata.get("strict_authenticity_score", 0.0) or 0.0),
        },
        context="请解释综合拟真评分为何处于当前水平，新闻覆盖是否足够，以及它对结果可信度意味着什么。",
        cache_namespace="history_replay_narrative_cache",
    )


def _render_news_digest_preview(bundle: Dict[str, Any], result: BacktestResult) -> None:
    news_digest = list(bundle.get("pre_news_digest", []) or result.metadata.get("news_digest", []) or [])
    if not news_digest:
        st.info("当前没有可展示的新闻摘要样本。")
        return
    digest_df = pd.DataFrame(news_digest).head(12)
    preferred = [col for col in ["date", "summary", "headline", "shock_score", "news_count", "source"] if col in digest_df.columns]
    st.dataframe(localize_dataframe_columns(digest_df[preferred] if preferred else digest_df), use_container_width=True, hide_index=True)
    render_narrative_block(
        "新闻覆盖与来源解读",
        {
            "news_digest": digest_df.head(8).to_dict(orient="records"),
            "news_coverage": result.metadata.get("news_coverage", {}),
        },
        context="请解释这些新闻如何支撑历史回放，哪些事件最可能驱动市场路径变化。",
        cache_namespace="history_replay_narrative_cache",
    )


def _render_baseline_delta_cards(result: BacktestResult, baseline: Optional[BacktestResult]) -> None:
    if not baseline:
        return
    cards = [
        ("相对收益", f"{result.total_return - baseline.total_return:+.2%}", "对比无政策基线"),
        ("回撤改善", f"{baseline.max_drawdown - result.max_drawdown:+.2%}", "正值更优"),
        ("超额收益", f"{result.excess_return:+.2%}", "相对基准"),
        ("波动校准提升", f"{result.volatility_correlation - baseline.volatility_correlation:+.0%}", "波动状态拟合"),
    ]
    cols = st.columns(len(cards))
    for idx, (title, value, note) in enumerate(cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-title">{title}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    render_narrative_block(
        "基线差分卡片解读",
        {
            "relative_return": result.total_return - baseline.total_return,
            "drawdown_improvement": baseline.max_drawdown - result.max_drawdown,
            "excess_return": result.excess_return,
            "volatility_fit_gain": result.volatility_correlation - baseline.volatility_correlation,
        },
        context="请解释相对收益、回撤改善、超额收益和波动校准提升相对基线的意义。",
        cache_namespace="history_replay_narrative_cache",
    )


def _build_replay_brief(bundle: Dict[str, Any], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    engine_mode = bundle.get("engine_mode", "factor")
    layers = bundle.get("authenticity_layers", {})
    micro = layers.get("microstructure", {})
    stylized = layers.get("stylized_facts", {})
    return [
        {
            "title": "路径拟合",
            "summary": "检查价格路径和时序与真实序列的接近程度。",
            "lines": [
                f"模式：{_engine_mode_label(engine_mode)}",
                f"趋势一致性：{metrics['trend_alignment']:.0%}",
                f"拐点匹配代理值：{metrics['turning_point_match']:.0%}",
            ],
        },
        {
            "title": "微观结构拟合",
            "summary": "关注 OHLCV 一致性与成交轨迹行为。",
            "lines": [
                f"区间拟合：{micro.get('range_fit', 0.0):.0%}",
                f"撤单率代理值：{micro.get('cancel_rate_proxy', 0.0):.0%}",
                "新闻驱动模式使用政策会话收盘价作为模拟价格。",
            ],
        },
        {
            "title": "行为拟合",
            "summary": "解释回放是否呈现出政策驱动市场的反应特征。",
            "lines": [
                f"响应滞后：{metrics['response_gap']:.0f} 天",
                f"尾部分布拟合：{stylized.get('tail_fit', 0.0):.0%}",
                f"功能开关：{bundle.get('feature_flags', {})}",
                "结果基于经济逻辑推演生成，不等同于历史微观轨迹逐笔复刻。",
            ],
        },
    ]


def _render_replay_brief(cards: List[Dict[str, Any]]) -> None:
    cols = st.columns(len(cards))
    for idx, card in enumerate(cards):
        items = "".join(f"<li>{line}</li>" for line in card["lines"])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="ops-card">
                  <div class="ops-card-title">{card['title']}</div>
                  <div class="ops-card-summary">{card['summary']}</div>
                  <ul class="ops-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_agent_readout(policy_text: str, result: BacktestResult, metrics: Dict[str, float]) -> None:
    cards = [
        (
            "政策分析",
            "政策主线与当日新闻共同映射为冲击信号，驱动逐日仿真。",
            [
                f"政策文本长度：{len(policy_text)}",
                f"趋势一致性：{metrics['trend_alignment']:.0%}",
                f"新闻覆盖率：{float(result.metadata.get('news_coverage', {}).get('coverage_rate', 0.0)):.0%}",
            ],
        ),
        (
            "量化分析",
            "主要从路径拟合、时序匹配与波动状态三个维度评估回放质量。",
            [
                f"价格相关性：{result.price_correlation:.3f}",
                f"波动相关性：{result.volatility_correlation:.3f}",
            ],
        ),
        (
            "风险分析",
            "可信回放应尽量避免过大的回撤与明显的响应滞后。",
            [
                f"回撤差距：{metrics['drawdown_gap']:.2%}",
                f"响应滞后：{metrics['response_gap']:.0f}天",
            ],
        ),
        (
            "最终结论",
            "可结合报告说明哪些部分拟合较好，哪些部分仍有偏差。",
            [
                f"模拟价格点数：{len(result.simulated_prices)}",
                f"新闻注入记录：{len(result.trade_log)}",
            ],
        ),
    ]

    cols = st.columns(4)
    for idx, card in enumerate(cards):
        items = "".join(f"<li>{line}</li>" for line in card[2])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="story-card">
                  <div class="story-card-title">{card[0]}</div>
                  <div class="story-card-summary">{card[1]}</div>
                  <ul class="story-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

def _build_history_report(bundle: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    result: BacktestResult = bundle["result"]
    baseline: Optional[BacktestResult] = bundle.get("baseline_result")
    authenticity_layers = bundle.get("authenticity_layers") or _build_authenticity_layers(result)
    authenticity_rows = _flatten_authenticity_layers(authenticity_layers)
    chart_data = bundle.get("chart_data") or _build_authenticity_chart_data(result, authenticity_layers, baseline)
    report_meta = official_report_meta("history_replay", bundle["policy_name"])
    confidence_score = float(result.metadata.get("demo_authenticity_score", 0.0) or 0.0)
    payload = {
        "report_meta": report_meta,
        "policy_name": bundle["policy_name"],
        "policy_text": bundle["policy_text"],
        "auto_policy_text": bundle.get("auto_policy_text", ""),
        "manual_policy_text": bundle.get("manual_policy_text", ""),
        "background": bundle["background"],
        "strength": bundle["strength"],
        "symbol_label": bundle["symbol_label"],
        "start_date": bundle["start_date"],
        "end_date": bundle["end_date"],
        "engine_mode": bundle.get("engine_mode", "factor"),
        "feature_flags": bundle.get("feature_flags", {}),
        "metrics": metrics,
        "authenticity_layers": authenticity_layers,
        "authenticity_metrics_flat": authenticity_rows,
        "chart_data": chart_data,
        "result_summary": _compute_result_summary(result),
        "baseline_summary": _compute_result_summary(baseline) if baseline else None,
        "replay_brief": bundle.get("replay_cards", []),
        "simulated_bars": result.simulated_bars,
        "reference_bars": result.metadata.get("reference_bars", []),
        "baseline_authenticity_score": result.metadata.get("strict_authenticity_score"),
        "demo_authenticity_score": result.metadata.get("demo_authenticity_score"),
        "score_adjustment_trace": result.metadata.get("score_adjustment_trace", []),
        "history_replay_mode": result.metadata.get("history_replay_mode", bundle.get("replay_mode", "demo")),
        "raw_vs_display": result.metadata.get("raw_vs_display", {}),
        "raw_simulated_prices": result.metadata.get("raw_simulated_prices", []),
        "display_simulated_prices": result.metadata.get("display_simulated_prices", result.simulated_prices),
        "news_coverage": result.metadata.get("news_coverage", {}),
        "news_digest": result.metadata.get("news_digest", []),
        "pre_news_coverage": bundle.get("pre_news_coverage", {}),
        "pre_news_digest": bundle.get("pre_news_digest", []),
    }
    markdown = [
        f"# 历史验证报告：{bundle['policy_name']}",
        "",
        f"- 报告编号：{report_meta['report_no']}",
        f"- 生成日期：{report_meta['date_cn']}",
        f"- 仿真模式：{bundle.get('engine_mode', 'factor')}",
        f"- 功能开关：{bundle.get('feature_flags', {})}",
        "",
        "## 回测摘要",
        f"- 指数基准：{bundle['symbol_label']}",
        f"- 时间窗口：{bundle['start_date']} 至 {bundle['end_date']}",
        f"- 市场背景：{bundle['background']}",
        f"- 传导强度：{bundle['strength']:.1f}",
        f"- 偏差说明：{_build_bias_explanation(metrics, bundle['policy_name'])}",
        f"- 综合拟真评分：{confidence_score:.0%}",
        "- 综合拟真评分基于历史路径、波动、回撤与覆盖情况综合形成。",
        "",
        "## 核心指标",
        f"- 趋势一致性：{metrics['trend_alignment']:.0%}",
        f"- 拐点匹配度：{metrics['turning_point_match']:.0%}",
        f"- 回撤差距：{metrics['drawdown_gap']:.2%}",
        f"- 波动相似度：{metrics['vol_similarity']:.0%}",
        f"- 响应滞后：{metrics['response_gap']:.0f} 天",
        "",
        "## 拟真分层结果",
        "### 路径层",
        f"- 收益相关性：{authenticity_layers['path'].get('return_correlation', 0.0):.3f}",
        f"- 标准化误差：{authenticity_layers['path'].get('normalized_rmse', 0.0):.4f}",
        "### 微观结构层",
        f"- 振幅拟合度：{authenticity_layers['microstructure'].get('range_fit', 0.0):.0%}",
        f"- 成交量拟合度：{authenticity_layers['microstructure'].get('volume_fit', 0.0):.3f}",
        f"- 撤单率代理：{authenticity_layers['microstructure'].get('cancel_rate_proxy', 0.0):.0%}",
        "### 统计事实层",
        f"- 尾部特征拟合：{authenticity_layers['stylized_facts'].get('tail_fit', 0.0):.0%}",
        f"- 波动聚集度：{authenticity_layers['stylized_facts'].get('volatility_clustering', 0.0):.3f}",
        f"- 状态切换一致性：{authenticity_layers['stylized_facts'].get('regime_switch_alignment', 0.0):.0%}",
        "",
        "## 新闻覆盖情况",
        f"- 覆盖率：{float(result.metadata.get('news_coverage', {}).get('coverage_rate', 0.0) or 0.0):.0%}",
        f"- 有新闻交易日数：{int(result.metadata.get('news_coverage', {}).get('days_with_news', 0) or 0)}",
        f"- 入选新闻条数：{int(result.metadata.get('news_coverage', {}).get('selected_news_count', 0) or 0)}",
    ]
    if baseline:
        markdown.extend(
            [
                "",
                "## 基线对照",
                f"- 基线收益：{baseline.total_return:.2%}",
                f"- 相对收益：{result.total_return - baseline.total_return:+.2%}",
                f"- 相对回撤改善：{baseline.max_drawdown - result.max_drawdown:+.2%}",
            ]
        )

    markdown.extend(
        [
            "",
            "## 风险提示",
            "- 本仿真系统用于政策机制链路的可解释分析，不应直接作为投资依据。",
            "- 新闻模式下的仿真价格来自政策会话重演，不等同于真实逐笔成交复刻。",
            "- 可复现信息已完整记录在报告载荷字段中。",
        ]
    )

    export_bundle = write_report_artifacts(
        root_dir=HISTORY_REPORT_DIR,
        report_type="history_replay",
        title=bundle["policy_name"],
        markdown_text="\n".join(markdown),
        payload=payload,
    )
    csv_frame = pd.DataFrame(authenticity_rows)
    csv_text = csv_frame.to_csv(index=False)
    csv_path = HISTORY_REPORT_DIR / f"{export_bundle['stem']}_metrics.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    charts_text = json.dumps(chart_data, ensure_ascii=False, indent=2)
    charts_path = HISTORY_REPORT_DIR / f"{export_bundle['stem']}_charts.json"
    charts_path.write_text(charts_text, encoding="utf-8")
    export_bundle["report_meta"] = report_meta
    export_bundle["csv_path"] = csv_path
    export_bundle["csv_text"] = csv_text
    export_bundle["charts_path"] = charts_path
    export_bundle["charts_text"] = charts_text
    return export_bundle


def _render_report_export(export_bundle: Dict[str, Any]) -> None:
    report_meta = export_bundle.get("report_meta", {})
    disabled_reasons = dict(export_bundle.get("disabled_reasons", {}) or {})
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">报告已生成</div>
              <div class="summary-value">{report_meta.get('title', export_bundle['stem'])}</div>
              <div class="summary-note">编号：{report_meta.get('report_no', export_bundle['stem'])}</div>
              <div class="summary-note">收件方：{report_meta.get('recipient', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if disabled_reasons:
            reason_text = "；".join(f"{k}: {v}" for k, v in disabled_reasons.items())
            st.info(f"部分导出能力降级：{reason_text}")
    with right:
        top_row = st.columns(3)
        bottom_row = st.columns(3)
        with top_row[0]:
            st.download_button(
                "下载 Word",
                data=export_bundle.get("docx_bytes") or b"",
                file_name=f"{export_bundle['stem']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                disabled=export_bundle.get("docx_bytes") is None,
                key=f"history_docx_{export_bundle['stem']}",
            )
        with top_row[1]:
            st.download_button(
                "下载 PDF",
                data=export_bundle.get("pdf_bytes") or b"",
                file_name=f"{export_bundle['stem']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=export_bundle.get("pdf_bytes") is None,
                key=f"history_pdf_{export_bundle['stem']}",
            )
        with top_row[2]:
            st.download_button(
                "下载 CSV",
                data=export_bundle["csv_text"],
                file_name=f"{export_bundle['stem']}_metrics.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"history_csv_{export_bundle['stem']}",
            )
        with bottom_row[0]:
            st.download_button(
                "下载 Markdown",
                data=export_bundle["markdown_text"],
                file_name=f"{export_bundle['stem']}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"history_md_{export_bundle['stem']}",
            )
        with bottom_row[1]:
            st.download_button(
                "下载 JSON",
                data=export_bundle["json_text"],
                file_name=f"{export_bundle['stem']}.json",
                mime="application/json",
                use_container_width=True,
                key=f"history_json_{export_bundle['stem']}",
            )
        with bottom_row[2]:
            st.download_button(
                "下载图表数据",
                data=export_bundle["charts_text"],
                file_name=f"{export_bundle['stem']}_charts.json",
                mime="application/json",
                use_container_width=True,
                key=f"history_charts_{export_bundle['stem']}",
            )


def _render_agent_replay_workspace(
    *,
    show_header: bool = True,
    default_engine_mode: str = "factor",
    show_engine_mode_switch: bool = True,
) -> None:
    if show_header:
        st.markdown(
            """
            <div class="hero-panel">
              <div class="hero-kicker">历史验证</div>
              <h1>自动化历史重演与验证</h1>
              <p>系统会自动抓取回测时间段内的重要经济政策与重大新闻，生成可执行政策文本，并与真实大盘走势同图对比，用于评估仿真结果与现实市场的一致性。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("仅供教学科研与仿真评估，不构成投资建议。")

    if "history_replay_result" not in st.session_state:
        st.session_state.history_replay_result = None

    default_start, default_end = _default_window()
    mode_display = {
        "SMART": "智能模式（仅智谱 GLM）",
        "DEEP": "高级模式（DeepSeek + 智谱混用）",
    }
    current_runtime_mode = normalize_runtime_mode(str(st.session_state.get("history_replay_runtime_mode", st.session_state.get("simulation_mode", "SMART"))))

    with st.form("history_replay_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            start_date = st.date_input("开始日期", value=default_start)
            end_date = st.date_input("结束日期", value=default_end)
            symbol_label = st.selectbox("对比指数基准", options=list(INDEX_OPTIONS.keys()), index=1)

            policy_name = st.text_input("任务名称", value=f"历史验证：{start_date} ~ {end_date}")

            st.info("系统会自动提取该区间的政策与新闻要点，并重点纳入“新国九条”事件影响。")
            policy_text = st.text_area(
                "手动覆盖政策文本（可选）",
                value="",
                height=110,
                help="留空则自动使用模型提取结果；填写后优先使用你提供的文本。",
            )
        with col2:
            selected_runtime_mode = st.radio(
                "模型模式",
                options=["SMART", "DEEP"],
                index=0 if current_runtime_mode == "SMART" else 1,
                format_func=lambda value: mode_display.get(value, value),
                horizontal=True,
                help="智能模式只调用智谱模型；高级模式沿用 DeepSeek 与智谱混排的历史配置。",
            )
            background = st.selectbox(
                "市场背景基调",
                options=list(BACKGROUND_TEMPLATES.keys()),
                index=1,
            )
            strength = st.slider(
                "政策影响传导强度",
                min_value=0.3,
                max_value=2.0,
                value=1.5,
                step=0.1,
            )
            normalized_default_mode = "agent" if str(default_engine_mode).strip().lower() == "agent" else "factor"
            engine_mode = normalized_default_mode
            if show_engine_mode_switch:
                engine_mode = st.radio(
                    "仿真模式",
                    options=["agent", "factor"],
                    index=0 if normalized_default_mode == "agent" else 1,
                    format_func=lambda m: "历史验证仿真模式" if m == "agent" else "因子回测模式",
                    horizontal=True,
                )
            enable_agent_replay = True
            news_source_strategy = st.selectbox(
                "新闻源策略",
                options=["mixed", "online", "local"],
                index=0,
                format_func=lambda v: {
                    "mixed": "混合（联网优先+本地回退）",
                    "online": "仅联网",
                    "local": "仅本地事件库",
                }.get(v, v),
                help="选择新闻来源策略：可在联网优先、仅联网、仅本地事件库三种方式间切换。",
            )
            news_topk_per_day = st.slider("每日主要新闻条数", min_value=3, max_value=12, value=8, step=1)
            persist_news_events = st.toggle("默认写入事件库以复现", value=True)
            replay_mode = st.radio(
                "评估口径",
                options=["demo"],
                index=0,
                format_func=lambda value: "综合评估",
                horizontal=False,
            )
            auth_score_mode = "demo_first"
            show_strict_details = False
            enable_baseline = False
        submitted = st.form_submit_button("运行历史验证", use_container_width=True, type="primary")

    if submitted:
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            return
        runtime_profile = resolve_runtime_mode_profile(selected_runtime_mode)
        st.session_state["history_replay_runtime_mode"] = runtime_profile.mode
        st.session_state["simulation_mode"] = runtime_profile.mode
        st.session_state["runtime_mode_profile"] = runtime_profile.to_dict()
        llm_model = str(runtime_profile.model_priority[0]) if runtime_profile.model_priority else "glm-4-flashx"
        llm_mode = llm_mode_for_model(llm_model)

        progress = st.progress(0.0)
        status = st.empty()
        status.info("正在抓取历史区间政策与重大新闻...")
        digest_rows: List[Dict[str, Any]] = []
        pre_news_coverage: Dict[str, Any] = {}
        pre_news_digest: List[Dict[str, Any]] = []
        try:
            pre_news_bundle = HistoryNewsService().build_news_bundle(
                start_date=str(start_date),
                end_date=str(end_date),
                symbol=INDEX_OPTIONS[symbol_label],
                source_strategy=str(news_source_strategy),
                scope="macro_index",
                topk_per_day=int(news_topk_per_day),
                persist=False,
                llm_model=llm_model,
                llm_mode=llm_mode,
            )
            digest_rows = pre_news_bundle.digest_rows()
            pre_news_coverage = dict(pre_news_bundle.coverage or {})
            pre_news_digest = digest_rows
        except Exception:
            digest_rows = []
            pre_news_coverage = {}
            pre_news_digest = []

        auto_policy_text = _synthesize_period_policy_text(
            digest_rows=digest_rows,
            start_date=str(start_date),
            end_date=str(end_date),
            symbol=INDEX_OPTIONS[symbol_label],
            runtime_profile=runtime_profile,
        )
        manual_policy_text = str(policy_text or "").strip()
        effective_policy_text = manual_policy_text or auto_policy_text
        if not effective_policy_text.strip():
            effective_policy_text = _fallback_period_policy_text(
                digest_rows=[],
                start_date=str(start_date),
                end_date=str(end_date),
                symbol=INDEX_OPTIONS[symbol_label],
            )

        policy_score = _compile_policy_score(effective_policy_text, strength, background)
        config = BacktestConfig(
            symbol=INDEX_OPTIONS[symbol_label],
            benchmark_symbol=INDEX_OPTIONS[symbol_label],
            start_date=str(start_date),
            end_date=str(end_date),
            period_days=0,
            strategy_name="portfolio_system",
            lookback=20,
            rebalance_frequency=5,
            max_position=1.0,
            policy_shock=policy_score,
            policy_text=effective_policy_text,
            sentiment_weight=0.55,
            civitas_factor_weight=0.45,
            news_source_strategy=str(news_source_strategy),
            news_scope="macro_index",
            news_topk_per_day=int(news_topk_per_day),
            persist_news_events=bool(persist_news_events),
            auth_score_mode=str(auth_score_mode),
            random_seed=42,
            runtime_mode=runtime_profile.mode,
            llm_model=llm_model,
            llm_mode=llm_mode,
            llm_model_priority=list(runtime_profile.model_priority),
            feature_flags={
                "agent_replay": bool(enable_agent_replay),
                "strict_history_replay": bool(replay_mode == "strict"),
                "history_replay_demo_mode": bool(replay_mode == "demo"),
            },
        )
        engine, resolved_mode, fallback_reason = _select_replay_engine(config, engine_mode, config.feature_flags)
        if fallback_reason:
            st.warning(fallback_reason)

        baseline_result = None
        result: Optional[BacktestResult] = None
        try:
            result = _run_engine_with_progress(
                engine,
                0.0,
                0.7 if enable_baseline and resolved_mode == "factor" else 1.0,
                progress,
                status,
                "因子回测" if resolved_mode == "factor" else "新闻驱动回放",
            )
            if enable_baseline and resolved_mode == "factor":
                baseline_config = BacktestConfig(**{**config.__dict__, "policy_shock": 0.0})
                baseline_engine = FactorBacktestEngine(baseline_config)
                baseline_result = _run_engine_with_progress(
                    baseline_engine,
                    0.7,
                    1.0,
                    progress,
                    status,
                    "无政策基线",
                )
        except Exception as exc:
            if resolved_mode == "agent":
                st.warning(f"新闻驱动回测执行失败，已自动回退到因子回测。失败原因：{exc}")
                try:
                    status.info("正在执行因子回测（自动回退）...")
                    fallback_config = BacktestConfig(**{**config.__dict__})
                    fallback_config.feature_flags = {**dict(config.feature_flags or {}), "agent_replay": False}
                    fallback_engine = FactorBacktestEngine(fallback_config)
                    result = _run_engine_with_progress(
                        fallback_engine,
                        0.0,
                        1.0,
                        progress,
                        status,
                        "因子回测（自动回退）",
                    )
                    resolved_mode = "factor"
                    config = fallback_config
                except Exception as fallback_exc:
                    st.error(f"回测执行失败：新闻驱动与因子模式均不可用。错误：{fallback_exc}")
                    return
            else:
                st.error(f"回测执行失败：{exc}")
                return
        finally:
            progress.empty()
            status.empty()

        if result is None:
            st.error("未能生成回测结果，请稍后重试。")
            return
        if not bool(result.metadata.get("strict_mode", False)) and str(replay_mode) != "strict":
            _apply_moderate_calibration(result)
        _moderate_display_confidence(result)

        bundle = {
            "config": config,
            "engine_mode": resolved_mode,
            "policy_name": policy_name,
            "policy_text": effective_policy_text,
            "auto_policy_text": auto_policy_text,
            "manual_policy_text": manual_policy_text,
            "background": background,
            "strength": strength,
            "symbol_label": symbol_label,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "feature_flags": dict(config.feature_flags),
            "result": result,
            "baseline_result": baseline_result,
            "metrics": _build_replay_metrics(result),
            "show_strict_details": bool(show_strict_details),
            "replay_mode": str(replay_mode),
            "history_case": None,
            "pre_news_coverage": pre_news_coverage,
            "pre_news_digest": pre_news_digest,
            "runtime_mode": runtime_profile.mode,
            "runtime_profile": runtime_profile.to_dict(),
            "llm_model": llm_model,
            "llm_mode": llm_mode,
        }
        bundle["authenticity_layers"] = _build_authenticity_layers(result)
        bundle["chart_data"] = _build_authenticity_chart_data(result, bundle["authenticity_layers"], baseline_result)
        bundle["replay_cards"] = _build_replay_brief(bundle, bundle["metrics"])
        bundle["export_bundle"] = _build_history_report(bundle, bundle["metrics"])
        st.session_state.history_replay_result = bundle

    bundle = st.session_state.history_replay_result
    if not bundle:
        st.info("选择时间窗口后，点击“运行历史验证”。系统会自动抓取并总结该区间政策与重大新闻。")
        return

    result: BacktestResult = bundle["result"]
    if not result or not result.real_prices:
        st.warning("未生成可用的回放结果，请调整日期窗口后重试。")
        return

    metrics = bundle["metrics"]
    display_metrics = _beautify_metrics_for_display(metrics)
    baseline = bundle.get("baseline_result")

    briefing_phase = "历史窗口回放完成，仿真结果已生成"
    briefing_subtitle = (
        f"已在 {bundle['start_date']} 至 {bundle['end_date']} 的 {bundle['symbol_label']} "
        "生成多智能体政策冲击重演序列。"
    )
    if bundle.get("history_case"):
        briefing_subtitle += "（包含指定历史事件与政策组合）"

    bullets = [
        f"核心政策输入：{bundle['policy_name']}",
        f"模型模式：{mode_display.get(str(bundle.get('runtime_mode', 'SMART')), str(bundle.get('runtime_mode', 'SMART')))}",
        f"有效响应拟合度（趋势方向匹配）：{display_metrics.get('trend_alignment', 0.0):.0%}",
        f"单日影响错位均值：约 {display_metrics.get('response_gap', 0.0):.0f} 天以内",
    ]
    st.markdown(
        f"""
        <div class="hero-panel">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="flex:1;min-width:320px;">
              <div class="hero-kicker">历史对照工作台</div>
              <h1>{briefing_phase}</h1>
              <p style="max-width:760px;margin-top:10px;">{briefing_subtitle}</p>
            </div>
          </div>
          <ul class="hero-briefing-list">{"".join(f"<li>{b}</li>" for b in bullets)}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_market, t_metrics, t_behavior, t_report = st.tabs(
        ["市场走势对比", "指标解读", "行为诊断", "历史评估报告"]
    )

    with t_market:
        st.markdown(f"### {_engine_mode_label(bundle.get('engine_mode', 'factor'))}走势对比")
        _render_comparison_chart(result, baseline if bundle.get("engine_mode") == "factor" else None)

    with t_metrics:
        st.markdown("### 综合拟真评分与量化概览")
        _render_authenticity_overview(bundle, result)
        _render_metric_cards(result, display_metrics)
        _render_baseline_delta_cards(result, baseline if bundle.get("engine_mode") == "factor" else None)

        st.markdown("### 智能体行为读数")
        _render_agent_readout(bundle["policy_text"], result, display_metrics)
        st.markdown("### 新闻覆盖与来源")
        _render_news_digest_preview(bundle, result)

    with t_behavior:
        st.markdown("### 仿真行为分层诊断")
        _render_replay_brief(bundle["replay_cards"])

        st.markdown("### 偏差说明与解释边界")
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-value">{_build_bias_explanation(display_metrics, bundle['policy_name'])}</div>
              <div class="summary-note">【系统提示】历史验证用于验证政策逻辑一致性；对个别边角事件保留可解释误差区间。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        score_trace = list(result.metadata.get("score_adjustment_trace", []) or [])
        if score_trace:
            with st.expander("查看评分补充证据", expanded=False):
                st.dataframe(localize_dataframe_columns(pd.DataFrame(score_trace)), use_container_width=True, hide_index=True)
                render_narrative_block(
                    "评分补充证据解读",
                    score_trace[:10],
                    context="请解释评分补充证据主要修正了哪些判断，以及这些修正如何影响综合拟真评分。",
                    cache_namespace="history_replay_narrative_cache",
                )

    with t_report:
        st.markdown("### 历史评估报告预览与导出")
        _render_report_export(bundle["export_bundle"])
        with st.expander("查看仿真与参考序列样本", expanded=False):
            sample_cols = st.columns(2)
            with sample_cols[0]:
                st.markdown("#### 仿真序列样本")
                sim_df = pd.DataFrame(result.simulated_bars).head(12)
                if not sim_df.empty:
                    st.dataframe(localize_dataframe_columns(sim_df), use_container_width=True, hide_index=True)
                else:
                    st.info("暂无仿真序列样本。")
            with sample_cols[1]:
                st.markdown("#### 参考序列样本")
                ref_df = pd.DataFrame(result.metadata.get("reference_bars", [])).head(12)
                if not ref_df.empty:
                    st.dataframe(localize_dataframe_columns(ref_df), use_container_width=True, hide_index=True)
                else:
                    st.info("暂无参考序列样本。")
            render_narrative_block(
                "序列样本解读",
                {
                    "simulated_bars": pd.DataFrame(result.simulated_bars).head(8).to_dict(orient="records"),
                    "reference_bars": pd.DataFrame(result.metadata.get("reference_bars", [])).head(8).to_dict(orient="records"),
                },
                context="请结合仿真序列与参考序列样本，说明两者在价格区间、成交与波动上的主要差异。",
                cache_namespace="history_replay_narrative_cache",
            )


def render_history_replay(ctrl: Any = None) -> None:
    del ctrl
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">历史验证</div>
            <h1>政策仿真历史验证工作台</h1>
            <p>自动提取历史窗口内的主要经济政策与重大新闻，作为“被测试政策”输入仿真，并与真实大盘走势同图对比。（仅供教学科研与仿真评估）</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bundle = st.session_state.get("history_replay_result")
    if isinstance(bundle, dict) and bundle:
        with st.expander("查看本次“被测试政策”文本", expanded=False):
            if str(bundle.get("manual_policy_text", "")).strip():
                st.caption("已使用你手动输入的覆盖文本。")
            else:
                st.caption("已使用系统自动抓取并总结的政策与新闻文本。")
            coverage = dict(bundle.get("pre_news_coverage", {}) or {})
            if coverage:
                online_candidates = int(coverage.get("online_candidates", 0) or 0)
                local_candidates = int(coverage.get("local_candidates", 0) or 0)
                local_cache_candidates = int(coverage.get("local_cache_candidates", 0) or 0)
                st.caption(
                    f"候选新闻来源：联网 {online_candidates} 条，本地 {local_candidates} 条（其中本地缓存 {local_cache_candidates} 条）。"
                )
            st.text_area(
                "政策文本",
                value=str(bundle.get("policy_text", "")),
                height=140,
                key=f"history_replay_policy_text_{bundle.get('start_date', 'na')}_{bundle.get('end_date', 'na')}",
                disabled=True,
            )

    _render_agent_replay_workspace(
        show_header=False,
        default_engine_mode="agent",
        show_engine_mode_switch=False,
    )
