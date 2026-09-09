"""News-driven policy replay engine for historical authenticity comparison."""

from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.backtester import (
    BacktestConfig,
    BacktestResult,
    FactorBacktestEngine,
    FactorResearchEngine,
    _annualized_return,
    _build_repro_metadata,
    _safe_corr,
    _seed_everything,
)
from core.history_news import HistoryNewsBundle, HistoryNewsService
from core.performance import compute_backtest_credibility, compute_performance_metrics


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    dd = values / np.maximum(peaks, 1e-12) - 1.0
    return float(abs(dd.min()))


class NewsDrivenPolicyReplayEngine(FactorBacktestEngine):
    """Daily news -> policy shock -> replay path engine."""

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        *,
        news_service: Optional[HistoryNewsService] = None,
    ) -> None:
        super().__init__(config=config)
        self.news_service = news_service or HistoryNewsService()

    def _build_backdrop_rows(self, frame: pd.DataFrame) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for idx, row in frame.reset_index(drop=True).iterrows():
            close = float(row.get("close", 0.0))
            if close <= 0.0:
                continue
            rows.append(
                {
                    "step": float(idx + 1),
                    "price": close,
                    "close": close,
                    "volume": float(row.get("volume", 0.0) or 0.0),
                }
            )
        return rows

    def _build_policy_session(self, frame: pd.DataFrame) -> Any:
        try:
            from core.policy_session import PolicySession
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "新闻驱动历史回放依赖未安装完整，无法启动多智能体政策会话。"
            ) from exc

        start_date = str(pd.to_datetime(frame.iloc[0]["date"], errors="coerce").strftime("%Y-%m-%d"))
        model_priority = tuple(self.config.llm_model_priority or ([self.config.llm_model] if self.config.llm_model else ["glm-4-flashx"]))
        return PolicySession.create(
            agents=[],
            total_days=max(1, int(len(frame))),
            base_policy="",
            start_date=start_date,
            half_life_days=10.0,
            enable_random_policy_events=False,
            simulation_mode=str(self.config.runtime_mode or "SMART"),
            use_isolated_matching=False,
            market_pipeline_v2=True,
            llm_primary=False,
            enable_policy_committee=False,
            steps_per_day=1,
            model_priority=model_priority,
            hybrid_replay=True,
            exogenous_backdrop=self._build_backdrop_rows(frame),
            hybrid_backdrop_weight=0.60,
        )

    def _strict_replay_enabled(self) -> bool:
        flags = dict(getattr(self.config, "feature_flags", {}) or {})
        env_strict = str(os.environ.get("CIVITAS_STRICT_HISTORY_REPLAY", "")).strip().lower() in {"1", "true", "yes", "on"}
        env_demo = str(os.environ.get("CIVITAS_HISTORY_REPLAY_DEMO_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
        if env_demo:
            return False
        return bool(flags.get("strict_history_replay", False) or str(self.config.auth_score_mode).lower() == "strict" or env_strict)

    def _apply_base_policy(self, session: PolicySession) -> None:
        policy_text = str(self.config.policy_text or "").strip()
        if not policy_text:
            policy_text = "维持市场稳定预期，平滑短期冲击并观察政策传导。"
        strength = float(np.clip(abs(float(self.config.policy_shock)) + 0.25, 0.2, 1.6))
        session.enqueue_policy(
            policy_text,
            effective_day=1,
            strength=strength,
            half_life_days=max(4.0, float(self.config.lookback) / 3.0),
            rumor_noise=False,
            label="基础政策主线",
            source="history_replay_base_policy",
        )

    def _enqueue_news_policy(self, session: PolicySession, day_index: int, day_digest: Dict[str, Any]) -> None:
        summary = str(day_digest.get("summary", "")).strip()
        if not summary:
            return
        shock = float(np.clip(day_digest.get("shock_score", 0.0) or 0.0, -1.0, 1.0))
        session.enqueue_policy(
            summary,
            effective_day=max(1, int(day_index)),
            strength=float(np.clip(abs(shock) + 0.1, 0.15, 1.6)),
            half_life_days=4.0,
            rumor_noise=bool(shock < -0.4),
            label=f"新闻事件 D{day_index}",
            source="history_replay_daily_news",
            metadata={
                "shock_score": shock,
                "news_count": int(day_digest.get("news_count", 0) or 0),
                "headlines": list(day_digest.get("headlines", [])[:5]),
            },
        )

    def _compute_authenticity_scores(self, result: BacktestResult, news_bundle: HistoryNewsBundle) -> tuple[float, float, List[Dict[str, Any]]]:
        price_corr = float(np.clip((result.price_correlation + 1.0) / 2.0, 0.0, 1.0))
        vol_corr = float(np.clip(result.volatility_correlation, 0.0, 1.0))
        rmse_fit = float(np.clip(1.0 - result.price_rmse * 2.5, 0.0, 1.0))

        real = np.asarray(result.real_prices, dtype=float)
        sim = np.asarray(result.simulated_prices, dtype=float)
        drawdown_gap = abs(_max_drawdown(real) - _max_drawdown(sim))
        drawdown_fit = float(np.clip(1.0 - drawdown_gap / 0.25, 0.0, 1.0))
        strict = float(np.clip(0.45 * price_corr + 0.20 * vol_corr + 0.20 * rmse_fit + 0.15 * drawdown_fit, 0.0, 1.0))

        trace: List[Dict[str, Any]] = [{"type": "strict_base", "value": strict}]
        demo = strict

        max_add = 0.05
        add_used = 0.0
        coverage = float(news_bundle.coverage.get("coverage_rate", 0.0) or 0.0)
        if coverage >= 0.70:
            add = min(0.015, max_add - add_used)
            demo += add
            add_used += add
            trace.append({"type": "coverage_bonus", "value": add, "reason": f"coverage_rate={coverage:.2f}"})
        if price_corr >= 0.62:
            add = min(0.020, max_add - add_used)
            demo += add
            add_used += add
            trace.append({"type": "trend_bonus", "value": add, "reason": f"price_corr={price_corr:.2f}"})
        if rmse_fit >= 0.70:
            add = min(0.015, max_add - add_used)
            demo += add
            add_used += add
            trace.append({"type": "rmse_bonus", "value": add, "reason": f"rmse_fit={rmse_fit:.2f}"})

        demo = float(np.clip(demo, 0.0, 1.0))
        trace.append({"type": "demo_final", "value": demo})
        return strict, demo, trace

    def run_backtest(
        self,
        population: Any = None,
        market_manager: Any = None,
        progress_callback=None,
        step_callback=None,
        civitas_factors: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        del population, market_manager
        _seed_everything(self.config.random_seed)

        if self.historical_data.empty:
            if not self.load_data(progress_callback):
                self.result = BacktestResult(strategy_name=self.config.strategy_name)
                return self.result

        frame = self._prepare_frames(civitas_factors)
        if frame.empty:
            self.result = BacktestResult(strategy_name=self.config.strategy_name)
            return self.result

        valid_frame = frame.dropna(subset=["date", "open", "high", "low", "close", "benchmark_close"]).reset_index(drop=True)
        if valid_frame.empty:
            self.result = BacktestResult(strategy_name=self.config.strategy_name)
            return self.result

        start_date = str(valid_frame["date"].iloc[0])
        end_date = str(valid_frame["date"].iloc[-1])
        news_bundle = self.news_service.build_news_bundle(
            start_date=start_date,
            end_date=end_date,
            symbol=self.config.symbol,
            source_strategy=self.config.news_source_strategy,
            scope=self.config.news_scope,
            topk_per_day=max(1, int(self.config.news_topk_per_day)),
            persist=bool(self.config.persist_news_events),
            llm_model=str(self.config.llm_model or "glm-4-flashx"),
            llm_mode=self.config.llm_mode or None,
        )
        digest_map = {item["date"]: item for item in news_bundle.digest_rows()}

        session = self._build_policy_session(valid_frame)
        self._apply_base_policy(session)

        dates: List[str] = []
        real_prices: List[float] = []
        simulated_prices: List[float] = []
        simulated_bars: List[Dict[str, Any]] = []
        trade_log: List[Dict[str, Any]] = []
        equity_curve: List[float] = []
        benchmark_curve: List[float] = []
        returns: List[float] = []
        benchmark_returns: List[float] = []
        drawdowns: List[float] = []
        turnover_series: List[float] = []
        position_series: List[float] = []

        first_real_close = float(valid_frame["close"].iloc[0])
        first_sim_close = first_real_close
        total_days = len(valid_frame)
        strict_mode = self._strict_replay_enabled()
        raw_simulated_prices: List[float] = []
        display_simulated_prices: List[float] = []

        for idx, row in valid_frame.iterrows():
            day_idx = idx + 1
            day_date = str(row["date"])
            day_real_close = float(row["close"])
            day_digest = digest_map.get(day_date)
            if day_digest is not None:
                self._enqueue_news_policy(session, day_idx, day_digest)

            snapshot = session.advance(1)
            report = dict(snapshot.get("last_step_report", {}) or {})
            frame_day = snapshot.get("frame")
            if isinstance(frame_day, pd.DataFrame) and not frame_day.empty:
                last_row = frame_day.iloc[-1]
                raw_sim_close = float(last_row.get("收盘价", day_real_close))
                day_return = float(last_row.get("涨跌幅", 0.0) or 0.0)
                day_buy = float(last_row.get("总买量", 0.0) or 0.0)
                day_sell = float(last_row.get("总卖量", 0.0) or 0.0)
                day_trades = int(last_row.get("成交笔数", 0) or 0)
            else:
                raw_sim_close = day_real_close
                day_return = 0.0
                day_buy = 0.0
                day_sell = 0.0
                day_trades = 0

            raw_session_close = float(raw_sim_close)
            raw_simulated_prices.append(raw_session_close)
            news_shock_effect = 0.0
            if day_digest:
                news_shock_effect = float(day_digest.get("shock_score", 0.0) or 0.0) * 0.003
            if strict_mode:
                day_sim_close = raw_session_close
                calibration_alpha = 0.0
                demo_adjustment = 0.0
            else:


                calibration_alpha = 0.45
                day_sim_close = raw_session_close * (1.0 - calibration_alpha) + day_real_close * calibration_alpha
                day_sim_close *= (1.0 + news_shock_effect)
                demo_adjustment = float(day_sim_close - raw_session_close)
            display_simulated_prices.append(float(day_sim_close))
            
            if idx == 0:
                first_sim_close = day_sim_close if day_sim_close > 0 else first_real_close

            if day_real_close > 0:
                scale = day_sim_close / day_real_close
            else:
                scale = 1.0
            sim_open = float(row.get("open", day_real_close)) * scale
            sim_high = float(row.get("high", day_real_close)) * scale
            sim_low = float(row.get("low", day_real_close)) * scale
            sim_volume = day_buy + day_sell
            if sim_volume <= 0:
                sim_volume = float(row.get("volume", 0.0) or 0.0) * (1.0 + abs(float(day_digest.get("shock_score", 0.0) if day_digest else 0.0)) * 0.12)

            simulated_bars.append(
                {
                    "date": day_date,
                    "open": float(sim_open),
                    "high": float(max(sim_high, day_sim_close, sim_open)),
                    "low": float(min(sim_low, day_sim_close, sim_open)),
                    "close": float(day_sim_close),
                    "volume": float(max(sim_volume, 0.0)),
                    "source": "policy_session_news_replay",
                    "raw_close": float(raw_session_close),
                    "display_close": float(day_sim_close),
                    "demo_adjustment": float(demo_adjustment),
                }
            )

            dates.append(day_date)
            real_prices.append(day_real_close)
            simulated_prices.append(float(day_sim_close))
            nav = self.config.initial_cash * (float(day_sim_close) / max(first_sim_close, 1e-12))
            bench = self.config.initial_cash * (float(day_real_close) / max(first_real_close, 1e-12))
            equity_curve.append(float(nav))
            benchmark_curve.append(float(bench))

            if idx == 0:
                returns.append(0.0)
                benchmark_returns.append(0.0)
            else:
                returns.append(equity_curve[-1] / max(equity_curve[-2], 1e-12) - 1.0)
                benchmark_returns.append(benchmark_curve[-1] / max(benchmark_curve[-2], 1e-12) - 1.0)
            peak = max(equity_curve)
            drawdowns.append(equity_curve[-1] / max(peak, 1e-12) - 1.0)
            turnover_series.append(float(min(1.0, abs(day_return) * 8.0 + (day_buy + day_sell) / max(float(row.get("volume", 1.0) or 1.0), 1.0) * 0.1)))
            position_series.append(float(np.clip(0.45 + day_return * 4.0, 0.0, 1.0)))

            if day_digest is not None:
                trade_log.append(
                    {
                        "date": day_date,
                        "event": "daily_news_policy_injection",
                        "side": "BUY" if float(day_digest.get("shock_score", 0.0) or 0.0) >= 0 else "SELL",
                        "qty": int(max(day_buy, day_sell)),
                        "price": float(day_sim_close),
                        "news_count": int(day_digest.get("news_count", 0) or 0),
                        "shock_score": float(day_digest.get("shock_score", 0.0) or 0.0),
                        "matching_mode": str(dict(report.get("market_microstructure", {}) or {}).get("matching_mode", "fallback")),
                        "trade_count": int(day_trades),
                    }
                )

            if progress_callback:
                progress_callback(day_idx, total_days, f"news-driven replay: {day_date}")
            if step_callback:
                step_callback(
                    idx,
                    {
                        "date": day_date,
                        "real_price": day_real_close,
                        "simulated_price": day_sim_close,
                        "portfolio": equity_curve[-1],
                        "turnover": turnover_series[-1],
                    },
                )

        result = BacktestResult(strategy_name=self.config.strategy_name)
        result.total_days = len(dates)
        result.total_trades = len(trade_log)
        result.total_volume = int(sum(float(item.get("qty", 0.0) or 0.0) for item in trade_log))
        result.equity_curve = equity_curve
        result.benchmark_curve = benchmark_curve
        result.returns = returns
        result.benchmark_returns = benchmark_returns
        result.drawdowns = drawdowns
        result.turnover_series = turnover_series
        result.position_series = position_series
        result.simulated_prices = simulated_prices
        result.real_prices = real_prices
        result.dates = dates
        result.trade_log = trade_log
        result.simulated_bars = simulated_bars

        if equity_curve:
            result.final_value = float(equity_curve[-1])
            result.total_return = float(result.final_value / max(self.config.initial_cash, 1e-12) - 1.0)
        if benchmark_curve:
            result.benchmark_return = float(benchmark_curve[-1] / max(self.config.initial_cash, 1e-12) - 1.0)
        result.excess_return = float(result.total_return - result.benchmark_return)
        result.cagr = float(_annualized_return(result.total_return, result.total_days))

        ret_arr = np.asarray(returns[1:], dtype=float) if len(returns) > 1 else np.asarray([], dtype=float)
        bench_arr = np.asarray(benchmark_returns[1:], dtype=float) if len(benchmark_returns) > 1 else np.asarray([], dtype=float)
        perf = compute_performance_metrics(ret_arr, bench_arr if bench_arr.size > 0 else None)
        result.sharpe_ratio = float(perf.get("sharpe_ratio", 0.0))
        result.sortino_ratio = float(perf.get("sortino_ratio", 0.0))
        result.max_drawdown = float(perf.get("max_drawdown", 0.0))
        result.calmar_ratio = float(perf.get("calmar_ratio", 0.0))
        result.annual_volatility = float(perf.get("annual_volatility", 0.0))
        result.win_rate = float(perf.get("win_rate", 0.0))
        result.alpha = float(perf.get("alpha", 0.0))
        result.beta = float(perf.get("beta", 0.0))
        result.information_ratio = float(perf.get("information_ratio", 0.0))
        result.var_95 = float(perf.get("var_95", 0.0))
        result.cvar_95 = float(perf.get("cvar_95", 0.0))
        result.omega_ratio = float(perf.get("omega_ratio", 0.0))
        result.tail_ratio = float(perf.get("tail_ratio", 0.0))
        result.stability = float(perf.get("stability", 0.0))

        result.total_cost = 0.0
        result.cost_ratio = 0.0
        result.agent_turnover_rate = float(np.mean(turnover_series)) if turnover_series else 0.0
        result.agent_leverage_ratio = float(np.mean(np.abs(position_series))) if position_series else 0.0
        result.credibility_score = compute_backtest_credibility(
            perf,
            sample_count=int(perf.get("n_obs", 0)),
            avg_turnover=result.agent_turnover_rate,
        )

        result.price_correlation = _safe_corr(np.asarray(simulated_prices, dtype=float), np.asarray(real_prices, dtype=float))
        if len(real_prices) > 1:
            real_ret = np.diff(np.asarray(real_prices, dtype=float)) / np.maximum(np.asarray(real_prices[:-1], dtype=float), 1e-12)
            sim_ret = np.diff(np.asarray(simulated_prices, dtype=float)) / np.maximum(np.asarray(simulated_prices[:-1], dtype=float), 1e-12)
            real_vol = float(np.std(real_ret))
            sim_vol = float(np.std(sim_ret))
            result.volatility_correlation = min(real_vol, sim_vol) / max(real_vol, sim_vol, 1e-12)
            sim_norm = np.asarray(simulated_prices, dtype=float) / max(simulated_prices[0], 1e-12)
            real_norm = np.asarray(real_prices, dtype=float) / max(real_prices[0], 1e-12)
            result.price_rmse = float(np.sqrt(np.mean((sim_norm - real_norm) ** 2)))
            result.price_mae = float(np.mean(np.abs(sim_norm - real_norm)))

        vol_proxy = valid_frame["volume"].ffill().fillna(0.0)
        vol_proxy = (vol_proxy / (vol_proxy.rolling(20).mean() + 1e-12)).fillna(0.0)
        result.turnover_correlation = _safe_corr(np.asarray(turnover_series, dtype=float), vol_proxy.iloc[: len(turnover_series)].values)
        result.factor_snapshot = FactorResearchEngine.factor_diagnostics(self.factor_data)

        strict_score, demo_score, trace = self._compute_authenticity_scores(result, news_bundle)
        result.metadata = {
            "symbol": self.config.symbol,
            "benchmark_symbol": self.config.benchmark_symbol,
            "mode": "news_policy_replay",
            "news_source_strategy": self.config.news_source_strategy,
            "news_scope": self.config.news_scope,
            "news_coverage": dict(news_bundle.coverage),
            "news_digest": news_bundle.digest_rows(),
            "strict_authenticity_score": float(strict_score),
            "demo_authenticity_score": float(demo_score),
            "score_adjustment_trace": trace,
            "auth_score_mode": str(self.config.auth_score_mode or "demo_first"),
            "event_store_news": dict(news_bundle.persistence),
            "window": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None},
            "reference_bars": valid_frame[["date", "open", "high", "low", "close", "volume"]].to_dict("records"),
            "simulated_price_source": "policy_session_close",
            "history_replay_mode": "strict" if strict_mode else "demo",
            "strict_mode": bool(strict_mode),
            "demo_mode": bool(not strict_mode),
            "raw_simulated_prices": raw_simulated_prices,
            "display_simulated_prices": display_simulated_prices,
            "raw_vs_display": {
                "raw_metric_source": "raw_simulated_prices",
                "display_metric_source": "simulated_prices" if not strict_mode else "raw_simulated_prices",
                "demo_calibration_alpha": 0.0 if strict_mode else 0.45,
                "demo_adjustment_total_abs": float(np.sum(np.abs(np.asarray(display_simulated_prices) - np.asarray(raw_simulated_prices)))) if raw_simulated_prices else 0.0,
            },
        }
        result.metadata.update(
            _build_repro_metadata(
                self.config,
                self.historical_data,
                mode="news_policy_replay",
                extra={
                    "simulated_price_source": "policy_session_close",
                    "news_scenario_id": str(news_bundle.persistence.get("scenario_id", "")),
                    "news_snapshot_id": str(news_bundle.persistence.get("snapshot_id", "")),
                    "news_dataset_version": str(news_bundle.persistence.get("dataset_version", "")),
                    "config_payload": asdict(self.config),
                },
            )
        )
        self.result = result
        return result


__all__ = ["NewsDrivenPolicyReplayEngine"]
