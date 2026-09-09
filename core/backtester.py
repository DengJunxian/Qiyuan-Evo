"""
历史回测与量化研究框架。

设计参考因子研究、事件驱动策略接口、成本滑点模型和可组合策略模板。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd

from config import GLOBAL_CONFIG
from core.data.market_data_provider import MarketDataProvider, MarketDataQuery
from core.experiment_manifest import attach_manifest_to_metadata
from core.performance import compute_backtest_credibility, compute_performance_metrics
from core.portfolio import PortfolioConstructionLayer, PortfolioConstraints, PortfolioInput

ProgressCallback = Optional[Callable[[int, int, str], None]]
StepCallback = Optional[Callable[[int, Dict[str, Any]], None]]


@dataclass
class BacktestConfig:
    """鍥炴祴閰嶇疆"""

    symbol: str = "sh000001"
    benchmark_symbol: str = "sh000001"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period_days: int = 756

    strategy_name: str = "momentum"
    lookback: int = 20
    rebalance_frequency: int = 5
    allow_short: bool = False
    max_position: float = 1.0
    signal_threshold: float = 0.05

    initial_cash: float = GLOBAL_CONFIG.DEFAULT_CASH
    commission_rate: float = GLOBAL_CONFIG.TAX_RATE_COMMISSION
    stamp_duty_rate: float = GLOBAL_CONFIG.TAX_RATE_STAMP
    slippage_bps: float = 5.0
    market_impact: float = 0.05

    policy_shock: float = 0.0
    policy_text: str = ""
    sentiment_weight: float = 0.5
    civitas_factor_weight: float = 0.5
    news_source_strategy: str = "mixed"
    news_scope: str = "macro_index"
    news_topk_per_day: int = 8
    persist_news_events: bool = True
    auth_score_mode: str = "demo_first"
    runtime_mode: str = "SMART"
    llm_model: str = ""
    llm_mode: str = ""
    llm_model_priority: List[str] = field(default_factory=list)

    tick_per_day: int = 4
    export_qlib_bundle: bool = False
    qlib_bundle_path: str = "outputs/backtest_qlib_bundle"
    random_seed: int = 42
    feature_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """鍥炴祴缁撴灉"""

    strategy_name: str = ""

    total_days: int = 0
    total_trades: int = 0
    total_volume: int = 0

    final_value: float = 0.0
    total_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    annual_volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    stability: float = 0.0
    credibility_score: float = 0.0

    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    win_rate: float = 0.0

    total_cost: float = 0.0
    cost_ratio: float = 0.0

    agent_turnover_rate: float = 0.0
    agent_leverage_ratio: float = 0.0

    price_correlation: float = 0.0
    turnover_correlation: float = 0.0
    volatility_correlation: float = 0.0
    price_rmse: float = 0.0
    price_mae: float = 0.0

    simulated_prices: List[float] = field(default_factory=list)
    real_prices: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    simulated_bars: List[Dict[str, Any]] = field(default_factory=list)

    equity_curve: List[float] = field(default_factory=list)
    benchmark_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    benchmark_returns: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    turnover_series: List[float] = field(default_factory=list)
    position_series: List[float] = field(default_factory=list)

    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    factor_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _get_calibration_grade(self) -> str:
        avg_corr = (
            self.price_correlation
            + self.turnover_correlation
            + self.volatility_correlation
        ) / 3
        if avg_corr >= 0.8:
            return "A (优秀)"
        if avg_corr >= 0.6:
            return "B (良好)"
        if avg_corr >= 0.4:
            return "C (一般)"
        return "D (需改进)"

    def get_summary(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "total_days": self.total_days,
            "total_trades": self.total_trades,
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "excess_return": f"{self.excess_return:.2%}",
            "price_correlation": f"{self.price_correlation:.4f}",
            "turnover_correlation": f"{self.turnover_correlation:.4f}",
            "volatility_correlation": f"{self.volatility_correlation:.4f}",
            "price_rmse": f"{self.price_rmse:.4f}",
            "price_mae": f"{self.price_mae:.4f}",
            "calibration_grade": self._get_calibration_grade(),
            "agent_turnover_rate": f"{self.agent_turnover_rate:.2%}",
            "agent_leverage_ratio": f"{self.agent_leverage_ratio:.2f}",
            "cost_ratio": f"{self.cost_ratio:.2%}",
        }


@dataclass
class TransactionCostModel:
    commission_rate: float
    stamp_duty_rate: float
    min_commission: float = 0.0

    def estimate(self, trade_value: float, is_sell: bool) -> float:
        if trade_value <= 0:
            return 0.0
        commission = max(trade_value * self.commission_rate, self.min_commission)
        stamp_duty = trade_value * self.stamp_duty_rate if is_sell else 0.0
        return commission + stamp_duty


@dataclass
class SlippageModel:
    slippage_bps: float
    market_impact: float

    def apply(
        self,
        mid_price: float,
        trade_value: float,
        daily_volume: float,
        volatility: float,
        side: str,
    ) -> float:
        if mid_price <= 0:
            return 0.0
        if trade_value <= 0:
            return mid_price

        base = self.slippage_bps / 10000.0
        liquidity_ratio = trade_value / max(daily_volume * mid_price, 1e-9)
        impact = self.market_impact * min(max(liquidity_ratio, 0.0), 1.0)
        vol_penalty = max(volatility, 0.0) * 0.3
        slippage = base + impact + vol_penalty

        signed = 1.0 if side == "buy" else -1.0
        return mid_price * (1.0 + signed * slippage)

class HistoricalDataLoader:
    """Historical data loader backed by the unified market data provider."""

    _provider = MarketDataProvider()

    @staticmethod
    def _to_date_str(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d")

    @staticmethod
    def _daily_projection(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df[["date", "open", "high", "low", "close", "volume"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        return out.reset_index(drop=True)

    @staticmethod
    def load_daily_data(
        symbol: str = "sh000001",
        period_days: int = 1095,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: ProgressCallback = None,
    ) -> pd.DataFrame:
        if progress_callback:
            progress_callback(0, 100, "Connecting market data providers...")

        try:
            if progress_callback:
                progress_callback(15, 100, f"Downloading daily bars for {symbol}...")

            query = MarketDataQuery(
                symbol=symbol,
                interval="1d",
                start=HistoricalDataLoader._to_date_str(start_date),
                end=HistoricalDataLoader._to_date_str(end_date),
                period_days=period_days,
                adjust="",
                market="CN",
            )
            frame = HistoricalDataLoader._provider.get_ohlcv(query, use_cache=True, freeze_snapshot=False)
            df = HistoricalDataLoader._daily_projection(frame)

            if df.empty:
                raise ValueError(f"no usable market data for {symbol}")

            df = df.reset_index(drop=True)

            if progress_callback:
                progress_callback(100, 100, f"Loaded {len(df)} rows")

            return df

        except Exception as exc:
            if progress_callback:
                progress_callback(100, 100, f"Load failed: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _build_intraday_fallback(daily: pd.DataFrame, date_value: Optional[str]) -> pd.DataFrame:
        if daily.empty:
            return pd.DataFrame()

        target_row = daily.iloc[-1]
        if date_value:
            matched = daily[daily["date"] == date_value]
            if not matched.empty:
                target_row = matched.iloc[-1]

        times = ["09:30", "10:30", "11:30", "13:30", "14:30", "15:00"]
        o = float(target_row["open"])
        h = float(target_row["high"])
        low_price = float(target_row["low"])
        c = float(target_row["close"])
        prices = np.linspace(o, c, len(times))
        prices[1] = o + (h - o) * 0.45
        prices[2] = h
        prices[3] = h - (h - low_price) * 0.35
        prices[4] = low_price + (c - low_price) * 0.5
        prices[5] = c
        return pd.DataFrame(
            {
                "time": times,
                "price": prices,
                "volume": [float(target_row["volume"]) / len(times)] * len(times),
            }
        )

    @staticmethod
    def load_intraday_data(
        symbol: str = "sh000001",
        date_value: Optional[str] = None,
        progress_callback: ProgressCallback = None,
    ) -> pd.DataFrame:
        if progress_callback:
            progress_callback(0, 100, "Loading minute bars...")

        start = None
        end = None
        if date_value:
            date_str = HistoricalDataLoader._to_date_str(date_value)
            if date_str:
                start = f"{date_str} 09:30:00"
                end = f"{date_str} 15:00:00"

        out = pd.DataFrame()
        try:
            query = MarketDataQuery(
                symbol=symbol,
                interval="1m",
                start=start,
                end=end,
                period_days=1,
                adjust="qfq",
                market="CN",
            )
            minute = HistoricalDataLoader._provider.get_ohlcv(query, use_cache=True, freeze_snapshot=False)
            if not minute.empty:
                ts = pd.to_datetime(minute["datetime"], errors="coerce")
                out = pd.DataFrame(
                    {
                        "time": ts.dt.strftime("%H:%M"),
                        "price": pd.to_numeric(minute["close"], errors="coerce"),
                        "volume": pd.to_numeric(minute["volume"], errors="coerce"),
                    }
                ).dropna(subset=["time", "price"])
        except Exception:
            out = pd.DataFrame()

        if out.empty:
            daily = HistoricalDataLoader.load_daily_data(symbol=symbol, period_days=2)
            out = HistoricalDataLoader._build_intraday_fallback(daily, date_value)

        if progress_callback:
            progress_callback(100, 100, f"Loaded {len(out)} intraday rows")

        return out

class FactorResearchEngine:
    """Factor research helpers for building backtest-ready feature frames."""
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / (loss + 1e-12)
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def build_features(
        price_frame: pd.DataFrame,
        policy_shock: float = 0.0,
        sentiment_weight: float = 0.5,
        civitas_factor_weight: float = 0.5,
        civitas_factors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = price_frame.copy()
        if df.empty:
            return pd.DataFrame()

        df["ret_1d"] = df["close"].pct_change()
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_20d"] = df["close"].pct_change(20)

        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()

        df["momentum_20"] = df["close"] / (ma20 + 1e-12) - 1.0
        df["reversal_z"] = (df["close"] - ma20) / (std20 + 1e-12)
        df["volatility_20"] = df["ret_1d"].rolling(20).std().fillna(0.0)

        vol_ma20 = df["volume"].rolling(20).mean()
        vol_std20 = df["volume"].rolling(20).std()
        df["volume_z"] = (df["volume"] - vol_ma20) / (vol_std20 + 1e-12)

        df["rsi_14"] = FactorResearchEngine._rsi(df["close"], 14)
        df["policy_shock_factor"] = float(policy_shock)

        base_sentiment = np.tanh(df["volume_z"].fillna(0.0) * 0.6 + policy_shock * 1.2)
        df["sentiment_factor"] = base_sentiment * sentiment_weight

        if civitas_factors is not None and not civitas_factors.empty:
            cf = civitas_factors.copy()
            if "date" in cf.columns:
                cf["date"] = pd.to_datetime(cf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                numeric_cols = [c for c in cf.columns if c != "date"]
                for col in numeric_cols:
                    cf[col] = pd.to_numeric(cf[col], errors="coerce")
                cf["civitas_factor"] = cf[numeric_cols].mean(axis=1).fillna(0.0)
                df = df.merge(cf[["date", "civitas_factor"]], on="date", how="left")
                df["civitas_factor"] = df["civitas_factor"].fillna(0.0)
            else:
                df["civitas_factor"] = 0.0
        else:
            df["civitas_factor"] = 0.0

        df["composite_alpha"] = (
            0.45 * df["momentum_20"].fillna(0.0)
            - 0.20 * df["reversal_z"].fillna(0.0)
            + 0.20 * df["sentiment_factor"].fillna(0.0)
            + 0.15 * civitas_factor_weight * df["civitas_factor"].fillna(0.0)
        )

        df["label_return_1d"] = df["close"].shift(-1) / (df["close"] + 1e-12) - 1.0
        return df

    @staticmethod
    def factor_diagnostics(factor_frame: pd.DataFrame) -> List[Dict[str, Any]]:
        if factor_frame.empty:
            return []

        target = factor_frame["label_return_1d"].fillna(0.0)
        candidates = [
            "momentum_20",
            "reversal_z",
            "volatility_20",
            "volume_z",
            "rsi_14",
            "policy_shock_factor",
            "sentiment_factor",
            "civitas_factor",
            "composite_alpha",
        ]

        rows: List[Dict[str, Any]] = []
        for col in candidates:
            if col not in factor_frame.columns:
                continue
            signal = factor_frame[col].fillna(0.0)
            ic = _safe_corr(signal.values, target.values)
            rank_ic = _safe_corr(signal.rank().values, target.rank().values)
            dispersion = float(signal.std())
            rows.append(
                {
                    "factor": col,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "dispersion": dispersion,
                }
            )

        rows.sort(key=lambda x: abs(x["ic"]), reverse=True)
        return rows

@dataclass
class StrategyContext:
    day_index: int
    date: str
    cash: float
    shares: float
    price: float
    portfolio_value: float
    current_weight: float
    config: BacktestConfig


class BaseStrategy:
    name = "base"

    def __init__(self, config: BacktestConfig):
        self.config = config

    def initialize(self, _context: StrategyContext) -> None:
        return None

    def handle_data(self, context: StrategyContext, row: pd.Series) -> float:
        del context, row
        return 0.0


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def handle_data(self, _context: StrategyContext, row: pd.Series) -> float:
        score = float(row.get("momentum_20", 0.0) + 0.4 * row.get("composite_alpha", 0.0))
        if abs(score) < self.config.signal_threshold:
            return 0.0
        return float(np.clip(score * 2.5, -self.config.max_position, self.config.max_position))


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def handle_data(self, _context: StrategyContext, row: pd.Series) -> float:
        score = float(-row.get("reversal_z", 0.0) + 0.15 * row.get("sentiment_factor", 0.0))
        if abs(score) < self.config.signal_threshold:
            return 0.0
        return float(np.clip(score * 0.55, -self.config.max_position, self.config.max_position))


class RiskParityStrategy(BaseStrategy):
    name = "risk_parity"

    def handle_data(self, _context: StrategyContext, row: pd.Series) -> float:
        vol = float(abs(row.get("volatility_20", 0.0)))
        direction = float(np.sign(row.get("composite_alpha", 0.0)))
        target_vol = 0.012
        gross = min(self.config.max_position, target_vol / max(vol, 0.004))
        return float(np.clip(direction * gross, -self.config.max_position, self.config.max_position))


class NewsDrivenStrategy(BaseStrategy):
    name = "news_driven"

    def handle_data(self, _context: StrategyContext, row: pd.Series) -> float:
        score = float(
            0.6 * row.get("sentiment_factor", 0.0)
            + 0.3 * row.get("policy_shock_factor", 0.0)
            + 0.1 * row.get("composite_alpha", 0.0)
        )
        if abs(score) < self.config.signal_threshold:
            return 0.0
        return float(np.clip(score * 2.2, -self.config.max_position, self.config.max_position))


class PortfolioSystemStrategy(BaseStrategy):
    """Policy/sentiment-aware portfolio construction over risky asset + cash."""

    name = "portfolio_system"

    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self._constraints = PortfolioConstraints(
            long_only=True,
            fully_invested=True,
            min_weight=0.0,
            max_weight=max(0.0, float(config.max_position)),
            sentiment_penalty=max(0.0, float(config.sentiment_weight)) * 0.15,
            policy_penalty=max(0.0, abs(float(config.policy_shock))) * 0.20,
            target_max_drawdown=0.20,
        )
        self._constructor = PortfolioConstructionLayer(
            method="mean_variance",
            constraints=self._constraints,
            risk_aversion=2.0,
        )

    def handle_data(self, context: StrategyContext, row: pd.Series) -> float:
        alpha = float(row.get("composite_alpha", 0.0))
        sentiment = float(row.get("sentiment_factor", 0.0))
        policy = float(row.get("policy_shock_factor", 0.0))
        volatility = max(float(abs(row.get("volatility_20", 0.0))), 0.002)


        expected_returns = pd.Series({"risky_asset": alpha + 0.35 * sentiment + 0.2 * policy, "cash": 0.0})
        cov = pd.DataFrame(
            [[volatility**2, 0.0], [0.0, 1e-6]],
            index=["risky_asset", "cash"],
            columns=["risky_asset", "cash"],
        )
        current = pd.Series(
            {
                "risky_asset": float(np.clip(context.current_weight, 0.0, 1.0)),
                "cash": float(np.clip(1.0 - context.current_weight, 0.0, 1.0)),
            }
        )
        input_data = PortfolioInput(
            expected_returns=expected_returns,
            cov_matrix=cov,
            current_weights=current,
            sentiment_risk=pd.Series({"risky_asset": abs(sentiment), "cash": 0.0}),
            policy_risk=pd.Series({"risky_asset": abs(policy), "cash": 0.0}),
        )
        weights = self._constructor.optimize(input_data)
        risky_weight = float(weights.get("risky_asset", 0.0))
        return float(np.clip(risky_weight, 0.0, self.config.max_position))


STRATEGY_REGISTRY = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "risk_parity": RiskParityStrategy,
    "news_driven": NewsDrivenStrategy,
    "portfolio_system": PortfolioSystemStrategy,
}


def build_strategy(config: BacktestConfig) -> BaseStrategy:
    strategy_cls = STRATEGY_REGISTRY.get(config.strategy_name, MomentumStrategy)
    return strategy_cls(config)


def _seed_everything(seed: int) -> None:
    normalized = int(seed) % (2**32 - 1)
    random.seed(normalized)
    np.random.seed(normalized)


def _hash_frame(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "0" * 64
    payload = pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _build_snapshot_info(
    frame: pd.DataFrame,
    *,
    symbol: str,
    benchmark_symbol: str,
    mode: str,
) -> Dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "mode": mode,
            "rows": 0,
            "start_date": None,
            "end_date": None,
            "sha256": "0" * 64,
            "snapshot_id": f"{mode}-empty",
            "symbol": symbol,
            "benchmark_symbol": benchmark_symbol,
        }

    digest = _hash_frame(frame)
    dates = pd.to_datetime(frame.get("date", pd.Series(dtype=str)), errors="coerce").dropna()
    start_date = dates.min().strftime("%Y-%m-%d") if not dates.empty else None
    end_date = dates.max().strftime("%Y-%m-%d") if not dates.empty else None
    return {
        "mode": mode,
        "rows": int(len(frame)),
        "start_date": start_date,
        "end_date": end_date,
        "sha256": digest,
        "snapshot_id": f"{mode}-{digest[:12]}",
        "symbol": symbol,
        "benchmark_symbol": benchmark_symbol,
    }


def _build_repro_metadata(
    config: BacktestConfig,
    frame: pd.DataFrame,
    *,
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config_hash = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    snapshot = _build_snapshot_info(
        frame,
        symbol=config.symbol,
        benchmark_symbol=config.benchmark_symbol,
        mode=mode,
    )
    metadata = {
        "replay_mode": mode,
        "config_hash": config_hash,
        "random_seed": int(config.random_seed),
        "feature_flags": dict(config.feature_flags or {}),
        "data_snapshot": snapshot,
    }
    if extra:
        metadata.update(extra)
    return attach_manifest_to_metadata(
        metadata,
        module=f"backtest_{mode}",
        seed=int(config.random_seed),
        dataset_snapshot_id=str(snapshot.get("snapshot_id", "")),
        feature_flags=dict(config.feature_flags or {}),
        config=asdict(config),
    )


class FactorBacktestEngine:
    """鍘嗗彶鍥炴祴寮曟搸"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.historical_data: pd.DataFrame = pd.DataFrame()
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        self.factor_data: pd.DataFrame = pd.DataFrame()
        self.result: BacktestResult = BacktestResult()

        self.current_day_index: int = 0
        self.is_running: bool = False

        self.cost_model = TransactionCostModel(
            commission_rate=self.config.commission_rate,
            stamp_duty_rate=self.config.stamp_duty_rate,
        )
        self.slippage_model = SlippageModel(
            slippage_bps=self.config.slippage_bps,
            market_impact=self.config.market_impact,
        )

    def load_data(self, progress_callback: ProgressCallback = None) -> bool:
        self.historical_data = HistoricalDataLoader.load_daily_data(
            symbol=self.config.symbol,
            period_days=self.config.period_days,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            progress_callback=progress_callback,
        )

        if self.historical_data.empty:
            self.benchmark_data = pd.DataFrame()
            return False

        if self.config.benchmark_symbol == self.config.symbol:
            self.benchmark_data = self.historical_data[["date", "close"]].copy()
            self.benchmark_data.rename(columns={"close": "benchmark_close"}, inplace=True)
            return True

        benchmark = HistoricalDataLoader.load_daily_data(
            symbol=self.config.benchmark_symbol,
            period_days=self.config.period_days,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            progress_callback=None,
        )
        if benchmark.empty:
            benchmark = self.historical_data[["date", "close"]].copy()

        benchmark = benchmark[["date", "close"]].copy()
        benchmark.rename(columns={"close": "benchmark_close"}, inplace=True)
        self.benchmark_data = benchmark
        return True

    def get_day_data(self, day_index: int) -> Optional[Dict[str, Any]]:
        if day_index >= len(self.historical_data):
            return None

        row = self.historical_data.iloc[day_index]
        return {
            "date": str(row.get("date", "")),
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)),
        }

    def _prepare_frames(self, civitas_factors: Optional[pd.DataFrame]) -> pd.DataFrame:
        merged = self.historical_data.copy()
        if self.benchmark_data.empty:
            self.benchmark_data = merged[["date", "close"]].rename(columns={"close": "benchmark_close"})

        merged = merged.merge(self.benchmark_data, on="date", how="left")
        merged["benchmark_close"] = merged["benchmark_close"].ffill().bfill()

        feature_frame = FactorResearchEngine.build_features(
            merged,
            policy_shock=self.config.policy_shock,
            sentiment_weight=self.config.sentiment_weight,
            civitas_factor_weight=self.config.civitas_factor_weight,
            civitas_factors=civitas_factors,
        )

        self.factor_data = feature_frame
        return feature_frame

    def run_backtest(
        self,
        population: Any = None,
        market_manager: Any = None,
        progress_callback: ProgressCallback = None,
        step_callback: StepCallback = None,
        civitas_factors: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        del market_manager
        _seed_everything(self.config.random_seed)

        if self.historical_data.empty:
            if not self.load_data(progress_callback):
                self.result = BacktestResult(strategy_name=self.config.strategy_name)
                return self.result

        frame = self._prepare_frames(civitas_factors)
        if frame.empty:
            self.result = BacktestResult(strategy_name=self.config.strategy_name)
            return self.result

        frame = frame.reset_index(drop=True)

        self.cost_model = TransactionCostModel(
            commission_rate=self.config.commission_rate,
            stamp_duty_rate=self.config.stamp_duty_rate,
        )
        self.slippage_model = SlippageModel(
            slippage_bps=self.config.slippage_bps,
            market_impact=self.config.market_impact,
        )

        strategy = build_strategy(self.config)

        cash = float(self.config.initial_cash)
        shares = 0.0
        total_cost = 0.0
        total_trades = 0
        total_volume = 0.0

        equity_curve: List[float] = []
        benchmark_curve: List[float] = []
        returns: List[float] = []
        benchmark_returns: List[float] = []
        drawdowns: List[float] = []
        turnover_series: List[float] = []
        position_series: List[float] = []
        simulated_prices: List[float] = []
        real_prices: List[float] = []
        dates: List[str] = []
        trade_log: List[Dict[str, Any]] = []

        valid_frame = frame.dropna(subset=["close", "benchmark_close"]).reset_index(drop=True)
        if valid_frame.empty:
            self.result = BacktestResult(strategy_name=self.config.strategy_name)
            return self.result

        first_close = float(valid_frame.loc[0, "close"])
        first_bench = float(valid_frame.loc[0, "benchmark_close"])

        strategy.initialize(
            StrategyContext(
                day_index=0,
                date=str(valid_frame.loc[0, "date"]),
                cash=cash,
                shares=shares,
                price=first_close,
                portfolio_value=cash,
                current_weight=0.0,
                config=self.config,
            )
        )

        self.is_running = True
        total_days = len(valid_frame)

        for idx, row in valid_frame.iterrows():
            if not self.is_running:
                break

            self.current_day_index = idx
            price = float(row["close"])
            bench_price = float(row["benchmark_close"])
            volume = float(row.get("volume", 0.0))
            volatility = float(abs(row.get("volatility_20", 0.0)))
            day_date = str(row.get("date", ""))

            pre_value = cash + shares * price
            current_weight = (shares * price / pre_value) if pre_value > 1e-12 else 0.0

            target_weight = current_weight
            rebalance_due = idx >= self.config.lookback and ((idx - self.config.lookback) % max(self.config.rebalance_frequency, 1) == 0)
            if rebalance_due:
                context = StrategyContext(
                    day_index=idx,
                    date=day_date,
                    cash=cash,
                    shares=shares,
                    price=price,
                    portfolio_value=pre_value,
                    current_weight=current_weight,
                    config=self.config,
                )
                target_weight = strategy.handle_data(context, row)
                if not self.config.allow_short:
                    target_weight = max(target_weight, 0.0)
                target_weight = float(np.clip(target_weight, -self.config.max_position, self.config.max_position))

            trade_value_target = (target_weight - current_weight) * pre_value
            trade_notional = 0.0
            turnover = 0.0
            if abs(trade_value_target) > pre_value * 1e-3:
                side = "buy" if trade_value_target > 0 else "sell"
                exec_price = self.slippage_model.apply(
                    mid_price=price,
                    trade_value=abs(trade_value_target),
                    daily_volume=volume,
                    volatility=volatility,
                    side=side,
                )

                qty = trade_value_target / max(exec_price, 1e-12)

                if not self.config.allow_short and shares + qty < 0:
                    qty = -shares

                if qty > 0:
                    max_affordable = cash / max(exec_price * (1 + self.config.commission_rate), 1e-12)
                    qty = min(qty, max_affordable)

                trade_notional = abs(qty * exec_price)
                cost = self.cost_model.estimate(trade_notional, is_sell=qty < 0)

                if qty > 0 and trade_notional + cost > cash:
                    scale = cash / max(trade_notional + cost, 1e-12)
                    qty *= max(min(scale, 1.0), 0.0)
                    trade_notional = abs(qty * exec_price)
                    cost = self.cost_model.estimate(trade_notional, is_sell=qty < 0)

                if abs(qty) > 1e-8:
                    shares += qty
                    cash -= qty * exec_price
                    cash -= cost
                    total_cost += cost
                    total_trades += 1
                    total_volume += abs(qty)
                    turnover = trade_notional / max(pre_value, 1e-12)

                    trade_log.append(
                        {
                            "date": day_date,
                            "side": "BUY" if qty > 0 else "SELL",
                            "qty": float(qty),
                            "price": float(exec_price),
                            "notional": float(trade_notional),
                            "cost": float(cost),
                            "target_weight": float(target_weight),
                        }
                    )

            value = cash + shares * price
            if value < 0:
                value = 0.0

            equity_curve.append(value)
            bench_nav = self.config.initial_cash * (bench_price / max(first_bench, 1e-12))
            benchmark_curve.append(bench_nav)

            real_prices.append(price)
            simulated_prices.append(first_close * (value / max(self.config.initial_cash, 1e-12)))
            dates.append(day_date)

            position_weight = (shares * price / value) if value > 1e-12 else 0.0
            position_series.append(position_weight)
            turnover_series.append(turnover)

            if len(equity_curve) > 1:
                ret = equity_curve[-1] / max(equity_curve[-2], 1e-12) - 1.0
                bench_ret = benchmark_curve[-1] / max(benchmark_curve[-2], 1e-12) - 1.0
            else:
                ret = 0.0
                bench_ret = 0.0

            returns.append(ret)
            benchmark_returns.append(bench_ret)

            peak = max(equity_curve)
            drawdown = value / max(peak, 1e-12) - 1.0
            drawdowns.append(drawdown)

            if progress_callback:
                progress_callback(idx + 1, total_days, f"鍥炴祴鏃ユ湡: {day_date}")

            if step_callback:
                step_callback(
                    idx,
                    {
                        "date": day_date,
                        "real_price": price,
                        "simulated_price": simulated_prices[-1],
                        "portfolio": value,
                        "turnover": turnover,
                    },
                )

        self.is_running = False

        result = BacktestResult(strategy_name=self.config.strategy_name)
        result.total_days = len(dates)
        result.total_trades = total_trades
        result.total_volume = int(total_volume)

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

        if equity_curve:
            result.final_value = equity_curve[-1]
            result.total_return = result.final_value / max(self.config.initial_cash, 1e-12) - 1.0
        if benchmark_curve:
            result.benchmark_return = benchmark_curve[-1] / max(self.config.initial_cash, 1e-12) - 1.0

        result.excess_return = result.total_return - result.benchmark_return
        result.cagr = _annualized_return(result.total_return, result.total_days)

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

        result.total_cost = float(total_cost)
        result.cost_ratio = total_cost / max(self.config.initial_cash, 1e-12)

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

        agent_count = _safe_population_size(population)
        result.metadata = {
            "symbol": self.config.symbol,
            "benchmark_symbol": self.config.benchmark_symbol,
            "agent_count": agent_count,
            "credibility_score": result.credibility_score,
            "tail_risk": {
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
                "omega_ratio": result.omega_ratio,
                "tail_ratio": result.tail_ratio,
            },
            "window": {
                "start": dates[0] if dates else None,
                "end": dates[-1] if dates else None,
            },
            "framework_notes": {
                "qlib": "factor generation + label building + research export",
                "zipline": "initialize/handle_data event-driven API style",
                "lean": "transaction cost and slippage model support",
                "bt": "strategy templating and pluggable execution stack",
            },
        }
        result.metadata.update(
            _build_repro_metadata(
                self.config,
                self.historical_data,
                mode="factor",
                extra={"simulated_price_source": "equity_curve_scaled"},
            )
        )

        if self.config.export_qlib_bundle:
            try:
                bundle_path = self.export_qlib_bundle(self.config.qlib_bundle_path)
                result.metadata["qlib_bundle_path"] = bundle_path
            except Exception as exc:  # pragma: no cover - file system edge path
                result.metadata["qlib_bundle_error"] = str(exc)

        self.result = result
        return result

    def export_qlib_bundle(self, output_dir: str) -> str:
        if self.factor_data.empty:
            raise ValueError("No factor data to export. Run backtest first.")

        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)

        export_frame = self.factor_data.copy()
        export_frame["instrument"] = self.config.symbol

        factor_cols = [
            c
            for c in export_frame.columns
            if c
            not in {
                "instrument",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "benchmark_close",
            }
        ]

        feature_cols = [
            "instrument",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            *factor_cols,
        ]
        feature_frame = export_frame[feature_cols].copy()

        label_frame = export_frame[["instrument", "date", "label_return_1d"]].copy()
        label_frame.rename(columns={"label_return_1d": "LABEL0"}, inplace=True)

        feature_path = base / "features.csv"
        label_path = base / "labels.csv"
        meta_path = base / "meta.json"

        feature_frame.to_csv(feature_path, index=False, encoding="utf-8-sig")
        label_frame.to_csv(label_path, index=False, encoding="utf-8-sig")

        meta = {
            "symbol": self.config.symbol,
            "generated_at": datetime.now().isoformat(),
            "rows": int(len(feature_frame)),
            "factors": [c for c in factor_cols if c != "label_return_1d"],
            "label": "LABEL0 = t+1 close return",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(base.resolve())

    def stop(self) -> None:
        self.is_running = False

    def get_progress(self) -> Tuple[int, int]:
        return self.current_day_index, len(self.historical_data)


class HistoricalBacktester(FactorBacktestEngine):
    """Backward-compatible alias for the factor backtest engine."""


class BacktestReportGenerator:
    """Build HTML backtest reports."""

    @staticmethod
    def generate_html_report(result: BacktestResult) -> str:
        summary = result.get_summary()
        top_factors = result.factor_snapshot[:5]

        factor_rows = "".join(
            [
                (
                    "<tr>"
                    f"<td style='padding:8px;border-bottom:1px solid #233040;'>{row['factor']}</td>"
                    f"<td style='padding:8px;text-align:right;border-bottom:1px solid #233040;'>{row['ic']:.4f}</td>"
                    f"<td style='padding:8px;text-align:right;border-bottom:1px solid #233040;'>{row['rank_ic']:.4f}</td>"
                    "</tr>"
                )
                for row in top_factors
            ]
        )

        if not factor_rows:
            factor_rows = "<tr><td colspan='3' style='padding:8px;color:#8aa0b8;'>No factor diagnostics available.</td></tr>"

        return f"""
        <div style="font-family:'Segoe UI',sans-serif;padding:20px;background:#0e141d;border:1px solid #233040;border-radius:12px;color:#d5deea;">
            <h2 style="margin:0 0 12px 0;color:#7fd1ff;">Historical Backtest Report</h2>
            <p style="margin:0 0 16px 0;color:#9cb0c6;">Strategy <b>{summary['strategy']}</b> | Calibration <b>{summary['calibration_grade']}</b></p>

            <div style="display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px;margin-bottom:16px;">
                <div style="background:#152233;padding:10px;border-radius:8px;"><div style="font-size:12px;color:#8aa0b8;">Total Return</div><div style="font-size:18px;font-weight:bold;">{summary['total_return']}</div></div>
                <div style="background:#152233;padding:10px;border-radius:8px;"><div style="font-size:12px;color:#8aa0b8;">CAGR</div><div style="font-size:18px;font-weight:bold;">{summary['cagr']}</div></div>
                <div style="background:#152233;padding:10px;border-radius:8px;"><div style="font-size:12px;color:#8aa0b8;">Sharpe</div><div style="font-size:18px;font-weight:bold;">{summary['sharpe_ratio']}</div></div>
                <div style="background:#152233;padding:10px;border-radius:8px;"><div style="font-size:12px;color:#8aa0b8;">Max Drawdown</div><div style="font-size:18px;font-weight:bold;">{summary['max_drawdown']}</div></div>
            </div>

            <h4 style="margin:12px 0 8px 0;color:#a6e47b;">Calibration Metrics</h4>
            <table style="width:100%;border-collapse:collapse;background:#101a27;border-radius:8px;overflow:hidden;">
                <tr style="background:#1c2a3b;">
                    <th style="padding:8px;text-align:left;">Metric</th>
                    <th style="padding:8px;text-align:right;">Value</th>
                </tr>
                <tr><td style="padding:8px;border-bottom:1px solid #233040;">Price Correlation</td><td style="padding:8px;text-align:right;border-bottom:1px solid #233040;">{summary['price_correlation']}</td></tr>
                <tr><td style="padding:8px;border-bottom:1px solid #233040;">Turnover Correlation</td><td style="padding:8px;text-align:right;border-bottom:1px solid #233040;">{summary['turnover_correlation']}</td></tr>
                <tr><td style="padding:8px;border-bottom:1px solid #233040;">Volatility Correlation</td><td style="padding:8px;text-align:right;border-bottom:1px solid #233040;">{summary['volatility_correlation']}</td></tr>
                <tr><td style="padding:8px;">Cost Ratio</td><td style="padding:8px;text-align:right;">{summary['cost_ratio']}</td></tr>
            </table>

            <h4 style="margin:16px 0 8px 0;color:#ffcf70;">Factor Diagnostics (Top IC)</h4>
            <table style="width:100%;border-collapse:collapse;background:#101a27;border-radius:8px;overflow:hidden;">
                <tr style="background:#1c2a3b;">
                    <th style="padding:8px;text-align:left;">Factor</th>
                    <th style="padding:8px;text-align:right;">IC</th>
                    <th style="padding:8px;text-align:right;">Rank IC</th>
                </tr>
                {factor_rows}
            </table>
        </div>
        """

    @staticmethod
    def build_tear_sheet_payload(
        result: BacktestResult,
        *,
        scenario_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from core.tear_sheet import build_standard_tear_sheet

        metadata = dict(result.metadata or {})
        if extra_metadata:
            metadata.update(extra_metadata)

        payload = build_standard_tear_sheet(
            scenario_name=scenario_name or result.strategy_name or "scenario",
            returns=result.returns[1:] if len(result.returns) > 1 else result.returns,
            benchmark_returns=(
                result.benchmark_returns[1:] if len(result.benchmark_returns) > 1 else result.benchmark_returns
            ),
            dates=result.dates[1:] if len(result.dates) > 1 else result.dates,
            metadata=metadata,
        )
        return payload.to_dict()

    @staticmethod
    def export_tear_sheet_files(
        result: BacktestResult,
        output_dir: str,
        *,
        scenario_name: Optional[str] = None,
    ) -> Dict[str, str]:
        from core.tear_sheet import (
            build_standard_tear_sheet,
            export_tear_sheet_html,
            export_tear_sheet_json,
        )

        payload = build_standard_tear_sheet(
            scenario_name=scenario_name or result.strategy_name or "scenario",
            returns=result.returns[1:] if len(result.returns) > 1 else result.returns,
            benchmark_returns=(
                result.benchmark_returns[1:] if len(result.benchmark_returns) > 1 else result.benchmark_returns
            ),
            dates=result.dates[1:] if len(result.dates) > 1 else result.dates,
            metadata=result.metadata,
        )

        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        safe_name = (payload.scenario_name or "scenario").replace(" ", "_")
        json_path = export_tear_sheet_json(payload, str(base / f"{safe_name}_tearsheet.json"))
        html_path = export_tear_sheet_html(payload, str(base / f"{safe_name}_tearsheet.html"))
        return {"json": json_path, "html": html_path}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    n = min(x.size, y.size)
    x0 = x[:n]
    y0 = y[:n]
    if np.std(x0) < 1e-12 or np.std(y0) < 1e-12:
        return 0.0
    c = np.corrcoef(x0, y0)[0, 1]
    if math.isnan(c):
        return 0.0
    return float(c)


def _annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    if periods <= 0:
        return 0.0
    growth = 1.0 + total_return
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / periods) - 1.0)


def _annualized_sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    ex = returns - rf / periods_per_year
    std = float(np.std(ex))
    if std < 1e-12:
        return 0.0
    return float(np.mean(ex) / std * np.sqrt(periods_per_year))


def _annualized_sortino(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    ex = returns - rf / periods_per_year
    downside = ex[ex < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else 0.0
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(ex) / downside_std * np.sqrt(periods_per_year))


def _alpha_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, rf: float = 0.0) -> Tuple[float, float]:
    if strategy_returns.size < 2 or benchmark_returns.size < 2:
        return 0.0, 0.0

    n = min(strategy_returns.size, benchmark_returns.size)
    sr = strategy_returns[:n]
    br = benchmark_returns[:n]

    var_b = float(np.var(br))
    if var_b < 1e-12:
        return 0.0, 0.0

    cov = float(np.cov(sr, br)[0, 1])
    beta = cov / var_b

    alpha_daily = float(np.mean(sr - rf / 252.0) - beta * np.mean(br - rf / 252.0))
    alpha_annual = alpha_daily * 252.0
    return alpha_annual, float(beta)


def _information_ratio(excess_returns: np.ndarray, periods_per_year: int = 252) -> float:
    if excess_returns.size < 2:
        return 0.0
    std = float(np.std(excess_returns))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))


def _safe_population_size(population: Any) -> int:
    if population is None:
        return 0

    if hasattr(population, "smart_agents"):
        agents = getattr(population, "smart_agents")
        try:
            return len(agents)
        except Exception:
            return 0

    if hasattr(population, "agents"):
        agents = getattr(population, "agents")
        try:
            return len(agents)
        except Exception:
            return 0

    try:
        return len(population)
    except Exception:
        return 0
