from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.backtester import (
    BacktestConfig,
    BacktestResult,
    FactorBacktestEngine,
    FactorResearchEngine,
    HistoricalDataLoader,
    SlippageModel,
    TransactionCostModel,
    _annualized_return,
    _build_repro_metadata,
    _safe_corr,
    _seed_everything,
)
from core.exchange.order_book import OrderBook
from core.performance import compute_backtest_credibility, compute_performance_metrics
from core.types import Order, OrderSide, OrderType
from core.event_store import EventStore

FEATURE_FLAG_EVENT_DRIVEN_V2 = "history_replay_event_driven_v2"
FEATURE_FLAG_ROLLING_CALIBRATION_V1 = "history_replay_rolling_calibration_v1"


class AgentReplayEngine(FactorBacktestEngine):
    """Deterministic agent-based historical replay engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        super().__init__(config)
        self.trade_tape: List[Dict[str, Any]] = []
        self.simulated_bars: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(self.config.random_seed)
        self.event_store: Optional[EventStore] = None
        self.dataset_version: str = ""
        self.snapshot_id: str = ""
        self.scenario_id: str = ""

    def configure_event_store(
        self,
        *,
        event_store: Optional[EventStore],
        dataset_version: str,
        snapshot_id: str = "",
        scenario_id: str = "",
    ) -> None:
        self.event_store = event_store
        self.dataset_version = str(dataset_version or "")
        self.snapshot_id = str(snapshot_id or "")
        self.scenario_id = str(scenario_id or "")

    def _load_from_event_store_if_enabled(self) -> bool:
        if self.event_store is None:
            return False
        if not self.dataset_version:
            return False
        if not self.config.feature_flags.get("event_store_v1", False):
            return False

        start_time = str(self.config.start_date or "")
        end_time = str(self.config.end_date or "")
        if self.scenario_id:
            frame = self.event_store.query_scenario_events(
                self.dataset_version,
                self.scenario_id,
                visible_at=end_time or None,
            )
            if frame is not None and not frame.empty and "payload_json" in frame.columns:
                payloads = frame["payload_json"].apply(lambda x: json.loads(str(x or "{}")))
                payload_frame = pd.json_normalize(payloads.tolist())
                frame = pd.concat([frame.drop(columns=["payload_json", "metadata_json"], errors="ignore"), payload_frame], axis=1)
        else:
            frame = self.event_store.load_market_bars(
                self.dataset_version,
                start_time=start_time or None,
                end_time=end_time or None,
                visible_at=end_time or None,
            )
        if frame is None or frame.empty:
            return False

        usable = frame.copy()
        rename_map = {"timestamp": "date"}
        for old, new in rename_map.items():
            if old in usable.columns and new not in usable.columns:
                usable[new] = usable[old]
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in usable.columns:
                return False
        usable["date"] = pd.to_datetime(usable["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume"]:
            usable[col] = pd.to_numeric(usable[col], errors="coerce")
        usable = usable.dropna(subset=required_cols).reset_index(drop=True)
        if usable.empty:
            return False
        self.historical_data = usable[required_cols].copy()
        self.benchmark_data = usable[["date", "close"]].rename(columns={"close": "benchmark_close"})
        return True

    def _visible_signal(self, visible_frame: pd.DataFrame) -> Dict[str, float]:
        if visible_frame.empty:
            return {
                "signal": 0.0,
                "participation_rate": 0.0,
                "urgency": 0.0,
                "slicing_rule": "twap_like",
                "time_horizon": 1,
            }

        last = visible_frame.iloc[-1]
        momentum = float(np.nan_to_num(last.get("momentum_20", 0.0), nan=0.0))
        alpha = float(np.nan_to_num(last.get("composite_alpha", 0.0), nan=0.0))
        sentiment = float(np.nan_to_num(last.get("sentiment_factor", 0.0), nan=0.0))
        policy = float(np.nan_to_num(last.get("policy_shock_factor", 0.0), nan=0.0))
        volatility = float(abs(np.nan_to_num(last.get("volatility_20", 0.0), nan=0.0)))
        trend = 0.0
        if len(visible_frame) >= 5:
            past_close = float(np.nan_to_num(visible_frame["close"].iloc[-5], nan=0.0))
            last_close = float(np.nan_to_num(visible_frame["close"].iloc[-1], nan=0.0))
            if past_close > 1e-12:
                trend = last_close / past_close - 1.0

        signal = 0.48 * alpha + 0.24 * momentum + 0.18 * sentiment + 0.10 * policy - 0.08 * volatility + 1.25 * trend
        signal = float(np.nan_to_num(signal, nan=0.0))
        signal = float(np.clip(signal, -self.config.max_position, self.config.max_position))
        urgency = float(np.clip(abs(signal) * 1.2, 0.0, 1.0))
        participation_rate = float(np.clip(0.04 + abs(signal) * 0.18, 0.0, 0.25))
        recent_volume_mean = float(np.nan_to_num(visible_frame["volume"].tail(10).mean(), nan=0.0))
        slicing_rule = "vwap_like" if float(last.get("volume", 0.0)) > recent_volume_mean else "twap_like"
        time_horizon = max(1, min(8, int(np.ceil(abs(signal) * 6.0)) or 1))
        return {
            "signal": signal,
            "participation_rate": participation_rate,
            "urgency": urgency,
            "slicing_rule": slicing_rule,
            "time_horizon": time_horizon,
        }

    @staticmethod
    def _build_intraday_path(row: pd.Series) -> List[float]:
        open_p = float(row.get("open", row.get("close", 0.0)))
        high_p = float(row.get("high", open_p))
        low_p = float(row.get("low", open_p))
        close_p = float(row.get("close", open_p))
        if close_p >= open_p:
            return [open_p, (open_p + high_p) / 2.0, high_p, (high_p + close_p) / 2.0, close_p]
        return [open_p, (open_p + low_p) / 2.0, low_p, (low_p + close_p) / 2.0, close_p]

    def _seed_liquidity(self, order_book: OrderBook, ref_price: float, slice_volume: float) -> None:
        spread = max(ref_price * 0.0015, 0.01)
        base_qty = max(50, int(slice_volume * 0.03))
        for level in range(2):
            qty = int(base_qty * (1.0 + 0.15 * level))
            ask = Order.create(
                agent_id=f"liquidity_ask_{level}",
                symbol=order_book.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=round(ref_price + spread * (level + 1), 2),
                quantity=qty,
            )
            bid = Order.create(
                agent_id=f"liquidity_bid_{level}",
                symbol=order_book.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=round(ref_price - spread * (level + 1), 2),
                quantity=qty,
            )
            order_book.add_order(ask)
            order_book.add_order(bid)

    @staticmethod
    def _trade_side(agent_id: str, trade) -> str:
        if trade.buyer_agent_id == agent_id:
            return "buy"
        if trade.seller_agent_id == agent_id:
            return "sell"
        return "unknown"

    def _apply_trade(self, trade, cash: float, shares: float, agent_id: str) -> tuple[float, float]:
        notional = float(trade.notional)
        if trade.buyer_agent_id == agent_id:
            cash -= notional + float(trade.buyer_fee)
            shares += float(trade.quantity)
        elif trade.seller_agent_id == agent_id:
            cash += notional - float(trade.seller_fee) - float(trade.seller_tax)
            shares -= float(trade.quantity)
        return cash, shares

    def _rolling_calibration_state(self, frame: pd.DataFrame, idx: int) -> Dict[str, Any]:
        enabled = bool(self.config.feature_flags.get(FEATURE_FLAG_ROLLING_CALIBRATION_V1, False))
        history = frame.iloc[:idx].dropna(subset=["close"]).reset_index(drop=True)
        if not enabled or len(history) < 8:
            return {
                "enabled": False,
                "signal_scale": 1.0,
                "train_size": 0,
                "validation_size": 0,
                "holdout_anchor": None,
                "direction_fit": 0.0,
                "volatility_scale": 1.0,
            }

        train_size = max(6, int(len(history) * 0.7))
        train_size = min(train_size, len(history) - 2)
        validation = history.iloc[train_size:].copy()
        train = history.iloc[:train_size].copy()
        if validation.empty or train.empty:
            return {
                "enabled": False,
                "signal_scale": 1.0,
                "train_size": int(len(train)),
                "validation_size": int(len(validation)),
                "holdout_anchor": None,
                "direction_fit": 0.0,
                "volatility_scale": 1.0,
            }

        train_ret = train["close"].pct_change().dropna()
        validation_ret = validation["close"].pct_change().dropna()
        train_trend = float(train["close"].iloc[-1] / max(float(train["close"].iloc[0]), 1e-12) - 1.0)
        validation_trend = float(validation["close"].iloc[-1] / max(float(validation["close"].iloc[0]), 1e-12) - 1.0)
        direction_fit = 1.0 if np.sign(train_trend) == np.sign(validation_trend) else 0.35
        train_vol = float(train_ret.std()) if not train_ret.empty else 0.0
        validation_vol = float(validation_ret.std()) if not validation_ret.empty else train_vol
        volatility_scale = float(np.clip(validation_vol / max(train_vol, 1e-12), 0.75, 1.35))
        signal_scale = float(np.clip(0.65 + 0.25 * direction_fit + 0.25 * volatility_scale, 0.55, 1.25))
        holdout_anchor = None
        if idx < len(frame):
            holdout_anchor = str(frame.iloc[idx].get("date", ""))

        return {
            "enabled": True,
            "signal_scale": signal_scale,
            "train_size": int(len(train)),
            "validation_size": int(len(validation)),
            "holdout_anchor": holdout_anchor,
            "direction_fit": direction_fit,
            "volatility_scale": volatility_scale,
            "train_window": {
                "start": str(train.iloc[0].get("date", "")),
                "end": str(train.iloc[-1].get("date", "")),
            },
            "validation_window": {
                "start": str(validation.iloc[0].get("date", "")),
                "end": str(validation.iloc[-1].get("date", "")),
            },
        }

    def _build_event_schedule(
        self,
        day_date: str,
        row: pd.Series,
        signal_info: Dict[str, float],
        rolling_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        open_p = float(row.get("open", row.get("close", 0.0)))
        high_p = float(row.get("high", open_p))
        low_p = float(row.get("low", open_p))
        close_p = float(row.get("close", open_p))
        signal = float(signal_info.get("signal", 0.0))
        urgency = float(signal_info.get("urgency", 0.0))
        policy_bias = float(np.nan_to_num(row.get("policy_shock_factor", self.config.policy_shock), nan=self.config.policy_shock))
        sentiment = float(np.nan_to_num(row.get("sentiment_factor", 0.0), nan=0.0))
        regime_scale = float(rolling_state.get("signal_scale", 1.0)) if rolling_state.get("enabled") else 1.0
        impulse_anchor = high_p if signal + policy_bias + sentiment >= 0 else low_p
        liquidity_anchor = float(np.mean([open_p, high_p, low_p, close_p]))
        close_anchor = float(np.mean([close_p, impulse_anchor]))
        raw_schedule = [
            {
                "event": "opening_auction",
                "timestamp": f"{day_date}T09:25:00",
                "anchor_price": open_p,
                "weight": 0.20 + 0.05 * urgency,
                "urgency_multiplier": 1.10,
                "prefer_market": urgency >= 0.75,
                "crossing_buffer": 0.0040,
            },
            {
                "event": "policy_impulse",
                "timestamp": f"{day_date}T10:05:00",
                "anchor_price": impulse_anchor,
                "weight": 0.24 + 0.06 * abs(policy_bias) + 0.05 * abs(signal),
                "urgency_multiplier": 1.15 + 0.10 * abs(policy_bias),
                "prefer_market": abs(policy_bias) >= 0.15,
                "crossing_buffer": 0.0045,
            },
            {
                "event": "midday_liquidity",
                "timestamp": f"{day_date}T13:15:00",
                "anchor_price": liquidity_anchor,
                "weight": 0.30 + 0.04 * regime_scale,
                "urgency_multiplier": 0.90,
                "prefer_market": False,
                "crossing_buffer": 0.0035,
            },
            {
                "event": "closing_rebalance",
                "timestamp": f"{day_date}T14:55:00",
                "anchor_price": close_anchor,
                "weight": 0.26 + 0.05 * urgency,
                "urgency_multiplier": 1.05,
                "prefer_market": urgency >= 0.60,
                "crossing_buffer": 0.0050,
            },
        ]
        total_weight = sum(max(float(item["weight"]), 0.01) for item in raw_schedule)
        for item in raw_schedule:
            item["weight"] = float(max(float(item["weight"]), 0.01) / max(total_weight, 1e-12))
        return raw_schedule

    def run_backtest(
        self,
        population: Any = None,
        market_manager: Any = None,
        progress_callback=None,
        step_callback=None,
        civitas_factors: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        del market_manager
        del population
        _seed_everything(self.config.random_seed)

        if self.historical_data.empty:
            loaded_from_event_store = self._load_from_event_store_if_enabled()
            if not loaded_from_event_store and not self.load_data(progress_callback):
                self.result = BacktestResult(strategy_name=self.config.strategy_name)
                return self.result

        frame = self._prepare_frames(civitas_factors)
        if frame.empty:
            self.result = BacktestResult(strategy_name=self.config.strategy_name)
            return self.result

        frame = frame.reset_index(drop=True)
        valid_frame = frame.dropna(subset=["open", "high", "low", "close", "benchmark_close"]).reset_index(drop=True)
        if valid_frame.empty:
            self.result = BacktestResult(strategy_name=self.config.strategy_name)
            return self.result

        self.cost_model = TransactionCostModel(
            commission_rate=self.config.commission_rate,
            stamp_duty_rate=self.config.stamp_duty_rate,
        )
        self.slippage_model = SlippageModel(
            slippage_bps=self.config.slippage_bps,
            market_impact=self.config.market_impact,
        )

        cash = float(self.config.initial_cash)
        shares = 0.0
        prev_close = float(valid_frame.loc[0, "close"])
        first_bench = float(valid_frame.loc[0, "benchmark_close"])

        equity_curve: List[float] = []
        benchmark_curve: List[float] = []
        returns: List[float] = []
        benchmark_returns: List[float] = []
        drawdowns: List[float] = []
        turnover_series: List[float] = []
        position_series: List[float] = []
        real_prices: List[float] = []
        simulated_prices: List[float] = []
        dates: List[str] = []
        trade_log: List[Dict[str, Any]] = []
        total_cost = 0.0
        total_trades = 0
        total_volume = 0.0
        simulated_bars: List[Dict[str, Any]] = []
        trade_tape: List[Dict[str, Any]] = []
        calibration_windows: List[Dict[str, Any]] = []
        event_count_total = 0
        replay_variant = "event_driven_v2" if self.config.feature_flags.get(FEATURE_FLAG_EVENT_DRIVEN_V2, False) else "legacy_v1"

        total_days = len(valid_frame)
        agent_id = f"agent-replay-{self.config.random_seed}"
        volume_baseline = float(np.nan_to_num(valid_frame["volume"].tail(20).mean(), nan=0.0))

        self.is_running = True
        for idx, row in valid_frame.iterrows():
            if not self.is_running:
                break

            day_date = str(row.get("date", ""))
            day_open = float(row.get("open", prev_close))
            day_high = float(row.get("high", day_open))
            day_low = float(row.get("low", day_open))
            day_close = float(row.get("close", day_open))
            day_volume = float(row.get("volume", 0.0))

            visible_frame = frame.iloc[:idx].dropna(subset=["close"]).reset_index(drop=True)
            signal_info = self._visible_signal(visible_frame)
            signal = float(signal_info["signal"])
            urgency = float(signal_info["urgency"])
            participation_rate = float(signal_info["participation_rate"])
            slicing_rule = str(signal_info["slicing_rule"])
            time_horizon = int(signal_info["time_horizon"])
            calibration_state = self._rolling_calibration_state(frame, idx)
            if calibration_state["enabled"]:
                signal = float(np.clip(signal * calibration_state["signal_scale"], -self.config.max_position, self.config.max_position))
                signal = float(np.nan_to_num(signal, nan=0.0, posinf=self.config.max_position, neginf=-self.config.max_position))
                signal_info["signal"] = signal
                calibration_windows.append(calibration_state)

            portfolio_value = cash + shares * prev_close
            current_weight = (shares * prev_close / portfolio_value) if portfolio_value > 1e-12 else 0.0
            desired_weight = float(np.clip(signal, -self.config.max_position, self.config.max_position))
            if not self.config.allow_short:
                desired_weight = max(0.0, desired_weight)

            target_value = desired_weight * portfolio_value
            target_qty = (target_value - shares * prev_close) / max(day_open, 1e-12)
            if target_qty > 0:
                max_affordable = cash / max(day_open * (1.0 + self.config.commission_rate), 1e-12)
                target_qty = min(target_qty, max_affordable)
            if not self.config.allow_short:
                target_qty = max(target_qty, -shares)

            if abs(target_qty) < 1e-8:
                bar_open = day_open
                bar_high = max(day_open, day_close)
                bar_low = min(day_open, day_close)
                bar_close = day_close
                bar_volume = 0
                simulated_bars.append(
                    {
                        "date": day_date,
                        "open": bar_open,
                        "high": bar_high,
                        "low": bar_low,
                        "close": bar_close,
                        "volume": bar_volume,
                        "source": "no_trade",
                    }
                )
                cash_value = cash + shares * bar_close
                equity_curve.append(cash_value)
                benchmark_nav = self.config.initial_cash * (day_close / max(first_bench, 1e-12))
                benchmark_curve.append(benchmark_nav)
                real_prices.append(day_close)
                simulated_prices.append(bar_close)
                dates.append(day_date)
                position_series.append((shares * bar_close / cash_value) if cash_value > 1e-12 else 0.0)
                turnover_series.append(0.0)
                returns.append(0.0 if len(equity_curve) == 1 else equity_curve[-1] / max(equity_curve[-2], 1e-12) - 1.0)
                benchmark_returns.append(0.0 if len(benchmark_curve) == 1 else benchmark_curve[-1] / max(benchmark_curve[-2], 1e-12) - 1.0)
                drawdowns.append(cash_value / max(max(equity_curve), 1e-12) - 1.0)
                prev_close = bar_close
                if progress_callback:
                    progress_callback(idx + 1, total_days, f"agent replay: {day_date}")
                continue

            order_book = OrderBook(symbol=self.config.symbol, prev_close=prev_close)
            event_schedule = self._build_event_schedule(day_date, row, signal_info, calibration_state) if replay_variant == "event_driven_v2" else []
            slices = len(event_schedule) if event_schedule else max(
                1,
                min(6, int(np.ceil(abs(target_qty) / max(participation_rate * max(day_volume, 1.0), 100.0)))),
            )
            path = self._build_intraday_path(row)
            remaining_qty = float(target_qty)
            day_trades = []
            day_turnover_value = 0.0

            for slice_idx in range(slices):
                event_info = event_schedule[slice_idx] if slice_idx < len(event_schedule) else None
                slice_price = float(event_info["anchor_price"]) if event_info else float(path[min(slice_idx, len(path) - 1)])
                self._seed_liquidity(order_book, slice_price, day_volume / max(len(path), 1))
                if event_info:
                    remaining_weight = sum(float(item["weight"]) for item in event_schedule[slice_idx:]) or 1.0
                    child_qty = remaining_qty * float(event_info["weight"]) / remaining_weight
                    child_urgency = float(np.clip(urgency * float(event_info["urgency_multiplier"]), 0.0, 1.0))
                    crossing_buffer = max(self.config.slippage_bps / 10000.0 + float(event_info["crossing_buffer"]), 0.003)
                    event_name = str(event_info["event"])
                    event_timestamp = str(event_info["timestamp"])
                else:
                    child_qty = remaining_qty / max(slices - slice_idx, 1)
                    child_urgency = urgency
                    crossing_buffer = max(self.config.slippage_bps / 10000.0 + 0.002, 0.003)
                    event_name = "slice"
                    event_timestamp = f"{day_date}T{9 + min(slice_idx, 5):02d}:30:00"
                child_qty = float(np.nan_to_num(child_qty, nan=0.0, posinf=0.0, neginf=0.0))
                child_side = OrderSide.BUY if child_qty > 0 else OrderSide.SELL
                child_qty_abs = max(1, int(round(abs(child_qty))))
                order_type = OrderType.MARKET if child_urgency >= 0.7 or bool(event_info and event_info.get("prefer_market")) else OrderType.LIMIT
                lower_limit, upper_limit = order_book.get_limit_prices()
                if child_side == OrderSide.BUY:
                    child_price = upper_limit if order_type == OrderType.MARKET else min(upper_limit, round(slice_price * (1.0 + crossing_buffer), 2))
                else:
                    child_price = lower_limit if order_type == OrderType.MARKET else max(lower_limit, round(slice_price * (1.0 - crossing_buffer), 2))

                order = Order.create(
                    agent_id=agent_id,
                    symbol=self.config.symbol,
                    side=child_side,
                    order_type=order_type,
                    price=child_price,
                    quantity=child_qty_abs,
                )
                trades = order_book.add_order(order)
                event_count_total += 1

                filled_qty = sum(int(trade.quantity) for trade in trades)
                remaining_qty -= filled_qty if child_side == OrderSide.BUY else -filled_qty

                if order.remaining_qty > 0:
                    order_book.cancel_order(order.order_id)

                for trade in trades:
                    cash, shares = self._apply_trade(trade, cash, shares, agent_id)
                    total_cost += float(trade.buyer_fee if trade.buyer_agent_id == agent_id else trade.seller_fee + trade.seller_tax)
                    total_trades += 1
                    total_volume += float(trade.quantity)
                    day_turnover_value += float(trade.notional)
                    day_trades.append(trade)
                    trade_tape.append(
                        {
                            "date": day_date,
                            "timestamp": event_timestamp,
                            "slice": slice_idx,
                            "event": event_name,
                            "price": float(trade.price),
                            "quantity": int(trade.quantity),
                            "side": self._trade_side(agent_id, trade),
                            "maker_id": trade.maker_id,
                            "taker_id": trade.taker_id,
                            "trade_id": trade.trade_id,
                        }
                    )
                    trade_log.append(
                        {
                            "date": day_date,
                            "timestamp": event_timestamp,
                            "slice": slice_idx,
                            "event": event_name,
                            "side": self._trade_side(agent_id, trade),
                            "qty": int(trade.quantity),
                            "price": float(trade.price),
                            "notional": float(trade.notional),
                            "order_type": order_type.value,
                            "urgency": child_urgency,
                            "participation_rate": participation_rate,
                            "slicing_rule": slicing_rule,
                            "time_horizon": time_horizon,
                            "calibration_signal_scale": float(calibration_state.get("signal_scale", 1.0)),
                        }
                    )

            trade_prices = [float(t.price) for t in day_trades]
            trade_qty = int(sum(t.quantity for t in day_trades))
            if trade_prices:
                bar_open = trade_prices[0]
                bar_high = max(trade_prices)
                bar_low = min(trade_prices)
                bar_close = trade_prices[-1]
            else:
                bar_open = day_open
                bar_high = max(day_open, day_close)
                bar_low = min(day_open, day_close)
                bar_close = day_close

            simulated_bars.append(
                {
                    "date": day_date,
                    "open": bar_open,
                    "high": bar_high,
                        "low": bar_low,
                        "close": bar_close,
                        "volume": trade_qty,
                        "source": "event_trade_tape" if replay_variant == "event_driven_v2" else "trade_tape",
                    }
                )

            prev_close = bar_close
            value = cash + shares * bar_close
            equity_curve.append(value)
            benchmark_nav = self.config.initial_cash * (day_close / max(first_bench, 1e-12))
            benchmark_curve.append(benchmark_nav)
            real_prices.append(day_close)
            simulated_prices.append(bar_close)
            dates.append(day_date)
            position_series.append((shares * bar_close / value) if value > 1e-12 else 0.0)
            turnover_series.append(day_turnover_value / max(portfolio_value, 1e-12))
            returns.append(0.0 if len(equity_curve) == 1 else equity_curve[-1] / max(equity_curve[-2], 1e-12) - 1.0)
            benchmark_returns.append(0.0 if len(benchmark_curve) == 1 else benchmark_curve[-1] / max(benchmark_curve[-2], 1e-12) - 1.0)
            drawdowns.append(value / max(max(equity_curve), 1e-12) - 1.0)

            if step_callback:
                step_callback(
                    idx,
                    {
                        "date": day_date,
                        "real_price": day_close,
                        "simulated_price": bar_close,
                        "portfolio": value,
                        "turnover": turnover_series[-1],
                        "trade_count": len(day_trades),
                    },
                )
            if progress_callback:
                progress_callback(idx + 1, total_days, f"agent replay: {day_date}")

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
        result.simulated_bars = simulated_bars

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
        calibration_summary = {
            "enabled": bool(self.config.feature_flags.get(FEATURE_FLAG_ROLLING_CALIBRATION_V1, False)),
            "window_count": len(calibration_windows),
            "latest_signal_scale": float(calibration_windows[-1]["signal_scale"]) if calibration_windows else 1.0,
            "latest_holdout_anchor": calibration_windows[-1]["holdout_anchor"] if calibration_windows else None,
        }
        if calibration_windows:
            calibration_summary["avg_signal_scale"] = float(
                np.mean([float(item["signal_scale"]) for item in calibration_windows])
            )
            calibration_summary["latest_window"] = {
                "train": calibration_windows[-1]["train_window"],
                "validation": calibration_windows[-1]["validation_window"],
            }
        result.metadata = {
            "symbol": self.config.symbol,
            "benchmark_symbol": self.config.benchmark_symbol,
            "agent_count": 1,
            "credibility_score": result.credibility_score,
            "trade_tape_rows": len(trade_tape),
            "simulated_bars": len(simulated_bars),
            "mode": "agent",
            "replay_variant": replay_variant,
            "rolling_calibration": calibration_summary,
            "event_schedule": {
                "mode": replay_variant,
                "events_per_day": 4 if replay_variant == "event_driven_v2" else None,
                "total_events": event_count_total,
            },
            "reference_bars": valid_frame[["date", "open", "high", "low", "close", "volume"]].to_dict("records"),
            "window": {
                "start": dates[0] if dates else None,
                "end": dates[-1] if dates else None,
            },
            "replay_params": {
                "slicing_rule": "mixed_twap_vwap",
                "volume_baseline": volume_baseline,
            },
            "event_store": {
                "enabled": bool(self.config.feature_flags.get("event_store_v1", False)),
                "dataset_version": self.dataset_version,
                "snapshot_id": self.snapshot_id,
                "scenario_id": self.scenario_id,
            },
        }
        result.metadata.update(
            _build_repro_metadata(
                self.config,
                self.historical_data,
                mode="agent",
                extra={
                    "simulated_price_source": "trade_tape_close",
                    "replay_variant": replay_variant,
                    "rolling_calibration_enabled": bool(self.config.feature_flags.get(FEATURE_FLAG_ROLLING_CALIBRATION_V1, False)),
                    "trade_tape_sha256": hashlib.sha256(
                        json.dumps(trade_tape, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest(),
                },
            )
        )
        self.result = result
        return result
