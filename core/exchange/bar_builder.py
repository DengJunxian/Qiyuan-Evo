"""Trade tape aggregation helpers for OHLCV bars and replay metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.exchange.trade_tape import TradeTapeRecord, aggregate_trade_tape_to_bars
from core.types import Candle, Trade


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class TradeTapeEntry:
    """Structured trade-tape record with execution context."""

    trade: Trade
    tick: int = 0
    phase: str = "continuous"
    event_type: str = "trade"
    queue_position: int = 0
    latency_ticks: int = 0
    market_timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def price(self) -> float:
        return float(self.trade.price)

    @property
    def quantity(self) -> int:
        return int(self.trade.quantity)

    @property
    def timestamp(self) -> float:
        return float(self.market_timestamp if self.market_timestamp else self.trade.timestamp)

    @property
    def buyer_fee(self) -> float:
        return float(self.trade.buyer_fee)

    @property
    def seller_fee(self) -> float:
        return float(self.trade.seller_fee)

    @property
    def seller_tax(self) -> float:
        return float(self.trade.seller_tax)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "trade": asdict(self.trade),
            "tick": int(self.tick),
            "phase": self.phase,
            "event_type": self.event_type,
            "queue_position": int(self.queue_position),
            "latency_ticks": int(self.latency_ticks),
            "market_timestamp": float(self.market_timestamp),
            "metadata": dict(self.metadata),
        }
        return out


def _as_trade(entry: Trade | TradeTapeEntry | TradeTapeRecord) -> Trade:
    if isinstance(entry, TradeTapeRecord):
        return Trade(
            trade_id=entry.trade_id,
            price=float(entry.price),
            quantity=int(entry.volume),
            maker_id=str(entry.metadata.get("maker_id", entry.sell_order_id)),
            taker_id=str(entry.metadata.get("taker_id", entry.buy_order_id)),
            maker_agent_id=str(entry.metadata.get("maker_agent_id", entry.sell_agent_id)),
            taker_agent_id=str(entry.metadata.get("taker_agent_id", entry.buy_agent_id)),
            buyer_agent_id=str(entry.buy_agent_id),
            seller_agent_id=str(entry.sell_agent_id),
            timestamp=float(entry.timestamp),
        )
    return entry.trade if isinstance(entry, TradeTapeEntry) else entry


def _as_entry(entry: Trade | TradeTapeEntry | TradeTapeRecord) -> TradeTapeEntry:
    if isinstance(entry, TradeTapeEntry):
        return entry
    if isinstance(entry, TradeTapeRecord):
        return TradeTapeEntry(
            trade=_as_trade(entry),
            tick=int(entry.tick),
            phase=str(entry.phase),
            event_type="trade",
            queue_position=int(entry.queue_position),
            latency_ticks=int(entry.latency_ticks),
            market_timestamp=float(entry.timestamp),
            metadata={
                "symbol": entry.symbol,
                "trading_day": entry.trading_day,
                "seed": int(entry.seed),
                "config_hash": entry.config_hash,
                "data_snapshot_hash": entry.data_snapshot_hash,
                **dict(entry.metadata or {}),
            },
        )
    return TradeTapeEntry(trade=entry, market_timestamp=float(entry.timestamp), metadata={})


@dataclass
class TradeTapeBarBuilder:
    """Convert trade tape into OHLCV bars and summary metrics."""

    seed: int = 42
    config_hash: str = ""
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    snapshot_info: Dict[str, Any] = field(default_factory=dict)
    bar_interval_seconds: int = 60

    def _group_key(
        self,
        entry: TradeTapeEntry,
        *,
        bar_key_fn: Optional[Callable[[TradeTapeEntry, int], Any]] = None,
        index: int = 0,
    ) -> Any:
        if bar_key_fn is not None:
            return bar_key_fn(entry, index)
        metadata_key = entry.metadata.get("bar_key")
        if metadata_key is not None:
            return metadata_key
        if entry.tick:
            return int(entry.tick)
        return int(max(0.0, entry.timestamp) // max(1, self.bar_interval_seconds))

    def build_bar(
        self,
        trade_tape: Sequence[Trade | TradeTapeEntry | TradeTapeRecord],
        *,
        symbol: str,
        step: int,
        timestamp: str,
        prev_close: float,
        open_price: Optional[float] = None,
        is_simulated: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Candle:
        entries = [_as_entry(entry) for entry in trade_tape]
        if not entries:
            price = float(max(1e-6, prev_close if prev_close > 0 else 1.0))
            return Candle(
                symbol=symbol,
                step=int(step),
                timestamp=str(timestamp),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,
                amount=0.0,
                is_simulated=is_simulated,
            )

        trades = [_as_trade(entry) for entry in entries]
        prices = [float(trade.price) for trade in trades]
        qtys = [int(trade.quantity) for trade in trades]
        amounts = [float(trade.price * trade.quantity) for trade in trades]
        open_v = float(open_price if open_price is not None else prices[0])
        high_v = float(max(prices))
        low_v = float(min(prices))
        close_v = float(prices[-1])
        volume = int(sum(qtys))
        amount = float(sum(amounts))

        bar = Candle(
            symbol=symbol,
            step=int(step),
            timestamp=str(timestamp),
            open=open_v,
            high=high_v,
            low=low_v,
            close=close_v,
            volume=volume,
            amount=amount,
            is_simulated=is_simulated,
        )
        if extra_metadata:
            setattr(bar, "metadata", dict(extra_metadata))
        return bar

    def build_bars_from_trade_tape(
        self,
        trade_tape: Sequence[Trade | TradeTapeEntry | TradeTapeRecord],
        *,
        symbol: str,
        prev_close: float,
        bar_key_fn: Optional[Callable[[TradeTapeEntry, int], Any]] = None,
        is_simulated: bool = True,
    ) -> List[Candle]:
        groups: Dict[Any, List[TradeTapeEntry]] = {}
        for index, item in enumerate(trade_tape):
            entry = _as_entry(item)
            key = self._group_key(entry, bar_key_fn=bar_key_fn, index=index)
            groups.setdefault(key, []).append(entry)

        bars: List[Candle] = []
        running_prev_close = float(prev_close)
        def _sort_key(value: Any) -> tuple[int, Any]:
            if isinstance(value, (int, float)):
                return (0, float(value))
            return (1, str(value))

        for step, key in enumerate(sorted(groups.keys(), key=_sort_key)):
            entries = groups[key]
            bar = self.build_bar(
                entries,
                symbol=symbol,
                step=step,
                timestamp=str(key),
                prev_close=running_prev_close,
                open_price=entries[0].price if entries else running_prev_close,
                is_simulated=is_simulated,
                extra_metadata={
                    "bar_key": key,
                    "trade_count": len(entries),
                },
            )
            bars.append(bar)
            running_prev_close = float(bar.close)
        return bars

    def build_bars_from_canonical_tape(
        self,
        trade_tape: Sequence[TradeTapeRecord],
        *,
        symbol: str,
        prev_close: float,
        freq: str = "1m",
        is_simulated: bool = True,
    ) -> List[Candle]:
        return aggregate_trade_tape_to_bars(
            trade_tape,
            freq=freq,
            symbol=symbol,
            prev_close=prev_close,
            is_simulated=is_simulated,
        )

    def build_replay_metrics(
        self,
        trade_tape: Sequence[Trade | TradeTapeEntry | TradeTapeRecord],
        bars: Sequence[Candle],
    ) -> Dict[str, Any]:
        entries = [_as_entry(item) for item in trade_tape]
        trades = [_as_trade(item) for item in entries]
        total_volume = int(sum(trade.quantity for trade in trades))
        total_amount = float(sum(trade.price * trade.quantity for trade in trades))
        closes = np.asarray([float(bar.close) for bar in bars], dtype=float)
        opens = np.asarray([float(bar.open) for bar in bars], dtype=float)
        high_low_ranges = np.asarray([float(bar.high - bar.low) for bar in bars], dtype=float)
        queue_positions = np.asarray([float(entry.queue_position) for entry in entries], dtype=float)
        latencies = np.asarray([float(entry.latency_ticks) for entry in entries], dtype=float)
        fee_total = float(sum(trade.buyer_fee + trade.seller_fee for trade in trades))
        tax_total = float(sum(trade.seller_tax for trade in trades))

        return {
            "trade_count": int(len(trades)),
            "bar_count": int(len(bars)),
            "total_volume": total_volume,
            "total_amount": total_amount,
            "vwap": float(total_amount / max(total_volume, 1)),
            "open_close_return": float((closes[-1] - opens[0]) / max(opens[0], 1e-12)) if closes.size and opens.size else 0.0,
            "realized_volatility": float(np.std(np.diff(closes) / np.maximum(closes[:-1], 1e-12))) if closes.size > 1 else 0.0,
            "average_range": float(np.mean(high_low_ranges)) if high_low_ranges.size else 0.0,
            "average_queue_position": float(np.mean(queue_positions)) if queue_positions.size else 0.0,
            "average_latency_ticks": float(np.mean(latencies)) if latencies.size else 0.0,
            "fee_total": fee_total,
            "tax_total": tax_total,
            "config_hash": self.config_hash or _stable_hash(
                {
                    "seed": self.seed,
                    "feature_flags": self.feature_flags,
                    "snapshot_info": self.snapshot_info,
                }
            ),
            "seed": int(self.seed),
            "snapshot_info": dict(self.snapshot_info),
            "feature_flags": dict(self.feature_flags),
        }


__all__ = ["TradeTapeEntry", "TradeTapeBarBuilder"]
