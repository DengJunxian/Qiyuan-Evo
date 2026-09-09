"""Trade tape as the canonical source for prices, bars, and replay metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.types import Candle, Trade


def stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deterministic_trade_id(
    *,
    seed: int,
    sequence: int,
    symbol: str,
    price: float,
    volume: int,
    buy_order_id: str,
    sell_order_id: str,
    timestamp: float,
    phase: str,
) -> str:
    digest = stable_hash(
        {
            "seed": int(seed),
            "sequence": int(sequence),
            "symbol": str(symbol),
            "price": round(float(price), 8),
            "volume": int(volume),
            "buy_order_id": str(buy_order_id),
            "sell_order_id": str(sell_order_id),
            "timestamp": round(float(timestamp), 6),
            "phase": str(phase),
        }
    )
    return f"tt_{digest[:24]}"


def _phase_from_metadata(phase: str, metadata: Mapping[str, Any]) -> str:
    raw = str(phase or metadata.get("phase") or metadata.get("market_phase") or "continuous")
    if raw == "call_auction":
        return "open_call"
    return raw


@dataclass(frozen=True)
class TradeTapeRecord:
    """Canonical execution record.

    The underlying matching engine may use implementation-specific trade ids.
    This record owns the deterministic replay id and all context needed to
    rebuild OHLCV and microstructure metrics.
    """

    trade_id: str
    timestamp: float
    tick: int
    trading_day: str
    symbol: str
    price: float
    volume: int
    buy_order_id: str
    sell_order_id: str
    buy_agent_id: str = ""
    sell_agent_id: str = ""
    aggressor_side: str = "unknown"
    phase: str = "continuous"
    seed: int = 42
    config_hash: str = ""
    data_snapshot_hash: str = ""
    queue_position: int = 0
    latency_ticks: int = 0
    spread: float = 0.0
    depth_imbalance: float = 0.0
    cancel_to_trade_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def amount(self) -> float:
        return float(self.price) * int(self.volume)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_trade(
        cls,
        trade: Trade,
        *,
        symbol: str,
        tick: int,
        trading_day: str,
        phase: str,
        seed: int,
        sequence: int,
        config_hash: str = "",
        data_snapshot_hash: str = "",
        queue_position: int = 0,
        latency_ticks: int = 0,
        market_timestamp: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TradeTapeRecord":
        meta = dict(metadata or {})
        buyer = str(getattr(trade, "buyer_agent_id", "") or "")
        seller = str(getattr(trade, "seller_agent_id", "") or "")
        buy_order_id = str(meta.get("buy_order_id") or "")
        sell_order_id = str(meta.get("sell_order_id") or "")
        if not buy_order_id or not sell_order_id:
            maker_id = str(getattr(trade, "maker_id", ""))
            taker_id = str(getattr(trade, "taker_id", ""))
            maker_agent = str(getattr(trade, "maker_agent_id", ""))
            if maker_agent and maker_agent == buyer:
                buy_order_id = buy_order_id or maker_id
                sell_order_id = sell_order_id or taker_id
            elif maker_agent and maker_agent == seller:
                buy_order_id = buy_order_id or taker_id
                sell_order_id = sell_order_id or maker_id
            else:
                buy_order_id = buy_order_id or str(meta.get("buyer_order_id") or taker_id)
                sell_order_id = sell_order_id or str(meta.get("seller_order_id") or maker_id)
        ts = float(market_timestamp if market_timestamp is not None else getattr(trade, "timestamp", 0.0))
        normalized_phase = _phase_from_metadata(phase, meta)
        aggressor = str(meta.get("aggressor_side") or "unknown")
        trade_id = deterministic_trade_id(
            seed=seed,
            sequence=sequence,
            symbol=symbol,
            price=float(trade.price),
            volume=int(trade.quantity),
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            timestamp=ts,
            phase=normalized_phase,
        )
        return cls(
            trade_id=trade_id,
            timestamp=ts,
            tick=int(tick),
            trading_day=str(trading_day),
            symbol=str(symbol),
            price=float(trade.price),
            volume=int(trade.quantity),
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            buy_agent_id=buyer,
            sell_agent_id=seller,
            aggressor_side=aggressor,
            phase=normalized_phase,
            seed=int(seed),
            config_hash=str(config_hash),
            data_snapshot_hash=str(data_snapshot_hash),
            queue_position=int(queue_position),
            latency_ticks=int(latency_ticks),
            spread=float(meta.get("spread", 0.0) or 0.0),
            depth_imbalance=float(meta.get("depth_imbalance", 0.0) or 0.0),
            cancel_to_trade_ratio=float(meta.get("cancel_to_trade_ratio", 0.0) or 0.0),
            metadata=meta,
        )


def _record_from_any(item: Any, *, default_symbol: str = "", seed: int = 42, sequence: int = 0) -> TradeTapeRecord:
    if isinstance(item, TradeTapeRecord):
        return item
    if isinstance(item, Mapping):
        payload = dict(item)
        if "volume" not in payload and "quantity" in payload:
            payload["volume"] = payload["quantity"]
        if "timestamp" not in payload and "market_timestamp" in payload:
            payload["timestamp"] = payload["market_timestamp"]
        payload.setdefault("trade_id", deterministic_trade_id(
            seed=int(payload.get("seed", seed)),
            sequence=int(payload.get("sequence", sequence)),
            symbol=str(payload.get("symbol", default_symbol)),
            price=float(payload.get("price", 0.0) or 0.0),
            volume=int(payload.get("volume", 0) or 0),
            buy_order_id=str(payload.get("buy_order_id", "")),
            sell_order_id=str(payload.get("sell_order_id", "")),
            timestamp=float(payload.get("timestamp", 0.0) or 0.0),
            phase=str(payload.get("phase", "continuous")),
        ))
        payload.setdefault("tick", int(sequence))
        payload.setdefault("trading_day", datetime.fromtimestamp(float(payload.get("timestamp", 0.0) or 0.0)).date().isoformat())
        payload.setdefault("symbol", default_symbol)
        payload.setdefault("buy_order_id", "")
        payload.setdefault("sell_order_id", "")
        payload.setdefault("volume", 0)
        return TradeTapeRecord(**{k: v for k, v in payload.items() if k in TradeTapeRecord.__dataclass_fields__})
    trade = getattr(item, "trade", item)
    metadata = dict(getattr(item, "metadata", {}) or {})
    timestamp = float(getattr(item, "market_timestamp", 0.0) or getattr(trade, "timestamp", 0.0) or 0.0)
    trading_day = str(metadata.get("trading_day") or datetime.fromtimestamp(timestamp).date().isoformat())
    return TradeTapeRecord.from_trade(
        trade,
        symbol=str(metadata.get("symbol") or default_symbol or getattr(trade, "symbol", "")),
        tick=int(getattr(item, "tick", sequence) or sequence),
        trading_day=trading_day,
        phase=str(getattr(item, "phase", metadata.get("phase", "continuous"))),
        seed=int(metadata.get("seed", seed) or seed),
        sequence=sequence,
        config_hash=str(metadata.get("config_hash", "")),
        data_snapshot_hash=str(metadata.get("data_snapshot_hash", "")),
        queue_position=int(getattr(item, "queue_position", 0) or 0),
        latency_ticks=int(getattr(item, "latency_ticks", 0) or 0),
        market_timestamp=timestamp,
        metadata=metadata,
    )


class TradeTape:
    """Append-only canonical execution tape."""

    def __init__(
        self,
        *,
        symbol: str,
        seed: int = 42,
        config_hash: str = "",
        data_snapshot_hash: str = "",
    ) -> None:
        self.symbol = str(symbol)
        self.seed = int(seed)
        self.config_hash = str(config_hash)
        self.data_snapshot_hash = str(data_snapshot_hash)
        self._records: List[TradeTapeRecord] = []

    def append(self, record: TradeTapeRecord) -> TradeTapeRecord:
        self._records.append(record)
        return record

    def append_trade(
        self,
        trade: Trade,
        *,
        tick: int,
        trading_day: str,
        phase: str,
        queue_position: int = 0,
        latency_ticks: int = 0,
        market_timestamp: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TradeTapeRecord:
        record = TradeTapeRecord.from_trade(
            trade,
            symbol=self.symbol,
            tick=int(tick),
            trading_day=trading_day,
            phase=phase,
            seed=self.seed,
            sequence=len(self._records),
            config_hash=self.config_hash,
            data_snapshot_hash=self.data_snapshot_hash,
            queue_position=queue_position,
            latency_ticks=latency_ticks,
            market_timestamp=market_timestamp,
            metadata=metadata,
        )
        return self.append(record)

    def extend_from_trades(self, trades: Iterable[Trade], **kwargs: Any) -> List[TradeTapeRecord]:
        return [self.append_trade(trade, **kwargs) for trade in trades]

    @property
    def records(self) -> List[TradeTapeRecord]:
        return list(self._records)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def hash(self) -> str:
        return stable_hash({"records": self.to_dicts()})

    def last_price(self, fallback: float = 0.0) -> float:
        return float(self._records[-1].price) if self._records else float(fallback)

    def aggregate(self, freq: str, *, prev_close: float = 0.0) -> List[Candle]:
        return aggregate_trade_tape_to_bars(self._records, freq=freq, symbol=self.symbol, prev_close=prev_close)


def _bucket_key(record: TradeTapeRecord, freq: str) -> str:
    ts = datetime.fromtimestamp(float(record.timestamp))
    normalized = str(freq or "1m").lower()
    if normalized in {"1d", "d", "day", "daily"}:
        return record.trading_day or ts.date().isoformat()
    if normalized in {"5m", "5min"}:
        minute = (ts.minute // 5) * 5
        return ts.replace(minute=minute, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    if normalized in {"1m", "m", "minute"}:
        return ts.replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    raise ValueError(f"unsupported tape bar frequency: {freq}")


def aggregate_trade_tape_to_bars(
    trade_tape: Sequence[Any],
    *,
    freq: str,
    symbol: str,
    prev_close: float = 0.0,
    is_simulated: bool = True,
) -> List[Candle]:
    """Aggregate OHLCV from tape only.

    Empty buckets are intentionally not emitted; consumers that need visual
    continuity must fill gaps outside evaluation paths and mark that output as
    visualization-only.
    """

    records = [_record_from_any(item, default_symbol=symbol, sequence=idx) for idx, item in enumerate(trade_tape)]
    groups: Dict[str, List[TradeTapeRecord]] = {}
    for record in records:
        groups.setdefault(_bucket_key(record, freq), []).append(record)

    bars: List[Candle] = []
    for step, key in enumerate(sorted(groups.keys())):
        rows = sorted(groups[key], key=lambda record: (record.timestamp, record.tick, record.trade_id))
        prices = [float(row.price) for row in rows]
        volumes = [int(row.volume) for row in rows]
        amount = float(sum(row.amount for row in rows))
        bar = Candle(
            symbol=str(symbol),
            step=int(step),
            timestamp=str(key),
            open=float(prices[0]),
            high=float(max(prices)),
            low=float(min(prices)),
            close=float(prices[-1]),
            volume=int(sum(volumes)),
            amount=amount,
            is_simulated=bool(is_simulated),
        )
        setattr(
            bar,
            "metadata",
            {
                "source": "trade_tape",
                "freq": str(freq),
                "trade_count": int(len(rows)),
                "tape_hash": stable_hash({"records": [row.to_dict() for row in rows]}),
                "prev_close": float(prev_close),
            },
        )
        bars.append(bar)
    return bars


__all__ = [
    "TradeTape",
    "TradeTapeRecord",
    "aggregate_trade_tape_to_bars",
    "deterministic_trade_id",
    "stable_hash",
]
