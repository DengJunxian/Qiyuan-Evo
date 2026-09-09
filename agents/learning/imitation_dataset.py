"""Imitation dataset builders from historical order-flow or bar proxies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ImitationSample:
    state: Dict[str, float]
    action: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImitationDataset:
    """Small data container for behavior cloning smoke tests."""

    def __init__(self, samples: Sequence[ImitationSample] | None = None, *, source: str = "") -> None:
        self.samples = list(samples or [])
        self.source = str(source or "unknown")

    @classmethod
    def from_bars(
        cls,
        bars: Iterable[Mapping[str, Any]] | pd.DataFrame,
        *,
        agent_group: str = "retail",
        return_threshold: float = 0.001,
    ) -> "ImitationDataset":
        frame = bars.copy() if isinstance(bars, pd.DataFrame) else pd.DataFrame(list(bars))
        if frame.empty:
            return cls([], source="bar_proxy")
        close = pd.to_numeric(frame.get("close", frame.get("price", pd.Series(dtype=float))), errors="coerce").ffill().bfill()
        volume = pd.to_numeric(frame.get("volume", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
        returns = close.pct_change().fillna(0.0)
        vol_norm = (volume / max(float(volume.mean() or 1.0), 1.0)).clip(lower=0.0, upper=10.0)
        samples: List[ImitationSample] = []
        for idx in range(1, len(frame)):
            ret = float(returns.iloc[idx])
            if ret > return_threshold:
                action = "BUY"
            elif ret < -return_threshold:
                action = "SELL"
            else:
                action = "HOLD"
            state = {
                "last_return": float(ret),
                "momentum": float(returns.iloc[max(0, idx - 5): idx + 1].sum()),
                "realized_volatility": float(returns.iloc[max(0, idx - 20): idx + 1].std() or 0.0),
                "volume_ratio": float(vol_norm.iloc[idx]),
                "price": float(close.iloc[idx]),
            }
            samples.append(
                ImitationSample(
                    state=state,
                    action=action,
                    weight=float(1.0 + min(abs(ret) * 100.0, 2.0)),
                    metadata={"index": int(idx), "agent_group": str(agent_group), "proxy": "bar_return_sign"},
                )
            )
        return cls(samples, source="bar_proxy")

    @classmethod
    def from_order_flow_proxy(
        cls,
        rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
        *,
        agent_group: str = "market_maker",
    ) -> "ImitationDataset":
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if frame.empty:
            return cls([], source="order_flow_proxy")
        samples: List[ImitationSample] = []
        for idx, row in frame.reset_index(drop=True).iterrows():
            buy = float(row.get("buy_volume", row.get("bid_volume", 0.0)) or 0.0)
            sell = float(row.get("sell_volume", row.get("ask_volume", 0.0)) or 0.0)
            imbalance = (buy - sell) / max(abs(buy) + abs(sell), 1.0)
            action = "BUY" if imbalance > 0.08 else "SELL" if imbalance < -0.08 else "HOLD"
            samples.append(
                ImitationSample(
                    state={
                        "order_flow_imbalance": float(imbalance),
                        "spread": float(row.get("spread", 0.0) or 0.0),
                        "depth_imbalance": float(row.get("depth_imbalance", imbalance) or 0.0),
                        "price": float(row.get("price", row.get("last_price", 0.0)) or 0.0),
                    },
                    action=action,
                    weight=float(1.0 + abs(imbalance)),
                    metadata={"index": int(idx), "agent_group": str(agent_group), "proxy": "order_flow_imbalance"},
                )
            )
        return cls(samples, source="order_flow_proxy")

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for sample in self.samples:
            rows.append({"action": sample.action, "weight": sample.weight, **sample.state, **sample.metadata})
        return pd.DataFrame(rows)

    def action_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"BUY": 0, "HOLD": 0, "SELL": 0}
        for sample in self.samples:
            counts[sample.action] = counts.get(sample.action, 0) + 1
        return counts


__all__ = ["ImitationDataset", "ImitationSample"]
