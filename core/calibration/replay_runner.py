"""Historical replay runner with Shanghai index scorecard output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core.calibration.losses import calibration_loss
from core.calibration.scorecard import ReplayScorecard, build_replay_scorecard
from core.data.macro_data_provider import MacroDataProvider, MacroPanel
from core.data.market_data_provider import MarketDataProvider, MarketDataQuery
from core.exchange.trade_tape import stable_hash


DEFAULT_PARAMS = {
    "trend_following": 0.55,
    "mean_reversion": 0.20,
    "macro_sensitivity": 0.35,
    "volatility_scale": 0.80,
    "liquidity_sensitivity": 0.25,
}


@dataclass(frozen=True)
class ReplayConfig:
    benchmark_symbol: str = "sh000001"
    start: str = "2024-01-01"
    end: str = "2024-03-31"
    seed: int = 42
    theta: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    freeze_snapshot: bool = False
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass
class ReplayResult:
    config: ReplayConfig
    real: pd.DataFrame
    simulated: pd.DataFrame
    macro: pd.DataFrame
    scorecard: ReplayScorecard
    loss: float
    config_hash: str
    data_snapshot_hash: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "real": self.real.to_dict(orient="records"),
            "simulated": self.simulated.to_dict(orient="records"),
            "macro_snapshot_hash": stable_hash({"macro": self.macro.to_dict(orient="records")}),
            "scorecard": self.scorecard.to_dict(),
            "loss": float(self.loss),
            "config_hash": self.config_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "seed": int(self.seed),
        }


class ReplayRunner:
    def __init__(
        self,
        *,
        market_provider: Optional[MarketDataProvider] = None,
        macro_provider: Optional[MacroDataProvider] = None,
    ) -> None:
        self.market_provider = market_provider or MarketDataProvider()
        self.macro_provider = macro_provider or MacroDataProvider()

    def _fallback_index_history(self, config: ReplayConfig) -> pd.DataFrame:
        dates = pd.bdate_range(start=config.start, end=config.end)
        if len(dates) == 0:
            dates = pd.bdate_range(start=pd.Timestamp(config.start), periods=30)
        rng = np.random.default_rng(int(config.seed))
        shocks = rng.normal(0.0004, 0.010, len(dates))
        close = 3000.0 * np.cumprod(1.0 + shocks)
        open_ = close * (1.0 - rng.normal(0.0, 0.0015, len(dates)))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.002, len(dates))))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.002, len(dates))))
        volume = np.clip(rng.normal(1_000_000, 100_000, len(dates)), 100_000, None)
        return pd.DataFrame(
            {
                "datetime": dates.strftime("%Y-%m-%d 00:00:00"),
                "date": dates.strftime("%Y-%m-%d"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": close * volume,
                "symbol": config.benchmark_symbol,
                "interval": "1d",
                "provider": "deterministic_index_fixture",
                "adjust": "",
                "market": "CN",
            }
        )

    def load_index_history(self, config: ReplayConfig) -> tuple[pd.DataFrame, str]:
        query = MarketDataQuery(
            symbol=config.benchmark_symbol,
            interval="1d",
            start=config.start,
            end=config.end,
            period_days=0,
            adjust="",
            market="CN",
        )
        try:
            real = self.market_provider.get_ohlcv(query, use_cache=True, freeze_snapshot=config.freeze_snapshot)
            if real.empty:
                raise ValueError("empty market data")
        except Exception:
            real = self._fallback_index_history(config)
        snapshot_hash = stable_hash({"index_history": real.to_dict(orient="records")})
        return real.reset_index(drop=True), snapshot_hash

    @staticmethod
    def _simulate_path(real: pd.DataFrame, macro: pd.DataFrame, config: ReplayConfig) -> pd.DataFrame:
        theta = {**DEFAULT_PARAMS, **dict(config.theta or {})}
        rng = np.random.default_rng(int(config.seed))
        close = pd.to_numeric(real["close"], errors="coerce").ffill().bfill().astype(float).to_numpy()
        real_returns = np.diff(close) / np.maximum(close[:-1], 1e-12) if close.size > 1 else np.asarray([], dtype=float)
        sim = [float(close[0] if close.size else 3000.0)]
        macro_frame = macro.reset_index(drop=True)
        for idx in range(1, len(close)):
            prev_real_ret = float(real_returns[idx - 2]) if idx >= 2 and real_returns.size else 0.0
            macro_row = macro_frame.iloc[min(idx, len(macro_frame) - 1)] if not macro_frame.empty else {}
            liquidity = float(getattr(macro_row, "get", lambda *_: 1.0)("social_financing", 1.0) or 1.0) - 1.0
            credit = float(getattr(macro_row, "get", lambda *_: 0.012)("credit_spread", 0.012) or 0.012) - 0.012
            northbound = float(getattr(macro_row, "get", lambda *_: 0.0)("northbound_flow", 0.0) or 0.0)
            noise = rng.normal(0.0, 0.0025 * float(theta["volatility_scale"]))
            ret = (
                float(theta["trend_following"]) * prev_real_ret
                - float(theta["mean_reversion"]) * ((sim[-1] - close[idx - 1]) / max(close[idx - 1], 1e-12))
                + float(theta["macro_sensitivity"]) * (0.18 * liquidity - 0.40 * credit + 0.015 * northbound)
                + noise
            )
            ret = float(np.clip(ret, -0.11, 0.11))
            sim.append(float(max(1e-6, sim[-1] * (1.0 + ret))))
        out = real[["date"]].copy()
        out["close"] = sim[: len(out)]
        out["open"] = out["close"].shift(1).fillna(out["close"])
        out["high"] = np.maximum(out["open"], out["close"]) * 1.002
        out["low"] = np.minimum(out["open"], out["close"]) * 0.998
        out["volume"] = pd.to_numeric(real.get("volume", pd.Series([0] * len(out))), errors="coerce").fillna(0.0)
        return out

    def run(self, config: ReplayConfig, *, macro_panel: Optional[MacroPanel] = None) -> ReplayResult:
        real, market_hash = self.load_index_history(config)
        macro_panel = macro_panel or self.macro_provider.load_macro_panel(config.start, config.end, seed=config.seed)
        macro = macro_panel.frame
        data_snapshot_hash = stable_hash({"market": market_hash, "macro": macro_panel.snapshot_hash})
        simulated = self._simulate_path(real, macro, config)
        scorecard = build_replay_scorecard(
            sim_close=simulated["close"].tolist(),
            real_close=real["close"].tolist(),
            benchmark_symbol=config.benchmark_symbol,
            replay_window={"start": config.start, "end": config.end},
            seed=config.seed,
            config_hash=config.config_hash,
            data_snapshot_hash=data_snapshot_hash,
            microstructure_metrics={"spread": 0.0, "depth_imbalance": 0.0},
            behavioral_metrics={"csad": 0.0},
            contagion_metrics={"panic_speed": 0.0},
            regulatory_metrics={"intervention_cost": 0.0},
        )
        loss = calibration_loss(sim_close=simulated["close"].tolist(), real_close=real["close"].tolist())
        return ReplayResult(
            config=config,
            real=real,
            simulated=simulated,
            macro=macro,
            scorecard=scorecard,
            loss=float(loss),
            config_hash=config.config_hash,
            data_snapshot_hash=data_snapshot_hash,
            seed=int(config.seed),
        )


def run_replay(
    *,
    theta: Optional[Mapping[str, Any]] = None,
    macro: Optional[pd.DataFrame] = None,
    initial_price: Optional[float] = None,
    seed: int = 42,
    benchmark_symbol: str = "sh000001",
    start: str = "2024-01-01",
    end: str = "2024-03-31",
) -> ReplayResult:
    del initial_price
    config = ReplayConfig(
        benchmark_symbol=benchmark_symbol,
        start=start,
        end=end,
        seed=int(seed),
        theta={str(k): float(v) for k, v in dict(theta or DEFAULT_PARAMS).items()},
    )
    macro_panel = None
    if macro is not None:
        macro_hash = stable_hash({"macro": macro.to_dict(orient="records")})
        macro_panel = MacroPanel(frame=macro.copy(), provider="caller_supplied", snapshot_hash=macro_hash)
    return ReplayRunner().run(config, macro_panel=macro_panel)


__all__ = ["DEFAULT_PARAMS", "ReplayConfig", "ReplayResult", "ReplayRunner", "run_replay"]
