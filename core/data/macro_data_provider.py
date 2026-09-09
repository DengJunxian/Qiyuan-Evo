"""Deterministic macro panel provider for replay calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.exchange.trade_tape import stable_hash


MACRO_COLUMNS = [
    "date",
    "cpi",
    "ppi",
    "social_financing",
    "m2",
    "repo_7d",
    "credit_spread",
    "northbound_flow",
    "margin_balance",
    "industry_index",
]


@dataclass
class MacroPanel:
    frame: pd.DataFrame
    provider: str
    snapshot_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "snapshot_hash": self.snapshot_hash,
            "rows": int(len(self.frame)),
            "columns": list(self.frame.columns),
        }


class MacroDataProvider:
    """Offline-safe macro panel for calibration and replay tests."""

    def load_macro_panel(self, start: str, end: str, *, seed: int = 42) -> MacroPanel:
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) == 0:
            dates = pd.bdate_range(start=pd.Timestamp(start), periods=1)
        rng = np.random.default_rng(int(seed))
        x = np.linspace(0.0, 1.0, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "cpi": 0.018 + 0.004 * np.sin(x * np.pi * 2.0),
                "ppi": 0.006 + 0.006 * np.cos(x * np.pi * 1.7),
                "social_financing": 1.0 + 0.08 * np.sin(x * np.pi * 1.2) + rng.normal(0.0, 0.005, len(dates)),
                "m2": 0.085 + 0.006 * np.cos(x * np.pi * 1.4),
                "repo_7d": 0.019 + 0.002 * np.sin(x * np.pi * 2.3),
                "credit_spread": 0.012 + 0.003 * np.cos(x * np.pi * 2.1),
                "northbound_flow": rng.normal(0.0, 1.0, len(dates)).cumsum() / max(len(dates), 1),
                "margin_balance": 1.0 + 0.03 * np.sin(x * np.pi * 1.8),
                "industry_index": 1.0 + 0.04 * np.cos(x * np.pi * 1.1),
            }
        )
        snapshot_hash = stable_hash({"macro_panel": frame.to_dict(orient="records")})
        return MacroPanel(frame=frame[MACRO_COLUMNS], provider="deterministic_macro_fixture", snapshot_hash=snapshot_hash)


__all__ = ["MACRO_COLUMNS", "MacroDataProvider", "MacroPanel"]
