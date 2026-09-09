"""Bank transmission channel for credit and liquidity conditions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from core.macro.state import MacroState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class BankSignal:
    """Per-step banking signal emitted to macro layer."""

    credit_spread: float
    liquidity_index: float
    lending_tightness: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class BankAgent:
    """Macro bank agent that links policy rate to credit and liquidity."""

    lending_tightness: float = 0.5
    balance_sheet_capacity: float = 1.0

    def step(self, macro_state: MacroState, liquidity_shock: float = 0.0, sentiment_shock: float = 0.0) -> BankSignal:
        """Transmit macro shocks into credit spread and liquidity index."""
        stress = _clip(
            0.5 * macro_state.unemployment
            + 0.4 * (macro_state.policy_rate + macro_state.credit_spread)
            - 0.3 * macro_state.liquidity_index
            - 0.2 * macro_state.sentiment_index
            - 0.2 * liquidity_shock
            - 0.1 * sentiment_shock,
            -1.0,
            1.0,
        )
        self.lending_tightness = _clip(0.70 * self.lending_tightness + 0.30 * (0.5 + stress), 0.0, 1.0)
        self.balance_sheet_capacity = _clip(
            self.balance_sheet_capacity + 0.25 * liquidity_shock - 0.10 * max(stress, 0.0),
            0.2,
            2.0,
        )

        macro_state.credit_spread = _clip(
            0.65 * macro_state.credit_spread + 0.35 * (0.01 + 0.03 * self.lending_tightness),
            0.0,
            0.2,
        )
        macro_state.liquidity_index = _clip(
            macro_state.liquidity_index + 0.18 * liquidity_shock - 0.12 * self.lending_tightness,
            0.1,
            3.0,
        )
        macro_state.sentiment_index = _clip(macro_state.sentiment_index + 0.04 * (1.0 - self.lending_tightness), 0.0, 1.0)

        return BankSignal(
            credit_spread=macro_state.credit_spread,
            liquidity_index=macro_state.liquidity_index,
            lending_tightness=self.lending_tightness,
        )


if __name__ == "__main__":
    macro = MacroState()
    bank = BankAgent()
    for step in range(3):
        sig = bank.step(macro, liquidity_shock=0.2, sentiment_shock=0.1)
        print(f"step={step+1}", sig.to_dict(), macro.to_dict())
