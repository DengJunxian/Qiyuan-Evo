"""Firm agents for sector-level macro transmission."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from core.macro.state import MacroState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class FirmSignal:
    """Per-step firm output used for macro aggregation."""

    firm_id: str
    sector: str
    earnings_expectation: float
    hiring_plan: float
    inventory: float
    financing_cost: float
    sector_outlook: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


@dataclass(slots=True)
class FirmAgent:
    """Macro firm agent with expectations, hiring and financing channels."""

    firm_id: str
    sector: str
    earnings_expectation: float = 0.0
    hiring_plan: float = 0.0
    inventory: float = 100.0
    financing_cost: float = 0.035
    sector_outlook: float = 0.0

    def step(self, macro_state: MacroState, social_sentiment: float, demand_signal: float) -> FirmSignal:
        """Update firm state by macro + sentiment + demand inputs."""
        demand_pressure = _clip(demand_signal / 10_000.0, -1.0, 1.0)
        sentiment_term = _clip(social_sentiment * 0.5, -0.5, 0.5)

        self.financing_cost = _clip(
            0.6 * self.financing_cost + 0.4 * (macro_state.policy_rate + macro_state.credit_spread),
            0.005,
            0.25,
        )
        self.earnings_expectation = _clip(
            0.55 * self.earnings_expectation
            + 0.25 * demand_pressure
            + 0.20 * (macro_state.sentiment_index - 0.5)
            - 0.40 * (self.financing_cost - 0.03),
            -1.0,
            1.0,
        )
        self.hiring_plan = _clip(0.5 * self.hiring_plan + 0.7 * self.earnings_expectation, -1.0, 1.0)
        self.inventory = _clip(self.inventory + 12.0 * (0.1 - demand_pressure), 10.0, 400.0)
        self.sector_outlook = _clip(0.6 * self.sector_outlook + 0.4 * (self.earnings_expectation + sentiment_term), -1.0, 1.0)

        return FirmSignal(
            firm_id=self.firm_id,
            sector=self.sector,
            earnings_expectation=self.earnings_expectation,
            hiring_plan=self.hiring_plan,
            inventory=self.inventory,
            financing_cost=self.financing_cost,
            sector_outlook=self.sector_outlook,
        )


if __name__ == "__main__":
    macro = MacroState()
    f = FirmAgent(firm_id="firm-001", sector="科技")
    for step in range(3):
        sig = f.step(macro, social_sentiment=0.15, demand_signal=8_000.0)
        print(f"step={step+1}", sig.to_dict())
