"""Household agents for macro-to-social transmission."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from core.macro.state import MacroState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class HouseholdSignal:
    """Per-step household output used by macro aggregation."""

    household_id: str
    consumption: float
    savings: float
    risk_preference: float
    news_exposure: float
    social_exposure: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


@dataclass(slots=True)
class HouseholdAgent:
    """Macro household agent with income/consumption/savings and exposure dynamics."""

    household_id: str
    income: float = 12_000.0
    propensity_to_consume: float = 0.62
    savings: float = 35_000.0
    risk_preference: float = 0.5
    news_exposure: float = 0.4
    social_exposure: float = 0.5

    def step(self, macro_state: MacroState, social_sentiment: float, policy_text: str = "") -> HouseholdSignal:
        """Update household state and return one-step macro signal."""
        inflation_drag = 1.0 - macro_state.inflation * 0.4
        labor_income_factor = 1.0 - macro_state.unemployment * 0.8 + macro_state.wage_growth * 0.6
        effective_income = max(100.0, self.income * inflation_drag * labor_income_factor)

        confidence = _clip(
            0.55 * macro_state.sentiment_index
            + 0.25 * social_sentiment
            + 0.20 * (1.0 - macro_state.unemployment),
            0.0,
            1.0,
        )
        policy_bias = 0.03 if ("降" in policy_text or "注入" in policy_text) else 0.0
        rumor_drag = -0.06 if ("谣言" in policy_text or "panic" in policy_text.lower()) else 0.0

        self.risk_preference = _clip(
            self.risk_preference + 0.10 * (confidence - 0.5) + policy_bias + rumor_drag,
            0.0,
            1.0,
        )
        self.news_exposure = _clip(self.news_exposure + 0.10 * abs(rumor_drag) + 0.03 * (1.0 - confidence), 0.0, 1.0)
        self.social_exposure = _clip(self.social_exposure + 0.08 * abs(social_sentiment), 0.0, 1.0)

        desired_consumption = effective_income * _clip(self.propensity_to_consume + 0.20 * confidence, 0.20, 0.95)
        consumption = max(0.0, min(desired_consumption, effective_income + self.savings * 0.10))
        self.savings = max(0.0, self.savings + effective_income - consumption)

        return HouseholdSignal(
            household_id=self.household_id,
            consumption=consumption,
            savings=self.savings,
            risk_preference=self.risk_preference,
            news_exposure=self.news_exposure,
            social_exposure=self.social_exposure,
        )


if __name__ == "__main__":
    macro = MacroState()
    h = HouseholdAgent(household_id="hh-001")
    for step in range(3):
        sig = h.step(macro, social_sentiment=0.2, policy_text="流动性注入")
        print(f"step={step+1}", sig.to_dict())
