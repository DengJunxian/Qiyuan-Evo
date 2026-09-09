"""Macro state and macro-to-agent DTO definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class MacroState:
    """Top-level macro state shared by social and micro simulation layers."""

    inflation: float = 0.02
    unemployment: float = 0.055
    wage_growth: float = 0.03
    credit_spread: float = 0.018
    liquidity_index: float = 1.0
    policy_rate: float = 0.0225
    fiscal_stimulus: float = 0.0
    sentiment_index: float = 0.5

    def apply_delta(self, **delta: float) -> "MacroState":
        """Apply additive deltas by field name and clamp to safe ranges."""
        for key, value in delta.items():
            if not hasattr(self, key):
                continue
            current = float(getattr(self, key))
            setattr(self, key, current + float(value))
        self.clamp_inplace()
        return self

    def clamp_inplace(self) -> "MacroState":
        """Clamp fields to simulation-stable ranges."""
        self.inflation = _clip(self.inflation, -0.05, 0.30)
        self.unemployment = _clip(self.unemployment, 0.0, 0.35)
        self.wage_growth = _clip(self.wage_growth, -0.20, 0.25)
        self.credit_spread = _clip(self.credit_spread, 0.0, 0.20)
        self.liquidity_index = _clip(self.liquidity_index, 0.10, 3.00)
        self.policy_rate = _clip(self.policy_rate, 0.0, 0.20)
        self.fiscal_stimulus = _clip(self.fiscal_stimulus, -0.50, 0.50)
        self.sentiment_index = _clip(self.sentiment_index, 0.0, 1.0)
        return self

    def to_dict(self) -> Dict[str, float]:
        """Serialize state to dict."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MacroState":
        """Build a macro state from dict-like payload."""
        kwargs: Dict[str, float] = {}
        for field_name in cls.__dataclass_fields__.keys():
            if field_name in payload:
                kwargs[field_name] = float(payload[field_name])
        return cls(**kwargs).clamp_inplace()


@dataclass(slots=True)
class MacroContextDTO:
    """Unified macro-social context payload injected into trading agents."""

    tick: int
    macro_state: MacroState
    policy_summary: str
    social_sentiment: float
    sector_outlook: Dict[str, float] = field(default_factory=dict)
    household_risk_shift: float = 0.0
    firm_hiring_signal: float = 0.0
    policy_event: Dict[str, Any] = field(default_factory=dict)
    policy_package: Dict[str, Any] = field(default_factory=dict)
    transmission_layers: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Serialize for transport through generic dict-based APIs."""
        return {
            "tick": int(self.tick),
            "policy_summary": self.policy_summary,
            "social_sentiment": float(self.social_sentiment),
            "sector_outlook": {k: float(v) for k, v in self.sector_outlook.items()},
            "household_risk_shift": float(self.household_risk_shift),
            "firm_hiring_signal": float(self.firm_hiring_signal),
            "policy_event": dict(self.policy_event),
            "policy_package": dict(self.policy_package),
            "transmission_layers": dict(self.transmission_layers),
            "macro_state": self.macro_state.to_dict(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MacroContextDTO":
        """Deserialize from dict payload."""
        macro_payload = payload.get("macro_state", {}) if isinstance(payload, Mapping) else {}
        macro_state = MacroState.from_mapping(macro_payload if isinstance(macro_payload, Mapping) else {})
        sector_payload = payload.get("sector_outlook", {}) if isinstance(payload, Mapping) else {}
        sector_outlook = {
            str(k): float(v)
            for k, v in (sector_payload.items() if isinstance(sector_payload, Mapping) else [])
        }
        return cls(
            tick=int(payload.get("tick", 0)),
            macro_state=macro_state,
            policy_summary=str(payload.get("policy_summary", "no_policy")),
            social_sentiment=float(payload.get("social_sentiment", 0.0)),
            sector_outlook=sector_outlook,
            household_risk_shift=float(payload.get("household_risk_shift", 0.0)),
            firm_hiring_signal=float(payload.get("firm_hiring_signal", 0.0)),
            policy_event=dict(payload.get("policy_event", {})) if isinstance(payload, Mapping) else {},
            policy_package=dict(payload.get("policy_package", {})) if isinstance(payload, Mapping) else {},
            transmission_layers=dict(payload.get("transmission_layers", {})) if isinstance(payload, Mapping) else {},
        )

    @classmethod
    def coerce(cls, payload: Any) -> "MacroContextDTO | None":
        """Coerce object to MacroContextDTO if possible."""
        if payload is None:
            return None
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, Mapping):
            return cls.from_payload(payload)
        return None

    def to_prompt_context(self) -> str:
        """Render concise prompt text for LLM-based trading agents."""
        m = self.macro_state
        return (
            f"[MacroContext@tick={self.tick}] policy='{self.policy_summary}' | "
            f"inflation={m.inflation:.3f}, unemployment={m.unemployment:.3f}, "
            f"wage_growth={m.wage_growth:.3f}, credit_spread={m.credit_spread:.3f}, "
            f"liquidity_index={m.liquidity_index:.3f}, policy_rate={m.policy_rate:.3f}, "
            f"fiscal_stimulus={m.fiscal_stimulus:.3f}, sentiment_index={m.sentiment_index:.3f}, "
            f"social_sentiment={self.social_sentiment:.3f}, "
            f"household_risk_shift={self.household_risk_shift:.3f}, "
            f"firm_hiring_signal={self.firm_hiring_signal:.3f}"
        )


if __name__ == "__main__":
    state = MacroState()
    state.apply_delta(liquidity_index=0.15, credit_spread=-0.003, sentiment_index=0.08)
    dto = MacroContextDTO(
        tick=1,
        macro_state=state,
        policy_summary="流动性注入 5000 亿",
        social_sentiment=0.24,
        sector_outlook={"金融": 0.3, "科技": 0.2},
    )
    print(dto.to_payload())
    print(dto.to_prompt_context())
