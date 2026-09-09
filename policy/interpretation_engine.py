"""Policy interpretation layer: structured policy package -> per-agent belief."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from policy.structured import PolicyPackage


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class AgentBelief:
    """Belief object consumed by trading agents."""

    expected_return: Dict[str, float]
    expected_risk: Dict[str, float]
    liquidity_score: Dict[str, float]
    confidence: float
    latency_bars: int
    disagreement_tags: List[str] = field(default_factory=list)
    expected_holding_period_shift: Dict[str, float] = field(default_factory=dict)
    leverage_pressure: Dict[str, float] = field(default_factory=dict)
    funding_stress: Dict[str, float] = field(default_factory=dict)
    sentiment_susceptibility: Dict[str, float] = field(default_factory=dict)
    sector_rotation_preference: Dict[str, float] = field(default_factory=dict)
    safe_haven_bias: Dict[str, float] = field(default_factory=dict)
    regulation_fear: Dict[str, float] = field(default_factory=dict)
    compliance_pressure: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


EventBelief = AgentBelief


class PolicyInterpretationEngine:
    """Map a single policy package into heterogeneous agent beliefs."""

    def __init__(
        self,
        *,
        default_symbols: Optional[Sequence[str]] = None,
        symbol_sector_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.default_symbols = list(default_symbols or ["A_SHARE_IDX"])
        self.symbol_sector_map = dict(symbol_sector_map or {})

    def _resolve_symbols(self, market_state: Optional[Mapping[str, Any]]) -> List[str]:
        if not market_state:
            return list(self.default_symbols)
        if isinstance(market_state.get("symbols"), list) and market_state.get("symbols"):
            return [str(symbol) for symbol in market_state.get("symbols", [])]
        prices = market_state.get("prices")
        if isinstance(prices, Mapping) and prices:
            return [str(symbol) for symbol in prices.keys()]
        return list(self.default_symbols)

    def _persona_key(self, persona: Any) -> str:
        key = str(getattr(persona, "archetype_key", "") or "").strip()
        if key:
            return key
        archetype = getattr(persona, "archetype", None)
        if archetype is not None:
            key = str(getattr(archetype, "key", "") or "").strip()
            if key:
                return key
        return "retail"

    def _persona_policy_sensitivity(self, persona: Any) -> float:
        archetype = getattr(persona, "archetype", None)
        if archetype is None:
            return 0.5
        value = getattr(archetype, "policy_channel_sensitivity", 0.5)
        return float(_clip(value, 0.0, 1.0))

    def _persona_latency(self, persona: Any) -> int:
        archetype = getattr(persona, "archetype", None)
        mutable_state = getattr(persona, "mutable_state", None)
        delay = int(getattr(mutable_state, "policy_reaction_delay", 0) or 0)
        if archetype is None:
            return max(0, delay)
        bars = int(getattr(archetype, "info_latency_bars", 1) or 1)
        return max(0, bars + delay)

    def _persona_liquidity_pref(self, persona: Any) -> float:
        archetype = getattr(persona, "archetype", None)
        if archetype is None:
            return 0.5
        return float(_clip(getattr(archetype, "liquidity_preference", 0.5), 0.0, 1.0))

    def _persona_rumor_sensitivity(self, persona: Any) -> float:
        archetype = getattr(persona, "archetype", None)
        if archetype is None:
            return 0.5
        return float(_clip(getattr(archetype, "rumor_sensitivity", 0.5), 0.0, 1.0))

    def _persona_benchmark_pressure(self, persona: Any) -> float:
        archetype = getattr(persona, "archetype", None)
        if archetype is None:
            return 0.5
        return float(_clip(getattr(archetype, "benchmark_tracking_pressure", 0.5), 0.0, 1.0))

    def _bars_since_policy(self, policy_pkg: PolicyPackage, market_state: Optional[Mapping[str, Any]]) -> int:
        current_tick = int((market_state or {}).get("tick", getattr(policy_pkg.event, "tick", 0)) or getattr(policy_pkg.event, "tick", 0))
        policy_tick = int(getattr(policy_pkg.event, "tick", 0) or 0)
        return max(0, current_tick - policy_tick)

    def _channel_weight(self, channel: Any, persona: Any) -> float:
        sensitivity = self._persona_policy_sensitivity(persona)
        rumor_sensitivity = self._persona_rumor_sensitivity(persona)
        liquidity_pref = self._persona_liquidity_pref(persona)
        benchmark_pressure = self._persona_benchmark_pressure(persona)
        channel_name = str(getattr(channel, "name", "") or getattr(channel, "channel_type", "") or "").strip().lower()

        if channel_name == "rumor_suppression":
            return float(_clip(0.20 + 0.80 * rumor_sensitivity, 0.0, 1.5))
        if channel_name in {"liquidity", "tax_frictions"}:
            return float(_clip(0.25 + 0.75 * sensitivity + 0.15 * (1.0 - liquidity_pref), 0.0, 1.5))
        if channel_name in {"rate", "credit_spread"}:
            return float(_clip(0.25 + 0.65 * sensitivity + 0.20 * benchmark_pressure, 0.0, 1.5))
        if channel_name == "fiscal_demand":
            return float(_clip(0.25 + 0.75 * sensitivity, 0.0, 1.5))
        if channel_name == "sentiment_confidence":
            return float(_clip(0.20 + 0.45 * sensitivity + 0.35 * rumor_sensitivity, 0.0, 1.5))
        return float(_clip(0.25 + 0.75 * sensitivity, 0.0, 1.5))

    def _channel_activation(self, channel: Any, *, bars_since_policy: int, latency_bars: int) -> float:
        channel_lag = int(getattr(channel, "lag_days", 0) or 0)
        effective_lag = max(0, latency_bars + channel_lag)
        if effective_lag <= 0:
            return 1.0
        return float(_clip((bars_since_policy + 1) / float(effective_lag + 1), 0.0, 1.0))

    def _resolve_sector(self, symbol: str) -> str:
        if symbol in self.symbol_sector_map:
            return str(self.symbol_sector_map[symbol])
        token = symbol.lower()
        if "bank" in token or "fin" in token:
            return "financials"
        if "chip" in token or "ai" in token or "tech" in token:
            return "growth"
        if "real" in token or "prop" in token:
            return "property"
        if "energy" in token or "new" in token:
            return "cyclical"
        if "cons" in token:
            return "consumer"
        return "state_owned" if token in {"a_share_idx", "index", "idx"} else "defensive"

    def _agent_effect_for_persona(
        self,
        package: PolicyPackage,
        persona_key: str,
    ) -> float:
        effects = dict(package.agent_class_effects or {})
        if persona_key in effects:
            return float(effects[persona_key])
        alias_map = {
            "retail_general": "retail",
            "retail_value": "retail",
            "retail_emotional": "retail",
            "retail_day_trader": "retail",
            "retail_swing": "retail",
            "retail_momentum_chaser": "retail",
            "retail_mean_reverter": "retail",
            "private_fund_discretionary": "institution",
            "mutual_fund": "institution",
            "pension_fund": "institution",
            "insurer": "institution",
            "long_term_institution": "institution",
            "quant_stock_selector": "quant",
            "quant_timing": "quant",
            "quant_arbitrage": "quant",
            "trend_trader": "quant",
            "event_driven_capital": "institution",
            "state_stabilization_fund": "state_stabilization",
            "policy_capital": "policy_capital",
            "market_maker": "market_maker",
            "media_propagator": "rumor_trader",
        }
        alias = alias_map.get(persona_key, "")
        if alias and alias in effects:
            return float(effects[alias])
        if "retail" in persona_key and "retail" in effects:
            return float(effects["retail"])
        if "quant" in persona_key and "quant" in effects:
            return float(effects["quant"])
        if "maker" in persona_key and "market_maker" in effects:
            return float(effects["market_maker"])
        if "stabilization" in persona_key and "state_stabilization" in effects:
            return float(effects["state_stabilization"])
        return 0.0

    def interpret(
        self,
        policy_pkg: PolicyPackage,
        persona: Any,
        market_state: Optional[Mapping[str, Any]] = None,
        portfolio: Optional[Mapping[str, Any]] = None,
    ) -> AgentBelief:
        symbols = self._resolve_symbols(market_state)
        market_effects = dict(policy_pkg.market_effects or {})
        persona_key = self._persona_key(persona)
        agent_bias = self._agent_effect_for_persona(policy_pkg, persona_key)
        policy_confidence = float(getattr(policy_pkg.uncertainty, "confidence", 0.5))
        sensitivity = self._persona_policy_sensitivity(persona)
        confidence = _clip(policy_confidence * (0.7 + 0.6 * sensitivity), 0.05, 0.99)
        liquidity_pref = self._persona_liquidity_pref(persona)
        latency_bars = self._persona_latency(persona)
        bars_since_policy = self._bars_since_policy(policy_pkg, market_state)
        market_bias = float(market_effects.get("market_bias", 0.0))
        liquidity_bias = float(market_effects.get("liquidity_bias", 0.0))
        volatility_bias = float(market_effects.get("volatility_bias", 0.0))

        channel_weights: Dict[str, float] = {}
        channel_activation: Dict[str, float] = {}
        channel_signal = 0.0
        channel_weight_total = 0.0
        for channel in list(getattr(policy_pkg, "channels", []) or []):
            name = str(getattr(channel, "name", "generic") or "generic")
            weight = self._channel_weight(channel, persona)
            activation = self._channel_activation(channel, bars_since_policy=bars_since_policy, latency_bars=latency_bars)
            impact = float(getattr(channel, "impact", 0.0) or 0.0)
            channel_weights[name] = float(weight)
            channel_activation[name] = float(activation)
            channel_signal += impact * weight * activation
            channel_weight_total += abs(impact) * max(weight * activation, 1e-6)
        normalized_channel_bias = channel_signal / channel_weight_total if channel_weight_total > 0 else 0.0
        effective_pass_through = float(sum(channel_activation.values()) / max(len(channel_activation), 1)) if channel_activation else 1.0
        effective_confidence = float(_clip(confidence * (0.55 + 0.45 * effective_pass_through), 0.05, 0.99))

        expected_return: Dict[str, float] = {}
        expected_risk: Dict[str, float] = {}
        liquidity_score: Dict[str, float] = {}
        expected_holding_period_shift: Dict[str, float] = {}
        leverage_pressure: Dict[str, float] = {}
        funding_stress: Dict[str, float] = {}
        sentiment_susceptibility: Dict[str, float] = {}
        sector_rotation_preference: Dict[str, float] = {}
        safe_haven_bias: Dict[str, float] = {}
        regulation_fear: Dict[str, float] = {}
        compliance_pressure: Dict[str, float] = {}
        disagreement_tags: List[str] = []
        compliance_signal = float(market_effects.get("compliance_pressure", 0.0) or 0.0)
        funding_signal = float(market_effects.get("funding_stress", 0.0) or 0.0)
        if not compliance_signal:
            compliance_signal = float((policy_pkg.agent_class_effects or {}).get("rumor_trader", 0.0)) * -0.35
        if not funding_signal:
            funding_signal = float(max(0.0, volatility_bias) + max(0.0, -liquidity_bias)) * 0.35

        for symbol in symbols:
            sector = self._resolve_sector(symbol)
            sector_bias = float((policy_pkg.sector_effects or {}).get(sector, 0.0))
            value = 0.30 * market_bias + 0.25 * agent_bias + 0.15 * sector_bias + 0.30 * normalized_channel_bias
            expected_return[symbol] = _clip(value * effective_confidence, -0.30, 0.30)

            risk_value = 0.18 + 0.42 * abs(volatility_bias) + 0.18 * (1.0 - effective_confidence) + 0.10 * (1.0 - effective_pass_through)
            expected_risk[symbol] = _clip(risk_value, 0.01, 1.0)

            liq = 0.50 + 0.40 * liquidity_bias * effective_pass_through - 0.25 * liquidity_pref
            liquidity_score[symbol] = _clip(liq, 0.0, 1.0)
            expected_holding_period_shift[symbol] = _clip(
                0.35 * confidence + 0.30 * sector_bias - 0.40 * abs(volatility_bias) - 0.20 * self._persona_benchmark_pressure(persona),
                -1.0,
                1.0,
            )
            leverage_pressure[symbol] = _clip(
                funding_signal + max(0.0, -liquidity_bias) * 0.55 + max(0.0, volatility_bias) * 0.35,
                0.0,
                1.0,
            )
            funding_stress[symbol] = _clip(funding_signal + max(0.0, -liquidity_bias) * 0.60, 0.0, 1.0)
            sentiment_susceptibility[symbol] = _clip(
                self._persona_rumor_sensitivity(persona) * (0.55 + 0.45 * abs(market_bias)),
                0.0,
                1.0,
            )
            sector_rotation_preference[symbol] = _clip(sector_bias + 0.35 * normalized_channel_bias, -1.0, 1.0)
            safe_haven_bias[symbol] = _clip(max(0.0, -value) + max(0.0, volatility_bias) * 0.35, 0.0, 1.0)
            regulation_fear[symbol] = _clip(compliance_signal + max(0.0, -agent_bias) * 0.25, 0.0, 1.0)
            compliance_pressure[symbol] = _clip(compliance_signal + 0.20 * self._persona_policy_sensitivity(persona), 0.0, 1.0)

        if effective_confidence < 0.35:
            disagreement_tags.append("low_policy_confidence")
        if abs(agent_bias) < 0.05:
            disagreement_tags.append("weak_agent_specific_signal")
        if volatility_bias < -0.15:
            disagreement_tags.append("tail_risk_elevated")
        if latency_bars > 2:
            disagreement_tags.append("slow_information_processing")
        if effective_pass_through < 0.5:
            disagreement_tags.append("lagged_policy_pass_through")
        if isinstance(portfolio, Mapping) and portfolio:
            disagreement_tags.append("position_constraint_active")

        metadata = {
            "persona_key": persona_key,
            "agent_bias": float(agent_bias),
            "market_bias": float(market_bias),
            "channel_bias": float(normalized_channel_bias),
            "volatility_bias": float(volatility_bias),
            "liquidity_bias": float(liquidity_bias),
            "bars_since_policy": int(bars_since_policy),
            "effective_pass_through": float(effective_pass_through),
            "channel_weights": channel_weights,
            "channel_activation": channel_activation,
            "policy_id": getattr(policy_pkg.event, "policy_id", ""),
        }
        return AgentBelief(
            expected_return=expected_return,
            expected_risk=expected_risk,
            liquidity_score=liquidity_score,
            confidence=float(effective_confidence),
            latency_bars=int(latency_bars),
            disagreement_tags=disagreement_tags,
            expected_holding_period_shift=expected_holding_period_shift,
            leverage_pressure=leverage_pressure,
            funding_stress=funding_stress,
            sentiment_susceptibility=sentiment_susceptibility,
            sector_rotation_preference=sector_rotation_preference,
            safe_haven_bias=safe_haven_bias,
            regulation_fear=regulation_fear,
            compliance_pressure=compliance_pressure,
            metadata=metadata,
        )

    def batch_interpret(
        self,
        policy_pkg: Optional[PolicyPackage],
        agents: Iterable[Any],
        market_state: Optional[Mapping[str, Any]] = None,
    ) -> List[AgentBelief]:
        if policy_pkg is None:
            return []
        beliefs: List[AgentBelief] = []
        for agent in agents:
            persona = getattr(agent, "persona", None)
            portfolio = getattr(agent, "portfolio", None)
            beliefs.append(self.interpret(policy_pkg, persona, market_state, portfolio))
        return beliefs
