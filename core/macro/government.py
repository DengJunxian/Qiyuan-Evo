"""Government and policy compiler for structured policy shocks."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from core.macro.state import MacroState
from policy.structured import PolicyEvent, PolicyPackage, StructuredPolicyParser, stable_json_hash


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class PolicyShock:
    """Structured policy shock compiled from free-form policy text."""

    policy_id: str
    policy_text: str
    policy_rate_delta: float = 0.0
    fiscal_stimulus_delta: float = 0.0
    liquidity_injection: float = 0.0
    credit_spread_delta: float = 0.0
    stamp_tax_delta: float = 0.0
    sentiment_delta: float = 0.0
    rumor_shock: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


class PolicyCompiler:
    """Compile policy text into structured and simulation-ready shocks."""

    _RE_NUMERIC = re.compile(r"(-?\d+(?:\.\d+)?)")

    def compile(self, policy_text: str, *, tick: int = 0) -> PolicyShock:
        text = str(policy_text or "").strip()
        lower = text.lower()
        shock = PolicyShock(policy_id=f"policy-{tick}-{uuid.uuid4().hex[:8]}", policy_text=text)

        if not text:
            return shock

        numeric_hint = self._extract_numeric_hint(text)

        if "印花税" in text and any(token in text for token in ("下调", "降低", "减免", "下调")):
            scale = 0.001 + 0.0002 * numeric_hint
            shock.stamp_tax_delta -= scale
            shock.liquidity_injection += 0.08 + 0.03 * numeric_hint
            shock.sentiment_delta += 0.08
            shock.credit_spread_delta -= 0.0015

        if any(token in text for token in ("流动性注入", "降准", "降息", "MLF", "再贷款")):
            scale = 0.08 + 0.04 * numeric_hint
            shock.liquidity_injection += scale
            shock.policy_rate_delta -= 0.001 * (1.0 + numeric_hint)
            shock.credit_spread_delta -= 0.002
            shock.sentiment_delta += 0.06

        if any(token in text for token in ("财政刺激", "基建", "补贴", "减税")):
            shock.fiscal_stimulus_delta += 0.05 + 0.02 * numeric_hint
            shock.sentiment_delta += 0.04

        if any(token in text for token in ("负面谣言", "谣言", "爆雷", "panic", "挤兑")):
            shock.rumor_shock -= 0.25 - 0.05 * min(numeric_hint, 1.0)
            shock.sentiment_delta -= 0.12
            shock.liquidity_injection -= 0.06
            shock.credit_spread_delta += 0.003

        if any(token in lower for token in ("rate hike", "hike", "tightening")) or "加息" in text:
            shock.policy_rate_delta += 0.0015 + 0.001 * numeric_hint
            shock.credit_spread_delta += 0.0025
            shock.sentiment_delta -= 0.05

        if all(
            getattr(shock, field) == 0.0
            for field in (
                "policy_rate_delta",
                "fiscal_stimulus_delta",
                "liquidity_injection",
                "credit_spread_delta",
                "stamp_tax_delta",
                "sentiment_delta",
                "rumor_shock",
            )
        ):

            shock.sentiment_delta = 0.01 if "支持" in text else -0.01 if "风险" in text else 0.0

        return shock

    def _extract_numeric_hint(self, text: str) -> float:
        match = self._RE_NUMERIC.search(text)
        if not match:
            return 0.0
        value = abs(float(match.group(1)))
        if value > 1000:
            return 1.0
        if value > 100:
            return 0.6
        if value > 10:
            return 0.3
        if value > 1:
            return 0.1
        return 0.0


class GovernmentAgent:
    """Government layer that compiles policy text and applies macro shocks."""

    def __init__(
        self,
        compiler: Optional[PolicyCompiler] = None,
        *,
        feature_flags: Optional[Mapping[str, bool]] = None,
        policy_parser: Optional[StructuredPolicyParser] = None,
    ):
        self.compiler = compiler or PolicyCompiler()
        self.feature_flags: Dict[str, bool] = {
            "structured_policy_parser_v1": True,
            "policy_transmission_layers_v1": True,
        }
        if feature_flags:
            for key, value in feature_flags.items():
                self.feature_flags[str(key)] = bool(value)
        self.policy_parser = policy_parser or StructuredPolicyParser(feature_flags=self.feature_flags)

    def _infer_policy_family(self, policy_text: str, shock: Optional[PolicyShock] = None) -> Tuple[str, str]:
        text = str(policy_text or "").lower()
        if shock is not None:
            if shock.stamp_tax_delta < 0:
                return "tax_cut", "easing"
            if shock.stamp_tax_delta > 0:
                return "tax_hike", "tightening"
            if shock.policy_rate_delta < 0 or shock.liquidity_injection > 0:
                return "liquidity_easing", "easing"
            if shock.policy_rate_delta > 0 or shock.credit_spread_delta > 0:
                return "tightening", "tightening"
            if shock.fiscal_stimulus_delta > 0:
                return "fiscal_stimulus", "easing"
            if shock.rumor_shock > 0 or "辟谣" in policy_text or "rumor" in text:
                return "rumor_refutation", "stabilizing"
        if "印花税" in policy_text or "stamp" in text:
            if "下调" in policy_text or "cut" in text:
                return "tax_cut", "easing"
            return "tax_hike", "tightening"
        if "降准" in policy_text or "liquidity" in text or "流动性" in policy_text:
            return "liquidity_easing", "easing"
        if "降息" in policy_text or "rate cut" in text:
            return "rate_cut", "easing"
        if "财政" in policy_text or "fiscal" in text or "补贴" in policy_text:
            return "fiscal_stimulus", "easing"
        if "辟谣" in policy_text or "rumor" in text or "澄清" in policy_text:
            return "rumor_refutation", "stabilizing"
        if "收紧" in policy_text or "tighten" in text or "加息" in policy_text:
            return "tightening", "tightening"
        if "平准" in policy_text or "维稳" in policy_text or "国家队" in policy_text:
            return "stabilization", "stabilizing"
        return "unclassified", "neutral"

    def _legacy_package(
        self,
        *,
        policy_text: str,
        shock: PolicyShock,
        tick: int,
        policy_type_hint: Optional[str],
        intensity: float,
        market_regime: Optional[str],
        snapshot_info: Optional[Mapping[str, Any]],
    ) -> PolicyPackage:
        policy_type, direction = self._infer_policy_family(policy_text, shock)
        if policy_type_hint:
            policy_type = str(policy_type_hint)
        channels = []
        def _pick_channel(family: str, channel_name: str, dir_hint: str) -> Any:
            family_channels = self.policy_parser._build_channels(family, 1.0, dir_hint, policy_text)
            for channel in family_channels:
                if channel.name == channel_name:
                    return channel
            return family_channels[0]

        if shock.liquidity_injection != 0.0:
            channels.append(_pick_channel("liquidity_easing", "liquidity", "easing" if shock.liquidity_injection > 0 else "tightening"))
        if shock.policy_rate_delta != 0.0:
            channels.append(_pick_channel("rate_cut", "rate", "easing" if shock.policy_rate_delta < 0 else "tightening"))
        if shock.fiscal_stimulus_delta != 0.0:
            channels.append(_pick_channel("fiscal_stimulus", "fiscal_demand", "easing"))
        if shock.stamp_tax_delta != 0.0:
            channels.append(_pick_channel("tax_cut" if shock.stamp_tax_delta < 0 else "tax_hike", "tax_frictions", "easing" if shock.stamp_tax_delta < 0 else "tightening"))
        if shock.credit_spread_delta != 0.0:
            channels.append(_pick_channel("tightening", "credit_spread", "tightening" if shock.credit_spread_delta > 0 else "easing"))
        if shock.sentiment_delta != 0.0:
            channels.append(_pick_channel("liquidity_easing", "sentiment_confidence", "easing" if shock.sentiment_delta > 0 else "tightening"))
        if shock.rumor_shock != 0.0:
            channels.append(_pick_channel("rumor_refutation", "rumor_suppression", "stabilizing" if shock.rumor_shock > 0 else "tightening"))

        if not channels:
            channels = self.policy_parser._build_channels("unclassified", 1.0, "neutral", policy_text)

        sector_effects, factor_effects, agent_class_effects, market_effects = self.policy_parser._build_layers(
            policy_type=policy_type,
            direction=direction,
            channels=channels,
            text=policy_text,
            market_regime=market_regime,
        )
        event = PolicyEvent(
            policy_id=shock.policy_id or f"policy-{tick}",
            raw_text=policy_text,
            policy_type=policy_type,
            policy_label="Legacy fallback",
            direction=direction,
            intensity=float(intensity),
            tick=int(tick),
            matched_tokens=[],
            source="legacy_compiler",
        )
        metadata = {
            "seed": 0,
            "config_hash": stable_json_hash(
                {
                    "policy_text": policy_text,
                    "tick": int(tick),
                    "intensity": float(intensity),
                    "policy_type_hint": policy_type_hint or "",
                    "mode": "legacy",
                }
            ),
            "snapshot_info": dict(snapshot_info or {}),
            "feature_flags": dict(self.feature_flags),
            "parser_mode": "legacy",
        }
        return PolicyPackage(
            event=event,
            channels=channels,
            uncertainty=self.policy_parser.parse(
                policy_text,
                tick=tick,
                policy_type_hint=policy_type_hint,
                intensity=intensity,
                market_regime=market_regime,
                snapshot_info=snapshot_info,
                fallback_used=True,
            ).uncertainty,
            sector_effects=sector_effects,
            factor_effects=factor_effects,
            agent_class_effects=agent_class_effects,
            market_effects=market_effects,
            metadata=metadata,
            parser_version="legacy_policy_compiler",
        )

    def compile_policy_package(
        self,
        policy_text: str,
        *,
        tick: int = 0,
        policy_type_hint: Optional[str] = None,
        intensity: float = 1.0,
        market_regime: Optional[str] = None,
        snapshot_info: Optional[Mapping[str, Any]] = None,
    ) -> PolicyPackage:
        """Compile free-form policy text into a structured policy package."""

        if not self.feature_flags.get("structured_policy_parser_v1", True):
            shock = self.compiler.compile(policy_text, tick=tick)
            return self._legacy_package(
                policy_text=policy_text,
                shock=shock,
                tick=tick,
                policy_type_hint=policy_type_hint,
                intensity=intensity,
                market_regime=market_regime,
                snapshot_info=snapshot_info,
            )

        package = self.policy_parser.parse(
            policy_text,
            tick=tick,
            policy_type_hint=policy_type_hint,
            intensity=intensity,
            market_regime=market_regime,
            snapshot_info=snapshot_info,
        )
        if package.uncertainty.confidence < 0.35 or package.event.policy_type == "unclassified":
            shock = self.compiler.compile(policy_text, tick=tick)
            return self._legacy_package(
                policy_text=policy_text,
                shock=shock,
                tick=tick,
                policy_type_hint=policy_type_hint,
                intensity=intensity,
                market_regime=market_regime,
                snapshot_info=snapshot_info,
            )
        return package

    def compile_policy_text(
        self,
        policy_text: str,
        *,
        tick: int = 0,
        policy_type_hint: Optional[str] = None,
        intensity: float = 1.0,
        market_regime: Optional[str] = None,
        snapshot_info: Optional[Mapping[str, Any]] = None,
    ) -> PolicyShock:
        """Compile free-form policy text into a structured policy shock."""
        package = self.compile_policy_package(
            policy_text,
            tick=tick,
            policy_type_hint=policy_type_hint,
            intensity=intensity,
            market_regime=market_regime,
            snapshot_info=snapshot_info,
        )
        shock_fields = package.to_policy_shock_fields()
        shock = PolicyShock(
            policy_id=package.event.policy_id,
            policy_text=policy_text,
            **shock_fields,
        )
        shock.metadata = {
            "policy_event": package.event.to_dict() if hasattr(package.event, "to_dict") else asdict(package.event),
            "policy_package": package.to_dict(),
            "reproducibility": {
                "seed": int(package.metadata.get("seed", 0)),
                "config_hash": str(package.metadata.get("config_hash", "")),
                "snapshot_info": dict(package.metadata.get("snapshot_info", {})),
            },
            "parser_mode": package.uncertainty.parser_mode,
            "feature_flags": dict(self.feature_flags),
        }
        return shock

    def apply_policy_shock(self, macro_state: MacroState, shock: PolicyShock) -> MacroState:
        """Apply policy shock to macro state."""
        macro_state.apply_delta(
            policy_rate=shock.policy_rate_delta,
            fiscal_stimulus=shock.fiscal_stimulus_delta,
            liquidity_index=shock.liquidity_injection,
            credit_spread=shock.credit_spread_delta,
            sentiment_index=shock.sentiment_delta,
        )

        macro_state.apply_delta(
            liquidity_index=-0.8 * shock.stamp_tax_delta,
            sentiment_index=-2.0 * shock.stamp_tax_delta,
        )
        if shock.rumor_shock != 0.0:
            macro_state.apply_delta(
                sentiment_index=shock.rumor_shock,
                liquidity_index=0.25 * shock.rumor_shock,
                unemployment=-0.06 * shock.rumor_shock,
            )
        macro_state.policy_rate = _clip(macro_state.policy_rate, 0.0, 0.20)
        macro_state.clamp_inplace()
        return macro_state


if __name__ == "__main__":
    gov = GovernmentAgent()
    macro = MacroState()
    for idx, text in enumerate(("印花税下调", "流动性注入 5000 亿", "突发负面谣言"), start=1):
        shock = gov.compile_policy_text(text, tick=idx)
        gov.apply_policy_shock(macro, shock)
        print(shock.to_dict())
        print(macro.to_dict())
