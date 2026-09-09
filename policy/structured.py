"""Structured policy parsing and transmission models.

This module keeps the legacy macro shock path intact while adding a typed
policy representation with explicit transmission channels and layered outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash for nested JSON-like data."""

    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _match_any(text: str, *patterns: str) -> bool:
    lower = text.lower()
    return any(pattern in text or pattern in lower for pattern in patterns)


def _first_numeric_hint(text: str) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    value = abs(float(match.group(1)))
    if value > 1000:
        return 1.0
    if value > 100:
        return 0.7
    if value > 10:
        return 0.35
    if value > 1:
        return 0.1
    return 0.0


@dataclass(slots=True)
class TransmissionChannel:
    """A single policy transmission channel."""

    name: str
    impact: float
    lag_days: int
    decay_half_life: float
    direction: str
    channel_type: str = "generic"
    macro_delta: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyUncertainty:
    """Parse confidence and ambiguity for a policy event."""

    confidence: float
    ambiguity: float
    rumor_risk: float
    source_reliability: float
    fallback_used: bool = False
    parser_mode: str = "structured"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyEvent:
    """Structured policy event produced by the parser."""

    policy_id: str
    raw_text: str
    policy_type: str
    policy_label: str
    direction: str
    intensity: float
    tick: int
    matched_tokens: List[str] = field(default_factory=list)
    implementation_timing: str = "immediate"
    target_scope: List[str] = field(default_factory=list)
    expected_lag_days: int = 0
    action_channels: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "structured_parser"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyPackage:
    """Structured policy package with layered transmission outputs."""

    event: PolicyEvent
    channels: List[TransmissionChannel]
    uncertainty: PolicyUncertainty
    sector_effects: Dict[str, float]
    factor_effects: Dict[str, float]
    agent_class_effects: Dict[str, float]
    market_effects: Dict[str, float]
    policy_schema: Dict[str, Any] = field(default_factory=dict)
    transmission_graph: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parser_version: str = "structured_policy_v1"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["top_layers"] = self.top_layers()
        return payload

    def to_policy_shock_fields(self) -> Dict[str, float]:
        fields = {
            "policy_rate_delta": 0.0,
            "fiscal_stimulus_delta": 0.0,
            "liquidity_injection": 0.0,
            "credit_spread_delta": 0.0,
            "stamp_tax_delta": 0.0,
            "sentiment_delta": 0.0,
            "rumor_shock": 0.0,
        }
        for channel in self.channels:
            for key, value in channel.macro_delta.items():
                if key in fields:
                    fields[key] += float(value)
        return fields

    def top_layers(self, limit: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        def _top(items: Mapping[str, float]) -> List[Tuple[str, float]]:
            ranked = sorted(items.items(), key=lambda kv: abs(kv[1]), reverse=True)
            return [(str(name), float(value)) for name, value in ranked[:limit]]

        return {
            "sector": _top(self.sector_effects),
            "factor": _top(self.factor_effects),
            "agent_class": _top(self.agent_class_effects),
        }


class StructuredPolicyParser:
    """Parse natural-language policy text into a structured policy package."""

    _PATTERNS: Sequence[Tuple[str, str, Tuple[str, ...], str]] = (
        ("liquidity_easing", "Liquidity easing", ("流动性", "注入", "降准", "rrr", "rrr cut", "liquidity"), "easing"),
        ("rate_cut", "Rate cut", ("降息", "下调利率", "cut rate", "rate cut", "policy rate"), "easing"),
        ("fiscal_stimulus", "Fiscal stimulus", ("财政", "补贴", "基建", "stimulus", "fiscal"), "easing"),
        ("tax_cut", "Tax cut", ("印花税下调", "减税", "税率下调", "tax cut", "stamp duty cut"), "easing"),
        ("tax_hike", "Tax hike", ("印花税上调", "加税", "税率上调", "tax hike", "stamp duty hike"), "tightening"),
        ("stabilization", "Stabilization", ("平准", "维稳", "国家队", "backstop", "stabilize"), "stabilizing"),
        ("rumor_refutation", "Rumor refutation", ("辟谣", "澄清", "压制谣言", "refute", "rumor suppression"), "stabilizing"),
        ("tightening", "Tightening", ("收紧", "加息", "tighten", "tightening"), "tightening"),
    )

    def __init__(self, *, feature_flags: Optional[Mapping[str, bool]] = None, seed: int = 0, version: str = "structured_policy_v1") -> None:
        self.feature_flags = dict(feature_flags or {})
        self.seed = int(seed)
        self.version = version

    def parse(
        self,
        policy_text: str,
        *,
        tick: int = 0,
        policy_type_hint: Optional[str] = None,
        intensity: float = 1.0,
        market_regime: Optional[str] = None,
        snapshot_info: Optional[Mapping[str, Any]] = None,
        fallback_used: bool = False,
    ) -> PolicyPackage:
        text = str(policy_text or "").strip()
        text_lower = text.lower()
        numeric_hint = _first_numeric_hint(text)
        base_intensity = _clip(float(intensity) * (1.0 + 0.2 * numeric_hint), 0.0, 2.0)
        preferred_policy_type: Optional[str] = None
        if _match_any(text, "印花税", "stamp duty", "税", "税率", "税费", "减税"):
            if _match_any(text, "上调", "提高", "加税", "hike", "提高税", "税率上调"):
                preferred_policy_type = "tax_hike"
            elif _match_any(text, "下调", "降低", "减免", "调整", "cut", "lower", "tax cut"):
                preferred_policy_type = "tax_cut"
        if preferred_policy_type is None and _match_any(text, "负面谣言", "恐慌", "panic", "selloff", "谣言", "rumor"):
            if not _match_any(text, "辟谣", "澄清", "压制", "refut", "稳定", "维稳", "rumor suppression"):
                preferred_policy_type = "tightening"

        matches: List[Tuple[str, str, str]] = []
        for policy_type, label, tokens, direction in self._PATTERNS:
            hits = [token for token in tokens if token in text or token in text_lower]
            if hits:
                matches.append((policy_type, label, direction))

        if policy_type_hint:
            hint = str(policy_type_hint).strip().lower()
            for policy_type, label, _, direction in self._PATTERNS:
                if hint == policy_type or hint == label.lower():
                    matches.insert(0, (policy_type, label, direction))
                    break

        if preferred_policy_type:
            for idx, item in enumerate(matches):
                if item[0] == preferred_policy_type:
                    matches.insert(0, matches.pop(idx))
                    break
            else:
                for policy_type, label, _, direction in self._PATTERNS:
                    if policy_type == preferred_policy_type:
                        matches.insert(0, (policy_type, label, direction))
                        break

        if matches:
            policy_type, policy_label, direction = matches[0]
        else:
            policy_type, policy_label, direction = "unclassified", "Unclassified", "neutral"
            fallback_used = True

        if len(matches) > 1 and policy_type != "unclassified":

            ambiguous = True
        else:
            ambiguous = False

        channels = self._build_channels(policy_type, base_intensity, direction, text)
        sector_effects, factor_effects, agent_class_effects, market_effects = self._build_layers(
            policy_type=policy_type,
            direction=direction,
            channels=channels,
            text=text,
            market_regime=market_regime,
        )

        confidence = 0.40 + 0.12 * len(matches) + 0.12 * len([token for token in self._PATTERNS if _match_any(text, *token[2])])
        if policy_type_hint:
            confidence += 0.08
        if fallback_used:
            confidence -= 0.18
        if ambiguous:
            confidence -= 0.12
        confidence = _clip(confidence, 0.10, 0.96)

        event = PolicyEvent(
            policy_id=f"policy-{tick}-{uuid.uuid4().hex[:8]}",
            raw_text=text,
            policy_type=policy_type,
            policy_label=policy_label,
            direction=direction,
            intensity=base_intensity,
            tick=int(tick),
            matched_tokens=self._matched_tokens(text),
            implementation_timing=self._implementation_timing(policy_type),
            target_scope=self._target_scope(policy_type, text),
            expected_lag_days=max((int(channel.lag_days) for channel in channels), default=0),
            action_channels=[channel.name for channel in channels],
            side_effects=self._side_effects(policy_type, direction),
            source="structured_parser" if not fallback_used else "legacy_fallback",
        )

        uncertainty = PolicyUncertainty(
            confidence=confidence,
            ambiguity=_clip(1.0 - confidence + (0.18 if ambiguous else 0.0), 0.0, 1.0),
            rumor_risk=_clip(0.25 + 0.35 * (1.0 - confidence) + (0.15 if "rumor" in text_lower or "谣言" in text else 0.0), 0.0, 1.0),
            source_reliability=_clip(0.90 if policy_type_hint else 0.75, 0.0, 1.0),
            fallback_used=bool(fallback_used),
            parser_mode="structured" if not fallback_used else "legacy",
        )

        snapshot = dict(snapshot_info or {})
        snapshot.setdefault("policy_text_length", len(text))
        snapshot.setdefault("matched_token_count", len(event.matched_tokens))
        snapshot.setdefault("market_regime", market_regime or "neutral")
        snapshot.setdefault("parser_mode", uncertainty.parser_mode)

        metadata = {
            "seed": int(self.seed),
            "config_hash": stable_json_hash(
                {
                    "seed": int(self.seed),
                    "version": self.version,
                    "policy_type_hint": policy_type_hint or "",
                    "policy_type": policy_type,
                    "policy_text": text,
                    "intensity": float(base_intensity),
                    "market_regime": market_regime or "neutral",
                    "feature_flags": dict(self.feature_flags),
                }
            ),
            "snapshot_info": snapshot,
            "feature_flags": dict(self.feature_flags),
            "parser_version": self.version,
        }

        policy_schema = self._build_policy_schema(event)
        transmission_graph = self._build_transmission_graph(
            event=event,
            channels=channels,
            sector_effects=sector_effects,
            factor_effects=factor_effects,
            agent_class_effects=agent_class_effects,
            market_effects=market_effects,
        )
        explanation = self._build_explanation(
            event=event,
            transmission_graph=transmission_graph,
            sector_effects=sector_effects,
            factor_effects=factor_effects,
            agent_class_effects=agent_class_effects,
            market_effects=market_effects,
        )

        return PolicyPackage(
            event=event,
            channels=channels,
            uncertainty=uncertainty,
            sector_effects=sector_effects,
            factor_effects=factor_effects,
            agent_class_effects=agent_class_effects,
            market_effects=market_effects,
            policy_schema=policy_schema,
            transmission_graph=transmission_graph,
            explanation=explanation,
            metadata=metadata,
            parser_version=self.version,
        )

    def _implementation_timing(self, policy_type: str) -> str:
        if policy_type in {"tax_cut", "tax_hike", "stabilization", "rumor_refutation"}:
            return "announcement_day"
        if policy_type in {"liquidity_easing", "rate_cut", "tightening"}:
            return "t+1 rollout"
        if policy_type == "fiscal_stimulus":
            return "phased_rollout"
        return "unspecified"

    def _target_scope(self, policy_type: str, text: str) -> List[str]:
        scopes: List[str] = []
        if any(token in text.lower() for token in ["bank", "credit", "financing", "margin", "融资"]):
            scopes.append("funding_conditions")
        if any(token in text.lower() for token in ["property", "real estate", "地产"]):
            scopes.append("property")
        if any(token in text.lower() for token in ["equity", "stock", "market", "股市", "市场"]):
            scopes.append("equity_market")
        if policy_type in {"stabilization", "rumor_refutation"}:
            scopes.append("expectations_management")
        if policy_type in {"fiscal_stimulus"}:
            scopes.append("real_economy")
        if not scopes:
            scopes.append("broad_market")
        return list(dict.fromkeys(scopes))

    def _side_effects(self, policy_type: str, direction: str) -> List[str]:
        side_effect_map = {
            "liquidity_easing": ["asset_valuation_repricing", "leverage_rebound"],
            "rate_cut": ["duration_extension", "sector_rotation"],
            "fiscal_stimulus": ["crowding_in", "execution_lag"],
            "tax_cut": ["turnover_spike", "speculation_rebound"],
            "tax_hike": ["liquidity_drop", "risk_aversion_jump"],
            "stabilization": ["moral_hazard", "state_flow_dependence"],
            "rumor_refutation": ["credibility_test", "short_squeeze"],
            "tightening": ["funding_stress", "valuation_compression"],
        }
        effects = list(side_effect_map.get(policy_type, ["interpretation_uncertainty"]))
        if direction == "tightening":
            effects.append("margin_constraint")
        return list(dict.fromkeys(effects))

    def _build_policy_schema(self, event: PolicyEvent) -> Dict[str, Any]:
        return {
            "schema_version": "policy_schema_v1",
            "policy_id": event.policy_id,
            "policy_type": event.policy_type,
            "policy_label": event.policy_label,
            "direction": event.direction,
            "intensity": float(event.intensity),
            "implementation_timing": event.implementation_timing,
            "target_scope": list(event.target_scope),
            "expected_lag_days": int(event.expected_lag_days),
            "action_channels": list(event.action_channels),
            "side_effects": list(event.side_effects),
            "source": event.source,
            "tick": int(event.tick),
        }

    def _build_transmission_graph(
        self,
        *,
        event: PolicyEvent,
        channels: Sequence[TransmissionChannel],
        sector_effects: Mapping[str, float],
        factor_effects: Mapping[str, float],
        agent_class_effects: Mapping[str, float],
        market_effects: Mapping[str, float],
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = [
            {
                "id": f"policy:{event.policy_id}",
                "layer": "policy",
                "label": event.policy_label,
                "type": event.policy_type,
                "intensity": float(event.intensity),
            }
        ]
        edges: List[Dict[str, Any]] = []
        for channel in channels:
            channel_id = f"channel:{channel.name}"
            nodes.append(
                {
                    "id": channel_id,
                    "layer": "channel",
                    "label": channel.name,
                    "impact": float(channel.impact),
                    "lag_days": int(channel.lag_days),
                    "channel_type": channel.channel_type,
                }
            )
            edges.append(
                {
                    "source": f"policy:{event.policy_id}",
                    "target": channel_id,
                    "weight": float(channel.impact),
                    "lag_days": int(channel.lag_days),
                    "relation": "activates",
                }
            )

        top_agents = sorted(agent_class_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_markets = sorted(market_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        behavior_templates = {
            "retail": "risk_appetite_shift",
            "institution": "allocation_rebalance",
            "market_maker": "quote_width_adjustment",
            "state_stabilization": "supportive_bid_bias",
            "rumor_trader": "sentiment_chasing",
            "etf_arbitrageur": "basket_hedging",
            "passive_fund": "tracking_flow_rebalance",
            "policy_capital": "stability_support",
            "leveraged_capital": "margin_repricing",
            "foreign_proxy": "cross_border_flow",
            "quant": "spread_capture",
        }
        for agent_name, weight in top_agents:
            agent_id = f"agent:{agent_name}"
            behavior_id = f"behavior:{agent_name}"
            order_flow_id = f"order_flow:{agent_name}"
            nodes.extend(
                [
                    {"id": agent_id, "layer": "agent", "label": agent_name, "weight": float(weight)},
                    {
                        "id": behavior_id,
                        "layer": "behavior_variable",
                        "label": behavior_templates.get(agent_name, "position_adjustment"),
                        "weight": float(weight),
                    },
                    {"id": order_flow_id, "layer": "order_flow", "label": "net_order_flow", "weight": float(weight)},
                ]
            )
            for channel in channels[: min(3, len(channels))]:
                edges.append(
                    {
                        "source": f"channel:{channel.name}",
                        "target": agent_id,
                        "weight": float(weight * channel.impact),
                        "lag_days": int(channel.lag_days),
                        "relation": "influences",
                    }
                )
            edges.extend(
                [
                    {"source": agent_id, "target": behavior_id, "weight": float(weight), "lag_days": 0, "relation": "changes"},
                    {"source": behavior_id, "target": order_flow_id, "weight": float(weight), "lag_days": 0, "relation": "generates"},
                ]
            )
            for market_name, market_weight in top_markets:
                market_id = f"market:{market_name}"
                if not any(node["id"] == market_id for node in nodes):
                    nodes.append(
                        {"id": market_id, "layer": "market_result", "label": market_name, "weight": float(market_weight)}
                    )
                edges.append(
                    {
                        "source": order_flow_id,
                        "target": market_id,
                        "weight": float(weight * market_weight),
                        "lag_days": 0,
                        "relation": "moves",
                    }
                )

        info_network = [
            {"channel": channel.name, "speed": max(1, int(channel.lag_days)), "coverage": "broad"}
            for channel in channels
            if channel.name in {"authority_signal", "risk_appetite", "sector_preference", "compliance_intensity"}
        ]
        funding_network = [
            {"channel": channel.name, "speed": max(1, int(channel.lag_days)), "constraint": "balance_sheet"}
            for channel in channels
            if channel.name in {"financing_cost", "liquidity_supply", "margin_leverage"}
        ]
        primary_path = []
        if channels and top_agents and top_markets:
            primary_path = [
                event.policy_label,
                channels[0].name,
                top_agents[0][0],
                behavior_templates.get(top_agents[0][0], "position_adjustment"),
                "net_order_flow",
                top_markets[0][0],
            ]
        return {
            "schema_version": "transmission_graph_v1",
            "nodes": nodes,
            "edges": edges,
            "network_layers": {
                "information_network": info_network,
                "credit_funding_network": funding_network,
            },
            "primary_path": primary_path,
            "layer_summaries": {
                "sector": sorted(sector_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3],
                "factor": sorted(factor_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3],
                "agent_class": top_agents,
                "market": top_markets,
            },
        }

    def _build_explanation(
        self,
        *,
        event: PolicyEvent,
        transmission_graph: Mapping[str, Any],
        sector_effects: Mapping[str, float],
        factor_effects: Mapping[str, float],
        agent_class_effects: Mapping[str, float],
        market_effects: Mapping[str, float],
    ) -> Dict[str, Any]:
        top_sector = sorted(sector_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        top_factor = sorted(factor_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        top_agent = sorted(agent_class_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        top_market = sorted(market_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        return {
            "headline": f"{event.policy_label} mainly works through {', '.join(event.action_channels[:2]) or 'market expectations'}",
            "primary_path": list(transmission_graph.get("primary_path", [])),
            "affected_agents": [{"name": name, "score": float(score)} for name, score in top_agent],
            "affected_sectors": [{"name": name, "score": float(score)} for name, score in top_sector],
            "affected_factors": [{"name": name, "score": float(score)} for name, score in top_factor],
            "market_results": [{"name": name, "score": float(score)} for name, score in top_market],
            "expected_lag_days": int(event.expected_lag_days),
            "side_effects": list(event.side_effects),
            "why_this_happened": {
                "policy": event.policy_label,
                "channels": list(event.action_channels[:3]),
                "agents": [name for name, _ in top_agent],
                "lags": {
                    "announcement": 0,
                    "channel": min((int(edge.get("lag_days", 0)) for edge in transmission_graph.get("edges", []) if str(edge.get("source", "")).startswith("policy:")), default=0),
                    "market": int(event.expected_lag_days),
                },
            },
        }

    def _matched_tokens(self, text: str) -> List[str]:
        tokens: List[str] = []
        for _, _, keywords, _ in self._PATTERNS:
            tokens.extend([token for token in keywords if token in text or token in text.lower()])

        seen: set[str] = set()
        unique: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique

    def _build_channels(
        self,
        policy_type: str,
        intensity: float,
        direction: str,
        text: str,
    ) -> List[TransmissionChannel]:
        sign = -1.0 if direction == "tightening" else 1.0
        if policy_type == "tax_hike":
            sign = -1.0
        if policy_type == "unclassified":
            sign = 0.0

        def channel(name: str, impact: float, lag: int, half_life: float, macro_delta: Dict[str, float], note: str) -> TransmissionChannel:
            scaled_delta = {k: float(v * intensity * sign) for k, v in macro_delta.items()}
            return TransmissionChannel(
                name=name,
                impact=_clip(impact * intensity * sign, -1.0, 1.0),
                lag_days=int(lag),
                decay_half_life=float(half_life),
                direction="supportive" if impact * intensity * sign >= 0 else "restrictive",
                channel_type=name,
                macro_delta=scaled_delta,
                note=note,
            )

        def _with_legacy_aliases(base_channels: List[TransmissionChannel]) -> List[TransmissionChannel]:
            """Append legacy channel aliases so old assertions/UI remain compatible."""
            by_name = {c.name: c for c in base_channels}

            def alias(alias_name: str, src_name: str, *, note: str) -> Optional[TransmissionChannel]:
                src = by_name.get(src_name)
                if src is None:
                    return None
                return TransmissionChannel(
                    name=alias_name,
                    impact=float(src.impact),
                    lag_days=int(src.lag_days),
                    decay_half_life=float(src.decay_half_life),
                    direction=src.direction,
                    channel_type="legacy_alias",
                    macro_delta=dict(src.macro_delta),
                    note=note,
                )

            aliases: List[TransmissionChannel] = []
            for alias_spec in (
                ("liquidity", "liquidity_supply", "Legacy alias for liquidity_supply"),
                ("rate", "financing_cost", "Legacy alias for financing_cost"),
                ("credit_spread", "financing_cost", "Legacy alias for financing_cost"),
            ):
                ch = alias(alias_spec[0], alias_spec[1], note=alias_spec[2])
                if ch is not None:
                    aliases.append(ch)

            if policy_type in {"tax_cut", "tax_hike"}:
                ch = alias("tax_frictions", "compliance_intensity", note="Legacy alias for tax friction channel")
                if ch is not None:
                    aliases.append(ch)
            if policy_type in {"rumor_refutation", "stabilization"}:
                for alias_spec in (
                    ("rumor_suppression", "compliance_intensity", "Legacy alias for rumor suppression"),
                    ("sentiment_confidence", "authority_signal", "Legacy alias for confidence signaling"),
                ):
                    ch = alias(alias_spec[0], alias_spec[1], note=alias_spec[2])
                    if ch is not None:
                        aliases.append(ch)
            return [*base_channels, *aliases]

        if policy_type in {"liquidity_easing", "rate_cut", "stabilization", "rumor_refutation"}:
            return _with_legacy_aliases([
                channel("liquidity_supply", 0.85, 0, 7.0, {"liquidity_injection": 0.08, "sentiment_delta": 0.05}, "Liquidity support"),
                channel("financing_cost", 0.70 if policy_type != "stabilization" else 0.35, 1, 10.0, {"policy_rate_delta": -0.0015, "credit_spread_delta": -0.0008}, "Lower financing cost"),
                channel("risk_appetite", 0.55, 1, 6.0, {"sentiment_delta": 0.05, "rumor_shock": 0.02}, "Risk appetite repair"),
                channel("margin_leverage", 0.45, 2, 8.0, {"credit_spread_delta": -0.0015, "liquidity_injection": 0.02}, "Margin conditions ease"),
                channel("authority_signal", 0.60 if policy_type != "stabilization" else 0.78, 0, 5.0, {"sentiment_delta": 0.08}, "Official confidence signal"),
                channel("compliance_intensity", 0.38 if policy_type != "stabilization" else 0.55, 0, 4.0, {"rumor_shock": 0.12, "sentiment_delta": 0.04}, "Compliance / rumor suppression"),
            ])

        if policy_type == "fiscal_stimulus":
            return _with_legacy_aliases([
                channel("sector_preference", 0.90, 2, 14.0, {"fiscal_stimulus_delta": 0.06, "sentiment_delta": 0.05}, "Sector demand support"),
                channel("liquidity_supply", 0.35, 1, 8.0, {"liquidity_injection": 0.03}, "Secondary liquidity effect"),
                channel("risk_appetite", 0.55, 0, 6.0, {"sentiment_delta": 0.06}, "Confidence uplift"),
                channel("financing_cost", 0.20, 2, 10.0, {"credit_spread_delta": -0.0006}, "Mild credit improvement"),
                channel("authority_signal", 0.28, 0, 7.0, {"sentiment_delta": 0.03}, "Policy commitment signal"),
            ])

        if policy_type in {"tax_cut", "tax_hike"}:
            return _with_legacy_aliases([
                channel("compliance_intensity", 0.90, 0, 5.0, {"stamp_tax_delta": -0.0005, "liquidity_injection": 0.03, "sentiment_delta": 0.03}, "Trading friction repricing"),
                channel("liquidity_supply", 0.30, 0, 6.0, {"liquidity_injection": 0.02}, "Turnover response"),
                channel("risk_appetite", 0.35, 0, 5.0, {"sentiment_delta": 0.04}, "Interpretation effect"),
                channel("sector_preference", 0.20, 1, 5.0, {"sentiment_delta": 0.02}, "Participation shift"),
            ])

        if policy_type == "tightening":
            return _with_legacy_aliases([
                channel("financing_cost", 0.80, 1, 10.0, {"policy_rate_delta": -0.0018, "credit_spread_delta": -0.0010}, "Higher financing cost"),
                channel("liquidity_supply", 0.65, 0, 8.0, {"liquidity_injection": 0.07, "sentiment_delta": 0.04}, "Liquidity withdrawal"),
                channel("margin_leverage", 0.55, 1, 12.0, {"credit_spread_delta": -0.0022}, "Funding stress"),
                channel("risk_appetite", 0.55, 1, 7.0, {"sentiment_delta": 0.05, "rumor_shock": 0.03}, "Risk repricing"),
                channel("authority_signal", 0.50, 0, 6.0, {"sentiment_delta": 0.07}, "Tighter official stance"),
                channel("compliance_intensity", 0.25, 1, 4.0, {"rumor_shock": 0.05}, "Compliance cooling"),
            ])

        return _with_legacy_aliases([
            channel("authority_signal", 0.0, 0, 5.0, {"sentiment_delta": 0.0}, "Neutral fallback"),
        ])

    def _build_layers(
        self,
        *,
        policy_type: str,
        direction: str,
        channels: Sequence[TransmissionChannel],
        text: str,
        market_regime: Optional[str],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        channel_map = {channel.name: channel.impact for channel in channels}

        sector_weights = {
            "financials": {"liquidity_supply": 0.55, "financing_cost": -0.55, "margin_leverage": -0.45, "authority_signal": 0.18},
            "growth": {"liquidity_supply": 0.62, "financing_cost": 0.48, "sector_preference": 0.28, "risk_appetite": 0.35},
            "consumer": {"sector_preference": 0.62, "risk_appetite": 0.24, "liquidity_supply": 0.12, "compliance_intensity": -0.08},
            "defensive": {"risk_appetite": -0.28, "authority_signal": 0.22, "compliance_intensity": 0.16},
            "cyclical": {"liquidity_supply": 0.35, "sector_preference": 0.56, "financing_cost": 0.22, "margin_leverage": -0.18},
            "property": {"financing_cost": 0.62, "liquidity_supply": 0.28, "margin_leverage": -0.24, "authority_signal": 0.12},
            "state_owned": {"authority_signal": 0.55, "liquidity_supply": 0.28, "risk_appetite": 0.18, "compliance_intensity": 0.14},
            "speculative": {"compliance_intensity": -0.62, "risk_appetite": 0.45, "margin_leverage": -0.28, "liquidity_supply": 0.15},
        }

        factor_weights = {
            "value": {"compliance_intensity": 0.18, "financing_cost": 0.22, "margin_leverage": 0.16, "authority_signal": 0.08},
            "growth": {"liquidity_supply": 0.42, "financing_cost": 0.35, "sector_preference": 0.28, "risk_appetite": 0.20},
            "momentum": {"risk_appetite": 0.58, "liquidity_supply": 0.22, "authority_signal": 0.18},
            "quality": {"financing_cost": 0.36, "authority_signal": 0.22, "compliance_intensity": 0.10},
            "low_vol": {"risk_appetite": -0.62, "authority_signal": 0.18, "liquidity_supply": 0.10},
            "liquidity": {"liquidity_supply": 0.88, "margin_leverage": 0.16},
            "size": {"liquidity_supply": 0.18, "sector_preference": 0.18, "compliance_intensity": -0.08},
            "sentiment": {"risk_appetite": 0.82, "authority_signal": 0.38, "compliance_intensity": -0.18},
            "turnover": {"compliance_intensity": -0.58, "liquidity_supply": 0.24, "risk_appetite": 0.16},
        }

        agent_weights = {
            "retail": {"risk_appetite": 0.65, "compliance_intensity": -0.22, "authority_signal": 0.18},
            "institution": {"liquidity_supply": 0.30, "financing_cost": 0.32, "authority_signal": 0.22, "compliance_intensity": 0.18},
            "market_maker": {"liquidity_supply": 0.52, "risk_appetite": -0.40, "margin_leverage": 0.18, "compliance_intensity": -0.12},
            "state_stabilization": {"authority_signal": 0.86, "liquidity_supply": 0.46, "risk_appetite": 0.32, "compliance_intensity": 0.38},
            "rumor_trader": {"authority_signal": -0.76, "risk_appetite": 0.42, "compliance_intensity": -0.22},
            "etf_arbitrageur": {"liquidity_supply": 0.55, "financing_cost": 0.18, "compliance_intensity": -0.15},
            "passive_fund": {"liquidity_supply": 0.36, "sector_preference": 0.20, "compliance_intensity": 0.28},
            "policy_capital": {"authority_signal": 0.92, "liquidity_supply": 0.40, "compliance_intensity": 0.36},
            "leveraged_capital": {"margin_leverage": -0.62, "financing_cost": -0.35, "risk_appetite": 0.25},
            "foreign_proxy": {"financing_cost": 0.24, "risk_appetite": 0.26, "authority_signal": 0.22},
            "quant": {"liquidity_supply": 0.30, "compliance_intensity": -0.12, "margin_leverage": 0.18},
        }

        sector_effects = {name: _layer_score(channel_map, weights) for name, weights in sector_weights.items()}
        factor_effects = {name: _layer_score(channel_map, weights) for name, weights in factor_weights.items()}
        agent_class_effects = {name: _layer_score(channel_map, weights) for name, weights in agent_weights.items()}
        fine_agent_aliases = {
            "retail_general": "retail",
            "retail_value": "retail",
            "retail_emotional": "retail",
            "retail_day_trader": "retail",
            "retail_swing": "retail",
            "retail_momentum_chaser": "retail",
            "retail_mean_reverter": "retail",
            "mutual_fund": "institution",
            "private_fund_discretionary": "institution",
            "long_term_institution": "institution",
            "pension_fund": "institution",
            "insurer": "institution",
            "quant_stock_selector": "quant",
            "quant_timing": "quant",
            "trend_trader": "quant",
            "event_driven_capital": "institution",
            "media_propagator": "rumor_trader",
            "state_stabilization_fund": "state_stabilization",
        }
        for alias, coarse in fine_agent_aliases.items():
            if coarse in agent_class_effects:
                agent_class_effects[alias] = float(agent_class_effects[coarse])


        regime_bias = {
            "risk_off": -0.10,
            "risk_on": 0.10,
            "inflation": -0.08,
            "liquidity": 0.12,
            "neutral": 0.0,
        }.get(str(market_regime or "neutral").lower(), 0.0)
        market_effects = {
            "market_bias": _clip(sum(channel_map.values()) + regime_bias, -1.0, 1.0),
            "liquidity_bias": _clip(channel_map.get("liquidity_supply", 0.0), -1.0, 1.0),
            "confidence_bias": _clip(channel_map.get("risk_appetite", 0.0) + channel_map.get("authority_signal", 0.0), -1.0, 1.0),
            "volatility_bias": _clip(-channel_map.get("margin_leverage", 0.0) + channel_map.get("compliance_intensity", 0.0) * 0.25, -1.0, 1.0),
        }
        if policy_type == "stabilization":
            market_effects["market_bias"] = _clip(market_effects["market_bias"] + 0.12, -1.0, 1.0)
        if direction == "tightening":
            market_effects["market_bias"] = _clip(market_effects["market_bias"] - 0.08, -1.0, 1.0)
        if "谣言" in text or "rumor" in text.lower():
            market_effects["volatility_bias"] = _clip(market_effects["volatility_bias"] - 0.06, -1.0, 1.0)

        return sector_effects, factor_effects, agent_class_effects, market_effects


def _layer_score(channel_map: Mapping[str, float], weights: Mapping[str, float]) -> float:
    score = 0.0
    for channel_name, weight in weights.items():
        score += float(channel_map.get(channel_name, 0.0)) * float(weight)
    return _clip(score, -1.0, 1.0)
