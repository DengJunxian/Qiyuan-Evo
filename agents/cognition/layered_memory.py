"""Layered memory and behavior card helpers for trader agents."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from core.behavioral_finance import InstitutionConstraintProfile, build_institution_constraint_profile


def _stable_json_hash(payload: Any) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    except TypeError:
        encoded = json.dumps(str(payload), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _market_lookup(market_state: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(market_state, Mapping):
        return market_state.get(key, default)
    return getattr(market_state, key, default)


@dataclass
class EpisodicMemoryItem:
    timestamp: float
    summary: str
    outcome: float
    source: str = "market"
    policy_tag: str = ""
    event_type: str = "decision"
    memory_horizon: str = "short_term"
    salience: float = 0.5
    confidence: float = 0.5
    decay: float = 0.92
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorCard:
    agent_id: str
    enabled: bool
    current_belief: str
    key_memories: List[str]
    current_constraints: Dict[str, Any]
    current_risk_budget: float
    current_narrative: str
    source_credibility: Dict[str, float]
    attention_fatigue: float
    imitation_threshold: float
    panic_threshold: float
    policy_delay_remaining: int
    seed: int
    config_hash: str
    snapshot_info: Dict[str, Any]
    memory_layers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayeredMemory:
    """Minimal layered memory with episodic, semantic, and procedural state."""

    agent_id: str
    seed: int = 0
    enabled: bool = False
    institution_type: str = "retail_swing"
    config: Dict[str, Any] = field(default_factory=dict)

    episodic: List[EpisodicMemoryItem] = field(default_factory=list)
    semantic: Dict[str, Any] = field(default_factory=dict)
    procedural: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self._rng = random.Random(self.seed)
        self.constraint_profile: InstitutionConstraintProfile = build_institution_constraint_profile(self.institution_type)
        self.config_hash = _stable_json_hash(
            {
                "agent_id": self.agent_id,
                "seed": self.seed,
                "enabled": bool(self.enabled),
                "institution_type": self.institution_type,
                "config": dict(self.config),
            }
        )
        self.semantic = {
            "source_credibility": {},
            "themes": {},
            "mid_term_narratives": {},
            "narrative_anchor": "",
            "policy_signature": "",
            "policy_delay_remaining": 0,
            "last_belief": "",
            "long_term_style": {
                "institution_type": self.institution_type,
                "dominant_theme": "neutral",
                "style_conviction": 0.50,
                "credibility_anchor": 0.50,
            },
            "memory_decay": {
                "short_term": 0.90,
                "mid_term": 0.95,
                "long_term": 0.985,
            },
        }
        self.procedural = {
            "base_attention_span": 3,
            "recent_loss_streak": 0,
            "recent_win_streak": 0,
            "trauma_level": 0.0,
            "attention_fatigue": 0.0,
        }

    def _episode_salience(self, *, outcome: float, event_type: str, policy_tag: str, source: str) -> float:
        salience = 0.35 + min(0.45, abs(float(outcome)) * 0.80)
        if event_type in {"loss", "gain"}:
            salience += 0.10
        if policy_tag:
            salience += 0.08
        if source not in {"market", ""}:
            salience += 0.05
        return _clip(salience, 0.05, 1.0)

    def _episode_confidence(self, *, source: str, outcome: float) -> float:
        credibility = self.semantic.setdefault("source_credibility", {})
        base = _safe_float(credibility.get(source, 0.5), 0.5)
        if outcome > 0:
            base = 0.60 * base + 0.40 * 0.80
        elif outcome < 0:
            base = 0.70 * base + 0.30 * 0.30
        return _clip(base, 0.05, 1.0)

    def _decay_memory_layers(self) -> None:
        decay_cfg = dict(self.semantic.get("memory_decay", {}))
        short_decay = _safe_float(decay_cfg.get("short_term", 0.90), 0.90)
        mid_decay = _safe_float(decay_cfg.get("mid_term", 0.95), 0.95)
        long_decay = _safe_float(decay_cfg.get("long_term", 0.985), 0.985)

        pruned: List[EpisodicMemoryItem] = []
        for item in self.episodic:
            item.salience = _clip(float(item.salience) * float(item.decay), 0.0, 1.0)
            if item.salience >= 0.03:
                pruned.append(item)
        self.episodic = pruned[-60:]

        themes = self.semantic.setdefault("themes", {})
        for key in list(themes.keys()):
            themes[key] = float(themes[key]) * mid_decay
            if themes[key] < 0.02:
                themes.pop(key, None)

        narratives = self.semantic.setdefault("mid_term_narratives", {})
        for key in list(narratives.keys()):
            narratives[key] = float(narratives[key]) * mid_decay
            if narratives[key] < 0.02:
                narratives.pop(key, None)

        long_term_style = self.semantic.setdefault("long_term_style", {})
        long_term_style["style_conviction"] = _clip(
            _safe_float(long_term_style.get("style_conviction", 0.50), 0.50) * long_decay,
            0.10,
            1.0,
        )
        long_term_style["credibility_anchor"] = _clip(
            _safe_float(long_term_style.get("credibility_anchor", 0.50), 0.50) * long_decay + 0.01,
            0.10,
            1.0,
        )
        if themes:
            self.semantic["narrative_anchor"] = max(themes.items(), key=lambda kv: kv[1])[0]

    def _memory_layers_snapshot(self) -> Dict[str, Any]:
        short_items = [
            {
                "summary": item.summary,
                "salience": round(float(item.salience), 4),
                "confidence": round(float(item.confidence), 4),
                "horizon": item.memory_horizon,
            }
            for item in sorted(self.episodic[-8:], key=lambda entry: float(entry.salience), reverse=True)[:3]
        ]
        mid_term = dict(
            sorted(
                (
                    (str(key), round(float(value), 4))
                    for key, value in dict(self.semantic.get("mid_term_narratives", {})).items()
                ),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:3]
        )
        long_term_style = dict(self.semantic.get("long_term_style", {}))
        return {
            "short_term": short_items or [{"summary": "cold_start", "salience": 0.0, "confidence": 0.5, "horizon": "short_term"}],
            "mid_term": mid_term,
            "long_term": long_term_style,
        }

    def _append_episode(
        self,
        *,
        summary: str,
        outcome: float,
        source: str = "market",
        policy_tag: str = "",
        event_type: str = "decision",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._decay_memory_layers()
        salience = self._episode_salience(outcome=outcome, event_type=event_type, policy_tag=policy_tag, source=source)
        confidence = self._episode_confidence(source=source, outcome=outcome)
        item = EpisodicMemoryItem(
            timestamp=time.time(),
            summary=str(summary),
            outcome=float(outcome),
            source=str(source or "market"),
            policy_tag=str(policy_tag or ""),
            event_type=str(event_type or "decision"),
            memory_horizon="short_term",
            salience=salience,
            confidence=confidence,
            decay=_safe_float(self.semantic.get("memory_decay", {}).get("short_term", 0.90), 0.90),
            metadata=dict(metadata or {}),
        )
        self.episodic.append(item)
        if len(self.episodic) > 60:
            self.episodic = self.episodic[-60:]

        theme_key = self._extract_theme(item.summary, item.policy_tag)
        if theme_key:
            themes = self.semantic.setdefault("themes", {})
            themes[theme_key] = float(themes.get(theme_key, 0.0)) + float(item.salience)
            narratives = self.semantic.setdefault("mid_term_narratives", {})
            narratives[theme_key] = float(narratives.get(theme_key, 0.0)) + float(item.salience * 0.60 + item.confidence * 0.40)
            self.semantic["narrative_anchor"] = max(themes.items(), key=lambda kv: kv[1])[0]
            long_term_style = self.semantic.setdefault("long_term_style", {})
            dominant_theme = str(long_term_style.get("dominant_theme", "neutral"))
            current_strength = _safe_float(long_term_style.get("style_conviction", 0.50), 0.50)
            if dominant_theme == "neutral" or float(narratives[theme_key]) >= current_strength:
                long_term_style["dominant_theme"] = theme_key
            long_term_style["style_conviction"] = _clip(current_strength * 0.94 + item.salience * 0.12, 0.10, 1.0)

        if item.source:
            credibility = self.semantic.setdefault("source_credibility", {})
            current = _safe_float(credibility.get(item.source, 0.5), 0.5)
            target = 0.75 if item.outcome > 0 else 0.25 if item.outcome < 0 else 0.5
            credibility[item.source] = _clip(current * 0.8 + target * 0.2, 0.0, 1.0)
            long_term_style = self.semantic.setdefault("long_term_style", {})
            long_term_style["credibility_anchor"] = _clip(
                _safe_float(long_term_style.get("credibility_anchor", 0.50), 0.50) * 0.92 + credibility[item.source] * 0.08,
                0.10,
                1.0,
            )

    def _extract_theme(self, summary: str, policy_tag: str) -> str:
        text = f"{summary} {policy_tag}".lower()
        rules = [
            ("policy", ["policy", "policy_shock", "降准", "降息", "印花税", "维稳", "辟谣"]),
            ("panic", ["panic", "fear", "恐慌", "selloff", "崩盘"]),
            ("liquidity", ["liquidity", "流动性", "资金", "资金面"]),
            ("rumor", ["rumor", "谣言", "辟谣"]),
            ("benchmark", ["benchmark", "基准", "tracking"]),
        ]
        for label, tokens in rules:
            if any(token in text for token in tokens):
                return label
        if summary:
            return summary.split()[0][:24]
        return ""

    def _policy_signal(self, market_state: Mapping[str, Any] | Any) -> float:
        text = " ".join(
            [
                str(_market_lookup(market_state, "policy_description", "")),
                str(_market_lookup(market_state, "policy_news", "")),
                str(_market_lookup(market_state, "news", "")),
            ]
        ).lower()
        shock = _safe_float(_market_lookup(market_state, "text_policy_shock", 0.0), 0.0)
        if any(token in text for token in ["降准", "降息", "流动性", "平准资金", "维稳", "支持"]):
            shock += 0.35
        if any(token in text for token in ["印花税", "收紧", "辟谣", "监管", "罚", "降温"]):
            shock -= 0.25
        if str(_market_lookup(market_state, "text_regime_bias", "")).lower() in {"bullish", "easing", "supportive"}:
            shock += 0.15
        if str(_market_lookup(market_state, "text_regime_bias", "")).lower() in {"bearish", "tightening", "restrictive"}:
            shock -= 0.15
        return _clip(shock, -1.0, 1.0)

    def _social_signal(self, social_signal: str) -> float:
        text = str(social_signal or "").lower()
        score = 0.0
        if any(token in text for token in ["panic", "selling", "bear", "rumor", "nervous", "恐慌"]):
            score -= 0.45
        if any(token in text for token in ["buy", "bull", "fomo", "support", "资金", "买入"]):
            score += 0.45
        return _clip(score, -1.0, 1.0)

    def _risk_budget(
        self,
        *,
        account_state: Mapping[str, Any] | Any,
        market_state: Mapping[str, Any] | Any,
    ) -> float:
        pnl_pct = _safe_float(_market_lookup(account_state, "pnl_pct", 0.0), 0.0)
        panic_level = _safe_float(_market_lookup(market_state, "panic_level", 0.0), 0.0)
        base = 1.0
        base -= self.procedural.get("trauma_level", 0.0) * 0.45
        base -= self.procedural.get("attention_fatigue", 0.0) * 0.20
        base -= max(0.0, -pnl_pct) * 1.5
        base -= panic_level * 0.15

        inst = self.constraint_profile
        if self.institution_type in {"mutual_fund", "pension_fund", "insurer"}:
            base *= 0.85 + (inst.liquidity_preference * 0.10)
        elif self.institution_type == "market_maker":
            base *= 0.70
        elif self.institution_type == "state_stabilization_fund":
            base *= 0.95
        elif self.institution_type == "prop_desk":
            base *= 1.05
        return _clip(base, 0.05, 1.0)

    def _update_internal_state(
        self,
        *,
        decision: Mapping[str, Any],
        market_state: Mapping[str, Any] | Any,
        account_state: Mapping[str, Any] | Any,
        emotional_state: str,
        social_signal: str,
    ) -> Dict[str, Any]:
        policy_signal = self._policy_signal(market_state)
        policy_signature = str(
            _market_lookup(market_state, "policy_description", "")
            or _market_lookup(market_state, "policy_news", "")
            or _market_lookup(market_state, "news", "")
        )
        if policy_signature and policy_signature != self.semantic.get("policy_signature"):
            delay = max(0, int(round(self.constraint_profile.policy_channel_sensitivity * 3)))
            self.semantic["policy_delay_remaining"] = max(int(self.semantic.get("policy_delay_remaining", 0)), delay)
            self.semantic["policy_signature"] = policy_signature

        if int(self.semantic.get("policy_delay_remaining", 0)) > 0:
            self.semantic["policy_delay_remaining"] = int(self.semantic.get("policy_delay_remaining", 0)) - 1
            policy_signal *= 0.5

        social_strength = self._social_signal(social_signal)
        panic_level = _safe_float(_market_lookup(market_state, "panic_level", 0.0), 0.0)
        trend = _safe_float(_market_lookup(market_state, "market_trend", 0.0), 0.0)
        source = str(_market_lookup(market_state, "news_source", "market") or "market")

        summary = (
            f"action={decision.get('action', 'HOLD')}; "
            f"trend={trend:.3f}; panic={panic_level:.3f}; "
            f"policy={policy_signal:.3f}; social={social_strength:.3f}; "
            f"emotion={emotional_state}"
        )
        pnl = _safe_float(_market_lookup(account_state, "pnl_pct", 0.0), 0.0)
        if pnl < 0:
            self.procedural["recent_loss_streak"] = int(self.procedural.get("recent_loss_streak", 0)) + 1
            self.procedural["recent_win_streak"] = 0
            self.procedural["trauma_level"] = _clip(
                self.procedural.get("trauma_level", 0.0) * 0.85 + min(1.0, abs(pnl) * 4.0) * 0.30,
                0.0,
                1.0,
            )
        else:
            self.procedural["recent_win_streak"] = int(self.procedural.get("recent_win_streak", 0)) + 1
            self.procedural["recent_loss_streak"] = 0
            self.procedural["trauma_level"] = _clip(self.procedural.get("trauma_level", 0.0) * 0.90, 0.0, 1.0)

        self.procedural["attention_fatigue"] = _clip(
            self.procedural.get("attention_fatigue", 0.0) * 0.90 + 0.05 + min(0.15, abs(panic_level) * 0.08),
            0.0,
            1.0,
        )

        if source:
            credibility = self.semantic.setdefault("source_credibility", {})
            current = _safe_float(credibility.get(source, 0.5), 0.5)
            credibility[source] = _clip(current * 0.9 + (0.8 if pnl >= 0 else 0.2) * 0.1, 0.0, 1.0)

        self._append_episode(
            summary=summary,
            outcome=pnl,
            source=source,
            policy_tag=policy_signature,
            event_type="decision",
            metadata={
                "decision": dict(decision),
                "market_state": dict(market_state) if isinstance(market_state, Mapping) else {},
                "account_state": dict(account_state) if isinstance(account_state, Mapping) else {},
            },
        )

        return {
            "policy_signal": policy_signal,
            "social_signal": social_strength,
            "trauma_level": float(self.procedural.get("trauma_level", 0.0)),
            "attention_fatigue": float(self.procedural.get("attention_fatigue", 0.0)),
            "risk_budget": self._risk_budget(account_state=account_state, market_state=market_state),
            "policy_delay_remaining": int(self.semantic.get("policy_delay_remaining", 0)),
        }

    def record_outcome(
        self,
        *,
        decision: Mapping[str, Any],
        outcome: Mapping[str, Any],
        market_state: Optional[Mapping[str, Any] | Any] = None,
        account_state: Optional[Mapping[str, Any] | Any] = None,
    ) -> None:
        """Update memory after trade feedback."""

        market_state = market_state or {}
        account_state = account_state or {}
        pnl = _safe_float(outcome.get("pnl", 0.0), 0.0)
        pnl_pct = _safe_float(outcome.get("pnl_pct", _market_lookup(outcome, "pnl_pct", 0.0)), 0.0)
        if pnl_pct == 0.0 and pnl != 0.0:
            market_value = _safe_float(_market_lookup(account_state, "market_value", 0.0), 0.0)
            pnl_pct = pnl / max(1.0, market_value if market_value > 0 else abs(pnl) * 10.0)

        source = str(
            outcome.get("source")
            or _market_lookup(market_state, "news_source", "")
            or _market_lookup(market_state, "policy_source", "")
            or "market"
        )
        event_type = "loss" if pnl < 0 else "gain" if pnl > 0 else "flat"
        self._append_episode(
            summary=f"outcome action={decision.get('action', 'HOLD')} pnl={pnl:.2f} pnl_pct={pnl_pct:.3f}",
            outcome=pnl_pct,
            source=source,
            policy_tag=str(_market_lookup(market_state, "policy_description", "")),
            event_type=event_type,
            metadata={"decision": dict(decision), "outcome": dict(outcome)},
        )

    def build_behavior_card(
        self,
        *,
        market_state: Mapping[str, Any] | Any,
        account_state: Mapping[str, Any] | Any,
        decision: Mapping[str, Any],
        emotional_state: str,
        social_signal: str,
        snapshot_info: Optional[Mapping[str, Any]] = None,
    ) -> BehaviorCard:
        """Render a traceable behavior card for downstream logs/tests."""

        state = self._update_internal_state(
            decision=decision,
            market_state=market_state,
            account_state=account_state,
            emotional_state=emotional_state,
            social_signal=social_signal,
        )
        current_belief = (
            f"risk_budget={state['risk_budget']:.2f}; "
            f"trauma={state['trauma_level']:.2f}; "
            f"policy_lag={state['policy_delay_remaining']}; "
            f"attention={state['attention_fatigue']:.2f}"
        )
        memory_layers = self._memory_layers_snapshot()
        key_memories = [item.summary for item in self.episodic[-3:]]
        if not key_memories:
            key_memories = ["cold_start"]
        narrative = self.semantic.get("narrative_anchor") or memory_layers.get("long_term", {}).get("dominant_theme", "neutral")
        if self.institution_type in {"mutual_fund", "pension_fund", "insurer"} and state["policy_delay_remaining"] > 0:
            narrative = f"{narrative}: policy lag"
        elif self.institution_type == "market_maker":
            narrative = f"{narrative}: inventory control"
        elif self.institution_type == "state_stabilization_fund":
            narrative = f"{narrative}: stabilisation bias"

        source_credibility = dict(sorted(self.semantic.get("source_credibility", {}).items(), key=lambda kv: kv[1], reverse=True)[:5])
        behavior_constraints = self.constraint_profile.to_dict()
        behavior_constraints.update(
            {
                "risk_budget": state["risk_budget"],
                "effective_policy_delay": state["policy_delay_remaining"],
                "trauma_level": state["trauma_level"],
                "attention_fatigue": state["attention_fatigue"],
                "memory_layers": memory_layers,
            }
        )

        credibility_term = max(source_credibility.values()) if source_credibility else 0.0
        imitation_threshold = _clip(
            0.55
            + 0.20 * (1.0 - self.constraint_profile.rumor_sensitivity)
            + 0.20 * state["attention_fatigue"]
            - 0.10 * credibility_term,
            0.10,
            0.95,
        )
        panic_threshold = _clip(
            0.70
            - 0.30 * state["trauma_level"]
            - 0.10 * self.constraint_profile.rumor_sensitivity
            - 0.05 * max(0.0, _safe_float(_market_lookup(market_state, "panic_level", 0.0), 0.0)),
            0.10,
            0.95,
        )

        snapshot = dict(snapshot_info or {})
        if not snapshot:
            snapshot = {
                "price": _safe_float(_market_lookup(market_state, "price", _market_lookup(market_state, "last_price", 0.0)), 0.0),
                "pnl_pct": _safe_float(_market_lookup(account_state, "pnl_pct", 0.0), 0.0),
                "policy_description": str(_market_lookup(market_state, "policy_description", "")),
            }

        return BehaviorCard(
            agent_id=self.agent_id,
            enabled=bool(self.enabled),
            current_belief=current_belief,
            key_memories=key_memories,
            current_constraints=behavior_constraints,
            current_risk_budget=float(state["risk_budget"]),
            current_narrative=narrative,
            source_credibility=source_credibility,
            attention_fatigue=float(state["attention_fatigue"]),
            imitation_threshold=float(imitation_threshold),
            panic_threshold=float(panic_threshold),
            policy_delay_remaining=int(state["policy_delay_remaining"]),
            seed=int(self.seed),
            config_hash=self.config_hash,
            snapshot_info=snapshot,
            memory_layers=memory_layers,
            metadata={
                "institution_type": self.institution_type,
                "enabled": bool(self.enabled),
                "behavior_version": "layered_memory_v2",
            },
        )

    def apply_decision_overlay(
        self,
        decision: Mapping[str, Any],
        *,
        market_state: Mapping[str, Any] | Any,
        account_state: Mapping[str, Any] | Any,
        emotional_state: str,
        social_signal: str,
        snapshot_info: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply memory/constraint overlays to a decision payload."""

        decision_out = dict(decision)
        if not self.enabled:
            risk_budget = self._risk_budget(account_state=account_state, market_state=market_state)
            current_delay = int(self.semantic.get("policy_delay_remaining", 0))
            source_credibility = dict(sorted(self.semantic.get("source_credibility", {}).items(), key=lambda kv: kv[1], reverse=True)[:5])
            memory_layers = self._memory_layers_snapshot()
            current_belief = (
                f"risk_budget={risk_budget:.2f}; "
                f"trauma={self.procedural.get('trauma_level', 0.0):.2f}; "
                f"policy_lag={current_delay}; "
                f"attention={self.procedural.get('attention_fatigue', 0.0):.2f}"
            )
            key_memories = [item.summary for item in self.episodic[-3:]] or ["cold_start"]
            narrative = self.semantic.get("narrative_anchor") or memory_layers.get("long_term", {}).get("dominant_theme", "neutral")
            behavior_constraints = self.constraint_profile.to_dict()
            behavior_constraints.update(
                {
                    "risk_budget": risk_budget,
                    "effective_policy_delay": current_delay,
                    "trauma_level": float(self.procedural.get("trauma_level", 0.0)),
                    "attention_fatigue": float(self.procedural.get("attention_fatigue", 0.0)),
                    "memory_layers": memory_layers,
                }
            )
            credibility_term = max(source_credibility.values()) if source_credibility else 0.0
            imitation_threshold = _clip(
                0.55
                + 0.20 * (1.0 - self.constraint_profile.rumor_sensitivity)
                + 0.20 * float(self.procedural.get("attention_fatigue", 0.0))
                - 0.10 * credibility_term,
                0.10,
                0.95,
            )
            panic_threshold = _clip(
                0.70
                - 0.30 * float(self.procedural.get("trauma_level", 0.0))
                - 0.10 * self.constraint_profile.rumor_sensitivity
                - 0.05 * max(0.0, _safe_float(_market_lookup(market_state, "panic_level", 0.0), 0.0)),
                0.10,
                0.95,
            )
            snapshot = dict(snapshot_info or {})
            if not snapshot:
                snapshot = {
                    "price": _safe_float(_market_lookup(market_state, "price", _market_lookup(market_state, "last_price", 0.0)), 0.0),
                    "pnl_pct": _safe_float(_market_lookup(account_state, "pnl_pct", 0.0), 0.0),
                    "policy_description": str(_market_lookup(market_state, "policy_description", "")),
                }
            card = BehaviorCard(
                agent_id=self.agent_id,
                enabled=False,
                current_belief=current_belief,
                key_memories=key_memories,
                current_constraints=behavior_constraints,
                current_risk_budget=float(risk_budget),
                current_narrative=narrative,
                source_credibility=source_credibility,
                attention_fatigue=float(self.procedural.get("attention_fatigue", 0.0)),
                imitation_threshold=float(imitation_threshold),
                panic_threshold=float(panic_threshold),
                policy_delay_remaining=current_delay,
                seed=int(self.seed),
                config_hash=self.config_hash,
                snapshot_info=snapshot,
                memory_layers=memory_layers,
                metadata={
                    "institution_type": self.institution_type,
                    "enabled": False,
                    "behavior_version": "layered_memory_v2",
                },
            )
            return {"decision": decision_out, "behavior_card": card.to_dict(), "behavior_context": {"enabled": False}}

        state = self._update_internal_state(
            decision=decision_out,
            market_state=market_state,
            account_state=account_state,
            emotional_state=emotional_state,
            social_signal=social_signal,
        )
        inst = self.constraint_profile
        action = str(decision_out.get("action", "HOLD")).upper()
        qty = int(decision_out.get("qty", decision_out.get("quantity", 0)) or 0)
        price = _safe_float(decision_out.get("price", _market_lookup(market_state, "price", _market_lookup(market_state, "last_price", 0.0))), 0.0)
        panic_level = _safe_float(_market_lookup(market_state, "panic_level", 0.0), 0.0)
        trend = _safe_float(_market_lookup(market_state, "market_trend", 0.0), 0.0)
        source_credibility = max(self.semantic.get("source_credibility", {}).values(), default=0.5)

        if state["policy_delay_remaining"] > 0:
            qty = int(qty * 0.75)
            if abs(state["policy_signal"]) > 0.25 and self.institution_type in {"mutual_fund", "pension_fund", "insurer"}:
                action = "HOLD"

        if self.procedural["trauma_level"] > 0.55 and action == "BUY":
            qty = int(qty * 0.45)
            if self.procedural["recent_loss_streak"] >= 3:
                action = "HOLD"

        if self.institution_type in {"mutual_fund", "pension_fund", "insurer"}:
            qty = int(qty * (0.55 + 0.45 * state["risk_budget"]))
            if panic_level > 0.55 and action == "BUY":
                qty = int(qty * 0.70)
            if self.procedural["trauma_level"] > 0.35 and trend < 0:
                action = "HOLD"

        elif self.institution_type == "market_maker":
            inventory = _safe_float(_market_lookup(account_state, "inventory", _market_lookup(account_state, "market_value", 0.0)), 0.0)
            inventory_pressure = min(1.0, abs(inventory) / max(1.0, inst.inventory_limit * 1000.0))
            qty = int(qty * max(0.20, 1.0 - inventory_pressure))
            if abs(inventory) > inst.inventory_limit and action == "BUY" and inventory > 0:
                action = "SELL"

        elif self.institution_type == "state_stabilization_fund":
            if panic_level > 0.55 and state["policy_signal"] > 0:
                action = "BUY"
                qty = max(qty, int(max(1.0, _safe_float(_market_lookup(account_state, "cash", 0.0), 0.0) / max(price, 1.0) * 0.05)))
            qty = int(qty * (0.80 + 0.20 * state["risk_budget"]))

        elif self.institution_type == "rumor_trader":
            if social_signal and state["social_signal"] < 0 and self.constraint_profile.rumor_sensitivity > 0.5:
                action = "SELL"
                qty = int(max(qty, 1) * (0.90 + 0.10 * self.constraint_profile.rumor_sensitivity))
            elif source_credibility < 0.35:
                qty = int(qty * 0.55)

        elif self.institution_type == "etf_arbitrageur":
            qty = int(qty * 0.60)
            if state["risk_budget"] < 0.35:
                action = "HOLD"

        elif self.institution_type == "prop_desk":
            qty = int(qty * (0.85 + 0.50 * state["risk_budget"]))

        if qty <= 0 and action != "HOLD":
            action = "HOLD"

        decision_out["action"] = action
        decision_out["qty"] = max(0, qty)
        decision_out["price"] = float(price)
        decision_out["risk_budget"] = float(state["risk_budget"])
        decision_out["policy_delay_remaining"] = int(state["policy_delay_remaining"])
        decision_out["source_credibility"] = float(source_credibility)

        card = self.build_behavior_card(
            market_state=market_state,
            account_state=account_state,
            decision=decision_out,
            emotional_state=emotional_state,
            social_signal=social_signal,
            snapshot_info=snapshot_info,
        )

        return {
            "decision": decision_out,
            "behavior_card": card.to_dict(),
            "behavior_context": {
                "enabled": True,
                "risk_budget": state["risk_budget"],
                "policy_signal": state["policy_signal"],
                "social_signal": state["social_signal"],
                "policy_delay_remaining": state["policy_delay_remaining"],
                "attention_fatigue": state["attention_fatigue"],
            },
        }
