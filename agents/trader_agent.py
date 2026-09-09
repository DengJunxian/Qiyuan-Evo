"""
交易智能体实现。

该模块实现感知、推理、决策、下单的认知闭环，并结合画像参数、
大模型推理、结构化订单和快慢思考机制完成交易决策。
"""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from agents.base_agent import BaseAgent, MarketSnapshot
from agents.brain import DeepSeekBrain
from core.types import Order, OrderSide, OrderType, OrderStatus, ExecutionPlan
from core.behavioral_finance import (
    ReferencePoints,
    ReferenceShiftConfig,
    build_institution_constraint_profile,
    behavioral_update_step,
    initialize_reference_points,
    predict_next_return,
)
from agents.persona import Persona, RiskAppetite
from core.society.network import SocialGraph, SentimentState
from agents.cognition.graph_storage import GraphMemoryBank
from agents.cognition.graph_builder import GraphExtractor
from agents.roles.news_analyst import NewsAnalyst
from agents.roles.quant_analyst import QuantAnalyst
from agents.roles.risk_analyst import RiskAnalyst


@dataclass
class LegacyDecisionAction:
    action: str
    amount: float
    target_price: Optional[float] = None


class TraderAgent(BaseAgent):
    """
    具备完整认知闭环的交易智能体。
    """

    def __init__(
        self,
        agent_id: str,
        cash_balance: float = 100_000.0,
        portfolio: Optional[Dict[str, int]] = None,
        psychology_profile: Optional[Dict[str, float]] = None,
        model_router: Optional[Any] = None,
        persona: Optional[Persona] = None,
        use_llm: bool = True,
        model_priority: Optional[List[str]] = None,
        execution_plan_enabled: bool = False,
        execution_seed: Optional[int] = None,
        news_config: Optional[Dict[str, Any]] = None,
        news_max_articles: int = 5,
        ram_cooldown_steps: int = 3
    ):
        super().__init__(agent_id, cash_balance, portfolio, psychology_profile)
        
        self.use_llm = use_llm
        self.execution_plan_enabled = bool(execution_plan_enabled)
        self.execution_seed = int(execution_seed if execution_seed is not None else 42)
        self.feature_flags: Dict[str, bool] = {}
        if isinstance(psychology_profile, dict):
            raw_flags = psychology_profile.get("feature_flags", {})
            if isinstance(raw_flags, dict):
                self.feature_flags = {str(key): bool(value) for key, value in raw_flags.items()}
        self.behavior_layer_enabled = bool(
            os.environ.get("CIVITAS_LAYERED_MEMORY_V1", "").strip().lower() in {"1", "true", "yes", "on"}
            or (isinstance(psychology_profile, dict) and bool(psychology_profile.get("behavior_layer_enabled", False)))
        )
        

        self.persona = persona if persona else Persona(name=agent_id)
        

        self.social_node_id: Optional[int] = None
        self.social_graph: Optional[SocialGraph] = None
        

        self.profile = {
            "risk_aversion": self._map_persona_to_risk_aversion(),
            "confidence_level": 0.5 + (self.persona.overconfidence * 0.5),
            "attention_span": 3 + int(self.persona.patience * 2),
            "loss_sensitivity": self.persona.loss_aversion,
        }
        if psychology_profile:
            self.profile.update(psychology_profile)
        self.institution_type = ""
        if isinstance(psychology_profile, dict) and psychology_profile.get("institution_type"):
            self.institution_type = str(psychology_profile.get("institution_type", "")).strip().lower().replace(" ", "_")


        if agent_id.startswith("Debate_"):
            from agents.debate_brain import DebateBrain
            self.brain = DebateBrain(
                agent_id=agent_id,
                persona={
                    "risk_preference": self.persona.risk_appetite.value,
                    "loss_aversion": self.profile.get("loss_sensitivity", 1.5),
                    "institution_type": self.institution_type,
                },
                api_key=None
            )
            if model_router is not None:
                self.brain.set_model_router(model_router)
        else:
            self.brain = DeepSeekBrain(
                agent_id=agent_id,
                persona={
                    "risk_preference": self.persona.risk_appetite.value,
                    "loss_aversion": self.profile.get("loss_sensitivity", 1.5),
                    "institution_type": self.institution_type,
                },
                model_router=model_router
            )
        if hasattr(self.brain, "layered_memory"):
            self.brain.layered_memory.enabled = self.behavior_layer_enabled
            if isinstance(psychology_profile, dict) and psychology_profile.get("institution_type"):
                institution_type = str(psychology_profile.get("institution_type", "")).strip().lower().replace(" ", "_")
                self.brain.layered_memory.institution_type = institution_type
                self.brain.layered_memory.constraint_profile = build_institution_constraint_profile(institution_type)

        if hasattr(self.brain, 'model_priority'):
            self.brain.model_priority = model_priority
        

        self.compliance_feedback: List[str] = []
        
        # 快慢思考模式状态跟踪
        self._last_news_count = 0
        self._last_social_sentiment = "neutral"
        self._fast_mode_consecutive_steps = 0
        self.fast_think_count = 0
        self.slow_think_count = 0
        self.last_thinking_path = "fast"
        self.last_behavior_card: Dict[str, Any] = {}
        
        # 图谱增强记忆层
        self.graph_memory = GraphMemoryBank(agent_id=self.agent_id)
        self.graph_extractor = GraphExtractor(
            model_router=self.brain.model_router if hasattr(self, 'brain') else model_router,
            model_priority=model_priority,
        )
        

        self.brain.state.confidence = self.profile.get("confidence_level", 0.5) * 100


        self.news_analyst = NewsAnalyst(config_paths=news_config, max_articles=news_max_articles)
        self.quant_analyst = QuantAnalyst()
        self.risk_analyst = RiskAnalyst()


        self._portfolio_value_history: List[float] = []
        self._last_cvar: Optional[float] = None
        self._ram_until_step: int = -1
        self._ram_last_trigger: str = ""
        self._ram_cooldown_steps: int = max(1, int(ram_cooldown_steps))


        self._price_history: List[float] = []
        self._wind_tunnel_confidence: float = 0.5
        self._wind_tunnel_records: List[Dict[str, Any]] = []
        self._recent_trade_outcomes: List[Dict[str, Any]] = []
        self._last_belief_reflection_step: int = 0
        self.reference_points: Optional[ReferencePoints] = None
        shift_profile = self.persona.reference_shift_profile() if hasattr(self.persona, "reference_shift_profile") else {}
        self.reference_shift_config = ReferenceShiftConfig(**shift_profile)
        base_risk = self.persona.base_risk_score() if hasattr(self.persona, "base_risk_score") else 0.5
        self.current_risk_appetite: float = float(base_risk)
        self.current_trading_intent: float = 0.0
        self.current_loss_aversion_intensity: float = float(self.persona.loss_aversion)
        self.last_behavioral_state: Dict[str, Any] = {}
        self.execution_config_hash = self._build_execution_config_hash()

    def _map_persona_to_risk_aversion(self) -> float:
        mapping = {
            RiskAppetite.CONSERVATIVE: 0.8,
            RiskAppetite.BALANCED: 0.5,
            RiskAppetite.AGGRESSIVE: 0.3,
            RiskAppetite.GAMBLER: 0.1
        }
        return mapping.get(self.persona.risk_appetite, 0.5)

    def _build_execution_config_hash(self) -> str:
        payload = {
            "agent_id": self.agent_id,
            "cash_balance": float(self.cash_balance),
            "portfolio": dict(self.portfolio),
            "persona": {
                "name": self.persona.name,
                "risk_appetite": getattr(self.persona.risk_appetite, "value", str(self.persona.risk_appetite)),
                "investment_horizon": getattr(self.persona.investment_horizon, "value", str(self.persona.investment_horizon)),
                "conformity": float(self.persona.conformity),
                "influence": float(self.persona.influence),
                "patience": float(self.persona.patience),
            },
            "execution_plan_enabled": bool(self.execution_plan_enabled),
            "execution_seed": int(self.execution_seed),
            "feature_flags": dict(self.feature_flags),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _feature_flag_enabled(self, name: str, default: bool = False) -> bool:
        if name in self.feature_flags:
            return bool(self.feature_flags[name])
        env_name = f"CIVITAS_{name.upper()}"
        raw = os.environ.get(env_name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _intent_execution_split_enabled(self) -> bool:
        return bool(
            self.execution_plan_enabled
            and self._feature_flag_enabled("trader_intent_execution_split_v1", False)
        )

    def _execution_constraints(self) -> Dict[str, Any]:
        if hasattr(self.persona, "agent_schema"):
            schema = self.persona.agent_schema()
            constraints = schema.get("constraints", {})
            if isinstance(constraints, dict):
                return dict(constraints)
        archetype = getattr(self.persona, "archetype", None)
        if archetype is not None and hasattr(archetype, "constraint_schema"):
            constraints = archetype.constraint_schema()
            if isinstance(constraints, dict):
                return dict(constraints)
        return {}

    def _derive_intent_trace(
        self,
        *,
        decision: Dict[str, Any],
        action: str,
        reference_price: float,
        target_qty: int,
        target_notional: Optional[float],
        urgency: float,
    ) -> Dict[str, Any]:
        desired_notional = float(target_notional) if target_notional is not None else 0.0
        if desired_notional <= 0.0 and target_qty > 0 and reference_price > 0:
            desired_notional = float(target_qty) * float(reference_price)
        conviction = max(0.0, min(1.0, 0.30 + abs(float(self.current_trading_intent)) * 0.45 + urgency * 0.25))
        return {
            "action": action,
            "desired_qty": int(max(0, target_qty)),
            "desired_notional": float(max(0.0, desired_notional)),
            "urgency": float(urgency),
            "conviction": float(conviction),
            "thesis": str(decision.get("thesis") or decision.get("reasoning") or decision.get("narrative") or ""),
            "llm_fields": sorted(str(key) for key in decision.keys()),
        }

    def _split_execution_spec(
        self,
        *,
        action: str,
        decision: Dict[str, Any],
        reference_price: float,
        target_qty: int,
        target_notional: Optional[float],
        urgency: float,
    ) -> Dict[str, Any]:
        constraints = self._execution_constraints()
        liquidity_pref = float(constraints.get("liquidity_preference", 0.5))
        benchmark_pressure = float(constraints.get("benchmark_tracking_pressure", 0.0))
        benchmark_tolerance = float(constraints.get("benchmark_deviation_tolerance", 0.2))
        flow_pressure = float(constraints.get("flow_pressure", 0.0))
        compliance_intensity = float(constraints.get("compliance_intensity", 0.5))
        execution_preference = str(constraints.get("execution_preference", "balanced")).strip().lower()
        order_horizon_bars = max(1, int(constraints.get("order_horizon_bars", 1)))
        risk_budget = max(0.05, min(1.0, 0.55 * float(self.current_risk_appetite) + 0.45 * float(self.persona.base_risk_score())))

        intent_trace = self._derive_intent_trace(
            decision=decision,
            action=action,
            reference_price=reference_price,
            target_qty=target_qty,
            target_notional=target_notional,
            urgency=urgency,
        )

        sizing_multiplier = 0.45 + 0.55 * risk_budget
        sizing_multiplier *= 0.82 + 0.18 * benchmark_tolerance
        sizing_multiplier *= 1.0 - min(0.35, benchmark_pressure * 0.20 + flow_pressure * 0.25)
        if action == "BUY":
            sizing_multiplier *= 1.0 - min(0.18, compliance_intensity * 0.10)
        adjusted_qty = int(max(0, round(target_qty * max(0.12, sizing_multiplier))))
        if adjusted_qty <= 0 and target_qty > 0:
            adjusted_qty = max(1, min(target_qty, int(round(target_qty * 0.25))))

        if adjusted_qty <= 0 and target_notional and target_notional > 0 and reference_price > 0:
            adjusted_qty = int(target_notional / reference_price)

        if execution_preference in {"passive_slicing", "patient"}:
            order_type = OrderType.POST_ONLY if liquidity_pref >= 0.7 else OrderType.LIMIT
        elif execution_preference in {"aggressive"} and urgency >= 0.72:
            order_type = OrderType.MARKET if action == "BUY" else OrderType.IOC
        elif execution_preference in {"opportunistic"} and urgency >= 0.65:
            order_type = OrderType.IOC
        else:
            order_type = OrderType.LIMIT

        slicing_rule = str(decision.get("slicing_rule") or "").strip().lower()
        if not slicing_rule:
            if adjusted_qty >= 800 or order_horizon_bars >= 3 or liquidity_pref >= 0.7:
                slicing_rule = "vwap-like" if execution_preference in {"passive_slicing", "patient"} else "twap-like"
            else:
                slicing_rule = "single"

        time_horizon = int(decision.get("time_horizon", order_horizon_bars if slicing_rule != "single" else 1))
        time_horizon = max(1, time_horizon)
        participation_rate = float(
            decision.get(
                "participation_rate",
                max(0.05, min(0.45, 0.12 + urgency * 0.20 + (1.0 - liquidity_pref) * 0.08)),
            )
        )
        max_slippage = float(decision.get("max_slippage", self._default_max_slippage(order_type, urgency)))
        cancel_replace_policy = str(
            decision.get("cancel_replace_policy") or self._default_cancel_replace_policy(order_type, slicing_rule)
        ).strip().lower()

        execution_trace = {
            "schema_version": "intent_execution_split_v1",
            "risk_budget": float(risk_budget),
            "sizing_multiplier": float(max(0.12, sizing_multiplier)),
            "liquidity_preference": float(liquidity_pref),
            "benchmark_tracking_pressure": float(benchmark_pressure),
            "benchmark_deviation_tolerance": float(benchmark_tolerance),
            "flow_pressure": float(flow_pressure),
            "compliance_intensity": float(compliance_intensity),
            "execution_preference": execution_preference,
        }
        return {
            "target_qty": int(max(0, adjusted_qty)),
            "target_notional": target_notional,
            "order_type": order_type,
            "slicing_rule": slicing_rule,
            "cancel_replace_policy": cancel_replace_policy,
            "time_horizon": int(time_horizon),
            "participation_rate": float(max(0.0, min(1.0, participation_rate))),
            "max_slippage": float(max(0.0, max_slippage)),
            "intent_trace": intent_trace,
            "execution_trace": execution_trace,
        }

    @staticmethod
    def _emotion_to_sentiment(emotion: str) -> float:
        mapping = {
            "Greedy": 0.65,
            "Confident": 0.35,
            "Neutral": 0.0,
            "Anxious": -0.35,
            "Regretful": -0.65,
            "Fearful": -0.8,
        }
        return float(mapping.get(str(emotion or "Neutral"), 0.0))

    def _derive_step_sentiment(self, perceived_data: Dict[str, Any], social_signal: str) -> float:
        snapshot: MarketSnapshot = perceived_data.get("snapshot")
        trend = float(getattr(snapshot, "market_trend", 0.0))
        emotion_term = self._emotion_to_sentiment(getattr(self, "emotional_state", "Neutral"))
        signal = (social_signal or "").lower()
        social_term = 0.0
        if "panic" in signal or "selling" in signal or "nervous" in signal:
            social_term -= 0.35
        if "buying" in signal or "fomo" in signal or "bull" in signal:
            social_term += 0.35
        sentiment = 0.45 * emotion_term + 0.35 * np.clip(trend, -1.0, 1.0) + 0.20 * social_term
        return float(np.clip(sentiment, -1.0, 1.0))

    def _estimate_peer_anchor(self, current_price: float) -> float:
        price = float(max(current_price, 1e-6))
        if self.social_graph is None or self.social_node_id is None:
            return price
        bearish_ratio = float(self.social_graph.get_bearish_ratio(self.social_node_id))
        bullish_ratio = float(self.social_graph.get_bullish_ratio(self.social_node_id))
        peer_shift = np.clip((bullish_ratio - bearish_ratio) * 0.05, -0.10, 0.10)
        return float(price * (1.0 + peer_shift))

    def _estimate_policy_anchor(self, snapshot: MarketSnapshot, current_price: float) -> tuple[float, float]:
        price = float(max(current_price, 1e-6))
        text_shock = float(getattr(snapshot, "text_policy_shock", 0.0))
        tax_rate = float(getattr(snapshot, "policy_tax_rate", 0.0))
        regime = str(getattr(snapshot, "text_regime_bias", "neutral")).lower()
        shock = text_shock - tax_rate * 0.5
        if regime in {"bullish", "easing", "supportive"}:
            shock += 0.2
        elif regime in {"bearish", "tightening", "restrictive"}:
            shock -= 0.2
        shock = float(np.clip(shock, -1.0, 1.0))
        return float(price * (1.0 + 0.08 * shock)), shock

    def update_behavioral_state(self, perceived_data: Dict[str, Any], social_signal: str) -> Dict[str, Any]:
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        current_price = float(snapshot.last_price)
        if self.reference_points is None:
            self.reference_points = initialize_reference_points(current_price)

        sentiment = self._derive_step_sentiment(perceived_data, social_signal)
        peer_anchor = self._estimate_peer_anchor(current_price)
        policy_anchor, policy_shock = self._estimate_policy_anchor(snapshot, current_price)
        step = behavioral_update_step(
            sentiment=sentiment,
            current_price=current_price,
            reference_points=self.reference_points,
            base_risk_appetite=self.current_risk_appetite,
            peer_anchor=peer_anchor,
            policy_anchor=policy_anchor,
            policy_shock=policy_shock,
            loss_aversion=float(self.persona.loss_aversion),
            shift_config=self.reference_shift_config,
            reference_weights=self.persona.reference_weights() if hasattr(self.persona, "reference_weights") else None,
        )
        self.reference_points = step.reference_points
        self.current_risk_appetite = float(step.risk_appetite)
        self.current_trading_intent = float(step.trading_intent)
        self.current_loss_aversion_intensity = float(step.loss_aversion_intensity)
        self.last_behavioral_state = {
            "sentiment": float(step.sentiment),
            "risk_appetite": float(step.risk_appetite),
            "trading_intent": float(step.trading_intent),
            "prospect_direction": float(step.prospect_direction),
            "loss_aversion_intensity": float(step.loss_aversion_intensity),
            "weighted_reference_return": float(step.weighted_reference_return),
            "reference_points": {
                "purchase_anchor": float(step.reference_points.purchase_anchor),
                "recent_high_anchor": float(step.reference_points.recent_high_anchor),
                "peer_anchor": float(step.reference_points.peer_anchor),
                "policy_anchor": float(step.reference_points.policy_anchor),
            },
        }
        return dict(self.last_behavioral_state)

    def _apply_behavioral_intent_overlay(
        self,
        decision_payload: Dict[str, Any],
        snapshot: MarketSnapshot,
    ) -> Dict[str, Any]:
        intent = float(self.current_trading_intent)
        risk = float(self.current_risk_appetite)
        decision = decision_payload.setdefault("decision", {})
        action = str(decision.get("action", "HOLD")).upper()
        qty = int(decision.get("qty", 0) or 0)
        price = float(decision.get("price", snapshot.last_price) or snapshot.last_price)

        if action == "HOLD":
            if intent >= 0.55 and self.cash_balance > snapshot.last_price:
                action = "BUY"
                qty = max(100, int((self.cash_balance * (0.06 + 0.16 * risk)) / snapshot.last_price) // 100 * 100)
                price = round(snapshot.last_price * 1.01, 2)
            elif intent <= -0.55:
                holding = int(self.portfolio.get(snapshot.symbol, 0))
                if holding > 0:
                    action = "SELL"
                    qty = max(100, int(holding * (0.25 + 0.45 * abs(intent))) // 100 * 100)
                    qty = min(qty, holding)
                    price = round(snapshot.last_price * 0.99, 2)

        decision["action"] = action
        decision["qty"] = max(0, int(qty))
        decision["price"] = float(price)
        decision_payload["behavioral_state"] = dict(self.last_behavioral_state)
        behavior_bundle = self.brain._apply_behavior_layer(
            decision=decision,
            market_state={
                "price": float(snapshot.last_price),
                "last_price": float(snapshot.last_price),
                "trend": getattr(snapshot, "market_trend", 0.0),
                "market_trend": getattr(snapshot, "market_trend", 0.0),
                "panic_level": float(getattr(snapshot, "panic_level", 0.0)),
                "policy_description": getattr(snapshot, "policy_description", ""),
                "policy_news": getattr(snapshot, "policy_news", ""),
                "text_sentiment_score": float(getattr(snapshot, "text_sentiment_score", 0.0)),
                "text_policy_shock": float(getattr(snapshot, "text_policy_shock", 0.0)),
                "text_regime_bias": getattr(snapshot, "text_regime_bias", "neutral"),
                "news_source": getattr(snapshot, "news_source", "market"),
            },
            account_state={
                "cash": float(self.cash_balance),
                "market_value": float(self.portfolio.get(snapshot.symbol, 0) * snapshot.last_price),
                "pnl_pct": float(getattr(self, "last_pnl_pct", 0.0) or 0.0),
            },
            emotional_state=str(getattr(self, "emotional_state", "Neutral")),
            social_signal=str(getattr(self, "_last_social_sentiment", "neutral")),
            snapshot_info={
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp,
                "price": float(snapshot.last_price),
                "market_trend": float(getattr(snapshot, "market_trend", 0.0)),
                "panic_level": float(getattr(snapshot, "panic_level", 0.0)),
            },
        )
        decision_payload["decision"] = behavior_bundle.get("decision", decision)
        decision_payload["behavior_card"] = behavior_bundle.get("behavior_card", {})
        decision_payload["behavior_context"] = behavior_bundle.get("behavior_context", {})
        self.last_behavior_card = dict(decision_payload["behavior_card"])
        return decision_payload

    def bind_social_node(self, node_id: int, graph: SocialGraph):
        """Bind this agent to a node in the Social Graph."""
        self.social_node_id = node_id
        self.social_graph = graph
        if node_id in graph.agents:
            graph.agents[node_id].agent_id = self.agent_id

        self.sync_social_semantic_profile()

    def _risk_tilt_from_persona(self) -> float:
        """Map discrete risk preference to a continuous tilt score in [-1, 1]."""
        mapping = {
            RiskAppetite.CONSERVATIVE: -0.8,
            RiskAppetite.BALANCED: 0.0,
            RiskAppetite.AGGRESSIVE: 0.6,
            RiskAppetite.GAMBLER: 0.9,
        }
        return float(mapping.get(self.persona.risk_appetite, 0.0))

    def _default_focus_topics(self) -> List[str]:
        """Build a stable cold-start topic profile before graph memory converges."""
        horizon_map = {
            "short-term": ["volatility", "turnover", "sentiment", "hot-theme"],
            "medium-term": ["policy", "liquidity", "earnings", "valuation"],
            "long-term": ["fundamental", "industry-trend", "fiscal", "rates"],
        }
        risk_map = {
            RiskAppetite.CONSERVATIVE: ["defensive", "dividend", "stable-growth"],
            RiskAppetite.BALANCED: ["balanced-allocation", "policy", "cycle"],
            RiskAppetite.AGGRESSIVE: ["growth", "technology", "beta"],
            RiskAppetite.GAMBLER: ["speculation", "high-vol", "leverage"],
        }

        horizon_key = str(self.persona.investment_horizon.value).lower()
        topics = horizon_map.get(horizon_key, ["policy", "risk"])
        topics = topics + risk_map.get(self.persona.risk_appetite, ["policy", "risk"])
        return list(dict.fromkeys(topics))

    def sync_social_semantic_profile(self) -> None:
        """Sync GraphRAG semantic profile to social graph node."""
        if self.social_graph is None or self.social_node_id is None:
            return

        dominant = self.graph_memory.get_dominant_narratives(top_k=6)
        default_focus = self._default_focus_topics()
        focus_topics = list(dict.fromkeys(default_focus + dominant[:3]))
        confidence = float(getattr(self.brain.state, "confidence", 50.0))
        confidence_bias = np.clip((confidence - 50.0) / 50.0, -1.0, 1.0)

        self.social_graph.update_semantic_profile(
            self.social_node_id,
            dominant_narratives=dominant if dominant else default_focus[:3],
            focus_topics=focus_topics,
            risk_tilt=self._risk_tilt_from_persona(),
            historical_risk_bias=float(confidence_bias) * 0.3,
        )
        self.social_graph.update_holdings(self.social_node_id, self.portfolio)
        self.social_graph.update_bdi_profile(
            self.social_node_id,
            beliefs=dominant if dominant else default_focus[:3],
            desires=focus_topics,
            intentions=[getattr(self, "emotional_state", "Neutral")],
        )

    async def perceive(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
    ) -> Dict[str, Any]:
        """感知阶段。"""

        span = int(self.profile.get("attention_span", 3))
        observed_news = public_news[:span] if public_news else []
        
        portfolio_value = self.get_total_value({market_snapshot.symbol: market_snapshot.last_price})
        initial = getattr(self, 'initial_cash', 100000)
        pnl = portfolio_value - initial
        pnl_pct = pnl / initial if initial > 0 else 0

        if portfolio_value:
            self._portfolio_value_history.append(portfolio_value)
            if len(self._portfolio_value_history) > 200:
                self._portfolio_value_history = self._portfolio_value_history[-200:]

        self._price_history.append(market_snapshot.last_price)
        if len(self._price_history) > 200:
            self._price_history = self._price_history[-200:]

        return {
            "snapshot": market_snapshot,
            "news": observed_news,
            "portfolio_value": portfolio_value,
            "cash": self.cash_balance,
            "pnl_pct": pnl_pct,
            "timestamp": market_snapshot.timestamp
        }

    async def reason(
        self,
        perceived_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reasoning stage with analyst reports and behavioral context."""
        analyst_reports = self._collect_analyst_reports(perceived_data)
        perceived_data["analyst_reports"] = analyst_reports
        self.update_emotional_state(perceived_data)
        self._update_ram_state(
            analyst_reports.get("risk", {}),
            getattr(self, "emotional_state", "Neutral"),
        )
        social_signal = self.perceive_social_signal(perceived_data)
        perceived_data["behavioral_state"] = self.update_behavioral_state(perceived_data, social_signal)
        return await self.reason_and_act(perceived_data, self.emotional_state, social_signal)

    def _needs_deep_thinking(self, perceived_data: Dict[str, Any], social_signal: str) -> bool:
        """Decide whether to invoke expensive deep-thinking path."""
        news = perceived_data.get("news", [])
        if len(news) > self._last_news_count:
            self._last_news_count = len(news)
            return True

        pnl_pct = perceived_data.get("pnl_pct", 0)
        utility = pnl_pct if pnl_pct > 0 else pnl_pct * 2.25
        if utility < -0.15 or utility > 0.10:
            return True

        current_social = "neutral"
        if "panic" in social_signal.lower() or "selling" in social_signal.lower():
            current_social = "bearish"
        elif "buy" in social_signal.lower() or "fomo" in social_signal.lower():
            current_social = "bullish"

        if current_social != self._last_social_sentiment:
            self._last_social_sentiment = current_social
            return True

        if self._fast_mode_consecutive_steps > 10:
            return True
        return False

    def _fast_think(
        self,
        perceived_data: Dict[str, Any],
        emotional_state: str,
        social_signal: str,
    ) -> Dict[str, Any]:
        """Fast rule-based decision path."""
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        trend_val = getattr(snapshot, "market_trend", 0)

        action = "HOLD"
        qty = 0
        price = 0.0
        last_price = snapshot.last_price
        behavioral_state = perceived_data.get("behavioral_state", self.last_behavioral_state)
        if isinstance(behavioral_state, dict):
            intent_score = float(behavioral_state.get("trading_intent", self.current_trading_intent))
            risk_score = float(behavioral_state.get("risk_appetite", self.current_risk_appetite))
        else:
            intent_score = self.current_trading_intent
            risk_score = self.current_risk_appetite

        if trend_val > 0.03 and self.cash_balance > last_price * 100:
            action = "BUY"
            qty = int((self.cash_balance * (0.06 + 0.18 * risk_score)) / last_price) // 100 * 100
            price = round(last_price * 1.02, 2)
        elif trend_val < -0.03:
            holding = self.portfolio.get(snapshot.symbol, 0)
            if holding > 0:
                action = "SELL"
                qty = int(holding * 0.2) // 100 * 100
                price = round(last_price * 0.98, 2)

        if "panic" in social_signal.lower() or emotional_state == "Fearful":
            holding = self.portfolio.get(snapshot.symbol, 0)
            if holding > 0:
                action = "SELL"
                qty = int(holding * 0.5) // 100 * 100
                if qty == 0 and holding > 0:
                    qty = holding
                price = round(last_price * 0.95, 2)

        if action == "HOLD":
            if intent_score >= 0.55 and self.cash_balance > last_price * 100:
                action = "BUY"
                qty = int((self.cash_balance * (0.05 + 0.18 * risk_score)) / last_price) // 100 * 100
                price = round(last_price * 1.01, 2)
            elif intent_score <= -0.55:
                holding = self.portfolio.get(snapshot.symbol, 0)
                if holding > 0:
                    action = "SELL"
                    qty = int(holding * (0.20 + 0.55 * abs(intent_score))) // 100 * 100
                    if qty <= 0:
                        qty = holding
                    price = round(last_price * 0.99, 2)

        if action == "BUY" and qty > 0:
            qty = max(100, int(qty * (0.85 + 0.6 * risk_score)))
        elif action == "SELL" and qty > 0:
            qty = max(100, int(qty * (0.85 + 0.6 * abs(min(0.0, intent_score)))))

        reasoning = f"(System 1) Fast response. Trend={trend_val:.3f}, Emotion={emotional_state}"
        decision_payload = {
            "decision": {"action": action, "qty": qty, "price": price},
            "reasoning": reasoning,
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp,
            "behavioral_state": dict(self.last_behavioral_state),
        }
        self.fast_think_count += 1
        self.last_thinking_path = "fast"
        decision_payload = self._apply_behavioral_intent_overlay(decision_payload, snapshot)
        self.last_behavior_card = dict(decision_payload.get("behavior_card", {}))
        return decision_payload

    async def _async_extract_and_store(self, text: str, current_time: float, is_news: bool = False):
        """后台抽取图谱，避免阻塞交易循环。"""
        triplets = await self.graph_extractor.extract_graph(text)
        if not triplets:
            return
            
        topics = set()
        for t in triplets:
            self.graph_memory.add_triplet(t["subject"], t["predicate"], t["target"], t["weight"])
            topics.add(t["subject"])
            

        if is_news and topics:
            capsule_topic = list(topics)[0]
            summary = f"[{capsule_topic}] 相关动态已发生: {text[:50]}..."
            self.graph_memory.add_capsule(capsule_topic, summary, current_time, ttl_seconds=3600)

    def _collect_analyst_reports(self, perceived_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集分析师报告，并在失败时返回防御性兜底结果。"""
        reports: Dict[str, Any] = {}
        snapshot: MarketSnapshot = perceived_data.get("snapshot")

        try:
            reports["news"] = self.news_analyst.analyze()
        except Exception as exc:
            reports["news"] = {
                "analyst": "NewsAnalyst",
                "status": "error",
                "error": str(exc),
                "events": [],
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
            }

        try:
            reports["quant"] = self.quant_analyst.analyze(snapshot)
        except Exception as exc:
            reports["quant"] = {
                "analyst": "QuantAnalyst",
                "status": "error",
                "error": str(exc),
                "csad": 0.0,
                "momentum": 0.0,
                "herding_intensity": 0.0,
            }

        try:
            reports["risk"] = self.risk_analyst.analyze(self._portfolio_value_history)
        except Exception as exc:
            reports["risk"] = {
                "analyst": "RiskAnalyst",
                "status": "error",
                "error": str(exc),
                "cvar": 0.0,
                "max_drawdown": 0.0,
            }

        return reports

    def _update_ram_state(self, risk_report: Dict[str, Any], emotional_state: str) -> None:
        """更新风险会议状态机。"""
        cvar = float(risk_report.get("cvar", 0.0) or 0.0)
        cvar_drop = None
        trigger_reason = ""

        if self._last_cvar is not None:
            cvar_drop = cvar - self._last_cvar
            if cvar_drop < -abs(self._last_cvar) * 0.5 and cvar < 0:
                trigger_reason = f"CVaR 下降: {self._last_cvar:.4f} -> {cvar:.4f}"

        if emotional_state == "Fearful":
            trigger_reason = trigger_reason or "恐惧情绪触发风险会议"

        if trigger_reason:
            self._ram_until_step = max(self._ram_until_step, self._step_count + self._ram_cooldown_steps)
            self._ram_last_trigger = trigger_reason

        self._last_cvar = cvar
        if cvar_drop is not None:
            risk_report["cvar_drop"] = cvar_drop
        risk_report["ram_active"] = self._ram_until_step >= self._step_count
        risk_report["ram_trigger_reason"] = self._ram_last_trigger

    def _beliefs_file_path(self) -> str:
        base_dir = os.path.join("data", "beliefs")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"beliefs_{self.agent_id}.json")

    def _persist_beliefs(self, new_beliefs: List[str]) -> None:
        """持久化交易信念，用于后续提示词增强。"""
        path = self._beliefs_file_path()
        beliefs = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    beliefs = data.get("beliefs", [])
            except Exception:
                beliefs = []
        for b in new_beliefs:
            b = str(b).strip()
            if b and b not in beliefs:
                beliefs.append(b)
        beliefs = beliefs[-10:]
        payload = {
            "agent_id": self.agent_id,
            "updated_at": time.time(),
            "beliefs": beliefs,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _reflect_and_persist_beliefs(self) -> None:
        """Summarize recent outcomes into compact investment beliefs."""
        if not self._recent_trade_outcomes:
            return
        if self._step_count - self._last_belief_reflection_step < 5:
            return

        recent = self._recent_trade_outcomes[-8:]
        prompt = {
            "recent_trades": recent,
            "instruction": "Extract up to 3 abstract investment beliefs as a JSON array.",
        }
        messages = [
            {"role": "system", "content": "You are a quantitative behavioral analyst. Return JSON array only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        beliefs: List[str] = []
        try:
            if getattr(self.brain, "model_router", None):
                priority = getattr(self.brain, "model_priority", None) or ["glm-4-flashx"]
                content, _, _ = self.brain.model_router.sync_call_with_fallback(
                    messages=messages,
                    priority_models=list(priority),
                    timeout_budget=20.0,
                )
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    beliefs = [str(x) for x in parsed if str(x).strip()]
        except Exception:
            beliefs = []

        if not beliefs:
            if self.brain.state.consecutive_losses >= 3:
                beliefs = ["In noisy regimes, reduce turnover and avoid overtrading."]
            elif self.brain.state.consecutive_wins >= 3:
                beliefs = ["Trend-following can work, but drawdown controls must remain active."]
            else:
                beliefs = ["When uncertainty is high, position discipline beats frequent trial-and-error."]

        self._persist_beliefs(beliefs)
        self._last_belief_reflection_step = self._step_count

    def _wind_tunnel_predict(self) -> Dict[str, Any]:
        """Internal wind-tunnel predictor used as pre-trade veto signal."""
        pred_return = predict_next_return(self._price_history)
        if pred_return is None:
            return {"valid": False}
        return {
            "valid": True,
            "predicted_return": pred_return,
            "confidence": self._wind_tunnel_confidence
        }

    async def reason_and_act(
        self,
        perceived_data: Dict[str, Any],
        emotional_state: str,
        social_signal: str
    ) -> Dict[str, Any]:
        """结合市场、情绪和社交信号生成交易动作。"""
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        if self._ram_until_step >= self._step_count:
            holding = self.portfolio.get(snapshot.symbol, 0)
            if holding > 0:
                return {
                    "decision": {
                        "action": "SELL",
                        "qty": holding,
                        "price": round(snapshot.last_price * 0.98, 2)
                    },
                    "reasoning": f"[RAM] 风险会议触发，强制避险清仓。原因：{self._ram_last_trigger}",
                    "symbol": snapshot.symbol,
                    "timestamp": snapshot.timestamp,
                    "analyst_reports": perceived_data.get("analyst_reports", {}),
                    "risk_mode": "RAM",
                    "behavior_card": {},
                    "behavior_context": {"enabled": False, "mode": "RAM"},
                }
            return {
                "decision": {"action": "HOLD"},
                "reasoning": f"[RAM] 风险会议触发，当前无持仓，保持观望。原因：{self._ram_last_trigger}",
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp,
                "analyst_reports": perceived_data.get("analyst_reports", {}),
                "risk_mode": "RAM",
                "behavior_card": {},
                "behavior_context": {"enabled": False, "mode": "RAM"},
            }



        if not self.use_llm:
            return self._fast_think(perceived_data, emotional_state, social_signal)


        if not self._needs_deep_thinking(perceived_data, social_signal):
            self._fast_mode_consecutive_steps += 1
            return self._fast_think(perceived_data, emotional_state, social_signal)
            

        self._fast_mode_consecutive_steps = 0
        self.slow_think_count += 1
        self.last_thinking_path = "slow"
        
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        news = perceived_data["news"]


        self.sync_social_semantic_profile()
        

        current_time = snapshot.timestamp
        graph_context = ""
        capsules = self.graph_memory.get_valid_capsules(current_time)
        extracted_keywords = []
        news_text = "; ".join(news) if news else ""
        
        if news_text:
            # 简单关键词提取逻辑用于查图
            if "利率" in news_text or "降准" in news_text:
                extracted_keywords.append("利率")
            if "liquidity" in news_text.lower():
                extracted_keywords.append("liquidity")
            if "政策" in news_text:
                extracted_keywords.append("政策")
            if not extracted_keywords:
                extracted_keywords = ["市场", "风险"]
        
        if capsules:
            graph_context = "【知识胶囊：宏观共识缓存】\n" + "\n".join(capsules)
        elif extracted_keywords:
            subgraph = self.graph_memory.retrieve_subgraph(extracted_keywords, depth=2)
            if subgraph:
                graph_context = "【私有认知图谱关联】\n" + subgraph
                

        if news_text and not capsules:
            asyncio.create_task(self._async_extract_and_store(news_text, current_time, is_news=True))
            

        market_state = {
            "price": snapshot.last_price,
            "trend": getattr(snapshot, "market_trend", "震荡"),
            "panic_level": getattr(snapshot, "panic_level", 0.5),
            "news": news_text if news_text else "no-major-news",
            "last_rejection_reason": self.compliance_feedback[-1] if self.compliance_feedback else None,
            "policy_description": getattr(snapshot, "policy_description", ""),
            "policy_tax_rate": getattr(snapshot, "policy_tax_rate", 0.0),
            "policy_news": getattr(snapshot, "policy_news", ""),
            "text_dominant_topic": getattr(snapshot, "text_dominant_topic", "uncategorized"),
            "text_sentiment_score": getattr(snapshot, "text_sentiment_score", 0.0),
            "text_panic_score": getattr(snapshot, "text_panic_score", 0.0),
            "text_greed_score": getattr(snapshot, "text_greed_score", 0.0),
            "text_policy_shock": getattr(snapshot, "text_policy_shock", 0.0),
            "text_regime_bias": getattr(snapshot, "text_regime_bias", "neutral"),
            "text_impact_paths": getattr(snapshot, "text_impact_paths", []),
            "graph_context": graph_context,
            "analyst_reports": perceived_data.get("analyst_reports", {}),
            "behavioral_state": perceived_data.get("behavioral_state", self.last_behavioral_state),
        }
        

        trend_val = snapshot.market_trend
        if trend_val > 0.02:
            market_state["trend"] = "上涨"
        elif trend_val < -0.02:
            market_state["trend"] = "下跌"
        else:
            market_state["trend"] = "震荡"


        account_state = {
            "cash": perceived_data["cash"],
            "market_value": perceived_data["portfolio_value"] - perceived_data["cash"],
            "pnl_pct": perceived_data["pnl_pct"],
            "risk_appetite": self.current_risk_appetite,
        }


        try:
            decision_output = await self.brain.think_async(
                market_state=market_state, 
                account_state=account_state,
                emotional_state=emotional_state,
                social_signal=social_signal
            )
            

            decision_output["symbol"] = snapshot.symbol
            decision_output["timestamp"] = snapshot.timestamp
            decision_output = self._apply_behavioral_intent_overlay(decision_output, snapshot)
            self.last_behavior_card = dict(decision_output.get("behavior_card", {}))
            return decision_output
            
        except Exception as e:
            print(f"Agent {self.agent_id} Brain Error: {e}")
            return {
                "decision": {"action": "HOLD"},
                "reasoning": f"Brain Failure: {e}",
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp
            }

    def update_emotional_state(self, perceived_data: Dict[str, Any]):
        """Update emotional state from PnL and panic indicators."""
        pnl_pct = perceived_data.get("pnl_pct", 0)
        snapshot = perceived_data.get("snapshot")
        panic_level = getattr(snapshot, "panic_level", 0)
        

        loss_threshold = -0.10 / (self.persona.loss_aversion / 2.25)
        panic_threshold = 0.6 * self.persona.patience
        
        if pnl_pct < loss_threshold:
            self.emotional_state = "Regretful" # 鎮旀仺 (澶т簭)
        elif pnl_pct < loss_threshold / 2:
            self.emotional_state = "Anxious"
        elif pnl_pct > 0.10:
            self.emotional_state = "Greedy"    # 璐┆ (澶ц禋)
        elif pnl_pct > 0.05:
            self.emotional_state = "Confident"
        elif panic_level > panic_threshold:
            self.emotional_state = "Fearful"   # 鎭愭儳 (甯傚満鎭愭厡)
        else:
            self.emotional_state = "Neutral"
            
    def perceive_social_signal(self, perceived_data: Dict[str, Any]) -> str:
        """Perceive social signal from graph neighbors and market trend."""

        if self.social_graph and self.social_node_id is not None:
            bearish_ratio = self.social_graph.get_bearish_ratio(self.social_node_id)
            

            panic_threshold = 1.0 - (self.persona.conformity * 0.8)
            
            if bearish_ratio > panic_threshold:
                return f"Panic Alert! {bearish_ratio:.0%} of neighbors are selling!"
            elif bearish_ratio > panic_threshold * 0.6:
                return "Neighbors are getting nervous."
                

        snapshot = perceived_data.get("snapshot")
        trend = getattr(snapshot, "market_trend", 0)
        
        if trend > 0.05:
            return "Everyone is buying! (FOMO)"
        elif trend < -0.05:
            return "Everyone is panic selling!"
        else:
            return "Market is quiet."

    def _reference_price_for_plan(self, reasoning_output: Dict[str, Any], decision: Dict[str, Any]) -> float:
        price = decision.get("price")
        if price is None:
            price = reasoning_output.get("price")
        if price is None and self._price_history:
            price = self._price_history[-1]
        return float(max(0.0, float(price or 0.0)))

    def _default_slicing_rule(self, action: str, qty: int, target_notional: Optional[float]) -> str:
        if qty >= 1000 or (target_notional is not None and target_notional > self.cash_balance * 0.2):
            horizon = str(getattr(self.persona.investment_horizon, "value", "")).lower()
            if "long" in horizon:
                return "vwap-like"
            return "twap-like"
        if action in {"BUY", "SELL"} and self.current_trading_intent > 0.75:
            return "twap-like"
        return "single"

    def _default_order_type(self, action: str, qty: int, slicing_rule: str) -> OrderType:
        if slicing_rule in {"twap-like", "vwap-like"}:
            return OrderType.LIMIT
        if action == "BUY" and self.current_trading_intent >= 0.85:
            return OrderType.MARKET
        if action == "SELL" and self.current_trading_intent <= -0.85:
            return OrderType.IOC
        return OrderType.LIMIT

    def _default_cancel_replace_policy(self, order_type: OrderType, slicing_rule: str) -> str:
        if order_type in {OrderType.MARKET, OrderType.IOC, OrderType.FOK}:
            return "none"
        if slicing_rule in {"twap-like", "vwap-like"}:
            return "cancel-replace"
        return "none"

    def _default_max_slippage(self, order_type: OrderType, urgency: float) -> float:
        if order_type == OrderType.MARKET:
            return min(0.05, 0.02 + 0.03 * urgency)
        if order_type == OrderType.IOC:
            return min(0.03, 0.01 + 0.02 * urgency)
        if order_type == OrderType.FOK:
            return 0.0
        if order_type == OrderType.POST_ONLY:
            return 0.005
        return min(0.02, 0.005 + 0.015 * urgency)

    def _build_execution_plan(self, reasoning_output: Dict[str, Any]) -> Optional[ExecutionPlan]:
        decision = dict(reasoning_output.get("decision", {}) or {})
        action = str(decision.get("action") or decision.get("side") or "HOLD").strip().upper()
        if action in {"", "HOLD"}:
            return None
        if action not in {"BUY", "SELL"}:
            return None

        symbol = str(reasoning_output.get("symbol", "UNKNOWN"))
        timestamp = float(reasoning_output.get("timestamp", time.time()))
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        reference_price = self._reference_price_for_plan(reasoning_output, decision)

        target_qty = decision.get("target_qty", decision.get("qty"))
        if target_qty is not None:
            target_qty = max(0, int(target_qty))
        target_notional = decision.get("target_notional")
        if target_notional is not None:
            target_notional = max(0.0, float(target_notional))
        if target_qty is None and target_notional is not None and reference_price > 0:
            target_qty = int(target_notional / reference_price)

        if target_qty is None:
            target_qty = 0

        legacy_mode = not self.execution_plan_enabled
        split_mode = self._intent_execution_split_enabled() and not legacy_mode

        urgency = float(decision.get("urgency", max(0.0, min(1.0, abs(self.current_trading_intent)))))
        if not 0.0 <= urgency <= 1.0:
            urgency = max(0.0, min(1.0, urgency))

        intent_trace: Dict[str, Any] = {}
        execution_trace: Dict[str, Any] = {}

        if split_mode:
            split_spec = self._split_execution_spec(
                action=action,
                decision=decision,
                reference_price=reference_price,
                target_qty=int(target_qty or 0),
                target_notional=target_notional,
                urgency=urgency,
            )
            target_qty = int(split_spec["target_qty"])
            target_notional = split_spec["target_notional"]
            order_type = split_spec["order_type"]
            slicing_rule = str(split_spec["slicing_rule"])
            cancel_replace_policy = str(split_spec["cancel_replace_policy"])
            time_horizon = int(split_spec["time_horizon"])
            participation_rate = float(split_spec["participation_rate"])
            max_slippage = float(split_spec["max_slippage"])
            intent_trace = dict(split_spec.get("intent_trace", {}))
            execution_trace = dict(split_spec.get("execution_trace", {}))

        if action == "BUY" and reference_price > 0:
            if legacy_mode:
                affordable = int(self.cash_balance / reference_price) // 100 * 100
            else:
                affordable = int(self.cash_balance / reference_price)
            target_qty = min(target_qty, affordable)
        elif action == "SELL":
            target_qty = min(target_qty, int(self.portfolio.get(symbol, 0)))

        if target_qty <= 0 and (target_notional is None or target_notional <= 0):
            return None

        if legacy_mode:
            order_type = OrderType.LIMIT
            slicing_rule = "single"
            cancel_replace_policy = "none"
            time_horizon = 1
            participation_rate = float(decision.get("participation_rate", 0.1 if slicing_rule == "single" else 0.2))
            max_slippage = float(decision.get("max_slippage", 0.01))
        elif not split_mode:
            order_type = OrderType.LIMIT
            if "order_type" in decision:
                try:
                    order_type = OrderType(str(decision["order_type"]).strip().lower().replace("_", "-"))
                except Exception:
                    order_type = OrderType.LIMIT
            else:
                order_type = self._default_order_type(action, target_qty, str(decision.get("slicing_rule", "")).lower())

            slicing_rule = str(decision.get("slicing_rule") or self._default_slicing_rule(action, target_qty, target_notional)).strip().lower()
            cancel_replace_policy = str(decision.get("cancel_replace_policy") or self._default_cancel_replace_policy(order_type, slicing_rule)).strip().lower()
            time_horizon = int(decision.get("time_horizon", 1 if slicing_rule == "single" else max(2, min(10, target_qty // 100 if target_qty >= 100 else 3))))
            participation_rate = float(decision.get("participation_rate", 0.1 if slicing_rule == "single" else 0.2))
            max_slippage = float(decision.get("max_slippage", self._default_max_slippage(order_type, urgency)))
        if time_horizon <= 0:
            time_horizon = 1

        if not 0.0 <= participation_rate <= 1.0:
            participation_rate = max(0.0, min(1.0, participation_rate))

        if max_slippage < 0:
            max_slippage = 0.0

        child_order_schedule = decision.get("child_order_schedule", [])
        if not isinstance(child_order_schedule, list):
            child_order_schedule = []
        child_order_schedule = [max(0, int(q)) for q in child_order_schedule if int(q) > 0]

        if not legacy_mode and not child_order_schedule and slicing_rule in {"twap-like", "vwap-like"}:
            plan_qty = max(target_qty, 1)
            parts = max(1, time_horizon)
            base = plan_qty // parts
            remainder = plan_qty % parts
            child_order_schedule = [base + (1 if i < remainder else 0) for i in range(parts) if base + (1 if i < remainder else 0) > 0]

        snapshot_info = {
            "symbol": symbol,
            "timestamp": timestamp,
            "last_price": reference_price,
            "cash_balance": float(self.cash_balance),
            "position": int(self.portfolio.get(symbol, 0)),
            "portfolio_value": float(self.get_total_value({symbol: reference_price}) if reference_price > 0 else self.cash_balance),
            "risk_appetite": float(self.current_risk_appetite),
            "trading_intent": float(self.current_trading_intent),
            "persona": {
                "risk_appetite": getattr(self.persona.risk_appetite, "value", str(self.persona.risk_appetite)),
                "investment_horizon": getattr(self.persona.investment_horizon, "value", str(self.persona.investment_horizon)),
                "archetype_key": getattr(self.persona, "archetype_key", ""),
                "participant_type": getattr(getattr(self.persona, "archetype", None), "participant_type", "generic"),
                "strategy_family": getattr(getattr(self.persona, "archetype", None), "strategy_family", "discretionary"),
                "constraints": self._execution_constraints(),
            },
        }

        plan = ExecutionPlan(
            symbol=symbol,
            agent_id=self.agent_id,
            action=action,
            side=side,
            target_qty=target_qty if target_qty > 0 else None,
            target_notional=target_notional,
            urgency=urgency,
            order_type=order_type,
            max_slippage=max_slippage,
            participation_rate=participation_rate,
            slicing_rule=slicing_rule,
            cancel_replace_policy=cancel_replace_policy,
            time_horizon=time_horizon,
            price=reference_price,
            timestamp=timestamp,
            child_order_schedule=child_order_schedule,
            seed=self.execution_seed,
            snapshot_info=snapshot_info,
        )

        config_payload = {
            "agent_id": self.agent_id,
            "execution_seed": self.execution_seed,
            "execution_plan_enabled": self.execution_plan_enabled,
            "symbol": symbol,
            "action": action,
            "side": plan.side.value,
            "target_qty": plan.target_qty,
            "target_notional": plan.target_notional,
            "urgency": plan.urgency,
            "order_type": plan.order_type.value,
            "max_slippage": plan.max_slippage,
            "participation_rate": plan.participation_rate,
            "slicing_rule": plan.slicing_rule,
            "cancel_replace_policy": plan.cancel_replace_policy,
            "time_horizon": plan.time_horizon,
            "snapshot_info": snapshot_info,
            "feature_flags": dict(self.feature_flags),
        }
        plan.config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        plan.metadata.update(
            {
                "config_hash": plan.config_hash,
                "execution_seed": self.execution_seed,
                "feature_flag_execution_plan": self.execution_plan_enabled,
                "feature_flag_trader_intent_execution_split_v1": split_mode,
                "snapshot_info": snapshot_info,
                "intent_trace": intent_trace,
                "execution_trace": execution_trace,
            }
        )
        return plan

    async def decide(
        self,
        reasoning_output: Dict[str, Any],
    ) -> Optional[Any]:
        """Convert reasoning output into an executable order or execution plan."""
        plan = self._build_execution_plan(reasoning_output)
        if plan is None:
            return None
        legacy_mode = not self.execution_plan_enabled

        wind_report = self._wind_tunnel_predict()
        decision = dict(reasoning_output.get("decision", {}) or {})
        decision["wind_tunnel"] = wind_report
        if wind_report.get("valid"):
            pred = float(wind_report.get("predicted_return", 0.0))
            conf = float(wind_report.get("confidence", 0.5))
            if plan.is_buy and pred < -0.005 and conf > 0.6:
                return None
            if plan.is_sell and pred > 0.005 and conf > 0.6:
                return None

        if plan.is_buy:
            cost = plan.price * plan.resolved_qty(plan.price)
            if cost > self.cash_balance:
                if legacy_mode:
                    adjusted_qty = int(self.cash_balance / plan.price) // 100 * 100 if plan.price > 0 else 0
                else:
                    adjusted_qty = int(self.cash_balance / plan.price) if plan.price > 0 else 0
                if adjusted_qty <= 0:
                    return None
                plan.target_qty = adjusted_qty
        elif plan.is_sell:
            holding = self.portfolio.get(plan.symbol, 0)
            plan.target_qty = min(int(plan.target_qty or 0), int(holding))
            if plan.target_qty <= 0:
                return None

        if self.execution_plan_enabled:
            plan.metadata["feature_flag_execution_plan"] = True
            return plan

        return plan.to_order()

    async def generate_trading_decision(self, market_data: Dict[str, Any], retrieved_context: str = "") -> Any:
        """Compatibility bridge for legacy simulation loop."""
        price = float(market_data.get("current_price", 0.0) or 0.0)
        if price <= 0:
            price = 1.0
        snapshot = MarketSnapshot(
            symbol=str(market_data.get("symbol", "A_SHARE_IDX")),
            last_price=price,
            best_bid=float(market_data.get("best_bid", price * 0.999) or price * 0.999),
            best_ask=float(market_data.get("best_ask", price * 1.001) or price * 1.001),
            bid_ask_spread=float(market_data.get("spread", max(price * 0.002, 0.01)) or max(price * 0.002, 0.01)),
            mid_price=price,
            total_volume=int(market_data.get("volume", 0) or 0),
            volatility=float(market_data.get("volatility", 0.0) or 0.0),
            market_trend=float(market_data.get("trend", 0.0) or 0.0),
            panic_level=float(market_data.get("panic_level", 0.0) or 0.0),
            timestamp=float(market_data.get("timestamp", time.time()) or time.time()),
            policy_description=str(market_data.get("latest_broadcast", "") or ""),
        )
        news = [str(market_data.get("latest_broadcast", "") or "market_stable")]
        result = await self.act(snapshot, news)
        if isinstance(result, ExecutionPlan):
            return result
        if isinstance(result, Order):
            action = "BUY" if result.side == OrderSide.BUY else "SELL"
            return LegacyDecisionAction(action=action, amount=float(result.quantity), target_price=float(result.price))
        return LegacyDecisionAction(action="HOLD", amount=0.0, target_price=price)

    def decide_from_belief(self, belief: Any, market_state: Optional[Dict[str, Any]] = None) -> Optional[ExecutionPlan]:
        """Derive an execution plan from policy interpretation belief."""
        if belief is None:
            return None
        expected_return = dict(getattr(belief, "expected_return", {}) or {})
        if not expected_return:
            return None
        symbol = str(max(expected_return.items(), key=lambda kv: abs(float(kv[1])))[0])
        alpha = float(expected_return.get(symbol, 0.0))
        confidence = float(getattr(belief, "confidence", 0.5) or 0.5)
        expected_risk = dict(getattr(belief, "expected_risk", {}) or {})
        liquidity_score = dict(getattr(belief, "liquidity_score", {}) or {})
        belief_metadata = dict(getattr(belief, "metadata", {}) or {})
        disagreement_tags = list(getattr(belief, "disagreement_tags", []) or [])
        latency_bars = int(getattr(belief, "latency_bars", 0) or 0)
        pass_through = float(belief_metadata.get("effective_pass_through", 1.0) or 0.0)
        channel_bias = float(belief_metadata.get("channel_bias", 0.0) or 0.0)
        if abs(alpha) < 1e-4:
            return None
        action = "BUY" if alpha > 0 else "SELL"
        ref_price = 0.0
        if market_state:
            prices = market_state.get("prices", {})
            if isinstance(prices, dict):
                ref_price = float(prices.get(symbol, 0.0) or 0.0)
            if ref_price <= 0:
                ref_price = float(market_state.get("last_price", 0.0) or 0.0)
        if ref_price <= 0 and self._price_history:
            ref_price = float(self._price_history[-1])
        if ref_price <= 0:
            ref_price = 1.0

        liquidity = float(liquidity_score.get(symbol, 0.5) or 0.5)
        risk_signal = float(expected_risk.get(symbol, 0.25) or 0.25)
        lag_penalty = max(0.25, 1.0 - 0.12 * float(latency_bars))
        disagreement_penalty = 1.0
        if "low_policy_confidence" in disagreement_tags:
            disagreement_penalty *= 0.70
        if "lagged_policy_pass_through" in disagreement_tags:
            disagreement_penalty *= 0.75
        if "weak_agent_specific_signal" in disagreement_tags:
            disagreement_penalty *= 0.85
        risk_penalty = max(0.35, 1.0 - 0.45 * max(0.0, risk_signal - 0.2))

        signal_strength = min(1.0, abs(alpha) / 0.12)
        pass_through_boost = max(0.20, min(1.0, pass_through))
        conviction = signal_strength * confidence * pass_through_boost * lag_penalty * disagreement_penalty * risk_penalty

        risk_budget = max(0.03, min(1.0, 0.10 + 0.90 * conviction))
        if action == "BUY":
            qty = int((self.cash_balance * risk_budget) / ref_price)
        else:
            qty = int(self.portfolio.get(symbol, 0) * risk_budget)
            if qty <= 0:
                qty = int(max(0, self.portfolio.get(symbol, 0)))
        qty = max(0, int(qty))
        if qty <= 0:
            return None

        if action == "BUY":
            urgency = max(0.05, min(1.0, 0.18 + 0.42 * conviction + 0.10 * max(0.0, channel_bias)))
            if liquidity < 0.45:
                order_type = "limit"
                slicing_rule = "twap-like" if qty >= 200 else "single"
                time_horizon = 4 if qty >= 200 else 2
                participation_rate = min(0.22, 0.08 + 0.12 * conviction)
            else:
                order_type = "limit" if conviction < 0.55 else "ioc"
                slicing_rule = "twap-like" if qty >= 400 else "single"
                time_horizon = 3 if qty >= 400 else 1
                participation_rate = min(0.35, 0.10 + 0.22 * conviction)
        else:
            stress_urgency = max(0.0, (1.0 - liquidity) * 0.35 + max(0.0, risk_signal - 0.2) * 0.45)
            urgency = max(0.10, min(1.0, 0.24 + 0.35 * conviction + stress_urgency))
            if liquidity < 0.40 or risk_signal > 0.55:
                order_type = "market" if urgency >= 0.70 else "ioc"
                slicing_rule = "single" if qty < 300 else "twap-like"
                time_horizon = 1 if qty < 300 else 2
                participation_rate = min(0.50, 0.18 + 0.30 * urgency)
            else:
                order_type = "limit"
                slicing_rule = "twap-like" if qty >= 400 else "single"
                time_horizon = 2 if qty >= 400 else 1
                participation_rate = min(0.35, 0.12 + 0.20 * conviction)

        max_slippage = max(0.0025, min(0.03, 0.004 + 0.012 * urgency + 0.006 * max(0.0, risk_signal - liquidity)))
        cancel_replace_policy = "cancel-replace" if order_type in {"limit", "post-only"} and time_horizon > 1 else "none"
        reasoning_output = {
            "symbol": symbol,
            "timestamp": time.time(),
            "decision": {
                "action": action,
                "qty": qty,
                "price": ref_price,
                "urgency": urgency,
                "order_type": order_type,
                "slicing_rule": slicing_rule,
                "cancel_replace_policy": cancel_replace_policy,
                "time_horizon": time_horizon,
                "participation_rate": participation_rate,
                "max_slippage": max_slippage,
            },
            "belief_metadata": belief_metadata,
        }
        plan = self._build_execution_plan(reasoning_output)
        if plan is None:
            return None
        plan.metadata["belief"] = {
            "confidence": confidence,
            "latency_bars": latency_bars,
            "disagreement_tags": disagreement_tags,
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "liquidity_score": liquidity_score,
            "effective_pass_through": pass_through,
            "channel_bias": channel_bias,
        }
        plan.metadata["belief_execution"] = {
            "conviction": float(conviction),
            "risk_budget": float(risk_budget),
            "lag_penalty": float(lag_penalty),
            "disagreement_penalty": float(disagreement_penalty),
            "risk_penalty": float(risk_penalty),
        }
        return plan

    async def update_memory(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        """Update memory and reinforcement signals after execution feedback."""
        pnl = float(outcome.get("pnl", 0.0) or 0.0)
        pnl_pct = 0.0
        self.brain.state.update_after_trade(pnl, pnl_pct)
        if hasattr(self.brain, "layered_memory"):
            try:
                self.brain.layered_memory.record_outcome(
                    decision=decision,
                    outcome=outcome,
                    market_state={
                        "news_source": outcome.get("source", "market"),
                        "panic_level": float(outcome.get("panic_level", 0.0) or 0.0),
                        "policy_description": outcome.get("policy_description", ""),
                    },
                    account_state={
                        "pnl_pct": pnl_pct,
                        "market_value": float(outcome.get("market_value", 0.0) or 0.0),
                    },
                )
            except Exception:
                pass

        if abs(pnl) > 1000:
            content = f"Decision: {decision}. Outcome: {outcome}"
            score = 1.0 if pnl > 0 else -1.0
            self.brain.memory.add_memory(content, score)
            extract_text = (
                f"交易复盘: action={decision.get('action')}, pnl={pnl}, confidence={self.brain.state.confidence}"
            )
            asyncio.create_task(self._async_extract_and_store(extract_text, time.time(), is_news=False))

        wind = decision.get("wind_tunnel", {})
        if wind.get("valid"):
            pred = float(wind.get("predicted_return", 0.0))
            actual = 0.0
            if len(self._price_history) >= 2:
                prev = self._price_history[-2]
                curr = self._price_history[-1]
                actual = (curr - prev) / prev if prev > 0 else 0.0
            if (pred * actual) > 0 and pnl > 0:
                self._wind_tunnel_confidence = min(1.0, self._wind_tunnel_confidence + 0.05)
            else:
                self._wind_tunnel_confidence = max(0.1, self._wind_tunnel_confidence - 0.02)

        self._recent_trade_outcomes.append({
            "decision": decision,
            "pnl": pnl,
            "status": outcome.get("status", "UNKNOWN"),
            "timestamp": time.time(),
        })
        if len(self._recent_trade_outcomes) > 20:
            self._recent_trade_outcomes = self._recent_trade_outcomes[-20:]

        if self.brain.state.consecutive_losses >= 3 or self.brain.state.consecutive_wins >= 3:
            self._reflect_and_persist_beliefs()

        status = outcome.get("status")
        if status == "REJECTED" or status == OrderStatus.REJECTED:
            reason = outcome.get("reason", "Unknown regulatory rejection")
            self.compliance_feedback.append(reason)
            if len(self.compliance_feedback) > 5:
                self.compliance_feedback.pop(0)
            content = f"REGULATORY REJECTION: {reason}. Action={decision.get('action')}"
            self.brain.memory.add_memory(content, -2.0)
            asyncio.create_task(self._async_extract_and_store(content, time.time(), is_news=False))
            self.brain.state.confidence *= 0.95

    # ------------------------------------------
    # 额外方法：面向交易智能体的专用需求
    # ------------------------------------------
    
    def get_psychology_description(self) -> str:
        """返回智能体心理画像摘要。"""
        return (
            f"Risk: {self.profile.get('risk_aversion', 0.5):.2f}, "
            f"Conf: {self.brain.state.confidence:.1f}"
        )

    def share_opinion(self) -> Dict[str, Any]:
        """Share local sentiment/opinion into social network."""
        sentiment = "neutral"
        if hasattr(self, "emotional_state"):
            if self.emotional_state in ("Greedy", "Confident"):
                sentiment = "bullish"
            elif self.emotional_state in ("Fearful", "Regretful", "Anxious"):
                sentiment = "bearish"

        if self.social_graph and self.social_node_id is not None:
            node = self.social_graph.agents.get(self.social_node_id)
            if node:
                if sentiment == "bearish":
                    node.sentiment_state = SentimentState.INFECTED
                elif sentiment == "bullish":
                    node.sentiment_state = SentimentState.BULLISH
                else:
                    node.sentiment_state = SentimentState.SUSCEPTIBLE

        return {
            "agent_id": self.agent_id,
            "sentiment": sentiment,
            "confidence": self.brain.state.confidence / 100.0,
            "emotional_state": getattr(self, "emotional_state", "Neutral"),
        }

    def receive_opinion(self, opinions: List[Dict[str, Any]]) -> str:
        """Receive neighbor opinions and summarize social pressure."""
        if not opinions:
            return "No social messages received."

        bearish_count = sum(1 for o in opinions if o.get("sentiment") == "bearish")
        bullish_count = sum(1 for o in opinions if o.get("sentiment") == "bullish")
        total = len(opinions)

        if bearish_count > total * 0.6:
            return f"{bearish_count}/{total} neighbors are panic selling."
        if bullish_count > total * 0.6:
            return f"{bullish_count}/{total} neighbors are aggressively buying."
        if bearish_count > bullish_count:
            return f"Circle is bearish ({bearish_count} vs {bullish_count})."
        if bullish_count > bearish_count:
            return f"Circle is bullish ({bullish_count} vs {bearish_count})."
        return "Circle is split and uncertain."

    def get_social_summary(self) -> str:
        """Return social sentiment summary for bound social graph."""
        if self.social_graph and self.social_node_id is not None:
            return self.social_graph.generate_social_summary(self.social_node_id)
        return "Not bound to social graph."
