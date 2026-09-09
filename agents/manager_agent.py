"""Manager orchestration agent.

This module replaces the old alias-only entry point with a real orchestration
class that aggregates analyst reports and produces a structured ExecutionIntent.
The implementation stays backward-compatible with BaseAgent-style callers and
keeps risky features behind feature flags.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from agents.base_agent import BaseAgent, MarketSnapshot
from agents.brain import build_runtime_metadata, resolve_feature_flags, stable_json_hash
from agents.roles.news_analyst import NewsAnalyst
from agents.roles.quant_analyst import QuantAnalyst
from agents.roles.risk_analyst import RiskAnalyst
from config import GLOBAL_CONFIG
from core.types import Order, OrderSide, OrderType


@dataclass
class ExecutionIntent:
    """Structured manager-approved execution request."""

    agent_id: str
    symbol: str
    action: str
    side: str
    target_notional: float = 0.0
    target_qty: int = 0
    urgency: str = "normal"
    order_type: str = "limit"
    max_slippage: float = 0.01
    participation_rate: float = 0.10
    slicing_rule: str = "single"
    cancel_replace_policy: str = "none"
    time_horizon: str = "intraday"
    approved: bool = True
    analyst_reports: Dict[str, Any] = field(default_factory=dict)
    debate: Dict[str, Any] = field(default_factory=dict)
    risk_review: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "symbol": self.symbol,
            "action": self.action,
            "side": self.side,
            "target_notional": float(self.target_notional),
            "target_qty": int(self.target_qty),
            "urgency": self.urgency,
            "order_type": self.order_type,
            "max_slippage": float(self.max_slippage),
            "participation_rate": float(self.participation_rate),
            "slicing_rule": self.slicing_rule,
            "cancel_replace_policy": self.cancel_replace_policy,
            "time_horizon": self.time_horizon,
            "approved": bool(self.approved),
            "analyst_reports": self.analyst_reports,
            "debate": self.debate,
            "risk_review": self.risk_review,
            "metadata": self.metadata,
        }

    def to_order(self) -> Optional[Order]:
        """Compatibility bridge for legacy code that still wants a single order."""
        if self.action.upper() == "HOLD":
            return None

        price = float(self.metadata.get("reference_price") or self.metadata.get("last_price") or 0.0)
        qty = int(self.target_qty)
        if qty <= 0 and self.target_notional > 0 and price > 0:
            qty = max(1, int(self.target_notional / price))
        if qty <= 0 or price <= 0:
            return None

        side = OrderSide.BUY if self.side.lower() == "buy" else OrderSide.SELL
        order_type = OrderType.MARKET if self.order_type.lower() in {"market", "ioc", "fok"} else OrderType.LIMIT
        order = Order(
            symbol=self.symbol,
            price=price,
            quantity=qty,
            side=side,
            order_type=order_type,
            agent_id=self.agent_id,
            timestamp=float(self.metadata.get("timestamp", time.time())),
        )
        order.reason = str(self.metadata.get("approval_reason", ""))
        return order


class ManagerAgent(BaseAgent):
    """Orchestrates analyst reports and emits a structured execution intent."""

    @staticmethod
    def plan_competition_task(task, team_config, *, experience_ids=(), feedback=()):
        """Layer A planning entry point; does not instantiate a trading account.

        The same Manager now supports task orchestration and its original
        ExecutionIntent API. Versioned prompt instructions change actual tools.
        """
        from core.team_evolution.config import prompt_checks
        from core.team_evolution.models import TaskPlan

        checks = prompt_checks(team_config.roles["planner"].system_prompt)
        graph = {role: [a for a, b in team_config.topology.communication_edges if b == role]
                 for role in team_config.topology.active_agents}
        subtasks = [{"subtask_id": role, "role_type": team_config.roles[role].role_type,
                     "tools": list(team_config.roles[role].tools), "checks": checks if role == "executor" else []}
                    for layer in team_config.topology.layers() for role in layer]
        return TaskPlan(subtasks, graph, {r: r for r in graph},
                        ["policy", "simulation", "risk"] + checks,
                        {"max_replans": 1, "max_tool_retries": team_config.tool_policy.get("max_retries", 0),
                         "budget_steps": task.constraints.get("max_steps", 100)},
                        decision_summary=f"{task.task_type}: 按执行契约安排 {len(subtasks)} 类分工；"
                                         f"额外核验 {checks}；中间反馈 {len(feedback)} 项。",
                        retrieved_experience_ids=list(experience_ids))

    def __init__(
        self,
        agent_id: str,
        cash_balance: float = GLOBAL_CONFIG.DEFAULT_CASH,
        portfolio: Optional[Dict[str, int]] = None,
        psychology_profile: Optional[Dict[str, float]] = None,
        persona: Optional[Any] = None,
        *,
        seed: Optional[int] = None,
        feature_flags: Optional[Mapping[str, bool]] = None,
        news_config: Optional[Dict[str, Any]] = None,
        news_max_articles: int = 5,
        use_debate: Optional[bool] = None,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            cash_balance=cash_balance,
            portfolio=portfolio,
            psychology_profile=psychology_profile,
        )
        self.persona = persona
        self.legacy_kwargs = dict(legacy_kwargs)
        self.seed = self._resolve_seed(seed)
        self._seed_rng(self.seed)
        self.feature_flags = resolve_feature_flags(feature_flags)
        if use_debate is not None:
            self.feature_flags["manager_debate_v1"] = bool(use_debate)

        self.news_analyst = NewsAnalyst(config_paths=news_config, max_articles=news_max_articles)
        self.quant_analyst = QuantAnalyst()
        self.risk_analyst = RiskAnalyst()

        self.last_reports: Dict[str, Any] = {}
        self.last_intent: Optional[ExecutionIntent] = None
        self.orchestration_history: List[Dict[str, Any]] = []
        self.config_hash = stable_json_hash(
            {
                "agent_id": self.agent_id,
                "seed": self.seed,
                "persona": getattr(self.persona, "to_dict", lambda: None)(),
                "feature_flags": dict(self.feature_flags),
                "legacy_kwargs": sorted(self.legacy_kwargs.keys()),
            }
        )
        self.runtime_metadata = build_runtime_metadata(
            seed=self.seed,
            config={
                "agent_id": self.agent_id,
                "cash_balance": float(self.cash_balance),
                "portfolio_size": len(self.portfolio),
                "persona": getattr(self.persona, "to_dict", lambda: None)(),
            },
            snapshot={},
            feature_flags=self.feature_flags,
        )
        self.runtime_metadata["config_hash"] = self.config_hash

    def _resolve_seed(self, seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        env_seed = os.environ.get("CIVITAS_SEED")
        if env_seed and env_seed.strip():
            try:
                return int(env_seed)
            except ValueError:
                pass
        return 0

    def _seed_rng(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    @staticmethod
    def _snapshot_to_dict(snapshot: MarketSnapshot) -> Dict[str, Any]:
        return {
            "symbol": snapshot.symbol,
            "last_price": float(snapshot.last_price),
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.bid_ask_spread,
            "market_trend": float(snapshot.market_trend),
            "panic_level": float(snapshot.panic_level),
            "timestamp": float(snapshot.timestamp),
            "policy_tax_rate": float(getattr(snapshot, "policy_tax_rate", 0.0)),
            "text_policy_shock": float(getattr(snapshot, "text_policy_shock", 0.0)),
            "text_regime_bias": str(getattr(snapshot, "text_regime_bias", "neutral")),
        }

    def _build_debate(self, reports: Dict[str, Any], score: float) -> Dict[str, Any]:
        bull_case = "Positive news and acceptable risk support accumulation."
        bear_case = "Risk or weak momentum argues for caution."
        if score > 0:
            winner = "bull"
        elif score < 0:
            winner = "bear"
        else:
            winner = "neutral"
        return {
            "enabled": True,
            "winner": winner,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "score": float(score),
            "reports": {
                "news": reports.get("news", {}),
                "quant": reports.get("quant", {}),
                "risk": reports.get("risk", {}),
            },
        }

    def _derive_action_score(self, reports: Dict[str, Any]) -> float:
        news = reports.get("news", {}) or {}
        quant = reports.get("quant", {}) or {}
        risk = reports.get("risk", {}) or {}

        news_score = float(news.get("sentiment_score", 0.0))
        quant_score = float(quant.get("momentum", 0.0)) + float(quant.get("herding_intensity", 0.0)) * 0.05
        risk_penalty = abs(float(risk.get("cvar", 0.0))) * 0.4 + float(risk.get("max_drawdown", 0.0)) * 0.25
        debate_bonus = 0.0
        if self.feature_flags.get("manager_debate_v1", False):
            debate_bonus = 0.02 if news_score >= 0 else -0.02
        return news_score * 0.45 + quant_score * 0.35 + debate_bonus - risk_penalty

    def _compose_intent(self, reasoning_output: Dict[str, Any]) -> ExecutionIntent:
        snapshot: MarketSnapshot = reasoning_output["snapshot"]
        reports = reasoning_output.get("analyst_reports", {})
        score = float(reasoning_output.get("decision_score", 0.0))
        risk = reports.get("risk", {}) or {}
        price = float(snapshot.last_price)
        portfolio_value = float(reasoning_output.get("portfolio_value") or self.get_total_value({snapshot.symbol: price}))

        risk_block = float(risk.get("max_drawdown", 0.0)) > 0.30 or float(risk.get("cvar", 0.0)) < -0.08
        if risk_block:
            action = "HOLD"
        elif score > 0.15:
            action = "BUY"
        elif score < -0.15:
            action = "SELL"
        else:
            action = "HOLD"

        urgency = "high" if abs(score) >= 0.35 else "normal" if abs(score) >= 0.18 else "low"
        approved = action != "HOLD" and not risk_block
        side = "buy" if action == "BUY" else "sell" if action == "SELL" else "none"
        target_notional = 0.0
        target_qty = 0
        if action != "HOLD" and price > 0:
            base_notional = portfolio_value * self._clamp(abs(score), 0.08, 0.25)
            target_notional = float(base_notional)
            target_qty = max(1, int(target_notional / price))

        if target_notional >= 250_000:
            slicing_rule = "vwap"
        elif target_notional >= 75_000:
            slicing_rule = "twap"
        else:
            slicing_rule = "single"

        order_type = "market" if urgency == "high" and approved else "limit"
        max_slippage = 0.005 if urgency == "high" else 0.01
        participation_rate = self._clamp(abs(score) * 0.2, 0.05, 0.25)
        cancel_replace_policy = "cancel_replace" if order_type == "limit" and approved else "no_requote"
        time_horizon = "intraday" if urgency != "low" else "multi_step"

        metadata = build_runtime_metadata(
            seed=self.seed,
            config={
                "agent_id": self.agent_id,
                "score": score,
                "risk_block": risk_block,
                "portfolio_value": portfolio_value,
            },
            snapshot=self._snapshot_to_dict(snapshot),
            feature_flags=self.feature_flags,
        )
        metadata.update(
            {
                "last_price": price,
                "reference_price": price,
                "decision_score": score,
                "approval_reason": "risk_veto" if risk_block else "manager_approval",
            }
        )

        return ExecutionIntent(
            agent_id=self.agent_id,
            symbol=snapshot.symbol,
            action=action,
            side=side,
            target_notional=target_notional,
            target_qty=target_qty,
            urgency=urgency,
            order_type=order_type,
            max_slippage=max_slippage,
            participation_rate=participation_rate,
            slicing_rule=slicing_rule,
            cancel_replace_policy=cancel_replace_policy,
            time_horizon=time_horizon,
            approved=approved,
            analyst_reports=reports,
            debate=reasoning_output.get("debate", {}),
            risk_review=risk,
            metadata=metadata,
        )

    async def perceive(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
        **context: Any,
    ) -> Dict[str, Any]:
        return {
            "snapshot": market_snapshot,
            "public_news": list(public_news),
            "context": dict(context),
            "metadata": {
                "seed": self.seed,
                "config_hash": self.config_hash,
                "feature_flags": dict(self.feature_flags),
                "snapshot": self._snapshot_to_dict(market_snapshot),
            },
        }

    async def reason(self, perceived_data: Dict[str, Any]) -> Dict[str, Any]:
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        context = perceived_data.get("context", {}) or {}
        reports = dict(context.get("analyst_reports", {}))

        if not self.feature_flags.get("manager_orchestration_v1", True):
            reasoning_output = {
                "snapshot": snapshot,
                "public_news": perceived_data.get("public_news", []),
                "analyst_reports": reports,
                "decision_score": 0.0,
                "debate": {},
                "portfolio_value": context.get("portfolio_value"),
                "metadata": perceived_data.get("metadata", {}),
            }
            self.last_reports = reports
            return reasoning_output

        if self.feature_flags.get("manager_external_analysts_v1", False) and not reports:
            reports["news"] = self.news_analyst.analyze()
            reports["quant"] = self.quant_analyst.analyze(
                snapshot,
                returns=context.get("returns"),
                price_series=context.get("price_series"),
            )
            reports["risk"] = self.risk_analyst.analyze(context.get("portfolio_values", []))
        else:
            reports.setdefault("news", context.get("news_report", {}))
            reports.setdefault("quant", context.get("quant_report", {}))
            reports.setdefault("risk", context.get("risk_report", {}))

        score = self._derive_action_score(reports)
        debate = self._build_debate(reports, score) if self.feature_flags.get("manager_debate_v1", False) else {}
        reasoning_output = {
            "snapshot": snapshot,
            "public_news": perceived_data.get("public_news", []),
            "analyst_reports": reports,
            "decision_score": score,
            "debate": debate,
            "portfolio_value": context.get("portfolio_value"),
            "metadata": perceived_data.get("metadata", {}),
        }
        self.last_reports = reports
        return reasoning_output

    async def decide(self, reasoning_output: Dict[str, Any]) -> ExecutionIntent:
        intent = self._compose_intent(reasoning_output)
        self.last_intent = intent
        self.orchestration_history.append(
            {
                "timestamp": time.time(),
                "intent": intent.to_dict(),
                "metadata": intent.metadata,
            }
        )
        return intent

    async def act(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
        **context: Any,
    ) -> ExecutionIntent:
        self._step_count += 1
        perceived = await self.perceive(market_snapshot, public_news, **context)
        reasoning = await self.reason(perceived)
        return await self.decide(reasoning)

    async def update_memory(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        self.memory.append(
            {
                "decision": dict(decision),
                "outcome": dict(outcome),
                "config_hash": self.config_hash,
                "seed": self.seed,
            }
        )
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def summary(self) -> Dict[str, Any]:
        """Compact reproducibility summary for logs/tests."""
        return {
            "agent_id": self.agent_id,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "feature_flags": dict(self.feature_flags),
            "snapshot": dict(self.runtime_metadata.get("snapshot", {})),
        }


__all__ = ["ExecutionIntent", "ManagerAgent"]
