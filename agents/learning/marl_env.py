"""Minimal MARL environment interface for optional training integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class MARLStepResult:
    observations: Dict[str, Dict[str, float]]
    rewards: Dict[str, float]
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MARLEnvironment:
    """Gym-like multi-agent interface, intentionally training-backend neutral."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        *,
        initial_state: Optional[Mapping[str, Any]] = None,
        reward_fn: Optional[Callable[[str, Mapping[str, Any], Mapping[str, Any]], float]] = None,
        horizon: int = 128,
        seed: int = 42,
    ) -> None:
        self.agent_ids = [str(agent_id) for agent_id in agent_ids]
        self.initial_state = dict(initial_state or {})
        self.reward_fn = reward_fn
        self.horizon = max(1, int(horizon))
        self.seed = int(seed)
        self.tick = 0
        self.state: Dict[str, Any] = dict(self.initial_state)

    def _observation_for(self, agent_id: str) -> Dict[str, float]:
        return {
            "tick": float(self.tick),
            "price": float(self.state.get("price", self.state.get("last_price", 100.0)) or 100.0),
            "sentiment_index": float(self.state.get("sentiment_index", 0.5) or 0.5),
            "inventory": float(self.state.get("inventory", {}).get(agent_id, 0.0) if isinstance(self.state.get("inventory"), Mapping) else 0.0),
            "event_shock": float(self.state.get("event_shock", 0.0) or 0.0),
        }

    def reset(self, *, seed: Optional[int] = None, state: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        if seed is not None:
            self.seed = int(seed)
        self.tick = 0
        self.state = {**self.initial_state, **dict(state or {})}
        return {agent_id: self._observation_for(agent_id) for agent_id in self.agent_ids}

    def step(self, actions: Mapping[str, Mapping[str, Any] | str]) -> MARLStepResult:
        self.tick += 1
        inventory = dict(self.state.get("inventory", {}) or {}) if isinstance(self.state.get("inventory"), Mapping) else {}
        net_flow = 0.0
        for agent_id in self.agent_ids:
            action_payload = actions.get(agent_id, "HOLD")
            action = str(action_payload.get("action", "HOLD") if isinstance(action_payload, Mapping) else action_payload).upper()
            qty = float(action_payload.get("target_qty", action_payload.get("amount", 1.0)) if isinstance(action_payload, Mapping) else 1.0)
            if action == "BUY":
                inventory[agent_id] = float(inventory.get(agent_id, 0.0) + qty)
                net_flow += qty
            elif action == "SELL":
                inventory[agent_id] = float(inventory.get(agent_id, 0.0) - qty)
                net_flow -= qty
        old_price = float(self.state.get("price", self.state.get("last_price", 100.0)) or 100.0)
        price = max(0.01, old_price * (1.0 + max(-0.05, min(0.05, net_flow * 0.00001))))
        self.state["price"] = float(price)
        self.state["inventory"] = inventory
        observations = {agent_id: self._observation_for(agent_id) for agent_id in self.agent_ids}
        rewards: Dict[str, float] = {}
        for agent_id in self.agent_ids:
            action_payload = actions.get(agent_id, "HOLD")
            action = str(action_payload.get("action", "HOLD") if isinstance(action_payload, Mapping) else action_payload).upper()
            if self.reward_fn is not None:
                rewards[agent_id] = float(self.reward_fn(agent_id, self.state, action_payload if isinstance(action_payload, Mapping) else {"action": action}))
            else:
                position = float(inventory.get(agent_id, 0.0))
                rewards[agent_id] = float(position * (price - old_price) - abs(position) * 0.00001)
        terminated = bool(self.tick >= self.horizon)
        return MARLStepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=False,
            info={"tick": int(self.tick), "net_flow": float(net_flow), "seed": int(self.seed)},
        )

    def observation_space_spec(self) -> Dict[str, List[str]]:
        keys = ["tick", "price", "sentiment_index", "inventory", "event_shock"]
        return {agent_id: list(keys) for agent_id in self.agent_ids}

    def action_space_spec(self) -> Dict[str, List[str]]:
        return {agent_id: ["BUY", "HOLD", "SELL"] for agent_id in self.agent_ids}


__all__ = ["MARLEnvironment", "MARLStepResult"]
