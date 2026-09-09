"""
Regulator Agent 顶层框架。

目标：
1. 将“海量交易 Agent + 市场波动”抽象为统一 RL 环境；
2. 用“宏观平稳度 + 崩盘损失率”构建奖励函数；
3. 支持动作空间：印花税、降准、降息、定向辟谣；
4. 提供基础训练循环，输出最优组合调控方案。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from config import GLOBAL_CONFIG
from core.optimization import bayesian_search, generate_optimization_report, nsga_search, sensitivity_analysis
from core.objective_discovery import ObjectiveDiscoveryEngine


DEFAULT_REGULATOR_FEATURE_FLAGS: Dict[str, bool] = {
    "regulator_real_env_v1": True,
    "regulator_toy_fallback_v1": True,
}


def _has_configured_live_model() -> bool:
    """Real env is only practical when at least one online model provider is configured."""
    deepseek_key = str(getattr(GLOBAL_CONFIG, "DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    zhipu_key = str(getattr(GLOBAL_CONFIG, "ZHIPU_API_KEY", "") or os.environ.get("ZHIPU_API_KEY", "")).strip()
    return bool(deepseek_key or zhipu_key)


def _run_coro_blocking(coro: Any, *, timeout: float) -> Any:
    """
    同步上下文下安全执行协程，避免“慢速 I/O + 快速状态机”耦合导致的阻塞死锁。

    设计要点：
    1. 当前线程没有事件循环时，直接 `asyncio.run`，路径最短、开销最低。
    2. 当前线程已存在运行中的事件循环时，不能再在同线程上 `future.result()` 阻塞等待，
       否则会卡住事件循环自身。这里改为在独立线程中新建事件循环执行协程，再同步回收结果。
    3. 通过 timeout 对桥接线程做硬约束，避免策略训练在异常协程上无限等待。
    """
    try:
        # 仅用于探测“当前线程是否已有运行中的 事件循环”
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - 由主线程统一抛出
            error_box["error"] = exc

    worker = threading.Thread(target=_runner, daemon=True, name="regulator-coro-bridge")
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError(f"Coroutine bridge timed out after {timeout:.1f}s")
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


# ==========================================================
# 1) 统一环境接口
# ==========================================================


@dataclass
class EnvObservation:
    """
    监管智能体可观测状态。

    字段说明：
    - macro_stability: 宏观平稳度，越大越稳定 (0~1)
    - crash_loss_rate: 崩盘损失率，越大越糟糕 (0~1)
    - volatility: 市场波动率，越大风险越高
    - panic_level: 市场恐慌因子 (0~1)
    """

    step: int
    macro_stability: float
    crash_loss_rate: float
    volatility: float
    panic_level: float
    agent_count: int
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegulatoryAction:
    """
    监管动作（组合调控）定义。
    """

    stamp_tax_rate: float
    reserve_cut_bps: int
    policy_rate_cut_bps: int
    rumor_refute_strength: float
    stabilization_capital: float = 0.0
    stabilization_timing: int = 0
    halt_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stamp_tax_rate": self.stamp_tax_rate,
            "reserve_cut_bps": self.reserve_cut_bps,
            "policy_rate_cut_bps": self.policy_rate_cut_bps,
            "rumor_refute_strength": self.rumor_refute_strength,
            "stabilization_capital": self.stabilization_capital,
            "stabilization_timing": self.stabilization_timing,
            "halt_enabled": self.halt_enabled,
        }

    def describe(self) -> str:
        return (
            f"印花税={self.stamp_tax_rate:.4%}, "
            f"降准={self.reserve_cut_bps}bps, "
            f"降息={self.policy_rate_cut_bps}bps, "
            f"定向辟谣强度={self.rumor_refute_strength:.2f}, "
            f"稳定资金={self.stabilization_capital:.2f}, "
            f"干预时点={self.stabilization_timing}, "
            f"halt={'on' if self.halt_enabled else 'off'}"
        )


class RegulatoryEnvironment(Protocol):
    """统一环境协议：可替换为真实沙盘或轻量仿真。"""

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        ...

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        ...


# ==========================================================
# 2) 奖励函数
# ==========================================================


@dataclass
class RewardFunction:
    """
    奖励函数：
    - 奖励宏观平稳度；
    - 惩罚崩盘损失率；
    - 额外轻惩罚高波动（防止“表面稳定、底层剧烈震荡”）。
    """

    w_macro_stability: float = 1.8
    w_crash_risk_penalty: float = 2.4
    w_volatility_penalty: float = 0.4
    w_liquidity_shortage_penalty: float = 0.8
    w_intervention_cost_penalty: float = 0.6
    w_welfare_confidence_reward: float = 0.7
    w_stability_improve: float = 0.6
    w_financing_function_reward: float = 0.5
    w_fairness_compliance_reward: float = 0.4

    def explain(self, obs: EnvObservation, prev_obs: Optional[EnvObservation] = None) -> Dict[str, Any]:
        liquidity_shortage = float(obs.extras.get("liquidity_shortage", 0.0)) if obs.extras else 0.0
        intervention_cost = float(obs.extras.get("intervention_cost", 0.0)) if obs.extras else 0.0
        welfare_confidence = float(obs.extras.get("welfare_confidence", 1.0 - obs.panic_level)) if obs.extras else float(1.0 - obs.panic_level)
        liquidity = float(np.clip(1.0 - liquidity_shortage, 0.0, 1.0))
        financing_function = (
            float(obs.extras.get("financing_function", 1.0 - 0.50 * obs.volatility - 0.35 * obs.panic_level))
            if obs.extras
            else float(1.0 - 0.50 * obs.volatility - 0.35 * obs.panic_level)
        )
        fairness_compliance = (
            float(obs.extras.get("fairness_compliance", 1.0 - 0.35 * obs.panic_level - 0.45 * obs.crash_loss_rate))
            if obs.extras
            else float(1.0 - 0.35 * obs.panic_level - 0.45 * obs.crash_loss_rate)
        )
        weighted_components = {
            "stability": self.w_macro_stability * float(obs.macro_stability),
            "crash_risk_penalty": -self.w_crash_risk_penalty * float(obs.crash_loss_rate),
            "volatility_penalty": -self.w_volatility_penalty * float(obs.volatility),
            "liquidity_penalty": -self.w_liquidity_shortage_penalty * float(liquidity_shortage),
            "financing_function_reward": self.w_financing_function_reward * float(np.clip(financing_function, 0.0, 1.0)),
            "fairness_compliance_reward": self.w_fairness_compliance_reward * float(np.clip(fairness_compliance, 0.0, 1.0)),
            "intervention_cost_penalty": -self.w_intervention_cost_penalty * float(intervention_cost),
            "welfare_confidence_reward": self.w_welfare_confidence_reward * float(welfare_confidence),
            "stability_improve_reward": 0.0,
        }
        if prev_obs is not None:
            weighted_components["stability_improve_reward"] = self.w_stability_improve * (
                float(obs.macro_stability) - float(prev_obs.macro_stability)
            )
        reward = float(sum(weighted_components.values()))
        return {
            "reward": reward,
            "raw_metrics": {
                "macro_stability": float(obs.macro_stability),
                "crash_risk": float(obs.crash_loss_rate),
                "volatility": float(obs.volatility),
                "panic_level": float(obs.panic_level),
                "liquidity": liquidity,
                "liquidity_shortage": float(liquidity_shortage),
                "financing_function": float(np.clip(financing_function, 0.0, 1.0)),
                "fairness_compliance": float(np.clip(fairness_compliance, 0.0, 1.0)),
                "intervention_cost": float(intervention_cost),
                "welfare_confidence": float(welfare_confidence),
            },
            "weighted_components": weighted_components,
        }

    def __call__(self, obs: EnvObservation, prev_obs: Optional[EnvObservation] = None) -> float:
        return float(self.explain(obs, prev_obs=prev_obs)["reward"])


def resolve_regulator_feature_flags(feature_flags: Optional[Mapping[str, Any]] = None) -> Dict[str, bool]:
    merged = dict(DEFAULT_REGULATOR_FEATURE_FLAGS)
    if feature_flags:
        for key in merged:
            if key in feature_flags:
                merged[key] = bool(feature_flags[key])
    return merged


def _derive_policy_support_metrics(
    *,
    volatility: float,
    crash_loss_rate: float,
    panic_level: float,
    stabilization_capital: float,
    rumor_refute_strength: float,
) -> Dict[str, float]:
    financing_function = float(
        np.clip(
            1.0
            - 0.42 * volatility
            - 0.28 * panic_level
            - 0.18 * crash_loss_rate
            + 0.15 * max(0.0, stabilization_capital),
            0.0,
            1.0,
        )
    )
    fairness_compliance = float(
        np.clip(
            1.0
            - 0.30 * panic_level
            - 0.30 * crash_loss_rate
            + 0.22 * max(0.0, rumor_refute_strength)
            + 0.08 * max(0.0, stabilization_capital),
            0.0,
            1.0,
        )
    )
    return {
        "financing_function": financing_function,
        "fairness_compliance": fairness_compliance,
    }


# ==========================================================
# 3) 轻量沙盘环境（默认）
# ==========================================================


class ToyRegulatoryMarketEnv(RegulatoryEnvironment):
    """
    轻量可训练环境。

    说明：
    - 该环境用于快速训练与框架验证；
    - 可在后续替换为 Civitas 真正的多智能体仿真环境适配器。
    """

    def __init__(
        self,
        agent_count: int = 10_000,
        episode_steps: int = 48,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.agent_count = agent_count
        self.episode_steps = episode_steps
        self.reward_fn = reward_fn or RewardFunction()
        self._rng = random.Random(42)
        self._step = 0
        self._obs = EnvObservation(
            step=0,
            macro_stability=0.5,
            crash_loss_rate=0.2,
            volatility=0.06,
            panic_level=0.4,
            agent_count=self.agent_count,
        )
        self._target_action = RegulatoryAction(
            stamp_tax_rate=0.0005,
            reserve_cut_bps=25,
            policy_rate_cut_bps=10,
            rumor_refute_strength=0.6,
        )

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        if seed is not None:
            self._rng.seed(seed)
        self._step = 0

        # 每个 episode 随机化“外部冲击强度”，模拟不同宏观周期。
        shock = self._rng.uniform(0.0, 0.35)
        self._obs = EnvObservation(
            step=0,
            macro_stability=float(np.clip(0.68 - 0.5 * shock, 0.0, 1.0)),
            crash_loss_rate=float(np.clip(0.08 + 0.6 * shock, 0.0, 1.0)),
            volatility=float(np.clip(0.02 + 0.20 * shock, 0.0, 1.0)),
            panic_level=float(np.clip(0.15 + 0.70 * shock, 0.0, 1.0)),
            agent_count=self.agent_count,
            extras={"shock": shock},
        )
        return self._obs

    def _action_distance(self, action: RegulatoryAction) -> float:
        # 将不同量纲归一化到 [0,1] 后计算距离，距离越小说明越接近“有效组合”。
        tax_diff = abs(action.stamp_tax_rate - self._target_action.stamp_tax_rate) / 0.0015
        rrr_diff = abs(action.reserve_cut_bps - self._target_action.reserve_cut_bps) / 100.0
        rate_diff = abs(action.policy_rate_cut_bps - self._target_action.policy_rate_cut_bps) / 50.0
        rumor_diff = abs(action.rumor_refute_strength - self._target_action.rumor_refute_strength)
        capital_diff = abs(action.stabilization_capital - self._target_action.stabilization_capital) / 1.0
        timing_diff = abs(action.stabilization_timing - self._target_action.stabilization_timing) / 10.0
        return float(
            np.clip(
                0.30 * tax_diff
                + 0.20 * rrr_diff
                + 0.15 * rate_diff
                + 0.15 * rumor_diff
                + 0.10 * capital_diff
                + 0.10 * timing_diff,
                0.0,
                1.0,
            )
        )

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        prev_obs = self._obs
        self._step += 1

        # 控制增益：动作越接近有效组合，越能压制波动和崩盘损失。
        distance = self._action_distance(action)
        control_gain = 1.0 - distance

        noise = self._rng.uniform(-0.015, 0.015)
        intervention_effect = float(np.clip(action.stabilization_capital, 0.0, 1.0)) * 0.15
        next_vol = float(
            np.clip(
                0.78 * prev_obs.volatility
                + 0.14 * (1.0 - control_gain)
                + 0.10 * prev_obs.panic_level
                - intervention_effect
                + noise,
                0.0,
                1.0,
            )
        )
        next_panic = float(
            np.clip(
                0.75 * prev_obs.panic_level
                + 0.16 * (1.0 - control_gain)
                - 0.18 * action.rumor_refute_strength
                - intervention_effect * 0.4
                + noise,
                0.0,
                1.0,
            )
        )
        next_crash = float(np.clip(0.80 * prev_obs.crash_loss_rate + 0.22 * max(0.0, next_vol + next_panic - 0.85) + noise, 0.0, 1.0))
        next_stability = float(np.clip(1.0 - (0.55 * next_vol + 0.35 * next_panic + 0.45 * next_crash), 0.0, 1.0))
        liquidity_shortage = float(np.clip(0.55 * next_panic + 0.35 * next_vol - 0.25 * intervention_effect, 0.0, 1.0))
        intervention_cost = float(
            np.clip(
                abs(action.stamp_tax_rate - 0.0005) * 120.0
                + abs(action.reserve_cut_bps) / 120.0
                + abs(action.policy_rate_cut_bps) / 80.0
                + max(0.0, action.stabilization_capital),
                0.0,
                2.5,
            )
        )
        welfare_confidence = float(np.clip(1.0 - 0.55 * next_panic - 0.25 * next_vol + 0.20 * intervention_effect, 0.0, 1.0))
        support_metrics = _derive_policy_support_metrics(
            volatility=next_vol,
            crash_loss_rate=next_crash,
            panic_level=next_panic,
            stabilization_capital=float(action.stabilization_capital),
            rumor_refute_strength=float(action.rumor_refute_strength),
        )

        self._obs = EnvObservation(
            step=self._step,
            macro_stability=next_stability,
            crash_loss_rate=next_crash,
            volatility=next_vol,
            panic_level=next_panic,
            agent_count=self.agent_count,
            extras={
                "control_gain": control_gain,
                "distance": distance,
                "liquidity_shortage": liquidity_shortage,
                "intervention_cost": intervention_cost,
                "welfare_confidence": welfare_confidence,
                **support_metrics,
            },
        )

        reward_breakdown = self.reward_fn.explain(self._obs, prev_obs=prev_obs)
        reward = float(reward_breakdown["reward"])
        done = bool(self._step >= self.episode_steps or next_crash >= 0.95)
        info = {"distance": distance, "control_gain": control_gain, "reward_breakdown": reward_breakdown}
        return self._obs, reward, done, info


# ==========================================================
# 4) 真实 Civitas 环境适配器（可选接入）
# ==========================================================


class CivitasMarketEnvAdapter(RegulatoryEnvironment):
    """
    真实沙盘适配器框架。

    通过 model_factory 注入具体的 CivitasModel/Controller 实例，
    使监管智能体把“成千上万交易 Agent + 市场波动”作为统一环境学习。
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        episode_steps: int = 24,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.model_factory = model_factory
        self.episode_steps = episode_steps
        self.reward_fn = reward_fn or RewardFunction()
        self.model: Any = None
        self._step = 0
        self._last_obs: Optional[EnvObservation] = None
        self._last_action: RegulatoryAction = RegulatoryAction(0.0005, 0, 0, 0.0)

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        self.model = self.model_factory()
        self._step = 0
        self._last_obs = self._build_observation(step=0)
        return self._last_obs

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        self._last_action = action
        self._apply_action(action)
        self._advance_model_one_step()
        self._step += 1

        obs = self._build_observation(step=self._step)
        reward_breakdown = self.reward_fn.explain(obs, prev_obs=self._last_obs)
        reward = float(reward_breakdown["reward"])
        self._last_obs = obs
        done = bool(self._step >= self.episode_steps or obs.crash_loss_rate > 0.45)
        return obs, reward, done, {"action": action.to_dict(), "reward_breakdown": reward_breakdown}

    def _apply_action(self, action: RegulatoryAction) -> None:
        """
        将 RL 动作映射到现有监管参数。

        映射策略（可按业务进一步精化）：
        - 印花税 -> policy_manager.tax.rate
        - 降准 -> liquidity_injection 增量
        - 降息 -> risk_free_rate 下调
        - 定向辟谣 -> panic_level 下降
        """
        if self.model is None:
            return

        try:
            market = getattr(self.model, "market_manager", None)
            if market is None:
                return

            pm = getattr(market, "policy_manager", None)
            if pm is not None and hasattr(pm, "set_policy_param"):
                pm.set_policy_param("tax", "rate", float(action.stamp_tax_rate))

            policy = getattr(market, "policy", None)
            if policy is not None:
                if hasattr(policy, "liquidity_injection"):
                    policy.liquidity_injection = float(
                        getattr(policy, "liquidity_injection", 0.0)
                        + action.reserve_cut_bps * 1_000_000.0
                        + max(0.0, action.stabilization_capital) * 5_000_000.0
                    )
                if hasattr(policy, "risk_free_rate"):
                    policy.risk_free_rate = float(max(0.0, getattr(policy, "risk_free_rate", 0.02) - action.policy_rate_cut_bps / 10_000.0))

            if hasattr(self.model, "panic_level"):
                self.model.panic_level = float(
                    np.clip(
                        self.model.panic_level
                        - 0.25 * action.rumor_refute_strength
                        - 0.20 * max(0.0, action.stabilization_capital),
                        0.0,
                        1.0,
                    )
                )
            if action.halt_enabled:
                regulatory_module = getattr(self.model, "regulatory_module", None)
                cb = getattr(regulatory_module, "circuit_breaker", None) if regulatory_module is not None else None
                if cb is not None and hasattr(cb, "is_halted"):
                    cb.is_halted = True
        except Exception:
            # 顶层框架保持容错，避免某个字段不存在导致训练中断。
            return

    def _advance_model_one_step(self) -> None:
        if self.model is None:
            return
        if hasattr(self.model, "async_step"):
            coro = self.model.async_step()
            self._run_coro(coro)
        elif hasattr(self.model, "step"):
            self.model.step()

    @staticmethod
    def _run_coro(coro: Any) -> Any:
        return _run_coro_blocking(coro, timeout=30.0)

    def _build_observation(self, step: int) -> EnvObservation:
        if self.model is None:
            return EnvObservation(
                step=step,
                macro_stability=0.5,
                crash_loss_rate=0.2,
                volatility=0.1,
                panic_level=0.5,
                agent_count=0,
            )

        price_history = list(getattr(self.model, "price_history", []))
        if len(price_history) >= 3:
            prices = np.array(price_history[-64:], dtype=float)
            returns = np.diff(prices) / np.clip(prices[:-1], 1e-8, None)
            volatility = float(np.std(returns))
            peak = np.maximum.accumulate(prices)
            drawdown = (peak - prices) / np.clip(peak, 1e-8, None)
            crash_loss_rate = float(np.max(drawdown))
        else:
            volatility = 0.0
            crash_loss_rate = 0.0

        panic_level = float(getattr(self.model, "panic_level", 0.0))
        macro_stability = float(np.clip(1.0 - (12.0 * volatility + 0.6 * crash_loss_rate + 0.35 * panic_level), 0.0, 1.0))
        liquidity_shortage = float(np.clip(0.60 * panic_level + 0.35 * volatility - 0.25 * max(0.0, self._last_action.stabilization_capital), 0.0, 1.0))
        intervention_cost = float(
            np.clip(
                abs(self._last_action.stamp_tax_rate - 0.0005) * 120.0
                + abs(self._last_action.reserve_cut_bps) / 120.0
                + abs(self._last_action.policy_rate_cut_bps) / 80.0
                + max(0.0, self._last_action.stabilization_capital),
                0.0,
                2.5,
            )
        )
        welfare_confidence = float(np.clip(1.0 - 0.55 * panic_level - 0.25 * volatility + 0.20 * max(0.0, self._last_action.stabilization_capital), 0.0, 1.0))
        support_metrics = _derive_policy_support_metrics(
            volatility=volatility,
            crash_loss_rate=crash_loss_rate,
            panic_level=panic_level,
            stabilization_capital=float(self._last_action.stabilization_capital),
            rumor_refute_strength=float(self._last_action.rumor_refute_strength),
        )

        return EnvObservation(
            step=step,
            macro_stability=macro_stability,
            crash_loss_rate=float(np.clip(crash_loss_rate, 0.0, 1.0)),
            volatility=float(np.clip(volatility, 0.0, 1.0)),
            panic_level=float(np.clip(panic_level, 0.0, 1.0)),
            agent_count=int(len(getattr(self.model, "agents", []))),
            extras={
                "price_len": len(price_history),
                "liquidity_shortage": liquidity_shortage,
                "intervention_cost": intervention_cost,
                "welfare_confidence": welfare_confidence,
                **support_metrics,
            },
        )


class SimulationControllerEnvAdapter(RegulatoryEnvironment):
    """
    基于 SimulationController 的训练环境适配器。

    适配思路：
    - reset: 创建一个新的 SimulationController（或兼容对象）作为 episode 环境；
    - step: 将监管动作映射到控制器的市场/政策参数，然后执行 run_tick；
    - observation: 从 controller.model 与 controller.market 聚合状态。
    """

    def __init__(
        self,
        controller_factory: Callable[[], Any],
        episode_steps: int = 24,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.controller_factory = controller_factory
        self.episode_steps = episode_steps
        self.reward_fn = reward_fn or RewardFunction()
        self.controller: Any = None
        self._step = 0
        self._last_obs: Optional[EnvObservation] = None
        self._last_action: RegulatoryAction = RegulatoryAction(0.0005, 0, 0, 0.0)

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        self.controller = self.controller_factory()
        self._step = 0
        self._last_obs = self._build_observation(step=0)
        return self._last_obs

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        self._last_action = action
        self._apply_action(action)
        self._run_controller_tick()
        self._step += 1

        obs = self._build_observation(step=self._step)
        reward_breakdown = self.reward_fn.explain(obs, prev_obs=self._last_obs)
        reward = float(reward_breakdown["reward"])
        self._last_obs = obs
        done = bool(self._step >= self.episode_steps or obs.crash_loss_rate > 0.45)
        return obs, reward, done, {"action": action.to_dict(), "reward_breakdown": reward_breakdown}

    def _apply_action(self, action: RegulatoryAction) -> None:
        if self.controller is None:
            return
        try:
            market = getattr(self.controller, "market", None)
            if market is None:
                return

            pm = getattr(market, "policy_manager", None)
            if pm is not None and hasattr(pm, "set_policy_param"):
                pm.set_policy_param("tax", "rate", float(action.stamp_tax_rate))

            policy = getattr(market, "policy", None)
            if policy is not None:
                if hasattr(policy, "liquidity_injection"):
                    policy.liquidity_injection = float(
                        getattr(policy, "liquidity_injection", 0.0)
                        + action.reserve_cut_bps * 1_000_000.0
                        + max(0.0, action.stabilization_capital) * 5_000_000.0
                    )
                if hasattr(policy, "risk_free_rate"):
                    policy.risk_free_rate = float(max(0.0, getattr(policy, "risk_free_rate", 0.02) - action.policy_rate_cut_bps / 10_000.0))

            if hasattr(market, "panic_level"):
                market.panic_level = float(
                    np.clip(
                        market.panic_level
                        - 0.25 * action.rumor_refute_strength
                        - 0.20 * max(0.0, action.stabilization_capital),
                        0.0,
                        1.0,
                    )
                )
            if action.halt_enabled:
                regulatory_module = getattr(market, "regulatory_module", None)
                cb = getattr(regulatory_module, "circuit_breaker", None) if regulatory_module is not None else None
                if cb is not None and hasattr(cb, "is_halted"):
                    cb.is_halted = True
            if hasattr(market, "current_news"):
                market.current_news = (
                    f"[RegulatorRefute] 辟谣强度={action.rumor_refute_strength:.2f}，"
                    f"税率={action.stamp_tax_rate:.4%}，降准={action.reserve_cut_bps}bps，降息={action.policy_rate_cut_bps}bps，"
                    f"稳定资金={action.stabilization_capital:.2f}。"
                )
        except Exception:
            return

    def _run_controller_tick(self) -> None:
        if self.controller is None:
            return
        run_tick = getattr(self.controller, "run_tick", None)
        if run_tick is None:
            return
        result = run_tick()
        if asyncio.iscoroutine(result):
            self._run_coro(result)

    @staticmethod
    def _run_coro(coro: Any) -> Any:
        return _run_coro_blocking(coro, timeout=60.0)

    def _build_observation(self, step: int) -> EnvObservation:
        if self.controller is None:
            return EnvObservation(
                step=step,
                macro_stability=0.5,
                crash_loss_rate=0.2,
                volatility=0.1,
                panic_level=0.5,
                agent_count=0,
            )

        model = getattr(self.controller, "model", None)
        market = getattr(self.controller, "market", None)
        price_history = list(getattr(model, "price_history", [])) if model is not None else []

        if len(price_history) >= 3:
            prices = np.array(price_history[-64:], dtype=float)
            returns = np.diff(prices) / np.clip(prices[:-1], 1e-8, None)
            volatility = float(np.std(returns))
            peak = np.maximum.accumulate(prices)
            drawdown = (peak - prices) / np.clip(peak, 1e-8, None)
            crash_loss_rate = float(np.max(drawdown))
        else:
            volatility = 0.0
            crash_loss_rate = 0.0

        panic_level = float(getattr(market, "panic_level", 0.0))
        macro_stability = float(np.clip(1.0 - (12.0 * volatility + 0.6 * crash_loss_rate + 0.35 * panic_level), 0.0, 1.0))
        liquidity_shortage = float(np.clip(0.60 * panic_level + 0.35 * volatility - 0.25 * max(0.0, self._last_action.stabilization_capital), 0.0, 1.0))
        intervention_cost = float(
            np.clip(
                abs(self._last_action.stamp_tax_rate - 0.0005) * 120.0
                + abs(self._last_action.reserve_cut_bps) / 120.0
                + abs(self._last_action.policy_rate_cut_bps) / 80.0
                + max(0.0, self._last_action.stabilization_capital),
                0.0,
                2.5,
            )
        )
        welfare_confidence = float(np.clip(1.0 - 0.55 * panic_level - 0.25 * volatility + 0.20 * max(0.0, self._last_action.stabilization_capital), 0.0, 1.0))
        support_metrics = _derive_policy_support_metrics(
            volatility=volatility,
            crash_loss_rate=crash_loss_rate,
            panic_level=panic_level,
            stabilization_capital=float(self._last_action.stabilization_capital),
            rumor_refute_strength=float(self._last_action.rumor_refute_strength),
        )

        return EnvObservation(
            step=step,
            macro_stability=macro_stability,
            crash_loss_rate=float(np.clip(crash_loss_rate, 0.0, 1.0)),
            volatility=float(np.clip(volatility, 0.0, 1.0)),
            panic_level=float(np.clip(panic_level, 0.0, 1.0)),
            agent_count=int(len(getattr(model, "agents", []))) if model is not None else 0,
            extras={
                "price_len": len(price_history),
                "controller_mode": str(getattr(self.controller, "mode", "UNKNOWN")),
                "liquidity_shortage": liquidity_shortage,
                "intervention_cost": intervention_cost,
                "welfare_confidence": welfare_confidence,
                **support_metrics,
            },
        )


# ==========================================================
# 5) Regulator RL 智能体（Q-Learning 基础版）
# ==========================================================


@dataclass
class TrainingSummary:
    episodes: int
    avg_episode_reward: float
    best_action: RegulatoryAction
    best_action_score: float
    top_actions: List[Tuple[RegulatoryAction, float]]
    q_states: int


class RegulatorAgent:
    """
    监管强化学习智能体（基础实现）。

    算法：
    - 离散动作空间 + 状态分桶 + Q-Learning
    - epsilon-greedy 探索
    """

    def __init__(
        self,
        action_space: Optional[List[RegulatoryAction]] = None,
        alpha: float = 0.15,
        gamma: float = 0.96,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.03,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = action_space or self.build_default_action_space()
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = defaultdict(self._q_init)
        self.action_reward_sum = np.zeros(len(self.action_space), dtype=float)
        self.action_count = np.zeros(len(self.action_space), dtype=float)

    def _q_init(self) -> np.ndarray:
        return np.zeros(len(self.action_space), dtype=float)

    @staticmethod
    def build_default_action_space() -> List[RegulatoryAction]:
        """
        构建组合调控动作空间。
        """
        stamp_tax_candidates = [0.0003, 0.0005, 0.0008, 0.0010]
        reserve_cut_candidates = [0, 25, 50]
        rate_cut_candidates = [0, 10, 25]
        rumor_refute_candidates = [0.0, 0.4, 0.7, 1.0]
        stabilization_capital_candidates = [0.0, 0.3, 0.6]
        stabilization_timing_candidates = [0, 3, 6]
        halt_candidates = [False, True]

        actions: List[RegulatoryAction] = []
        for tax in stamp_tax_candidates:
            for rrr in reserve_cut_candidates:
                for rate in rate_cut_candidates:
                    for rumor in rumor_refute_candidates:
                        for capital in stabilization_capital_candidates:
                            for timing in stabilization_timing_candidates:
                                for halt in halt_candidates:
                                    actions.append(
                                        RegulatoryAction(
                                            stamp_tax_rate=float(tax),
                                            reserve_cut_bps=int(rrr),
                                            policy_rate_cut_bps=int(rate),
                                            rumor_refute_strength=float(rumor),
                                            stabilization_capital=float(capital),
                                            stabilization_timing=int(timing),
                                            halt_enabled=bool(halt),
                                        )
                                    )
        return actions

    @staticmethod
    def build_compact_action_space() -> List[RegulatoryAction]:
        """Smaller action set for quick A/B and Pareto evaluation."""
        return [
            RegulatoryAction(0.0005, 0, 0, 0.0, 0.0, 0, False),
            RegulatoryAction(0.0003, 25, 10, 0.7, 0.3, 2, False),
            RegulatoryAction(0.0003, 50, 25, 1.0, 0.6, 1, False),
            RegulatoryAction(0.0010, 0, 0, 0.2, 0.0, 0, True),
            RegulatoryAction(0.0008, 0, 0, 0.4, 0.2, 4, False),
        ]

    @staticmethod
    def discretize_state(obs: EnvObservation) -> Tuple[int, int, int, int]:
        """
        连续状态离散化。

        说明：基础版用分桶降低状态维度，便于稳定训练和可解释输出。
        """
        vol_bucket = int(np.digitize(obs.volatility, bins=[0.01, 0.02, 0.04, 0.08, 0.15]))
        crash_bucket = int(np.digitize(obs.crash_loss_rate, bins=[0.02, 0.05, 0.10, 0.20, 0.35]))
        stability_bucket = int(np.digitize(obs.macro_stability, bins=[0.2, 0.4, 0.6, 0.8]))
        panic_bucket = int(np.digitize(obs.panic_level, bins=[0.1, 0.3, 0.5, 0.7, 0.9]))
        return vol_bucket, crash_bucket, stability_bucket, panic_bucket

    def _select_action_index(self, state: Tuple[int, int, int, int]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.action_space))
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def _update_q(
        self,
        state: Tuple[int, int, int, int],
        action_idx: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
    ) -> None:
        q = self.q_table[state][action_idx]
        q_next_max = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * q_next_max
        self.q_table[state][action_idx] = q + self.alpha * (td_target - q)

    def train(
        self,
        env: RegulatoryEnvironment,
        episodes: int = 500,
        max_steps_per_episode: int = 96,
    ) -> TrainingSummary:
        episode_rewards: List[float] = []

        for ep in range(episodes):
            obs = env.reset(seed=ep)
            state = self.discretize_state(obs)
            total_reward = 0.0

            for _ in range(max_steps_per_episode):
                action_idx = self._select_action_index(state)
                action = self.action_space[action_idx]

                next_obs, reward, done, _info = env.step(action)
                next_state = self.discretize_state(next_obs)
                self._update_q(state, action_idx, reward, next_state)

                self.action_reward_sum[action_idx] += reward
                self.action_count[action_idx] += 1.0
                total_reward += reward
                state = next_state
                if done:
                    break

            episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        mean_scores = self.action_reward_sum / np.clip(self.action_count, 1.0, None)
        best_idx = int(np.argmax(mean_scores))
        ranked_idx = np.argsort(mean_scores)[::-1][:5]
        top_actions = [(self.action_space[int(i)], float(mean_scores[int(i)])) for i in ranked_idx]

        return TrainingSummary(
            episodes=episodes,
            avg_episode_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            best_action=self.action_space[best_idx],
            best_action_score=float(mean_scores[best_idx]),
            top_actions=top_actions,
            q_states=len(self.q_table),
        )


# ==========================================================
# 6) 一键运行入口
# ==========================================================


def run_regulatory_optimization(
    episodes: int = 1200,
    max_steps_per_episode: int = 72,
    use_toy_env: bool = True,
    env: Optional[RegulatoryEnvironment] = None,
) -> TrainingSummary:
    """
    一键训练入口。

    默认使用 Toy 环境做框架验证；接入真实沙盘时传入 env 参数即可。
    """
    if env is None:
        if not use_toy_env:
            raise ValueError("use_toy_env=False 时必须显式传入真实环境实例 env。")
        env = ToyRegulatoryMarketEnv()

    agent = RegulatorAgent()
    return agent.train(env=env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)


def build_simulation_controller_factory(
    deepseek_key: str,
    zhipu_key: Optional[str] = None,
    mode: str = "FAST",
    quant_manager: Optional[Any] = None,
    regulatory_module: Optional[Any] = None,
) -> Callable[[], Any]:
    """
    构建 SimulationController 工厂。

    说明：
    - 延迟导入，避免在未安装完整依赖或仅跑单测时触发重型初始化；
    - 供 SimulationControllerEnvAdapter 使用。
    """

    def _factory() -> Any:
        from core.scheduler import SimulationController

        return SimulationController(
            deepseek_key=deepseek_key,
            zhipu_key=zhipu_key,
            mode=mode,
            quant_manager=quant_manager,
            regulatory_module=regulatory_module,
        )

    return _factory


def run_controller_regulatory_optimization(
    controller_factory: Callable[[], Any],
    episodes: int = 120,
    max_steps_per_episode: int = 24,
) -> TrainingSummary:
    """
    在真实 SimulationController 环境上训练监管智能体。
    """
    env = SimulationControllerEnvAdapter(
        controller_factory=controller_factory,
        episode_steps=max_steps_per_episode,
    )
    agent = RegulatorAgent()
    return agent.train(env=env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)


def run_real_regulatory_optimization(
    *,
    env_factory: Callable[[], RegulatoryEnvironment],
    episodes: int = 300,
    max_steps_per_episode: int = 48,
    seed: int = 42,
) -> TrainingSummary:
    """Real-simulation-first optimization entrypoint (toy env remains fallback elsewhere)."""
    agent = RegulatorAgent(action_space=RegulatorAgent.build_compact_action_space(), seed=int(seed))
    env = env_factory()
    return agent.train(
        env=env,
        episodes=int(episodes),
        max_steps_per_episode=int(max_steps_per_episode),
    )


def training_summary_to_dict(summary: TrainingSummary) -> Dict[str, Any]:
    """
    将训练结果转换为可 JSON 序列化结构。
    """
    return {
        "episodes": int(summary.episodes),
        "avg_episode_reward": float(summary.avg_episode_reward),
        "best_action_score": float(summary.best_action_score),
        "best_action": summary.best_action.to_dict(),
        "best_action_description": summary.best_action.describe(),
        "top_actions": [
            {
                "score": float(score),
                "action": action.to_dict(),
                "description": action.describe(),
            }
            for action, score in summary.top_actions
        ],
        "q_states": int(summary.q_states),
    }


def _action_signature(action: RegulatoryAction) -> str:
    return hashlib.sha256(json.dumps(action.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def build_default_real_env_factory(
    *,
    max_steps_per_episode: int,
    reward_fn: Optional[RewardFunction] = None,
) -> Callable[[], RegulatoryEnvironment]:
    """
    Default real-environment entrypoint.

    This keeps the interface stable while making the real simulation the preferred path.
    If the controller cannot be constructed in the local runtime, the caller decides whether
    to fall back to the toy environment via feature flags.
    """

    def _factory() -> RegulatoryEnvironment:
        controller_factory = build_simulation_controller_factory(
            deepseek_key=str(getattr(GLOBAL_CONFIG, "DEEPSEEK_API_KEY", "") or ""),
            zhipu_key=str(getattr(GLOBAL_CONFIG, "ZHIPU_API_KEY", "") or "") or None,
            mode="FAST",
        )
        return SimulationControllerEnvAdapter(
            controller_factory=controller_factory,
            episode_steps=int(max_steps_per_episode),
            reward_fn=reward_fn,
        )

    return _factory


def _toy_env_factory(*, max_steps_per_episode: int, reward_fn: Optional[RewardFunction] = None) -> Callable[[], RegulatoryEnvironment]:
    return lambda: ToyRegulatoryMarketEnv(episode_steps=int(max_steps_per_episode), reward_fn=reward_fn)


def _evaluate_action_bundle(
    *,
    env_factory: Callable[[], RegulatoryEnvironment],
    action: RegulatoryAction,
    max_steps_per_episode: int,
    seed: int,
) -> Dict[str, Any]:
    env = env_factory()
    obs = env.reset(seed=seed)
    total_reward = 0.0
    metrics_trace: List[Dict[str, float]] = []
    reward_breakdowns: List[Dict[str, float]] = []
    done = False
    for _ in range(max_steps_per_episode):
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        liquidity_shortage = float(obs.extras.get("liquidity_shortage", 0.0)) if obs.extras else 0.0
        intervention_cost = float(obs.extras.get("intervention_cost", 0.0)) if obs.extras else 0.0
        welfare_confidence = float(obs.extras.get("welfare_confidence", 1.0 - obs.panic_level)) if obs.extras else float(1.0 - obs.panic_level)
        financing_function = float(obs.extras.get("financing_function", 1.0 - 0.50 * obs.volatility - 0.35 * obs.panic_level)) if obs.extras else float(1.0 - 0.50 * obs.volatility - 0.35 * obs.panic_level)
        fairness_compliance = float(obs.extras.get("fairness_compliance", 1.0 - 0.35 * obs.panic_level - 0.45 * obs.crash_loss_rate)) if obs.extras else float(1.0 - 0.35 * obs.panic_level - 0.45 * obs.crash_loss_rate)
        metrics_trace.append(
            {
                "macro_stability": float(obs.macro_stability),
                "crash_risk": float(obs.crash_loss_rate),
                "volatility": float(obs.volatility),
                "liquidity_shortage": liquidity_shortage,
                "liquidity": float(np.clip(1.0 - liquidity_shortage, 0.0, 1.0)),
                "intervention_cost": intervention_cost,
                "welfare_confidence": welfare_confidence,
                "financing_function": float(np.clip(financing_function, 0.0, 1.0)),
                "fairness_compliance": float(np.clip(fairness_compliance, 0.0, 1.0)),
            }
        )
        reward_data = info.get("reward_breakdown") if isinstance(info, dict) else None
        if isinstance(reward_data, dict):
            reward_breakdowns.append(
                {
                    key: float(value)
                    for key, value in dict(reward_data.get("weighted_components", {})).items()
                    if isinstance(value, (int, float))
                }
            )
        if done:
            break
    frame = np.array(
        [
            [
                row["macro_stability"],
                row["crash_risk"],
                row["volatility"],
                row["liquidity"],
                row["intervention_cost"],
                row["welfare_confidence"],
                row["financing_function"],
                row["fairness_compliance"],
            ]
            for row in metrics_trace
        ],
        dtype=float,
    )
    if frame.size == 0:
        frame = np.zeros((1, 8), dtype=float)
    reward_component_means: Dict[str, float] = {}
    if reward_breakdowns:
        keys = sorted({key for row in reward_breakdowns for key in row.keys()})
        reward_component_means = {
            key: float(np.mean([row.get(key, 0.0) for row in reward_breakdowns]))
            for key in keys
        }
    return {
        "action": action.to_dict(),
        "action_description": action.describe(),
        "action_signature": _action_signature(action),
        "avg_reward": float(total_reward / max(1, len(metrics_trace))),
        "episode_reward": float(total_reward),
        "macro_stability": float(np.mean(frame[:, 0])),
        "crash_risk": float(np.mean(frame[:, 1])),
        "volatility": float(np.mean(frame[:, 2])),
        "liquidity": float(np.mean(frame[:, 3])),
        "intervention_cost": float(np.mean(frame[:, 4])),
        "welfare_confidence": float(np.mean(frame[:, 5])),
        "financing_function": float(np.mean(frame[:, 6])),
        "fairness_compliance": float(np.mean(frame[:, 7])),
        "reward_breakdown": reward_component_means,
        "steps": int(len(metrics_trace)),
        "terminated": bool(done),
    }


def _blackbox_parameter_space() -> Dict[str, Tuple[float, float]]:
    return {
        "stamp_tax_rate": (0.0003, 0.0010),
        "reserve_cut_bps": (0.0, 50.0),
        "policy_rate_cut_bps": (0.0, 25.0),
        "rumor_refute_strength": (0.0, 1.0),
        "stabilization_capital": (0.0, 0.8),
        "stabilization_timing": (0.0, 6.0),
    }


def _params_to_action(params: Mapping[str, Any]) -> RegulatoryAction:
    return RegulatoryAction(
        stamp_tax_rate=float(params.get("stamp_tax_rate", 0.0005)),
        reserve_cut_bps=int(round(float(params.get("reserve_cut_bps", 0.0)))),
        policy_rate_cut_bps=int(round(float(params.get("policy_rate_cut_bps", 0.0)))),
        rumor_refute_strength=float(np.clip(float(params.get("rumor_refute_strength", 0.0)), 0.0, 1.0)),
        stabilization_capital=float(np.clip(float(params.get("stabilization_capital", 0.0)), 0.0, 1.0)),
        stabilization_timing=int(round(float(params.get("stabilization_timing", 0.0)))),
        halt_enabled=bool(params.get("halt_enabled", False)),
    )


def _action_to_params(action: RegulatoryAction) -> Dict[str, float]:
    return {
        "stamp_tax_rate": float(action.stamp_tax_rate),
        "reserve_cut_bps": float(action.reserve_cut_bps),
        "policy_rate_cut_bps": float(action.policy_rate_cut_bps),
        "rumor_refute_strength": float(action.rumor_refute_strength),
        "stabilization_capital": float(action.stabilization_capital),
        "stabilization_timing": float(action.stabilization_timing),
    }


def _rule_baseline_action() -> RegulatoryAction:
    return RegulatoryAction(
        stamp_tax_rate=0.0005,
        reserve_cut_bps=25,
        policy_rate_cut_bps=10,
        rumor_refute_strength=0.65,
        stabilization_capital=0.25,
        stabilization_timing=1,
        halt_enabled=False,
    )


def _static_policy_simulator(
    *,
    env_factory: Callable[[], RegulatoryEnvironment],
    max_steps_per_episode: int,
    seed: int,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _simulate(params: Dict[str, Any]) -> Dict[str, Any]:
        row = _evaluate_action_bundle(
            env_factory=env_factory,
            action=_params_to_action(params),
            max_steps_per_episode=int(max_steps_per_episode),
            seed=int(seed),
        )
        row["max_drawdown"] = float(row.get("crash_risk", 0.0))
        row["panic"] = float(np.clip(1.0 - float(row.get("welfare_confidence", 0.0)), 0.0, 1.0))
        row["market_function_score"] = float(row.get("financing_function", row.get("liquidity", 0.0)))
        row["growth_proxy_score"] = float(np.clip(0.45 + 0.20 * row.get("avg_reward", 0.0) + 0.20 * row.get("liquidity", 0.0), 0.0, 1.0))
        row["financial_stability_score"] = float(
            np.clip(
                0.55 * row.get("macro_stability", 0.0)
                + 0.25 * (1.0 - row.get("crash_risk", 0.0))
                + 0.20 * (1.0 - row.get("volatility", 0.0)),
                0.0,
                1.0,
            )
        )
        row["confidence_score"] = float(row.get("welfare_confidence", 0.0))
        row["fairness_compliance_score"] = float(row.get("fairness_compliance", 0.0))
        row["financing_function_score"] = float(row.get("financing_function", 0.0))
        row["tracking_rmse"] = float(max(0.0, row.get("volatility", 0.0) * 0.10 + row.get("crash_risk", 0.0) * 0.05))
        return row

    return _simulate


def _validate_blackbox_against_rule(
    *,
    env_factory: Callable[[], RegulatoryEnvironment],
    best_params: Mapping[str, Any],
    rule_action: RegulatoryAction,
    max_steps_per_episode: int,
    seed: int,
) -> Dict[str, Any]:
    windows: List[Dict[str, Any]] = []
    stable_wins = 0
    for idx, window_seed in enumerate((int(seed), int(seed) + 101), start=1):
        bo_row = _evaluate_action_bundle(
            env_factory=env_factory,
            action=_params_to_action(best_params),
            max_steps_per_episode=int(max_steps_per_episode),
            seed=window_seed,
        )
        rule_row = _evaluate_action_bundle(
            env_factory=env_factory,
            action=rule_action,
            max_steps_per_episode=int(max_steps_per_episode),
            seed=window_seed,
        )
        reward_win = float(bo_row.get("avg_reward", 0.0)) >= float(rule_row.get("avg_reward", 0.0))
        stability_win = float(bo_row.get("macro_stability", 0.0)) >= float(rule_row.get("macro_stability", 0.0))
        if reward_win and stability_win:
            stable_wins += 1
        windows.append(
            {
                "window_id": f"fixed_replay_window_{idx}",
                "seed": int(window_seed),
                "bo_avg_reward": float(bo_row.get("avg_reward", 0.0)),
                "rule_avg_reward": float(rule_row.get("avg_reward", 0.0)),
                "bo_macro_stability": float(bo_row.get("macro_stability", 0.0)),
                "rule_macro_stability": float(rule_row.get("macro_stability", 0.0)),
                "reward_win": bool(reward_win),
                "stability_win": bool(stability_win),
            }
        )
    return {
        "windows": windows,
        "stable_win_count": int(stable_wins),
        "required_windows": 2,
        "promote_blackbox_default": bool(stable_wins >= 2),
    }


def _run_blackbox_policy_optimization(
    *,
    env_factory: Callable[[], RegulatoryEnvironment],
    q_summary: Mapping[str, Any],
    max_steps_per_episode: int,
    seed: int,
    episodes: int,
) -> Dict[str, Any]:
    parameter_space = _blackbox_parameter_space()
    simulator = _static_policy_simulator(
        env_factory=env_factory,
        max_steps_per_episode=int(max_steps_per_episode),
        seed=int(seed),
    )
    initial_params = [_action_to_params(_rule_baseline_action())]
    for item in list(q_summary.get("top_actions", []) or [])[:3]:
        action_payload = item.get("action") if isinstance(item, Mapping) else None
        if isinstance(action_payload, Mapping):
            initial_params.append(dict(action_payload))
    n_iter = max(8, min(24, int(episodes) // 3 if int(episodes) > 0 else 8))
    bo = bayesian_search(
        simulator=simulator,
        parameter_space=parameter_space,
        n_iter=n_iter,
        seed=int(seed),
        initial_params=initial_params,
    )
    nsga = nsga_search(
        simulator=simulator,
        parameter_space=parameter_space,
        population_size=max(8, min(14, n_iter)),
        generations=2,
        seed=int(seed) + 17,
    )
    rule_action = _rule_baseline_action()
    rule_row = _evaluate_action_bundle(
        env_factory=env_factory,
        action=rule_action,
        max_steps_per_episode=int(max_steps_per_episode),
        seed=int(seed),
    )
    best_params = dict(bo.best.get("params", {})) if bo.best else {}
    validation = _validate_blackbox_against_rule(
        env_factory=env_factory,
        best_params=best_params,
        rule_action=rule_action,
        max_steps_per_episode=int(max_steps_per_episode),
        seed=int(seed),
    )
    sensitivity_rows = sensitivity_analysis(
        simulator=simulator,
        best_params=best_params,
        parameter_space=parameter_space,
        step_fraction=0.08,
    )[:12]
    report = generate_optimization_report(
        input_scenario={
            "kind": "static_policy_package",
            "replay_window": "fixed_seed_windows",
            "method_preference": "BO default, NSGA-II Pareto, Q-learning baseline retained",
        },
        data_snapshot={
            "seed": int(seed),
            "max_steps_per_episode": int(max_steps_per_episode),
            "env_factory": getattr(env_factory, "__name__", "callable_env_factory"),
        },
        parameter_space=parameter_space,
        q_learning_result=q_summary,
        bayesian_result=bo.to_dict(),
        nsga_result=nsga.to_dict(),
        rule_baseline=rule_row,
        counterfactual={
            "rule_baseline": rule_row,
            "bo_best": bo.best,
            "q_learning_best": q_summary.get("best_action", {}),
        },
        validation=validation,
        sensitivity=sensitivity_rows,
    )
    return {
        "bayesian_optimization": bo.to_dict(),
        "nsga_ii": nsga.to_dict(),
        "rule_baseline": rule_row,
        "validation": validation,
        "sensitivity_analysis": sensitivity_rows,
        "optimization_report": report,
        "production_path": "bayesian_blackbox" if validation["promote_blackbox_default"] else "q_learning_baseline",
    }


def _pareto_frontier(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frontier: List[Dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = (
                float(other.get("macro_stability", 0.0)) >= float(row.get("macro_stability", 0.0))
                and float(other.get("liquidity", 0.0)) >= float(row.get("liquidity", 0.0))
                and float(other.get("intervention_cost", 0.0)) <= float(row.get("intervention_cost", 0.0))
            )
            strictly_better = (
                float(other.get("macro_stability", 0.0)) > float(row.get("macro_stability", 0.0))
                or float(other.get("liquidity", 0.0)) > float(row.get("liquidity", 0.0))
                or float(other.get("intervention_cost", 0.0)) < float(row.get("intervention_cost", 0.0))
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda x: (float(x.get("intervention_cost", 0.0)), -float(x.get("macro_stability", 0.0)), -float(x.get("liquidity", 0.0))))
    return frontier


def _recommend_policy_bundle(
    *,
    baseline_row: Mapping[str, Any],
    candidates: List[Dict[str, Any]],
    pareto: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pool = pareto if pareto else candidates
    if not pool:
        return {
            "recommended_action": {},
            "action_signature": "",
            "action_description": "No candidate available",
            "scorecard": {},
            "evidence_chain": [],
            "tradeoff_summary": "No candidate available",
            "side_effects": [],
        }

    def _score(row: Mapping[str, Any]) -> float:
        return float(
            0.28 * float(row.get("macro_stability", 0.0))
            + 0.20 * float(row.get("liquidity", 0.0))
            + 0.17 * float(row.get("financing_function", 0.0))
            + 0.17 * float(row.get("fairness_compliance", 0.0))
            + 0.10 * float(row.get("welfare_confidence", 0.0))
            - 0.08 * float(row.get("intervention_cost", 0.0))
        )

    recommended = max(pool, key=_score)
    scorecard = {
        "macro_stability": float(recommended.get("macro_stability", 0.0)),
        "liquidity": float(recommended.get("liquidity", 0.0)),
        "financing_function": float(recommended.get("financing_function", 0.0)),
        "fairness_compliance": float(recommended.get("fairness_compliance", 0.0)),
        "welfare_confidence": float(recommended.get("welfare_confidence", 0.0)),
        "intervention_cost": float(recommended.get("intervention_cost", 0.0)),
        "composite_score": _score(recommended),
    }
    evidence_metrics = [
        ("macro_stability", "Stability improved"),
        ("liquidity", "Liquidity supply strengthened"),
        ("financing_function", "Financing function recovered"),
        ("fairness_compliance", "Fairness/compliance improved"),
        ("intervention_cost", "Intervention cost changed"),
    ]
    evidence_chain: List[Dict[str, Any]] = []
    for metric, interpretation in evidence_metrics:
        baseline_value = float(baseline_row.get(metric, 0.0))
        candidate_value = float(recommended.get(metric, 0.0))
        evidence_chain.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": float(candidate_value - baseline_value),
                "interpretation": interpretation,
            }
        )
    side_effects = [
        item
        for item in (
            "Higher intervention cost" if float(recommended.get("intervention_cost", 0.0)) > float(baseline_row.get("intervention_cost", 0.0)) else "",
            "Volatility remains elevated" if float(recommended.get("volatility", 0.0)) > 0.08 else "",
            "Crash risk still non-trivial" if float(recommended.get("crash_risk", 0.0)) > 0.10 else "",
        )
        if item
    ]
    return {
        "recommended_action": dict(recommended.get("action", {})),
        "action_signature": str(recommended.get("action_signature", "")),
        "action_description": str(recommended.get("action_description", "")),
        "scorecard": scorecard,
        "evidence_chain": evidence_chain,
        "tradeoff_summary": (
            f"Selected action balances stability={scorecard['macro_stability']:.3f}, "
            f"liquidity={scorecard['liquidity']:.3f}, financing={scorecard['financing_function']:.3f}, "
            f"fairness={scorecard['fairness_compliance']:.3f} against cost={scorecard['intervention_cost']:.3f}."
        ),
        "side_effects": side_effects,
    }


def _objective_discovery_for_regulator(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    market_rows: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    close = 3000.0
    for idx, row in enumerate(rows, start=1):
        stability = float(row.get("macro_stability", 0.0) or 0.0)
        volatility = float(row.get("volatility", 0.0) or 0.0)
        panic = float(np.clip(1.0 - float(row.get("welfare_confidence", 1.0 - volatility) or 0.0), 0.0, 1.0))
        close *= 1.0 + 0.002 * stability - 0.0015 * float(row.get("crash_risk", 0.0) or 0.0)
        market_rows.append(
            {
                "step": idx,
                "close": close,
                "open": close / max(1.0 + 0.001 * stability, 1e-9),
                "high": close * (1.0 + max(volatility, 0.002)),
                "low": close * (1.0 - max(volatility, 0.002)),
                "volume": 1_000_000.0 * (1.0 + panic),
                "panic_level": panic,
                "csad": float(np.clip(0.04 + 0.12 * panic + volatility, 0.0, 1.0)),
                "买入量": 500_000.0 * (1.0 + stability),
                "卖出量": 500_000.0 * (1.0 + panic),
            }
        )
        reports.append(
            {
                "macro_state": {
                    "liquidity_index": float(row.get("liquidity", 0.0) or 0.0),
                    "credit_spread": float(row.get("crash_risk", 0.0) or 0.0) * 0.01,
                    "unemployment": max(0.0, 0.08 - 0.03 * stability),
                    "wage_growth": max(0.0, 0.02 * stability),
                    "inflation": 0.02 + 0.01 * volatility,
                },
                "microstructure_metrics": {
                    "spread_pct": max(0.0, volatility * 0.20),
                    "depth_bid_total": 800_000.0 * float(row.get("liquidity", 0.0) or 0.0),
                    "depth_ask_total": 750_000.0 * float(row.get("liquidity", 0.0) or 0.0),
                    "cancel_to_trade_ratio": max(0.0, panic * 0.35),
                    "slippage_bps": max(0.0, volatility * 10_000.0),
                },
                "realism_diagnostics": {
                    "microstructure_score": float(row.get("liquidity", 0.0) or 0.0),
                    "liquidity_thinness": float(np.clip(1.0 - float(row.get("liquidity", 0.0) or 0.0), 0.0, 1.0)),
                },
            }
        )
    return ObjectiveDiscoveryEngine().discover(market_rows, reports=reports, top_k=8).to_dict()


def _objective_contribution(row: Mapping[str, Any], discovered: Mapping[str, Any]) -> Dict[str, Any]:
    weights = dict(discovered.get("weight_decomposition", {}) or {})
    proxy_values = {
        "liquidity_index": float(row.get("liquidity", 0.0) or 0.0),
        "financing_function": float(row.get("financing_function", 0.0) or 0.0),
        "fairness_compliance": float(row.get("fairness_compliance", 0.0) or 0.0),
        "welfare_confidence": float(row.get("welfare_confidence", 0.0) or 0.0),
        "intervention_cost": float(row.get("intervention_cost", 0.0) or 0.0),
        "realized_volatility": float(row.get("volatility", 0.0) or 0.0),
        "max_drawdown": float(row.get("crash_risk", 0.0) or 0.0),
        "panic_level": 1.0 - float(row.get("welfare_confidence", 0.0) or 0.0),
    }
    contributions = {
        name: float(weight) * float(proxy_values.get(name, row.get(name, 0.0) or 0.0))
        for name, weight in weights.items()
    }
    return {
        "weighted_sum": float(sum(contributions.values())),
        "top_contributions": contributions,
    }


def run_regulatory_closed_loop(
    *,
    episodes: int = 300,
    max_steps_per_episode: int = 48,
    seed: int = 42,
    top_k: int = 5,
    use_toy_env: Optional[bool] = None,
    env: Optional[RegulatoryEnvironment] = None,
    env_factory: Optional[Callable[[], RegulatoryEnvironment]] = None,
    feature_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run regulator optimization in a simulation loop and output:
    - training summary
    - counterfactual A/B bundles
    - Pareto frontier (stability vs liquidity vs cost)
    """
    flags = resolve_regulator_feature_flags(feature_flags)
    reward_fn = RewardFunction()
    toy_factory = _toy_env_factory(max_steps_per_episode=int(max_steps_per_episode), reward_fn=reward_fn)
    env_selection: Dict[str, Any] = {
        "requested_use_toy_env": use_toy_env,
        "feature_flags": flags,
        "selected_path": "",
        "fallback_used": False,
    }
    if env_factory is not None:
        active_env_factory = env_factory
        env_selection["selected_path"] = "custom_env_factory"
    elif env is not None:
        active_env_factory = lambda: env
        env_selection["selected_path"] = "custom_env"
    elif use_toy_env is True:
        active_env_factory = toy_factory
        env_selection["selected_path"] = "toy_explicit"
    elif flags.get("regulator_real_env_v1", True):
        if _has_configured_live_model():
            active_env_factory = build_default_real_env_factory(
                max_steps_per_episode=int(max_steps_per_episode),
                reward_fn=reward_fn,
            )
            env_selection["selected_path"] = "real_env_factory"
        elif flags.get("regulator_toy_fallback_v1", True):
            active_env_factory = toy_factory
            env_selection["selected_path"] = "toy_fallback"
            env_selection["fallback_used"] = True
            env_selection["fallback_reason"] = "missing_live_model_keys"
        else:
            raise ValueError("Real regulatory environment requested but no live model keys are configured.")
    elif flags.get("regulator_toy_fallback_v1", True):
        active_env_factory = toy_factory
        env_selection["selected_path"] = "toy_fallback"
        env_selection["fallback_used"] = True
        env_selection["fallback_reason"] = "real_env_feature_disabled"
    else:
        raise ValueError("No regulatory environment available under current feature flags.")

    agent = RegulatorAgent(action_space=RegulatorAgent.build_compact_action_space(), seed=int(seed))
    try:
        summary = agent.train(
            env=active_env_factory(),
            episodes=int(episodes),
            max_steps_per_episode=int(max_steps_per_episode),
        )
    except Exception as exc:
        if env_selection["selected_path"] == "real_env_factory" and flags.get("regulator_toy_fallback_v1", True):
            active_env_factory = toy_factory
            env_selection["selected_path"] = "toy_fallback"
            env_selection["fallback_used"] = True
            env_selection["fallback_reason"] = str(exc)
            summary = agent.train(
                env=active_env_factory(),
                episodes=int(episodes),
                max_steps_per_episode=int(max_steps_per_episode),
            )
        else:
            raise
    summary_dict = training_summary_to_dict(summary)

    baseline = RegulatoryAction(0.0005, 0, 0, 0.0, 0.0, 0, False)
    evaluated: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, action in enumerate([baseline, *[x[0] for x in summary.top_actions[: max(1, int(top_k))]]]):
        sig = _action_signature(action)
        if sig in seen:
            continue
        seen.add(sig)
        evaluated.append(
            _evaluate_action_bundle(
                env_factory=active_env_factory,
                action=action,
                max_steps_per_episode=int(max_steps_per_episode),
                seed=int(seed + idx),
            )
        )

    baseline_row = evaluated[0] if evaluated else {}
    candidates = evaluated[1:] if len(evaluated) > 1 else []
    ab_rows: List[Dict[str, Any]] = []
    for row in candidates:
        ab_rows.append(
            {
                "action": row["action"],
                "action_description": row["action_description"],
                "macro_stability_delta": float(row["macro_stability"] - float(baseline_row.get("macro_stability", 0.0))),
                "liquidity_delta": float(row["liquidity"] - float(baseline_row.get("liquidity", 0.0))),
                "financing_function_delta": float(row["financing_function"] - float(baseline_row.get("financing_function", 0.0))),
                "fairness_compliance_delta": float(row["fairness_compliance"] - float(baseline_row.get("fairness_compliance", 0.0))),
                "cost_delta": float(row["intervention_cost"] - float(baseline_row.get("intervention_cost", 0.0))),
                "reward_delta": float(row["avg_reward"] - float(baseline_row.get("avg_reward", 0.0))),
            }
        )

    pareto = _pareto_frontier(candidates if candidates else evaluated)
    discovered_objectives = _objective_discovery_for_regulator(evaluated)
    for row in evaluated:
        row["objective_contribution"] = _objective_contribution(row, discovered_objectives)
    recommendation = _recommend_policy_bundle(
        baseline_row=baseline_row,
        candidates=candidates,
        pareto=pareto,
    )
    recommended_sig = str(recommendation.get("action_signature", ""))
    recommended_row = next((row for row in evaluated if str(row.get("action_signature", "")) == recommended_sig), {})
    recommendation["discovered_objective_contribution"] = _objective_contribution(recommended_row, discovered_objectives)
    recommendation["discovered_objectives"] = discovered_objectives
    reproducibility = {
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "env_selection": env_selection,
        "config_hash": hashlib.sha256(
            json.dumps(
                {
                    "seed": int(seed),
                    "episodes": int(episodes),
                    "max_steps_per_episode": int(max_steps_per_episode),
                    "top_k": int(top_k),
                    "feature_flags": flags,
                    "env_selection": env_selection.get("selected_path", ""),
                    "action_space": [a.to_dict() for a in agent.action_space],
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }
    blackbox_result = _run_blackbox_policy_optimization(
        env_factory=active_env_factory,
        q_summary=summary_dict,
        max_steps_per_episode=int(max_steps_per_episode),
        seed=int(seed),
        episodes=int(episodes),
    )
    return {
        "training_summary": summary_dict,
        "counterfactual_ab": {
            "baseline": baseline_row,
            "candidates": candidates,
            "deltas": ab_rows,
        },
        "pareto_frontier": pareto,
        "multi_objective_pareto_frontier": blackbox_result["nsga_ii"].get("pareto_frontier", []),
        "blackbox_optimization": blackbox_result,
        "optimization_report": blackbox_result["optimization_report"],
        "default_production_path": blackbox_result["production_path"],
        "recommendation": recommendation,
        "discovered_objectives": discovered_objectives,
        "reproducibility": reproducibility,
    }


if __name__ == "__main__":
    summary = run_regulatory_optimization(episodes=600, max_steps_per_episode=64, use_toy_env=True)
    print("=== Regulator Agent Training Summary ===")
    print(f"Episodes: {summary.episodes}")
    print(f"Avg Episode Reward: {summary.avg_episode_reward:.4f}")
    print(f"Q States: {summary.q_states}")
    print(f"Best Action Score: {summary.best_action_score:.4f}")
    print(f"Best Action: {summary.best_action.describe()}")
    print("Top Action Bundles:")
    for idx, (action, score) in enumerate(summary.top_actions, start=1):
        print(f"{idx}. score={score:.4f} -> {action.describe()}")
