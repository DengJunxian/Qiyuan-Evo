"""Policy experiment session API for Streamlit-friendly stepwise simulation."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from core.event_store import EventStore
from core.experiment_events import EventDigest, ExperimentEvent, ScenarioEventQueue
from engine.simulation_loop import MarketEnvironment


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _run_sync(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("当前存在运行中的事件循环，请改用 advance_async()")
    return asyncio.run(awaitable)


def _coerce_date(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(timestamp).normalize()


@dataclass(slots=True)
class PolicyPlan:
    """One policy item inside a session timeline."""

    policy_id: str
    policy_text: str
    effective_day: int
    strength: float = 1.0
    half_life_days: float = 14.0
    rumor_noise: bool = False
    label: str = ""
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_day: int = 0

    def intensity_for_day(self, day_index: int) -> float:
        if day_index < self.effective_day:
            return 0.0
        age = max(0, int(day_index) - int(self.effective_day))
        half_life = max(float(self.half_life_days), 1e-6)
        decay = math.pow(0.5, age / half_life)
        return float(max(0.0, self.strength) * decay)

    def state_for_day(self, day_index: int) -> str:
        if day_index < self.effective_day:
            return "queued"
        return "active"

    def to_timeline_row(self, day_index: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "政策编号": self.policy_id,
            "政策文本": self.policy_text,
            "政策标签": self.label or "未命名政策",
            "生效日": int(self.effective_day),
            "基础强度": float(self.strength),
            "半衰期(天)": float(self.half_life_days),
            "是否谣言噪声": bool(self.rumor_noise),
            "来源": self.source,
            "创建日": int(self.created_day),
            "元数据": dict(self.metadata),
        }
        if day_index is not None:
            payload["当前状态"] = self.state_for_day(day_index)
            payload["当前强度"] = round(self.intensity_for_day(day_index), 4)
        return payload


@dataclass(slots=True)
class PolicySessionConfig:
    """Configuration for a stepwise policy experiment session."""

    total_days: int = 100
    base_policy: str = ""
    start_day: int = 1
    half_life_days: float = 14.0
    start_date: Any = None
    enable_random_policy_events: bool = False
    simulation_mode: str = "SMART"
    use_isolated_matching: bool = True
    market_pipeline_v2: bool = True
    llm_primary: bool = False
    deep_reasoning_pause_s: float = 0.0
    enable_policy_committee: bool = False
    runner_symbol: str = "A_SHARE_IDX"
    steps_per_day: int = 1
    model_priority: Sequence[str] = field(default_factory=lambda: ("glm-4-flashx",))
    hybrid_replay: bool = False
    exogenous_backdrop: Optional[Sequence[Mapping[str, Any]] | str] = None
    hybrid_backdrop_weight: float = 0.0
    event_dataset_version: str = "policy_lab_runtime"
    persist_runtime_events: bool = True


class PolicySession:
    """Stateful policy experiment session for Streamlit-style stepwise execution."""

    def __init__(
        self,
        *,
        environment: Optional[MarketEnvironment] = None,
        agents: Optional[Sequence[Any]] = None,
        config: Optional[PolicySessionConfig] = None,
        event_store: Optional[EventStore] = None,
    ) -> None:
        self.config = config or PolicySessionConfig()
        self.start_date = _coerce_date(self.config.start_date)
        self.calendar = pd.bdate_range(start=self.start_date, periods=max(1, int(self.config.total_days)))
        self._current_day = 0
        self._status = "idle"
        self._stopped = False
        self._daily_rows: List[Dict[str, Any]] = []
        self._policy_plans: List[PolicyPlan] = []
        self._event_queue = ScenarioEventQueue(
            event_store=event_store,
            dataset_version=str(self.config.event_dataset_version),
            persist_events=bool(self.config.persist_runtime_events),
            event_bus=getattr(environment, "event_bus", None) if environment is not None else None,
        )
        self._last_event_digest: EventDigest = self._event_queue.digest_for_time(0)
        self._last_result: Dict[str, Any] = {}
        self._environment = environment or self._build_environment(list(agents or []))
        self._event_queue.event_bus = getattr(self._environment, "event_bus", None)

        if self.config.base_policy:
            self.enqueue_policy(
                self.config.base_policy,
                effective_day=max(1, int(self.config.start_day)),
                strength=1.0,
                half_life_days=float(self.config.half_life_days),
                label="基础政策",
                source="base_policy",
            )

    @classmethod
    def create(
        cls,
        *,
        agents: Optional[Sequence[Any]] = None,
        environment: Optional[MarketEnvironment] = None,
        total_days: int = 100,
        base_policy: str = "",
        start_day: int = 1,
        half_life_days: float = 14.0,
        start_date: Any = None,
        enable_random_policy_events: bool = False,
        simulation_mode: str = "SMART",
        use_isolated_matching: bool = True,
        market_pipeline_v2: bool = True,
        llm_primary: bool = False,
        deep_reasoning_pause_s: float = 0.0,
        enable_policy_committee: bool = False,
        runner_symbol: str = "A_SHARE_IDX",
        steps_per_day: int = 1,
        model_priority: Optional[Sequence[str]] = None,
        hybrid_replay: bool = False,
        exogenous_backdrop: Optional[Sequence[Mapping[str, Any]] | str] = None,
        hybrid_backdrop_weight: float = 0.0,
        event_dataset_version: str = "policy_lab_runtime",
        persist_runtime_events: bool = True,
    ) -> "PolicySession":
        config = PolicySessionConfig(
            total_days=int(total_days),
            base_policy=str(base_policy or ""),
            start_day=int(start_day),
            half_life_days=float(half_life_days),
            start_date=start_date,
            enable_random_policy_events=bool(enable_random_policy_events),
            simulation_mode=str(simulation_mode or "SMART"),
            use_isolated_matching=bool(use_isolated_matching),
            market_pipeline_v2=bool(market_pipeline_v2),
            llm_primary=bool(llm_primary),
            deep_reasoning_pause_s=float(deep_reasoning_pause_s or 0.0),
            enable_policy_committee=bool(enable_policy_committee),
            runner_symbol=str(runner_symbol or "A_SHARE_IDX"),
            steps_per_day=max(1, int(steps_per_day)),
            model_priority=tuple(model_priority or ("glm-4-flashx",)),
            hybrid_replay=bool(hybrid_replay),
            exogenous_backdrop=exogenous_backdrop,
            hybrid_backdrop_weight=float(hybrid_backdrop_weight or 0.0),
            event_dataset_version=str(event_dataset_version or "policy_lab_runtime"),
            persist_runtime_events=bool(persist_runtime_events),
        )
        return cls(environment=environment, agents=agents, config=config)

    def _build_environment(self, agents: Sequence[Any]) -> MarketEnvironment:
        return MarketEnvironment(
            list(agents),
            use_isolated_matching=bool(self.config.use_isolated_matching),
            market_pipeline_v2=bool(self.config.market_pipeline_v2),
            simulation_mode=str(self.config.simulation_mode),
            llm_primary=bool(self.config.llm_primary),
            deep_reasoning_pause_s=float(self.config.deep_reasoning_pause_s),
            enable_policy_committee=bool(self.config.enable_policy_committee),
            enable_random_policy_events=bool(self.config.enable_random_policy_events),
            runner_symbol=str(self.config.runner_symbol),
            steps_per_day=int(self.config.steps_per_day),
            model_priority=list(self.config.model_priority),
            hybrid_replay=bool(self.config.hybrid_replay),
            exogenous_backdrop=self.config.exogenous_backdrop,
            hybrid_backdrop_weight=float(self.config.hybrid_backdrop_weight or 0.0),
        )

    def _mode_curve_calibration(self) -> Dict[str, float]:
        mode = str(self.config.simulation_mode or "SMART").strip().upper()
        if mode == "DEEP":
            return {
                "max_daily_move": 0.015,
                "raw_weight": 0.36,
                "policy_scale": 0.00065,
                "flow_scale": 0.00045,
                "sentiment_scale": 0.00038,
                "csad_scale": 0.00028,
                "mean_reversion_scale": 0.18,
                "backdrop_weight": 0.72,
            }
        return {
            "max_daily_move": 0.0075,
            "raw_weight": 0.18,
            "policy_scale": 0.00032,
            "flow_scale": 0.0002,
            "sentiment_scale": 0.00022,
            "csad_scale": 0.00015,
            "mean_reversion_scale": 0.28,
            "backdrop_weight": 0.82,
        }

    @property
    def environment(self) -> MarketEnvironment:
        return self._environment

    @property
    def current_day(self) -> int:
        return self._current_day

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == "running"

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def policies(self) -> List[PolicyPlan]:
        return list(self._policy_plans)

    @property
    def events(self) -> List[ExperimentEvent]:
        return list(self._event_queue.events)

    def start(self) -> "PolicySession":
        if not self._stopped:
            self._status = "running"
        return self

    def stop(self) -> "PolicySession":
        self._status = "stopped"
        self._stopped = True
        return self

    def pause(self) -> "PolicySession":
        if not self._stopped:
            self._status = "paused"
        return self

    def resume(self) -> "PolicySession":
        if not self._stopped:
            self._status = "running"
        return self

    def enqueue_policy(
        self,
        policy_text: str,
        *,
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life_days: Optional[float] = None,
        rumor_noise: bool = False,
        label: str = "",
        source: str = "manual",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        text = str(policy_text or "").strip()
        if not text:
            raise ValueError("policy_text 不能为空")
        day = int(effective_day) if effective_day is not None else (self._current_day + 1 if self.is_running else 1)
        day = max(1, day)
        policy_id = f"policy_{uuid.uuid4().hex[:10]}"
        plan = PolicyPlan(
            policy_id=policy_id,
            policy_text=text,
            effective_day=day,
            strength=float(max(0.0, strength)),
            half_life_days=float(half_life_days if half_life_days is not None else self.config.half_life_days),
            rumor_noise=bool(rumor_noise),
            label=str(label or text[:20]),
            source=str(source or "manual"),
            metadata=dict(metadata or {}),
            created_day=int(self._current_day),
        )
        self._policy_plans.append(plan)
        self._event_queue.append_event(
            ExperimentEvent.create(
                event_id=policy_id,
                event_type="policy",
                title=str(label or text[:20] or "政策事件"),
                raw_text=text,
                effective_day=day,
                strength=float(max(0.0, strength)),
                half_life=float(half_life_days if half_life_days is not None else self.config.half_life_days),
                scope=str(dict(metadata or {}).get("scope", "broad_market")),
                channels=list(dict(metadata or {}).get("channels", []) or []),
                confidence=float(dict(metadata or {}).get("confidence", 0.92) or 0.92),
                source=str(source or "manual"),
                metadata={
                    "policy_plan_id": policy_id,
                    "rumor_noise": bool(rumor_noise),
                    **dict(metadata or {}),
                },
                created_day=int(self._current_day),
                created_tick=int(getattr(self._environment, "simulation_time", 0) or 0),
            )
        )
        return policy_id

    add_policy = enqueue_policy
    append_policy = enqueue_policy

    def append_event(
        self,
        *,
        event_type: str,
        title: str,
        raw_text: str,
        effective_day: Optional[int] = None,
        effective_tick: Optional[int] = None,
        visibility_time: Any = "",
        strength: float = 1.0,
        half_life: float = 7.0,
        scope: str = "broad_market",
        channels: Optional[Sequence[str]] = None,
        confidence: float = 0.75,
        source: str = "manual",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Append a runtime event to the experiment timeline."""
        text = str(raw_text or "").strip()
        if not text:
            raise ValueError("raw_text 不能为空")
        day = int(effective_day) if effective_day is not None else (self._current_day + 1 if self.is_running else 1)
        day = max(1, day)
        return self._event_queue.append(
            event_type=event_type,
            title=str(title or text[:24]),
            raw_text=text,
            effective_day=day,
            effective_tick=effective_tick,
            strength=float(max(0.0, strength)),
            half_life=float(max(half_life, 1e-6)),
            scope=str(scope or "broad_market"),
            channels=list(channels or []),
            confidence=float(confidence),
            source=str(source or "manual"),
            metadata=dict(metadata or {}),
            created_day=int(self._current_day),
            created_tick=int(getattr(self._environment, "simulation_time", 0) or 0),
        )

    def append_news_event(
        self,
        raw_text: str,
        *,
        title: str = "重大新闻",
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 5.0,
        scope: str = "broad_market",
        confidence: float = 0.78,
        source: str = "runtime_news",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.append_event(
            event_type="major_news",
            title=title,
            raw_text=raw_text,
            effective_day=effective_day,
            strength=strength,
            half_life=half_life,
            scope=scope,
            channels=["information_network", "expectations"],
            confidence=confidence,
            source=source,
            metadata=metadata,
        )

    def append_macro_event(
        self,
        raw_text: str,
        *,
        title: str = "宏观冲击",
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 8.0,
        scope: str = "funding",
        confidence: float = 0.82,
        source: str = "runtime_macro",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.append_event(
            event_type="macro_shock",
            title=title,
            raw_text=raw_text,
            effective_day=effective_day,
            strength=strength,
            half_life=half_life,
            scope=scope,
            channels=["macro_state", "funding_conditions"],
            confidence=confidence,
            source=source,
            metadata=metadata,
        )

    def append_regime_shift(
        self,
        raw_text: str,
        *,
        title: str = "状态切换",
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 10.0,
        scope: str = "broad_market",
        confidence: float = 0.72,
        source: str = "runtime_regime",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.append_event(
            event_type="regime_shift",
            title=title,
            raw_text=raw_text,
            effective_day=effective_day,
            strength=strength,
            half_life=half_life,
            scope=scope,
            channels=["regime_transition", "risk_budget"],
            confidence=confidence,
            source=source,
            metadata=metadata,
        )

    def append_rumor_event(
        self,
        raw_text: str,
        *,
        title: str = "市场谣言",
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 3.0,
        scope: str = "expectations",
        confidence: float = 0.45,
        source: str = "runtime_rumor",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.append_event(
            event_type="rumor",
            title=title,
            raw_text=raw_text,
            effective_day=effective_day,
            strength=strength,
            half_life=half_life,
            scope=scope,
            channels=["rumor_network", "social_contagion"],
            confidence=confidence,
            source=source,
            metadata=metadata,
        )

    def append_refute_event(
        self,
        raw_text: str,
        *,
        title: str = "官方澄清",
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 4.0,
        scope: str = "expectations",
        confidence: float = 0.90,
        source: str = "runtime_refute",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.append_event(
            event_type="refute",
            title=title,
            raw_text=raw_text,
            effective_day=effective_day,
            strength=strength,
            half_life=half_life,
            scope=scope,
            channels=["authority_signal", "rumor_suppression"],
            confidence=confidence,
            source=source,
            metadata=metadata,
        )

    def append_regulatory_action(
        self,
        raw_text: str,
        *,
        title: str = "监管行动",
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 6.0,
        scope: str = "broad_market",
        confidence: float = 0.88,
        source: str = "runtime_regulatory",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.append_event(
            event_type="regulatory_action",
            title=title,
            raw_text=raw_text,
            effective_day=effective_day,
            strength=strength,
            half_life=half_life,
            scope=scope,
            channels=["regulatory_signal", "compliance_intensity"],
            confidence=confidence,
            source=source,
            metadata=metadata,
        )

    def _active_policies(self, day_index: int) -> List[PolicyPlan]:
        return [plan for plan in self._policy_plans if plan.intensity_for_day(day_index) > 1e-6]

    def _queued_policies(self, day_index: int) -> List[PolicyPlan]:
        return [plan for plan in self._policy_plans if plan.effective_day > day_index]

    def _build_day_policy_text(self, day_index: int) -> tuple[str, float, List[Dict[str, Any]]]:
        active = self._active_policies(day_index)
        if not active:
            return "", 0.0, []
        text_parts: List[str] = []
        timeline: List[Dict[str, Any]] = []
        total_strength = 0.0
        rumor_noise = False
        for plan in active:
            intensity = plan.intensity_for_day(day_index)
            total_strength += intensity
            text_parts.append(plan.policy_text)
            timeline.append(plan.to_timeline_row(day_index))
            rumor_noise = rumor_noise or plan.rumor_noise
        combined_text = "；".join(text_parts)
        if rumor_noise:
            combined_text = f"{combined_text}。市场传出相关谣言扰动，情绪波动加大。"
        strength_cap = 2.2 if str(self.config.simulation_mode or "SMART").strip().upper() == "DEEP" else 1.4
        return combined_text, float(_clip(total_strength, 0.0, strength_cap)), timeline

    def _resolve_day_prices(
        self,
        report: Dict[str, Any],
        active_timeline: List[Dict[str, Any]],
    ) -> tuple[float, float]:
        close_price = float(report.get("new_price", report.get("old_price", 0.0)) or 0.0)
        old_price = float(report.get("old_price", close_price) or close_price)
        if old_price <= 0.0:
            base_price = close_price if close_price > 0.0 else 100.0
            return float(base_price), float(base_price)
        if not active_timeline:
            return float(old_price), float(close_price)

        macro_state = report.get("macro_state", {}) if isinstance(report.get("macro_state"), dict) else {}
        diagnostics = report.get("behavioral_diagnostics", {}) if isinstance(report.get("behavioral_diagnostics"), dict) else {}
        policy_input = report.get("policy_input", {}) if isinstance(report.get("policy_input"), dict) else {}
        buy_volume = float(report.get("buy_volume", 0.0) or 0.0)
        sell_volume = float(report.get("sell_volume", 0.0) or 0.0)
        policy_strength = sum(float(item.get("当前强度", item.get("基础强度", 0.0)) or 0.0) for item in active_timeline)
        sentiment_index = float(macro_state.get("sentiment_index", 0.0) or 0.0)
        csad = float(diagnostics.get("csad", 0.0) or 0.0)
        flow_imbalance = (buy_volume - sell_volume) / max(abs(buy_volume) + abs(sell_volume), 1.0)
        policy_bias = float(policy_input.get("policy_intensity", 0.0) or 0.0)
        calibration = self._mode_curve_calibration()
        sentiment_edge = sentiment_index - 0.5
        raw_return = (close_price - old_price) / old_price if old_price > 0 else 0.0
        target_return = (
            0.0002
            + calibration["policy_scale"] * policy_strength
            + 0.00018 * policy_bias
            + calibration["sentiment_scale"] * sentiment_edge
            + calibration["flow_scale"] * flow_imbalance
            - calibration["csad_scale"] * csad
        )
        if abs(target_return) < 1e-6:
            target_return = 0.00015 * max(policy_strength, 0.2)
        target_return -= calibration["mean_reversion_scale"] * raw_return
        blended_target = (
            calibration["backdrop_weight"] * raw_return
            + (1.0 - calibration["backdrop_weight"]) * target_return
        )
        if abs(raw_return) > 1e-9:
            calibrated_return = (
                calibration["raw_weight"] * raw_return
                + (1.0 - calibration["raw_weight"]) * blended_target
            )
        else:
            calibrated_return = blended_target
        calibrated_return = _clip(
            calibrated_return,
            -float(calibration["max_daily_move"]),
            float(calibration["max_daily_move"]),
        )
        close_price = old_price * (1.0 + calibrated_return)
        return float(old_price), float(close_price)

    def _record_for_day(self, day_index: int, report: Dict[str, Any], active_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        trade_date = self.calendar[min(max(day_index - 1, 0), len(self.calendar) - 1)]
        raw_close_price = float(report.get("new_price", report.get("old_price", 0.0)) or 0.0)
        raw_old_price = float(report.get("old_price", raw_close_price) or raw_close_price)
        old_price, close_price = self._resolve_day_prices(report, active_timeline)
        market_return = float((close_price - old_price) / old_price) if old_price > 0 else 0.0
        event_digest = dict(report.get("runtime_event_digest", {}) or self._last_event_digest.to_dict())
        row = {
            "交易日序号": int(day_index),
            "交易日": trade_date.strftime("%Y-%m-%d"),
            "收盘价": round(close_price, 4),
            "涨跌幅": round(market_return, 6),
            "总买量": float(report.get("buy_volume", 0.0) or 0.0),
            "总卖量": float(report.get("sell_volume", 0.0) or 0.0),
            "成交笔数": int(report.get("trade_count", 0) or 0),
            "活跃政策数": len(active_timeline),
            "政策文本": str(report.get("policy_input", {}).get("policy_text", "")),
            "政策强度": float(report.get("policy_input", {}).get("policy_intensity", 0.0) or 0.0),
            "政策来源": str(report.get("policy_input", {}).get("policy_source", "")),
            "事件摘要": str(event_digest.get("combined_text", "")),
            "活跃事件数": int(event_digest.get("active_count", 0) or 0),
            "恐慌度": float(max(0.0, 1.0 - float(report.get("macro_state", {}).get("sentiment_index", 0.0) or 0.0))),
            "羊群度": float(report.get("behavioral_diagnostics", {}).get("csad", 0.0) or 0.0),
            "阶段": "运行中" if self.is_running else self.status,
            "显示平滑参数": {
                "visualization_only": bool(active_timeline),
                "evaluation_allowed": False if active_timeline else bool(report.get("evaluation_allowed", True)),
                "raw_old_price": float(raw_old_price),
                "raw_close_price": float(raw_close_price),
                "display_old_price": float(old_price),
                "display_close_price": float(close_price),
                "price_source": str(report.get("price_source", "policy_session_report")),
                "note": "PolicySession display smoothing is excluded from reward, optimizer, and replay scorecard.",
            },
            "原始报告": dict(report),
            "政策时间轴": list(active_timeline),
            "事件时间轴": list(event_digest.get("active_events", [])),
        }
        if "macro_state" in report:
            row["宏观状态"] = dict(report.get("macro_state", {}))
        if "policy_transmission_chain" in report:
            row["政策传导链"] = dict(report.get("policy_transmission_chain", {}))
        if "thinking_stats" in report:
            row["思考统计"] = dict(report.get("thinking_stats", {}))
        if "market_state" in report:
            row["市场状态"] = dict(report.get("market_state", {}))
        return row

    def _summary_payload(self) -> Dict[str, Any]:
        frame = self.to_frame()
        event_digest = self._event_queue.digest_for_time(self._current_day)
        if frame.empty:
            return {
                "状态": self.status,
                "当前交易日": int(self._current_day),
                "总交易日": int(self.config.total_days),
                "已完成交易日": 0,
                "累计收益率": 0.0,
                "最大回撤": 0.0,
                "最新收盘价": 0.0,
                "政策数量": len(self._policy_plans),
                "事件数量": len(self._event_queue.events),
                "活跃事件数": len(event_digest.active_events),
                "待生效事件数": len(event_digest.queued_events),
                "自动随机政策事件": bool(self.config.enable_random_policy_events),
            }
        close = frame["收盘价"].astype(float)
        returns = close.pct_change().fillna(0.0)
        drawdown = close / close.cummax() - 1.0
        return {
            "状态": self.status,
            "当前交易日": int(self._current_day),
            "总交易日": int(self.config.total_days),
            "已完成交易日": int(len(frame)),
            "累计收益率": float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0),
            "最大回撤": float(abs(drawdown.min())),
            "最新收盘价": float(close.iloc[-1]),
            "初始收盘价": float(close.iloc[0]),
            "日均波动率": float(returns.std()),
            "政策数量": len(self._policy_plans),
            "事件数量": len(self._event_queue.events),
            "活跃事件数": len(event_digest.active_events),
            "待生效事件数": len(event_digest.queued_events),
            "已生效政策数": len([plan for plan in self._policy_plans if plan.effective_day <= self._current_day]),
            "待生效政策数": len(self._queued_policies(self._current_day)),
            "自动随机政策事件": bool(self.config.enable_random_policy_events),
        }

    def _timeline_payload(self) -> List[Dict[str, Any]]:
        return [plan.to_timeline_row(self._current_day) for plan in self._policy_plans]

    def _event_timeline_payload(self) -> List[Dict[str, Any]]:
        return self._event_queue.timeline(self._current_day)

    def to_frame(self) -> pd.DataFrame:
        if not self._daily_rows:
            return pd.DataFrame(
                columns=[
                    "交易日序号",
                    "交易日",
                    "收盘价",
                    "涨跌幅",
                    "总买量",
                    "总卖量",
                    "成交笔数",
                    "活跃政策数",
                    "政策文本",
                    "政策强度",
                    "政策来源",
                    "事件摘要",
                    "活跃事件数",
                    "恐慌度",
                    "羊群度",
                    "阶段",
                ]
            )
        frame = pd.DataFrame(self._daily_rows)
        return frame.drop(columns=["原始报告"], errors="ignore").reset_index(drop=True)

    def _structured_report_sections(self) -> Dict[str, Any]:
        summary = self._summary_payload()
        frame = self.to_frame()
        timeline = self._timeline_payload()
        total_return = float(summary.get("累计收益率", 0.0) or 0.0)
        max_drawdown = float(summary.get("最大回撤", 0.0) or 0.0)
        avg_vol = float(summary.get("日均波动率", 0.0) or 0.0)
        avg_active_policy = (
            float(frame["活跃政策数"].astype(float).mean()) if (not frame.empty and "活跃政策数" in frame.columns) else 0.0
        )
        avg_panic = float(frame["恐慌度"].astype(float).mean()) if (not frame.empty and "恐慌度" in frame.columns) else 0.0
        start_panic = float(frame["恐慌度"].astype(float).iloc[0]) if (not frame.empty and "恐慌度" in frame.columns) else 0.0
        end_panic = float(frame["恐慌度"].astype(float).iloc[-1]) if (not frame.empty and "恐慌度" in frame.columns) else 0.0
        direction = "正向" if total_return > 0.005 else ("负向" if total_return < -0.005 else "中性")

        amplitude = abs(total_return)
        if amplitude >= 0.08:
            impact_level = "高"
        elif amplitude >= 0.03:
            impact_level = "中"
        else:
            impact_level = "低"

        one_line = (
            f"本轮政策组合对市场形成{direction}影响，累计收益率 {total_return:.2%}，"
            f"最大回撤 {max_drawdown:.2%}，整体影响等级为{impact_level}。"
        )
        market_feedback = (
            f"样本期内平均活跃政策数 {avg_active_policy:.2f}，平均恐慌度 {avg_panic:.4f}，"
            f"日均波动率 {avg_vol:.4f}，反映市场在政策冲击下的再定价过程。"
        )
        transmission_evidence = [
            f"政策时序证据：当前共记录 {len(timeline)} 条政策，已完成 {summary.get('已完成交易日', 0)} 个交易日观察。",
            (
                f"情绪传导证据：恐慌度由 {start_panic:.4f} 变动至 {end_panic:.4f}，"
                f"变化 {end_panic - start_panic:+.4f}。"
            ),
            f"价格反馈证据：累计收益率 {total_return:.2%}，最大回撤 {max_drawdown:.2%}。",
        ]
        key_risks = [
            "政策预期透支后可能触发估值回吐。",
            "外部冲击（地缘、流动性、监管）可能改变既有路径。",
            "政策叠加过密时，传导链条可能出现边际递减。",
        ]
        side_effects = [
            "短期成交放大可能伴随波动率抬升。",
            "板块分化可能导致指数表现与微观体感不一致。",
            "情绪修复不均衡时，可能出现阶段性拥挤交易。",
        ]
        risk_level = "高" if max_drawdown >= 0.15 else ("中" if max_drawdown >= 0.08 else "低")
        recommendations = {
            "short_term": [
                "维持政策沟通连续性，稳定关键预期变量。",
                "对高波动时段设置流动性与风险监测阈值。",
            ],
            "mid_term": [
                "滚动评估政策组合的边际效应，及时做强度校准。",
                "跟踪跨市场传导，防止单一指标误判政策效果。",
            ],
            "long_term": [
                "建立政策-市场反馈闭环，沉淀可复用干预模板。",
                "将宏观、行业与情绪指标纳入统一评估基线。",
            ],
        }
        monitoring_kpis = [
            {
                "name": "累计收益率",
                "value": round(total_return, 6),
                "display": f"{total_return:.2%}",
                "signal": "positive" if total_return > 0 else ("negative" if total_return < 0 else "neutral"),
            },
            {
                "name": "最大回撤",
                "value": round(max_drawdown, 6),
                "display": f"{max_drawdown:.2%}",
                "signal": "negative" if max_drawdown >= 0.1 else "neutral",
            },
            {
                "name": "日均波动率",
                "value": round(avg_vol, 6),
                "display": f"{avg_vol:.4f}",
                "signal": "warning" if avg_vol >= 0.02 else "stable",
            },
            {
                "name": "平均恐慌度",
                "value": round(avg_panic, 6),
                "display": f"{avg_panic:.4f}",
                "signal": "warning" if avg_panic >= 0.55 else "stable",
            },
            {
                "name": "平均活跃政策数",
                "value": round(avg_active_policy, 4),
                "display": f"{avg_active_policy:.2f}",
                "signal": "dense" if avg_active_policy >= 2.0 else "normal",
            },
        ]
        return {
            "one_line_conclusion": one_line,
            "impact_evaluation": {
                "overall_verdict": one_line,
                "impact_direction": direction,
                "impact_level": impact_level,
                "transmission_mechanism_evidence": transmission_evidence,
                "market_feedback": market_feedback,
            },
            "risk_assessment": {
                "risk_level": risk_level,
                "key_risks": key_risks,
                "side_effects": side_effects,
            },
            "action_recommendations": recommendations,
            "monitoring_kpis": monitoring_kpis,
        }

    def build_report_payload(self) -> Dict[str, Any]:
        frame = self.to_frame()
        sections = self._structured_report_sections()
        return {
            "标题": "政策试验台会话报告",
            "会话摘要": self._summary_payload(),
            "政策时间轴": self._timeline_payload(),
            "事件时间轴": self._event_timeline_payload(),
            "最新事件摘要": self._last_event_digest.to_dict(),
            "日度结果": frame.to_dict(orient="records"),
            "已完成交易日": int(len(frame)),
            "当前交易日": int(self._current_day),
            "总交易日": int(self.config.total_days),
            "自动随机政策事件": bool(self.config.enable_random_policy_events),
            "一句话结论": sections["one_line_conclusion"],
            "impact_evaluation": sections["impact_evaluation"],
            "risk_assessment": sections["risk_assessment"],
            "action_recommendations": sections["action_recommendations"],
            "monitoring_kpis": sections["monitoring_kpis"],
        }

    def _fallback_report_markdown(self) -> str:
        payload = self.build_report_payload()
        summary = payload["会话摘要"]
        timeline = payload["政策时间轴"]
        frame = self.to_frame()
        impact = dict(payload.get("impact_evaluation", {}) or {})
        risk = dict(payload.get("risk_assessment", {}) or {})
        recommendations = dict(payload.get("action_recommendations", {}) or {})
        kpis = list(payload.get("monitoring_kpis", []) or [])
        top_policies = timeline[:8]
        lines = [
            "# 政策试验台会话报告",
            "",
            "## 会话摘要",
            f"- 状态：{summary['状态']}",
            f"- 当前交易日：{summary['当前交易日']}/{summary['总交易日']}",
            f"- 已完成交易日：{summary['已完成交易日']}",
            f"- 累计收益率：{summary['累计收益率']:.2%}",
            f"- 最大回撤：{summary['最大回撤']:.2%}",
            f"- 最新收盘价：{summary['最新收盘价']:.2f}",
            f"- 政策数量：{summary['政策数量']}",
            f"- 自动随机政策事件：{'关闭' if not summary['自动随机政策事件'] else '开启'}",
            "",
            "## 一句话结论",
            f"- {payload.get('一句话结论', '')}",
            "",
            "## 政策时间轴",
        ]
        if top_policies:
            for item in top_policies:
                lines.append(
                    f"- 第{item['生效日']}日 {item['政策标签']}：强度 {item.get('当前强度', item['基础强度']):.3f}，半衰期 {item['半衰期(天)']:.1f} 天"
                )
        else:
            lines.append("- 当前会话尚未安排政策。")
        lines.extend(
            [
                "",
                "## 日度结果概览",
                f"- 已记录日数：{len(frame)}",
            ]
        )
        if not frame.empty:
            tail = frame.tail(5)
            for _, row in tail.iterrows():
                lines.append(
                    f"- {row['交易日']} | 收盘价 {float(row['收盘价']):.2f} | 涨跌幅 {float(row['涨跌幅']):+.2%} | 活跃政策 {int(row['活跃政策数'])}"
                )
        lines.extend(
            [
                "",
                "## 政策影响评价",
                f"- 影响方向：{impact.get('impact_direction', '中性')}",
                f"- 影响等级：{impact.get('impact_level', '中')}",
                f"- 市场反馈：{impact.get('market_feedback', '')}",
                "",
                "## 传导机制证据",
            ]
        )
        for evidence in impact.get("transmission_mechanism_evidence", []) or []:
            lines.append(f"- {evidence}")
        lines.extend(
            [
                "",
                "## 风险与副作用",
                f"- 风险等级：{risk.get('risk_level', '中')}",
                "- 关键风险：",
            ]
        )
        for item in risk.get("key_risks", []) or []:
            lines.append(f"  - {item}")
        lines.append("- 潜在副作用：")
        for item in risk.get("side_effects", []) or []:
            lines.append(f"  - {item}")
        lines.extend(["", "## 分阶段建议", "- 短期建议："])
        for item in recommendations.get("short_term", []) or []:
            lines.append(f"  - {item}")
        lines.append("- 中期建议：")
        for item in recommendations.get("mid_term", []) or []:
            lines.append(f"  - {item}")
        lines.append("- 长期建议：")
        for item in recommendations.get("long_term", []) or []:
            lines.append(f"  - {item}")
        lines.extend(["", "## 关键监测指标"])
        for item in kpis:
            lines.append(f"- {item.get('name', '指标')}：{item.get('display', item.get('value', ''))}（{item.get('signal', 'neutral')}）")
        return "\n".join(lines)

    def _llm_report(self) -> Optional[str]:
        try:
            from core.inference.api_backend import APIBackend
        except Exception:
            return None

        payload = self.build_report_payload()
        prompt = "\n".join(
            [
                "请根据下面的政策试验台会话结果，生成一份面向政策制定者和评委展示的中文 Markdown 报告。",
                "要求：必须是自然语言 Markdown，不要输出 JSON，不要输出代码块。",
                "不要逐项抄录数据；必须解释政策实施后可能出现的市场状态、主体行为变化、风险触发条件和监测重点。",
                "报告必须包含以下章节：",
                "1) 一句话结论",
                "2) 政策影响评价",
                "3) 传导机制证据",
                "4) 市场反馈",
                "5) 风险与副作用",
                "6) 分阶段建议（短期/中期/长期）",
                "7) 关键监测指标",
                "请强调政策影响会随时间衰减，说明中途追加政策后市场变化的机制，不要使用套话式总结。",
                f"数据：{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}",
            ]
        )
        backend = APIBackend(
            model=str(self.config.model_priority[0]) if self.config.model_priority else "glm-4-flashx",
            max_tokens=900,
            temperature=0.25,
        )
        response = str(
            backend.generate(
                prompt,
                system_prompt="你是政策试验台的中文评估专家，擅长从仿真数据关系推导政策后果和治理含义。",
                timeout_budget=25.0,
                fallback_response="",
            )
            or ""
        ).strip()
        if not response or response.startswith("[API Error]"):
            return None
        return response

    def generate_report(self, *, use_llm: bool = False) -> Dict[str, Any]:
        llm_report = self._llm_report() if use_llm else None
        report_text = llm_report or self._fallback_report_markdown()
        return {
            "报告正文": report_text,
            "报告数据": self.build_report_payload(),
            "是否使用大模型": bool(llm_report),
        }

    async def advance_async(self, days: int = 1) -> Dict[str, Any]:
        if self._stopped:
            return self._result_payload()
        if self._status == "idle":
            self.start()

        steps = max(0, int(days))
        for _ in range(steps):
            if self._current_day >= int(self.config.total_days):
                self._status = "completed"
                self._stopped = True
                break

            self._current_day += 1
            day_text, day_strength, active_timeline = self._build_day_policy_text(self._current_day)
            event_digest = self._event_queue.digest_for_time(
                self._current_day,
                tick=int(getattr(self._environment, "simulation_time", 0) or 0) + 1,
            )
            self._last_event_digest = event_digest
            event_text = event_digest.to_policy_text()
            scheduled_text = event_text or day_text
            scheduled_strength = float(event_digest.aggregate_strength if event_text else day_strength)
            if scheduled_text:
                self._environment.schedule_policy_shock(
                    scheduled_text,
                    intensity=float(scheduled_strength),
                    policy_id=f"session_day_{self._current_day}",
                    source="policy_session_events",
                    metadata={
                        "day_index": int(self._current_day),
                        "active_policies": list(active_timeline),
                        "runtime_event_digest": event_digest.to_dict(),
                    },
                )
            report = dict(await self._environment.simulation_step())
            report["runtime_event_digest"] = event_digest.to_dict()
            report.setdefault("policy_input", {})
            if isinstance(report["policy_input"], dict):
                report["policy_input"]["runtime_event_digest"] = event_digest.to_dict()
            daily_row = self._record_for_day(self._current_day, report, active_timeline)
            self._daily_rows.append(daily_row)
            self._last_result = report

        if self._current_day >= int(self.config.total_days):
            self._status = "completed"
            self._stopped = True
        return self._result_payload()

    def advance(self, days: int = 1) -> Dict[str, Any]:
        return _run_sync(self.advance_async(days=days))

    def run(self) -> Dict[str, Any]:
        remaining = max(0, int(self.config.total_days) - int(self._current_day))
        return self.advance(days=remaining)

    def _result_payload(self) -> Dict[str, Any]:
        frame = self.to_frame()
        summary = self._summary_payload()
        active_policies = [plan.to_timeline_row(self._current_day) for plan in self._active_policies(self._current_day)]
        queued_policies = [plan.to_timeline_row(self._current_day) for plan in self._queued_policies(self._current_day)]
        event_digest = self._event_queue.digest_for_time(self._current_day)
        report_payload = self.build_report_payload()
        result = {
            "frame": frame,
            "current_day": int(self._current_day),
            "active_policies": active_policies,
            "queued_policies": queued_policies,
            "active_events": list(event_digest.active_events),
            "queued_events": list(event_digest.queued_events),
            "expired_events": list(event_digest.expired_events),
            "event_timeline": self._event_timeline_payload(),
            "event_digest": event_digest.to_dict(),
            "summary": summary,
            "report_payload": report_payload,
            "status": self.status,
        }
        if self._last_result:
            result["last_step_report"] = dict(self._last_result)
        return result


__all__ = ["PolicyPlan", "PolicySession", "PolicySessionConfig", "ExperimentEvent", "EventDigest"]
