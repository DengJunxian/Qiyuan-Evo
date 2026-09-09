"""Deterministic natural-language runtime event compiler."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from core.events.runtime_event import RuntimeEvent, normalize_event_type


def _contains(text: str, keywords: Sequence[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower or keyword in text for keyword in keywords)


class RuntimeEventCompiler:
    """Compile user text into structured runtime events without requiring an LLM."""

    def compile_text(
        self,
        raw_text: str,
        *,
        timestamp: Optional[float] = None,
        trading_day: str = "",
        source: str = "manual",
        event_type: str = "",
        credibility: Optional[float] = None,
        structured_payload: Optional[Mapping[str, Any]] = None,
    ) -> RuntimeEvent:
        text = str(raw_text or "").strip()
        inferred_type = normalize_event_type(event_type or self._infer_type(text))
        payload = {**self._infer_payload(text, inferred_type), **dict(structured_payload or {})}
        sectors = list(payload.pop("affected_sectors", []))
        groups = list(payload.pop("affected_agent_groups", []))
        duration = int(payload.pop("expected_duration", 5))
        strength = float(payload.pop("shock_strength", 1.0))
        conf = float(credibility if credibility is not None else payload.pop("credibility", self._default_credibility(inferred_type)))
        ts = float(timestamp if timestamp is not None else datetime.now().timestamp())
        return RuntimeEvent.create(
            raw_text=text,
            event_type=inferred_type,
            timestamp=ts,
            trading_day=trading_day,
            source=source,
            credibility=conf,
            affected_sectors=sectors,
            affected_agent_groups=groups,
            expected_duration=duration,
            shock_strength=strength,
            structured_payload=payload,
            title=self._title(inferred_type, text),
        )

    def compile(self, event: RuntimeEvent | Mapping[str, Any] | str, **kwargs: Any) -> RuntimeEvent:
        if isinstance(event, RuntimeEvent):
            return event
        if isinstance(event, Mapping):
            payload = dict(event)
            return RuntimeEvent.create(
                raw_text=str(payload.get("raw_text", payload.get("text", ""))),
                event_type=str(payload.get("event_type", kwargs.get("event_type", "major_news"))),
                timestamp=payload.get("timestamp", kwargs.get("timestamp")),
                trading_day=str(payload.get("trading_day", kwargs.get("trading_day", ""))),
                source=str(payload.get("source", kwargs.get("source", "manual"))),
                credibility=float(payload.get("credibility", payload.get("confidence", kwargs.get("credibility", 0.75))) or 0.75),
                affected_sectors=payload.get("affected_sectors", ()),
                affected_agent_groups=payload.get("affected_agent_groups", ()),
                expected_duration=int(payload.get("expected_duration", kwargs.get("expected_duration", 5)) or 5),
                shock_strength=float(payload.get("shock_strength", payload.get("strength", kwargs.get("shock_strength", 1.0))) or 1.0),
                structured_payload=dict(payload.get("structured_payload", payload.get("payload", {})) or {}),
                title=str(payload.get("title", "")),
                event_id=str(payload.get("event_id", "")),
            )
        return self.compile_text(str(event), **kwargs)

    @staticmethod
    def _infer_type(text: str) -> str:
        if _contains(text, ("谣言", "传闻", "rumor")):
            return "rumor"
        if _contains(text, ("辟谣", "澄清", "refute", "clarification")):
            return "refute"
        if _contains(text, ("监管", "处罚", "窗口指导", "regulator", "regulatory")):
            return "regulatory_action"
        if _contains(text, ("印花税", "降准", "降息", "政策", "流动性投放", "稳定资金", "国家队", "policy")):
            return "policy"
        if _contains(text, ("外盘", "美股", "汇率", "海外", "冲击", "macro", "external")):
            return "macro_shock"
        return "major_news"

    @staticmethod
    def _default_credibility(event_type: str) -> float:
        if event_type == "rumor":
            return 0.45
        if event_type in {"policy", "regulatory_action", "refute"}:
            return 0.88
        return 0.72

    @staticmethod
    def _title(event_type: str, text: str) -> str:
        prefix = {
            "policy": "runtime policy",
            "major_news": "runtime news",
            "macro_shock": "runtime macro shock",
            "rumor": "runtime rumor",
            "refute": "runtime refute",
            "regulatory_action": "runtime regulatory action",
        }.get(event_type, "runtime event")
        return f"{prefix}: {text[:20]}" if text else prefix

    @staticmethod
    def _infer_payload(text: str, event_type: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "compiler": "runtime_event_compiler_v1",
            "channels": ["event_context", "agent_belief"],
            "scope": "broad_market",
            "sentiment_impact": 0.0,
            "funding_stress": 0.0,
            "liquidity_impact": 0.0,
            "compliance_pressure": 0.0,
            "affected_agent_groups": [],
            "affected_sectors": [],
            "expected_duration": 5,
            "shock_strength": 1.0,
        }
        if event_type == "rumor":
            payload.update(
                sentiment_impact=-0.18,
                funding_stress=0.08,
                affected_agent_groups=["retail", "leveraged_flow"],
                channels=["social_contagion", "agent_belief", "order_flow"],
                expected_duration=3,
                shock_strength=0.85,
            )
        elif event_type == "refute":
            payload.update(
                sentiment_impact=0.12,
                compliance_pressure=0.05,
                affected_agent_groups=["retail", "leveraged_flow"],
                channels=["authority_signal", "agent_belief", "report_narrative"],
                expected_duration=4,
            )
        elif event_type == "policy":
            payload.update(
                sentiment_impact=0.10,
                liquidity_impact=0.12,
                affected_agent_groups=["institution", "policy_capital", "retail"],
                channels=["macro_state", "policy_context", "order_flow"],
                expected_duration=10,
            )
        elif event_type == "regulatory_action":
            payload.update(
                sentiment_impact=-0.04,
                compliance_pressure=0.22,
                affected_agent_groups=["regulator", "quant", "leveraged_flow"],
                channels=["regulator_observation", "agent_belief", "order_flow"],
                expected_duration=6,
            )
        elif event_type == "macro_shock":
            payload.update(
                sentiment_impact=-0.10,
                funding_stress=0.16,
                affected_agent_groups=["foreign", "institution", "leveraged_flow"],
                channels=["macro_state", "order_flow", "report_narrative"],
                expected_duration=8,
            )
        else:
            payload.update(
                sentiment_impact=0.05 if _contains(text, ("利好", "上涨", "支持", "positive")) else -0.05 if _contains(text, ("利空", "下跌", "风险", "negative")) else 0.0,
                affected_agent_groups=["retail", "institution"],
                channels=["news_heat", "agent_belief", "report_narrative"],
                expected_duration=5,
            )

        if _contains(text, ("新能源", "光伏", "电动车")):
            payload["affected_sectors"].append("new_energy")
        if _contains(text, ("地产", "房地产")):
            payload["affected_sectors"].append("real_estate")
        if _contains(text, ("券商", "证券")):
            payload["affected_sectors"].append("brokerage")
        if _contains(text, ("银行", "金融")):
            payload["affected_sectors"].append("financials")
        if _contains(text, ("流动性", "降准", "投放")):
            payload["liquidity_impact"] = max(float(payload["liquidity_impact"]), 0.20)
        if _contains(text, ("印花税", "减税")):
            payload["sentiment_impact"] = max(float(payload["sentiment_impact"]), 0.16)
            payload["affected_agent_groups"] = list(dict.fromkeys([*payload["affected_agent_groups"], "retail", "quant"]))
        if _contains(text, ("强平", "融资", "保证金")):
            payload["funding_stress"] = max(float(payload["funding_stress"]), 0.25)
            payload["affected_agent_groups"] = list(dict.fromkeys([*payload["affected_agent_groups"], "leveraged_flow"]))
        return payload


__all__ = ["RuntimeEventCompiler"]
