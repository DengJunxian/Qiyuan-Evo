import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.model_router import ModelRouter
from data_flywheel.schemas import ExtractedEntity, RawArticle, SeedEvent
from data_flywheel.text_factor_pipeline import TextFactorPipeline

logger = logging.getLogger(__name__)


class NlpProcessor:
    """LLM-first NLP processor with deterministic factorization fallback."""

    SYSTEM_PROMPT = """你是资深金融情报分析师。
请阅读给定财经文本并仅输出 JSON:
{
  "entities": [{"name": "...", "entity_type": "company|sector|indicator|policy|person", "confidence": 0.0}],
  "sentiment": 0.0,
  "sentiment_label": "利好|利空|中性",
  "impact_level": "low|medium|high|critical",
  "affected_sectors": ["..."],
  "summary": "一句话摘要"
}
"""

    def __init__(self, model_router: ModelRouter, factor_pipeline: Optional[TextFactorPipeline] = None):
        self.router = model_router
        self.factor_pipeline = factor_pipeline or TextFactorPipeline()

    def process(self, article: RawArticle) -> SeedEvent:
        logger.info("Processing NLP for article: %s", article.title[:40])
        user_prompt = f"标题: {article.title}\n正文: {article.content}"
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        llm_data: Optional[Dict[str, Any]] = None
        try:
            content, _, _ = self.router.sync_call_with_fallback(
                messages=messages,
                priority_models=["deepseek-chat", "glm-4-flashx", "deepseek-reasoner"],
                timeout_budget=20.0,
            )
            llm_data = self._parse_json_response(content)
            if llm_data is None:
                logger.warning("Failed to parse LLM output for '%s'; fallback mode.", article.title)
        except Exception as exc:
            logger.error("NLP processing failed for '%s': %s", article.title, exc)

        if llm_data is None:
            return self._fallback_process(article)
        return self._build_seed_event(article, llm_data)

    def process_batch(self, articles: List[RawArticle]) -> List[SeedEvent]:
        return [self.process(a) for a in articles]

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                return None
        return None

    def _build_seed_event(self, article: RawArticle, data: Dict[str, Any]) -> SeedEvent:
        entities: List[ExtractedEntity] = []
        for e in data.get("entities", []) or []:
            if isinstance(e, dict) and "name" in e and "entity_type" in e:
                entities.append(
                    ExtractedEntity(
                        name=str(e.get("name", "Unknown")),
                        entity_type=str(e.get("entity_type", "other")),
                        confidence=float(e.get("confidence", 0.5)),
                    )
                )

        factors = self.factor_pipeline.analyze(article.title, article.content, llm_payload=data)
        sentiment = self._safe_float(data.get("sentiment"), factors.get("sentiment_score", 0.0))
        sentiment = max(-1.0, min(1.0, sentiment))

        sentiment_label = str(data.get("sentiment_label", "")).strip()
        if not sentiment_label:
            sentiment_label = self._label_from_sentiment(sentiment)

        impact_level = str(data.get("impact_level", "")).strip()
        if not impact_level:
            impact_level = self._infer_impact_level(factors, len(entities))

        return SeedEvent(
            source=article.source,
            source_url=article.source_url,
            title=article.title,
            summary=str(data.get("summary") or article.title),
            entities=entities,
            sentiment=sentiment,
            sentiment_label=sentiment_label,
            impact_level=impact_level,
            affected_sectors=self._normalize_str_list(data.get("affected_sectors")),
            raw_text=article.content,
            created_at=article.published_at or article.fetched_at,
            text_factors=factors,
        )

    def _fallback_process(self, article: RawArticle) -> SeedEvent:
        text = f"{article.title} {article.content}"
        bullish_keywords = ("上涨", "增加", "利好", "支持", "鼓励", "降准", "降息", "回暖")
        bearish_keywords = ("下跌", "减少", "利空", "限制", "处罚", "违约", "爆雷", "崩盘")

        bull_count = sum(1 for k in bullish_keywords if k in text)
        bear_count = sum(1 for k in bearish_keywords if k in text)

        if bull_count > bear_count:
            sentiment = 0.5
        elif bear_count > bull_count:
            sentiment = -0.5
        else:
            sentiment = 0.0

        factors = self.factor_pipeline.analyze(
            article.title,
            article.content,
            llm_payload={"sentiment": sentiment, "entities": [], "affected_sectors": []},
        )
        impact_level = "medium" if (bull_count + bear_count) > 0 else "low"

        return SeedEvent(
            source=article.source,
            source_url=article.source_url,
            title=article.title,
            summary=article.title,
            sentiment=sentiment,
            sentiment_label=self._label_from_sentiment(sentiment),
            impact_level=impact_level,
            raw_text=article.content,
            created_at=article.published_at or article.fetched_at,
            text_factors=factors,
        )

    def _infer_impact_level(self, factors: Dict[str, Any], entity_count: int) -> str:
        financial = factors.get("financial_factors", {}) or {}
        shock = self._safe_float(financial.get("policy_shock"), 0.0)
        panic = self._safe_float(financial.get("panic_index"), 0.0)
        greed = self._safe_float(financial.get("greed_index"), 0.0)
        intensity = max(shock, panic, greed)
        if intensity > 0.85 or (shock > 0.7 and entity_count >= 3):
            return "critical"
        if intensity > 0.65:
            return "high"
        if intensity > 0.35:
            return "medium"
        return "low"

    def _max_impact(self, lhs: str, rhs: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return lhs if order.get(lhs, 0) >= order.get(rhs, 0) else rhs

    def _normalize_str_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _label_from_sentiment(self, score: float) -> str:
        if score > 0.15:
            return "利好"
        if score < -0.15:
            return "利空"
        return "中性"
