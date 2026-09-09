"""Text factorization pipeline for policy/news/social streams.

This module keeps heavy dependencies optional:
- BERTopic (topic modeling)
- FinBERT via transformers pipeline (finance sentiment)

If optional packages are not available, it falls back to deterministic
rule-based extraction so the Civitas data flywheel stays runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import re


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class TopicSignal:
    topic: str
    score: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "score": _clamp(float(self.score), 0.0, 1.0),
            "source": self.source,
        }


class TextFactorPipeline:
    """Build topic/sentiment/financial factors from a single text item."""


    TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "monetary_policy": ("降准", "降息", "mlf", "lpr", "央行", "liquidity", "rate cut"),
        "fiscal_policy": ("财政", "专项债", "国债", "刺激", "基建", "subsidy"),
        "regulation": ("监管", "处罚", "合规", "反垄断", "停牌", "问询函"),
        "real_estate": ("地产", "房企", "按揭", "土地", "烂尾"),
        "technology": ("算力", "芯片", "人工智能", "AI", "半导体"),
        "banking_finance": ("银行", "券商", "保险", "信贷", "流动性"),
        "risk_event": ("违约", "爆雷", "暴跌", "crash", "default", "panic"),
        "global_macro": ("美元", "美联储", "通胀", "cpi", "ppi", "oil", "geopolitics"),
    }

    POSITIVE_WORDS: Tuple[str, ...] = (
        "利好", "增长", "反弹", "回暖", "上调", "修复", "创新高", "超预期",
        "improve", "beat", "surge", "rally", "bullish",
    )
    NEGATIVE_WORDS: Tuple[str, ...] = (
        "利空", "下滑", "下跌", "违约", "暴雷", "风险", "裁员", "亏损", "处罚",
        "miss", "drop", "crash", "bearish", "default", "downgrade",
    )

    _finbert_pipeline: Any = None
    _bertopic_model: Any = None
    _docs: List[str]

    def __init__(
        self,
        enable_finbert: bool = True,
        enable_bertopic: bool = True,
        max_topic_docs: int = 300,
        min_docs_for_topic_model: int = 24,
    ) -> None:
        self.enable_finbert = enable_finbert
        self.enable_bertopic = enable_bertopic
        self.max_topic_docs = max_topic_docs
        self.min_docs_for_topic_model = min_docs_for_topic_model
        self._docs = []

    def analyze(
        self,
        title: str,
        content: str,
        llm_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = f"{title}\n{content}".strip()
        llm_payload = llm_payload or {}

        topic_signals = self._extract_topics(text, llm_payload)
        sentiment_score, sentiment_components = self._extract_sentiment(text, llm_payload)
        factors = self._compose_financial_factors(topic_signals, sentiment_score, text)
        impact_paths = self._build_impact_paths(topic_signals, llm_payload)

        sentiment_label = "neutral"
        if sentiment_score > 0.15:
            sentiment_label = "positive"
        elif sentiment_score < -0.15:
            sentiment_label = "negative"

        return {
            "topic_signals": [s.to_dict() for s in topic_signals],
            "dominant_topic": topic_signals[0].topic if topic_signals else "uncategorized",
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "sentiment_components": sentiment_components,
            "financial_factors": factors,
            "impact_paths": impact_paths,
        }

    def _extract_topics(self, text: str, llm_payload: Dict[str, Any]) -> List[TopicSignal]:
        by_llm = self._topics_from_llm_payload(llm_payload)
        if by_llm:
            return by_llm

        bertopic_topics = self._topics_from_bertopic(text)
        if bertopic_topics:
            return bertopic_topics

        return self._topics_from_keywords(text)

    def _topics_from_llm_payload(self, payload: Dict[str, Any]) -> List[TopicSignal]:
        signals: List[TopicSignal] = []

        for sector in payload.get("affected_sectors", []) or []:
            if isinstance(sector, str) and sector.strip():
                signals.append(
                    TopicSignal(topic=sector.strip(), score=0.68, source="llm_affected_sectors")
                )

        for ent in payload.get("entities", []) or []:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            etype = str(ent.get("entity_type", "")).strip()
            conf = _clamp(_safe_float(ent.get("confidence", 0.55)), 0.0, 1.0)
            if name:
                topic = f"{etype}:{name}" if etype else name
                signals.append(TopicSignal(topic=topic, score=max(0.45, conf), source="llm_entities"))


        merged: Dict[str, TopicSignal] = {}
        for s in signals:
            prev = merged.get(s.topic)
            if prev is None or s.score > prev.score:
                merged[s.topic] = s

        out = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return out[:6]

    def _topics_from_bertopic(self, text: str) -> List[TopicSignal]:
        if not self.enable_bertopic:
            return []

        self._docs.append(text)
        if len(self._docs) > self.max_topic_docs:
            self._docs = self._docs[-self.max_topic_docs :]

        if len(self._docs) < self.min_docs_for_topic_model:
            return []


        try:
            from bertopic import BERTopic  # type: ignore
        except Exception:
            return []

        try:
            if self._bertopic_model is None:
                self._bertopic_model = BERTopic(verbose=False, calculate_probabilities=False)
                topics, _ = self._bertopic_model.fit_transform(self._docs)
            else:
                topics, _ = self._bertopic_model.transform(self._docs)

            if not topics:
                return []

            topic_id = int(topics[-1])
            if topic_id < 0:
                return []

            topic_words = self._bertopic_model.get_topic(topic_id) or []
            labels = [str(w[0]) for w in topic_words[:3] if isinstance(w, tuple) and w]
            label = " / ".join(labels) if labels else f"topic_{topic_id}"
            return [TopicSignal(topic=label, score=0.72, source="bertopic")]
        except Exception:
            return []

    def _topics_from_keywords(self, text: str) -> List[TopicSignal]:
        lowered = text.lower()
        scored: List[TopicSignal] = []
        for topic, kws in self.TOPIC_KEYWORDS.items():
            hits = 0
            for kw in kws:
                if kw.lower() in lowered:
                    hits += 1
            if hits > 0:

                score = 1.0 - math.exp(-0.6 * hits)
                scored.append(TopicSignal(topic=topic, score=score, source="keyword"))

        scored.sort(key=lambda x: x.score, reverse=True)
        if scored:
            return scored[:5]
        return [TopicSignal(topic="uncategorized", score=0.2, source="fallback")]

    def _extract_sentiment(
        self,
        text: str,
        llm_payload: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}

        llm_score = _safe_float(llm_payload.get("sentiment"), 0.0)
        components["llm"] = _clamp(llm_score, -1.0, 1.0)

        keyword_score = self._keyword_sentiment(text)
        components["keyword"] = keyword_score

        finbert_score = self._finbert_sentiment(text)
        if finbert_score is not None:
            components["finbert"] = finbert_score


        weighted_sum = 0.0
        total_weight = 0.0
        weights = {"llm": 0.45, "keyword": 0.25, "finbert": 0.30}
        for key, value in components.items():
            w = weights.get(key, 0.2)
            weighted_sum += value * w
            total_weight += w

        merged = weighted_sum / total_weight if total_weight > 0 else 0.0
        return (_clamp(merged, -1.0, 1.0), components)

    def _keyword_sentiment(self, text: str) -> float:
        lowered = text.lower()
        pos_hits = sum(1 for w in self.POSITIVE_WORDS if w.lower() in lowered)
        neg_hits = sum(1 for w in self.NEGATIVE_WORDS if w.lower() in lowered)
        if pos_hits == 0 and neg_hits == 0:
            return 0.0
        return _clamp((pos_hits - neg_hits) / max(1.0, (pos_hits + neg_hits)), -1.0, 1.0)

    def _finbert_sentiment(self, text: str) -> Optional[float]:
        if not self.enable_finbert:
            return None

        try:
            if self._finbert_pipeline is None:
                from transformers import pipeline  # type: ignore

                self._finbert_pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                )
            result = self._finbert_pipeline(text[:512])[0]
        except Exception:
            return None

        label = str(result.get("label", "")).lower()
        score = _safe_float(result.get("score"), 0.0)
        if "positive" in label:
            return _clamp(score, 0.0, 1.0)
        if "negative" in label:
            return _clamp(-score, -1.0, 0.0)
        return 0.0

    def _compose_financial_factors(
        self,
        topic_signals: List[TopicSignal],
        sentiment_score: float,
        text: str,
    ) -> Dict[str, Any]:
        topic_map = {s.topic: s.score for s in topic_signals}
        risk_weight = self._sum_topic_weight(topic_map, ("risk_event", "regulation"))
        policy_weight = self._sum_topic_weight(topic_map, ("monetary_policy", "fiscal_policy", "regulation"))
        growth_weight = self._sum_topic_weight(topic_map, ("technology", "banking_finance", "real_estate"))

        neg_hits = sum(1 for w in self.NEGATIVE_WORDS if w.lower() in text.lower())
        pos_hits = sum(1 for w in self.POSITIVE_WORDS if w.lower() in text.lower())
        tone_imbalance = _clamp((neg_hits - pos_hits) / max(1.0, pos_hits + neg_hits), -1.0, 1.0)

        panic_index = _clamp((-sentiment_score * 0.55) + (risk_weight * 0.35) + (max(0.0, tone_imbalance) * 0.20), 0.0, 1.0)
        greed_index = _clamp((sentiment_score * 0.55) + (growth_weight * 0.35) + (max(0.0, -tone_imbalance) * 0.20), 0.0, 1.0)
        policy_shock = _clamp((policy_weight * 0.65) + (abs(sentiment_score) * 0.35), 0.0, 1.0)

        regime_bias = "neutral"
        if panic_index - greed_index > 0.15:
            regime_bias = "risk_off"
        elif greed_index - panic_index > 0.15:
            regime_bias = "risk_on"

        return {
            "panic_index": panic_index,
            "greed_index": greed_index,
            "policy_shock": policy_shock,
            "regime_bias": regime_bias,
        }

    def _sum_topic_weight(self, topic_map: Dict[str, float], candidates: Iterable[str]) -> float:
        total = 0.0
        for name in candidates:
            total += _safe_float(topic_map.get(name, 0.0))
        return _clamp(total, 0.0, 1.0)

    def _build_impact_paths(
        self,
        topic_signals: List[TopicSignal],
        llm_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []
        sectors = [s for s in (llm_payload.get("affected_sectors") or []) if isinstance(s, str) and s.strip()]
        entities = [e for e in (llm_payload.get("entities") or []) if isinstance(e, dict)]

        for t in topic_signals[:3]:
            if sectors:
                for sector in sectors[:3]:
                    paths.append(
                        {
                            "source": t.topic,
                            "relation": "impacts_sector",
                            "target": sector.strip(),
                            "weight": round(_clamp(t.score, 0.1, 1.0), 3),
                        }
                    )
            elif entities:
                for ent in entities[:3]:
                    name = str(ent.get("name", "")).strip()
                    if not name:
                        continue
                    paths.append(
                        {
                            "source": t.topic,
                            "relation": "impacts_entity",
                            "target": name,
                            "weight": round(_clamp(t.score * _safe_float(ent.get("confidence"), 0.7), 0.1, 1.0), 3),
                        }
                    )

        return paths[:12]

