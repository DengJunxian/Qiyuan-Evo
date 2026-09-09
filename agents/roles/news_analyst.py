"""
NewsAnalyst —— 新闻情报分析师。

论文思想映射（FINCON / QuantAgents）：
1) 角色解耦：将“新闻获取 + 情感提取”从交易主流程中剥离，形成独立专家角色。
2) 结构化输出：以 JSON 报告返回，供 ManagerAgent 统一融合决策。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from data_flywheel.betta_spider import BettaSpider
from data_flywheel.schemas import SeedEvent

logger = logging.getLogger(__name__)


class NewsAnalyst:
    """
    新闻情报分析师：复用 BettaSpider 流水线（抓取 + NLP + SeedEvent）。
    输出 JSON 结构化报告，支持容错与降级。
    """

    def __init__(self, config_paths: Optional[Dict[str, Any]] = None, max_articles: int = 5):
        """
        Args:
            config_paths: BettaSpider 数据源配置（可选）
            max_articles: 单次抓取的新闻数量上限
        """
        self._offline = bool((config_paths or {}).get("offline", False))
        self._spider = None if self._offline else BettaSpider(config_paths=config_paths)
        self._max_articles = max(1, int(max_articles))

    def analyze(self) -> Dict[str, Any]:
        """
        运行一次 BettaSpider 并汇总情感结果。

        Returns:
            JSON 风格字典，包含事件列表、情感均值与风险影响等级。
        """
        if self._offline:
            result = self._summarize_events([])
            result["source"] = "offline_no_external_news"
            return result
        try:
            events = self._spider.run_once(max_articles=self._max_articles)
        except Exception as exc:
            logger.warning("NewsAnalyst failed to fetch/process news: %s", exc)
            return {
                "analyst": "NewsAnalyst",
                "status": "error",
                "error": str(exc),
                "events": [],
                "sentiment_score": 0.0,
                "sentiment_label": "中性",
                "impact_level": "low",
                "source": "BettaSpider",
            }

        return self._summarize_events(events)

    def _summarize_events(self, events: List[SeedEvent]) -> Dict[str, Any]:
        if not events:
            return {
                "analyst": "NewsAnalyst",
                "status": "ok",
                "events": [],
                "sentiment_score": 0.0,
                "sentiment_label": "中性",
                "impact_level": "low",
                "source": "BettaSpider",
            }

        sentiments = [float(e.sentiment) for e in events]
        avg_sentiment = sum(sentiments) / max(1, len(sentiments))
        sentiment_label = self._label_sentiment(avg_sentiment)
        impact_level = self._max_impact_level([e.impact_level for e in events])
        text_topics: List[str] = []
        panic_scores: List[float] = []
        greed_scores: List[float] = []
        shock_scores: List[float] = []
        for event in events:
            factors = getattr(event, "text_factors", {}) or {}
            topic = factors.get("dominant_topic")
            if isinstance(topic, str) and topic:
                text_topics.append(topic)
            financial = factors.get("financial_factors", {}) or {}
            if isinstance(financial, dict):
                panic_scores.append(float(financial.get("panic_index", 0.0)))
                greed_scores.append(float(financial.get("greed_index", 0.0)))
                shock_scores.append(float(financial.get("policy_shock", 0.0)))

        dominant_topic = max(set(text_topics), key=text_topics.count) if text_topics else "uncategorized"
        avg_panic = sum(panic_scores) / max(1, len(panic_scores))
        avg_greed = sum(greed_scores) / max(1, len(greed_scores))
        avg_shock = sum(shock_scores) / max(1, len(shock_scores))

        return {
            "analyst": "NewsAnalyst",
            "status": "ok",
            "events": [e.to_dict() for e in events],
            "sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label,
            "impact_level": impact_level,
            "text_factors": {
                "dominant_topic": dominant_topic,
                "panic_index": avg_panic,
                "greed_index": avg_greed,
                "policy_shock": avg_shock,
            },
            "source": "BettaSpider",
        }

    @staticmethod
    def _label_sentiment(score: float) -> str:
        if score > 0.1:
            return "利好"
        if score < -0.1:
            return "利空"
        return "中性"

    @staticmethod
    def _max_impact_level(levels: List[str]) -> str:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        best = "low"
        for level in levels:
            if level in order and order[level] > order[best]:
                best = level
        return best
