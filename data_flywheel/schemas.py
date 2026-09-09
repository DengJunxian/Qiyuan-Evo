"""
BettaFish 数据飞轮 — 统一数据模型

定义从"原始文章抓取"到"种子事件输出"全链路的数据结构。
所有结构均可序列化为 JSON，便于持久化和跨进程传输。
"""

import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class RawArticle:
    """
    原始文章 — 从数据源抓取的未经处理的文本

    Attributes:
        title: 文章标题
        content: 正文内容（可能为截断摘要）
        source: 来源名称（如"财联社电报"、"东方财富研报"）
        source_url: 原文链接
        published_at: 发布时间（ISO 8601 字符串）
        fetched_at: 抓取时间
    """
    title: str
    content: str
    source: str
    source_url: str = ""
    published_at: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedEntity:
    """
    提取的实体

    Attributes:
        name: 实体名称（如"中国人民银行"、"房地产"、"LPR"）
        entity_type: 实体类别
            - "company"   公司/机构
            - "sector"    行业板块
            - "indicator" 经济指标
            - "policy"    政策工具
            - "person"    人物
        confidence: LLM 给出的置信度 (0.0 ~ 1.0)
    """
    name: str
    entity_type: str
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SeedEvent:
    """
    种子事件 — BettaFish 数据飞轮的标准输出格式

    这是整个情报摄入管道的最终产物，可直接推入
    PolicyCommittee.interpret() 进行政策解读。

    Attributes:
        event_id: 事件唯一标识 (UUID)
        source: 信息来源
        source_url: 原文链接
        title: 事件标题
        summary: 经 LLM 生成的简要摘要
        entities: 提取的实体列表
        sentiment: 情感极性分数 (-1.0 极度利空 ~ 1.0 极度利好)
        sentiment_label: 情感标签 ("利好" / "利空" / "中性")
        impact_level: 影响等级 ("low" / "medium" / "high" / "critical")
        affected_sectors: 受影响的行业板块列表
        raw_text: 原始文本
        created_at: 原始发布时间
        processed_at: 处理完成时间
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    source_url: str = ""
    title: str = ""
    summary: str = ""
    entities: List[ExtractedEntity] = field(default_factory=list)
    sentiment: float = 0.0
    sentiment_label: str = "中性"
    impact_level: str = "low"
    affected_sectors: List[str] = field(default_factory=list)
    raw_text: str = ""
    created_at: str = ""
    processed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


    text_factors: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    #  序列化 / 反序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典"""
        data = asdict(self)
        return data

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedEvent":
        """从字典恢复 SeedEvent"""
        payload = dict(data)
        # 处理嵌套的 entities 列表
        entities_raw = payload.pop("entities", [])
        entities = [
            ExtractedEntity(**e) if isinstance(e, dict) else e
            for e in entities_raw
        ]
        text_factors = payload.get("text_factors")
        if text_factors is None or not isinstance(text_factors, dict):
            payload["text_factors"] = {}
        return cls(entities=entities, **payload)

    @classmethod
    def from_json(cls, json_str: str) -> "SeedEvent":
        """从 JSON 字符串恢复 SeedEvent"""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    #  对接 PolicyCommittee
    # ------------------------------------------------------------------

    def to_policy_input(self) -> str:
        """
        将种子事件转换为 PolicyCommittee.interpret() 可接收的政策文本

        输出格式示例:
            【财经快讯 | 来源: 财联社电报 | 情感: 利好(0.72) | 影响: high】
            央行宣布降准50个基点，释放长期资金约1万亿元。
            涉及实体: 中国人民银行(政策工具), 银行业(行业板块)
            影响板块: 银行, 房地产, 基建

        Returns:
            格式化的政策文本字符串
        """
        # 构建实体描述
        entity_strs = [
            f"{e.name}({e.entity_type})" for e in self.entities
        ]
        entity_line = ", ".join(entity_strs) if entity_strs else "未识别"

        # 构建板块描述
        sector_line = ", ".join(self.affected_sectors) if self.affected_sectors else "未明确"

        factor_line = "文本因子: N/A"
        if self.text_factors:
            dominant_topic = self.text_factors.get("dominant_topic", "unknown")
            financial = self.text_factors.get("financial_factors", {}) or {}
            panic = float(financial.get("panic_index", 0.0))
            greed = float(financial.get("greed_index", 0.0))
            shock = float(financial.get("policy_shock", 0.0))
            factor_line = (
                f"文本因子: topic={dominant_topic}, "
                f"panic={panic:.2f}, greed={greed:.2f}, shock={shock:.2f}"
            )

        policy_text = (
            f"【财经快讯 | 来源: {self.source} | "
            f"情感: {self.sentiment_label}({self.sentiment:.2f}) | "
            f"影响: {self.impact_level}】\n"
            f"{self.title}\n"
            f"{self.summary}\n"
            f"涉及实体: {entity_line}\n"
            f"影响板块: {sector_line}\n"
            f"{factor_line}"
        )
        return policy_text
