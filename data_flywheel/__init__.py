"""
BettaFish 数据飞轮 — 分布式情报摄入模块

负责从外部信息源持续采集财经异动文本，
经 LLM 驱动的清洗管道（实体提取 + 情感极性判断）后，
输出标准化的"种子事件流"JSON，直接对接 PolicyCommittee。
"""

from data_flywheel.schemas import SeedEvent, RawArticle, ExtractedEntity
from data_flywheel.nlp_processor import NlpProcessor
from data_flywheel.betta_spider import BettaSpider

__all__ = [
    "SeedEvent",
    "RawArticle",
    "ExtractedEntity",
    "NlpProcessor",
    "BettaSpider",
]
