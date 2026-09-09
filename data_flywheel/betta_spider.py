import logging
import os
import time
from typing import List

from config import GLOBAL_CONFIG
from core.model_router import ModelRouter
from data_flywheel.schemas import SeedEvent
from data_flywheel.sources.base_source import BaseSource
from data_flywheel.sources.news_api_source import NewsApiSource
from data_flywheel.sources.rss_source import RssSource
from data_flywheel.nlp_processor import NlpProcessor
from data_flywheel.event_graph_store import EventGraphStore
from data_flywheel.seed_store import SeedStore
from core.runtime_paths import resolve_runtime_path, running_under_pytest

logger = logging.getLogger(__name__)
_RSS_IMPORT_WARNED = False


class BettaSpider:
    """
    BettaFish 数据飞轮 — 主编排引擎

    负责协调各个组件：
    1. 轮询注册的所有数据源获取最新资讯。
    2. 调用 NLP 处理器进行实体提取与情感分析。
    3. 将结果输出并持久化至 SeedStore。
    """

    def __init__(self, config_paths: dict = None):
        """
        初始化飞轮。如果 config_paths 未提供，将使用内置默认配置，并优先加载 GLOBAL_CONFIG
        """
        self.sources: List[BaseSource] = []
        
        # 1. 挂载数据源
        self._init_sources(config_paths)

        # 2. 挂载 NLP 处理器
        api_key = GLOBAL_CONFIG.DEEPSEEK_API_KEY
        zhipu_key = GLOBAL_CONFIG.ZHIPU_API_KEY
        router = ModelRouter(deepseek_key=api_key, zhipu_key=zhipu_key)
        self.nlp = NlpProcessor(model_router=router)

        # 3. 挂载持久化存储
        store_path = resolve_runtime_path("data/seed_events.jsonl", env_var="CIVITAS_SEED_STORE_PATH")
        if config_paths and "output_path" in config_paths:
             store_path = config_paths["output_path"]
        self._reset_test_artifacts_if_needed(store_path)
        self.store = SeedStore(store_path)
        graph_path = resolve_runtime_path("data/event_graph.graphml", env_var="CIVITAS_EVENT_GRAPH_PATH")
        if config_paths and "graph_path" in config_paths:
            graph_path = config_paths["graph_path"]
        self.event_graph = EventGraphStore(graph_path=graph_path)

        logger.info(f"BettaSpider initialized with {len(self.sources)} sources.")

    @staticmethod
    def _reset_test_artifacts_if_needed(path_like) -> None:
        """仅在 pytest 下清理显式测试输出，避免旧样本污染当前断言。"""

        if not running_under_pytest():
            return
        target = str(path_like or "").strip()
        if not target:
            return
        normalized = target.replace("\\", "/").lower()
        if not normalized.startswith("tests/"):
            return
        if os.path.isfile(target):
            os.remove(target)

    def _init_sources(self, config: dict):
        """根据配置初始化数据源列表"""
        global _RSS_IMPORT_WARNED
        if not config:
            config = {
                "enable_mock_source": True,
                "rss_feeds": [
                    "https://rsshub.app/cls/telegraph",
                    "https://rsshub.app/eastmoney/report"
                ]
            }

        # 模拟数据源
        if config.get("enable_mock_source", True):
            self.sources.append(NewsApiSource(use_mock=True))

        # RSS 数据源
        rss_feeds = config.get("rss_feeds", [])
        if rss_feeds:
            try:
                import feedparser
                self.sources.append(RssSource(feed_urls=rss_feeds))
            except ImportError:
                if not _RSS_IMPORT_WARNED:
                    logger.warning("feedparser not installed, skipping RSS sources. Run `pip install feedparser`")
                    _RSS_IMPORT_WARNED = True

    def run_once(self, max_articles: int = 5) -> List[SeedEvent]:
        """
        执行一次完整的采集-清洗-存储流水线
        """
        logger.info("Starting a single run of BettaSpider...")
        all_articles = []

        # 1. 采集
        for source in self.sources:
            try:
                articles = source.fetch(max_items=max_articles)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching from source {source.source_name}: {e}")

        if not all_articles:
            logger.info("No new articles fetched.")
            return []

        logger.info(f"Fetched {len(all_articles)} raw articles. Starting NLP processing...")
        
        # 2. 清洗
        new_events = []
        for article in all_articles:
            event = self.nlp.process(article)
            new_events.append(event)
            try:
                self.event_graph.ingest(event)
            except Exception as e:
                logger.warning("Event graph ingest failed for '%s': %s", event.title, e)
            
        # 3. 持久化
        saved_count = self.store.append_batch(new_events)
        logger.info(f"Run complete. Generated {len(new_events)} seed events, {saved_count} saved to store.")
        
        return new_events

    def run_loop(self, interval_seconds: int = 300, max_articles: int = 5):
        """
        在一个无限循环中运行（作为后台服务）
        """
        logger.info(f"Starting BettaSpider loop with {interval_seconds}s interval...")
        try:
            while True:
                self.run_once(max_articles)
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("BettaSpider loop stopped by user.")

    def push_to_sandbox(self, committee) -> None:
        """
        【集成接口】
        读取最新的种子事件，推送到沙箱的政策委员会 (PolicyCommittee) 中进行解读。
        
        Args:
            committee: PolicyCommittee 实例
        """
        logger.info("Pushing latest seed events to sandbox...")
        events = self.store.read_latest(n=3, min_impact="medium")
        
        if not events:
            logger.info("No significant events found to push.")
            return
            
        for event in events:
            policy_text = event.to_policy_input()
            logger.info(f"Pushing event: {event.title}")
            try:
                 committee.interpret(policy_text)
            except Exception as e:
                 logger.error(f"Sandbox rejected event '{event.title}': {e}")
