"""
数据源适配层

提供统一的数据源接口，支持 RSS 订阅与新闻 API 等多种信息渠道。
"""

from data_flywheel.sources.base_source import BaseSource
from data_flywheel.sources.rss_source import RssSource
from data_flywheel.sources.news_api_source import NewsApiSource

__all__ = ["BaseSource", "RssSource", "NewsApiSource"]
