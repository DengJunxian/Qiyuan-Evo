"""
数据源抽象基类

所有数据源（RSS、新闻API、社交媒体等）必须继承此基类，
实现统一的 fetch() 接口，返回 RawArticle 列表。
"""

from abc import ABC, abstractmethod
from typing import List

from data_flywheel.schemas import RawArticle


class BaseSource(ABC):
    """
    数据源抽象基类

    子类需实现:
        - source_name: 返回数据源的可读名称
        - fetch(): 执行一次数据抓取，返回原始文章列表
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """数据源名称标识（如 'rss_cls', 'newsapi_eastmoney'）"""
        ...

    @abstractmethod
    def fetch(self, max_items: int = 50) -> List[RawArticle]:
        """
        执行一次数据抓取

        Args:
            max_items: 单次最大抓取条目数

        Returns:
            原始文章列表
        """
        ...
