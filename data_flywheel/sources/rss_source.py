import time
import logging
from typing import List, Set
from datetime import datetime, timezone

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None

from data_flywheel.schemas import RawArticle
from data_flywheel.sources.base_source import BaseSource

logger = logging.getLogger(__name__)


class RssSource(BaseSource):
    """
    RSS/Atom 订阅源适配器

    支持多个 Feed URL 轮询抓取，并基于文章 URL 进行基础去重。
    """

    def __init__(self, feed_urls: List[str], name: str = "RSS_Poller"):
        if not FEEDPARSER_AVAILABLE:
            raise ImportError(
                "feedparser 库未安装，请运行: pip install feedparser "
                "来支持 RSS 数据飞轮。"
            )

        self._name = name
        self.feed_urls = feed_urls
        self.seen_urls: Set[str] = set()

    @property
    def source_name(self) -> str:
        return self._name

    def fetch(self, max_items: int = 50) -> List[RawArticle]:
        """从所有配置的 RSS 源中抓取最新文章"""
        articles: List[RawArticle] = []
        
        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                

                if feed.bozo and hasattr(feed, 'bozo_exception'):
                    logger.warning(f"Error parsing feed {url}: {feed.bozo_exception}")
                    continue

                feed_title = feed.feed.get('title', 'Unknown RSS Feed')
                
                for entry in feed.entries:
                    link = entry.get('link', '')
                    

                    if link in self.seen_urls:
                        continue
                        
                    title = entry.get('title', '')

                    content = entry.get('description', entry.get('summary', ''))
                    

                    published_at = ""
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:

                        dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
                        published_at = dt.isoformat()
                    else:
                        published_at = entry.get('published', '')

                    article = RawArticle(
                        title=title,
                        content=content,
                        source=f"{self._name} ({feed_title})",
                        source_url=link,
                        published_at=published_at
                    )
                    
                    self.seen_urls.add(link)
                    articles.append(article)
                    
                    if len(articles) >= max_items:
                        break

            except Exception as e:
                logger.error(f"Failed to fetch RSS feed {url}: {e}")
                
            if len(articles) >= max_items:
                break
                

        if len(self.seen_urls) > 5000:

            self.seen_urls = set(list(self.seen_urls)[-2000:])
            
        return articles
