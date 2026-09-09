import logging
from typing import List

from data_flywheel.schemas import RawArticle
from data_flywheel.sources.base_source import BaseSource

logger = logging.getLogger(__name__)


class NewsApiSource(BaseSource):
    """
    新闻 API 适配器

    真实环境可通过 httpx 调用如财联社/东方财富的开放 API。
    本实现提供一个模拟模式（Mock Mode），内置预定义的财经异动样本，
    用于离线开发、测试和沙箱演示。
    """

    # 内置的模拟新闻样本
    MOCK_ARTICLES = [
        {
            "title": "央行：下调金融机构存款准备金率0.5个百分点",
            "content": "为释放长期流动性，中国人民银行决定于近期下调金融机构存款准备金率0.5个百分点（不含已执行5%存款准备金率的金融机构）。本次下调后，金融机构加权平均存款准备金率约为6.6%。此举预计将向市场释放长期资金约1万亿元，有助于降低实体经济融资成本，支持稳增长。",
            "source": "财联社电报",
        },
        {
            "title": "国家新闻出版署发布新规征求意见稿，网络游戏板块重挫",
            "content": "国家新闻出版署草拟了《网络游戏管理办法（草案征求意见稿）》，提出限制游戏过度使用和高额消费等条款。消息一出，港股及A股网络游戏板块全线闪崩，多家头部游戏公司股价跌幅超过10%，引发市场对传媒及互联网板块盈利前景的担忧。",
            "source": "东方财富网",
        },
        {
            "title": "重磅利好！房地产调控优化，多地取消限购",
            "content": "多部委联合发文，优化房地产市场平稳健康发展政策，明确提出'认房不认贷'。随即，一线及强二线城市相继宣布全面或部分取消限购限售政策。市场普遍认为，此举将极大提振购房者信心，房地产开发及相关建材、家居板块有望迎来估值修复。",
            "source": "同花顺财经",
        },
        {
            "title": "上交所：对某量化私募机构采取限制交易措施",
            "content": "上海证券交易所发布公告，某量化私募机构在开盘后短短1分钟内，通过计算机程序自动生成并下达大量市价卖出订单，导致多只宽基ETF价格瞬间大幅异常波动，严重影响市场正常交易秩序。上交所决定对该机构名下账户采取限制交易三个月的纪律处分，并启动公开谴责程序。",
            "source": "证监会发布",
        },
        {
            "title": "美国非农就业数据远超预期，美联储降息预期大幅降温",
            "content": "美国劳工部最新公布的数据显示，上月非农新增就业人数达到35.3万人，几乎是市场预期的两倍。同时失业率维持在3.7%的低位。强劲的劳动力市场数据直接打压了市场对美联储近期降息的预期，美债收益率飙升，美元指数走强，可能对亚太新兴市场股市造成流动性虹吸效应。",
            "source": "新浪财经",
        }
    ]

    def __init__(self, api_key: str = "", use_mock: bool = True, name: str = "NewsAPI"):
        self._name = name
        self.api_key = api_key
        self.use_mock = use_mock
        self._mock_index = 0

    @property
    def source_name(self) -> str:
        return f"{self._name}{'(Mock)' if self.use_mock else ''}"

    def fetch(self, max_items: int = 50) -> List[RawArticle]:
        """抓取新闻（支持模拟模式和真实API模式）"""
        if self.use_mock:
            return self._fetch_mock(max_items)
        else:
            return self._fetch_real(max_items)

    def _fetch_mock(self, max_items: int) -> List[RawArticle]:
        """从预定义的样本中按顺序“吐出”新闻"""
        articles = []
        

        remaining = len(self.MOCK_ARTICLES) - self._mock_index
        if remaining <= 0:
            logger.info(f"[{self.source_name}] No more mock articles available.")
            return articles
            
        count = min(max_items, remaining)
        
        for i in range(count):
            item = self.MOCK_ARTICLES[self._mock_index]
            article = RawArticle(
                title=item["title"],
                content=item["content"],
                source=item["source"],
                source_url=f"mock://news/{self._mock_index}"
            )
            articles.append(article)
            self._mock_index += 1
            
        logger.info(f"[{self.source_name}] Fetched {len(articles)} mock articles.")
        return articles

    def _fetch_real(self, max_items: int) -> List[RawArticle]:
        """真实API调用（AkShare接入）"""
        articles = []
        try:
            import akshare as ak
            logger.info(f"[{self.source_name}] Requesting real data from AkShare (Sina Global News)...")

            df = ak.stock_info_global_sina()
            
            if df is None or df.empty:
                logger.warning(f"[{self.source_name}] Retrieved empty dataframe from AkShare.")
                return articles
                
            # '时间' typically exists but sometimes indices vary. We'll use iloc or column names
            # Sina typically returns ['时间', '内容']
            columns = df.columns.tolist()
            content_col = '内容' if '内容' in columns else columns[1]
            time_col = '时间' if '时间' in columns else columns[0]
            
            count = min(max_items, len(df))
            for i in range(count):
                row = df.iloc[i]
                content = str(row[content_col])
                

                title = content[:30] + "..." if len(content) > 30 else content
                
                article = RawArticle(
                    title=title,
                    content=content,
                    source=f"新浪财经({row[time_col]})",
                    source_url=f"akshare://sina_global/{i}"
                )
                articles.append(article)
                
            logger.info(f"[{self.source_name}] Fetched {len(articles)} real articles from AkShare.")
            
        except Exception as e:
            logger.error(f"[{self.source_name}] Failed to fetch from AkShare: {e}")
            
        return articles
