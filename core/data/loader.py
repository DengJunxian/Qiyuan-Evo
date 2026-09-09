"""
高频市场数据加载与重放模块

使用 akshare 获取 A 股分钟级 K 线数据，
支持流式重放以初始化订单簿背景流动性。

注意事项:
- akshare 的 period 参数需要字符串格式 "1" | "5" | "15" | "30" | "60"
- 需要处理网络超时和 API 限制

作者: Civitas Economica Team
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import List, Optional, AsyncIterator
from datetime import datetime

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None
    print("[警告] akshare 未安装，部分功能不可用")

from tenacity import retry, stop_after_attempt, wait_exponential

from core.exchange.order_book import OrderBook, Order


# ==========================================
# 数据结构
# ==========================================

@dataclass
class Tick:
    """
    分钟级 Tick 数据
    
    Attributes:
        timestamp: 时间戳
        datetime_str: 日期时间字符串
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量
        amount: 成交额
    """
    timestamp: float
    datetime_str: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float = 0.0
    
    @property
    def mid_price(self) -> float:
        """估算中间价"""
        return (self.high + self.low) / 2
    
    @property
    def vwap(self) -> float:
        """估算 VWAP"""
        if self.volume == 0:
            return self.close
        return self.amount / self.volume if self.amount > 0 else self.close


# ==========================================
# 分钟数据加载器
# ==========================================

class MinuteDataLoader:
    """
    A 股分钟级数据加载器
    
    使用 akshare 的 stock_zh_a_hist_min_em 接口获取数据。
    支持 1/5/15/30/60 分钟周期。
    
    CRITICAL: 
    - akshare 的 period 参数需要字符串格式，如 "1" 而非整数
    - 接口有请求频率限制，需要适当延迟
    """
    
    # 支持的周期
    VALID_PERIODS = ["1", "5", "15", "30", "60"]
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def load(
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = "1",
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        加载分钟级 K 线数据
        
        Args:
            symbol: 股票代码 (如 "000001" 代表平安银行)
            start_date: 开始日期 (格式: "YYYY-MM-DD HH:MM:SS" 或 "YYYY-MM-DD")
            end_date: 结束日期
            period: 周期，"1" | "5" | "15" | "30" | "60" 分钟
            adjust: 复权类型，"" 不复权 | "qfq" 前复权 | "hfq" 后复权
            
        Returns:
            DataFrame，包含 datetime, open, high, low, close, volume, amount 列
            
        Raises:
            ValueError: 参数错误
            ConnectionError: 网络错误
        """
        if ak is None:
            raise ImportError("akshare 未安装，请运行: pip install akshare")
        
        # 参数校验
        if period not in MinuteDataLoader.VALID_PERIODS:
            raise ValueError(
                f"无效的周期参数: {period}。"
                f"有效值: {MinuteDataLoader.VALID_PERIODS}"
            )
        
        try:
            print(f"[*] 正在加载 {symbol} 的 {period} 分钟数据...")
            print(f"    时间范围: {start_date} 至 {end_date}")
            
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period=period,
                adjust=adjust
            )
            
            if df is None or df.empty:
                print("[!] 未获取到数据")
                return pd.DataFrame()
            
            # 标准化列名
            df = MinuteDataLoader._standardize_columns(df)
            
            print(f"[OK] 成功加载 {len(df)} 条分钟数据")
            return df
            
        except Exception as e:
            print(f"[!] 数据加载失败: {e}")
            raise
    
    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化 DataFrame 列名
        
        akshare 返回的列名可能是中文或不同格式，
        统一转换为英文列名。
        """
        column_mapping = {
            "时间": "datetime",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "date": "datetime",
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount"
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    @staticmethod
    def load_index_daily(
        symbol: str = "sh000001",
        period_days: int = 365
    ) -> pd.DataFrame:
        """
        加载指数日线数据
        
        用于上证指数等指数的历史数据加载。
        
        Args:
            symbol: 指数代码，如 "sh000001" (上证指数)
            period_days: 加载天数
            
        Returns:
            日线数据 DataFrame
        """
        if ak is None:
            raise ImportError("akshare 未安装")
        
        try:
            print(f"[*] 正在加载 {symbol} 近 {period_days} 天日线数据...")
            
            df = ak.stock_zh_index_daily(symbol=symbol)
            
            if df is None or df.empty:
                print("[!] 未获取到数据")
                return pd.DataFrame()
            
            # 取最近 N 天
            df = df.tail(period_days).reset_index(drop=True)
            
            # 标准化列名
            df = MinuteDataLoader._standardize_columns(df)
            
            print(f"[OK] 成功加载 {len(df)} 个交易日数据")
            return df
            
        except Exception as e:
            print(f"[!] 数据加载失败: {e}")
            raise


# ==========================================
# 市场数据重放器
# ==========================================

class MarketReplay:
    """
    历史市场数据流式重放器
    
    用于：
    1. 逐 tick 重放历史数据，驱动仿真
    2. 初始化订单簿背景流动性（噪声交易者）
    
    Features:
    - 支持异步流式重放
    - 可调节重放速度
    - 自动生成噪声交易者订单
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "A_SHARE_IDX"
    ):
        """
        初始化重放器
        
        Args:
            data: 历史数据 DataFrame，需包含 datetime, open, high, low, close, volume 列
            symbol: 交易标的代码
        """
        self.data = data.copy()
        self.symbol = symbol
        self.current_index = 0
        self._ticks: List[Tick] = []
        
        # 预处理数据
        self._preprocess()
    
    def _preprocess(self) -> None:
        """预处理数据，转换为 Tick 对象列表"""
        self._ticks = []
        
        for idx, row in self.data.iterrows():
            # 解析时间
            dt_str = str(row.get("datetime", row.get("date", "")))
            try:
                if " " in dt_str:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                else:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d")
                ts = dt.timestamp()
            except Exception:
                ts = time.time()
            
            tick = Tick(
                timestamp=ts,
                datetime_str=dt_str,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=int(row.get("volume", 0)),
                amount=float(row.get("amount", 0))
            )
            self._ticks.append(tick)
    
    def __len__(self) -> int:
        return len(self._ticks)
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self) -> Tick:
        if self.current_index >= len(self._ticks):
            raise StopIteration
        tick = self._ticks[self.current_index]
        self.current_index += 1
        return tick
    
    async def stream_ticks(
        self,
        delay: float = 0.0
    ) -> AsyncIterator[Tick]:
        """
        异步流式重放 ticks
        
        Args:
            delay: 每个 tick 之间的延迟（秒），0 表示无延迟
            
        Yields:
            Tick 对象
        """
        for tick in self._ticks:
            yield tick
            if delay > 0:
                await asyncio.sleep(delay)
    
    def reset(self) -> None:
        """重置重放位置"""
        self.current_index = 0
    
    # ------------------------------------------
    # 流动性注入
    # ------------------------------------------
    
    def inject_liquidity(
        self,
        order_book: OrderBook,
        tick: Optional[Tick] = None,
        num_levels: int = 5,
        qty_per_level: int = 100,
        spread_bps: float = 10.0
    ) -> List[Order]:
        """
        注入背景流动性（噪声交易者订单）
        
        根据历史 tick 数据或当前价格，在订单簿中生成
        多个价位的挂单，模拟真实市场的流动性。
        
        Args:
            order_book: 目标订单簿
            tick: 参考 tick 数据（用于确定价格中心）
            num_levels: 每侧档位数量
            qty_per_level: 每档基础挂单量
            spread_bps: 价差基点 (1 bps = 0.01%)
            
        Returns:
            生成的订单列表
        """
        # 确定参考价格
        if tick is not None:
            ref_price = tick.close
        else:
            ref_price = order_book.last_price or order_book.prev_close
        
        orders: List[Order] = []
        spread = ref_price * spread_bps / 10000  # 转换为价格
        
        # 生成卖单 (高于参考价)
        for i in range(num_levels):
            price = round(ref_price + spread * (i + 1), 2)
            # 随机化数量
            qty = int(qty_per_level * (0.8 + random.random() * 0.4))
            
            order = Order.create(
                agent_id=f"noise_ask_{i}",
                symbol=order_book.symbol,
                side="sell",
                order_type="limit",
                price=price,
                quantity=qty
            )
            orders.append(order)
            order_book.add_order(order)
        
        # 生成买单 (低于参考价)
        for i in range(num_levels):
            price = round(ref_price - spread * (i + 1), 2)
            qty = int(qty_per_level * (0.8 + random.random() * 0.4))
            
            order = Order.create(
                agent_id=f"noise_bid_{i}",
                symbol=order_book.symbol,
                side="buy",
                order_type="limit",
                price=price,
                quantity=qty
            )
            orders.append(order)
            order_book.add_order(order)
        
        return orders
    
    def generate_noise_orders(
        self,
        tick: Tick,
        num_orders: int = 5,
        max_qty: int = 100
    ) -> List[Order]:
        """
        根据 tick 生成噪声交易者订单
        
        用于在仿真过程中持续注入流动性。
        
        Args:
            tick: 当前 tick 数据
            num_orders: 生成订单数量
            max_qty: 最大单笔数量
            
        Returns:
            订单列表
        """
        orders: List[Order] = []
        
        for i in range(num_orders):
            # 随机买卖方向
            side = "buy" if random.random() > 0.5 else "sell"
            
            # 价格在 tick 的 high/low 范围内随机
            if side == "buy":
                # 买单价格在 low 附近
                price = round(
                    tick.low + random.random() * (tick.mid_price - tick.low),
                    2
                )
            else:
                # 卖单价格在 high 附近
                price = round(
                    tick.mid_price + random.random() * (tick.high - tick.mid_price),
                    2
                )
            
            qty = random.randint(10, max_qty)
            
            order = Order.create(
                agent_id=f"noise_trader_{i}_{int(time.time() * 1000) % 10000}",
                symbol=self.symbol,
                side=side,
                order_type="limit",
                price=price,
                quantity=qty
            )
            orders.append(order)
        
        return orders


# ==========================================
# 便捷函数
# ==========================================

def load_shanghai_index(
    days: int = 365
) -> pd.DataFrame:
    """
    加载上证指数历史数据
    
    Args:
        days: 加载天数
        
    Returns:
        日线数据 DataFrame
    """
    return MinuteDataLoader.load_index_daily(
        symbol="sh000001",
        period_days=days
    )


def create_replay_from_index(
    days: int = 30
) -> MarketReplay:
    """
    从上证指数数据创建重放器
    
    Args:
        days: 加载天数
        
    Returns:
        MarketReplay 实例
    """
    data = load_shanghai_index(days)
    return MarketReplay(data, symbol="SH000001")


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("市场数据加载与重放测试")
    print("=" * 60)
    
    # 注意：实际运行需要网络连接
    try:
        # 1. 测试日线数据加载
        print("\n[测试 1] 加载上证指数日线数据...")
        df = load_shanghai_index(days=10)
        if not df.empty:
            print(f"数据形状: {df.shape}")
            print(df.tail(3))
        
        # 2. 测试重放器
        print("\n[测试 2] 创建重放器...")
        replay = MarketReplay(df, symbol="SH000001")
        print(f"总 ticks: {len(replay)}")
        
        # 3. 测试流动性注入
        print("\n[测试 3] 注入初始流动性...")
        from core.exchange.order_book import OrderBook
        
        ob = OrderBook(symbol="SH000001", prev_close=3000.0)
        
        for tick in replay:
            orders = replay.inject_liquidity(ob, tick, num_levels=3)
            print(f"  {tick.datetime_str}: 注入 {len(orders)} 个订单")
            print(f"  盘口: {ob.get_depth(2)}")
            break  # 只测试第一个
        
    except Exception as e:
        print(f"\n[!] 测试失败（可能需要网络连接）: {e}")
        
        # 使用模拟数据测试
        print("\n[备选] 使用模拟数据测试...")
        mock_data = pd.DataFrame({
            "datetime": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "open": [3000, 3010, 3005],
            "high": [3020, 3025, 3015],
            "low": [2990, 3000, 2995],
            "close": [3010, 3005, 3008],
            "volume": [1000000, 1200000, 950000]
        })
        
        replay = MarketReplay(mock_data, symbol="SH000001")
        print(f"模拟数据 ticks: {len(replay)}")
        
        for tick in replay:
            print(f"  {tick.datetime_str}: O={tick.open}, H={tick.high}, L={tick.low}, C={tick.close}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
