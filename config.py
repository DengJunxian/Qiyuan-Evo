import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_ZHIPU_API_KEY = ""
DEFAULT_DEEPSEEK_API_KEY = ""


def _load_local_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env without overriding shell env."""

    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


_load_local_env_file()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class SimConfig:
    """
    Civitas Economica 全局仿真配置
    
    采用不可变数据类 (frozen=True) 防止运行时意外修改核心参数。
    """
    
    # --- 市场微观结构参数 ---
    
    # 印花税率
    # 2023年8月28日起，A股证券交易印花税实施减半征收，由0.1%降至0.05%。
    # 注意：仅向卖方征收。
    TAX_RATE_STAMP: float = 0.0005 
    
    # 交易佣金
    # 券商收取的净佣金，通常在 0.01% - 0.03% 之间，双向收取。
    TAX_RATE_COMMISSION: float = 0.00025
    
    # 涨跌停限制
    # 主板一般为 10%，科创板/创业板为 20%。此处默认为主板设定。
    PRICE_LIMIT: float = 0.10
    
    # T+1 交易制度
    # 为真时，当日买入的股票次日才能卖出。
    # 为假时，允许 T+0 回转交易。
    T_PLUS_1: bool = True
    
    # --- 仿真环境配置 ---
    
    PROJECT_NAME: str = "Civitas A-Share: 数治观澜"
    VERSION: str = "0.2.0-Alpha"
    
    # 基础货币 (用于未指定时的默认值)
    DEFAULT_CASH: float = 100_000.0
    
    # --- 外部接口配置 ---
    

    API_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_API_KEY: Optional[str] = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", DEFAULT_DEEPSEEK_API_KEY).strip()
    )
    

    # 智谱 GLM 接口，快速模式专用
    ZHIPU_API_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"
    ZHIPU_API_KEY: Optional[str] = field(
        default_factory=lambda: (
            os.environ.get("ZHIPUAI_API_KEY", "").strip()
            or os.environ.get("ZHIPU_API_KEY", "").strip()
            or DEFAULT_ZHIPU_API_KEY
        )
    )
    LLM_DEFAULT_PROVIDER: str = field(
        default_factory=lambda: os.environ.get("LLM_DEFAULT_PROVIDER", "auto")
    )
    LLM_TIMEOUT_SECONDS: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 20.0)
    )
    LLM_MAX_RETRIES: int = field(
        default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2)
    )
    
    # --- 模型配置 ---
    
    # DeepSeek 模型
    MODEL_DEEPSEEK_REASONER: str = "deepseek-reasoner"
    MODEL_DEEPSEEK_CHAT: str = "deepseek-chat"
    MODEL_DEEPSEEK_V4_PRO: str = "deepseek-v4-pro"
    MODEL_DEEPSEEK_V4_FLASH: str = "deepseek-v4-flash"
    

    # 智谱 GLM 模型，快速模式使用
    MODEL_ZHIPU_FLASHX: str = "glm-4-flashx"
    MODEL_ZHIPU_FLASHX_250414: str = "glm-4-flashx-250414"
    
    # 超时设置 (秒)
    API_TIMEOUT_REASONER: float = 30.0  # 推理模型超时
    API_TIMEOUT_CHAT: float = 15.0  # 对话模型超时
    API_TIMEOUT_FLASH: float = 15.0  # 快速模型超时
    
    # --- 仿真模式配置 ---
    
    # 默认仿真模式：智能、快速、深度
    DEFAULT_SIMULATION_MODE: str = "SMART"
    
    # 各模式时间预算（秒/天）
    TIME_BUDGET_SMART: float = 9999.0
    TIME_BUDGET_FAST: float = 9999.0
    TIME_BUDGET_DEEP: float = 9999.0
    
    # 每步深度思考的最大智能体数
    MAX_DEEP_AGENTS_SMART: int = 3
    MAX_DEEP_AGENTS_FAST: int = 2
    MAX_DEEP_AGENTS_DEEP: int = 5
    
    # --- 天数循环配置 ---
    
    # 仿真天数循环阈值，每 N 天自动暂停
    SIMULATION_DAY_CYCLE: int = 100

# 实例化全局配置对象
GLOBAL_CONFIG = SimConfig()
