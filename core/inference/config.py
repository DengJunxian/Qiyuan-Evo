"""
推理配置模块

定义推理模式和全局配置。
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from config import DEFAULT_DEEPSEEK_API_KEY


class InferenceMode(str, Enum):
    """
    推理模式
    
    - LITE: 纯 API 模式，无需本地 GPU
    - STANDARD: 1.5B 本地模型 + API (适合 8GB VRAM)
    - ENTERPRISE: 完整 vLLM 后端 (需要 24GB+ VRAM)
    """
    LITE = "lite"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"


class AgentTier(str, Enum):
    """
    Agent 推理层级
    
    - TIER_1: 本地推理 (快速、低成本)
    - TIER_2: 云端 API (高质量、高成本)
    - TIER_RULE: 纯规则引擎 (无 LLM)
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_RULE = "tier_rule"


@dataclass
class InferenceConfig:
    """
    推理配置
    
    Attributes:
        mode: 推理模式 (lite/standard/enterprise)
        api_key: DeepSeek API Key
        api_base_url: API 端点
        local_model_path: 本地模型路径 (GGUF 格式)
        vllm_model_name: vLLM 模型名称
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        tier_2_threshold: 触发 Tier 2 的恐慌阈值
    """
    mode: InferenceMode = InferenceMode.LITE
    
    # 接口 配置
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY") or DEFAULT_DEEPSEEK_API_KEY)
    api_base_url: str = "https://api.deepseek.com/v1"
    
    # 本地模型配置 (Standard 模式)
    local_model_path: Optional[str] = None
    n_ctx: int = 2048  # 上下文长度
    n_gpu_layers: int = -1  # -1 = 全部 offload 到 GPU
    
    # vLLM 配置，企业模式使用
    vllm_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_quantization: Optional[str] = "awq"
    
    # 生成参数
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 路由策略
    tier_2_threshold: float = 0.8  # panic_level > 0.8 触发 第二层
    institutional_always_tier_2: bool = True  # 机构始终用 第二层
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "api_base_url": self.api_base_url,
            "local_model_path": self.local_model_path,
            "vllm_model_name": self.vllm_model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """从环境变量加载配置"""
        mode_str = os.getenv("CIVITAS_INFERENCE_MODE", "lite").lower()
        mode = InferenceMode(mode_str) if mode_str in [m.value for m in InferenceMode] else InferenceMode.LITE
        
        return cls(
            mode=mode,
            api_key=os.getenv("DEEPSEEK_API_KEY") or DEFAULT_DEEPSEEK_API_KEY,
            local_model_path=os.getenv("CIVITAS_LOCAL_MODEL_PATH"),
            vllm_model_name=os.getenv("CIVITAS_VLLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        )
    
    @classmethod
    def lite(cls) -> "InferenceConfig":
        """创建 Lite 模式配置"""
        return cls(mode=InferenceMode.LITE)
    
    @classmethod
    def standard(cls, model_path: str) -> "InferenceConfig":
        """创建 Standard 模式配置"""
        return cls(
            mode=InferenceMode.STANDARD,
            local_model_path=model_path
        )
    
    @classmethod
    def enterprise(cls, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") -> "InferenceConfig":
        """创建 Enterprise 模式配置"""
        return cls(
            mode=InferenceMode.ENTERPRISE,
            vllm_model_name=model_name
        )


# 默认配置
DEFAULT_CONFIG = InferenceConfig.from_env()
