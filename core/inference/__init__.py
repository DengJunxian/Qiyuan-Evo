"""
推理引擎模块

支持多种部署模式:
- Lite: 100% 云端 API
- Standard: 1.5B 本地 + API
- Enterprise: 完整 vLLM 后端
"""

from core.inference.config import InferenceConfig, InferenceMode
from core.inference.tiered_router import TieredRouter

__all__ = ["InferenceConfig", "InferenceMode", "TieredRouter"]
