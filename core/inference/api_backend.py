"""
API 推理后端

使用 DeepSeek API 进行推理。
"""

import os
import time
from types import ModuleType
from typing import Optional, List, Any, cast

from config import DEFAULT_DEEPSEEK_API_KEY, DEFAULT_ZHIPU_API_KEY
from core.runtime_mode import llm_mode_for_model
from core.llm.router import sync_llm_complete

_openai_module: Optional[ModuleType]
try:
    import openai as _openai_module
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    _openai_module = None

_FALLBACK_BACKOFF_UNTIL = 0.0


def _fallback_backoff_seconds() -> float:
    raw = str(os.getenv("CIVITAS_API_FALLBACK_BACKOFF_SECONDS", "")).strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 180.0


def _fallback_backoff_active() -> bool:
    return time.monotonic() < _FALLBACK_BACKOFF_UNTIL


def _mark_fallback_backoff() -> None:
    global _FALLBACK_BACKOFF_UNTIL
    _FALLBACK_BACKOFF_UNTIL = max(
        _FALLBACK_BACKOFF_UNTIL,
        time.monotonic() + _fallback_backoff_seconds(),
    )


class APIBackend:
    """
    DeepSeek API 推理后端
    
    使用 OpenAI 兼容的 API 进行推理。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-reasoner",
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        self.model = model
        self.api_key = api_key or self._default_api_key_for_model(model)
        self.base_url = base_url or self._default_base_url_for_model(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._client: Optional[Any] = None

    @staticmethod
    def _default_api_key_for_model(model: str) -> str:
        name = str(model or "").lower()
        if name.startswith("glm-"):
            return (
                os.getenv("ZHIPUAI_API_KEY", "").strip()
                or os.getenv("ZHIPU_API_KEY", "").strip()
                or DEFAULT_ZHIPU_API_KEY
            )
        return os.getenv("DEEPSEEK_API_KEY") or DEFAULT_DEEPSEEK_API_KEY

    @staticmethod
    def _default_base_url_for_model(model: str) -> str:
        name = str(model or "").lower()
        if name.startswith("glm-"):
            return os.getenv("ZHIPU_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()
        return os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1").strip()

    def _mode_for_model(self) -> str:
        return llm_mode_for_model(self.model)
        
    def _get_client(self) -> Any:
        if self._client is None:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai 库未安装，请运行: pip install openai")
            if not self.api_key:
                raise ValueError("模型 API Key 未设置")
            module: Optional[ModuleType] = cast(Optional[ModuleType], _openai_module)
            if module is None:
                raise ImportError("openai 客户端不可用")
            self._client = module.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        fallback = kwargs.get("fallback_response")

        if fallback is not None and _fallback_backoff_active():
            return str(fallback)

        try:
            routed = sync_llm_complete(
                messages,
                mode=kwargs.get("mode") or self._mode_for_model(),
                task_type=kwargs.get("task_type"),
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=kwargs.get("timeout_budget", kwargs.get("timeout")),
                fallback_response=fallback,
            )
            if routed.ok or fallback is not None:
                if not routed.ok and str(routed.provider).lower() == "offline":
                    _mark_fallback_backoff()
                return routed.text
        except Exception:
            if fallback is not None:
                _mark_fallback_backoff()
                return str(fallback)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if fallback is not None:
                _mark_fallback_backoff()
                return str(fallback)
            return f"[API Error] {e}"
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """批量生成 (串行，未来可优化为并行)"""
        return [self.generate(p, system_prompt) for p in prompts]
