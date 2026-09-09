"""Environment-backed LLM configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config import DEFAULT_DEEPSEEK_API_KEY, DEFAULT_ZHIPU_API_KEY


def _load_local_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from local .env without overriding shell env."""

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
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True, slots=True)
class LLMSettings:
    deepseek_api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "").strip() or DEFAULT_DEEPSEEK_API_KEY
    )
    zhipu_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("ZHIPUAI_API_KEY", "").strip()
            or os.environ.get("ZHIPU_API_KEY", "").strip()
            or DEFAULT_ZHIPU_API_KEY
        )
    )
    default_provider: str = field(default_factory=lambda: os.environ.get("LLM_DEFAULT_PROVIDER", "auto").strip() or "auto")
    timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 20.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))
    deepseek_base_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com").strip())
    zhipu_base_url: str = field(default_factory=lambda: os.environ.get("ZHIPU_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip())


def load_llm_settings() -> LLMSettings:
    return LLMSettings()
