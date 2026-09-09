import asyncio
import json
import random
import re
from typing import Any, Dict, Optional, Callable, Tuple, Type

import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_THINK_TAGS_RE = re.compile(
    r"<(think|analysis|reasoning|system|assistant|tool)>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


def strip_nonstandard_tags(text: str) -> str:
    """移除模型输出中的 <think> 等非标准标签，避免干扰 JSON 解析。"""
    return _THINK_TAGS_RE.sub("", text or "").strip()


def _remove_control_chars(text: str) -> str:
    """去除不可见控制字符，避免 JSON 解析失败。"""
    return _CONTROL_CHARS_RE.sub("", text or "")


def _extract_json_candidate(text: str) -> str:
    """
    从原始文本中截取最可能的 JSON 片段。
    如果未找到明显 JSON，返回清理后的原文本。
    """
    if not text:
        return ""

    # 优先从代码块中取 JSON
    fenced = _CODE_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()

    # 从首个 { 或 [ 开始，尝试提取结构化片段
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not start_candidates:
        return text.strip()

    start = min(start_candidates)
    return text[start:].strip()


def _auto_close_json(text: str) -> str:
    """
    自动闭合被截断的 JSON 结构：
    1) 识别字符串上下文，避免误配
    2) 平衡 { } 与 [ ]
    3) 如果字符串未闭合，则补一个引号
    """
    if not text:
        return ""

    stack = []
    in_string = False
    escape = False
    out = []

    for ch in text:
        out.append(ch)
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()

    # 如果字符串未闭合，补一个引号
    if in_string:
        out.append('"')

    # 补齐未闭合的结构
    while stack:
        opener = stack.pop()
        out.append("}" if opener == "{" else "]")

    return "".join(out)


def _try_fix_json(raw_text: str) -> str:
    """
    尝试修复被模型输出破坏的 JSON 文本：
    - 过滤 <think> 等标签与代码围栏
    - 移除不可见控制字符
    - 自动闭合被截断的 { 或 [
    """
    cleaned = strip_nonstandard_tags(raw_text)
    cleaned = _remove_control_chars(cleaned)
    candidate = _extract_json_candidate(cleaned)
    return _auto_close_json(candidate)


def safe_json_loads(raw_text: str) -> Any:
    """安全 JSON 解析：失败时尝试修复后再解析。"""
    try:
        return json.loads(raw_text)
    except Exception:
        fixed = _try_fix_json(raw_text)
        return json.loads(fixed)


def async_retry_with_jitter(
    max_attempts: int = 3,
    base: float = 0.4,
    jitter: float = 0.2,
    max_wait: float = 6.0,
    retry_exceptions: Tuple[Type[BaseException], ...] = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        httpx.TimeoutException,
    ),
) -> Callable:
    """
    带抖动的指数退避重试装饰器，避免惊群效应。
    wait = min(max_wait, base * 2^n + random(0, jitter))
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    wait = min(max_wait, base * (2 ** (attempt - 1)) + random.uniform(0, jitter))
                    await asyncio.sleep(wait)
        return wrapper
    return decorator


class LLMClient:
    """
    轻量 LLM 客户端封装，提供：
    - 统一调用入口
    - 抖动指数退避
    """
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @async_retry_with_jitter()
    async def chat(
        self,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
