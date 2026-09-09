"""Natural-language rendering helpers for structured UI payloads."""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import streamlit as st

from core.runtime_mode import llm_mode_for_model
from core.ui_text import zh_metric_name


BANNED_NARRATIVE_PHRASES = [
    "建议先看趋势方向，再结合波动和风险项判断执行节奏",
    "需要结合更多信息进一步判断",
    "总体来看仍需持续观察",
    "建议综合判断",
    "需关注风险变化",
    "数据表明",
    "综上所述",
    "整体来看",
]

GENERIC_WORDS = ("趋势", "波动", "风险", "节奏")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _compact_payload(value: Any, *, depth: int = 0, max_depth: int = 3, max_items: int = 8) -> Any:
    if depth >= max_depth:
        if _is_scalar(value):
            return value
        return str(value)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = f"trimmed_{len(value) - max_items}_items"
                break
            out[str(key)] = _compact_payload(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return out
    if isinstance(value, list):
        head = value[:max_items]
        out = [_compact_payload(item, depth=depth + 1, max_depth=max_depth, max_items=max_items) for item in head]
        if len(value) > max_items:
            out.append(f"trimmed_{len(value) - max_items}_items")
        return out
    if _is_scalar(value):
        return value
    return str(value)


def _flatten_payload(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 32,
    out: List[Tuple[str, str]] | None = None,
) -> List[Tuple[str, str]]:
    if out is None:
        out = []
    if len(out) >= max_items:
        return out
    if depth >= max_depth or _is_scalar(value):
        key = prefix or "value"
        out.append((key, str(value)))
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            if len(out) >= max_items:
                break
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_payload(item, prefix=next_prefix, depth=depth + 1, max_depth=max_depth, max_items=max_items, out=out)
        return out
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if len(out) >= max_items:
                break
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_payload(item, prefix=next_prefix, depth=depth + 1, max_depth=max_depth, max_items=max_items, out=out)
        return out
    key = prefix or "value"
    out.append((key, str(value)))
    return out


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _format_number(value: float) -> str:
    if abs(value) < 1 and value != 0:
        return f"{value:.2%}" if abs(value) <= 0.35 else f"{value:.3f}"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _metric_label(path: str) -> str:
    raw = str(path).replace("]", "").split(".")[-1].split("[")[-1]
    return zh_metric_name(raw)


def _collect_numbers(value: Any, *, prefix: str = "", out: List[Tuple[str, float]] | None = None) -> List[Tuple[str, float]]:
    if out is None:
        out = []
    if len(out) >= 80:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_numbers(item, prefix=next_prefix, out=out)
        return out
    if isinstance(value, list):
        for idx, item in enumerate(value[:24]):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _collect_numbers(item, prefix=next_prefix, out=out)
        return out
    if _is_number(value):
        out.append((prefix or "value", float(value)))
    return out


def _collect_metric_names(payload: Any) -> List[str]:
    names: List[str] = []
    for key, value in _flatten_payload(payload, max_items=40):
        if _is_number(value):
            label = _metric_label(key)
            if label not in names and label not in {"value", "数值"}:
                names.append(label)
    return names


def _payload_has_numbers(payload: Any) -> bool:
    return bool(_collect_numbers(payload))


def _text_has_number(text: str) -> bool:
    return bool(re.search(r"[-+]?\d+(?:\.\d+)?%?", text or ""))


def _recent_narratives() -> List[str]:
    try:
        recent = st.session_state.get("_recent_ui_narratives", [])
    except Exception:
        return []
    return [str(item) for item in recent if str(item).strip()]


def _remember_narrative(text: str) -> None:
    try:
        recent = list(st.session_state.get("_recent_ui_narratives", []))
        recent.append(str(text or ""))
        st.session_state["_recent_ui_narratives"] = recent[-8:]
    except Exception:
        return


def _narrative_llm_disabled() -> bool:
    try:
        return bool(st.session_state.get("_ui_narrative_llm_disabled", False))
    except Exception:
        return False


def _disable_narrative_llm() -> None:
    try:
        st.session_state["_ui_narrative_llm_disabled"] = True
    except Exception:
        return


def _normalized_similarity(left: str, right: str) -> float:
    clean_left = re.sub(r"\s+", "", left or "")
    clean_right = re.sub(r"\s+", "", right or "")
    if not clean_left or not clean_right:
        return 0.0
    return difflib.SequenceMatcher(None, clean_left, clean_right).ratio()


def is_low_value_narrative(text: str, payload: Any) -> bool:
    body = str(text or "").strip()
    if not body:
        return True
    if any(phrase in body for phrase in BANNED_NARRATIVE_PHRASES):
        return True
    if len(body) < 80:
        return True
    if _payload_has_numbers(payload) and not _text_has_number(body):
        return True

    metric_names = _collect_metric_names(payload)
    if len(metric_names) > 3:
        mentioned = sum(1 for name in metric_names if name and name in body)
        if mentioned < 2:
            return True

    for recent in _recent_narratives()[-8:]:
        if _normalized_similarity(body, recent) >= 0.88:
            return True

    if all(word in body for word in GENERIC_WORDS) and not _text_has_number(body):
        return True
    return False


def _context_payload(
    *,
    module_id: str = "",
    chart_title: str = "",
    business_question: str = "",
    key_metrics: Any = None,
    trend_summary: str = "",
    threshold_rules: Any = None,
    policy_context: str = "",
    user_visible_units: Any = None,
) -> Dict[str, Any]:
    return {
        "module_id": module_id or "通用指标模块",
        "chart_title": chart_title,
        "business_question": business_question,
        "key_metrics": key_metrics or {},
        "trend_summary": trend_summary,
        "threshold_rules": threshold_rules or {},
        "policy_context": policy_context,
        "user_visible_units": user_visible_units or {},
    }


def _specific_value_list(payload: Any, limit: int = 6) -> List[str]:
    numbers = _collect_numbers(payload)
    if not numbers:
        return []
    ranked = sorted(numbers, key=lambda item: abs(item[1]), reverse=True)[:limit]
    return [f"{_metric_label(path)}={_format_number(value)}" for path, value in ranked]


def _fallback_narrative(title: str, payload: Any, context: str = "", **context_kwargs: Any) -> str:
    numbers = _collect_numbers(payload)
    metric_names = _collect_metric_names(payload)
    values = _specific_value_list(payload, limit=6)
    ctx_kwargs = dict(context_kwargs)
    ctx_kwargs.setdefault("chart_title", title)
    ctx_kwargs.setdefault("business_question", context)
    ctx = _context_payload(**ctx_kwargs)
    question = str(ctx.get("business_question") or context or "该模块用于判断政策冲击后的市场状态和治理含义。")
    policy_context = str(ctx.get("policy_context") or "政策实验")

    if not numbers:
        return (
            f"从【{title}】看，这个模块主要回答“{question}”。当前结构化载荷没有给出可计算数值，"
            f"因此前端只保留定性解释：它用于把{policy_context}中的传导环节、主体行为和风险线索组织成可复盘证据。"
            "后续一旦生成指标序列，系统会优先比较最新值、极值和阈值触发情况，再形成政策含义。"
        )

    latest_path, latest_value = numbers[-1]
    max_path, max_value = max(numbers, key=lambda item: item[1])
    min_path, min_value = min(numbers, key=lambda item: item[1])
    top_abs = sorted(numbers, key=lambda item: abs(item[1]), reverse=True)[:2]
    latest_label = _metric_label(latest_path)
    max_label = _metric_label(max_path)
    min_label = _metric_label(min_path)
    abnormal = "、".join(f"{_metric_label(path)}为{_format_number(value)}" for path, value in top_abs)

    threshold_text = ""
    threshold_rules = ctx.get("threshold_rules") or {}
    if isinstance(threshold_rules, Mapping):
        hits = []
        for raw_key, threshold in threshold_rules.items():
            if not _is_number(threshold):
                continue
            key = str(raw_key)
            matched = next((value for path, value in numbers if key in path or zh_metric_name(key) == _metric_label(path)), None)
            if matched is None:
                continue
            threshold_value = float(threshold)
            direction = "超过" if matched >= threshold_value else "低于"
            hits.append(f"{zh_metric_name(key)}{direction}阈值{_format_number(threshold_value)}")
        if hits:
            threshold_text = "；阈值检查显示：" + "、".join(hits[:2])

    if len(numbers) >= 2:
        first_value = numbers[0][1]
        change = latest_value - first_value
        change_text = f"区间变化约{_format_number(change)}"
    else:
        change_text = "当前为单点观测"

    if any("panic" in path or "恐慌" in _metric_label(path) for path, _ in numbers):
        state = "风险扩散"
    elif latest_value >= 0:
        state = "修复或稳定"
    else:
        state = "承压"

    metric_hint = "、".join(metric_names[:3]) if metric_names else "核心指标"
    value_hint = "；关键数值包括：" + "、".join(values[:4]) if values else ""
    return (
        f"从【{title}】看，这个指标模块回答的是“{question}”。最新观测中，{latest_label}为{_format_number(latest_value)}，"
        f"{max_label}最高达到{_format_number(max_value)}，{min_label}最低为{_format_number(min_value)}，{change_text}{threshold_text}。"
        f"当前最突出的两个异常读数是{abnormal}{value_hint}。对{policy_context}而言，{metric_hint}共同指向市场更接近“{state}”状态，"
        "政策含义是：若高风险读数继续抬升，应优先比较不同干预时点的成本和稳定性；若价格修复但流动性没有同步改善，则需要把方案重点放在预期管理和微观流动性修复上。"
    )


def _build_prompt(
    title: str,
    payload: Any,
    context: str,
    *,
    retry_values: Sequence[str] | None = None,
    **context_kwargs: Any,
) -> str:
    compact = _compact_payload(payload)
    serialized = json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) > 5000:
        serialized = f"{serialized[:5000]}...(truncated)"
    ctx_kwargs = dict(context_kwargs)
    ctx_kwargs.setdefault("chart_title", title)
    ctx_kwargs.setdefault("business_question", context)
    ctx = _context_payload(**ctx_kwargs)
    banned = "；".join(BANNED_NARRATIVE_PHRASES)
    retry_line = ""
    if retry_values:
        retry_line = "本次重写必须引用以下具体数值：" + "、".join(retry_values)
    return "\n".join(
        [
            "请把下面结构化指标写成面向政策制定者的中文仿真后果研判。",
            "要求：",
            "1. 只输出中文自然语言，不输出 JSON、代码块、键名清单。",
            "2. 至少引用 2 个具体指标或数值，但不能逐项抄数据；要解释它们之间的关系、方向和可能原因。",
            "3. 必须说明这张图或这个指标模块回答了什么政策问题，以及政策实施后可能出现的市场状态。",
            "4. 必须给出一个具体风险判断、触发条件或政策含义，指出下一步应重点监测什么。",
            "5. 表达要有差异化，不允许套话，不允许重复历史输出，不要说“作为 AI 模型”。",
            f"6. 不允许使用这些表达：{banned}",
            "7. 如果数据存在拐点、背离、阈值接近或结构性分化，必须优先解释这些机制。",
            retry_line,
            f"模块上下文：{json.dumps(ctx, ensure_ascii=False, sort_keys=True, default=str)}",
            f"标题：{title}",
            f"原始上下文：{context or '无'}",
            f"数据：{serialized}",
        ]
    )


def _llm_narrative(title: str, payload: Any, context: str = "", **context_kwargs: Any) -> str:
    if _narrative_llm_disabled():
        return ""
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        _disable_narrative_llm()
        return ""

    fallback = _fallback_narrative(title, payload, context, **context_kwargs)
    system_prompt = (
        "你是金融政策仿真系统的政策后果研判助手，面向政策制定者解释仿真结果。"
        "你的输出必须从数据关系推导政策含义，避免照抄数值和模板化表述。"
    )
    prompts = [
        _build_prompt(title, payload, context, **context_kwargs),
        _build_prompt(title, payload, context, retry_values=_specific_value_list(payload, limit=5), **context_kwargs),
    ]
    model_name = "glm-4-flashx"
    try:
        runtime_profile = st.session_state.get("runtime_mode_profile", {})
        priority = runtime_profile.get("model_priority", []) if isinstance(runtime_profile, dict) else []
        if priority:
            model_name = str(priority[0])
    except Exception:
        model_name = "glm-4-flashx"
    model_mode = llm_mode_for_model(model_name)

    for prompt in prompts:
        try:
            backend = APIBackend(model=model_name, max_tokens=420, temperature=0.35)
            content = str(
                backend.generate(
                    prompt,
                    system_prompt=system_prompt,
                    mode=model_mode,
                    task_type="ui_narrative",
                    max_tokens=420,
                    temperature=0.35,
                    timeout_budget=18.0,
                    fallback_response=fallback,
                )
                or ""
            ).strip()
        except Exception:
            _disable_narrative_llm()
            content = ""
        if not content or content.startswith("[API Error]"):
            _disable_narrative_llm()
            continue
        if content == fallback:
            _disable_narrative_llm()
            return ""
        stripped = content.lstrip()
        if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```"):
            continue
        if not is_low_value_narrative(content, payload):
            return content
    return ""


def narrate_payload(
    title: str,
    payload: Any,
    *,
    context: str = "",
    cache_namespace: str = "ui_narrative_cache",
    module_id: str = "",
    chart_title: str = "",
    business_question: str = "",
    key_metrics: Any = None,
    trend_summary: str = "",
    threshold_rules: Any = None,
    policy_context: str = "",
    user_visible_units: Any = None,
) -> str:
    compact = _compact_payload(payload)
    context_kwargs = {
        "module_id": module_id,
        "chart_title": chart_title or title,
        "business_question": business_question or context,
        "key_metrics": key_metrics,
        "trend_summary": trend_summary,
        "threshold_rules": threshold_rules,
        "policy_context": policy_context,
        "user_visible_units": user_visible_units,
    }
    cache_key = hashlib.sha256(
        json.dumps(
            {"title": title, "context": context, "context_kwargs": context_kwargs, "payload": compact},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    cache: Dict[str, str] = st.session_state.setdefault(cache_namespace, {})
    if cache_key in cache:
        return cache[cache_key]

    text = _llm_narrative(title, compact, context, **context_kwargs)
    if not text or is_low_value_narrative(text, compact):
        text = _fallback_narrative(title, compact, context, **context_kwargs)
    cache[cache_key] = text
    _remember_narrative(text)
    return text


def render_narrative_block(
    title: str,
    payload: Any,
    *,
    context: str = "",
    cache_namespace: str = "ui_narrative_cache",
    label: str = "大模型解读",
    module_id: str = "",
    chart_title: str = "",
    business_question: str = "",
    key_metrics: Any = None,
    trend_summary: str = "",
    threshold_rules: Any = None,
    policy_context: str = "",
    user_visible_units: Any = None,
) -> str:
    text = narrate_payload(
        title,
        payload,
        context=context,
        cache_namespace=cache_namespace,
        module_id=module_id,
        chart_title=chart_title,
        business_question=business_question,
        key_metrics=key_metrics,
        trend_summary=trend_summary,
        threshold_rules=threshold_rules,
        policy_context=policy_context,
        user_visible_units=user_visible_units,
    )
    html_body = _narrative_text_to_html(text)
    st.markdown(
        f"""
        <div class="insight-block">
          <div class="insight-block-label">{label}</div>
          <div class="insight-block-title">{title}</div>
          <div class="insight-block-body">{html_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return text


def _narrative_text_to_html(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines()]
    chunks: List[str] = []
    list_items: List[str] = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            chunks.append("<ul>" + "".join(list_items) + "</ul>")
            list_items = []

    for line in lines:
        if not line:
            flush_list()
            continue
        if line.startswith("- "):
            list_items.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        flush_list()
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            chunks.append(f"<p><strong>{html.escape(line[2:-2])}</strong></p>")
            continue
        safe = html.escape(line).replace("**", "")
        chunks.append(f"<p>{safe}</p>")
    flush_list()
    return "".join(chunks)


__all__ = [
    "BANNED_NARRATIVE_PHRASES",
    "_fallback_narrative",
    "is_low_value_narrative",
    "narrate_payload",
    "render_narrative_block",
]
