"""Runtime mode profiles for UI and simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class RuntimeModeProfile:
    """Execution profile for SMART/DEEP runtime selection."""

    mode: str
    label: str
    competition_safe_mode: bool
    market_pipeline_v2: bool
    llm_primary: bool
    use_live_api: bool
    enable_policy_committee: bool
    fast_slow_trigger: bool
    pause_for_llm_seconds: float
    model_priority: Tuple[str, ...]
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "label": self.label,
            "competition_safe_mode": self.competition_safe_mode,
            "market_pipeline_v2": self.market_pipeline_v2,
            "llm_primary": self.llm_primary,
            "use_live_api": self.use_live_api,
            "enable_policy_committee": self.enable_policy_committee,
            "fast_slow_trigger": self.fast_slow_trigger,
            "pause_for_llm_seconds": float(self.pause_for_llm_seconds),
            "model_priority": list(self.model_priority),
            "summary": self.summary,
        }


_SMART_PROFILE = RuntimeModeProfile(
    mode="SMART",
    label="智能模式",
    competition_safe_mode=True,
    market_pipeline_v2=True,
    llm_primary=True,
    use_live_api=True,
    enable_policy_committee=False,
    fast_slow_trigger=True,
    pause_for_llm_seconds=0.1,
    model_priority=("glm-4-flashx", "glm-4-flashx-250414"),
    summary="仅接入智谱 GLM 模型，适合低延迟政策理解、结构化抽取和常规推演。",
)

_DEEP_PROFILE = RuntimeModeProfile(
    mode="DEEP",
    label="高级模式",
    competition_safe_mode=False,
    market_pipeline_v2=True,
    llm_primary=True,
    use_live_api=True,
    enable_policy_committee=True,
    fast_slow_trigger=True,
    pause_for_llm_seconds=0.35,
    model_priority=("deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner", "deepseek-chat", "glm-4-flashx"),
    summary="沿用 DeepSeek + 智谱混合调度，启用推理优先、对话回退与委员会式校验。",
)


def normalize_runtime_mode(mode: str) -> str:
    normalized = str(mode or "SMART").strip().upper()
    if normalized in {"DEEP", "ADVANCED", "HIGH", "高级", "高级模式"}:
        return "DEEP"
    return "SMART"


def resolve_runtime_mode_profile(mode: str) -> RuntimeModeProfile:
    normalized = normalize_runtime_mode(mode)
    if normalized == "DEEP":
        return _DEEP_PROFILE
    return _SMART_PROFILE


def llm_mode_for_model(model_name: str) -> str:
    name = str(model_name or "").strip().lower()
    if name.startswith("glm-"):
        return "smart"
    if "v4-pro" in name or "reasoner" in name:
        return "slow"
    return "fast"


def merge_mode_feature_flags(
    mode: str,
    base_flags: Mapping[str, bool] | None = None,
) -> Dict[str, bool]:
    profile = resolve_runtime_mode_profile(mode)
    flags: Dict[str, bool] = dict(base_flags or {})
    flags["competition_safe_mode"] = bool(profile.competition_safe_mode)
    flags["market_pipeline_v2"] = bool(profile.market_pipeline_v2)
    flags["runtime_llm_primary"] = bool(profile.llm_primary)
    flags["runtime_use_live_api"] = bool(profile.use_live_api)
    flags["runtime_policy_committee_v1"] = bool(profile.enable_policy_committee)
    flags["runtime_fast_slow_trigger_v1"] = bool(profile.fast_slow_trigger)
    return flags
