"""Competition-mode compliance helpers and export artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


COMPETITION_MODE_FLAG = "competition_mode_v1"
_DEFAULT_FEATURE_FLAGS: Dict[str, bool] = {
    COMPETITION_MODE_FLAG: True,
}

_CONFIG_EXTENSIONS = {".py", ".json", ".toml", ".yaml", ".yml", ".ini", ".env", ".cfg", ".md"}
_KEYWORD_MAP = {
    "deepseek": "model_provider",
    "zhipu": "model_provider",
    "glm": "model_provider",
    "openai": "model_provider",
    "anthropic": "model_provider",
    "api_key": "credential_reference",
    "browser": "tooling",
    "playwright": "tooling",
    "search": "tooling",
    "reasoner": "model_variant",
    "chat": "model_variant",
}
_EXCLUDED_PATH_PARTS = {
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".tmp",
    "tmp",
    "outputs",
}


def resolve_competition_feature_flags(feature_flags: Optional[Mapping[str, Any]] = None) -> Dict[str, bool]:
    merged = dict(_DEFAULT_FEATURE_FLAGS)
    if feature_flags:
        for key in merged:
            if key in feature_flags:
                merged[key] = bool(feature_flags[key])
    return merged


def competition_mode_enabled(
    *,
    feature_flags: Optional[Mapping[str, Any]] = None,
    requested_mode: bool = False,
) -> bool:
    flags = resolve_competition_feature_flags(feature_flags)
    return bool(flags.get(COMPETITION_MODE_FLAG, True) and requested_mode)


def _iter_candidate_files(project_root: Path) -> Iterable[Path]:
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts.intersection(_EXCLUDED_PATH_PARTS):
            continue
        if any(part.startswith(".") for part in path.parts if part not in {".", ".."}):
            continue
        if path.suffix.lower() in _CONFIG_EXTENSIONS:
            yield path


def scan_model_and_tool_references(project_root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for path in _iter_candidate_files(project_root):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lowered = text.lower()
        matched_keywords = sorted({key for key in _KEYWORD_MAP if key in lowered})
        if not matched_keywords:
            continue
        compliance_risk = "medium" if any(key in lowered for key in ("api_key", "browser", "playwright")) else "low"
        findings.append(
            {
                "path": str(path),
                "keywords": matched_keywords,
                "categories": sorted({_KEYWORD_MAP[key] for key in matched_keywords}),
                "competition_risk": compliance_risk,
                "isolation_action": "route_to_competition_mode_offline_bundle" if compliance_risk != "low" else "document_only",
            }
        )
    findings.sort(key=lambda item: (item["competition_risk"] != "medium", item["path"]))
    return findings


def build_ai_tool_usage_manifest(
    *,
    project_root: Path,
    feature_flags: Optional[Mapping[str, Any]] = None,
    materials_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    findings = scan_model_and_tool_references(project_root)
    flags = resolve_competition_feature_flags(feature_flags)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "feature_flags": flags,
        "competition_mode_enabled": bool(flags.get(COMPETITION_MODE_FLAG, True)),
        "tools": findings,
        "isolation_strategy": {
            "default_mode": "offline_demo_bundle",
            "non_compliant_paths": [item["path"] for item in findings if item["competition_risk"] != "low"],
            "fallback_rule": "competition mode exports reproducible local artifacts and avoids online-only dependencies during defense.",
        },
        "materials_context": dict(materials_context or {}),
    }


def build_technical_route_template(
    *,
    manifest: Mapping[str, Any],
    app_flow: Optional[List[str]] = None,
) -> str:
    flow = app_flow or [
        "政策输入",
        "传导解释",
        "市场演化",
        "真实性报告",
        "A/B 对比",
        "监管优化",
    ]
    non_compliant = manifest.get("isolation_strategy", {}).get("non_compliant_paths", [])
    lines = [
        "# 技术路径说明模板",
        "",
        "## 一、系统定位",
        "- 本系统面向政策推演、市场演化解释、真实性验证与监管优化。",
        "- 综合展示模式默认采用离线可复现链路，避免现场运行依赖外部在线服务。",
        "",
        "## 二、综合展示链路",
    ]
    lines.extend([f"- {idx + 1}. {item}" for idx, item in enumerate(flow)])
    lines.extend(
        [
            "",
            "## 三、AI 工具使用说明",
            "- 所有模型与工具路径均通过 competition mode 做隔离说明。",
            "- 在线能力仅用于研发/扩展模式；综合展示默认使用离线数据与本地导出材料。",
            "",
            "## 四、合规隔离方案",
            f"- 非比赛合规路径数量：{len(non_compliant)}",
            "- 隔离原则：在线模型、浏览器工具、外部检索能力不作为默认展示路径。",
            "- 替代策略：统一导出 JSON/CSV/图表/Markdown/PDF 材料供现场复用。",
            "",
            "## 五、可复现性声明",
            f"- feature flags: {json.dumps(dict(manifest.get('feature_flags', {})), ensure_ascii=False, sort_keys=True)}",
            "- 输出材料包含配置快照、结构化清单与导出清单，支持断网复演。",
        ]
    )
    return "\n".join(lines)


def write_competition_compliance_artifacts(
    *,
    root_dir: Path,
    project_root: Path,
    feature_flags: Optional[Mapping[str, Any]] = None,
    materials_context: Optional[Mapping[str, Any]] = None,
    app_flow: Optional[List[str]] = None,
) -> Dict[str, Any]:
    root_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_ai_tool_usage_manifest(
        project_root=project_root,
        feature_flags=feature_flags,
        materials_context=materials_context,
    )
    route_template = build_technical_route_template(manifest=manifest, app_flow=app_flow)
    manifest_path = root_dir / "ai_tool_usage_manifest.json"
    route_path = root_dir / "technical_route_template.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    route_path.write_text(route_template, encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "technical_route_template": route_template,
        "technical_route_template_path": route_path,
    }
