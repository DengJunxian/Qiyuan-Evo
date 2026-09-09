"""Experiment registry and reproducibility metadata panels."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import streamlit as st

from core.ui_text import localize_dataframe_columns, zh_provider_name


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload or {}), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def provider_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": platform.python_version()}
    for name in ("streamlit", "plotly", "pandas", "numpy", "akshare", "yfinance", "docx", "reportlab", "pyarrow"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except Exception:
            versions[name] = "not_installed"
    return versions


def cpp_extension_status() -> Dict[str, Any]:
    try:
        from core.exchange import order_book_cpp

        available = bool(getattr(order_book_cpp, "_civitas_lob", None) is not None)
        return {
            "available": available,
            "engine": "C++ 撮合扩展" if available else "Python 撮合回退",
            "message": "C++ 扩展可用" if available else "未检测到 C++ 扩展，已使用 Python 撮合回退。",
        }
    except Exception as exc:
        return {"available": False, "engine": "Python 撮合回退", "message": f"撮合扩展检测失败：{exc}"}


def build_experiment_registry_entry(
    *,
    experiment_id: str = "",
    scenario_name: str = "",
    config_hash: str = "",
    data_snapshot_id: str = "",
    seed: int = 42,
    selected_benchmark: str = "sh000001",
    status: str = "created",
    created_at: Optional[str] = None,
    parameter_set_id: str = "default_calibration_v1",
) -> Dict[str, Any]:
    created = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not config_hash:
        config_hash = stable_payload_hash(
            {
                "scenario_name": scenario_name,
                "seed": int(seed),
                "selected_benchmark": selected_benchmark,
                "created_at": created,
            }
        )
    if not experiment_id:
        experiment_id = f"exp_{stable_payload_hash({'scenario': scenario_name, 'config_hash': config_hash, 'data_snapshot_id': data_snapshot_id, 'seed': int(seed), 'parameter_set_id': parameter_set_id})[:16]}"
    return {
        "experiment_id": str(experiment_id),
        "scenario_name": str(scenario_name or "research_workbench"),
        "config_hash": str(config_hash),
        "data_snapshot_id": str(data_snapshot_id or "synthetic_or_cached_snapshot"),
        "data_snapshot_hash": str(data_snapshot_id or "synthetic_or_cached_snapshot"),
        "parameter_set_id": str(parameter_set_id or "default_calibration_v1"),
        "seed": int(seed),
        "created_at": created,
        "selected_benchmark": str(selected_benchmark or "sh000001"),
        "status": str(status or "created"),
    }


def build_reproducibility_meta(
    *,
    experiment_id: str = "",
    data_snapshot_hash: str = "",
    config_hash: str = "",
    random_seed: int = 42,
    llm_provider_chain: Optional[list[str]] = None,
    calibration_parameter_set_id: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "experiment_id": str(experiment_id or ""),
        "data_snapshot_hash": str(data_snapshot_hash or "synthetic_or_cached_snapshot"),
        "config_hash": str(config_hash or stable_payload_hash(extra or {})),
        "git_commit_hash": safe_git_commit(),
        "random_seed": int(random_seed),
        "provider_versions": provider_versions(),
        "llm_provider_chain": list(llm_provider_chain or ["GLM-4-flashx", "DeepSeek", "offline_fallback"]),
        "calibration_parameter_set_id": str(calibration_parameter_set_id or "default_calibration_v1"),
        "cpp_extension": cpp_extension_status(),
        "extra": dict(extra or {}),
    }


def render_experiment_registry(
    registry: Any,
    *,
    key_prefix: str = "experiment_registry",
) -> pd.DataFrame:
    if isinstance(registry, Mapping):
        rows = [dict(registry)]
    elif isinstance(registry, list):
        rows = [dict(item) for item in registry if isinstance(item, Mapping)]
    else:
        rows = []
    frame = pd.DataFrame(rows)
    st.markdown("### 实验登记信息")
    if frame.empty:
        st.info("暂无实验登记信息。")
        return frame
    cols = [
        "experiment_id",
        "scenario_name",
        "config_hash",
        "data_snapshot_id",
        "data_snapshot_hash",
        "parameter_set_id",
        "seed",
        "created_at",
        "selected_benchmark",
        "status",
    ]
    st.dataframe(localize_dataframe_columns(frame[[col for col in cols if col in frame.columns]]), use_container_width=True, hide_index=True)
    return frame


def render_reproducibility_panel(
    metadata: Mapping[str, Any],
    *,
    key_prefix: str = "repro",
) -> Dict[str, Any]:
    meta = dict(metadata or {})
    st.markdown("### 可复现信息")
    cols = st.columns(4)
    cols[0].metric("数据快照", str(meta.get("data_snapshot_hash", ""))[:12] or "-")
    cols[1].metric("配置哈希", str(meta.get("config_hash", ""))[:12] or "-")
    cols[2].metric("随机种子", str(meta.get("random_seed", "")))
    cols[3].metric("代码版本", str(meta.get("git_commit_hash", ""))[:12] or "unknown")

    provider_frame = pd.DataFrame(
        [{"provider": zh_provider_name(str(key)), "version": value} for key, value in dict(meta.get("provider_versions", {}) or {}).items()]
    )
    left, right = st.columns([1.0, 1.0])
    with left:
        st.markdown("#### 依赖版本")
        st.dataframe(localize_dataframe_columns(provider_frame), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### 运行链路")
        chain = " -> ".join(zh_provider_name(str(item)) for item in list(meta.get("llm_provider_chain", []) or [])) or "-"
        st.markdown(f"- 大模型与优化链路：{chain}")
        st.markdown(f"- 校准参数集：`{meta.get('calibration_parameter_set_id', '')}`")
        with st.expander("技术细节（可展开）", expanded=False):
            st.json(
                {
                    "llm_provider_chain": list(meta.get("llm_provider_chain", []) or []),
                    "calibration_parameter_set_id": meta.get("calibration_parameter_set_id", ""),
                    "cpp_extension": meta.get("cpp_extension", {}),
                },
                expanded=False,
            )
    return meta


__all__ = [
    "build_experiment_registry_entry",
    "build_reproducibility_meta",
    "cpp_extension_status",
    "provider_versions",
    "render_experiment_registry",
    "render_reproducibility_panel",
    "safe_git_commit",
    "stable_payload_hash",
]
