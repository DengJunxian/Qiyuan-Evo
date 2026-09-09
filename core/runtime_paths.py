"""运行时路径辅助工具。

用于在测试或验收过程中隔离临时工件，避免反复改写仓库中已跟踪的数据文件。
"""

from __future__ import annotations

import os
from pathlib import Path


def running_under_pytest() -> bool:
    """判断当前是否处于 pytest 运行上下文。"""

    return "PYTEST_CURRENT_TEST" in os.environ


def resolve_runtime_path(default_path: str | Path, *, env_var: str | None = None) -> Path:
    """解析运行时路径。

    优先级：
    1. 显式环境变量覆盖
    2. pytest 隔离目录
    3. 仓库默认路径
    """

    if env_var:
        override = str(os.environ.get(env_var, "") or "").strip()
        if override:
            return Path(override)

    base_path = Path(default_path)
    if running_under_pytest():
        pytest_root = Path(os.environ.get("CIVITAS_PYTEST_RUNTIME_ROOT", ".pytest_tmp/runtime_artifacts"))
        return pytest_root / base_path

    return base_path
