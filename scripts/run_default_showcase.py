"""Run and persist the internal default showcase experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.default_showcase import write_default_showcase_artifact  # noqa: E402


def main() -> None:
    target = write_default_showcase_artifact(project_root=PROJECT_ROOT)
    payload = json.loads(target.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": payload["run"]["status"],
                "run_id": payload["run"]["run_id"],
                "policy_rows": len(payload["policy"]["path"]),
                "history_days": payload["history"]["total_days"],
                "output": str(target),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
