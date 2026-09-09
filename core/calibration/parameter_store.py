"""Small deterministic parameter-set store for replay calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from core.exchange.trade_tape import stable_hash
from core.reproducibility import parameter_set_id


@dataclass(frozen=True)
class ParameterSet:
    name: str
    params: Dict[str, float]
    seed: int = 42
    config_hash: str = ""
    data_snapshot_hash: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_hash(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "params": self.params,
                "seed": int(self.seed),
                "config_hash": self.config_hash,
                "data_snapshot_hash": self.data_snapshot_hash,
            }
        )

    @property
    def parameter_set_id(self) -> str:
        return parameter_set_id(
            {
                "name": self.name,
                "params": self.params,
                "seed": int(self.seed),
                "config_hash": self.config_hash,
                "data_snapshot_hash": self.data_snapshot_hash,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parameter_hash"] = self.parameter_hash
        payload["parameter_set_id"] = self.parameter_set_id
        return payload


class ParameterStore:
    def __init__(self, root_dir: str | Path = "outputs/parameter_store") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, parameter_set: ParameterSet) -> Path:
        path = self.root_dir / f"{parameter_set.name}_{parameter_set.parameter_hash[:12]}.json"
        path.write_text(json.dumps(parameter_set.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, path_or_name: str | Path) -> Optional[ParameterSet]:
        path = Path(path_or_name)
        if not path.exists():
            matches = sorted(self.root_dir.glob(f"{path_or_name}_*.json"))
            if not matches:
                return None
            path = matches[-1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("parameter_hash", None)
        payload.pop("parameter_set_id", None)
        return ParameterSet(
            name=str(payload.get("name", path.stem)),
            params={str(k): float(v) for k, v in dict(payload.get("params", {}) or {}).items()},
            seed=int(payload.get("seed", 42)),
            config_hash=str(payload.get("config_hash", "")),
            data_snapshot_hash=str(payload.get("data_snapshot_hash", "")),
            created_at_utc=str(payload.get("created_at_utc", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @staticmethod
    def from_mapping(
        name: str,
        params: Mapping[str, Any],
        *,
        seed: int,
        config_hash: str,
        data_snapshot_hash: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ParameterSet:
        return ParameterSet(
            name=str(name),
            params={str(k): float(v) for k, v in dict(params or {}).items()},
            seed=int(seed),
            config_hash=str(config_hash),
            data_snapshot_hash=str(data_snapshot_hash),
            metadata=dict(metadata or {}),
        )


__all__ = ["ParameterSet", "ParameterStore"]
