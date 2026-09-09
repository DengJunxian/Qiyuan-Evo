"""Unified experiment manifest for reproducible Civitas runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import hashlib
import json
import subprocess

from core.reproducibility import experiment_id as build_experiment_id, parameter_set_id as build_parameter_set_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class ExperimentManifest:
    """Run-level metadata used by backtest/replay/policy/regulator/report pipelines."""

    run_id: str
    timestamp: str
    git_commit: str
    config_hash: str
    seed: int
    dataset_snapshot_id: str
    module: str
    experiment_id: str = ""
    data_snapshot_hash: str = ""
    parameter_set_id: str = ""
    code_version: str = ""
    model_version: str = ""
    data_slice: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @staticmethod
    def build(
        *,
        module: str,
        seed: int,
        dataset_snapshot_id: str,
        config: Optional[Mapping[str, Any]] = None,
        feature_flags: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
        model_version: str = "",
        data_slice: Optional[Mapping[str, Any]] = None,
        parent_run_id: str = "",
        notes: Optional[Mapping[str, Any]] = None,
        git_commit: Optional[str] = None,
        config_hash: Optional[str] = None,
        timestamp: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "ExperimentManifest":
        clean_config = dict(config or {})
        clean_flags = {str(k): bool(v) for k, v in dict(feature_flags or {}).items()}
        clean_artifacts = dict(artifacts or {})
        clean_data_slice = dict(data_slice or {})
        clean_notes = dict(notes or {})
        manifest_timestamp = str(timestamp or _utc_now_iso())
        resolved_git_commit = str(git_commit or _safe_git_commit())
        manifest_config_hash = str(
            config_hash
            or _stable_hash(
                {
                    "module": module,
                    "seed": int(seed),
                    "dataset_snapshot_id": str(dataset_snapshot_id or ""),
                    "model_version": str(model_version or ""),
                    "data_slice": clean_data_slice,
                    "config": clean_config,
                    "feature_flags": clean_flags,
                }
            )
        )
        data_snapshot_hash = str(dataset_snapshot_id or "")
        parameter_id = str(
            clean_notes.get("parameter_set_id")
            or clean_config.get("parameter_set_id")
            or build_parameter_set_id(clean_config.get("calibration_parameters", clean_config))
        )
        resolved_experiment_id = build_experiment_id(
            module=str(module),
            config_hash_value=manifest_config_hash,
            data_snapshot_hash_value=data_snapshot_hash,
            seed=int(seed),
            parameter_set_id_value=parameter_id,
        )
        manifest_run_id = str(
            run_id
            or f"{module}-{manifest_timestamp.replace(':', '').replace('-', '').replace('+00:00', 'Z')}-{manifest_config_hash[:12]}"
        )
        return ExperimentManifest(
            run_id=manifest_run_id,
            timestamp=manifest_timestamp,
            git_commit=resolved_git_commit,
            config_hash=manifest_config_hash,
            seed=int(seed),
            dataset_snapshot_id=str(dataset_snapshot_id or ""),
            module=str(module),
            experiment_id=resolved_experiment_id,
            data_snapshot_hash=data_snapshot_hash,
            parameter_set_id=parameter_id,
            code_version=resolved_git_commit,
            model_version=str(model_version or ""),
            data_slice=clean_data_slice,
            feature_flags=clean_flags,
            config=clean_config,
            artifacts=clean_artifacts,
            parent_run_id=str(parent_run_id or ""),
            notes=clean_notes,
        )


def write_experiment_manifest(manifest: ExperimentManifest, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{manifest.run_id}.manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentManifest(**payload)


def attach_manifest_to_metadata(
    metadata: Optional[Mapping[str, Any]],
    *,
    module: str,
    seed: int,
    dataset_snapshot_id: str,
    feature_flags: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    model_version: str = "",
    data_slice: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta = dict(metadata or {})
    manifest = ExperimentManifest.build(
        module=module,
        seed=int(seed),
        dataset_snapshot_id=str(dataset_snapshot_id),
        feature_flags=feature_flags,
        config=config,
        artifacts=artifacts,
        model_version=model_version,
        data_slice=data_slice,
    )
    meta["experiment_manifest"] = manifest.to_dict()
    meta["experiment_manifest_id"] = manifest.run_id
    meta["experiment_id"] = manifest.experiment_id
    meta["experiment_config_hash"] = manifest.config_hash
    meta["code_version"] = manifest.code_version
    meta["model_version"] = manifest.model_version
    meta["data_slice"] = manifest.data_slice
    meta["seed"] = int(seed)
    meta["dataset_snapshot_id"] = str(dataset_snapshot_id)
    meta["data_snapshot_hash"] = manifest.data_snapshot_hash
    meta["parameter_set_id"] = manifest.parameter_set_id
    return meta
