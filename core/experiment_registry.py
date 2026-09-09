"""Append-only experiment registry for reproducible demos and reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.reproducibility import (
    LLMCallRecord,
    build_reproducibility_envelope,
    canonicalize_for_hash,
    experiment_id as build_experiment_id,
    safe_git_commit,
    stable_hash,
    utc_now_iso,
)


DEFAULT_REGISTRY_PATH = Path("outputs") / "experiment_registry" / "registry.jsonl"


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    module: str
    scenario_name: str
    config_hash: str
    data_snapshot_hash: str
    parameter_set_id: str
    seed: int
    benchmark_symbol: str = "sh000001"
    status: str = "created"
    created_at_utc: str = field(default_factory=utc_now_iso)
    git_commit: str = field(default_factory=safe_git_commit)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_experiment_record(
    *,
    module: str,
    scenario_name: str = "",
    config: Mapping[str, Any] | None = None,
    dataset: Any = None,
    parameters: Mapping[str, Any] | None = None,
    seed: int | None = None,
    benchmark_symbol: str = "sh000001",
    status: str = "created",
    llm_calls: Sequence[LLMCallRecord | Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    notes: Mapping[str, Any] | None = None,
) -> ExperimentRecord:
    envelope = build_reproducibility_envelope(
        module=module,
        config=config,
        dataset=dataset,
        parameters=parameters,
        seed=seed,
        llm_calls=llm_calls,
        extra={"scenario_name": scenario_name, "benchmark_symbol": benchmark_symbol},
    )
    return ExperimentRecord(
        experiment_id=envelope.experiment_id,
        module=str(module),
        scenario_name=str(scenario_name or module),
        config_hash=envelope.config_hash,
        data_snapshot_hash=envelope.data_snapshot_hash,
        parameter_set_id=envelope.parameter_set_id,
        seed=int(envelope.random_seed),
        benchmark_symbol=str(benchmark_symbol or "sh000001"),
        status=str(status or "created"),
        git_commit=envelope.git_commit,
        llm_calls=list(envelope.llm_calls),
        metrics=canonicalize_for_hash(dict(metrics or {})),
        artifacts=canonicalize_for_hash(dict(artifacts or {})),
        notes=canonicalize_for_hash(dict(notes or {})),
    )


def create_experiment_record_from_hashes(
    *,
    module: str,
    scenario_name: str = "",
    config_hash: str,
    data_snapshot_hash: str,
    parameter_set_id: str = "",
    seed: int = 42,
    benchmark_symbol: str = "sh000001",
    status: str = "created",
    llm_calls: Sequence[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    notes: Mapping[str, Any] | None = None,
) -> ExperimentRecord:
    exp_id = build_experiment_id(
        module=module,
        config_hash_value=str(config_hash),
        data_snapshot_hash_value=str(data_snapshot_hash),
        seed=int(seed),
        parameter_set_id_value=str(parameter_set_id or ""),
    )
    return ExperimentRecord(
        experiment_id=exp_id,
        module=str(module),
        scenario_name=str(scenario_name or module),
        config_hash=str(config_hash),
        data_snapshot_hash=str(data_snapshot_hash),
        parameter_set_id=str(parameter_set_id or ""),
        seed=int(seed),
        benchmark_symbol=str(benchmark_symbol or "sh000001"),
        status=str(status or "created"),
        llm_calls=[canonicalize_for_hash(dict(item)) for item in list(llm_calls or [])],
        metrics=canonicalize_for_hash(dict(metrics or {})),
        artifacts=canonicalize_for_hash(dict(artifacts or {})),
        notes=canonicalize_for_hash(dict(notes or {})),
    )


def append_experiment_record(record: ExperimentRecord, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return path


def load_experiment_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[ExperimentRecord]:
    path = Path(registry_path)
    if not path.exists():
        return []
    records: list[ExperimentRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        records.append(ExperimentRecord(**payload))
    return records


def upsert_experiment_record(record: ExperimentRecord, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    path = Path(registry_path)
    existing = [item for item in load_experiment_registry(path) if item.experiment_id != record.experiment_id]
    existing.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True, default=str) for item in existing) + "\n",
        encoding="utf-8",
    )
    return path


def attach_experiment_to_report(
    report_payload: Mapping[str, Any],
    *,
    experiment_id: str,
    report_type: str = "report",
) -> dict[str, Any]:
    payload = canonicalize_for_hash(dict(report_payload or {}))
    payload["experiment_id"] = str(experiment_id)
    payload["report_experiment_id"] = str(experiment_id)
    payload["report_type"] = str(report_type)
    payload["report_hash"] = stable_hash({"report_type": report_type, "experiment_id": experiment_id, "payload": payload})
    return payload


def records_to_frame_rows(records: Iterable[ExperimentRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "ExperimentRecord",
    "append_experiment_record",
    "attach_experiment_to_report",
    "create_experiment_record",
    "create_experiment_record_from_hashes",
    "load_experiment_registry",
    "records_to_frame_rows",
    "upsert_experiment_record",
]
