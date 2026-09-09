"""Unified Event Store for replay/policy/calibration scenarios.

Storage layout:
    data/event_store/<dataset_version>/
      events/
        event_type=<name>/date=<YYYY-MM-DD>/*.parquet
      snapshots/<snapshot_id>/manifest.json
      scenarios/<scenario_id>.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
import hashlib
import json

import pandas as pd

from core.runtime_paths import resolve_runtime_path


class EventType(str, Enum):
    MARKET_BAR = "market_bar"
    TRADE_TAPE = "trade_tape"
    POLICY = "policy"
    MACRO = "macro"
    NEWS = "news"
    RUMOR = "rumor"
    REFUTE = "refute"
    REGIME = "regime"
    REGULATORY_ACTION = "regulatory_action"


def _to_iso(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz="UTC")
    return ts.isoformat()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_event_type(value: str | EventType) -> str:
    if isinstance(value, EventType):
        return value.value
    text = str(value).strip().lower()
    aliases = {
        "bar": EventType.MARKET_BAR.value,
        "bars": EventType.MARKET_BAR.value,
        "trade": EventType.TRADE_TAPE.value,
        "trades": EventType.TRADE_TAPE.value,
    }
    return aliases.get(text, text)


@dataclass
class EventRecord:
    """Canonical event record with visibility guardrails."""

    timestamp: str
    source: str
    confidence: float
    visibility_time: str
    event_type: str | EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "event_id": self.event_id or self.stable_id(),
            "event_type": _normalize_event_type(self.event_type),
            "timestamp": _to_iso(self.timestamp),
            "visibility_time": _to_iso(self.visibility_time),
            "source": str(self.source),
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "payload_json": json.dumps(self.payload or {}, ensure_ascii=False, sort_keys=True, default=str),
            "metadata_json": json.dumps(self.metadata or {}, ensure_ascii=False, sort_keys=True, default=str),
        }
        return payload

    def stable_id(self) -> str:
        payload = {
            "event_type": _normalize_event_type(self.event_type),
            "timestamp": _to_iso(self.timestamp),
            "visibility_time": _to_iso(self.visibility_time),
            "source": str(self.source),
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "payload": self.payload or {},
            "metadata": self.metadata or {},
        }
        return _stable_hash(payload)

    @staticmethod
    def from_row(row: Mapping[str, Any]) -> "EventRecord":
        payload_raw = row.get("payload_json", "{}")
        metadata_raw = row.get("metadata_json", "{}")
        payload = payload_raw if isinstance(payload_raw, dict) else json.loads(str(payload_raw or "{}"))
        metadata = metadata_raw if isinstance(metadata_raw, dict) else json.loads(str(metadata_raw or "{}"))
        return EventRecord(
            event_id=str(row.get("event_id", "")),
            event_type=str(row.get("event_type", "")),
            timestamp=_to_iso(row.get("timestamp")),
            visibility_time=_to_iso(row.get("visibility_time")),
            source=str(row.get("source", "")),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            payload=dict(payload),
            metadata=dict(metadata),
        )


@dataclass
class SnapshotManifest:
    snapshot_id: str
    dataset_version: str
    timestamp: str
    seed: int
    config_hash: str
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    event_counts: Dict[str, int] = field(default_factory=dict)
    scenario_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioManifest:
    scenario_id: str
    dataset_version: str
    snapshot_id: str
    created_at: str
    seed: int
    config_hash: str
    start_time: str
    end_time: str
    event_types: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventStore:
    """Parquet-backed event store with snapshot/scenario manifests."""

    def __init__(self, root_dir: str | Path = "data/event_store") -> None:
        self.root_dir = resolve_runtime_path(root_dir, env_var="CIVITAS_EVENT_STORE_ROOT")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_root(self, dataset_version: str) -> Path:
        root = self.root_dir / str(dataset_version)
        root.mkdir(parents=True, exist_ok=True)
        (root / "events").mkdir(parents=True, exist_ok=True)
        (root / "snapshots").mkdir(parents=True, exist_ok=True)
        (root / "scenarios").mkdir(parents=True, exist_ok=True)
        return root

    def _event_partition_dir(self, dataset_version: str, event_type: str, event_date: str) -> Path:
        root = self._dataset_root(dataset_version) / "events"
        path = root / f"event_type={event_type}" / f"date={event_date}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _normalize_timestamp(value: Any) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            ts = pd.Timestamp.now(tz="UTC")
        return ts

    def append_events(
        self,
        dataset_version: str,
        events: Sequence[EventRecord | Mapping[str, Any]],
        *,
        seed: Optional[int] = None,
        config_hash: str = "",
        snapshot_id: str = "",
    ) -> List[Path]:
        written: List[Path] = []
        if not events:
            return written

        normalized: List[EventRecord] = []
        for item in events:
            if isinstance(item, EventRecord):
                normalized.append(item)
            else:
                normalized.append(
                    EventRecord(
                        event_id=str(item.get("event_id", "")),
                        event_type=str(item.get("event_type", "")),
                        timestamp=str(item.get("timestamp", "")),
                        visibility_time=str(item.get("visibility_time", item.get("timestamp", ""))),
                        source=str(item.get("source", "")),
                        confidence=float(item.get("confidence", 0.0) or 0.0),
                        payload=dict(item.get("payload", {})),
                        metadata=dict(item.get("metadata", {})),
                    )
                )

        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for record in normalized:
            row = record.to_dict()
            ts = self._normalize_timestamp(row["timestamp"])
            day = ts.strftime("%Y-%m-%d")
            event_type = _normalize_event_type(row["event_type"])
            row["event_type"] = event_type
            row["timestamp"] = _to_iso(row["timestamp"])
            row["visibility_time"] = _to_iso(row["visibility_time"])
            row["dataset_version"] = str(dataset_version)
            row["seed"] = int(seed) if seed is not None else None
            row["config_hash"] = str(config_hash or "")
            row["snapshot_id"] = str(snapshot_id or "")
            grouped.setdefault((event_type, day), []).append(row)

        for (event_type, day), rows in grouped.items():
            frame = pd.DataFrame(rows)
            partition = self._event_partition_dir(dataset_version, event_type, day)
            file_id = _stable_hash(
                {
                    "event_type": event_type,
                    "date": day,
                    "rows": len(rows),
                    "config_hash": config_hash,
                    "seed": seed,
                }
            )[:16]
            file_path = partition / f"events_{file_id}.parquet"
            frame.to_parquet(file_path, index=False)
            written.append(file_path)
        return written

    def _iter_parquet_files(self, dataset_version: str, event_types: Optional[Sequence[str]] = None) -> Iterator[Path]:
        root = self._dataset_root(dataset_version) / "events"
        if not root.exists():
            return iter(())
        target_types = {_normalize_event_type(x) for x in event_types} if event_types else None
        for file_path in root.rglob("*.parquet"):
            if target_types is None:
                yield file_path
                continue
            parent = file_path.parent.parent.name if file_path.parent.parent else ""
            if not parent.startswith("event_type="):
                continue
            file_type = parent.split("=", 1)[-1]
            if file_type in target_types:
                yield file_path

    def query_events(
        self,
        dataset_version: str,
        *,
        event_types: Optional[Sequence[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        visible_at: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> pd.DataFrame:
        files = list(self._iter_parquet_files(dataset_version, event_types=event_types))
        if not files:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "event_type",
                    "timestamp",
                    "visibility_time",
                    "source",
                    "confidence",
                    "payload_json",
                    "metadata_json",
                ]
            )
        frames = [pd.read_parquet(path) for path in files]
        frame = pd.concat(frames, axis=0, ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame["visibility_time"] = pd.to_datetime(frame["visibility_time"], errors="coerce", utc=True)
        if event_types:
            targets = {_normalize_event_type(x) for x in event_types}
            frame = frame[frame["event_type"].isin(targets)]
        if start_time:
            frame = frame[frame["timestamp"] >= self._normalize_timestamp(start_time)]
        if end_time:
            frame = frame[frame["timestamp"] <= self._normalize_timestamp(end_time)]
        if visible_at:
            frame = frame[frame["visibility_time"] <= self._normalize_timestamp(visible_at)]
        if source:
            frame = frame[frame["source"] == str(source)]
        frame = frame[frame["confidence"] >= float(min_confidence)]
        frame = frame.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        return frame

    def iter_visible_events(
        self,
        dataset_version: str,
        *,
        visible_at: str,
        event_types: Optional[Sequence[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> Iterator[EventRecord]:
        frame = self.query_events(
            dataset_version,
            event_types=event_types,
            start_time=start_time,
            end_time=end_time,
            visible_at=visible_at,
            min_confidence=min_confidence,
        )
        for _, row in frame.iterrows():
            yield EventRecord.from_row(row.to_dict())

    def load_market_bars(
        self,
        dataset_version: str,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        visible_at: Optional[str] = None,
    ) -> pd.DataFrame:
        frame = self.query_events(
            dataset_version,
            event_types=[EventType.MARKET_BAR.value],
            start_time=start_time,
            end_time=end_time,
            visible_at=visible_at,
        )
        if frame.empty:
            return frame
        payloads = frame["payload_json"].apply(lambda x: json.loads(str(x or "{}")))
        payload_frame = pd.json_normalize(payloads.tolist())
        out = pd.concat([frame.drop(columns=["payload_json", "metadata_json"]), payload_frame], axis=1)
        return out.reset_index(drop=True)

    def create_snapshot(
        self,
        dataset_version: str,
        *,
        seed: int,
        config_hash: str,
        feature_flags: Optional[Mapping[str, Any]] = None,
        scenario_id: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SnapshotManifest:
        frame = self.query_events(dataset_version)
        counts = frame["event_type"].value_counts().to_dict() if not frame.empty else {}
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        digest = _stable_hash(
            {
                "dataset_version": dataset_version,
                "seed": int(seed),
                "config_hash": str(config_hash),
                "event_counts": counts,
                "scenario_id": str(scenario_id or ""),
            }
        )[:16]
        snapshot_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{digest}"
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            dataset_version=str(dataset_version),
            timestamp=timestamp,
            seed=int(seed),
            config_hash=str(config_hash),
            feature_flags={str(k): bool(v) for k, v in dict(feature_flags or {}).items()},
            event_counts={str(k): int(v) for k, v in counts.items()},
            scenario_id=str(scenario_id or ""),
            metadata=dict(metadata or {}),
        )
        folder = self._dataset_root(dataset_version) / "snapshots" / snapshot_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def get_snapshot(self, dataset_version: str, snapshot_id: str) -> Optional[SnapshotManifest]:
        path = self._dataset_root(dataset_version) / "snapshots" / str(snapshot_id) / "manifest.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SnapshotManifest(**payload)

    def write_scenario_manifest(
        self,
        dataset_version: str,
        *,
        scenario_id: str,
        snapshot_id: str,
        start_time: str,
        end_time: str,
        seed: int,
        config_hash: str,
        event_types: Optional[Sequence[str]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        feature_flags: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ScenarioManifest:
        manifest = ScenarioManifest(
            scenario_id=str(scenario_id),
            dataset_version=str(dataset_version),
            snapshot_id=str(snapshot_id),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            seed=int(seed),
            config_hash=str(config_hash),
            start_time=_to_iso(start_time),
            end_time=_to_iso(end_time),
            event_types=[_normalize_event_type(x) for x in list(event_types or [])],
            filters=dict(filters or {}),
            feature_flags={str(k): bool(v) for k, v in dict(feature_flags or {}).items()},
            metadata=dict(metadata or {}),
        )
        path = self._dataset_root(dataset_version) / "scenarios" / f"{manifest.scenario_id}.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def get_scenario_manifest(self, dataset_version: str, scenario_id: str) -> Optional[ScenarioManifest]:
        path = self._dataset_root(dataset_version) / "scenarios" / f"{scenario_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ScenarioManifest(**payload)

    def query_scenario_events(self, dataset_version: str, scenario_id: str, *, visible_at: Optional[str] = None) -> pd.DataFrame:
        manifest = self.get_scenario_manifest(dataset_version, scenario_id)
        if manifest is None:
            return pd.DataFrame()
        return self.query_events(
            dataset_version,
            event_types=manifest.event_types or None,
            start_time=manifest.start_time,
            end_time=manifest.end_time,
            visible_at=visible_at,
        )


__all__ = [
    "EventType",
    "EventRecord",
    "EventStore",
    "ScenarioManifest",
    "SnapshotManifest",
]
