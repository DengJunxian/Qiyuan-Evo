"""Reproducibility primitives for experiments, snapshots, and LLM calls.

The abstractions in this module are intentionally small and dependency-light.
They borrow the "message/replay identity" idea from ABIDES-style simulations
and the data-environment-agent-evaluation lineage used by FinRL, but keep the
implementation local so CI can run without network, GPUs, or API keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is present in normal installs
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is present in normal installs
    pd = None  # type: ignore[assignment]

DEFAULT_RANDOM_SEED = int(os.environ.get("CIVITAS_RANDOM_SEED", "42") or 42)
SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "bearer",
    "token",
    "secret",
    "password",
    "credential",
)
_AUTH_HEADER_RE = re.compile(r"authorization\s*:\s*bearer\s+[^\s,;]+", re.IGNORECASE)
_BEARER_RE = re.compile(r"bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(sk|api[_-]?key)[-_A-Za-z0-9]{8,}", re.IGNORECASE)


def _redact_sensitive(value: Any) -> str:
    text = str(value or "")
    text = _AUTH_HEADER_RE.sub("[REDACTED_AUTH_HEADER]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _API_KEY_RE.sub("[REDACTED_KEY]", text)
    return text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonicalize_for_hash(value: Any) -> Any:
    """Convert common Python objects into stable, JSON-hashable values."""

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(token in key_text.lower() for token in SENSITIVE_KEYWORDS):
                out[key_text] = "[REDACTED]"
            else:
                out[key_text] = canonicalize_for_hash(item)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [canonicalize_for_hash(item) for item in value]
    if pd is not None and isinstance(value, pd.DataFrame):
        return dataframe_fingerprint(value)
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return canonicalize_for_hash(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    return value


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(
        canonicalize_for_hash(payload),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, Any] | None) -> str:
    return stable_hash({"config": dict(config or {})})


def seed_everything(seed: int | None = None) -> int:
    """Seed the Python/numpy RNG stack and return the resolved seed.

    `PYTHONHASHSEED` only affects new interpreter processes, but setting it here
    still makes spawned child processes inherit the same value.
    """

    resolved = int(DEFAULT_RANDOM_SEED if seed is None else seed)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    if np is not None:
        np.random.seed(resolved)
    return resolved


def dataframe_fingerprint(frame: Any) -> dict[str, Any]:
    if pd is None or not isinstance(frame, pd.DataFrame):
        return {"type": type(frame).__name__, "hash": stable_hash(str(frame))}
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    csv_text = normalized.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return {
        "type": "dataframe",
        "rows": int(len(normalized)),
        "columns": [str(col) for col in normalized.columns],
        "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
    }


def dataset_snapshot_hash(dataset: Any, *, namespace: str = "dataset") -> str:
    """Hash a dataset snapshot without persisting raw rows in experiment metadata."""

    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if path.exists() and path.is_file():
            return stable_hash(
                {
                    "namespace": namespace,
                    "path_name": path.name,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return stable_hash({"namespace": namespace, "fingerprint": canonicalize_for_hash(dataset)})


def parameter_set_id(params: Mapping[str, Any] | None, *, namespace: str = "calibration") -> str:
    digest = stable_hash({"namespace": namespace, "params": dict(params or {})})
    return f"pset_{digest[:16]}"


def experiment_id(
    *,
    module: str,
    config_hash_value: str,
    data_snapshot_hash_value: str,
    seed: int,
    parameter_set_id_value: str = "",
) -> str:
    digest = stable_hash(
        {
            "module": str(module),
            "config_hash": str(config_hash_value),
            "data_snapshot_hash": str(data_snapshot_hash_value),
            "seed": int(seed),
            "parameter_set_id": str(parameter_set_id_value or ""),
        }
    )
    return f"exp_{digest[:20]}"


def safe_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """Log-safe LLM call metadata.

    The record stores provider/model/routing/latency and deliberately excludes
    prompts, completions, API keys, headers, and tool payloads.
    """

    provider: str
    model: str
    fallback_chain: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    ok: bool = True
    error_type: str = ""
    cache_hit: bool = False
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["error_type"] = _redact_sensitive(payload.get("error_type", ""))
        return payload


def llm_call_record_from_response(response: Any, *, error_type: str = "", cache_hit: bool = False) -> LLMCallRecord:
    return LLMCallRecord(
        provider=str(getattr(response, "provider", "unknown") or "unknown"),
        model=str(getattr(response, "model", "unknown") or "unknown"),
        fallback_chain=[str(item) for item in list(getattr(response, "fallback_chain", []) or [])],
        latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
        ok=bool(getattr(response, "ok", False)),
        error_type=_redact_sensitive(error_type),
        cache_hit=bool(cache_hit),
    )


@dataclass(frozen=True, slots=True)
class ReproducibilityEnvelope:
    experiment_id: str
    config_hash: str
    data_snapshot_hash: str
    parameter_set_id: str
    random_seed: int
    git_commit: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_reproducibility_envelope(
    *,
    module: str,
    config: Mapping[str, Any] | None,
    dataset: Any,
    parameters: Mapping[str, Any] | None = None,
    seed: int | None = None,
    llm_calls: Sequence[LLMCallRecord | Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> ReproducibilityEnvelope:
    resolved_seed = seed_everything(seed)
    cfg_hash = config_hash(config)
    snap_hash = dataset_snapshot_hash(dataset)
    pset_id = parameter_set_id(parameters)
    exp_id = experiment_id(
        module=module,
        config_hash_value=cfg_hash,
        data_snapshot_hash_value=snap_hash,
        seed=resolved_seed,
        parameter_set_id_value=pset_id,
    )
    serialized_calls: list[dict[str, Any]] = []
    for call in list(llm_calls or []):
        if isinstance(call, LLMCallRecord):
            serialized_calls.append(call.to_dict())
        else:
            serialized_calls.append(canonicalize_for_hash(dict(call)))
    return ReproducibilityEnvelope(
        experiment_id=exp_id,
        config_hash=cfg_hash,
        data_snapshot_hash=snap_hash,
        parameter_set_id=pset_id,
        random_seed=resolved_seed,
        git_commit=safe_git_commit(),
        llm_calls=serialized_calls,
        extra=canonicalize_for_hash(dict(extra or {})),
    )


def replay_signature(payload: Any) -> str:
    """Compact digest for deterministic replay assertions."""

    return stable_hash({"replay": payload})


__all__ = [
    "DEFAULT_RANDOM_SEED",
    "LLMCallRecord",
    "ReproducibilityEnvelope",
    "build_reproducibility_envelope",
    "canonicalize_for_hash",
    "config_hash",
    "dataframe_fingerprint",
    "dataset_snapshot_hash",
    "experiment_id",
    "llm_call_record_from_response",
    "parameter_set_id",
    "replay_signature",
    "safe_git_commit",
    "seed_everything",
    "stable_hash",
    "stable_json_dumps",
    "utc_now_iso",
]
