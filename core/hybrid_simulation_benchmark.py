"""Hybrid smart/vectorized simulation benchmark utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence
import hashlib
import json
import time
import tracemalloc

import numpy as np

from core.types import AgentDecisionEnvelope


@dataclass
class HybridBenchmarkConfig:
    agent_counts: Sequence[int] = (1000, 5000, 10000, 50000)
    steps: int = 8
    smart_ratio: float = 0.02
    batch_size: int = 64
    seed: int = 42
    feature_flags: Dict[str, bool] = field(
        default_factory=lambda: {
            "hybrid_scheduler_v1": True,
            "prompt_cache_v1": True,
            "batched_inference_v1": True,
            "state_compression_v1": True,
            "representative_cluster_v1": True,
        }
    )


@dataclass
class HybridBenchmarkRow:
    agent_count: int
    steps: int
    smart_agents: int
    vectorized_agents: int
    avg_step_latency_ms: float
    p95_step_latency_ms: float
    memory_peak_mb: float
    orders_per_step: float
    determinism_hash: str
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridSimulationBenchmark:
    """Synthetic but deterministic benchmark for hybrid scheduler evolution."""

    def __init__(self, config: HybridBenchmarkConfig | None = None) -> None:
        self.config = config or HybridBenchmarkConfig()

    @staticmethod
    def _determinism_hash(payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _compress_state(state: np.ndarray) -> np.ndarray:
        return state.astype(np.float32, copy=False)

    def _smart_inference(
        self,
        *,
        smart_count: int,
        step: int,
        rng: np.random.Generator,
        prompt_cache: Dict[str, AgentDecisionEnvelope],
    ) -> List[AgentDecisionEnvelope]:
        out: List[AgentDecisionEnvelope] = []
        for i in range(smart_count):
            archetype = f"smart-{i % 16}-step-{step % 4}"
            if self.config.feature_flags.get("prompt_cache_v1", True) and archetype in prompt_cache:
                cached = prompt_cache[archetype]
                out.append(
                    AgentDecisionEnvelope(
                        agent_id=f"smart-{i}",
                        symbol="A_SHARE_IDX",
                        action=cached.action,
                        side=cached.side,
                        target_qty=cached.target_qty,
                        urgency=cached.urgency,
                        confidence=cached.confidence,
                        order_type=cached.order_type,
                        metadata={"agent_kind": "smart", "cache_hit": True},
                    )
                )
                continue
            action = "BUY" if rng.random() > 0.5 else "SELL"
            envelope = AgentDecisionEnvelope(
                agent_id=f"smart-{i}",
                symbol="A_SHARE_IDX",
                action=action,
                side="buy" if action == "BUY" else "sell",
                target_qty=int(rng.integers(100, 2000)),
                urgency=float(rng.uniform(0.3, 0.95)),
                confidence=float(rng.uniform(0.4, 0.95)),
                order_type="limit",
                metadata={"agent_kind": "smart", "cache_hit": False},
            )
            prompt_cache[archetype] = envelope
            out.append(envelope)
        return out

    def _vectorized_inference(
        self,
        *,
        vector_count: int,
        step: int,
        rng: np.random.Generator,
    ) -> List[AgentDecisionEnvelope]:
        signs = rng.standard_normal(vector_count) + (0.05 if step % 2 == 0 else -0.05)
        qty = rng.integers(50, 500, size=vector_count)
        urgency = np.clip(np.abs(signs) * 0.6, 0.05, 0.75)
        confidence = np.clip(0.7 - np.abs(signs) * 0.2, 0.2, 0.9)
        out: List[AgentDecisionEnvelope] = []
        for i in range(vector_count):
            action = "BUY" if signs[i] >= 0 else "SELL"
            out.append(
                AgentDecisionEnvelope(
                    agent_id=f"vec-{i}",
                    symbol="A_SHARE_IDX",
                    action=action,
                    side="buy" if action == "BUY" else "sell",
                    target_qty=int(qty[i]),
                    urgency=float(urgency[i]),
                    confidence=float(confidence[i]),
                    order_type="limit",
                    metadata={"agent_kind": "vectorized"},
                )
            )
        return out

    def _run_single(self, agent_count: int) -> HybridBenchmarkRow:
        rng = np.random.default_rng(self.config.seed + agent_count)
        smart_count = max(1, int(agent_count * float(self.config.smart_ratio)))
        vector_count = max(0, int(agent_count - smart_count))

        prompt_cache: Dict[str, AgentDecisionEnvelope] = {}
        state = rng.normal(0.0, 1.0, size=(agent_count, 8))
        if self.config.feature_flags.get("state_compression_v1", True):
            state = self._compress_state(state)

        step_latencies: List[float] = []
        orders_total = 0
        determinism_payload: Dict[str, Any] = {"agent_count": agent_count, "steps": self.config.steps, "actions": []}

        tracemalloc.start()
        for step in range(int(self.config.steps)):
            t0 = time.perf_counter()
            smart_envelopes = self._smart_inference(
                smart_count=smart_count,
                step=step,
                rng=rng,
                prompt_cache=prompt_cache,
            )
            vector_envelopes = self._vectorized_inference(
                vector_count=vector_count,
                step=step,
                rng=rng,
            )
            envelopes = [*smart_envelopes, *vector_envelopes]
            orders_total += len(envelopes)

            if self.config.feature_flags.get("batched_inference_v1", True):
                batch_size = max(1, int(self.config.batch_size))
                for i in range(0, len(envelopes), batch_size):
                    _ = envelopes[i : i + batch_size]
            if self.config.feature_flags.get("representative_cluster_v1", True):
                _ = np.mean(state[:, :4], axis=0)

            determinism_payload["actions"].append(
                hashlib.sha256(
                    json.dumps(
                        [(e.action, e.side.value, e.target_qty) for e in envelopes[: min(200, len(envelopes))]],
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
            )
            step_latencies.append((time.perf_counter() - t0) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return HybridBenchmarkRow(
            agent_count=int(agent_count),
            steps=int(self.config.steps),
            smart_agents=int(smart_count),
            vectorized_agents=int(vector_count),
            avg_step_latency_ms=float(np.mean(step_latencies)) if step_latencies else 0.0,
            p95_step_latency_ms=float(np.percentile(step_latencies, 95)) if step_latencies else 0.0,
            memory_peak_mb=float(peak / (1024.0 * 1024.0)),
            orders_per_step=float(orders_total / max(1, int(self.config.steps))),
            determinism_hash=self._determinism_hash(determinism_payload),
            feature_flags=dict(self.config.feature_flags),
        )

    def run(self) -> Dict[str, Any]:
        rows = [self._run_single(int(count)).to_dict() for count in self.config.agent_counts]
        payload = {
            "seed": int(self.config.seed),
            "config_hash": self._determinism_hash(asdict(self.config)),
            "rows": rows,
            "feature_flags": dict(self.config.feature_flags),
        }
        return payload


__all__ = ["HybridBenchmarkConfig", "HybridBenchmarkRow", "HybridSimulationBenchmark"]

