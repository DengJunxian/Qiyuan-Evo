"""Evolution and ecology metrics helpers for multi-agent market simulation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class StrategyGenome:
    """Compact strategy genome used by evolutionary operators."""

    analyst_weight_vector: Tuple[float, float, float]
    memory_span: int
    stop_loss_threshold: float
    order_aggressiveness: float
    social_susceptibility: float
    risk_aversion: float
    debate_participation: float

    @staticmethod
    def random(rng: random.Random | None = None) -> "StrategyGenome":
        rng = rng or random.Random()
        raw = [rng.uniform(0.1, 1.0) for _ in range(3)]
        total = sum(raw) or 1.0
        weights = (raw[0] / total, raw[1] / total, raw[2] / total)
        return StrategyGenome(
            analyst_weight_vector=weights,
            memory_span=int(rng.randint(8, 120)),
            stop_loss_threshold=float(-rng.uniform(0.02, 0.20)),
            order_aggressiveness=float(rng.uniform(0.05, 0.95)),
            social_susceptibility=float(rng.uniform(0.05, 0.95)),
            risk_aversion=float(rng.uniform(0.05, 0.95)),
            debate_participation=float(rng.uniform(0.0, 1.0)),
        )

    def normalized_weights(self) -> Tuple[float, float, float]:
        w1, w2, w3 = [max(1e-9, float(v)) for v in self.analyst_weight_vector]
        total = w1 + w2 + w3
        return (w1 / total, w2 / total, w3 / total)

    def crossover(self, other: "StrategyGenome", rng: random.Random | None = None) -> "StrategyGenome":
        rng = rng or random.Random()
        alpha = rng.uniform(0.25, 0.75)
        wa = self.normalized_weights()
        wb = other.normalized_weights()
        child_w = tuple(alpha * a + (1.0 - alpha) * b for a, b in zip(wa, wb))
        return StrategyGenome(
            analyst_weight_vector=child_w,
            memory_span=int(round(alpha * self.memory_span + (1.0 - alpha) * other.memory_span)),
            stop_loss_threshold=float(alpha * self.stop_loss_threshold + (1.0 - alpha) * other.stop_loss_threshold),
            order_aggressiveness=float(alpha * self.order_aggressiveness + (1.0 - alpha) * other.order_aggressiveness),
            social_susceptibility=float(alpha * self.social_susceptibility + (1.0 - alpha) * other.social_susceptibility),
            risk_aversion=float(alpha * self.risk_aversion + (1.0 - alpha) * other.risk_aversion),
            debate_participation=float(alpha * self.debate_participation + (1.0 - alpha) * other.debate_participation),
        ).mutated(mutation_rate=0.0, mutation_scale=0.0, rng=rng)

    def mutated(
        self,
        *,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.10,
        rng: random.Random | None = None,
    ) -> "StrategyGenome":
        rng = rng or random.Random()

        def maybe_mut(value: float, lo: float, hi: float) -> float:
            if rng.random() > mutation_rate:
                return float(value)
            span = (hi - lo) * mutation_scale
            return _clip(float(value) + rng.uniform(-span, span), lo, hi)

        w = list(self.normalized_weights())
        for i in range(3):
            w[i] = maybe_mut(w[i], 0.01, 0.98)
        ws = sum(w) or 1.0
        weights = (w[0] / ws, w[1] / ws, w[2] / ws)

        memory = int(round(maybe_mut(float(self.memory_span), 4.0, 256.0)))
        stop_loss = maybe_mut(self.stop_loss_threshold, -0.40, -0.005)

        return StrategyGenome(
            analyst_weight_vector=weights,
            memory_span=max(4, memory),
            stop_loss_threshold=float(stop_loss),
            order_aggressiveness=maybe_mut(self.order_aggressiveness, 0.01, 1.00),
            social_susceptibility=maybe_mut(self.social_susceptibility, 0.01, 1.00),
            risk_aversion=maybe_mut(self.risk_aversion, 0.01, 1.00),
            debate_participation=maybe_mut(self.debate_participation, 0.00, 1.00),
        )

    def diffuse_from(self, peer: "StrategyGenome", strength: float = 0.08) -> "StrategyGenome":
        beta = _clip(strength, 0.0, 1.0)
        w0 = self.normalized_weights()
        w1 = peer.normalized_weights()
        blended_w = tuple((1.0 - beta) * a + beta * b for a, b in zip(w0, w1))
        return StrategyGenome(
            analyst_weight_vector=blended_w,
            memory_span=int(round((1.0 - beta) * self.memory_span + beta * peer.memory_span)),
            stop_loss_threshold=(1.0 - beta) * self.stop_loss_threshold + beta * peer.stop_loss_threshold,
            order_aggressiveness=(1.0 - beta) * self.order_aggressiveness + beta * peer.order_aggressiveness,
            social_susceptibility=(1.0 - beta) * self.social_susceptibility + beta * peer.social_susceptibility,
            risk_aversion=(1.0 - beta) * self.risk_aversion + beta * peer.risk_aversion,
            debate_participation=(1.0 - beta) * self.debate_participation + beta * peer.debate_participation,
        )

    def signature_key(self, precision: int = 1) -> str:
        w = self.normalized_weights()
        return "|".join(
            [
                f"w:{round(w[0], precision)},{round(w[1], precision)},{round(w[2], precision)}",
                f"mem:{int(round(self.memory_span, -1))}",
                f"sl:{round(self.stop_loss_threshold, precision + 1)}",
                f"agg:{round(self.order_aggressiveness, precision)}",
                f"soc:{round(self.social_susceptibility, precision)}",
                f"ra:{round(self.risk_aversion, precision)}",
                f"deb:{round(self.debate_participation, precision)}",
            ]
        )


class EvolutionOperators:
    """Selection / crossover / mutation / diffusion operators."""

    def __init__(self, mutation_rate: float = 0.15, mutation_scale: float = 0.10, seed: int | None = None) -> None:
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.rng = random.Random(seed)

    def selection(self, score_map: Mapping[str, float], survival_rate: float = 0.5) -> List[str]:
        if not score_map:
            return []
        keep = max(1, int(math.ceil(len(score_map) * _clip(survival_rate, 0.05, 1.0))))
        ranked = sorted(score_map.items(), key=lambda kv: float(kv[1]), reverse=True)
        return [agent_id for agent_id, _ in ranked[:keep]]

    def crossover(self, parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
        return parent_a.crossover(parent_b, rng=self.rng)

    def mutation(self, genome: StrategyGenome) -> StrategyGenome:
        return genome.mutated(
            mutation_rate=self.mutation_rate,
            mutation_scale=self.mutation_scale,
            rng=self.rng,
        )

    def local_diffusion(
        self,
        genome_map: MutableMapping[str, StrategyGenome],
        adjacency: Mapping[str, Sequence[str]],
        strength: float = 0.08,
    ) -> Dict[str, StrategyGenome]:
        updated: Dict[str, StrategyGenome] = {}
        for node_id, genome in genome_map.items():
            neighbors = [n for n in adjacency.get(node_id, []) if n in genome_map]
            if not neighbors:
                updated[node_id] = genome
                continue
            peer_id = self.rng.choice(neighbors)
            updated[node_id] = genome.diffuse_from(genome_map[peer_id], strength=strength)
        genome_map.update(updated)
        return updated


@dataclass(slots=True)
class StrategyEcologyConfig:
    """Discrete FinEvo-style population dynamics knobs."""

    selection_beta: float = 0.35
    innovation_mu: float = 0.08
    perturbation_gamma: float = 0.12
    diversity_floor: float = 0.03
    min_share: float = 0.002
    noise_scale: float = 0.05
    transition_threshold: float = 0.025
    top_cohort_bias: float = 0.20
    seed: int = 0
    version: str = "strategy_ecology_v1"

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]] = None) -> "StrategyEcologyConfig":
        if payload is None:
            return cls()
        allowed = {
            "selection_beta",
            "innovation_mu",
            "perturbation_gamma",
            "diversity_floor",
            "min_share",
            "noise_scale",
            "transition_threshold",
            "top_cohort_bias",
            "seed",
            "version",
        }
        kwargs: Dict[str, Any] = {key: value for key, value in dict(payload).items() if key in allowed}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StrategyCohortState:
    """One strategy cohort in the population simplex."""

    cohort_id: str
    label: str
    agent_count: int
    population_share: float
    capital_share: float
    fitness: float
    mean_wealth_return: float
    target_share: float
    selection_force: float = 0.0
    innovation_force: float = 0.0
    perturbation_force: float = 0.0
    representative_genome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StrategyEcologySnapshot:
    """One strategy ecology observation and force decomposition."""

    tick: int
    cohorts: Dict[str, StrategyCohortState]
    dominant_cohort: Dict[str, Any]
    force_breakdown: Dict[str, Any]
    transition_events: List[Dict[str, Any]]
    policy_regime: str
    config_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": int(self.tick),
            "cohorts": {key: value.to_dict() for key, value in self.cohorts.items()},
            "dominant_cohort": dict(self.dominant_cohort),
            "force_breakdown": dict(self.force_breakdown),
            "transition_events": list(self.transition_events),
            "policy_regime": str(self.policy_regime),
            "config_hash": str(self.config_hash),
            "metadata": dict(self.metadata),
        }


class StrategyEcologyEngine:
    """Track strategy-cohort shares with selection, innovation, and perturbation."""

    def __init__(self, config: Optional[StrategyEcologyConfig | Mapping[str, Any]] = None) -> None:
        if isinstance(config, StrategyEcologyConfig):
            self.config = config
        else:
            self.config = StrategyEcologyConfig.from_mapping(config)
        self.rng = random.Random(int(self.config.seed))
        self.previous_shares: Dict[str, float] = {}
        self.snapshots: List[StrategyEcologySnapshot] = []
        self.config_hash = _stable_hash(self.config.to_dict())

    @staticmethod
    def _normalize(values: Mapping[str, float], *, floor: float = 0.0) -> Dict[str, float]:
        if not values:
            return {}
        floored = {str(k): max(float(floor), max(0.0, float(v))) for k, v in values.items()}
        total = sum(floored.values())
        if total <= 1e-12:
            n = float(max(1, len(floored)))
            return {key: 1.0 / n for key in floored}
        return {key: float(value / total) for key, value in floored.items()}

    @staticmethod
    def _softmax(values: Mapping[str, float], *, temperature: float = 0.10) -> Dict[str, float]:
        if not values:
            return {}
        scale = max(float(temperature), 1e-6)
        max_val = max(float(v) for v in values.values())
        exps = {key: math.exp((float(value) - max_val) / scale) for key, value in values.items()}
        total = sum(exps.values()) or 1.0
        return {key: float(value / total) for key, value in exps.items()}

    @staticmethod
    def _cohort_policy_bias(cohort_id: str, context: Mapping[str, Any]) -> float:
        key = str(cohort_id).lower()
        regime = str(context.get("policy_regime", context.get("direction", "neutral")) or "neutral").lower()
        rumor_pressure = float(context.get("rumor_pressure", 0.0) or 0.0)
        sentiment_shift = float(context.get("sentiment_shift", 0.0) or 0.0)
        affected = {str(item).lower() for item in context.get("affected_cohorts", []) or []}
        bias = 0.0
        if "rumor" in key:
            bias += 0.30 * rumor_pressure
        if "retail" in key or "day" in key or "swing" in key:
            bias += 0.16 * rumor_pressure + 0.10 * sentiment_shift
        if "state" in key or "stabilization" in key or "pension" in key or "insurer" in key:
            bias -= 0.08 * rumor_pressure
            if regime in {"stabilizing", "supportive"}:
                bias += 0.14
        if "market_maker" in key or "etf" in key:
            bias -= 0.05 * abs(rumor_pressure)
            if regime in {"mixed", "neutral"}:
                bias += 0.03
        if key in affected or any(token and token in key for token in affected):
            bias += 0.12 if regime != "restrictive" else -0.08
        if regime == "restrictive" and not ("rumor" in key or "state" in key):
            bias -= 0.05
        if regime == "supportive" and ("retail" in key or "prop" in key):
            bias += 0.06
        return float(_clip(bias, -0.35, 0.35))

    def update(
        self,
        *,
        tick: int,
        cohort_records: Sequence[Mapping[str, Any]],
        policy_context: Optional[Mapping[str, Any]] = None,
    ) -> StrategyEcologySnapshot:
        context = dict(policy_context or {})
        policy_regime = str(context.get("policy_regime", context.get("direction", "neutral")) or "neutral")
        records = [dict(item) for item in cohort_records if str(item.get("cohort_id", "")).strip()]
        if not records:
            snapshot = StrategyEcologySnapshot(
                tick=int(tick),
                cohorts={},
                dominant_cohort={},
                force_breakdown={"selection": 0.0, "innovation": 0.0, "perturbation": 0.0},
                transition_events=[],
                policy_regime=policy_regime,
                config_hash=self.config_hash,
                metadata={"status": "empty"},
            )
            self.snapshots.append(snapshot)
            return snapshot

        counts: Dict[str, int] = {}
        capital: Dict[str, float] = {}
        returns: Dict[str, List[float]] = {}
        labels: Dict[str, str] = {}
        genomes: Dict[str, Dict[str, int]] = {}
        for item in records:
            cid = str(item.get("cohort_id", "") or "").strip()
            labels[cid] = str(item.get("label", cid) or cid)
            counts[cid] = counts.get(cid, 0) + 1
            capital[cid] = capital.get(cid, 0.0) + max(0.0, float(item.get("capital", 0.0) or 0.0))
            returns.setdefault(cid, []).append(float(item.get("wealth_return", 0.0) or 0.0))
            genome = str(item.get("genome_signature", "") or "")
            if genome:
                genomes.setdefault(cid, {})[genome] = genomes.setdefault(cid, {}).get(genome, 0) + 1

        total_count = float(sum(counts.values()) or 1.0)
        observed_shares = {cid: count / total_count for cid, count in counts.items()}
        current_shares = dict(self.previous_shares) if self.previous_shares else dict(observed_shares)
        for cid, share in observed_shares.items():
            current_shares[cid] = 0.55 * float(current_shares.get(cid, share)) + 0.45 * float(share)
        current_shares = self._normalize(current_shares, floor=0.0)

        capital_share = self._normalize(capital, floor=0.0)
        fitness = {
            cid: float(sum(samples) / max(1, len(samples)))
            for cid, samples in returns.items()
        }
        for cid in current_shares:
            fitness.setdefault(cid, 0.0)

        mean_fit = sum(current_shares.get(cid, 0.0) * fitness.get(cid, 0.0) for cid in current_shares)
        selection_force = {
            cid: float(self.config.selection_beta * current_shares[cid] * (fitness.get(cid, 0.0) - mean_fit))
            for cid in current_shares
        }

        soft = self._softmax(fitness)
        n = max(1, len(current_shares))
        floor_total = min(0.85, float(self.config.diversity_floor) * n)
        innovation_target = {
            cid: float(self.config.diversity_floor + (1.0 - floor_total) * soft.get(cid, 1.0 / n))
            for cid in current_shares
        }
        innovation_target = self._normalize(innovation_target)
        innovation_force = {
            cid: float(self.config.innovation_mu * (innovation_target.get(cid, 0.0) - current_shares[cid]))
            for cid in current_shares
        }

        volatility = abs(float(context.get("sentiment_volatility", 0.0) or 0.0))
        policy_strength = abs(float(context.get("policy_intensity", context.get("aggregate_strength", 0.0)) or 0.0))
        perturb_raw: Dict[str, float] = {}
        for cid in current_shares:
            noise = self.rng.normalvariate(0.0, float(self.config.noise_scale))
            bias = self._cohort_policy_bias(cid, context)
            scale = 1.0 + 0.35 * policy_strength + 0.50 * volatility
            perturb_raw[cid] = current_shares[cid] * (bias + noise) * scale
        raw_total = sum(perturb_raw.values())
        perturbation_force = {
            cid: float(self.config.perturbation_gamma * (perturb_raw[cid] - current_shares[cid] * raw_total))
            for cid in current_shares
        }

        target = {
            cid: current_shares[cid]
            + selection_force.get(cid, 0.0)
            + innovation_force.get(cid, 0.0)
            + perturbation_force.get(cid, 0.0)
            for cid in current_shares
        }
        target = self._normalize(target, floor=float(self.config.min_share))

        transition_events: List[Dict[str, Any]] = []
        for cid, share in target.items():
            old_share = current_shares.get(cid, 0.0)
            delta = float(share - old_share)
            if abs(delta) >= float(self.config.transition_threshold):
                transition_events.append(
                    {
                        "cohort_id": cid,
                        "label": labels.get(cid, cid),
                        "from_share": float(old_share),
                        "to_share": float(share),
                        "delta": delta,
                        "direction": "expand" if delta > 0 else "contract",
                    }
                )

        cohorts: Dict[str, StrategyCohortState] = {}
        for cid, share in sorted(target.items(), key=lambda item: item[1], reverse=True):
            rep_genome = ""
            if cid in genomes and genomes[cid]:
                rep_genome = max(genomes[cid].items(), key=lambda item: item[1])[0]
            cohorts[cid] = StrategyCohortState(
                cohort_id=cid,
                label=labels.get(cid, cid),
                agent_count=int(counts.get(cid, 0)),
                population_share=float(current_shares.get(cid, 0.0)),
                capital_share=float(capital_share.get(cid, 0.0)),
                fitness=float(fitness.get(cid, 0.0)),
                mean_wealth_return=float(fitness.get(cid, 0.0)),
                target_share=float(share),
                selection_force=float(selection_force.get(cid, 0.0)),
                innovation_force=float(innovation_force.get(cid, 0.0)),
                perturbation_force=float(perturbation_force.get(cid, 0.0)),
                representative_genome=rep_genome,
            )

        dominant_id = max(cohorts, key=lambda cid: cohorts[cid].target_share) if cohorts else ""
        dominant = cohorts[dominant_id].to_dict() if dominant_id else {}
        force_breakdown = {
            "selection": float(sum(abs(v) for v in selection_force.values())),
            "innovation": float(sum(abs(v) for v in innovation_force.values())),
            "perturbation": float(sum(abs(v) for v in perturbation_force.values())),
            "by_cohort": {
                cid: {
                    "selection": float(selection_force.get(cid, 0.0)),
                    "innovation": float(innovation_force.get(cid, 0.0)),
                    "perturbation": float(perturbation_force.get(cid, 0.0)),
                    "current_share": float(current_shares.get(cid, 0.0)),
                    "target_share": float(target.get(cid, 0.0)),
                }
                for cid in cohorts
            },
        }
        snapshot = StrategyEcologySnapshot(
            tick=int(tick),
            cohorts=cohorts,
            dominant_cohort=dominant,
            force_breakdown=force_breakdown,
            transition_events=transition_events,
            policy_regime=policy_regime,
            config_hash=self.config_hash,
            metadata={
                "version": self.config.version,
                "cohort_count": len(cohorts),
                "agent_count": int(sum(counts.values())),
                "policy_context": context,
            },
        )
        self.previous_shares = {cid: state.target_share for cid, state in cohorts.items()}
        self.snapshots.append(snapshot)
        return snapshot

    def latest(self) -> Optional[StrategyEcologySnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    def save_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "tick",
                    "cohort_id",
                    "label",
                    "agent_count",
                    "population_share",
                    "capital_share",
                    "fitness",
                    "target_share",
                    "selection_force",
                    "innovation_force",
                    "perturbation_force",
                    "policy_regime",
                    "dominant",
                    "config_hash",
                ]
            )
            for snapshot in self.snapshots:
                dominant = str(snapshot.dominant_cohort.get("cohort_id", ""))
                for cohort in snapshot.cohorts.values():
                    writer.writerow(
                        [
                            int(snapshot.tick),
                            cohort.cohort_id,
                            cohort.label,
                            int(cohort.agent_count),
                            float(cohort.population_share),
                            float(cohort.capital_share),
                            float(cohort.fitness),
                            float(cohort.target_share),
                            float(cohort.selection_force),
                            float(cohort.innovation_force),
                            float(cohort.perturbation_force),
                            snapshot.policy_regime,
                            cohort.cohort_id == dominant,
                            snapshot.config_hash,
                        ]
                    )
        return target

    def save_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "report_type": "strategy_ecology",
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "latest": self.latest().to_dict() if self.latest() is not None else {},
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }
        payload["report_hash"] = _stable_hash(
            {
                "config_hash": self.config_hash,
                "snapshot_count": len(self.snapshots),
                "latest_tick": payload["latest"].get("tick", 0) if isinstance(payload["latest"], dict) else 0,
            }
        )
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target


def entropy_from_labels(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts: Dict[str, int] = {}
    for item in labels:
        counts[item] = counts.get(item, 0) + 1
    n = float(len(labels))
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 1e-12:
            entropy -= p * math.log(p)
    return float(entropy)


def hhi_from_shares(shares: Sequence[float]) -> float:
    vals = [max(0.0, float(x)) for x in shares]
    s = sum(vals)
    if s <= 1e-12:
        return 0.0
    return float(sum((v / s) ** 2 for v in vals))


def build_sentiment_coalitions(sentiments: Mapping[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for node_id, value in sentiments.items():
        s = float(value)
        if s >= 0.2:
            out[node_id] = 1
        elif s <= -0.2:
            out[node_id] = -1
        else:
            out[node_id] = 0
    return out


def coalition_persistence(prev: Mapping[str, int], curr: Mapping[str, int]) -> float:
    common = set(prev.keys()) & set(curr.keys())
    if not common:
        return 0.0
    match = sum(1 for node_id in common if int(prev[node_id]) == int(curr[node_id]))
    return float(match / max(1, len(common)))


def approx_modularity(adjacency: Mapping[str, Sequence[str]], coalition: Mapping[str, int]) -> float:
    total_edges = 0
    intra_edges = 0
    for src, neighbors in adjacency.items():
        src_group = coalition.get(src, 0)
        for dst in neighbors:
            if src >= dst:
                continue
            total_edges += 1
            if coalition.get(dst, 0) == src_group:
                intra_edges += 1
    if total_edges == 0:
        return 0.0
    base = 1.0 / max(1.0, float(len(set(coalition.values()))))
    return float((intra_edges / total_edges) - base)


def phase_change_score(prev_entropy: float, curr_entropy: float, prev_return: float, curr_return: float) -> float:
    entropy_jump = abs(float(curr_entropy) - float(prev_entropy))
    return_jump = abs(float(curr_return) - float(prev_return))
    return float(0.7 * entropy_jump + 0.3 * return_jump)


@dataclass(slots=True)
class EcologyMetricsRow:
    tick: int
    entropy: float
    hhi: float
    modularity: float
    phase_changes: float
    coalition_persistence: float


@dataclass(slots=True)
class EcologyMetricsTracker:
    rows: List[EcologyMetricsRow] = field(default_factory=list)

    def record(self, row: EcologyMetricsRow) -> None:
        self.rows.append(row)

    def latest(self) -> Dict[str, float]:
        if not self.rows:
            return {
                "entropy": 0.0,
                "hhi": 0.0,
                "modularity": 0.0,
                "phase_changes": 0.0,
                "coalition_persistence": 0.0,
            }
        r = self.rows[-1]
        return {
            "entropy": float(r.entropy),
            "hhi": float(r.hhi),
            "modularity": float(r.modularity),
            "phase_changes": float(r.phase_changes),
            "coalition_persistence": float(r.coalition_persistence),
        }

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "entropy", "hhi", "modularity", "phase_changes", "coalition_persistence"])
            for row in self.rows:
                writer.writerow(
                    [
                        int(row.tick),
                        float(row.entropy),
                        float(row.hhi),
                        float(row.modularity),
                        float(row.phase_changes),
                        float(row.coalition_persistence),
                    ]
                )
        return p
