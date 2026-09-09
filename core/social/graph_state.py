"""Social graph state for sentiment diffusion, exposure tracking, and observation export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SocialNodeType(str, Enum):
    OFFICIAL_MEDIA = "official_media"
    FINANCIAL_MEDIA = "financial_media"
    REGULATOR_VOICE = "regulator_voice"
    BROKER_RESEARCH = "broker_research"
    SOCIAL_MEDIA = "social_media"
    KOL_SOCIAL = "kol_social"
    RUMOR_KOL = "rumor_kol"
    RETAIL_DAY_TRADER = "retail_day_trader"
    RETAIL = "retail_day_trader"
    INSTITUTION = "institution"
    RUMOR_SOURCE = "rumor_source"


DEFAULT_NODE_PROFILES: Dict[str, Dict[str, float]] = {
    SocialNodeType.OFFICIAL_MEDIA.value: {
        "source_credibility": 0.96,
        "propagation_delay": 0,
        "decay_rate": 0.12,
        "amplification": 1.20,
        "contradiction_sensitivity": 0.94,
        "news_exposure": 0.92,
        "social_exposure": 0.60,
        "rumor_sensitivity": 0.18,
    },
    SocialNodeType.REGULATOR_VOICE.value: {
        "source_credibility": 0.99,
        "propagation_delay": 0,
        "decay_rate": 0.08,
        "amplification": 1.32,
        "contradiction_sensitivity": 0.98,
        "news_exposure": 0.88,
        "social_exposure": 0.54,
        "rumor_sensitivity": 0.10,
    },
    SocialNodeType.FINANCIAL_MEDIA.value: {
        "source_credibility": 0.84,
        "propagation_delay": 1,
        "decay_rate": 0.14,
        "amplification": 1.16,
        "contradiction_sensitivity": 0.80,
        "news_exposure": 0.90,
        "social_exposure": 0.68,
        "rumor_sensitivity": 0.22,
    },
    SocialNodeType.BROKER_RESEARCH.value: {
        "source_credibility": 0.72,
        "propagation_delay": 1,
        "decay_rate": 0.16,
        "amplification": 1.08,
        "contradiction_sensitivity": 0.66,
        "news_exposure": 0.80,
        "social_exposure": 0.72,
        "rumor_sensitivity": 0.34,
    },
    SocialNodeType.SOCIAL_MEDIA.value: {
        "source_credibility": 0.38,
        "propagation_delay": 0,
        "decay_rate": 0.22,
        "amplification": 1.42,
        "contradiction_sensitivity": 0.36,
        "news_exposure": 0.50,
        "social_exposure": 0.96,
        "rumor_sensitivity": 0.82,
    },
    SocialNodeType.KOL_SOCIAL.value: {
        "source_credibility": 0.42,
        "propagation_delay": 0,
        "decay_rate": 0.24,
        "amplification": 1.62,
        "contradiction_sensitivity": 0.38,
        "news_exposure": 0.56,
        "social_exposure": 0.94,
        "rumor_sensitivity": 0.78,
    },
    SocialNodeType.RUMOR_KOL.value: {
        "source_credibility": 0.18,
        "propagation_delay": 0,
        "decay_rate": 0.28,
        "amplification": 1.76,
        "contradiction_sensitivity": 0.20,
        "news_exposure": 0.42,
        "social_exposure": 0.98,
        "rumor_sensitivity": 0.96,
    },
    SocialNodeType.RETAIL_DAY_TRADER.value: {
        "source_credibility": 0.30,
        "propagation_delay": 1,
        "decay_rate": 0.26,
        "amplification": 0.90,
        "contradiction_sensitivity": 0.42,
        "news_exposure": 0.48,
        "social_exposure": 0.88,
        "rumor_sensitivity": 0.72,
    },
    SocialNodeType.INSTITUTION.value: {
        "source_credibility": 0.78,
        "propagation_delay": 2,
        "decay_rate": 0.14,
        "amplification": 0.82,
        "contradiction_sensitivity": 0.74,
        "news_exposure": 0.86,
        "social_exposure": 0.52,
        "rumor_sensitivity": 0.28,
    },
    SocialNodeType.RUMOR_SOURCE.value: {
        "source_credibility": 0.12,
        "propagation_delay": 0,
        "decay_rate": 0.30,
        "amplification": 1.78,
        "contradiction_sensitivity": 0.18,
        "news_exposure": 0.34,
        "social_exposure": 0.98,
        "rumor_sensitivity": 1.00,
    },
}


def _profile_for(node_type: str | SocialNodeType) -> Dict[str, float]:
    key = str(node_type.value if isinstance(node_type, SocialNodeType) else node_type)
    return dict(DEFAULT_NODE_PROFILES.get(key, DEFAULT_NODE_PROFILES[SocialNodeType.RETAIL_DAY_TRADER.value]))


@dataclass(slots=True)
class GraphNodeState:
    """Node-level social state."""

    node_id: str
    node_type: str = SocialNodeType.RETAIL_DAY_TRADER.value
    sentiment: float = 0.0
    news_exposure: float = 0.4
    social_exposure: float = 0.5
    source_credibility: float = 0.5
    propagation_delay: int = 1
    decay_rate: float = 0.15
    amplification: float = 1.0
    contradiction_sensitivity: float = 0.5
    rumor_sensitivity: float = 0.5
    belief_strength: float = 0.0
    reach_score: float = 0.0
    narrative_heat: float = 0.0
    first_seen_tick: int = -1
    last_seen_tick: int = -1
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    observation_state: Dict[str, Any] = field(default_factory=dict)

    def apply_profile(self, node_type: str | SocialNodeType, **overrides: Any) -> None:
        self.node_type = str(node_type.value if isinstance(node_type, SocialNodeType) else node_type)
        profile = _profile_for(self.node_type)
        self.source_credibility = float(overrides.get("source_credibility", profile["source_credibility"]))
        self.propagation_delay = max(0, int(overrides.get("propagation_delay", profile["propagation_delay"])))
        self.decay_rate = float(overrides.get("decay_rate", profile["decay_rate"]))
        self.amplification = float(overrides.get("amplification", profile["amplification"]))
        self.contradiction_sensitivity = float(
            overrides.get("contradiction_sensitivity", profile["contradiction_sensitivity"])
        )
        self.news_exposure = float(overrides.get("news_exposure", profile["news_exposure"]))
        self.social_exposure = float(overrides.get("social_exposure", profile["social_exposure"]))
        self.rumor_sensitivity = float(overrides.get("rumor_sensitivity", profile["rumor_sensitivity"]))

    def record_event(self, event: Mapping[str, Any], *, limit: int = 8) -> None:
        payload = dict(event)
        self.recent_events.append(payload)
        if len(self.recent_events) > limit:
            self.recent_events = self.recent_events[-limit:]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["node_type"] = str(self.node_type)
        return payload


@dataclass(slots=True)
class EdgeProfile:
    """Typed social-edge channels for contagion."""

    trust_edge: float = 0.5
    position_similarity_edge: float = 0.5
    news_exposure_edge: float = 0.5
    institution_affiliation_edge: float = 0.0
    propagation_delay_edge: float = 1.0
    decay_edge: float = 1.0
    amplification_edge: float = 1.0
    contradiction_edge: float = 0.5


@dataclass(slots=True)
class SocialGraphState:
    """Container for social graph topology and dynamic sentiment states."""

    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    nodes: Dict[str, GraphNodeState] = field(default_factory=dict)
    edge_profiles: Dict[Tuple[str, str], EdgeProfile] = field(default_factory=dict)

    @staticmethod
    def _edge_key(src: str, dst: str) -> Tuple[str, str]:
        return (str(src), str(dst))

    def ensure_node(self, node_id: str, node_type: str | SocialNodeType | None = None) -> GraphNodeState:
        """Create or return an existing graph node."""
        key = str(node_id)
        if key not in self.nodes:
            node = GraphNodeState(node_id=key)
            node.apply_profile(node_type or SocialNodeType.RETAIL_DAY_TRADER.value)
            self.nodes[key] = node
        elif node_type is not None:
            self.nodes[key].apply_profile(node_type)
        self.adjacency.setdefault(key, [])
        return self.nodes[key]

    def set_node_profile(self, node_id: str, node_type: str | SocialNodeType, **overrides: Any) -> GraphNodeState:
        node = self.ensure_node(node_id, node_type)
        node.apply_profile(node_type, **overrides)
        return node

    def add_edge(self, src: str, dst: str, *, bidirectional: bool = True) -> None:
        """Add an edge between two nodes."""
        self.ensure_node(src)
        self.ensure_node(dst)
        if dst not in self.adjacency[src]:
            self.adjacency[src].append(dst)
        self.edge_profiles.setdefault(self._edge_key(src, dst), EdgeProfile())
        if bidirectional and src not in self.adjacency[dst]:
            self.adjacency[dst].append(src)
            self.edge_profiles.setdefault(self._edge_key(dst, src), EdgeProfile())

    def set_edge_profile(
        self,
        src: str,
        dst: str,
        *,
        trust_edge: float | None = None,
        position_similarity_edge: float | None = None,
        news_exposure_edge: float | None = None,
        institution_affiliation_edge: float | None = None,
        propagation_delay_edge: float | None = None,
        decay_edge: float | None = None,
        amplification_edge: float | None = None,
        contradiction_edge: float | None = None,
        bidirectional: bool = False,
    ) -> None:
        """Set edge profile fields; optionally mirror to reverse edge."""
        self.add_edge(src, dst, bidirectional=False)
        profile = self.edge_profiles.setdefault(self._edge_key(src, dst), EdgeProfile())
        if trust_edge is not None:
            profile.trust_edge = _clip(trust_edge, 0.0, 1.0)
        if position_similarity_edge is not None:
            profile.position_similarity_edge = _clip(position_similarity_edge, 0.0, 1.0)
        if news_exposure_edge is not None:
            profile.news_exposure_edge = _clip(news_exposure_edge, 0.0, 1.0)
        if institution_affiliation_edge is not None:
            profile.institution_affiliation_edge = _clip(institution_affiliation_edge, 0.0, 1.0)
        if propagation_delay_edge is not None:
            profile.propagation_delay_edge = float(max(0.0, propagation_delay_edge))
        if decay_edge is not None:
            profile.decay_edge = float(max(0.0, decay_edge))
        if amplification_edge is not None:
            profile.amplification_edge = float(max(0.0, amplification_edge))
        if contradiction_edge is not None:
            profile.contradiction_edge = _clip(contradiction_edge, 0.0, 1.0)

        if bidirectional:
            self.add_edge(dst, src, bidirectional=False)
            rev = self.edge_profiles.setdefault(self._edge_key(dst, src), EdgeProfile())
            rev.trust_edge = profile.trust_edge
            rev.position_similarity_edge = profile.position_similarity_edge
            rev.news_exposure_edge = profile.news_exposure_edge
            rev.institution_affiliation_edge = profile.institution_affiliation_edge
            rev.propagation_delay_edge = profile.propagation_delay_edge
            rev.decay_edge = profile.decay_edge
            rev.amplification_edge = profile.amplification_edge
            rev.contradiction_edge = profile.contradiction_edge

    def get_edge_profile(self, src: str, dst: str) -> EdgeProfile:
        """Return edge profile; fallback to default profile."""
        return self.edge_profiles.get(self._edge_key(src, dst), EdgeProfile())

    def set_sentiment(self, node_id: str, sentiment: float) -> None:
        """Set node sentiment in [-1, 1]."""
        node = self.ensure_node(node_id)
        node.sentiment = _clip(sentiment, -1.0, 1.0)

    def record_observation(self, node_id: str, event: Mapping[str, Any]) -> None:
        node = self.ensure_node(node_id)
        node.record_event(event)
        node.observation_state = dict(event)

    def build_observation_payload(
        self,
        node_id: str,
        *,
        current_tick: int = 0,
        topic: str | None = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        node = self.ensure_node(node_id)
        recent_events = node.recent_events[-limit:]
        payload = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "current_tick": int(current_tick),
            "sentiment": float(node.sentiment),
            "source_credibility": float(node.source_credibility),
            "propagation_delay": int(node.propagation_delay),
            "decay_rate": float(node.decay_rate),
            "amplification": float(node.amplification),
            "contradiction_sensitivity": float(node.contradiction_sensitivity),
            "rumor_sensitivity": float(node.rumor_sensitivity),
            "belief_strength": float(node.belief_strength),
            "reach_score": float(node.reach_score),
            "narrative_heat": float(node.narrative_heat),
            "first_seen_tick": int(node.first_seen_tick),
            "last_seen_tick": int(node.last_seen_tick),
            "topic": str(topic or node.observation_state.get("topic", "")),
            "recent_events": recent_events,
            "memory_seed": {
                "dominant_signal": node.observation_state.get("dominant_signal", 0.0),
                "rumor_pressure": node.observation_state.get("rumor_pressure", 0.0),
                "refutation_pressure": node.observation_state.get("refutation_pressure", 0.0),
                "source_credibility": node.observation_state.get("source_credibility", node.source_credibility),
                "first_seen_tick": node.first_seen_tick,
            },
        }
        node.observation_state = dict(payload)
        return payload

    def mean_sentiment(self) -> float:
        """Return average sentiment across nodes."""
        if not self.nodes:
            return 0.0
        return sum(node.sentiment for node in self.nodes.values()) / len(self.nodes)

    def neighbors(self, node_id: str) -> List[GraphNodeState]:
        """Return neighbor node states."""
        return [self.nodes[n] for n in self.adjacency.get(node_id, []) if n in self.nodes]

    def node_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts

    def graph_signature(self) -> str:
        payload = {
            "nodes": {node_id: node.to_dict() for node_id, node in sorted(self.nodes.items())},
            "edges": {
                f"{src}->{dst}": asdict(profile)
                for (src, dst), profile in sorted(self.edge_profiles.items())
            },
        }
        return _stable_hash(payload)

    @classmethod
    def ring(cls, node_ids: Iterable[str]) -> "SocialGraphState":
        """Build a simple ring graph as a deterministic default topology."""
        ids = [str(item) for item in node_ids]
        graph = cls()
        if not ids:
            return graph
        for item in ids:
            graph.ensure_node(item)
        if len(ids) == 1:
            return graph
        for idx, src in enumerate(ids):
            dst = ids[(idx + 1) % len(ids)]
            graph.add_edge(src, dst, bidirectional=True)
        return graph


if __name__ == "__main__":
    g = SocialGraphState.ring(["a", "b", "c"])
    g.set_node_profile("a", SocialNodeType.REGULATOR_VOICE)
    g.set_sentiment("a", -0.2)
    g.set_sentiment("b", 0.4)
    g.set_sentiment("c", 0.1)
    print("mean_sentiment", g.mean_sentiment())
