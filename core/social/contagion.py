"""Social contagion engine for sentiment diffusion and propagation reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.macro.state import MacroState
from core.social.graph_state import DEFAULT_NODE_PROFILES, GraphNodeState, SocialGraphState, SocialNodeType


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _profile_for(node_type: str | SocialNodeType) -> Dict[str, float]:
    key = str(node_type.value if isinstance(node_type, SocialNodeType) else node_type)
    return dict(DEFAULT_NODE_PROFILES.get(key, DEFAULT_NODE_PROFILES[SocialNodeType.RETAIL_DAY_TRADER.value]))


def _reliability_band(credibility: float) -> str:
    if credibility >= 0.8:
        return "high"
    if credibility >= 0.45:
        return "medium"
    return "low"


@dataclass(slots=True)
class SocialMessage:
    """Structured social message used by the enriched propagation engine."""

    topic: str
    source_id: str
    source_type: str
    message_id: str = ""
    kind: str = "rumor"
    polarity: float = 0.0
    strength: float = 1.0
    credibility: float = 0.5
    created_tick: int = 0
    scheduled_tick: int = 0
    decay: float = 0.1
    amplification: float = 1.0
    audience_tags: List[str] = field(default_factory=list)
    coverage_ratio: float = 1.0
    source_reliability_band: str = "medium"
    diffusion_velocity: float = 1.0
    rebuttal_of: str = ""
    parent_message_id: str = ""
    root_message_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SocialPost:
    """Social-media post used by the recommendation and cascade layer."""

    post_id: str
    author_id: str
    topic: str
    source_type: str = SocialNodeType.SOCIAL_MEDIA.value
    kind: str = "opinion"
    polarity: float = 0.0
    strength: float = 1.0
    credibility: float = 0.5
    created_tick: int = 0
    upvotes: int = 0
    downvotes: int = 0
    repost_of: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SocialPost":
        return cls(
            post_id=str(payload.get("post_id", payload.get("id", f"post_{uuid.uuid4().hex[:8]}"))),
            author_id=str(payload.get("author_id", payload.get("source_id", "synthetic_source"))),
            topic=str(payload.get("topic", "market")),
            source_type=str(payload.get("source_type", SocialNodeType.SOCIAL_MEDIA.value)),
            kind=str(payload.get("kind", "opinion")),
            polarity=float(payload.get("polarity", 0.0)),
            strength=float(payload.get("strength", 1.0)),
            credibility=float(payload.get("credibility", 0.5)),
            created_tick=int(payload.get("created_tick", 0)),
            upvotes=int(payload.get("upvotes", payload.get("likes", 0)) or 0),
            downvotes=int(payload.get("downvotes", payload.get("unlikes", 0)) or 0),
            repost_of=str(payload.get("repost_of", "")),
            content=str(payload.get("content", payload.get("raw_text", ""))),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_message(self, *, target_id: str, current_tick: int, hot_score: float, rank: int) -> SocialMessage:
        chain_depth = int(self.metadata.get("cascade_depth", 1) or 1)
        if self.repost_of:
            chain_depth = max(chain_depth, 2)
        return SocialMessage(
            message_id=f"rec_{self.post_id}_{target_id}_{current_tick}",
            topic=self.topic,
            source_id=self.author_id,
            source_type=self.source_type,
            kind=self.kind,
            polarity=float(self.polarity),
            strength=float(max(0.05, self.strength) * (0.55 + min(1.0, hot_score))),
            credibility=float(_clip(self.credibility, 0.0, 1.0)),
            created_tick=int(self.created_tick),
            scheduled_tick=int(current_tick),
            decay=0.16,
            amplification=1.0 + min(1.0, hot_score),
            coverage_ratio=0.5,
            source_reliability_band=_reliability_band(float(self.credibility)),
            diffusion_velocity=1.0,
            parent_message_id=str(self.repost_of),
            root_message_id=str(self.repost_of or self.post_id),
            metadata={
                **dict(self.metadata),
                "recommendation": True,
                "recommendation_target": str(target_id),
                "source_post_id": self.post_id,
                "hot_score": float(hot_score),
                "recommendation_rank": int(rank),
                "cascade_depth": int(chain_depth),
            },
        )


@dataclass(slots=True)
class MediaRecommendationEngine:
    """TwinMarket-inspired hot-score recommender for social exposure."""

    top_k: int = 3
    similarity_threshold: float = 0.18
    half_life_ticks: float = 14.0
    time_decay_power: float = 1.8
    centrality_weight: float = 0.35
    engagement_weight: float = 1.0
    seed: int = 0

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]] = None) -> "MediaRecommendationEngine":
        if not payload:
            return cls()
        allowed = {
            "top_k",
            "similarity_threshold",
            "half_life_ticks",
            "time_decay_power",
            "centrality_weight",
            "engagement_weight",
            "seed",
        }
        return cls(**{key: value for key, value in dict(payload).items() if key in allowed})

    def _centrality(self, graph: SocialGraphState) -> Dict[str, float]:
        if not graph.nodes:
            return {}
        denom = max(1.0, float(len(graph.nodes) - 1))
        incoming: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        for _, neighbors in graph.adjacency.items():
            for dst in neighbors:
                if dst in incoming:
                    incoming[dst] += 1
        return {
            node_id: float((len(graph.adjacency.get(node_id, [])) + incoming.get(node_id, 0)) / (2.0 * denom))
            for node_id in graph.nodes
        }

    def hot_score(self, post: SocialPost | Mapping[str, Any], *, current_tick: int) -> float:
        item = post if isinstance(post, SocialPost) else SocialPost.from_mapping(post)
        engagement = max(0.0, float(item.upvotes - item.downvotes))
        age = max(0.0, float(current_tick - item.created_tick))
        time_term = (age / max(float(self.half_life_ticks), 1e-6) + 1.0) ** float(self.time_decay_power)
        return float((math.log10(engagement + 1.0) + 0.15 * abs(item.polarity) + 0.05) / time_term)

    def recommendation_trace(
        self,
        graph: SocialGraphState,
        posts: Sequence[SocialPost | Mapping[str, Any]],
        *,
        current_tick: int,
    ) -> List[Dict[str, Any]]:
        normalized = [post if isinstance(post, SocialPost) else SocialPost.from_mapping(post) for post in posts]
        centrality = self._centrality(graph)
        rows: List[Dict[str, Any]] = []
        for target_id in graph.nodes:
            candidates: List[Tuple[float, SocialPost, float]] = []
            for post in normalized:
                if post.author_id == target_id:
                    continue
                edge_info = graph.get_edge_profile(target_id, post.author_id)
                author_neighbors = graph.adjacency.get(target_id, [])
                connected = post.author_id in author_neighbors or target_id in graph.adjacency.get(post.author_id, [])
                similarity = max(
                    float(edge_info.position_similarity_edge),
                    float(edge_info.news_exposure_edge) * 0.75,
                    0.35 if connected else 0.0,
                )
                if similarity < float(self.similarity_threshold) and centrality.get(post.author_id, 0.0) < 0.35:
                    continue
                hot = self.hot_score(post, current_tick=current_tick)
                score = hot * (0.35 + similarity) * (1.0 + self.centrality_weight * centrality.get(post.author_id, 0.0))
                score *= 1.0 + self.engagement_weight * min(1.0, max(0.0, post.upvotes - post.downvotes) / 50.0)
                candidates.append((float(score), post, float(hot)))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for rank, (score, post, hot) in enumerate(candidates[: max(1, int(self.top_k))], start=1):
                rows.append(
                    {
                        "target_id": target_id,
                        "post_id": post.post_id,
                        "author_id": post.author_id,
                        "topic": post.topic,
                        "score": float(score),
                        "hot_score": float(hot),
                        "rank": int(rank),
                        "author_centrality": float(centrality.get(post.author_id, 0.0)),
                        "repost_of": post.repost_of,
                    }
                )
        return rows

    def build_messages(
        self,
        graph: SocialGraphState,
        posts: Sequence[SocialPost | Mapping[str, Any]],
        *,
        current_tick: int,
    ) -> Tuple[List[SocialMessage], List[Dict[str, Any]]]:
        normalized = [post if isinstance(post, SocialPost) else SocialPost.from_mapping(post) for post in posts]
        by_id = {post.post_id: post for post in normalized}
        trace = self.recommendation_trace(graph, normalized, current_tick=current_tick)
        messages: List[SocialMessage] = []
        for row in trace:
            post = by_id.get(str(row.get("post_id", "")))
            if post is None:
                continue
            message = post.to_message(
                target_id=str(row["target_id"]),
                current_tick=current_tick,
                hot_score=float(row.get("hot_score", 0.0)),
                rank=int(row.get("rank", 1)),
            )
            if post.author_id in graph.nodes:
                message.source_type = graph.nodes[post.author_id].node_type
            messages.append(message)
        return messages, trace


@dataclass(slots=True)
class PropagationTrace:
    """A single source-to-target propagation record."""

    topic: str
    kind: str
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    created_tick: int
    received_tick: int
    delay: int
    source_credibility: float
    target_receptivity: float
    amplification: float
    decay: float
    signal: float
    belief_delta: float
    refuted: bool = False
    audience_tags: List[str] = field(default_factory=list)
    coverage_ratio: float = 1.0
    source_reliability_band: str = "medium"
    diffusion_velocity: float = 1.0
    rebuttal_of: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContagionSnapshot:
    """Outcome of one social contagion step."""

    mean_sentiment: float
    stressed_nodes: List[str]
    node_sentiment: Dict[str, float]
    edge_channel_means: Dict[str, float] = field(default_factory=dict)
    propagation_chain: List[Dict[str, Any]] = field(default_factory=list)
    node_influence: Dict[str, float] = field(default_factory=dict)
    narrative_heat: Dict[str, float] = field(default_factory=dict)
    observation_packets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_rankings: List[Dict[str, Any]] = field(default_factory=list)
    rumor_suppression: Dict[str, float] = field(default_factory=dict)
    opinion_leaders: List[Dict[str, Any]] = field(default_factory=list)
    cascade_metrics: Dict[str, float] = field(default_factory=dict)
    recommendation_trace: List[Dict[str, Any]] = field(default_factory=list)
    clarification_metrics: Dict[str, float] = field(default_factory=dict)
    bdi_observation_packets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SocialContagionEngine:
    """Diffuses sentiment through graph topology and macro anchors."""

    contagion_strength: float = 0.55
    self_memory: float = 0.35
    macro_anchor: float = 0.25
    trust_weight: float = 0.35
    position_similarity_weight: float = 0.25
    news_exposure_weight: float = 0.20
    institution_affiliation_weight: float = 0.20
    feature_flag: bool = False
    seed: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    current_tick: int = 0
    event_queue: List[SocialMessage] = field(default_factory=list)
    recommendation_engine: Optional[MediaRecommendationEngine] = None

    def __post_init__(self) -> None:
        if self.recommendation_engine is None:
            rec_cfg = dict(self.config.get("recommendation", {}) or {}) if isinstance(self.config, Mapping) else {}
            rec_cfg.setdefault("seed", int(self.seed))
            self.recommendation_engine = MediaRecommendationEngine.from_mapping(rec_cfg)

    def _edge_weight(self, graph: SocialGraphState, src: str, dst: str) -> Dict[str, float]:
        profile = graph.get_edge_profile(src, dst) if hasattr(graph, "get_edge_profile") else None
        trust = float(getattr(profile, "trust_edge", 0.5)) if profile is not None else 0.5
        position = float(getattr(profile, "position_similarity_edge", 0.5)) if profile is not None else 0.5
        news = float(getattr(profile, "news_exposure_edge", 0.5)) if profile is not None else 0.5
        institution = float(getattr(profile, "institution_affiliation_edge", 0.0)) if profile is not None else 0.0
        delay = float(getattr(profile, "propagation_delay_edge", 1.0)) if profile is not None else 1.0
        decay = float(getattr(profile, "decay_edge", 1.0)) if profile is not None else 1.0
        amplification = float(getattr(profile, "amplification_edge", 1.0)) if profile is not None else 1.0
        contradiction = float(getattr(profile, "contradiction_edge", 0.5)) if profile is not None else 0.5
        edge_score = (
            self.trust_weight * trust
            + self.position_similarity_weight * position
            + self.news_exposure_weight * news
            + self.institution_affiliation_weight * institution
        )
        return {
            "edge_score": _clip(edge_score, 0.0, 1.0),
            "trust_edge": _clip(trust, 0.0, 1.0),
            "position_similarity_edge": _clip(position, 0.0, 1.0),
            "news_exposure_edge": _clip(news, 0.0, 1.0),
            "institution_affiliation_edge": _clip(institution, 0.0, 1.0),
            "propagation_delay_edge": max(0.0, delay),
            "decay_edge": max(0.0, decay),
            "amplification_edge": max(0.0, amplification),
            "contradiction_edge": _clip(contradiction, 0.0, 1.0),
        }

    def enqueue_message(self, message: SocialMessage | Mapping[str, Any]) -> SocialMessage:
        if isinstance(message, SocialMessage):
            payload = message
        else:
            credibility = float(message.get("credibility", 0.5))
            payload = SocialMessage(
                message_id=str(message.get("message_id", "")),
                topic=str(message.get("topic", "market")),
                source_id=str(message.get("source_id", "synthetic_source")),
                source_type=str(message.get("source_type", SocialNodeType.RUMOR_SOURCE.value)),
                kind=str(message.get("kind", "rumor")),
                polarity=float(message.get("polarity", 0.0)),
                strength=float(message.get("strength", 1.0)),
                credibility=credibility,
                created_tick=int(message.get("created_tick", self.current_tick)),
                scheduled_tick=int(message.get("scheduled_tick", self.current_tick)),
                decay=float(message.get("decay", 0.1)),
                amplification=float(message.get("amplification", 1.0)),
                audience_tags=[str(item) for item in message.get("audience_tags", [])],
                coverage_ratio=float(message.get("coverage_ratio", 1.0)),
                source_reliability_band=str(message.get("source_reliability_band", _reliability_band(credibility))),
                diffusion_velocity=float(message.get("diffusion_velocity", 1.0)),
                rebuttal_of=str(message.get("rebuttal_of", "")),
                parent_message_id=str(message.get("parent_message_id", "")),
                root_message_id=str(message.get("root_message_id", "")),
                metadata=dict(message.get("metadata", {})),
            )
        self.event_queue.append(payload)
        return payload

    def _default_message_from_rumor_shock(self, rumor_shock: float, macro_state: MacroState) -> SocialMessage:
        polarity = float(_clip(rumor_shock, -1.0, 1.0))
        if abs(polarity) < 1e-9:
            polarity = float(_clip((macro_state.sentiment_index - 0.5) * 2.0, -1.0, 1.0))
        return SocialMessage(
            topic="policy_sentiment",
            source_id="synthetic_rumor_source",
            source_type=SocialNodeType.RUMOR_SOURCE.value,
            kind="rumor" if polarity <= 0 else "support",
            polarity=polarity,
            strength=max(0.1, abs(polarity)),
            credibility=0.12,
            created_tick=self.current_tick,
            scheduled_tick=self.current_tick,
            decay=0.24,
            amplification=1.55,
            audience_tags=[SocialNodeType.SOCIAL_MEDIA.value, SocialNodeType.KOL_SOCIAL.value, SocialNodeType.RETAIL_DAY_TRADER.value],
            coverage_ratio=0.95,
            source_reliability_band="low",
            diffusion_velocity=1.4,
            metadata={"rumor_shock": float(rumor_shock)},
        )

    def _default_message_from_macro(self, macro_state: MacroState) -> SocialMessage:
        polarity = float(_clip((macro_state.sentiment_index - 0.5) * 2.0, -1.0, 1.0))
        return SocialMessage(
            topic="macro_sentiment",
            source_id="macro_anchor",
            source_type=SocialNodeType.OFFICIAL_MEDIA.value,
            kind="policy",
            polarity=polarity,
            strength=max(0.1, abs(polarity)),
            credibility=0.62,
            created_tick=self.current_tick,
            scheduled_tick=self.current_tick,
            decay=0.12,
            amplification=1.10,
            audience_tags=[SocialNodeType.OFFICIAL_MEDIA.value, SocialNodeType.FINANCIAL_MEDIA.value, SocialNodeType.INSTITUTION.value],
            coverage_ratio=0.75,
            source_reliability_band="high",
            diffusion_velocity=0.9,
            metadata={"macro_anchor": True},
        )

    def _legacy_step(self, graph: SocialGraphState, macro_state: MacroState, rumor_shock: float = 0.0) -> ContagionSnapshot:
        """Run the legacy diffusion logic unchanged for backward compatibility."""
        if not graph.nodes:
            return ContagionSnapshot(mean_sentiment=0.0, stressed_nodes=[], node_sentiment={})

        new_values: Dict[str, float] = {}
        edge_stats = {
            "trust_edge": 0.0,
            "position_similarity_edge": 0.0,
            "news_exposure_edge": 0.0,
            "institution_affiliation_edge": 0.0,
            "edge_score": 0.0,
        }
        edge_count = 0.0

        for node_id, node in graph.nodes.items():
            neighbor_ids = graph.adjacency.get(node_id, [])
            weighted_sum = 0.0
            weight_total = 0.0
            weighted_news_edge = 0.0
            weighted_trust_edge = 0.0
            for neighbor_id in neighbor_ids:
                neighbor = graph.nodes.get(neighbor_id)
                if neighbor is None:
                    continue
                channels = self._edge_weight(graph, node_id, neighbor_id)
                w = channels["edge_score"]
                weighted_sum += w * neighbor.sentiment
                weight_total += w
                weighted_news_edge += w * channels["news_exposure_edge"]
                weighted_trust_edge += w * channels["trust_edge"]
                edge_stats["trust_edge"] += channels["trust_edge"]
                edge_stats["position_similarity_edge"] += channels["position_similarity_edge"]
                edge_stats["news_exposure_edge"] += channels["news_exposure_edge"]
                edge_stats["institution_affiliation_edge"] += channels["institution_affiliation_edge"]
                edge_stats["edge_score"] += channels["edge_score"]
                edge_count += 1.0

            neighborhood = weighted_sum / weight_total if weight_total > 1e-8 else 0.0
            avg_news_edge = weighted_news_edge / weight_total if weight_total > 1e-8 else 0.0
            avg_trust_edge = weighted_trust_edge / weight_total if weight_total > 1e-8 else 0.0
            macro_term = (macro_state.sentiment_index - 0.5) * 2.0
            exposure_term = (
                (0.35 + 0.65 * avg_news_edge) * node.news_exposure * rumor_shock
                + (0.25 + 0.75 * avg_trust_edge) * node.social_exposure * neighborhood
            )
            updated = (
                self.self_memory * node.sentiment
                + self.contagion_strength * neighborhood
                + self.macro_anchor * macro_term
                + exposure_term
            )
            new_values[node_id] = _clip(updated, -1.0, 1.0)

        for node_id, sentiment in new_values.items():
            graph.nodes[node_id].sentiment = sentiment

        stressed = sorted([node_id for node_id, value in new_values.items() if value < -0.35])
        mean_sentiment = sum(new_values.values()) / len(new_values)
        channel_means = {}
        for key, value in edge_stats.items():
            channel_means[key] = float(value / edge_count) if edge_count > 0 else 0.0
        return ContagionSnapshot(
            mean_sentiment=mean_sentiment,
            stressed_nodes=stressed,
            node_sentiment=new_values,
            edge_channel_means=channel_means,
        )

    def _normalize_messages(
        self,
        messages: Optional[Iterable[SocialMessage | Mapping[str, Any]]],
    ) -> List[SocialMessage]:
        normalized: List[SocialMessage] = []
        if not messages:
            return normalized
        for message in messages:
            if isinstance(message, SocialMessage):
                normalized.append(message)
            else:
                normalized.append(
                    SocialMessage(
                        message_id=str(message.get("message_id", "")),
                        topic=str(message.get("topic", "market")),
                        source_id=str(message.get("source_id", "synthetic_source")),
                        source_type=str(message.get("source_type", SocialNodeType.RUMOR_SOURCE.value)),
                        kind=str(message.get("kind", "rumor")),
                        polarity=float(message.get("polarity", 0.0)),
                        strength=float(message.get("strength", 1.0)),
                        credibility=float(message.get("credibility", 0.5)),
                        created_tick=int(message.get("created_tick", self.current_tick)),
                        scheduled_tick=int(message.get("scheduled_tick", self.current_tick)),
                        decay=float(message.get("decay", 0.1)),
                        amplification=float(message.get("amplification", 1.0)),
                        audience_tags=[str(item) for item in message.get("audience_tags", [])],
                        coverage_ratio=float(message.get("coverage_ratio", 1.0)),
                        source_reliability_band=str(message.get("source_reliability_band", _reliability_band(float(message.get("credibility", 0.5))))),
                        diffusion_velocity=float(message.get("diffusion_velocity", 1.0)),
                        rebuttal_of=str(message.get("rebuttal_of", "")),
                        parent_message_id=str(message.get("parent_message_id", "")),
                        root_message_id=str(message.get("root_message_id", "")),
                        metadata=dict(message.get("metadata", {})),
                    )
                )
        return normalized

    def _normalize_posts(
        self,
        posts: Optional[Iterable[SocialPost | Mapping[str, Any]]],
    ) -> List[SocialPost]:
        if not posts:
            return []
        normalized: List[SocialPost] = []
        for post in posts:
            normalized.append(post if isinstance(post, SocialPost) else SocialPost.from_mapping(post))
        return normalized

    def messages_from_event_digest(self, digest: Any, *, tick: Optional[int] = None) -> List[SocialMessage]:
        """Map runtime event digests to social messages without importing event modules."""
        if digest is None:
            return []
        if hasattr(digest, "to_dict"):
            payload = digest.to_dict()
        elif isinstance(digest, Mapping):
            payload = dict(digest)
        else:
            return []
        events = list(payload.get("active_events", []) or [])
        by_type = payload.get("by_type", {}) if isinstance(payload.get("by_type", {}), Mapping) else {}
        if not events:
            for items in by_type.values():
                if isinstance(items, list):
                    events.extend(items)
        current_tick = int(self.current_tick if tick is None else tick)
        messages: List[SocialMessage] = []
        for idx, event in enumerate(events):
            if not isinstance(event, Mapping):
                continue
            event_type = str(event.get("event_type", event.get("type", "major_news")) or "major_news").lower()
            strength = float(event.get("current_strength", event.get("strength", 1.0)) or 1.0)
            confidence = float(_clip(event.get("confidence", 0.75), 0.0, 1.0))
            title = str(event.get("title", event.get("raw_text", event_type)) or event_type)
            topic = str(event.get("topic", title[:32]) or title[:32])
            impact = event.get("impact", {}) if isinstance(event.get("impact", {}), Mapping) else {}
            social_delta = impact.get("social_delta", {}) if isinstance(impact.get("social_delta", {}), Mapping) else {}
            sentiment_hint = float(impact.get("sentiment_impact", social_delta.get("news_attention", 0.0)) or 0.0)
            metadata = {
                "runtime_event_id": str(event.get("event_id", "")),
                "runtime_event_type": event_type,
                "event_title": title,
                "target_mode": "top_centrality" if event_type in {"major_news", "rumor"} else "",
                "target_top_k": 5 if event_type == "rumor" else 3,
            }
            if event_type == "rumor":
                messages.append(
                    SocialMessage(
                        message_id=f"runtime_rumor_{current_tick}_{idx}",
                        topic=topic,
                        source_id="runtime_rumor_source",
                        source_type=SocialNodeType.RUMOR_SOURCE.value,
                        kind="rumor",
                        polarity=-abs(0.35 + 0.35 * strength),
                        strength=max(0.1, strength),
                        credibility=min(0.35, confidence * 0.45),
                        created_tick=current_tick,
                        scheduled_tick=current_tick,
                        amplification=1.35,
                        audience_tags=[SocialNodeType.KOL_SOCIAL.value, SocialNodeType.SOCIAL_MEDIA.value, SocialNodeType.RETAIL_DAY_TRADER.value],
                        coverage_ratio=0.9,
                        source_reliability_band="low",
                        diffusion_velocity=1.4,
                        metadata=metadata,
                    )
                )
            elif event_type == "refute":
                messages.append(
                    SocialMessage(
                        message_id=f"runtime_refute_{current_tick}_{idx}",
                        topic=topic,
                        source_id="runtime_regulator_voice",
                        source_type=SocialNodeType.REGULATOR_VOICE.value,
                        kind="refutation",
                        polarity=abs(0.45 + 0.25 * strength),
                        strength=max(0.1, strength),
                        credibility=max(0.85, confidence),
                        created_tick=current_tick,
                        scheduled_tick=current_tick,
                        amplification=1.25,
                        coverage_ratio=0.95,
                        source_reliability_band="high",
                        diffusion_velocity=0.8,
                        rebuttal_of=str(event.get("rebuttal_of", topic)),
                        metadata={**metadata, "target_mode": ""},
                    )
                )
            elif event_type in {"policy", "regulatory_action"}:
                messages.append(
                    SocialMessage(
                        message_id=f"runtime_policy_{current_tick}_{idx}",
                        topic=topic,
                        source_id="runtime_official_media",
                        source_type=SocialNodeType.OFFICIAL_MEDIA.value,
                        kind="policy",
                        polarity=float(_clip(sentiment_hint or 0.20 * strength, -1.0, 1.0)),
                        strength=max(0.1, strength),
                        credibility=max(0.75, confidence),
                        created_tick=current_tick,
                        scheduled_tick=current_tick,
                        coverage_ratio=0.85,
                        source_reliability_band="high",
                        metadata={**metadata, "target_mode": ""},
                    )
                )
            else:
                polarity = float(_clip(sentiment_hint if abs(sentiment_hint) > 1e-9 else 0.10 * strength, -1.0, 1.0))
                messages.append(
                    SocialMessage(
                        message_id=f"runtime_news_{current_tick}_{idx}",
                        topic=topic,
                        source_id="runtime_financial_media",
                        source_type=SocialNodeType.FINANCIAL_MEDIA.value,
                        kind="news",
                        polarity=polarity,
                        strength=max(0.1, strength),
                        credibility=max(0.55, confidence),
                        created_tick=current_tick,
                        scheduled_tick=current_tick,
                        amplification=1.10,
                        coverage_ratio=0.70,
                        source_reliability_band=_reliability_band(max(0.55, confidence)),
                        metadata=metadata,
                    )
                )
        return messages

    def _propagate_message(
        self,
        graph: SocialGraphState,
        message: SocialMessage,
        macro_state: MacroState,
    ) -> List[PropagationTrace]:
        traces: List[PropagationTrace] = []
        source_node = graph.nodes.get(message.source_id)
        source_profile = source_node if source_node is not None else GraphNodeState(node_id=message.source_id)
        source_profile.apply_profile(message.source_type)

        metadata_targets = [str(item) for item in message.metadata.get("recommended_targets", []) or []]
        if message.metadata.get("recommendation_target"):
            metadata_targets.append(str(message.metadata.get("recommendation_target")))
        if metadata_targets:
            target_ids = [target_id for target_id in dict.fromkeys(metadata_targets) if target_id in graph.nodes]
        elif str(message.metadata.get("target_mode", "")) == "top_centrality":
            top_k = max(1, int(message.metadata.get("target_top_k", 3) or 3))
            degree_rows = sorted(
                (
                    (
                        node_id,
                        len(graph.adjacency.get(node_id, []))
                        + sum(1 for neighbors in graph.adjacency.values() if node_id in neighbors),
                    )
                    for node_id in graph.nodes
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            target_ids = [node_id for node_id, _ in degree_rows[:top_k]]
        elif message.source_id in graph.nodes:
            target_ids = [message.source_id] + graph.adjacency.get(message.source_id, [])
        else:
            target_ids = list(graph.nodes.keys())

        if message.kind == "refutation":
            polarity = abs(float(message.polarity))
        else:
            polarity = float(message.polarity)

        base_strength = max(0.0, float(message.strength))
        for target_id in target_ids:
            target = graph.ensure_node(target_id)
            if message.audience_tags and target.node_type not in set(message.audience_tags):
                continue
            edge_profile = graph.get_edge_profile(message.source_id, target_id) if message.source_id in graph.nodes else None
            edge_info = self._edge_weight(graph, message.source_id, target_id) if message.source_id in graph.nodes else {
                "trust_edge": 0.5,
                "position_similarity_edge": 0.5,
                "news_exposure_edge": 0.5,
                "institution_affiliation_edge": 0.0,
                "propagation_delay_edge": 1.0,
                "decay_edge": 1.0,
                "amplification_edge": 1.0,
                "contradiction_edge": 0.5,
                "edge_score": 0.5,
            }
            delay = int(
                max(
                    0,
                    source_profile.propagation_delay
                    + target.propagation_delay
                    + int(round(edge_info["propagation_delay_edge"])),
                )
            )
            received_tick = self.current_tick + delay
            decay_window = max(0, received_tick - message.created_tick)
            decay = math.exp(-(target.decay_rate + edge_info["decay_edge"] * 0.05) * decay_window)
            credibility = _clip(
                message.credibility * source_profile.source_credibility * edge_info["trust_edge"],
                0.0,
                1.0,
            )
            amplification = max(
                0.0,
                source_profile.amplification
                * target.amplification
                * edge_info["amplification_edge"]
                * message.amplification,
            )
            amplification *= max(0.05, float(message.coverage_ratio))
            delay_factor = 1.0 / max(1.0, 1.0 + delay)
            velocity_factor = 1.0 / max(0.5, float(message.diffusion_velocity))
            contradiction = 1.0 - target.contradiction_sensitivity * edge_info["contradiction_edge"]
            contradiction = _clip(contradiction, 0.15, 1.15)
            macro_bias = 1.0 + 0.15 * (macro_state.sentiment_index - 0.5)
            signal = polarity * base_strength * credibility * amplification * decay * delay_factor * velocity_factor * contradiction * macro_bias
            belief_delta = signal * (0.55 + 0.45 * target.social_exposure)
            if message.kind == "refutation":
                belief_delta = abs(belief_delta) * (1.0 + 0.4 * max(0.0, target.rumor_sensitivity))
                signal = abs(signal)
            else:
                belief_delta = float(belief_delta)

            target.sentiment = _clip(target.sentiment + belief_delta, -1.0, 1.0)
            target.belief_strength = _clip(target.belief_strength + abs(belief_delta), 0.0, 1.0)
            target.reach_score += abs(signal) * amplification
            target.narrative_heat += abs(signal)
            if target.first_seen_tick < 0:
                target.first_seen_tick = received_tick
            target.last_seen_tick = received_tick
            target.record_event(
                {
                    "message_id": message.message_id,
                    "topic": message.topic,
                    "kind": message.kind,
                    "source_id": message.source_id,
                    "source_type": message.source_type,
                    "signal": signal,
                    "belief_delta": belief_delta,
                    "received_tick": received_tick,
                    "source_credibility": credibility,
                    "audience_tags": list(message.audience_tags),
                    "coverage_ratio": float(message.coverage_ratio),
                    "source_reliability_band": message.source_reliability_band,
                    "rebuttal_of": message.rebuttal_of,
                    "parent_message_id": message.parent_message_id,
                    "root_message_id": message.root_message_id,
                    "recommendation": bool(message.metadata.get("recommendation", False)),
                }
            )
            target.observation_state = {
                "message_id": message.message_id,
                "topic": message.topic,
                "kind": message.kind,
                "dominant_signal": signal,
                "rumor_pressure": max(0.0, -signal) if message.kind != "refutation" else 0.0,
                "refutation_pressure": abs(signal) if message.kind == "refutation" else 0.0,
                "source_credibility": credibility,
                "received_tick": received_tick,
                "audience_tags": list(message.audience_tags),
                "coverage_ratio": float(message.coverage_ratio),
                "rebuttal_of": message.rebuttal_of,
                "parent_message_id": message.parent_message_id,
                "root_message_id": message.root_message_id,
                "recommendation": bool(message.metadata.get("recommendation", False)),
            }
            trace_metadata = dict(message.metadata)
            trace_metadata.setdefault("message_id", message.message_id)
            trace_metadata.setdefault("parent_message_id", message.parent_message_id)
            trace_metadata.setdefault("root_message_id", message.root_message_id or message.message_id)
            traces.append(
                PropagationTrace(
                    topic=message.topic,
                    kind=message.kind,
                    source_id=message.source_id,
                    source_type=message.source_type,
                    target_id=target_id,
                    target_type=target.node_type,
                    created_tick=message.created_tick,
                    received_tick=received_tick,
                    delay=delay,
                    source_credibility=credibility,
                    target_receptivity=target.social_exposure,
                    amplification=amplification,
                    decay=decay,
                    signal=signal,
                    belief_delta=belief_delta,
                    refuted=message.kind == "refutation",
                    audience_tags=list(message.audience_tags),
                    coverage_ratio=float(message.coverage_ratio),
                    source_reliability_band=message.source_reliability_band,
                    diffusion_velocity=float(message.diffusion_velocity),
                    rebuttal_of=message.rebuttal_of,
                    metadata=trace_metadata,
                )
            )
            graph.record_observation(target_id, traces[-1].to_dict())

        return traces

    def _build_snapshot(
        self,
        graph: SocialGraphState,
        traces: List[PropagationTrace],
        macro_state: MacroState,
        *,
        rumor_shock: float,
    ) -> ContagionSnapshot:
        node_sentiment = {node_id: float(node.sentiment) for node_id, node in graph.nodes.items()}
        stressed = sorted([node_id for node_id, value in node_sentiment.items() if value < -0.35])
        mean_sentiment = sum(node_sentiment.values()) / len(node_sentiment) if node_sentiment else 0.0

        edge_channel_means: Dict[str, float] = {
            "trust_edge": 0.0,
            "position_similarity_edge": 0.0,
            "news_exposure_edge": 0.0,
            "institution_affiliation_edge": 0.0,
            "edge_score": 0.0,
            "propagation_delay": 0.0,
            "amplification": 0.0,
            "decay": 0.0,
        }
        if traces:
            edge_channel_means["trust_edge"] = float(sum(t.source_credibility for t in traces) / len(traces))
            edge_channel_means["position_similarity_edge"] = float(sum(t.target_receptivity for t in traces) / len(traces))
            edge_channel_means["news_exposure_edge"] = float(
                sum(graph.get_edge_profile(t.source_id, t.target_id).news_exposure_edge if t.source_id in graph.nodes else 0.5 for t in traces)
                / len(traces)
            )
            edge_channel_means["institution_affiliation_edge"] = float(
                sum(graph.get_edge_profile(t.source_id, t.target_id).institution_affiliation_edge if t.source_id in graph.nodes else 0.0 for t in traces)
                / len(traces)
            )
            edge_channel_means["edge_score"] = float(sum(abs(t.signal) for t in traces) / len(traces))
            edge_channel_means["propagation_delay"] = float(sum(t.delay for t in traces) / len(traces))
            edge_channel_means["amplification"] = float(sum(t.amplification for t in traces) / len(traces))
            edge_channel_means["decay"] = float(sum(t.decay for t in traces) / len(traces))

        influence: Dict[str, float] = {}
        narrative_heat: Dict[str, float] = {}
        source_totals: Dict[str, float] = {}
        source_counts: Dict[str, int] = {}
        rumor_heat_before = 0.0
        refutation_heat = 0.0

        for trace in traces:
            influence[trace.target_id] = influence.get(trace.target_id, 0.0) + abs(trace.signal)
            narrative_heat[trace.topic] = narrative_heat.get(trace.topic, 0.0) + abs(trace.signal)
            source_totals[trace.source_id] = source_totals.get(trace.source_id, 0.0) + abs(trace.signal)
            source_counts[trace.source_id] = source_counts.get(trace.source_id, 0) + 1
            if trace.kind == "rumor":
                rumor_heat_before += abs(trace.signal)
            if trace.kind == "refutation":
                refutation_heat += abs(trace.signal)

        source_rankings = [
            {
                "source_id": source_id,
                "total_influence": float(total),
                "mean_influence": float(total / max(1, source_counts[source_id])),
                "spread_count": int(source_counts[source_id]),
                "source_type": next((trace.source_type for trace in traces if trace.source_id == source_id), ""),
            }
            for source_id, total in sorted(source_totals.items(), key=lambda item: item[1], reverse=True)
        ]

        observation_packets = {
            node_id: graph.build_observation_payload(node_id, current_tick=self.current_tick)
            for node_id in graph.nodes.keys()
        }
        bdi_packets: Dict[str, Dict[str, Any]] = {}
        for node_id, packet in observation_packets.items():
            memory_seed = dict(packet.get("memory_seed", {}) or {})
            signal = float(memory_seed.get("dominant_signal", packet.get("sentiment", 0.0)) or 0.0)
            rumor_pressure = float(memory_seed.get("rumor_pressure", 0.0) or 0.0)
            refutation_pressure = float(memory_seed.get("refutation_pressure", 0.0) or 0.0)
            bdi_packets[node_id] = {
                "belief": {
                    "topic": packet.get("topic", ""),
                    "sentiment": float(packet.get("sentiment", 0.0)),
                    "source_credibility": float(memory_seed.get("source_credibility", packet.get("source_credibility", 0.0)) or 0.0),
                    "rumor_pressure": rumor_pressure,
                    "refutation_pressure": refutation_pressure,
                },
                "desire": {
                    "risk_repair": float(_clip(refutation_pressure - rumor_pressure, -1.0, 1.0)),
                    "information_need": float(_clip(abs(signal) + rumor_pressure, 0.0, 1.0)),
                    "social_confirmation": float(_clip(packet.get("belief_strength", 0.0), 0.0, 1.0)),
                },
                "intention": {
                    "trading_bias": float(_clip(signal, -1.0, 1.0)),
                    "communication_action": "clarify" if refutation_pressure > rumor_pressure else ("repost" if rumor_pressure > 0 else "observe"),
                },
            }

        rumor_heat_after = max(0.0, rumor_heat_before - refutation_heat)
        rumor_suppression = {
            "rumor_heat_before": float(rumor_heat_before),
            "refutation_heat": float(refutation_heat),
            "rumor_heat_after": float(rumor_heat_after),
            "delta": float(rumor_heat_after - rumor_heat_before),
            "suppression_ratio": float(1.0 - rumor_heat_after / max(rumor_heat_before, 1e-12)) if rumor_heat_before > 0 else 0.0,
        }
        unique_targets = {trace.target_id for trace in traces}
        unique_sources = {trace.source_id for trace in traces}
        cascade_depths = [int(trace.metadata.get("cascade_depth", 1) or 1) for trace in traces]
        recommendation_trace = [
            {
                "target_id": trace.target_id,
                "source_id": trace.source_id,
                "topic": trace.topic,
                "post_id": trace.metadata.get("source_post_id", ""),
                "hot_score": float(trace.metadata.get("hot_score", 0.0) or 0.0),
                "rank": int(trace.metadata.get("recommendation_rank", 0) or 0),
                "signal": float(trace.signal),
                "received_tick": int(trace.received_tick),
            }
            for trace in traces
            if bool(trace.metadata.get("recommendation", False))
        ]
        pos = [value for value in node_sentiment.values() if value >= 0.0]
        neg = [value for value in node_sentiment.values() if value < 0.0]
        pos_mean = sum(pos) / len(pos) if pos else 0.0
        neg_mean = sum(neg) / len(neg) if neg else 0.0
        clarification_latency = 0.0
        rumor_ticks = [trace.created_tick for trace in traces if trace.kind == "rumor"]
        refute_ticks = [trace.received_tick for trace in traces if trace.kind == "refutation"]
        if rumor_ticks and refute_ticks:
            clarification_latency = float(max(0, min(refute_ticks) - min(rumor_ticks)))
        cascade_metrics = {
            "max_depth": float(max(cascade_depths) if cascade_depths else 0),
            "coverage": float(len(unique_targets) / max(1, len(graph.nodes))),
            "reproduction_rate": float(len(traces) / max(1, len(unique_sources))),
            "polarization": float(abs(pos_mean - neg_mean)),
            "recommendation_count": float(len(recommendation_trace)),
        }
        clarification_metrics = {
            "has_refutation": float(1.0 if refute_ticks else 0.0),
            "clarification_latency": float(clarification_latency),
            "suppression_ratio": float(rumor_suppression["suppression_ratio"]),
            "refutation_heat": float(refutation_heat),
        }
        incoming: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        for _, neighbors in graph.adjacency.items():
            for dst in neighbors:
                if dst in incoming:
                    incoming[dst] += 1
        opinion_leaders = []
        for node_id, node in graph.nodes.items():
            degree = len(graph.adjacency.get(node_id, [])) + incoming.get(node_id, 0)
            influence_score = 0.45 * float(degree) + 0.35 * float(node.reach_score) + 0.20 * float(node.narrative_heat)
            opinion_leaders.append(
                {
                    "node_id": node_id,
                    "node_type": node.node_type,
                    "degree": int(degree),
                    "reach_score": float(node.reach_score),
                    "narrative_heat": float(node.narrative_heat),
                    "influence_score": float(influence_score),
                }
            )
        opinion_leaders.sort(key=lambda item: item["influence_score"], reverse=True)
        snapshot = ContagionSnapshot(
            mean_sentiment=mean_sentiment,
            stressed_nodes=stressed,
            node_sentiment=node_sentiment,
            edge_channel_means=edge_channel_means,
            propagation_chain=[trace.to_dict() for trace in traces],
            node_influence=influence,
            narrative_heat=narrative_heat,
            observation_packets=observation_packets,
            source_rankings=source_rankings,
            rumor_suppression=rumor_suppression,
            opinion_leaders=opinion_leaders[:10],
            cascade_metrics=cascade_metrics,
            recommendation_trace=recommendation_trace,
            clarification_metrics=clarification_metrics,
            bdi_observation_packets=bdi_packets,
            metadata={
                "feature_flag": bool(self.feature_flag),
                "seed": int(self.seed),
                "config_hash": _stable_hash(
                    {
                        "seed": int(self.seed),
                        "feature_flag": bool(self.feature_flag),
                        "config": dict(self.config),
                        "graph_signature": graph.graph_signature(),
                        "tick": int(self.current_tick),
                        "rumor_shock": float(rumor_shock),
                    }
                ),
                "snapshot_info": {
                    "node_count": len(graph.nodes),
                    "edge_count": sum(len(v) for v in graph.adjacency.values()),
                    "tick": int(self.current_tick),
                    "node_type_distribution": graph.node_type_counts(),
                    "macro_sentiment_index": float(macro_state.sentiment_index),
                    "source_layers": sorted({trace.source_type for trace in traces}),
                },
            },
        )
        return snapshot

    def step(
        self,
        graph: SocialGraphState,
        macro_state: MacroState,
        rumor_shock: float = 0.0,
        *,
        messages: Optional[Iterable[SocialMessage | Mapping[str, Any]]] = None,
        posts: Optional[Iterable[SocialPost | Mapping[str, Any]]] = None,
        tick: Optional[int] = None,
    ) -> ContagionSnapshot:
        """Run one diffusion step and update graph sentiments in place."""
        if not graph.nodes:
            return ContagionSnapshot(mean_sentiment=0.0, stressed_nodes=[], node_sentiment={})

        self.current_tick = int(self.current_tick + 1 if tick is None else tick)

        if not self.feature_flag:
            return self._legacy_step(graph, macro_state, rumor_shock=rumor_shock)

        active_messages = list(self.event_queue)
        active_messages.extend(self._normalize_messages(messages))
        active_posts = self._normalize_posts(posts)
        if active_posts and self.recommendation_engine is not None:
            recommended_messages, _ = self.recommendation_engine.build_messages(
                graph,
                active_posts,
                current_tick=self.current_tick,
            )
            active_messages.extend(recommended_messages)

        if abs(rumor_shock) > 1e-12:
            active_messages.append(self._default_message_from_rumor_shock(rumor_shock, macro_state))
        if not active_messages:
            active_messages.append(self._default_message_from_macro(macro_state))

        due_messages = [message for message in active_messages if message.scheduled_tick <= self.current_tick]
        self.event_queue = [message for message in active_messages if message.scheduled_tick > self.current_tick]

        traces: List[PropagationTrace] = []
        for message in due_messages:
            traces.extend(self._propagate_message(graph, message, macro_state))

        if not traces:
            return self._legacy_step(graph, macro_state, rumor_shock=rumor_shock)

        snapshot = self._build_snapshot(graph, traces, macro_state, rumor_shock=rumor_shock)
        return snapshot

    def write_report(
        self,
        snapshot: ContagionSnapshot,
        graph: SocialGraphState,
        path: str | Path,
        *,
        title: str = "social_propagation",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        report = build_social_propagation_report(snapshot, graph, title=title, metadata=metadata or self.config)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def build_social_propagation_report(
    snapshot: ContagionSnapshot,
    graph: SocialGraphState,
    *,
    title: str = "social_propagation",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = snapshot.to_dict()
    report = {
        "report_type": title,
        "feature_flag": bool(snapshot.metadata.get("feature_flag", False)),
        "seed": int(snapshot.metadata.get("seed", 0)),
        "config_hash": str(snapshot.metadata.get("config_hash", "")),
        "snapshot_info": dict(snapshot.metadata.get("snapshot_info", {})),
        "node_type_distribution": graph.node_type_counts(),
        "mean_sentiment": float(snapshot.mean_sentiment),
        "stressed_nodes": list(snapshot.stressed_nodes),
        "edge_channel_means": dict(snapshot.edge_channel_means),
        "propagation_chain": list(snapshot.propagation_chain),
        "node_influence": dict(snapshot.node_influence),
        "narrative_heat": dict(snapshot.narrative_heat),
        "rumor_suppression": dict(snapshot.rumor_suppression),
        "opinion_leaders": list(snapshot.opinion_leaders),
        "cascade_metrics": dict(snapshot.cascade_metrics),
        "recommendation_trace": list(snapshot.recommendation_trace),
        "clarification_metrics": dict(snapshot.clarification_metrics),
        "bdi_observation_packets": dict(snapshot.bdi_observation_packets),
        "observation_packets": dict(snapshot.observation_packets),
        "source_rankings": list(snapshot.source_rankings),
        "heatmap_rows": [
            {
                "source_id": trace.get("source_id", ""),
                "source_type": trace.get("source_type", ""),
                "target_id": trace.get("target_id", ""),
                "target_type": trace.get("target_type", ""),
                "signal": float(trace.get("signal", 0.0)),
                "rebuttal_of": trace.get("rebuttal_of", ""),
            }
            for trace in list(snapshot.propagation_chain)
        ],
        "snapshot": payload,
        "metadata": dict(metadata or {}),
    }
    report["report_hash"] = _stable_hash(
        {
            "report_type": title,
            "config_hash": report["config_hash"],
            "node_count": report["snapshot_info"].get("node_count", 0),
            "chain_size": len(report["propagation_chain"]),
            "metadata": report["metadata"],
        }
    )
    return report


def write_social_propagation_report(
    snapshot: ContagionSnapshot,
    graph: SocialGraphState,
    path: str | Path,
    *,
    title: str = "social_propagation",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    report = build_social_propagation_report(snapshot, graph, title=title, metadata=metadata)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    from core.social.graph_state import SocialGraphState

    graph = SocialGraphState.ring(["a", "b", "c", "d"])
    graph.set_node_profile("a", SocialNodeType.OFFICIAL_MEDIA)
    graph.set_sentiment("a", -0.4)
    graph.set_sentiment("b", 0.1)
    graph.set_sentiment("c", 0.2)
    graph.set_sentiment("d", 0.0)
    macro = MacroState(sentiment_index=0.45)
    engine = SocialContagionEngine(feature_flag=True)
    snap = engine.step(
        graph,
        macro_state=macro,
        rumor_shock=-0.3,
        messages=[{"topic": "policy", "source_id": "a", "source_type": "official_media", "kind": "policy", "polarity": 0.4}],
    )
    print(json.dumps(snap.to_dict(), ensure_ascii=False, indent=2))
