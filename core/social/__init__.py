"""Social-layer graph state and contagion engines."""

from core.social.contagion import (
    ContagionSnapshot,
    MediaRecommendationEngine,
    PropagationTrace,
    SocialContagionEngine,
    SocialMessage,
    SocialPost,
    build_social_propagation_report,
    write_social_propagation_report,
)
from core.social.graph_state import (
    DEFAULT_NODE_PROFILES,
    EdgeProfile,
    GraphNodeState,
    SocialGraphState,
    SocialNodeType,
)

__all__ = [
    "ContagionSnapshot",
    "DEFAULT_NODE_PROFILES",
    "EdgeProfile",
    "GraphNodeState",
    "MediaRecommendationEngine",
    "PropagationTrace",
    "SocialContagionEngine",
    "SocialGraphState",
    "SocialMessage",
    "SocialPost",
    "SocialNodeType",
    "build_social_propagation_report",
    "write_social_propagation_report",
]
