# agents/cognition 模块
"""
认知架构与行为内核

包含:
- 前景理论效用函数
- DeepSeek R1 推理包装器
- 增强型 RAG 记忆
- 认知 Agent 整合
"""

from agents.cognition.utility import ProspectTheory, calculate_prospect_value
from agents.cognition.llm_brain import DeepSeekReasoner, ReasoningResult
from agents.cognition.memory import TraumaMemory
from agents.cognition.cognitive_agent import CognitiveAgent
from agents.cognition.graph_storage import GraphMemoryBank, KnowledgeCapsule
from agents.cognition.graph_builder import GraphExtractor
from agents.cognition.layered_memory import BehaviorCard, LayeredMemory

__all__ = [
    "ProspectTheory",
    "calculate_prospect_value",
    "DeepSeekReasoner",
    "ReasoningResult",
    "TraumaMemory",
    "CognitiveAgent",
    "GraphMemoryBank",
    "KnowledgeCapsule",
    "GraphExtractor",
    "BehaviorCard",
    "LayeredMemory"
]
