"""
Civitas-Economica: Ebbinghaus 认知记忆模块

本模块结合 ChromaDB 实现了代理具有随时间及特定属性衰减的记忆池。
根据代理的 Persona 计算对不同宏观政策和事件的遗忘率(Memory Strength S)。
提取记忆时根据余弦相似度和艾宾浩斯记忆保留率 (Ebbinghaus Retention Score R) 计算最终得分。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from agents.trading_agent_core import GLOBAL_CONFIG, CognitiveBias, InvestmentHorizon, Persona

logger = logging.getLogger("civitas.agents.ebbinghaus_memory")
logger.setLevel(logging.DEBUG)


class EbbinghausMemoryBank:
    """
    包装 ChromaDB 的认知记忆银行。
    每个代理实例持有一个独立的 Collection 以存储其特有的记忆和状态。
    结合艾宾浩斯遗忘曲线对记忆内容进行动态遗忘模拟。
    """

    def __init__(self, agent_id: str, db_path: str = "./chroma_db"):
        """
        初始化 Agent 的记忆库。

        Args:
            agent_id (str): 代理的唯一标识符
            db_path (str): ChromaDB 本地持久化路径
        """
        self.agent_id = agent_id
        
        # 为了高效执行，我们可以使用持久性客户端
        self._chroma_client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        
        # 每个代理有自己专属的 Collection，使用 cosine 距离作为度量标准
        collection_name = f"agent_memory_{self.agent_id}".replace("-", "_")
        try:
            self.collection = self._chroma_client.get_or_create_collection(
                name=collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to get or create collection for {agent_id}: {e}")
            raise

        self._memory_counter = 0
        
        # 性能瓶颈优化方案
        # 缓存同一智能体在记忆总数未变时的查询结果，避免每 Ticks 都要进行数百次高消耗的 ChromaDB/SQLite 查询
        self._chroma_query_cache: Dict[Tuple[str, int], Any] = {}

    def calculate_memory_strength(self, policy_content: str, agent_persona: Persona) -> float:
        """
        动态计算特定记忆条目的话题强度 (Memory Strength, S)。
        
        S = S_base + (alpha * relevance_score)
        如果政策内容与代理的 InvestmentHorizon 或 CognitiveBias 严重冲突或高度一致，
        S 必须受到严厉惩罚或大幅增强。

        Args:
            policy_content (str): 宏观政策或环境事件文本。
            agent_persona (Persona): 代理的心理画像。

        Returns:
            float: 该记忆的初始强度 (S > 0)
        """
        s_base = GLOBAL_CONFIG.get("ebbinghaus_base_decay_rate", 0.1)
        alpha = GLOBAL_CONFIG.get("ebbinghaus_alpha", 0.5)

        # 这里使用一个模拟语义检查来估算相关性分数
        # 在真实应用中，可以是对 embedding 向量或 大模型 进行短问答
        relevance_score = 0.5  # 基础相关性

        content_lower = policy_content.lower()

        # 1. 模拟对 InvestmentHorizon 的影响
        if agent_persona.investment_horizon == InvestmentHorizon.Short_term:
            if "短期" in content_lower or "日内" in content_lower or "short-term" in content_lower:
                relevance_score += 0.3
            elif "长期" in content_lower or "十年" in content_lower or "long-term" in content_lower:
                relevance_score -= 0.3
        elif agent_persona.investment_horizon == InvestmentHorizon.Long_term:
            if "长期" in content_lower or "十年" in content_lower or "long-term" in content_lower:
                relevance_score += 0.3
            elif "短期" in content_lower or "日内" in content_lower or "short-term" in content_lower:
                relevance_score -= 0.3

        # 2. 模拟 CognitiveBias 对记忆的影响
        # 例如，损失厌恶者对“下跌”、“风险”等词汇会有极深的印象
        if agent_persona.cognitive_bias == CognitiveBias.Loss_Aversion:
            if "风险" in content_lower or "下跌" in content_lower or "崩盘" in content_lower or "risk" in content_lower:
                relevance_score += 0.4
        elif agent_persona.cognitive_bias == CognitiveBias.Herding:
            if "普遍" in content_lower or "大众" in content_lower or "一致预期" in content_lower:
                relevance_score += 0.4

        # policy_sensitivity 作为整体敏感性权重
        sensitivity = agent_persona.policy_sensitivity
        
        # 相关性分数被限制在 [0.1, 1.0] 范围内，防止 S 为负或无穷小
        relevance_score = max(0.1, min(1.0, relevance_score))
        
        S = s_base + (alpha * sensitivity * relevance_score)
        
        # 防止 S 为 0 导致除 0 错误
        return max(S, 0.01)

    def add_memory(self, timestamp: int, content: str, content_embedding: Optional[List[float]], memory_strength: float) -> str:
        """
        向认知库中插入记忆。

        Args:
            timestamp (int): 当前系统/仿真时钟时间
            content (str): 记忆的具体文本内容
            content_embedding (Optional[List[float]]): 显式传入的内容向量。如果在 ChromaDB 设置中启用了默认模型，则可以为 None。
            memory_strength (float): 已计算出的记忆强度 S

        Returns:
            str: 插入的文档 ID
        """
        doc_id = f"mem_{self._memory_counter}_{timestamp}"
        self._memory_counter += 1

        metadata = {
            "timestamp": timestamp,
            "S": memory_strength
        }

        try:
            if content_embedding:
                self.collection.add(
                    documents=[content],
                    embeddings=[content_embedding],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            else:
                self.collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            logger.debug(f"[{self.agent_id}] 成功记录记忆片段: {doc_id} S={memory_strength:.3f}")
        except Exception as e:
            logger.error(f"Failed to add memory to collection {self.collection.name}: {e}")

        return doc_id

    def retrieve_context(self, current_simulation_time: int, query_embedding: Optional[List[float]], query_text: str, top_k: int = 5) -> str:
        """
        核心检索机制：
        基于余弦相似度结合艾宾浩斯衰减度，对回忆的线索进行打分。
        过滤掉那些遗忘保留率 (R) 跌破阈值的远古记忆。

        Args:
            current_simulation_time (int): 检索发生的系统时间
            query_embedding (Optional[List[float]]): 检索的表示向量
            query_text (str): 检索的文本
            top_k (int): 最高返回条数候选数量

        Returns:
            str: 将检索出的最优记忆拼接而成的提示上下文。
        """
        # 由于可能出现有大量已被彻底遗忘的记忆，在查询时预抓取一个较大的池子。
        fetch_k = top_k * 3 

        # 缓存键：查询文本与当前的记忆总数，如果记忆数量没变，直接复用 ChromaDB 候选集，极大地提升并发性能
        cache_key = (query_text, self._memory_counter)
        if cache_key in getattr(self, "_chroma_query_cache", {}):
            results = self._chroma_query_cache[cache_key]
        else:
            # 从 ChromaDB 拉取候选条目
            try:
                if query_embedding:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=fetch_k,
                        include=["documents", "distances", "metadatas"]
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query_text],
                        n_results=fetch_k,
                        include=["documents", "distances", "metadatas"]
                    )
                
                # 初始化缓存字典并防止内存泄漏
                if not hasattr(self, "_chroma_query_cache"):
                    self._chroma_query_cache = {}
                self._chroma_query_cache.clear()
                self._chroma_query_cache[cache_key] = results
                
            except Exception as e:
                logger.error(f"Failed to query collection {self.collection.name}: {e}")
                return "记忆检索失败或记忆为空。"

        if not results['documents'] or len(results['documents'][0]) == 0:
            return "您的记忆中没有相关参考信息。"

        docs = results['documents'][0]
        distances = results['distances'][0]
        metas = results['metadatas'][0]

        cosine_weight = GLOBAL_CONFIG.get("retrieval_cosine_weight", 0.6)
        retention_weight = GLOBAL_CONFIG.get("retrieval_retention_weight", 0.4)
        retention_threshold = GLOBAL_CONFIG.get("ebbinghaus_retention_threshold", 0.1)

        scored_memories = []

        for doc, dist, meta in zip(docs, distances, metas):
            mem_time = meta.get("timestamp", 0)
            S = meta.get("S", 0.1)

            # Chroma 的余弦距离范围约在 [0, 2]，这里统一按 1 - 距离计算余弦相似度。
            # 默认向量模型有时返回的并非标准余弦距离，因此此处只做稳定近似。
            cosine_similarity = 1.0 - dist
            # 对相似度进行规范化限制
            cosine_similarity = max(0.0, min(1.0, cosine_similarity))

            time_elapsed = current_simulation_time - mem_time
            if time_elapsed < 0:
                time_elapsed = 0

            # 数学约束：R = exp(-记忆年龄 / 记忆强度)
            R = math.exp(-time_elapsed / S)

            # 阈值拦截：如果保留分 R 跌破了界限，代理就完全忘记了这件事
            if R < retention_threshold:
                continue

            # 融合得分
            final_score = (cosine_weight * cosine_similarity) + (retention_weight * R)
            scored_memories.append((final_score, doc, meta, R))

        # 按综合最终打分从高到低排序
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        selected_memories = scored_memories[:top_k]

        if not selected_memories:
            return "你感觉有些东西似乎发生过，但已经彻底回忆不起来了。"

        context_lines = []
        for i, (score, doc, meta, r_score) in enumerate(selected_memories, start=1):
            t_stamp = meta.get('timestamp')
            context_lines.append(f"({i}) 记忆于[T={t_stamp}] (记忆清晰度: {r_score:.2f}): {doc}")

        return "\n".join(context_lines)
