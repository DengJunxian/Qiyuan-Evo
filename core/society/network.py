"""
社会网络拓扑与语义驱动情绪传染模型。

本模块实现从“纯拓扑概率传播”升级为：
1. 语义驱动感染：感染概率与双方主导叙事余弦相似度挂钩；
2. 轻量推荐分发：邻居信息先经过 Hot Score 排序后再进入感染计算；
3. 信息茧房/回音壁：历史曝光会提升后续同源信息权重，形成强化回路。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class SentimentState(str, Enum):
    """SIRS + Bullish 扩展状态。"""

    SUSCEPTIBLE = "S"
    INFECTED = "I"
    RECOVERED = "R"
    BULLISH = "B"


@dataclass
class AgentNode:
    """
    社交网络中的智能体节点。

    语义相关字段说明：
    - dominant_narratives: 来自 GraphRAG 的主导叙事概念
    - focus_topics: 该节点偏好接收的话题（推荐过滤）
    - source_exposure_count: 历史曝光计数（形成回音壁）
    """

    node_id: int
    agent_id: str = ""
    sentiment_state: SentimentState = SentimentState.SUSCEPTIBLE
    base_lambda: float = 2.25
    lambda_coeff: float = 2.25
    conformity: float = 0.5
    influence: float = 0.5
    state_duration: int = 0
    sentiment_history: List[SentimentState] = field(default_factory=list)

    dominant_narratives: List[str] = field(default_factory=list)
    focus_topics: List[str] = field(default_factory=list)
    risk_tilt: float = 0.0
    historical_risk_bias: float = 0.0
    feed_capacity: int = 3
    source_exposure_count: Dict[int, int] = field(default_factory=dict)

    # BDI 抽象
    beliefs: List[str] = field(default_factory=list)
    desires: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)

    # 持仓组合，用于 Jaccard 相似度与动态边权
    holdings: Dict[str, int] = field(default_factory=dict)


class SocialGraph:
    """
    小世界社交网络 (Watts-Strogatz)。
    """

    _DEFAULT_TOPIC_POOL = [
        "流动性",
        "降准",
        "通胀",
        "地产",
        "科技成长",
        "监管趋严",
        "风险偏好",
        "外资流向",
        "财政刺激",
        "消费复苏",
        "地缘风险",
        "AI产业",
    ]

    def __init__(
        self,
        n_agents: int = 1000,
        k: int = 6,
        p: float = 0.3,
        seed: int | None = None,
    ):
        self.n_agents = n_agents
        self.k = k
        self.p = p
        self.graph = nx.watts_strogatz_graph(n_agents, k, p, seed=seed)
        for u, v in self.graph.edges():
            self.graph[u][v]["weight"] = 1.0
        self.agents: Dict[int, AgentNode] = {}
        self.tick = 0
        self._init_agents(seed=seed)

    def _init_agents(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        for node_id in self.graph.nodes():
            conformity = float(np.clip(rng.normal(0.5, 0.15), 0.1, 0.95))
            influence = float(min(0.95, rng.pareto(2.0) * 0.2))
            base_lambda = float(np.clip(rng.normal(2.25, 0.5), 1.0, 4.0))
            focus_count = int(rng.integers(2, 5))
            dominant_count = int(rng.integers(1, 4))
            focus_topics = random.sample(self._DEFAULT_TOPIC_POOL, k=focus_count)
            dominant = random.sample(focus_topics + self._DEFAULT_TOPIC_POOL, k=dominant_count)
            self.agents[node_id] = AgentNode(
                node_id=node_id,
                agent_id=f"agent_{node_id}",
                conformity=conformity,
                influence=influence,
                base_lambda=base_lambda,
                lambda_coeff=base_lambda,
                dominant_narratives=list(dict.fromkeys(dominant)),
                focus_topics=list(dict.fromkeys(focus_topics)),
                risk_tilt=float(np.clip(rng.normal(0.0, 0.4), -1.0, 1.0)),
                historical_risk_bias=0.0,
                feed_capacity=int(rng.integers(2, 5)),
            )

    def update_semantic_profile(
        self,
        node_id: int,
        *,
        dominant_narratives: Optional[List[str]] = None,
        focus_topics: Optional[List[str]] = None,
        risk_tilt: Optional[float] = None,
        historical_risk_bias: Optional[float] = None,
    ) -> None:
        """更新节点的语义画像（由外层 Agent/GraphRAG 同步）。"""
        node = self.agents.get(node_id)
        if node is None:
            return

        if dominant_narratives is not None:
            cleaned = [str(x).strip() for x in dominant_narratives if str(x).strip()]
            if cleaned:
                node.dominant_narratives = list(dict.fromkeys(cleaned))

        if focus_topics is not None:
            cleaned = [str(x).strip() for x in focus_topics if str(x).strip()]
            if cleaned:
                node.focus_topics = list(dict.fromkeys(cleaned))

        if risk_tilt is not None:
            node.risk_tilt = float(np.clip(risk_tilt, -1.0, 1.0))

        if historical_risk_bias is not None:
            node.historical_risk_bias = float(np.clip(historical_risk_bias, -1.0, 1.0))

    def update_bdi_profile(
        self,
        node_id: int,
        *,
        beliefs: Optional[List[str]] = None,
        desires: Optional[List[str]] = None,
        intentions: Optional[List[str]] = None,
    ) -> None:
        """
        更新节点的 BDI 抽象状态，用于后续的意图传播与群体行为分析。
        """
        node = self.agents.get(node_id)
        if node is None:
            return
        if beliefs is not None:
            node.beliefs = [str(x).strip() for x in beliefs if str(x).strip()]
        if desires is not None:
            node.desires = [str(x).strip() for x in desires if str(x).strip()]
        if intentions is not None:
            node.intentions = [str(x).strip() for x in intentions if str(x).strip()]

    def update_holdings(self, node_id: int, holdings: Dict[str, int]) -> None:
        """
        更新节点持仓，用于 Jaccard 相似度和动态边权计算。
        """
        node = self.agents.get(node_id)
        if node is None:
            return
        cleaned = {str(k): int(v) for k, v in (holdings or {}).items() if int(v) != 0}
        node.holdings = cleaned

    def _portfolio_jaccard(self, a_node_id: int, b_node_id: int) -> float:
        """
        计算两节点持仓组合的 Jaccard 相似度。
        """
        a = self.agents.get(a_node_id)
        b = self.agents.get(b_node_id)
        if a is None or b is None:
            return 0.0
        a_set = {k for k, v in a.holdings.items() if v != 0}
        b_set = {k for k, v in b.holdings.items() if v != 0}
        if not a_set and not b_set:
            return 0.0
        return len(a_set & b_set) / max(1, len(a_set | b_set))

    def update_edge_weights_by_portfolio(
        self,
        *,
        weight_floor: float = 0.2,
        weight_ceiling: float = 2.0,
        smoothing: float = 0.3,
    ) -> None:
        """
        基于持仓 Jaccard 相似度更新网络连边权重，提升相似持仓间的信息传播概率。
        """
        for u, v in self.graph.edges():
            sim = self._portfolio_jaccard(u, v)
            target = 1.0 + sim
            old = float(self.graph[u][v].get("weight", 1.0))
            new_weight = old * (1 - smoothing) + target * smoothing
            new_weight = float(np.clip(new_weight, weight_floor, weight_ceiling))
            self.graph[u][v]["weight"] = new_weight

    def get_edge_weight(self, u: int, v: int) -> float:
        return float(self.graph[u][v].get("weight", 1.0))

    def get_bearish_ratio_weighted(self, node_id: int) -> float:
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        total_w = 0.0
        bearish_w = 0.0
        for n in neighbors:
            w = self.get_edge_weight(node_id, n)
            total_w += w
            if self.agents[n].sentiment_state == SentimentState.INFECTED:
                bearish_w += w
        return bearish_w / total_w if total_w > 0 else 0.0

    def get_bullish_ratio_weighted(self, node_id: int) -> float:
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        total_w = 0.0
        bullish_w = 0.0
        for n in neighbors:
            w = self.get_edge_weight(node_id, n)
            total_w += w
            if self.agents[n].sentiment_state == SentimentState.BULLISH:
                bullish_w += w
        return bullish_w / total_w if total_w > 0 else 0.0

    def get_neighbors(self, node_id: int) -> List[int]:
        return list(self.graph.neighbors(node_id))

    def get_neighbor_sentiments(self, node_id: int) -> Dict[SentimentState, int]:
        neighbors = self.get_neighbors(node_id)
        counts = {state: 0 for state in SentimentState}
        for n in neighbors:
            counts[self.agents[n].sentiment_state] += 1
        return counts

    def get_bearish_ratio(self, node_id: int) -> float:
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        bearish = sum(1 for n in neighbors if self.agents[n].sentiment_state == SentimentState.INFECTED)
        return bearish / len(neighbors)

    def get_bullish_ratio(self, node_id: int) -> float:
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        bullish = sum(1 for n in neighbors if self.agents[n].sentiment_state == SentimentState.BULLISH)
        return bullish / len(neighbors)

    @staticmethod
    def _topic_vector(topics: List[str], dim: int = 64) -> np.ndarray:
        """将主题列表映射为固定维度向量，用于轻量余弦相似度。"""
        vec = np.zeros(dim, dtype=float)
        if not topics:
            return vec
        for topic in topics:
            key = str(topic).strip().lower()
            if not key:
                continue
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def get_narrative_similarity(self, a_node_id: int, b_node_id: int) -> float:
        """
        计算两节点主导叙事余弦相似度，返回 [0, 1]。
        """
        a = self.agents.get(a_node_id)
        b = self.agents.get(b_node_id)
        if a is None or b is None:
            return 0.0
        # 语义相似度以“主导叙事”为核心，同时引入关注主题作为补充上下文，
        # 可降低冷启动阶段因图谱稀疏导致的全零相似。
        a_topics = list(dict.fromkeys(a.dominant_narratives + a.focus_topics))
        b_topics = list(dict.fromkeys(b.dominant_narratives + b.focus_topics))
        va = self._topic_vector(a_topics)
        vb = self._topic_vector(b_topics)
        if float(np.linalg.norm(va)) == 0 or float(np.linalg.norm(vb)) == 0:
            return 0.0
        sim = float(np.dot(va, vb))
        return float(np.clip(sim, 0.0, 1.0))

    def generate_social_summary(self, node_id: int) -> str:
        sentiments = self.get_neighbor_sentiments(node_id)
        total = max(1, len(self.get_neighbors(node_id)))
        bearish = sentiments[SentimentState.INFECTED]
        bullish = sentiments[SentimentState.BULLISH]
        neutral = sentiments[SentimentState.SUSCEPTIBLE]

        if bearish > total * 0.5:
            return f"你关注的账户中有{bearish}个处于强烈悲观，负面叙事正在扩散。"
        if bullish > total * 0.5:
            return f"你关注的账户中有{bullish}个明显看多，乐观叙事占据上风。"
        if bearish > bullish:
            return f"社交圈偏谨慎，悲观信号({bearish})多于乐观信号({bullish})。"
        if bullish > bearish:
            return f"社交圈偏积极，乐观信号({bullish})多于悲观信号({bearish})。"
        return f"社交圈整体中性，观望账户约{neutral}个。"

    def get_most_influential_node(self) -> Optional[int]:
        if not self.agents:
            return None
        return max(self.agents.items(), key=lambda x: x[1].influence)[0]

    def get_network_stats(self) -> Dict:
        states = [a.sentiment_state for a in self.agents.values()]
        return {
            "total_nodes": self.n_agents,
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(1, self.n_agents),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "sentiment_distribution": {
                "susceptible": sum(1 for s in states if s == SentimentState.SUSCEPTIBLE),
                "infected": sum(1 for s in states if s == SentimentState.INFECTED),
                "recovered": sum(1 for s in states if s == SentimentState.RECOVERED),
                "bullish": sum(1 for s in states if s == SentimentState.BULLISH),
            },
            "avg_lambda": float(np.mean([a.lambda_coeff for a in self.agents.values()])),
        }


class LightweightRecSys:
    """
    轻量推荐模块：根据 Hot Score 为目标节点挑选信息源。

    评分项（全部归一化到 0~1）：
    - 语义相似度 semantic_similarity
    - 主题匹配 topic_match（关注列表 vs 对方叙事）
    - 风偏亲和 risk_affinity（历史风险偏好接近度）
    - 影响力 influence_factor
    - 回音壁强化 echo_boost（历史曝光次数）
    """

    def __init__(
        self,
        w_semantic: float = 0.35,
        w_topic: float = 0.25,
        w_risk: float = 0.15,
        w_influence: float = 0.15,
        w_echo: float = 0.10,
    ):
        self.w_semantic = w_semantic
        self.w_topic = w_topic
        self.w_risk = w_risk
        self.w_influence = w_influence
        self.w_echo = w_echo

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa = {x.strip().lower() for x in a if str(x).strip()}
        sb = {x.strip().lower() for x in b if str(x).strip()}
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def hot_score(
        self,
        target: AgentNode,
        source: AgentNode,
        semantic_similarity: float,
    ) -> float:
        topic_match = self._jaccard(target.focus_topics, source.dominant_narratives)
        target_risk = float(np.clip(target.risk_tilt + target.historical_risk_bias, -1.0, 1.0))
        source_risk = float(np.clip(source.risk_tilt + source.historical_risk_bias, -1.0, 1.0))
        risk_affinity = 1.0 - min(1.0, abs(target_risk - source_risk))
        influence_factor = float(np.clip(source.influence, 0.0, 1.0))
        echo_boost = min(1.0, target.source_exposure_count.get(source.node_id, 0) / 5.0)

        score = (
            self.w_semantic * semantic_similarity
            + self.w_topic * topic_match
            + self.w_risk * risk_affinity
            + self.w_influence * influence_factor
            + self.w_echo * echo_boost
        )
        return float(np.clip(score, 0.0, 1.0))


class InformationDiffusion:
    """
    语义驱动 SIRS 情绪传播引擎。
    """

    def __init__(
        self,
        social_graph: SocialGraph,
        beta: float = 0.3,
        gamma: float = 0.1,
        delta: float = 0.05,
        lambda_amplifier: float = 0.5,
        recsys: Optional[LightweightRecSys] = None,
    ):
        self.graph = social_graph
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_amplifier = lambda_amplifier
        self.recsys = recsys or LightweightRecSys()
        self.history: List[Dict] = []

    def _rank_semantic_feed(self, target: AgentNode, infected_neighbors: List[AgentNode]) -> List[Tuple[AgentNode, float, float]]:
        """
        针对目标节点做推荐排序，返回 (源节点, hot_score, semantic_similarity)。

        这里是“信息分发层”，目标是替代“所有邻居等权注入”的旧逻辑。
        """
        ranked: List[Tuple[AgentNode, float, float]] = []
        for source in infected_neighbors:
            sim = self.graph.get_narrative_similarity(target.node_id, source.node_id)
            hot = self.recsys.hot_score(target, source, sim)
            ranked.append((source, hot, sim))

        ranked.sort(key=lambda x: x[1], reverse=True)
        feed_cap = max(1, target.feed_capacity)
        chosen = ranked[:feed_cap]

        # 历史曝光计数用于回音壁强化：高分内容更容易被持续看见
        for source, _, _ in chosen:
            target.source_exposure_count[source.node_id] = target.source_exposure_count.get(source.node_id, 0) + 1

        return chosen

    def compute_infection_signal(self, target_node_id: int) -> Dict[str, float]:
        """
        计算目标节点在当前时刻的“有效感染信号”。

        返回：
        - pressure: 语义+推荐加权后的感染压力
        - avg_hot_score: 推荐热度均值
        - avg_similarity: 语义相似均值
        - infected_neighbor_ratio: 拓扑层感染邻居比例
        """
        target = self.graph.agents[target_node_id]
        neighbors = self.graph.get_neighbors(target_node_id)
        if not neighbors:
            return {
                "pressure": 0.0,
                "avg_hot_score": 0.0,
                "avg_similarity": 0.0,
                "infected_neighbor_ratio": 0.0,
            }

        infected_neighbors = [
            self.graph.agents[n]
            for n in neighbors
            if self.graph.agents[n].sentiment_state == SentimentState.INFECTED
        ]
        infected_ratio = self.graph.get_bearish_ratio_weighted(target_node_id)
        if not infected_neighbors:
            return {
                "pressure": 0.0,
                "avg_hot_score": 0.0,
                "avg_similarity": 0.0,
                "infected_neighbor_ratio": infected_ratio,
            }

        feed = self._rank_semantic_feed(target, infected_neighbors)
        hot_scores = [item[1] for item in feed]
        similarities = [item[2] for item in feed]
        # 核心感染强度：必须同时满足“语义匹配”和“推荐命中”
        contributions = [hot * sim for (_, hot, sim) in feed]
        base_pressure = float(np.mean(contributions)) if contributions else 0.0

        # 再乘上拓扑感染密度，避免单点异常信息过强
        semantic_pressure = base_pressure * (0.5 + 0.5 * infected_ratio)
        # 将拓扑感染密度作为弱先验注入，避免语义冷启动时期传播几乎停滞。
        pressure = semantic_pressure + 0.35 * infected_ratio
        pressure = float(np.clip(pressure, 0.0, 1.0))

        return {
            "pressure": pressure,
            "avg_hot_score": float(np.mean(hot_scores)) if hot_scores else 0.0,
            "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "infected_neighbor_ratio": infected_ratio,
        }

    def update_sentiment_propagation(self) -> Dict:
        self.graph.tick += 1
        self.graph.update_edge_weights_by_portfolio()
        new_infected = 0
        new_recovered = 0
        new_susceptible = 0
        semantic_sims: List[float] = []
        hot_scores: List[float] = []
        state_changes: List[Tuple[int, SentimentState]] = []

        for node_id, agent in self.graph.agents.items():
            new_state, diagnostics = self._update_single_agent(agent)
            if diagnostics is not None:
                semantic_sims.append(diagnostics["avg_similarity"])
                hot_scores.append(diagnostics["avg_hot_score"])

            if new_state != agent.sentiment_state:
                state_changes.append((node_id, new_state))
                if new_state == SentimentState.INFECTED:
                    new_infected += 1
                elif new_state == SentimentState.RECOVERED:
                    new_recovered += 1
                elif new_state == SentimentState.SUSCEPTIBLE:
                    new_susceptible += 1

        for node_id, new_state in state_changes:
            agent = self.graph.agents[node_id]
            agent.sentiment_history.append(agent.sentiment_state)
            agent.sentiment_state = new_state
            agent.state_duration = 0

        for agent in self.graph.agents.values():
            agent.state_duration += 1

        self._update_lambda_by_herding()
        dist = self.graph.get_network_stats()["sentiment_distribution"]
        stats = {
            "tick": self.graph.tick,
            "new_infected": new_infected,
            "new_recovered": new_recovered,
            "new_susceptible": new_susceptible,
            "avg_semantic_similarity": float(np.mean(semantic_sims)) if semantic_sims else 0.0,
            "avg_hot_score": float(np.mean(hot_scores)) if hot_scores else 0.0,
            **dist,
        }
        self.history.append(stats)
        return stats

    def _update_single_agent(self, agent: AgentNode) -> Tuple[SentimentState, Optional[Dict[str, float]]]:
        state = agent.sentiment_state
        diagnostics: Optional[Dict[str, float]] = None

        if state == SentimentState.SUSCEPTIBLE:
            diagnostics = self.compute_infection_signal(agent.node_id)
            infection_prob = float(np.clip(self.beta * diagnostics["pressure"] * agent.conformity, 0.0, 1.0))
            if random.random() < infection_prob:
                return SentimentState.INFECTED, diagnostics
            return state, diagnostics

        if state == SentimentState.INFECTED:
            # 感染者恢复概率随持续时间上升，但在强回音壁中会被抑制
            diagnostics = self.compute_infection_signal(agent.node_id)
            # 至少经历一个完整传播周期后再允许恢复，避免种子节点首轮即回落
            if agent.state_duration <= 0:
                return state, diagnostics
            echo_strength = min(1.0, np.mean(list(agent.source_exposure_count.values())) / 8.0) if agent.source_exposure_count else 0.0
            recovery_prob = self.gamma * (1 + agent.state_duration * 0.1) * (1.0 - 0.4 * echo_strength)
            recovery_prob = float(np.clip(recovery_prob, 0.0, 1.0))
            if random.random() < recovery_prob:
                return SentimentState.RECOVERED, diagnostics
            return state, diagnostics

        if state == SentimentState.RECOVERED:
            # 免疫衰减：在语义匹配的传播环境下更容易再次变为易感
            diagnostics = self.compute_infection_signal(agent.node_id)
            decay_prob = self.delta * (1.0 + 0.5 * diagnostics["pressure"])
            decay_prob = float(np.clip(decay_prob, 0.0, 1.0))
            if random.random() < decay_prob:
                return SentimentState.SUSCEPTIBLE, diagnostics
            return state, diagnostics

        # BULLISH 或其他扩展状态先保持不变
        return state, diagnostics

    def _update_lambda_by_herding(self) -> None:
        """
        动态损失厌恶更新：
        旧版只看“感染邻居比例”，新版叠加语义感染压力，反映茧房放大效应。
        """
        for node_id, agent in self.graph.agents.items():
            bearish_ratio = self.graph.get_bearish_ratio_weighted(node_id)
            semantic_pressure = self.compute_infection_signal(node_id)["pressure"]
            herding_factor = 1 + self.lambda_amplifier * (0.6 * bearish_ratio + 0.4 * semantic_pressure)
            new_lambda = agent.base_lambda * herding_factor
            agent.lambda_coeff = float(np.clip(new_lambda, agent.base_lambda * 0.8, agent.base_lambda * 2.0))

    def inject_panic(self, n_seeds: int = 10, method: str = "random") -> List[int]:
        if method == "random":
            seeds = random.sample(list(self.graph.agents.keys()), min(n_seeds, len(self.graph.agents)))
        elif method == "influential":
            sorted_agents = sorted(self.graph.agents.items(), key=lambda x: x[1].influence, reverse=True)
            seeds = [a[0] for a in sorted_agents[:n_seeds]]
        elif method == "clustered":
            start_node = random.choice(list(self.graph.agents.keys()))
            seeds = [start_node] + self.graph.get_neighbors(start_node)[: max(0, n_seeds - 1)]
        else:
            seeds = random.sample(list(self.graph.agents.keys()), min(n_seeds, len(self.graph.agents)))

        for node_id in seeds:
            self.graph.agents[node_id].sentiment_state = SentimentState.INFECTED
            self.graph.agents[node_id].state_duration = 0
        return seeds

    def get_propagation_stats(self) -> Dict:
        if not self.history:
            return {}
        infected_counts = [h["infected"] for h in self.history]
        peak = max(infected_counts) if infected_counts else 0
        return {
            "total_ticks": len(self.history),
            "peak_infected": peak,
            "peak_tick": infected_counts.index(peak) if infected_counts else 0,
            "current_stats": self.history[-1],
        }

    def simulate(self, n_ticks: int = 100, initial_infected: int = 10, seed: int = 42) -> List[Dict]:
        # 固定随机种子，避免测试和演示中的传播结果抖动
        random.seed(seed)
        np.random.seed(seed)
        self.inject_panic(initial_infected, method="influential")
        for _ in range(n_ticks):
            self.update_sentiment_propagation()
        return self.history


class NDlibBackendAdapter:
    """Optional NDlib adapter to run standard epidemic models on SocialGraph."""

    _MODEL_MAP = {
        "SIR": ("SIRModel", "epidemics"),
        "SIS": ("SISModel", "epidemics"),
        "IC": ("IndependentCascadesModel", "epidemics"),
    }

    def __init__(
        self,
        social_graph: SocialGraph,
        *,
        model_name: str = "SIR",
        beta: float = 0.3,
        gamma: float = 0.1,
    ):
        self.social_graph = social_graph
        self.model_name = model_name.upper()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._model = None
        self._status_name = {0: SentimentState.SUSCEPTIBLE, 1: SentimentState.INFECTED, 2: SentimentState.RECOVERED}
        self._init_model()

    def _init_model(self) -> None:
        try:  # pragma: no cover - optional dependency
            from ndlib.models import ModelConfig as model_config
            from ndlib.models import epidemics as epidemic_models
        except Exception:
            self._model = None
            return

        model_entry = self._MODEL_MAP.get(self.model_name, self._MODEL_MAP["SIR"])
        model_cls = getattr(epidemic_models, model_entry[0], None)
        if model_cls is None:
            self._model = None
            return

        model = model_cls(self.social_graph.graph)
        config = model_config.Configuration()
        config.add_model_parameter("beta", self.beta)
        if self.model_name in {"SIR", "SIS"}:
            config.add_model_parameter("lambda", self.gamma)
        model.set_initial_status(config)
        self._model = model

    @property
    def available(self) -> bool:
        return self._model is not None

    def seed_infected(self, node_ids: List[int]) -> None:
        if self._model is None:
            for node_id in node_ids:
                if node_id in self.social_graph.agents:
                    self.social_graph.agents[node_id].sentiment_state = SentimentState.INFECTED
            return
        for node_id in node_ids:
            if node_id in self.social_graph.graph:
                self._model.status[node_id] = 1

    def step(self) -> Dict[str, int]:
        if self._model is None:
            local = InformationDiffusion(self.social_graph, beta=self.beta, gamma=self.gamma, delta=0.0)
            stats = local.update_sentiment_propagation()
            return {
                "susceptible": int(stats.get("susceptible", 0)),
                "infected": int(stats.get("infected", 0)),
                "recovered": int(stats.get("recovered", 0)),
            }

        iteration = self._model.iteration()
        status = iteration.get("status", {})
        for node_id, state_id in status.items():
            mapped = self._status_name.get(int(state_id), SentimentState.SUSCEPTIBLE)
            if node_id in self.social_graph.agents:
                self.social_graph.agents[node_id].sentiment_state = mapped

        dist = self.social_graph.get_network_stats()["sentiment_distribution"]
        return {
            "susceptible": int(dist.get("susceptible", 0)),
            "infected": int(dist.get("infected", 0)),
            "recovered": int(dist.get("recovered", 0)),
        }


class EoNBackendAdapter:
    """Optional EoN adapter for numerically robust network SIR simulations."""

    def __init__(self, social_graph: SocialGraph):
        self.social_graph = social_graph

    def simulate_sir(
        self,
        *,
        beta: float = 0.3,
        gamma: float = 0.1,
        tmax: int = 60,
        initial_infected: Optional[List[int]] = None,
    ) -> Dict[str, List[float]]:
        initial = initial_infected or []
        if not initial:
            initial = [random.choice(list(self.social_graph.agents.keys()))] if self.social_graph.agents else []

        try:  # pragma: no cover - optional dependency
            import EoN

            t, S, I, R = EoN.fast_SIR(
                self.social_graph.graph,
                beta,
                gamma,
                initial_infecteds=initial,
                tmax=tmax,
                return_full_data=False,
            )
            return {
                "time": [float(x) for x in np.asarray(t).tolist()],
                "S": [float(x) for x in np.asarray(S).tolist()],
                "I": [float(x) for x in np.asarray(I).tolist()],
                "R": [float(x) for x in np.asarray(R).tolist()],
                "backend": "EoN",
            }
        except Exception:
            return self._fallback_sir(beta=beta, gamma=gamma, tmax=tmax, initial_infected=initial)

    def _fallback_sir(self, *, beta: float, gamma: float, tmax: int, initial_infected: List[int]) -> Dict[str, List[float]]:
        n = max(1, self.social_graph.n_agents)
        i0 = float(min(len(initial_infected), n))
        s = float(n) - i0
        i = i0
        r = 0.0

        times = [0.0]
        s_hist = [s]
        i_hist = [i]
        r_hist = [r]

        for t in range(1, max(tmax, 1) + 1):
            new_inf = beta * s * i / n
            new_rec = gamma * i
            new_inf = float(np.clip(new_inf, 0.0, s))
            new_rec = float(np.clip(new_rec, 0.0, i))

            s -= new_inf
            i += new_inf - new_rec
            r += new_rec

            times.append(float(t))
            s_hist.append(float(s))
            i_hist.append(float(i))
            r_hist.append(float(r))

        return {"time": times, "S": s_hist, "I": i_hist, "R": r_hist, "backend": "fallback"}
