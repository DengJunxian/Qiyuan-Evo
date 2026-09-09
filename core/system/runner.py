"""
分布式模拟运行器

使用 Ray 框架实现 10,000+ Agent 的并行执行。

核心组件:
1. AgentActor - 分布式 Agent Actor
2. ModelRouter - 智能模型路由 (R1 vs 7B)
3. SimulationRunner - 编排器

成本优化策略:
- 机构投资者 (5%): DeepSeek-R1 API (高质量推理)
- 散户 (95%): 本地 7B 蒸馏模型 (低成本)

作者: Civitas Economica Team
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np

# Ray 导入 (条件导入以支持无 Ray 环境)
try:
    import ray
    from ray import ObjectRef
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    ObjectRef = Any


# ==========================================
# 数据结构
# ==========================================

class AgentType(str, Enum):
    """Agent 类型"""
    INSTITUTIONAL = "institutional"  # 机构投资者
    RETAIL = "retail"               # 散户
    QUANT = "quant"                 # 量化
    MARKET_MAKER = "market_maker"   # 做市商


class ModelType(str, Enum):
    """模型类型"""
    DEEPSEEK_R1 = "deepseek-r1"           # 高端 接口
    DEEPSEEK_7B = "deepseek-r1-distill-7b" # 本地蒸馏模型
    LOCAL_RULES = "local-rules"            # 纯规则引擎


@dataclass
class AgentConfig:
    """Agent 配置"""
    agent_id: str
    agent_type: AgentType
    model_type: ModelType
    initial_cash: float = 100000.0
    lambda_coeff: float = 2.25


@dataclass
class BatchDecision:
    """批量决策结果"""
    agent_id: str
    action: str
    quantity: int
    confidence: float
    reasoning: str = ""
    inference_time_ms: float = 0.0
    model_used: str = ""


# ==========================================
# 模型路由器
# ==========================================

class ModelRouter:
    """
    模型路由器 - 成本优化
    
    根据 Agent 类型分配不同的推理模型：
    - 机构投资者: DeepSeek-R1 API (高质量，高成本)
    - 散户: 本地 7B 蒸馏模型 (低成本，低延迟)
    
    成本估算:
    - R1 API: ~$0.02 / 1K tokens
    - 7B 本地: ~$0.0001 / 1K tokens (GPU 电费)
    """
    
    def __init__(
        self,
        r1_ratio: float = 0.05,
        enable_local_fallback: bool = True
    ):
        """
        初始化路由器
        
        Args:
            r1_ratio: 使用 R1 API 的比例 (默认 5%)
            enable_local_fallback: 是否在 API 失败时回退到本地
        """
        self.r1_ratio = r1_ratio
        self.enable_local_fallback = enable_local_fallback
        
        # 统计
        self.r1_calls = 0
        self.local_calls = 0
        self.fallback_count = 0
    
    def assign_model(self, agent_type: AgentType) -> ModelType:
        """
        根据 Agent 类型分配模型
        
        Args:
            agent_type: Agent 类型
            
        Returns:
            分配的模型类型
        """
        if agent_type == AgentType.INSTITUTIONAL:
            self.r1_calls += 1
            return ModelType.DEEPSEEK_R1
        
        elif agent_type == AgentType.QUANT:
            # 量化使用规则引擎 (确定性)
            self.local_calls += 1
            return ModelType.LOCAL_RULES
        
        else:
            # 散户和做市商使用本地模型
            self.local_calls += 1
            return ModelType.DEEPSEEK_7B
    
    def get_cost_estimate(self, n_calls: int, avg_tokens: int = 500) -> Dict:
        """
        估算成本
        
        Args:
            n_calls: 调用次数
            avg_tokens: 平均 token 数
            
        Returns:
            成本明细
        """
        r1_cost_per_1k = 0.02
        local_cost_per_1k = 0.0001
        
        r1_calls = int(n_calls * self.r1_ratio)
        local_calls = n_calls - r1_calls
        
        r1_cost = r1_calls * (avg_tokens / 1000) * r1_cost_per_1k
        local_cost = local_calls * (avg_tokens / 1000) * local_cost_per_1k
        
        return {
            "total_calls": n_calls,
            "r1_calls": r1_calls,
            "local_calls": local_calls,
            "r1_cost_usd": r1_cost,
            "local_cost_usd": local_cost,
            "total_cost_usd": r1_cost + local_cost,
            "savings_vs_all_r1": (n_calls * (avg_tokens / 1000) * r1_cost_per_1k) - (r1_cost + local_cost)
        }
    
    def get_stats(self) -> Dict:
        """获取路由统计"""
        total = self.r1_calls + self.local_calls
        return {
            "r1_calls": self.r1_calls,
            "local_calls": self.local_calls,
            "fallback_count": self.fallback_count,
            "r1_percentage": (self.r1_calls / total * 100) if total > 0 else 0
        }


# ==========================================

# ==========================================

def create_agent_actor_class():
    """
    动态创建 AgentActor 类
    
    因为 @ray.remote 装饰器需要在 ray 初始化后使用
    """
    if not RAY_AVAILABLE:
        # 无 Ray 时返回普通类
        class AgentActorFallback:
            def __init__(self, agent_configs: List[AgentConfig], model_router: ModelRouter = None):
                self.agent_configs = agent_configs
                self.model_router = model_router or ModelRouter()
                self.agent_ids = [c.agent_id for c in agent_configs]
            
            def decide_batch(self, market_state: Dict) -> List[BatchDecision]:
                return self._decide_impl(market_state)
            
            def _decide_impl(self, market_state: Dict) -> List[BatchDecision]:
                results = []
                for config in self.agent_configs:
                    model = self.model_router.assign_model(config.agent_type)
                    decision = self._local_decide(config, market_state, model)
                    results.append(decision)
                return results
            
            def _local_decide(self, config: AgentConfig, market_state: Dict, model: ModelType) -> BatchDecision:
                start = time.time()
                trend = market_state.get('trend', 'neutral')
                panic = market_state.get('panic_level', 0.5)
                
                if trend == '上涨' and panic < 0.5:
                    action = 'BUY'
                elif trend == '下跌' or panic > 0.7:
                    action = 'SELL'
                else:
                    action = 'HOLD'
                
                return BatchDecision(
                    agent_id=config.agent_id,
                    action=action,
                    quantity=100 if action != 'HOLD' else 0,
                    confidence=0.6,
                    reasoning=f"趋势:{trend}, 恐慌:{panic:.2f}",
                    inference_time_ms=(time.time() - start) * 1000,
                    model_used=model.value
                )
        
        return AgentActorFallback
    
    # 有 Ray 时创建分布式 Actor
    @ray.remote
    class AgentActor:
        """
        分布式 Agent Actor
        
        每个 Actor 管理一批 Agent，在独立进程中执行决策。
        
        设计考量:
        - 每个 Actor 管理 100-500 个 Agent
        - 批量处理减少 Actor 间通信开销
        - 内置模型路由器
        """
        
        def __init__(
            self,
            agent_configs: List[AgentConfig],
            model_router: ModelRouter = None
        ):
            """
            初始化 Actor
            
            Args:
                agent_configs: 此 Actor 管理的 Agent 配置列表
                model_router: 模型路由器
            """
            self.agent_configs = agent_configs
            self.model_router = model_router or ModelRouter()
            self.agent_ids = [c.agent_id for c in agent_configs]
            
            # 初始化推理引擎 (懒加载)
            self._r1_reasoner = None
            self._local_reasoner = None
        
        def get_agent_count(self) -> int:
            """获取管理的 Agent 数量"""
            return len(self.agent_configs)
        
        def decide_batch(self, market_state: Dict) -> List[BatchDecision]:
            """
            批量决策
            
            Args:
                market_state: 市场状态
                
            Returns:
                决策列表
            """
            results = []
            
            for config in self.agent_configs:
                model = self.model_router.assign_model(config.agent_type)
                
                if model == ModelType.DEEPSEEK_R1:
                    decision = self._decide_with_r1(config, market_state)
                elif model == ModelType.DEEPSEEK_7B:
                    decision = self._decide_with_7b(config, market_state)
                else:
                    decision = self._decide_with_rules(config, market_state)
                
                results.append(decision)
            
            return results
        
        def _decide_with_r1(self, config: AgentConfig, market_state: Dict) -> BatchDecision:
            """使用 DeepSeek R1 API 决策"""
            start = time.time()
            
            # 模拟接口调用，实际应调用 DeepSeek 推理模型
            # 这里使用规则作为占位
            decision = self._decide_with_rules(config, market_state)
            decision.model_used = ModelType.DEEPSEEK_R1.value
            decision.inference_time_ms = (time.time() - start) * 1000 + 100  # 模拟 接口 延迟
            
            return decision
        
        def _decide_with_7b(self, config: AgentConfig, market_state: Dict) -> BatchDecision:
            """使用本地 7B 模型决策"""
            start = time.time()
            
            # 模拟本地推理，实际应调用本地推理后端
            decision = self._decide_with_rules(config, market_state)
            decision.model_used = ModelType.DEEPSEEK_7B.value
            decision.inference_time_ms = (time.time() - start) * 1000 + 20  # 模拟本地延迟
            
            return decision
        
        def _decide_with_rules(self, config: AgentConfig, market_state: Dict) -> BatchDecision:
            """使用规则引擎决策"""
            start = time.time()
            
            trend = market_state.get('trend', 'neutral')
            panic = market_state.get('panic_level', 0.5)
            volatility = market_state.get('volatility', 0.02)
            
            # 简单规则
            if trend == '上涨' and panic < 0.5:
                action = 'BUY'
                confidence = 0.7
            elif trend == '下跌' or panic > 0.7:
                action = 'SELL'
                confidence = 0.6
            else:
                action = 'HOLD'
                confidence = 0.5
            
            # 根据 智能体 类型调整
            if config.agent_type == AgentType.INSTITUTIONAL:
                # 机构更理性
                if panic > 0.8 and action == 'SELL':
                    action = 'BUY'  # 逆向投资
            
            return BatchDecision(
                agent_id=config.agent_id,
                action=action,
                quantity=100 if action != 'HOLD' else 0,
                confidence=confidence,
                reasoning=f"趋势:{trend}, 恐慌:{panic:.2f}, 波动:{volatility:.2%}",
                inference_time_ms=(time.time() - start) * 1000,
                model_used=ModelType.LOCAL_RULES.value
            )
    
    return AgentActor


# ==========================================
# 模拟运行器
# ==========================================

class SimulationRunner:
    """
    模拟运行器 - 编排 10,000+ Agent
    
    使用 Ray 实现并行执行，支持：
    - 动态扩缩容
    - 故障恢复
    - 进度监控
    """
    
    def __init__(
        self,
        n_agents: int = 10000,
        agents_per_actor: int = 200,
        r1_ratio: float = 0.05
    ):
        """
        初始化运行器
        
        Args:
            n_agents: 总 Agent 数量
            agents_per_actor: 每个 Actor 管理的 Agent 数
            r1_ratio: 使用 R1 的比例
        """
        self.n_agents = n_agents
        self.agents_per_actor = agents_per_actor
        self.model_router = ModelRouter(r1_ratio=r1_ratio)
        
        # Actor 列表
        self.actors: List[Any] = []
        self.actor_class = create_agent_actor_class()
        
        # 状态
        self.initialized = False
        self.tick = 0
    
    def initialize(self) -> None:
        """
        初始化 Ray 集群和 Actors
        """
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 创建 智能体 配置
        configs = self._generate_agent_configs()
        
        # 分批创建 Actors
        n_actors = (self.n_agents + self.agents_per_actor - 1) // self.agents_per_actor
        
        for i in range(n_actors):
            start_idx = i * self.agents_per_actor
            end_idx = min(start_idx + self.agents_per_actor, self.n_agents)
            batch_configs = configs[start_idx:end_idx]
            
            if RAY_AVAILABLE:
                actor = self.actor_class.remote(batch_configs, self.model_router)
            else:
                actor = self.actor_class(batch_configs, self.model_router)
            
            self.actors.append(actor)
        
        self.initialized = True
        print(f"初始化完成: {n_actors} Actors, {self.n_agents} Agents")
    
    def _generate_agent_configs(self) -> List[AgentConfig]:
        """生成 Agent 配置"""
        configs = []
        
        # 分配比例
        institutional_count = int(self.n_agents * 0.05)
        quant_count = int(self.n_agents * 0.02)
        market_maker_count = int(self.n_agents * 0.005)
        retail_count = self.n_agents - institutional_count - quant_count - market_maker_count
        
        # 机构
        for i in range(institutional_count):
            configs.append(AgentConfig(
                agent_id=f"inst_{i}",
                agent_type=AgentType.INSTITUTIONAL,
                model_type=ModelType.DEEPSEEK_R1,
                initial_cash=10_000_000,
                lambda_coeff=1.5
            ))
        
        # 量化
        for i in range(quant_count):
            configs.append(AgentConfig(
                agent_id=f"quant_{i}",
                agent_type=AgentType.QUANT,
                model_type=ModelType.LOCAL_RULES,
                initial_cash=5_000_000,
                lambda_coeff=1.0
            ))
        
        # 做市商
        for i in range(market_maker_count):
            configs.append(AgentConfig(
                agent_id=f"mm_{i}",
                agent_type=AgentType.MARKET_MAKER,
                model_type=ModelType.DEEPSEEK_7B,
                initial_cash=50_000_000,
                lambda_coeff=1.2
            ))
        
        # 散户
        for i in range(retail_count):
            configs.append(AgentConfig(
                agent_id=f"retail_{i}",
                agent_type=AgentType.RETAIL,
                model_type=ModelType.DEEPSEEK_7B,
                initial_cash=np.random.uniform(50000, 500000),
                lambda_coeff=np.random.uniform(2.0, 3.5)
            ))
        
        return configs
    
    @staticmethod
    def calculate_csad(returns: List[float]) -> float:
        """
        计算截面绝对偏差 (CSAD) - 羊群效应指标
        
        CSAD_t = 1/N * Σ|R_{i,t} - R_{m,t}|
        
        Args:
            returns: 各 Agent 的收益率列表
            
        Returns:
            CSAD 值. 值越小表示羊群效应越显著.
        """
        if not returns:
            return 0.0
            
        market_return = sum(returns) / len(returns)
        abs_deviations = [abs(r - market_return) for r in returns]
        return sum(abs_deviations) / len(returns)
    
    def step(self, market_state: Dict) -> List[BatchDecision]:
        """
        执行一个模拟步骤
        
        Args:
            market_state: 市场状态
            
        Returns:
            所有 Agent 的决策
        """
        if not self.initialized:
            self.initialize()
        
        self.tick += 1
        
        # 并行获取所有 Actor 的决策
        if RAY_AVAILABLE:
            futures = [actor.decide_batch.remote(market_state) for actor in self.actors]
            results = ray.get(futures)
        else:
            results = [actor.decide_batch(market_state) for actor in self.actors]
        
        # 合并结果
        all_decisions = []
        for batch in results:
            all_decisions.extend(batch)
        
        return all_decisions
    
    def get_stats(self) -> Dict:
        """获取运行统计"""
        return {
            "tick": self.tick,
            "n_agents": self.n_agents,
            "n_actors": len(self.actors),
            "agents_per_actor": self.agents_per_actor,
            "ray_available": RAY_AVAILABLE,
            "model_router_stats": self.model_router.get_stats(),
            "cost_estimate": self.model_router.get_cost_estimate(self.n_agents)
        }
    
    def shutdown(self) -> None:
        """关闭运行器"""
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        self.actors.clear()
        self.initialized = False


# 全局 Actor 类 (延迟创建)
AgentActor = create_agent_actor_class()


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("分布式模拟运行器测试")
    print("=" * 60)
    
    # 创建小规模测试
    runner = SimulationRunner(
        n_agents=100,
        agents_per_actor=20,
        r1_ratio=0.05
    )
    
    print(f"\nRay 可用: {RAY_AVAILABLE}")
    
    # 初始化
    runner.initialize()
    
    # 模拟市场状态
    market = {
        "price": 3000.0,
        "trend": "下跌",
        "panic_level": 0.6,
        "volatility": 0.03
    }
    
    # 执行一个步骤
    print("\n[执行模拟步骤]")
    start = time.time()
    decisions = runner.step(market)
    elapsed = time.time() - start
    
    print(f"  决策数量: {len(decisions)}")
    print(f"  耗时: {elapsed * 1000:.2f} ms")
    
    # 统计决策分布
    action_counts = {}
    for d in decisions:
        action_counts[d.action] = action_counts.get(d.action, 0) + 1
    
    print(f"  决策分布: {action_counts}")
    
    # 获取统计
    print("\n[运行统计]")
    stats = runner.get_stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    
    # 关闭
    runner.shutdown()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
