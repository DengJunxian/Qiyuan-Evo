"""
认知 Agent 整合模块

将前景理论、DeepSeek 推理和 RAG 记忆三大模块整合为
完整的 CognitiveAgent，实现"硅基投资者"的模拟。

核心特性:
1. 效用函数对 LLM 决策的覆盖
2. 创伤记忆的自动检索与注入
3. 完整的思维链记录

作者: Civitas Economica Team
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from agents.cognition.utility import (
    ProspectTheory, InvestorType,
    ConfidenceTracker, AnchorTracker
)
from agents.cognition.llm_brain import (
    DeepSeekReasoner, LocalReasoner, ReasoningResult
)
from agents.cognition.memory import TraumaMemory


# ==========================================
# 数据结构
# ==========================================

@dataclass
class CognitiveDecision:
    """
    认知决策结果
    
    包含原始 LLM 决策、前景理论分析、覆盖逻辑和最终动作。
    """
    # 最终决策
    final_action: str
    final_quantity: int = 0
    final_confidence: float = 0.5
    
    # 大模型 原始决策
    llm_action: str = ""
    llm_reasoning: str = ""
    
    # 前景理论分析
    prospect_value: float = 0.0
    pain_gain_ratio: float = 0.0
    decision_bias: str = ""
    
    # 覆盖信息
    was_overridden: bool = False
    override_reason: Optional[str] = None
    
    # 情绪状态
    fear_level: float = 0.0
    greed_level: float = 0.0
    
    # 记忆上下文
    memory_context: str = ""
    
    # 元数据
    inference_time_ms: float = 0.0
    model_used: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "action": self.final_action,
            "qty": self.final_quantity,
            "confidence": self.final_confidence,
            "was_overridden": self.was_overridden,
            "override_reason": self.override_reason,
            "prospect_value": self.prospect_value,
            "fear_level": self.fear_level,
            "greed_level": self.greed_level,
            "model_used": self.model_used
        }


# ==========================================
# 认知 智能体
# ==========================================

class CognitiveAgent:
    """
    认知 Agent - "硅基投资者"
    
    整合三大认知模块:
    1. **前景理论效用**: 模拟损失厌恶、处置效应等非理性行为
    2. **DeepSeek R1 推理**: 获取类人思维链 (CoT)
    3. **创伤记忆 RAG**: 高波动时触发历史教训回忆
    
    核心机制:
    - LLM 提供初始决策建议
    - 前景理论计算效用值
    - 当效用值超过阈值时，覆盖 LLM 决策
    
    Attributes:
        agent_id: Agent 唯一标识
        investor_type: 投资者类型
        prospect: 前景理论计算器
        reasoner: LLM 推理引擎
        memory: RAG 记忆库
    """
    
    def __init__(
        self,
        agent_id: str,
        investor_type: InvestorType = InvestorType.NORMAL,
        api_key: Optional[str] = None,
        use_local_reasoner: bool = False,
        lambda_override: Optional[float] = None,
        risk_aversion_lambda: float = 2.25,
        reference_point: float = 0.0
    ):
        """
        初始化认知 Agent
        
        Args:
            agent_id: Agent 唯一标识
            investor_type: 投资者类型预设
            api_key: DeepSeek API Key
            use_local_reasoner: 是否使用本地规则引擎 (无 API 调用)
            lambda_override: 覆盖损失厌恶系数 (可选)
            risk_aversion_lambda: 损失厌恶系数 (默认2.25)
            reference_point: 参考点/持仓成本
        """
        self.agent_id = agent_id
        self.investor_type = investor_type
        self.reference_point = reference_point
        
        # 初始化前景理论计算器
        if lambda_override is not None:
            self.prospect = ProspectTheory(lambda_coeff=lambda_override)
        else:
            self.prospect = ProspectTheory(investor_type=investor_type, lambda_coeff=risk_aversion_lambda)
        
        # 初始化推理引擎
        if use_local_reasoner:
            self.reasoner = LocalReasoner()
        else:
            self.reasoner = DeepSeekReasoner(api_key=api_key)
        
        # 初始化记忆库
        self.memory = TraumaMemory()
        
        # 初始化心理跟踪器
        self.confidence_tracker = ConfidenceTracker()
        self.anchor_tracker = None  # 将在第一次 make_decision 或持仓时初始化
        
        # 决策历史
        self.decision_history: List[CognitiveDecision] = []
        
        # 覆盖阈值配置
        self.fear_threshold = -0.5
        self.greed_threshold = 0.3    # 贪婪覆盖阈值
    
    def calculate_psychological_value(self, current_price: float) -> float:
        """
        计算当前价格相对于参考点(持仓成本)的心理效用值 V(x)
        
        基于前景理论公式：
        - 盈利时: V = (price - ref)^0.88
        - 亏损时: V = -lambda * (ref - price)^0.88
        
        Args:
            current_price: 当前市场价格
        Returns:
            float: 心理效用值 (正=快乐, 负=痛苦)
        """
        if self.reference_point <= 0:
            return 0.0
        pnl_pct = (current_price - self.reference_point) / self.reference_point
        return self.prospect.calculate_value(pnl_pct)
    
    async def prepare_decision_prompt_async(self, market_state: Dict, account_state: Dict) -> Optional[List[Dict]]:
        """
        [并行化支持] 阶段1: 异步准备提示词 (包含异步记忆检索)
        """
        # 如果是本地推理，直接返回 None
        if isinstance(self.reasoner, LocalReasoner):
            return None
            
        if hasattr(self.reasoner, "build_messages"):
            # 异步检索记忆上下文
            memory_context = await self.memory.get_context_for_decision_async(market_state)
            # 构建 提示词
            return self.reasoner.build_messages(market_state, account_state, memory_context)
            
        return None

    def prepare_decision_prompt(self, market_state: Dict, account_state: Dict) -> Optional[List[Dict]]:
        """
        [并行化支持] 阶段1: 准备提示词 (同步 - 兼容旧接口)
        """
        # 如果是本地推理，直接返回 None
        if isinstance(self.reasoner, LocalReasoner):
            return None
            
        if hasattr(self.reasoner, "build_messages"):
            # 同步检索
            memory_context = self.memory.get_context_for_decision(market_state)
            return self.reasoner.build_messages(market_state, account_state, memory_context)
        return None

    def finalize_decision_from_result(
        self, 
        result: ReasoningResult, 
        market_state: Dict,
        account_state: Dict
    ) -> CognitiveDecision:
        """
        [并行化支持] 阶段2: 根据 LLM 结果生成最终决策
        """
        # 1. 获取 大模型 建议
        # result 已经是 ReasoningResult 对象
        raw_decision = result.decision
        
        # 2. 前景理论效用计算



        pnl = account_state.get("pnl_pct", 0)
        utility = self.prospect.calculate_utility(pnl)
        
        # 3. 覆盖逻辑
        final_action = raw_decision.action
        


        if utility < -150:

            if final_action != "SELL":
                final_action = "SELL"
        



        



        

        return raw_decision

    def make_decision(
        self,
        market_state: Dict,
        account_state: Dict,
        symbol: str = "000001"
    ) -> ReasoningResult:
        """
        执行认知决策过程 (同步模式 - 兼容旧代码)
        """
        start_time = time.time()
        
        pnl_pct = account_state.get('pnl_pct', 0)
        current_price = market_state.get('price', 0)
        avg_cost = account_state.get('avg_cost', 0)
        
        # 初始化/更新锚点
        if self.anchor_tracker is None and avg_cost > 0:
            self.anchor_tracker = AnchorTracker(initial_cost=avg_cost, reference_point=avg_cost)
        elif self.anchor_tracker:
            self.anchor_tracker.update(current_price)
            
        # 获取心理状态描述
        confidence_desc = self.confidence_tracker.get_description()
        anchor_desc = self.anchor_tracker.get_bias_description(current_price) if self.anchor_tracker else ""
        
        # ========== 阶段 1: 记忆检索 ==========
        memory_context = self.memory.get_context_for_decision(market_state)
        
        # ========== 阶段 2: 大模型 推理 ==========
        # 更新参考点并计算心理效用
        if avg_cost > 0:
            self.reference_point = avg_cost
        psychological_value = self.calculate_psychological_value(current_price)
        
        # 注入心理状态到 market_state (供 Brain 提示词 使用)
        market_state["_psychological_value"] = psychological_value
        market_state["_risk_aversion"] = self.prospect.lambda_coeff
        
        # 计算前景值用于本地推理器
        prospect_value = self.prospect.calculate_value(pnl_pct)
        
        if isinstance(self.reasoner, LocalReasoner):
            result = self.reasoner.reason(
                market_state, account_state,
                prospect_value=prospect_value
            )
        else:
            result = self.reasoner.reason(
                market_state, account_state,
                symbol=symbol,
                lambda_coeff=self.prospect.lambda_coeff,
                memory_context=memory_context,
                confidence_desc=confidence_desc,
                anchor_desc=anchor_desc,
                csad=market_state.get('csad', None)
            )
        
        llm_action = result.decision.action
        llm_quantity = result.decision.quantity
        
        # ========== 阶段 3: 前景理论分析 ==========
        prospect_result = self.prospect.calculate_full(pnl_pct)
        
        # ========== 阶段 4: 效用覆盖逻辑 ==========
        final_action, override_reason = self.prospect.should_override_decision(
            llm_action=llm_action,
            pnl=pnl_pct,
            fear_threshold=self.fear_threshold,
            greed_threshold=self.greed_threshold
        )
        
        was_overridden = (final_action != llm_action)
        
        # 如果被覆盖为 持有，数量设为 0
        if was_overridden and final_action == "HOLD":
            final_quantity = 0
        else:
            final_quantity = llm_quantity
        
        # ========== 构建决策结果 ==========
        decision = CognitiveDecision(
            final_action=final_action,
            final_quantity=final_quantity,
            final_confidence=result.decision.confidence,
            llm_action=llm_action,
            llm_reasoning=result.thought_chain,
            prospect_value=prospect_result.subjective_value,
            pain_gain_ratio=prospect_result.pain_gain_ratio,
            decision_bias=prospect_result.decision_bias,
            was_overridden=was_overridden,
            override_reason=override_reason,
            fear_level=result.decision.fear_level,
            greed_level=result.decision.greed_level,
            memory_context=memory_context,
            inference_time_ms=(time.time() - start_time) * 1000,
            model_used=result.model_used
        )
        
        # 记录历史
        self.decision_history.append(decision)
        if len(self.decision_history) > 50:
            self.decision_history.pop(0)
        
        return decision
    
    def record_outcome(
        self,
        pnl: float,
        market_summary: str,
        volatility: float = 0.0
    ) -> None:
        """
        记录交易结果
        
        将结果写入记忆库，供未来决策参考。
        
        Args:
            pnl: 盈亏比例
            market_summary: 市场概况
            volatility: 当时的波动率
        """
        if pnl < -0.05:
            content = f"[亏损 {pnl:.1%}] {market_summary}。教训：{self._generate_lesson(pnl)}"
        elif pnl > 0.05:
            content = f"[盈利 {pnl:.1%}] {market_summary}。成功经验。"
        else:
            content = f"[持平] {market_summary}。"
        
        self.memory.add_memory(content, outcome=pnl, volatility=volatility)
        
        # 更新自信心
        self.confidence_tracker.update(pnl)
    
    def _generate_lesson(self, pnl: float) -> str:
        """生成教训描述"""
        if pnl < -0.10:
            return "重大亏损，需要严格止损纪律"
        elif pnl < -0.05:
            return "中等亏损，应该更谨慎"
        else:
            return "小幅亏损，需要优化入场时机"
    
    def get_emotional_profile(self) -> Dict:
        """
        获取情绪画像
        
        Returns:
            包含情绪包袱、风险偏好等信息的字典
        """
        baggage, baggage_desc = self.memory.get_emotional_baggage()
        
        avg_fear = 0.0
        avg_greed = 0.0
        if self.decision_history:
            avg_fear = sum(d.fear_level for d in self.decision_history) / len(self.decision_history)
            avg_greed = sum(d.greed_level for d in self.decision_history) / len(self.decision_history)
        
        return {
            "agent_id": self.agent_id,
            "investor_type": self.investor_type.value,
            "lambda_coeff": self.prospect.lambda_coeff,
            "emotional_baggage": baggage,
            "baggage_description": baggage_desc,
            "avg_fear_level": avg_fear,
            "avg_greed_level": avg_greed,
            "total_decisions": len(self.decision_history),
            "override_rate": sum(1 for d in self.decision_history if d.was_overridden) / max(1, len(self.decision_history))
        }
    
    def __repr__(self) -> str:
        return (
            f"CognitiveAgent(id={self.agent_id}, "
            f"type={self.investor_type.value}, "
            f"λ={self.prospect.lambda_coeff})"
        )


# ==========================================
# 工厂函数
# ==========================================

def create_panic_retail_agent(agent_id: str, **kwargs) -> CognitiveAgent:
    """创建恐慌型散户 Agent (λ=3.0)"""
    # 显式传递 lambda 以覆盖默认值 2.25
    return CognitiveAgent(
        agent_id, 
        InvestorType.PANIC_RETAIL, 
        risk_aversion_lambda=3.0,
        **kwargs
    )


def create_normal_agent(agent_id: str, **kwargs) -> CognitiveAgent:
    """创建普通投资者 Agent"""
    return CognitiveAgent(agent_id, InvestorType.NORMAL, **kwargs)


def create_quant_agent(agent_id: str, **kwargs) -> CognitiveAgent:
    """创建纪律型量化 Agent (λ=1.0)"""
    return CognitiveAgent(
        agent_id, 
        InvestorType.DISCIPLINED_QUANT, 
        risk_aversion_lambda=1.0,
        **kwargs
    )


def create_population(
    size: int = 10,
    panic_ratio: float = 0.3,
    quant_ratio: float = 0.1
) -> List[CognitiveAgent]:
    """
    创建异质性 Agent 群体
    
    Args:
        size: 群体规模
        panic_ratio: 恐慌散户比例
        quant_ratio: 量化比例
        
    Returns:
        CognitiveAgent 列表
    """
    agents = []
    
    num_panic = int(size * panic_ratio)
    num_quant = int(size * quant_ratio)
    num_normal = size - num_panic - num_quant
    
    for i in range(num_panic):
        agents.append(create_panic_retail_agent(
            f"panic_{i}", use_local_reasoner=True
        ))
    
    for i in range(num_quant):
        agents.append(create_quant_agent(
            f"quant_{i}", use_local_reasoner=True
        ))
    
    for i in range(num_normal):
        agents.append(create_normal_agent(
            f"normal_{i}", use_local_reasoner=True
        ))
    
    return agents



