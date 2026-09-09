"""
认知 Agent 工具库

包含:
1. 投资者类型定义
2. 前景理论参数包装器
3. 心理状态跟踪器
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

# 复用 core.behavioral_finance 中的基础函数，但在本模块增强封装
from core.behavioral_finance import ProspectTheoryParams, prospect_value, prospect_utility, probability_weight

class InvestorType(Enum):
    """投资者类型"""
    NORMAL = "normal"             # 普通投资者 (基准)
    PANIC_RETAIL = "panic"        # 恐慌型散户 (高羊群, 高损失厌恶)
    DISCIPLINED_QUANT = "quant"   # 纪律型量化 (理性, 低偏差)
    INSTITUTION = "institution"   # 机构投资者 (理性, 深度思考)

@dataclass
class ProspectValue:
    """前景理论估值结果"""
    subjective_value: float       # 综合效用值
    reference_point: float        # 参考点
    pain_gain_ratio: float = 0.0  # 痛苦/快乐比率
    decision_bias: str = ""       # 偏差描述

class ProspectTheory:
    """前景理论计算器包装器"""
    
    def __init__(self, investor_type: InvestorType = InvestorType.NORMAL, lambda_coeff: Optional[float] = None):
        self.params = ProspectTheoryParams()
        self.investor_type = investor_type
        
        # 根据类型调整参数
        if lambda_coeff:
            self.params.lambda_ = lambda_coeff
        elif investor_type == InvestorType.PANIC_RETAIL:
            self.params.lambda_ = 3.5  # 更高的损失厌恶
            self.params.alpha = 0.88
            self.params.beta = 0.88
        elif investor_type == InvestorType.DISCIPLINED_QUANT or investor_type == InvestorType.INSTITUTION:
            self.params.lambda_ = 1.0  # 风险中性/理性
            self.params.alpha = 1.0    # 线性效用
            self.params.beta = 1.0
        
        # 保存系数供外部访问
        self.lambda_coeff = self.params.lambda_
            
    def calculate_utility(self, pnl_pct: float) -> float:
        """计算效用值 (简化版)"""
        return prospect_utility(pnl_pct, loss_aversion=self.params.lambda_)

    def calculate_value(self, pnl_pct: float) -> float:
        """计算前景值 V(x)"""
        return prospect_value(pnl_pct, reference=0, params=self.params)
        
    def calculate_full(self, pnl_pct: float) -> ProspectValue:
        """计算完整的前景理论指标"""
        val = self.calculate_value(pnl_pct)
        
        # 计算痛苦/快乐比率 (即便在盈利时，也可能衡量潜在损失的痛苦)
        if pnl_pct > 0:
             # 如果是盈利，计算如果亏损同等金额会有多痛苦
             pain = abs(prospect_value(-pnl_pct, 0, self.params))
             gain = val
             ratio = pain / gain if gain > 0 else 999.0
             bias = f"损失厌恶: 失去当前浮盈的痛苦是获得的 {ratio:.1f} 倍"
        elif pnl_pct < 0:
             # 如果是亏损，计算当前的痛苦与同等盈利的快乐之比
             pain = abs(val)
             gain = prospect_value(-pnl_pct, 0, self.params)
             ratio = pain / gain if gain > 0 else 999.0
             bias = f"痛苦倍数: 当前亏损的痛苦是同等盈利快乐的 {ratio:.1f} 倍"
        else:
             ratio = self.params.lambda_
             bias = "中性"
             
        return ProspectValue(
            subjective_value=val,
            reference_point=0,
            pain_gain_ratio=ratio,
            decision_bias=bias
        )

    def should_override_decision(
        self, 
        llm_action: str, 
        pnl: float,
        fear_threshold: float = -0.5,
        greed_threshold: float = 0.3
    ) -> Tuple[str, Optional[str]]:
        """
        判断是否应该覆盖 LLM 的决策
        
        基于前景理论效用值：
        - 极度痛苦时 (Utility < fear_threshold) -> 可能恐慌卖出或死扛 (取决于具体逻辑，这里简化为 HOLD 或 SELL)
        - 极度贪婪时 (Utility > greed_threshold) -> 可能急于落袋为安
        
        Args:
            llm_action: LLM 建议的动作
            pnl: 当前盈亏比例
            fear_threshold: 恐惧阈值 (负数)
            greed_threshold: 贪婪阈值 (正数)
            
        Returns:
            (final_action, reason)
        """
        utility = self.calculate_value(pnl)
        
        # 场景1：处置效应 - 盈利时急于卖出
        if pnl > 0.05 and llm_action == "BUY" and self.investor_type == InvestorType.PANIC_RETAIL:
             # 散户在盈利时往往厌恶风险，不愿意加仓
             return "HOLD", "处置效应覆盖：盈利时厌恶风险，拒绝加仓"

        # 场景2：损失厌恶 - 亏损时死扛
        # 前景理论认为在亏损区域，人倾向于风险偏好 (赌徒心理)，希望能回本
        if pnl < -0.10 and llm_action == "SELL" and self.investor_type == InvestorType.PANIC_RETAIL:
             # 如果亏损严重，且 大模型 建议止损，散户可能因为损失厌恶而死扛

             if utility > fear_threshold * 1.5: # 还没到崩溃点
                 return "HOLD", "损失厌恶覆盖：不愿确认亏损，选择死扛"
        

        # 效用值极低（如 -2.0 以下），心理防线崩溃
        if utility < fear_threshold:
             if llm_action == "BUY":
                 return "HOLD", f"极度恐惧覆盖(V={utility:.2f})：心理崩溃，无法执行买入"
             elif llm_action == "HOLD":
                 # 可能会恐慌卖出
                 return "SELL", f"极度恐惧覆盖(V={utility:.2f})：心理崩溃，恐慌止损"

        return llm_action, None

def weight_probability(p: float) -> Tuple[float, float]:
    """计算概率权重 (Gain/Loss)"""
    w_gain = probability_weight(p, is_gain=True)
    w_loss = probability_weight(p, is_gain=False)
    return w_gain, w_loss

class ConfidenceTracker:
    """信心跟踪器"""
    def __init__(self, initial: float = 0.5, decay: float = 0.95):
        self.confidence = initial
        self.decay = decay
        
    def update(self, pnl: float):
        """根据盈亏更新信心"""
        if pnl > 0:
            # 盈利增强信心
            boost = min(0.1, pnl * 2)
            self.confidence = min(1.0, self.confidence + boost)
        else:
            # 亏损打击信心
            penalty = min(0.05, abs(pnl) * 1.0)
            self.confidence = max(0.0, self.confidence - penalty)
            
    def get_description(self) -> str:
        if self.confidence > 0.8:
            return "极度自信"
        if self.confidence > 0.6:
            return "比较自信"
        if self.confidence < 0.2:
            return "极度沮丧"
        if self.confidence < 0.4:
            return "缺乏信心"
        return "心态平和"

class AnchorTracker:
    """锚定效应跟踪器"""
    def __init__(self, initial_cost: float, reference_point: float):
        self.initial_cost = initial_cost
        self.reference_point = reference_point # 心理参考点
        
    def update(self, current_price: float):
        # 锚点会随时间缓慢移动
        alpha = 0.05
        self.reference_point = (1 - alpha) * self.reference_point + alpha * current_price
        
    def get_bias_description(self, current_price: float) -> str:
        diff_pct = (current_price - self.reference_point) / self.reference_point
        if abs(diff_pct) < 0.02:
            return "价格接近心理锚点"
        elif diff_pct > 0:
            return f"价格高于心理锚点 {diff_pct:.1%}"
        else:
            return f"价格低于心理锚点 {abs(diff_pct):.1%}"

# 工厂函数
def create_panic_retail(): return InvestorType.PANIC_RETAIL
def create_normal_investor(): return InvestorType.NORMAL
def create_disciplined_quant(): return InvestorType.DISCIPLINED_QUANT

def calculate_prospect_value(
    pnl_pct: float,
    investor_type: InvestorType = InvestorType.NORMAL,
    lambda_coeff: Optional[float] = None
) -> ProspectValue:
    """
    模块级便捷函数：计算前景理论综合估值
    
    Args:
        pnl_pct: 盈亏百分比
        investor_type: 投资者类型
        lambda_coeff: 自定义损失厌恶系数（可选）
        
    Returns:
        ProspectValue 对象，包含效用值、参考点、痛苦/快乐比率
    """
    pt = ProspectTheory(investor_type=investor_type, lambda_coeff=lambda_coeff)
    return pt.calculate_full(pnl_pct)
