"""
SOP 政策委员会

受 MetaGPT 框架启发，实现标准化操作程序（SOP）的政策解析流水线。

三阶段流程：
1. 宏观分析员 → 将自然语言政策拆解为经济目标
2. 量化建模师 → 将经济目标映射为具体参数
3. 合规审查员 → 校验参数边界，确保合法合规
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIConnectionError, APITimeoutError, RateLimitError

from config import GLOBAL_CONFIG


class CommitteeRole(Enum):
    """委员会角色"""
    MACRO_ANALYST = "macro_analyst"       # 宏观分析员
    QUANT_MODELER = "quant_modeler"       # 量化建模师
    COMPLIANCE = "compliance_officer"     # 合规审查员


@dataclass
class PolicyGoal:
    """政策目标"""
    target: str          # 目标对象（如"融资成本"）
    direction: str       # 方向
    magnitude: str       # 幅度
    timeframe: str       # 时间框架
    confidence: float    # 置信度 0-1


@dataclass
class ParameterAdjustment:
    """参数调整"""
    param_name: str      # 参数名称
    current_value: float # 当前值
    new_value: float     # 新值
    rationale: str       # 调整理由


@dataclass
class ComplianceCheck:
    """合规检查结果"""
    passed: bool
    violations: List[str]
    warnings: List[str]
    adjusted_params: Dict[str, float]


@dataclass
class PolicyInterpretationResult:
    """政策解读完整结果"""
    original_text: str
    goals: List[PolicyGoal]
    parameters: Dict[str, float]
    compliance: ComplianceCheck
    reasoning_chain: List[str]
    final_state: Dict


class PolicyCommittee:
    """
    政策委员会
    
    通过多 Agent 流水线解析政策文本，
    确保参数生成的准确性和合规性
    """
    
    # 系统角色 提示词
    ROLE_PROMPTS = {
        CommitteeRole.MACRO_ANALYST: """你是一位资深宏观经济分析师，专注于解读中国政策。

你的职责：
1. 分析政策文本的核心意图
2. 识别影响的经济领域（货币、财政、监管等）
3. 提炼政策目标（如"降低融资成本"、"稳定市场信心"）
4. 判断政策力度（温和/适中/强力）

输出格式（JSON）：
{
    "policy_type": "货币政策/财政政策/监管政策/其他",
    "core_intent": "政策核心意图描述",
    "goals": [
        {
            "target": "目标对象",
            "direction": "increase/decrease/stabilize",
            "magnitude": "mild/moderate/significant",
            "timeframe": "immediate/short-term/long-term",
            "confidence": 0.8
        }
    ],
    "affected_sectors": ["银行", "房地产", ...]
}""",

        CommitteeRole.QUANT_MODELER: """你是一位量化金融建模师，专注于将宏观政策转化为市场参数。

你的职责：
1. 将政策目标映射为具体的数值参数
2. 基于历史经验估算参数变化幅度
3. 考虑政策传导的时滞效应

可调整的参数及其范围：
- tax_rate: 交易税率 (0.0001 ~ 0.003)
- risk_free_rate: 无风险利率 (0.01 ~ 0.05)
- liquidity_injection: 流动性注入概率 (0 ~ 0.3)
- fear_factor: 恐慌因子 (-0.5 ~ 0.5)
- volatility_multiplier: 波动率倍数 (0.5 ~ 2.0)
- margin_ratio: 保证金比例 (0.3 ~ 1.0)

输出格式（JSON）：
{
    "parameter_changes": [
        {
            "param": "参数名",
            "old_value": 0.001,
            "new_value": 0.0005,
            "rationale": "降低交易成本以刺激流动性"
        }
    ],
    "expected_impact": {
        "market_direction": "bullish/bearish/neutral",
        "volatility_change": "increase/decrease/stable",
        "duration_days": 30
    }
}""",

        CommitteeRole.COMPLIANCE: """你是一位金融合规审查员，专注于参数边界校验。

你的职责：
1. 检查参数是否超出法定范围
2. 识别可能的政策冲突
3. 确保系统稳定性

合规规则：
- 税率不能为负
- 涨跌幅限制不能超过 ±10%
- 保证金比例不能低于 30%
- 任何单一参数变化幅度不宜超过 50%

输出格式（JSON）：
{
    "passed": true/false,
    "violations": ["违规项描述", ...],
    "warnings": ["警告项描述", ...],
    "adjusted_params": {
        "param_name": "合规后的值"
    }
}"""
    }
    
    # 参数边界（硬性约束）
    PARAM_BOUNDS = {
        "tax_rate": (0.0, 0.01),
        "risk_free_rate": (0.0, 0.10),
        "liquidity_injection": (0.0, 0.5),
        "fear_factor": (-1.0, 1.0),
        "volatility_multiplier": (0.1, 5.0),
        "margin_ratio": (0.3, 1.0),
        "price_limit": (0.05, 0.20)
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        
        if self._api_key:
            from core.model_router import ModelRouter
            self.model_router = ModelRouter(
                deepseek_key=self._api_key,
                zhipu_key=GLOBAL_CONFIG.ZHIPU_API_KEY
            )
        else:
            self.model_router = None
        
        # 解析历史
        self.interpretation_history: List[PolicyInterpretationResult] = []
    
    def _call_agent(self, role: CommitteeRole, context: str) -> Dict:
        """调用单个 Agent"""
        if not self.model_router:
            return {"error": "API 未连接"}
        
        messages = [
            {"role": "system", "content": self.ROLE_PROMPTS[role]},
            {"role": "user", "content": context}
        ]
        
        content, _, _ = self.model_router.sync_call_with_fallback(
            messages,
            priority_models=["deepseek-reasoner", "glm-4-flashx", "deepseek-chat"],
            timeout_budget=30.0
        )
        
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return {"raw": content}
        
        return {"raw": content}
    
    def interpret(self, policy_text: str) -> PolicyInterpretationResult:
        """
        完整的三阶段政策解读流程
        
        Args:
            policy_text: 政策原文
            
        Returns:
            PolicyInterpretationResult
        """
        reasoning_chain = []
        
        # ===== 阶段1: 宏观分析 =====
        reasoning_chain.append("📊 阶段1: 宏观分析员正在解读政策...")
        
        try:
            macro_result = self._call_agent(
                CommitteeRole.MACRO_ANALYST,
                f"请分析以下政策文本：\n\n{policy_text}"
            )
            reasoning_chain.append(f"宏观分析结果: {json.dumps(macro_result, ensure_ascii=False, indent=2)}")
        except Exception as e:
            macro_result = {"error": str(e), "goals": []}
            reasoning_chain.append(f"宏观分析失败: {e}")
        
        # 提取政策目标
        goals = []
        for g in macro_result.get("goals", []):
            goals.append(PolicyGoal(
                target=g.get("target", ""),
                direction=g.get("direction", "stabilize"),
                magnitude=g.get("magnitude", "moderate"),
                timeframe=g.get("timeframe", "short-term"),
                confidence=g.get("confidence", 0.5)
            ))
        
        # ===== 阶段2: 量化建模 =====
        reasoning_chain.append("📈 阶段2: 量化建模师正在映射参数...")
        
        quant_context = f"""
基于宏观分析结果：
{json.dumps(macro_result, ensure_ascii=False)}

请将政策目标映射为具体的市场参数调整。
"""
        
        try:
            quant_result = self._call_agent(
                CommitteeRole.QUANT_MODELER,
                quant_context
            )
            reasoning_chain.append(f"量化建模结果: {json.dumps(quant_result, ensure_ascii=False, indent=2)}")
        except Exception as e:
            quant_result = {"error": str(e), "parameter_changes": []}
            reasoning_chain.append(f"量化建模失败: {e}")
        
        # 收集参数变更
        param_adjustments = {}
        for change in quant_result.get("parameter_changes", []):
            param = change.get("param", "")
            new_val = change.get("new_value", 0)
            if param:
                param_adjustments[param] = new_val
        
        # ===== 阶段3: 合规审查 =====
        reasoning_chain.append("🔍 阶段3: 合规审查员正在校验参数...")
        
        compliance_context = f"""
请审查以下参数调整是否合规：

原始政策：{policy_text}

拟调整参数：
{json.dumps(param_adjustments, ensure_ascii=False, indent=2)}
"""
        
        try:
            compliance_raw = self._call_agent(
                CommitteeRole.COMPLIANCE,
                compliance_context
            )
            reasoning_chain.append(f"合规审查结果: {json.dumps(compliance_raw, ensure_ascii=False, indent=2)}")
        except Exception as e:
            compliance_raw = {"passed": True, "violations": [], "warnings": [str(e)]}
            reasoning_chain.append(f"合规审查异常: {e}")
        
        # 硬性边界校验
        violations = list(compliance_raw.get("violations", []))
        adjusted_params = dict(param_adjustments)
        
        for param, value in param_adjustments.items():
            if param in self.PARAM_BOUNDS:
                low, high = self.PARAM_BOUNDS[param]
                if value < low or value > high:
                    violations.append(f"{param} = {value} 超出范围 [{low}, {high}]")
                    adjusted_params[param] = max(low, min(high, value))
        
        compliance = ComplianceCheck(
            passed=len(violations) == 0,
            violations=violations,
            warnings=compliance_raw.get("warnings", []),
            adjusted_params=adjusted_params
        )
        
        # ===== 生成最终状态 =====
        final_state = self._build_final_state(adjusted_params, macro_result)
        
        result = PolicyInterpretationResult(
            original_text=policy_text,
            goals=goals,
            parameters=adjusted_params,
            compliance=compliance,
            reasoning_chain=reasoning_chain,
            final_state=final_state
        )
        
        self.interpretation_history.append(result)
        
        return result
    
    def _build_final_state(self, params: Dict, macro: Dict) -> Dict:
        """构建最终政策状态"""
        state = {
            "tax_rate": params.get("tax_rate", 0.001),
            "liquidity_injection": params.get("liquidity_injection", 0.0),
            "fear_factor": params.get("fear_factor", 0.0),
            "volatility_multiplier": params.get("volatility_multiplier", 1.0),
            "initial_news": macro.get("core_intent", "政策调整中"),
            "affected_sectors": macro.get("affected_sectors", [])
        }
        return state
    
    def get_reasoning_chain(self, index: int = -1) -> List[str]:
        """获取推理链"""
        if not self.interpretation_history:
            return []
        return self.interpretation_history[index].reasoning_chain
    
    def get_interpretation_summary(self, index: int = -1) -> str:
        """获取解读摘要"""
        if not self.interpretation_history:
            return "尚无政策解读记录"
        
        result = self.interpretation_history[index]
        
        summary_parts = [
            f"📜 **原始政策**: {result.original_text[:100]}...",
            "",
            "📊 **识别到的政策目标**:"
        ]
        
        for goal in result.goals:
            summary_parts.append(f"  - {goal.target}: {goal.direction} ({goal.magnitude})")
        
        summary_parts.append("")
        summary_parts.append("⚙️ **参数调整**:")
        for param, value in result.parameters.items():
            summary_parts.append(f"  - {param}: {value}")
        
        if result.compliance.violations:
            summary_parts.append("")
            summary_parts.append("⚠️ **合规警告**:")
            for v in result.compliance.violations:
                summary_parts.append(f"  - {v}")
        
        return "\n".join(summary_parts)


# 便捷函数
def interpret_policy(policy_text: str, api_key: Optional[str] = None) -> Dict:
    """
    便捷的政策解读函数
    
    Args:
        policy_text: 政策文本
        api_key: API 密钥
        
    Returns:
        可直接用于仿真的参数字典
    """
    committee = PolicyCommittee(api_key)
    result = committee.interpret(policy_text)
    return result.final_state
