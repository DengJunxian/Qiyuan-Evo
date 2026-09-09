"""
多角色辩论决策系统

受 TradingAgents 项目启发，实现 Bull vs Bear 内心辩论机制。
通过多人格对抗提升决策质量和可解释性。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


from agents.brain import DeepSeekBrain, ThoughtRecord
from core.validator import (
    RiskCommittee,
    PolicyCommittee,
    aggregate_analyst_cards,
    coerce_analyst_card,
    export_decision_trace,
    validate_analyst_card,
)


class DebateRole(Enum):
    """辩论角色"""
    BULL = "bull"       # 看多派 - 激进，寻找机会
    BEAR = "bear"       # 看空派 - 保守，识别风险
    RISK_MGR = "risk_manager"  # 风控经理 - 审核，一票否决


@dataclass
class DebateMessage:
    """辩论消息"""
    role: DebateRole
    content: str
    timestamp: float = field(default_factory=time.time)
    emotion_score: float = 0.0


@dataclass
class DebateRecord:
    """辩论完整记录"""
    agent_id: str
    timestamp: float
    market_context: Dict
    debate_rounds: List[DebateMessage]
    final_decision: Dict
    risk_approval: bool
    reasoning_summary: str


class DebateBrain(DeepSeekBrain):
    """
    多角色辩论决策大脑
    
    架构设计:
    1. Bull Agent: 寻找买入机会，强调趋势与收益
    2. Bear Agent: 识别风险信号，强调保本与止损
    3. Risk Manager: 审核最终决策，可一票否决
    
    辩论流程:
    Round 1: Bull 陈述 → Bear 反驳
    Round 2: Bear 陈述 → Bull 反驳
    Final:   综合双方观点做出决策
    Risk:    风控审核，可能否决
    """
    
    # 类级别辩论历史存储
    debate_history: Dict[str, List[DebateRecord]] = {}
    
    # 角色 提示词 模板
    ROLE_PROMPTS = {
        DebateRole.BULL: """你是这位投资者内心的「贪婪人格」，代号"牛牛"。
你的核心信念：
- 财富属于敢于冒险的人
- 每次下跌都是上车机会
- 要敢于追涨，抓住趋势
- 损失只是暂时的，坚持就是胜利

你的任务：从当前市场信息中找出所有利好因素，论证为什么应该买入或持有。
语气应当自信、乐观、甚至有点冲动。""",

        DebateRole.BEAR: """你是这位投资者内心的「恐惧人格」，代号"空空"。
你的核心信念：
- 保住本金是第一要务
- 利好出尽是利空
- 宁可错过，不可做错
- 落袋为安，见好就收

你的任务：从当前市场信息中找出所有风险因素，论证为什么应该卖出或观望。
语气应当谨慎、担忧、甚至有点悲观。""",

        DebateRole.RISK_MGR: """你是这位投资者的「风控经理」人格。
你的职责：
- 审核 Bull 和 Bear 的辩论
- 判断最终决策是否合理
- 检查仓位控制是否符合规范
- 确保不会发生灾难性亏损

你拥有一票否决权。如果决策风险过高，你必须否决并给出理由。
审核标准：
1. 单笔交易不超过总资产的20%
2. 亏损超过15%必须止损
3. 极端行情（跌停/涨停）不追高杀跌"""
    }
    
    def __init__(self, agent_id: str, persona: Dict, api_key: Optional[str] = None):
        super().__init__(agent_id, persona, api_key)
        
        # 初始化辩论历史
        if agent_id not in DebateBrain.debate_history:
            DebateBrain.debate_history[agent_id] = []
        
        # 辩论参数
        self.debate_rounds = 2
        self.risk_committee = RiskCommittee()
        self.policy_committee = PolicyCommittee()
    
    def _build_role_system_prompt(self, role: DebateRole) -> str:
        """构建角色系统提示词"""
        base_persona = f"""
你正在扮演一位投资者（{self.agent_id}）内心的一个人格。

投资者基本信息：
- 风险偏好: {self.persona.get('risk_preference', '稳健')}
- 损失厌恶系数: {self.persona.get('loss_aversion', 2.0):.1f}

{self.ROLE_PROMPTS[role]}
"""
        return base_persona
    
    def _build_debate_context(
        self, 
        market_state: Dict, 
        account_state: Dict,
        previous_messages: List[DebateMessage]
    ) -> str:
        """构建辩论上下文"""
        
        # 市场信息
        context = f"""
## 当前市场状态

- 最新价格: {market_state.get('last_price', 0):.2f}
- 昨收价格: {market_state.get('prev_close', 0):.2f}
- 涨跌幅: {market_state.get('change_pct', 0):.2%}
- 成交量变化: {market_state.get('volume_ratio', 1):.1f}倍
- 市场恐慌指数: {market_state.get('panic_level', 0):.2f}
- 最新消息: {market_state.get('news', '无')}

## 账户状态

- 可用资金: ¥{account_state.get('cash', 0):,.0f}
- 持仓市值: ¥{account_state.get('market_value', 0):,.0f}
- 持仓成本: ¥{account_state.get('cost_basis', 0):,.0f}
- 当前盈亏: {account_state.get('pnl_pct', 0):.2%}
"""
        
        # 添加已有辩论记录
        if previous_messages:
            context += "\n## 已有辩论记录\n\n"
            for msg in previous_messages:
                role_name = {
                    DebateRole.BULL: "🐂 牛牛(看多)",
                    DebateRole.BEAR: "🐻 空空(看空)",
                    DebateRole.RISK_MGR: "🛡️ 风控经理"
                }.get(msg.role, "未知")
                context += f"**{role_name}**: {msg.content}\n\n"
        
        return context
    
    def _call_role_api(self, role: DebateRole, context: str) -> str:
        """调用 API 获取角色发言"""
        if not self.model_router:
            return f"[{role.value}] 路由未连接"
        
        messages = [
            {"role": "system", "content": self._build_role_system_prompt(role)},
            {"role": "user", "content": context + "\n\n请发表你的观点（100-200字）："}
        ]
        
        content, _, _ = self.model_router.sync_call_with_fallback(
            messages,
            timeout_budget=15.0
        )
        return content
    
    def _synthesize_decision(
        self, 
        debate_messages: List[DebateMessage],
        market_state: Dict,
        account_state: Dict
    ) -> Tuple[Dict, str]:
        """综合辩论结果做出决策"""
        
        # 汇总辩论内容
        debate_summary = "\n".join([
            f"{'🐂' if m.role == DebateRole.BULL else '🐻'} {m.content}" 
            for m in debate_messages
        ])
        
        synthesis_prompt = f"""
作为这位投资者的理性决策中心，你需要综合内心两个人格的辩论，做出最终决策。

## 辩论回顾
{debate_summary}

## 决策要求
1. 权衡双方观点的合理性
2. 考虑当前账户状态
3. 做出 BUY/SELL/HOLD 决策

请用 JSON 格式输出：
{{"action": "BUY/SELL/HOLD", "qty": 数量, "reason": "综合理由"}}
"""
        
        messages = [
            {"role": "system", "content": "你是一个理性的投资决策者，善于综合不同观点。"},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        content, reasoning, _ = self.model_router.sync_call_with_fallback(
            messages,
            timeout_budget=20.0
        )
        
        decision = self._extract_json(content)
        return decision, reasoning or ""
    
    def _risk_review(
        self, 
        decision: Dict, 
        account_state: Dict,
        debate_messages: List[DebateMessage]
    ) -> Tuple[bool, str]:
        """风控审核"""
        
        # 基础规则校验
        action = decision.get('action', 'HOLD')
        qty = decision.get('qty', 0)
        cash = account_state.get('cash', 0)
        market_value = account_state.get('market_value', 0)
        pnl_pct = account_state.get('pnl_pct', 0)
        
        total_assets = cash + market_value
        
        # 规则1: 单笔不超过20%
        if action == 'BUY' and qty * 3000 > total_assets * 0.2:  # 假设价格3000
            return False, "💀 否决：单笔交易超过总资产20%，风险过高"
        
        # 规则2: 亏损超过15%必须止损
        if pnl_pct < -0.15 and action != 'SELL':
            return False, "💀 否决：亏损已超15%，必须止损，不得加仓或持有"
        
        # 规则3: 极端行情不追高杀跌
        # (这里简化处理)
        
        # 通过 大模型 进行更智能的审核
        if self.model_router:
            review_prompt = f"""
作为风控经理，审核以下交易决策：

决策: {decision}
账户盈亏: {pnl_pct:.2%}
可用资金: ¥{cash:,.0f}

辩论摘要:
{chr(10).join([f"- {m.role.value}: {m.content[:100]}..." for m in debate_messages[:4]])}

请判断是否批准（JSON格式）：
{{"approved": true/false, "reason": "理由"}}
"""
            messages = [
                {"role": "system", "content": self.ROLE_PROMPTS[DebateRole.RISK_MGR]},
                {"role": "user", "content": review_prompt}
            ]
            
            content, _, _ = self.model_router.sync_call_with_fallback(
                messages,
                timeout_budget=10.0,
                fallback_response='{"approved": true, "reason": "风控服务降级，默认通过"}'
            )
            
            result = self._extract_json(content)
            return result.get('approved', True), result.get('reason', '风控请求处理')
        
        return True, "✅ 风控通过"
    
    def think_with_debate(
        self,
        market_state: Dict,
        account_state: Dict
    ) -> Dict:
        """Debate flow upgraded to structured evidence cards + calibrated manager decision."""
        debate_messages: List[DebateMessage] = []

        context = self._build_debate_context(market_state, account_state, [])
        try:
            bull_speech = self._call_role_api(
                DebateRole.BULL,
                context + "\n\nProvide bullish case in concise form.",
            )
        except Exception as exc:
            bull_speech = f"debate_error:{exc}"
        debate_messages.append(DebateMessage(role=DebateRole.BULL, content=bull_speech, emotion_score=0.4))

        context = self._build_debate_context(market_state, account_state, debate_messages)
        try:
            bear_speech = self._call_role_api(
                DebateRole.BEAR,
                context + "\n\nProvide bearish case in concise form.",
            )
        except Exception as exc:
            bear_speech = f"debate_error:{exc}"
        debate_messages.append(DebateMessage(role=DebateRole.BEAR, content=bear_speech, emotion_score=-0.4))

        if self.debate_rounds >= 2:
            context = self._build_debate_context(market_state, account_state, debate_messages)
            try:
                risk_speech = self._call_role_api(
                    DebateRole.RISK_MGR,
                    context + "\n\nProvide risk committee perspective.",
                )
            except Exception as exc:
                risk_speech = f"debate_error:{exc}"
            debate_messages.append(DebateMessage(role=DebateRole.RISK_MGR, content=risk_speech, emotion_score=0.0))
        else:
            risk_speech = "risk_manager_not_activated"

        panic = float(market_state.get('panic_level', 0.0) or 0.0)
        pnl_pct = float(account_state.get('pnl_pct', 0.0) or 0.0)

        analyst_cards = [
            coerce_analyst_card(
                {
                    "analyst_id": f"{self.agent_id}_bull",
                    "thesis": "Bull case from debate",
                    "evidence": [
                        {"type": "price", "content": bull_speech[:280], "weight": 0.70},
                        {"type": "news", "content": str(market_state.get('news', '')), "weight": 0.55},
                    ],
                    "time_horizon": "intraday",
                    "risk_tags": ["overcrowding"] if panic < 0.35 else [],
                    "confidence": 0.62,
                    "counterarguments": [bear_speech[:180]],
                    "recommended_action": "buy",
                },
                analyst_id=f"{self.agent_id}_bull",
            ),
            coerce_analyst_card(
                {
                    "analyst_id": f"{self.agent_id}_bear",
                    "thesis": "Bear case from debate",
                    "evidence": [
                        {"type": "risk", "content": bear_speech[:280], "weight": 0.72},
                        {"type": "macro", "content": f"panic={panic:.2f}", "weight": 0.60},
                    ],
                    "time_horizon": "swing",
                    "risk_tags": ["panic", "liquidity"] if panic > 0.45 else ["liquidity"],
                    "confidence": 0.60,
                    "counterarguments": [bull_speech[:180]],
                    "recommended_action": "sell",
                },
                analyst_id=f"{self.agent_id}_bear",
            ),
            coerce_analyst_card(
                {
                    "analyst_id": f"{self.agent_id}_risk_committee",
                    "thesis": "Risk manager synthesis",
                    "evidence": [
                        {"type": "risk", "content": risk_speech[:280], "weight": 0.80},
                        {"type": "risk", "content": f"pnl_pct={pnl_pct:.3f}", "weight": 0.70},
                    ],
                    "time_horizon": "intraday",
                    "risk_tags": ["panic", "liquidity", "overcrowding"],
                    "confidence": 0.68,
                    "counterarguments": ["debate may overfit current narrative"],
                    "recommended_action": "reduce_risk" if panic > 0.5 or pnl_pct < -0.08 else "hold",
                },
                analyst_id=f"{self.agent_id}_risk_committee",
            ),
        ]
        validated_cards: List[Dict] = []
        for idx, card in enumerate(analyst_cards):
            valid, _ = validate_analyst_card(card)
            if valid:
                validated_cards.append(card)
            else:
                validated_cards.append(
                    coerce_analyst_card({}, analyst_id=f"{self.agent_id}_fallback_{idx}")
                )
        analyst_cards = validated_cards

        risk_alert = self.risk_committee.assess(
            {
                "cvar": float(market_state.get("risk_cvar", pnl_pct - 0.02)),
                "max_drawdown": float(market_state.get("max_drawdown", pnl_pct)),
                "turnover": float(market_state.get("turnover", abs(panic) * 1.2)),
                "crowding": float(market_state.get("crowding", abs(panic))),
                "volatility_spike": float(market_state.get("volatility_spike", 1.0 + abs(panic) * 2.0)),
            }
        ).to_dict()
        policy_update = self.policy_committee.translate(str(market_state.get("policy_description", "") or market_state.get("news", "")))

        manager_final_card = aggregate_analyst_cards(analyst_cards, risk_alert=risk_alert)
        decision = self._decision_from_manager_card(manager_final_card, account_state, market_state)

        risk_approved, risk_reason = self._risk_review(decision, account_state, debate_messages)
        if not risk_approved:
            decision = {"action": "HOLD", "qty": 0, "price": float(market_state.get('last_price', 0.0) or 0.0)}

        reasoning_summary = (
            f"debate_manager_action={manager_final_card.get('recommended_action', 'hold')}; "
            f"calibrated_conf={manager_final_card.get('calibrated_confidence', 0.0):.3f}; "
            f"risk_level={risk_alert.get('level', 'normal')}"
        )

        debate_record = DebateRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            market_context=market_state,
            debate_rounds=debate_messages,
            final_decision=decision,
            risk_approval=risk_approved,
            reasoning_summary=reasoning_summary,
        )
        DebateBrain.debate_history[self.agent_id].append(debate_record)
        if len(DebateBrain.debate_history[self.agent_id]) > 10:
            DebateBrain.debate_history[self.agent_id].pop(0)

        avg_emotion = sum(m.emotion_score for m in debate_messages) / len(debate_messages) if debate_messages else 0.0
        thought_record = ThoughtRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            reasoning_content=self._format_debate_for_display(debate_messages),
            emotion_score=avg_emotion,
            decision=decision,
            market_context=market_state,
        )
        DeepSeekBrain.thought_history[self.agent_id].append(thought_record)

        trace_path = export_decision_trace(
            {
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "debate_messages": [
                    {"role": msg.role.value, "content": msg.content, "emotion_score": msg.emotion_score}
                    for msg in debate_messages
                ],
                "analyst_cards": analyst_cards,
                "manager_final_card": manager_final_card,
                "risk_alert": risk_alert,
                "policy_conditions": policy_update,
                "decision": decision,
            }
        )

        return {
            "decision": decision,
            "debate_messages": debate_messages,
            "risk_approved": risk_approved,
            "risk_reason": risk_reason,
            "emotion_score": avg_emotion,
            "reasoning": reasoning_summary,
            "analyst_cards": analyst_cards,
            "contradiction_matrix": manager_final_card.get("contradiction_matrix", {}),
            "manager_final_card": manager_final_card,
            "risk_alerts": risk_alert,
            "policy_conditions": policy_update,
            "calibration": manager_final_card.get("calibration", {}),
            "decision_trace_path": trace_path,
        }

    def _format_debate_for_display(self, messages: List[DebateMessage]) -> str:
        """格式化辩论记录供展示"""
        lines = ["=== 内心辩论记录 ===\n"]
        
        for i, msg in enumerate(messages):
            role_emoji = {
                DebateRole.BULL: "🐂",
                DebateRole.BEAR: "🐻",
                DebateRole.RISK_MGR: "🛡️"
            }.get(msg.role, "❓")
            
            role_name = {
                DebateRole.BULL: "牛牛(看多)",
                DebateRole.BEAR: "空空(看空)",
                DebateRole.RISK_MGR: "风控"
            }.get(msg.role, "未知")
            
            lines.append(f"{role_emoji} **{role_name}**:")
            lines.append(msg.content)
            lines.append("")
        
        return "\n".join(lines)
    
    # 重写 think 方法以支持辩论模式
    def think(self, market_state: Dict, account_state: Dict) -> Dict:
        """重写思考方法，使用辩论机制"""
        result = self.think_with_debate(market_state, account_state)
        
        return {
            "decision": result["decision"],
            "reasoning": result["reasoning"],
            "raw_content": self._format_debate_for_display(result["debate_messages"]),
            "emotion_score": result["emotion_score"],
            "analyst_cards": result.get("analyst_cards", []),
            "contradiction_matrix": result.get("contradiction_matrix", {}),
            "manager_final_card": result.get("manager_final_card", {}),
            "risk_alerts": result.get("risk_alerts", {}),
            "policy_conditions": result.get("policy_conditions", {}),
            "calibration": result.get("calibration", {}),
            "decision_trace_path": result.get("decision_trace_path", ""),
        }
