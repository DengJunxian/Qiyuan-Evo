
import hashlib
import json
import os
import re
import time
from collections import OrderedDict, defaultdict, deque
import numpy as np
from typing import Dict, List, Any, Optional, Mapping
from dataclasses import dataclass, field
from openai import OpenAI
from config import GLOBAL_CONFIG
from core.validator import (
    PolicyCommittee,
    RiskCommittee,
    aggregate_analyst_cards,
    coerce_analyst_card,
    export_decision_trace,
    validate_analyst_card,
)
from agents.cognition.layered_memory import LayeredMemory


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def stable_json_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 hash for a JSON-serializable payload."""
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    except TypeError:
        encoded = json.dumps(str(payload), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion for noisy market/account payloads."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_feature_flags(overrides: Optional[Mapping[str, bool]] = None) -> Dict[str, bool]:
    """Resolve feature flags from environment defaults plus explicit overrides."""
    defaults = {
        "manager_orchestration_v1": True,
        "manager_external_analysts_v1": False,
        "manager_debate_v1": False,
        "manager_metadata_v1": True,
        "population_protocol_v1": True,
    }
    flags = {name: env_flag(f"CIVITAS_{name.upper()}", default) for name, default in defaults.items()}
    if overrides:
        for name, value in overrides.items():
            flags[str(name)] = bool(value)
    return flags


def build_runtime_metadata(
    *,
    seed: int,
    config: Mapping[str, Any],
    snapshot: Optional[Mapping[str, Any]] = None,
    feature_flags: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """Create reproducible metadata for logs, reports, and tests."""
    snapshot_payload = dict(snapshot or {})
    feature_payload = dict(feature_flags or {})
    config_payload = dict(config)
    config_hash = stable_json_hash(
        {
            "seed": int(seed),
            "config": config_payload,
            "feature_flags": feature_payload,
            "snapshot": snapshot_payload,
        }
    )
    return {
        "seed": int(seed),
        "config_hash": config_hash,
        "config": config_payload,
        "snapshot": snapshot_payload,
        "feature_flags": feature_payload,
    }

# --- 1. 向量记忆模块 ---

@dataclass
class MemoryFragment:
    """记忆碎片：存储单次交易的上下文与结果"""
    content: str
    vector: np.ndarray
    timestamp: float
    outcome_score: float  # -1.0 (惨败) ~ 1.0 (大胜)

class VectorMemory:
    """
    基于 Numpy 的轻量级 RAG 记忆库。
    用于存储和检索"过去的教训"。
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.fragments: List[MemoryFragment] = []
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """
        [模拟] 生成确定性的伪向量。
        在生产环境中，此处应调用 client.embeddings.create()。
        """
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.rand(self.dimension)
        return vec / np.linalg.norm(vec) # 归一化

    def add_memory(self, text: str, outcome: float):
        """写入记忆"""
        vector = self._mock_embedding(text)
        fragment = MemoryFragment(
            content=text,
            vector=vector,
            timestamp=time.time(),
            outcome_score=outcome
        )
        self.fragments.append(fragment)
        # 保持记忆库不过大，保留最近100条
        if len(self.fragments) > 100:
            self.fragments.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索最相关的记忆"""
        if not self.fragments:
            return []
            
        query_vec = self._mock_embedding(query)
        scores = []
        
        for frag in self.fragments:
            # 余弦相似度
            cosine_sim = np.dot(query_vec, frag.vector)
            scores.append((cosine_sim, frag.content))
            
        # 按相似度降序排列
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]


# --- 2. 智能体状态 ---

@dataclass
class AgentState:
    """
    智能体持久化状态
    
    实现 OODA 循环中的状态持久化，跟踪信心指数和人格演化。
    """
    confidence: float = 50.0  # 信心指数 (0-100)
    consecutive_losses: int = 0  # 连续亏损次数
    consecutive_wins: int = 0  # 连续盈利次数
    total_trades: int = 0  # 总交易次数
    total_pnl: float = 0.0  # 累计盈亏
    
    # 人格演化相关
    evolved_risk_preference: Optional[str] = None  # 演化后的风险偏好（覆盖初始设定）
    trauma_events: int = 0  # 创伤事件次数（如单日亏损超过10%）
    
    def update_after_trade(self, pnl: float, pnl_pct: float):
        """
        交易后更新状态
        
        Args:
            pnl: 盈亏金额
            pnl_pct: 盈亏百分比
        """
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # 信心衰减（指数衰减）
            self.confidence *= 0.92
            
            # 创伤事件检测
            if pnl_pct < -0.10:  # 单日亏损超过10%
                self.trauma_events += 1
                self.confidence *= 0.8  # 额外信心打击
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # 信心恢复（渐进恢复）
            self.confidence = min(100, self.confidence * 1.05 + 1)
        
        # 人格演化检测
        self._check_personality_evolution()
    
    def _check_personality_evolution(self):
        """
        基于经验的人格演化
        
        模拟"创伤学习"：如果一个激进型Agent经历多次破产，
        它应自动演化为保守型。
        """
        # 规则1: 连续亏损3次且信心低于30 -> 演化为保守型
        if self.consecutive_losses >= 3 and self.confidence < 30:
            self.evolved_risk_preference = "极度保守"
        
        # 规则2: 经历2次以上创伤事件 -> 演化为保守型
        elif self.trauma_events >= 2:
            self.evolved_risk_preference = "保守"
        
        # 规则3: 连续盈利5次且信心高于80 -> 演化为激进型
        elif self.consecutive_wins >= 5 and self.confidence > 80:
            self.evolved_risk_preference = "激进"
        
        # 规则4: 信心恢复到中等水平 -> 可能恢复原本风格
        elif 40 < self.confidence < 70 and self.trauma_events == 0:
            self.evolved_risk_preference = None  # 恢复原本设定
    
    @property
    def decision_modifier(self) -> float:
        """
        决策修正系数
        
        低信心的Agent即使看到利好信号，也会降低下单比例。
        
        Returns:
            修正系数 (0.0 - 1.0)
        """
        # 信心映射到决策力度
        # 信心50 -> 系数1.0
        # 信心25 -> 系数0.5
        # 信心0 -> 系数0.2 (即使完全丧失信心，仍保留20%决策力)
        return max(0.2, self.confidence / 50.0)
    
    @property
    def herd_susceptibility(self) -> float:
        """
        羊群效应易感性
        
        低信心的Agent更容易受到羊群效应的影响。
        
        Returns:
            易感性系数 (0.0 - 1.0)，越高越容易跟风
        """
        # 信心低 -> 易感性高
        return max(0.0, min(1.0, (100 - self.confidence) / 80))

# --- 2. 思维链记录 ---

@dataclass
class ThoughtRecord:
    """单次思考的完整记录，用于fMRI可视化"""
    agent_id: str
    timestamp: float
    reasoning_content: str  # 完整思维链
    emotion_score: float
    decision: Dict
    market_context: Dict = field(default_factory=dict)

# --- 3. 多模型智能体大脑 ---

class DeepSeekBrain:
    """
    多模型驱动的智能体大脑。
    
    支持模型：
    - DeepSeek: deepseek-reasoner, deepseek-chat
    - 智谱: glm-4-flashx（快速模式或降级）
    
    特性：
    1. 多模型路由与自动降级
    2. 快速思考模式（本地规则引擎）
    3. 提取 reasoning_content (思维链)
    4. 强制 JSON 输出 (结构化决策)
    5. 前景理论人格植入
    6. 情绪分析与思维链历史记录
    7. 决策缓存（相似prompt复用）
    """
    
    # 类级别的思维链历史存储（用于fMRI可视化）
    # 使用 defaultdict 和 deque 自动管理内存，限制每个 智能体 保留最近 20 条记录
    thought_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    
    # 类级别的决策缓存（相似市场状态复用决策）
    # 使用 OrderedDict 实现 最近最少使用 缓存
    decision_cache: OrderedDict = OrderedDict()
    CACHE_MAX_SIZE = 100  # 最大缓存条目
    CACHE_TTL_STEPS = 5   # 缓存有效步数
    
    @classmethod
    def clear_memory(cls):
        """显式清空类级别的静态内存"""
        cls.thought_history.clear()
        cls.decision_cache.clear()
    
    def __init__(self, agent_id: str, persona: Dict, api_key: Optional[str] = None, model_router = None):
        self.agent_id = agent_id
        self.persona = persona
        self.memory = VectorMemory()
        self.state = AgentState()  # 持久化状态
        self.layered_memory_enabled = env_flag("CIVITAS_LAYERED_MEMORY_V1", False)
        self.layered_memory = LayeredMemory(
            agent_id=self.agent_id,
            seed=self._derive_layered_memory_seed(),
            enabled=self.layered_memory_enabled,
            institution_type=self._infer_institution_type(),
            config={"persona": self.persona},
        )
        self._api_healthy = False
        self._last_error = None
        
        # 智能体重要性等级（用于混合调度）
        # 0=普通, 1=重要（资金大/粉丝多）, 2=核心（意见领袖）
        self.importance_level = 0
        
        # 多模型路由器（可选，由外部注入）
        self.model_router = model_router
        
        # 初始化思维链历史
        if agent_id not in DeepSeekBrain.thought_history:
            DeepSeekBrain.thought_history[agent_id] = []
            
        self.model_priority = None
        
        self._api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        if self.model_router is None:
            from core.model_router import ModelRouter
            self.model_router = ModelRouter(
                deepseek_key=self._api_key,
                zhipu_key=GLOBAL_CONFIG.ZHIPU_API_KEY
            )
        self._api_healthy = True
        self.risk_committee = RiskCommittee()
        self.policy_committee = PolicyCommittee()

    def _derive_layered_memory_seed(self) -> int:
        payload = {
            "agent_id": self.agent_id,
            "persona": self.persona,
        }
        digest = stable_json_hash(payload)
        return int(digest[:8], 16)

    def _infer_institution_type(self) -> str:
        persona = self.persona if isinstance(self.persona, dict) else {}
        candidates = [
            str(persona.get("institution_type", "")),
            str(persona.get("role", "")),
            str(persona.get("agent_class", "")),
            str(self.agent_id),
        ]
        label = " ".join(candidates).lower()
        mapping = {
            "state_stabilization_fund": ["state", "stabil", "national team", "national_team", "证金", "平准"],
            "pension_fund": ["pension", "养老"],
            "mutual_fund": ["mutual", "公募", "fund_manager", "asset_manager"],
            "insurer": ["insurer", "insurance", "保险"],
            "market_maker": ["maker", "做市"],
            "prop_desk": ["prop", "desk", "自营"],
            "etf_arbitrageur": ["etf", "arb", "arbitrage"],
            "rumor_trader": ["rumor", "谣言"],
            "retail_day_trader": ["day_trader", "daytrader", "intraday"],
            "retail_swing": ["retail", "swing", "个人", "散户"],
        }
        for institution, tokens in mapping.items():
            if any(token in label for token in tokens):
                return institution
        return "retail_swing"

    def _apply_behavior_layer(
        self,
        *,
        decision: Dict[str, Any],
        market_state: Dict[str, Any],
        account_state: Dict[str, Any],
        emotional_state: str,
        social_signal: str,
        snapshot_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not hasattr(self, "layered_memory") or self.layered_memory is None:
            return {"decision": dict(decision), "behavior_card": {}, "behavior_context": {"enabled": False}}
        return self.layered_memory.apply_decision_overlay(
            decision,
            market_state=market_state,
            account_state=account_state,
            emotional_state=emotional_state,
            social_signal=social_signal,
            snapshot_info=snapshot_info,
        )
    
    def _generate_cache_key(self, market_state: Dict, account_state: Dict) -> str:
        """
        生成缓存键：基于市场核心状态的哈希
        只使用影响决策的关键字段，忽略细微变化
        """
        # 将价格离散化到0.5%区间
        price = market_state.get("price", 3000)
        price_bucket = round(price / (price * 0.005)) * (price * 0.005)
        
        # 核心状态拼接
        key_parts = [
            f"trend:{market_state.get('trend', 'N/A')}",
            f"panic:{round(market_state.get('panic_level', 0) * 10)}",  # 10%精度
            f"price_bucket:{int(price_bucket)}",
            f"persona:{self.persona.get('risk_preference', 'N/A')}",
            f"pnl_sign:{1 if account_state.get('pnl_pct', 0) > 0 else -1}"
        ]
        return "|".join(key_parts)
    
    def _check_cache(self, cache_key: str, current_step: int) -> Optional[Dict]:
        """检查缓存是否命中且未过期"""
        if cache_key in DeepSeekBrain.decision_cache:
            cached = DeepSeekBrain.decision_cache[cache_key]
            if current_step - cached.get("step", 0) <= self.CACHE_TTL_STEPS:
                return cached.get("decision")
        return None
    
    def _update_cache(self, cache_key: str, decision: Dict, current_step: int):
        """更新缓存"""
        # 如果 键 已存在，移动到末尾 (最近使用)
        if cache_key in DeepSeekBrain.decision_cache:
            DeepSeekBrain.decision_cache.move_to_end(cache_key)
        
        DeepSeekBrain.decision_cache[cache_key] = {
            "decision": decision,
            "step": current_step
        }
        
        # 缓存大小控制
        if len(DeepSeekBrain.decision_cache) > self.CACHE_MAX_SIZE:
            # 移除最老的条目
            DeepSeekBrain.decision_cache.popitem(last=False)
    
    def _init_client(self):
        """初始化API客户端"""
        try:
            self.client = OpenAI(
                api_key=self._api_key,
                base_url=GLOBAL_CONFIG.API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
            )
            self._api_healthy = True
        except Exception as e:
            self._last_error = str(e)
            self._api_healthy = False
            print(f"[Brain Init Error] Agent {self.agent_id}: {e}")
    
    def set_model_router(self, router):
        """设置多模型路由器"""
        self.model_router = router
        self._api_healthy = True
    
    def health_check(self) -> bool:
        """
        API 健康检查
        
        Returns:
            bool: API是否可用
        """
        if not self.model_router:
            return False
        self._api_healthy = True
        return True
    
    @property
    def is_healthy(self) -> bool:
        return self._api_healthy
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _build_system_prompt(self) -> str:
        """
        构建基于前景理论 (Prospect Theory) 的系统提示词
        区分机构 (Institution) 和散户 (Retail) 的 CoT 深度
        """
        agent_type = self.persona.get("agent_type", "retail")

        beliefs_block = ""
        beliefs_path = os.path.join("data", "beliefs", f"beliefs_{self.agent_id}.json")
        if os.path.exists(beliefs_path):
            try:
                with open(beliefs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    beliefs = data.get("beliefs", [])
                if beliefs:
                    beliefs_text = "\n".join([f"- {b}" for b in beliefs])
                    beliefs_block = f"\n【交易信仰(长期强化)】\n{beliefs_text}\n"
            except Exception:
                beliefs_block = ""
        
        if agent_type == "institution":
            # ===== 机构投资者 提示词: 强调逻辑、理性、结构化 CoT =====
            return f"""你现在是A股市场中的一名【机构投资者】，ID为 {self.agent_id}。

【角色设定】
1. 决策风格: 理性、客观、数据驱动。你使用 DeepSeek R1 级别的深度推理。
2. 风险偏好: {self.persona.get('risk_preference', '稳健')}。注重回撤控制和夏普比率。
3. 目标: 战胜基准指数，寻找阿尔法收益。

【思考格式 - Chain-of-Thought (CoT)】
请务必严格按照以下 XML 结构进行思考：
<reasoning>
1. 宏观分析: [分析政策、新闻对市场的影响]
2. 资金面分析: [分析成交量、流动性、市场情绪]
3. 估值与技术: [分析当前价格是否合理，技术形态如何]
4. 风险对冲策略: [分析潜在风险点及对策]
5. 决策推导: [综合以上因素，得出最终操作建议]
</reasoning>

【最终输出】
在思考结束后，输出严格的 JSON 格式决策：
{{
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "000001",
    "price": 0.0,
    "qty": 0,
    "confidence": 0.0
}}
""" + beliefs_block
        
        # ===== 散户投资者 提示词: 强调情绪、前景理论、非理性 =====
        loss_aversion = self.persona.get('loss_aversion', 2.25)
        risk_preference = self.state.evolved_risk_preference or self.persona.get('risk_preference', '保守')
        
        confidence_desc = ""
        if self.state.confidence < 30:
            confidence_desc = "你目前信心极度低迷，对市场充满恐惧，甚至恐慌。"
        elif self.state.confidence < 50:
            confidence_desc = "你目前信心不足，倾向于观望为主，容易受惊吓。"
        elif self.state.confidence > 80:
            confidence_desc = "你目前信心爆棚，甚至有点过度自信，愿意激进下注。"
            
        return f"""你现在是A股市场中的一名【个人投资者(散户)】，ID为 {self.agent_id}。

【人格设定】
1. 核心心理：你深受"前景理论"影响。你是一个**非理性**的人。
   - **损失厌恶**: 你对损失感到极度痛苦（痛苦程度是盈利快乐的 {loss_aversion} 倍）。
   - 亏损时: 你倾向于冒险（死扛、补仓），试图回本，不愿承认失败。
   - 盈利时: 你倾向于保守（落袋为安），害怕煮熟的鸭子飞了。
2. 投资风格: {risk_preference}。
3. 当前心态: {confidence_desc}

【决策输入】
你将看到当前的"心理效用值(Psychological Value)"。
- 如果值为负且很大(如 -2.0): 代表你极度痛苦，处于心理崩溃边缘。
- 如果值为正(如 1.0): 代表你比较快乐，但小心过度自信。

【输出要求】
请先进行一段内心独白(模拟散户的真实心理活动)，然后输出 JSON 决策。

JSON 格式示例：
{{
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "000001",
    "price": 10.5,
    "qty": 100,
    "confidence": 0.8
}}
""" + beliefs_block


    def _analyze_emotion(self, reasoning: str, decision: Dict) -> float:
        """
        分析思维链中的情绪倾向
        
        通过关键词匹配和决策行为推断情绪分数
        
        Returns:
            float: -1.0(极度恐惧) ~ 1.0(极度贪婪)
        """

        fear_keywords = ['恐慌', '担心', '风险', '亏损', '下跌', '危险', '割肉', '止损', '逃离', '害怕']
        # 贪婪关键词
        greed_keywords = ['机会', '抄底', '上涨', '盈利', '加仓', '牛市', '暴涨', '翻倍', '贪婪', '冲']
        
        fear_count = sum(1 for kw in fear_keywords if kw in reasoning)
        greed_count = sum(1 for kw in greed_keywords if kw in reasoning)
        
        # 基于关键词的基础分数
        keyword_score = (greed_count - fear_count) / max(greed_count + fear_count, 1)
        
        # 基于决策的修正
        action = decision.get('action', 'HOLD')
        action_modifier = {'BUY': 0.3, 'SELL': -0.3, 'HOLD': 0}.get(action, 0)
        
        # 综合分数
        emotion_score = 0.6 * keyword_score + 0.4 * action_modifier
        return max(-1.0, min(1.0, emotion_score))

    def think(self, market_state: Dict, account_state: Dict) -> Dict:
        """
        执行思考过程
        
        Returns:
            Dict: 包含 'decision' (JSON)、'reasoning' (Str) 和 'emotion_score' (Float)
        """
        # 1. 检索记忆
        context_query = f"当前行情:{market_state['trend']}, 盈亏:{account_state['pnl_pct']:.2%}"
        past_lessons = self.memory.retrieve(context_query)
        lessons_text = "\n".join([f"- {lesson}" for lesson in past_lessons]) if past_lessons else "无相关记忆。"

        # 2. 构建用户提示词
        # 获取政策分析 (如果有)
        policy_desc = market_state.get('policy_description', '无')
        policy_reasoning_raw = market_state.get('policy_reasoning', '')
        policy_summary = ""
        if policy_reasoning_raw:
            # 截取推理链的前500字符作为摘要
            policy_summary = policy_reasoning_raw[:500] + "..." if len(policy_reasoning_raw) > 500 else policy_reasoning_raw
        
        user_prompt = f"""
        【市场环境】
        - 价格: {market_state['price']:.2f}
        - 趋势: {market_state['trend']}
        - 恐慌指数: {market_state['panic_level']:.2f}
        - 最新消息: {market_state['news']}
        
        【当前政策】
        - 政策描述: {policy_desc}
        - 分析摘要: {policy_summary if policy_summary else '暂无政策分析'}
        
        【账户状态】
        - 可用资金: {account_state['cash']:.2f}
        - 持仓市值: {account_state['market_value']:.2f}
        - 当前浮动盈亏: {account_state['pnl_pct']:.2%} (注意：你对这个数字非常敏感！)
        
        【闪回记忆】
        {lessons_text}
        
        请基于你的人设和当前政策环境做出交易决策。
        """

        # 3. 调用多模型路由器的同步 回退 处理机制 (内置重试与降级，绝不崩溃)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        
        content_raw, reasoning, model_used = self.model_router.sync_call_with_fallback(
            messages,
            priority_models=self.model_priority or ["glm-4-flashx"],
            timeout_budget=15.0
        )
        
        # 判断逻辑异常降级
        decision_json = self._extract_json(content_raw)
        
        # 处理空推理内容
        if not reasoning:
            # 去除 JSON 部分，剩余内容作为思维链
            reasoning = re.sub(r'```json.*?```', '', content_raw, flags=re.DOTALL).strip()
            if not reasoning:
                reasoning = content_raw
        
        # 5. 情绪分析
        emotion_score = self._analyze_emotion(reasoning, decision_json)
        
        # 6. 记录思维链历史（用于fMRI可视化）
        thought_record = ThoughtRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            reasoning_content=reasoning,
            emotion_score=emotion_score,
            decision=decision_json,
            market_context=market_state
        )
        DeepSeekBrain.thought_history[self.agent_id].append(thought_record)
        
        return {
            "decision": decision_json,
            "reasoning": reasoning,
            "raw_content": content_raw,
            "emotion_score": emotion_score
        }
    
    def _fallback_decision(self, error_msg: str) -> Dict:
        """API不可用时的兜底决策"""
        return {
            "decision": {"action": "HOLD", "qty": 0},
            "reasoning": f"大脑死机，系统异常: {error_msg}",
            "raw_content": "",
            "emotion_score": 0.0
        }

    def _extract_json(self, text: str) -> Dict:
        """
        鲁棒的 JSON 提取器，处理 LLM 可能输出的 Markdown 格式
        """
        try:
            # 尝试直接解析
            return json.loads(text)
        except Exception:
            pass
            
        try:
            # 提取 ```json ... ``` 块
            match = re.search(r"```json(.*?)```", text, re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
                return json.loads(clean_text)
        except Exception:
            pass
            
        # 兜底策略
        return {"action": "HOLD", "qty": 0, "error": "JSON_PARSE_FAIL"}
    
    # --- 新增：异步思考方法（支持多模型路由） ---
    
    def _build_local_analyst_cards(
        self,
        market_state: Dict[str, Any],
        account_state: Dict[str, Any],
        emotional_state: str,
        social_signal: str,
    ) -> List[Dict[str, Any]]:
        """Build deterministic analyst cards used by the manager aggregation layer."""
        price = float(market_state.get("price", market_state.get("last_price", 0.0)) or 0.0)
        panic = float(market_state.get("panic_level", 0.0) or 0.0)
        trend = str(market_state.get("trend", "neutral")).lower()
        pnl_pct = float(account_state.get("pnl_pct", 0.0) or 0.0)
        sentiment = float(market_state.get("text_sentiment_score", 0.0) or 0.0)
        liquidity = float(market_state.get("liquidity_index", 1.0) or 1.0)

        news_action = "buy" if ("up" in trend or "上涨" in trend or sentiment > 0.15) else "sell" if ("down" in trend or "下跌" in trend or sentiment < -0.15) else "hold"
        macro_action = "buy" if liquidity > 1.05 and panic < 0.4 else "reduce_risk" if panic > 0.6 else "hold"
        risk_action = "reduce_risk" if panic > 0.55 or pnl_pct < -0.08 else "hold"

        cards = [
            {
                "analyst_id": f"{self.agent_id}_news_analyst",
                "thesis": f"News/price combined view at price={price:.2f}, trend={trend}",
                "evidence": [
                    {"type": "news", "content": str(market_state.get("news", "no_news")), "weight": 0.7},
                    {"type": "price", "content": f"trend={trend}, sentiment={sentiment:.2f}", "weight": 0.6},
                ],
                "time_horizon": "intraday",
                "risk_tags": ["panic"] if panic > 0.5 else [],
                "confidence": 0.55 + min(0.35, abs(sentiment)),
                "counterarguments": ["headline risk can reverse quickly", "microstructure noise"],
                "recommended_action": news_action,
            },
            {
                "analyst_id": f"{self.agent_id}_macro_analyst",
                "thesis": "Macro and policy transmission analysis",
                "evidence": [
                    {"type": "macro", "content": f"liquidity={liquidity:.2f}", "weight": 0.7},
                    {"type": "macro", "content": f"panic={panic:.2f}", "weight": 0.6},
                ],
                "time_horizon": "weekly",
                "risk_tags": ["liquidity"] if liquidity < 0.9 else [],
                "confidence": 0.50 + 0.30 * max(0.0, min(1.0, liquidity - 0.8)),
                "counterarguments": ["macro impulses can lag", "policy implementation uncertainty"],
                "recommended_action": macro_action,
            },
            {
                "analyst_id": f"{self.agent_id}_risk_analyst",
                "thesis": f"Risk-first view under emotion={emotional_state}, social={social_signal}",
                "evidence": [
                    {"type": "risk", "content": f"pnl_pct={pnl_pct:.3f}", "weight": 0.8},
                    {"type": "social", "content": str(social_signal), "weight": 0.6},
                ],
                "time_horizon": "intraday",
                "risk_tags": ["panic", "overcrowding"] if panic > 0.45 else ["liquidity"] if liquidity < 0.9 else [],
                "confidence": 0.60 + 0.30 * max(0.0, min(1.0, panic)),
                "counterarguments": ["defensive posture can miss rebounds"],
                "recommended_action": risk_action,
            },
        ]
        return [coerce_analyst_card(card, analyst_id=card.get("analyst_id", "analyst")) for card in cards]

    def _decision_from_manager_card(
        self,
        manager_card: Dict[str, Any],
        account_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map manager final card into order-like decision payload."""
        action_map = {"buy": "BUY", "sell": "SELL", "hold": "HOLD", "reduce_risk": "SELL"}
        final_action = str(manager_card.get("recommended_action", "hold")).lower()
        action = action_map.get(final_action, "HOLD")
        price = float(market_state.get("price", market_state.get("last_price", 0.0)) or 0.0)
        if price <= 0:
            return {"action": "HOLD", "qty": 0, "price": 0.0}
        cash = float(account_state.get("cash", 0.0) or 0.0)
        market_value = float(account_state.get("market_value", 0.0) or 0.0)

        if action == "BUY":
            qty = int(max(0.0, (cash * 0.12) / price))
        elif final_action == "reduce_risk":
            qty = int(max(0.0, (market_value * 0.30) / price))
        elif action == "SELL":
            qty = int(max(0.0, (market_value * 0.20) / price))
        else:
            qty = 0
        return {
            "action": action,
            "qty": qty,
            "price": price,
            "confidence": float(manager_card.get("calibrated_confidence", 0.5)),
        }

    async def think_async(
        self,
        market_state: Dict,
        account_state: Dict,
        model_priority: List[str] = None,
        timeout_budget: float = 15.0,
        emotional_state: str = "Neutral",
        social_signal: str = "Neutral"
    ) -> Dict:
        """Structured evidence flow: analyst cards -> manager card -> calibrated decision."""
        analyst_schema = {
            "type": "object",
            "properties": {
                "thesis": {"type": "string", "default": "neutral thesis"},
                "evidence": {"type": "array", "default": []},
                "time_horizon": {"type": "string", "default": "swing"},
                "risk_tags": {"type": "array", "default": []},
                "confidence": {"type": "number", "default": 0.5},
                "counterarguments": {"type": "array", "default": []},
                "recommended_action": {"type": "string", "default": "hold"},
            },
        }
        local_cards = self._build_local_analyst_cards(market_state, account_state, emotional_state, social_signal)
        analyst_cards: List[Dict[str, Any]] = []
        model_used = "local_structured"

        role_prompts = [
            ("news_analyst", "Focus on news and price evidence. Output only strict JSON."),
            ("macro_analyst", "Focus on macro and policy evidence. Output only strict JSON."),
            ("risk_analyst", "Focus on risk and social evidence. Output only strict JSON."),
        ]
        priority = self.model_priority or model_priority or ["glm-4-flashx"]
        try:
            per_role_timeout = float(os.environ.get("CIVITAS_AGENT_LLM_CALL_TIMEOUT_SECONDS", "9.0") or 9.0)
        except (TypeError, ValueError):
            per_role_timeout = 9.0
        per_role_timeout = max(6.0, min(12.0, per_role_timeout))

        if self.model_router and hasattr(self.model_router, "call_with_schema"):
            for idx, (role_name, role_prompt) in enumerate(role_prompts):
                fallback_card = local_cards[idx] if idx < len(local_cards) else local_cards[-1]
                messages = [
                    {"role": "system", "content": role_prompt},
                    {
                        "role": "user",
                        "content": self._build_user_prompt(market_state, account_state, emotional_state, social_signal),
                    },
                ]
                try:
                    parsed_card, sub_model_used, _ = await self.model_router.call_with_schema(
                        messages=messages,
                        json_schema=analyst_schema,
                        priority_models=priority,
                        timeout_budget=per_role_timeout,
                        fallback_obj=fallback_card,
                    )
                    parsed_card["analyst_id"] = f"{self.agent_id}_{role_name}"
                    normalized_card = coerce_analyst_card(parsed_card, analyst_id=parsed_card["analyst_id"])
                    valid, _ = validate_analyst_card(normalized_card)
                    analyst_cards.append(normalized_card if valid else fallback_card)
                    model_used = sub_model_used
                except Exception:
                    analyst_cards.append(fallback_card)
        else:
            analyst_cards = local_cards

        risk_metrics = {
            "cvar": float(market_state.get("risk_cvar", account_state.get("pnl_pct", 0.0) - 0.02)),
            "max_drawdown": float(market_state.get("max_drawdown", account_state.get("pnl_pct", 0.0))),
            "turnover": float(market_state.get("turnover", abs(float(market_state.get("panic_level", 0.0))) * 1.2)),
            "crowding": float(market_state.get("crowding", min(1.0, abs(float(market_state.get("panic_level", 0.0)))))),
            "volatility_spike": float(market_state.get("volatility_spike", 1.0 + abs(float(market_state.get("panic_level", 0.0))) * 2.0)),
        }
        risk_alert = self.risk_committee.assess(risk_metrics).to_dict()

        policy_event = str(market_state.get("policy_description", "") or market_state.get("news", ""))
        policy_update = self.policy_committee.translate(policy_event)

        manager_final_card = aggregate_analyst_cards(analyst_cards, risk_alert=risk_alert)
        decision_json = self._decision_from_manager_card(manager_final_card, account_state, market_state)
        behavior_bundle = self._apply_behavior_layer(
            decision=decision_json,
            market_state=market_state,
            account_state=account_state,
            emotional_state=emotional_state,
            social_signal=social_signal,
            snapshot_info={
                "price": _safe_float(market_state.get("price", market_state.get("last_price", 0.0)), 0.0),
                "trend": market_state.get("trend", "neutral"),
                "panic_level": _safe_float(market_state.get("panic_level", 0.0), 0.0),
                "policy_description": policy_event,
            },
        )
        decision_json = behavior_bundle["decision"]
        behavior_card = behavior_bundle["behavior_card"]

        reasoning = (
            f"manager_action={manager_final_card.get('recommended_action', 'hold')}; "
            f"signal={manager_final_card.get('aggregated_signal', 0.0):.3f}; "
            f"calibrated_conf={manager_final_card.get('calibrated_confidence', 0.0):.3f}; "
            f"risk_level={risk_alert.get('level', 'normal')}; "
            f"behavior_budget={behavior_card.get('current_risk_budget', 0.0):.3f}"
        )
        emotion_score = self._analyze_emotion(reasoning, decision_json)

        decision_trace = {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "market_state": market_state,
            "account_state": account_state,
            "analyst_cards": analyst_cards,
            "manager_final_card": manager_final_card,
            "contradiction_matrix": manager_final_card.get("contradiction_matrix", {}),
            "risk_alert": risk_alert,
            "policy_conditions": policy_update,
            "decision": decision_json,
            "behavior_card": behavior_card,
            "model_used": model_used,
        }
        trace_path = export_decision_trace(decision_trace)

        thought_record = ThoughtRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            reasoning_content=reasoning,
            emotion_score=emotion_score,
            decision=decision_json,
            market_context=market_state,
        )
        DeepSeekBrain.thought_history[self.agent_id].append(thought_record)
        if len(DeepSeekBrain.thought_history[self.agent_id]) > 20:
            DeepSeekBrain.thought_history[self.agent_id].pop(0)

        return {
            "decision": decision_json,
            "reasoning": reasoning,
            "raw_content": json.dumps({"manager_final_card": manager_final_card}, ensure_ascii=False),
            "emotion_score": emotion_score,
            "model_used": model_used,
            "analyst_cards": analyst_cards,
            "contradiction_matrix": manager_final_card.get("contradiction_matrix", {}),
            "manager_final_card": manager_final_card,
            "risk_alerts": risk_alert,
            "policy_conditions": policy_update,
            "calibration": manager_final_card.get("calibration", {}),
            "behavior_card": behavior_card,
            "behavior_context": behavior_bundle.get("behavior_context", {}),
            "decision_trace_path": trace_path,
        }

    def _build_user_prompt(
        self, 
        market_state: Dict, 
        account_state: Dict,
        emotional_state: str = "Neutral",
        social_signal: str = "Neutral"
    ) -> str:
        """构建用户提示词 (增强版)"""
        # 检索记忆
        context_query = f"当前行情:{market_state.get('trend', '未知')}, 盈亏:{account_state.get('pnl_pct', 0):.2%}"
        past_lessons = self.memory.retrieve(context_query)
        lessons_text = "\n".join([f"- {lesson}" for lesson in past_lessons]) if past_lessons else "无相关记忆。"
        
        policy_desc = market_state.get('policy_description', '无')
        policy_reasoning_raw = market_state.get('policy_reasoning', '')
        policy_summary = ""
        if policy_reasoning_raw:
            policy_summary = policy_reasoning_raw[:500] + "..." if len(policy_reasoning_raw) > 500 else policy_reasoning_raw
        
        # 监管反馈注入
        rejection_reason = market_state.get('last_rejection_reason')
        regulatory_block = ""
        if rejection_reason:
            regulatory_block = f"""
        【监管警告】
        你的上一个订单被风控系统拦截了！
        原因: {rejection_reason}
        请反思你的行为。如果是"高频(OTR High)"，请降低下单频率；如果是"价格异常"，请检查你的挂单价格是否合理。
        """

        graph_context = market_state.get('graph_context', '')
        
        return f"""
        【市场环境】
        - 价格: {market_state.get('price', 0):.2f}
        - 趋势: {market_state.get('trend', '未知')}
        - 恐慌指数: {market_state.get('panic_level', 0):.2f}
        - 最新消息: {market_state.get('news', '无')}
        - text_topic: {market_state.get('text_dominant_topic', 'uncategorized')}
        - text_sentiment: {market_state.get('text_sentiment_score', 0):.2f}
        - text_panic_greed: {market_state.get('text_panic_score', 0):.2f}/{market_state.get('text_greed_score', 0):.2f}
        - text_policy_shock: {market_state.get('text_policy_shock', 0):.2f} ({market_state.get('text_regime_bias', 'neutral')})
        {regulatory_block}
        
        【当前政策】
        - 政策描述: {policy_desc}
        - 分析摘要: {policy_summary if policy_summary else '暂无政策分析'}
        
        【个人状态】
        - 情绪状态: {emotional_state} (这会显著影响你的风险偏好!)
        - 社交信号: {social_signal} (你的朋友圈正在做什么?)
        - 心理效用值: {market_state.get('_psychological_value', 0):.3f} (前景理论计算结果，负=痛苦/正=快乐)
        - 损失厌恶系数λ: {market_state.get('_risk_aversion', 2.25)} (越高你越怕亏钱)
        - 可用资金: {account_state.get('cash', 0):.2f}
        - 持仓市值: {account_state.get('market_value', 0):.2f}
        - 当前浮动盈亏: {account_state.get('pnl_pct', 0):.2%}
        
        【私有认知图谱及宏观共识】
        {graph_context if graph_context else "尚未形成有效图谱记忆。"}
        
        【闪回记忆(Vector)】
        {lessons_text}
        
        请作为一名真实的投资者，基于你的人设、当前情绪和社交压力，做出交易决策。
        """
    
    # --- 新增：快速思考方法（本地规则引擎） ---
    
    def think_fast(self, market_state: Dict, account_state: Dict) -> Dict:
        """
        快速思考 - 使用本地规则引擎，无API调用
        
        基于简单启发式规则和人格设定生成决策。
        适用于快速模式下减少API调用。
        
        Returns:
            Dict: 包含 'decision'、'reasoning'、'emotion_score'
        """
        price = market_state.get('price', 3000)
        trend = market_state.get('trend', '震荡')
        panic_level = market_state.get('panic_level', 0.5)
        text_sentiment = market_state.get('text_sentiment_score', 0.0)
        text_policy_shock = market_state.get('text_policy_shock', 0.0)
        text_regime = market_state.get('text_regime_bias', 'neutral')
        cash = account_state.get('cash', 100000)
        market_value = account_state.get('market_value', 0)
        pnl_pct = account_state.get('pnl_pct', 0)
        
        # 提取人格参数
        risk_pref = self.state.evolved_risk_preference or self.persona.get('risk_preference', '稳健')
        loss_aversion = self.persona.get('loss_aversion', 2.25)
        confidence = self.state.confidence
        
        # 基于规则的决策逻辑
        action = "HOLD"
        qty = 0
        reasoning_parts = []
        
        # 规则1: 趋势跟踪
        if trend == "上涨":
            trend_signal = 0.3
            reasoning_parts.append("市场上涨趋势明显")
        elif trend == "下跌":
            trend_signal = -0.3
            reasoning_parts.append("市场处于下跌趋势")
        else:
            trend_signal = 0.0
            reasoning_parts.append("市场震荡整理")
        
        # 规则2: 恐慌指数
        if panic_level > 0.7:
            panic_signal = -0.4
            reasoning_parts.append(f"恐慌指数高达{panic_level:.2f}，市场情绪极度悲观")
        elif panic_level < 0.3:
            panic_signal = 0.2
            reasoning_parts.append("市场情绪较为乐观")
        else:
            panic_signal = 0.0


        if text_regime == "risk_off":
            text_signal = -0.2 - (0.2 * min(1.0, abs(text_policy_shock)))
            reasoning_parts.append(
                f"NLP regime={text_regime}, sentiment={text_sentiment:.2f}, shock={text_policy_shock:.2f}"
            )
        elif text_regime == "risk_on":
            text_signal = 0.1 + (0.2 * min(1.0, text_policy_shock))
            reasoning_parts.append(
                f"NLP regime={text_regime}, sentiment={text_sentiment:.2f}, shock={text_policy_shock:.2f}"
            )
        else:
            text_signal = 0.15 * float(text_sentiment)
        
        # 规则3: 盈亏反应（前景理论）
        if pnl_pct < -0.05:
            # 亏损时倾向冒险（死扛）
            pnl_signal = 0.1 if risk_pref == "激进" else -0.1
            reasoning_parts.append(f"当前亏损{pnl_pct:.2%}，{'选择继续持有等待回本' if pnl_signal > 0 else '考虑止损'}")
        elif pnl_pct > 0.05:
            # 盈利时倾向保守（落袋为安）
            pnl_signal = -0.2 * loss_aversion / 2.25
            reasoning_parts.append(f"当前盈利{pnl_pct:.2%}，考虑部分获利了结")
        else:
            pnl_signal = 0.0
        
        # 规则4: 信心影响
        confidence_factor = confidence / 100.0
        reasoning_parts.append(f"当前信心指数: {confidence:.0f}")
        
        # 综合信号
        total_signal = (trend_signal + panic_signal + text_signal + pnl_signal) * confidence_factor
        
        # 决策阈值
        if total_signal > 0.3:
            action = "BUY"
            qty = int(cash * 0.2 / price)  # 20%仓位
            reasoning_parts.append(f"综合信号偏多({total_signal:.2f})，决定买入")
        elif total_signal < -0.3:
            action = "SELL"
            qty = int(market_value * 0.3 / price) if price > 0 else 0  # 30%持仓
            reasoning_parts.append(f"综合信号偏空({total_signal:.2f})，决定卖出")
        else:
            action = "HOLD"
            reasoning_parts.append(f"信号不明确({total_signal:.2f})，选择观望")
        
        # 情绪分数
        emotion_score = np.clip(total_signal, -1.0, 1.0)
        
        # 组装决策
        decision = {
            "action": action,
            "ticker": "000001",
            "price": float(price),
            "qty": max(0, qty),
            "confidence": abs(total_signal)
        }

        behavior_bundle = self._apply_behavior_layer(
            decision=decision,
            market_state={
                "price": float(price),
                "last_price": float(price),
                "trend": trend,
                "market_trend": trend,
                "panic_level": float(panic_level),
                "policy_description": str(text_regime),
                "policy_news": str(text_regime),
                "text_sentiment_score": float(text_sentiment),
                "text_policy_shock": float(text_policy_shock),
                "text_regime_bias": str(text_regime),
                "news_source": "local_rules",
            },
            account_state={
                "cash": float(cash),
                "market_value": float(market_value),
                "pnl_pct": float(pnl_pct),
            },
            emotional_state=str(emotional_state),
            social_signal=str(social_signal),
            snapshot_info={
                "price": float(price),
                "trend": trend,
                "panic_level": float(panic_level),
                "pnl_pct": float(pnl_pct),
            },
        )
        decision = behavior_bundle.get("decision", decision)
        behavior_card = behavior_bundle.get("behavior_card", {})

        reasoning = "【快速决策模式】\n" + "\n".join([f"• {p}" for p in reasoning_parts])
        reasoning += f"\n• behavior_budget={behavior_card.get('current_risk_budget', 0.0):.3f}"

        # 记录思维链历史
        thought_record = ThoughtRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            reasoning_content=reasoning,
            emotion_score=emotion_score,
            decision=decision,
            market_context=market_state
        )
        DeepSeekBrain.thought_history[self.agent_id].append(thought_record)
        if len(DeepSeekBrain.thought_history[self.agent_id]) > 20:
            DeepSeekBrain.thought_history[self.agent_id].pop(0)

        return {
            "decision": decision,
            "reasoning": reasoning,
            "raw_content": "",
            "emotion_score": emotion_score,
            "model_used": "local_rules",
            "behavior_card": behavior_card,
            "behavior_context": behavior_bundle.get("behavior_context", {}),
        }
