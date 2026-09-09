import json
from typing import Any, Dict, List


def get_diagnostic_tools() -> List[Dict[str, Any]]:
    """返回给大模型的探针工具定义。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_agent_state",
                "description": "获取指定智能体的资金、持仓、盈亏与风险偏好",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "智能体 ID"},
                    },
                    "required": ["agent_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_agent_thought_history",
                "description": "获取指定智能体最近思维链记录",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "智能体 ID"},
                        "limit": {"type": "integer", "description": "条数上限", "default": 3},
                    },
                    "required": ["agent_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_agent_memory",
                "description": "查询指定智能体图谱记忆",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "智能体 ID"},
                        "topic": {"type": "string", "description": "主题关键词"},
                    },
                    "required": ["agent_id", "topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_agent_memory",
                "description": "query_agent_memory 的别名",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "智能体 ID"},
                        "topic": {"type": "string", "description": "主题关键词"},
                    },
                    "required": ["agent_id", "topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_lob_log",
                "description": "查询撮合引擎成交日志",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "明细条数上限", "default": 50},
                        "detail": {"type": "boolean", "description": "是否返回明细", "default": False},
                        "agent_id": {"type": "string", "description": "按智能体过滤"},
                        "current_step": {"type": "boolean", "description": "仅查询当前步缓冲", "default": False},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_interview",
                "description": "对指定智能体发起访谈并返回合成回答",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "受访智能体 ID"},
                        "question": {"type": "string", "description": "访谈问题"},
                        "topic": {"type": "string", "description": "可选主题关键词"},
                    },
                    "required": ["agent_id", "question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_snapshot",
                "description": "获取当前市场快照",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


class ProbeExecutor:
    def __init__(self, agents_map: Dict[str, Any], market_env: Any):
        self.agents = agents_map
        self.market_env = market_env

    def _extract_risk_preference(self, agent: Any) -> str:
        """统一提取风险偏好，兼容 dict persona 与对象 persona。"""
        persona = getattr(agent, "persona", None)
        if isinstance(persona, dict):
            return str(persona.get("risk_preference", "未知"))
        if persona is None:
            return "未知"
        if hasattr(persona, "risk_preference"):
            return str(getattr(persona, "risk_preference"))
        if hasattr(persona, "risk_appetite"):
            risk_appetite = getattr(persona, "risk_appetite")
            return str(getattr(risk_appetite, "value", risk_appetite))
        return "未知"

    def execute_tool(self, name: str, kwargs: Dict[str, Any]) -> str:
        """按工具名路由到具体实现。"""
        try:
            if name == "get_agent_state":
                return self.get_agent_state(**kwargs)
            if name == "get_agent_thought_history":
                return self.get_agent_thought_history(**kwargs)
            if name in {"query_agent_memory", "search_agent_memory"}:
                return self.query_agent_memory(**kwargs)
            if name == "query_lob_log":
                return self.query_lob_log(**kwargs)
            if name == "send_interview":
                return self.send_interview(**kwargs)
            if name == "get_market_snapshot":
                return self.get_market_snapshot()
            return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"Error executing {name}: {exc}"}, ensure_ascii=False)

    def get_agent_state(self, agent_id: str) -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."}, ensure_ascii=False)

        state_info = {
            "cash": getattr(agent, "cash_balance", 0.0),
            "portfolio": getattr(agent, "portfolio", {}),
            "total_pnl": getattr(agent, "total_pnl", 0.0) if hasattr(agent, "total_pnl") else 0.0,
            "risk_preference": self._extract_risk_preference(agent),
        }

        if hasattr(agent, "brain") and hasattr(agent.brain, "state"):
            state_info["confidence"] = getattr(agent.brain.state, "confidence", 50.0)
            evolved = getattr(agent.brain.state, "evolved_risk_preference", None)
            if evolved:
                state_info["risk_preference"] = evolved

        return json.dumps(state_info, ensure_ascii=False)

    def get_agent_thought_history(self, agent_id: str, limit: int = 3) -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."}, ensure_ascii=False)
        if not hasattr(agent, "brain"):
            return json.dumps({"error": f"Agent {agent_id} has no brain component."}, ensure_ascii=False)

        from agents.brain import DeepSeekBrain

        history = DeepSeekBrain.thought_history.get(agent_id, [])
        if not history:
            return json.dumps({"message": "No thought history available."}, ensure_ascii=False)

        records = []
        for item in history[-max(1, int(limit)):]:
            records.append(
                {
                    "timestamp": item.timestamp,
                    "emotion_score": item.emotion_score,
                    "reasoning": item.reasoning_content,
                    "decision": item.decision,
                }
            )
        return json.dumps(records, ensure_ascii=False)

    def query_agent_memory(self, agent_id: str, topic: str) -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."}, ensure_ascii=False)
        if not hasattr(agent, "graph_memory"):
            return json.dumps({"error": f"Agent {agent_id} doesn't have GraphMemoryBank enabled."}, ensure_ascii=False)

        subgraph = agent.graph_memory.retrieve_subgraph([topic], depth=2)
        if not subgraph:
            return json.dumps({"message": f"No graph memory found for topic '{topic}'."}, ensure_ascii=False)
        return json.dumps({"graph_subgraph": subgraph}, ensure_ascii=False)

    def query_lob_log(
        self,
        limit: int = 50,
        detail: bool = False,
        agent_id: str = "",
        current_step: bool = False,
    ) -> str:
        if not self.market_env or not hasattr(self.market_env, "engine"):
            return json.dumps({"error": "Market environment or engine not bound."}, ensure_ascii=False)

        engine = getattr(self.market_env, "engine", None)
        if engine is None:
            return json.dumps({"error": "Matching engine not available in market_env."}, ensure_ascii=False)

        trades = engine.step_trades_buffer if current_step else engine.trades_history
        trades = trades or []

        if agent_id:
            trades = [
                t
                for t in trades
                if agent_id
                in {
                    getattr(t, "buyer_agent_id", ""),
                    getattr(t, "seller_agent_id", ""),
                    getattr(t, "maker_agent_id", ""),
                    getattr(t, "taker_agent_id", ""),
                }
            ]

        total_qty = sum(getattr(t, "quantity", 0) for t in trades)
        total_notional = sum(getattr(t, "price", 0.0) * getattr(t, "quantity", 0) for t in trades)
        timestamps = [getattr(t, "timestamp", 0) for t in trades]

        buyer_stats: Dict[str, int] = {}
        seller_stats: Dict[str, int] = {}
        for trade in trades:
            buyer = getattr(trade, "buyer_agent_id", "")
            seller = getattr(trade, "seller_agent_id", "")
            qty = int(getattr(trade, "quantity", 0) or 0)
            if buyer:
                buyer_stats[buyer] = buyer_stats.get(buyer, 0) + qty
            if seller:
                seller_stats[seller] = seller_stats.get(seller, 0) + qty

        result: Dict[str, Any] = {
            "summary": {
                "trade_count": len(trades),
                "total_qty": total_qty,
                "total_notional": total_notional,
                "time_range": [min(timestamps), max(timestamps)] if timestamps else None,
                "top_buyers": sorted(buyer_stats.items(), key=lambda x: x[1], reverse=True)[:5],
                "top_sellers": sorted(seller_stats.items(), key=lambda x: x[1], reverse=True)[:5],
            },
            "source": "step_buffer" if current_step else "trades_history",
            "filtered_agent": agent_id or None,
        }

        if detail:
            recent = trades[-max(1, int(limit)) :] if limit else trades
            result["detail"] = [
                {
                    "trade_id": getattr(t, "trade_id", ""),
                    "price": getattr(t, "price", 0.0),
                    "quantity": getattr(t, "quantity", 0),
                    "buyer_agent_id": getattr(t, "buyer_agent_id", ""),
                    "seller_agent_id": getattr(t, "seller_agent_id", ""),
                    "maker_agent_id": getattr(t, "maker_agent_id", ""),
                    "taker_agent_id": getattr(t, "taker_agent_id", ""),
                    "timestamp": getattr(t, "timestamp", 0),
                }
                for t in recent
            ]

        return json.dumps(result, ensure_ascii=False)

    def send_interview(self, agent_id: str, question: str, topic: str = "") -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."}, ensure_ascii=False)

        from agents.brain import DeepSeekBrain

        risk_pref = self._extract_risk_preference(agent)
        history = DeepSeekBrain.thought_history.get(agent_id, [])
        recent = history[-2:] if history else []
        thought_summaries = [
            {
                "timestamp": r.timestamp,
                "emotion_score": r.emotion_score,
                "reasoning": r.reasoning_content,
                "decision": r.decision,
            }
            for r in recent
        ]

        memory_snippet = None
        if topic and hasattr(agent, "graph_memory"):
            try:
                memory_snippet = agent.graph_memory.retrieve_subgraph([topic], depth=2)
            except Exception:
                memory_snippet = None

        response = (
            f"我倾向于 {risk_pref} 风格。"
            f"针对“{question}”，我的判断基于近期思维记录与图谱记忆。"
        )
        if thought_summaries:
            response += "近期思维显示我更关注波动与仓位风险。"
        if memory_snippet:
            response += f"图谱记忆提示“{topic}”相关因果链路需要优先关注。"

        payload = {
            "interviewee": agent_id,
            "question": question,
            "persona": {"risk_preference": risk_pref},
            "recent_thoughts": thought_summaries,
            "memory_snippet": memory_snippet,
            "response": response,
        }
        return json.dumps(payload, ensure_ascii=False)

    def get_market_snapshot(self) -> str:
        if not self.market_env:
            return json.dumps({"error": "Market environment not bound to ProbeExecutor."}, ensure_ascii=False)

        try:
            state = {
                "price": getattr(self.market_env, "last_price", getattr(self.market_env, "current_price", 0.0)),
                "trend": getattr(self.market_env, "trend", "未知"),
                "panic_level": getattr(self.market_env, "panic_level", 0.0),
                "news_count": len(getattr(self.market_env, "recent_news", [])),
                "timestamp": getattr(self.market_env, "current_time", getattr(self.market_env, "timestamp", 0)),
            }
            return json.dumps(state, ensure_ascii=False)
        except Exception:
            return json.dumps({"error": "Could not extract standard market state."}, ensure_ascii=False)
