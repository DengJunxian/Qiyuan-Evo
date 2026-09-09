import json
import logging
from typing import Dict, Any, List, Optional

from agents.diagnostic.tools import get_diagnostic_tools, ProbeExecutor
from config import GLOBAL_CONFIG
from core.llm_client import LLMClient, safe_json_loads, strip_nonstandard_tags

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    自治复盘诊断 Agent（ReAct 风格）。
    输出 Thought/Action/Observation/Final 结构，Thought 为简短摘要。
    """

    @staticmethod
    def synthesize_evidence(task, artifacts, critiques):
        """Build a report from immutable tool evidence; never generate market facts."""
        from copy import deepcopy
        labels = {"policy": "政策传导", "simulation": "市场实验", "risk": "风险测量",
                  "control": "同种子对照", "reproducibility": "独立复现",
                  "historical_comparison": "历史偏差", "quant": "量化指标",
                  "counterfactual_search": "干预方案比较"}
        conclusions = []
        for name, item in sorted(artifacts.items()):
            data = item["data"]
            summary = labels.get(name, name)
            if name == "policy":
                summary += f"：{data['event']['policy_label']}；方向 {data['event']['direction']}"
            elif name == "simulation":
                summary += f"：累计收益 {data['cumulative_return']:.4%}，来源 {data['engine']}"
            elif name == "risk":
                summary += f"：最大回撤 {data['max_drawdown']:.4%}，CVaR {data['cvar']:.4%}"
            elif name == "historical_comparison":
                summary += f"：归一化 RMSE {data['normalized_rmse']:.6f}，方向一致率 {data['direction_consistency']:.2%}"
            elif name == "counterfactual_search":
                summary += f"：预算内推荐 {data['recommended']}，相对不干预回撤降低 {data['risk_reduction']:.4%}"
            conclusions.append({"conclusion": summary, "evidence": [name],
                                "confidence": item["confidence"], "caveat": item["caveat"],
                                "source": item["source"], "facts": deepcopy(data)})
        return {"schema_version": "qiyuan_report_v1", "task_id": task.task_id,
                "objective": task.objective, "conclusions": conclusions,
                "unresolved_issues": deepcopy(critiques),
                "decision_summary": "结论从工具证据生成；所有未满足项保留在审查结果中。"}
    def __init__(
        self,
        agents_map: Dict[str, Any],
        market_env: Any = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "deepseek-chat"
    ):
        self.api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        self.base_url = base_url or GLOBAL_CONFIG.API_BASE_URL
        self.model_name = model_name

        self.client = LLMClient(api_key=self.api_key, base_url=self.base_url)
        self.executor = ProbeExecutor(agents_map, market_env)

        self.system_prompt = (
            "你是一位资深的‘金融风控与市场回溯专家’。你的职责是深入沙箱微观层，利用 ReAct 架构拆解市场异动。"
            "分析准则：\n"
            "1. 优先级：先查 LOB 成交日志定位嫌疑成交，再查相关智能体资金状态，最后通过‘微观访谈 (send_interview)’下探行为心理。\n"
            "2. 深度：不仅要说明‘发生了什么’，更要深挖‘为什么’（例如：Agent 是否因为图谱记忆中的利空共识而恐慌抛售）。\n"
            "3. 纪律：所有的结论必须基于工具返回的 Observation 证据。严禁幻觉。\n"
            "输出格式：Thought（每步分析摘要）、Action、Action Input、Observation、Final Result。"
        )

        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _sanitize_history(self) -> List[Dict[str, Any]]:
        clean_history = []
        for msg in self.history:
            clean_msg = {"role": msg["role"]}
            if "content" in msg:
                clean_msg["content"] = msg["content"]
            if "tool_calls" in msg:
                clean_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                clean_msg["name"] = msg["name"]
            clean_history.append(clean_msg)
        return clean_history

    def _summarize_thought(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """生成短版 Thought，避免暴露完整推理链。"""
        if tool_name == "query_lob_log":
            return "需要先查成交日志以确定关键交易与参与者。"
        if tool_name in ("query_agent_memory", "search_agent_memory"):
            return "需要检索该智能体的图谱记忆以确认认知关联。"
        if tool_name == "send_interview":
            return "需要向关键个体发起质询以获取微观原因。"
        if tool_name == "get_agent_thought_history":
            return "需要查看近期思维链路以定位决策动因。"
        if tool_name == "get_agent_state":
            return "需要核对智能体当前状态以校验风险暴露。"
        if tool_name == "get_market_snapshot":
            return "需要获取市场快照以确认宏观压力。"
        return f"需要调用 {tool_name} 获取证据。"

    async def chat(self, user_message: str) -> str:
        """接收用户文本输入，执行 ReAct 工具调用并返回可读结果"""
        self.history.append({"role": "user", "content": user_message})
        tools = get_diagnostic_tools()

        # 防止无限循环
        max_tool_iterations = 10
        trace_lines: List[str] = []

        for _ in range(max_tool_iterations):
            try:
                response = await self.client.chat(
                    model=self.model_name,
                    messages=self._sanitize_history(),
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3
                )

                message = response.choices[0].message

                if message.tool_calls:
                    tool_calls_record = []
                    for t in message.tool_calls:
                        tool_calls_record.append({
                            "id": t.id,
                            "type": t.type,
                            "function": {
                                "name": t.function.name,
                                "arguments": t.function.arguments
                            }
                        })
                    self.history.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": tool_calls_record
                    })

                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            func_args = safe_json_loads(tool_call.function.arguments)
                        except Exception:
                            func_args = {}

                        # ReAct 输出
                        trace_lines.append(f"Thought: {self._summarize_thought(func_name, func_args)}")
                        trace_lines.append(f"Action: {func_name}")
                        trace_lines.append(
                            "Action Input: " + json.dumps(func_args, ensure_ascii=False)
                        )

                        tool_result_str = self.executor.execute_tool(func_name, func_args)
                        trace_lines.append(f"Observation: {tool_result_str}")

                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": tool_result_str
                        })
                else:
                    final_text = strip_nonstandard_tags(message.content or "")
                    self.history.append({"role": "assistant", "content": final_text})
                    if trace_lines:
                        return "\n".join(trace_lines + [f"Final: {final_text}"])
                    return final_text

            except Exception as e:
                logger.error(f"ReportAgent Chat Error: {e}")
                err_msg = f"与大模型通信或执行探针时发生错误: {e}"
                self.history.append({"role": "assistant", "content": err_msg})
                return err_msg

        return "诊断助手进行了过多内部探针调用，已强行中断防止阻挠。请重试或简化你的问题。"

    def clear_history(self):
        """清空对话历史"""
        self.history = [{"role": "system", "content": self.system_prompt}]
