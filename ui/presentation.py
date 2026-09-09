"""Chinese display vocabulary and summaries derived only from recorded evidence."""
import re

from core.team_evolution.evaluation import valid_evidence

ROLE_NAMES = {"planner": "任务规划员", "analyst": "政策分析员", "executor": "仿真执行员",
              "critic": "风险评审员", "reporter": "报告撰写员", "verifier": "独立复核员",
              "historian": "历史分析员", "quant": "量化分析员", "counterfactual": "方案比较员",
              "evolution_controller": "演进控制器"}
EVIDENCE_NAMES = {"policy": "政策解析", "simulation": "市场仿真", "control": "无政策对照",
                  "reproducibility": "独立复核", "historical_comparison": "历史行情对照",
                  "counterfactual_search": "干预方案比较", "quant": "量化分析", "risk": "风险测量"}
TOOL_NAMES = {"market_simulation": "市场仿真", "historical_replay": "历史回放", "compile_policy": "政策解析",
              "policy_compile": "政策解析", "control_simulation": "无政策对照", "risk_analysis": "风险测量", "replicate": "独立复跑",
              "evidence_review": "证据审查", "synthesize_report": "报告汇总", "plan_task": "任务分解",
              "review_evidence": "证据审查", "report": "报告汇总", "quantitative_analysis": "量化分析",
              "independent_replicate": "独立复跑", "counterfactual_compare": "干预方案比较", "structured_report": "报告汇总", "quant_analysis": "量化分析",
              **EVIDENCE_NAMES}
STATE_NAMES = dict(IDLE="待命", PLANNING="规划中", RUNNING="执行中", WAITING="等待上游",
                   REVIEWING="审查中", FAILED="执行失败", COMPLETED="已完成", EVOLVING="调整中")
ACTION_NAMES = {"prompt_update": "修改提示词", "tool_routing": "调整工具策略",
                "spawn_reroute": "增加专业角色", "split_parallelize": "拆分并行任务"}
PROFILE_NAMES = {"REPRODUCIBLE DEMO": "本地可复现实验", "ONLINE LIVE": "联网优化", "OFFLINE FALLBACK": "离线备用"}
TASK_TITLES = {"policy_report": "印花税下调：补齐政策影响评估",
               "historical_analysis": "2024 年政策窗口：核对历史走势",
               "regulatory_planning": "恐慌抛售：比较三种干预方案"}
PLAN_NAMES = {"no_action": "暂不干预", "clarification": "信息澄清", "liquidity_support": "流动性支持"}


def chinese(value):
    """Display only: retain raw identifiers in downloadable records."""
    text = str(value)
    labels = {**EVIDENCE_NAMES, **TOOL_NAMES, **ROLE_NAMES, **PLAN_NAMES,
              "Tax cut": "减税", "Liquidity easing": "流动性宽松", "Tightening": "政策收紧",
              "easing": "宽松", "tightening": "收紧", "Critic": "评审员", "Prompt": "提示词",
              "task_handoff": "移交任务", "review_feedback": "反馈审查结果", "parallel": "并行", "serial": "串行", "success": "成功", "failed": "失败", "completed": "已完成",
              "policy_analyst": "政策分析", "simulation_executor": "仿真执行", "risk_critic": "风险审查", "planner_orchestrator": "任务规划", "synthesizer": "报告汇总"}
    for key in sorted(labels, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z_])" + re.escape(key) + r"(?![A-Za-z_])", lambda _: labels[key], text)
    return text


def evidence_progress(trace):
    expected = trace.get("task", {}).get("expected_outputs", [])
    found = [k for k in expected if valid_evidence(k, trace.get("artifacts", {}).get(k, {}))]
    return len(found), len(expected)


def outcome_summary(trace):
    data = {k: v.get("data", {}) for k, v in trace.get("artifacts", {}).items()}
    if "historical_comparison" in data:
        h = data["historical_comparison"]
        return f'回放与观测行情的方向一致率为 {h["direction_consistency"]:.1%}，归一化均方根误差为 {h["normalized_rmse"]:.4f}。'
    if "counterfactual_search" in data:
        c = data["counterfactual_search"]
        return f'比较 {len(c["plans"])} 个方案后，推荐{PLAN_NAMES.get(c["recommended"], c["recommended"])}；相对不干预的最大回撤降低 {c["risk_reduction"]*100:.2f} 个百分点。'
    if "control" in data:
        return f'政策组与无政策组的期末收益差为 {data["control"]["return_difference"]*100:+.2f} 个百分点。'
    if "simulation" in data:
        return f'本轮模拟累计收益 {data["simulation"]["cumulative_return"]:+.2%}，其余验证见任务记录。'
    return "实验尚未形成市场结果。"


def action_summary(action):
    before, after = action["before"], action["after"]
    if action["target"] == "topology":
        added = sorted(set(after["active_agents"]) - set(before["active_agents"]))
        retired = sorted(set(before["active_agents"]) - set(after["active_agents"]))
        parts = (["加入" + "、".join(ROLE_NAMES.get(k, k) for k in added)] if added else [])
        if retired: parts.append("移除" + "、".join(ROLE_NAMES.get(k, k) for k in retired))
        if after["scheduling_policy"] != before["scheduling_policy"]: parts.append("改为" + chinese(after["scheduling_policy"]) + "执行")
        return "；".join(parts) or "更新任务之间的依赖关系"
    if action["action_type"] == "tool_routing":
        old, new = before["executor"], after["executor"]
        if old["preferred"] != new["preferred"]:
            return chinese(old["preferred"]) + " → " + chinese(new["preferred"])
        if old.get("cache") != new.get("cache"):
            return "相同输入允许复用计算结果；独立复核仍重新计算" if new["cache"] else "关闭计算结果复用"
        return "更新工具选择顺序"
    if action["action_type"] == "prompt_update":
        # Prompt schema differs between optimizer versions; use recorded text only.
        match = re.search(r'<plan_contract>(.*?)</plan_contract>', str(after), re.S)
        if match:
            import json
            try:
                checks = json.loads(match[1])["checks"]
                return "任务计划增加：" + "、".join(chinese(k) for k in checks)
            except (ValueError, KeyError):
                pass
        return "把评审指出的遗漏加入下一轮任务计划"
    return chinese(action["rationale"])
