"""Frozen baseline and executable planner prompt contracts."""
import json
import re

from .models import AgentRoleSpec, AgentTeamConfig, TeamTopology

CONTRACT_START = "<plan_contract>"
CONTRACT_END = "</plan_contract>"
ALLOWED_CHECKS = {"control", "replicate", "historical_comparison", "counterfactual_search", "sensitivity"}


def planner_prompt(checks=()) -> str:
    return ("你是启元任务规划员。解析任务，安排政策分析、确定性仿真、独立风险审查和证据报告。"
            "不得修改价格事实或评价规则。只输出任务计划和决策摘要。\n"
            "以下机器可读执行契约决定要安排的额外核验任务：\n"
            + CONTRACT_START + json.dumps({"checks": sorted(checks)}, ensure_ascii=False) + CONTRACT_END)


def prompt_checks(prompt: str) -> list[str]:
    match = re.search(r"<plan_contract>(.*?)</plan_contract>", prompt, re.S)
    if not match:
        raise ValueError("Planner prompt is missing its executable plan_contract")
    data = json.loads(match.group(1))
    checks = data.get("checks")
    if not isinstance(checks, list) or any(x not in ALLOWED_CHECKS for x in checks):
        raise ValueError("Unrecognized planner instruction")
    return sorted(set(checks))


def baseline_config() -> AgentTeamConfig:
    definitions = {
        "planner": ("planner", planner_prompt(), []),
        "analyst": ("policy_analyst", "将政策编译为 PolicyPackage，保留渠道、对象、时滞和不确定性。", ["policy_compile"]),
        "executor": ("simulation_executor", "调用 Civitas，输出实测价格、风险与来源。", ["market_simulation", "historical_replay"]),
        "critic": ("risk_critic", "独立检查证据、完成度、复现与报告一致性；逐项归因。", ["risk_analysis"]),
        "reporter": ("reporter", "所有结论绑定 evidence / confidence / caveat / source。", ["structured_report"]),
    }
    roles = {k: AgentRoleSpec(k, kind, "v1", prompt, tools, [kind])
             for k, (kind, prompt, tools) in definitions.items()}
    names = list(roles)
    edges = [list(pair) for pair in zip(names, names[1:])]
    return AgentTeamConfig(1, roles, TeamTopology(names, edges, {a: [b] for a, b in edges}),
                           {"executor": {"preferred": "market_simulation", "cache": False},
                            "max_retries": 0}, baseline=True)


def rebuild_routing(config: AgentTeamConfig):
    config.topology.routing = {a: [b for source, b in config.topology.communication_edges if source == a]
                               for a in config.topology.active_agents}
    config.validate()
