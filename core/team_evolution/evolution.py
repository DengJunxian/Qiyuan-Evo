"""Evidence-triggered candidate generation. Only the service may promote a tested candidate."""
from __future__ import annotations

import difflib
import time
import uuid

from .backend import PromptOptimizer
from .config import prompt_checks, rebuild_routing
from .models import AgentRoleSpec, AgentTeamConfig, EvolutionAction
from .store import now


class EvolutionController:
    def __init__(self, *, cloud_optimizer=False):
        self.optimizer = PromptOptimizer(enabled=cloud_optimizer)

    async def propose(self, task, trace, statistics):
        start = time.perf_counter()
        config = AgentTeamConfig.from_dict(trace.team_config)
        before = config.to_dict()
        evolution_id = "evo-" + uuid.uuid4().hex
        checks = set(prompt_checks(config.roles["planner"].system_prompt))
        feedback = [c for c in trace.critiques if c["attribution"] == "planner"]
        for item in feedback:
            if item["recommendation"] != "repair_tool_execution":
                checks.add(item["recommendation"])
        config.baseline = False
        actions = []
        def action(target, kind, old, new, reason):
            if old == new:
                return
            actions.append(EvolutionAction(evolution_id + "-" + str(len(actions) + 1), target, kind, old, new,
                                           reason, [trace.run_id], "同输入评估候选；不预设增益").to_dict())

        optimizer_info = {"backend": "LOCAL FALLBACK", "provider_called": False, "token_usage": 0,
                          "reason": "No new prompt instruction required", "latency": 0.0}
        if checks != set(prompt_checks(config.roles["planner"].system_prompt)):
            old = config.roles["planner"].system_prompt
            optimizer_start = time.perf_counter()
            new, optimizer_info = await self.optimizer.optimize(old, str(feedback), sorted(checks),
                                                                prompt_version=config.roles["planner"].prompt_version)
            optimizer_info["latency"] = time.perf_counter() - optimizer_start
            config.roles["planner"].system_prompt = new
            config.roles["planner"].prompt_version = f"v{config.version + 1}"
            action("planner", "prompt_update", old, new, "Critic 的缺失证据反馈转化为可执行规划契约")
            actions[-1]["diff"] = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile="old_prompt", tofile="new_prompt", lineterm=""))

        old_tools = before["tool_policy"]
        if task.task_type == "historical_analysis" and feedback:
            choices = {name: statistics.get("executor:" + name) for name in ("market_simulation", "historical_replay")}
            replay = choices["historical_replay"]
            market = choices["market_simulation"]
            # Try an unobserved task-appropriate tool, then exploit measured quality.
            if replay is None or market is None or replay["average_quality"] >= market["average_quality"]:
                config.tool_policy["executor"] = {"preferred": "historical_replay", "cache": True,
                                                   "reason": "explore unobserved replay tool" if replay is None else "higher observed quality",
                                                   "history": choices}
        elif feedback:
            config.tool_policy["executor"] = {"preferred": "market_simulation", "cache": True,
                                               "reason": "reuse identical deterministic calls; replicate always bypasses cache"}
        if trace.errors:
            config.tool_policy["max_retries"] = min(1, int(task.constraints.get("max_tool_retries", 1)))
        action("tool_policy", "tool_routing", old_tools, config.tool_policy,
               "读取按任务上下文聚合的成功率、质量、时延和失败次数，探索/复用工具策略")

        specialists = []
        if task.task_type == "historical_analysis" and "historical_comparison" in checks:
            specialists = [("historian", "historical_analyst", ["historical_comparison"]),
                           ("quant", "quant_analyst", ["quantitative_analysis"])]
        elif task.task_type == "regulatory_planning" and "counterfactual_search" in checks:
            specialists = [("counterfactual", "counterfactual_analyst", ["market_simulation", "counterfactual_compare"])]
        elif "replicate" in checks:
            specialists = [("verifier", "reproducibility_reviewer", ["independent_replicate"])]
        new_roles = []
        for name, role_type, tools in specialists:
            if name not in config.roles:
                config.roles[name] = AgentRoleSpec(name, role_type, "v1", "完成独立核验，返回工具证据和来源。", tools, [role_type])
                config.topology.active_agents.append(name)
                new_roles.append(name)
        if new_roles:
            config.topology.communication_edges = [e for e in config.topology.communication_edges if e != ["executor", "critic"]]
            for role in new_roles:
                config.topology.communication_edges.extend([["executor", role], [role, "critic"]])
            config.topology.scheduling_policy = "parallel" if len(new_roles) > 1 else "serial"
            rebuild_routing(config)
            action("topology", "split_parallelize" if len(new_roles) > 1 else "spawn_reroute",
                   before["topology"], config.topology.to_dict(),
                   f"将遗漏任务交给实际执行的专家 {new_roles}；审查等待所有依赖完成")
        executor = config.roles["executor"]
        executor.skills = ["historical_replay" if config.tool_policy["executor"]["preferred"] == "historical_replay" else "policy_simulation"]
        if "control" in checks and "control_simulation" not in executor.tools:
            executor.tools.append("control_simulation")
        if "replicate" in checks and "verifier" not in config.topology.active_agents and "independent_replicate" not in executor.tools:
            executor.tools.append("independent_replicate")
        for role, spec in config.roles.items():
            spec.performance_history = [v | {"tool": k.split(":", 1)[1]} for k, v in statistics.items() if k.startswith(role + ":")]
        config.version += 1
        config.validate()
        return config, {"evolution_id": evolution_id, "timestamp": now(), "actions": actions,
                        "source_trace_ids": [trace.run_id], "optimizer": optimizer_info,
                        "controller_trace": {"agent": "evolution_controller", "role_type": "evolution_controller",
                                             "status": "completed", "latency": time.perf_counter() - start,
                                             "decision_summary": f"由 {len(feedback)} 条规划反馈生成 {len(actions)} 个配置变更",
                                             "tool_statistics": statistics}, "promotion": {}}
