"""Agent handlers executed by the selected backend. UI never constructs traces."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from contextvars import ContextVar
import time

from .config import prompt_checks
from .evaluation import evaluate
from .models import AgentMessage, digest
from .store import now
from .tools import FinancialTools, evidence, policy_compile


class TeamRuntime:
    def __init__(self, task, config, trace, run_dir, *, experience_ids=(), observer=None):
        self.task, self.config, self.trace = task, config, trace
        self.tools = FinancialTools(task, trace, run_dir)
        self.experience_ids = experience_ids
        self.checks = prompt_checks(config.roles["planner"].system_prompt)
        self.route = config.tool_policy["executor"]
        self.replanned = False
        self.observer = observer
        self.started = time.perf_counter()
        self.event_context = ContextVar("competition_agent_event", default=None)

    def observe(self, event):
        if self.observer:
            try:
                self.observer(event, self.trace)
            except Exception:
                pass

    async def tool(self, role, name, fn, args=None, *, cache=None):
        if len(self.trace.tool_calls) >= self.task.constraints.get("max_steps", 100):
            raise RuntimeError("Tool budget exhausted")
        try:
            return await self.tools.call(role, name, fn, args=args,
                                         cache=self.route.get("cache", False) if cache is None else cache,
                                         retries=self.config.tool_policy.get("max_retries", 0))
        finally:
            self.observe("tool_finished")

    def put(self, key, data, source, **kwargs):
        self.trace.artifacts[key] = evidence(key, data, source, **kwargs)
        event = self.event_context.get()
        if event is not None and key not in event["evidence_ids"]:
            event["evidence_ids"].append(key)

    def simulation(self):
        return self.trace.artifacts["simulation"]["data"]

    async def execute_simulation(self, role="executor"):
        name = self.route["preferred"]
        if name not in {"market_simulation", "historical_replay"}:
            raise ValueError(f"Unregistered simulation tool: {name}")
        fn = getattr(self.tools, name)
        data = await self.tool(role, name, fn)
        self.put("simulation", data, data["engine"], caveat=data["assumptions"])

    async def replicate(self, role):
        first = self.simulation()
        fn = self.tools.historical_replay if "ReplayRunner" in first["engine"] else self.tools.market_simulation
        second = await self.tool(role, "independent_replicate", fn, cache=False)
        data = {"first_hash": digest(first), "replicate_hash": digest(second),
                "matched": first == second, "independent_execution": True,
                "seed": self.task.seed, "scope": "entire deterministic simulation artifact; independent process for order-book runs"}
        self.put("reproducibility", data, first["engine"])

    async def history(self, role):
        data = await self.tool(role, "historical_comparison", lambda: self.tools.historical_comparison(self.simulation()))
        self.put("historical_comparison", data, self.tools.manifest["source_url"],
                 caveat="观测行情已冻结；回放为条件模型，宏观变量为合成假设。归一化误差不代表因果识别。")

    async def quant(self, role):
        data = await self.tool(role, "quantitative_analysis", lambda: self.tools.quantitative_analysis(self.simulation()["prices"]))
        self.put("quant", data, "agents.roles.quant_analyst.QuantAnalyst.analyze")

    async def control(self, role):
        data = await self.tool(role, "control_simulation", lambda: self.tools.market_simulation(policy_text=""), {"policy_text": ""})
        self.put("control", {**data, "return_difference": self.simulation()["cumulative_return"] - data["cumulative_return"]},
                 "engine.simulation_loop.MarketEnvironment / same-seed no-policy control")

    async def counterfactual(self, role):
        import math
        candidates = deepcopy(self.task.inputs.get("interventions", []))
        if len(candidates) < 3 or len(candidates) > 6:
            raise ValueError("Counterfactual task requires 3–6 explicit interventions")
        if len({x["name"] for x in candidates}) != len(candidates):
            raise ValueError("Intervention names must be unique")
        base = self.simulation()
        plans = []
        budget = float(self.task.constraints["max_intervention_cost"])
        for candidate in candidates:
            cost = float(candidate["cost"])
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Intervention costs must be finite and nonnegative")
            text = candidate["policy_text"]
            if not text:
                result = base
            else:
                result = await self.tool(role, "market_simulation", lambda t=text: self.tools.market_simulation(intervention=t),
                                         {"intervention": text})
            plans.append({**candidate, "prices": result["prices"], "risk": result["risk"],
                          "feasible": cost <= budget,
                          "objective": result["risk"]["max_drawdown"] + 0.001 * cost})
        feasible = [x for x in plans if x["feasible"]]
        if not feasible:
            raise ValueError("No feasible intervention")
        best = min(feasible, key=lambda p: (p["objective"], p["cost"], p["name"]))
        data = {"plans": plans, "recommended": best["name"], "constraint_satisfaction": best["cost"] <= budget,
                "intervention_cost": best["cost"], "budget": budget,
                "risk_reduction": base["risk"]["max_drawdown"] - best["risk"]["max_drawdown"],
                "objective_definition": "maximum drawdown + 0.001 * normalized intervention cost",
                "seed": self.task.seed, "intervention_day": 4}
        data = await self.tool(role, "counterfactual_compare", lambda: data)
        self.put("counterfactual_search", data, "Civitas same-seed intervention experiments",
                 caveat="成本为预先定义的归一化预算单位；推荐基于本场景目标函数，风险改善可为零。")

    async def dispatch(self, role, incoming):
        start = time.perf_counter()
        spec = self.config.roles[role]
        event = {"event_id": f"event-{len(self.trace.agent_events) + 1}", "agent": role,
                 "role_type": spec.role_type, "prompt_version": spec.prompt_version,
                 "prompt_hash": digest(spec.system_prompt), "started_at": now(),
                 "started_offset": start - self.started, "status": "running", "decision_summary": "",
                 "input_summary": deepcopy(incoming), "task": self.task.task_id, "tools": spec.tools,
                 "evidence_ids": []}
        self.trace.agent_events.append(event)
        for parent, output in incoming.items():
            self.trace.messages.append(AgentMessage(parent, role, "task_handoff", self.task.task_id,
                                                     output, now()).to_dict())
        self.observe("agent_started")
        context_token = self.event_context.set(event)
        try:
            if role == "planner":
                from agents.manager_agent import ManagerAgent
                plan = ManagerAgent.plan_competition_task(self.task, self.config, experience_ids=self.experience_ids,
                                                          feedback=self.trace.errors if self.replanned else ())
                if self.replanned:
                    self.trace.plan["replan"] = plan.to_dict()
                else:
                    self.trace.plan = plan.to_dict()
                event["decision_summary"] = plan.decision_summary
            elif role == "analyst":
                data = await self.tool(role, "policy_compile", lambda: policy_compile(self.task))
                self.put("policy", data, "policy.structured.StructuredPolicyParser.parse",
                         confidence=data["uncertainty"]["confidence"],
                         caveat="规则解析的传导方向和强度是仿真假设，保留 PolicyPackage 中的不确定性与副作用。")
            elif role == "executor":
                await self.execute_simulation()
                if "control" in self.checks:
                    await self.control(role)
                if "replicate" in self.checks and "verifier" not in self.config.topology.active_agents:
                    await self.replicate(role)
                if "historical_comparison" in self.checks and "historian" not in self.config.topology.active_agents:
                    await self.history(role)
                    await self.quant(role)
                if "counterfactual_search" in self.checks and "counterfactual" not in self.config.topology.active_agents:
                    await self.counterfactual(role)
            elif role == "verifier":
                await self.replicate(role)
            elif role == "historian":
                await self.history(role)
            elif role == "quant":
                await self.quant(role)
            elif role == "counterfactual":
                await self.counterfactual(role)
            elif role == "critic":
                from agents.roles.risk_analyst import RiskAnalyst
                from agents.diagnostic.diagnostic_agent import DiagnosticAgent
                # Re-plan only on execution failure, not to hide a deficient baseline prompt.
                if "simulation" not in self.trace.artifacts and not self.config.baseline and not self.replanned:
                    self.replanned = True
                    failure_count = len(self.trace.errors)
                    self.trace.retries += 1
                    await self.dispatch("planner", {"critic": {"message_type": "replan_request",
                                                               "problem": "simulation evidence missing", "retry_budget": 1}})
                    await self.dispatch("executor", {"planner": {"message_type": "retry_assignment", "budget": 1}})
                    for specialist in ("verifier", "historian", "quant", "counterfactual"):
                        if specialist in self.config.topology.active_agents:
                            await self.dispatch(specialist, {"executor": {"message_type": "recovery_handoff"}})
                    if "simulation" in self.trace.artifacts:
                        for error in self.trace.errors[:failure_count]:
                            error["recoverable"] = True
                if "simulation" in self.trace.artifacts:
                    data = await self.tool(role, "risk_analysis", lambda: RiskAnalyst().analyze(self.simulation()["prices"]))
                    self.put("risk", data, "agents.roles.risk_analyst.RiskAnalyst.analyze")
                self.trace.critiques = DiagnosticAgent.review_competition_evidence(self.task, self.trace.artifacts)
            elif role == "reporter":
                from agents.report.report_agent import ReportAgent
                self.trace.report = await self.tool(role, "structured_report", lambda: ReportAgent.synthesize_evidence(
                    self.task, self.trace.artifacts, self.trace.critiques))
            else:
                raise ValueError(f"No implementation for agent {role}")
            event["status"] = "completed"
            if not event["decision_summary"]:
                event["decision_summary"] = f"{spec.role_type} 已执行；证据数量 {len(self.trace.artifacts)}"
        except Exception as exc:
            event["status"] = "failed"
            event["decision_summary"] = f"执行失败：{type(exc).__name__}；下游可继续审查和报告"
            self.trace.errors.append({"agent": role, "type": type(exc).__name__, "recoverable": False})
        finally:
            event["latency"] = time.perf_counter() - start
            event["finished_at"] = now()
            event["evidence_ids"].sort()
            event["tool_call_ids"] = [x["call_id"] for x in self.trace.tool_calls if x["agent"] == role]
            self.observe("agent_finished")
            self.event_context.reset(context_token)
        # Pass compact evidence references through the real DAG, never hidden thoughts.
        return {"agent": role, "status": event["status"], "decision_summary": event["decision_summary"],
                "evidence_ids": sorted(self.trace.artifacts),
                "critique_count": len(self.trace.critiques)}

    async def run(self, backend):
        start = time.perf_counter()
        self.started = start
        self.trace.runtime["workflow_invoked"] = backend.name == "OPENJIUWEN"
        try:
            await backend.execute(self.config.topology, self.dispatch, self.trace.run_id)
            self.trace.runtime["workflow_completed"] = True
        except Exception as exc:
            self.trace.errors.append({"agent": "backend", "type": type(exc).__name__, "recoverable": False})
            self.trace.runtime["workflow_error"] = type(exc).__name__
            # Never silently rerun partially executed workflow under another backend.
        self.trace.latency = time.perf_counter() - start
        self.trace.evaluation = evaluate(self.task, self.trace)
        self.trace.token_usage = self.trace.evaluation["token_usage"]
        self.trace.semantic_hash = digest({"task": self.task.to_dict(), "artifacts": self.trace.artifacts,
                                          "report": self.trace.report,
                                          "evaluation": {k: v for k, v in self.trace.evaluation.items() if k != "latency"}})
        return self.trace
