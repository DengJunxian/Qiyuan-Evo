"""Real openJiuwen DAG execution, with explicitly labelled local fallback.

APIs verified against the installed 0.1.17.post1 distribution, not inferred
from older tutorials. No API key is needed for deterministic Workflow nodes.
"""
from __future__ import annotations

import asyncio
from importlib import metadata
import os
from typing import Awaitable, Callable

from .models import TeamTopology


def installed_version() -> str | None:
    try:
        return metadata.version("openjiuwen")
    except metadata.PackageNotFoundError:
        return None


class LocalFallbackBackend:
    name = "LOCAL FALLBACK"

    def __init__(self, reason="explicit local backend"):
        self.reason = reason

    def info(self):
        return {"backend": self.name, "package_version": installed_version(),
                "sdk_imported": False, "reason": self.reason,
                "workflow_invoked": False, "workflow_completed": False,
                "llm_mode": "LOCAL FALLBACK", "llm_reason": "deterministic role implementations"}

    async def execute(self, topology: TeamTopology, dispatch: Callable[[str, dict], Awaitable[dict]], run_id: str):
        results = {}
        for layer in topology.layers():
            async def call(role):
                incoming = {a: results[a] for a, b in topology.communication_edges if b == role}
                return role, await dispatch(role, incoming)
            if topology.scheduling_policy == "parallel":
                values = await asyncio.gather(*(call(role) for role in layer))
            else:
                values = [await call(role) for role in layer]
            results.update(values)
        return {"completed_agents": sorted(results)}


class OpenJiuwenBackend:
    name = "OPENJIUWEN"

    def __init__(self):
        from openjiuwen.core.workflow import Workflow, WorkflowCard, WorkflowComponent, Start, End
        from openjiuwen.core.session.workflow import create_workflow_session

        self.Workflow, self.Card, self.Component = Workflow, WorkflowCard, WorkflowComponent
        self.Start, self.End, self.session_factory = Start, End, create_workflow_session

    def info(self):
        return {"backend": self.name, "package_version": installed_version(),
                "sdk_imported": True, "reason": "native Workflow DAG with one component per task agent",
                "workflow_invoked": False, "workflow_completed": False,
                "llm_mode": "LOCAL FALLBACK", "llm_reason": "deterministic roles; workflow needs no API key"}

    async def execute(self, topology: TeamTopology, dispatch, run_id: str):
        topology.layers()
        component_base = self.Component

        class AgentComponent(component_base):
            def __init__(self, role):
                super().__init__()
                self.role = role

            async def invoke(self, inputs, session, context):
                return {"result": await dispatch(self.role, inputs or {})}

        flow = self.Workflow(card=self.Card(id=run_id, name="Qiyuan Agent Team", version="1"))
        flow.set_start_comp("start", self.Start(), inputs_schema={"task_id": "${task_id}"})
        for role in topology.active_agents:
            parents = [a for a, b in topology.communication_edges if b == role]
            flow.add_workflow_comp(role, AgentComponent(role), wait_for_all=True,
                                   inputs_schema={a: "${" + a + ".result}" for a in parents})
            if not parents:
                flow.add_connection("start", role)
        for source, target in topology.communication_edges:
            flow.add_connection(source, target)
        sinks = [a for a in topology.active_agents if not topology.routing.get(a)]
        flow.set_end_comp("end", self.End(), inputs_schema={a: "${" + a + ".result}" for a in sinks})
        for role in sinks:
            flow.add_connection(role, "end")
        result = await flow.invoke({"task_id": run_id}, self.session_factory(session_id=run_id))
        return result.result


def select_backend(requested="auto"):
    if requested not in {"auto", "openjiuwen", "local"}:
        raise ValueError("backend must be auto, openjiuwen or local")
    if requested == "local":
        return LocalFallbackBackend()
    try:
        return OpenJiuwenBackend()
    except Exception as exc:
        # Explicit SDK mode is strict, for CI/competition acceptance runs.
        if requested == "openjiuwen":
            raise RuntimeError(f"openJiuwen unavailable: {type(exc).__name__}") from exc
        return LocalFallbackBackend(f"openJiuwen import failed: {type(exc).__name__}")


class PromptOptimizer:
    """Official feedback optimizer when configured; executable-contract repair offline."""

    def __init__(self, *, enabled=False):
        self.enabled = enabled

    async def optimize(self, old_prompt: str, feedback: str, required_checks: list[str], *, prompt_version=None):
        from .config import planner_prompt, prompt_checks
        from .models import digest
        from .store import now
        local = planner_prompt(required_checks)
        request_model = os.environ.get("QIYUAN_LLM_MODEL", "deepseek-chat")
        metadata = {"requested_model": request_model, "temperature": 0,
                    "prompt_version": prompt_version, "timestamp": now(),
                    "input_hash": digest({"old_prompt": old_prompt, "feedback": feedback,
                                          "required_checks": required_checks}),
                    "prompt_hash": digest(old_prompt), "provider_response_hash": None}
        def audit(info, output):
            return metadata | info | {"model": request_model if info["provider_called"] else "none",
                                      "finished_at": now(), "response_hash": digest(output)}
        key = os.environ.get("QIYUAN_LLM_API_KEY", "").strip()
        if not self.enabled or not key or key.startswith("your_"):
            return local, audit({"backend": "LOCAL FALLBACK", "reason": "API key missing or cloud optimizer disabled",
                                 "token_usage": 0, "provider_called": False}, local)
        try:
            from openjiuwen.core.foundation.llm import ModelRequestConfig, ModelClientConfig
            from openjiuwen.dev_tools.prompt_builder.builder.feedback_prompt_builder import FeedbackPromptBuilder

            builder = FeedbackPromptBuilder(
                ModelRequestConfig(model=request_model, temperature=0),
                ModelClientConfig(client_provider="OpenAI", api_key=key,
                                  api_base=os.environ.get("QIYUAN_LLM_BASE_URL", "https://api.deepseek.com/v1"),
                                  verify_ssl=True))
            response = await asyncio.wait_for(builder.build(
                old_prompt, feedback=feedback + "\n保留且完整输出执行契约：\n" + local,
                language="zh-CN"), timeout=30)
            metadata["provider_response_hash"] = digest(response)
            from core.llm_client import strip_nonstandard_tags
            response = strip_nonstandard_tags(response or "")
            if prompt_checks(response) != sorted(set(required_checks)):
                raise ValueError("Optimizer did not preserve requested execution contract")
            return response, audit({"backend": "OPENJIUWEN", "provider_called": True,
                              "api": "FeedbackPromptBuilder.build", "token_usage": None,
                              "reason": "SDK builder returns text without usage; token count unavailable"}, response)
        except Exception as exc:
            return local, audit({"backend": "LOCAL FALLBACK", "provider_called": True,
                           "token_usage": None, "error_type": type(exc).__name__,
                           "reason": "Cloud prompt optimization failed; deterministic contract repair applied"}, local)
