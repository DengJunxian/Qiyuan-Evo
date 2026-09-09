import asyncio
from importlib.metadata import version

import pytest

from core.competition_service import CompetitionService, run_sync
from core.team_evolution.backend import OpenJiuwenBackend, LocalFallbackBackend, PromptOptimizer
from core.team_evolution.config import planner_prompt, prompt_checks
from core.team_evolution.models import TeamTopology


@pytest.mark.integration
def test_native_workflow_executes_dependency_dag_and_parallel_branches():
    backend = OpenJiuwenBackend()  # Deliberately not skip/fallback when installed SDK is broken.
    assert backend.info()["package_version"] == version("openjiuwen")
    graph = TeamTopology(["planner", "historian", "quant", "reporter"],
                         [["planner", "historian"], ["planner", "quant"], ["historian", "reporter"], ["quant", "reporter"]],
                         {"planner": ["historian", "quant"], "historian": ["reporter"], "quant": ["reporter"], "reporter": []}, "parallel")
    async def execute(engine):
        active = set()
        overlap = []
        seen = {}
        async def dispatch(role, incoming):
            seen[role] = set(incoming)
            active.add(role)
            if role in {"historian", "quant"}:
                await asyncio.sleep(0.04)
                overlap.append(len(active))
            active.remove(role)
            return {"role": role}
        await engine.execute(graph, dispatch, "native-test")
        return seen, overlap
    seen, overlap = run_sync(execute(backend))
    assert seen["reporter"] == {"historian", "quant"}
    assert max(overlap) == 2
    assert run_sync(execute(LocalFallbackBackend()))[0] == seen


@pytest.mark.integration
def test_native_sdk_is_in_the_real_task_path(tmp_path):
    service = CompetitionService(tmp_path, backend="openjiuwen")
    result = service.run_before_after("historical_analysis")
    trace = result["evolved"]
    assert trace["runtime"]["backend"] == "OPENJIUWEN"
    assert trace["runtime"]["workflow_invoked"] and trace["runtime"]["workflow_completed"]
    assert trace["evaluation"]["task_success"], trace["errors"]
    assert len(trace["agent_events"]) == 7


@pytest.mark.integration
def test_cloud_model_failure_returns_explicit_fallback(monkeypatch):
    from openjiuwen.dev_tools.prompt_builder.builder.feedback_prompt_builder import FeedbackPromptBuilder
    async def unavailable(*args, **kwargs):
        raise ConnectionError("test outage")
    monkeypatch.setenv("QIYUAN_LLM_API_KEY", "test-credential-not-real")
    monkeypatch.setattr(FeedbackPromptBuilder, "build", unavailable)
    prompt, info = run_sync(PromptOptimizer(enabled=True).optimize(planner_prompt(), "missing repeat", ["replicate"]))
    assert prompt_checks(prompt) == ["replicate"]
    assert info["backend"] == "LOCAL FALLBACK" and info["provider_called"]
    assert info["token_usage"] is None
    assert "test-credential" not in str(info)


def test_prompt_usage_is_observed_from_sdk_callbacks_across_event_loops(monkeypatch):
    import json
    from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient
    from openjiuwen.core.foundation.llm.schema.message import AssistantMessage, UsageMetadata
    received_configs=[]
    async def measured(self, *args, **kwargs):
        received_configs.append(self.model_client_config)
        return AssistantMessage(content=planner_prompt(['replicate']),
                                reasoning_content='private reasoning must not be exported',
                                usage_metadata=UsageMetadata(input_tokens=100,output_tokens=20,total_tokens=120))
    monkeypatch.setenv('QIYUAN_LLM_API_KEY','test-credential-never-exported')
    monkeypatch.setattr(OpenAIModelClient,'invoke',measured)
    for _ in range(2):
        _,audit=run_sync(PromptOptimizer(enabled=True).optimize(planner_prompt(),'独立复核',['replicate']))
        assert audit['backend']=='OPENJIUWEN',audit
        assert audit['token_usage']==120 and audit['usage']['input_tokens']==100
        assert audit['response_received'] and len(audit['request_messages_hash'])==64
        assert 'test-credential' not in json.dumps(audit)
        assert 'private reasoning' not in json.dumps(audit)
    assert all(not c.use_shared_llm_http_client for c in received_configs)
