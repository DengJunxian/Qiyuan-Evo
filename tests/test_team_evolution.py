from copy import deepcopy
import json

import pytest

from core.competition_service import CompetitionService, run_sync
from core.team_evolution.benchmark import benchmark_tasks, get_task, load_dataset
from core.team_evolution.config import baseline_config, prompt_checks, planner_prompt
from core.team_evolution.evaluation import evaluate, review_evidence
from core.team_evolution.models import AgentTeamConfig, ExecutionTrace, TeamTopology, digest
from core.team_evolution.store import context_key
from core.team_evolution.tools import FinancialTools


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    service = CompetitionService(tmp_path_factory.mktemp("competition"), backend="local")
    service.run_benchmark(rounds=1)
    saved = json.loads((service.store.root / "before_after.json").read_text())
    comparisons = {x["baseline"]["task"]["task_type"]: x for x in saved}
    return service, comparisons


@pytest.mark.parametrize("kind", ["policy_report", "historical_analysis", "regulatory_planning"])
def test_real_tasks_complete_with_evidence_and_executed_agents(suite, kind):
    _, comparisons = suite
    result = comparisons[kind]
    trace = result["evolved"]
    assert trace["evaluation"]["task_success"], trace["errors"]
    assert not trace["errors"]
    assert len({x["role_type"] for x in trace["agent_events"]}) >= 3
    assert set(trace["team_config"]["topology"]["active_agents"]) == {x["agent"] for x in trace["agent_events"]}
    assert trace["evaluation"]["active_agent_count"] == len(trace["agent_events"])
    assert trace["messages"] and trace["tool_calls"]
    assert json.loads(json.dumps(trace, allow_nan=False)) == trace
    assert trace["task"]["seed"] == result["baseline"]["task"]["seed"] == 42
    assert trace["runtime"]["dataset_hash"] == result["baseline"]["runtime"]["dataset_hash"]
    assert trace["task"]["model_config"] == result["baseline"]["task"]["model_config"]
    assert result["promotion"]["accepted"]
    assert trace["artifacts"]["reproducibility"]["data"]["matched"]
    assert trace["team_config"]["roles"]["planner"]["prompt_version"] != "v1"
    assert trace["team_config"]["topology"] != result["baseline"]["team_config"]["topology"]
    assert trace["team_config"]["tool_policy"] != result["baseline"]["team_config"]["tool_policy"]


def test_scorecard_is_computed_from_trace_and_exposes_zero_tokens(suite):
    for result in suite[1].values():
        data = result["evolved"]
        trace = ExecutionTrace(**data)
        score = evaluate(get_task(data["task"]["task_type"]), trace)
        assert score == trace.evaluation
        assert score["tool_calls"] == len(trace.tool_calls)
        assert score["token_usage"] == 0
        assert result["evolution_gain"]["token_reduction"] is None
        assert trace.evaluation["quality_score"] - result["baseline"]["evaluation"]["quality_score"] == result["evolution_gain"]["quality_gain"]


def test_prompt_has_executable_effect_without_version_shortcuts():
    from agents.manager_agent import ManagerAgent
    task = get_task("policy_report")
    config = baseline_config()
    a = ManagerAgent.plan_competition_task(task, config)
    config.roles["planner"].system_prompt = planner_prompt(["control", "replicate"])
    b = ManagerAgent.plan_competition_task(task, config)
    # Same version: behavior depends on prompt content, never the v2 label.
    assert config.roles["planner"].prompt_version == "v1"
    assert a.expected_evidence != b.expected_evidence
    assert "replicate" in next(s for s in b.subtasks if s["subtask_id"] == "executor")["checks"]
    with pytest.raises(ValueError):
        prompt_checks('<plan_contract>{"checks":["invent_prices"]}</plan_contract>')


def test_candidate_config_does_not_alias_source_snapshot():
    original = baseline_config().to_dict()
    candidate = AgentTeamConfig.from_dict(original)
    candidate.tool_policy["executor"]["cache"] = True
    candidate.roles["executor"].tools.append("new_tool")
    candidate.topology.communication_edges.append(["critic", "planner"])
    assert original == baseline_config().to_dict()


def test_benchmark_artifacts_are_complete_and_trace_backed(suite):
    service = suite[0]
    for name in ("summary", "run_trace", "evolution_history", "before_after", "team_topology", "prompt_registry"):
        assert json.loads((service.store.root / (name + ".json")).read_text())
    summary = service.get_benchmark_summary()
    assert len(summary["tasks"]) == 3
    for row in summary["tasks"]:
        assert service.get_execution_trace(row["evolved_run_id"])["evaluation"] == row["evolved"]


def test_baseline_does_not_read_evolution_memory(suite, monkeypatch):
    service = suite[0]
    def forbidden(*args, **kwargs):
        raise AssertionError("baseline read evolution memory")
    monkeypatch.setattr(service.store, "retrieve", forbidden)
    monkeypatch.setattr(service.store, "current_config", forbidden)
    trace = service.run_baseline("policy_report")
    assert trace["team_config"] == baseline_config().to_dict()
    assert trace["plan"]["retrieved_experience_ids"] == []
    assert trace["runtime"]["baseline_experience_reads"] == 0


def test_history_reuse_and_no_unjustified_promotion(suite):
    service = suite[0]
    second = service.run_before_after("historical_analysis")
    assert second["evolved"]["plan"]["retrieved_experience_ids"]
    assert second["evolved"]["team_config"]["version"] == 2
    assert second["promotion"]["accepted"] is False
    assert second["candidate"] is None
    assert any(c["tool"] == "historical_replay" for c in second["evolved"]["tool_calls"])
    assert any(c["tool"] == "market_simulation" for c in second["baseline"]["tool_calls"])
    assert service.store.tool_statistics(service._key(get_task("historical_analysis")))["executor:historical_replay"]["calls"] > 0


def test_evolution_lineage_includes_real_traces_and_prompt_diffs(suite):
    service = suite[0]
    for history in service.get_evolution_history():
        assert history["controller_trace"]["role_type"] == "evolution_controller"
        assert history["controller_trace"]["status"] == "completed"
        for run_id in history["source_trace_ids"]:
            assert service.get_execution_trace(run_id)
        for action in history["actions"]:
            assert action["evaluation_before"] and action["evaluation_after"]
            for run_id in action["source_trace_ids"]:
                assert service.get_execution_trace(run_id)
            if action["action_type"] == "prompt_update":
                assert action["diff"] and action["before"] != action["after"]
                assert action["accepted"] is True
    assert len(service.store.prompt_registry()) == 3


def test_same_artifact_reproduced_from_saved_config(suite):
    service, comparisons = suite
    for kind in ("policy_report", "historical_analysis"):
        original = comparisons[kind]["evolved"]
        rerun = service.rerun_execution(original["run_id"])
        assert rerun["semantic_match"]
        assert rerun["rerun"]["artifacts"] == original["artifacts"]
        assert rerun["rerun"]["run_id"] != original["run_id"]


def test_observed_data_provenance_and_financial_results(suite):
    frame, manifest = load_dataset()
    assert len(frame) == manifest["rows"] == 18
    assert manifest["kind"] == "observed_market_data"
    result = suite[1]["historical_analysis"]["evolved"]["artifacts"]
    assert result["simulation"]["data"]["macro_provider"] == "deterministic_macro_fixture"
    assert result["historical_comparison"]["data"]["source"]["sha256"] == manifest["sha256"]
    policy = suite[1]["policy_report"]["evolved"]["artifacts"]["simulation"]["data"]
    assert policy["trades"] > 0
    assert policy["cumulative_return"] == policy["prices"][-1] / policy["prices"][0] - 1
    planning = suite[1]["regulatory_planning"]["evolved"]["artifacts"]["counterfactual_search"]["data"]
    chosen = next(p for p in planning["plans"] if p["name"] == planning["recommended"])
    assert chosen["cost"] <= planning["budget"]


def test_critic_detects_recomputed_hash_with_contradictory_risk(suite):
    data = deepcopy(suite[1]["policy_report"]["evolved"])
    data["artifacts"]["risk"]["data"]["max_drawdown"] = 0.99
    data["artifacts"]["risk"]["content_hash"] = digest(data["artifacts"]["risk"]["data"])
    assert any(c["dimension"] == "consistency" and c["severity"] == "critical"
               for c in review_evidence(get_task("policy_report"), data["artifacts"]))


def test_execution_failure_produces_report_and_failed_score(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("simulated API/tool outage")
    monkeypatch.setattr(FinancialTools, "market_simulation", unavailable)
    trace = CompetitionService(tmp_path, backend="local").run_baseline("policy_report")
    assert trace["errors"] and not trace["evaluation"]["task_success"]
    assert trace["report"]["conclusions"]
    assert next(x for x in trace["agent_events"] if x["agent"] == "reporter")["status"] == "completed"


def test_bounded_replan_is_executed_and_recorded(tmp_path, monkeypatch):
    task = get_task("policy_report")
    task.expected_outputs = ["policy", "simulation", "risk"]
    config = baseline_config()
    config.baseline = False
    config.tool_policy["executor"]["preferred"] = "historical_replay"
    real = FinancialTools.historical_replay
    calls = []
    def transient(self):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("transient outage")
        return real(self)
    monkeypatch.setattr(FinancialTools, "historical_replay", transient)
    service = CompetitionService(tmp_path, backend="local")
    trace = run_sync(service._run(task, config, "evolved")).to_dict()
    assert len(calls) == 2
    assert trace["retries"] == 1 and "replan" in trace["plan"]
    assert sum(e["agent"] == "planner" for e in trace["agent_events"]) == 2
    assert sum(e["agent"] == "executor" for e in trace["agent_events"]) == 2
    assert all(e["recoverable"] for e in trace["errors"])
    assert trace["evaluation"]["task_success"]


def test_non_improving_candidate_is_rejected_and_registry_retains_it(tmp_path, monkeypatch):
    service = CompetitionService(tmp_path, backend="local")
    real = service.controller.propose
    async def ineffective(*args, **kwargs):
        _, history = await real(*args, **kwargs)
        config = baseline_config()
        config.baseline = False
        config.version = 2
        return config, history
    monkeypatch.setattr(service.controller, "propose", ineffective)
    comparison = service.run_before_after("historical_analysis")
    assert comparison["candidate"] is not None
    assert not comparison["promotion"]["accepted"]
    assert comparison["evolved"]["team_config"]["version"] == 1
    assert all(p["accepted"] is False for p in service.store.prompt_registry())


def test_tool_cache_records_hits_and_replicate_bypasses_it(tmp_path):
    task = get_task("policy_report")
    trace = ExecutionTrace("cache-test", task.task_id, task.to_dict(), {}, {}, "test")
    tools = FinancialTools(task, trace, tmp_path)
    calls = []
    def compute():
        calls.append(1)
        return {"value": 7}
    async def exercise():
        await tools.call("executor", "test_computation", compute, cache=True)
        await tools.call("executor", "test_computation", compute, cache=True)
        await tools.call("executor", "test_computation", compute, cache=False)
    run_sync(exercise())
    assert len(calls) == 2
    assert [x["cache_hit"] for x in trace.tool_calls] == [False, True, False]


def test_local_fallback_for_missing_package(monkeypatch):
    import core.team_evolution.backend as backend
    def missing():
        raise ImportError("package unavailable")
    monkeypatch.setattr(backend, "OpenJiuwenBackend", missing)
    actual = backend.select_backend("auto")
    assert actual.info()["backend"] == "LOCAL FALLBACK"
    assert actual.info()["sdk_imported"] is False
    with pytest.raises(RuntimeError):
        backend.select_backend("openjiuwen")


def test_prompt_optimizer_missing_key_is_explicit(monkeypatch):
    from core.team_evolution.backend import PromptOptimizer
    monkeypatch.delenv("QIYUAN_LLM_API_KEY", raising=False)
    prompt, info = run_sync(PromptOptimizer(enabled=True).optimize(planner_prompt(), "missing repeat", ["replicate"]))
    assert prompt_checks(prompt) == ["replicate"]
    assert info["backend"] == "LOCAL FALLBACK" and not info["provider_called"]
    assert info["token_usage"] == 0


def test_task_context_separates_datasets_models_and_constraints():
    t = get_task("policy_report")
    key = context_key(t, "dataset-a")
    assert key != context_key(t, "dataset-b")
    t.constraints["simulation_days"] += 1
    assert key != context_key(t, "dataset-a")


def test_topology_rejects_cycles_and_disabled_agents():
    config = baseline_config()
    config.topology.communication_edges.append(["reporter", "planner"])
    with pytest.raises(ValueError, match="cycle"):
        config.validate()
    config = baseline_config()
    config.roles["critic"].enabled = False
    with pytest.raises(ValueError, match="disabled"):
        config.validate()


def test_sqlite_promotion_compare_and_swap(tmp_path):
    from core.team_evolution.store import ExperienceStore
    store = ExperienceStore(tmp_path)
    config = baseline_config()
    config.version = 2
    history = {"evolution_id": "one", "actions": [], "promotion": {"accepted": True}}
    assert store.commit_evolution("key", history, config, base_version=1, accepted=True)
    history = {"evolution_id": "two", "actions": [], "promotion": {"accepted": True}}
    config.version = 3
    assert not store.commit_evolution("key", history, config, base_version=1, accepted=True)
    assert store.current_config("key").version == 2


def test_baseline_mode_conflicts_rejected(tmp_path):
    with pytest.raises(ValueError):
        CompetitionService(tmp_path, backend="local").run_competition_task("policy_report", baseline_mode=True, evolved_mode=True)
