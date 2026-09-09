"""Final acceptance: real default demo, durable identity, and honest failure paths."""
from copy import deepcopy
import io
import json
from pathlib import Path
import time
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

from core.competition_cockpit import DEFAULT_COMPETITION_DEMO, DemoJobs, validate_comparison
from core.competition_service import run_sync
from core.team_evolution.backend import PromptOptimizer
from core.team_evolution.config import planner_prompt
from core.team_evolution.evidence_bundle import bundle_files, bundle_zip
from core.team_evolution.models import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def default_demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-demo")
    jobs = DemoJobs(root)
    job_id = jobs.start(**{k: DEFAULT_COMPETITION_DEMO[k] for k in ("task_type", "profile", "scale")})
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        job = jobs.snapshot(job_id)
        if job["status"] in {"COMPLETED", "FAILED"}:
            assert job["status"] == "COMPLETED", job["error"]
            return root, jobs, job
        time.sleep(.1)
    pytest.fail("Default demo exceeded acceptance time budget")


def test_default_demo_has_real_critic_failure_and_three_executed_changes(default_demo):
    _, _, job = default_demo
    result = validate_comparison(job["result"])
    a, b = result["baseline"], result["evolved"]
    assert result["seed"] == DEFAULT_COMPETITION_DEMO["seed"]
    assert a["task"] == b["task"]
    assert {c["dimension"] for c in a["critiques"]} == {"control", "reproducibility"}
    assert not a["evaluation"]["task_success"] and b["evaluation"]["task_success"]
    assert {x["action_type"] for x in result["evolution_history"][0]["actions"]} == {"prompt_update", "tool_routing", "spawn_reroute"}
    assert len(a["agent_events"]) == 5 and len(b["agent_events"]) == 6
    assert b["plan"]["retrieved_experience_ids"]
    assert any(e["agent"] == "verifier" and e["status"] == "completed" for e in b["agent_events"])
    assert b["runtime"]["backend"] == "OPENJIUWEN" and b["runtime"]["workflow_completed"]


def test_evidence_bundle_reuses_canonical_artifacts_and_rejects_tampering(default_demo):
    _, _, job = default_demo
    root = Path(job["output_dir"])
    manifest = json.loads((root / "competition_demo_summary.json").read_text())
    assert {"baseline_trace.json", "evolved_trace.json", "evaluation.json", "evolution_history.json",
            "prompt_diff.json", "topology_before.json", "topology_after.json", "benchmark_summary.json"} <= manifest["artifacts"].keys()
    files = bundle_files(root, manifest)
    assert manifest["experiment_id"] == job["result"]["experiment_id"]
    assert manifest["artifacts"]["baseline_trace.json"]["path"].startswith("runs/")
    with zipfile.ZipFile(io.BytesIO(bundle_zip(root, manifest))) as archive:
        assert "competition_demo_summary.json" in archive.namelist()
        assert all(path in archive.namelist() for path in files)
    changed = deepcopy(manifest)
    changed["artifacts"]["baseline_trace.json"]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"): bundle_files(root, changed)
    changed = deepcopy(manifest); changed["experiment_id"] = "foreign"
    with pytest.raises(ValueError, match="identity mismatch"): bundle_files(root, changed)
    changed = deepcopy(job["result"]); changed["evolution_gain"]["quality_gain"] = .99
    with pytest.raises(ValueError, match="gain"): validate_comparison(changed)


def test_restart_recovers_saved_experiment_and_browser_refresh_restores_it(default_demo, monkeypatch):
    root, _, saved = default_demo
    restored_jobs = DemoJobs(root)
    job = restored_jobs.snapshot(saved["job_id"])
    assert job["recovered"] and job["status"] == "COMPLETED"
    assert job["result"]["experiment_id"] == saved["result"]["experiment_id"]
    import ui.team_evolution as ui
    monkeypatch.setattr(ui, "get_jobs", lambda: restored_jobs)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.query_params["demo"] = saved["job_id"]
    app.run()
    assert not app.exception
    assert app.session_state["ev_result"]["experiment_id"] == saved["result"]["experiment_id"]
    assert "恢复已完成实验" in app.session_state["ev_origin"]
    app.button(key="top_entry_技术与复现").click().run()
    assert not app.exception
    refreshed = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    refreshed.query_params.update(app.query_params)
    refreshed.run()
    assert not refreshed.exception and refreshed.session_state["entry"] == "技术与复现"
    assert refreshed.session_state["ev_result"]["experiment_id"] == saved["result"]["experiment_id"]


def test_interrupted_process_is_not_presented_as_live_or_completed(tmp_path):
    jobs = DemoJobs(tmp_path)
    record = {"job_id": "demo-" + "1" * 32, "status": "RUNNING", "action": "compare",
              "output_dir": str(tmp_path), "profile": "REPRODUCIBLE DEMO", "scale": "答辩演示"}
    jobs._persist(record)
    recovered = DemoJobs(tmp_path).snapshot(record["job_id"])
    assert recovered["status"] == "FAILED" and recovered["error"]["type"] == "InterruptedRun"
    with pytest.raises(KeyError): jobs.snapshot("../../secret")


def test_cloud_prompt_metadata_records_input_output_and_model_without_credentials(monkeypatch):
    from openjiuwen.dev_tools.prompt_builder.builder.feedback_prompt_builder import FeedbackPromptBuilder
    async def response(*args, **kwargs):
        return planner_prompt(["replicate"])
    monkeypatch.setenv("QIYUAN_LLM_API_KEY", "test-credential-never-exported")
    monkeypatch.setenv("QIYUAN_LLM_MODEL", "test-model")
    monkeypatch.setattr(FeedbackPromptBuilder, "build", response)
    output, metadata = run_sync(PromptOptimizer(enabled=True).optimize(planner_prompt(), "missing repeat", ["replicate"], prompt_version="v1"))
    assert metadata["model"] == "test-model" and metadata["temperature"] == 0
    assert metadata["prompt_version"] == "v1" and metadata["backend"] == "OPENJIUWEN"
    assert metadata["response_hash"] == metadata["provider_response_hash"] == digest(output)
    assert len(metadata["input_hash"]) == 64 and metadata["timestamp"] <= metadata["finished_at"]
    assert "test-credential" not in json.dumps(metadata)
    monkeypatch.delenv("QIYUAN_LLM_API_KEY")
    _, local = run_sync(PromptOptimizer(enabled=True).optimize(planner_prompt(), "repeat", ["replicate"], prompt_version="v1"))
    assert not local["provider_called"] and local["model"] == "none" and local["provider_response_hash"] is None
