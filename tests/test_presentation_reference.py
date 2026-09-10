from copy import deepcopy
from pathlib import Path
import shutil

from streamlit.testing.v1 import AppTest

from core.competition_cockpit import ExperimentArchive, reference_directory
from core.team_evolution.evidence_bundle import bundle_zip
from core.team_evolution.provenance import source_provenance
from ui.presentation import evidence_progress, outcome_summary

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks/competition/reference"


def test_bundled_cases_are_current_measured_pairs_with_downloadable_evidence(monkeypatch):
    monkeypatch.delenv("QIYUAN_REFERENCE_DIR", raising=False)
    assert reference_directory() == REFERENCE
    archive = ExperimentArchive()
    cases = archive.comparisons()
    assert len(cases) == 3 and not archive.issues
    assert {c["baseline"]["task"]["task_type"] for c in cases} == {"policy_report", "historical_analysis", "regulatory_planning"}
    current = source_provenance()["source_hash"]
    experiences = {e["experience_id"]: e for e in archive.experiences()}
    for case in cases:
        assert case["baseline"]["task"] == case["evolved"]["task"]
        assert case["evolved"]["runtime"]["source_hash"] == current
        assert case["evolved"]["runtime"]["workflow_completed"]
        assert case["evolved"]["runtime"]["backend"] == "OPENJIUWEN"
        for eid in case["evolved"]["plan"]["retrieved_experience_ids"]:
            assert experiences[eid]["times_reused"] >= 1
        manifest = archive.read(f'comparisons/{case["baseline"]["run_id"]}/competition_demo_summary.json')
        assert bundle_zip(REFERENCE, manifest).startswith(b"PK")


def test_display_uses_recorded_facts_and_retains_zero_effects():
    for case in ExperimentArchive(REFERENCE).comparisons():
        trace = case["evolved"]
        untouched = deepcopy(trace)
        complete, total = evidence_progress(trace)
        assert complete == total == len(trace["task"]["expected_outputs"])
        summary = outcome_summary(trace)
        if trace["task"]["task_type"] == "historical_analysis":
            actual = trace["artifacts"]["historical_comparison"]["data"]["direction_consistency"]
            assert f"{actual:.1%}" in summary
        else:
            assert "0.00" in summary
        assert trace == untouched


def test_corrupt_packaged_case_is_not_displayed(tmp_path):
    destination = tmp_path / "reference"
    shutil.copytree(REFERENCE, destination)
    with (destination / "before_after.json").open("a") as output:
        output.write(" ")
    archive = ExperimentArchive(destination)
    assert archive.comparisons() == [] and archive.issues


def test_cockpit_deep_link_needs_neither_outputs_nor_online_runtime(monkeypatch):
    monkeypatch.delenv("QIYUAN_REFERENCE_DIR", raising=False)
    def unavailable(*args, **kwargs): raise ConnectionError("offline")
    monkeypatch.setattr(ExperimentArchive, "runtime", unavailable)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.query_params["page"] = "自演进驾驶舱"
    app.run()
    assert not app.exception
    assert app.session_state["ev_scale"] == "完整实验"
    assert not app.caption  # No developer narration under the home modules.
    text = " ".join(item.value for item in app.get("html"))
    assert "3/5" in text and "5/5" in text
    assert "AGENT EVOLUTION COCKPIT" not in text and "QUALITY" not in text
    assert app.button(key="ev_to_collaboration").label == "开始讲解：团队如何协作"
    app.button(key="ev_to_collaboration").click().run()
    assert not app.exception
    assert app.radio(key="ev_trace_phase").options == ["初始团队", "改进后团队", "候选配置复评"]


def test_running_case_does_not_inherit_previous_completed_label(monkeypatch):
    from ui import team_evolution
    task = ExperimentArchive(REFERENCE).comparisons()[1]["baseline"]["task"]
    job = {"job_id":"demo-"+"a"*32, "status":"RUNNING", "profile":"OFFLINE FALLBACK",
           "scale":"完整实验", "task":task, "action":"compare", "observations":[]}
    class RunningJobs:
        def snapshot(self, _): return deepcopy(job)
    monkeypatch.setattr(team_evolution, "get_jobs", lambda: RunningJobs())
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.query_params["demo"] = job["job_id"]
    app.run()
    assert not app.exception
    text = " ".join(item.value for item in app.get("html"))
    assert "已完成实验" not in text
    assert "本地备用工作流" in text and "离线备用" in text
    assert any("2024 年政策窗口" in x.value for x in app.markdown)
    assert app.button(key="ev_demo").disabled
