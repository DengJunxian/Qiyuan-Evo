"""Verify measured competition evidence without rerunning or changing experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(root):
    from core.competition_cockpit import ExperimentArchive
    from core.team_evolution.backend import installed_version
    from core.team_evolution.evidence_bundle import bundle_files
    from core.team_evolution.models import digest
    from core.team_evolution.provenance import source_provenance
    from core.team_evolution.store import now
    archive = ExperimentArchive(root)
    pairs = archive.comparisons()
    checks = {}
    checks["valid_comparisons"] = bool(pairs) and not archive.issues
    groups = defaultdict(list)
    for c in pairs:
        groups[c["baseline"]["task"]["task_type"]].append(c)
    checks["three_task_families"] = set(groups) == {"policy_report", "historical_analysis", "regulatory_planning"}
    traces = [c[m] for c in pairs for m in ("baseline", "evolved")]
    checks["source_matches_current"] = all(t["runtime"]["source_hash"] == source_provenance()["source_hash"] for t in traces)
    checks["native_sdk_actually_executed"] = all(t["runtime"]["backend"] == "OPENJIUWEN" and
        t["runtime"]["workflow_invoked"] and t["runtime"]["workflow_completed"] and
        t["runtime"]["package_version"] == installed_version() for t in traces)
    checks["declared_agents_executed"] = all(
        set(t["team_config"]["topology"]["active_agents"]) == {e["agent"] for e in t["agent_events"] if e["status"] == "completed"}
        and len({e["role_type"] for e in t["agent_events"]}) >= 3 for t in traces)
    checks["plan_and_actual_handoffs"] = all(t["plan"]["subtasks"] and t["plan"]["assignments"] and
        {tuple(e) for e in t["team_config"]["topology"]["communication_edges"]} <=
        {(m["sender"], m["receiver"]) for m in t["messages"]} for t in traces)
    checks["baseline_memory_isolation"] = all(not c["baseline"]["plan"]["retrieved_experience_ids"] for c in pairs)
    checks["next_run_uses_prompt_content"] = all(
        next(e for e in c["evolved"]["agent_events"] if e["agent"] == "planner")["prompt_hash"] ==
        digest(c["evolved"]["team_config"]["roles"]["planner"]["system_prompt"]) and
        c["baseline"]["team_config"]["roles"]["planner"]["system_prompt"] != c["evolved"]["team_config"]["roles"]["planner"]["system_prompt"]
        for c in pairs)
    checks["actual_tool_routing_changed"] = any(
        {x["tool"] for x in c["baseline"]["tool_calls"] if x["agent"] == "executor"} !=
        {x["tool"] for x in c["evolved"]["tool_calls"] if x["agent"] == "executor"} for c in pairs)
    checks["actual_topology_and_scheduling_changed"] = all(c["baseline"]["team_config"]["topology"] != c["evolved"]["team_config"]["topology"] for c in pairs) and any(
        c["evolved"]["team_config"]["topology"]["scheduling_policy"] == "parallel" for c in pairs)
    checks["successful_experience_retrieved"] = all(c["evolved"]["plan"]["retrieved_experience_ids"] for c in pairs)
    checks["reports_preserve_evidence_facts"] = all(claim["facts"] == t["artifacts"][claim["evidence"][0]]["data"] and
        {"conclusion", "evidence", "confidence", "caveat", "source"} <= claim.keys() for t in traces for claim in t["report"]["conclusions"])
    checks["evolved_success_and_independent_replication"] = all(c["evolved"]["evaluation"]["task_success"] and
        c["evolved"]["artifacts"]["reproducibility"]["data"]["matched"] and
        c["evolved"]["artifacts"]["reproducibility"]["data"]["independent_execution"] for c in pairs)
    checks["repeat_semantic_stability"] = all(len({c[m]["semantic_hash"] for c in group}) == 1
        for group in groups.values() for m in ("baseline", "evolved"))
    manifest_checks = []
    controller_checks = []
    for c in pairs:
        manifest = archive.read(f'comparisons/{c["baseline"]["run_id"]}/competition_demo_summary.json')
        try:
            manifest_checks.append(bool(bundle_files(archive.root, manifest)))
        except (TypeError, KeyError, OSError, ValueError):
            manifest_checks.append(False)
        for h in c["evolution_history"]:
            trace = archive.read(f'runs/{h["evolution_id"]}/run_trace.json', {})
            controller_checks.append(trace.get("experiment_id") == c["experiment_id"] and
                any(e["agent"] == "evolution_controller" and e["status"] == "completed" for e in trace.get("agent_events", [])))
    checks["bundle_identity_and_hashes"] = bool(manifest_checks) and all(manifest_checks)
    checks["evolution_controller_has_own_trace"] = bool(controller_checks) and all(controller_checks)
    return {"timestamp": now(), "ready": all(checks.values()), "checks": checks, "comparisons": len(pairs),
            "source_hash": source_provenance()["source_hash"], "archive_issues": archive.issues,
            "experiments": [{"experiment_id": c["experiment_id"], "task_id": c["task_id"],
                             "baseline_run_id": c["baseline"]["run_id"], "evolved_run_id": c["evolved"]["run_id"]} for c in pairs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "competition_readiness_final")
    args = parser.parse_args()
    from core.team_evolution.store import write_json
    result = verify(args.input)
    write_json(args.input / "evidence_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not result["ready"])


if __name__ == "__main__":
    raise SystemExit(main())
