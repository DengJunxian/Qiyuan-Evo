"""Manifest of one experiment, reusing canonical traces instead of copying them.

Artifact paths are relative to the CompetitionService store. JSON pointers select
the already-exported paired topology/history. Every file is content-addressed.
"""
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from .store import write_json


def export_evidence_bundle(root, target, comparison):
    root, target = Path(root).resolve(), Path(target).resolve()
    eid = comparison["experiment_id"]
    a, b = comparison["baseline"], comparison["evolved"]
    evaluation = {"experiment_id": eid, "rules": a["task"]["evaluation_rules"],
                  "baseline": a["evaluation"], "evolved": b["evaluation"],
                  "evolution_gain": comparison["evolution_gain"],
                  "candidate": comparison["candidate"]["evaluation"] if comparison.get("candidate") else None}
    write_json(target / "evaluation.json", evaluation)
    write_json(target / "prompt_diff.json", {"experiment_id": eid, "changes": [
        action for h in comparison["evolution_history"] for action in h["actions"]
        if action["action_type"] == "prompt_update"]})
    write_json(target / "benchmark_summary.json", {"experiment_id": eid, "task": a["task"],
               "baseline_run_id": a["run_id"], "evolved_run_id": b["run_id"],
               "baseline": a["evaluation"], "evolved": b["evaluation"],
               "promotion": comparison["promotion"], "scope": "one paired training-task experiment"})
    files = {"baseline_trace.json": (root / "runs" / a["run_id"] / "run_trace.json", ""),
             "evolved_trace.json": (root / "runs" / b["run_id"] / "run_trace.json", ""),
             "evaluation.json": (target / "evaluation.json", ""),
             "evolution_history.json": (target / "evolution_history.json", ""),
             "prompt_diff.json": (target / "prompt_diff.json", ""),
             "topology_before.json": (target / "team_topology.json", "/before"),
             "topology_after.json": (target / "team_topology.json", "/after"),
             "benchmark_summary.json": (target / "benchmark_summary.json", ""),
             "before_after.json": (target / "before_after.json", "")}
    for h in comparison["evolution_history"]:
        files[f'controller_{h["evolution_id"]}.json'] = (root / "runs" / h["evolution_id"] / "run_trace.json", "")
    manifest = {"schema_version": "qiyuan_evidence_bundle_v1", "experiment_id": eid,
                "task_id": comparison["task_id"], "seed": comparison["seed"],
                "dataset_hash": comparison["dataset_hash"], "source_hash": a["runtime"]["source_hash"],
                "artifact_base": "CompetitionService output directory",
                "artifacts": {name: {"path": str(path.relative_to(root)), "json_pointer": pointer,
                                     "sha256": sha256(path.read_bytes()).hexdigest()}
                              for name, (path, pointer) in files.items()}}
    write_json(target / "competition_demo_summary.json", manifest)
    return manifest


def bundle_files(root, manifest):
    """Validate content and containment before downloads or acceptance checks."""
    root = Path(root).resolve()
    files = {}
    for item in manifest["artifacts"].values():
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence path is outside its experiment store")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Evidence artifact hash mismatch")
        value = json.loads(raw)
        members = value if isinstance(value, list) else [value]
        if any(x.get("experiment_id") != manifest["experiment_id"] for x in members):
            raise ValueError("Evidence experiment identity mismatch")
        files[item["path"]] = raw
    return files


def bundle_zip(root, manifest):
    files = bundle_files(root, manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("competition_demo_summary.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for path, raw in files.items():
            archive.writestr(path, raw)
    return buffer.getvalue()
