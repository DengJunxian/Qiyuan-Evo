"""Package the first recorded experiment of each family, without selecting by score."""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package(source, destination):
    from core.competition_cockpit import ExperimentArchive
    from core.team_evolution.benchmark import benchmark_tasks
    from core.team_evolution.evidence_bundle import bundle_files
    from core.team_evolution.provenance import source_provenance
    from core.team_evolution.store import write_json
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty; existing reference experiments are immutable")
    archive = ExperimentArchive(source)
    comparisons = archive.comparisons()
    selected = [next(c for c in comparisons if c["baseline"]["task"]["task_type"] == t.task_type)
                for t in benchmark_tasks()]
    current = source_provenance()
    if any(c["baseline"]["runtime"]["source_hash"] != current["source_hash"] for c in selected):
        raise ValueError("Run all three tasks with the current source before packaging")
    destination.mkdir(parents=True, exist_ok=True)
    traces = {}
    for comparison in selected:
        relative = Path("comparisons") / comparison["baseline"]["run_id"] / "competition_demo_summary.json"
        manifest = archive.read(str(relative))
        for name, raw in bundle_files(source, manifest).items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        write_json(destination / relative, manifest)
        for mode in ("baseline", "candidate", "evolved"):
            trace = comparison.get(mode)
            if trace: traces[trace["run_id"]] = trace
    run_ids = set(traces)
    experiences = [r for r in archive.experiences() if r["experience_id"] in run_ids]
    write_json(destination / "before_after.json", selected)
    write_json(destination / "run_trace.json", list(traces.values()))
    write_json(destination / "experiences.json", experiences)
    write_json(destination / "evolution_history.json", [h for c in selected for h in c["evolution_history"]])
    write_json(destination / "source_manifest.json", current)
    write_json(destination / "summary.json", {"scope": "first recorded pair of each fixed task family",
               "experiments": [{"experiment_id": c["experiment_id"], "task_id": c["task_id"],
                                "seed": c["seed"], "evolution_gain": c["evolution_gain"]} for c in selected]})
    files = {str(p.relative_to(destination)): sha256(p.read_bytes()).hexdigest()
             for p in sorted(destination.rglob("*.json"))}
    write_json(destination / "reference_manifest.json", {"schema_version": "qiyuan_reference_v1",
               "selection": "first recorded pair per family; no score filtering", "source_hash": current["source_hash"],
               "experiment_ids": [c["experiment_id"] for c in selected], "files": files})
    check = ExperimentArchive(destination)
    assert len(check.comparisons()) == 3 and not check.issues
    return {"destination": str(destination), "experiments": 3, "files": len(files),
            "bytes": sum(p.stat().st_size for p in destination.rglob("*.json"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "benchmarks/competition/reference")
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.destination), ensure_ascii=False))
