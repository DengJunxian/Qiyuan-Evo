"""Reproduce the Qiyuan team-evolution benchmark using real service calls."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "competition_benchmark")
    parser.add_argument("--backend", choices=["auto", "local", "openjiuwen"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3, help="rounds per task for --task all --mode compare")
    parser.add_argument("--task", choices=["all", "policy_report", "historical_analysis", "regulatory_planning"], default="all")
    parser.add_argument("--mode", choices=["baseline", "evolved", "compare"], default="compare")
    parser.add_argument("--cloud-optimizer", action="store_true")
    args = parser.parse_args()
    from core.competition_service import CompetitionService
    from core.team_evolution.store import write_json
    service = CompetitionService(args.output, backend=args.backend, cloud_optimizer=args.cloud_optimizer)
    if not 1 <= args.rounds <= 10:
        parser.error("--rounds must be between 1 and 10")
    if args.task != "all":
        method = {"baseline": service.run_baseline, "evolved": service.run_evolved, "compare": service.run_before_after}[args.mode]
        result = method(args.task, seed=args.seed)
        write_json(args.output / (args.task + "_" + args.mode + ".json"), result)
        print(json.dumps(result.get("evolution_gain", result.get("evaluation")), ensure_ascii=False))
        return int(not result["evolved"]["evaluation"]["task_success"]) if args.mode == "compare" else int(bool(result["errors"]))
    elif args.mode == "compare":
        result = service.run_benchmark(seed=args.seed, rounds=args.rounds,
                                       progress=lambda task, r: print(f"Running {task}, round {r}", flush=True))
        print(json.dumps({"output": str(args.output.resolve()), "backend": result["measured_backends"],
                          "all_successful": result["all_evolved_tasks_successful"],
                          "comparisons": len(result["tasks"])}, ensure_ascii=False))
        return int(not result["all_evolved_tasks_successful"])
    else:
        from core.team_evolution.benchmark import benchmark_tasks
        method = service.run_baseline if args.mode == "baseline" else service.run_evolved
        traces = [method(task) for task in benchmark_tasks(args.seed)]
        result = {"mode": args.mode, "seed": args.seed,
                  "runs": [{"task_id": t["task_id"], "run_id": t["run_id"],
                            "experiment_id": t["experiment_id"], "evaluation": t["evaluation"]} for t in traces]}
        write_json(args.output / ("all_" + args.mode + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
        return int(any(t["errors"] for t in traces))


if __name__ == "__main__":
    raise SystemExit(main())
