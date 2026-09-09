"""Run exactly the cockpit demo through CompetitionService and save its observations."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["reproducible", "online", "offline"], default="reproducible")
    from core.competition_cockpit import DEFAULT_COMPETITION_DEMO
    parser.add_argument("--task", choices=["policy_report", "historical_analysis", "regulatory_planning"], default=DEFAULT_COMPETITION_DEMO["task_type"])
    parser.add_argument("--scale", choices=["demo", "full"], default="demo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from core.competition_cockpit import DemoJobs
    jobs = DemoJobs()
    job_id = jobs.start(task_type=args.task, profile={"reproducible": "REPRODUCIBLE DEMO", "online": "ONLINE LIVE", "offline": "OFFLINE FALLBACK"}[args.profile],
                        scale="答辩演示" if args.scale == "demo" else "完整实验", output_dir=args.output)
    phase = None
    while True:
        job = jobs.snapshot(job_id)
        event = job["observations"][-1]["event"] if job["observations"] else job["status"]
        if event != phase:
            print(event, flush=True)
            phase = event
        if job["status"] in {"COMPLETED", "FAILED"}:
            print(json.dumps({"status": job["status"], "output": job["output_dir"], "error": job["error"],
                              "gain": job["result"]["evolution_gain"] if job["result"] else None}, ensure_ascii=False))
            return 0 if job["status"] == "COMPLETED" else 1
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
