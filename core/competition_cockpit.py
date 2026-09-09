"""Presentation service: read verified experiments and run bounded real demos.

Streamlit consumes this module; it never constructs market results or scores.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import sqlite3
import threading
import uuid

from core.competition_service import CompetitionService
from core.team_evolution.backend import select_backend
from core.team_evolution.benchmark import benchmark_tasks, get_task
from core.team_evolution.config import baseline_config
from core.team_evolution.models import digest
from core.team_evolution.provenance import source_provenance
from core.team_evolution.store import now, write_json

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DemoProfile:
    name: str
    backend: str
    cloud_optimizer: bool
    description: str


PROFILES = {p.name: p for p in (
    DemoProfile("REPRODUCIBLE DEMO", "auto", False, "冻结数据与确定性角色；真实执行可用的 Workflow。"),
    DemoProfile("ONLINE LIVE", "auto", True, "实时执行；请求云端 Prompt 优化，缺密钥或服务故障时明确回退。"),
    DemoProfile("OFFLINE FALLBACK", "local", False, "本地 DAG 与确定性工具；不访问云端模型。"),
)}
DEMO_SCALES = {"答辩演示": 6, "完整实验": 12}
DEFAULT_COMPETITION_DEMO = {"task_type": "policy_report", "seed": 42, "scale": "答辩演示",
                            "profile": "REPRODUCIBLE DEMO",
                            "failure_case": "baseline lacks control and independent reproducibility evidence",
                            "evaluation_rules": "competition_v1"}


def demo_task(task_type="policy_report", *, scale="答辩演示", seed=42):
    task = get_task(task_type, seed)
    task.constraints["simulation_days"] = DEMO_SCALES[scale]
    return task


def reference_directory():
    configured = os.environ.get("QIYUAN_REFERENCE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = ROOT / "benchmarks" / "competition" / "reference"
    if (bundled / "reference_manifest.json").is_file():
        return bundled
    preferred = ROOT / "outputs" / "competition_cockpit_reference"
    return preferred if (preferred / "before_after.json").exists() else ROOT / "outputs" / "competition_benchmark"


def capabilities():
    config = baseline_config()
    roles = {k: asdict(v) for k, v in config.roles.items()}
    roles["evolution_controller"] = {"role_type": "evolution_controller", "implementation": "EvolutionController.propose"}
    mechanisms = {"prompt": "PromptOptimizer.optimize", "tool": "EvolutionController.propose: tool_routing",
                  "topology": "EvolutionController.propose: spawn_reroute / split_parallelize"}
    return {"roles": roles, "role_count": len({x["role_type"] for x in roles.values()}),
            "tasks": [t.to_dict() for t in benchmark_tasks()], "task_count": len(benchmark_tasks()),
            "mechanisms": mechanisms, "mechanism_count": len(mechanisms),
            "comparison": callable(CompetitionService.run_before_after), "profiles": [asdict(p) for p in PROFILES.values()]}


def validate_comparison(value):
    """Reject mismatched pairs before the UI can present a comparison."""
    a, b = value["baseline"], value["evolved"]
    if a["task"] != b["task"] or a["task"]["seed"] != value["seed"]:
        raise ValueError("Comparison task/seed mismatch")
    for key in ("dataset_hash", "source_hash"):
        if a["runtime"].get(key) != b["runtime"].get(key):
            raise ValueError(f"Comparison {key} mismatch")
    if value["dataset_hash"] != a["runtime"]["dataset_hash"] or value["model_config"] != a["task"]["model_config"]:
        raise ValueError("Comparison dataset/model mismatch")
    for trace in (a, b):
        if not trace["evaluation"] or not trace["agent_events"]:
            raise ValueError("Incomplete execution trace")
        from core.team_evolution.models import AgentTeamConfig, ExecutionTrace, TaskSpec
        from core.team_evolution.evaluation import evaluate
        AgentTeamConfig.from_dict(trace["team_config"]).validate()
        spec = TaskSpec(**trace["task"])
        if evaluate(spec, ExecutionTrace(**trace)) != trace["evaluation"]:
            raise ValueError("Artifact score does not agree with its ExecutionTrace")
    from core.team_evolution.evaluation import evolution_gain
    if value["evolution_gain"] != evolution_gain(a["evaluation"], b["evaluation"]):
        raise ValueError("Comparison gain does not agree with its measured scores")
    if value.get("experiment_id"):
        if any(t.get("experiment_id") != value["experiment_id"] for t in (a, b)):
            raise ValueError("Comparison experiment identity mismatch")
    return value


class ExperimentArchive:
    """Read-only access; opening archived evidence never rewrites its manifest."""
    def __init__(self, root=None):
        self.root = Path(root or reference_directory()).resolve()
        self.issues = []

    def read(self, name, default=None):
        path = self.root / name
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.issues.append(f"{name}: {type(exc).__name__}")
            return default

    def comparisons(self):
        manifest = self.read("reference_manifest.json")
        if (self.root / "reference_manifest.json").exists():
            try:
                if not manifest or not manifest["files"]:
                    raise ValueError("Missing reference manifest")
                for name, expected in manifest["files"].items():
                    path = (self.root / name).resolve()
                    if not path.is_relative_to(self.root) or sha256(path.read_bytes()).hexdigest() != expected:
                        raise ValueError("Reference content mismatch")
            except (OSError, KeyError, TypeError, ValueError):
                self.issues.append("内置案例校验失败，请重新运行或恢复案例文件。")
                return []
        values = self.read("before_after.json", [])
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            self.issues.append("before_after.json: invalid schema")
            return []
        for p in sorted((self.root / "comparisons").glob("*/before_after.json")):
            values += [self.read(str(p.relative_to(self.root)), {})]
        valid, seen = [], set()
        for value in values:
            try:
                identifier = value["baseline"]["run_id"]
                if identifier not in seen:
                    valid.append(validate_comparison(value))
                    seen.add(identifier)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
                self.issues.append(f"Comparison unavailable: {type(exc).__name__}")
        return sorted(valid, key=lambda c: c["baseline"]["agent_events"][0].get("started_at", ""))

    def table(self, name):
        if name not in {"runs", "experiences", "evolutions", "prompts"}:
            raise ValueError("Unknown archive table")
        db_path = self.root / "experience.sqlite3"
        if not db_path.is_file():
            return []
        try:
            with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
                return [json.loads(row[0]) for row in db.execute(f"SELECT payload FROM {name}")]
        except (sqlite3.Error, ValueError) as exc:
            self.issues.append(f"{name}: {type(exc).__name__}")
            return []

    def traces(self):
        values = self.table("runs") or self.read("run_trace.json", [])
        if not values:
            values = [c[m] for c in self.comparisons() for m in ("baseline", "candidate", "evolved") if c.get(m)]
        return {v["run_id"]: v for v in values if isinstance(v, dict) and "run_id" in v}

    def history(self):
        return self.table("evolutions") or self.read("evolution_history.json", [])

    def experiences(self):
        traces = self.traces()
        reused = Counter(eid for t in traces.values() for eid in t.get("plan", {}).get("retrieved_experience_ids", []))
        rows = self.table("experiences") or self.read("experiences.json", [])
        for row in rows:
            eid = row["experience_id"]
            row["times_reused"] = reused[eid]
            row["used_by"] = [t["run_id"] for t in traces.values() if eid in t.get("plan", {}).get("retrieved_experience_ids", [])]
        return rows

    def runtime(self, profile="REPRODUCIBLE DEMO"):
        try:
            info = select_backend(PROFILES[profile].backend).info()
        except Exception as exc:
            return {"backend": "UNAVAILABLE", "package_version": None, "sdk_imported": False, "reason": type(exc).__name__}
        info["profile"] = profile
        info["status"] = "READY" if info["sdk_imported"] else "FALLBACK"
        return info


def configuration_identity(config):
    return digest({"topology": config["topology"], "tool_policy": config["tool_policy"],
                   "prompts": {k: [v["prompt_version"], v["system_prompt"]] for k, v in config["roles"].items()}})


def generations(comparisons, task):
    """Only observed baseline/promoted generations for identical experiment input."""
    rows, seen = [], set()
    for comparison in comparisons:
        if comparison["baseline"]["task"] != task:
            continue
        for mode in ("baseline", "evolved"):
            trace = comparison[mode]
            key = configuration_identity(trace["team_config"])
            if key not in seen:
                seen.add(key)
                rows.append({"generation": len(rows), "config": trace["team_config"], "trace": trace,
                             "quality": trace["evaluation"]["quality_score"], "source": mode})
    return rows


class DemoJobs:
    """One real background job at a time; no Streamlit calls from worker threads."""
    def __init__(self, output_root=None):
        self.output_root = Path(output_root or ROOT / "outputs" / "competition_demos")
        self._jobs = {}
        self._lock = threading.RLock()

    def _persist(self, job):
        # Small durable index: observations and traces have their own canonical files.
        record = {k: v for k, v in job.items() if k not in {"observations", "result"}}
        write_json(self.output_root / ".jobs" / (job["job_id"] + ".json"), record)

    def start(self, *, task_type="policy_report", profile="REPRODUCIBLE DEMO", scale="答辩演示",
              action="compare", output_dir=None, saved_trace=None):
        if profile not in PROFILES or action not in {"compare", "baseline", "evolved", "reproduce"}:
            raise ValueError("Unknown demo profile/action")
        task = demo_task(task_type, scale=scale)
        if action == "reproduce":
            from core.team_evolution.models import TaskSpec
            if not saved_trace:
                raise ValueError("A saved ExecutionTrace is required for reproduction")
            task = TaskSpec(**deepcopy(saved_trace["task"]))
        job_id = "demo-" + uuid.uuid4().hex
        root = Path(output_dir) if output_dir else self.output_root / job_id
        with self._lock:
            if any(j["status"] in {"QUEUED", "RUNNING"} for j in self._jobs.values()):
                raise RuntimeError("Another experiment is already running")
            self._jobs[job_id] = {"job_id": job_id, "status": "QUEUED", "profile": profile,
                                  "scale": scale, "action": action, "task": task.to_dict(),
                                  "output_dir": str(root.resolve()), "observations": [], "result": None,
                                  "started_at": now(), "error": None}
            try:
                self._persist(self._jobs[job_id])
            except OSError:
                self._jobs.pop(job_id)
                raise
        thread = threading.Thread(target=self._run, args=(job_id, task, saved_trace), daemon=True)
        thread.start()
        return job_id

    def _run(self, job_id, task, saved_trace):
        def observe(item):
            with self._lock:
                self._jobs[job_id]["observations"].append(deepcopy(item))
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "RUNNING"
        try:
            self._persist(job)
            profile = PROFILES[job["profile"]]
            if saved_trace and saved_trace["runtime"].get("source_hash") != source_provenance()["source_hash"]:
                raise ValueError("Source revision differs from the saved trace; select a current artifact or run a fresh experiment")
            service = CompetitionService(job["output_dir"], backend=profile.backend,
                                         cloud_optimizer=profile.cloud_optimizer, observer=observe)
            if job["action"] == "reproduce":
                from core.team_evolution.models import ExecutionTrace
                service.store.save_run(ExecutionTrace(**saved_trace))
                result = service.rerun_execution(saved_trace["run_id"])
            else:
                fn = {"compare": service.run_before_after, "baseline": service.run_baseline, "evolved": service.run_evolved}[job["action"]]
                result = fn(task)
            # UI reads only after final artifacts have been committed.
            name = "before_after" if job["action"] == "compare" else "result"
            write_json(Path(job["output_dir"]) / (name + ".json"), result)
            write_json(Path(job["output_dir"]) / "evolution_history.json", service.get_evolution_history())
            write_json(Path(job["output_dir"]) / "event_stream.json", job["observations"])
            if job["action"] == "compare":
                manifest = Path(job["output_dir"]) / "comparisons" / result["baseline"]["run_id"] / "competition_demo_summary.json"
                write_json(Path(job["output_dir"]) / "competition_demo_summary.json", json.loads(manifest.read_text()))
            write_json(Path(job["output_dir"]) / "demo_manifest.json",
                       {k: job[k] for k in ("job_id", "profile", "scale", "action", "task", "started_at")} |
                       {"finished_at": now(), "runtime": service.get_openjiuwen_runtime_info(),
                        "experiment_id": result.get("experiment_id", result.get("rerun", {}).get("experiment_id")),
                        "source_hash": service.provenance["source_hash"]})
            with self._lock:
                if job["action"] == "compare":
                    result_path = str(Path("comparisons") / result["baseline"]["run_id"] / "before_after.json")
                elif job["action"] in {"baseline", "evolved"}:
                    result_path = str(Path("runs") / result["run_id"] / "run_trace.json")
                else:
                    result_path = job_id + "_reproduction.json"
                    write_json(Path(job["output_dir"]) / result_path, result)
                completed = job | {"status": "COMPLETED", "result": result, "finished_at": now(), "result_path": result_path}
                self._persist(completed)
                job.update(completed)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:300]}
            try:
                write_json(Path(job["output_dir"]) / "demo_failure.json", error)
            except OSError as storage_error:
                error["artifact_write_error"] = type(storage_error).__name__
            with self._lock:
                job.update(status="FAILED", error=error, finished_at=now())
                try:
                    self._persist(job)
                except OSError:
                    pass  # Original failure remains visible in this process.

    def snapshot(self, job_id):
        if not re.fullmatch(r"demo-[a-f0-9]{32}", job_id):
            raise KeyError("Invalid demo identifier")
        with self._lock:
            if job_id not in self._jobs:
                try:
                    record = json.loads((self.output_root / ".jobs" / (job_id + ".json")).read_text())
                    record.update(observations=[], result=None, recovered=True)
                    if record["status"] == "COMPLETED":
                        record["result"] = json.loads((Path(record["output_dir"]) / record["result_path"]).read_text())
                        if record["action"] == "compare":
                            validate_comparison(record["result"])
                    elif record["status"] in {"QUEUED", "RUNNING"}:
                        record.update(status="FAILED", error={"type": "InterruptedRun",
                                      "message": "服务进程曾中断；已保存的 trace 保留，请重新运行。"})
                    self._jobs[job_id] = record
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    raise KeyError("Demo record unavailable") from exc
            return deepcopy(self._jobs[job_id])
