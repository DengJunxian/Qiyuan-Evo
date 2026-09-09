"""Public competition service. All UI/CLI data comes from measured executions."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import threading
import uuid

from core.team_evolution.backend import LocalFallbackBackend, select_backend
from core.team_evolution.benchmark import benchmark_tasks, get_task, load_dataset
from core.team_evolution.config import baseline_config
from core.team_evolution.evaluation import evolution_gain
from core.team_evolution.evolution import EvolutionController
from core.team_evolution.models import AgentTeamConfig, BeforeAfterComparison, ExecutionTrace, EvolutionSnapshot, TaskSpec, digest
from core.team_evolution.runtime import TeamRuntime
from core.team_evolution.provenance import source_provenance
from core.team_evolution.store import ExperienceStore, context_key, write_json, now


def run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Also usable inside notebook/Streamlit environments with an active event loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class CompetitionService:
    def __init__(self, output_dir=None, *, backend="auto", cloud_optimizer=False, observer=None):
        root = output_dir or Path(__file__).resolve().parents[1] / "outputs" / "competition_benchmark"
        self.store = ExperienceStore(root)
        self.backend = select_backend(backend)
        self.controller = EvolutionController(cloud_optimizer=cloud_optimizer)
        self.cloud_optimizer = cloud_optimizer
        self.observer = observer
        self.lock = threading.RLock()
        _, self.dataset = load_dataset()
        self.provenance = source_provenance()
        write_json(self.store.root / "source_manifest.json", self.provenance)

    def observe(self, event, trace=None, **detail):
        """Publish immutable observations; presentation failures never affect execution."""
        if self.observer is not None:
            try:
                self.observer({"event": event, "timestamp": now(),
                               "trace": trace.to_dict() if trace is not None else None,
                               **deepcopy(detail)})
            except Exception:
                pass

    def _task(self, task, seed=None):
        if isinstance(task, str):
            return get_task(task, 42 if seed is None else seed)
        data = task.to_dict() if isinstance(task, TaskSpec) else deepcopy(task)
        if seed is not None:
            data["seed"] = seed
        spec = TaskSpec(**data)
        if spec.dataset_id != self.dataset["dataset_id"]:
            raise ValueError("Unknown dataset: register and freeze it before running")
        if spec.model_config.get("mode") != "deterministic":
            raise ValueError("Current task roles use deterministic tools; cloud models are supported for prompt optimization only")
        return spec

    def _key(self, task):
        return digest({"task_context": context_key(task, self.dataset["sha256"]),
                       "source_hash": self.provenance["source_hash"]})

    async def _run(self, task, config, mode, experience_ids=(), *, experiment_id=None):
        config.validate()
        run_id = "run-" + uuid.uuid4().hex
        run_dir = self.store.root / "runs" / run_id
        run_dir.mkdir(parents=True)
        runtime = self.backend.info() | {"dataset_hash": self.dataset["sha256"],
                                        "source_hash": self.provenance["source_hash"],
                                        "python": self.provenance["python"],
                                        "prompt_optimizer_requested": self.cloud_optimizer,
                                        "baseline_experience_reads": 0 if mode == "baseline" else None}
        trace = ExecutionTrace(run_id, task.task_id, task.to_dict(), config.to_dict(), runtime, mode)
        trace.experiment_id = experiment_id or "exp-" + uuid.uuid4().hex
        self.observe("run_started", trace)
        result = await TeamRuntime(task, config, trace, run_dir, experience_ids=experience_ids,
                                   observer=self.observe).run(self.backend)
        self.store.save_run(result)
        self.store.remember(task, result, self._key(task), eligible=mode == "evolved" and config.version > 1)
        self.observe("run_saved", result)
        return result

    def _select_evolved(self, task):
        key = self._key(task)
        experiences = self.store.retrieve(task, key)
        current = self.store.current_config(key)
        if experiences:
            chosen = AgentTeamConfig.from_dict(experiences[0]["successful_strategy"])
            # Use current promoted version when tied; older experiences cannot roll it back.
            config = current if current and current.version >= chosen.version else chosen
        else:
            config = current or baseline_config()
        config.baseline = False
        return config, [x["experience_id"] for x in experiences]

    def run_competition_task(self, task, *, baseline_mode=False, evolved_mode=False, seed=None):
        if baseline_mode and evolved_mode:
            raise ValueError("baseline_mode and evolved_mode are mutually exclusive")
        return self.run_baseline(task, seed=seed) if baseline_mode else self.run_evolved(task, seed=seed)

    def run_baseline(self, task, *, seed=None):
        with self.lock:
            spec = self._task(task, seed)
            return run_sync(self._run(spec, baseline_config(), "baseline")).to_dict()

    def run_evolved(self, task, *, seed=None):
        with self.lock:
            spec = self._task(task, seed)
            config, experiences = self._select_evolved(spec)
            return run_sync(self._run(spec, config, "evolved", experiences)).to_dict()

    async def _compare(self, task):
        experiment_id = "exp-" + uuid.uuid4().hex
        baseline = await self._run(task, baseline_config(), "baseline", experiment_id=experiment_id)
        self.observe("baseline_evaluated", baseline)
        config, experiences = self._select_evolved(task)
        incumbent = baseline if config.version == 1 else await self._run(task, config, "evolved", experiences, experiment_id=experiment_id)
        controller_started = now()
        self.observe("evolution_started", incumbent, controller={"agent": "evolution_controller",
                     "status": "running", "started_at": controller_started,
                     "source_trace_ids": [incumbent.run_id]})
        proposed, history = await self.controller.propose(task, incumbent, self.store.tool_statistics(self._key(task)))
        history["experiment_id"] = experiment_id
        history["controller_trace"].update(started_at=controller_started, finished_at=now())
        self.observe("candidate_proposed", incumbent, history=history, candidate_config=proposed.to_dict())
        candidate = await self._run(task, proposed, "candidate", experiences, experiment_id=experiment_id) if history["actions"] else None
        a, b = incumbent.evaluation, candidate.evaluation if candidate else incumbent.evaluation
        accepted = bool(candidate and not candidate.errors and b["quality_score"] > a["quality_score"] + 1e-9
                        and b["evidence_coverage"] >= a["evidence_coverage"]
                        and b["critic_pass_rate"] >= a["critic_pass_rate"]
                        and b["task_success"] >= a["task_success"])
        history["candidate_run_id"] = candidate.run_id if candidate else None
        history["evaluation_before"], history["evaluation_after"] = a, b
        history["promotion"] = {"accepted": accepted,
                                "rule": "strict quality gain; no coverage/critic/success regression; no execution errors",
                                "reason": "candidate improved measured score" if accepted else "no qualifying measured improvement",
                                "comparison_scope": "same task, seed, dataset and role-model configuration; training-task comparison"}
        history["training_cost"] = {"candidate_tool_calls": b["tool_calls"] if candidate else 0,
                                    "candidate_latency": b["latency"] if candidate else 0,
                                    "optimizer_token_usage": history["optimizer"]["token_usage"],
                                    "controller_latency": history["controller_trace"]["latency"]}
        for action in history["actions"]:
            action["evaluation_before"], action["evaluation_after"] = a, b
            action["source_trace_ids"] = [incumbent.run_id] + ([candidate.run_id] if candidate else [])
        accepted = self.store.commit_evolution(self._key(task), history, proposed,
                                               base_version=config.version, accepted=accepted)
        # Evolution is a real executed controller, with its own ExecutionTrace.
        # Its training overhead is not folded into the baseline/evolved task score.
        controller_event = history["controller_trace"]
        evo_trace = ExecutionTrace(history["evolution_id"], task.task_id, task.to_dict(), proposed.to_dict(),
                                   LocalFallbackBackend("application-level evolution controller").info()
                                   | {"prompt_optimizer": history["optimizer"]}, "evolution")
        evo_trace.agent_events = [controller_event]
        evo_trace.experiment_id = experiment_id
        evo_trace.tool_calls = [{"tool": "prompt_optimization", "agent": "evolution_controller",
                                "status": "completed", "backend": history["optimizer"]["backend"],
                                "token_usage": history["optimizer"]["token_usage"],
                                "latency": history["optimizer"]["latency"]}] if any(
                                    a["action_type"] == "prompt_update" for a in history["actions"]) else []
        evo_trace.artifacts = {"candidate_config": proposed.to_dict(), "promotion": history["promotion"],
                               "evolution_actions": history["actions"]}
        evo_trace.token_usage = history["optimizer"]["token_usage"]
        evo_trace.latency = controller_event["latency"]
        self.store.save_run(evo_trace)
        self.observe("evolution_decided", evo_trace, history=history)
        if accepted:
            self.store.remember(task, candidate, self._key(task), eligible=True)
            # Execute the NEXT run from persisted config, proving promotion changes runtime.
            final_config, experience_ids = self._select_evolved(task)
            evolved = await self._run(task, final_config, "evolved", experience_ids, experiment_id=experiment_id)
        elif config.version == 1:
            evolved = await self._run(task, config, "evolved", experiences, experiment_id=experiment_id)
        else:
            evolved = incumbent
        snapshot = EvolutionSnapshot(evolved.team_config["version"],
                                     {r: s["prompt_version"] for r, s in evolved.team_config["roles"].items()},
                                     evolved.team_config["tool_policy"], evolved.team_config["topology"], evolved.evaluation)
        comparison = BeforeAfterComparison(task.task_id, task.seed, self.dataset["sha256"], task.model_config,
                                           baseline.to_dict(), evolved.to_dict(), evolution_gain(baseline.evaluation, evolved.evaluation),
                                           [history], candidate.to_dict() if candidate else None, history["promotion"], experiment_id).to_dict()
        target = self.store.root / "comparisons" / baseline.run_id
        write_json(target / "before_after.json", comparison)
        write_json(target / "evolution_snapshot.json", snapshot.to_dict())
        write_json(target / "evolution_history.json", [history])
        write_json(target / "team_topology.json", {"experiment_id": experiment_id, "before": baseline.team_config["topology"], "after": evolved.team_config["topology"]})
        from core.team_evolution.evidence_bundle import export_evidence_bundle
        export_evidence_bundle(self.store.root, target, comparison)
        self.observe("comparison_saved", evolved, artifact=str(target / "before_after.json"), promotion=history["promotion"])
        return comparison

    def run_before_after(self, task, *, seed=None):
        with self.lock:
            return run_sync(self._compare(self._task(task, seed)))

    def get_execution_trace(self, run_id):
        return self.store.get_run(run_id)

    def rerun_execution(self, run_id):
        """Replay an immutable historical configuration, not today's evolved config."""
        with self.lock:
            saved = self.get_execution_trace(run_id)
            if saved["mode"] == "evolution":
                raise ValueError("Evolution decisions are reviewed through history; rerun a task execution instead")
            task = self._task(saved["task"])
            if saved["runtime"].get("source_hash") != self.provenance["source_hash"]:
                raise ValueError("Source/configuration changed since this run; rerun from the recorded source revision")
            config = AgentTeamConfig.from_dict(saved["team_config"])
            trace = run_sync(self._run(task, config, "replay"))
            return {"original_run_id": run_id, "rerun": trace.to_dict(),
                    "semantic_match": trace.semantic_hash == saved["semantic_hash"]}

    def get_team_topology(self, run_id):
        return self.get_execution_trace(run_id)["team_config"]["topology"]

    def get_evolution_history(self):
        return self.store.history()

    def get_openjiuwen_runtime_info(self):
        import os
        key = os.environ.get("QIYUAN_LLM_API_KEY", "").strip()
        return self.backend.info() | {"cloud_prompt_optimizer_enabled": self.cloud_optimizer,
                                      "optimizer_api_key_available": bool(key and not key.startswith("your_")),
                                      "runtime_note": "workflow_invoked/completed are recorded per actual run"}

    def get_benchmark_summary(self):
        import json
        path = self.store.root / "summary.json"
        return json.loads(path.read_text()) if path.exists() else {"status": "not_run", "tasks": []}

    def run_benchmark(self, *, seed=42, rounds=3, progress=None):
        if not 1 <= rounds <= 10:
            raise ValueError("rounds must be between 1 and 10")
        comparisons, rows = [], []
        for task in benchmark_tasks(seed):
            for round_index in range(1, rounds + 1):
                if progress:
                    progress(task.task_id, round_index)
                result = self.run_before_after(task)
                comparisons.append(result)
                rows.append({"task_id": task.task_id, "round": round_index,
                             "experiment_id": result["experiment_id"],
                             "baseline": result["baseline"]["evaluation"], "evolved": result["evolved"]["evaluation"],
                             "gain": result["evolution_gain"], "promotion": result["promotion"],
                             "baseline_run_id": result["baseline"]["run_id"], "evolved_run_id": result["evolved"]["run_id"],
                             "team_version": result["evolved"]["team_config"]["version"],
                             "experience_ids": result["evolved"]["plan"].get("retrieved_experience_ids", [])})
        traces = {x[mode]["run_id"]: x[mode] for x in comparisons for mode in ("baseline", "evolved", "candidate") if x[mode]}
        controller_traces = [self.store.get_run(h["evolution_id"]) for x in comparisons for h in x["evolution_history"]]
        summary = {"schema_version": "qiyuan_benchmark_v1", "seed": seed, "rounds": rounds,
                   "dataset": self.dataset, "runtime": self.get_openjiuwen_runtime_info(), "tasks": rows,
                   "source_hash": self.provenance["source_hash"],
                   "measured_backends": sorted({x["runtime"]["backend"] for x in traces.values()}),
                   "all_evolved_tasks_successful": all(r["evolved"]["task_success"] for r in rows),
                   "comparison_scope": "training tasks; successive reuse does not establish held-out generalization",
                   "reproducibility_scope": "semantic hashes and financial artifacts; wall clock/UUIDs deliberately vary"}
        for name, payload in {"summary": summary, "run_trace": list(traces.values()),
                              "evolution_history": self.store.history(), "before_after": comparisons,
                              "evolution_trace": controller_traces,
                              "team_topology": [{"task_id": x["task_id"], "before": x["baseline"]["team_config"]["topology"],
                                                 "after": x["evolved"]["team_config"]["topology"]} for x in comparisons],
                              "prompt_registry": self.store.prompt_registry()}.items():
            write_json(self.store.root / (name + ".json"), payload)
        return summary
