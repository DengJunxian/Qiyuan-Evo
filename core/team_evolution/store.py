"""Transactional experience, prompt, configuration and trace persistence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .models import AgentTeamConfig, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as f:
        f.write(raw + "\n")
        tmp = f.name
    os.replace(tmp, path)


def context_key(task, dataset_hash):
    return digest({"type": task.task_type, "dataset": dataset_hash, "model": task.model_config,
                   "constraints": task.constraints, "rules": task.evaluation_rules,
                   "expected_outputs": task.expected_outputs})


class ExperienceStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "experience.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experiences(id TEXT PRIMARY KEY, context TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS configurations(context TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evolutions(id TEXT PRIMARY KEY, context TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS prompts(id TEXT PRIMARY KEY, context TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def save_run(self, trace):
        payload = trace.to_dict()
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES (?,?)", (trace.run_id, json.dumps(payload, ensure_ascii=False, allow_nan=False)))
        write_json(self.root / "runs" / trace.run_id / "run_trace.json", payload)

    def get_run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(f"Unknown run: {run_id}")
        return json.loads(row[0])

    def remember(self, task, trace, key, *, eligible=False):
        e = {"experience_id": trace.run_id, "task_fingerprint": task.fingerprint, "task_type": task.task_type,
             "objective": task.objective, "context_key": key, "team_configuration": trace.team_config,
             "prompt_versions": {r: c["prompt_version"] for r, c in trace.team_config["roles"].items()},
             "tools_used": trace.tool_calls, "key_failures": trace.errors, "critic_feedback": trace.critiques,
             "final_score": trace.evaluation, "cost": {"token_usage": trace.token_usage, "currency": None},
             "latency": trace.latency, "eligible": eligible,
             "successful_strategy": trace.team_config if eligible and trace.evaluation["task_success"] else None}
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO experiences VALUES (?,?,?)", (trace.run_id, key, json.dumps(e, ensure_ascii=False, allow_nan=False)))

    def retrieve(self, task, key, limit=3):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM experiences WHERE context=?", (key,)).fetchall()
        values = [json.loads(row[0]) for row in rows]
        # Reuse only successful accepted strategies; failed traces remain available to optimizer.
        values = [x for x in values if x["successful_strategy"]]
        def rank(x):
            exact = x["task_fingerprint"] == task.fingerprint
            return (x["final_score"]["quality_score"], exact, x["team_configuration"]["version"])
        return sorted(values, key=rank, reverse=True)[:limit]

    def tool_statistics(self, key):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM experiences WHERE context=?", (key,)).fetchall()
        aggregates = {}
        for row in rows:
            exp = json.loads(row[0])
            for call in exp["tools_used"]:
                identity = call["agent"] + ":" + call["tool"]
                s = aggregates.setdefault(identity, {"calls": 0, "successes": 0, "quality_sum": 0.0,
                                                     "latency_sum": 0.0, "tokens": 0, "unknown_tokens": False})
                s["calls"] += 1
                s["successes"] += int(call["status"] == "ok")
                s["quality_sum"] += exp["final_score"]["quality_score"] if call["status"] == "ok" else 0
                s["latency_sum"] += call["latency"]
                s["tokens"] += call.get("token_usage") or 0
                s["unknown_tokens"] |= call.get("token_usage") is None
        return {k: {"calls": s["calls"], "success_rate": s["successes"] / s["calls"],
                    "average_quality": s["quality_sum"] / s["calls"],
                    "average_latency": s["latency_sum"] / s["calls"],
                    "average_token_cost": None if s["unknown_tokens"] else s["tokens"] / s["calls"],
                    "failure_count": s["calls"] - s["successes"]} for k, s in aggregates.items()}

    def current_config(self, key):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM configurations WHERE context=?", (key,)).fetchone()
        return AgentTeamConfig.from_dict(json.loads(row[0])) if row else None

    def commit_evolution(self, key, history, config, *, base_version, accepted):
        """Compare-and-swap promotion prevents concurrent candidates overwriting a newer team."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM configurations WHERE context=?", (key,)).fetchone()
            current = row[0] if row else 1
            if accepted and current != base_version:
                accepted = False
                history["promotion"]["reason"] = "configuration changed concurrently"
            history["promotion"]["accepted"] = accepted
            for action in history["actions"]:
                action["accepted"] = accepted
            if accepted:
                db.execute("INSERT OR REPLACE INTO configurations VALUES (?,?,?)",
                           (key, config.version, json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False)))
            db.execute("INSERT INTO evolutions VALUES (?,?,?)", (history["evolution_id"], key, json.dumps(history, ensure_ascii=False, allow_nan=False)))
            for action in history["actions"]:
                if action["action_type"] == "prompt_update":
                    db.execute("INSERT INTO prompts VALUES (?,?,?)", (action["evolution_id"], key, json.dumps(action, ensure_ascii=False, allow_nan=False)))
        write_json(self.root / "evolutions" / (history["evolution_id"] + ".json"), history)
        return accepted

    def history(self, key=None):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM evolutions" + (" WHERE context=?" if key else ""), (key,) if key else ()).fetchall()
        return [json.loads(row[0]) for row in rows]

    def prompt_registry(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM prompts")]
