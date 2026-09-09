"""Versioned JSON contracts shared by orchestration, evaluation and the UI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any


def digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class JSONModel:
    def to_dict(self) -> dict:
        result = asdict(self)
        # Fail on invalid metrics instead of emitting JavaScript-incompatible NaN.
        json.dumps(result, allow_nan=False)
        return result


@dataclass
class TaskSpec(JSONModel):
    task_id: str
    task_type: str
    objective: str
    constraints: dict
    expected_outputs: list[str]
    evaluation_rules: dict
    seed: int = 42
    dataset_id: str = "sse_2024_policy_window_v1"
    model_config: dict = field(default_factory=lambda: {"mode": "deterministic", "model": "none"})
    inputs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.task_type not in {"policy_report", "historical_analysis", "regulatory_planning"}:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        if not self.objective.strip() or not 0 <= self.seed < 2**32:
            raise ValueError("Objective and a valid deterministic seed are required")
        if not self.expected_outputs or len(set(self.expected_outputs)) != len(self.expected_outputs):
            raise ValueError("Expected outputs must be nonempty and unique")
        weights = self.evaluation_rules.get("weights", {})
        names = {"completeness", "evidence", "consistency", "task_quality", "reproducibility"}
        if set(weights) != names or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in weights.values()):
            raise ValueError("Invalid evaluation weights")
        if abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError("Evaluation weights must sum to one")
        if not 0 <= self.evaluation_rules.get("success_threshold", -1) <= 1:
            raise ValueError("Invalid success threshold")
        if not 4 <= int(self.constraints.get("simulation_days", 12)) <= 60:
            raise ValueError("simulation_days must be between 4 and 60")
        if any(k.lower() in {"api_key", "authorization", "password", "secret", "access_token"} for k in self.model_config):
            raise ValueError("Credentials belong in the environment, never serialized task configuration")

    @property
    def fingerprint(self) -> str:
        data = self.to_dict()
        data.pop("task_id")
        return digest(data)


@dataclass
class AgentRoleSpec(JSONModel):
    role_id: str
    role_type: str
    prompt_version: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    model_config: dict = field(default_factory=lambda: {"mode": "deterministic"})
    enabled: bool = True
    performance_history: list[dict] = field(default_factory=list)


@dataclass
class TaskPlan(JSONModel):
    subtasks: list[dict]
    dependency_graph: dict[str, list[str]]
    assignments: dict[str, str]
    expected_evidence: list[str]
    stop_conditions: dict
    decision_summary: str = ""
    retrieved_experience_ids: list[str] = field(default_factory=list)


@dataclass
class AgentMessage(JSONModel):
    sender: str
    receiver: str
    message_type: str
    task_id: str
    payload_summary: dict
    timestamp: str


@dataclass
class Critique(JSONModel):
    dimension: str
    problem: str
    evidence: list[str]
    severity: str
    attribution: str
    recommendation: str


@dataclass
class TeamTopology(JSONModel):
    active_agents: list[str]
    communication_edges: list[list[str]]
    routing: dict[str, list[str]]
    scheduling_policy: str = "serial"

    def layers(self) -> list[list[str]]:
        remaining, done, layers = set(self.active_agents), set(), []
        if len(remaining) != len(self.active_agents):
            raise ValueError("Duplicate active agent")
        if any(a not in remaining or b not in remaining or a == b
               for a, b in self.communication_edges):
            raise ValueError("Invalid communication edge")
        while remaining:
            ready = sorted(a for a in remaining if all(
                source in done for source, target in self.communication_edges if target == a))
            if not ready:
                raise ValueError("Team topology contains a cycle")
            layers.append(ready)
            remaining.difference_update(ready)
            done.update(ready)
        return layers


@dataclass
class AgentTeamConfig(JSONModel):
    version: int
    roles: dict[str, AgentRoleSpec]
    topology: TeamTopology
    tool_policy: dict
    baseline: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTeamConfig":
        from copy import deepcopy
        # Candidate edits must never mutate the immutable source trace/snapshot.
        data = deepcopy(data)
        return cls(version=data["version"], roles={k: AgentRoleSpec(**v) for k, v in data["roles"].items()},
                   topology=TeamTopology(**data["topology"]), tool_policy=data["tool_policy"],
                   baseline=data.get("baseline", False))

    def validate(self):
        self.topology.layers()
        active = self.topology.active_agents
        if not {"planner", "executor", "critic", "reporter"}.issubset(active):
            raise ValueError("Planner, executor, critic and reporter are mandatory")
        if any(a not in self.roles or not self.roles[a].enabled for a in active):
            raise ValueError("Active role missing or disabled")


@dataclass
class CompetitionScorecard(JSONModel):
    task_success: bool
    quality_score: float
    evidence_coverage: float
    critic_pass_rate: float
    reproducibility: float | None
    steps: int
    retries: int
    tool_calls: int
    token_usage: int | None
    latency: float
    active_agent_count: int
    dimensions: dict = field(default_factory=dict)


@dataclass
class ExecutionTrace(JSONModel):
    run_id: str
    task_id: str
    task: dict
    team_config: dict
    runtime: dict
    mode: str
    agent_events: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    retries: int = 0
    token_usage: int | None = 0
    latency: float = 0.0
    evaluation: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    critiques: list[dict] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    semantic_hash: str = ""
    experiment_id: str = ""


@dataclass
class EvolutionAction(JSONModel):
    evolution_id: str
    target: str
    action_type: str
    before: Any
    after: Any
    rationale: str
    source_trace_ids: list[str]
    expected_gain: str
    accepted: bool | None = None
    evaluation_before: dict = field(default_factory=dict)
    evaluation_after: dict = field(default_factory=dict)
    diff: str = ""


@dataclass
class EvolutionSnapshot(JSONModel):
    version: int
    prompt_versions: dict
    tool_policy: dict
    topology: dict
    benchmark_metrics: dict


@dataclass
class BeforeAfterComparison(JSONModel):
    task_id: str
    seed: int
    dataset_hash: str
    model_config: dict
    baseline: dict
    evolved: dict
    evolution_gain: dict
    evolution_history: list[dict]
    candidate: dict | None = None
    promotion: dict = field(default_factory=dict)
    experiment_id: str = ""
