"""Frozen competition tasks and evaluator rules, independent of team versions."""
from copy import deepcopy
import json
from pathlib import Path
from hashlib import sha256

from .models import TaskSpec

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "competition"
RULES = {"version": "competition_v1", "success_threshold": 0.85,
         "weights": {"completeness": 0.30, "evidence": 0.20, "consistency": 0.20,
                     "task_quality": 0.20, "reproducibility": 0.10}}


def load_dataset():
    import pandas as pd
    manifest = json.loads((FIXTURE_ROOT / "sse_2024_policy_window.manifest.json").read_text())
    path = FIXTURE_ROOT / manifest["file"]
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != manifest["sha256"]:
        raise ValueError("Frozen benchmark dataset hash mismatch")
    frame = pd.read_csv(path)
    if len(frame) != manifest["rows"] or frame["date"].duplicated().any():
        raise ValueError("Invalid benchmark dataset")
    return frame, manifest


def benchmark_tasks(seed=42):
    common = {"seed": seed, "evaluation_rules": deepcopy(RULES),
              "constraints": {"max_steps": 100, "max_intervention_cost": 1.0, "simulation_days": 12}}
    return [
        TaskSpec("policy_report_v1", "policy_report", "评估印花税下调对市场流动性、风险和交易行为的影响，形成证据报告。",
                 expected_outputs=["policy", "simulation", "risk", "control", "reproducibility"],
                 inputs={"policy_text": "证券交易印花税下调，降低交易税费，改善市场流动性。"}, **deepcopy(common)),
        TaskSpec("historical_analysis_v1", "historical_analysis", "回放 2024 年 9 月政策窗口，量化模拟路径相对上证指数观测走势的偏差。",
                 expected_outputs=["policy", "simulation", "risk", "historical_comparison", "quant", "reproducibility"],
                 inputs={"policy_text": "降准与降息，增加流动性供给，支持资本市场。"}, **deepcopy(common)),
        TaskSpec("regulatory_planning_v1", "regulatory_planning", "针对恐慌抛售场景设计预算内的监管干预，比较方案风险与干预成本。",
                 expected_outputs=["policy", "simulation", "risk", "counterfactual_search", "reproducibility"],
                 inputs={"policy_text": "负面谣言引发恐慌抛售，信用收紧。",
                         "interventions": [
                             {"name": "no_action", "policy_text": "", "cost": 0.0},
                             {"name": "clarification", "policy_text": "发布澄清公告，辟谣，稳定市场预期。", "cost": 0.2},
                             {"name": "liquidity_support", "policy_text": "注入流动性，国家队维稳，稳定市场。", "cost": 0.8}]},
                 **deepcopy(common)),
    ]


def get_task(task_id: str, seed=42) -> TaskSpec:
    for task in benchmark_tasks(seed):
        if task.task_id == task_id or task.task_type == task_id:
            return task
    raise ValueError(f"Unknown benchmark task: {task_id}")
