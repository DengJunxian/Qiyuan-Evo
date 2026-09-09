"""Fixed, transparent scoring of artifacts and traces; no mode-dependent rewards."""
from __future__ import annotations

import math
from .models import CompetitionScorecard, Critique, digest

REPAIRS = {"control": "control", "reproducibility": "replicate",
           "historical_comparison": "historical_comparison", "quant": "historical_comparison",
           "counterfactual_search": "counterfactual_search"}


def valid_evidence(name, item):
    try:
        data = item["data"]
        if not item["source"] or not data or item["content_hash"] != digest(data):
            return False
        if name == "simulation":
            return len(data["prices"]) >= 2 and all(math.isfinite(p) and p > 0 for p in data["prices"])
        if name == "risk":
            return data["status"] == "ok" and data["sample_size"] >= 2
        if name == "policy":
            return bool(data["event"]["raw_text"] and data["channels"])
        if name == "reproducibility":
            return data["matched"] and data["first_hash"] == data["replicate_hash"] and data["independent_execution"]
        if name == "historical_comparison":
            return data["observations"] >= 2 and data["source"]["kind"] == "observed_market_data"
        if name == "counterfactual_search":
            return len(data["plans"]) >= 3 and data["constraint_satisfaction"] and data["recommended"] in [x["name"] for x in data["plans"]]
        if name == "control":
            return len(data["prices"]) >= 2 and "return_difference" in data
        if name == "quant":
            return data["status"] == "ok"
        return True
    except (KeyError, TypeError, ValueError):
        return False


def review_evidence(task, artifacts):
    issues = []
    for name in task.expected_outputs:
        if name not in artifacts or not valid_evidence(name, artifacts[name]):
            issues.append(Critique(name, f"缺少有效的 {name} 证据", [name] if name in artifacts else [],
                                   "high", "planner" if name in REPAIRS else "executor",
                                   REPAIRS.get(name, "repair_tool_execution")).to_dict())
    if "simulation" in artifacts and "risk" in artifacts:
        from agents.roles.risk_analyst import RiskAnalyst
        expected = RiskAnalyst().analyze(artifacts["simulation"]["data"]["prices"])
        actual = artifacts["risk"]["data"]
        if any(not math.isclose(expected[k], actual.get(k, math.inf), abs_tol=1e-10)
               for k in ("cvar", "max_drawdown")):
            issues.append(Critique("consistency", "风险报告与价格路径不一致", ["simulation", "risk"],
                                   "critical", "critic", "recompute_risk").to_dict())
    return issues


def evaluate(task, trace):
    artifacts, report = trace.artifacts, trace.report
    complete = sum(valid_evidence(k, artifacts.get(k, {})) for k in task.expected_outputs) / len(task.expected_outputs)
    claims = report.get("conclusions", [])
    fields = {"conclusion", "evidence", "confidence", "caveat", "source", "facts"}
    schema = sum(fields.issubset(c) for c in claims) / max(1, len(claims))
    supported = [c for c in claims if c.get("evidence") and all(
        k in artifacts and valid_evidence(k, artifacts[k]) for k in c["evidence"])]
    coverage = len(supported) / max(1, len(claims))
    consistent = sum(c.get("facts") == artifacts[c["evidence"][0]]["data"] for c in supported) / max(1, len(claims))
    if any(c["dimension"] == "consistency" for c in trace.critiques):
        consistent = 0.0
    repetition = artifacts.get("reproducibility")
    reproducibility = float(valid_evidence("reproducibility", repetition)) if repetition else None
    task_quality = complete
    specific = {}
    if task.task_type == "historical_analysis":
        hist = artifacts.get("historical_comparison", {}).get("data", {})
        if hist:
            task_quality = (1 / (1 + hist["normalized_rmse"]) + hist["direction_consistency"]) / 2
            specific.update(normalized_rmse=hist["normalized_rmse"], normalized_mae=hist["normalized_mae"],
                            direction_consistency=hist["direction_consistency"])
        else:
            task_quality = 0.0
    if task.task_type == "regulatory_planning":
        cf = artifacts.get("counterfactual_search", {}).get("data", {})
        if cf:
            # Zero risk reduction remains zero. No action can be the best plan.
            task_quality = float(cf["constraint_satisfaction"]) * min(1.0, len(cf["plans"]) / 3)
            specific.update(risk_reduction=cf["risk_reduction"], intervention_cost=cf["intervention_cost"])
        else:
            task_quality = 0.0
    dimensions = {"completeness": complete * schema, "evidence": coverage, "consistency": consistent,
                  "task_quality": task_quality, "reproducibility": reproducibility or 0.0}
    quality = sum(task.evaluation_rules["weights"][k] * v for k, v in dimensions.items())
    critical = any(c["severity"] in {"critical", "high"} for c in trace.critiques)
    success = complete == 1 and not critical and quality >= task.evaluation_rules["success_threshold"] and not any(
        not e.get("recoverable", False) for e in trace.errors)
    pass_rate = max(0.0, 1 - len(trace.critiques) / (len(task.expected_outputs) + 1))
    tokens = None if any(x.get("token_usage") is None for x in trace.tool_calls) else sum(x["token_usage"] for x in trace.tool_calls)
    active = {x["agent"] for x in trace.agent_events if x["status"] in {"completed", "failed"}}
    return CompetitionScorecard(success, quality, coverage, pass_rate, reproducibility,
                               len(trace.agent_events), trace.retries, len(trace.tool_calls), tokens,
                               trace.latency, len(active), {**dimensions, **specific, "schema_completeness": schema}).to_dict()


def evolution_gain(before, after):
    def reduction(key):
        a, b = before.get(key), after.get(key)
        return None if a is None or b is None or a == 0 else (a - b) / a
    return {"quality_gain": after["quality_score"] - before["quality_score"],
            "success_gain": int(after["task_success"]) - int(before["task_success"]),
            "step_reduction": reduction("steps"), "token_reduction": reduction("token_usage"),
            "latency_reduction": reduction("latency")}
