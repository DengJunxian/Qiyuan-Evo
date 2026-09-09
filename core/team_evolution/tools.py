"""Layer A tool boundary: reuse financial engines, measure every invocation."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .benchmark import load_dataset
from .models import digest
from .store import now

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evidence(name, data, source, confidence=1.0, caveat="仿真实验结果；置信度表示证据完整程度，不是投资收益概率。"):
    return {"evidence_id": name, "data": data, "source": source,
            "confidence": confidence, "caveat": caveat, "content_hash": digest(data)}


def policy_compile(task):
    from policy.structured import StructuredPolicyParser
    package = StructuredPolicyParser(seed=task.seed).parse(task.inputs.get("policy_text", task.objective), tick=1)
    data = package.to_dict()
    # Stable event identity for this frozen input (does not alter market facts).
    data["event"]["timestamp"] = 0.0
    old_id = data["event"]["policy_id"]
    stable_id = "policy-" + task.fingerprint[:16]
    data = json.loads(json.dumps(data).replace(old_id, stable_id))
    return data


class FinancialTools:
    def __init__(self, task, trace, run_dir):
        self.task, self.trace, self.run_dir = task, trace, Path(run_dir)
        self.cache = {}
        self.frame, self.manifest = load_dataset()

    async def call(self, role, name, function, *, args=None, cache=False, retries=0):
        args = args or {}
        key = digest({"name": name, "args": args, "task": self.task.fingerprint,
                      "dataset": self.manifest["sha256"]})
        for attempt in range(retries + 1):
            start = time.perf_counter()
            cached = cache and key in self.cache
            record = {"call_id": f"tool-{len(self.trace.tool_calls) + 1}", "agent": role,
                      "tool": name, "arguments": args, "cache_hit": cached, "attempt": attempt,
                      "token_usage": 0, "status": "running", "started_at": now()}
            self.trace.tool_calls.append(record)
            try:
                result = deepcopy(self.cache[key]) if cached else await asyncio.to_thread(function)
                digest(result)
                self.cache[key] = deepcopy(result)
                record.update(status="ok", result_hash=digest(result))
                return result
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                self.trace.errors.append({"agent": role, "tool": name, "type": type(exc).__name__,
                                          "recoverable": attempt < retries})
                if attempt == retries:
                    raise
                self.trace.retries += 1
            finally:
                record["latency"] = time.perf_counter() - start
                record["finished_at"] = now()

    def market_simulation(self, *, policy_text=None, intervention="", intensity=1.0):
        request = {"seed": self.task.seed,
                   "policy_text": self.task.inputs.get("policy_text", self.task.objective) if policy_text is None else policy_text,
                   "intervention": intervention, "intensity": intensity,
                   "days": min(60, max(4, int(self.task.constraints.get("simulation_days", 12))))}
        # A separate directory even for replicate calls: no accidental cache reuse.
        import tempfile
        work = Path(tempfile.mkdtemp(prefix="simulation-", dir=self.run_dir))
        shutil.copyfile(PROJECT_ROOT / "config.yaml", work / "config.yaml")
        source = work / "request.json"
        target = work / "result.json"
        source.write_text(json.dumps(request, ensure_ascii=False))
        env = dict(os.environ)
        env.update(PYTHONPATH=str(PROJECT_ROOT), PYTHONHASHSEED=str(self.task.seed),
                   CIVITAS_RANDOM_SEED=str(self.task.seed), CIVITAS_SEED=str(self.task.seed),
                   CIVITAS_VECTOR_FAST_AGENTS="true")
        # Prevent ambient keys from enabling any model inside deterministic tools.
        for name in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY", "OPENAI_API_KEY"):
            env[name] = ""
        with (work / "worker.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "core.team_evolution.simulation_worker", str(source), str(target)],
                                    cwd=work, env=env, stdout=log, stderr=log, timeout=90, check=False)
        if result.returncode or not target.exists():
            raise RuntimeError(f"Civitas worker failed; see {work / 'worker.log'}")
        return json.loads(target.read_text())

    def historical_replay(self):
        from core.calibration.replay_runner import ReplayConfig, ReplayRunner
        from core.data.macro_data_provider import MacroDataProvider
        from agents.roles.risk_analyst import RiskAnalyst
        real = self.frame.copy()
        config = ReplayConfig(seed=self.task.seed, start=real.iloc[0]["date"], end=real.iloc[-1]["date"])
        # Exact alignment to observed trading dates, not calendar/business-day position.
        macro = MacroDataProvider().load_macro_panel(config.start, config.end, seed=self.task.seed)
        panel = macro.frame.set_index("date").loc[real["date"]].reset_index()
        simulated = ReplayRunner._simulate_path(real, panel, config)
        prices = simulated["close"].tolist()
        return {"engine": "core.calibration.replay_runner.ReplayRunner._simulate_path",
                "seed": self.task.seed, "prices": prices, "rows": simulated.to_dict(orient="records"),
                "risk": RiskAnalyst().analyze(prices), "cumulative_return": prices[-1] / prices[0] - 1,
                "dataset_hash": self.manifest["sha256"], "dates": real["date"].tolist(),
                "macro_provider": macro.provider, "macro_hash": digest(panel.to_dict(orient="records")),
                "assumptions": "One-step conditional replay uses lagged observed close; synthetic macro panel; not an out-of-sample forecast."}

    def historical_comparison(self, simulation):
        import numpy as np
        from core.eval.replay_scorecard import rmse, returns
        observed = self.frame["close"].tolist()
        sim = simulation["prices"]
        # Generic market-run horizon is resampled explicitly for shape comparison.
        aligned = np.interp(np.linspace(0, 1, len(observed)), np.linspace(0, 1, len(sim)), sim)
        actual = np.asarray(observed) / observed[0]
        predicted = aligned / aligned[0]
        errors = predicted - actual
        direction = float(np.mean(np.sign(returns(predicted)) == np.sign(returns(actual))))
        return {"normalized_rmse": rmse(predicted, actual), "normalized_mae": float(np.abs(errors).mean()),
                "direction_consistency": direction, "observed_prices": observed,
                "simulated_normalized": predicted.tolist(), "observed_normalized": actual.tolist(),
                "dates": self.frame["date"].tolist(), "observations": len(observed),
                "source": self.manifest, "alignment": "normalized index on trading-date grid",
                "error_explanation": {"mean_bias": float(errors.mean()), "maximum_absolute_error": float(np.abs(errors).max()),
                                      "model_assumptions": simulation["assumptions"]}}

    def quantitative_analysis(self, prices):
        from agents.base_agent import MarketSnapshot
        from agents.roles.quant_analyst import QuantAnalyst
        return QuantAnalyst().analyze(MarketSnapshot("A_SHARE_IDX", prices[-1]), price_series=prices)
