"""Isolated Layer B worker. Every price is produced by existing Civitas code.

Separate processes isolate legacy global RNGs, caches and IPC lifecycles.
Run artifacts are written to a private working directory, never UI session data.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys


async def simulate(request):
    from core.reproducibility import seed_everything
    from core.runtime_mode import resolve_runtime_mode_profile
    from core.simulation_factory import build_policy_agents
    from engine.simulation_loop import MarketEnvironment
    from agents.roles.risk_analyst import RiskAnalyst

    seed = int(request["seed"])
    seed_everything(seed)
    profile = replace(resolve_runtime_mode_profile("SMART"), llm_primary=False, use_live_api=False)
    agents = build_policy_agents(profile, seed=seed, offline=True)
    env = MarketEnvironment(agents, llm_primary=False, enable_policy_committee=False,
                            enable_random_policy_events=False, use_isolated_matching=True,
                            market_pipeline_v2=True, steps_per_day=1,
                            enable_strategy_ecology=False, evolution_interval_days=10000)
    rows = []
    try:
        initial = env.current_price
        # Explicit initial liquidity condition: resting orders, not generated prices.
        # Use existing BufferedIntent and order-book matching for every execution.
        from simulation_runner import BufferedIntent
        for level in range(1, 21):
            for side, sign in (("buy", -1), ("sell", 1)):
                env.simulation_runner.submit_intent(BufferedIntent(
                    intent_id=f"initial-depth-{side}-{level}", agent_id=f"passive_{side}",
                    side=side, quantity=300, price=round(initial * (1 + sign * level * 0.0002), 2),
                    symbol="A_SHARE_IDX", activate_step=1,
                    metadata={"source": "fixed_benchmark_initial_depth"}))
        for day in range(int(request["days"])):
            policy = request["policy_text"] if day == 0 else ""
            if day == 3 and request.get("intervention"):
                policy = request["intervention"]
            if policy:
                env.schedule_policy_shock(policy, intensity=float(request.get("intensity", 1)),
                                          policy_id=f"benchmark-policy-{day}", source="competition_tool")
            result = await env.simulation_step()
            if result["matching_mode"] != "isolated_ipc":
                raise RuntimeError("Civitas matching failed; refusing to report impact-model fallback as order-book output")
            rows.append({"step": day + 1, "close": float(result["new_price"]),
                         "trade_count": int(result.get("trade_count", 0)),
                         "csad": float(result.get("behavioral_diagnostics", {}).get("csad", 0)),
                         "matching_mode": result["matching_mode"],
                         "llm_call_count": int(result.get("llm_call_count", 0))})
        prices = [initial] + [row["close"] for row in rows]
        if any(row["llm_call_count"] for row in rows):
            raise RuntimeError("Deterministic simulator unexpectedly invoked LLM")
        return {"engine": "engine.simulation_loop.MarketEnvironment", "seed": seed,
                "rows": rows, "prices": prices, "risk": RiskAnalyst().analyze(prices),
                "cumulative_return": prices[-1] / prices[0] - 1,
                "trades": sum(r["trade_count"] for r in rows),
                "market_participants": len(agents), "matching": "Civitas A-share order book / isolated IPC",
                "assumptions": "Six existing heuristic TraderAgents plus fixed two-sided initial resting depth (20 levels, 300 shares/level); synthetic macro/social states; fixed policy mapping."}
    finally:
        env.close()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    output = asyncio.run(simulate(request))
    Path(sys.argv[2]).write_text(json.dumps(output, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
