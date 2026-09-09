"""Minimal demo scenario: strategy genome + hybrid replay + abuse sandbox."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.simulation_loop import MarketEnvironment


class DemoAgent:
    def __init__(self, agent_id: str, action: str) -> None:
        self.agent_id = agent_id
        self._action = action
        self.persona = type("PersonaStub", (), {"risk_tolerance": 0.5})()
        self.memory_bank = None

    async def generate_trading_decision(self, market_data, retrieved_context):
        _ = retrieved_context
        amount = 120.0 if self._action == "BUY" else 90.0 if self._action == "SELL" else 0.0
        return type(
            "Action",
            (),
            {
                "action": self._action,
                "amount": amount,
                "target_price": float(market_data.get("current_price", 100.0)),
            },
        )()


async def main() -> None:
    backdrop = Path(__file__).parent / "backdrop.csv"
    agents = [DemoAgent("buyer", "BUY"), DemoAgent("seller", "SELL")]
    env = MarketEnvironment(
        agents,
        use_isolated_matching=True,
        runner_symbol="DEMO",
        hybrid_replay=True,
        exogenous_backdrop=backdrop,
        hybrid_backdrop_weight=0.4,
        enable_abuse_agents=True,
        abuse_agent_scale=1.0,
    )
    try:
        for _ in range(5):
            report = await env.simulation_step()
            print(report["tick"], report["new_price"], report["matching_mode"])
        print("ecology:", report.get("ecology_metrics_path"))
        print("abuse:", report.get("market_abuse_report_path"))
        print("intervention:", report.get("intervention_effect_report_path"))
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
