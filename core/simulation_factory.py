"""Shared Civitas participant construction for policy UI and team tools."""
from __future__ import annotations


def build_policy_agents(runtime_profile, *, seed=42, offline=False):
    from agents.persona import Persona
    from agents.trader_agent import TraderAgent

    specs = [("retail_day_trader", 240_000.0), ("retail_swing", 320_000.0),
             ("retail_momentum_chaser", 200_000.0), ("mutual_fund", 1_800_000.0),
             ("quant_arbitrage", 1_400_000.0), ("market_maker", 2_200_000.0)]
    cutoff = max(3, int(round(len(specs) * 0.67)))
    return [TraderAgent(
        agent_id=f"deep_{i:02d}", cash_balance=cash,
        portfolio={"A_SHARE_IDX": int(200 + 40 * i)},
        psychology_profile={"feature_flags": {"trader_intent_execution_split_v1": True},
                            "institution_type": archetype},
        persona=Persona.from_archetype(archetype, name=f"{archetype}_{i:02d}"),
        use_llm=bool(runtime_profile.llm_primary and i < cutoff),
        model_priority=list(runtime_profile.model_priority), execution_plan_enabled=True,
        execution_seed=seed + i,
        news_config={"offline": True} if offline else None,
    ) for i, (archetype, cash) in enumerate(specs)]
