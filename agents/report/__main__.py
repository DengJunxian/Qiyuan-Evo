import asyncio
import time

from agents.report.report_agent import ReportAgent
from agents.persona import Persona
from agents.trader_agent import TraderAgent
from core.types import Trade


async def async_main():
    print("=====================================================")
    print("      Civitas 复盘诊断 (Report Agent) - ReAct 终端     ")
    print("=====================================================")
    print("正在初始化虚拟测试环境...\n")

    # 构建模拟撮合引擎
    class MockEngine:
        def __init__(self):
            self.trades_history = []
            self.step_trades_buffer = []

    engine = MockEngine()
    # 简单构造一些成交记录
    engine.trades_history.append(
        Trade(
            trade_id="T001",
            price=3200.0,
            quantity=100,
            maker_id="m1",
            taker_id="t1",
            maker_agent_id="Inst_Profit",
            taker_agent_id="Retail_Loss",
            buyer_agent_id="Inst_Profit",
            seller_agent_id="Retail_Loss",
            timestamp=time.time()
        )
    )

    # 构建模拟市场环境
    class MockMarketEnv:
        last_price = 3200.5
        trend = "震荡"
        panic_level = 0.6
        current_time = 1718000000.0
        def __init__(self, engine):
            self.engine = engine

    market_env = MockMarketEnv(engine)

    # 构建模拟智能体映射
    agents_map = {}

    p1 = Persona(name="Retail_Loss")
    a1 = TraderAgent(agent_id="Retail_Loss", persona=p1)
    a1.cash_balance = 5000
    a1.total_pnl = -45000
    a1.brain.state.confidence = 20
    agents_map["Retail_Loss"] = a1

    p2 = Persona(name="Inst_Profit")
    a2 = TraderAgent(agent_id="Inst_Profit", persona=p2)
    a2.cash_balance = 80000
    a2.total_pnl = 15000
    a2.brain.state.confidence = 85
    agents_map["Inst_Profit"] = a2

    try:
        report_agent = ReportAgent(agents_map=agents_map, market_env=market_env)
    except Exception as e:
        print(f"初始化 ReportAgent 失败 (请检查 config.yaml 中 API 配置): {e}")
        return

    print("环境就绪。已挂载工具 (Tools):")
    print(" - get_agent_state")
    print(" - get_agent_thought_history")
    print(" - query_agent_memory")
    print(" - search_agent_memory")
    print(" - query_lob_log")
    print(" - send_interview")
    print(" - get_market_snapshot\n")

    print("可用虚拟 Agent_ID 测试: 'Retail_Loss', 'Inst_Profit'")
    print("输入 'exit' 退出。")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n[You] > ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue

            print("\n[Report Agent] 诊断中，请稍候...")
            response = await report_agent.chat(user_input)
            print("-" * 50)
            print(f"[Report Agent]\n{response}")
            print("-" * 50)

        except KeyboardInterrupt:
            break


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
