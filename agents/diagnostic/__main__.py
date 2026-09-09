import asyncio
import sys

from agents.diagnostic.diagnostic_agent import DiagnosticAgent
from agents.persona import Persona
from agents.trader_agent import TraderAgent

async def async_main():
    print("=====================================================")
    print("      Civitas 诊断探针 (Diagnostic Agent) - 调试终端     ")
    print("=====================================================")
    print("正在初始化虚拟测试环境...\n")
    
    # 构建 模拟 市场环境
    class MockMarketEnv:
        last_price = 3200.5
        trend = "震荡"
        panic_level = 0.6
        current_time = 1718000000.0
        
    market_env = MockMarketEnv()
    
    # 构建模拟智能体映射
    agents_map = {}
    
    # 智能体 1: 激进亏钱
    p1 = Persona(name="Retail_Loss", risk_preference="激进")
    a1 = TraderAgent(agent_id="Retail_Loss", persona=p1)
    a1.cash_balance = 5000
    a1.total_pnl = -45000
    a1.brain.state.confidence = 20
    agents_map["Retail_Loss"] = a1
    
    # 智能体 2: 稳健赚钱
    p2 = Persona(name="Inst_Profit", risk_preference="稳健")
    a2 = TraderAgent(agent_id="Inst_Profit", persona=p2)
    a2.cash_balance = 80000
    a2.total_pnl = 15000
    a2.brain.state.confidence = 85
    agents_map["Inst_Profit"] = a2
    
    # 初始化诊断助手
    try:
        diag_agent = DiagnosticAgent(agents_map=agents_map, market_env=market_env)
    except Exception as e:
        print(f"初始化诊断大模型失败 (请检查 config.yaml 中 API 配置): {e}")
        return
        
    print("环境就绪。已挂载探针 (Tools):")
    print(" - get_agent_state")
    print(" - get_agent_thought_history")
    print(" - query_agent_memory")
    print(" - get_market_snapshot\n")
    
    print("可用虚拟 Agent_ID 测试：'Retail_Loss', 'Inst_Profit'")
    print("输入 'exit' 退出。")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n[You] > ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input.strip():
                continue
                
            print("\n[Diagnostic Agent] 诊断中，请稍候...")
            response = await diag_agent.chat(user_input)
            print("-" * 50)
            print(f"[Diagnostic Agent]\n{response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
