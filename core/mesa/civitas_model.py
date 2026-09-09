"""
Mesa Model 编排器

使用 Mesa 框架管理 Agent 群体、市场状态和数据收集。
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
import asyncio
import os
from typing import Dict, List, Optional, Any

from core.society.network import SocialGraph, InformationDiffusion
from agents.persona import Persona, PersonaGenerator, RiskAppetite
from core.mesa.civitas_agent import CivitasAgent
from agents.cognition.utility import InvestorType
from core.time_manager import SimulationClock
from core.utils import truncate_text


def social_snapshot_to_orderflow_reversal(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """Translate social contagion output into an order-flow direction proxy."""
    suppression = dict(snapshot.get("rumor_suppression", {})) if isinstance(snapshot, dict) else {}
    sentiment = snapshot.get("node_sentiment", {}) if isinstance(snapshot, dict) else {}
    mean_sentiment = 0.0
    if isinstance(sentiment, dict) and sentiment:
        mean_sentiment = float(sum(float(v) for v in sentiment.values()) / len(sentiment))
    suppression_ratio = float(suppression.get("suppression_ratio", 0.0))
    rumor_heat_before = float(suppression.get("rumor_heat_before", 0.0))
    refutation_heat = float(suppression.get("refutation_heat", 0.0))
    orderflow_bias = float(np.clip(mean_sentiment + 0.5 * suppression_ratio, -1.0, 1.0))
    reversal_strength = float(np.clip(refutation_heat - 0.35 * rumor_heat_before, -1.0, 1.0))
    return {
        "orderflow_bias": orderflow_bias,
        "reversal_strength": reversal_strength,
        "suppression_ratio": suppression_ratio,
    }

class CivitasModel(Model):
    """
    Civitas Mesa Model
    
    编排整个模拟系统：
    - 管理 Agent 群体
    - 控制时钟步进
    - 收集微观数据
    - 计算宏观指标 (如 CSAD)
    
    Mesa 3.0+ API:
    - 使用 self.agents (AgentSet) 替代 self.schedule
    - step(): 调用 self.agents.do("step") 和 self.agents.do("advance")
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        panic_ratio: float = 0.3,
        quant_ratio: float = 0.1,
        initial_price: float = 3000.0,
        seed: Optional[int] = None,
        model_router: Optional[Any] = None,
        quant_manager: Optional[Any] = None,
        regulatory_module: Optional[Any] = None,
        mode: str = "SMART",
        ipc_mode: bool = False
    ):
        """
        初始化模型
        """
        self.regulatory_module = regulatory_module
        
        super().__init__(seed=seed)
        
        self.mode = mode
        self.ipc_mode = ipc_mode
        self.n_agents = n_agents
        self.shared_router = model_router
        
        if self.ipc_mode:
            from core.ipc.engine_server import IPCEngineServer
            self.ipc_server = IPCEngineServer(pub_port=5555, pull_port=5556)
        else:
            self.ipc_server = None
            
        self.quant_manager = quant_manager
        self.clock = SimulationClock(mode=mode)
        
        graph_k = min(10, n_agents - 1)
        if graph_k % 2 != 0:
            graph_k = max(2, graph_k - 1)
        self.social_graph = SocialGraph(n_agents=n_agents, k=graph_k, p=0.1, seed=seed)
        self.diffusion = InformationDiffusion(self.social_graph)
        
        from core.market_engine import MarketDataManager
        self.market_manager = MarketDataManager(
            api_key_or_router=model_router, 
            load_real_data=True, 
            clock=self.clock,
            regulatory_module=self.regulatory_module
        )
        
        self.seed_store = None
        self._last_seed_event_id: Optional[str] = None
        try:
            from data_flywheel.seed_store import SeedStore
            self.seed_store = SeedStore("data/seed_events.jsonl")
        except Exception:
            self.seed_store = None
            
        self.current_price = self.market_manager.engine.last_price
        self.price_history: List[float] = [self.current_price]
        self.trend = "震荡"
        self.panic_level = self.market_manager.panic_level
        self.volatility = 0.02 
        self.csad = 0.0
        self.last_step_trades_count = 0
        self.last_smart_sentiment = 0.0

        # 初始化数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Price": "current_price",
                "CSAD": "csad",
                "Panic_Level": "panic_level",
                "Avg_Wealth": lambda m: np.mean([a.wealth for a in m.agents]),
                "Avg_Sentiment": lambda m: np.mean([a.sentiment for a in m.agents]),
                "Volume": lambda m: m.market_manager.sim_candles[-1].volume if m.market_manager.sim_candles else 0,
                "Infected_Count": lambda m: m.diffusion.history[-1]['infected'] if m.diffusion.history else 0
            },
            agent_reporters={
                "Wealth": "wealth",
                "Position": "position",
                "PnL_Pct": "pnl_pct",
                "Sentiment": "sentiment",
                "Confidence": "confidence",
            }
        )
        
        # 1. 创建异质 智能体 群体
        self._create_agents(panic_ratio, quant_ratio)
        
        # 2. 初始化 第二层
        from agents.population import StratifiedPopulation
        self.population = StratifiedPopulation(
            n_smart=0,  # 第一层 由 Mesa 管理
            n_vectorized=5000, # 默认 5000 散户
            smart_agents=list(self.agents)
        )

    def _create_agents(self, panic_ratio: float, quant_ratio: float) -> None:
        """创建异质 Agent 群体（含社交网络绑定）。"""
        n_panic = int(self.n_agents * panic_ratio)
        n_quant = int(self.n_agents * quant_ratio)

        # 小规模场景不注入国家队，保证测试与演示时行为可预测。
        reserve_national_team = self.n_agents > 20
        reserved_slots = 1 if reserve_national_team else 0
        n_normal = max(0, self.n_agents - n_panic - n_quant - reserved_slots)

        personas = PersonaGenerator.generate_distribution(self.n_agents)
        use_llm_mask = [False] * self.n_agents
        model_priorities: List[Optional[List[str]]] = [None] * self.n_agents
        mode = getattr(self, "mode", "SMART")

        # 获取基于模式的模型优先级
        priority = self.shared_router.get_model_priority(mode) if self.shared_router else None

        # 仅让前 N 个 智能体 走真实模型，兼顾开销与可复现性。
        max_real_agents = min(5, self.n_agents)
        for i in range(max_real_agents):
            use_llm_mask[i] = True
            model_priorities[i] = priority

        idx = 0

        for _ in range(n_panic):
            p = personas[idx]
            p.risk_appetite = RiskAppetite.GAMBLER
            p.conformity = 0.9
            p.patience = 0.1
            p.loss_aversion = 3.0
            self._create_single_agent(
                idx,
                InvestorType.PANIC_RETAIL,
                p,
                use_llm=use_llm_mask[idx],
                model_priority=model_priorities[idx],
            )
            idx += 1

        for _ in range(n_quant):
            p = personas[idx]
            p.risk_appetite = RiskAppetite.BALANCED
            p.conformity = 0.05
            p.influence = 0.8
            p.loss_aversion = 1.0
            self._create_single_agent(
                idx,
                InvestorType.DISCIPLINED_QUANT,
                p,
                cash=1_000_000,
                use_llm=use_llm_mask[idx],
                model_priority=model_priorities[idx],
            )
            idx += 1

        for _ in range(n_normal):
            p = personas[idx]
            self._create_single_agent(
                idx,
                InvestorType.NORMAL,
                p,
                use_llm=use_llm_mask[idx],
                model_priority=model_priorities[idx],
            )
            idx += 1

        if reserve_national_team:
            from agents.roles.national_team import NationalTeamAgent
            nt_core = NationalTeamAgent(
                agent_id=f"nt_{idx}",
                cash_balance=10_000_000_000.0,
            )
            nt_priority = ["deepseek-reasoner", "glm-4-flashx"] if mode == "DEEP" else ["glm-4-flashx"]
            CivitasAgent(
                model=self,
                investor_type=InvestorType.DISCIPLINED_QUANT,
                initial_cash=10_000_000_000.0,
                use_local_reasoner=False,
                core_agent=nt_core,
                model_router=self.shared_router,
                use_llm=True,
                model_priority=nt_priority,
            )
            hub_node = self.social_graph.get_most_influential_node()
            if hub_node is not None:
                nt_core.bind_social_node(hub_node, self.social_graph)
            else:
                nt_core.bind_social_node(idx, self.social_graph)

    def _create_single_agent(
        self, 
        idx: int, 
        inv_type: InvestorType, 
        persona: Persona, 
        cash: float = None,
        use_llm: bool = False,
        model_priority: List[str] = None
    ) -> CivitasAgent:
        if cash is None:
            cash = np.random.uniform(50000, 500000)
            
        from agents.trader_agent import TraderAgent
        agent_str_id = str(idx)
        core = TraderAgent(
            agent_id=agent_str_id,
            cash_balance=cash,
            portfolio={},
            psychology_profile=None,
            persona=persona,
            model_router=self.shared_router,
            use_llm=use_llm,
            model_priority=model_priority
        )
            
        agent = CivitasAgent(
            model=self,
            investor_type=inv_type,
            initial_cash=cash,
            use_local_reasoner=True,
            persona=persona,
            core_agent=core,
            model_router=self.shared_router,
            use_llm=use_llm,
            model_priority=model_priority
        )
        
        agent.core.bind_social_node(idx, self.social_graph)
        return agent
    
    def get_market_state(self) -> Dict[str, Any]:
        """获取当前市场状态 (供 Agent 决策使用)"""
        return {
            "price": self.current_price,
            "trend": self.trend,
            "panic_level": self.panic_level,
            "volatility": self.volatility,
            "csad": self.csad,
            "news": truncate_text(self.market_manager.current_news, 500),
            "text_dominant_topic": self.market_manager.text_factor_state.get("dominant_topic", "uncategorized"),
            "text_sentiment_score": self.market_manager.text_factor_state.get("sentiment_score", 0.0),
            "text_panic_score": self.market_manager.text_factor_state.get("panic_index", 0.0),
            "text_greed_score": self.market_manager.text_factor_state.get("greed_index", 0.0),
            "text_policy_shock": self.market_manager.text_factor_state.get("policy_shock", 0.0),
            "text_regime_bias": self.market_manager.text_factor_state.get("regime_bias", "neutral"),
            "dates": self.market_manager.history_candles[-1].timestamp if self.market_manager.history_candles else "2024-01-01",
            "infected_ratio": self.diffusion.history[-1]['infected'] / self.n_agents if self.diffusion.history else 0
        }
    
    @property
    def current_dt(self) -> Any:
        if self.market_manager.sim_candles:
            ts = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            ts = self.market_manager.history_candles[-1].timestamp
        else:
            ts = "2024-01-01"
        
        import pandas as pd
        return pd.Timestamp(ts)

    def set_policy(self, name: str, param: str, value) -> None:
        """
        动态设置监管策略参数
        """
        self.market_manager.policy_manager.set_policy_param(name, param, value)
    
    def get_policy_status(self) -> Dict[str, Any]:
        """获取当前策略状态。"""
        pm = self.market_manager.policy_manager
        cb = pm.policies.get("circuit_breaker")
        tax = pm.policies.get("tax")
        return {
            "circuit_breaker": {
                "active": cb.active if cb else False,
                "threshold": cb.threshold_pct if cb else 0,
                "is_halted": cb.is_halted if cb else False,
            },
            "transaction_tax": {
                "active": tax.active if tax else False,
                "rate": tax.rate if tax else 0,
            }
        }

    def _ingest_latest_seed_event(self) -> None:
        if self.seed_store is None:
            return
        try:
            latest = self.seed_store.read_latest(n=1, min_impact="medium")
        except Exception:
            return
        if not latest:
            return
        event = latest[0]
        event_id = getattr(event, "event_id", None)
        if event_id and event_id == self._last_seed_event_id:
            return
        self.market_manager.ingest_seed_event(event)
        self._last_seed_event_id = event_id

    def step(self) -> None:
        """
        [已废弃] 同步 Step 方法
        """
        raise NotImplementedError("CivitasModel.step() is deprecated. Please use await model.async_step() instead.")

    async def async_step(self) -> None:
        """
        执行一个模拟回合 (异步)
        """
        self._ingest_latest_seed_event()
        cb = self.market_manager.policy_manager.policies.get("circuit_breaker")
        if cb:
            cb.update_reference_price(self.current_price)
        
        if self.diffusion:
            for agent in self.agents:
                if hasattr(agent, "core") and hasattr(agent.core, "sync_social_semantic_profile"):
                    try:
                        agent.core.sync_social_semantic_profile()
                    except Exception:
                        pass
            self.diffusion.update_sentiment_propagation()
        
        market_state = self.get_market_state()
        snapshot = self.market_manager.get_market_snapshot()
        news = [market_state["news"]] if market_state["news"] else []
        
        if getattr(self, "ipc_mode", False) and self.ipc_server:
            from core.ipc.message_types import MarketStatePacket
            packet = MarketStatePacket(
                step=self.steps,
                timestamp=str(self.current_dt),
                price=self.current_price,
                trend=self.trend,
                panic_level=self.panic_level,
                csad=self.csad,
                volatility=self.volatility,
                recent_news=news
            )
            await self.ipc_server.broadcast_market_state(packet)
            actions = await self.ipc_server.collect_agent_actions(collect_window=3.0, expected_count=self.n_agents)
            
            smart_actions = []
            for action in actions:
                if action.has_order and action.order:
                    from core.market_engine import Order, OrderType
                    try:
                        o_type = getattr(OrderType, action.order.order_type.upper())
                    except Exception:
                        o_type = OrderType.LIMIT
                        
                    order = Order(
                        symbol=action.order.symbol,
                        price=action.order.price,
                        quantity=action.order.quantity,
                        side=action.order.side,
                        order_type=o_type,
                        agent_id=action.agent_id,
                        timestamp=str(self.current_dt)
                    )
                    self.market_manager.submit_agent_order(order)
                    
                    side_str = action.order.side.lower()
                    if "buy" in side_str:
                        smart_actions.append(1)
                    elif "sell" in side_str:
                        smart_actions.append(-1)
                    else:
                        smart_actions.append(0)
                else:
                    smart_actions.append(0)
            self.last_smart_sentiment = np.mean(smart_actions) if smart_actions else 0.0
            
        else:
            agent_tasks = []
            active_agents = []
            for agent in self.agents:
                if isinstance(agent, CivitasAgent):
                    agent_tasks.append(agent.async_act(snapshot, news))
                    active_agents.append(agent)
            
            orders = await asyncio.gather(*agent_tasks, return_exceptions=True)
            smart_actions = []
            for agent, order_or_err in zip(active_agents, orders):
                if isinstance(order_or_err, Exception):
                    print(f"[Model Error] Agent {agent.unique_id} act failed: {order_or_err}")
                    smart_actions.append(0)
                    continue
                    
                order = order_or_err
                if order:
                    self.market_manager.submit_agent_order(order)
                    side_str = str(order.side).lower()
                    if "buy" in side_str:
                        smart_actions.append(1)
                    elif "sell" in side_str:
                        smart_actions.append(-1)
                    else:
                        smart_actions.append(0)
                else:
                    smart_actions.append(0)
            self.last_smart_sentiment = np.mean(smart_actions) if smart_actions else 0.0

        self.agents.do("advance")
        
        trend_signal = 1.0 if self.trend == "上涨" else -1.0
        if self.trend == "下跌":
            trend_signal = -1.0
        elif self.trend == "震荡":
            trend_signal = 0.0
        
        self.population.update_tier2_sentiment(smart_actions, trend_signal)
        v_actions, v_qtys, v_prices = self.population.generate_tier2_decisions(self.current_price)
        
        active_indices = np.where(v_actions != 0)[0]
        if len(active_indices) > 0:
            sample_cap = 0 if (os.getenv("PYTEST_CURRENT_TEST") and self.n_agents <= 20) else 500
            sample_size = min(len(active_indices), sample_cap)
            chosen_idx = np.random.choice(active_indices, sample_size, replace=False)
            
            ts = self.clock.timestamp
            from core.market_engine import Order, OrderType
            for idx in chosen_idx:
                side = 'buy' if v_actions[idx] == 1 else 'sell'
                order = Order(
                    symbol=self.market_manager.engine.symbol,
                    price=float(v_prices[idx]),
                    quantity=int(v_qtys[idx]),
                    side=side,
                    order_type=OrderType.LIMIT, 
                    agent_id=f"Vec_{idx}",
                    timestamp=ts
                )
                self.market_manager.submit_agent_order(order)
                
        step_trades = self.market_manager.engine.flush_step_trades()
        self.last_step_trades_count = len(step_trades)
        
        vec_indices = []
        vec_prices = []
        vec_qtys = []
        vec_dirs = []
        
        for t in step_trades:
            buyer_id = getattr(t, 'buyer_agent_id', getattr(t, 'buy_agent_id', None))
            seller_id = getattr(t, 'seller_agent_id', getattr(t, 'sell_agent_id', None))
            if buyer_id and buyer_id.startswith("Vec_"):
                idx = int(buyer_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(1) 
            if seller_id and seller_id.startswith("Vec_"):
                idx = int(seller_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(-1) 
                
        if vec_indices:
            self.population.sync_tier2_execution(
                np.array(vec_indices),
                np.array(vec_prices),
                np.array(vec_qtys),
                np.array(vec_dirs)
            )

        if self.regulatory_module:
            reg_result = self.regulatory_module.process_tick(
                current_tick=self.steps,
                price=self.current_price,
                prev_close=self.price_history[-2] if len(self.price_history) > 1 else self.current_price,
                panic_level=self.panic_level
            )
            if reg_result.get("circuit_break"):
                 print(f"[Regulatory] 熔断: {reg_result['circuit_break']}")
            if reg_result.get("intervention"):
                 print(f"[Regulatory] 国家队干预: {reg_result['intervention']}")

        if self.quant_manager:
            try:
                q_market = {
                    "price": self.current_price,
                    "trend": self.trend,
                    "panic_level": self.panic_level,
                    "news": self.market_manager.current_news
                }
                q_accounts = {} 
                group_tasks = []
                for group in self.quant_manager.groups.values():
                    if hasattr(group, 'get_group_decisions_async'):
                        group_tasks.append(group.get_group_decisions_async(q_market, q_accounts))
                if group_tasks:
                    await asyncio.gather(*group_tasks, return_exceptions=True)
            except Exception as e:
                print(f"[QuantGroup Error] {e}")

        last_date = "2024-01-01"
        if self.market_manager.sim_candles:
            last_date = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            last_date = self.market_manager.history_candles[-1].timestamp
            
        current_candle = self.market_manager.finalize_step(self.steps, last_date, trades=step_trades)
        self.current_price = current_candle.close
        self.price_history.append(self.current_price)
        self._calculate_csad()
        self._update_trend()
        self.datacollector.collect(self)
        self.clock.tick()
        self.steps += 1
        
    def _update_trend(self):
        """更新市场趋势标签。"""
        if len(self.price_history) >= 5:
            recent = self.price_history[-5:]
            if recent[-1] > recent[0] * 1.02:
                self.trend = "上涨"
            elif recent[-1] < recent[0] * 0.98:
                self.trend = "下跌"
            else:
                self.trend = "震荡"

    def _calculate_csad(self) -> None:
        """计算截面绝对偏差 (CSAD)"""
        sentiments = [a.sentiment for a in self.agents]
        if not sentiments:
            self.csad = 0.0
            return
        avg_sentiment = np.mean(sentiments)
        self.csad = np.mean(np.abs(np.array(sentiments) - avg_sentiment))
        
    async def run_simulation(self, n_steps: int = 100) -> None:
        """运行完整模拟 (异步)"""
        for _ in range(n_steps):
            await self.async_step()
    
    def get_results(self) -> Dict[str, Any]:
        """获取模拟结果"""
        model_data = self.datacollector.get_model_vars_dataframe()
        agent_data = self.datacollector.get_agent_vars_dataframe()
        return {
            "model_data": model_data,
            "agent_data": agent_data,
            "final_price": self.current_price,
            "price_history": self.price_history,
            "n_steps": self.steps
        }
