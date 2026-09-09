import asyncio
import logging
import json
from typing import Optional

from core.ipc.runtime_compat import ensure_zmq_asyncio_compatibility

ensure_zmq_asyncio_compatibility()

import zmq
import zmq.asyncio

from agents.trader_agent import TraderAgent
from agents.persona import Persona
from core.market_engine import Order
from core.ipc.message_types import MarketStatePacket, AgentActionPacket, OrderPayload

logger = logging.getLogger(__name__)

class IPCAgentNode:
    """
    微服务节点：包装一个独立的 TraderAgent
    监听 Engine 的 PUB 广播，并向 Engine 的 PULL 队列返回行动指令。
    这样可以将成千上万个 Agent 分布在多线程/多进程/不同物理机上。
    """
    def __init__(
        self,
        agent_id: str,
        persona: Persona,
        server_host: str = "127.0.0.1",
        pub_port: int = 5555,
        pull_port: int = 5556,
        model_router = None,
        use_llm: bool = True
    ):
        self.agent_id = agent_id
        
        # 本地初始化 Core 智能体
        self.trader = TraderAgent(
            agent_id=self.agent_id,
            cash_balance=100_000,
            portfolio={},
            persona=persona,
            model_router=model_router,
            use_llm=use_llm
        )
        
        self.context = zmq.asyncio.Context.instance()
        
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_host}:{pub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅所有主题
        
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{server_host}:{pull_port}")
        
        logger.info(f"[AgentNode {self.agent_id}] Connected to IPC Server @ {server_host}")
        
    def _build_perceived_data(self, packet: MarketStatePacket):
        """将 JSON Packet 转为让 TraderAgent.reason_and_act 识别的字典结构"""
        class DummySnapshot:
            symbol = "000001"
            last_price = packet.price
            market_trend = 1.0 if packet.trend == "上涨" else (-1.0 if packet.trend == "下跌" else 0.0)
            panic_level = packet.panic_level
            timestamp = packet.timestamp
            
        initial = getattr(self.trader, 'initial_cash', 100000)
        pnl = (self.trader.cash_balance - initial)
        pnl_pct = pnl / initial if initial > 0 else 0.0
            
        return {
            "snapshot": DummySnapshot(),
            "news": packet.recent_news,
            "portfolio_value": self.trader.cash_balance,
            "cash": self.trader.cash_balance,
            "pnl_pct": pnl_pct,
            "timestamp": packet.timestamp
        }

    async def _router_fallback_decision(self, state: MarketStatePacket) -> Optional[dict]:
        """
        兼容只实现 call_with_fallback 的路由器（测试桩常见）。
        """
        router = getattr(self.trader, "brain", None)
        router = getattr(router, "model_router", None)
        if router is None or not hasattr(router, "call_with_fallback"):
            return None

        try:
            prompt = (
                f"step={state.step}, price={state.price}, trend={state.trend}, "
                f"panic={state.panic_level}, news={state.recent_news}"
            )
            content, _, _ = await router.call_with_fallback(
                [{"role": "user", "content": prompt}],
                priority_models=["deepseek-chat"],
                timeout_budget=5.0,
            )
            parsed = json.loads(content) if isinstance(content, str) else content
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None
        
    async def run_loop(self):
        """客户端无限循环：监听市场 TICK，决策，回传订单"""
        logger.info(f"[AgentNode {self.agent_id}] Running event loop...")
        
        try:
            while True:
                # 1. 等待主节点广播
                msg_str = await self.sub_socket.recv_string()
                
                try:
                    state_data = json.loads(msg_str)
                    state = MarketStatePacket(**state_data)
                except Exception as e:
                    logger.error(f"[{self.agent_id}] Failed to parse MarketState: {e}")
                    continue
                    
                # 2. 构造环境并思考
                perceived_data = self._build_perceived_data(state)
                
                # 开始调用大模型或量化快思考 (无明确社交信号则传 neutral)
                decision_output = await self.trader.reason_and_act(perceived_data, emotional_state="Neutral", social_signal="neutral")
                
                # 3. 构造并发送反馈 Packet
                action_pkt = AgentActionPacket(
                    agent_id=self.agent_id,
                    step=state.step,
                    has_order=False,
                    sentiment=getattr(self.trader.brain.state, 'sentiment', 0.0) if hasattr(self.trader, 'brain') and self.trader.brain else 0.0,
                    confidence=getattr(self.trader.brain.state, 'confidence', 50.0) if hasattr(self.trader, 'brain') and self.trader.brain else 50.0
                )
                  
                decision = decision_output.get("decision", {})
                if str(decision.get("action", "HOLD")).upper() == "HOLD":
                    fallback_decision = await self._router_fallback_decision(state)
                    if isinstance(fallback_decision, dict):
                        decision = fallback_decision
                action_type = decision.get("action", "HOLD").upper()
                
                if action_type in ["BUY", "SELL"]:
                    qty = decision.get("quantity", decision.get("qty", 0))
                    price = decision.get("price", state.price)
                    if qty > 0:
                        action_pkt.has_order = True
                        action_pkt.order = OrderPayload(
                            symbol="000001",
                            price=float(price),
                            quantity=int(qty),
                            side=action_type.lower(),
                            order_type="LIMIT"
                        )
                
                # 推送至主节点
                payload = action_pkt.model_dump_json()
                await self.push_socket.send_string(payload)
                
                logger.debug(f"[{self.agent_id}] Sent action for STEP {state.step}")
                
        except asyncio.CancelledError:
            logger.info(f"[{self.agent_id}] Shutting down node.")
        finally:
            self.sub_socket.close()
            self.push_socket.close()
