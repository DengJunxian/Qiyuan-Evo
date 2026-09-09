import asyncio
import logging
import json
from typing import List

from core.ipc.runtime_compat import ensure_zmq_asyncio_compatibility

ensure_zmq_asyncio_compatibility()

import zmq
import zmq.asyncio

from core.ipc.message_types import MarketStatePacket, AgentActionPacket

logger = logging.getLogger(__name__)

class IPCEngineServer:
    """
    负责在 IPC 模式下替代传统 for 循环收集订单的主节点通信组件。
    - 使用 PUB 广播全局市场快照
    - 使用 PULL 汇聚不同机器、不同进程推回的群体交易动作
    """
    def __init__(self, pub_port: int = 5555, pull_port: int = 5556):
        self.pub_port = pub_port
        self.pull_port = pull_port
        
        self.context = zmq.asyncio.Context.instance()
        
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{self.pub_port}")
        
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{self.pull_port}")
        
        logger.info(f"[IPC EngineServer] Bound PUB(tcp://*:{self.pub_port}) and PULL(tcp://*:{self.pull_port})")

    async def broadcast_market_state(self, state: MarketStatePacket):
        """主沙箱向所有微服务节点发出滴答广播"""
        payload = state.model_dump_json()
        await self.pub_socket.send_string(payload)
        logger.debug(f"[IPC EngineServer] Broadcasted STEP {state.step}")

    async def collect_agent_actions(self, collect_window: float = 2.0, expected_count: int = 0) -> List[AgentActionPacket]:
        """
        在指定时间窗口内收集从各个节点 PUSH 回来的动作。
        如果收集到足够的 expected_count 则提前返回（0 表示严格等待至窗口期结束）。
        """
        collected = []
        start_time = asyncio.get_event_loop().time()
        
        logger.debug(f"[IPC EngineServer] Start collecting actions for {collect_window} seconds...")
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= collect_window:
                break
                
            # 剩余等待时间（毫秒）
            remains_ms = max(int((collect_window - elapsed) * 1000), 1)
            
            # 使用 Poller 监听是否有数据到达，防止死锁
            events = await self.pull_socket.poll(timeout=remains_ms)
            if events:
                msg_str = await self.pull_socket.recv_string()
                try:
                    data = json.loads(msg_str)
                    packet = AgentActionPacket(**data)
                    collected.append(packet)
                    
                    if expected_count > 0 and len(collected) >= expected_count:
                        break
                except Exception as e:
                    logger.error(f"[IPC EngineServer] Invalid packet received: {e[:200]}")
            else:
                # 超时无数据，跳出
                break
                
        logger.debug(f"[IPC EngineServer] Collected {len(collected)} actions.")
        return collected

    def close(self):
        self.pub_socket.close()
        self.pull_socket.close()
