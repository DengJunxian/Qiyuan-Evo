"""
仿真调度器与独立撮合子进程。

核心职责：
1. 主进程仅负责发送交易意图与时间推进命令，不直接触碰撮合状态机。
2. 子进程维护独立的 LOB 状态、离散时间和订单执行缓冲。
3. 通过 simulation_ipc 的异步命令响应机制完成跨进程通讯。
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
import time
import csv
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from core.time_manager import SimulationClock
from core.types import Order, OrderSide, OrderType
from core.exchange.trade_tape import TradeTape

from simulation_ipc import FileSystemIPC, IPCEnvelope

try:
    from core.exchange.order_book_cpp import OrderBookCPP, _civitas_lob

    if _civitas_lob is None:
        raise ImportError("C++ extension _civitas_lob not available")

    ORDER_BOOK_CLASS = OrderBookCPP
except Exception:
    from core.exchange.order_book import OrderBook as ORDER_BOOK_CLASS


@dataclass(slots=True)
class BufferedIntent:
    """
    LLM 侧生成的交易意图。

    这里不是“订单已提交”，而是“可在未来某个离散时间点被引擎消费的候选意图”。
    """

    intent_id: str
    agent_id: str
    side: str
    quantity: int
    price: float
    symbol: str = "A_SHARE_IDX"
    order_type: str = "limit"
    intent_type: str = "order"
    activate_step: Optional[int] = None
    expire_step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_order(self, *, timestamp: float) -> Order:
        return Order(
            symbol=self.symbol,
            price=float(self.price),
            quantity=int(self.quantity),
            side=OrderSide(self.side.lower()),
            order_type=OrderType(self.order_type.lower()),
            agent_id=self.agent_id,
            timestamp=timestamp,
            order_id=self.intent_id,
            metadata=dict(self.metadata or {}),
        )


@dataclass(slots=True)
class ExogenousBackdropPoint:
    step: int
    price: float
    volume: float


class OASISRunner:
    """
    独立子进程内运行的极速撮合执行器。

    设计原则：
    - 所有订单执行都必须由 advance_time 驱动，避免 LLM I/O 直接阻塞撮合主循环。
    - submit_intent 仅负责入缓冲，不立即触发成交。
    """

    def __init__(self, ipc_root: str | Path, symbol: str = "A_SHARE_IDX", prev_close: float = 3000.0):
        self.ipc = FileSystemIPC(ipc_root)
        self.clock = SimulationClock()
        self.order_book = ORDER_BOOK_CLASS(symbol=symbol, prev_close=prev_close)
        self.symbol = str(symbol)
        self.prev_close = float(prev_close)
        self.trade_tape = TradeTape(symbol=symbol, seed=42, config_hash="simulation_runner_v1")
        self.intent_buffer: Deque[BufferedIntent] = deque()
        self._running = True

    def run(self, idle_sleep: float = 0.01) -> None:
        while self._running:
            claimed = self.ipc.claim_next_command()
            if claimed is None:
                time.sleep(idle_sleep)
                continue

            envelope, processing_path = claimed
            try:
                payload = self._handle_command(envelope)
                self.ipc.send_response(
                    correlation_id=envelope.message_id,
                    message_type=f"{envelope.message_type}_ack",
                    payload=payload,
                )
                self.ipc.acknowledge_command(processing_path)
            except Exception as exc:
                self.ipc.send_response(
                    correlation_id=envelope.message_id,
                    message_type=f"{envelope.message_type}_error",
                    payload={"error": str(exc)},
                )
                self.ipc.reject_command(processing_path, str(exc))

    def _handle_command(self, envelope: IPCEnvelope) -> Dict[str, Any]:
        if envelope.message_type == "submit_intent":
            intent = BufferedIntent(**envelope.payload)
            self.intent_buffer.append(intent)
            return {
                "accepted": True,
                "buffer_size": len(self.intent_buffer),
                "intent_id": intent.intent_id,
                "current_step": self.clock.ticks,
            }

        if envelope.message_type == "advance_time":
            steps = int(envelope.payload.get("steps", 1))
            return self._advance_time(steps)

        if envelope.message_type == "get_snapshot":
            return self._snapshot()

        if envelope.message_type == "shutdown":
            self._running = False
            return {"stopped": True, "current_step": self.clock.ticks}

        raise ValueError(f"Unsupported command type: {envelope.message_type}")

    def _advance_time(self, steps: int) -> Dict[str, Any]:
        """
        推进离散时间并消费当前时点可执行的全部意图。

        之所以按 step 逐个推进，而不是一次性跳过，是为了保证未来可扩展到:
        - 逐步政策注入
        - 逐步外生事件
        - 同步 GraphRAG / Diagnostic Agent 探针
        """
        all_trades: List[Dict[str, Any]] = []
        all_tape: List[Dict[str, Any]] = []
        executed_intents: List[str] = []
        cancelled_orders: List[Dict[str, Any]] = []

        for _ in range(steps):
            self.clock.tick()
            current_step = self.clock.ticks

            ready_intents: List[BufferedIntent] = []
            deferred: Deque[BufferedIntent] = deque()

            while self.intent_buffer:
                intent = self.intent_buffer.popleft()
                if intent.expire_step is not None and current_step > intent.expire_step:
                    continue
                if intent.activate_step is None or intent.activate_step <= current_step:
                    ready_intents.append(intent)
                else:
                    deferred.append(intent)

            self.intent_buffer = deferred

            for intent in ready_intents:
                intent_kind = str(getattr(intent, "intent_type", "order") or "order").lower()
                side = str(getattr(intent, "side", "")).lower()
                if intent_kind == "cancel" or side == "cancel":
                    target_order_id = str(intent.metadata.get("target_order_id", "") or "")
                    cancelled = bool(target_order_id) and bool(self.order_book.cancel_order(target_order_id))
                    executed_intents.append(intent.intent_id)
                    if cancelled:
                        cancelled_orders.append(
                            {
                                "agent_id": intent.agent_id,
                                "target_order_id": target_order_id,
                                "timestamp": self.clock.timestamp,
                            }
                        )
                    continue

                order = intent.to_order(timestamp=self.clock.timestamp)
                trades = self.order_book.add_order(order)
                executed_intents.append(intent.intent_id)
                for trade in trades:
                    tape_record = self.trade_tape.append_trade(
                        trade,
                        tick=int(self.clock.ticks),
                        trading_day=time.strftime("%Y-%m-%d", time.localtime(float(self.clock.timestamp))),
                        phase="continuous",
                        market_timestamp=float(self.clock.timestamp),
                        metadata={
                            "source": "simulation_runner",
                            "intent_id": intent.intent_id,
                            "intent_type": intent.intent_type,
                            "aggressor_side": intent.side,
                            "seed": 42,
                            "symbol": self.symbol,
                            **dict(intent.metadata or {}),
                        },
                    )
                    all_trades.append(
                        {
                            "trade_id": trade.trade_id,
                            "price": trade.price,
                            "quantity": trade.quantity,
                            "buyer_agent_id": trade.buyer_agent_id,
                            "seller_agent_id": trade.seller_agent_id,
                            "maker_id": trade.maker_id,
                            "taker_id": trade.taker_id,
                            "timestamp": trade.timestamp,
                        }
                    )
                    all_tape.append(tape_record.to_dict())

        snapshot = self._snapshot()
        tape_last_price = self.trade_tape.last_price(float(snapshot.get("last_price") or self.prev_close))
        snapshot.update(
            {
                "executed_intents": executed_intents,
                "trade_count": len(all_trades),
                "trades": all_trades,
                "trade_tape": all_tape,
                "canonical_trade_tape": all_tape,
                "tape_last_price": float(tape_last_price),
                "trade_tape_hash": self.trade_tape.hash(),
                "price_source": "trade_tape" if all_tape else "previous_trade_tape_price",
                "cancel_count": len(cancelled_orders),
                "canceled_orders": cancelled_orders,
            }
        )
        return snapshot

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "current_step": self.clock.ticks,
            "buffer_size": len(self.intent_buffer),
            "last_price": getattr(self.order_book, "last_price", None),
            "best_bid": self.order_book.get_best_bid(),
            "best_ask": self.order_book.get_best_ask(),
            "depth": self.order_book.get_depth(5),
            "activity_stats": self.order_book.get_activity_stats() if hasattr(self.order_book, "get_activity_stats") else {},
            "trade_tape_size": len(self.trade_tape.records),
            "trade_tape_hash": self.trade_tape.hash(),
        }


def _runner_entry(ipc_root: str, symbol: str, prev_close: float) -> None:
    runner = OASISRunner(ipc_root=ipc_root, symbol=symbol, prev_close=prev_close)
    runner.run()


class SimulationRunner:
    """
    主进程侧的调度器门面。

    该类不保存撮合核心状态，只负责：
    - 启停子进程
    - 向子进程发送命令
    - 等待并解析响应
    """

    def __init__(
        self,
        ipc_root: str | Path | None = None,
        *,
        symbol: str = "A_SHARE_IDX",
        prev_close: float = 3000.0,
        response_timeout: float = 5.0,
    ):
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        if ipc_root is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="civitas-sim-ipc-")
            ipc_root = self._temp_dir.name

        self.ipc_root = Path(ipc_root)
        self.ipc = FileSystemIPC(self.ipc_root)
        self.symbol = symbol
        self.prev_close = prev_close
        self.response_timeout = response_timeout
        self.process: Optional[mp.Process] = None
        self._exogenous_backdrop: List[ExogenousBackdropPoint] = []

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return

        self.process = mp.Process(
            target=_runner_entry,
            args=(str(self.ipc_root), self.symbol, self.prev_close),
            daemon=True,
            name="OASISRunnerProcess",
        )
        self.process.start()

    def stop(self) -> Dict[str, Any]:
        response: Dict[str, Any] = {}
        if self.process is None:
            return response

        if self.process.is_alive():
            response = self._request("shutdown", {})
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)

        self.process = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        return response

    def submit_intent(self, intent: BufferedIntent) -> Dict[str, Any]:
        return self._request("submit_intent", asdict(intent))

    def submit_batch(self, intents: List[BufferedIntent], *, advance_steps: int = 1) -> Dict[str, Any]:
        submitted = 0
        accepted = 0
        max_buffer_size = 0
        acks: List[Dict[str, Any]] = []
        for intent in intents:
            ack = self.submit_intent(intent)
            acks.append(ack)
            submitted += 1
            if bool(ack.get("accepted", False)):
                accepted += 1
            max_buffer_size = max(max_buffer_size, int(ack.get("buffer_size", 0) or 0))
        snapshot = self.advance_time(max(1, int(advance_steps)))
        snapshot["submitted_count"] = int(submitted)
        snapshot["accepted_count"] = int(accepted)
        snapshot["max_buffer_size"] = int(max_buffer_size)
        snapshot["acks"] = acks
        return snapshot

    def advance_time(self, steps: int = 1) -> Dict[str, Any]:
        return self._request("advance_time", {"steps": steps})

    def get_snapshot(self) -> Dict[str, Any]:
        return self._request("get_snapshot", {})

    def set_exogenous_backdrop(self, series: List[Dict[str, Any]]) -> int:
        backdrop: List[ExogenousBackdropPoint] = []
        for row in series:
            step = int(row.get("step", len(backdrop) + 1))
            price = float(row.get("price", row.get("close", 0.0)) or 0.0)
            volume = float(row.get("volume", row.get("qty", 0.0)) or 0.0)
            if price <= 0:
                continue
            backdrop.append(ExogenousBackdropPoint(step=step, price=price, volume=max(0.0, volume)))
        backdrop.sort(key=lambda item: item.step)
        self._exogenous_backdrop = backdrop
        return len(self._exogenous_backdrop)

    def load_exogenous_backdrop_csv(self, path: str | Path) -> int:
        p = Path(path)
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return self.set_exogenous_backdrop(rows)

    def get_exogenous_backdrop_point(self, step: int) -> Optional[Dict[str, float]]:
        if not self._exogenous_backdrop:
            return None
        target = int(step)
        selected = self._exogenous_backdrop[-1]
        for item in self._exogenous_backdrop:
            if item.step <= target:
                selected = item
            else:
                break
        return {
            "step": float(selected.step),
            "price": float(selected.price),
            "volume": float(selected.volume),
        }

    def _request(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Simulation runner process is not running.")

        envelope = self.ipc.send_command(message_type, payload)
        response = self.ipc.wait_for_response(envelope.message_id, timeout=self.response_timeout)
        if response is None:
            raise TimeoutError(f"Timed out waiting for response to {message_type}.")
        if response.message_type.endswith("_error"):
            raise RuntimeError(response.payload.get("error", f"{message_type} failed"))
        return response.payload
