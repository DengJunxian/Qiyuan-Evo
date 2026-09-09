"""
基于文件系统的异步命令-响应 IPC 通道。

设计目标：
1. 将慢速 LLM 推理侧与快速撮合侧做进程级物理隔离。
2. 通过落盘队列提供天然缓冲，避免瞬时高峰将撮合进程阻塞。
3. 保持实现简单，默认不依赖 Redis，方便本地与测试环境直接运行。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> float:
    """统一生成时间戳，便于测试时替换。"""
    return time.time()


@dataclass(slots=True)
class IPCEnvelope:
    """标准 IPC 包结构。"""

    message_id: str
    channel: str
    message_type: str
    payload: Dict[str, Any]
    created_at: float = field(default_factory=_utc_now)
    correlation_id: Optional[str] = None
    source: str = "unknown"
    target: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        channel: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        source: str = "unknown",
        target: str = "unknown",
    ) -> "IPCEnvelope":
        return cls(
            message_id=str(uuid.uuid4()),
            channel=channel,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            source=source,
            target=target,
        )


class FileSystemIPC:
    """
    文件系统 IPC 实现。

    目录布局：
    - commands/pending: 待消费命令
    - commands/processing: 已被某个引擎实例锁定的命令
    - responses: 响应结果
    - dead_letter: 无法解析或处理失败的消息
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.commands_pending = self.root / "commands" / "pending"
        self.commands_processing = self.root / "commands" / "processing"
        self.responses = self.root / "responses"
        self.dead_letter = self.root / "dead_letter"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for folder in (
            self.commands_pending,
            self.commands_processing,
            self.responses,
            self.dead_letter,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def send_command(
        self,
        message_type: str,
        payload: Dict[str, Any],
        *,
        source: str = "controller",
        target: str = "engine",
    ) -> IPCEnvelope:
        """
        将命令写入待消费队列。

        文件名带时间戳前缀，保证近似 FIFO；消息体保留唯一 message_id 便于响应关联。
        """
        envelope = IPCEnvelope.create(
            channel="command",
            message_type=message_type,
            payload=payload,
            source=source,
            target=target,
        )
        file_name = f"{envelope.created_at:.6f}-{envelope.message_id}.json"
        self._atomic_write_json(self.commands_pending / file_name, envelope.to_dict())
        return envelope

    def claim_next_command(self) -> Optional[tuple[IPCEnvelope, Path]]:
        """
        由撮合引擎进程轮询并抢占下一条命令。

        使用原子 rename 将消息从 pending 移入 processing，实现轻量级文件锁。
        """
        pending_files = sorted(self.commands_pending.glob("*.json"))
        for pending_path in pending_files:
            processing_path = self.commands_processing / pending_path.name
            try:
                os.replace(pending_path, processing_path)
            except FileNotFoundError:
                continue

            try:
                raw = self._load_json(processing_path)
                return IPCEnvelope(**raw), processing_path
            except Exception:
                self._move_to_dead_letter(processing_path)
        return None

    def acknowledge_command(self, processing_path: Path) -> None:
        """引擎消费成功后删除 processing 文件。"""
        if processing_path.exists():
            processing_path.unlink()

    def reject_command(self, processing_path: Path, reason: str) -> None:
        """
        处理失败的命令转入死信队列，便于事后排障。
        """
        try:
            raw = self._load_json(processing_path)
        except Exception:
            raw = {"reason": "unreadable_command"}
        raw["error"] = reason
        target = self.dead_letter / processing_path.name
        self._atomic_write_json(target, raw)
        if processing_path.exists():
            processing_path.unlink()

    def _move_to_dead_letter(self, source_path: Path) -> None:
        target = self.dead_letter / source_path.name
        if source_path.exists():
            os.replace(source_path, target)

    def send_response(
        self,
        *,
        correlation_id: str,
        message_type: str,
        payload: Dict[str, Any],
        source: str = "engine",
        target: str = "controller",
    ) -> IPCEnvelope:
        """撮合进程回写响应。"""
        envelope = IPCEnvelope.create(
            channel="response",
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            source=source,
            target=target,
        )
        path = self.responses / f"{correlation_id}.json"
        self._atomic_write_json(path, envelope.to_dict())
        return envelope

    def wait_for_response(
        self,
        correlation_id: str,
        *,
        timeout: float = 5.0,
        poll_interval: float = 0.02,
        consume: bool = True,
    ) -> Optional[IPCEnvelope]:
        """
        控制侧等待响应。

        这里采用短轮询而非文件监听，原因是跨平台兼容性更稳定，且当前消息体很小。
        """
        deadline = _utc_now() + timeout
        path = self.responses / f"{correlation_id}.json"
        while _utc_now() < deadline:
            if path.exists():
                try:
                    raw = self._load_json(path)
                except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError):
                    time.sleep(poll_interval)
                    continue
                if consume:
                    try:
                        path.unlink(missing_ok=True)
                    except PermissionError:
                        pass
                return IPCEnvelope(**raw)
            time.sleep(poll_interval)
        return None

    def list_pending_commands(self) -> List[Path]:
        return sorted(self.commands_pending.glob("*.json"))
