import json
import os
import logging
from typing import List, Optional
from datetime import datetime

from data_flywheel.schemas import SeedEvent
from core.runtime_paths import resolve_runtime_path

logger = logging.getLogger(__name__)


class SeedStore:
    """
    种子事件存储模块
    
    负责将处理完毕的 SeedEvent 以 JSONL 格式持久化，
    支持按时间、来源过滤查询，供回放或沙箱摄入使用。
    """

    def __init__(self, file_path: str = "data/seed_events.jsonl"):
        if str(file_path) == "data/seed_events.jsonl":
            self.file_path = str(resolve_runtime_path(file_path, env_var="CIVITAS_SEED_STORE_PATH"))
        else:
            self.file_path = file_path
        self._ensure_dir()

    def _ensure_dir(self):
        """确保存储路径中的目录结构存在"""
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory for SeedStore: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")

    def append(self, event: SeedEvent) -> bool:
        """向文件尾部追加单条事件"""
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                # 写入不带换行的 JSON 字符串并手动添加换行符
                f.write(event.to_json(indent=None) + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to append event to store: {e}")
            return False

    def append_batch(self, events: List[SeedEvent]) -> int:
        """批量追加事件，返回成功写入的数量"""
        count = 0
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                for event in events:
                    f.write(event.to_json(indent=None) + "\n")
                    count += 1
            return count
        except Exception as e:
            logger.error(f"Failed to append batch to store: {e}")
            return count

    def read_latest(self, n: int = 10, min_impact: Optional[str] = None) -> List[SeedEvent]:
        """
        读取最近写入的 N 条事件。
        如果指定了 min_impact，则过滤掉影响过小的事件。
        """
        if not os.path.exists(self.file_path):
            return []

        events = []
        impact_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        min_level = impact_order.get(min_impact, 0) if min_impact else 0

        try:
            # 对于大型文件，如果需要极致性能可以考虑从末尾向前读取。
            # 这里为简单起见读取全部内容后取切片。
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    event = SeedEvent.from_json(line)
                    event_level = impact_order.get(event.impact_level, 0)
                    
                    if min_impact and event_level < min_level:
                        continue
                        
                    events.append(event)
                    if len(events) >= n:
                        break
                except Exception as e:
                    logger.warning(f"Failed to parse line in SeedStore: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to read from SeedStore: {e}")

        # events 中是从新到旧的顺序
        return events

