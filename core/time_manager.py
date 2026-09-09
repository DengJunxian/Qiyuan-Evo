"""
仿真时间管理器 (SimulationClock)

提供确定性的仿真时间，替代系统时间 (time.time())，确保仿真可复现。
"""

import time
from datetime import datetime, timedelta
from typing import Optional

class SimulationClock:
    """
    仿真时钟
    
    维护当前仿真时间 (Tick 和 DateTime)。
    """
    MODE_STEP_SECONDS = {
        "FAST": 1.0,
        "SMART": 3.0,
        "DEEP": 6.0,
    }

    def __init__(self, start_time: Optional[datetime] = None, mode: str = "SMART"):
        if start_time is None:
            # 默认从 2024-01-01 09:30:00 开始
            self._current_time = datetime(2024, 1, 1, 9, 30, 0)
        else:
            self._current_time = start_time
            
        self._tick_count = 0
        self._mode = "SMART"
        self._time_step = timedelta(seconds=3) # 默认每 Tick 3 秒 (类似 A 股快照)
        self.configure_mode(mode)
        
    @property
    def now(self) -> datetime:
        """获取当前仿真时间 (datetime)"""
        return self._current_time
    
    @property
    def timestamp(self) -> float:
        """获取当前仿真时间戳 (float)"""
        return self._current_time.timestamp()
    
    @property
    def ticks(self) -> int:
        """获取当前 Tick 数"""
        return self._tick_count
        
    def tick(self, steps: int = 1) -> None:
        """推进仿真时间"""
        self._current_time += self._time_step * steps
        self._tick_count += steps
        
    def set_time(self, new_time: datetime) -> None:
        """强制设置时间 (用于跳跃或初始化)"""
        self._current_time = new_time
        
    def set_time_step(self, seconds: float) -> None:
        """设置时间步长"""
        self._time_step = timedelta(seconds=seconds)

    @property
    def time_step_seconds(self) -> float:
        return float(self._time_step.total_seconds())

    @property
    def mode(self) -> str:
        return self._mode

    def configure_mode(self, mode: str) -> None:
        """
        Align clock speed with scheduler/model mode.
        FAST   -> high-frequency stress replay
        SMART  -> default mixed-fidelity simulation
        DEEP   -> slower deliberative simulation
        """
        mode_name = str(mode or "SMART").upper()
        seconds = self.MODE_STEP_SECONDS.get(mode_name, self.MODE_STEP_SECONDS["SMART"])
        self._mode = mode_name
        self._time_step = timedelta(seconds=seconds)

# 全局单例 (可选，但推荐在 Model 中传递实例)
