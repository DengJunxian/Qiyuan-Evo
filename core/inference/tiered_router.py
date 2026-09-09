"""
分层推理路由器

根据 Agent 类型和市场状态，智能路由到不同推理后端。
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from core.inference.config import InferenceConfig, InferenceMode, AgentTier


@dataclass
class InferenceRequest:
    """推理请求"""
    agent_id: str
    agent_type: str
    prompt: str
    market_state: Dict[str, Any]
    priority: int = 0


@dataclass
class InferenceResult:
    """推理结果"""
    agent_id: str
    content: str
    tier_used: AgentTier
    latency_ms: float
    tokens_generated: int = 0
    model_used: str = ""


class TieredRouter:
    """
    分层推理路由器
    
    根据配置的 mode 和请求特征，将推理请求路由到合适的后端：
    
    - LITE 模式: 所有请求 → API
    - STANDARD 模式: 散户 → 本地 1.5B, 其他 → API
    - ENTERPRISE 模式: 散户 → vLLM 7B, 机构/极端行情 → vLLM 32B 或 API
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig.from_env()
        
        # 后端实例 (懒加载)
        self._api_backend: Optional[Any] = None
        self._local_backend: Optional[Any] = None
        self._vllm_backend: Optional[Any] = None
        
        # 统计
        self.stats = {
            "tier_1_calls": 0,
            "tier_2_calls": 0,
            "tier_rule_calls": 0,
            "total_latency_ms": 0.0
        }
    
    def _init_backends(self) -> None:
        """懒加载推理后端"""
        if self.config.mode == InferenceMode.LITE:
            self._init_api_backend()
            
        elif self.config.mode == InferenceMode.STANDARD:
            self._init_api_backend()
            self._init_local_backend()
            
        elif self.config.mode == InferenceMode.ENTERPRISE:
            self._init_api_backend()
            self._init_vllm_backend()
    
    def _init_api_backend(self) -> None:
        """初始化 API 后端"""
        if self._api_backend is None:
            from core.inference.api_backend import APIBackend
            self._api_backend = APIBackend(
                api_key=self.config.api_key,
                base_url=self.config.api_base_url
            )
    
    def _init_local_backend(self) -> None:
        """初始化本地 llama.cpp 后端"""
        if self._local_backend is None and self.config.local_model_path:
            try:
                from core.inference.llama_cpp_backend import LlamaCppBackend
                self._local_backend = LlamaCppBackend(
                    model_path=self.config.local_model_path,
                    n_ctx=self.config.n_ctx,
                    n_gpu_layers=self.config.n_gpu_layers
                )
            except Exception as e:
                print(f"[Warning] 无法加载本地模型: {e}, 回退到 API")
    
    def _init_vllm_backend(self) -> None:
        """初始化 vLLM 后端"""
        if self._vllm_backend is None:
            try:
                from core.inference.vllm_backend import VLLMBackend
                self._vllm_backend = VLLMBackend(
                    model_name=self.config.vllm_model_name,
                    tensor_parallel_size=self.config.vllm_tensor_parallel,
                    gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                    quantization=self.config.vllm_quantization
                )
            except Exception as e:
                print(f"[Warning] 无法初始化 vLLM: {e}, 回退到 API")
    
    def determine_tier(self, request: InferenceRequest) -> AgentTier:
        """
        确定请求应该使用的推理层级
        
        规则:
        1. 量化 Agent → 纯规则 (无 LLM)
        2. 机构 Agent → Tier 2 (API)
        3. panic_level > threshold → Tier 2
        4. 其他 → Tier 1 (本地)
        """
        agent_type = request.agent_type.lower()
        panic_level = request.market_state.get("panic_level", 0.5)
        
        # 量化 智能体 使用规则引擎
        if agent_type == "quant":
            return AgentTier.TIER_RULE
        
        # 机构 智能体 始终使用 第二层
        if agent_type == "institutional" and self.config.institutional_always_tier_2:
            return AgentTier.TIER_2
        
        # 极端行情使用 第二层
        if panic_level > self.config.tier_2_threshold:
            return AgentTier.TIER_2
        
        # 其他使用 第一层
        return AgentTier.TIER_1
    
    async def call_with_fallback(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        [Async] 带自动降级的异步批量调用
        由于 TieredRouter 内部逻辑主要是同步 (调用 APIBackend.generate)，
        这里使用 run_in_executor 将其放入线程池执行，模拟异步 IO。
        
        Args:
            requests: 推理请求列表
            
        Returns:
            List[InferenceResult]
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        # 简单实现：并发执行这一批请求
        # 注意: 如果 request 数量很大，建议分批或控制 semaphore
        tasks = []
        for req in requests:
            # 将每个 route 调用包装到线程池
            # 注意：路由调用内部会同步访问接口，所以需要线程池
            tasks.append(loop.run_in_executor(None, self.route, req))
            
        results = await asyncio.gather(*tasks)
        return list(results)
    
    def route(self, request: InferenceRequest) -> InferenceResult:
        """
        路由并执行推理请求
        """
        # 确保后端已初始化
        if self._api_backend is None:
            self._init_backends()
        
        tier = self.determine_tier(request)
        start_time = time.time()
        
        try:
            if tier == AgentTier.TIER_RULE:
                result = self._execute_rule(request)
            elif tier == AgentTier.TIER_1:
                result = self._execute_tier_1(request)
            else:
                result = self._execute_tier_2(request)
                
        except Exception as e:
            # 失败时回退
            result = InferenceResult(
                agent_id=request.agent_id,
                content=f"[Error] 推理失败: {e}",
                tier_used=tier,
                latency_ms=0
            )
        
        latency = (time.time() - start_time) * 1000
        result.latency_ms = latency
        
        # 更新统计
        self._update_stats(tier, latency)
        
        return result
    
    def _execute_rule(self, request: InferenceRequest) -> InferenceResult:
        """执行规则引擎推理"""
        self.stats["tier_rule_calls"] += 1
        
        # 简单规则逻辑
        panic = request.market_state.get("panic_level", 0.5)
        trend = request.market_state.get("trend", "震荡")
        
        if trend == "上涨" and panic < 0.5:
            action = "BUY"
        elif trend == "下跌" or panic > 0.7:
            action = "SELL"
        else:
            action = "HOLD"
        
        content = f'{{"action": "{action}", "confidence": 0.6, "reasoning": "基于量化规则"}}'
        
        return InferenceResult(
            agent_id=request.agent_id,
            content=content,
            tier_used=AgentTier.TIER_RULE,
            latency_ms=0,
            model_used="rule_engine"
        )
    
    def _execute_tier_1(self, request: InferenceRequest) -> InferenceResult:
        """执行 Tier 1 推理 (本地模型)"""
        self.stats["tier_1_calls"] += 1
        
        # 根据模式选择后端
        if self.config.mode == InferenceMode.STANDARD and self._local_backend:
            content = self._local_backend.generate(request.prompt)
            model_used = "llama.cpp"
        elif self.config.mode == InferenceMode.ENTERPRISE and self._vllm_backend:
            content = self._vllm_backend.generate(request.prompt)
            model_used = "vllm"
        else:
            # Lite 模式或后端不可用，使用 接口
            if self._api_backend is None:
                raise RuntimeError("API 后端未初始化")
            content = self._api_backend.generate(request.prompt)
            model_used = "api_fallback"
        
        return InferenceResult(
            agent_id=request.agent_id,
            content=content,
            tier_used=AgentTier.TIER_1,
            latency_ms=0,
            model_used=model_used
        )
    
    def _execute_tier_2(self, request: InferenceRequest) -> InferenceResult:
        """执行 Tier 2 推理 (云端 API)"""
        self.stats["tier_2_calls"] += 1
        if self._api_backend is None:
            raise RuntimeError("API 后端未初始化")
        content = self._api_backend.generate(request.prompt)
        
        return InferenceResult(
            agent_id=request.agent_id,
            content=content,
            tier_used=AgentTier.TIER_2,
            latency_ms=0,
            model_used="deepseek_api"
        )
    
    def _update_stats(self, tier: AgentTier, latency_ms: float) -> None:
        self.stats["total_latency_ms"] += latency_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = self.stats["tier_1_calls"] + self.stats["tier_2_calls"] + self.stats["tier_rule_calls"]
        return {
            **self.stats,
            "total_calls": total,
            "tier_1_ratio": self.stats["tier_1_calls"] / total if total > 0 else 0,
            "tier_2_ratio": self.stats["tier_2_calls"] / total if total > 0 else 0,
            "avg_latency_ms": self.stats["total_latency_ms"] / total if total > 0 else 0
        }
    
    def batch_route(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """批量路由 (未来可优化为并行)"""
        return [self.route(req) for req in requests]
