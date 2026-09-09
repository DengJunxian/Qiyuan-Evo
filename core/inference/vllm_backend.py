"""
vLLM 高性能推理后端

使用 vLLM 实现高吞吐量推理，支持:
- PagedAttention (内存优化)
- Continuous Batching (低延迟)
- Tensor Parallelism (多卡并行)

适用于 Enterprise 模式 (24GB+ VRAM)。
"""

from typing import Optional, Dict, List, Any

# 条件导入
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


class VLLMBackend:
    """
    vLLM 高性能推理后端
    
    特点:
    - PagedAttention: 解决 KV Cache 碎片化
    - Continuous Batching: 迭代级调度，低延迟
    - 支持 AWQ/GPTQ 量化
    
    要求:
    - NVIDIA GPU (24GB+ VRAM)
    - CUDA 11.8+
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        quantization: Optional[str] = None,
        max_model_len: int = 4096,
        trust_remote_code: bool = True
    ):
        """
        初始化 vLLM 后端
        
        Args:
            model_name: HuggingFace 模型名称
            tensor_parallel_size: 张量并行度 (GPU 数量)
            gpu_memory_utilization: GPU 显存使用率
            quantization: 量化方式 ("awq", "gptq", None)
            max_model_len: 最大序列长度
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM 未安装。请运行:\n"
                "pip install vllm\n"
                "注意: 需要 CUDA 11.8+ 和 24GB+ GPU"
            )
        
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        
        self._llm: Optional[LLM] = None
    
    def _load_model(self) -> LLM:
        """懒加载模型"""
        if self._llm is None:
            print(f"[vLLM] 加载模型: {self.model_name}")
            print(f"[vLLM] 配置: TP={self.tensor_parallel_size}, "
                  f"GPU Mem={self.gpu_memory_utilization:.0%}, "
                  f"Quantization={self.quantization}")
            
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                quantization=self.quantization,
                max_model_len=self.max_model_len,
                trust_remote_code=self.trust_remote_code
            )
            print("[vLLM] 模型加载完成")
        return self._llm
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样
            
        Returns:
            生成的文本
        """
        llm = self._load_model()
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        try:
            outputs = llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            return f"[vLLM Error] {e}"
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        批量生成 (利用 Continuous Batching)
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成文本列表
        """
        llm = self._load_model()
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        try:
            outputs = llm.generate(prompts, sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        except Exception as e:
            return [f"[vLLM Error] {e}"] * len(prompts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "vllm_available": VLLM_AVAILABLE
        }


def check_vllm_requirements() -> Dict[str, Any]:
    """检查 vLLM 运行要求"""
    import torch
    
    result = {
        "vllm_installed": VLLM_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "gpu_memory_gb": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            result["gpu_names"].append(torch.cuda.get_device_name(i))
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            result["gpu_memory_gb"].append(round(mem_gb, 1))
    
    return result


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Backend 检查")
    print("=" * 60)
    
    if not VLLM_AVAILABLE:
        print("vLLM 未安装")
        print("安装命令: pip install vllm")
    else:
        print("vLLM 可用")
    
    print("\n[系统要求检查]")
    try:
        reqs = check_vllm_requirements()
        for k, v in reqs.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  检查失败: {e}")
