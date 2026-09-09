"""
llama.cpp 本地推理后端

使用 llama-cpp-python 运行 GGUF 量化模型。
适用于 Standard 模式 (8GB VRAM)。

推荐模型:
- Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (~1GB)
- Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2GB)
"""

import os
from typing import Optional, Dict, List

# 条件导入
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class LlamaCppBackend:
    """
    llama.cpp 本地推理后端
    
    特点:
    - 支持 GGUF 量化模型
    - 可在 CPU 或低显存 GPU 上运行
    - 适合 1.5B-7B 小模型
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False
    ):
        """
        初始化 llama.cpp 后端
        
        Args:
            model_path: GGUF 模型文件路径
            n_ctx: 上下文长度
            n_gpu_layers: GPU offload 层数 (-1 = 全部)
            n_threads: CPU 线程数 (None = 自动)
            verbose: 是否打印详细信息
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python 未安装。请运行:\n"
                "pip install llama-cpp-python\n"
                "或 (GPU 加速):\n"
                "CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        
        self._model: Optional[Llama] = None
        self.n_threads = n_threads
        self.verbose = verbose
    
    def _load_model(self) -> Llama:
        """懒加载模型"""
        if self._model is None:
            print(f"[LlamaCpp] 加载模型: {self.model_path}")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose
            )
            print("[LlamaCpp] 模型加载完成")
        return self._model
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样
            stop: 停止词列表
            
        Returns:
            生成的文本
        """
        model = self._load_model()
        
        try:
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["</s>", "[/INST]", "```"],
                echo=False
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            return f"[LlamaCpp Error] {e}"
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Chat 格式生成
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            
        Returns:
            助手回复
        """
        model = self._load_model()
        
        try:
            output = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[LlamaCpp Error] {e}"
    
    def unload(self) -> None:
        """释放模型"""
        if self._model is not None:
            del self._model
            self._model = None
            print("[LlamaCpp] 模型已释放")


def get_recommended_models() -> Dict[str, str]:
    """获取推荐的 GGUF 模型列表"""
    return {
        "qwen-1.5b": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen-3b": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "qwen-7b": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
    }


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("LlamaCpp Backend 测试")
    print("=" * 60)
    
    if not LLAMA_CPP_AVAILABLE:
        print("llama-cpp-python 未安装")
        print("安装命令: pip install llama-cpp-python")
    else:
        print("llama-cpp-python 可用")
        print("\n推荐模型:")
        for name, url in get_recommended_models().items():
            print(f"  {name}: {url}")
