"""
反事实推理模块 (Counterfactual Reasoning)

支持"平行宇宙"实验：在同一历史时刻分叉两个仿真环境，
比较不同政策下的市场表现差异。

本次重构：从"伪仿真"升级为真实的 Agent-Based 平行宇宙运行。
"""

import copy
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

# 引用真实模型
from core.mesa.civitas_model import CivitasModel
from core.policy import PolicyManager

@dataclass
class UniverseConfig:
    """宇宙配置快照：用于重建等效的仿真环境"""
    seed: int
    n_agents: int
    panic_ratio: float
    quant_ratio: float
    initial_price: float
    # 可以添加更多参数以确保一致性

@dataclass
class CounterfactualResult:
    """反事实实验结果 (增强版)"""
    universe_a_name: str
    universe_b_name: str
    fork_point: int  # 分叉点（K线索引）
    
    # K 线数据
    candles_a: pd.DataFrame
    candles_b: pd.DataFrame
    
    # 社交/网络数据
    network_stats_a: List[Dict[str, Any]]
    network_stats_b: List[Dict[str, Any]]
    
    # 对比指标
    final_price_a: float
    final_price_b: float
    price_divergence: float  # 价格分歧度
    
    volatility_a: float
    volatility_b: float
    
    cumulative_return_a: float
    cumulative_return_b: float
    
    # 推理分析
    analysis: str = ""
    

class ParallelUniverseEngine:
    """
    平行宇宙仿真引擎 (Real Agent-Based)
    
    核心功能：
    1. 接受基础模型配置
    2. 实例化两个独立的 CivitasModel (Model A & Model B)
    3. 确保随机种子一致
    4. 注入不同 Policy
    5. 并发运行并收集数据
    """
    
    def __init__(self):
        pass
    
    def _create_mirrored_models(self, config: UniverseConfig) -> Tuple[CivitasModel, CivitasModel]:
        """
        创建两个初始状态完全一致的镜像世界
        
        Args:
            config: 宇宙配置
            
        Returns:
            (model_a, model_b)
        """

        model_a = CivitasModel(
            n_agents=config.n_agents,
            panic_ratio=config.panic_ratio,
            quant_ratio=config.quant_ratio,
            initial_price=config.initial_price,
            seed=config.seed
        )
        

        model_b = CivitasModel(
            n_agents=config.n_agents,
            panic_ratio=config.panic_ratio,
            quant_ratio=config.quant_ratio,
            initial_price=config.initial_price,
            seed=config.seed
        )
        
        return model_a, model_b
    
    def _apply_policy_to_model(self, model: CivitasModel, policy_config: Optional[Dict[str, Any]]):
        """将政策配置应用到模型"""
        if not policy_config:
            return
            
        pm = model.market_manager.policy_manager
        
        # 处理印花税
        if "tax_rate" in policy_config:
            tax = pm.policies.get("tax")
            if tax:
                tax.rate = policy_config.get("tax_rate", 0.001)
                tax.active = True
                
        # 处理熔断
        if "circuit_breaker" in policy_config:
            cb = pm.policies.get("circuit_breaker")
            cb_conf = policy_config["circuit_breaker"]
            if cb:
                cb.threshold_pct = cb_conf.get("threshold", 0.10)
                cb.active = cb_conf.get("active", True)
                
        # 处理初始新闻/舆情注入
        if "initial_news" in policy_config:
            news_item = policy_config["initial_news"]
            model.market_manager.add_news(news_item)
            
    async def run_counterfactual_experiment(
        self,
        base_config: UniverseConfig,
        policy_a: Optional[Dict[str, Any]],
        policy_b: Dict[str, Any],
        n_steps: int = 30,
        policy_a_name: str = "Baseline",
        policy_b_name: str = "Treatment"
    ) -> CounterfactualResult:
        """
        运行反事实实验 (真实仿真)
        
        Args:
            base_config: 基础环境配置
            policy_a: 宇宙A的政策 (通常为 None 或基准)
            policy_b: 宇宙B的政策 (实验组)
            n_steps: 运行步数
            
        Returns:
            实验结果对比
        """
        # 1. 创建镜像世界
        model_a, model_b = self._create_mirrored_models(base_config)
        
        # 2. 注入政策
        self._apply_policy_to_model(model_a, policy_a)
        self._apply_policy_to_model(model_b, policy_b)
        
        # 3. 并发运行
        # 使用 asyncio.gather 并行跑两个模型的 step 循环
        async def run_model(model: CivitasModel, steps: int):
            for _ in range(steps):
                await model.async_step()
            return model
            
        print(f"Starting Parallel Universe Simulation ({n_steps} steps)...")
        tasks = [
            run_model(model_a, n_steps),
            run_model(model_b, n_steps)
        ]
        
        await asyncio.gather(*tasks)
        print("Simulation Completed.")
        
        # 4. 收集数据
        def get_candles_df(model: CivitasModel) -> pd.DataFrame:
            # 转换 Candle 对象列表为 DataFrame
            data = []
            for c in model.market_manager.sim_candles:
                data.append({
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                })
            if not data:

                return pd.DataFrame({"close": [base_config.initial_price]})
            return pd.DataFrame(data)
            
        candles_a = get_candles_df(model_a)
        candles_b = get_candles_df(model_b)
        
        # 获取网络统计

        net_stats_a = model_a.diffusion.history if hasattr(model_a.diffusion, 'history') else []
        net_stats_b = model_b.diffusion.history if hasattr(model_b.diffusion, 'history') else []
        
        # 计算统计指标
        price_a = candles_a.iloc[-1]["close"]
        price_b = candles_b.iloc[-1]["close"]
        
        ret_a = (price_a - base_config.initial_price) / base_config.initial_price
        ret_b = (price_b - base_config.initial_price) / base_config.initial_price
        
        vol_a = candles_a["close"].pct_change().std() * np.sqrt(252) if len(candles_a) > 1 else 0
        vol_b = candles_b["close"].pct_change().std() * np.sqrt(252) if len(candles_b) > 1 else 0

        # 5. 构建结果
        result = CounterfactualResult(
            universe_a_name=policy_a_name,
            universe_b_name=policy_b_name,
            fork_point=0, # 从头开始跑
            candles_a=candles_a,
            candles_b=candles_b,
            network_stats_a=net_stats_a,
            network_stats_b=net_stats_b,
            final_price_a=price_a,
            final_price_b=price_b,
            price_divergence=abs(price_a - price_b) / base_config.initial_price,
            volatility_a=vol_a,
            volatility_b=vol_b,
            cumulative_return_a=ret_a,
            cumulative_return_b=ret_b,
        )
        
        # 6. 生成分析报告
        result.analysis = self._generate_analysis(result, policy_b_name)
        
        return result
    
    def _generate_analysis(self, result: CounterfactualResult, policy: str) -> str:
        """生成差异分析报告"""
        lines = [
            "## 平行宇宙反事实推演报告",
            "",
            f"**政策干预**: {policy}",
            "",
            "### 市场表现对比",
            f"| 指标 | {result.universe_a_name} | {result.universe_b_name} |",
            "|------|------------------------|------------------------|",
            f"| 最终价格 | {result.final_price_a:.2f} | {result.final_price_b:.2f} |",
            f"| 累计收益 | {result.cumulative_return_a:+.2%} | {result.cumulative_return_b:+.2%} |",
            f"| 年化波动率 | {result.volatility_a:.2%} | {result.volatility_b:.2%} |",
            "",
            f"**价格分歧度**: {result.price_divergence:.2%}",
            "",
            "### 社会网络影响",
        ]
        
        # 简单的网络对比分析
        final_inf_a = result.network_stats_a[-1].get('infected', 0) if result.network_stats_a else 0
        final_inf_b = result.network_stats_b[-1].get('infected', 0) if result.network_stats_b else 0
        
        lines.append(f"- **{result.universe_a_name}** 最终恐慌感染人数: {final_inf_a}")
        lines.append(f"- **{result.universe_b_name}** 最终恐慌感染人数: {final_inf_b}")
        
        diff = final_inf_b - final_inf_a
        if diff > 10:
             lines.append(f"⚠️ 警告：政策导致恐慌情绪显著扩散 (+{diff} 人)")
        elif diff < -10:
             lines.append(f"✅ 正面：政策有效抑制了恐慌传播 ({diff} 人)")
        
        return "\n".join(lines)



