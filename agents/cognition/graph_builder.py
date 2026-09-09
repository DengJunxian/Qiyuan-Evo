import json
import re
import logging
from typing import List, Dict, Any, Optional, Sequence

logger = logging.getLogger(__name__)

class GraphExtractor:
    """
    使用大模型（通过 ModelRouter）从杂乱的宏观资讯或交易复盘
    提取出标准化的高质量概念三元组。
    """
    def __init__(self, model_router: Any, model_priority: Optional[Sequence[str]] = None):
        self.model_router = model_router
        self.model_priority = list(model_priority or ["glm-4-flashx"])

    def _extract_json_array(self, text: str) -> List[Dict[str, Any]]:
        try:
            # 优先提取 ```json ... ``` 包裹的内容
            match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
                
            # 否则尝试直接找数组格式
            match = re.search(r"(\[.*?\])", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse triplet JSON array: {e}")
        return []

    async def extract_graph(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        异步抽取文本中的三元组逻辑
        """
        if not self.model_router:
            logger.error("GraphExtractor requires a valid model_router.")
            return []

        system_prompt = """你是一个专业的金融认知图谱抽取器。
你需要从提供的文本中，精准提取核心概念、实体及其内在逻辑联系。
尽可能抽象化概念（如“基准利率”、“流动性不足”、“科技股”），而不是具体的数值。
你必须严格输出一个 JSON 对象的数组，不要包含其他额外文本。
格式规范：
[
    {"subject": "主体概念", "predicate": "动作或关系 (如 导致、属于、利好)", "target": "客体概念", "weight": 权重小数(0.1-1.0)}
]
"""
        
        user_prompt = f"请提取以下文本中的核心逻辑三元组:\n{text}"
        if context:
            user_prompt = f"背景:\n{context}\n\n" + user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            content, _, _ = await self.model_router.call_with_fallback(
                messages=messages,
                priority_models=list(self.model_priority),
                timeout_budget=15.0
            )
            
            triplets = self._extract_json_array(content)
            # 数据结构简单清洗
            valid_triplets = []
            for t in triplets:
                if isinstance(t, dict) and "subject" in t and "predicate" in t and "target" in t:
                    t["weight"] = float(t.get("weight", 1.0))
                    valid_triplets.append(t)
            return valid_triplets

        except Exception as e:
            logger.error(f"GraphExtraction Error: {e}")
            return []
