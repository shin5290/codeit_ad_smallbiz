"""
Prompt Processor Node
한글 사용자 입력 → Z-Image Turbo용 영어 프롬프트 변환 노드

역할:
- PromptTemplateManager를 사용하여 프롬프트 생성
- 스타일 자동 감지 및 반영
- 업종 자동 감지
"""

from typing import Dict, Any, Optional, Literal
from .base import BaseNode
from ..prompt import PromptTemplateManager


class PromptProcessorNode(BaseNode):
    """
    한글 입력을 영어 프롬프트로 변환하는 노드

    Inputs:
        - user_input (str): 한글 사용자 입력
        - style (str, optional): 스타일 힌트 (realistic, semi_realistic, anime)

    Outputs:
        - prompt (str): 생성된 영어 프롬프트
        - detected_style (str): 감지/확정된 스타일
        - industry (str): 감지된 업종

    Example:
        node = PromptProcessorNode()
        result = node.execute({
            "user_input": "카페 신메뉴 딸기라떼 홍보",
            "style": "realistic"
        })
        # result = {
        #     "prompt": "Professional commercial photography...",
        #     "detected_style": "realistic",
        #     "industry": "s1_hot_cooking"
        # }
    """

    def __init__(self, default_style: str = "realistic"):
        """
        Args:
            default_style: 기본 스타일 (style 입력이 없을 때 사용)
        """
        super().__init__("PromptProcessorNode")
        self.default_style = default_style
        self._prompt_manager = None

    @property
    def prompt_manager(self) -> PromptTemplateManager:
        """Lazy initialization of PromptTemplateManager"""
        if self._prompt_manager is None:
            self._prompt_manager = PromptTemplateManager()
        return self._prompt_manager

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        한글 입력 → 영어 프롬프트 생성

        Args:
            inputs: {
                "user_input": str,      # 필수
                "style": str (optional)  # 스타일 힌트
            }

        Returns:
            {
                "prompt": str,           # 생성된 영어 프롬프트
                "detected_style": str,   # 감지/확정된 스타일
                "industry": str          # 감지된 업종
            }
        """
        user_input = inputs["user_input"]
        style = inputs.get("style", self.default_style)

        # PromptTemplateManager를 통한 프롬프트 생성
        result = self.prompt_manager.generate_detailed_prompt(
            user_input=user_input,
            style=style
        )

        # GPT가 감지한 스타일로 업데이트 (유효한 경우만)
        detected_style = result.get("style", style)
        valid_styles = ["realistic", "ultra_realistic", "semi_realistic", "anime"]
        if detected_style not in valid_styles:
            detected_style = style

        return {
            "prompt": result["positive"],
            "detected_style": detected_style,
            "industry": result.get("industry", "unknown"),
            "text_data": result.get("text_overlay", None)  # GPT가 추출한 텍스트 데이터
        }

    def get_required_inputs(self) -> list:
        return ["user_input"]

    def get_output_keys(self) -> list:
        return ["prompt", "detected_style", "industry", "text_data"]
