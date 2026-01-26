"""
Product Layout Analyzer Node
제품 배치 위치 자동 분석 (GPT-4V)

작성자: 이현석

GPT-4V가 배경 이미지를 분석하여:
1. 제품을 배치하기 좋은 위치 결정
2. 적절한 제품 크기(scale) 제안
3. 배경의 빈 공간, 색상 대비 고려
"""

import os
import json
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from src.utils.logging import get_logger
from .base import BaseNode

load_dotenv()
logger = get_logger(__name__)


class ProductLayoutAnalyzerNode(BaseNode):
    """
    제품 배치 분석 노드 (GPT-4V)

    배경 이미지를 분석하여 제품의 최적 배치 위치와 크기를 결정

    Inputs:
        - background (PIL.Image): 배경 이미지 (AI 생성)
        - foreground (PIL.Image): 제품 이미지 (RGBA)
        - context (str): 추가 컨텍스트 (예: "카페 신메뉴", "음식점 이벤트")

    Outputs:
        - position (tuple): (x, y) 제품 배치 위치
        - scale (float): 제품 크기 비율 (0.3 ~ 0.9)
        - reasoning (str): GPT의 배치 결정 이유
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 2):
        super().__init__("ProductLayoutAnalyzerNode")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries

        if not self.api_key:
            logger.warning(f"[{self.node_name}] No OpenAI API key found")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        배경 분석 및 제품 배치 결정

        Returns:
            {
                "position": (x, y),
                "scale": float,
                "reasoning": str
            }
        """
        background = inputs["background"]
        foreground = inputs["foreground"]
        context = inputs.get("context", "")

        logger.info(f"[{self.node_name}] Analyzing background for product placement...")
        logger.info(f"   Background: {background.size}, Foreground: {foreground.size}")
        logger.info(f"   Context: {context or 'None'}")

        # OpenAI API 키 확인
        if not self.api_key:
            logger.warning(f"[{self.node_name}] No API key, using center placement")
            return self._get_default_layout(background.size, foreground.size)

        # GPT-4V 호출 (재시도 포함)
        for attempt in range(self.max_retries + 1):
            try:
                result = self._call_gpt_vision(background, foreground, context)
                return result
            except Exception as e:
                logger.warning(f"GPT-4V call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt == self.max_retries:
                    logger.warning(f"All retries exhausted, using default layout")
                    return self._get_default_layout(background.size, foreground.size)

    def _call_gpt_vision(self, background: Image.Image, foreground: Image.Image, context: str) -> Dict[str, Any]:
        """GPT-4V로 배경 분석 및 배치 결정"""
        client = OpenAI(api_key=self.api_key)

        # 이미지를 base64로 인코딩
        bg_base64 = self._encode_image_to_base64(background)
        fg_base64 = self._encode_image_to_base64(foreground)

        # 프롬프트 생성
        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size

        prompt = f"""You are an expert in product photography and advertising composition.

**Background Image Analysis Task:**
Analyze the provided background image and determine the optimal placement for a product.

**Background Size:** {bg_width}x{bg_height} pixels
**Product Size (before scaling):** {fg_width}x{fg_height} pixels
**Context:** {context or "General product placement"}

**Instructions:**
1. Identify empty/clean areas in the background where the product can be placed
2. Consider visual balance, rule of thirds, and composition principles
3. Ensure the product doesn't cover important background elements
4. Choose a scale that makes the product prominent but not overwhelming

**Return JSON:**
{{
    "position_x": <x coordinate, 0 to {bg_width}>,
    "position_y": <y coordinate, 0 to {bg_height}>,
    "scale": <float between 0.3 and 0.9>,
    "reasoning": "<brief explanation of placement decision>"
}}

**Example reasoning:** "Placed product in the right third to balance the composition. Dark background on the left provides contrast. Scale 0.7 makes product prominent without overwhelming the scene."

Analyze the background and decide the best placement for the product.
"""

        # GPT-4V 호출
        response = client.chat.completions.create(
            model="gpt-4o",  # gpt-4o는 vision 지원
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{bg_base64}",
                                "detail": "low"  # 비용 절감
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{fg_base64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.3
        )

        # 응답 파싱
        content = response.choices[0].message.content

        # JSON 추출 (코드 블록 제거)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        layout_data = json.loads(content)

        # 좌표 검증
        position_x = max(0, min(layout_data["position_x"], bg_width))
        position_y = max(0, min(layout_data["position_y"], bg_height))
        scale = max(0.3, min(0.9, layout_data["scale"]))

        logger.info(f"[{self.node_name}] GPT-4V placement decision:")
        logger.info(f"   Position: ({position_x}, {position_y})")
        logger.info(f"   Scale: {scale}")
        logger.info(f"   Reasoning: {layout_data['reasoning']}")

        return {
            "position": (position_x, position_y),
            "scale": scale,
            "reasoning": layout_data["reasoning"]
        }

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """PIL Image를 base64로 인코딩"""
        import base64

        # 메모리 절약을 위해 리사이즈 (GPT-4V는 고해상도 불필요)
        max_size = 512
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.copy().resize(new_size, Image.Resampling.LANCZOS)

        # base64 인코딩
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _get_default_layout(self, bg_size: Tuple[int, int], fg_size: Tuple[int, int]) -> Dict[str, Any]:
        """기본 배치 (중앙, 0.7 스케일)"""
        bg_width, bg_height = bg_size
        fg_width, fg_height = fg_size

        # 중앙 배치
        scale = 0.7
        scaled_width = int(fg_width * scale)
        scaled_height = int(fg_height * scale)

        position_x = (bg_width - scaled_width) // 2
        position_y = (bg_height - scaled_height) // 2

        return {
            "position": (position_x, position_y),
            "scale": scale,
            "reasoning": "Default center placement (API key not available)"
        }

    def get_required_inputs(self) -> list:
        return ["background", "foreground"]

    def get_output_keys(self) -> list:
        return ["position", "scale", "reasoning"]
