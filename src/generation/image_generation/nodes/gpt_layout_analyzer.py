"""
GPT Layout Analyzer Node
GPT-4V를 사용하여 이미지를 분석하고 최적의 텍스트 레이아웃 결정

작성자: 이현석
"""

import os
import base64
import json
from io import BytesIO
from typing import Dict, Any, Optional

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseNode
from ..tools.text_layout_tools import (
    TEXT_OVERLAY_TOOL,
    get_analysis_prompt,
    validate_layout_spec,
    get_default_layout
)

load_dotenv()


class GPTLayoutAnalyzerNode(BaseNode):
    """
    GPT-4V 이미지 분석 및 텍스트 레이아웃 결정 노드

    Inputs:
        - image (PIL.Image): 분석할 이미지
        - text_data (Dict[str, str]): 오버레이할 텍스트 데이터
            예: {"product_name": "딸기라떼", "tagline": "신메뉴 출시"}
        - image_context (str, optional): 이미지 컨텍스트 힌트
            예: "카페 메뉴 광고"

    Outputs:
        - layout_spec (Dict): GPT-4V가 결정한 레이아웃 명세
            {
                "layers": [
                    {
                        "text": "딸기라떼",
                        "position": {"x": 0.5, "y": 0.2, "anchor": "center"},
                        "font": {"family": "NanumGothicBold", "size": 80},
                        "color": {"r": 255, "g": 255, "b": 255, "a": 1.0},
                        "effects": {...},
                        "reasoning": "..."
                    }
                ]
            }

    Example:
        node = GPTLayoutAnalyzerNode()
        result = node.execute({
            "image": pil_image,
            "text_data": {"product_name": "딸기라떼", "tagline": "신메뉴 출시"},
            "image_context": "카페 메뉴 광고"
        })
        layout_spec = result["layout_spec"]
    """

    def __init__(self, model: str = "gpt-4o", max_retries: int = 2):
        """
        Args:
            model: GPT 모델 이름 (gpt-4o, gpt-4-turbo 등)
            max_retries: GPT 호출 실패 시 재시도 횟수
        """
        super().__init__("GPTLayoutAnalyzerNode")

        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        GPT-4V를 사용하여 이미지 분석 및 레이아웃 결정

        Args:
            inputs: {
                "image": PIL.Image,
                "text_data": Dict[str, str] or None,
                "image_context": str (optional)
            }

        Returns:
            {
                "layout_spec": Dict or None  # GPT-4V가 생성한 레이아웃 명세 (text_data 없으면 None)
            }
        """
        image = inputs["image"]
        text_data = inputs.get("text_data", None)
        image_context = inputs.get("image_context", "")

        # text_data가 없으면 스킵
        if not text_data:
            print(f"[{self.node_name}] ⏭️  No text_data provided, skipping text overlay")
            return {"layout_spec": None}

        print(f"[{self.node_name}] Analyzing image with GPT-4V...")
        print(f"   Text data: {text_data}")
        print(f"   Context: {image_context or 'None'}")

        # 1. 이미지를 base64로 인코딩
        image_base64 = self._encode_image_to_base64(image)

        # 2. 분석 프롬프트 생성
        analysis_prompt = get_analysis_prompt(text_data, image_context)

        # 3. GPT-4V 호출 (재시도 로직 포함)
        layout_spec = None
        for attempt in range(self.max_retries + 1):
            try:
                layout_spec = self._call_gpt_vision(image_base64, analysis_prompt)
                break
            except Exception as e:
                print(f"⚠️ GPT-4V call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt == self.max_retries:
                    print(f"⚠️ All retries exhausted, using fallback layout")
                    layout_spec = get_default_layout(text_data, image.size)

        # 4. 레이아웃 검증
        try:
            validate_layout_spec(layout_spec)
            print(f"[{self.node_name}] ✅ Layout analysis complete")
            print(f"   Layers: {len(layout_spec['layers'])}")
        except ValueError as e:
            print(f"⚠️ Layout validation failed: {e}")
            print(f"⚠️ Using fallback layout")
            layout_spec = get_default_layout(text_data, image.size)

        return {
            "layout_spec": layout_spec
        }

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        PIL 이미지를 base64 문자열로 인코딩

        Args:
            image: PIL.Image 객체

        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        # RGB 변환 (RGBA나 다른 모드 처리)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # JPEG로 인코딩 (품질 85, GPT-4V 최적)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        # base64 인코딩
        return base64.b64encode(image_bytes).decode("utf-8")

    def _call_gpt_vision(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """
        GPT-4V API 호출 (Tool Calling 사용)

        Args:
            image_base64: base64 인코딩된 이미지
            prompt: 분석 프롬프트

        Returns:
            Dict: GPT가 반환한 레이아웃 명세

        Raises:
            Exception: API 호출 실패 또는 tool call 없음
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"  # 고해상도 분석
                            }
                        }
                    ]
                }
            ],
            tools=[TEXT_OVERLAY_TOOL],
            tool_choice={"type": "function", "function": {"name": "apply_text_overlay"}}
        )

        # Tool call 추출
        message = response.choices[0].message

        if not message.tool_calls:
            raise ValueError("GPT-4V did not return any tool calls")

        tool_call = message.tool_calls[0]
        if tool_call.function.name != "apply_text_overlay":
            raise ValueError(f"Unexpected function call: {tool_call.function.name}")

        # JSON 파싱
        layout_spec = json.loads(tool_call.function.arguments)

        # 디버깅: GPT의 reasoning 출력
        for i, layer in enumerate(layout_spec.get("layers", [])):
            reasoning = layer.get("reasoning", "No reasoning provided")
            print(f"   Layer {i+1}: \"{layer.get('text', '')}\" - {reasoning}")

        return layout_spec

    def get_required_inputs(self) -> list:
        return ["image", "text_data"]

    def get_output_keys(self) -> list:
        return ["layout_spec"]
