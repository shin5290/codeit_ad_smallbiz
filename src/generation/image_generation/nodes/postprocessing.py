"""
Postprocessing Nodes
후처리 노드 모음

- BackgroundCompositeNode: 전경 + 배경 합성
- SolidBackgroundNode: 단색/그라디언트 배경 생성

작성자: 이현석
"""

from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

from src.utils.logging import get_logger

from .base import BaseNode

logger = get_logger(__name__)


class BackgroundCompositeNode(BaseNode):
    """
    배경 합성 노드

    전경(배경 제거된 제품) + 배경 이미지 → 최종 합성

    Inputs:
        - foreground (PIL.Image): 전경 이미지 (RGBA)
        - background (PIL.Image): 배경 이미지 (RGB/RGBA)
        - position (tuple): (x, y) 전경 위치 (None이면 중앙)
        - scale (float): 전경 스케일 (1.0 = 원본 크기)

    Outputs:
        - image (PIL.Image): 합성된 이미지 (RGB)
    """

    def __init__(self):
        super().__init__("BackgroundCompositeNode")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        배경 합성 실행

        Returns:
            {"image": PIL.Image (RGB)}
        """
        foreground = inputs["foreground"]
        background = inputs["background"]
        position = inputs.get("position", None)
        scale = inputs.get("scale", 1.0)

        logger.info(f"[{self.node_name}] Compositing foreground + background...")
        logger.info(f"   Foreground: {foreground.size}, Background: {background.size}")
        logger.info(f"   Scale: {scale}, Position: {position or 'center'}")

        # 1. 배경을 RGB로 변환
        if background.mode == "RGBA":
            # 알파 채널을 흰색 배경에 합성
            bg_rgb = Image.new("RGB", background.size, (255, 255, 255))
            bg_rgb.paste(background, mask=background.split()[3])
            background = bg_rgb
        elif background.mode != "RGB":
            background = background.convert("RGB")

        # 2. 전경을 RGBA로 변환
        if foreground.mode != "RGBA":
            foreground = foreground.convert("RGBA")

        # 3. 전경 스케일링
        if scale != 1.0:
            new_width = int(foreground.width * scale)
            new_height = int(foreground.height * scale)
            foreground = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"   Scaled foreground to {new_width}x{new_height}")

        # 4. 전경이 배경보다 크면 배경 크기에 맞춤
        if foreground.width > background.width or foreground.height > background.height:
            # 배경에 맞게 비율 유지하며 축소
            ratio = min(
                background.width / foreground.width,
                background.height / foreground.height
            ) * 0.9  # 여백 10%
            new_width = int(foreground.width * ratio)
            new_height = int(foreground.height * ratio)
            foreground = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"   Foreground too large, resized to {new_width}x{new_height}")

        # 5. 위치 결정
        if position is None:
            # 중앙 배치
            x = (background.width - foreground.width) // 2
            y = (background.height - foreground.height) // 2
            position = (x, y)

        # 6. 합성
        composite = background.copy()
        composite.paste(foreground, position, mask=foreground.split()[3])

        logger.info(f"[{self.node_name}] ✅ Composite complete at position {position}")

        return {"image": composite}

    def get_required_inputs(self) -> list:
        return ["foreground", "background"]

    def get_output_keys(self) -> list:
        return ["image"]


class SolidBackgroundNode(BaseNode):
    """
    단색 배경 생성 노드

    Inputs:
        - width (int): 배경 너비
        - height (int): 배경 높이
        - color (tuple/str): RGB 색상 (r, g, b) 또는 hex "#RRGGBB"
        - gradient (bool): 그라디언트 활성화 (기본: False)

    Outputs:
        - background (PIL.Image): 생성된 배경 (RGB)
    """

    def __init__(self):
        super().__init__("SolidBackgroundNode")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        단색 배경 생성

        Returns:
            {"background": PIL.Image (RGB)}
        """
        width = inputs["width"]
        height = inputs["height"]
        color = inputs.get("color", (255, 255, 255))  # 기본 흰색
        gradient = inputs.get("gradient", False)

        # Hex 색상 → RGB 변환
        if isinstance(color, str):
            color = self._hex_to_rgb(color)

        logger.info(f"[{self.node_name}] Creating background {width}x{height}, color={color}")

        if gradient:
            # 간단한 세로 그라디언트
            background = self._create_gradient(width, height, color)
        else:
            # 단색 배경
            background = Image.new("RGB", (width, height), color)

        logger.info(f"[{self.node_name}] ✅ Background created")

        return {"background": background}

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Hex 색상을 RGB로 변환"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _create_gradient(self, width: int, height: int, base_color: Tuple[int, int, int]) -> Image.Image:
        """세로 그라디언트 생성 (어두운 → 밝은)"""
        # 베이스 색상에서 밝은 버전 생성
        r, g, b = base_color
        bright_color = (
            min(255, int(r * 1.3)),
            min(255, int(g * 1.3)),
            min(255, int(b * 1.3))
        )

        # 그라디언트 생성
        gradient_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            ratio = y / height
            gradient_array[y, :, 0] = int(base_color[0] * (1 - ratio) + bright_color[0] * ratio)
            gradient_array[y, :, 1] = int(base_color[1] * (1 - ratio) + bright_color[1] * ratio)
            gradient_array[y, :, 2] = int(base_color[2] * (1 - ratio) + bright_color[2] * ratio)

        return Image.fromarray(gradient_array, mode="RGB")

    def get_required_inputs(self) -> list:
        return ["width", "height"]

    def get_output_keys(self) -> list:
        return ["background"]
