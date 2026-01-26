"""
Text Overlay Node
GPT-4V가 결정한 레이아웃 명세에 따라 이미지에 텍스트 렌더링

작성자: 이현석
"""

from typing import Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFilter
import copy

from src.utils.logging import get_logger

from .base import BaseNode
from ..tools.font_loader import load_font

logger = get_logger(__name__)


class TextOverlayNode(BaseNode):
    """
    텍스트 오버레이 노드 (GPT-4V 레이아웃 기반)

    Inputs:
        - image (PIL.Image): 원본 이미지
        - layout_spec (Dict): GPT-4V가 생성한 레이아웃 명세
            {
                "layers": [
                    {
                        "text": "딸기라떼",
                        "position": {"x": 0.5, "y": 0.2, "anchor": "center"},
                        "font": {"family": "NanumGothicBold", "size": 80},
                        "color": {"r": 255, "g": 255, "b": 255, "a": 1.0},
                        "effects": {
                            "stroke": {...},
                            "shadow": {...},
                            "background_box": {...}
                        }
                    }
                ]
            }

    Outputs:
        - image (PIL.Image): 텍스트가 합성된 최종 이미지

    Example:
        node = TextOverlayNode()
        result = node.execute({
            "image": pil_image,
            "layout_spec": layout_spec
        })
        final_image = result["image"]
    """

    def __init__(self):
        super().__init__("TextOverlayNode")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        레이아웃 명세에 따라 텍스트 오버레이

        Args:
            inputs: {
                "image": PIL.Image,
                "layout_spec": Dict or None
            }

        Returns:
            {
                "image": PIL.Image  # 텍스트가 합성된 이미지 (layout_spec 없으면 원본 그대로)
            }
        """
        image = inputs["image"]
        layout_spec = inputs.get("layout_spec", None)

        # layout_spec이 없으면 원본 이미지 그대로 반환
        if not layout_spec:
            logger.info(f"[{self.node_name}] ⏭️  No layout_spec provided, skipping text overlay")
            return {"image": image}

        logger.info(f"[{self.node_name}] Applying text overlay...")

        # 이미지 복사 (원본 보존)
        canvas = image.copy()

        # 각 레이어를 순차적으로 렌더링
        layers = layout_spec.get("layers", [])
        for i, layer in enumerate(layers):
            logger.info(f"   Rendering layer {i+1}/{len(layers)}: \"{layer.get('text', '')}\"")
            canvas = self._render_layer(canvas, layer)

        logger.info(f"[{self.node_name}] ✅ Text overlay complete")

        return {"image": canvas}

    def _render_layer(self, canvas: Image.Image, layer: Dict[str, Any]) -> Image.Image:
        """
        단일 텍스트 레이어 렌더링

        Args:
            canvas: 현재 캔버스 이미지
            layer: 레이어 명세

        Returns:
            Image.Image: 텍스트가 렌더링된 이미지
        """
        text = layer["text"]
        position_spec = layer["position"]
        font_spec = layer["font"]
        color_spec = layer["color"]
        effects = layer.get("effects", {})

        # 1. 폰트 로드
        font = load_font(font_spec["family"], font_spec["size"])

        # 2. 절대 좌표 계산
        width, height = canvas.size
        abs_x, abs_y = self._calculate_absolute_position(
            width, height,
            position_spec["x"], position_spec["y"],
            position_spec["anchor"],
            text, font
        )

        # 3. RGBA 색상 변환
        text_color = self._rgba_tuple(color_spec)

        # 4. 효과 렌더링
        canvas = self._apply_effects(
            canvas, text, (abs_x, abs_y), font, text_color, effects
        )

        return canvas

    def _calculate_absolute_position(
        self,
        image_width: int,
        image_height: int,
        rel_x: float,
        rel_y: float,
        anchor: str,
        text: str,
        font
    ) -> Tuple[int, int]:
        """
        정규화 좌표 (0.0-1.0) → 절대 좌표 (픽셀) 변환

        Args:
            image_width, image_height: 이미지 크기
            rel_x, rel_y: 정규화 좌표 (0.0-1.0)
            anchor: 앵커 포인트 (예: "center", "top_left")
            text: 텍스트 내용
            font: PIL 폰트 객체

        Returns:
            (x, y): 절대 좌표
        """
        # 1. 정규화 좌표 → 절대 좌표
        x = int(rel_x * image_width)
        y = int(rel_y * image_height)

        # 2. 텍스트 경계 박스 계산
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 3. 앵커 포인트 보정
        anchor_offsets = {
            "top_left": (0, 0),
            "top_center": (-text_width // 2, 0),
            "top_right": (-text_width, 0),
            "center_left": (0, -text_height // 2),
            "center": (-text_width // 2, -text_height // 2),
            "center_right": (-text_width, -text_height // 2),
            "bottom_left": (0, -text_height),
            "bottom_center": (-text_width // 2, -text_height),
            "bottom_right": (-text_width, -text_height),
        }

        offset_x, offset_y = anchor_offsets.get(anchor, (0, 0))
        return x + offset_x, y + offset_y

    def _rgba_tuple(self, color_spec: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """RGBA dict → tuple 변환"""
        return (
            color_spec["r"],
            color_spec["g"],
            color_spec["b"],
            int(color_spec["a"] * 255)
        )

    def _apply_effects(
        self,
        canvas: Image.Image,
        text: str,
        position: Tuple[int, int],
        font,
        text_color: Tuple[int, int, int, int],
        effects: Dict[str, Any]
    ) -> Image.Image:
        """
        텍스트 효과 적용 (shadow, stroke, background_box)

        렌더링 순서:
        1. Background box (가장 뒤)
        2. Shadow (중간)
        3. Stroke (외곽선)
        4. Main text (가장 앞)
        """
        # RGBA 캔버스로 변환 (투명도 지원)
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")

        # Separate layers for shadow and main text to avoid blurring text/box
        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)

        text_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        # 1. Background Box
        bg_box = effects.get("background_box", {})
        if bg_box.get("enabled", False):
            self._draw_background_box(text_draw, text, position, font, bg_box)

        # 2. Shadow
        shadow = effects.get("shadow", {})
        if shadow.get("enabled", False):
            shadow_color = self._rgba_tuple(shadow["color"])
            offset_x = shadow.get("offset_x", 2)
            offset_y = shadow.get("offset_y", 2)
            blur = shadow.get("blur", 4)

            # 그림자 위치
            shadow_pos = (position[0] + offset_x, position[1] + offset_y)

            # 그림자 렌더링
            shadow_draw.text(shadow_pos, text, font=font, fill=shadow_color)

            # 블러 적용
            if blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur // 2))

        # 3. Main Text with Stroke (외곽선)
        stroke_spec = effects.get("stroke", {})
        if stroke_spec.get("enabled", False):
            stroke_width = stroke_spec.get("width", 2)
            stroke_color = self._rgba_tuple(stroke_spec["color"])

            # Stroke + Fill을 동시에 렌더링 (PIL 내부에서 stroke를 먼저 그린 후 fill)
            text_draw.text(
                position,
                text,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color
            )
        else:
            # Stroke 없으면 일반 텍스트만
            text_draw.text(position, text, font=font, fill=text_color)

        # Composite in order: shadow -> text
        canvas = Image.alpha_composite(canvas, shadow_layer)
        canvas = Image.alpha_composite(canvas, text_layer)

        return canvas

    def _draw_background_box(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        position: Tuple[int, int],
        font,
        bg_box_spec: Dict[str, Any]
    ):
        """배경 박스 그리기"""
        padding = bg_box_spec.get("padding", 10)
        bg_color = self._rgba_tuple(bg_box_spec["color"])
        border_radius = bg_box_spec.get("border_radius", 0)

        # 텍스트 경계 박스
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 배경 박스 좌표
        x1 = position[0] - padding
        y1 = position[1] - padding
        x2 = position[0] + text_width + padding
        y2 = position[1] + text_height + padding

        # 둥근 모서리 사각형
        if border_radius > 0:
            draw.rounded_rectangle(
                [(x1, y1), (x2, y2)],
                radius=border_radius,
                fill=bg_color
            )
        else:
            draw.rectangle([(x1, y1), (x2, y2)], fill=bg_color)

    def get_required_inputs(self) -> list:
        return ["image", "layout_spec"]

    def get_output_keys(self) -> list:
        return ["image"]
