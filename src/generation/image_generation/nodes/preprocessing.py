"""
Preprocessing Nodes
전처리 노드 모음

- BackgroundRemovalNode: 배경 제거 (rembg)

작성자: 이현석
"""

from typing import Dict, Any
from PIL import Image
import io

from src.utils.logging import get_logger

from .base import BaseNode

logger = get_logger(__name__)

# rembg는 lazy import (메모리 절약)
_REMBG_SESSION = None


def get_rembg_session(model: str = "u2net"):
    """전역 rembg 세션 (lazy initialization)"""
    global _REMBG_SESSION

    if _REMBG_SESSION is None:
        try:
            from rembg import new_session
            # u2net: 범용 (176MB)
            # u2net_human_seg: 인물 특화 (176MB)
            # isnet-general-use: 고품질 (176MB)
            _REMBG_SESSION = new_session(model)
            logger.info(f"[BackgroundRemoval] rembg session initialized ({model})")
        except ImportError:
            logger.error("[BackgroundRemoval] rembg not installed. Run: pip install rembg")
            raise

    return _REMBG_SESSION


class BackgroundRemovalNode(BaseNode):
    """
    배경 제거 노드 (rembg)

    메모리: ~500MB VRAM
    속도: 1-2초/이미지

    Inputs:
        - image (PIL.Image): 원본 이미지
        - alpha_matting (bool): 알파 매팅 활성화 (디테일 향상, 느림)
        - alpha_matting_foreground_threshold (int): 전경 임계값 (0-255)
        - alpha_matting_background_threshold (int): 배경 임계값 (0-255)

    Outputs:
        - foreground (PIL.Image): 배경이 제거된 RGBA 이미지
        - mask (PIL.Image): 알파 마스크 (L mode)
    """

    def __init__(self, model: str = "u2net"):
        """
        Args:
            model: rembg 모델 선택
                - "u2net": 범용 (기본)
                - "u2net_human_seg": 인물 특화
                - "isnet-general-use": 고품질
        """
        super().__init__("BackgroundRemovalNode")
        self.model = model

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        배경 제거 실행

        Returns:
            {
                "foreground": PIL.Image (RGBA),
                "mask": PIL.Image (L mode)
            }
        """
        image = inputs["image"]
        alpha_matting = inputs.get("alpha_matting", False)
        alpha_matting_fg = inputs.get("alpha_matting_foreground_threshold", 240)
        alpha_matting_bg = inputs.get("alpha_matting_background_threshold", 10)

        logger.info(f"[{self.node_name}] Removing background...")
        logger.info(f"   Model: {self.model}, Alpha matting: {alpha_matting}")

        # rembg 세션 가져오기
        session = get_rembg_session(self.model)

        # 배경 제거
        try:
            from rembg import remove

            # PIL Image → bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # 배경 제거 실행
            output_bytes = remove(
                img_bytes.read(),
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_fg,
                alpha_matting_background_threshold=alpha_matting_bg,
            )

            # bytes → PIL Image
            output_image = Image.open(io.BytesIO(output_bytes))

            # RGBA로 변환 (알파 채널 보장)
            if output_image.mode != "RGBA":
                output_image = output_image.convert("RGBA")

            # 알파 마스크 추출
            mask = output_image.split()[3]  # A 채널

            logger.info(f"[{self.node_name}] ✅ Background removed")

            # VRAM 최적화: rembg 세션 언로드
            # Text2ImageNode가 로드되기 전에 메모리 확보
            global _REMBG_SESSION
            if _REMBG_SESSION is not None:
                del _REMBG_SESSION
                _REMBG_SESSION = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                logger.info(f"[{self.node_name}] Released rembg session to free VRAM")

            return {
                "foreground": output_image,
                "mask": mask
            }

        except Exception as e:
            logger.error(f"[{self.node_name}] Background removal failed: {e}")
            # 실패 시 원본 반환
            rgba_image = image.convert("RGBA")
            mask = Image.new("L", image.size, 255)  # 전체 불투명 마스크
            return {
                "foreground": rgba_image,
                "mask": mask
            }

    def get_required_inputs(self) -> list:
        return ["image"]

    def get_output_keys(self) -> list:
        return ["foreground", "mask"]
