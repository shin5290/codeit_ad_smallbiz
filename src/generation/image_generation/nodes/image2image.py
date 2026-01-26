"""
Image-to-Image Generation Node (Z-Image Turbo)
Z-Image Turbo I2I를 사용한 이미지 스타일 변환

특징:
- 빠른 속도 (8 steps)
- 원본 구도 유지하면서 스타일 변환
- 제품 사진 → 애니메이션 스타일
- 일반 사진 → 초사실적 스타일
"""

import gc
import threading
from typing import Dict, Any, Optional

import torch
from PIL import Image

from src.utils.logging import get_logger

from .base import BaseNode
from ..config import aspect_ratio_templates
from ..shared_cache import get_i2i_pipeline, flush_shared_cache

logger = get_logger(__name__)

# ==============================================================================
# ★ 전역 상태 관리 (공유 캐시 사용)
# ==============================================================================
_EXECUTION_LOCK = threading.Lock()
_EXECUTION_COUNT = 0


class Image2ImageNode(BaseNode):
    """
    Z-Image Turbo Image-to-Image 노드

    사용 케이스:
    1. 스타일 변환: 사실적 제품 사진 → 애니메이션 스타일
    2. 품질 향상: 일반 사진 → 초사실적 고품질 사진
    3. 구도 유지 재생성: 스케치 → 완성된 그림

    Example:
        node = Image2ImageNode()
        result = node.execute({
            "prompt": "anime style, vibrant colors, illustrated product",
            "reference_image": product_photo,  # PIL.Image
            "strength": 0.6,  # 변형 강도 (0.3~0.7 권장)
            "aspect_ratio": "1:1",
            "num_inference_steps": 8,
            "seed": 42
        })
        image = result["image"]
    """

    def __init__(self, device: Optional[str] = None, auto_unload: bool = False):
        super().__init__("Image2ImageNode")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_unload = auto_unload

    def load_pipeline(self):
        """공유 캐시를 사용하여 I2I 파이프라인 로드"""
        logger.info(f"[{self.node_name}] Loading I2I pipeline (shared cache)...")
        return get_i2i_pipeline(self.device)

    def get_generator_device(self, pipe):
        """Generator 디바이스 결정 (CPU offload 고려)"""
        if hasattr(pipe, "_execution_device"):
            return pipe._execution_device
        return self.device

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """I2I 이미지 생성"""
        global _EXECUTION_COUNT

        with _EXECUTION_LOCK:
            # 입력 추출
            prompt = inputs.get("prompt", "")
            reference_image = inputs.get("reference_image")  # PIL.Image
            strength = inputs.get("strength", 0.6)  # 기본값 0.6
            aspect_ratio = inputs.get("aspect_ratio", "1:1")
            num_inference_steps = inputs.get("num_inference_steps", 8)
            seed = inputs.get("seed", None)

            # 필수 파라미터 검증
            if reference_image is None:
                raise ValueError("reference_image is required for Image2Image generation")

            if not isinstance(reference_image, Image.Image):
                raise ValueError("reference_image must be a PIL.Image.Image object")

            if reference_image.mode != "RGB":
                reference_image = reference_image.convert("RGB")

            # 해상도 결정
            width, height = aspect_ratio_templates.get_size(aspect_ratio)

            # 입력 이미지 리사이즈
            reference_image = reference_image.resize((width, height), Image.Resampling.LANCZOS)

            logger.info(f"[{self.node_name}] Input: {width}x{height}, strength={strength}")

            # 파이프라인 로드
            pipe = self.load_pipeline()

            # 제너레이터 생성 및 시드 추출
            exec_device = self.get_generator_device(pipe)

            if seed is None:
                import random
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=exec_device).manual_seed(seed)

            logger.info(f"[{self.node_name}] Generating I2I ({width}x{height}, seed={seed}, strength={strength})...")

            # I2I 생성
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    image=reference_image,
                    strength=strength,  # 변형 강도 (0.0~1.0)
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0,  # ZIT는 CFG 미사용
                    generator=generator,
                    output_type="pil"
                ).images[0]

            # 주기적 메모리 정리 (5회마다)
            _EXECUTION_COUNT += 1
            if _EXECUTION_COUNT % 5 == 0:
                logger.info(f"[{self.node_name}] 🧹 Periodic Memory Cleanup (Count: {_EXECUTION_COUNT})")
                gc.collect()
                torch.cuda.empty_cache()

            # auto_unload가 True일 때만 강제 종료
            if self.auto_unload:
                self.flush_global()

            return {"image": image, "seed": seed, "width": width, "height": height}

    def flush_global(self):
        """전역 캐시 완전 초기화 (공유 캐시 사용)"""
        flush_shared_cache()

    def get_required_inputs(self):
        return ['prompt', 'reference_image']

    def get_output_keys(self):
        return ["image", "seed", "width", "height"]
