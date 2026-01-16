"""
Z-Image Turbo Text2Image Node
Z-Image Turbo 모델을 사용한 고속 이미지 생성 노드

특징:
- 8 steps로 고품질 이미지 생성 (~1-2초)
- 긴 프롬프트 지원 (CLIP 77 토큰 제한 없음, T5 기반)
- LoRA를 통한 스타일 전환 지원
- Negative Prompt 미지원 (CFG 미사용)
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
from PIL import Image

from .base import BaseNode
from ..config import (
    model_config,
    generation_config,
    aspect_ratio_templates,
)

import os

# Z-Image Turbo 모델 경로
ZIT_MODELS_DIR = Path(os.getenv("ZIT_MODELS_DIR", "/opt/ai-models/zit"))
ZIT_BASE_MODEL = ZIT_MODELS_DIR / "Z-Image-Turbo-Base"
ZIT_LORA_DIR = ZIT_MODELS_DIR / "lora"

# 스타일별 LoRA 매핑
STYLE_LORA_MAP = {
    "realistic": None,  # 베이스 모델 그대로 사용
    "ultra_realistic": None,  # 베이스 모델 그대로 사용
    "semi_realistic": "OB半写实肖像画2.0 OB Semi-Realistic Portraits z- image turbo(1).safetensors",
    "anime": "Anime-Z.safetensors",
}


class Text2ImageNode(BaseNode):
    """
    Z-Image Turbo를 사용한 Text-to-Image 노드

    SDXL 대비 장점:
    - 8 steps로 고속 생성 (SDXL 40 steps 대비 5배 빠름)
    - 긴 프롬프트 지원 (T5 인코더, 토큰 제한 없음)
    - LoRA로 스타일 전환 간편

    Example:
        node = Text2ImageNode()
        result = node.execute({
            "prompt": "귀여운 곰 캐릭터가 헬스장에서 운동하는 광고",
            "aspect_ratio": "1:1",
            "style": "anime",
            "seed": 42
        })
        image = result["image"]  # PIL.Image
    """

    # 클래스 변수: 파이프라인 캐시 (스타일별)
    _pipe_cache = {}
    _current_lora = None

    def __init__(self, device: Optional[str] = None, auto_unload: bool = True):
        """
        Args:
            device: 실행할 디바이스 ("cuda", "cpu" 등)
            auto_unload: 이미지 생성 완료 후 자동으로 파이프라인 언로드 (기본: True)
        """
        super().__init__("Text2ImageNode")

        self.device = device or model_config.DEVICE
        self.auto_unload = auto_unload
        self.pipe = None

    def _load_pipeline(self, style: str = "realistic"):
        """
        Z-Image Turbo 파이프라인 로드

        Args:
            style: 스타일 (realistic, semi_realistic, anime)
        """
        from diffusers import ZImagePipeline

        # 이미 로드된 파이프라인이 있고 같은 스타일이면 재사용
        if self.pipe is not None and Text2ImageNode._current_lora == STYLE_LORA_MAP.get(style):
            return

        print(f"[{self.node_name}] Loading Z-Image Turbo pipeline...")

        # 베이스 모델 로드
        if not ZIT_BASE_MODEL.exists():
            raise FileNotFoundError(
                f"Z-Image Turbo model not found: {ZIT_BASE_MODEL}\n"
                "Please run: python src/generation/image_generation/download_model_zit.py"
            )

        # 파이프라인 로드 (bfloat16 권장)
        self.pipe = ZImagePipeline.from_pretrained(
            str(ZIT_BASE_MODEL),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.pipe.to(self.device)

        # LoRA 적용 (스타일에 따라)
        lora_file = STYLE_LORA_MAP.get(style)
        if lora_file:
            lora_path = ZIT_LORA_DIR / lora_file
            if lora_path.exists():
                print(f"[{self.node_name}] Loading LoRA: {lora_file}")
                self.pipe.load_lora_weights(str(lora_path))
                Text2ImageNode._current_lora = lora_file
            else:
                print(f"[{self.node_name}] Warning: LoRA not found: {lora_path}")
                Text2ImageNode._current_lora = None
        else:
            # LoRA 없이 베이스 모델 사용
            if Text2ImageNode._current_lora is not None:
                self.pipe.unload_lora_weights()
            Text2ImageNode._current_lora = None

        print(f"[{self.node_name}] Pipeline loaded successfully! (style: {style})")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        텍스트 프롬프트로부터 이미지 생성

        Z-Image Turbo는 Negative Prompt를 지원하지 않습니다.
        모든 제약은 Positive Prompt에 포함해야 합니다.

        Args:
            inputs: 입력 데이터
                - prompt (str, 필수): 생성할 이미지 설명 (긴 자연어 권장)
                - aspect_ratio (str, 선택): "1:1", "16:9" 등 (기본: "1:1")
                - style (str, 선택): 스타일 ("realistic", "semi_realistic", "anime")
                - num_inference_steps (int, 선택): 생성 스텝 수 (기본: 8)
                - seed (int, 선택): 랜덤 시드 (재현성 위해)

        Returns:
            출력 데이터
                - image (PIL.Image): 생성된 이미지
                - seed (int): 사용된 시드값
                - width (int): 이미지 너비
                - height (int): 이미지 높이
        """
        # 입력 파라미터 추출
        prompt = inputs["prompt"]
        aspect_ratio = inputs.get("aspect_ratio", generation_config.DEFAULT_ASPECT_RATIO)
        style = inputs.get("style", "realistic")
        num_inference_steps = inputs.get("num_inference_steps", 8)  # Z-Image Turbo 기본값
        seed = inputs.get("seed", None)

        # 파이프라인 로드 (스타일에 맞게)
        self._load_pipeline(style)

        # 해상도 가져오기
        width, height = aspect_ratio_templates.get_size(aspect_ratio)

        # 시드 설정
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 이미지 생성
        print(f"[{self.node_name}] Generating {width}x{height} image...")
        print(f"[{self.node_name}] Prompt: {prompt[:100]}...")

        # Z-Image Turbo는 negative_prompt 미지원 (CFG 미사용)
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps + 1,  # Z-Image는 n+1 steps = n forwards
            guidance_scale=0.0,  # Turbo 모델은 guidance 0 권장
            generator=generator,
        )

        image = result.images[0]

        # 자동 언로드
        if self.auto_unload:
            print(f"[{self.node_name}] Auto-unloading pipeline to free memory...")
            self.unload_pipeline()

        return {
            "image": image,
            "seed": seed,
            "width": width,
            "height": height,
        }

    def get_required_inputs(self) -> list:
        """필수 입력: prompt만 필수"""
        return ["prompt"]

    def get_output_keys(self) -> list:
        """출력: image, seed, width, height"""
        return ["image", "seed", "width", "height"]

    def unload_pipeline(self):
        """파이프라인 언로드 (메모리 절약)"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print(f"[{self.node_name}] Pipeline unloaded")
