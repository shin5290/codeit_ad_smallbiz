"""
Image Generation Nodes
텍스트나 이미지를 입력받아 새로운 이미지를 생성하는 노드들
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from .base import BaseNode
from ..config import (
    model_config,
    generation_config,
    aspect_ratio_templates,
)

# 모델 캐시 디렉토리
# VM 공용 스토리지 경로 (환경변수로 오버라이드 가능)
import os
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/ai-models/sdxl"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class Text2ImageNode(BaseNode):
    """
    텍스트 프롬프트로부터 이미지를 생성하는 노드

    SDXL 모델과 개선된 VAE를 사용하여 고품질 이미지 생성

    Example:
        node = Text2ImageNode()
        result = node.execute({
            "prompt": "cozy cafe interior",
            "aspect_ratio": "16:9",
            "negative_prompt": "blurry, low quality",
            "num_inference_steps": 40,
            "guidance_scale": 7.5,
            "seed": 42
        })
        image = result["image"]  # PIL.Image
    """

    # 클래스 변수: 모든 인스턴스가 공유하는 VAE 캐시
    _vae_cache = None

    def __init__(self, device: Optional[str] = None, model_id: Optional[str] = None, auto_unload: bool = True):
        """
        Args:
            device: 실행할 디바이스 ("cuda", "cpu" 등)
                    None이면 config.py의 설정 사용
            model_id: 사용할 SDXL 모델 ID (HuggingFace repo)
                      None이면 기본 SDXL (SG161222/RealVisXL_V4.0)
                      예: "SG161222/RealVisXL_V4.0", "cagliostrolab/animagine-xl-3.1"

                      모델은 자동으로 image_generation/models/ 에 다운로드되어 캐싱됨
            auto_unload: 이미지 생성 완료 후 자동으로 파이프라인 언로드 (기본: True)
        """
        super().__init__("Text2ImageNode")

        # 디바이스 설정
        self.device = device or model_config.DEVICE

        # 모델 ID 설정
        self.model_id = model_id or model_config.MODEL_ID

        # 자동 언로드 설정
        self.auto_unload = auto_unload

        # 파이프라인 (처음엔 None, 매 실행마다 로드/언로드)
        self.pipe = None

    def _load_pipeline(self):
        """
        SDXL 파이프라인 로드 (models/ 폴더에서 캐싱)

        - 로컬 models/ 폴더에 모델이 있으면 그것을 사용
        - 없으면 HuggingFace에서 다운로드하여 models/ 폴더에 저장
        """
        if self.pipe is not None:
            return  # 이미 로드됨

        print(f"[{self.node_name}] Loading SDXL pipeline from {self.model_id}...")

        # 로컬 모델 경로 확인
        local_model_path = MODELS_DIR / self.model_id.replace("/", "--")

        # 개선된 VAE 로드 (클래스 변수로 캐싱하여 재사용)
        if Text2ImageNode._vae_cache is None:
            print(f"[{self.node_name}] Loading VAE (first time)...")
            Text2ImageNode._vae_cache = AutoencoderKL.from_pretrained(
                str(MODELS_DIR / "madebyollin--sdxl-vae-fp16-fix"),
                local_files_only=True,
                torch_dtype=getattr(torch, model_config.DTYPE),
            )
        else:
            print(f"[{self.node_name}] Using cached VAE...")

        vae = Text2ImageNode._vae_cache

        if not local_model_path.exists():
            raise FileNotFoundError(
                f"Local SDXL model not found: {local_model_path}"
            )

        print(f"[{self.node_name}] Loading local SDXL model: {local_model_path}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            str(local_model_path),
            vae=vae,
            local_files_only=True,
            torch_dtype=getattr(torch, model_config.DTYPE),
            use_safetensors=True,
        )

        # GPU로 이동
        self.pipe.to(self.device)

        print(f"[{self.node_name}] Pipeline loaded successfully!")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        텍스트 프롬프트로부터 이미지 생성

        Args:
            inputs: 입력 데이터
                - prompt (str, 필수): 생성할 이미지 설명
                - aspect_ratio (str, 선택): "1:1", "16:9" 등 (기본: "1:1")
                - negative_prompt (str, 선택): 제외할 요소 (지정하지 않으면 style에 따라 자동 설정)
                - style (str, 선택): 스타일 ("ultra_realistic", "semi_realistic", "anime")
                - num_inference_steps (int, 선택): 생성 스텝 수 (기본: 40)
                - guidance_scale (float, 선택): CFG 스케일 (기본: 7.5)
                - seed (int, 선택): 랜덤 시드 (재현성 위해)
                - industry (str, 선택): 업종 ("cafe", "restaurant" 등)

        Returns:
            출력 데이터
                - image (PIL.Image): 생성된 이미지
                - seed (int): 사용된 시드값
                - width (int): 이미지 너비
                - height (int): 이미지 높이
        """
        # 파이프라인 로드 (lazy loading)
        self._load_pipeline()

        # 입력 파라미터 추출
        prompt = inputs["prompt"]
        aspect_ratio = inputs.get("aspect_ratio", generation_config.DEFAULT_ASPECT_RATIO)
        style = inputs.get("style", "ultra_realistic")
        # negative_prompt가 명시적으로 지정되지 않으면 스타일에 따라 자동 설정
        negative_prompt = inputs.get("negative_prompt", generation_config.get_negative_prompt(style))
        num_inference_steps = inputs.get("num_inference_steps", generation_config.DEFAULT_STEPS)
        guidance_scale = inputs.get("guidance_scale", generation_config.DEFAULT_GUIDANCE_SCALE)
        seed = inputs.get("seed", None)
        industry = inputs.get("industry", None)

        # 해상도 가져오기 (aspect_ratio_templates에서)
        width, height = aspect_ratio_templates.get_size(aspect_ratio)

        # 업종별 스타일 적용 (있을 경우)
        if industry and industry in generation_config.INDUSTRY_STYLES:
            style = generation_config.INDUSTRY_STYLES[industry]
            # 프롬프트에 스타일 접미사 추가
            prompt = f"{prompt}, {style['style_suffix']}"
            # 네거티브 프롬프트에 추가 네거티브 추가
            negative_prompt = f"{negative_prompt}, {style['negative_add']}"

        # 시드 설정 (재현성)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 이미지 생성
        print(f"[{self.node_name}] Generating {width}x{height} image...")
        print(f"[{self.node_name}] Prompt: {prompt[:100]}...")  # 앞부분만 출력

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]

        # 자동 언로드 (메모리 절약)
        if self.auto_unload:
            print(f"[{self.node_name}] Auto-unloading pipeline to free memory...")
            self.unload_pipeline()

        # 실제 사용된 시드 반환 (없었으면 None)
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
        """
        파이프라인 언로드 (메모리 절약)

        사용 후 메모리를 확보하고 싶을 때 호출
        """
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()  # GPU 메모리 정리
            print(f"[{self.node_name}] Pipeline unloaded")
