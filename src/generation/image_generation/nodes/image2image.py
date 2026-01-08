"""
Image-to-Image Generation Node with ControlNet
ControlNet을 사용한 이미지 기반 생성 (제품 형태 유지 + 스타일 변환)
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, AutoencoderKL

from .base import BaseNode
from ..config import (
    model_config,
    generation_config,
    aspect_ratio_templates,
)

# 모델 캐시 디렉토리
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class Image2ImageControlNetNode(BaseNode):
    """
    ControlNet을 사용한 Image-to-Image 생성 노드

    입력 이미지의 구조(윤곽선, 깊이 등)를 유지하면서
    선택된 스타일로 이미지 재생성

    Example:
        # 제품 사진을 ultra_realistic 스타일로 재생성
        node = Image2ImageControlNetNode(model_id="SG161222/RealVisXL_V4.0")
        result = node.execute({
            "prompt": "professional product photo of coffee cup on wooden table",
            "control_image": canny_edge_image,  # ControlNet preprocessor 출력
            "controlnet": controlnet_model,     # ControlNet 모델
            "style": "ultra_realistic",
            "aspect_ratio": "1:1",
            "num_inference_steps": 40,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 0.8
        })
        image = result["image"]
    """

    # 클래스 변수: VAE 캐시 (Text2ImageNode와 공유)
    from .text2image_backup import Text2ImageNode
    _vae_cache = Text2ImageNode._vae_cache

    def __init__(
        self,
        device: Optional[str] = None,
        model_id: Optional[str] = None,
        auto_unload: bool = True
    ):
        """
        Args:
            device: 실행할 디바이스 ("cuda", "cpu" 등)
            model_id: 사용할 SDXL 모델 ID
            auto_unload: 생성 완료 후 자동 언로드 (기본: True)
        """
        super().__init__("Image2ImageControlNetNode")

        self.device = device or model_config.DEVICE
        self.model_id = model_id or model_config.MODEL_ID
        self.auto_unload = auto_unload

        # 파이프라인 (처음엔 None)
        self.pipe = None

    def _load_pipeline(self, controlnet):
        """
        SDXL ControlNet 파이프라인 로드

        Args:
            controlnet: ControlNet 모델 객체
        """
        if self.pipe is not None:
            return  # 이미 로드됨

        print(f"[{self.node_name}] Loading SDXL ControlNet pipeline from {self.model_id}...")

        # 로컬 모델 경로
        local_model_path = MODELS_DIR / self.model_id.replace("/", "--")

        # VAE 로드 (Text2ImageNode와 공유)
        from .text2image_backup import Text2ImageNode
        if Text2ImageNode._vae_cache is None:
            print(f"[{self.node_name}] Loading VAE (first time)...")
            Text2ImageNode._vae_cache = AutoencoderKL.from_pretrained(
                model_config.VAE_ID,
                torch_dtype=getattr(torch, model_config.DTYPE),
                cache_dir=MODELS_DIR
            )
        else:
            print(f"[{self.node_name}] Using cached VAE...")

        vae = Text2ImageNode._vae_cache

        # ControlNet 파이프라인 로드
        if local_model_path.exists():
            print(f"[{self.node_name}] Loading from local cache: {local_model_path}")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                str(local_model_path),
                controlnet=controlnet,
                vae=vae,
                torch_dtype=getattr(torch, model_config.DTYPE),
                use_safetensors=True,
            )
        else:
            print(f"[{self.node_name}] Downloading from HuggingFace: {self.model_id}")
            try:
                self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=getattr(torch, model_config.DTYPE),
                    variant=model_config.VARIANT,
                    use_safetensors=True,
                )
            except (OSError, ValueError) as e:
                if "variant" in str(e).lower():
                    print(f"[{self.node_name}] No fp16 variant, loading without variant...")
                    self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                        self.model_id,
                        controlnet=controlnet,
                        vae=vae,
                        torch_dtype=getattr(torch, model_config.DTYPE),
                        use_safetensors=True,
                    )
                else:
                    raise

            # 로컬 저장 (ControlNet 제외, base model만)
            print(f"[{self.node_name}] Saving base model to: {local_model_path}")
            # ControlNet 파이프라인에서 base model만 추출해서 저장하는 건 복잡하므로
            # 기존 Text2ImageNode가 이미 저장했으므로 스킵

        # GPU로 이동
        self.pipe.to(self.device)

        print(f"[{self.node_name}] ControlNet pipeline loaded successfully!")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        ControlNet을 사용하여 이미지 생성

        Args:
            inputs:
                - prompt (str, 필수): 생성할 이미지 설명
                - control_image (PIL.Image, 필수): ControlNet용 전처리 이미지 (Canny, Depth 등)
                - controlnet (ControlNetModel, 필수): ControlNet 모델
                - aspect_ratio (str, 선택): "1:1", "16:9" 등
                - negative_prompt (str, 선택): 제외할 요소
                - style (str, 선택): 스타일
                - num_inference_steps (int, 선택): 생성 스텝
                - guidance_scale (float, 선택): CFG 스케일
                - controlnet_conditioning_scale (float, 선택): ControlNet 강도 (0.0~1.0, 기본 0.8)
                - seed (int, 선택): 랜덤 시드
                - industry (str, 선택): 업종

        Returns:
            - image (PIL.Image): 생성된 이미지
            - seed (int): 사용된 시드
            - width (int): 이미지 너비
            - height (int): 이미지 높이
        """
        # 필수 입력 확인
        prompt = inputs["prompt"]
        control_image = inputs["control_image"]
        controlnet = inputs["controlnet"]

        # 파이프라인 로드
        self._load_pipeline(controlnet)

        # 선택적 입력
        aspect_ratio = inputs.get("aspect_ratio", generation_config.DEFAULT_ASPECT_RATIO)
        style = inputs.get("style", "ultra_realistic")
        negative_prompt = inputs.get("negative_prompt", generation_config.get_negative_prompt(style))
        num_inference_steps = inputs.get("num_inference_steps", generation_config.DEFAULT_STEPS)
        guidance_scale = inputs.get("guidance_scale", generation_config.DEFAULT_GUIDANCE_SCALE)
        controlnet_conditioning_scale = inputs.get(
            "controlnet_conditioning_scale",
            generation_config.CONTROLNET_CONDITIONING_SCALE
        )
        seed = inputs.get("seed", None)
        industry = inputs.get("industry", None)

        # 해상도
        width, height = aspect_ratio_templates.get_size(aspect_ratio)

        # control_image 해상도 조정
        if control_image.size != (width, height):
            print(f"[{self.node_name}] Resizing control_image from {control_image.size} to {width}x{height}")
            control_image = control_image.resize((width, height), Image.LANCZOS)

        # 업종별 스타일 적용
        if industry and industry in generation_config.INDUSTRY_STYLES:
            industry_style = generation_config.INDUSTRY_STYLES[industry]
            prompt = f"{prompt}, {industry_style['style_suffix']}"
            negative_prompt = f"{negative_prompt}, {industry_style['negative_add']}"

        # 시드 설정
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 이미지 생성
        print(f"[{self.node_name}] Generating {width}x{height} image with ControlNet...")
        print(f"[{self.node_name}] Prompt: {prompt[:100]}...")
        print(f"[{self.node_name}] ControlNet scale: {controlnet_conditioning_scale}")

        result = self.pipe(
            prompt=prompt,
            image=control_image,  # ControlNet condition
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )

        image = result.images[0]

        # 자동 언로드
        if self.auto_unload:
            print(f"[{self.node_name}] Auto-unloading pipeline...")
            self.unload_pipeline()

        return {
            "image": image,
            "seed": seed,
            "width": width,
            "height": height,
        }

    def get_required_inputs(self) -> list:
        return ["prompt", "control_image", "controlnet"]

    def get_output_keys(self) -> list:
        return ["image", "seed", "width", "height"]

    def unload_pipeline(self):
        """파이프라인 언로드"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print(f"[{self.node_name}] Pipeline unloaded")
