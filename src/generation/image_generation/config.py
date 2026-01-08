"""
Image Generation Configuration
SDXL 기반 이미지 생성을 위한 설정 파일
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """모델 관련 설정"""

    # SDXL 모델 설정
    MODEL_TYPE: str = "sdxl"
    MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # VAE 설정 (개선된 VAE 사용)
    VAE_ID: str = "madebyollin/sdxl-vae-fp16-fix"

    # 데이터 타입
    DTYPE: str = "float16"
    VARIANT: str = "fp16"

    # 디바이스
    DEVICE: str = "cuda"


@dataclass
class AspectRatioTemplates:
    """해상도 템플릿 (비율별)"""

    # 비율별 해상도 (SDXL 최적화)
    RATIOS: dict = None

    def __post_init__(self):
        if self.RATIOS is None:
            self.RATIOS = {
                "1:1": (1024, 1024),    # 정사각형 (SNS 프로필, 썸네일)
                "3:4": (896, 1152),     # 세로 (Instagram 피드, 포스터)
                "4:3": (1152, 896),     # 가로 (프레젠테이션, 배너)
                "16:9": (1344, 768),    # 와이드 (유튜브 썸네일, 웹 배너)
                "9:16": (768, 1344),    # 세로 (Instagram Story, 모바일)
            }

    def get_size(self, ratio: str) -> Tuple[int, int]:
        """비율에 맞는 해상도 반환"""
        return self.RATIOS.get(ratio, (1024, 1024))


@dataclass
class GenerationConfig:
    """이미지 생성 하이퍼파라미터"""

    # 기본 생성 설정
    DEFAULT_STEPS: int = 40
    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_ASPECT_RATIO: str = "1:1"  # 기본 비율

    # 품질 설정 - 공통 네거티브 프롬프트
    NEGATIVE_PROMPT_BASE: str = (
        "low quality, blurry, distorted, ugly, deformed, bad anatomy, "
        "bad hands, extra fingers, missing fingers, fused fingers, too many fingers, "
        "mutated hands, poorly drawn hands, malformed limbs, "
        "watermark, text overlay, signature, logo, amateur photo, "
        "low resolution, oversaturated colors"
    )

    # 스타일별 추가 네거티브 프롬프트
    STYLE_NEGATIVE_PROMPTS: dict = None

    # 기본 네거티브 프롬프트 (Realistic 스타일용)
    NEGATIVE_PROMPT: str = (
        NEGATIVE_PROMPT_BASE + ", "
        "cartoon, anime style, 3d render, plastic looking, artificial"
    )

    # ControlNet 설정
    CONTROLNET_CONDITIONING_SCALE: float = 0.8

    # 업종별 스타일 프리셋
    INDUSTRY_STYLES: dict = None

    def __post_init__(self):
        # 스타일별 네거티브 프롬프트 초기화
        if self.STYLE_NEGATIVE_PROMPTS is None:
            self.STYLE_NEGATIVE_PROMPTS = {
                "ultra_realistic": self.NEGATIVE_PROMPT_BASE + ", cartoon, anime style, 3d render, plastic looking, artificial",
                "semi_realistic": self.NEGATIVE_PROMPT_BASE + ", cartoon, extreme anime style, 3d render",
                "anime": self.NEGATIVE_PROMPT_BASE + ", photorealistic, photograph, 3d render, plastic looking",
            }

        # 업종별 스타일 프리셋 초기화
        if self.INDUSTRY_STYLES is None:
            self.INDUSTRY_STYLES = {
                "cafe": {
                    "style_suffix": "warm lighting, cozy atmosphere, wooden furniture, coffee cups",
                    "negative_add": "cold atmosphere, sterile environment",
                },
                "restaurant": {
                    "style_suffix": "elegant dining, food presentation, ambient lighting",
                    "negative_add": "messy, unappetizing",
                },
                "retail": {
                    "style_suffix": "clean display, product showcase, bright lighting",
                    "negative_add": "cluttered, dark, disorganized",
                },
                "service": {
                    "style_suffix": "professional, clean, modern interior",
                    "negative_add": "unprofessional, messy",
                },
            }

    def get_negative_prompt(self, style: str = "ultra_realistic") -> str:
        """스타일에 맞는 네거티브 프롬프트 반환"""
        return self.STYLE_NEGATIVE_PROMPTS.get(style, self.NEGATIVE_PROMPT)


@dataclass
class PreprocessConfig:
    """전처리 설정"""

    # 배경 제거
    REMOVE_BG_ENABLED: bool = True

    # 이미지 품질 분석 임계값
    MIN_RESOLUTION: int = 512
    MIN_BRIGHTNESS: float = 0.3
    MAX_BRIGHTNESS: float = 0.9

    # 리사이징
    TARGET_SIZE: tuple = (1024, 1024)


@dataclass
class PostprocessConfig:
    """후처리 설정"""

    # 압축
    JPEG_QUALITY: int = 95
    PNG_COMPRESSION: int = 6

    # 텍스트 오버레이
    FONT_SIZE: int = 48
    FONT_COLOR: tuple = (255, 255, 255)  # RGB
    TEXT_POSITION: str = "bottom"  # top, bottom, center
    TEXT_PADDING: int = 20


# 전역 설정 인스턴스
model_config = ModelConfig()
aspect_ratio_templates = AspectRatioTemplates()
generation_config = GenerationConfig()
preprocess_config = PreprocessConfig()
postprocess_config = PostprocessConfig()
