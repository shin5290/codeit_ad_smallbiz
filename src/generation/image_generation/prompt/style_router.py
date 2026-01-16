"""
Style Router (Z-Image Turbo용)
스타일에 따라 적절한 LoRA 및 프롬프트 구성 안내

Z-Image Turbo에서는:
- 스타일은 주로 LoRA로 적용
- 프롬프트에는 기본적인 스타일 힌트만 제공
- Negative prompt 미지원
"""

from typing import Dict, Optional
from .prompt_templates import PromptStructure


class StyleRouter:
    """
    Z-Image Turbo 스타일 라우터

    지원 스타일 (LoRA 기반):
    - realistic: 실사 사진 스타일 (베이스 모델)
    - ultra_realistic: realistic과 동일
    - semi_realistic: 반실사 스타일 (LoRA)
    - anime: 애니메이션 스타일 (LoRA)
    """

    # 스타일별 프롬프트 시작 문구
    STYLE_PREFIXES = {
        "realistic": "Professional commercial photography of",
        "ultra_realistic": "Professional commercial photography of",
        "semi_realistic": "Highly detailed digital artwork of",
        "anime": "Vibrant anime style illustration of"
    }

    # 스타일별 기술 설명 (자연어 형태)
    STYLE_DESCRIPTIONS = {
        "realistic": "shot on Canon EOS R5 with 85mm f/1.4 lens, natural lighting, shallow depth of field, professional color grading",
        "ultra_realistic": "shot on Canon EOS R5 with 85mm f/1.4 lens, natural lighting, shallow depth of field, professional color grading",
        "semi_realistic": "cinematic lighting, painterly style, detailed rendering, soft edges with sharp details",
        "anime": "clean bold outlines, flat color shading, vibrant saturated colors, anime aesthetic"
    }

    # 스타일별 텍스처/디테일 키워드
    STYLE_TEXTURES = {
        "realistic": "skin texture, fabric weave, fine details, natural imperfections",
        "ultra_realistic": "skin texture, fabric weave, fine details, natural imperfections",
        "semi_realistic": "painterly texture, artistic details, soft blending",
        "anime": "clean lines, smooth shading, sparkle effects"
    }

    # 스타일별 LoRA 파일 매핑 (text2image.py와 일치)
    STYLE_LORA_MAP = {
        "realistic": None,  # 베이스 모델 사용
        "ultra_realistic": None,  # 베이스 모델 사용
        "semi_realistic": "OB半写实肖像画2.0 OB Semi-Realistic Portraits z- image turbo(1).safetensors",
        "anime": "Anime-Z.safetensors",
    }

    @classmethod
    def get_style_prefix(cls, style: str) -> str:
        """스타일별 프롬프트 시작 문구 반환"""
        return cls.STYLE_PREFIXES.get(style, cls.STYLE_PREFIXES["realistic"])

    @classmethod
    def get_style_description(cls, style: str) -> str:
        """스타일별 기술 설명 반환"""
        return cls.STYLE_DESCRIPTIONS.get(style, cls.STYLE_DESCRIPTIONS["realistic"])

    @classmethod
    def get_style_textures(cls, style: str) -> str:
        """스타일별 텍스처 키워드 반환"""
        return cls.STYLE_TEXTURES.get(style, cls.STYLE_TEXTURES["realistic"])

    @classmethod
    def get_lora_file(cls, style: str) -> Optional[str]:
        """스타일에 해당하는 LoRA 파일명 반환"""
        return cls.STYLE_LORA_MAP.get(style)

    @classmethod
    def build_subject_phrase(
        cls,
        style: str,
        subject: str,
        additional_context: str = ""
    ) -> str:
        """
        스타일에 맞는 Subject Phrase 생성

        Args:
            style: 스타일 (realistic, semi_realistic, anime)
            subject: 주요 피사체
            additional_context: 추가 맥락 (on marble table, in cafe 등)

        Returns:
            str: 스타일에 맞는 Subject Phrase
        """
        prefix = cls.get_style_prefix(style)

        if additional_context:
            return f"{prefix} {subject}. {additional_context}"
        else:
            return f"{prefix} {subject}"

    @classmethod
    def build_prompt(
        cls,
        style: str,
        subject: str,
        setting: str = "",
        lighting: str = "",
        atmosphere: str = "",
        details: str = ""
    ) -> PromptStructure:
        """
        스타일에 맞는 완전한 PromptStructure 생성

        Args:
            style: 스타일
            subject: 주요 피사체
            setting: 배경/환경
            lighting: 조명
            atmosphere: 분위기
            details: 디테일

        Returns:
            PromptStructure: 완성된 프롬프트 구조
        """
        structure = PromptStructure()

        # 1. Subject Phrase (스타일별 prefix 적용)
        structure.subject_phrase = cls.build_subject_phrase(style, subject)

        # 2. Setting
        structure.setting = setting

        # 3. Lighting
        structure.lighting = lighting if lighting else "natural soft lighting"

        # 4. Atmosphere
        structure.atmosphere = atmosphere

        # 5. Style (스타일별 기술 설명)
        structure.style = cls.get_style_description(style)

        # 6. Details (스타일별 텍스처 + 추가 디테일)
        style_textures = cls.get_style_textures(style)
        if details:
            structure.details = f"{style_textures}, {details}"
        else:
            structure.details = style_textures

        return structure

    @classmethod
    def build_full_prompt(
        cls,
        style: str,
        subject: str,
        action: str = "",
        setting: str = "",
        lighting: str = "",
        atmosphere: str = "",
        details: str = ""
    ) -> str:
        """
        완전한 Z-Image Turbo 프롬프트 문자열 생성

        Args:
            style: 스타일
            subject: 주요 피사체
            action: 동작/상태
            setting: 배경/환경
            lighting: 조명
            atmosphere: 분위기
            details: 추가 디테일

        Returns:
            str: 완전한 프롬프트 문자열
        """
        parts = []

        # 1. Subject + Action (가장 중요 - 맨 앞)
        prefix = cls.get_style_prefix(style)
        if action:
            parts.append(f"{prefix} {subject} {action}")
        else:
            parts.append(f"{prefix} {subject}")

        # 2. Setting
        if setting:
            parts.append(setting)

        # 3. Lighting
        if lighting:
            parts.append(lighting)
        else:
            parts.append("natural soft lighting")

        # 4. Atmosphere
        if atmosphere:
            parts.append(atmosphere)

        # 5. Style description
        parts.append(cls.get_style_description(style))

        # 6. Textures
        parts.append(cls.get_style_textures(style))

        # 7. Additional details
        if details:
            parts.append(details)

        # 문장으로 연결
        return ". ".join(parts) + "."

    @classmethod
    def get_supported_styles(cls) -> list:
        """지원하는 스타일 목록 반환"""
        return list(cls.STYLE_PREFIXES.keys())

    @classmethod
    def is_valid_style(cls, style: str) -> bool:
        """유효한 스타일인지 확인"""
        return style in cls.STYLE_PREFIXES

    @classmethod
    def normalize_style(cls, style: str) -> str:
        """
        스타일 정규화

        Args:
            style: 입력 스타일

        Returns:
            str: 정규화된 스타일 (유효하지 않으면 "realistic")
        """
        if cls.is_valid_style(style):
            return style
        return "realistic"
