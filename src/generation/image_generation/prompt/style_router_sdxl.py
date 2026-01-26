"""
Style Router
스타일에 따라 적절한 프롬프트 빌더로 분기
"""

from typing import Dict, Optional
from .prompt_templates import PromptStructure


class StyleRouter:
    """
    스타일별 프롬프트 빌더 라우터

    지원 스타일:
    - realistic: 실사 사진 스타일 (RealVisXL V4.0)
    - semi_realistic: 반실사 스타일 (BSS Equinox)
    - anime: 애니메이션 스타일 (Animagine XL 3.1)
    """

    # 스타일별 기본 Medium/Prefix
    STYLE_PREFIXES = {
        "realistic": "Professional commercial photography of",
        "semi_realistic": "Highly detailed digital artwork of",
        "anime": "anime style illustration of"
    }

    # 스타일별 기술 키워드
    STYLE_TECHNICAL = {
        "realistic": [
            "shot on Canon EOS R5",
            "85mm lens",
            "natural lighting",
            "shallow depth of field",
            "professional color grading"
        ],
        "semi_realistic": [
            "highly detailed",
            "cinematic lighting",
            "digital painting",
            "semi-realistic style",
            "soft rendering"
        ],
        "anime": [
            "anime aesthetic",
            "vibrant colors",
            "clean lineart",
            "detailed anime art",
            "modern anime style"
        ]
    }

    # 스타일별 품질 키워드
    STYLE_QUALITY = {
        "realistic": ["high resolution", "sharp focus", "detailed"],
        "semi_realistic": ["detailed", "polished", "artistic"],
        "anime": ["studio quality", "detailed", "vibrant"]
    }

    @classmethod
    def get_style_prefix(cls, style: str) -> str:
        """스타일별 프롬프트 시작 문구 반환"""
        return cls.STYLE_PREFIXES.get(style, cls.STYLE_PREFIXES["realistic"])

    @classmethod
    def get_style_technical(cls, style: str) -> list:
        """스타일별 기술 키워드 반환"""
        return cls.STYLE_TECHNICAL.get(style, cls.STYLE_TECHNICAL["realistic"])

    @classmethod
    def get_style_quality(cls, style: str) -> list:
        """스타일별 품질 키워드 반환"""
        return cls.STYLE_QUALITY.get(style, cls.STYLE_QUALITY["realistic"])

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
            return f"{prefix} {subject}, {additional_context}"
        else:
            return f"{prefix} {subject}"

    @classmethod
    def build_prompt(
        cls,
        style: str,
        subject: str,
        setting: str = "",
        lighting: str = "",
        color: str = "",
        composition: list = None,
        details: list = None
    ) -> PromptStructure:
        """
        스타일에 맞는 완전한 PromptStructure 생성

        Args:
            style: 스타일
            subject: 주요 피사체
            setting: 배경/환경
            lighting: 조명
            color: 색감
            composition: 구도 키워드
            details: 디테일 키워드

        Returns:
            PromptStructure: 완성된 프롬프트 구조
        """
        structure = PromptStructure()

        # 1. Subject Phrase (스타일별 prefix 적용)
        structure.subject_phrase = cls.build_subject_phrase(style, subject)

        # 2. Setting
        structure.setting = setting

        # 3. Lighting
        structure.lighting = lighting

        # 4. Color
        structure.color = color

        # 5. Composition
        structure.composition = composition or []

        # 6. Style (스타일별 품질 키워드)
        structure.style = cls.get_style_quality(style)

        # 7. Details
        structure.details = details or []

        # 8. Technical (스타일별 기술 키워드)
        structure.technical = cls.get_style_technical(style)[:2]

        return structure

    @classmethod
    def get_supported_styles(cls) -> list:
        """지원하는 스타일 목록 반환"""
        return list(cls.STYLE_PREFIXES.keys())

    @classmethod
    def is_valid_style(cls, style: str) -> bool:
        """유효한 스타일인지 확인"""
        return style in cls.STYLE_PREFIXES
