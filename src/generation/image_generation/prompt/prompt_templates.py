"""
Z-Image Turbo Prompt Templates
Z-Image Turbo 최적화 프롬프트 구조

핵심 원칙:
1. 긴 자연어 문장 형태
2. Negative prompt 미지원
3. 가중치 문법 미지원
4. 주요 요소는 프롬프트 앞부분에 배치
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptStructure:
    """
    Z-Image Turbo 최적화 프롬프트 구조

    자연어 문장 형태로 구성 (키워드 나열이 아닌 설명적 문장)
    키워드 나열보다 설명적 문장 선호

    레이어 구성:
    - subject_phrase: 주요 피사체 (문장형, 가장 중요 - 맨 앞에)
    - action: 동작/상태 (문장형)
    - setting: 배경/환경 (문장형)
    - lighting: 조명 (문장형)
    - atmosphere: 분위기/무드 (문장형)
    - style: 스타일/매체 (문장형)
    - details: 디테일/텍스처 (문장형)
    """

    # 자연어 구문 (Natural Language - 모두 문장형)
    subject_phrase: str = ""
    action: str = ""
    setting: str = ""
    lighting: str = ""
    atmosphere: str = ""
    style: str = ""
    details: str = ""

    def build(self) -> str:
        """
        자연어 프롬프트 조립

        Z-Image Turbo 권장 구조:
        [Subject + Action] [Setting] [Lighting] [Atmosphere] [Style] [Details]

        Returns:
            str: 완성된 Z-Image Turbo 프롬프트
        """
        parts = []

        # 1. Subject Phrase (가장 중요 - 맨 앞에)
        if self.subject_phrase:
            parts.append(self.subject_phrase)

        # 2. Action (동작/상태)
        if self.action:
            parts.append(self.action)

        # 3. Setting (배경/환경)
        if self.setting:
            parts.append(self.setting)

        # 4. Lighting (조명)
        if self.lighting:
            parts.append(self.lighting)

        # 5. Atmosphere (분위기)
        if self.atmosphere:
            parts.append(self.atmosphere)

        # 6. Style (스타일/매체)
        if self.style:
            parts.append(self.style)

        # 7. Details (디테일/텍스처)
        if self.details:
            parts.append(self.details)

        # 자연스러운 문장으로 연결
        # Z-Image는 ". " 또는 ", " 연결 모두 가능
        # 문장형이므로 ". "로 연결하는 것이 더 자연스러움
        result = ". ".join(filter(None, parts))

        # 마지막에 마침표 추가
        if result and not result.endswith("."):
            result += "."

        return result

    def to_dict(self) -> Dict:
        """디버깅용 딕셔너리 변환"""
        return {
            "subject_phrase": self.subject_phrase,
            "action": self.action,
            "setting": self.setting,
            "lighting": self.lighting,
            "atmosphere": self.atmosphere,
            "style": self.style,
            "details": self.details
        }


class ZITPromptBuilder:
    """
    Z-Image Turbo 프롬프트 빌더

    특징:
    - 긴 자연어 문장 생성
    - Negative prompt 미지원
    - 스타일은 LoRA로 적용 (프롬프트에는 최소한만)
    """

    # 스타일별 기본 프롬프트 시작 문구
    STYLE_OPENERS = {
        "realistic": "Professional commercial photography",
        "semi_realistic": "Highly detailed digital artwork",
        "anime": "Vibrant anime style illustration"
    }

    # 스타일별 기술 설명
    STYLE_TECHNICAL = {
        "realistic": "shot on Canon EOS R5 with 85mm lens, natural lighting, shallow depth of field",
        "semi_realistic": "cinematic lighting, painterly style, detailed rendering",
        "anime": "clean bold outlines, flat color shading, vibrant colors"
    }

    def __init__(self, industry_template: Dict = None):
        """
        Args:
            industry_template: industries.yaml에서 로드한 업종 템플릿 (선택)
        """
        self.template = industry_template.get('prompt_template', {}) if industry_template else {}

    def build_from_user_input(
        self,
        user_input: Dict,
        style: str = "realistic"
    ) -> PromptStructure:
        """
        사용자 입력 기반 Z-Image Turbo 프롬프트 생성

        Args:
            user_input: 파싱된 사용자 요청
                {
                    "product": "strawberry latte",
                    "theme": "warm",
                    "mood": "cozy"
                }
            style: 스타일 (realistic, semi_realistic, anime)

        Returns:
            PromptStructure: 완성된 프롬프트 구조
        """
        structure = PromptStructure()

        # 1. Subject Phrase (가장 중요)
        structure.subject_phrase = self._build_subject_phrase(user_input, style)

        # 2. Action
        structure.action = self._build_action(user_input)

        # 3. Setting
        structure.setting = self._build_setting(user_input)

        # 4. Lighting
        structure.lighting = self._build_lighting(user_input)

        # 5. Atmosphere
        structure.atmosphere = self._build_atmosphere(user_input)

        # 6. Style/Technical
        structure.style = self.STYLE_TECHNICAL.get(style, self.STYLE_TECHNICAL["realistic"])

        # 7. Details
        structure.details = self._build_details(user_input)

        return structure

    def _build_subject_phrase(self, user_input: Dict, style: str) -> str:
        """주요 피사체 문장 생성"""
        opener = self.STYLE_OPENERS.get(style, self.STYLE_OPENERS["realistic"])

        # 핵심 피사체 결정
        main_subject = (
            user_input.get('product') or
            user_input.get('dish') or
            user_input.get('item') or
            user_input.get('subject') or
            'subject'
        )

        return f"{opener} of {main_subject}"

    def _build_action(self, user_input: Dict) -> str:
        """동작/상태 문장 생성"""
        action = (
            user_input.get('activity') or
            user_input.get('state') or
            user_input.get('presentation') or
            ''
        )

        if action:
            return f"shown {action}"
        return ""

    def _build_setting(self, user_input: Dict) -> str:
        """배경/환경 문장 생성"""
        surface = user_input.get('surface', '')
        setting = user_input.get('setting', '')

        if surface and setting:
            return f"placed on {surface} in {setting}"
        elif surface:
            return f"placed on {surface}"
        elif setting:
            return f"in {setting}"
        return ""

    def _build_lighting(self, user_input: Dict) -> str:
        """조명 문장 생성"""
        # 템플릿에서 조명 구문 가져오기
        lighting_phrases = self.template.get('lighting_phrases', [])

        if lighting_phrases:
            return lighting_phrases[0]

        # 기본 조명
        time = user_input.get('time', 'day')
        if time == 'morning':
            return "soft morning light streaming through window"
        elif time == 'evening':
            return "warm golden hour lighting"
        else:
            return "natural soft lighting"

    def _build_atmosphere(self, user_input: Dict) -> str:
        """분위기 문장 생성"""
        mood = user_input.get('mood', '')
        theme = user_input.get('theme', '')

        if mood and theme:
            return f"{mood} and {theme} atmosphere"
        elif mood:
            return f"{mood} atmosphere"
        elif theme:
            return f"{theme} mood"
        return ""

    def _build_details(self, user_input: Dict) -> str:
        """디테일/텍스처 문장 생성"""
        # 템플릿에서 디테일 가져오기
        details_keywords = self.template.get('details_keywords', [])

        if details_keywords:
            return f"with {', '.join(details_keywords[:3])}"

        # 기본 디테일
        return "with fine textures and sharp details"


# ============================================
# 유틸리티 함수 (ZIT용)
# ============================================

def build_simple_prompt(
    subject: str,
    style: str = "realistic",
    additional_context: str = ""
) -> str:
    """
    간단한 Z-Image Turbo 프롬프트 생성

    Args:
        subject: 주요 피사체
        style: 스타일 (realistic, semi_realistic, anime)
        additional_context: 추가 맥락

    Returns:
        str: Z-Image Turbo 프롬프트
    """
    openers = {
        "realistic": "Professional commercial photography of",
        "semi_realistic": "Highly detailed digital artwork of",
        "anime": "Vibrant anime style illustration of"
    }

    opener = openers.get(style, openers["realistic"])

    if additional_context:
        return f"{opener} {subject}. {additional_context}."
    else:
        return f"{opener} {subject}."


# ============================================
# 하위 호환성을 위한 Legacy 클래스
# ============================================

class HybridPromptBuilder(ZITPromptBuilder):
    """
    Legacy 호환성 - ZITPromptBuilder의 별칭

    기존 코드에서 HybridPromptBuilder를 import하는 경우를 위해 유지
    """
    pass


class NegativePromptBuilder:
    """
    Legacy 호환성 - Z-Image Turbo는 Negative Prompt 미지원

    이 클래스는 호환성을 위해 유지되지만,
    항상 빈 문자열을 반환합니다.
    """

    @classmethod
    def build(cls, industry: str = "", style: str = "realistic") -> str:
        """
        Z-Image Turbo는 Negative Prompt를 지원하지 않습니다.

        Returns:
            str: 항상 빈 문자열
        """
        # Z-Image Turbo는 CFG를 사용하지 않으므로 negative prompt 미지원
        return ""
