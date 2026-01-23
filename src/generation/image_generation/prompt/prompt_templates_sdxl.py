"""
SDXL Hybrid Prompting System
벤치마크 분석 기반 자연어 + 키워드 혼합 구조

핵심 원칙:
1. Subject + Setting = 자연어 문장
2. Composition + Technical = 키워드
3. Quality tags 제거
4. Negative prompt 최소화 (5-7개)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.logging import get_logger


@dataclass
class PromptStructure:
    """
    SDXL 최적화 프롬프트 구조

    레이어 구성:
    - subject_phrase: 자연어 주요 피사체 (문장형)
    - setting: 배경/환경 (문장형)
    - composition: 구도 (키워드)
    - lighting: 조명 (구문/키워드 혼합)
    - style: 스타일 (키워드)
    - color: 색감 (구문)
    - details: 디테일 (키워드)
    - technical: 기술적 요소 (키워드)
    """

    # 자연어 구문 (Natural Language Phrases)
    subject_phrase: str = ""
    setting: str = ""
    lighting: str = ""
    color: str = ""

    # 키워드 (Keywords)
    composition: List[str] = None
    style: List[str] = None
    details: List[str] = None
    technical: List[str] = None

    def __post_init__(self):
        """기본값 초기화"""
        if self.composition is None:
            self.composition = []
        if self.style is None:
            self.style = []
        if self.details is None:
            self.details = []
        if self.technical is None:
            self.technical = []

    def build(self) -> str:
        """
        계층적 프롬프트 조립

        구조: [Medium + Subject] [Setting], [Keywords...]

        Returns:
            str: 완성된 SDXL 프롬프트
        """
        parts = []

        # 1. Subject Phrase (자연어 - 가장 중요)
        if self.subject_phrase:
            parts.append(self.subject_phrase)

        # 2. Setting (자연어 - 맥락 제공)
        if self.setting:
            parts.append(self.setting)

        # 3. Lighting (구문 - 분위기)
        if self.lighting:
            parts.append(self.lighting)

        # 4. Color (구문 - 색감)
        if self.color:
            parts.append(self.color)

        # 5. Composition (키워드 - 구도)
        if self.composition:
            parts.extend(self.composition)

        # 6. Style (키워드 - 스타일)
        if self.style:
            parts.extend(self.style)

        # 7. Details (키워드 - 디테일)
        if self.details:
            parts.extend(self.details)

        # 8. Technical (키워드 - 기술)
        if self.technical:
            parts.extend(self.technical)

        # ✅ 강화된 중복 제거
        seen_words = set()
        unique_parts = []

        for part in parts:
            if not part:
                continue

            part_lower = part.lower().strip()

            # 전체 일치 체크
            if part_lower in seen_words:
                continue

            # 단어 포함 관계 체크 (더 강력)
            words_in_part = set(part_lower.split())

            # 이미 본 단어들과 50% 이상 겹치면 스킵
            overlap = len(words_in_part & seen_words)
            if overlap > 0 and overlap / len(words_in_part) > 0.5:
                continue

            # 추가
            unique_parts.append(part)
            seen_words.add(part_lower)
            seen_words.update(words_in_part)

        # Join with ", " (SDXL 선호 구분자)
        return ", ".join(unique_parts)

    def to_dict(self) -> Dict:
        """디버깅용 딕셔너리 변환"""
        return {
            "subject_phrase": self.subject_phrase,
            "setting": self.setting,
            "lighting": self.lighting,
            "color": self.color,
            "composition": self.composition,
            "style": self.style,
            "details": self.details,
            "technical": self.technical
        }


class HybridPromptBuilder:
    """
    Hybrid Prompting 빌더

    업종별 템플릿을 기반으로 동적 프롬프트 생성
    """

    def __init__(self, industry_template: Dict):
        """
        Args:
            industry_template: industries.yaml에서 로드한 업종 템플릿
        """
        self.template = industry_template.get('prompt_template', {})

        self.logger = get_logger(__name__)

    def build_from_user_input(
        self,
        user_input: Dict,
        composition_type: Optional[str] = None
    ) -> PromptStructure:
        """
        사용자 입력 기반 프롬프트 생성

        Args:
            user_input: 파싱된 사용자 요청
                {
                    "product": "strawberry latte",
                    "theme": "warm",
                    "mood": "cozy"
                }
            composition_type: 구도 타입 (overhead, 45_degree 등)

        Returns:
            PromptStructure: 완성된 프롬프트 구조
        """
        structure = PromptStructure()

        # 1. Subject Phrase 생성 (자연어)
        structure.subject_phrase = self._build_subject_phrase(user_input)

        # 2. Setting 선택
        structure.setting = self._select_setting(user_input)

        # 3. Lighting 선택
        structure.lighting = self._select_lighting(user_input)

        # 4. Color 선택
        structure.color = self._select_color(user_input)

        # 5. Composition (키워드)
        structure.composition = self._select_composition(composition_type)

        # 6. Style (키워드)
        structure.style = self._select_style()

        # 7. Details (키워드)
        structure.details = self._select_details()

        # 8. Technical (키워드)
        structure.technical = self._select_technical()

        return structure

    def _build_subject_phrase(self, user_input: Dict) -> str:
        """
        Subject 자연어 문장 생성

        템플릿: "Professional food photography of {product} on {surface}"

        NOTE: user_input은 이미 영어로 번역된 상태
              (prompt_manager.extract_keywords_english()의 출력)

        Args:
            user_input: 영어 키워드 딕셔너리
                예: {"product": "strawberry latte", "surface": "marble table"}

        Returns:
            str: Subject 자연어 문장
        """
        patterns = self.template.get('subject_patterns', [])
        if not patterns:
            return ""

        # 첫 번째 패턴 사용
        pattern = patterns[0]

        # 핵심 피사체 결정 (우선순위)
        main_subject = (
            user_input.get('product') or
            user_input.get('dish') or
            user_input.get('item') or
            user_input.get('flowers') or
            user_input.get('subject') or
            'subject'
        )

        # 액션/상태 결정
        action_or_state = (
            user_input.get('activity') or
            user_input.get('state') or
            user_input.get('plating') or
            user_input.get('arrangement') or
            user_input.get('presentation') or
            user_input.get('display_method') or
            ''
        )

        # 템플릿 변수 기본값 설정
        defaults = {
            'drink_name': user_input.get('product', user_input.get('drink_name', main_subject)),
            'product': main_subject,
            'surface': user_input.get('surface', 'table'),
            'setting': user_input.get('setting', 'interior'),
            'decoration': user_input.get('decoration', 'detail'),
            'container': user_input.get('container', 'container'),
            'activity': user_input.get('activity', action_or_state or 'activity'),
            'focus': user_input.get('focus', 'detail'),
            'person_type': user_input.get('person_type', 'person'),
            'item': main_subject,
            'state': action_or_state or user_input.get('state', 'prepared'),
            'quality': user_input.get('quality', 'high quality'),
            'service': user_input.get('service', 'service'),
            'detail': user_input.get('detail', 'detail'),
            'process': user_input.get('process', 'process'),
            'presentation': action_or_state or user_input.get('presentation', 'display'),
            'texture_detail': user_input.get('texture_detail', 'texture'),
            'feature': user_input.get('feature', 'feature'),
            'dish': main_subject,
            'plating': action_or_state or user_input.get('plating', 'plated'),
            'cuisine_style': user_input.get('cuisine_style', 'gourmet'),
            'garnish': user_input.get('garnish', 'garnish'),
            'style': user_input.get('style', 'style'),
            'result': user_input.get('result', 'result'),
            'technique': user_input.get('technique', 'technique'),
            'stylist': user_input.get('stylist', 'stylist'),
            'design': user_input.get('design', 'design'),
            'artist': user_input.get('artist', 'artist'),
            'flowers': main_subject,
            'arrangement': action_or_state or user_input.get('arrangement', 'arrangement'),
            'display_method': action_or_state or user_input.get('display_method', 'displayed'),
            'display': action_or_state or user_input.get('display', 'display'),
            'subject': main_subject,
            'location': user_input.get('location', 'location'),
            'interaction': user_input.get('interaction', 'interaction')
        }

        # {변수} 치환 (KeyError 방지)
        try:
            result = pattern.format(**defaults)
        except KeyError as e:
            # 누락된 변수가 있으면 구체적인 fallback
            self.logger.error(f"Warning: Missing template variable {e}, using fallback")
            if action_or_state:
                result = f"Professional photograph of {main_subject} {action_or_state}"
            else:
                result = f"Professional photograph of {main_subject}"

        return result

    def _select_setting(self, user_input: Dict) -> str:
        """
        Setting 구문 선택 (자연어 연결어 포함)
        """
        patterns = self.template.get('setting_patterns', [])
        if not patterns:
            return ""

        # 사용자 mood에 따라 선택 (간단한 로직)
        mood = user_input.get('mood', 'neutral')
        selected_pattern = patterns[1] if mood == 'cozy' and len(patterns) > 1 else patterns[0]

        # 자연어 연결어 추가 (with, in, at 등)
        # 이미 연결어가 있으면 그대로, 없으면 추가
        if not any(conn in selected_pattern.lower() for conn in [' with ', ' in ', ' at ', ' featuring ']):
            # "interior" → "interior with" 같은 변환은 하지 않고 그대로 유지
            pass

        return selected_pattern

    def _select_lighting(self, user_input: Dict) -> str:
        """
        Lighting 구문 선택 (자연어 연결어 강화)
        """
        phrases = self.template.get('lighting_phrases', [])
        if not phrases:
            return ""

        # 시간대에 따라 선택
        time = user_input.get('time', 'day')
        selected_phrase = phrases[0]  # 기본

        if time == 'morning' and len(phrases) > 2:
            selected_phrase = phrases[2]  # "bright morning light"
        elif time == 'afternoon' and len(phrases) > 1:
            selected_phrase = phrases[1]  # "warm afternoon sunlight"

        # 자연어 강화: 연결어가 없으면 추가
        if not any(conn in selected_phrase.lower() for conn in [' from ', ' with ', ' creating ', ' streaming ']):
            # "natural light" → "natural light streaming from window" 같은 확장은
            # 템플릿에서 이미 되어 있다고 가정하고 그대로 반환
            pass

        return selected_phrase

        # 시간대에 따라 선택
        time = user_input.get('time', 'day')
        if time == 'morning' and len(phrases) > 2:
            return phrases[2]  # "bright morning light"
        elif time == 'afternoon':
            return phrases[1]  # "warm afternoon sunlight"

        return phrases[0]  # 기본

    def _select_color(self, user_input: Dict) -> str:
        """Color 구문 선택"""
        phrases = self.template.get('color_phrases', [])
        if not phrases:
            return ""

        # 테마에 따라
        theme = user_input.get('theme', 'neutral')
        if theme == 'warm' and len(phrases) > 1:
            return phrases[1]  # "soft pink and beige tones"

        return phrases[0]

    def _select_composition(self, composition_type: Optional[str]) -> List[str]:
        """
        Composition 키워드 선택 (필수)

        최소 1개 보장
        """
        keywords = self.template.get('composition_keywords', [])

        # 키워드 없으면 기본값
        if not keywords:
            return ["centered composition"]  # 기본 composition

        if composition_type == 'overhead':
            result = [kw for kw in keywords if 'overhead' in kw or 'top' in kw]
            if result:
                return result
        elif composition_type == '45_degree':
            result = [kw for kw in keywords if '45' in kw or 'angle' in kw]
            if result:
                return result

        # 기본: 무조건 1개 반환
        return [keywords[0]]

    def _select_style(self) -> List[str]:
        """Style 키워드 선택"""
        keywords = self.template.get('style_keywords', [])
        # 1개만 선택 (길이 축소)
        return keywords[:1] if keywords else []

    def _select_details(self) -> List[str]:
        """Details 키워드 선택"""
        keywords = self.template.get('details_keywords', [])
        # 1-2개만 선택 (길이 축소)
        return keywords[:2] if keywords else []

    def _select_technical(self) -> List[str]:
        """
        Technical 키워드 선택

        렌즈는 1개만 선택 (중복 방지)
        """
        keywords = self.template.get('technical_keywords', [])
        if not keywords:
            return []

        # 렌즈와 비렌즈 분리
        import re
        lens_pattern = re.compile(r'\d+(?:-\d+)?mm')

        lens_keywords = []
        other_keywords = []

        for kw in keywords:
            if lens_pattern.search(kw):
                lens_keywords.append(kw)
            else:
                other_keywords.append(kw)

        # 렌즈 1개만 선택
        selected = []
        if lens_keywords:
            selected.append(lens_keywords[0])  # 첫 번째 렌즈만

        # 나머지 기술 요소 2개
        selected.extend(other_keywords[:2])

        return selected[:3]  # 최대 3개


class NegativePromptBuilder:
    """
    Negative Prompt 빌더

    텍스트 생성 방지를 최우선으로 함
    """

    # 텍스트 방지 키워드 (최우선 - 항상 포함)
    ANTI_TEXT_KEYWORDS = [
        "text",
        "letters",
        "words",
        "watermark",
        "signature",
        "logo",
        "caption",
        "writing",
        "typography",
        "font",
        "label"
    ]

    # 모든 업종 공통 (Base)
    COMMON_NEGATIVE = [
        "low quality",
        "blurry",
        "deformed",
        "ugly"
    ]

    # 스타일별 Negative
    STYLE_NEGATIVE = {
        "realistic": ["cartoon", "illustration", "painting", "anime", "drawing"],
        "semi_realistic": ["cartoon", "anime", "sketch", "unfinished"],
        "anime": ["photorealistic", "photograph", "realistic", "3d render"]
    }

    # 업종별 추가 Negative
    INDUSTRY_NEGATIVE = {
        "cafe": ["artificial", "plastic-looking"],
        "gym": ["poor anatomy", "static pose"],
        "laundry": ["dirty", "wrinkled"],
        "bakery": ["unappealing", "flat"],  # "artificial" 제거 (중복)
        "restaurant": ["unappetizing", "fake food"],
        "hair_salon": ["messy", "unprofessional"],
        "nail_salon": ["uneven", "sloppy"],
        "flower_shop": ["wilted", "artificial flowers"],
        "clothing_store": ["poor fit", "wrinkled"],
        "general": ["generic", "amateur"]
    }

    @classmethod
    def build(cls, industry: str, style: str = "realistic") -> str:
        """
        Negative Prompt 생성

        우선순위:
        1. 텍스트 방지 키워드 (필수)
        2. 스타일별 Negative
        3. 업종별 Negative
        4. 공통 Negative

        Args:
            industry: 업종 코드 (cafe, gym 등)
            style: 스타일 (realistic, anime, illustration)

        Returns:
            str: Negative prompt
        """
        negatives = []

        # 1. 텍스트 방지 (최우선 - 핵심 5개)
        negatives.extend(cls.ANTI_TEXT_KEYWORDS[:5])

        # 2. 스타일별 Negative
        style_specific = cls.STYLE_NEGATIVE.get(style, cls.STYLE_NEGATIVE["realistic"])
        negatives.extend(style_specific[:3])

        # 3. 업종별 Negative
        industry_specific = cls.INDUSTRY_NEGATIVE.get(industry, [])
        negatives.extend(industry_specific[:2])

        # 4. 공통 Negative
        negatives.extend(cls.COMMON_NEGATIVE[:2])

        # 중복 제거
        seen = set()
        unique_negatives = []
        for neg in negatives:
            if neg.lower() not in seen:
                seen.add(neg.lower())
                unique_negatives.append(neg)

        return ", ".join(unique_negatives)


# ============================================
# 유틸리티 함수
# ============================================

def remove_quality_tags(prompt: str) -> str:
    """
    Quality spam 태그 제거

    SDXL에서 역효과를 내는 태그들:
    - masterpiece, best quality, high quality
    - 8k, 4k, ultra detailed
    - award-winning 등
    """
    quality_spam = [
        "masterpiece",
        "best quality",
        "high quality",
        "highest quality",
        "ultra detailed",
        "highly detailed",
        "8k",
        "4k",
        "16k",
        "award-winning",
        "award winning",
        "professional quality"
    ]

    # 소문자 변환 후 제거
    prompt_lower = prompt.lower()
    for spam in quality_spam:
        prompt_lower = prompt_lower.replace(spam.lower(), "")

    # 연속 쉼표 정리
    import re
    prompt_clean = re.sub(r',\s*,', ',', prompt_lower)
    prompt_clean = re.sub(r'^\s*,\s*|\s*,\s*$', '', prompt_clean)

    return prompt_clean.strip()


def apply_prompt_weights(prompt: str, weights: Dict[str, float]) -> str:
    """
    프롬프트 가중치 적용

    SDXL은 가중치에 매우 민감 (1.1 ~ 1.4 권장)

    Args:
        prompt: 원본 프롬프트
        weights: {"keyword": 1.3} 형태

    Returns:
        str: 가중치가 적용된 프롬프트
    """
    result = prompt

    for keyword, weight in weights.items():
        if keyword in result:
            # (keyword:weight) 형식으로 치환
            weighted = f"({keyword}:{weight:.1f})"
            result = result.replace(keyword, weighted, 1)  # 첫 번째만

    return result
