"""
광고 문구 프롬프트 관리자 (v3.2.0)
industries.yaml 기반 업종별 광고 문구 생성

작성자: 배현석 -> 신승목
버전: 3.2.0
핵심 변경 사항: 소상공인 전체 업종으로 확장

사용 예시:
    from src.generation.text_generation.prompt_manager import PromptTemplateManager

    manager = PromptTemplateManager()

    # 업종 자동 감지
    industry = manager.detect_industry("삼겹살 맛집 홍보")
    # 결과: "s1_hot_cooking"

    # 이미지 프롬프트 생성
    prompts = manager.generate_image_prompt("카페 신메뉴 딸기라떼 홍보")
    # 결과: {"positive": "...", "negative": "", "industry": "s3_emotional"}
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
import yaml


class IndustryConfigLoader:
    """
    industries.yaml 로더 (v3.0.0 계층 구조)

    6개 등급 (S-E), 18개 하위 그룹 지원
    """

    GRADES = ['s_grade', 'a_grade', 'b_grade', 'c_grade', 'd_grade', 'e_grade']

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 파일 경로 (None이면 기본 경로 사용)
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "industries.yaml"
        else:
            self.config_path = Path(config_path)

        self.config = self._load_config()
        self._subgroup_cache = self._build_subgroup_cache()
        self._keyword_map = self._build_keyword_map()

    def _load_config(self) -> Dict:
        """YAML 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_subgroup_cache(self) -> Dict[str, Tuple[str, Dict]]:
        """하위 그룹 코드 → (등급, 데이터) 캐시 생성"""
        cache = {}
        for grade_key in self.GRADES:
            grade_data = self.config.get(grade_key, {})
            for key, value in grade_data.items():
                # 하위 그룹은 dict이고 prompt_template 또는 keyword_priorities를 가짐
                if isinstance(value, dict) and ('prompt_template' in value or 'keyword_priorities' in value):
                    cache[key] = (grade_key, value)
        return cache

    def _build_keyword_map(self) -> Dict[str, List[str]]:
        """업종 감지용 키워드 맵 생성"""
        keyword_map = {}

        for subgroup_code, (grade_key, subgroup_data) in self._subgroup_cache.items():
            keywords = []

            # 1. korean_keywords 필드 (한글)
            korean_keywords = subgroup_data.get('korean_keywords', [])
            keywords.extend(korean_keywords)

            # 2. keywords 필드 (영어)
            eng_keywords = subgroup_data.get('keywords', [])
            keywords.extend([kw.lower() for kw in eng_keywords])

            # 3. businesses 필드에서 추출
            businesses = subgroup_data.get('businesses', [])
            for business in businesses:
                # 영어 업종명
                words = business.lower().replace('/', ' ').replace('_', ' ').split()
                keywords.extend(words)

            keyword_map[subgroup_code] = list(set(keywords))

        return keyword_map

    def get_industry(self, industry_code: str) -> Optional[Dict]:
        """
        업종 설정 가져오기

        Args:
            industry_code: 하위 그룹 코드 (s1_hot_cooking, a1_beauty 등)

        Returns:
            Dict: 업종 설정 또는 None
        """
        if industry_code in self._subgroup_cache:
            return self._subgroup_cache[industry_code][1]

        # 레거시 코드 호환
        legacy_mapping = {
            "cafe": "s3_emotional",
            "bakery": "s3_emotional",
            "restaurant": "s1_hot_cooking",
            "gym": "a2_wellness",
            "hair_salon": "a1_beauty",
            "nail_salon": "a1_beauty",
            "flower_shop": "a4_delicate_care",
            "clothing_store": "a3_fashion",
            "general": "s3_emotional"
        }

        if industry_code in legacy_mapping:
            mapped_code = legacy_mapping[industry_code]
            if mapped_code in self._subgroup_cache:
                return self._subgroup_cache[mapped_code][1]

        # 기본값
        return self._subgroup_cache.get('s3_emotional')

    def detect_industry(self, user_input: str) -> str:
        """
        사용자 입력에서 업종 자동 감지

        Args:
            user_input: 사용자 입력 텍스트

        Returns:
            str: 감지된 하위 그룹 코드 또는 "s3_emotional" (기본값)
        """
        user_input_lower = user_input.lower()

        scores = {}
        for subgroup_code, keywords in self._keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                scores[subgroup_code] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return "s3_emotional"

    def get_grade_for_industry(self, industry_code: str) -> Optional[str]:
        """업종의 등급 반환"""
        if industry_code in self._subgroup_cache:
            return self._subgroup_cache[industry_code][0]
        return None

    def get_all_subgroups(self) -> List[str]:
        """모든 하위 그룹 코드 리스트"""
        return list(self._subgroup_cache.keys())

    def get_tone_guide(self) -> Dict:
        """톤 가이드 반환"""
        return self.config.get('tone_guide', {})

    def get_platform_guide(self) -> Dict:
        """플랫폼 가이드 반환"""
        return self.config.get('platform_guide', {})


class AdCopyPromptBuilder:
    """
    광고 문구 생성용 프롬프트 빌더

    AIDA 프레임워크 기반:
    - Attention: 주목 끌기
    - Interest: 관심 유발
    - Desire: 욕구 자극
    - Action: 행동 유도
    """

    def __init__(self, industry_config: Dict, tone_guide: Dict = None):
        """
        Args:
            industry_config: 업종 설정 (industries.yaml에서 로드)
            tone_guide: 톤 가이드 (선택)
        """
        self.config = industry_config
        self.tone_guide = tone_guide or {}

    def build_system_prompt(self, tone: str = "warm", max_length: int = 20) -> str:
        """
        시스템 프롬프트 생성

        Args:
            tone: 톤 (warm, professional, friendly, energetic, practical, respectful)
            max_length: 최대 글자 수

        Returns:
            str: 시스템 프롬프트
        """
        # 기본 프롬프트
        base_prompt = f"""당신은 소상공인을 위한 전문 광고 카피라이터입니다.
AIDA 프레임워크를 활용하여 짧고 임팩트 있는 광고 문구를 만들어주세요.

규칙:
- {max_length}자 이내 (공백 포함)
- 번호, 특수문자 없이 문구만 작성
- 한국어로 작성 (사용자가 다른 언어 요청 시 제외)
- 광고 문구 1개만 생성"""

        # 업종별 키워드 우선순위 (한국어 우선, 없으면 영어)
        keyword_priorities = self.config.get('keyword_priorities_ko', self.config.get('keyword_priorities', {}))
        if keyword_priorities:
            base_prompt += "\n\n키워드 우선순위 (이 순서대로 강조):\n"
            # 딕셔너리인 경우 (예: sensory_35: [...], temperature_25: [...])
            if isinstance(keyword_priorities, dict):
                for i, (priority_key, keywords) in enumerate(list(keyword_priorities.items())[:5], 1):
                    # 키에서 카테고리명 추출 (예: sensory_35 -> sensory)
                    category = priority_key.rsplit('_', 1)[0] if '_' in priority_key else priority_key
                    if isinstance(keywords, list) and keywords:
                        base_prompt += f"{i}. {category}: {', '.join(keywords[:3])}\n"
            # 리스트인 경우 (기존 방식 호환)
            elif isinstance(keyword_priorities, list):
                for i, kw in enumerate(keyword_priorities[:5], 1):
                    base_prompt += f"{i}. {kw}\n"

        # 업종별 브랜드 보이스 (한국어 우선)
        brand_voice = self.config.get('brand_voice_ko', self.config.get('brand_voice', ''))
        if brand_voice:
            base_prompt += f"\n브랜드 보이스:\n{brand_voice}"

        # 업종별 CTA (한국어 우선)
        cta = self.config.get('cta_ko', self.config.get('cta', []))
        if cta:
            if isinstance(cta, list):
                base_prompt += f"\n\nCTA 스타일:\n{', '.join(cta[:3])}"
            else:
                base_prompt += f"\n\nCTA 스타일:\n{cta}"

        # 제외 키워드 (한국어 우선)
        exclude = self.config.get('exclude_keywords_ko', self.config.get('exclude_keywords', []))
        if exclude:
            if isinstance(exclude, list):
                base_prompt += f"\n\n피해야 할 표현: {', '.join(exclude[:5])}"
            else:
                base_prompt += f"\n\n피해야 할 표현: {exclude}"

        # 톤 가이드 적용
        tone_desc = self._get_tone_description(tone)
        base_prompt += f"\n\n톤 앤 매너:\n{tone_desc}"

        return base_prompt

    def _get_tone_description(self, tone: str) -> str:
        """톤 설명 반환"""
        # YAML에서 로드된 톤 가이드 사용
        if tone in self.tone_guide:
            tone_data = self.tone_guide[tone]
            desc = tone_data.get('description_ko', tone_data.get('description', ''))
            keywords = tone_data.get('keywords', [])
            if desc and keywords:
                return f"{desc}\n핵심 키워드: {', '.join(keywords[:3])}"
            return desc

        # 기본 톤 설명
        tone_styles = {
            "warm": "따뜻하고 감성적인 톤으로 작성하세요. 편안하고 아늑한 느낌을 주세요.",
            "professional": "전문적이고 신뢰감 있는 톤으로 작성하세요. 격식 있고 세련된 느낌을 주세요.",
            "friendly": "친근하고 편안한 톤으로 작성하세요. 대화하듯 자연스러운 느낌을 주세요.",
            "energetic": "활기차고 역동적인 톤으로 작성하세요. 열정적이고 긍정적인 느낌을 주세요.",
            "practical": "실용적이고 명확한 톤으로 작성하세요. 구체적인 혜택을 강조하세요.",
            "respectful": "정중하고 신뢰감 있는 톤으로 작성하세요. 전문성을 강조하세요."
        }

        return tone_styles.get(tone, tone_styles["warm"])

    def build_user_prompt(self, user_input: str, max_length: int = 20) -> str:
        """
        사용자 프롬프트 생성

        Args:
            user_input: 사용자 입력
            max_length: 최대 글자 수

        Returns:
            str: 사용자 프롬프트
        """
        # 업종별 프롬프트 템플릿 사용
        template = self.config.get('prompt_template', '')

        prompt = f"""다음 내용으로 광고 문구를 만들어주세요:

{user_input}

요구사항:
- {max_length}자 이내
- 광고 문구만 작성 (설명, 번호 등 제외)
- 감성적이면서도 명확한 메시지 전달"""

        if template:
            prompt += f"\n\n참고 프롬프트 구조:\n{template}"

        prompt += "\n\n광고 문구:"

        return prompt

    def get_example_copies(self) -> List[str]:
        """업종별 예시 광고 문구 반환"""
        return self.config.get('example_copies', [])


class PromptTemplateManager:
    """
    통합 프롬프트 관리자

    기능:
    1. 업종 자동 감지
    2. 광고 문구 생성 프롬프트 제공
    3. 이미지 생성 프롬프트 제공 (image_generation 모듈 연동)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: industries.yaml 경로 (None이면 기본 경로)
        """
        self.loader = IndustryConfigLoader(config_path)
        self.tone_guide = self.loader.get_tone_guide()

    def detect_industry(self, user_input: str) -> str:
        """
        업종 자동 감지

        Args:
            user_input: 사용자 입력

        Returns:
            str: 감지된 업종 코드
        """
        return self.loader.detect_industry(user_input)

    def get_ad_copy_prompt(
        self,
        user_input: str,
        tone: str = "warm",
        max_length: int = 20,
        industry: Optional[str] = None
    ) -> Dict[str, str]:
        """
        광고 문구 생성용 프롬프트 반환

        Args:
            user_input: 사용자 입력
            tone: 톤 (warm, professional, friendly, energetic, practical, respectful)
            max_length: 최대 글자 수
            industry: 업종 코드 (None이면 자동 감지)

        Returns:
            Dict: {
                "system_prompt": "...",
                "user_prompt": "...",
                "industry": "s1_hot_cooking",
                "examples": ["예시1", "예시2"]
            }
        """
        # 업종 감지
        if industry is None:
            industry = self.detect_industry(user_input)

        # 업종 설정 로드
        industry_config = self.loader.get_industry(industry) or {}

        # 프롬프트 빌더 생성
        builder = AdCopyPromptBuilder(industry_config, self.tone_guide)

        return {
            "system_prompt": builder.build_system_prompt(tone, max_length),
            "user_prompt": builder.build_user_prompt(user_input, max_length),
            "industry": industry,
            "examples": builder.get_example_copies()
        }

    def generate_image_prompt(
        self,
        user_input: str,
        style: str = "realistic",
        industry: Optional[str] = None
    ) -> Dict[str, str]:
        """
        이미지 생성 프롬프트 반환

        Note: 이 메서드는 image_generation 모듈과의 호환성을 위해 제공됩니다.
              실제 이미지 프롬프트 생성은 image_generation.prompt 모듈을 사용하세요.

        Args:
            user_input: 사용자 입력
            style: 이미지 스타일 (realistic, semi_realistic, anime)
            industry: 업종 코드 (None이면 자동 감지)

        Returns:
            Dict: {
                "positive": "...",
                "negative": "",  # Z-Image Turbo는 negative 미지원
                "industry": "s1_hot_cooking"
            }
        """
        # 업종 감지
        if industry is None:
            industry = self.detect_industry(user_input)

        # image_generation 모듈 연동 시도
        try:
            from src.generation.image_generation.prompt.config_loader import PromptGenerator

            generator = PromptGenerator()
            result = generator.generate(
                industry=industry,
                user_input={"product": user_input, "style": style}
            )

            return {
                "positive": result.get("positive", ""),
                "negative": result.get("negative", ""),  # Z-Image Turbo는 빈 문자열
                "industry": industry
            }

        except ImportError:
            # image_generation 모듈 없이 기본 프롬프트 생성
            return self._generate_basic_image_prompt(user_input, style, industry)

    def _generate_basic_image_prompt(
        self,
        user_input: str,
        style: str,
        industry: str
    ) -> Dict[str, str]:
        """기본 이미지 프롬프트 생성 (image_generation 모듈 없이)"""

        style_openers = {
            "realistic": "Professional commercial photography of",
            "semi_realistic": "Highly detailed digital artwork of",
            "anime": "Vibrant anime style illustration of"
        }

        opener = style_openers.get(style, style_openers["realistic"])

        # 간단한 프롬프트 생성
        positive = f"{opener} {user_input}. High quality, detailed, professional lighting."

        return {
            "positive": positive,
            "negative": "",  # Z-Image Turbo는 negative 미지원
            "industry": industry
        }

    def get_industry_info(self, industry: str) -> Dict:
        """
        업종 정보 조회

        Args:
            industry: 업종 코드

        Returns:
            Dict: 업종 정보
        """
        config = self.loader.get_industry(industry)
        if not config:
            return {}

        return {
            "name": config.get("name", ""),
            "name_ko": config.get("name_ko", ""),
            "grade": self.loader.get_grade_for_industry(industry),
            "keyword_priorities": config.get("keyword_priorities", []),
            "brand_voice": config.get("brand_voice", ""),
            "example_copies": config.get("example_copies", [])
        }

    def get_all_industries(self) -> List[str]:
        """모든 업종 코드 리스트"""
        return self.loader.get_all_subgroups()

    def get_recommended_tone(self, industry: str) -> str:
        """
        업종에 맞는 추천 톤 반환

        Args:
            industry: 업종 코드

        Returns:
            str: 추천 톤
        """
        grade = self.loader.get_grade_for_industry(industry)

        # 등급별 기본 추천 톤
        grade_tones = {
            "s_grade": "warm",      # 감각/맛 자극 → 따뜻하고 감성적
            "a_grade": "friendly",  # 시각적 변화 → 친근하고 편안
            "b_grade": "energetic", # 공간 경험 → 활기차고 역동적
            "c_grade": "professional",  # 신뢰 기반 → 전문적
            "d_grade": "practical",     # 목적 중심 → 실용적
            "e_grade": "respectful"     # 인프라 → 정중함
        }

        return grade_tones.get(grade, "warm")


# ============================================
# 편의 함수
# ============================================

def create_manager(config_path: Optional[str] = None) -> PromptTemplateManager:
    """
    PromptTemplateManager 인스턴스 생성 헬퍼

    Args:
        config_path: industries.yaml 경로

    Returns:
        PromptTemplateManager 인스턴스
    """
    return PromptTemplateManager(config_path)


def detect_industry(user_input: str) -> str:
    """
    업종 자동 감지 편의 함수

    Args:
        user_input: 사용자 입력

    Returns:
        str: 감지된 업종 코드
    """
    manager = PromptTemplateManager()
    return manager.detect_industry(user_input)


# ============================================
# 테스트 코드
# ============================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("📝 PromptTemplateManager 테스트")
#     print("=" * 70)

#     # 매니저 생성
#     manager = PromptTemplateManager()

#     # 테스트 케이스
#     test_cases = [
#         "삼겹살 맛집 홍보, 불판에서 지글지글 굽는 모습",
#         "카페 봄 시즌 딸기 음료 출시 홍보",
#         "미용실 봄 시즌 헤어컬러 이벤트",
#         "헬스장 PT 프로그램 홍보",
#         "법률사무소 무료 상담 이벤트",
#         "꽃집 봄 시즌 꽃다발 할인"
#     ]

#     for test_input in test_cases:
#         print(f"\n{'='*70}")
#         print(f"입력: {test_input}")
#         print("-" * 70)

#         # 업종 감지
#         industry = manager.detect_industry(test_input)
#         print(f"감지된 업종: {industry}")

#         # 업종 정보
#         info = manager.get_industry_info(industry)
#         print(f"업종명: {info.get('name_ko', 'N/A')}")
#         print(f"등급: {info.get('grade', 'N/A')}")
#         print(f"추천 톤: {manager.get_recommended_tone(industry)}")

#         # 광고 문구 프롬프트
#         prompts = manager.get_ad_copy_prompt(test_input, max_length=20)
#         print(f"\n[시스템 프롬프트 미리보기]")
#         print(prompts["system_prompt"][:200] + "...")

#         if info.get('example_copies'):
#             print(f"\n[예시 광고 문구]")
#             for ex in info['example_copies'][:2]:
#                 print(f"  - {ex}")

#     print(f"\n{'='*70}")
#     print("✅ 테스트 완료!")
#     print(f"{'='*70}")
