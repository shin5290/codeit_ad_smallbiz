"""
Configuration Loader
industries.yaml을 로드하고 Hybrid Prompting 시스템과 통합
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from src.generation.image_generation.prompt.prompt_templates import (
    HybridPromptBuilder, NegativePromptBuilder)
from src.generation.image_generation.prompt.style_router import StyleRouter


class IndustryConfigLoader:
    """
    industries.yaml 로더 (v3.0.0 계층 구조 지원)

    한글 키워드는 YAML의 korean_keywords 필드에서 직접 로드
    """

    # 등급 목록 상수
    GRADES = ['s_grade', 'a_grade', 'b_grade', 'c_grade', 'd_grade', 'e_grade']

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            self.config_path = Path(__file__).parent / "config" / "industries.yaml"
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()

        # 캐시: 하위 그룹 코드 → (등급, 하위그룹 데이터) 매핑
        self._subgroup_cache = self._build_subgroup_cache()

        # 캐시: 업종명 → 하위 그룹 코드 매핑
        self._business_to_subgroup = self._build_business_mapping()

        # 캐시: 하위 그룹별 키워드 맵 (영어 + 한글)
        self._keyword_map = self._build_detection_keywords()

    def _load_config(self) -> Dict:
        """YAML 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_subgroup_cache(self) -> Dict[str, tuple]:
        """하위 그룹 코드로 빠르게 접근할 수 있는 캐시 생성"""
        cache = {}
        for grade_key in self.GRADES:
            grade_data = self.config.get(grade_key, {})
            for key, value in grade_data.items():
                # 하위 그룹은 dict이고 prompt_template을 가짐
                if isinstance(value, dict) and 'prompt_template' in value:
                    cache[key] = (grade_key, value)
        return cache

    def _build_business_mapping(self) -> Dict[str, str]:
        """개별 업종명 → 하위 그룹 코드 매핑"""
        mapping = {}
        for subgroup_key, (grade_key, subgroup_data) in self._subgroup_cache.items():
            businesses = subgroup_data.get('businesses', [])
            for business in businesses:
                # 업종명 정규화 (소문자, 공백 제거)
                normalized = business.lower().replace(' ', '_').replace('/', '_')
                mapping[normalized] = subgroup_key
                # 원본도 추가
                mapping[business.lower()] = subgroup_key
        return mapping

    def get_industry(self, industry_code: str) -> Optional[Dict]:
        """
        업종 설정 가져오기 (v3.0.0 호환)

        Args:
            industry_code: 하위 그룹 코드 (s1_hot_cooking, a1_beauty 등)
                          또는 레거시 코드 (cafe, gym 등)

        Returns:
            Dict: 업종 설정 또는 None
        """
        # 1. 새 구조에서 하위 그룹 코드로 직접 찾기
        if industry_code in self._subgroup_cache:
            return self._subgroup_cache[industry_code][1]

        # 2. 레거시 코드 호환 (cafe → s3_emotional 등)
        legacy_mapping = self._get_legacy_mapping()
        if industry_code in legacy_mapping:
            mapped_code = legacy_mapping[industry_code]
            if mapped_code in self._subgroup_cache:
                return self._subgroup_cache[mapped_code][1]

        # 3. 개별 업종명으로 찾기
        normalized_code = industry_code.lower().replace(' ', '_')
        if normalized_code in self._business_to_subgroup:
            subgroup_code = self._business_to_subgroup[normalized_code]
            return self._subgroup_cache[subgroup_code][1]

        # 4. 없으면 general (s4_neat_variety 또는 기본값)
        return self._subgroup_cache.get('s4_neat_variety')

    def _get_legacy_mapping(self) -> Dict[str, str]:
        """레거시 업종 코드 → 신규 하위 그룹 코드 매핑"""
        return {
            # S등급
            "cafe": "s3_emotional",
            "bakery": "s3_emotional",
            "restaurant": "s1_hot_cooking",

            # A등급
            "gym": "a2_wellness",
            "hair_salon": "a1_beauty",
            "nail_salon": "a1_beauty",
            "flower_shop": "a4_delicate_care",
            "clothing_store": "a3_fashion",

            # C등급
            "laundry": "a4_delicate_care",  # 세탁소는 A4로 이동

            # 기본값
            "general": "s4_neat_variety"
        }

    def get_all_industries(self) -> Dict[str, Dict]:
        """모든 하위 그룹 설정 반환"""
        return {k: v for k, (_, v) in self._subgroup_cache.items()}

    def get_all_subgroups(self) -> List[str]:
        """사용 가능한 하위 그룹 코드 리스트"""
        return list(self._subgroup_cache.keys())

    def get_industry_names(self) -> List[str]:
        """사용 가능한 하위 그룹 코드 리스트 (레거시 호환)"""
        return self.get_all_subgroups()

    def get_grade_info(self, grade_key: str) -> Optional[Dict]:
        """등급 정보 가져오기 (core_strategy, characteristics 등)"""
        grade_data = self.config.get(grade_key, {})
        if not grade_data:
            return None
        return {
            'name': grade_data.get('name', ''),
            'name_ko': grade_data.get('name_ko', ''),
            'characteristics': grade_data.get('characteristics', ''),
            'core_strategy': grade_data.get('core_strategy', ''),
            'total_businesses': grade_data.get('total_businesses', 0)
        }

    def detect_industry(self, user_input: str) -> str:
        """
        사용자 입력에서 업종 자동 감지 (v3.0.0)

        YAML의 keywords + korean_keywords 필드 활용

        Args:
            user_input: 사용자 입력 텍스트

        Returns:
            str: 감지된 하위 그룹 코드 또는 "s4_neat_variety" (기본값)
        """
        user_input_lower = user_input.lower()

        # 초기화 시 생성된 키워드 캐시 활용
        scores = {}
        for subgroup_code, keywords in self._keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                scores[subgroup_code] = score

        # 가장 높은 점수의 하위 그룹 반환
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return "s4_neat_variety"  # 기본값

    def _build_detection_keywords(self) -> Dict[str, List[str]]:
        """
        감지용 키워드 맵 구축

        YAML의 korean_keywords 필드를 직접 활용 (하드코딩 제거)
        """
        keyword_map = {}

        for subgroup_code, (grade_key, subgroup_data) in self._subgroup_cache.items():
            keywords = []

            # 1. YAML의 keywords 필드 (영어)
            yaml_keywords = subgroup_data.get('keywords', [])
            keywords.extend([kw.lower() for kw in yaml_keywords])

            # 2. YAML의 korean_keywords 필드 (한글) - 직접 로드
            korean_keywords = subgroup_data.get('korean_keywords', [])
            keywords.extend(korean_keywords)

            # 3. businesses 필드에서 추출
            businesses = subgroup_data.get('businesses', [])
            for business in businesses:
                words = business.lower().replace('/', ' ').split()
                keywords.extend(words)

            keyword_map[subgroup_code] = list(set(keywords))

        return keyword_map


class PromptGenerator:
    """
    통합 프롬프트 생성기 (v3.0.0 호환)

    IndustryConfigLoader + HybridPromptBuilder 통합
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 설정 파일 경로 (None이면 자동으로 config/industries.yaml 사용)
        """
        self.loader = IndustryConfigLoader(config_path)

    def generate(
        self,
        industry: str,
        user_input: Dict,
        composition: Optional[str] = None,
        apply_weights: bool = False,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        완전한 프롬프트 생성 (Positive + Negative)

        Args:
            industry: 업종 코드 (cafe, gym 등)
            user_input: 사용자 입력
                {
                    "style": "realistic",  # realistic, semi_realistic, anime
                    "product": "strawberry latte",
                    "theme": "warm",
                    "mood": "cozy",
                    "time": "morning"
                }
            composition: 구도 타입 (overhead, 45_degree 등)
            apply_weights: 가중치 적용 여부
            weights: 가중치 딕셔너리 {"keyword": 1.3}

        Returns:
            Dict: {
                "positive": "...",
                "negative": "...",
                "style": "...",
                "structure": {...}  # 디버깅용
            }
        """
        # 0. 스타일 추출 (기본값: realistic)
        style = user_input.get("style", "realistic")
        if not StyleRouter.is_valid_style(style):
            style = "realistic"

        # 1. 업종 설정 로드 (v3.0.0 호환)
        industry_config = self.loader.get_industry(industry)
        if not industry_config:
            # fallback: 기본값 사용
            industry_config = self.loader.get_industry("s4_neat_variety")

        # 2. Hybrid Prompt 빌드 (업종 템플릿 기반)
        builder = HybridPromptBuilder(industry_config)
        structure = builder.build_from_user_input(user_input, composition)

        # 3. 스타일별 Subject Phrase 교체
        main_subject = (
            user_input.get('product') or
            user_input.get('dish') or
            user_input.get('item') or
            user_input.get('subject') or
            user_input.get('person_type') or  # 캐릭터, 사람 등
            user_input.get('character') or    # 캐릭터 명시적 지정
            'subject'
        )
        structure.subject_phrase = StyleRouter.build_subject_phrase(
            style=style,
            subject=main_subject,
            additional_context=structure.setting
        )

        # 스타일별 Technical 키워드 적용(현재 get_style_technical 제거)
        # structure.technical = StyleRouter.get_style_technical(style)[:2]

        # 4. 프롬프트 조립
        positive_prompt = structure.build()

        # 5. 가중치 적용 (옵션)
        if apply_weights and weights:
            from .prompt_templates import apply_prompt_weights
            positive_prompt = apply_prompt_weights(positive_prompt, weights)

        # 6. Negative Prompt 생성 (스타일 반영)
        negative_prompt = NegativePromptBuilder.build(
            industry=industry,
            style=style
        )

        return {
            "positive": positive_prompt,
            "negative": negative_prompt,
            "style": style,
            "structure": structure.to_dict()
        }

    def generate_simple(
        self,
        industry: str,
        product: str,
        **kwargs
    ) -> str:
        """
        간단한 프롬프트 생성 (Positive만)

        Args:
            industry: 업종
            product: 주요 상품/서비스
            **kwargs: 추가 파라미터 (theme, mood, time 등)

        Returns:
            str: Positive prompt
        """
        user_input = {"product": product, **kwargs}
        result = self.generate(industry, user_input)
        return result["positive"]

    def get_industry_info(self, industry: str) -> Dict:
        """
        업종 정보 조회

        Returns:
            Dict: {
                "name": "...",
                "name_ko": "...",
                "description": "...",
                "keywords": [...],
                "required_layers": [...]
            }
        """
        config = self.loader.get_industry(industry)
        if not config:
            return {}

        return {
            "name": config.get("name", ""),
            "name_ko": config.get("name_ko", ""),
            "description": config.get("description", ""),
            "keywords": config.get("keywords", []),
            "required_layers": config.get("required_layers", [])
        }


# ============================================
# 전역 인스턴스 (다른 모듈에서 import 가능)
# ============================================

if __name__ == "__main__":
    try:
        industry_config = IndustryConfigLoader()
    except Exception as e:
        print(f"⚠️  industries.yaml 로드 실패: {e}")
        industry_config = None
