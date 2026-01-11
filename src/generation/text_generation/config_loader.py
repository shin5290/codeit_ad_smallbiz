"""
Configuration Loader
industries.yaml을 로드하고 Hybrid Prompting 시스템과 통합
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from prompt_templates import HybridPromptBuilder, NegativePromptBuilder, PromptStructure


class IndustryConfigLoader:
    """
    industries.yaml 로더
    """
    
    def __init__(self, config_path: str = "src/generation/text_generation/config/industries.yaml"):
        """
        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """YAML 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_industry(self, industry_code: str) -> Optional[Dict]:
        """
        업종 설정 가져오기
        
        Args:
            industry_code: 업종 코드 (cafe, gym 등)
        
        Returns:
            Dict: 업종 설정 또는 None
        """
        return self.config.get(industry_code)
    
    def get_all_industries(self) -> Dict[str, Dict]:
        """모든 업종 설정 반환"""
        # metadata, common_negative 제외
        exclude = ['metadata', 'common_negative']
        return {k: v for k, v in self.config.items() if k not in exclude}
    
    def get_industry_names(self) -> List[str]:
        """사용 가능한 업종 코드 리스트"""
        return list(self.get_all_industries().keys())
    
    def detect_industry(self, user_input: str) -> str:
        """
        사용자 입력에서 업종 자동 감지
        
        Args:
            user_input: 사용자 입력 텍스트
        
        Returns:
            str: 감지된 업종 코드 또는 "general"
        """
        user_input_lower = user_input.lower()
        
        # 업종별 키워드 (우선순위 순)
        industry_keywords = {
            "cafe": ["카페", "커피", "라떼", "아메리카노", "에스프레소", "카푸치노", "음료"],
            "bakery": ["빵", "베이커리", "크루아상", "바게트", "제과", "제빵", "케이크"],
            "gym": ["헬스", "헬스장", "운동", "근육", "스쿼트", "웨이트", "피트니스", "짐", "gym"],
            "restaurant": ["레스토랑", "식당", "음식", "요리", "파스타", "스테이크", "메뉴"],
            "hair_salon": ["미용실", "헤어", "염색", "커트", "미용"],
            "nail_salon": ["네일", "매니큐어", "손톱"],
            "flower_shop": ["꽃", "플라워", "장미", "꽃집", "부케"],
            "clothing_store": ["옷", "의류", "패션", "셔츠"],
            "laundry": ["세탁", "빨래", "린넨", "다림질"]
        }
        
        # 점수 계산
        scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input)
            if score > 0:
                scores[industry] = score
        
        # 가장 높은 점수의 업종 반환
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return "general"


class PromptGenerator:
    """
    통합 프롬프트 생성기
    
    IndustryConfigLoader + HybridPromptBuilder 통합
    """
    
    def __init__(self, config_path: str = "src/generation/text_generation/config/industries.yaml"):
        """
        Args:
            config_path: YAML 설정 파일 경로
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
                "structure": {...}  # 디버깅용
            }
        """
        # 1. 업종 설정 로드
        industry_config = self.loader.get_industry(industry)
        if not industry_config:
            raise ValueError(f"Unknown industry: {industry}")
        
        # 2. Hybrid Prompt 빌드
        builder = HybridPromptBuilder(industry_config)
        structure = builder.build_from_user_input(user_input, composition)
        
        # 3. 프롬프트 조립
        positive_prompt = structure.build()
        
        # 4. 가중치 적용 (옵션)
        if apply_weights and weights:
            from prompt_templates import apply_prompt_weights
            positive_prompt = apply_prompt_weights(positive_prompt, weights)
        
        # 5. Negative Prompt 생성
        negative_prompt = NegativePromptBuilder.build(
            industry=industry,
            style="realistic"
        )
        
        # 주의: 업종별 Negative는 이미 NegativePromptBuilder.build()에 포함됨!
        # 추가하지 말 것!
        
        return {
            "positive": positive_prompt,
            "negative": negative_prompt,
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
# 예시 사용법
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("SDXL Prompt Generator - Test Suite")
    print("=" * 70)
    
    # 1. Config Loader 테스트
    print("\n[Test 1] Industry Config Loader")
    print("-" * 70)
    
    loader = IndustryConfigLoader()
    industries = loader.get_industry_names()
    print(f"Available industries: {', '.join(industries)}")
    
    cafe_config = loader.get_industry("cafe")
    print(f"\nCafe templates available:")
    print(f"  - Subject patterns: {len(cafe_config['prompt_template']['subject_patterns'])}")
    print(f"  - Setting patterns: {len(cafe_config['prompt_template']['setting_patterns'])}")
    print(f"  - Lighting phrases: {len(cafe_config['prompt_template']['lighting_phrases'])}")
    
    # 2. Prompt Generator 테스트
    print("\n\n[Test 2] Prompt Generation - Cafe")
    print("-" * 70)
    
    generator = PromptGenerator()
    
    # Cafe 예시
    cafe_result = generator.generate(
        industry="cafe",
        user_input={
            "product": "strawberry latte",
            "surface": "marble table",
            "theme": "warm",
            "mood": "cozy",
            "time": "morning"
        },
        composition="overhead"
    )
    
    print("\n✅ Positive Prompt:")
    print(cafe_result["positive"])
    print("\n❌ Negative Prompt:")
    print(cafe_result["negative"])
    
    # 3. Gym 예시
    print("\n\n[Test 3] Prompt Generation - Gym")
    print("-" * 70)
    
    gym_result = generator.generate(
        industry="gym",
        user_input={
            "person_type": "athletic man",
            "activity": "barbell squat",
            "focus": "muscle definition"
        }
    )
    
    print("\n✅ Positive Prompt:")
    print(gym_result["positive"])
    print("\n❌ Negative Prompt:")
    print(gym_result["negative"])
    
    # 4. 가중치 적용 테스트
    print("\n\n[Test 4] Weighted Prompt - Laundry")
    print("-" * 70)
    
    laundry_result = generator.generate(
        industry="laundry",
        user_input={
            "item": "white linens",
            "state": "freshly laundered",
            "quality": "crisp and clean"
        },
        apply_weights=True,
        weights={
            "freshly laundered": 1.3,
            "crisp": 1.2
        }
    )
    
    print("\n✅ Weighted Positive Prompt:")
    print(laundry_result["positive"])
    
    # 5. Simple Generation
    print("\n\n[Test 5] Simple Generation")
    print("-" * 70)
    
    simple_prompt = generator.generate_simple(
        industry="bakery",
        product="croissant",
        theme="rustic",
        time="morning"
    )
    
    print("\n✅ Simple Prompt:")
    print(simple_prompt)
    
    # 6. Industry Info
    print("\n\n[Test 6] Industry Information")
    print("-" * 70)
    
    cafe_info = generator.get_industry_info("cafe")
    print(f"\nIndustry: {cafe_info['name']} ({cafe_info['name_ko']})")
    print(f"Description: {cafe_info['description']}")
    print(f"Keywords: {', '.join(cafe_info['keywords'][:5])}...")
    print(f"Required layers: {', '.join(cafe_info['required_layers'])}")
    
    # 7. All Industries
    print("\n\n[Test 7] All Industries Quick Test")
    print("-" * 70)
    
    test_cases = {
        "cafe": {"product": "latte"},
        "gym": {"activity": "workout", "person_type": "athlete"},
        "laundry": {"item": "shirt", "state": "clean"},
        "bakery": {"product": "bread"},
        "restaurant": {"dish": "pasta"}
    }
    
    for industry, user_input in test_cases.items():
        result = generator.generate(industry=industry, user_input=user_input)
        prompt = result["positive"]
        print(f"\n{industry.upper()}: {prompt[:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


# ============================================
# 전역 인스턴스 (다른 모듈에서 import 가능)
# ============================================

try:
    industry_config = IndustryConfigLoader()
except Exception as e:
    print(f"⚠️  industries.yaml 로드 실패: {e}")
    industry_config = None