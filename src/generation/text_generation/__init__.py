"""
텍스트 생성 모듈 (광고 문구 생성)

이 모듈은 소상공인 광고 생성 서비스의 광고 문구 생성 기능을 제공합니다.

의존성 로드 순서:
1. prompt_manager (IndustryConfigLoader, AdCopyPromptBuilder, PromptTemplateManager)
2. text_generator (TextGenerator)
3. ad_generator (generate_advertisement)
4. evaluate_prompt (CivitaiEnhancedEvaluator)

사용 예시:
    from src.generation.text_generation import generate_advertisement

    result = generate_advertisement(
        user_input="카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
        tone="warm",
        max_length=20
    )

작성자: 배현석 -> 신승목
버전: 3.0.0
"""

# 1단계: 프롬프트 관리자 (industries.yaml 의존)
from src.generation.text_generation.prompt_manager import (
    # 설정 로더
    IndustryConfigLoader,
    # 프롬프트 빌더
    AdCopyPromptBuilder,
    # 통합 관리자
    PromptTemplateManager,
    # 편의 함수
    create_manager,
    detect_industry,
)

# 2단계: 텍스트 생성기 (PromptTemplateManager 의존)
from src.generation.text_generation.text_generator import TextGenerator

# 3단계: 통합 API (TextGenerator, PromptTemplateManager 의존)
from src.generation.text_generation.ad_generator import (
    generate_advertisement,
    test_without_api,
)

# 4단계: 프롬프트 평가 시스템 (독립적, 선택적 import)
from src.generation.text_generation.evaluate_prompt import (
    CivitaiEnhancedEvaluator,
    EvaluationResult,
    save_md_report,
)

# 모듈 메타데이터
__version__ = "3.0.0"
__author__ = "배현석, 신승목"
__all__ = [
    # 1단계: 프롬프트 관리자
    "IndustryConfigLoader",
    "AdCopyPromptBuilder",
    "PromptTemplateManager",
    "create_manager",
    "detect_industry",
    # 2단계: 텍스트 생성기
    "TextGenerator",
    # 3단계: 통합 API
    "generate_advertisement",
    "test_without_api",
    # 4단계: 평가 시스템
    "CivitaiEnhancedEvaluator",
    "EvaluationResult",
    "save_md_report",
]
