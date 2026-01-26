"""
텍스트 생성 모듈

광고 문구 생성 및 프롬프트 관리 기능을 제공합니다.

작성자: 배현석
버전: 1.0
"""

# text_generator.py (먼저 로드)
from src.generation.text_generation.text_generator import TextGenerator

# prompt_manager.py (ad_generator보다 먼저 로드)
from src.generation.text_generation.prompt_manager import (
    PromptTemplateManager,
    IndustryConfigLoader,
    AdCopyPromptBuilder,
    create_manager,
    detect_industry,
)

# evaluate_prompt.py
from src.generation.text_generation.evaluate_prompt import (
    CivitaiEnhancedEvaluator,
    EvaluationResult,
    save_md_report,
)

# ad_generator.py (TextGenerator, PromptTemplateManager에 의존하므로 마지막에 로드)
from src.generation.text_generation.ad_generator import (
    generate_advertisement,
    test_without_api
)

__all__ = [
    # 텍스트 생성
    "TextGenerator",
    # 광고 생성 통합 함수
    "generate_advertisement",
    "test_without_api",
    # 프롬프트 관리
    "PromptTemplateManager",
    "IndustryConfigLoader",
    "AdCopyPromptBuilder",
    "create_manager",
    "detect_industry",
    # 프롬프트 평가
    "CivitaiEnhancedEvaluator",
    "EvaluationResult",
    "save_md_report",
]
