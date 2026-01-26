"""
설정 관리 모듈

환경변수 및 설정값 관리
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class Settings:
    """애플리케이션 설정"""

    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    serpapi_api_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    naver_client_id: str = field(default_factory=lambda: os.getenv("NAVER_CLIENT_ID", ""))
    naver_client_secret: str = field(default_factory=lambda: os.getenv("NAVER_CLIENT_SECRET", ""))

    # Model Settings
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "gpt-4o-mini"
    rerank_model: str = "rerank-multilingual-v2.0"

    # RAG Settings
    vector_store_path: str = "vector_store"
    top_k_retrieval: int = 10  # FAISS에서 가져올 후보 수
    top_k_rerank: int = 3  # Rerank 후 최종 수
    similarity_threshold: float = 0.7

    # Agent Settings
    agent_max_iterations: int = 3
    agent_timeout: int = 30  # seconds

    # Self-Refine Settings
    refine_threshold: float = 7.0  # 10점 만점 기준
    max_refine_iterations: int = 2

    # Memory Settings
    max_conversation_history: int = 10  # 최대 대화 턴 수

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths
    data_path: str = "data"
    cases_file: str = "cases.json"

    def validate(self) -> list[str]:
        """설정 유효성 검사"""
        errors = []

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is not set")

        return errors

    @property
    def is_valid(self) -> bool:
        """설정이 유효한지 확인"""
        return len(self.validate()) == 0


@lru_cache()
def get_settings() -> Settings:
    """싱글톤 Settings 인스턴스 반환"""
    return Settings()


# 기본 프롬프트 템플릿
SYSTEM_PROMPT = """당신은 소상공인 온라인 마케팅 전문 컨설턴트입니다.

역할:
- 상황을 파악하고 명확한 전략을 수립합니다
- 성공 사례를 근거로 제시합니다
- 실행 가능한 요구사항을 정리합니다

규칙:
1. 결과를 보장하지 않습니다 (법적 리스크)
2. 과장 표현 금지 ("반드시", "100%", "절대" 등)
3. 항상 근거를 명시합니다 (출처, 날짜, 성과 지표)
4. 구체적 숫자/예시를 제공합니다
5. 다음 단계를 명확히 안내합니다
"""

CRITIQUE_PROMPT = """다음 상담 답변을 평가하세요.

답변:
{draft}

체크리스트 (각 항목 0-10점):
1. 구체성: 추상적 조언이 아닌 실행 가능한 내용
2. 근거: 출처, 사례, 데이터 명시 여부
3. 정확성: 과장 표현, 보장성 발언 없음
4. 완성도: 다음 단계가 명확함

JSON 형식으로 답하세요:
{{
    "scores": {{
        "specificity": 점수,
        "evidence": 점수,
        "accuracy": 점수,
        "completeness": 점수
    }},
    "total_score": 평균점수,
    "issues": ["문제점1", "문제점2"],
    "suggestions": ["개선안1", "개선안2"]
}}
"""

REFINE_PROMPT = """초안:
{draft}

문제점:
{issues}

개선 방향:
{suggestions}

위 피드백을 반영하여 답변을 개선하세요.

개선 규칙:
- 구체적인 숫자/예시 추가
- 모든 주장에 근거 명시
- "반드시", "100%", "절대" 같은 단어 제거
- 다음 단계 명확히
"""
