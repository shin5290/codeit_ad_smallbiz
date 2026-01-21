# 백엔드 연동 가이드

## 개요

챗봇의 RAG 기반 Knowledge Base를 백엔드의 `ConsultingService`와 연동하는 방법입니다.

## 연동 방법

### 1. consulting_knowledge_base.py 수정

백엔드의 `consulting_knowledge_base.py`에서 `MockKnowledgeBase` 대신 `SmallBizKnowledgeBase`를 사용합니다.

```python
# consulting_knowledge_base.py

# 기존 import 유지
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ConsultingKnowledgeBase 추상 클래스는 그대로 유지...

# =====================================================
# SmallBizKnowledgeBase 연동
# =====================================================

def get_knowledge_base() -> ConsultingKnowledgeBase:
    """지식베이스 인스턴스 반환 (싱글톤)"""
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        try:
            # SmallBizKnowledgeBase 사용 (RAG 기반)
            from src.generation.chat_bot.rag import SmallBizKnowledgeBase
            _knowledge_base_instance = SmallBizKnowledgeBase(use_reranker=False)
            logger.info("SmallBizKnowledgeBase initialized")
        except ImportError as e:
            # 폴백: MockKnowledgeBase
            logger.warning(f"SmallBizKnowledgeBase import failed: {e}")
            _knowledge_base_instance = MockKnowledgeBase()
            logger.info("Fallback to MockKnowledgeBase")
    return _knowledge_base_instance
```

### 2. Import 경로

```python
# 방법 1: rag 모듈에서 직접 import
from src.generation.chat_bot.rag import SmallBizKnowledgeBase, get_knowledge_base

# 방법 2: knowledge_base 모듈에서 import
from src.generation.chat_bot.rag.knowledge_base import SmallBizKnowledgeBase
```

### 3. 사용 예시

```python
# 싱글톤 인스턴스 사용
from src.generation.chat_bot.rag import get_knowledge_base

kb = get_knowledge_base()

# 검색
results = kb.search(
    query="카페 인스타그램 마케팅 전략",
    category="faq",  # 선택사항
    limit=3
)

# 결과 형식
# [
#     {
#         "content": "검색된 문서 내용...",
#         "source": "매장명 또는 문서 제목",
#         "score": 0.85,
#         "category": "faq",
#         "metadata": {
#             "industry": "cafe",
#             "location": "강남",
#             "title": "OO카페",
#             "rating": 4.5,
#             "keywords": ["카페", "인스타그램", "마케팅"]
#         }
#     },
#     ...
# ]
```

## API 인터페이스

### ConsultingKnowledgeBase.search()

```python
def search(
    self,
    query: str,                    # 검색 쿼리
    category: Optional[str] = None, # 카테고리 필터
    limit: int = 3                 # 반환할 문서 수
) -> List[Dict]:
    """
    Returns:
        [
            {
                "content": str,      # 문서 내용
                "source": str,       # 출처 (매장명/문서 제목)
                "score": float,      # 유사도 점수 (0~1)
                "category": str,     # 카테고리
                "metadata": dict     # 추가 메타데이터
            },
            ...
        ]
    """
```

### 카테고리 종류

- `faq`: 일반 FAQ
- `generation_guide`: 광고 생성 가이드
- `modification_guide`: 수정 가이드
- `industry_tips`: 업종별 팁

## Agent 기능 (선택)

트렌드/실시간 정보 검색이 필요한 경우:

```python
from src.generation.chat_bot.rag import get_knowledge_base

# Agent 통합 버전
kb = get_knowledge_base(use_agent=True)

# 확장 검색 (웹 검색 포함)
result = kb.search_with_agent(
    query="요즘 유행하는 카페 마케팅 트렌드",
    limit=3
)

# 결과 형식
# {
#     "rag_results": [...],      # 벡터DB 검색 결과
#     "web_results": [...],      # 웹 검색 결과
#     "combined_answer": "..."   # 통합 답변
# }
```

## 환경 설정

### 필수 환경 변수

```bash
export OPENAI_API_KEY="your-api-key"
export LLM_MODEL="gpt-4o"  # 선택, 기본값: gpt-4o
export TAVILY_API_KEY="your-tavily-key"  # 웹 검색용, 선택
```

### 필수 패키지

```bash
pip install langchain langchain-openai langchain-community chromadb sentence-transformers
```

## 벡터스토어 위치

```
src/generation/chat_bot/data/vectorstore/chroma_db/
```

벡터스토어가 없으면 `data/` 폴더의 스크립트로 생성해야 합니다:

```bash
cd src/generation/chat_bot/data
python 06_build_vectorstore.py
```

## 문의

챗봇 관련 문의: @배현석
