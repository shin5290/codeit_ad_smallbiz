# RAG 파이프라인 아키텍처 설계서

**작성일**: 2026-01-15
**담당**: 백엔드 로직 (진수경)
**버전**: 1.0
**목적**: 광고 생성 시스템의 RAG 기반 챗봇 파이프라인 아키텍처 명세

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [RAG 파이프라인 플로우](#2-rag-파이프라인-플로우)
3. [의도 분석 (Intent Analysis)](#3-의도-분석-intent-analysis)
4. [VectorDB 검색 전략](#4-vectordb-검색-전략)
5. [생성 파이프라인 통합](#5-생성-파이프라인-통합)
6. [데이터베이스 구조](#6-데이터베이스-구조)
7. [API 엔드포인트 설계](#7-api-엔드포인트-설계)
8. [백엔드 아키텍처 다이어그램](#8-백엔드-아키텍처-다이어그램)

---

## 1. 시스템 개요

### 1.1 프로젝트 목적

중소기업 광고 생성 플랫폼으로, 사용자의 자연어 입력을 받아 맞춤형 광고 콘텐츠(텍스트/이미지)를 생성하는 시스템입니다.

### 1.2 핵심 컴포넌트

```
┌─────────────────────────────────────────────────────────────┐
│                         사용자 입력                            │
│                  (텍스트 + 이미지 선택사항)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG 챗봇 파이프라인                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 1. 의도 분석 (Intent Analysis)                       │     │
│  │    - 생성 (Generation)                              │     │
│  │    - 수정 (Modification)                            │     │
│  │    - 상담 (Consulting)                              │     │
│  └────────────────────────────────────────────────────┘     │
│                       ↓                                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 2. VectorDB 검색                                    │     │
│  │    A. PostgreSQL 벡터 검색 (대화 히스토리.)              │     │
│  │    B. 정적 파일 VectorDB (상담 지식베이스)               │     │
│  └────────────────────────────────────────────────────┘     │
│                       ↓                                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 3.1 상담 파이프라인 (GPT-4o-mini )                     │     │
│  │    - 컨텍스트: 검색 결과 + 워크플로우 상태                 │     │
│  │    - 출력: 챗봇 응답 + 추출된 정보                       │     │
│  └────────────────────────────────────────────────────┘     │
│                       ↓                                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 3.2 광고 생성 파이프라인                                │     │
│  │    - 텍스트 생성: ad_generator.generate_advertisement │     │
│  │    - 이미지 생성: generator.generate_and_save_image   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 주요 특징

- **RAG (Retrieval-Augmented Generation)**: 대화 히스토리와 지식베이스를 검색하여 응답 품질 향상
- **의도 기반 라우팅**: 사용자 의도에 따라 생성/수정/상담 플로우 분기
- **이중 VectorDB 전략**:
  - PostgreSQL: 대화 히스토리 (사용자별, 세션별)
  - 정적 파일 VectorDB: 상담 지식베이스 (팀원 담당)
- **워크플로우 상태 관리**: 광고 생성에 필요한 정보 수집 추적

---

## 2. RAG 파이프라인 플로우

### 2.1 전체 사이클

```
                    [사용자 질문]
                        ↓
             [의도 분석 (Intent Analysis)]
                        ↓
┌───────────────┬───────────────┬───────────────┐
│     생성       │     수정       │     상담       │
│ (Generation)  │(Modification) │(Consulting)   │
└───────┬───────┴───────┬───────┴───────┬───────┘
        ↓               ↓               ↓
  [VectorDB 검색: PostgreSQL (대화/생성 히스토리)]
        +               +               +
    [VectorDB 검색: 정적 파일 (상담 지식베이스)]
        ↓               ↓               ↓
    [ 광고 생성 파이프라인 호출 ]     [LLM 호출 - 컨텍스트 기반 응답 생성]
        ↓               ↓               ↓
  [광고 생성]     [기존 광고 수정]        [상담 응답]
        ↓               ↓               ↓
    [DB 저장: GenerationHistory, ChatHistory]
                        ↓
                   [사용자 응답]
                        ↓
                  [다음 사용자 질문] → (사이클 반복)
```

### 2.2 단계별 상세 설명

#### Step 1: 사용자 입력 수신

**엔드포인트**: `POST /api/chat/message`

**입력**:
```json
{
  "session_id": "sess_abc123",
  "message": "카페 광고 이미지를 만들고 싶어요",
  "image": null  // 선택사항
}
```

**처리**:
- 세션 확보 (없으면 자동 생성)
- 이미지 업로드 처리 (있는 경우)
- 메시지 저장 (PostgreSQL ChatHistory)

#### Step 2: 의도 분석 (Intent Analysis)

**LLM 프롬프트 예시**:
```
시스템: 당신은 광고 제작 어시스턴트입니다. 사용자 메시지를 분석하여 의도를 파악하세요.

의도 분류:
- generation: 새로운 광고를 만들고 싶어하는 경우
- modification: 기존 광고를 수정하고 싶어하는 경우
- consulting: 광고 제작 방법이나 조언을 구하는 경우

사용자 메시지: "{user_message}"

JSON 형식으로 응답:
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "extracted_info": {
    "ad_type": "text|image",
    "business_type": "카페",
    ...
  }
}
```

**출력**:
```json
{
  "intent": "generation",
  "confidence": 0.95,
  "extracted_info": {
    "ad_type": "image",
    "business_type": "카페"
  }
}
```

#### Step 3: VectorDB 검색 (이중 전략)

##### 3.1 PostgreSQL 벡터 검색 (대화/생성 히스토리)

**목적**:
- 사용자의 과거 대화 맥락 이해
- 이전 광고 생성 이력 참조

**검색 대상**:
- `ChatHistory` 테이블: 세션별 대화 메시지
- `GenerationHistory` 테이블: 광고 생성 이력

**검색 쿼리**:
```sql
-- 유사 대화 검색 (pgvector 사용 시)
SELECT content, role, created_at
FROM chat_history
WHERE session_id = '{session_id}'
ORDER BY created_at DESC
LIMIT 10;

-- 유사 생성 이력 검색
SELECT output_text, output_image_id, prompt, style, industry
FROM generation_history
WHERE session_id = '{session_id}'
  AND content_type = '{ad_type}'
ORDER BY created_at DESC
LIMIT 5;
```

**검색 결과 예시**:
```python
{
  "recent_conversations": [
    {"role": "user", "content": "카페 광고 만들고 싶어요", "timestamp": "..."},
    {"role": "assistant", "content": "어떤 스타일을 원하시나요?", "timestamp": "..."}
  ],
  "previous_generations": [
    {
      "output_text": "따뜻한 커피 한 잔으로 시작하는 하루",
      "style": "ultra_realistic",
      "industry": "cafe"
    }
  ]
}
```

##### 3.2 정적 파일 VectorDB 검색 (상담 지식베이스)

**목적**:
- 광고 제작 가이드라인 참조
- 업종별 베스트 프랙티스 제공
- FAQ 응답

**검색 대상**:
- 정적 문서 (Markdown, PDF 등)
- 팀원이 관리하는 별도 VectorDB (ChromaDB 또는 Pinecone)

**검색 쿼리** (ChromaDB 예시):
```python
results = chroma_collection.query(
    query_texts=[user_message],
    n_results=3,
    where={"document_type": "consulting_guide"}
)
```

**검색 결과 예시**:
```python
{
  "documents": [
    {
      "content": "카페 광고는 따뜻한 분위기와 커피 향을 연상시키는 이미지가 효과적입니다...",
      "source": "cafe_advertising_guide.md",
      "score": 0.89
    }
  ]
}
```

**중요**: 이 부분은 다른 팀원이 담당하며, API 인터페이스만 정의하면 됨.

#### Step 4: LLM 호출 (컨텍스트 통합)

**컨텍스트 구성**:
```python
context = {
    "user_message": message,
    "intent": intent_analysis_result,
    "workflow_state": {
        "ad_type": "image",
        "business_type": "카페",
        "is_complete": False,
        "missing_info": ["style", "aspect_ratio"]
    },
    "conversation_history": recent_conversations,
    "generation_history": previous_generations,
    "knowledge_base": consulting_guide_results
}
```

**LLM 프롬프트**:
```
시스템: 당신은 광고 제작 어시스턴트입니다.

사용자 정보:
- 의도: {intent}
- 현재 수집된 정보: {workflow_state}
- 과거 대화: {conversation_history}
- 이전 생성 이력: {generation_history}
- 관련 가이드: {knowledge_base}

사용자 메시지: "{user_message}"

다음을 수행하세요:
1. 누락된 정보가 있으면 자연스럽게 질문하세요.
2. 모든 정보가 모였으면 "ready_to_generate": true를 반환하세요.
3. JSON 형식으로 응답:

{
  "assistant_message": "어떤 스타일의 이미지를 원하시나요? 사실적인 스타일과 애니메이션 스타일 중 선택해주세요.",
  "extracted_info": {
    "ad_type": "image",
    "business_type": "카페"
  },
  "ready_to_generate": false,
  "missing_info": ["style", "aspect_ratio"]
}
```

**LLM 응답**:
```json
{
  "assistant_message": "카페 광고 이미지를 만들어드리겠습니다! 어떤 스타일을 원하시나요?\n\n1. 사실적인 스타일 (ultra_realistic)\n2. 세미 사실적 스타일 (semi_realistic)\n3. 애니메이션 스타일 (anime)",
  "extracted_info": {
    "ad_type": "image",
    "business_type": "카페"
  },
  "ready_to_generate": false,
  "missing_info": ["style", "aspect_ratio"]
}
```

#### Step 5: 광고 생성 파이프라인 (조건부)

**조건**: `ready_to_generate == true`

**엔드포인트**: `POST /api/chat/generate`

**처리 플로우**:
```python
# services.py: handle_chat_generate()
1. 워크플로우 상태 검증 (필수 정보 확인)
2. handle_generate_pipeline() 호출
   ├─ ingest_user_message(): 입력 수집/저장
   ├─ generate_contents(): 광고 생성
   │   ├─ text_generation.ad_generator.generate_advertisement()
   │   └─ image_generation.generator.generate_and_save_image()
   └─ persist_generation_result(): DB 저장
3. Task 진행률 업데이트 (0% → 100%)
4. 결과 반환
```

**생성 결과**:
```json
{
  "session_id": "sess_abc123",
  "output": {
    "content_type": "image",
    "output_text": "따뜻한 커피 한 잔으로 시작하는 하루",
    "image": "abc123def456.png"
  }
}
```

#### Step 6: 결과 저장 및 응답

**DB 저장**:
- `ChatHistory`: 어시스턴트 응답 메시지
- `GenerationHistory`: 광고 생성 이력 (메타데이터 포함)
- `ImageMatching`: 생성된 이미지 파일 정보

**사용자 응답**:
```json
{
  "session_id": "sess_abc123",
  "assistant_message": "카페 광고 이미지가 생성되었습니다!",
  "ready_to_generate": false,
  "workflow_state": {
    "ad_type": "image",
    "business_type": "카페",
    "style": "ultra_realistic",
    "is_complete": true
  },
  "generated_content": {
    "image": "abc123def456.png",
    "text": "따뜻한 커피 한 잔으로 시작하는 하루"
  }
}
```

---

## 3. 의도 분석 (Intent Analysis)

### 3.1 의도 분류

| 의도 | 설명 | 예시 |
|-----|------|------|
| **생성 (Generation)** | 새로운 광고를 만들고 싶은 경우 | "카페 광고 만들어줘", "이미지 광고 생성해줘" |
| **수정 (Modification)** | 기존 광고를 수정하고 싶은 경우 | "이미지를 더 밝게 만들어줘", "텍스트를 변경해줘" |
| **상담 (Consulting)** | 광고 제작 방법이나 조언을 구하는 경우 | "카페 광고는 어떻게 만들어?", "효과적인 광고 문구는?" |

### 3.2 의도별 VectorDB 검색 전략

#### 3.2.1 생성 (Generation)

**검색 대상**:
1. **PostgreSQL**:
   - 세션별 대화 히스토리 (이전 광고 생성 맥락 파악)
   - 유사 업종의 생성 이력 (참고용)
2. **정적 파일 VectorDB**:
   - 업종별 광고 템플릿
   - 광고 제작 가이드라인

**검색 쿼리 예시**:
```python
# PostgreSQL
recent_conversations = get_recent_messages(session_id, limit=10)
similar_generations = search_similar_generations(
    business_type="카페",
    ad_type="image",
    limit=5
)

# 정적 파일 VectorDB (팀원 담당)
knowledge_results = consulting_vectordb.search(
    query="카페 광고 이미지 제작 가이드",
    filters={"category": "generation_guide"},
    limit=3
)
```

#### 3.2.2 수정 (Modification)

**검색 대상**:
1. **PostgreSQL**:
   - 현재 세션의 최근 생성 이력 (수정할 대상 찾기)
   - 수정 요청 패턴 학습 (과거 수정 이력)
2. **정적 파일 VectorDB**:
   - 수정 가능한 파라미터 설명
   - 수정 예시 문서

**검색 쿼리 예시**:
```python
# PostgreSQL
latest_generation = get_latest_generation(session_id)
modification_history = search_modification_patterns(
    user_input="더 밝게",
    limit=5
)

# 정적 파일 VectorDB
modification_guide = consulting_vectordb.search(
    query="이미지 밝기 조절 방법",
    filters={"category": "modification_guide"},
    limit=2
)
```

#### 3.2.3 상담 (Consulting)

**검색 대상**:
1. **PostgreSQL**:
   - 세션별 대화 히스토리 (대화 맥락 유지)
2. **정적 파일 VectorDB** (주요):
   - FAQ 문서
   - 광고 제작 가이드
   - 업종별 베스트 프랙티스

**검색 쿼리 예시**:
```python
# PostgreSQL
recent_conversations = get_recent_messages(session_id, limit=10)

# 정적 파일 VectorDB (팀원 담당 - 가장 중요)
consulting_results = consulting_vectordb.search(
    query="카페 광고 효과적인 방법",
    filters={"category": "faq"},
    limit=5
)
```

### 3.3 모든 의도에서 PostgreSQL 벡터 검색이 필요한 이유

**핵심 개념**: "대화는 맥락이다"

1. **대화 히스토리 추적**:
   - 사용자가 이전에 무엇을 물었는지
   - 어시스턴트가 무엇을 답변했는지
   - 현재 워크플로우 진행 상태

2. **생성 이력 참조**:
   - 사용자가 이전에 만든 광고
   - 선호하는 스타일/업종
   - 수정 요청 패턴

3. **개인화**:
   - 사용자별 맞춤 응답
   - 세션별 상태 관리

**예시**: 상담 의도에서도 PostgreSQL 검색이 필요한 경우
```
사용자: "카페 광고는 어떻게 만들어?"
-> PostgreSQL 검색: 이 사용자가 이전에 카페 광고를 만든 적이 있는지 확인
-> 있다면: "이전에 만드신 카페 광고처럼 이런 방식으로 제작할 수 있습니다."
-> 없다면: "일반적인 카페 광고 제작 가이드를 안내해드리겠습니다."
```

---

## 4. VectorDB 검색 전략

### 4.1 PostgreSQL 벡터 검색 (대화/생성 히스토리)

#### 4.1.1 구현 방식

**옵션 1**: pgvector 확장 사용 (권장)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE chat_history
ADD COLUMN embedding vector(1536);

-- 유사도 검색
SELECT content, role, created_at
FROM chat_history
WHERE session_id = '{session_id}'
ORDER BY embedding <-> '{query_embedding}'::vector
LIMIT 10;
```

**옵션 2**: 단순 시간순 정렬 (현재 구현)
```sql
SELECT content, role, created_at
FROM chat_history
WHERE session_id = '{session_id}'
ORDER BY created_at DESC
LIMIT 10;
```

#### 4.1.2 검색 대상 테이블

**ChatHistory 테이블**:
```python
{
  "id": 123,
  "session_id": "sess_abc123",
  "role": "user",
  "content": "카페 광고 만들고 싶어요",
  "image_id": null,
  "created_at": "2026-01-15T10:00:00"
}
```

**GenerationHistory 테이블**:
```python
{
  "id": 456,
  "session_id": "sess_abc123",
  "content_type": "image",
  "input_text": "카페 광고 이미지",
  "output_text": "따뜻한 커피 한 잔으로 시작하는 하루",
  "prompt": "A cozy cafe with warm lighting...",
  "style": "ultra_realistic",
  "industry": "cafe",
  "seed": 42,
  "aspect_ratio": "1:1",
  "created_at": "2026-01-15T10:05:00"
}
```

#### 4.1.3 검색 함수 (process_db.py)

```python
def get_chat_history_by_session(
    db: Session,
    session_id: str,
    limit: int = 10
) -> List[models.ChatHistory]:
    """세션별 최근 대화 히스토리 조회"""
    return (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.session_id == session_id)
        .order_by(models.ChatHistory.created_at.desc())
        .limit(limit)
        .all()
    )

def get_generation_history_by_session(
    db: Session,
    session_id: str,
    limit: int = 5
) -> List[models.GenerationHistory]:
    """세션별 광고 생성 이력 조회"""
    return (
        db.query(models.GenerationHistory)
        .filter(models.GenerationHistory.session_id == session_id)
        .order_by(models.GenerationHistory.created_at.desc())
        .limit(limit)
        .all()
    )
```

### 4.2 정적 파일 VectorDB (상담 지식베이스)

#### 4.2.1 담당 분리

**이 부분은 다른 팀원이 담당**:
- VectorDB 선택 (ChromaDB, Pinecone, Weaviate 등)
- 정적 문서 관리 (Markdown, PDF 등)
- 임베딩 생성 및 저장
- 검색 API 구현

**백엔드 로직 담당 (나)**:
- API 인터페이스 정의
- 검색 결과 통합 로직

#### 4.2.2 API 인터페이스 (예시)

```python
# chatbot.py (가상의 인터페이스)

class ConsultingKnowledgeBase:
    """상담 지식베이스 인터페이스"""

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 3
    ) -> List[dict]:
        """
        정적 문서 검색

        Args:
            query: 검색 쿼리
            category: 문서 카테고리 (faq, generation_guide, modification_guide 등)
            limit: 반환할 문서 수

        Returns:
            [
                {
                    "content": "문서 내용...",
                    "source": "파일명",
                    "score": 0.89,
                    "metadata": {...}
                }
            ]
        """
        # 팀원이 구현할 부분
        pass
```

#### 4.2.3 통합 예시

```python
# chatbot.py: RAGChatbot.process_message()

# 1. PostgreSQL 검색
recent_conversations = conv_manager.get_recent_messages(db, session_id, limit=10)
generation_history = conv_manager.get_generation_history(db, session_id, limit=5)

# 2. 정적 파일 VectorDB 검색 (팀원 API 호출)
knowledge_results = consulting_knowledge_base.search(
    query=user_message,
    category="faq" if intent == "consulting" else "generation_guide",
    limit=3
)

# 3. 컨텍스트 통합
context = {
    "recent_conversations": recent_conversations,
    "generation_history": generation_history,
    "knowledge_base": knowledge_results
}

# 4. LLM 호출
llm_response = orchestrator.generate_response(user_message, context)
```

### 4.3 검색 전략 비교 정리

| VectorDB | 담당 | 검색 대상 | 주요 용도 | 구현 우선순위 |
|---------|------|----------|----------|-------------|
| **PostgreSQL** | 백엔드 로직 (나) | 대화/생성 히스토리 | 맥락 유지, 개인화 | HIGH (필수) |
| **정적 파일 VectorDB** | 상담 챗봇 팀원 | FAQ, 가이드 문서 | 상담 응답, 지식 제공 | MEDIUM (협업) |

---

## 5. 생성 파이프라인 통합

### 5.1 생성 파이프라인 개요

**위치**: `src/generation/`

**구성 요소**:
1. **텍스트 생성**: `text_generation/ad_generator.py`
   - `generate_advertisement()`: 광고 문구 생성
2. **이미지 생성**: `image_generation/generator.py`
   - `generate_and_save_image()`:  프롬프트 + Stable Diffusion 기반 이미지 생성

### 5.2 LLM 호출 = 생성 파이프라인 호출

**중요한 개념 정정**:

사용자가 "LLM 호출 = 생성 파이프라인"이라고 이해했지만, 정확히는:

```
LLM 호출 ≠ 생성 파이프라인
LLM 호출 = 챗봇 응답 생성 (GPT-4o-mini)
생성 파이프라인 = 광고 생성 (텍스트 모델 + 이미지 모델)
```

**올바른 플로우**:
```
사용자 질문
    ↓
[LLM 호출 1]: 의도 분석 + 정보 수집 (GPT-4o-mini)
    ↓
(정보가 모두 수집되면)
    ↓
[생성 파이프라인 호출]:
    ├─ [텍스트 모델]: 광고 문구 생성 (Qwen 등)
    └─ [이미지 모델]: 이미지 생성 (Stable Diffusion)
    ↓
[LLM 호출 2]: 결과 설명 생성 (GPT-4o-mini)
```

### 5.3 통합 코드 (services.py)

#### 5.3.1 현재 구현된 함수

```python
# services.py: handle_generate_pipeline()

async def handle_generate_pipeline(
    *,
    db: Session,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    image: Optional[UploadFile] = None,
    task_id: str,
    generation_type: str,  # "text" or "image"
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
):
    """
    광고 생성 파이프라인 통합 서비스

    단계:
    1. ingest_user_message(): 입력 수집/저장
    2. generate_contents(): 광고 생성
       ├─ text_generation.ad_generator.generate_advertisement()
       └─ image_generation.generator.generate_and_save_image()
    3. persist_generation_result(): DB 저장
    4. Task 진행률 업데이트 (0% → 100%)
    """
    pass  # 구현 생략 (코드 참조)
```

#### 5.3.2 RAG 챗봇 통합

```python
# services.py: handle_chat_generate()

async def handle_chat_generate(
    *,
    db: Session,
    session_id: str,
    user_id: Optional[int],
    task_id: str,
):
    """
    챗봇을 통해 수집된 정보로 광고 생성

    단계:
    1. 워크플로우 상태 조회
    2. 필수 정보 검증
    3. handle_generate_pipeline() 호출
    4. 워크플로우 초기화
    """
    # 1. 워크플로우 상태 조회
    chatbot = get_chatbot()
    workflow_state = chatbot.get_workflow_state(session_id)

    if not workflow_state.is_complete:
        raise HTTPException(
            status_code=400,
            detail=f"필수 정보가 부족합니다: {', '.join(workflow_state.get_missing_info())}"
        )

    # 2. 광고 생성 파이프라인 호출
    result = await handle_generate_pipeline(
        db=db,
        input_text=workflow_state.user_input,
        session_id=session_id,
        user_id=user_id,
        image=None,
        task_id=task_id,
        generation_type=workflow_state.ad_type,
        style=workflow_state.style,
        aspect_ratio=workflow_state.aspect_ratio,
    )

    # 3. 워크플로우 초기화
    chatbot.reset_workflow(session_id)

    return result
```

### 5.4 생성 파이프라인 상세 플로우

```
handle_chat_generate()
    ↓
handle_generate_pipeline()
    ↓
┌─────────────────────────────────────────┐
│ Step 1: ingest_user_message()           │
│  - 세션 확보                              │
│  - 이미지 저장 (있는 경우)                   │
│  - ChatHistory 저장                      │
│  [Progress: 0% → 5%]                    │
└─────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ Step 2: generate_contents()               │
│  ├─ text_generation                       │
│  │   └─ ad_generator.generate_advertisement()│
│  │       - 광고 문구 생성                     │
│  │       - 프롬프트 생성                      │
│  │  [Progress: 5% → 30%]                  │
│  │                                        │
│  └─ image_generation (if ad_type="image") │
│      └─ generator.generate_and_save_image()│
│          - Stable Diffusion 이미지 생성      │
│          - ControlNet (참고 이미지 사용)      │
│      [Progress: 30% → 70%]                │
└───────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Step 3: persist_generation_result()     │
│  - ChatHistory 저장 (assistant 응답)      │
│  - GenerationHistory 저장                │
│  - ImageMatching 저장 (이미지 메타데이터)    │
│  [Progress: 70% → 90%]                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Step 4: complete_task()                 │
│  - Task 상태: DONE                       │
│  - 결과 반환                              │
│  [Progress: 90% → 100%]                 │
└─────────────────────────────────────────┘
```

---

## 6. 데이터베이스 구조

### 6.1 ERD (Entity Relationship Diagram)

```
┌─────────────┐
│    User     │
│─────────────│
│ user_id PK  │
│ login_id    │
│ login_pw    │
│ name        │
│ created_at  │
└──────┬──────┘
       │ 1
       │
       │ N
┌──────▼────────────┐
│   ChatSession     │
│───────────────────│
│ session_id PK     │
│ user_id FK        │ (nullable)
│ created_at        │
└──────┬───────┬────┘
       │ 1     │ 1
       │       │
       │ N     │ N
┌──────▼─────────┐ ┌──────▼───────────────────┐
│  ChatHistory   │ │   GenerationHistory      │
│────────────────│ │──────────────────────────│
│ id PK          │ │ id PK                    │
│ session_id FK  │ │ session_id FK            │
│ role           │ │ content_type             │
│ content        │ │ input_text               │
│ image_id FK    │ │ output_text              │
│ created_at     │ │ prompt                   │
└────────┬───────┘ │ input_image_id FK        │
         │         │ output_image_id FK       │
         │         │ generation_method        │
         │         │ style, industry, seed    │
         │         │ aspect_ratio, created_at │
         │ N       └────────┬─────────┬───────┘
         │                  │ N       │ N
         │                  │         │
         │ 1                │ 1       │ 1
┌────────▼──────────────────▼─────────▼───┐
│          ImageMatching                   │
│──────────────────────────────────────────│
│ id PK                                    │
│ file_hash (unique)                       │
│ file_directory                           │
│ created_at                               │
└──────────────────────────────────────────┘
```

### 6.2 테이블 상세

#### User (사용자)
```sql
CREATE TABLE user (
    user_id INTEGER PRIMARY KEY,
    login_id VARCHAR(50) UNIQUE NOT NULL,
    login_pw VARCHAR(255) NOT NULL,  -- bcrypt hashed
    name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### ChatSession (채팅 세션)
```sql
CREATE TABLE chat_session (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id INTEGER,  -- nullable (게스트 지원)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE SET NULL
);
```

#### ChatHistory (대화 히스토리)
```sql
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    image_id INTEGER,  -- nullable
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_session(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (image_id) REFERENCES image_matching(id) ON DELETE SET NULL
);

CREATE INDEX idx_chat_history_session ON chat_history(session_id, created_at);
```

#### GenerationHistory (생성 이력)
```sql
CREATE TABLE generation_history (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    content_type VARCHAR(20) NOT NULL,  -- 'text' or 'image'
    input_text TEXT,
    output_text TEXT,
    prompt TEXT,  -- 이미지 생성용 프롬프트
    input_image_id INTEGER,  -- nullable
    output_image_id INTEGER,  -- nullable
    generation_method VARCHAR(50),  -- 'canny', 'depth', 'openpose' (ControlNet)
    style VARCHAR(50),  -- 'ultra_realistic', 'semi_realistic', 'anime'
    industry VARCHAR(50),  -- 'cafe', 'restaurant', 'retail' 등
    seed INTEGER,
    aspect_ratio VARCHAR(10),  -- '1:1', '16:9', '9:16', '4:3'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_session(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (input_image_id) REFERENCES image_matching(id) ON DELETE SET NULL,
    FOREIGN KEY (output_image_id) REFERENCES image_matching(id) ON DELETE SET NULL
);

CREATE INDEX idx_generation_history_session ON generation_history(session_id, created_at);
```

#### ImageMatching (이미지 파일 레지스트리)
```sql
CREATE TABLE image_matching (
    id INTEGER PRIMARY KEY,
    file_hash VARCHAR(128) UNIQUE NOT NULL,  -- SHA-256 hash
    file_directory TEXT NOT NULL,  -- 디스크 경로
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_image_matching_hash ON image_matching(file_hash);
```

### 6.3 벡터 검색 확장 (pgvector)

**설치** (선택사항):
```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- ChatHistory에 임베딩 컬럼 추가
ALTER TABLE chat_history
ADD COLUMN embedding vector(1536);  -- OpenAI embedding 차원

-- 유사도 검색 인덱스
CREATE INDEX idx_chat_history_embedding
ON chat_history
USING ivfflat (embedding vector_cosine_ops);
```

**검색 쿼리**:
```sql
-- 유사 대화 검색
SELECT content, role, created_at
FROM chat_history
WHERE session_id = '{session_id}'
ORDER BY embedding <-> '{query_embedding}'::vector
LIMIT 10;
```

---

## 7. API 엔드포인트 설계

### 7.1 챗봇 메시지

#### POST /api/chat/message

**요청**:
```http
POST /api/chat/message HTTP/1.1
Content-Type: multipart/form-data
Authorization: Bearer {jwt_token}

message: "카페 광고 이미지를 만들고 싶어요"
session_id: "sess_abc123"  (optional)
image: [binary]  (optional)
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "assistant_message": "카페 광고 이미지를 만들어드리겠습니다! 어떤 스타일을 원하시나요?\n\n1. 사실적인 스타일 (ultra_realistic)\n2. 세미 사실적 스타일 (semi_realistic)\n3. 애니메이션 스타일 (anime)",
  "ready_to_generate": false,
  "workflow_state": {
    "ad_type": "image",
    "business_type": "카페",
    "style": null,
    "aspect_ratio": null,
    "is_complete": false,
    "missing_info": ["style", "aspect_ratio"]
  }
}
```

### 7.2 광고 생성

#### POST /api/chat/generate

**요청**:
```http
POST /api/chat/generate HTTP/1.1
Content-Type: application/x-www-form-urlencoded
Authorization: Bearer {jwt_token}

session_id=sess_abc123
```

**응답**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 7.3 작업 상태 조회

#### GET /api/task/{task_id}

**요청**:
```http
GET /api/task/550e8400-e29b-41d4-a716-446655440000 HTTP/1.1
Authorization: Bearer {jwt_token}
```

**응답** (진행 중):
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "generating",
  "progress": 45,
  "result": null,
  "error": null
}
```

**응답** (완료):
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "done",
  "progress": 100,
  "result": {
    "session_id": "sess_abc123",
    "output": {
      "content_type": "image",
      "output_text": "따뜻한 커피 한 잔으로 시작하는 하루",
      "image": "abc123def456.png"
    }
  },
  "error": null
}
```

### 7.4 워크플로우 상태 조회

#### GET /api/chat/workflow/{session_id}

**요청**:
```http
GET /api/chat/workflow/sess_abc123 HTTP/1.1
Authorization: Bearer {jwt_token}
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "ad_type": "image",
  "business_type": "카페",
  "user_input": null,
  "style": "ultra_realistic",
  "aspect_ratio": "1:1",
  "platform": null,
  "target_audience": null,
  "is_complete": true,
  "missing_info": []
}
```

### 7.5 워크플로우 초기화

#### POST /api/chat/workflow/{session_id}/reset

**요청**:
```http
POST /api/chat/workflow/sess_abc123/reset HTTP/1.1
Authorization: Bearer {jwt_token}
```

**응답**:
```json
{
  "message": "워크플로우가 초기화되었습니다."
}
```

### 7.6 대화 히스토리 조회

#### GET /api/chat/history/{session_id}

**요청**:
```http
GET /api/chat/history/sess_abc123?limit=20 HTTP/1.1
Authorization: Bearer {jwt_token}
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "messages": [
    {
      "role": "user",
      "content": "카페 광고 이미지를 만들고 싶어요",
      "timestamp": "2026-01-15T10:00:00",
      "image_id": null
    },
    {
      "role": "assistant",
      "content": "어떤 스타일을 원하시나요?",
      "timestamp": "2026-01-15T10:00:05",
      "image_id": null
    }
  ]
}
```

---

## 8. 백엔드 아키텍처 다이어그램

### 8.1 레이어 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                     │
│                    (FastAPI Routers)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ /chat/*     │  │ /generate/* │  │ /task/*     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│                      Service Layer                         │
│                    (services.py)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ handle_chat_message()                                │  │
│  │  - 챗봇 대화 처리                                       │  │
│  │  - 의도 분석                                           │  │
│  │  - VectorDB 검색                                      │  │
│  │  - LLM 호출                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ handle_chat_generate()                               │  │
│  │  - 워크플로우 상태 검증                                   │  │
│  │  - 생성 파이프라인 호출                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ handle_generate_pipeline()                           │  │
│  │  - 입력 수집 (ingest_user_message)                     │  │
│  │  - 광고 생성 (generate_contents)                       │  │
│  │  - 결과 저장 (persist_generation_result)               │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────┐
│                      Business Logic Layer                 │
│  ┌──────────────────────┐  ┌─────────────────────────┐    │
│  │ chatbot.py           │  │ generation/             │    │
│  │ (미구현 - 필요)         │  │ ├─ text_generation/     │    │
│  │                      │  │ │  └─ ad_generator.py   │    │
│  │ - RAGChatbot         │  │ └─ image_generation/    │    │
│  │ - ConversationManager│  │    └─ generator.py      │    │
│  │ - LLMOrchestrator    │  └─────────────────────────┘    │
│  │ - WorkflowState      │                                 │
│  └──────────────────────┘                                 │
└───────────────────────┬───────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│                      Data Access Layer                     │
│                    (process_db.py, models.py)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ process_db.py                                        │  │
│  │  - get_chat_history_by_session()                     │  │
│  │  - get_generation_history_by_session()               │  │
│  │  - save_chat_message()                               │  │
│  │  - save_generation_history()                         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ models.py                                            │  │
│  │  - User, ChatSession, ChatHistory                    │  │
│  │  - GenerationHistory, ImageMatching                  │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│                      Database Layer                        │
│  ┌────────────────────┐         ┌──────────────────────┐   │
│  │ PostgreSQL         │         │ 정적 파일 VectorDB     │   │
│  │ - ChatHistory      │         │ (팀원 담당)            │   │
│  │ - GenerationHistory│         │ - ChromaDB?          │   │
│  │ - (pgvector?)      │         │ - Pinecone?          │   │
│  └────────────────────┘         └──────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### 8.2 RAG 파이프라인 플로우 (상세)

```
┌────────────────┐
│ 사용자 입력       │
│ "카페 광고"      │
└────────┬───────┘
         │
         ▼
┌────────────────────────────────────────┐
│ routers/chat.py                        │
│ POST /api/chat/message                 │
│  - 입력 수신                             │
│  - 인증 확인                             │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ services.handle_chat_message()         │
│  ├─ 세션 확보 (normalize_session_id)     │
│  ├─ 이미지 처리 (save_uploaded_image)     │
│  └─ chatbot.process_message() 호출      │
└────────┬───────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────┐
│ chatbot.RAGChatbot.process_message()  │
│ (미구현 - 필요)                          │
│  ┌─────────────────────────────────┐  │
│  │ 1. 의도 분석                      │  │
│  │    LLM 호출 → intent 판단         │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 2. VectorDB 검색                 │  │
│  │    A. PostgreSQL                │  │
│  │       - get_chat_history()      │  │
│  │       - get_generation_history()│  │
│  │    B. 정적 파일 VectorDB          │  │
│  │       - knowledge_base.search() │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 3. 워크플로우 상태 업데이트           │  │
│  │    - extracted_info 반영         │  │
│  │    - is_complete 체크            │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 4. LLM 호출 (응답 생성)            │  │
│  │    - 컨텍스트 통합                 │  │
│  │    - 챗봇 응답 생성                │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 5. 대화 저장                      │  │
│  │    - save_chat_message()        │  │
│  └─────────────────────────────────┘  │
└────────┬──────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 사용자 응답                               │
│ {                                      │
│   "assistant_message": "...",          │
│   "ready_to_generate": false,          │
│   "workflow_state": {...}              │
│ }                                      │
└────────────────────────────────────────┘

(ready_to_generate = true 일 때)

         │
         ▼
┌────────────────────────────────────────┐
│ POST /api/chat/generate                │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ services.handle_chat_generate()        │
│  ├─ 워크플로우 검증                        │
│  └─ handle_generate_pipeline() 호출     │
└────────┬───────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────┐
│ services.handle_generate_pipeline()   │
│  ┌─────────────────────────────────┐  │
│  │ 1. ingest_user_message()        │  │
│  │    [Progress: 0% → 5%]          │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 2. generate_contents()          │  │
│  │    ├─ ad_generator              │  │
│  │    │  .generate_advertisement() │  │
│  │    │  [Progress: 5% → 30%]      │  │
│  │    └─ generator                 │  │
│  │       .generate_and_save_image()│  │
│  │       [Progress: 30% → 70%]     │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 3. persist_generation_result()  │  │
│  │    [Progress: 70% → 90%]        │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ 4. complete_task()              │  │
│  │    [Progress: 90% → 100%]       │  │
│  └─────────────────────────────────┘  │
└────────┬──────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 생성 완료                                │
│ {                                      │
│   "session_id": "sess_abc123",         │
│   "output": {                          │
│     "content_type": "image",           │
│     "output_text": "...",              │
│     "image": "abc123.png"              │
│   }                                    │
│ }                                      │
└────────────────────────────────────────┘
```

### 8.3 파일 구조

```
src/backend/
├── __init__.py
├── models.py                    # SQLAlchemy 모델 정의
├── schemas.py                   # Pydantic 스키마
├── process_db.py                # DB 접근 함수
├── services.py                  # 비즈니스 로직
├── task.py                      # Task 관리
├── chatbot.py                   # ⚠️ 미구현 (필요)
│   ├── RAGChatbot               # RAG 챗봇 메인
│   ├── ConversationManager      # 대화 관리자
│   ├── LLMOrchestrator          # LLM 호출 관리
│   └── WorkflowStateManager     # 워크플로우 상태 관리
└── routers/
    ├── __init__.py
    ├── auth.py                  # 인증 관련 엔드포인트
    ├── chat.py                  # 챗봇 엔드포인트
    └── generate.py              # 광고 생성 엔드포인트

src/generation/
├── text_generation/
│   ├── ad_generator.py          # ✅ 구현됨
│   └── text_generator.py
└── image_generation/
    ├── generator.py             # ✅ 구현됨
    ├── workflow.py
    └── nodes/
        ├── text2image.py
        ├── image2image.py
        └── controlnet.py
```

---

## 9. 핵심 개념 정리

### 9.1 RAG 사이클

```
[사용자 질문]
    ↓
[의도 분석 (Intent Analysis)]
    ↓
[VectorDB 검색 (PostgreSQL + 정적 파일)]
    ↓
[LLM 호출 (GPT-4o-mini: 챗봇 응답)] = [생성 필요 시: 광고 생성 파이프라인]
    ↓
[응답 반환]
    ↓
[다음 사용자 질문] → (사이클 반복)
```

### 9.2 VectorDB 이중 전략

| 구분 | PostgreSQL | 정적 파일 VectorDB |
|-----|-----------|------------------|
| **담당** | 백엔드 로직 (나) | 상담 챗봇 팀원 |
| **검색 대상** | 대화/생성 히스토리 | FAQ, 가이드 문서 |
| **용도** | 맥락 유지, 개인화 | 상담 응답, 지식 제공 |
| **모든 의도에 필요** | ✅ YES | 의도별 차등 |

### 9.3 LLM 호출 vs 생성 파이프라인

```
LLM 호출 (GPT-4o-mini):
  - 목적: 챗봇 응답 생성, 의도 분석
  - 입력: 사용자 메시지 + 컨텍스트
  - 출력: 챗봇 메시지 + extracted_info

생성 파이프라인 (Qwen + Stable Diffusion):
  - 목적: 광고 콘텐츠 생성
  - 입력: 광고 정보 (업종, 스타일, 비율 등)
  - 출력: 광고 문구 + 이미지
```

### 9.4 의도별 처리 흐름

#### 생성 (Generation)
```
의도 분석 → PostgreSQL (대화/생성 히스토리 검색)
         → 정적 파일 VectorDB (생성 가이드 검색)
         → LLM (정보 수집)
         → ready_to_generate = true
         → 생성 파이프라인 호출
```

#### 수정 (Modification)
```
의도 분석 → PostgreSQL (최근 생성 이력 조회)
         → 정적 파일 VectorDB (수정 가이드 검색)
         → LLM (수정 요청 파싱)
         → 생성 파이프라인 재호출 (파라미터 변경)
```

#### 상담 (Consulting)
```
의도 분석 → PostgreSQL (대화 히스토리 검색)
         → 정적 파일 VectorDB (FAQ, 가이드 검색) ⭐ 주요
         → LLM (상담 응답 생성)
         → 응답 반환 (생성 없음)
```

---

## 10. 다음 단계

### 10.1 구현 우선순위

#### HIGH (필수)
1. ✅ **현재 코드 분석 완료**
2. **chatbot.py 구현**
   - RAGChatbot 클래스
   - ConversationManager (PostgreSQL 검색)
   - LLMOrchestrator (GPT-4o-mini 호출)
   - WorkflowState 관리
3. **정적 파일 VectorDB 인터페이스 정의**
   - 팀원과 API 협의
   - ConsultingKnowledgeBase 인터페이스 작성

#### MEDIUM (개선)
4. **수정/컨펌 플로우 추가**
   - handle_chat_revise() 구현
   - handle_chat_confirm() 구현
   - WorkflowState에 phase 추가
5. **pgvector 도입 검토**
   - 벡터 검색 성능 향상
   - 유사도 기반 검색

#### LOW (최적화)
6. **캐싱 전략**
   - Redis 도입
   - 반복 질문 캐싱
7. **모니터링**
   - LLM 응답 품질 추적
   - 생성 성공률 추적

### 10.2 팀원 협업 포인트

**상담 챗봇 팀원**:
- 정적 파일 VectorDB 구축
- FAQ 문서 관리
- 검색 API 구현

**백엔드 로직 (나)**:
- chatbot.py 구현
- PostgreSQL 벡터 검색
- 생성 파이프라인 통합

**협업 인터페이스**:
```python
# consulting_knowledge_base.py (팀원 구현)

class ConsultingKnowledgeBase:
    def search(self, query: str, category: str, limit: int) -> List[dict]:
        """정적 문서 검색"""
        pass

# chatbot.py (내가 구현)
knowledge_base = ConsultingKnowledgeBase()
results = knowledge_base.search(
    query=user_message,
    category="faq",
    limit=3
)
```

---

**문서 작성**: Claude Code (Backend Developer Agent)
**검토 필요**: 백엔드 로직 담당자
**마지막 업데이트**: 2026-01-15
