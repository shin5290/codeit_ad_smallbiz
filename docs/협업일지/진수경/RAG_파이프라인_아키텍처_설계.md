# RAG 파이프라인 아키텍처 설계서

**작성일**: 2026-01-15
**담당**: 백엔드 로직 (진수경)
**버전**: 1.1 (현재 백엔드 구현 기준 반영)
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
│  │ 1. 의도 분석 (Intent Analysis, LLM)                  │     │
│  │    - 최근 대화 3~5턴만 사용                           │     │
│  │    - intent / generation_type / target_generation_id │     │
│  └────────────────────────────────────────────────────┘     │
│                       ↓                                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 2. 분기                                            │     │
│  │   A) consulting: PostgreSQL 검색 + 정적 KB → LLM 응답 │     │
│  │   B) generation/modification: 전체 히스토리 →        │     │
│  │      refine_generation_input(LLM) → generate_contents │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 주요 특징

- **RAG (Retrieval-Augmented Generation)**: 대화 히스토리와 지식베이스를 검색하여 응답 품질 향상
- **의도 기반 라우팅**: 사용자 의도에 따라 생성/수정/상담 플로우 분기
- **이중 VectorDB 전략**:
  - PostgreSQL: 대화 히스토리 (사용자별, 세션별)
  - 정적 파일 VectorDB: 상담 지식베이스 (팀원 담당)

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
[전체 히스토리 조회] [전체 히스토리 조회] [유사 대화 + 지식베이스]
        ↓               ↓               ↓
[refine_generation_input (LLM)] [refine_generation_input (LLM)] [상담 응답 LLM]
        ↓               ↓               ↓
[generate_contents]     [handle_chat_revise]   [상담 응답]
        ↓               ↓               ↓
[DB 저장: GenerationHistory, ChatHistory]     [ChatHistory 저장]
                        ↓
                   [사용자 응답]
                        ↓
                  [다음 사용자 질문] → (사이클 반복)
```

### 2.2 단계별 상세 설명

#### Step 1: 사용자 입력 수신

**엔드포인트**: `POST /api/chat/message/stream`

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

추가 규칙:
- generation_type은 intent가 generation일 때만 명확히 선택
- 수정 요청이면 target_generation_id를 선택 (없으면 null)

사용자 메시지: "{user_message}"

JSON 형식으로 응답:
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "generation_type": "image|text|null",
  "target_generation_id": 123 or null
}
```

**출력**:
```json
{
  "intent": "generation",
  "confidence": 0.92,
  "generation_type": "image",
  "target_generation_id": null
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
-- 최근 대화 조회
SELECT content, role, created_at
FROM chat_history
WHERE session_id = '{session_id}'
ORDER BY created_at DESC
LIMIT 5;

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

#### Step 4: 맥락 정제 / 상담 LLM 호출

**A) generation/modification용 맥락 정제 (LLM)**  
의도 분석 결과와 전체 히스토리를 기반으로 `refine_generation_input()`을 호출합니다.

```python
refined_input = await refine_generation_input(
    intent=intent,
    generation_type=generation_type,
    target_generation_id=target_generation_id,
    chat_history=all_chat_history,
    generation_history=all_generation_history,
)
```

- 번호/대명사 참조를 실제 문구로 치환
- 수정 요청은 **이전 생성물 복사 금지**, 요구사항 요약으로 변환
- 결과는 `refined_input` 단일 텍스트로 반환

**B) consulting용 응답 생성 (LLM)**  
최근 대화 + 지식베이스를 컨텍스트로 상담 응답을 생성합니다.

```python
context = {
    "recent_conversations": recent_conversations,
    "generation_history": generation_history,
    "knowledge_base": consulting_guide_results,
}
assistant_message = await generate_consulting_response(message, context)
```

#### Step 5: 광고 생성 파이프라인

**조건**: `intent in {"generation", "modification"}`

**엔드포인트**:
- `POST /api/chat/message/stream`: intent 분석 후 SSE 스트리밍으로 진행 상태/결과 전송

**처리 플로우**:
```python
# services.py
1. intent 분석 (선택) → generation_type / target_generation_id 결정
2. refine_generation_input() 호출 (전체 히스토리 기반)
3. generate_contents() 실행
   ├─ text_generation.text_generator.generate_ad_copy()
   └─ image_generation.generator.generate_and_save_image()
4. persist_generation_result(): DB 저장
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
- `GenerationHistory`: 광고 생성 이력 (`input_text`는 refined_input 저장)
- `ImageMatching`: 생성된 이미지 파일 정보

**사용자 응답 (generation/modification, SSE)**:
```
data: {"type":"meta","session_id":"sess_abc123","intent":"generation"}

data: {"type":"progress","stage":"analyzing","message":"요청을 정리하고 있습니다."}

data: {"type":"progress","stage":"generating","message":"광고를 생성하고 있습니다."}

data: {"type":"done","assistant_message":"광고가 생성되었습니다.","output":{"content_type":"image","output_text":"따뜻한 커피 한 잔으로 시작하는 하루","image":"abc123def456.png"}}
```

---

## 3. 의도 분석 (Intent Analysis)

### 3.1 의도 분류

| 의도 | 설명 | 예시 |
|-----|------|------|
| **생성 (Generation)** | 새로운 광고를 만들고 싶은 경우 | "카페 광고 만들어줘", "이미지 광고 생성해줘" |
| **수정 (Modification)** | 기존 광고를 수정하고 싶은 경우 | "이미지를 더 밝게 만들어줘", "텍스트를 변경해줘" |
| **상담 (Consulting)** | 광고 제작 방법이나 조언을 구하는 경우 | "카페 광고는 어떻게 만들어?", "효과적인 광고 문구는?" |

**현재 구현 기준**:
- 의도 분석은 최근 3~5턴만 참고 (경량)
- `generation_input` 정제는 이 단계에서 수행하지 않음

### 3.2 의도별 VectorDB 검색 전략

#### 3.2.1 생성 (Generation)

**검색 대상**:
1. **PostgreSQL**:
   - 세션별 최근 대화 히스토리 (최근 5턴)
   - 세션별 최근 생성 이력 (최근 5건)
2. **정적 파일 VectorDB**:
   - generation/modification에는 현재 사용하지 않음 (consulting 전용)

**검색 쿼리 예시**:
```python
# PostgreSQL
recent_conversations = get_recent_messages(session_id, limit=5)
generation_history = get_generation_history(session_id, limit=5)
```

#### 3.2.2 수정 (Modification)

**검색 대상**:
1. **PostgreSQL**:
   - 현재 세션의 최근 생성 이력 (수정할 대상 찾기)
   - target_generation_id가 있으면 해당 항목 우선
2. **정적 파일 VectorDB**:
   - 현재 수정 플로우에서는 사용하지 않음

**검색 쿼리 예시**:
```python
# PostgreSQL
latest_generation = get_latest_generation(session_id)
target_generation_id = intent_result.get("target_generation_id")
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
recent_conversations = get_recent_messages(
    session_id=session_id,
    limit=5,
)

# 정적 파일 VectorDB (팀원 담당 - 가장 중요)
consulting_results = consulting_vectordb.search(
    query="카페 광고 효과적인 방법",
    filters={"category": "faq"},
    limit=5
)
```

### 3.3 모든 의도에서 PostgreSQL 대화/생성 이력 조회가 필요한 이유

**핵심 개념**: "대화는 맥락이다"

1. **대화 히스토리 추적**:
   - 사용자가 이전에 무엇을 물었는지
   - 어시스턴트가 무엇을 답변했는지
   - 현재 대화 진행 상태

2. **생성 이력 참조**:
   - 사용자가 이전에 만든 광고
   - 선호하는 스타일/업종
   - 수정 요청 패턴

3. **개인화**:
   - 사용자별 맞춤 응답
   - 세션별 상태 관리

**예시**: 상담 의도에서도 PostgreSQL 조회가 필요한 경우
```
사용자: "카페 광고는 어떻게 만들어?"
-> PostgreSQL 검색: 이 사용자가 이전에 카페 광고를 만든 적이 있는지 확인
-> 있다면: "이전에 만드신 카페 광고처럼 이런 방식으로 제작할 수 있습니다."
-> 없다면: "일반적인 카페 광고 제작 가이드를 안내해드리겠습니다."
```

---

## 4. 검색 전략

### 4.1 PostgreSQL 검색 (대화/생성 히스토리)

#### 4.1.1 구현 방식

**현재 구현**: 단순 시간순 정렬 (최근 5턴)
```sql
SELECT content, role, created_at
FROM chat_history
WHERE session_id = '{session_id}'
ORDER BY created_at DESC
LIMIT 5;
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
    limit: Optional[int] = 10
) -> List[models.ChatHistory]:
    """세션별 최근 대화 히스토리 조회"""
    query = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.session_id == session_id)
        .order_by(models.ChatHistory.created_at.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    return query.all()

def get_generation_history_by_session(
    db: Session,
    session_id: str,
    limit: Optional[int] = 5
) -> List[models.GenerationHistory]:
    """세션별 광고 생성 이력 조회"""
    query = (
        db.query(models.GenerationHistory)
        .filter(models.GenerationHistory.session_id == session_id)
        .order_by(models.GenerationHistory.created_at.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    return query.all()
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

# 1. 의도 분석용 컨텍스트 (최근 5턴)
recent_conversations = conv_manager.get_recent_messages(db, session_id, limit=5)
generation_history = conv_manager.get_generation_history(db, session_id, limit=5)

# 2. 의도 분석
intent_result = orchestrator.analyze_intent(user_message, {
    "recent_conversations": recent_conversations,
    "generation_history": generation_history,
})

# 3. 분기
if intent_result["intent"] == "consulting":
    knowledge_results = consulting_knowledge_base.search(
        query=user_message,
        category="faq",
        limit=3
    )
    context = {
        "recent_conversations": recent_conversations,
        "generation_history": generation_history,
        "knowledge_base": knowledge_results,
    }
    assistant_message = orchestrator.generate_consulting_response(user_message, context)
else:
    full_conversations = conv_manager.get_full_messages(db, session_id)
    full_generations = conv_manager.get_full_generation_history(db, session_id)
    refined_input = orchestrator.refine_generation_input(
        intent=intent_result["intent"],
        generation_type=intent_result.get("generation_type"),
        target_generation_id=intent_result.get("target_generation_id"),
        chat_history=full_conversations,
        generation_history=full_generations,
    )
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
1. **텍스트 생성**: `text_generation/text_generator.py`
   - `generate_ad_copy()`: 광고 문구 생성
2. **이미지 생성**: `image_generation/generator.py`
   - `generate_and_save_image()`:  프롬프트 + Stable Diffusion 기반 이미지 생성

### 5.2 LLM 호출과 생성 파이프라인 관계

현재 구조는 **LLM 호출과 생성 파이프라인을 분리**합니다.

```
LLM 호출 = 의도 분석 + 맥락 정제(또는 상담 응답)
생성 파이프라인 = 광고 콘텐츠 생성 (텍스트/이미지 모델)
```

**올바른 플로우**:
```
사용자 질문
    ↓
[LLM 호출 1]: 의도 분석 (최근 대화 3~5턴)
    ↓
┌───────────────────────────────────────────┐
│ consulting → [LLM 상담 응답 생성]          │
└───────────────────────────────────────────┘
┌───────────────────────────────────────────┐
│ generation/modification →                 │
│ [LLM 맥락 정제(refine_generation_input)]  │
│     ↓                                     │
│ [생성 파이프라인 호출]                     │
└───────────────────────────────────────────┘
```

### 5.3 통합 코드 (services.py)

#### 5.3.1 현재 구현된 함수

```python
# services.py: _execute_generation_pipeline()

async def _execute_generation_pipeline(
    *,
    db: Session,
    input_text: str,
    generation_input: str,
    generation_type: str,  # "text" or "image"
    session_id: Optional[str],
    user_id: Optional[int],
    style: Optional[str],
    aspect_ratio: Optional[str],
    ingest: Optional[IngestResult] = None,
    image: Optional[UploadFile] = None,
):
    """
    광고 생성 파이프라인 실행

    단계:
    1. ingest_user_message(): 입력 수집/저장 (ingest가 없을 때만)
    2. generate_contents(): 광고 생성
       ├─ text_generation.text_generator.generate_ad_copy()
       └─ image_generation.generator.generate_and_save_image()
    3. persist_generation_result(): DB 저장
    4. SSE progress/done 이벤트 전송 (상위 스트리밍 핸들러)
    """
    pass  # 구현 생략 (코드 참조)
```

### 5.4 생성 파이프라인 상세 플로우

```
_run_generation_for_intent()
    ↓
_execute_generation_pipeline()
    ↓
┌─────────────────────────────────────────┐
│ Step 1: ingest_user_message()           │
│  - 세션 확보                              │
│  - 이미지 저장 (있는 경우)                   │
│  - ChatHistory 저장                      │
│  - SSE progress 이벤트 전송 (analyzing)  │
└─────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ Step 2: generate_contents()               │
│  ├─ text_generation                       │
│  │   └─ text_generator.generate_ad_copy() │
│  │       - 광고 문구 생성                     │
│  │       - 프롬프트 생성                      │
│  │  - SSE progress 이벤트 전송 (generating) │
│  │                                        │
│  └─ image_generation (if ad_type="image") │
│      └─ generator.generate_and_save_image()│
│          - Stable Diffusion 이미지 생성      │
│          - ControlNet (참고 이미지 사용)      │
└───────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Step 3: persist_generation_result()     │
│  - ChatHistory 저장 (assistant 응답)      │
│  - GenerationHistory 저장                │
│  - ImageMatching 저장 (이미지 메타데이터)    │
│  - SSE done 이벤트 전송                   │
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

## 7. API 엔드포인트 설계

### 7.1 챗봇 메시지

#### POST /api/chat/message/stream

**요청**:
```http
POST /api/chat/message/stream HTTP/1.1
Content-Type: multipart/form-data
Authorization: Bearer {jwt_token}

message: "카페 광고 이미지를 만들고 싶어요"
session_id: "sess_abc123"  (optional)
image: [binary]  (optional)
```

**응답 (SSE)**:
```
data: {"type":"meta","session_id":"sess_abc123","intent":"generation"}

data: {"type":"progress","stage":"analyzing","message":"요청을 정리하고 있습니다."}

data: {"type":"progress","stage":"generating","message":"광고를 생성하고 있습니다."}

data: {"type":"done","assistant_message":"광고가 생성되었습니다.","output":{"content_type":"image","output_text":"따뜻한 커피 한 잔으로 시작하는 하루","image":"abc123def456.png"}}
```

### 7.2 대화 히스토리 조회

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
│  ┌─────────────┐                                            │
│  │ /chat/*     │                                            │
│  └─────────────┘                                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│                      Service Layer                         │
│                    (services.py)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ handle_chat_message_stream()                         │  │
│  │  - 챗봇 대화 처리                                       │  │
│  │  - 의도 분석 + 분기 처리                                │  │
│  │  - generation/modification은 스트리밍으로 결과 전송       │  │
│  │  - consulting은 LLM 응답 생성                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ _execute_generation_pipeline()                        │  │
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
│  │                      │  │ ├─ text_generation/     │    │
│  │                      │  │ │  └─ text_generator.py │    │
│  │ - RAGChatbot         │  │ └─ image_generation/    │    │
│  │ - ConversationManager│  │    └─ generator.py      │    │
│  │ - LLMOrchestrator    │  └─────────────────────────┘    │
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
│  │  - save_generation_history()                         │  │
│  │  - save_image_from_hash()                            │  │
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
│  │ - 키워드 검색       │         │ - Pinecone?          │   │
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
│ POST /api/chat/message/stream          │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ services.handle_chat_message_stream()  │
│  ├─ 세션 확보                           │
│  ├─ 이미지 처리                          │
│  └─ chatbot.process_message() 호출      │
└────────┬───────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────┐
│ chatbot.RAGChatbot.process_message()  │
│  ├─ 의도 분석 (최근 3~5턴)              │
│  ├─ consulting → 상담 응답 생성         │
│  └─ generation/modification →          │
│     파이프라인 실행 + 결과 스트리밍      │
└────────┬──────────────────────────────┘
         │
         ├── consulting 응답 스트리밍
         │
         └── generation/modification
             ▼
┌───────────────────────────────────────┐
│ _run_generation_for_intent()          │
│  ├─ refine_generation_input() (LLM)   │
│  ├─ generate_contents()               │
│  ├─ persist_generation_result()       │
│  └─ SSE done 이벤트 전송               │
└───────────────────────────────────────┘
```

### 8.3 파일 구조

```
main.py                         # FastAPI 엔드포인트 (스트리밍)
src/backend/
├── __init__.py
├── models.py                    # SQLAlchemy 모델 정의
├── schemas.py                   # Pydantic 스키마
├── process_db.py                # DB 접근 함수
├── services.py                  # 비즈니스 로직
├── chatbot.py                   # ✅ 구현됨
│   ├── RAGChatbot               # RAG 챗봇 메인
│   ├── ConversationManager      # 대화 관리자
│   └── LLMOrchestrator          # LLM 호출 관리

src/generation/
├── text_generation/
│   └── text_generator.py        # ✅ 구현됨
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

### 9.1 RAG 사이클 (현재 구현)

```
[사용자 질문]
    ↓
[의도 분석 (최근 3~5턴)]
    ↓
┌───────────────────────────────────────────┐
│ consulting → 최근 대화 + 지식베이스       │
│              → 상담 응답 LLM              │
└───────────────────────────────────────────┘
┌───────────────────────────────────────────┐
│ generation/modification → 전체 히스토리   │
│              → refine_generation_input LLM│
│              → 생성 파이프라인 호출       │
└───────────────────────────────────────────┘
```

### 9.2 VectorDB 이중 전략 (현재)

| 구분 | PostgreSQL | 정적 파일 VectorDB |
|-----|-----------|------------------|
| **담당** | 백엔드 로직 (나) | 상담 챗봇 팀원 |
| **검색 대상** | 대화/생성 히스토리 | FAQ, 가이드 문서 |
| **용도** | 맥락 유지, 개인화 | 상담 응답, 지식 제공 |
| **모든 의도에 필요** | ✅ YES | consulting 중심 |

### 9.3 LLM 호출 vs 생성 파이프라인

```
LLM 호출 (GPT-4o-mini):
  - 목적: 의도 분석, refined_input 정제, 상담 응답 생성
  - 입력: 사용자 메시지 + 컨텍스트 (히스토리/KB)
  - 출력: intent/generation_type/target_generation_id
          또는 refined_input
          또는 assistant_message

생성 파이프라인 (TextGenerator + Stable Diffusion):
  - 목적: 광고 콘텐츠 생성
  - 입력: refined_input (+ 업로드 이미지)
  - 출력: 광고 문구 + 이미지
```

### 9.4 의도별 처리 흐름

#### 생성 (Generation)
```
의도 분석 → 전체 히스토리 조회 → refine_generation_input
         → generate_contents → 결과 저장
```

#### 수정 (Modification)
```
의도 분석 → target_generation_id 선택
         → 전체 히스토리 조회 → refine_generation_input
         → handle_chat_revise (수정 재생성)
```

#### 상담 (Consulting)
```
의도 분석 → 유사 대화 + 지식베이스 검색
         → generate_consulting_response
         → 응답 반환 (생성 없음)
```

---

## 10. 다음 단계

### 10.1 개선 우선순위 (현 상태 기준)
1. **정적 파일 VectorDB 품질 개선** (검색 정확도/컨텍스트 정제)
2. **수정 요청 파서 고도화** (제약 조건/수정 누적 처리)
3. **테스트/관측 강화** (단위/통합, 응답 품질 로깅)

### 10.2 팀원 협업 포인트

**상담 챗봇 팀원**:
- 정적 파일 VectorDB 구축
- FAQ 문서 관리
- 검색 API 구현

**백엔드 로직 (나)**:
- consulting 컨텍스트 통합 로직 유지/개선
- refine_generation_input 프롬프트 관리
- 생성 파이프라인/로그/테스트 정비

**협업 인터페이스**:
```python
# consulting_knowledge_base.py (팀원 구현)

class ConsultingKnowledgeBase:
    def search(self, query: str, category: str, limit: int) -> List[dict]:
        """정적 문서 검색"""
        pass

# chatbot.py (백엔드)
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
**마지막 업데이트**: 2026-01-25
