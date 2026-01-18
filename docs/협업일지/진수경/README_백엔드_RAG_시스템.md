# 백엔드 RAG 시스템 문서 요약

**작성일**: 2026-01-15
**담당**: 백엔드 로직 (진수경)
**목적**: RAG 기반 챗봇 시스템 아키텍처 및 리팩토링 계획 종합 가이드
**버전**: 1.1 (현재 구현 반영)

---

## 문서 구조

이 폴더에는 RAG (Retrieval-Augmented Generation) 기반 광고 생성 챗봇 시스템의 전체 아키텍처와 구현 계획이 담겨 있습니다.

### 주요 문서

1. **[RAG_파이프라인_아키텍처_설계.md](./RAG_파이프라인_아키텍처_설계.md)** (55KB)
   - RAG 시스템의 전체 아키텍처 설계
   - 의도 분석, VectorDB 검색 전략, 생성 파이프라인 통합
   - 사용자가 이해한 구조를 기반으로 작성

2. **[백엔드_리팩토링_계획서.md](./백엔드_리팩토링_계획서.md)** (55KB)
   - 현재 코드 분석 및 문제점 파악
   - 단계별 리팩토링 계획 (6단계, 12-18일 소요)
   - 구체적인 코드 예시 및 구현 가이드

---

## 핵심 개념 요약

### RAG 파이프라인 플로우

```
[사용자 질문]
    ↓
[의도 분석 (최근 3~5턴)]
    ↓
┌───────────────────────────────────────────┐
│ consulting → 유사 대화 + 지식베이스       │
│              → 상담 응답 LLM              │
└───────────────────────────────────────────┘
┌───────────────────────────────────────────┐
│ generation/modification → 전체 히스토리   │
│              → refine_generation_input LLM│
│              → 생성 파이프라인 호출       │
└───────────────────────────────────────────┘
```

**핵심**: 생성/수정 intent는 **refine_generation_input(LLM)**으로 입력을 정제한 뒤 생성 파이프라인으로 넘김

### 이중 VectorDB 전략

| VectorDB | 담당 | 검색 대상 | 모든 의도에 필요 |
|---------|------|----------|----------------|
| **PostgreSQL** | 백엔드 로직 (나) | 대화/생성 히스토리 | ✅ YES |
| **정적 파일 VectorDB** | 상담 챗봇 팀원 | FAQ, 가이드 문서 | 의도별 차등 |

**중요**: PostgreSQL 검색은 모든 의도(생성/수정/상담)에서 필요합니다. 대화 맥락을 유지하고 개인화된 응답을 제공하기 위함입니다.

### 의도별 처리

#### 생성 (Generation)
- PostgreSQL: 대화/생성 히스토리 → 맥락 파악
- **refine_generation_input(LLM)**으로 입력 정제
- 생성 파이프라인 호출 → generate_contents 실행
- 결과: 광고 결과물 반환

#### 수정 (Modification)
- PostgreSQL: 최근 생성 이력 → 수정할 대상 찾기
- **refine_generation_input(LLM)**으로 요청 정제
- handle_chat_revise → 생성 파이프라인 재호출
- 결과: 수정된 광고 결과물 반환

#### 상담 (Consulting)
- PostgreSQL: 대화 히스토리 → 맥락 유지
- 정적 파일 VectorDB: FAQ, 가이드 → 지식 제공 (주요)
- **백엔드 RAG 챗봇에서 LLM 호출** (상담 응답 생성)
- 결과: 상담 응답 (광고 생성 없음)

---

## 현재 상태 분석

### 구현 완료 (✅)

1. **DB 모델** (`models.py`)
   - User, ChatSession, ChatHistory, GenerationHistory, ImageMatching

2. **광고 생성 파이프라인** (`services.py`)
   - `ingest_user_message()`: 입력 수집/저장
   - `generate_contents()`: 텍스트/이미지 광고 생성
   - `persist_generation_result()`: DB 저장
   - `_execute_generation_pipeline()`: 파이프라인 실행

3. **RAG 챗봇 핵심 로직** (`chatbot.py`)
   - RAGChatbot / ConversationManager / LLMOrchestrator 구현
   - analyze_intent + consulting 응답 생성
   - refine_generation_input 추가

4. **수정 플로우** (`services.py`)
   - `handle_chat_revise()` 구현

### 미구현/스텁 (❌/⚠️)

1. **정적 파일 VectorDB 고도화**
   - 지식베이스 구축/정확도 개선 (팀원 협업)

---

## 개선 계획 요약 (현 상태)

### 완료됨
- RAGChatbot / ConversationManager / LLMOrchestrator 구현
- refine_generation_input 도입 (generation/modification 입력 정제)
- handle_chat_revise 구현
- `/api/chat/message/stream` 흐름 정리

### 진행/예정
- 정적 파일 VectorDB 검색 품질 개선
- 테스트/관측/로깅 강화

---

## 협업 포인트

### 상담 챗봇 팀원

**협업 내용**:
- 정적 파일 VectorDB 구축 (ChromaDB, Pinecone 등)
- FAQ 문서 관리
- 검색 API 구현

**인터페이스**:
```python
class ConsultingKnowledgeBase:
    def search(self, query: str, category: str, limit: int) -> List[dict]:
        """정적 문서 검색"""
        pass
```

**협업 시점**: 지식베이스 구축 일정과 병행

### 프론트엔드 팀

**협업 내용**:
- 새 API 엔드포인트 통합 (`/chat/*`)
- SSE progress 이벤트 처리

**협업 시점**: API 변경 시점마다 공유

---

## 주요 파일 위치

```
main.py                  # FastAPI 엔드포인트 (스트리밍)
src/backend/
├── models.py               # ✅ DB 모델 (확장 필요)
├── process_db.py           # ✅ DB CRUD 함수 (확장 필요)
├── services.py             # ✅ 비즈니스 로직
├── chatbot.py              # ✅ RAG 챗봇 구현
│   ├── RAGChatbot          # Intent 분석 + 분기 처리
│   ├── ConversationManager # PostgreSQL 대화 관리
│   └── LLMOrchestrator     # Intent 분석 + 상담/정제

src/generation/
├── text_generation/
│   └── text_generator.py   # ✅ 텍스트 광고 생성
└── image_generation/
    └── generator.py        # ✅ 이미지 광고 생성
```

---

## API 엔드포인트 요약

### 현재 구현됨 (✅)

```
POST /api/auth/signup       # 회원가입
POST /api/auth/login        # 로그인
POST /api/chat/message/stream # 챗봇 메시지 스트리밍
POST /api/chat/session      # 세션 발급
GET  /api/chat/history/{sid}   # 대화 히스토리 조회
GET  /api/chat/generation/{sid}# 생성 이력 조회
```

**참고**: `/api/chat/message`는 비스트리밍을 제거했고, `/api/chat/message/stream`만 지원합니다.

---

## 다음 액션 아이템

### 우선순위 높은 개선
1. **정적 파일 VectorDB 품질 개선**
   - 검색 정확도/컨텍스트 정제 개선
2. **수정 파서 고도화**
   - 누적 수정 제약/번호 참조 케이스 보강
3. **테스트/관측 강화**
   - 단위/통합 테스트, 응답 품질 로그

---

## 참고 문서


### 프로젝트 전체 문서

- `docs/doc/시스템_아키텍처_설계서.md`: 전체 시스템 아키텍처 (있는 경우)

---

## 버전 히스토리

| 날짜 | 버전 | 변경 사항 |
|-----|------|----------|
| 2026-01-15 | 1.0 | 초기 문서 작성 (RAG 아키텍처 설계 + 리팩토링 계획) |
