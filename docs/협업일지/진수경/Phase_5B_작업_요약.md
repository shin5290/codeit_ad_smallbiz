# Phase 5-B: pgvector 통합 작업 요약

**작업일**: 2026-01-16
**담당**: Backend Developer
**상태**: ✅ 완료

---

## 작업 개요

백엔드 리팩토링 계획서의 Phase 5-B에 따라 pgvector를 통합하여 대화 히스토리의 의미 기반 유사도 검색 기능을 구현했습니다.

---

## 구현 내용

### 1. 데이터베이스 스키마 확장

#### 파일: `/home/spai0416/codeit_ad_smallbiz/src/backend/models.py`

**변경 사항**:
- `pgvector.sqlalchemy.Vector` import 추가
- `ChatHistory` 모델에 `embedding` 컬럼 추가
  - 타입: `Vector(1536)` (OpenAI text-embedding-3-small 모델 차원)
  - Nullable: True (기존 데이터 호환성)

```python
# pgvector: 임베딩 벡터 (OpenAI text-embedding-3-small: 1536 차원)
embedding = Column(Vector(1536), nullable=True)
```

### 2. Alembic 마이그레이션 파일 생성

#### 파일: `/home/spai0416/codeit_ad_smallbiz/alembic/versions/002_add_pgvector_embedding.py`

**구현 내용**:
1. PostgreSQL pgvector 확장 설치 (`CREATE EXTENSION vector`)
2. `chat_history` 테이블에 `embedding` 컬럼 추가
3. IVFFlat 인덱스 생성 (빠른 유사도 검색용)
   - 인덱스 타입: `ivfflat`
   - 연산자: `vector_cosine_ops` (코사인 유사도)
   - 파라미터: `lists = 100` (클러스터 수)

**마이그레이션 실행**:
```bash
alembic upgrade head
```

### 3. 임베딩 생성 유틸리티

#### 파일: `/home/spai0416/codeit_ad_smallbiz/src/utils/embedding.py` (신규)

**주요 함수**:

1. **`generate_embedding(text, api_key, model)`**
   - 텍스트를 1536차원 벡터로 변환 (동기)
   - OpenAI Embedding API 사용
   - 에러 핸들링 (인증, Rate Limit, API 오류)

2. **`generate_embedding_async(text, api_key, model)`**
   - 비동기 버전
   - FastAPI 등 비동기 환경에서 사용

3. **`calculate_cosine_similarity(vec1, vec2)`**
   - 두 벡터의 코사인 유사도 계산
   - NumPy 기반 구현

**사용 예시**:
```python
from src.utils.embedding import generate_embedding

embedding = generate_embedding("카페 광고 만들어줘")
print(len(embedding))  # 1536
```

### 4. ConversationManager 업데이트

#### 파일: `/home/spai0416/codeit_ad_smallbiz/src/backend/chatbot.py`

**4.1. `add_message()` 함수 확장**

**변경 사항**:
- `generate_embedding` 파라미터 추가 (기본값: True)
- user 메시지 저장 시 자동으로 임베딩 생성 및 저장
- assistant 메시지는 임베딩 생성하지 않음 (성능 최적화)
- 에러 발생 시에도 메시지 저장 (임베딩은 optional)

**동작**:
1. user 메시지 + `generate_embedding=True` → 임베딩 생성 및 저장
2. assistant 메시지 → 임베딩 생성하지 않음
3. 임베딩 생성 실패 → 경고 로그, 메시지는 저장

**4.2. `search_similar_messages()` 함수 구현**

**기능**:
- pgvector 기반 코사인 유사도 검색
- 쿼리 텍스트와 유사한 대화 검색
- 세션별 필터링 지원

**파라미터**:
- `query`: 검색 쿼리 텍스트
- `session_id`: 세션 ID (None이면 전체 검색)
- `limit`: 최대 결과 수
- `similarity_threshold`: 유사도 임계값 (0.0~1.0)

**알고리즘**:
1. 쿼리 텍스트를 임베딩 벡터로 변환
2. pgvector SQL 쿼리 실행 (`<->` 코사인 거리 연산자)
3. 유사도 = 1 - 코사인 거리
4. 유사도 내림차순 정렬 후 반환

**SQL 쿼리**:
```sql
SELECT
    id, session_id, role, content, created_at,
    1 - (embedding <-> :query_embedding) AS similarity
FROM chat_history
WHERE
    role = 'user'
    AND embedding IS NOT NULL
    AND (:session_id IS NULL OR session_id = :session_id)
    AND 1 - (embedding <-> :query_embedding) >= :similarity_threshold
ORDER BY similarity DESC
LIMIT :limit
```

**폴백 전략**:
- 임베딩 생성 실패 시 → 최근 메시지 반환
- 벡터 검색 실패 시 → 최근 메시지 반환

---

## 생성된 파일

### 코드 파일
1. `/home/spai0416/codeit_ad_smallbiz/src/utils/embedding.py` (신규)
2. `/home/spai0416/codeit_ad_smallbiz/alembic/versions/002_add_pgvector_embedding.py` (신규)

### 문서 파일
1. `/home/spai0416/codeit_ad_smallbiz/docs/doc/pgvector_통합_가이드.md` (신규)
   - 상세 사용법, 성능 최적화, 트러블슈팅
2. `/home/spai0416/codeit_ad_smallbiz/docs/doc/Phase_5B_pgvector_설치_가이드.md` (신규)
   - 설치 및 설정 가이드
3. `/home/spai0416/codeit_ad_smallbiz/docs/doc/Phase_5B_작업_요약.md` (이 문서)

### 수정된 파일
1. `/home/spai0416/codeit_ad_smallbiz/src/backend/models.py`
   - ChatHistory에 embedding 컬럼 추가
2. `/home/spai0416/codeit_ad_smallbiz/src/backend/chatbot.py`
   - ConversationManager 업데이트
3. `/home/spai0416/codeit_ad_smallbiz/docs/협업일지/진수경/백엔드_리팩토링_계획서.md`
   - Phase 5-B 체크리스트 업데이트

---

## 설치 및 설정

### 1. Python 패키지 설치

```bash
pip install pgvector>=0.2.3
pip install numpy>=1.24.0
```

### 2. PostgreSQL pgvector 확장 설치

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install postgresql-server-dev-14
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
sudo systemctl restart postgresql
```

#### macOS
```bash
brew install pgvector
```

### 3. 마이그레이션 실행

```bash
cd /home/spai0416/codeit_ad_smallbiz
alembic upgrade head
```

### 4. 설치 확인

```sql
-- PostgreSQL에 접속 후
SELECT * FROM pg_extension WHERE extname = 'vector';
\d chat_history
```

---

## 사용 예시

### 1. 메시지 저장 (자동 임베딩)

```python
from src.backend.chatbot import ConversationManager

conv = ConversationManager()

# user 메시지 저장 시 자동 임베딩 생성
message_id = conv.add_message(
    db=db,
    session_id="session_123",
    role="user",
    content="카페 광고 만들어줘"
)
```

### 2. 유사 메시지 검색

```python
# 코사인 유사도 기반 검색
similar_messages = conv.search_similar_messages(
    db=db,
    query="광고 어떻게 만들어?",
    session_id="session_123",
    limit=5,
    similarity_threshold=0.7
)

# 결과
for msg in similar_messages:
    print(f"유사도: {msg['similarity']:.2f}, 내용: {msg['content']}")
```

### 3. 임베딩 직접 생성

```python
from src.utils.embedding import generate_embedding

embedding = generate_embedding("카페 광고 만들어줘")
print(len(embedding))  # 1536
```

---

## 성능 고려사항

### 1. 인덱스 최적화

**IVFFlat lists 파라미터**:
- 현재 설정: `lists = 100`
- 권장값: `rows / 1000`
- 데이터 증가 시 조정 필요

**조정 방법**:
```sql
DROP INDEX idx_chat_history_embedding;
CREATE INDEX idx_chat_history_embedding
ON chat_history
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);  -- 데이터 양에 따라
```

### 2. 검색 정확도 조정

**probes 값 조정**:
```sql
-- 세션별로 설정 (검색 전에)
SET ivfflat.probes = 10;  -- 기본값: 1
-- 값이 클수록 정확하지만 느림
```

### 3. 임베딩 생성 비용

- OpenAI API 비용 발생 (text-embedding-3-small: $0.00002/1K tokens)
- user 메시지만 임베딩 생성 (assistant는 생략)
- 필요 시 `generate_embedding=False`로 비활성화 가능

---

## 향후 개선 사항

### 1. RAG 챗봇 통합

현재는 기능 구현만 완료, 실제 RAG 파이프라인에 통합 필요:

```python
# chatbot.py: process_message() 내부에서 활용 (예정)
similar_messages = self.conv.search_similar_messages(
    db, user_message, session_id, limit=3
)

context["similar_conversations"] = similar_messages
# → LLM 프롬프트에 유사 대화 추가
```

### 2. 성능 모니터링

- 검색 속도 측정
- 임베딩 생성 성공률 추적
- 유사도 분포 분석

### 3. 하이브리드 검색

- pgvector (의미 검색) + PostgreSQL FTS (키워드 검색) 결합
- 더 정확한 검색 결과

### 4. HNSW 인덱스 고려

- IVFFlat보다 빠른 검색 (메모리 더 사용)
- pgvector 0.5.0+ 필요

```sql
CREATE INDEX idx_chat_history_embedding
ON chat_history
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 트러블슈팅

### 1. pgvector 확장 설치 실패

```
ERROR: extension "vector" is not available
```

**해결**:
- PostgreSQL 버전 확인 (11+ 필요)
- pgvector 재설치
- PostgreSQL 재시작

### 2. 임베딩 생성 실패

```
ConversationManager: embedding generation returned None
```

**해결**:
- `.env` 파일에 `OPENAI_API_KEY` 확인
- API 키 유효성 확인
- 네트워크 연결 확인

### 3. 검색 성능 저하

**해결**:
- 인덱스 확인 (`\d chat_history`)
- `VACUUM ANALYZE chat_history` 실행
- `probes` 값 조정

---

## 테스트 체크리스트

### 단위 테스트 (향후 작성 필요)

- [ ] `generate_embedding()` 함수 테스트
- [ ] `ConversationManager.add_message()` 임베딩 저장 테스트
- [ ] `ConversationManager.search_similar_messages()` 검색 테스트

### 통합 테스트 (향후 작성 필요)

- [ ] 메시지 저장 → 검색 플로우
- [ ] 임베딩 생성 실패 시 폴백 동작
- [ ] 세션별 검색 필터링

### 성능 테스트 (향후 실행 필요)

- [ ] 1,000개 메시지 검색 속도
- [ ] 10,000개 메시지 검색 속도
- [ ] 임베딩 생성 응답 시간

---

## 참고 문서

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- `/home/spai0416/codeit_ad_smallbiz/docs/doc/pgvector_통합_가이드.md`
- `/home/spai0416/codeit_ad_smallbiz/docs/협업일지/진수경/백엔드_리팩토링_계획서.md`

---

## 요약

Phase 5-B를 통해 다음을 달성했습니다:

✅ **pgvector 통합 완료**
- ChatHistory 테이블에 임베딩 저장 기능 추가
- 코사인 유사도 기반 의미 검색 구현
- IVFFlat 인덱스로 빠른 검색 지원

✅ **확장 가능한 아키텍처**
- 임베딩 생성 유틸리티 분리 (재사용 가능)
- 폴백 전략으로 안정성 확보
- 성능 튜닝 파라미터 노출

✅ **프로덕션 준비**
- Alembic 마이그레이션 제공
- 상세한 설치 및 사용 가이드
- 에러 핸들링 및 로깅

**다음 단계**: Phase 6 (테스트 및 배포) 또는 RAG 챗봇에 유사도 검색 통합

---

**작성자**: Backend Developer
**최종 업데이트**: 2026-01-16
**상태**: ✅ 완료
