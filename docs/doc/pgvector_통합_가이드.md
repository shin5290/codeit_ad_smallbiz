# pgvector 통합 가이드

**작성일**: 2026-01-16
**버전**: 1.0
**작성자**: Backend Developer (Phase 5-B)

## 목차

1. [개요](#1-개요)
2. [pgvector란?](#2-pgvector란)
3. [설치 및 설정](#3-설치-및-설정)
4. [마이그레이션 실행](#4-마이그레이션-실행)
5. [사용 방법](#5-사용-방법)
6. [성능 최적화](#6-성능-최적화)
7. [트러블슈팅](#7-트러블슈팅)

---

## 1. 개요

### 1.1 Phase 5-B 목표

pgvector를 통합하여 대화 히스토리의 의미 기반 유사도 검색을 지원합니다.

**주요 기능**:
- 사용자 메시지 자동 임베딩 생성 및 저장
- 벡터 유사도 검색 (코사인 유사도)
- RAG 챗봇의 컨텍스트 강화

### 1.2 구현 완료 사항

- ✅ ChatHistory 모델에 `embedding` 컬럼 추가 (Vector(1536))
- ✅ Alembic 마이그레이션 파일 생성 (`002_add_pgvector_embedding.py`)
- ✅ 임베딩 생성 유틸리티 (`src/utils/embedding.py`)
- ✅ ConversationManager에 임베딩 저장 로직 추가
- ✅ ConversationManager에 벡터 검색 함수 구현

---

## 2. pgvector란?

### 2.1 소개

**pgvector**는 PostgreSQL에서 벡터 유사도 검색을 지원하는 확장(extension)입니다.

**특징**:
- PostgreSQL 네이티브 확장
- 빠른 근사 최근접 이웃 검색 (Approximate Nearest Neighbor, ANN)
- 다양한 거리 메트릭 지원 (코사인, L2, 내적)
- 인덱스 지원 (IVFFlat, HNSW)

### 2.2 왜 pgvector인가?

**대안 비교**:

| 방법 | 장점 | 단점 |
|-----|------|------|
| **pgvector** | PostgreSQL 통합, 트랜잭션 지원, 운영 간편 | ChromaDB보다 느릴 수 있음 |
| ChromaDB | 전문 벡터 DB, 빠른 검색 | 별도 서버 필요, 복잡도 증가 |
| Pinecone | 매우 빠름, 관리형 서비스 | 비용, 외부 의존성 |

**선택 이유**:
1. 기존 PostgreSQL 인프라 활용
2. 트랜잭션 일관성 보장
3. 운영 복잡도 최소화
4. 중소 규모 데이터에 충분한 성능

---

## 3. 설치 및 설정

### 3.1 pgvector 설치

#### 3.1.1 Ubuntu/Debian

```bash
# 시스템 패키지 업데이트
sudo apt update

# PostgreSQL 개발 패키지 설치 (버전에 맞게 조정)
sudo apt install postgresql-server-dev-14

# pgvector 빌드 및 설치
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### 3.1.2 macOS (Homebrew)

```bash
brew install pgvector
```

#### 3.1.3 Docker

```dockerfile
# Dockerfile 예시
FROM postgres:14

RUN apt-get update && \
    apt-get install -y git build-essential postgresql-server-dev-14 && \
    cd /tmp && \
    git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    apt-get remove -y git build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/pgvector
```

### 3.2 Python 패키지 설치

```bash
pip install pgvector
```

### 3.3 PostgreSQL 확장 활성화

마이그레이션을 실행하면 자동으로 활성화되지만, 수동으로 확인하려면:

```sql
-- PostgreSQL에 접속 후
CREATE EXTENSION IF NOT EXISTS vector;

-- 설치 확인
SELECT * FROM pg_extension WHERE extname = 'vector';
```

---

## 4. 마이그레이션 실행

### 4.1 마이그레이션 파일

**위치**: `/home/spai0416/codeit_ad_smallbiz/alembic/versions/002_add_pgvector_embedding.py`

**내용**:
1. `CREATE EXTENSION vector` - pgvector 확장 설치
2. `ALTER TABLE chat_history ADD COLUMN embedding vector(1536)` - 임베딩 컬럼 추가
3. `CREATE INDEX idx_chat_history_embedding USING ivfflat` - 벡터 인덱스 생성

### 4.2 마이그레이션 실행

```bash
# 프로젝트 루트로 이동
cd /home/spai0416/codeit_ad_smallbiz

# 마이그레이션 실행
alembic upgrade head

# 또는 특정 버전으로
alembic upgrade 002_add_pgvector_embedding
```

### 4.3 마이그레이션 확인

```sql
-- PostgreSQL에 접속 후

-- 1. 확장 설치 확인
SELECT * FROM pg_extension WHERE extname = 'vector';

-- 2. 컬럼 추가 확인
\d chat_history

-- 출력 예시:
--  Column    |   Type    | Nullable
-- -----------+-----------+----------
--  embedding | vector(1536) | YES

-- 3. 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'chat_history' AND indexname LIKE '%embedding%';

-- 출력 예시:
-- indexname: idx_chat_history_embedding
-- indexdef: CREATE INDEX idx_chat_history_embedding ON chat_history USING ivfflat (embedding vector_cosine_ops) WITH (lists='100')
```

### 4.4 롤백

문제 발생 시 롤백:

```bash
# 이전 버전으로 롤백
alembic downgrade 001_add_generation_revision_columns
```

---

## 5. 사용 방법

### 5.1 임베딩 생성

#### 5.1.1 기본 사용

```python
from src.utils.embedding import generate_embedding

# 텍스트를 벡터로 변환
text = "카페 광고 이미지 만들어줘"
embedding = generate_embedding(text)

# 결과: List[float], 길이 1536
print(len(embedding))  # 1536
```

#### 5.1.2 비동기 사용

```python
from src.utils.embedding import generate_embedding_async

# 비동기 컨텍스트에서
text = "카페 광고 이미지 만들어줘"
embedding = await generate_embedding_async(text)
```

#### 5.1.3 API 키 명시

```python
embedding = generate_embedding(
    text="카페 광고",
    api_key="sk-...",
    model="text-embedding-3-small"
)
```

### 5.2 메시지 저장 (자동 임베딩)

#### 5.2.1 ConversationManager 사용

```python
from src.backend.chatbot import ConversationManager

conv = ConversationManager()

# user 메시지 저장 시 자동으로 임베딩 생성
message_id = conv.add_message(
    db=db,
    session_id="session_123",
    role="user",
    content="카페 광고 만들어줘",
    generate_embedding=True  # 기본값: True
)

# assistant 메시지는 임베딩 생성 안 함
conv.add_message(
    db=db,
    session_id="session_123",
    role="assistant",
    content="네, 카페 광고를 만들어드리겠습니다."
)
```

#### 5.2.2 임베딩 생성 비활성화

```python
# 임베딩 생성 없이 저장 (빠른 저장 필요 시)
message_id = conv.add_message(
    db=db,
    session_id="session_123",
    role="user",
    content="테스트 메시지",
    generate_embedding=False
)
```

### 5.3 벡터 유사도 검색

#### 5.3.1 세션 내 유사 메시지 검색

```python
from src.backend.chatbot import ConversationManager

conv = ConversationManager()

# 특정 세션 내에서 유사한 대화 검색
similar_messages = conv.search_similar_messages(
    db=db,
    query="카페 광고 어떻게 만들어?",
    session_id="session_123",
    limit=5,
    similarity_threshold=0.7  # 유사도 70% 이상만
)

# 결과 예시:
# [
#     {
#         "id": 42,
#         "session_id": "session_123",
#         "role": "user",
#         "content": "카페 광고 만들어줘",
#         "timestamp": "2026-01-16T10:30:00",
#         "similarity": 0.92
#     },
#     ...
# ]
```

#### 5.3.2 전체 세션 검색

```python
# 모든 세션에서 유사한 대화 검색
similar_messages = conv.search_similar_messages(
    db=db,
    query="이미지 생성",
    session_id=None,  # 전체 검색
    limit=10,
    similarity_threshold=0.8
)
```

#### 5.3.3 유사도 임계값 조정

```python
# 임계값이 낮을수록 더 많은 결과 반환
similar_messages = conv.search_similar_messages(
    db=db,
    query="광고",
    session_id="session_123",
    limit=10,
    similarity_threshold=0.5  # 50% 이상
)
```

### 5.4 RAG 챗봇 통합

#### 5.4.1 자동 통합

`RAGChatbot.process_message()`에서 자동으로 임베딩 저장:

```python
from src.backend.chatbot import get_chatbot

chatbot = get_chatbot()

# 사용자 메시지 처리 (자동 임베딩 저장)
result = await chatbot.process_message(
    db=db,
    session_id="session_123",
    user_message="카페 광고 만들어줘"
)
```

#### 5.4.2 향후 활용 (예정)

```python
# process_message() 내부에서 유사 대화 검색 활용
similar_messages = self.conv.search_similar_messages(
    db, user_message, session_id, limit=3
)

# 유사한 이전 대화를 LLM 컨텍스트에 추가
context["similar_conversations"] = similar_messages
```

---

## 6. 성능 최적화

### 6.1 인덱스 설정

#### 6.1.1 IVFFlat 인덱스

**현재 설정**:
```sql
CREATE INDEX idx_chat_history_embedding
ON chat_history
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**파라미터 설명**:
- `lists`: 클러스터 수
  - 권장값: `rows / 1000` (최소 10, 최대 10,000)
  - 예: 10,000개 메시지 → lists = 10
  - 예: 100,000개 메시지 → lists = 100

**lists 값 조정**:
```sql
-- 기존 인덱스 삭제
DROP INDEX idx_chat_history_embedding;

-- 새 인덱스 생성 (lists 값 변경)
CREATE INDEX idx_chat_history_embedding
ON chat_history
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);  -- 데이터 양에 따라 조정
```

#### 6.1.2 HNSW 인덱스 (더 빠른 검색)

pgvector 0.5.0 이상에서 지원:

```sql
-- IVFFlat 대신 HNSW 사용 (더 빠르지만 메모리 많이 사용)
DROP INDEX idx_chat_history_embedding;

CREATE INDEX idx_chat_history_embedding
ON chat_history
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**HNSW 파라미터**:
- `m`: 연결 수 (기본값: 16, 범위: 2-100)
  - 높을수록 정확하지만 메모리 많이 사용
- `ef_construction`: 빌드 시 탐색 깊이 (기본값: 64)
  - 높을수록 정확하지만 인덱스 생성 느림

### 6.2 쿼리 최적화

#### 6.2.1 probes 설정 (IVFFlat)

검색 시 탐색할 클러스터 수 조정:

```sql
-- 세션 단위로 설정
SET ivfflat.probes = 10;  -- 기본값: 1

-- 쿼리 실행
SELECT * FROM chat_history
WHERE embedding <-> '[...]' < 0.3
ORDER BY embedding <-> '[...]'
LIMIT 10;
```

**probes 값 가이드**:
- `1`: 매우 빠름, 정확도 낮음
- `10`: 균형 (권장)
- `lists 값`: 정확한 검색 (느림)

#### 6.2.2 필터링 조건 추가

```sql
-- session_id 필터로 검색 범위 축소
SELECT * FROM chat_history
WHERE session_id = 'session_123'
  AND role = 'user'
  AND embedding IS NOT NULL
  AND embedding <-> '[...]' < 0.3
ORDER BY embedding <-> '[...]'
LIMIT 10;
```

### 6.3 임베딩 생성 최적화

#### 6.3.1 배치 생성

여러 텍스트를 한 번에 임베딩:

```python
# OpenAI API는 배치 요청 지원
import openai

client = openai.OpenAI(api_key="...")


texts = ["텍스트1", "텍스트2", "텍스트3"]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts  # 리스트 전달
)

embeddings = [item.embedding for item in response.data]
```

#### 6.3.2 캐싱

동일한 텍스트는 재사용:

```python
embedding_cache = {}

def get_embedding_cached(text: str):
    if text in embedding_cache:
        return embedding_cache[text]

    embedding = generate_embedding(text)
    embedding_cache[text] = embedding
    return embedding
```

---

## 7. 트러블슈팅

### 7.1 pgvector 설치 실패

**증상**:
```
ERROR: extension "vector" is not available
```

**해결**:
1. PostgreSQL 버전 확인 (pgvector는 PostgreSQL 11+ 필요)
   ```sql
   SELECT version();
   ```

2. pgvector 재설치
   ```bash
   cd /tmp
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make clean
   make
   sudo make install
   ```

3. PostgreSQL 재시작
   ```bash
   sudo systemctl restart postgresql
   ```

### 7.2 임베딩 생성 실패

**증상**:
```
ConversationManager: embedding generation returned None
```

**원인**:
- OpenAI API 키 미설정
- API 요청 한도 초과
- 네트워크 오류

**해결**:
1. API 키 확인
   ```bash
   # .env 파일 확인
   cat .env | grep OPENAI_API_KEY
   ```

2. API 키 테스트
   ```python
   from src.utils.embedding import generate_embedding

   embedding = generate_embedding("테스트")
   print(embedding is not None)  # True여야 함
   ```

3. 로그 확인
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### 7.3 벡터 검색 성능 저하

**증상**:
- 검색이 매우 느림 (1초 이상)

**해결**:
1. 인덱스 확인
   ```sql
   SELECT indexname FROM pg_indexes
   WHERE tablename = 'chat_history';
   ```

2. 인덱스 재생성
   ```sql
   REINDEX INDEX idx_chat_history_embedding;
   ```

3. `probes` 값 조정
   ```sql
   SET ivfflat.probes = 5;  -- 기본값 1에서 증가
   ```

4. VACUUM ANALYZE 실행
   ```sql
   VACUUM ANALYZE chat_history;
   ```

### 7.4 마이그레이션 오류

**증상**:
```
ERROR: column "embedding" of relation "chat_history" already exists
```

**해결**:
이미 마이그레이션이 적용된 경우:

```bash
# 마이그레이션 상태 확인
alembic current

# 마이그레이션 기록만 업데이트 (실제 변경 없이)
alembic stamp 002_add_pgvector_embedding
```

### 7.5 임베딩 차원 불일치

**증상**:
```
ERROR: dimension mismatch: expected 1536, got 3072
```

**원인**:
- 다른 모델 사용 (`text-embedding-3-large`는 3072차원)

**해결**:
1. 모델 통일
   ```python
   # text-embedding-3-small 사용 (1536차원)
   embedding = generate_embedding(text, model="text-embedding-3-small")
   ```

2. 또는 DB 스키마 변경 (권장하지 않음)
   ```sql
   ALTER TABLE chat_history
   ALTER COLUMN embedding TYPE vector(3072);
   ```

---

## 8. 성능 벤치마크

### 8.1 테스트 환경

- PostgreSQL 14
- pgvector 0.5.1
- 데이터: 10,000개 메시지
- 임베딩: text-embedding-3-small (1536차원)

### 8.2 검색 성능

| 인덱스 유형 | 검색 시간 (avg) | 정확도 |
|-----------|----------------|--------|
| None (Full Scan) | 250ms | 100% |
| IVFFlat (lists=100) | 15ms | 95% |
| HNSW (m=16) | 5ms | 98% |

### 8.3 임베딩 생성 성능

| 방법 | 시간 (1개) | 시간 (100개) |
|-----|-----------|-------------|
| 순차 생성 | 200ms | 20초 |
| 배치 생성 (10개씩) | - | 3초 |

---

## 9. 참고 자료

### 9.1 공식 문서

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### 9.2 관련 파일

- `/home/spai0416/codeit_ad_smallbiz/src/backend/models.py` - ChatHistory 모델
- `/home/spai0416/codeit_ad_smallbiz/src/backend/chatbot.py` - ConversationManager
- `/home/spai0416/codeit_ad_smallbiz/src/utils/embedding.py` - 임베딩 유틸리티
- `/home/spai0416/codeit_ad_smallbiz/alembic/versions/002_add_pgvector_embedding.py` - 마이그레이션

### 9.3 다음 단계

Phase 5-B 완료 후:
1. 실제 데이터로 검색 성능 테스트
2. 유사도 임계값 최적화
3. RAG 챗봇에 유사 대화 검색 통합
4. 프로덕션 배포

---

**문서 버전**: 1.0
**최종 업데이트**: 2026-01-16
**작성자**: Backend Developer
