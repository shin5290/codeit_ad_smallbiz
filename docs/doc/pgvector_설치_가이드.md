# Phase 5-B: pgvector 통합 설치 가이드

**작성일**: 2026-01-16
**버전**: 1.0

## 필수 Python 패키지 설치

Phase 5-B를 실행하기 위해 다음 패키지를 설치해야 합니다:

```bash
# pgvector Python 클라이언트
pip install pgvector>=0.2.3

# NumPy (코사인 유사도 계산용)
pip install numpy>=1.24.0
```

## PostgreSQL pgvector 확장 설치

### Ubuntu/Debian

```bash
# PostgreSQL 개발 패키지 설치
sudo apt update
sudo apt install postgresql-server-dev-14  # PostgreSQL 버전에 맞게 조정

# pgvector 빌드 및 설치
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# PostgreSQL 재시작
sudo systemctl restart postgresql
```

### macOS

```bash
brew install pgvector
```

### Docker

```dockerfile
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

## 마이그레이션 실행

```bash
# 프로젝트 루트로 이동
cd /home/spai0416/codeit_ad_smallbiz

# 마이그레이션 실행
alembic upgrade head
```

## 설치 확인

```sql
-- PostgreSQL에 접속 후
CREATE EXTENSION IF NOT EXISTS vector;

-- 확인
SELECT * FROM pg_extension WHERE extname = 'vector';

-- chat_history 테이블 확인
\d chat_history
```

## 주요 구현 파일

### 1. 모델 (models.py)

- ChatHistory 테이블에 `embedding vector(1536)` 컬럼 추가
- OpenAI text-embedding-3-small 모델 사용 (1536차원)

### 2. 임베딩 유틸리티 (src/utils/embedding.py)

- `generate_embedding()`: 동기 임베딩 생성
- `generate_embedding_async()`: 비동기 임베딩 생성
- `calculate_cosine_similarity()`: 코사인 유사도 계산

### 3. 대화 관리자 (src/backend/chatbot.py)

**ConversationManager 업데이트**:
- `add_message()`: 사용자 메시지 저장 시 자동 임베딩 생성
- `search_similar_messages()`: pgvector 기반 유사도 검색

### 4. 마이그레이션 (alembic/versions/002_add_pgvector_embedding.py)

- pgvector 확장 설치
- embedding 컬럼 추가
- IVFFlat 인덱스 생성 (빠른 유사도 검색)

## 사용 예시

### 임베딩 생성

```python
from src.utils.embedding import generate_embedding

embedding = generate_embedding("카페 광고 만들어줘")
print(len(embedding))  # 1536
```

### 메시지 저장 (자동 임베딩)

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

### 유사 메시지 검색

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
# [
#     {
#         "content": "카페 광고 만들어줘",
#         "similarity": 0.92,
#         ...
#     }
# ]
```

## 성능 최적화

### IVFFlat 인덱스 파라미터 조정

```sql
-- lists 값 조정 (데이터 크기에 따라)
-- 권장: rows / 1000
DROP INDEX idx_chat_history_embedding;
CREATE INDEX idx_chat_history_embedding
ON chat_history
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);  -- 데이터 양에 맞게 조정
```

### 검색 시 probes 조정

```python
# process_db.py 또는 검색 쿼리 전에
db.execute("SET ivfflat.probes = 10")  # 기본값: 1
```

## 트러블슈팅

### pgvector 확장 설치 실패

```
ERROR: extension "vector" is not available
```

**해결**:
1. PostgreSQL 버전 확인 (11+ 필요)
2. pgvector 재설치
3. PostgreSQL 재시작

### 임베딩 생성 실패

```
ConversationManager: embedding generation returned None
```

**해결**:
1. `.env` 파일에 `OPENAI_API_KEY` 설정 확인
2. API 키 유효성 확인
3. 네트워크 연결 확인

## 다음 단계

Phase 5-B 완료 후:
1. 실제 사용자 데이터로 검색 테스트
2. 유사도 임계값 최적화
3. RAG 챗봇에 유사 대화 검색 통합
4. 성능 모니터링 및 튜닝

## 참고 문서

- `/home/spai0416/codeit_ad_smallbiz/docs/doc/pgvector_통합_가이드.md` - 상세 사용 가이드
- `/home/spai0416/codeit_ad_smallbiz/docs/협업일지/진수경/백엔드_리팩토링_계획서.md` - 전체 리팩토링 계획

---

**작성자**: Backend Developer
**최종 업데이트**: 2026-01-16
