# RAG 기반 상담 챗봇 시스템

> **LangChain + Chroma + Agent + FastAPI(SSE)** 소상공인 마케팅 상담 챗봇 (백엔드 전용)

## 🛠️ 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| **Framework** | LangChain |
| **Vector DB** | Chroma |
| **Embedding** | intfloat/multilingual-e5-large |
| **Reranker** | BAAI/bge-reranker-v2-m3 (선택) |
| **LLM** | GPT-4o-mini / GPT-4o |
| **Agent** | LangChain Agent (ReAct) |
| **API** | FastAPI |

---

## 📊 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      사용자 질문                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 IntentRouter (LangChain)                     │
│         "트렌드" → Agent | 기타 → RAG                        │
└─────────────────────────────────────────────────────────────┘
            ↓                    ↓
    ┌──────────────┐    ┌──────────────┐
    │     RAG      │    │    Agent     │
    │   (Chroma)   │    │   (ReAct)    │
    │   + E5 임베딩 │    │  + Tools     │
    └──────────────┘    └──────────────┘
            ↓                    ↓
            └──────────────┬─────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Self-Refine (LangChain)                   │
│              초안 → 비평 → 개선 (점수 < 7점 시)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       최종 답변                               │
│              + method + sources + memory                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 모듈별 상세

### 1. `data/` - 데이터 파이프라인

**파이프라인 흐름:**
```
01_crawl_naver.py → raw/naver_places.json
      ↓
02_split_data.py → raw/{train,val,test}_places.json
      ↓
03_build_documents_v5.py → processed/documents_v5.jsonl
      ↓
06_build_vectorstore.py → data/vectorstore/chroma_db/
```

**Retrieval 평가 결과 (multilingual-e5-large):**
| 메트릭 | Vector Only | + Reranker |
|--------|-------------|------------|
| Recall@1 | 0.8533 | 0.8733 |
| Recall@3 | 0.9033 | 0.9333 |
| MRR | 0.8858 | 0.9060 |

**LLM-as-a-Judge 평가 결과 (프롬프트 엔지니어링):**
| 항목 | 최종 점수 | 설명 |
|------|-----------|------|
| Specificity | 8.50 / 10 | 숫자, 구체적 예시 포함 |
| Evidence | 9.00 / 10 | 출처, 사례 근거 명시 |
| Structure | 10.00 / 10 | 응답 구조 일관성 |
| Safety | 10.00 / 10 | 과장/허위 정보 없음 |

> 4회 iteration으로 베이스라인(Evidence 6.33) → 최종(9.00) 개선. 주요 변경: 항목별 출처 태깅, k=7 확장, 숫자 하한선 적용.

**임베딩 모델 VRAM 최적화:**

이미지 생성 모델(~20GB)과 함께 운영 시 VRAM 부족 문제가 발생하여 최적화 실험을 진행했습니다.

| 옵션 | VRAM | Recall@1 | Latency | 비고 |
|------|------|----------|---------|------|
| **GPU (FP32, 기본)** | 2.2GB | 0.8533 | ~0.3초 | 기준 |
| **GPU (FP16 양자화)** | 1.1GB | 0.8533 | ~0.3초 | 정확도 유지 |
| **GPU (INT8 양자화)** | ~0.6GB | 0.84 (추정) | ~0.3초 | 1-2% 정확도 감소 가능 |
| **OpenAI API** | 0GB | 0.79 | ~0.5초 | 정확도 6%p 감소, 벡터스토어 재구축 필요 |
| **CPU** | 0GB | 0.8533 | ~1-2초 | 정확도 유지, latency 증가 |

**환경 (GCP):**
- GPU: NVIDIA L4 (VRAM 23GB)
- 이미지 생성 모델: ~20GB
- 가용 VRAM: ~3GB (메모리 파편화로 실제 여유 더 적음)

**최종 결정: CPU 전환**

선택 이유:
1. **정확도 유지**: OpenAI API 전환 시 R@1이 0.8533 → 0.79로 6%p 하락. 100번 질문 중 6번 더 잘못된 문서 검색.
2. **VRAM 완전 해제**: FP16(1.1GB), INT8(0.6GB)도 메모리 파편화 환경에서 OOM 위험 존재. CPU는 VRAM 0GB로 완전 해결.
3. **허용 가능한 latency**: 임베딩 latency +1~2초 증가하나, LLM 응답 시간(3~5초)이 전체의 70% 차지하여 체감 영향 적음.
4. **작업량 최소**: 벡터스토어 재구축 없이 코드 1줄 수정으로 적용 가능.

```python
# rag/chain.py - E5Embeddings
self.model = SentenceTransformer(model_name, device="cpu")  # VRAM 0GB
```

**CPU 성능 최적화 (큐잉 + 마이크로배치):**

동시 사용자 증가에 대비하여 큐잉과 마이크로배치를 적용했습니다.

| 설정 | 값 | 이유 |
|------|-----|------|
| **threads** | 2 | 벤치마크 결과 최적 (4 이상은 오버헤드) |
| **batch_wait_ms** | 50ms | 요청 모으는 대기 시간 |
| **max_batch_size** | 8 | 배치당 최대 쿼리 수 |

**벤치마크 결과 (CPU, threads=2):**
| 시나리오 | batch_size | concurrency | p95 latency | per_sentence |
|----------|------------|-------------|-------------|--------------|
| 단일 요청 | 1 | 1 | 200ms | 200ms |
| 동시 5명 | 1 | 5 | 815ms | 815ms |
| 마이크로배치 | 4 | 1 | 342ms | **87ms** |
| 마이크로배치 | 8 | 1 | 470ms | **53ms** |

**설계 원칙:**
1. **동시 encode 금지**: Lock으로 CPU 경쟁 방지 (conc 높으면 모두 느려짐)
2. **큐잉**: 요청을 순차 처리하여 예측 가능한 latency 보장
3. **마이크로배치**: 50ms 동안 요청 모아서 배치 처리 → 문장당 처리 효율 향상
4. **p95 기준 설계**: 95% 사용자가 200ms 내 응답 받도록 보장

```python
# rag/chain.py - E5Embeddings 설정
E5Embeddings(
    device="cpu",           # VRAM 0GB
    batch_wait_ms=50,       # 50ms 대기 후 배치 처리
    max_batch_size=8,       # 최대 8개 묶어서 처리
    enable_micro_batch=True # 마이크로배치 활성화
)
```

---

### 2. `rag/` - RAG 시스템 (LangChain)

**핵심 구성:**
```python
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Chroma 벡터스토어 + E5 임베딩
vectorstore = Chroma(
    persist_directory="data/vectorstore/chroma_db",
    embedding_function=E5Embeddings(),
)

# Retriever (메타데이터 필터링 지원)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"industry": "cafe"}}
)

# RAG Chain
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
)
```

**E5 임베딩 prefix 규칙:**
- 문서: `"passage: " + text`
- 쿼리: `"query: " + text`

---

### 3. `agent/` - Agent 시스템 (LangChain Agent)

**Agent 구조:**
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

# Tools 정의
tools = [
    Tool(
        name="web_search",
        func=web_search,
        description="최신 트렌드, 뉴스 검색에 사용"
    ),
    Tool(
        name="rag_search",
        func=rag_search,
        description="소상공인 마케팅 사례 검색에 사용"
    ),
]

# ReAct Agent 생성
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

**Agent 사용 시점:**
- "요즘", "최근", "트렌드" 키워드 포함
- 실시간 정보 필요 시

---

### 4. `refine/` - Self-Refine (실험 완료)

**LangChain으로 Self-Refine 구현 (2단계 루프 테스트 완료, 최종 점수 7~9.2/10):**
```python
from langchain.chains import SequentialChain

# 1. 초안 생성
draft_chain = LLMChain(llm=llm, prompt=draft_prompt)

# 2. 자체 평가
critique_chain = LLMChain(llm=llm, prompt=critique_prompt)

# 3. 개선
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

# Sequential 실행
self_refine = SequentialChain(
    chains=[draft_chain, critique_chain, refine_chain],
    input_variables=["question", "context"],
    output_variables=["final_answer"],
)
```

**평가 기준 (10점 만점):**
- 구체성: 숫자, 예시 포함
- 근거: 출처, 사례 명시
- 정확성: 과장 표현 없음
- 완성도: 다음 단계 명확

---

### 5. 통합 라우팅 (백엔드)

```python
from chat_bot.agent.agent import SmallBizConsultant
from chat_bot.rag.prompts import UserContext

consultant = SmallBizConsultant()

result = consultant.consult(
    query="요즘 카페 SNS 트렌드 알려줘",
    user_context=UserContext(industry="cafe", location="강남"),
)
print(result["method"])  # agent 또는 rag
print(result["answer"])
```

---

## 🚀 Quick Start

### 1. 환경 설정
```bash
# 패키지 설치
pip install langchain langchain-openai langchain-community chromadb sentence-transformers

# 환경변수 설정
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="..."  # Agent 웹 검색용 (없으면 DuckDuckGo 폴백)
```

### 2. 벡터스토어 생성
```bash
cd chat_bot/data
python 06_build_vectorstore.py
```

### 3. RAG 테스트
```bash
python -m rag.chain
```

### 4. 간단 사용 예시
```python
from chat_bot.rag.chain import SmallBizRAG
from chat_bot.agent.agent import TrendAgent, SmallBizConsultant
from chat_bot.rag.prompts import UserContext

user_ctx = UserContext(industry="cafe", location="강남", budget=300000, goal="신규 고객 유치")

# RAG only
rag = SmallBizRAG()
rag_result = rag.query("카페 신메뉴 홍보 방법", user_context=user_ctx)
print(rag_result["answer"])

# 트렌드 질문 (Agent)
agent = TrendAgent()
trend_result = agent.run("요즘 유행하는 카페 마케팅", user_context=user_ctx)
print(trend_result["answer"])

# 의도별 라우팅
consultant = SmallBizConsultant()
print(consultant.consult("2024년 네이버/인스타 예산 배분 추천"))
```

### 5. FastAPI 연동 (통합 서버)
```bash
# 레포 루트에서 실행
uvicorn main:app --host 0.0.0.0 --port 9000
```
- SSE 채팅: `POST /chat/message/stream` (폼 필드 `message`, 선택: `session_id`, `image`)
- 세션·히스토리: `POST /chat/session`, `GET /chat/history`, `GET /chat/generation/{session_id}`
- 테스트 페이지: `/` (챗 UI), `/admin` (관리자; `is_admin=True` 계정 필요)
- 서버 스타트업에서 DB 초기화 + 이미지 생성 모델 preload 수행

---

## 📝 개발 로드맵

- [x] Phase 1: 데이터 수집 (592개 매장)
- [x] Phase 2: 문서 생성 및 최적화
- [x] Phase 3: Retrieval 평가 (R@1 = 0.8533)
- [x] Phase 4: Chroma 벡터스토어 구축
- [x] Phase 5: LangChain RAG 기본 구현
- [x] Phase 6: LangChain Agent 구현
- [x] Phase 7: Self-Refine 체인 (실험 완료)
- [x] Phase 8: FastAPI 연동 및 라우팅 정리 (SSE 포함)
- [ ] Phase 9: 평가 및 최적화

---

## 🔑 주요 클래스

| 클래스 | 위치 | 역할 |
|--------|------|------|
| `SmallBizRAG` | `rag/chain.py` | Chroma + LangChain RAG 파이프라인 |
| `PromptBuilder` / `IntentRouter` | `rag/prompts.py` | 태스크 분류 + 프롬프트 조립 |
| `TrendAgent` | `agent/agent.py` | 웹 검색 + RAG 사례 통합 에이전트 |
| `SmallBizConsultant` | `agent/agent.py` | 의도 기반 라우팅(RAG/Agent) |
| `SelfRefiner` | `refine/self_refine.py` | Self-Refine 실험 체인 |

---

## 📁 디렉토리 구조

```
chat_bot/
├── README.md
├── RAG_구축_체크리스트.md
├── chat_bot_기획.md
├── requirements.txt
├── __init__.py
├── config/                       # 설정
│   └── settings.py
├── data/                         # 데이터 파이프라인
│   ├── 01_crawl_naver.py         # 네이버 플레이스 크롤링
│   ├── 02_split_data.py          # 데이터 정제/분리
│   ├── 03_build_documents_v5.py  # 문서 생성
│   ├── 06_build_vectorstore.py   # Chroma 벡터스토어 생성 (output: data/vectorstore/)
│   ├── processed/                # 처리된 문서/코어 데이터
│   └── vectorstore/              # 생성된 Chroma DB (출력)
├── evaluation/                   # 평가 스크립트/결과
│   ├── 04_evaluate_embeddings.py
│   ├── 05_evaluate_reranker.py
│   ├── build_responses.py
│   ├── evaluate_prompts.py
│   └── results/
├── rag/                          # RAG 시스템
│   ├── chain.py                  # LangChain RAG 체인
│   └── prompts.py                # 프롬프트/의도 분류
├── agent/                        # LangChain Agent
│   └── agent.py
├── refine/                       # Self-Refine 실험
│   └── self_refine.py
├── api/                          # FastAPI 서버 초안
│   └── endpoints.py
└── core/                         # 추후 확장용 (현재 비워둠)
```

---

**작성일:** 2025-01-17
**담당자:** 배현석
**Framework:** LangChain + Chroma + FastAPI (SSE)
