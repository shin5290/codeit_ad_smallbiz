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

**RAG 검색 성능 평가 결과 (v5.9, 200개 쿼리):**
| 메트릭 | Baseline (Dense E5) | 비고 |
|--------|---------------------|------|
| **Recall@1** | **63.0%** | Top 1 정답률 |
| **Recall@5** | **88.5%** | Top 5 정답률 ✅ |
| **Recall@10** | **94.0%** | Top 10 정답률 ✅ |
| **MRR** | **73.6%** | 평균 순위 |
| **Success Rate** | **98.0%** | 답변 생성 가능률 ✅ |
| **Answer Quality** | **3.98/5** | LLM-as-Judge 평가 |

**실험한 개선 방법 (모두 실패):**
- Metadata Filtering: R@5 하락 (88.5% → 79.5%)
- Hybrid Search (BM25+E5): 성능 저하 (R@1 -45%)
- BGE Reranker: Latency 80배 증가 (0.27초 → 22.85초)
- Query Rewriting: 성능 저하 (R@5 -6.2%)

**최종 결정**: Baseline (Dense E5 only) 채택 → Simple is Best

**End-to-End 시스템 평가 결과 (200개 쿼리):**
| 메트릭 | 결과 | 비고 |
|--------|------|------|
| **Intent 정확도** | **91.5%** | 라우팅 정확도 ✅ |
| **비용** | **$0.0016/쿼리** | 월 1만 쿼리 $16 ✅ |
| **Latency** | **8.5초** | 개선 필요 ⚠️ (목표: 2-3초) |
| **Self-Refine 효율** | **36% 개선** | 25% 쿼리만 적용 |

**Route 분포:**
- doc_rag (사례 검색): 60.5%
- marketing_counsel (전략 조언): 39.0%
- trend_web (웹 검색): 0.5%

> 상세 평가 결과: [evaluation/results/eval_summary.md](evaluation/results/eval_summary.md)

**LLM-as-a-Judge 평가 결과 (프롬프트 엔지니어링):**
| 항목 | 최종 점수 | 설명 |
|------|-----------|------|
| Specificity | 8.50 / 10 | 숫자, 구체적 예시 포함 |
| Evidence | 9.00 / 10 | 출처, 사례 근거 명시 |
| Structure | 10.00 / 10 | 응답 구조 일관성 |
| Safety | 10.00 / 10 | 과장/허위 정보 없음 |

> 4회 iteration으로 베이스라인(Evidence 6.33) → 최종(9.00) 개선. 주요 변경: 항목별 출처 태깅, k=7 확장, 숫자 하한선 적용.

**임베딩 모델 VRAM 최적화:**

이미지 생성 모델(~20GB)과 함께 운영 시 VRAM 부족 문제가 발생하여 최적화를 진행했습니다.

| 항목 | Before (GPU) | After (CPU + 최적화) |
|------|-------------|---------------------|
| VRAM | 2.2GB | **0GB** |
| Recall@1 | 0.8533 | **0.8533** (유지) |
| Latency (단일) | ~25ms | ~200ms |
| OOM 위험 | 있음 | **없음** |

**최종 결정**: CPU 전환 + 큐잉 + 마이크로배치
- 정확도 유지 (OpenAI API는 6%p 하락)
- VRAM 완전 해제 (메모리 파편화 문제 해결)
- 동시 사용자 대응 (Lock + 마이크로배치)

> 상세 벤치마크, 실험 과정, 설계 원칙: [docs/EMBEDDING_OPTIMIZATION.md](docs/EMBEDDING_OPTIMIZATION.md)

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
- [x] Phase 3: Retrieval 평가 (Baseline 채택)
- [x] Phase 4: Chroma 벡터스토어 구축
- [x] Phase 5: LangChain RAG 기본 구현
- [x] Phase 6: LangChain Agent 구현
- [x] Phase 7: Self-Refine 체인 (조건부 적용)
- [x] Phase 8: FastAPI 연동 및 라우팅 정리 (SSE 포함)
- [x] **Phase 9: 종합 평가 완료** (RAG + End-to-End)

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
├── README.md                     # 이 파일
├── requirements.txt
├── __init__.py
│
├── config/                       # 설정
│   └── settings.py
│
├── data/                         # 데이터 파이프라인
│   ├── 01_crawl_naver.py         # 네이버 플레이스 크롤링
│   ├── 02_split_data.py          # 데이터 정제/분리
│   ├── 03_build_documents_v5.py  # 문서 생성
│   ├── 06_build_vectorstore.py   # Chroma 벡터스토어 생성
│   ├── processed/                # 처리된 문서 (documents_v5.jsonl)
│   └── vectorstore/              # Chroma DB (chroma_db/)
│
├── docs/                         # 문서
│   └── EMBEDDING_OPTIMIZATION.md # 임베딩 최적화 기록
│
├── evaluation/                   # 평가 시스템
│   ├── 01_generate_queries.py    # 평가 쿼리 생성
│   ├── 02_evaluate_recall.py     # Recall@K 평가
│   ├── 03_evaluate_hybrid_reranker.py
│   ├── 04_evaluate_advanced_metrics.py
│   ├── 05_evaluate_query_rewriting.py
│   ├── 06_end_to_end_eval.py     # End-to-End 시스템 평가
│   ├── README.md                 # 평가 가이드
│   ├── FINAL_EVALUATION_RESULTS.md  # RAG 평가 결과
│   └── results/
│       ├── queries_final.json    # 평가 쿼리 200개
│       ├── end_to_end_results.json
│       └── eval_summary.md       # 종합 평가 요약
│
├── rag/                          # RAG 시스템 (LangChain)
│   ├── chain.py                  # SmallBizRAG 클래스
│   └── prompts.py                # IntentRouter, UserContext
│
├── agent/                        # Agent 시스템 (LangChain)
│   └── agent.py                  # TrendAgent, SmallBizConsultant
│
├── refine/                       # Self-Refine (조건부 적용)
│   └── self_refine.py            # SelfRefiner 클래스
│
├── api/                          # FastAPI 연동
│   └── endpoints.py
│
└── core/                         # 확장용 (비워둠)
```

---

## 📊 프로젝트 요약

**기간**: 2026-01-20 ~ 2026-01-27 (1주)
**담당자**: 배현석
**Framework**: LangChain + Chroma + FastAPI (SSE)

**핵심 성과**:
- RAG 검색 정확도: Recall@5 **88.5%**
- Intent 라우팅 정확도: **91.5%**
- 운영 비용: 월 1만 쿼리 **$16** (매우 저렴)
- 평가 완료: 200개 쿼리 종합 테스트

**최종 업데이트**: 2026-01-27
