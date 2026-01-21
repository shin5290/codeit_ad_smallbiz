# RAG 상담 챗봇 구축 체크리스트 (백엔드 전용)

> **최종 수정:** 2025-01-21  
> **Framework:** LangChain + Chroma (FastAPI 연동 완료, SSE 스트리밍)  

---

## ✅ Phase 0: 데이터 수집
- [x] 네이버 플레이스 크롤링 (592개)
- [x] 데이터 품질 확인 및 원본 저장 (`data/raw/…`)

산출물: `data/01_crawl_naver.py`, 원본 JSON

## ✅ Phase 1: 문서/벡터스토어
- [x] 데이터 정제/중복 제거 (`data/02_split_data.py`)
- [x] 문서 생성 v5 (`data/03_build_documents_v5.py`)
- [x] 임베딩 평가 (`evaluation/04_evaluate_embeddings.py`)
- [x] Reranker 평가 (`evaluation/05_evaluate_reranker.py`)
- [x] Chroma 벡터스토어 생성 (`data/06_build_vectorstore.py` → `data/vectorstore/chroma_db`)

산출물: `data/processed/documents_v5.jsonl`, `data/vectorstore/chroma_db/`

## ✅ Phase 2: RAG 기본 파이프라인
- [x] E5 임베딩 + Chroma Retriever
- [x] 메타데이터 필터링
- [x] GPT-4o 계열 LLM 연동
- [x] RAG 클래스 (`SmallBizRAG`)

산출물: `rag/chain.py`

## ✅ Phase 3: 프롬프트 엔지니어링
- [x] 11개 규칙 System Prompt
- [x] 태스크별 프롬프트 (recommend/ad_copy/strategy/trend/photo_guide/general)
- [x] IntentRouter, UserContext, 출처 포맷팅
- [x] 평가 스크립트 (`build_responses.py`, `evaluate_prompts.py`)
- [x] LLM-as-a-Judge 평가 완료

산출물: `rag/prompts.py`, `evaluation/*.py`, 평가 결과 JSON/CSV/MD

### 📊 LLM-as-a-Judge 평가 결과

**최종 스코어 (Iteration 4):**
| 항목 | 점수 |
|------|------|
| Specificity (구체성) | 8.50 / 10 |
| Evidence (근거) | 9.00 / 10 |
| Structure (구조) | 10.00 / 10 |
| Safety (안전성) | 10.00 / 10 |
| Rule Violations | 0 |

**개선 히스토리:**
| Iteration | 주요 변경 | Specificity | Evidence | Structure | Safety |
|-----------|-----------|-------------|----------|-----------|--------|
| 1 | 베이스라인 | 7.67 | 6.33 | 9.00 | 9.33 |
| 2 | 숫자·근거 요구(느슨) | 7.50 | 6.00 | 9.00 | 9.50 |
| 3 | 숫자 상향, 출처 요구 | 8.25 | 6.75 | 9.25 | 9.25 |
| 4 | 항목별 출처 부착, k=7 | **8.50** | **9.00** | **10.00** | **10.00** |

**효과가 컸던 변경:**
- 항목별 출처 태깅: 각 bullet 끝에 `(출처: {제목}({지역}))` 강제 → Evidence/Structure 개선
- 검색 폭 확장: k=3→7로 사례 밀도 증가 → 근거 점수 상승
- 숫자 하한선: 주요 항목에 숫자 2개 이상 요구 → Specificity 개선

## ✅ Phase 4: Agent
- [x] TrendAgent (웹 검색 + RAG 사례 하이브리드)
- [x] Tool Calling + 중복 호출 방지
- [x] SmallBizConsultant: 의도별 라우팅(RAG/Agent)

산출물: `agent/agent.py`

## ✅ Phase 5: Self-Refine (실험 완료)
- [x] Critique/Refine 프롬프트 보완
- [x] LangChain SequentialChain 연결 및 단독/통합 테스트 (최종 점수 7~9.2/10)
- [ ] API 후처리 적용 여부 결정 (선택)

산출물: `refine/self_refine.py` (테스트 로그)

## ✅ Phase 6: FastAPI 연동
- [x] `/chat/message/stream` SSE 스트리밍 엔드포인트 연동
- [x] 세션/대화/생성이력 API 정리 (`/chat/session`, `/chat/history`, `/chat/generation`)
- [x] 서버 스타트업 훅에서 DB 초기화 및 이미지 모델 preload (`main.py`)
- [x] 관리자/인증 라우터 통합 및 `is_admin` 권한 필드 추가

산출물: `main.py`, `src/backend/routers/chat.py`, `src/backend/services.py`, `src/backend/chatbot.py`, `src/frontend/test.html`, `src/frontend/admin.html`

---

## 📊 진행률 (현재 기준)
```
Phase 0: ████████████████████ 100%  데이터 수집
Phase 1: ████████████████████ 100%  문서/벡터스토어
Phase 2: ████████████████████ 100%  RAG 기본
Phase 3: ████████████████████ 100%  프롬프트
Phase 4: ████████████████████ 100%  Agent
Phase 5: ████████████████████ 100%  Self-Refine 실험
Phase 6: ████████████████████ 100%  FastAPI 연동
```

---

## 📁 파일 구조 (요약)
```
chat_bot/
├── data/                    # 수집/정제/문서화/벡터스토어
├── evaluation/              # 임베딩·프롬프트·응답 평가
├── rag/                     # SmallBizRAG + 프롬프트
├── agent/agent.py           # TrendAgent + 라우터
├── refine/self_refine.py    # Self-Refine 실험
├── api/endpoints.py         # FastAPI 초안
├── config/settings.py       # 설정
├── README.md
└── chat_bot_기획.md
```
