# RAG 기반 상담 챗봇 백엔드 기획

> **작성일:** 2025-01-14  
> **최종 수정:** 2025-01-21  
> **범위:** 백엔드 전용 (FastAPI + RAG + Agent) — Streamlit/UI, MCP는 제외

---

## 🎯 목표
- 소상공인 마케팅 상담용 RAG/Agent 백엔드 구축
- FastAPI에 바로 연결 가능한 구조 유지
- 데이터 파이프라인·평가 스크립트 재현성 확보

## 🔍 현재 상태
- 데이터 수집/정제/문서화 완료 (`data/01~03`)
- Chroma 벡터스토어 생성 스크립트 정리 (`data/06_build_vectorstore.py`)
- LangChain RAG 체인 정리 (`rag/chain.py`)
- 트렌드 Agent (웹 검색 + RAG 사례) 정리 (`agent/agent.py`)
- 평가 스크립트 정리 (`evaluation/04`, `evaluation/05`, `build_responses.py`, `evaluate_prompts.py`)
- Self-Refine 실험 완료 (`refine/self_refine.py`, 점수 7~9.2/10 확인)
- FastAPI SSE 라우팅 및 서버 연동 완료 (`/chat/message/stream`, 세션/히스토리/생성이력 API, `main.py` 스타트업 훅)

## 🗺️ 아키텍처 (백엔드)
```
사용자 → FastAPI (상담 엔드포인트)
                    ↓
           ┌────────────────┐
           │ Intent Router  │  ← rag.prompts.IntentRouter
           └────────────────┘
            ↓             ↓
        RAG (rag.chain)   Agent (agent.agent)
            ↓             ↓
      답변 + 출처        답변 + 웹/사례 근거
```

## 🛠️ 주요 컴포넌트
- `data/01_crawl_naver.py` : 네이버 플레이스 크롤링
- `data/02_split_data.py` : 정제/중복제거/코어 분리
- `data/03_build_documents_v5.py` : 문서 JSONL 생성
- `data/06_build_vectorstore.py` : Chroma DB 생성 (`data/vectorstore/chroma_db`)
- `rag/chain.py` : LangChain 기반 SmallBizRAG (E5 임베딩, 프롬프트 엔지니어링)
- `agent/agent.py` : TrendAgent + SmallBizConsultant (의도별 라우팅)
- `evaluation/04_evaluate_embeddings.py`, `05_evaluate_reranker.py` : 성능 평가

## 🚧 남은 작업 (백로그)
- Self-Refine를 프로덕션 후처리에 적용할지 결정 (선택)
- 배포/운영 환경용 로깅·모니터링·헬스체크 정리 (`/health`, 버전 정보 등)

## ⚙️ 실행 요약
```bash
# 벡터스토어 생성
python data/06_build_vectorstore.py

# RAG 체인 단독 테스트
python -m rag.chain

# Agent 테스트
python agent/agent.py
```
