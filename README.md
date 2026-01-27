# CodeIt Ad SmallBiz

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Repository 소개

**코드잇 4기 4팀** 고급 프로젝트
**"생성모델을 활용한 소상공인 광고 컨텐츠 생성 서비스"**

AI 기술을 활용하여 소상공인이 업종에 무관하게 손쉽게 광고 이미지와 마케팅 문구를 생성할 수 있는 웹 서비스를 제공합니다.

**프로젝트 기간**: 2025-12-29 ~ 2026-01-28 (총 31일)

---
## 결과 보고서 및 협업 일지
[결과보고서 다운로드](https://naver.me/G02ABIM7)

**협업일지**
[배현석]()
[신승목](https://github.com/shin5290/codeit_ad_smallbiz/tree/alpha/docs/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EC%8B%A0%EC%8A%B9%EB%AA%A9)
[이유노]()
[이현석](https://github.com/shin5290/codeit_ad_smallbiz/tree/alpha/docs/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EC%9D%B4%ED%98%84%EC%84%9D)
[진수경]()

---

## 목차

- [팀 구성](#팀-구성)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [시스템 아키텍처](#시스템-아키텍처)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [API 문서](#api-문서)
- [개발 가이드](#개발-가이드)
- [배포](#배포)

---

## 팀 구성

| 이름 | 역할 | 담당 업무 |
|------|------|-----------|
| **진수경** | 백엔드 + 프론트엔드 | 비즈니스 로직 통합 + 백엔드 API + 프론트 |
| **이현석** | AI 이미지 | Z-Image Turbo 기반 이미지 생성 모듈 |
| **배현석** | AI 텍스트, 챗봇 | 프롬프트/문구 생성 템플릿, 챗봇 |
| **이유노** | 광고 문구 | 광고 문구 파이프라인 |
| **신승목** | 인프라, 테스트 | 환경 조성, 문서화, 업종 확장 테스트 |

---

## 주요 기능

### 핵심 기능

- **RAG 기반 대화형 챗봇**
  - 5개 Intent 분류 (chitchat, trend_web, task_action, marketing_counsel, doc_rag)
  - Intent별 라우팅 (LLM 직접 응답 / Agent 웹검색 / RAG 검색 / 생성 모듈)
  - Slot-Filling (업종/지역/예산/목표 자동 추출 및 질문)
  - Tool Calling Agent (Tavily/DuckDuckGo 웹검색 + RAG 하이브리드)
  - Self-Refine (답변 품질 자동 평가 7점 미만 시 개선)
  - 프롬프트 엔지니어링 (태스크별: recommend, strategy, trend, photo_guide 등)
  - SSE 스트리밍 응답

- **AI 광고 이미지 생성**
  - Z-Image Turbo 모델 기반 고속 이미지 생성 (~1-2초, 8 steps)
  - Text-to-Image (T2I) 및 Image-to-Image (I2I) 지원
  - LoRA를 통한 스타일 전환
  - 다양한 스타일: ultra_realistic, semi_realistic, anime

- **AI 광고 문구 생성**
  - GPT-4o-mini 기반 맞춤형 마케팅 카피 생성
  - 톤 앤 매너 설정 (warm, professional, friendly, energetic, practical, respectful)
  - 길이 조절 가능 (10~200자)
  - industries.yaml 기반 247개 업종별 최적화 프롬프트 (S~E 등급, 18개 하위 그룹)
  - AIDA 프레임워크 기반 카피라이팅

- **사용자 인증 및 관리**
  - JWT 기반 인증 시스템 (HttpOnly 쿠키)
  - 회원가입, 로그인, 정보 수정, 회원 탈퇴
  - bcrypt 비밀번호 해싱

- **생성 이력 관리**
  - 세션별 채팅 히스토리 저장
  - 생성된 이미지/텍스트 이력 조회
  - 이미지 파일 해시 기반 중복 방지

- **다양한 이미지 비율**
  - 1:1 (1024x1024) - SNS 프로필, 썸네일
  - 3:4 (896x1152) - Instagram 피드, 포스터
  - 4:3 (1152x896) - 프레젠테이션, 배너
  - 16:9 (1344x768) - 유튜브 썸네일, 웹 배너
  - 9:16 (768x1344) - Instagram Story, 모바일

---

## 기술 스택

### 프론트엔드
- **Framework**: Svelte 4.x
- **Build Tool**: Vite
- **Deployment**: Vercel

### 백엔드
- **Framework**: FastAPI 0.104+
- **Language**: Python 3.10+
- **ASGI Server**: Uvicorn
- **Validation**: Pydantic v2, pydantic-settings
- **Authentication**: python-jose (JWT), bcrypt, passlib

### AI 모델

**이미지 생성**
- Z-Image Turbo (8 steps 고속 생성)
- Diffusers, Transformers
- PyTorch 2.6.0+ (CUDA 12.6)
- LoRA 스타일 전환

**텍스트 생성**
- OpenAI GPT-4o-mini
- OpenAI API 1.6+
- industries.yaml 기반 247개 업종 분류 (S~E 등급)
- AIDA 프레임워크 기반 프롬프트 엔지니어링
- PyYAML (업종 설정 로드)

**RAG 챗봇**
- LangChain (langchain, langchain-openai, langchain-community)
- ChromaDB 벡터스토어 (영구 저장, 메타데이터 필터링)
- Sentence-Transformers (intfloat/multilingual-e5-large)
- BGE Reranker (BAAI/bge-reranker-v2-m3, 선택적)
- Tool Calling Agent (웹 검색: Tavily/DuckDuckGo + RAG 검색)
- Self-Refine (답변 품질 자동 평가 및 개선)
- Intent Router (5개 인텐트: chitchat, trend_web, task_action, marketing_counsel, doc_rag)
- Slot-Filling (업종/지역/예산 자동 추출)

### 데이터베이스
- **RDBMS**: PostgreSQL 15
- **ORM**: SQLAlchemy 2.0+
- **Migration**: Alembic
- **Schema**: 5개 테이블
  - User: 사용자 계정 정보
  - ChatSession: 채팅 세션
  - ChatHistory: 채팅 메시지 로그
  - GenerationHistory: 광고/이미지 생성 이력
  - ImageMatching: 이미지 파일 레지스트리

### 인프라
- **Compute**: GCP VM (g2-standard-4)
- **GPU**: NVIDIA L4 (24GB VRAM)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 로컬 파일 시스템 (200GB)

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 (웹 브라우저)                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   프론트엔드 (Svelte)                        │
│  - Vercel 자동 배포                                          │
│  - 챗봇 UI, SSE 스트리밍 수신                                │
│  - 회원가입/로그인 UI                                        │
│  - 생성 이력 조회                                            │
│  - 관리자 대시보드 (admin.html)                              │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API + SSE (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              GCP VM 인스턴스 (g2-standard-4)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         FastAPI Backend (main.py)                    │   │
│  │  - 포트: 8000                                         │   │
│  │  - Lifespan: DB 초기화, 모델/RAG Preload             │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│                 ▼                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        API 라우터 레이어 (routers/)                   │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  auth.py: /auth/*                              │  │   │
│  │  │  - 회원가입, 로그인/로그아웃, 회원정보 관리     │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  chat.py: /chat/*                              │  │   │
│  │  │  - 챗봇 메시지 스트리밍, 세션/히스토리 관리     │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  admin.py: /admin/*                            │  │   │
│  │  │  - 사용자 관리, 생성 이력 조회 (관리자 전용)    │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│                 ▼                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        비즈니스 로직 레이어                           │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  services.py                                   │  │   │
│  │  │  - 인증: register_user, authenticate_user      │  │   │
│  │  │  - 생성: generate_contents, persist_generation │  │   │
│  │  │  - 스트림: handle_chat_message_stream          │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  chatbot.py (RAG 챗봇)                         │  │   │
│  │  │  - ConversationManager: 대화 히스토리 관리      │  │   │
│  │  │  - LLMOrchestrator: Intent 분석, Refinement    │  │   │
│  │  │  - ConsultingService: 상담 응답 스트리밍        │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  process_db.py                                 │  │   │
│  │  │  - User, ChatSession, ChatHistory CRUD         │  │   │
│  │  │  - GenerationHistory, ImageMatching CRUD       │  │   │
│  │  │  - Admin: list_users, delete_users_by_ids      │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────┬───────────────────┬───────────────────────────┘   │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌─────────────────┐ ┌─────────────────────────────────┐    │
│  │ Text Generation │ │ Image Generation                │    │
│  │ (GPT-4o-mini)   │ │ (Z-Image Turbo)                 │    │
│  │                 │ │                                 │    │
│  │ text_generator  │ │ generator.py → workflow.py     │    │
│  │   .py           │ │   → nodes/text2image.py        │    │
│  │                 │ │   → nodes/image2image.py       │    │
│  │                 │ │   → prompt/prompt_manager.py   │    │
│  └─────────────────┘ └─────────────────────────────────┘    │
│         │                   │                               │
│         └─────────┬─────────┘                               │
│                   ▼                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Preload Layer                                │   │
│  │  - preload.py: 이미지 생성 모델 GPU 사전 로딩        │   │
│  │  - rag_preload.py: RAG 벡터스토어/임베딩 사전 로딩   │   │
│  └──────────────────────────────────────────────────────┘   │
│                   │                                         │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │         Data Layer                                   │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  PostgreSQL (adbizdb)                          │  │   │
│  │  │  - User (is_admin 포함)                        │  │   │
│  │  │  - ChatSession, ChatHistory                    │  │   │
│  │  │  - GenerationHistory, ImageMatching            │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  File Storage (data/)                          │  │   │
│  │  │  - generated/ (생성된 이미지)                   │  │   │
│  │  │  - uploads/ (업로드된 이미지)                   │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 빠른 시작

### 사전 요구사항

- **Python**: 3.10 이상
- **PostgreSQL**: 15 이상
- **NVIDIA GPU**: L4 또는 동등 이상 (선택사항)
- **CUDA**: 12.1 이상 (GPU 사용 시)
- **API Keys**: OpenAI API Key

### 1. 저장소 클론

```bash
git clone https://github.com/shin5290/codeit_ad_smallbiz.git
cd codeit_ad_smallbiz
```

### 2. 가상환경 설정

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
```

필수 환경 변수:
```ini
DATABASE_URL=postgresql+psycopg2://aduser:your_password@localhost:5432/adbizdb
JWT_SECRET_KEY=<64자_hex_문자열>
OPENAI_API_KEY=sk-proj-...
```

### 5. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. 접속 확인

- **API 문서**: http://localhost:8000/docs

---

## 프로젝트 구조

```
codeit_ad_smallbiz/
├── main.py                               # FastAPI 앱 진입점
├── alembic.ini                           # Alembic 설정
├── requirements.txt                      # Python 의존성
├── run.sh                                # 실행 스크립트
├── README.md
│
├── alembic/                              # DB 마이그레이션
│   ├── env.py
│   ├── script.py.mako
│   └── versions/                         # 마이그레이션 파일
│       ├── 001_add_generation_revision_columns.py
│       ├── 002_add_pgvector_embedding.py
│       ├── 003_remove_revision_columns.py
│       ├── 004_remove_pgvector_embedding.py
│       ├── 005_expand_generation_seed_bigint.py
│       └── 006_add_admin_role.py
│
│
├── src/
│   ├── backend/                          # 백엔드 핵심 로직
│   │   ├── __init__.py                   # 모듈 초기화
│   │   ├── models.py                     # SQLAlchemy ORM 모델 (5개 테이블)
│   │   ├── schemas.py                    # Pydantic 요청/응답 스키마
│   │   ├── process_db.py                 # DB CRUD 함수
│   │   ├── services.py                   # 비즈니스 로직 오케스트레이션
│   │   ├── chatbot.py                    # RAG 챗봇 (ConversationManager/LLMOrchestrator/ConsultingService)
│   │   ├── consulting_knowledge_base.py  # 상담용 지식 베이스
│   │   ├── rag_preload.py                # RAG 모델 프리로드
│   │   └── routers/                      # API 라우터
│   │       ├── __init__.py
│   │       ├── auth.py                   # 인증 API (/auth/*)
│   │       ├── chat.py                   # 채팅 API (/chat/*)
│   │       └── admin.py                  # 관리자 API (/admin/*)
│   │
│   ├── generation/                       # AI 생성 모듈
│   │   ├── __init__.py
│   │   ├── text_generation/              # 광고 문구 생성
│   │   │   ├── __init__.py               # 모듈 export (TextGenerator, PromptTemplateManager 등)
│   │   │   ├── text_generator.py         # GPT-4o-mini 기반 카피 생성 (use_industry_config 지원)
│   │   │   ├── ad_generator.py           # 광고 생성 통합 API
│   │   │   ├── prompt_manager.py         # 업종별 프롬프트 관리 (IndustryConfigLoader, AdCopyPromptBuilder)
│   │   │   ├── industries.yaml           # 247개 업종 설정 (S~E 등급, 18개 하위 그룹)
│   │   │   └── evaluate_prompt.py        # Civitai 기반 프롬프트 평가 (CivitaiEnhancedEvaluator)
│   │   │
│   │   ├── image_generation/             # 이미지 생성
│   │   │   ├── generator.py              # 메인 진입점 (generate_and_save_image)
│   │   │   ├── generator_sdxl.py         # SDXL 버전 (대체)
│   │   │   ├── workflow.py               # 생성 워크플로우
│   │   │   ├── preload.py                # 모델 프리로드 관리
│   │   │   ├── shared_cache.py           # 모델 캐시 관리
│   │   │   ├── nodes/                    # 생성 노드
│   │   │   │   ├── base.py               # 기본 노드
│   │   │   │   ├── text2image.py         # T2I 노드 (Z-Image Turbo)
│   │   │   │   ├── image2image.py        # I2I 노드
│   │   │   │   ├── controlnet.py         # ControlNet 노드
│   │   │   │   ├── preprocessing.py      # 전처리
│   │   │   │   ├── postprocessing.py     # 후처리
│   │   │   │   ├── text_overlay.py       # 텍스트 오버레이
│   │   │   │   ├── save_image.py         # 이미지 저장
│   │   │   │   ├── prompt_processor.py   # 프롬프트 처리
│   │   │   │   ├── gpt_layout_analyzer.py      # GPT 레이아웃 분석
│   │   │   │   ├── product_layout_analyzer.py  # 제품 레이아웃 분석
│   │   │   │   └── sdxl/                 # SDXL 전용 노드
│   │   │   │       ├── text2image_sdxl.py
│   │   │   │       ├── image2image_sdxl.py
│   │   │   │       └── controlnet_sdxl.py
│   │   │   ├── prompt/                   # 프롬프트 관리
│   │   │   │   ├── prompt_manager.py     # 한글→영어 프롬프트 변환 (GPT)
│   │   │   │   ├── prompt_templates.py   # 프롬프트 템플릿
│   │   │   │   ├── style_router.py       # 스타일 라우터
│   │   │   │   ├── config_loader.py      # 설정 로더
│   │   │   │   └── input_parser.py       # 입력 파서
│   │   │   └── tools/                    # 유틸리티 도구
│   │   │       ├── font_loader.py        # 폰트 로더
│   │   │       └── text_layout_tools.py  # 텍스트 레이아웃 도구
│   │   │
│   │   └── chat_bot/                     # RAG 기반 상담 챗봇
│   │       ├── rag/                      # RAG 파이프라인
│   │       │   ├── chain.py              # LangChain RAG 체인 (SmallBizRAG, E5Embeddings)
│   │       │   ├── knowledge_base.py     # 백엔드 연동 Knowledge Base (SmallBizKnowledgeBase)
│   │       │   └── prompts.py            # 프롬프트 엔지니어링 (PromptBuilder, IntentRouter, SlotChecker)
│   │       ├── agent/                    # 에이전트
│   │       │   └── agent.py              # TrendAgent (웹검색+RAG), SmallBizConsultant (통합 라우터)
│   │       ├── refine/                   # Self-Refine 모듈
│   │       │   └── self_refine.py        # SelfRefiner (답변 품질 자동 개선)
│   │       ├── api/                      # API 엔드포인트
│   │       │   └── endpoints.py          # FastAPI 상담 챗봇 API (/consult, /health)
│   │       ├── data/                     # 데이터 파이프라인
│   │       │   ├── 01_crawl_naver.py     # 네이버 플레이스 크롤러 (전국 1000개+)
│   │       │   ├── 02_split_data.py      # 데이터 분할
│   │       │   ├── 03_build_documents_v5.py # 문서 구축
│   │       │   ├── 06_build_vectorstore.py  # Chroma 벡터스토어 빌드
│   │       │   ├── processed/            # 처리된 데이터 (documents_v5.jsonl)
│   │       │   └── vectorstore/          # ChromaDB 벡터스토어 (chroma_db/)
│   │       ├── evaluation/               # 평가 스크립트
│   │       │   ├── 04_evaluate_embeddings.py  # 임베딩 모델 평가 (Recall@K, MRR)
│   │       │   ├── 05_evaluate_reranker.py    # Reranker 평가
│   │       │   ├── build_responses.py         # 응답 생성
│   │       │   └── evaluate_prompts.py        # 프롬프트 평가
│   │       └── config/                   # 설정
│   │           └── settings.py           # RAG/Agent 설정 (API키, 모델, 임계값)
│   │
│   ├── utils/                            # 유틸리티
│   │   ├── config.py                     # Pydantic Settings 환경 설정
│   │   ├── security.py                   # JWT, bcrypt 인증
│   │   ├── session.py                    # 세션 관리
│   │   ├── image.py                      # 이미지 파일 처리
│   │   └── logging.py                    # 로깅 설정
│   │
│   └── frontend/                         # 프론트엔드 정적 파일
│       ├── main.html                     # 메인 페이지
│       ├── admin.html                    # 관리자 페이지
│       └── static/                       # 정적 파일 (CSS, JS)
│
├── data/                                 # 데이터 저장소
│   ├── generated/                        # 생성된 이미지
│   └── uploads/                          # 업로드된 이미지
│
└── docs/                                 # 문서
    └── doc/
        └── 시스템_아키텍처_설계서.md
```

---

## API 문서

### 주요 엔드포인트

서버 실행 후 http://localhost:8000/docs 에서 전체 API 문서를 확인할 수 있습니다.

#### 1. 인증 API (`/auth/*`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/auth/me` | 현재 로그인 사용자 정보 |
| POST | `/auth/signup` | 회원가입 |
| POST | `/auth/login` | 로그인 (OAuth2 폼) |
| POST | `/auth/logout` | 로그아웃 |
| PUT | `/auth/user` | 회원정보 수정 |
| DELETE | `/auth/user` | 회원 탈퇴 |

#### 2. 채팅 API (`/chat/*`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/chat/message/stream` | 챗봇 메시지 (SSE 스트리밍) |
| POST | `/chat/session` | 세션 조회/생성 |
| GET | `/chat/history` | 유저 대화 히스토리 (페이징) |
| GET | `/chat/history/{session_id}` | 세션별 대화 히스토리 |
| GET | `/chat/generation/{session_id}` | 세션별 생성 이력 |

#### 3. 이미지 API (`/images/*`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/images/{file_hash}` | 이미지 파일 서빙 |

#### 4. 관리자 API (`/admin/*`)

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/admin/users` | 사용자 목록 조회 (필터링/페이징) | 관리자 |
| POST | `/admin/users/delete` | 사용자 일괄 삭제 | 관리자 |
| GET | `/admin/generations` | 생성 이력 조회 (필터링/페이징) | 관리자 |

**관리자 API 쿼리 파라미터**:

`GET /admin/users`:
- `limit`: 페이지 크기 (기본 15, 최대 200)
- `offset`: 오프셋
- `user_id`: 사용자 ID 필터
- `login_id`: 로그인 ID 필터
- `name`: 이름 필터
- `is_admin`: 관리자 여부 필터
- `start_date`, `end_date`: 가입일 범위 필터

`GET /admin/generations`:
- `page`: 페이지 번호 (1부터 시작)
- `limit`: 페이지 크기 (기본 5, 최대 50)
- `user_id`: 사용자 ID 필터
- `login_id`: 로그인 ID 필터
- `session_id`: 세션 ID 필터
- `content_type`: 콘텐츠 타입 필터 (image/text)
- `start_date`, `end_date`: 생성일 범위 필터

---

## 핵심 플로우

### 1. 챗봇 메시지 처리 플로우 (SSE 스트리밍)

```
사용자 메시지 입력
        │
        ▼
┌─────────────────────────────┐
│  /chat/message/stream       │
│  (POST, SSE)                │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  1. ingest_user_message()   │
│  - 세션 확보/생성           │
│  - 이미지 저장 (있으면)      │
│  - 채팅 히스토리 저장        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  2. LLM analyze_intent()    │
│  - Intent 분류              │
│  - 플랫폼/비율 감지         │
│  - 스타일/업종 감지         │
│  - strength 감지 (수정시)    │
└─────────────────────────────┘
        │
        ▼
    ┌───┴───┐
    │Intent │
    └───┬───┘
        │
   ┌────┼────┬────────────┐
   ▼    ▼    ▼            ▼
consulting  generation  modification
   │         │            │
   ▼         ▼            ▼
┌──────┐  ┌──────────┐  ┌────────────────┐
│상담   │  │refine +  │  │refine +        │
│스트림 │  │generate  │  │target ID 탐색 +│
│      │  │          │  │generate        │
└──────┘  └──────────┘  └────────────────┘
   │         │            │
   └────┬────┴────────────┘
        │
        ▼
┌─────────────────────────────┐
│  3. persist_generation()    │
│  - 이미지 DB 저장           │
│  - 채팅 히스토리 저장        │
│  - 생성 이력 저장           │
└─────────────────────────────┘
        │
        ▼
    SSE done 이벤트
```

### 2. Intent 분석 상세

**analyze_intent() 반환값**:
```json
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "generation_type": "image|text|null",
  "aspect_ratio": "1:1|16:9|9:16|4:3" or null,
  "style": "ultra_realistic|semi_realistic|anime" or null,
  "industry": "cafe|restaurant|..." or null,
  "strength": 0.0-1.0 or null,
  "text_tone": "warm|professional|friendly|energetic" or null,
  "text_max_length": 10-200 or null
}
```

**플랫폼별 비율 매핑**:
| 플랫폼 | aspect_ratio |
|--------|--------------|
| Instagram 피드 | 1:1 |
| YouTube 썸네일 | 16:9 |
| Instagram Story | 9:16 |
| Facebook | 4:3 |
| 포스터/전단지 | 4:3 |
| 배너 | 16:9 |

**수정 강도(strength) 매핑**:
| 표현 | strength |
|------|----------|
| 살짝, 약간, 조금 | 0.35 |
| 좀, 적당히 (기본) | 0.55 |
| 많이, 크게, 확실하게 | 0.75 |
| 완전히 다르게, 처음부터 | 0.9 |

### 3. 이미지 생성 플로우

```
한글 사용자 입력
        │
        ▼
┌─────────────────────────────┐
│  PromptTemplateManager      │
│  generate_detailed_prompt() │
│  - GPT로 영어 프롬프트 생성 │
│  - 스타일 감지              │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  ImageGenerationWorkflow    │
│  - T2I: Text2ImageNode      │
│  - I2I: Image2ImageNode     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Z-Image Turbo 추론         │
│  - 8 steps                  │
│  - LoRA 스타일 적용         │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  이미지 저장                │
│  - SHA-256 해시 파일명      │
│  - data/generated/{hash}.jpg │
└─────────────────────────────┘
```

---

## 개발 가이드

### 코드 스타일

- **스타일**: PEP 8
- **Import 순서**: 표준 라이브러리 → 서드파티 → 로컬
- **Import 경로**: 절대 경로 사용 (`from src.backend import ...`)

```python
# 올바른 예시
from typing import Optional
from fastapi import FastAPI
from src.backend.services import register_user
from src.utils.config import settings
```

### 환경 설정

```python
from src.utils.config import settings

db_url = settings.DATABASE_URL
api_key = settings.OPENAI_API_KEY
```

---

## 배포

### 프로덕션 배포 (GCP VM)

```bash
# 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 프론트엔드 배포 (Vercel)

Vercel Dashboard에서 환경 변수 설정:
- `VITE_API_URL`: 백엔드 API URL

---

## 라이선스

이 프로젝트는 **Apache License 2.0** 하에 배포됩니다.

---

**CodeIt 4기 4팀** | 2025-12-29 ~ 2026-01-28

