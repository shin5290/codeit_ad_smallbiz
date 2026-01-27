# CodeIt Ad SmallBiz

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Repository 소개

**코드잇 4기 4팀** 고급 프로젝트
**"생성모델을 활용한 소상공인 광고 컨텐츠 생성 서비스"**

AI를 활용해 소상공인이 업종에 맞는 광고 이미지와 마케팅 문구를 생성하고,
필요 시 RAG 기반 상담 답변을 받을 수 있는 웹 서비스를 제공합니다.

**프로젝트 기간**: 2025-12-29 ~ 2026-01-28 (총 31일)

---
## 결과 보고서 및 협업 일지
**결과보고서** 

[다운로드](https://naver.me/G02ABIM7)

**협업일지**

[배현석]() |
[신승목](https://github.com/shin5290/codeit_ad_smallbiz/tree/alpha/docs/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EC%8B%A0%EC%8A%B9%EB%AA%A9) |
[이유노]() |
[이현석](https://github.com/shin5290/codeit_ad_smallbiz/tree/alpha/docs/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EC%9D%B4%ED%98%84%EC%84%9D) |
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
- [핵심 플로우](#핵심-플로우)
- [개발 가이드](#개발-가이드)
- [배포](#배포)

---

## 팀 구성

| 이름 | 역할 | 담당 업무 |
|------|------|-----------|
| **진수경** | 백엔드 + 프론트엔드 | 비즈니스 로직 통합 + 백엔드 API + 프론트 |
| **이현석** | AI 이미지 | Z-Image Turbo 기반 이미지 생성 모듈 |
| **배현석** | AI 텍스트, 챗봇 | 프롬프트/문구 생성 템플릿, 상담 챗봇 |
| **이유노** | 광고 문구 | 광고 문구 파이프라인 |
| **신승목** | 인프라, 테스트 | 환경 조성, 문서화, 업종 확장 테스트 |

---

## 주요 기능

### 핵심 기능

- **LLM 기반 Intent 분석 (generation/modification/consulting)**
  - 요청 의도/스타일/비율/업종/수정 강도 자동 추출
  - SSE 스트리밍으로 진행률/결과 제공

- **RAG 기반 상담 응답 (컨설팅 모드)**
  - LangChain + Chroma + E5 임베딩 기반 사례 검색
  - Slot-Filling(업종/지역 등) 기반 질문 보완
  - 실패 시 LLM/Consultant fallback

- **AI 광고 이미지 생성 (Z-Image Turbo)**
  - Text-to-Image(T2I), Image-to-Image(I2I), 배경 제거+합성 워크플로우
  - LoRA 스타일 전환, 텍스트 오버레이, 레이아웃 분석 노드
  - 진행률 이벤트 제공

- **AI 광고 문구 생성**
  - GPT-4o-mini 기반 카피 생성
  - industries.yaml(247개 업종) 기반 톤/키워드 최적화
  - 길이/해시태그 요구 자동 반영

- **사용자 인증 및 세션 관리**
  - JWT(HttpOnly 쿠키), 회원가입/로그인/수정/탈퇴
  - 게스트 세션 → 로그인 세션 귀속

- **관리자 기능**
  - 사용자/세션/메시지/생성 이력 조회
  - 서버 로그 파일 조회/스트리밍

- **이미지 파일 관리**
  - SHA-256 해시 기반 중복 방지
  - `/images/{hash}` 서빙 + 썸네일 지원

### 지원 이미지 비율
- 1:1 (1024x1024)
- 3:4 (896x1152)
- 4:3 (1152x896)
- 16:9 (1344x768)
- 9:16 (768x1344)

---

## 기술 스택

### 프론트엔드
- **UI**: HTML/CSS/Vanilla JS
- **렌더링**: marked + DOMPurify (Markdown 안전 렌더링)
- **통신**: Fetch API + SSE 스트리밍
- **정적 파일 서빙**: FastAPI StaticFiles

### 백엔드
- **Framework**: FastAPI 0.128
- **Language**: Python 3.10+
- **ASGI Server**: Uvicorn
- **Validation**: Pydantic v2, pydantic-settings
- **Authentication**: python-jose (JWT), bcrypt, passlib

### AI 모델

**이미지 생성**
- Z-Image Turbo (8 steps 고속 생성)
- Diffusers, Transformers, PyTorch
- LoRA 스타일 전환
- rembg 기반 배경 제거

**텍스트 생성**
- OpenAI GPT-4o-mini
- industries.yaml 기반 업종별 프롬프트
- AIDA 프레임워크 기반 카피라이팅

**RAG 상담**
- LangChain
- ChromaDB 벡터스토어
- Sentence-Transformers (intfloat/multilingual-e5-large)
- 선택적 Reranker (BAAI/bge-reranker-v2-m3)

### 데이터베이스
- **RDBMS**: PostgreSQL 15
- **ORM**: SQLAlchemy 2.0
- **Migration**: Alembic

### 로깅/스토리지
- **Logs**: /mnt/logs (일자별 로그 폴더)
- **Images**: /mnt/data/uploads, /mnt/data/generated

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────┐
│                 사용자 브라우저                  │
└───────────────────────────┬──────────────────┘
                            │ HTTPS
                            ▼
┌──────────────────────────────────────────────┐
│                 FastAPI Backend              │
│  - 정적 파일 (main.html/admin.html)            │
│  - API + SSE 스트리밍                          │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│                 Routers Layer                │
│  /auth, /chat, /admin, /images               │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│                Services / Orchestration       │
│   - intent 분석 / generation / modification    │
│   - consulting 스트리밍                         │
└───────┬───────────────┬───────────────┬───────┘
        │               │               │
        ▼               ▼               ▼
┌──────────────┐  ┌───────────────┐  ┌────────────┐
│ Text Gen     │  │ Image Gen     │  │ RAG/Consult│
│ GPT-4o-mini  │  │ Z-Image Turbo │  │ LangChain  │
└──────────────┘  └───────────────┘  └────────────┘
        │               │               │
        └──────┬────────┴───────┬───────┘
               ▼                ▼
┌───────────────────────┐  ┌────────────────────┐
│ PostgreSQL(SQLAlchemy)│  │/mnt/data,/mnt/logs │
└───────────────────────┘  └────────────────────┘
```

---

## 빠른 시작

### 사전 요구사항

- **Python**: 3.10 이상
- **PostgreSQL**: 15 이상
- **NVIDIA GPU**: 선택 (이미지 생성 가속)
- **CUDA**: PyTorch 설치 버전에 맞는 12.x
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

`.env` 파일을 프로젝트 루트에 생성하고 아래 항목을 설정합니다.

```ini
DATABASE_URL=postgresql+psycopg2://aduser:your_password@localhost:5432/adbizdb
JWT_SECRET_KEY=<64자_hex_문자열>
OPENAI_API_KEY=sk-proj-...

# 선택
JWT_EXPIRE_MINUTES=60
COOKIE_SECURE=false
LOG_ROOT=/mnt/logs
```

### 5. 마이그레이션 (선택)

```bash
alembic upgrade head
```

### 6. 서버 실행

```bash
./run.sh
# 또는
uvicorn main:app --host 0.0.0.0 --port 9000
```

### 7. 접속 확인

- **서비스**: http://localhost:9000/
- **API 문서**: http://localhost:9000/docs

---

## 프로젝트 구조

```
codeit_ad_smallbiz/
├── main.py                         # FastAPI 앱 진입점
├── run.sh                          # 실행 스크립트
├── alembic/                        # DB 마이그레이션
├── requirements.txt                # Python 의존성
├── src/
│   ├── backend/
│   │   ├── models.py               # ORM 모델
│   │   ├── schemas.py              # Pydantic 스키마
│   │   ├── process_db.py           # DB CRUD
│   │   ├── services.py             # 비즈니스 로직
│   │   ├── chatbot.py              # RAG/LLM 오케스트레이터
│   │   ├── rag_preload.py          # RAG 프리로드
│   │   └── routers/                # auth/chat/admin 라우터
│   ├── generation/
│   │   ├── text_generation/        # 광고 문구 생성
│   │   │   ├── text_generator.py   # GPT-4o-mini 카피 생성
│   │   │   ├── ad_generator.py     # 광고 생성 통합 API
│   │   │   ├── prompt_manager.py   # 업종별 프롬프트 관리
│   │   │   ├── evaluate_prompt.py  # 프롬프트 평가
│   │   │   └── industries.yaml     # 247개 업종 설정
│   │   ├── image_generation/       # 이미지 생성 (Z-Image Turbo)
│   │   │   ├── generator.py        # 생성 파이프라인 진입점
│   │   │   ├── workflow.py         # 노드 워크플로우
│   │   │   ├── config.py           # 이미지 생성 설정
│   │   │   ├── preload.py          # 모델 프리로드
│   │   │   ├── shared_cache.py     # 모델 캐시 관리
│   │   │   ├── nodes/              # T2I/I2I/레이아웃/저장 노드
│   │   │   ├── prompt/             # 프롬프트/스타일 설정
│   │   │   └── tools/              # 폰트/텍스트 레이아웃 도구
│   │   └── chat_bot/               # RAG 기반 상담 챗봇
│   │       ├── rag/                # SmallBizRAG 체인
│   │       ├── agent/              # TrendAgent / SmallBizConsultant
│   │       ├── refine/             # Self-Refine
│   │       ├── api/                # 상담 API 엔드포인트
│   │       ├── config/             # RAG/Agent 설정
│   │       ├── data/               # 문서/벡터스토어 데이터
│   │       └── evaluation/         # 평가 스크립트
│   ├── utils/
│   │   ├── config.py               # 환경 설정
│   │   ├── security.py             # JWT/bcrypt
│   │   ├── session.py              # 세션 관리
│   │   ├── image.py                # 이미지 처리/서빙
│   │   ├── logging.py              # 로깅 설정
│   │   ├── admin_logs.py           # 로그 조회 유틸
│   │   └── intent_keywords.py      # Intent 키워드 매핑
│   └── frontend/
│       ├── main.html               # 메인 페이지
│       ├── admin.html              # 관리자 페이지
│       └── static/                 # CSS/JS/이미지
└── docs/
```

---

## API 문서

서버 실행 후 http://localhost:9000/docs 에서 전체 API 문서를 확인할 수 있습니다.

### 1. 인증 API (`/auth/*`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/auth/me` | 현재 로그인 사용자 정보 |
| POST | `/auth/signup` | 회원가입 |
| POST | `/auth/login` | 로그인 (OAuth2 폼) |
| POST | `/auth/logout` | 로그아웃 |
| PUT | `/auth/user` | 회원정보 수정 |
| DELETE | `/auth/user` | 회원 탈퇴 |

### 2. 채팅 API (`/chat/*`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/chat/message/stream` | 챗봇 메시지 (SSE 스트리밍) |
| POST | `/chat/session` | 세션 조회/생성 |
| GET | `/chat/history` | 유저 대화 히스토리 (페이징) |

### 3. 이미지 API (`/images/*`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/images/{file_hash}` | 이미지 파일 서빙 (size=thumb 지원) |

### 4. 관리자 API (`/admin/*`)

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/admin/users` | 사용자 목록 조회 | 관리자 |
| POST | `/admin/users/delete` | 사용자 일괄 삭제 | 관리자 |
| GET | `/admin/sessions` | 세션 목록 조회 | 관리자 |
| GET | `/admin/sessions/{session_id}` | 세션 상세 조회 | 관리자 |
| GET | `/admin/generations` | 생성 이력 조회 | 관리자 |
| GET | `/admin/logs/dates` | 로그 날짜 목록 | 관리자 |
| GET | `/admin/logs/files` | 로그 파일 목록 | 관리자 |
| GET | `/admin/logs/tail` | 로그 tail 조회 | 관리자 |
| GET | `/admin/logs/full` | 로그 전체 조회 | 관리자 |
| GET | `/admin/logs/current` | 현재 로그 정보 | 관리자 |
| GET | `/admin/logs/stream/current` | 현재 로그 스트림 | 관리자 |

---

## 핵심 플로우

### 1. 챗봇 메시지 처리 (SSE)

```
사용자 메시지
   │
   ▼
/chat/message/stream
   │
   ├─ ingest_user_message()
   │   - 세션 확보/생성
   │   - 이미지 저장
   │   - 히스토리 저장
   │
   ├─ analyze_intent()
   │   - generation/modification/consulting
   │   - 스타일/비율/업종/강도 추출
   │
   ├─ consulting → RAG 스트리밍
   ├─ generation → generate_contents()
   └─ modification → target 이력 선택 + generate_contents()

   └─ persist_generation_result()
       - 이미지/히스토리/생성 이력 저장
```

### 2. 이미지 생성 파이프라인

```
PromptProcessorNode
   │
   ├─ Text2ImageNode / Image2ImageNode
   ├─ (선택) BackgroundRemoval + Composite
   ├─ (선택) GPTLayoutAnalyzer + TextOverlay
   └─ SaveImageNode
```

---

## 개발 가이드

### 코드 스타일

- **스타일**: PEP 8
- **Import 순서**: 표준 라이브러리 → 서드파티 → 로컬
- **Import 경로**: 절대 경로 사용 (`from src.backend import ...`)

```python
from typing import Optional
from fastapi import FastAPI
from src.backend.services import register_user
from src.utils.config import settings
```

### 환경 설정

```python
from src.utils.config import settings

settings.DATABASE_URL
settings.OPENAI_API_KEY
```

---

## 배포

- **단일 FastAPI 서버**가 프론트 정적 파일과 API를 함께 제공합니다.
- 기본 실행 포트는 9000이며 필요 시 변경 가능합니다.

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

---

## 라이선스

이 프로젝트는 **Apache License 2.0** 하에 배포됩니다.

---

**CodeIt 4기 4팀** | 2025-12-29 ~ 2026-01-28
