# Final Project Architecture

프로젝트 목표: 생성형 AI 기술을 활용하여 소상공인이 광고 컨텐츠를 손쉽게 제작할 수 있도록 서비스를 개발

## **Q. 소상공인이 컨텐츠를 제작할 때 겪는 어려움이 무엇인가?**

1. **시간 부족 (1순위)**

매장 운영, 재고 관리 등 핵심 업무로 인해 마케팅에 할애할 시간이 부족

직원이 없거나 적어서 모든 걸 직접 해야하는 상황

1. **비용 부담 (2순위)**

전문 디자이너나 마케팅 대행사를 고용하기에는 비용이 너무 높음

디지털 마케팅 도구 도입 비용도 부담

1. **기술과 지식 부족 (3순위)**

디지털 마케팅 자체를 어디서부터 시작해야할지 모름

스마트폰이나 컴퓨터 활용 능력이 부족한 경우가 많음 (중장년층)

디자인 툴 사용법을 배우는 것 자체가 진입장벽

무엇을 만들어야 효과적인지 판단하기 어려움

1. **컨텐츠 품질 판단 어려움 (4순위)**

트렌드를 따라가기가 힘듬

만든 결과물이 좋은지 스스로 판단하기 어려움

## **타겟 선정 전략**

업종별 분포:
음식점/카페, 도소매업, 생활서비스업, 숙박업 등

1. **음식점/카페**
장점: 수가 가장 많아 시장 규모가 크다, 비주얼 마케팅 수요가 높다, 온라인 마케팅 전환율이 높다
단점: 경쟁이 치열하다, 이미 관련 서비스가 많다.
2. **오프라인 중심 + 온라인 마케팅 미경험자**
(소규모 제조업체, 동내 소매점-꽃집, 철물점 등, 생활 서비스업-미용실, 세탁소 등)
장점: 블루오션(경쟁이 낮다), 디지털 전환 지원 정책 혜택 대상, 니즈는 있지만 해결책이 없는 시장
단점: 디지털 리터러시가 낮아 교육이 필요할 수 있다, 시장 규모가 상대적으로 작을 수 있다.

## **기존 도구들과의 차별화(Canva 고려)**

1. 완전 자동화 (Canva는 직접 사용자가 편집해야 함)
2. 브랜드 일관성 자동 유지 (Canva 같은 경우 유료 서비스)
3. 한국 소상공인 맞춤 기능 - 한글 폰트 최적화, 한국 식당/카페 트렌드 반영 템플릿, 네이버 블로그, 인스타그램 등 한국 플랫폼 최적화 사이즈

즉, Canva는 쉬운 디자인 틀, 우리는 디자인이 필요 없는 자동화 서비스

## **팀에게 제안**

4주라는 제한된 시간동안 가장 좋은 서비스를 만들고 싶기에,

대상을 제한하되 퀄리티를 높이면 어떨까? - 카페/음식점 (가능하다면 온라인 마케팅 미경험자 추가)

**Q. 카페/음식점인 이유?**

**1. 4주라는 시간 제약**

모든 업종 커버하려면 프롬프트 템플릿만 최소 30-50개 필요/ 각 업종별 테스트/검증 시간 부족

결과: 어디에도 제대로 안 맞는 서비스

**2. 음식점/카페의 명확한 니즈**

- 신메뉴 출시 → SNS 홍보 이미지 (주 1-2회)
- 이벤트/할인 → 배너 이미지 (월 2-3회)
- 일상 포스팅 → 감성 이미지 (주 3-5회)

→ 패턴이 명확하고 반복적 = AI 자동화에 최적

**3. 레퍼런스가 풍부**

인스타그램 #카페스타그램 #먹스타그램에 수백만 개 이미지

학습 데이터 풍부 = AI 품질 높음

**4. 빠른 검증 가능**

주변 카페/음식점에 직접 테스트 가능

실시간 피드백 → 즉시 개선

**5. 확장 용이**

잘 만들어진 음식점 서비스는 다른 업종으로 확장 쉬움

예: 음식점 → 미용실 (비슷한 구조)

## **구현했으면 하는 기능**

1. 초간단 입력(시간 절약) - 광고 문구 또는 요청사항에 맞춰 이미지 제작
2. 음식점/카페에 특화된 AI 프롬프트
3. 브랜드 일관성 자동 유지
4. 메뉴 기반 자동 키워드 추천
5. 오늘의 추천
6. 챗봇 서비스 - 대화 관리, 의도 파악, 제안 생성 

## **서비스 아키텍처 (Streamlit 활용)**

```python
┌─────────────────────────────────────────────────┐
│          사용자 (소상공인)                         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│   Frontend (Streamlit) - UI ONLY ⭐              │
│                                                  │
│   파일: 단 3개!                                   │
│   - app.py (메인)                                │
│   - ui_components.py (모든 UI 컴포넌트)           │
│   - backend_client.py (API 호출)                 │
│                                                  │
│   역할: 화면만 담당!                              │
│   - 버튼 누르면 → API 호출                        │
│   - API 응답 받으면 → 화면에 표시                 │
│   - 로직 없음! 순수 UI만!                         │
└──────────────────┬──────────────────────────────┘
                   │ HTTP REST API
                   │ (완전 분리!)
                   ▼
┌─────────────────────────────────────────────────┐
│   Backend (FastAPI) - 모든 로직 ⭐                │
│                                                  │
│   ┌────────────────────────────────────────┐   │
│   │  API Layer                             │   │
│   │  - main.py                             │   │
│   │  - routes.py (모든 엔드포인트)          │   │
│   └────────────┬───────────────────────────┘   │
│                │                                │
│   ┌────────────▼───────────────────────────┐   │
│   │  Business Logic Layer                  │   │
│   │  - services.py (비즈니스 로직 통합)     │   │
│   └────────────┬───────────────────────────┘   │
│                │                                │
│   ┌────────────▼───────────────────────────┐   │
│   │  Integration Layer                     │   │
│   │  - image_pipeline.py (이미지 생성)      │   │
│   │  - prompt_engine.py (프롬프트)          │   │
│   └────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ GPU    │ │ OpenAI │ │   DB   │
    │ Server │ │  API   │ │        │
    └────────┘ └────────┘ └────────┘
```

## 프로젝트 구조

```python
cafe-ai-marketing/
│
├── frontend/                     # ① Frontend (Streamlit UI만)
│   ├── app.py                    # 메인 앱
│   ├── ui_components.py          # 모든 UI 컴포넌트 통합! 
│   └── backend_client.py         # API 호출 클라이언트 
│
├── backend/                      # ② Backend (FastAPI 전체)
│   ├── main.py                   # FastAPI 앱 
│   ├── routes.py                 # 모든 API 엔드포인트 
│   ├── services.py               # 비즈니스 로직 통합 
│   ├── models.py                 # DB 모델 
│   └── schemas.py                # Pydantic 스키마 
│
├── models/                       # ③④ AI 모델
│   ├── image_pipeline.py         # 이미지 생성 통합 
│   ├── prompt_engine.py          # 프롬프트 & 텍스트 통합 
│   ├── preprocessor.py           # 전처리 통합 
│   └── postprocessor.py          # 후처리 통합
│
├── utils/                        # ⑤ 공통 유틸 (그대로 유지)
│   ├── database.py
│   ├── storage.py
│   ├── validators.py
│   └── config.py
│
├── deployment/                   # ⑤ 배포 (그대로 유지)
│   ├── docker/
│   │   ├── Dockerfile.frontend
│   │   ├── Dockerfile.backend
│   │   └── docker-compose.yml
│   └── scripts/
│       └── deploy.sh
│
├── tests/                        # ⑤ 테스트 (그대로 유지)
│   ├── test_api.py
│   ├── test_models.py
│   └── test_integration.py
│
├── requirements.txt
├── .env
└── README.md
```

## 역할 분배

### ① Frontend 담당 (1명)

**담당:**

```python
frontend/
├── app.py                    # 메인 앱
├── ui_components.py          # 모든 UI 컴포넌트
└── backend_client.py         # API 클라이언트
```

**주요 작업:**

1. 화면 그리기
2. 버튼, 입력창, 이미지 표시
3. API 호출 후 결과 표시

**산출물:**

- ✅ `app.py` (100줄) - Streamlit 메인 앱
- ✅ `ui_components.py` (200줄) - 모든 UI 컴포넌트 통합
    - render_sidebar() - 브랜드 설정 UI
    - render_input_form() - 입력 폼
    - render_result() - 결과 표시
    - render_chatbot() - 챗봇 UI
    - render_history() - 이력 페이지
- ✅ `backend_client.py` (100줄) - FastAPI 통신 클라이언트
    - generate_image() - 이미지 생성 요청
    - chat() - 챗봇 대화
    - get_history() - 이력 조회
    - get_recommendations() - 오늘의 추천

**기술 스택:**

필수:

- Python 3.9+
- Streamlit 1.30+
- requests (HTTP 클라이언트)

선택:

- streamlit-chat (챗봇 UI, 선택사항)
- Pillow (이미지 표시)

```toml
# frontend/app.py (완전히 단순화!)

import streamlit as st
from backend_client import BackendClient
from ui_components import (
    render_sidebar,
    render_input_form,
    render_result,
    render_chatbot
)

st.set_page_config(title="카페 AI 마케팅", layout="wide")

# API 클라이언트
api = BackendClient("http://localhost:8000")

# 사이드바
render_sidebar(api)

# 메인
st.title("☕ 카페 AI 마케팅")

# 오늘의 추천
recommendations = api.get_recommendations()
for rec in recommendations:
    st.info(f"{rec['title']}: {rec['suggestion']}")

# 입력 폼
user_request = render_input_form()

# 생성 버튼
if st.button("✨ 생성"):
    with st.spinner("생성 중..."):
        result = api.generate_image(user_request)
        render_result(result)
```

```toml
# frontend/ui_components.py (모든 컴포넌트 통합!)

import streamlit as st

def render_sidebar(api):
    """사이드바 (브랜드 설정)"""
    with st.sidebar:
        st.header("🎨 브랜드 설정")
        
        logo = st.file_uploader("로고 업로드")
        if logo:
            api.upload_logo(logo)
            st.success("✅ 저장!")
        
        primary_color = st.color_picker("주 색상")
        # ...

def render_input_form():
    """입력 폼"""
    user_request = st.text_area(
        "광고 내용",
        placeholder="예: 딸기라떼 신메뉴"
    )
    return user_request

def render_result(result):
    """결과 표시"""
    st.image(result["image_url"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**키워드:**", ", ".join(result["keywords"]))
    with col2:
        st.write("**해시태그:**", " ".join(result["hashtags"]))

def render_chatbot(api):
    """챗봇 UI"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if user_input := st.chat_input("질문하기"):
        response = api.chat(user_input)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["message"]
        })
```

### ② Backend 담당 (1명)

**역할:** "모든 AI를 하나로 묶는 접착제 + 챗봇"

**담당:**

```python
backend/
├── main.py          # FastAPI 앱
├── routes.py        # 모든 API 엔드포인트
├── services.py      # 비즈니스 로직 (③④ 통합)
├── models.py        # DB 모델
└── schemas.py       # 요청/응답 스키마
```

**주요 작업:**

1. FastAPI 서버 전체
2. 모든 비즈니스 로직
3. ③④ 통합 레이어
4. DB 관리

**예시 코드:**

```python
# backend/main.py (매우 단순!)

from fastapi import FastAPI
from routes import router

app = FastAPI(title="Cafe AI Marketing API")
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"status": "ok"}
```

```python
# backend/routes.py (모든 엔드포인트 통합!)

from fastapi import APIRouter, BackgroundTasks
from schemas import GenerateRequest, GenerateResponse, StatusResponse
from services import ImageGenerationService

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate_image(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """이미지 생성 요청"""
    service = ImageGenerationService()
    task_id = service.create_task(request)
    
    background_tasks.add_task(
        service.process_generation,
        task_id
    )
    
    return {"task_id": task_id, "status": "pending"}

@router.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """상태 확인"""
    service = ImageGenerationService()
    return service.get_status(task_id)

@router.post("/chat")
async def chat(request: ChatRequest):
    """챗봇"""
    service = ChatService()
    return service.chat(request.message)

@router.get("/recommendations")
async def get_recommendations():
    """오늘의 추천"""
    service = RecommendationService()
    return service.get_recommendations()

@router.get("/history/{user_id}")
async def get_history(user_id: str):
    """생성 이력"""
    service = HistoryService()
    return service.get_history(user_id)
```

```python
# backend/services.py (비즈니스 로직 통합!)

from models.image_pipeline import ImagePipeline
from models.prompt_engine import PromptEngine
from utils.database import Database

class ImageGenerationService:
    """이미지 생성 비즈니스 로직"""
    
    def __init__(self):
        self.image_pipeline = ImagePipeline()  # ③
        self.prompt_engine = PromptEngine()    # ④
        self.db = Database()
        self.tasks = {}  # 실제론 Redis
    
    def create_task(self, request):
        """작업 생성"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {"status": "pending"}
        return task_id
    
    async def process_generation(self, task_id):
        """실제 생성 처리"""
        try:
            self.tasks[task_id]["status"] = "processing"
            
            # 1. 프롬프트 생성 (④)
            prompts = self.prompt_engine.generate(
                user_request=request.user_request
            )
            
            # 2. 이미지 생성 (③)
            image = await self.image_pipeline.generate(
                prompt=prompts["prompt"]
            )
            
            # 3. 키워드 (④)
            keywords = self.prompt_engine.generate_keywords(
                request.user_request
            )
            
            # 4. 저장
            image_url = self.storage.save(image)
            self.db.save_generation(...)
            
            # 5. 완료
            self.tasks[task_id] = {
                "status": "completed",
                "image_url": image_url,
                "keywords": keywords
            }
            
        except Exception as e:
            self.tasks[task_id] = {
                "status": "failed",
                "error": str(e)
            }
    
    def get_status(self, task_id):
        """상태 조회"""
        return self.tasks.get(task_id)

class ChatService:
    """챗봇 서비스"""
    def chat(self, message):
        # GPT 호출
        pass

class RecommendationService:
    """오늘의 추천"""
    def get_recommendations(self):
        # 날짜/요일 기반 추천
        pass
```

**산출물:**

- ✅ `main.py` (50줄) - FastAPI 앱 초기화
- ✅ `routes.py` (200줄) - 모든 API 엔드포인트
    - POST /api/v1/generate - 이미지 생성 요청
    - GET /api/v1/status/{task_id} - 작업 상태 확인
    - POST /api/v1/chat - 챗봇 대화
    - GET /api/v1/recommendations - 오늘의 추천
    - GET /api/v1/history/{user_id} - 생성 이력
    - POST /api/v1/brand/logo - 로고 업로드
- ✅ `services.py` (300줄) - 비즈니스 로직 통합
    - ImageGenerationService - ③④ 통합 + 워크플로우
    - ChatService - 챗봇 로직
    - RecommendationService - 추천 알고리즘
    - StorageService - 파일 저장 (S3/로컬)
- ✅ `models.py` (100줄) - SQLAlchemy DB 모델
    - User, ImageGeneration, ChatHistory, Brand
- ✅ `schemas.py` (100줄) - Pydantic 스키마
    - 요청/응답 검증
- ✅ API 문서 - Swagger 자동 생성 (/docs)

**기술 스택:**

핵심:

- Python 3.9+
- FastAPI 0.100+
- SQLAlchemy 2.0+ (ORM)
- Pydantic 2.0+ (검증)
- PostgreSQL 15+

추가:

- asyncio (비동기 처리)
- Redis (작업 큐, 선택사항)
- Celery (백그라운드 작업, 선택사항)
- uvicorn (ASGI 서버)

개발 도구:

- pytest (테스트)
- Black (코드 포매터)
- mypy (타입 체크)

### **③ 이미지 생성 담당 (1명)**

**담당:**

```python
models/
├── image_pipeline.py        # 통합 파이프라인
├── preprocessor.py          # 전처리 통합
└── postprocessor.py         # 후처리 통합
```

**주요 작업:**

1. SDXL/Flux/ControlNet 통합
2. 자동 모델 선택
3. 전처리/후처리

**예시 코드:**

```python
# models/image_pipeline.py (모든 파이프라인 통합!)

from diffusers import StableDiffusionXLPipeline
import torch

class ImagePipeline:
    """이미지 생성 통합 파이프라인"""
    
    def __init__(self):
        self._load_models()
    
    def _load_models(self):
        """모델 로드"""
        self.sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Flux, ControlNet도 여기서 로드
    
    def select_model(self, prompt: str):
        """자동 모델 선택"""
        if "고품질" in prompt or "professional" in prompt:
            return "flux"
        elif "배경" in prompt or "스타일" in prompt:
            return "controlnet"
        return "sdxl"  # 기본
    
    async def generate(self, prompt: str, negative_prompt: str = ""):
        """이미지 생성"""
        
        # 1. 모델 선택
        model = self.select_model(prompt)
        
        # 2. 전처리 (필요시)
        from preprocessor import preprocess
        prompt = preprocess(prompt)
        
        # 3. 생성
        if model == "sdxl":
            image = self.sdxl(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50
            ).images[0]
        
        # 4. 후처리
        from postprocessor import postprocess
        image = postprocess(image)
        
        return image
```

**산출물:**

- ✅ `image_pipeline.py` (300줄) - 통합 이미지 생성 파이프라인
    - ImagePipeline 클래스
        - select_model() - 자동 모델 선택 알고리즘
        - generate() - 메인 생성 함수
        - _load_models() - 모델 초기화
    - 3개 파이프라인 구현:
        - SDXL (기본)
        - Flux.1 (고품질)
        - ControlNet (입력 이미지 기반)
- ✅ `preprocessor.py` (100줄) - 전처리 모듈 통합
    - remove_background() - 배경 제거
    - adjust_brightness() - 밝기 자동 조정
    - analyze_quality() - 품질 분석
    - detect_faces() - 얼굴 감지 (선택사항)
- ✅ `postprocessor.py` (100줄) - 후처리 모듈 통합
    - resize_image() - 1080x1080 리사이징
    - insert_logo() - 로고 자동 삽입
    - add_text_overlay() - 텍스트 오버레이
    - optimize_quality() - 품질 최적화
    - compress() - 용량 압축
- ✅ 모델 선택 로직 문서
- ✅ 성능 벤치마크 결과 (모델별 생성 시간)

**기술 스택:**

핵심:

- PyTorch 2.0+
- Diffusers 0.25+ (HuggingFace)
- CUDA 11.8+ / cuDNN 8+
- transformers (모델 로드)

모델:

- Stable Diffusion XL 1.0 (기본)
- Flux.1-dev (고품질, 선택사항)
- ControlNet (Canny, Depth, Pose)

이미지 처리:

- Pillow (PIL)
- OpenCV (cv2)
- rembg (배경 제거)
- numpy

GPU 최적화:

- torch.compile (선택사항)
- xformers (메모리 최적화)
- bitsandbytes (양자화, 선택사항)

하드웨어 요구사항:

- GPU: NVIDIA RTX 3090 이상 (24GB VRAM 권장)
- RAM: 32GB 이상
- Storage: 50GB+ (모델 저장)
- 

### ④ 텍스트 & 프롬프트 담당 (1명)

**담당:**

```python
models/
└── prompt_engine.py         # 모든 프롬프트/텍스트 통합
```

**주요 작업:**

1. 이미지 프롬프트 생성
2. GPT 텍스트 생성
3. 키워드/해시태그

```python
# models/prompt_engine.py (모든 기능 통합!)

import openai
from datetime import datetime

class PromptEngine:
    """프롬프트 & 텍스트 생성 통합"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self._load_templates()
    
    def _load_templates(self):
        """템플릿 로드"""
        self.keywords = {
            "라떼": ["부드러운", "크리미한"],
            "딸기": ["상큼한", "달콤한"],
            # ...
        }
    
    def generate(self, user_request: str):
        """이미지 프롬프트 생성"""
        
        # 1. 기본 프롬프트
        prompt = f"Professional food photography of {user_request}, "
        
        # 2. 메뉴 타입 감지
        menu_type = self._detect_menu_type(user_request)
        if menu_type == "커피":
            prompt += "ceramic cup, latte art, "
        
        # 3. 계절 키워드
        season = self._get_season()
        if season == "봄":
            prompt += "cherry blossom pink, fresh spring, "
        
        # 4. 카페 분위기
        prompt += "Korean cafe aesthetic, warm lighting, high quality"
        
        negative = "blurry, low quality, ugly"
        
        return {
            "prompt": prompt,
            "negative_prompt": negative
        }
    
    def generate_keywords(self, user_request: str):
        """키워드 자동 추천"""
        keywords = []
        
        # 메뉴에서 키워드 추출
        for menu, kws in self.keywords.items():
            if menu in user_request:
                keywords.extend(kws)
        
        # 계절 키워드
        season = self._get_season()
        if season == "봄":
            keywords.extend(["봄의", "신선한"])
        
        return keywords
    
    def generate_hashtags(self, keywords: list):
        """해시태그 생성"""
        hashtags = ["#카페", "#카페스타그램"]
        for kw in keywords[:5]:
            hashtags.append(f"#{kw}")
        return hashtags
    
    def generate_ad_copy(self, menu_name: str):
        """GPT 광고 카피"""
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"{menu_name}에 대한 광고 카피 3가지"
            }]
        )
        return response.choices[0].message.content
    
    def _detect_menu_type(self, text):
        """메뉴 타입 감지"""
        if any(k in text for k in ["라떼", "커피"]):
            return "커피"
        return "음료"
    
    def _get_season(self):
        """현재 계절"""
        month = datetime.now().month
        if 3 <= month <= 5:
            return "봄"
        elif 12 <= month or month <= 2:
            return "겨울"
        return "여름"
```

**산출물:**

- ✅ `prompt_engine.py` (300줄) - 통합 프롬프트 엔진
    - PromptEngine 클래스
        - generate() - 이미지 프롬프트 생성 (메인)
        - generate_keywords() - 키워드 자동 추천
        - generate_hashtags() - 해시태그 생성
        - generate_ad_copy() - GPT 광고 카피
        - _detect_menu_type() - 메뉴 타입 감지
        - _get_season() - 계절 자동 감지
        - _load_templates() - 템플릿 로드
- ✅ `templates/` - 프롬프트 템플릿 DB
    - cafe_keywords.json - 카페 키워드 150개+
    - menu_types.json - 메뉴 타입별 키워드
    - seasonal_keywords.json - 계절별 키워드
    - style_templates.json - 스타일별 템플릿
    - text_templates.json - 텍스트 생성 템플릿
- ✅ 프롬프트 품질 가이드 문서
- ✅ 키워드 DB 관리 가이드

**기술 스택:**

핵심:

- Python 3.9+
- OpenAI API (GPT-5-mini 또는 GPT-4)
- JSON (템플릿 관리)

텍스트 처리:

- re (정규표현식)
- datetime (계절 판단)
- typing (타입 힌트)

선택사항:

- pandas (키워드 분석, 선택사항)
- konlpy (한글 형태소 분석, 선택사항)

### ⑤ Infrastructure 담당 (1명)

**담당:**

```python
utils/              # 공통 유틸
deployment/         # Docker, Cloud
tests/              # 테스트
```

**주요 작업:**

1. **데이터베이스 관리**
2. **파일 저장 (로컬/S3)**
3. **오늘의 추천 (기능 5)**
4. **검증 함수**
5. **배포 설정**

**산출물:**

1. **Docker 관련 (deployment/docker/)**
- ✅ `Dockerfile.frontend` - Streamlit 컨테이너
- ✅ `Dockerfile.backend` - FastAPI 컨테이너
- ✅ `Dockerfile.gpu` - GPU 서버 컨테이너 (선택)
- ✅ `docker-compose.yml` - 전체 서비스 오케스트레이션
- ✅ `.dockerignore` - 불필요 파일 제외

**2. 배포 스크립트 (deployment/scripts/)**

- ✅ `setup_gpu.sh` - GPU 서버 초기 셋업
- ✅ `deploy_frontend.sh` - Streamlit Cloud 배포
- ✅ `deploy_backend.sh` - Cloud Run 배포
- ✅ `migrate_db.sh` - DB 마이그레이션

**3. CI/CD (.github/workflows/)**

- ✅ `test.yml` - 자동 테스트
- ✅ `deploy.yml` - 자동 배포
- ✅ `lint.yml` - 코드 품질 체크

**4. 공통 유틸 (utils/)**

- ✅ `database.py`
- ✅ `storage.py`
- ✅ `validators.py`
- ✅ `config.py`
- ✅ `logging_config.py`

**5. 테스트 (tests/)**

- ✅ `test_api.py` - API 테스트
- ✅ `test_models.py` - AI 모델 테스트
- ✅ `test_integration.py` - 통합 테스트

**기술 스택:**

Docker, Docker Compose, NVIDIA Docker
GCP (Cloud Run, Cloud Storage, Compute Engine)
PostgreSQL, Redis
GitHub Actions, pytest

## 협업 플로우
```
사용자 입력
↓
Frontend (Streamlit)
↓ HTTP: POST /api/v1/generate
Backend (FastAPI) - [routes.py](http://routes.py/)
↓
Backend - [services.py](http://services.py/)
↓
├→ prompt_engine.py (④) → 프롬프트 생성
↓
├→ image_pipeline.py (③) → 이미지 생성
↓
└→ storage + DB 저장
↓
Backend → Frontend
↓
사용자에게 표시
```
