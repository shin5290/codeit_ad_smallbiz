# Image Generation Module

## 📌 개요

소상공인을 위한 광고 이미지를 자동으로 생성하는 모듈입니다.
ComfyUI 스타일의 노드 기반 아키텍처를 채택하여 유연하고 확장 가능한 이미지 생성 파이프라인을 구축합니다.

---

## 🎯 주요 목표

1. **자동 이미지 생성**: 텍스트 프롬프트 기반 광고 이미지 생성
2. **노드 기반 워크플로우**: 유연한 전처리/생성/후처리 파이프라인
3. **업종별 최적화**: 카페, 음식점, 소매업 등 업종별 스타일 프리셋
4. **다양한 비율 지원**: 1:1, 3:4, 4:3, 16:9, 9:16 해상도 템플릿

---

## 🏗️ 아키텍처

### **노드 기반 시스템 (ComfyUI 스타일)**

```
사용자 입력
    ↓
[전처리 노드들]
    - 배경 제거 노드
    - 밝기 조정 노드
    - 품질 분석 노드
    ↓
[생성 노드]
    - Text2Image 노드 (SDXL)
    - ControlNet 노드 (옵션)
    ↓
[후처리 노드들]
    - 리사이즈 노드
    - 텍스트 오버레이 노드
    - 압축 노드
    ↓
최종 이미지 출력
```

### **핵심 컴포넌트**

1. **BaseNode** (`nodes/base.py`)
   - 모든 노드의 추상 베이스 클래스
   - `process()` 메서드 정의
   - 입력/출력 표준화

2. **ImageGenerationWorkflow** (`workflow.py`)
   - 노드들을 연결하여 실행
   - 동적 워크플로우 구성
   - 중간 결과 확인 가능

3. **UnifiedImageGenerator** (`generator.py`)
   - 외부(Backend)에서 호출하는 메인 인터페이스
   - 자동 모델 선택 로직
   - 입력 분석 및 워크플로우 생성

---

## 🔧 기술 스택

### **모델**
- **Primary**: SDXL (stabilityai/stable-diffusion-xl-base-1.0)
- **VAE**: madebyollin/sdxl-vae-fp16-fix (품질 개선)
- **ControlNet**: diffusers/controlnet-canny-sdxl-1.0 (구조 유지)

### **이유**
- ✅ L4 22GB GPU에서 안정적 동작 (~7-8GB VRAM)
- ✅ 적당한 이미지 생성
- ✅ 풍부한 ControlNet 지원
- ✅ 검증된 안정성

### **의존성**
```
diffusers
transformers
accelerate
safetensors
pillow
opencv-python
numpy
rembg
```

---

## 📁 폴더 구조

```
src/generation/image_generation/
├── __init__.py
├── Image_README.md              # 이 문서
├── config.py                    # 모델/생성 설정
├── generator.py                 # UnifiedImageGenerator 메인 클래스
├── workflow.py                  # ImageGenerationWorkflow
├── nodes/
│   ├── __init__.py
│   ├── base.py                  # BaseNode 추상 클래스
│   ├── generation.py            # Text2ImageNode, ControlNetNode
│   ├── preprocessing.py         # 전처리 노드들
│   ├── postprocessing.py        # 후처리 노드들
│   └── controlnet_prep.py       # ControlNet 전처리 노드
├── utils.py                     # 헬퍼 함수
└── test_model*.py               # 테스트 스크립트
```

---

## 🎨 주요 기능

### **1. 해상도 템플릿**

사용자가 용도에 맞는 비율 선택:

| 비율 | 해상도 | 용도 |
|------|--------|------|
| 1:1 | 1024x1024 | SNS 프로필, 썸네일 |
| 3:4 | 896x1152 | Instagram 피드, 포스터 |
| 4:3 | 1152x896 | 프레젠테이션, 배너 |
| 16:9 | 1344x768 | 유튜브 썸네일, 웹 배너 |
| 9:16 | 768x1344 | Instagram Story, 모바일 |

### **2. 업종별 스타일 프리셋**

- **카페**: 따뜻한 조명, 아늑한 분위기, 커피 컵
- **음식점**: 우아한 다이닝, 음식 프레젠테이션
- **소매업**: 깔끔한 디스플레이, 밝은 조명
- **서비스업**: 전문적, 모던한 인테리어

### **3. 자동 모델 선택**

```python
입력 분석:
- 이미지 없음 → Text2Image (컨셉 이미지)
- 이미지 있음 → ControlNet (제품 구조 유지, 스타일만 변경)
```

---

## 🔄 워크플로우 예시

### **기본 Text2Image 워크플로우**

```python
workflow = ImageGenerationWorkflow()
workflow.add_node(Text2ImageNode(model="sdxl"))
workflow.add_node(ResizeNode(ratio="16:9"))
workflow.add_node(TextOverlayNode(text="특별 할인!"))

result = workflow.execute({
    "prompt": "professional coffee shop advertisement",
    "industry": "cafe"
})
```

### **ControlNet 워크플로우 (제품 이미지 변환)**

```python
workflow = ImageGenerationWorkflow()
workflow.add_node(RemoveBackgroundNode())
workflow.add_node(AdjustBrightnessNode(factor=1.2))
workflow.add_node(CannyEdgeNode())  # ControlNet 전처리
workflow.add_node(ControlNetNode(model="canny"))
workflow.add_node(CompressNode(quality=95))

result = workflow.execute({
    "prompt": "professional product photo, studio lighting",
    "input_image": user_product_image,
    "industry": "retail"
})
```

---

## 🎯 Backend API 연동 인터페이스

### **입력 형식**
```python
{
    "prompt": str,                    # 사용자 프롬프트
    "image": Optional[bytes],         # 입력 이미지 (ControlNet용)
    "industry": str,                  # 업종 (cafe, restaurant, retail, service)
    "aspect_ratio": str,              # 비율 (1:1, 3:4, 4:3, 16:9, 9:16)
    "style": Optional[str],           # 추가 스타일 키워드
    "overlay_text": Optional[str],    # 오버레이 텍스트
}
```

### **출력 형식**
```python
{
    "image_url": str,           # GCS 저장 경로
    "method": str,              # "t2i" or "controlnet"
    "metadata": {
        "model": str,
        "steps": int,
        "guidance_scale": float,
        "resolution": tuple,
        "generation_time": float,
    }
}
```

---

## 🚀 사용 예시

```python
from image_generation.generator import UnifiedImageGenerator

# 초기화 (한 번만)
generator = UnifiedImageGenerator(model_type="sdxl")

# 텍스트→이미지 생성
result = generator.generate(
    prompt="cozy coffee shop interior with latte art",
    industry="cafe",
    aspect_ratio="1:1",
)

# 제품 이미지 변환 (ControlNet)
result = generator.generate(
    prompt="professional product photo, clean background",
    input_image=product_image,
    industry="retail",
    aspect_ratio="4:3",
)
```

---

## 📊 성능 목표

- **생성 시간**: ~30-60초 (SDXL 40 steps 기준)
- **품질**: 상업적 사용 가능 수준
- **VRAM**: ~7-8GB (L4 GPU에서 안정적)
- **확장성**: 새로운 노드 추가 용이

---

## 🔜 향후 확장 계획

1. **LoRA 지원**: 특정 스타일 강화
2. **IP-Adapter**: 참조 이미지 스타일 전이
3. **Upscale 노드**: 고해상도 출력
4. **A/B 테스트**: 여러 버전 동시 생성
5. **캐싱**: 자주 사용하는 프롬프트 결과 캐싱

---

## 📝 개발 진행 상황

- [x] SDXL 모델 테스트 및 확정
- [x] config.py 작성 (해상도 템플릿, 설정)
- [ ] nodes/base.py (BaseNode)
- [ ] workflow.py (ImageGenerationWorkflow)
- [ ] nodes/generation.py (Text2ImageNode)
- [ ] nodes/preprocessing.py
- [ ] nodes/postprocessing.py
- [ ] generator.py (UnifiedImageGenerator)
- [ ] Backend API 통합 테스트

---

## 👥 담당자

**이현석** - 이미지 생성 모듈 전체 담당

---

**최종 수정일**: 2025-12-31
