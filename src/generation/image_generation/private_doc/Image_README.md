# Image Generation Module

## 📌 개요

소상공인을 위한 광고 이미지를 자동으로 생성하는 모듈입니다.
ComfyUI 스타일의 노드 기반 아키텍처를 채택하여 유연하고 확장 가능한 이미지 생성 파이프라인을 구축합니다.

---

## 🎯 주요 목표

1. **자동 이미지 생성**: 텍스트 프롬프트 기반 광고 이미지 생성
2. **노드 기반 워크플로우**: 유연한 전처리/생성/후처리 파이프라인
3. **스타일별 최적화**: Ultra Realistic, Semi Realistic, Anime 스타일 지원
4. **다양한 비율 지원**: 1:1, 3:4, 4:3, 16:9, 9:16 해상도 템플릿
5. **멀티 모델 시스템**: 스타일별 전문 체크포인트 모델 자동 전환

---

## 🏗️ 아키텍처

### **노드 기반 시스템 (ComfyUI 스타일)**

```
사용자 입력
    ↓
[전처리 노드들] (향후 구현)
    - 배경 제거 노드
    - 밝기 조정 노드
    - 품질 분석 노드
    ↓
[생성 노드] ✅ 구현 완료
    - Text2ImageNode (SDXL)
    - 멀티 모델 지원 (RealVisXL, Equinox, Animagine)
    - 로컬 캐싱 및 자동 언로드
    ↓
[후처리 노드들] (향후 구현)
    - 리사이즈 노드
    - 텍스트 오버레이 노드
    - 압축 노드
    ↓
최종 이미지 출력
```

### **핵심 컴포넌트**

1. **BaseNode** (`nodes/base.py`) ✅
   - 모든 노드의 추상 베이스 클래스
   - `process()` 메서드 정의
   - 입력/출력 표준화
   - 메타데이터 자동 추적 (실행 시간, 상태, 에러)

2. **ImageGenerationWorkflow** (`workflow.py`) ✅
   - 노드들을 연결하여 실행
   - 동적 워크플로우 구성
   - 메타데이터 수집 및 리포트
   - 에러 핸들링

3. **Text2ImageNode** (`nodes/generation.py`) ✅
   - SDXL 파이프라인 lazy loading
   - 멀티 모델 지원 (model_id 파라미터)
   - 로컬 캐싱 (models/ 폴더)
   - 자동 언로드 (메모리 관리)
   - Variant fallback (fp16 미지원 모델 대응)

4. **UnifiedImageGenerator** (`generator.py`) 🚧
   - 외부(Backend)에서 호출하는 메인 인터페이스
   - 자동 모델 선택 로직
   - 입력 분석 및 워크플로우 생성

---

## 🔧 기술 스택

### **모델 시스템**

#### **현재 사용 중인 모델**
1. **Ultra Realistic**: SG161222/RealVisXL_V4.0 (~6.5GB)
   - 포토리얼리즘 전문 모델
   - 제빵소, 바리스타, 헤어샵 등 실사 이미지

2. **Semi Realistic**: John6666/bss-equinox-il-semi-realistic-model-v25-sdxl (~6.5GB)
   - 균형잡힌 리얼리즘
   - 꽃집, 서점 등 일반적인 광고 이미지

3. **Anime**: cagliostrolab/animagine-xl-3.1 (~6.5GB)
   - 애니메이션 스타일 전문
   - 캐릭터 일러스트, 캐주얼한 분위기

#### **공통 VAE**
- **madebyollin/sdxl-vae-fp16-fix**: 품질 개선 및 메모리 효율화

### **메모리 관리**
- ✅ L4 22GB GPU에서 안정적 동작
- ✅ 로컬 캐싱으로 재다운로드 방지
- ✅ 자동 언로드로 모델 교체 시 메모리 최적화
- ✅ Variant fallback으로 호환성 보장

### **의존성**
```
diffusers
transformers
accelerate
safetensors
peft  # LoRA 지원용
pillow
opencv-python (향후)
numpy
rembg (향후)
```

---

## 📁 폴더 구조

```
src/generation/image_generation/
├── __init__.py
├── private_doc/
│   └── Image_README.md              # 이 문서
├── config.py                        # ✅ 모델/생성 설정
├── generator.py                     # 🚧 UnifiedImageGenerator 메인 클래스
├── workflow.py                      # ✅ ImageGenerationWorkflow
├── nodes/
│   ├── __init__.py
│   ├── base.py                      # ✅ BaseNode 추상 클래스
│   ├── generation.py                # ✅ Text2ImageNode
│   ├── preprocessing.py             # 🚧 전처리 노드들
│   └── postprocessing.py            # 🚧 후처리 노드들
├── models/                          # 로컬 모델 캐시 (gitignore)
│   ├── SG161222--RealVisXL_V4.0/
│   ├── John6666--bss-equinox-il-semi-realistic-model-v25-sdxl/
│   ├── cagliostrolab--animagine-xl-3.1/
│   └── stabilityai--stable-diffusion-xl-base-1.0/
├── test_images/                     # 테스트 결과물
└── test_workflow.py                 # ✅ 테스트 스크립트
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

config.py의 INDUSTRY_STYLES:
- **카페**: 따뜻한 조명, 아늑한 분위기, 커피 컵
- **음식점**: 우아한 다이닝, 음식 프레젠테이션
- **소매업**: 깔끔한 디스플레이, 밝은 조명
- **서비스업**: 전문적, 모던한 인테리어

### **3. Negative Prompt 최적화**

손가락 품질 개선:
```python
NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, ugly, deformed, bad anatomy, "
    "bad hands, extra fingers, missing fingers, fused fingers, too many fingers, "
    "mutated hands, poorly drawn hands, malformed limbs, "
    "watermark, text overlay, signature, logo, amateur photo, "
    "low resolution, oversaturated colors, cartoon, anime style, "
    "3d render, plastic looking, artificial"
)
```

---

## 🔄 워크플로우 예시

### **기본 Text2Image 워크플로우**

```python
from workflow import ImageGenerationWorkflow
from nodes.generation import Text2ImageNode

# Ultra Realistic 스타일
workflow = ImageGenerationWorkflow(name="AdGeneration")
workflow.add_node(Text2ImageNode(
    model_id="SG161222/RealVisXL_V4.0",
    auto_unload=True
))

result = workflow.run({
    "prompt": "professional bakery interior, fresh croissants and bread",
    "aspect_ratio": "4:3",
    "num_inference_steps": 40,
    "guidance_scale": 8.0,
    "seed": 1000
})

# result["image"]: PIL.Image
# result["seed"]: 사용된 시드
# result["width"], result["height"]: 해상도
```

### **스타일별 자동 모델 선택**

```python
# Anime 스타일
workflow = ImageGenerationWorkflow(name="AnimeAd")
workflow.add_node(Text2ImageNode(
    model_id="cagliostrolab/animagine-xl-3.1",
    auto_unload=True
))

result = workflow.run({
    "prompt": "anime style character illustration of cheerful barista",
    "aspect_ratio": "3:4",
})
```

---

## 🎯 Backend API 연동 인터페이스

### **현재 사용 가능한 인터페이스**

```python
from workflow import ImageGenerationWorkflow
from nodes.generation import Text2ImageNode
from config import generation_config

# 워크플로우 생성
workflow = ImageGenerationWorkflow(name="AdGeneration")
workflow.add_node(Text2ImageNode(
    model_id="SG161222/RealVisXL_V4.0",  # 스타일에 따라 선택
    auto_unload=True
))

# 이미지 생성
result = workflow.run({
    "prompt": image_prompt,  # 배현석님의 PromptTemplateManager에서 생성
    "aspect_ratio": aspect_ratio,
    "negative_prompt": generation_config.NEGATIVE_PROMPT
})

# 결과 사용
image = result["image"]  # PIL.Image
seed = result["seed"]
width = result["width"]
height = result["height"]
```

### **입력 형식**
```python
{
    "prompt": str,                    # 필수: 생성할 이미지 설명
    "aspect_ratio": str,              # 기본: "1:1"
    "negative_prompt": str,           # 기본: config.NEGATIVE_PROMPT
    "num_inference_steps": int,       # 기본: 40
    "guidance_scale": float,          # 기본: 7.5
    "seed": Optional[int],            # 재현성 위해 (None이면 랜덤)
    "industry": Optional[str],        # 업종 프리셋 적용
}
```

### **출력 형식**
```python
{
    "image": PIL.Image,         # 생성된 이미지 객체
    "seed": int,               # 사용된 시드값
    "width": int,              # 이미지 너비
    "height": int,             # 이미지 높이
}
```

---

## 📊 성능 및 메모리

### **생성 속도**
- **모든 스타일 (40 steps)**: ~15-20초 (L4 GPU 기준)
- 스타일에 관계없이 일정한 속도

### **메모리 사용량**
- **모델 로드**: 약 6-7GB VRAM
- **이미지 생성**: 추가 2-3GB VRAM
- **총**: 약 10GB (L4 22GB에서 안정적)
- **자동 언로드**: 생성 완료 후 즉시 메모리 해제

### **로컬 캐싱**
- 모델은 `models/` 폴더에 저장
- 재실행 시 다운로드 없이 즉시 로드
- 약 20GB 디스크 공간 사용 (3개 모델 + VAE)

---

## 🚀 테스트 스크립트

### **test_workflow.py**

9가지 테스트 케이스:
- **Ultra Realistic** (3): 베이커리, 바리스타, 헤어샵
- **Semi Realistic** (3): 꽃집, 꽃집 직원, 서점
- **Anime** (3): 카페, 바리스타, 제빵사

각 스타일별로 자동으로 모델 전환하며 테스트:
```bash
python test_workflow.py
```

결과는 `test_images/` 폴더에 저장됨

---

## 📝 개발 진행 상황

### **✅ 완료**
- [x] SDXL 모델 테스트 및 확정
- [x] FLUX vs SDXL 비교 (SDXL 선택)
- [x] config.py 작성 (해상도 템플릿, negative prompt, 업종 프리셋)
- [x] nodes/base.py (BaseNode + NodeMetadata)
- [x] workflow.py (ImageGenerationWorkflow + 메타데이터 수집)
- [x] nodes/generation.py (Text2ImageNode + 멀티 모델)
- [x] 로컬 모델 캐싱 시스템
- [x] 자동 언로드 메모리 관리
- [x] Variant fallback 처리
- [x] 테스트 스크립트 (9개 케이스)

### **🚧 진행 중**
- [ ] generator.py (UnifiedImageGenerator) - 백엔드 연동용
- [ ] nodes/preprocessing.py (배경 제거, 이미지 품질 분석)
- [ ] nodes/postprocessing.py (텍스트 오버레이, 압축)

### **📋 계획**
- [ ] Backend API 통합 테스트
- [ ] 프롬프트 최적화 (배현석님 TextGenerator 연동)
- [ ] 이미지 저장 로직 (신승목님 storage 연동)
- [ ] 에러 처리 강화
- [ ] 성능 모니터링

---

## 🔜 향후 확장 계획

1. **전처리 노드**: 배경 제거, 밝기 조정
2. **후처리 노드**: 텍스트 오버레이, 압축, 워터마크
3. **LoRA 추가**: 특정 스타일 강화
4. **Upscale**: 고해상도 출력
5. **캐싱**: 자주 사용하는 프롬프트 결과 캐싱
6. **모니터링**: 생성 시간, 메모리 사용량 추적

---

## 👥 담당자

**이현석** - 이미지 생성 모듈 전체 담당

---

**최종 수정일**: 2026-01-05
