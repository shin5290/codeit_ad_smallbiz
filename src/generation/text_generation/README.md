# 텍스트 생성 모듈 (광고 문구 + 이미지 프롬프트)

**작성자**: 배현석
**버전**: 1.0
**담당**: 광고 문구 생성 + 이미지 프롬프트 생성

---

## 목차

1. [개요](#개요)
2. [주요 기능](#주요-기능)
3. [파일 구조](#파일-구조)
4. [설치 및 설정](#설치-및-설정)
5. [사용법](#사용법)
6. [백엔드 통합 가이드](#백엔드-통합-가이드)
7. [테스트](#테스트)
8. [API 호출 부분 (JupyterHub 테스트용)](#api-호출-부분-jupyterhub-테스트용)

---

## 개요

이 모듈은 **소상공인 광고 생성 서비스**의 핵심 기능을 담당합니다:

1. **광고 문구 생성** (GPT-4o-mini)
2. **이미지 생성 프롬프트 작성** (한글 → 영어 키워드 추출 → SDXL 프롬프트)

---

## 주요 기능

### 1. 광고 문구 생성 (`text_generator.py`)

- GPT-4o-mini API를 사용하여 20자 이내 광고 문구 생성
- 4가지 톤 앤 매너 지원: `warm`, `professional`, `friendly`, `energetic`
- 후처리 및 Fallback 로직 포함

### 2. 이미지 프롬프트 생성 (`prompt_manager.py`)

- 한글 사용자 입력 → 영어 키워드 추출 (GPT-4o-mini)
- 10개 업종 자동 감지: `cafe`, `gym`, `bakery`, `restaurant` 등
- Positive/Negative 프롬프트 자동 생성

### 3. 통합 함수 (`ad_generator.py`)

백엔드가 호출할 **단일 함수**:

```python
generate_advertisement(user_input, tone, max_length, style)
```

**반환값**:
```python
{
    "ad_copy": "따뜻한 겨울, 새로운 맛",           # 광고 문구 [1개]
    "positive_prompt": "Professional food...",   # 이미지 프롬프트 [1/2]
    "negative_prompt": "cartoon, blurry...",     # 이미지 프롬프트 [2/2]
    "industry": "cafe",                          # 업종 [1개]
    "status": "success"
}
```

---

## 파일 구조

```
src/generation/text_generation/
├── ad_generator.py           # 🔥 백엔드가 호출할 메인 함수
├── text_generator.py         # 광고 문구 생성
├── prompt_manager.py         # 키워드 추출 + 프롬프트 생성
├── prompt_templates.py       # SDXL Hybrid Prompting 시스템
├── config_loader.py          # YAML 기반 프롬프트 생성기
├── industries.yaml           # 10개 업종 템플릿 (Civitai 벤치마크 기반)
├── test_basic.py             # API 연결 테스트
├── test_integration.py       # 통합 테스트
└── README.md                 # 이 문서
```

---

## 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install openai python-dotenv pyyaml
```

### 2. `.env` 파일 설정

프로젝트 루트에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. API 연결 테스트

```bash
cd src/generation/text_generation
python test_basic.py
```

**예상 출력**:
```
✅ API 키 로드 성공
✅ GPT 응답 성공
🎉 모든 테스트 통과! API 연결 정상 작동합니다.
```

---

## 사용법

### 기본 사용법 (Python)

```python
from src.generation.text_generation.ad_generator import generate_advertisement

# 광고 생성
result = generate_advertisement(
    user_input="카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌, 겨울",
    tone="warm",           # optional (기본값: "warm")
    max_length=20,         # optional (기본값: 20)
    style="realistic"      # optional (기본값: "realistic")
)

# 결과 확인
print(result)
```

**출력 예시**:
```python
{
    "ad_copy": "따뜻한 겨울, 새로운 맛",
    "positive_prompt": "Professional food photography of strawberry latte on marble table, minimalist cafe interior with natural light, soft natural window light streaming from left, warm pastel pink and beige tones, overhead shot, professional food photography, creamy foam texture, delicate latte art, 85mm lens, f/1.8 aperture, bokeh background",
    "negative_prompt": "cartoon, illustration, painting, low quality, artificial, plastic-looking",
    "industry": "cafe",
    "status": "success"
}
```

### 파라미터 설명

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `user_input` | str | ✅ | - | 한글 사용자 요청 (예: "카페 신메뉴 딸기라떼 홍보") |
| `tone` | str | ❌ | "warm" | 광고 문구 톤 (`warm`, `professional`, `friendly`, `energetic`) |
| `max_length` | int | ❌ | 20 | 광고 문구 최대 길이 (10~30 권장) |
| `style` | str | ❌ | "realistic" | 이미지 스타일 (`realistic`, `anime` 등) |

---

## 백엔드 통합 가이드

### 백엔드 요구사항

백엔드(진수경)에게 전달할 데이터:

1. **텍스트 생성** [1개]: `ad_copy`
2. **프롬프트 생성** [2개]: `positive_prompt`, `negative_prompt`
3. **업종** [1개]: `industry`

### `services.py` 통합 예제

```python
from src.generation.text_generation.ad_generator import generate_advertisement

def create_advertisement(user_input: str):
    """광고 생성 API 엔드포인트"""

    # 1. 배현석 파트 호출 (텍스트 + 프롬프트 생성)
    result = generate_advertisement(
        user_input=user_input,
        tone="warm",
        max_length=20
    )

    if result["status"] != "success":
        # 실패 시 에러 반환
        return {"error": result.get("error", "Unknown error")}

    # 2. 이현석님한테 프롬프트 전달 (이미지 생성)
    image_result = generate_image_with_leehs(
        positive_prompt=result["positive_prompt"],
        negative_prompt=result["negative_prompt"]
    )

    # 3. 최종 결과 통합
    return {
        "ad_copy": result["ad_copy"],
        "positive_prompt": result["positive_prompt"],
        "negative_prompt": result["negative_prompt"],
        "industry": result["industry"],
        "image_path": image_result["path"],  # 이현석님 결과
        "status": "success"
    }
```

---

## 테스트

### 1. 통합 테스트 실행

```bash
cd src/generation/text_generation
python test_integration.py
```

**예상 출력**:
```
🧪 백엔드 통합 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 백엔드 요구사항:
   - 텍스트 생성: ad_copy [1개]
   - 프롬프트 생성: positive_prompt, negative_prompt [2개]
   - 업종: industry [1개]

테스트 케이스 1/3: 카페 - 딸기라떼
✅ 성공!

{
    "ad_copy": "따뜻한 겨울, 새로운 맛",
    "positive_prompt": "Professional food photography of strawberry...",
    "negative_prompt": "cartoon, illustration, low quality...",
    "industry": "cafe",
    "status": "success"
}

🎉 모든 테스트 통과!
✅ 백엔드 통합 준비 완료!
```

### 2. 개별 모듈 테스트

**광고 문구만 테스트**:
```bash
python text_generator.py
```

**프롬프트만 테스트**:
```bash
python prompt_manager.py
```

---

## API 호출 부분 (JupyterHub 테스트용)

⚠️ **중요**: 실제 GPT API는 **JupyterHub에서만 호출**하세요!

### JupyterHub에서 테스트 방법

```python
# JupyterHub 노트북에서 실행

import sys
sys.path.append('/path/to/project')  # 프로젝트 경로 추가

from src.generation.text_generation.ad_generator import generate_advertisement

# 광고 생성 (실제 API 호출)
result = generate_advertisement(
    user_input="카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
    tone="warm"
)

print(result)
```

### API 호출 위치

| 파일 | 라인 | API 호출 내용 |
|------|------|--------------|
| `text_generator.py` | 53~61 | GPT-4o-mini 호출 (광고 문구 생성) |
| `prompt_manager.py` | 73~81 | GPT-4o-mini 호출 (키워드 추출) |

**주석 처리 예시** (로컬 테스트 시):
```python
# 53~61 라인을 주석 처리하고 더미 데이터 사용
# response = self.client.chat.completions.create(...)
ad_copy = "테스트 광고 문구"  # 더미 데이터
```

---

## 지원 업종

현재 10개 업종 지원 (`industries.yaml`):

1. **cafe** - 카페/커피숍
2. **gym** - 헬스장/피트니스
3. **bakery** - 베이커리/제과점
4. **restaurant** - 레스토랑/식당
5. **hair_salon** - 미용실
6. **nail_salon** - 네일샵
7. **flower_shop** - 꽃집
8. **clothing_store** - 옷가게
9. **laundry** - 세탁소
10. **general** - 일반 업종

업종은 사용자 입력에서 **자동 감지**됩니다.

---

## 문제 해결

### 1. API 키 오류

```
❌ OPENAI_API_KEY가 설정되지 않았습니다.
```

**해결**: `.env` 파일 확인 및 API 키 재설정

### 2. YAML 로드 실패

```
⚠️ industries.yaml 로드 실패
```

**해결**: `industries.yaml` 파일 경로 확인
```python
# config_loader.py:288
generator = PromptGenerator(config_path="src/generation/text_generation/industries.yaml")
```

### 3. Import 오류

```
ModuleNotFoundError: No module named 'text_generator'
```

**해결**: Python 경로 확인
```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
```

---

## 다음 단계

1. **백엔드 통합** (진수경): `services.py`에서 `generate_advertisement()` 호출
2. **이미지 생성 통합** (이현석): `positive_prompt`, `negative_prompt` 받아서 이미지 생성
3. **JupyterHub 테스트**: 실제 API로 end-to-end 테스트

---

## 연락처

- **작성자**: 배현석
- **담당**: 광고 문구 생성 + 이미지 프롬프트 생성
- **문의**: 문제 발생 시 팀 채널에 공유

---

**마지막 업데이트**: 2026-01-11
