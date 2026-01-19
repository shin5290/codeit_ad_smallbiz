# Image Generation Module (Z-Image Turbo Integration)

**Last Updated:** 2026-01-16  
**Document Status:** Production Ready

---

## 1. 개요 (Overview)

본 문서는 소상공인 광고 이미지 자동 생성을 위한 모듈의 기술 명세서입니다.
기존 **SDXL (Stable Diffusion XL)** 기반 시스템의 한계를 극복하고, 광고 이미지로서의 **설득력(Quality)**과 **운영 효율성(Efficiency)**을 동시에 확보하기 위해, 2026년 1월부로 **Z-Image Turbo (ZIT)** 기반으로 엔진을 전면 교체하였습니다.

---

## 2. 왜 SDXL에서 Z-Image Turbo로 전환했는가? (Migration Rationale)

안정적인 SDXL 파이프라인을 두고, 32.9GB에 달하는 거대 모델인 Z-Image Turbo 도입을 결정한 배경에는 다음과 같은 기술적/운영적 고민이 있었습니다.

### 2.1. 퀄리티와 프롬프트 이해도의 한계 극복
* **문제점 (SDXL):** SDXL은 생성 속도는 빠르지만, 텍스트 인코더(CLIP)의 77토큰 한계로 인해 상세한 프롬프트를 정확히 반영하지 못하는 경우가 빈번했습니다.
* **해결책 (ZIT):** **Qwen 2.5 (LLM)**를 텍스트 인코더로 사용하는 ZIT를 도입했습니다. 긴 문맥과 추상적인 묘사를 정확하게 이미지로 구현할 수 있게 되었으며, 생성 속도는 소폭 감소했으나 광고 소재로서의 가치는 비약적으로 상승했습니다.

### 2.2. "One Model, Multi-Style" 전략을 통한 운영 효율화
* **문제점 (SDXL):** 실사, 반실사, 애니메이션 등 스타일 변경 시마다 약 6GB 크기의 체크포인트 모델을 통째로 교체(Unload/Load)해야 했습니다. 이는 잦은 I/O 오버헤드와 모델 관리 복잡도를 유발했습니다.
* **해결책 (ZIT):** 단일 **Base Model (약 20.5GB)** 하나만 메모리에 상주시킨 후, 스타일 변경 시에는 수백 MB 수준의 경량 **LoRA 어댑터**만 동적으로 교체하는 구조로 변경했습니다. 이를 통해 확장성은 높이고 관리 포인트는 일원화했습니다.

### 2.3. 메모리 스와핑 방지 및 연속성 확보
* **문제점 (SDXL):** 제한된 VRAM 내에서 여러 모델을 운용하기 위해 잦은 `auto_unload`가 발생했고, 재로딩 시간으로 인해 전체 서비스 응답 속도가 저하되었습니다.
* **해결책 (ZIT):** 초기 로딩 시간(약 1~3분)을 감수하더라도, 한 번 로드된 모델을 VRAM에 상주시키는 **Singleton 패턴**을 적용했습니다. 결과적으로 첫 생성 이후에는 **대기 시간 없는 즉시 생성(Zero-Loading Time)**이 가능해져 전체 처리량(Throughput)이 향상되었습니다.

---

## 3. 인프라 최적화 및 트러블슈팅 (Troubleshooting Log)

* **환경:** GCP L4 Instance (vCPU 4 / RAM 16GB / VRAM 24GB)
* **도전 과제:** 시스템 물리 RAM(16GB)보다 훨씬 큰 모델(초기 32.9GB)을 구동하고 안정화해야 함.

### 3.1. 메모리 초과(OOM) 및 시스템 프리징 해결
* **현상:** 24GB 모델 로드 중 16GB 시스템 RAM 고갈로 프로세스 강제 종료(Kill) 또는 인스턴스 프리징(Freezing) 발생.
* **해결 과정:**
    1.  **Swap 메모리 확장:** 디스크 기반 가상 메모리를 24GB 추가 할당하여 로딩 시 발생하는 메모리 스파이크를 흡수할 버퍼 확보.
    2.  **경량화 모델 발굴 (Key Solution):** 기존 32.9GB 모델 대신, L4 VRAM(24GB)에 적재 가능한 **20.5GB BF16 Optimized 모델 (`dimitribarbot/Z-Image-Turbo-BF16`)**을 도입.
    3.  **VRAM Full Load 전략:** 느린 CPU Offload 대신 모델 전체를 VRAM에 상주(`pipe.to("cuda")`)시켜 추론 속도 극대화.
    4.  **이중 안전장치 (Fallback):** VRAM 로드 실패 시, 자동으로 `enable_model_cpu_offload()` 모드로 전환되어 시스템 다운 방지.

### 3.2. 아키텍처 호환성 문제 해결
* **현상:** ZIT 로드 시 `DeiTConfig`, `norm_type` 등의 에러 발생으로 구동 실패.
* **해결 과정:**
    * `AutoModel`, `AutoTokenizer`를 활용하여 Qwen 구조 자동 감지 로직 적용.
    * **Self-Repair 로직 구현:** 배포된 모델 설정 파일(`config.json`)에 `norm_type` 키 누락 버그 발견. 코드 실행 시 이를 감지하고 자동으로 패치(`rms_norm` 주입)하는 기능을 개발하여 영구 해결.

### 3.3. 프로덕션 레벨 안정성 확보
* **Concurrency 제어:** `threading.Lock`을 도입하여 멀티 쓰레드 환경에서 안전한 GPU 접근 보장.
* **메모리 파편화 방지:** `PYTORCH_ALLOC_CONF` 환경변수 설정으로 장시간 운용 시 메모리 조각화 완화.
* **주기적 리소스 정리:** 매 요청마다 메모리를 비우는 대신, 일정 횟수(5회)마다 가비지 컬렉션을 수행하여 성능과 안정성의 균형 확보.

---

## 4. 기술 스택 (Tech Stack)

### Current System (Z-Image Turbo)

| 구성 요소 | 상세 내용 | 비고 |
| :--- | :--- | :--- |
| **Base Model** | `dimitribarbot/Z-Image-Turbo-BF16` | 20.5GB (S3-DiT) |
| **Text Encoder** | Qwen 2.5 (LLM) | Long Prompt 지원 |
| **VAE** | SDXL Compatible VAE | Tiling/Slicing 적용 |
| **LoRA** | Custom Style LoRAs | 동적 스위칭 지원 |

### Legacy System (SDXL) - *Archived*
*(참고: 현재 사용하지 않으나 이력 관리를 위해 명시함)*

* **Ultra Realistic:** `SG161222/RealVisXL_V4.0`
* **Semi Realistic:** `John6666/bss-equinox-il-semi-realistic-model-v25-sdxl`
* **Anime:** `cagliostrolab/animagine-xl-3.1`

---

## 5. 핵심 로직 구조 (Architecture)

### Text2ImageNode (`nodes/text2image.py`)
이미지 생성의 핵심 엔진으로, 다음과 같은 4단계 프로세스로 동작합니다.

1.  **Initialize (Singleton)**
    * 전역 변수(`_GLOBAL_PIPE`)를 확인하여 파이프라인이 없을 때만 1회 로딩.
    * `_fix_zit_config()`를 통해 설정 파일의 무결성 검사 및 자동 복구.
2.  **Optimize**
    * `PYTORCH_ALLOC_CONF` 설정 및 VAE Tiling/Slicing 적용으로 메모리 효율 증대.
3.  **Style Application**
    * Base Model은 유지하고, 요청된 스타일의 LoRA만 가볍게 교체 (`unload` -> `load`).
4.  **Inference**
    * Thread Lock 하에서 안전하게 이미지 생성.
    * 주기적인 메모리 정리로 안정성 유지.