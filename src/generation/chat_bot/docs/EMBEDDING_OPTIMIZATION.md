# E5 임베딩 모델 최적화 기록

> 이미지 생성 모델과 함께 운영 시 VRAM 부족 문제를 해결하기 위한 임베딩 모델 최적화 과정을 기록합니다.

---

## 1. 문제 상황

### 환경
- **GPU**: NVIDIA L4 (VRAM 23GB)
- **이미지 생성 모델**: ~20GB VRAM 사용
- **E5 임베딩 모델**: intfloat/multilingual-e5-large (~2.2GB VRAM)

### 문제
```
이미지 모델 (20GB) + 임베딩 모델 (2.2GB) = 22.2GB
가용 VRAM: 23GB
여유: 0.8GB → 메모리 파편화로 OOM 발생
```

실제 `nvidia-smi` 기록:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA L4                      On  |   00000000:00:03.0 Off |                    0 |
| N/A   65C    P0             35W /   72W |   22308MiB /  23034MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

---

## 2. 검토한 옵션

| 옵션 | VRAM | Recall@1 | Latency | 비고 |
|------|------|----------|---------|------|
| **GPU (FP32, 기본)** | 2.2GB | 0.8533 | ~25ms | 기준 |
| **GPU (FP16 양자화)** | 1.1GB | 0.8533 | ~25ms | 정확도 유지 |
| **GPU (INT8 양자화)** | ~0.6GB | 0.84 (추정) | ~25ms | 1-2% 정확도 감소 가능 |
| **OpenAI API (text-embedding-3-small)** | 0GB | 0.79 | ~500ms | 정확도 6%p 감소, 벡터스토어 재구축 필요 |
| **CPU** | 0GB | 0.8533 | ~200ms | 정확도 유지, latency 증가 |

### 각 옵션 상세 분석

#### FP16 양자화
- VRAM 2.2GB → 1.1GB (50% 감소)
- 남은 여유: ~1.9GB
- **문제**: 메모리 파편화 환경에서 여전히 OOM 위험

#### INT8 양자화
- VRAM 2.2GB → ~0.6GB (73% 감소)
- 남은 여유: ~2.4GB
- **문제**: 정확도 1-2% 감소 가능, 여전히 파편화 위험

#### OpenAI API
- VRAM 0GB (완전 해제)
- **문제**:
  - Recall@1: 0.8533 → 0.79 (**6%p 하락**)
  - 100번 질문 중 6번 더 잘못된 문서 검색
  - 벡터스토어 재구축 필요 (임베딩 차원 다름)

#### CPU
- VRAM 0GB (완전 해제)
- 정확도 유지 (동일 모델)
- Latency 증가: ~25ms → ~200ms
- **장점**: 벡터스토어 재구축 불필요, 코드 1줄 수정

---

## 3. 실험 환경

### 하드웨어
- **GPU**: NVIDIA L4 (VRAM 23GB)
- **CPU**: Intel Xeon (GCP 환경)
- **RAM**: 충분

### 소프트웨어
- Python 3.10
- sentence-transformers
- PyTorch 2.x

### 모델
- `intfloat/multilingual-e5-large` (560M parameters)

---

## 4. GPU 벤치마크 결과

```
device=cuda, batch_size=10, batches=10
avg: 0.025s, p95: 0.040s, min/max: 0.023/0.035
```

| 메트릭 | 값 |
|--------|-----|
| 평균 | 25ms |
| p95 | 40ms |
| 최소 | 23ms |
| 최대 | 35ms |

**결론**: GPU 사용 시 매우 빠름 (~25ms). 하지만 VRAM 2.2GB 필요.

---

## 5. CPU 벤치마크 결과

### 5.1 테스트 쿼리 설계

#### Short Texts (온라인 쿼리 시나리오)
```python
short_texts = [
    "연남 카페 추천해줘",
    "강남역 맛집 어디야?",
    "비건 베이커리 있나요?",
    "여의도 회식 장소 추천",
    "파스타 맛있는 데 알려줘",
] * 20  # 100개 샘플
```

**선택 이유**:
1. **지역 + 업종 조합**: "연남 카페", "강남역 맛집" 등 실제 검색 패턴
2. **다양한 업종**: 카페, 맛집, 베이커리, 회식 장소, 파스타 등
3. **길이**: 10~20자 내외 (실제 모바일 입력 길이)

#### Long Texts (오프라인 문서 임베딩 시나리오)
```python
long_texts = [
    (
        "이 매장은 브런치와 스페셜티 커피를 함께 제공하며, 평일 오후에 방문객이 많고 "
        "주차 공간이 협소합니다. 시그니처 메뉴로는 수제 치아바타 샌드위치와 싱글 오리진 핸드드립이 있습니다. "
        "주말에는 예약을 권장하며, 반려동물 동반이 가능합니다."
    )
] * 20  # 긴 문장 샘플
```

**선택 이유**:
1. **문서 임베딩 시나리오**: 벡터스토어 구축 시 문서 청크 길이와 유사
2. **실제 매장 정보 구조**: 메뉴, 특징, 주차, 예약 정보 포함
3. **길이**: ~150자 (실제 문서 청크 평균 길이)

### 5.2 스레드 수 (threads) 비교

| threads | short, bs=1, conc=1 | 개선율 |
|---------|---------------------|--------|
| 1 | 328ms | 기준 |
| **2** | **187ms** | **43% 개선** |
| 4 | 182ms | 44% (2와 거의 동일) |

**분석**:
- threads=1 → 2: 성능 2배 가까이 개선
- threads=2 → 4: 추가 개선 거의 없음 (CPU 코어 간 오버헤드)
- **threads=4 문제**: long + concurrency 높을 때 급격히 느려짐 (173초, 318초)

### 5.3 배치 크기 (batch_size) 비교

| batch_size | 총 시간 | 문장당 시간 |
|------------|---------|-------------|
| 1 | 185ms | 185ms |
| 4 | 313ms | **78ms** |
| 8 | 423ms | **53ms** |
| 16 | 598ms | **37ms** |

**분석**:
- 배치가 클수록 문장당 처리 효율 향상
- 하지만 온라인 쿼리는 1개씩 들어오므로 batch_size=1이 현실적
- 마이크로배치로 여러 요청을 모으면 효율 개선 가능

### 5.4 동시성 (concurrency) 비교

| concurrency | threads=2, short, bs=1 | 증가율 |
|-------------|------------------------|--------|
| 1 | 187ms | 기준 |
| 5 | 771ms | 4.1배 |
| 10 | 1,550ms | 8.3배 |

**분석**:
- 동시 요청 증가 시 latency 선형 이상으로 증가
- CPU 경쟁으로 모든 요청이 느려짐
- **결론**: 동시 encode 금지, 큐잉으로 순차 처리 필요

### 5.5 전체 결과 (threads=2)

#### Short Texts
```
[short]
  bs= 1 conc= 1 | avg=0.187s p95=0.194s min/max=0.181/0.195 | per_sentence≈187.1ms
  bs= 1 conc= 5 | avg=0.771s p95=0.815s min/max=0.727/0.836 | per_sentence≈770.5ms
  bs= 1 conc=10 | avg=1.550s p95=1.594s min/max=1.438/1.598 | per_sentence≈1549.6ms
  bs= 4 conc= 1 | avg=0.349s p95=0.418s min/max=0.304/0.430 | per_sentence≈87.2ms
  bs= 8 conc= 1 | avg=0.422s p95=0.450s min/max=0.400/0.454 | per_sentence≈52.7ms
```

#### Long Texts
```
[long]
  bs= 1 conc= 1 | avg=0.456s p95=0.482s min/max=0.434/0.485 | per_sentence≈456.2ms
  bs= 1 conc= 5 | avg=2.315s p95=2.398s min/max=2.204/2.406 | per_sentence≈2315.0ms
  bs= 4 conc= 1 | avg=1.174s p95=1.193s min/max=1.154/1.196 | per_sentence≈293.5ms
  bs= 8 conc= 1 | avg=2.104s p95=2.111s min/max=2.099/2.113 | per_sentence≈263.0ms
```

---

## 6. 실험 코드

```python
import os
import time
import statistics
import torch
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
DEVICE = "cpu"  # 또는 "cuda"

THREADS = [1, 2, 4]          # OMP/torch 스레드 수 sweep
BATCH_SIZES = [1, 4, 8]      # 배치 크기 sweep
CONCURRENCY = [1, 5, 10]     # 동시에 들어오는 요청 수 가정
NUM_BATCHES = 3              # 각 조합당 반복 횟수

# 테스트 쿼리 (온라인 시나리오)
short_texts = [
    "연남 카페 추천해줘",
    "강남역 맛집 어디야?",
    "비건 베이커리 있나요?",
    "여의도 회식 장소 추천",
    "파스타 맛있는 데 알려줘",
] * 20  # 100개 샘플

# 테스트 문서 (오프라인 시나리오)
long_texts = [
    (
        "이 매장은 브런치와 스페셜티 커피를 함께 제공하며, 평일 오후에 방문객이 많고 "
        "주차 공간이 협소합니다. 시그니처 메뉴로는 수제 치아바타 샌드위치와 싱글 오리진 핸드드립이 있습니다. "
        "주말에는 예약을 권장하며, 반려동물 동반이 가능합니다."
    )
] * 20


def percentile(values, p):
    """p in [0,100]. Safe percentile for small samples."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(xs) else f
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def benchmark_single(model, texts, batch_size):
    """단일 배치 벤치마크"""
    batch = texts[:batch_size]
    prefixed = ["query: " + t for t in batch]

    start = time.perf_counter()
    _ = model.encode(prefixed, normalize_embeddings=True)
    elapsed = time.perf_counter() - start

    return elapsed


def benchmark_concurrent(model, texts, batch_size, concurrency, num_batches):
    """동시성 벤치마크"""
    times = []

    def worker():
        return benchmark_single(model, texts, batch_size)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(num_batches):
            futures = [executor.submit(worker) for _ in range(concurrency)]
            batch_times = [f.result() for f in futures]
            times.extend(batch_times)

    return times


def run_benchmark():
    cases = [("short", short_texts), ("long", long_texts)]

    for num_threads in THREADS:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)

        print(f"\n=== threads={num_threads} ===")

        model = SentenceTransformer(MODEL, device=DEVICE)

        for case_name, texts in cases:
            print(f"[{case_name}]")

            for bs in BATCH_SIZES:
                for conc in CONCURRENCY:
                    times = benchmark_concurrent(model, texts, bs, conc, NUM_BATCHES)

                    avg = statistics.mean(times)
                    p95 = percentile(times, 95)
                    mn, mx = min(times), max(times)
                    per = avg * 1000 / bs  # ms per sentence

                    print(
                        f"  bs={bs:2d} conc={conc:2d} | "
                        f"avg={avg:.3f}s p95={p95:.3f}s min/max={mn:.3f}/{mx:.3f} | "
                        f"per_sentence≈{per:.1f}ms"
                    )


if __name__ == "__main__":
    run_benchmark()
```

---

## 7. 최종 결정: CPU 전환

### 선택 이유

1. **정확도 유지**: OpenAI API 전환 시 R@1이 0.8533 → 0.79로 6%p 하락
   - 100번 질문 중 6번 더 잘못된 문서 검색
   - RAG에서 잘못된 문서 = 잘못된 답변

2. **VRAM 완전 해제**:
   - FP16(1.1GB), INT8(0.6GB)도 메모리 파편화 환경에서 OOM 위험
   - CPU는 VRAM 0GB로 완전 해결

3. **허용 가능한 latency**:
   - 임베딩: ~25ms → ~200ms (+175ms)
   - LLM 응답: 3~5초 (전체의 70%)
   - 체감 영향 적음

4. **작업량 최소**:
   - 벡터스토어 재구축 불필요
   - 코드 1줄 수정으로 적용

### 코드 변경

```python
# rag/chain.py - E5Embeddings
self.model = SentenceTransformer(model_name, device="cpu")  # VRAM 0GB
```

---

## 8. CPU 성능 최적화: 큐잉 + 마이크로배치

### 설계 원칙

| 원칙 | 설명 | 근거 |
|------|------|------|
| **동시 encode 금지** | Lock으로 CPU 경쟁 방지 | conc=5 → 4배 느림 |
| **큐잉** | 순차 처리로 예측 가능한 latency | 안정성 |
| **마이크로배치** | 50ms 대기 후 배치 처리 | 문장당 효율 향상 |
| **p95 기준** | 95% 사용자 응답 시간 보장 | 서비스 SLA |

### 최적 설정

| 설정 | 값 | 이유 |
|------|-----|------|
| **threads** | 2 | 벤치마크 결과 최적 (4 이상은 오버헤드) |
| **batch_wait_ms** | 50ms | 요청 모으는 대기 시간 |
| **max_batch_size** | 8 | 배치당 최대 쿼리 수 |

### 예상 성능

| 시나리오 | batch_size | p95 latency | per_sentence |
|----------|------------|-------------|--------------|
| 단일 요청 | 1 | 200ms | 200ms |
| 마이크로배치 | 4 | 342ms | **87ms** |
| 마이크로배치 | 8 | 470ms | **53ms** |

### 구현

```python
# rag/chain.py - E5Embeddings

# CPU 스레드 최적화
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)

class E5Embeddings:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",
        batch_wait_ms: int = 50,      # 50ms 대기
        max_batch_size: int = 8,       # 최대 8개 배치
        enable_micro_batch: bool = True,
    ):
        # Lock으로 동시 encode 방지
        self._lock = threading.Lock()

        # 마이크로배치 워커 스레드
        if enable_micro_batch:
            self._batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self._batch_thread.start()
```

---

## 9. 결론

### Before vs After

| 항목 | Before (GPU) | After (CPU + 최적화) |
|------|-------------|---------------------|
| VRAM | 2.2GB | **0GB** |
| Recall@1 | 0.8533 | **0.8533** (유지) |
| Latency (단일) | ~25ms | ~200ms |
| Latency (마이크로배치) | - | ~53ms/문장 |
| OOM 위험 | 있음 | **없음** |

### 핵심 트레이드오프

```
GPU (25ms) vs CPU (200ms) → +175ms latency
하지만 LLM 응답이 3~5초이므로 전체의 5% 미만 영향
```

### 최종 설정

```python
E5Embeddings(
    device="cpu",           # VRAM 0GB
    batch_wait_ms=50,       # 50ms 대기 후 배치 처리
    max_batch_size=8,       # 최대 8개 묶어서 처리
    enable_micro_batch=True # 마이크로배치 활성화
)
```

---

**작성일**: 2025-01-22
**담당자**: 배현석
**환경**: GCP NVIDIA L4 (23GB VRAM)
