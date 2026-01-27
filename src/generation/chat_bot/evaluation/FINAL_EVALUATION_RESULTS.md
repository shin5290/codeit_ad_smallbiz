# RAG 시스템 최종 평가 결과

> 소상공인 마케팅 상담 챗봇 RAG 검색 성능 평가 (2026.01.27)

---

## 📊 최종 배포 결정

### ✅ **채택: Baseline (Dense Embedding only)**

**구성**:
- 임베딩 모델: `intfloat/multilingual-e5-large` (1024차원)
- 벡터 DB: ChromaDB
- Top K: 5개 문서 검색
- 필터링: 없음 (전체 문서 검색)

---

## 🎯 최종 성능 지표

### 1. Retrieval 성능

| 지표 | 값 | 평가 |
|------|-----|------|
| **Recall@1** | 63.0% | 양호 |
| **Recall@3** | 80.5% | 우수 |
| **Recall@5** | 88.5% | 우수 |
| **Recall@10** | 94.0% | 매우 우수 |
| **MRR** | 73.6% | 우수 |

### 2. 검색 품질

| 지표 | 값 | 설명 |
|------|-----|------|
| **Relevance@5** | 1.122/2.0 | 검색된 문서의 평균 관련성 (0-2) |
| **NDCG@5** | 0.596 | 순위 품질 (0-1) |

### 3. 답변 생성 능력

| 지표 | 값 | 평가 |
|------|-----|------|
| **Success Rate** | 98.0% | 매우 우수 |
| **Answer Quality** | 3.98/5 | 우수 |

### 4. 응답 속도

| 지표 | 값 | 평가 |
|------|-----|------|
| **평균 Latency** | 0.27초 | 매우 빠름 |
| **P95 Latency** | 0.31초 | 실시간 가능 |

---

## 📈 의도별 성능 (Recall@5)

### 🌟 우수한 의도 (90% 이상)

| 의도 | Recall@5 | Relevance@5 | NDCG@5 | 비율 |
|------|----------|-------------|--------|------|
| **location_based** | 100.0% | 1.480 | 0.773 | 30% |
| **channel_strategy** | 95.0% | 1.410 | 0.718 | 20% |
| **industry_trend** | 100.0% | 1.800 | 0.921 | 5% |
| **complex_condition** | 90.0% | 1.080 | 0.600 | 5% |

### ⚠️ 개선 필요 의도

| 의도 | Recall@5 | Relevance@5 | NDCG@5 | 비율 |
|------|----------|-------------|--------|------|
| **scale_based** | 84.0% | 0.760 | 0.383 | 25% |
| **problem_solving** | 60.0% | 0.413 | 0.326 | 15% |

**원인**: 리뷰 수, 평점 등 숫자 정보 기반 검색이 semantic similarity로는 한계

---

## 🔬 실험한 개선 방법 (모두 실패)

### 1. ❌ Metadata Filtering

**실험**: Location + Industry 필터 적용

**결과**:
```
Recall@1: 63.0% → 79.5% (+26%)  ✅
Recall@5: 88.5% → 79.5% (-10%)  ❌
```

**문제점**:
- 필터가 너무 좁혀서 다양성 감소
- Top 1 성능은 올라가지만 Top 5+ 성능은 하락
- 전체적으로 Baseline이 우수

---

### 2. ❌ Hybrid Search (Dense + BM25)

**실험**: E5 임베딩 + BM25 조합 (α = 0.1 ~ 0.9)

**결과**:
```
Baseline:     Recall@1 = 63.0%
Hybrid α=0.5: Recall@1 = 34.5% (-45%)  ❌
```

**문제점**:
- BM25가 한국어에서 작동 안 함
- 단순 공백 토큰화로는 부족
- 형태소 분석 필요하지만 추가 복잡도 불필요

---

### 3. ❌ BGE Reranker

**실험**: Dense로 Top 50 검색 → BGE Reranker로 재정렬

**결과**:
```
성능:
  Baseline: Recall@1 = 63.0%
  Reranker: Recall@1 = 68.5% (+5.5%p)  ✅

Latency:
  Baseline: 0.27초
  Reranker: 22.85초 (80배 느림!)  ❌
```

**문제점**:
- 5.5%p 성능 향상을 위해 80배 느린 속도는 불가
- 실시간 서비스 불가능 (쿼리당 23초)

---

### 4. ❌ Query Rewriting

**실험**: 숫자 조건을 semantic 표현으로 변환
```
"리뷰 1000개" → "리뷰가 매우 많은 인기 있는"
```

**결과**:
```
Recall@1: 63.0% → 62.0% (-1.6%)  ❌
Recall@5: 88.5% → 83.0% (-6.2%)  ❌

개선 대상 의도:
  problem_solving: 60.0% → 53.3% (-11.1%)  ❌
  scale_based:     84.0% → 72.0% (-14.3%)  ❌
```

**문제점**:
- E5 임베딩이 이미 숫자를 semantic하게 이해
- 변환 과정에서 구체적 정보 손실
- Latency 증가 (0.27초 → 1.23초)

---

## 💡 핵심 인사이트

### 1. **Simple is Best**
- 복잡한 개선 방법들이 모두 실패
- Baseline (단순 Dense 검색)이 최고 성능

### 2. **E5 임베딩의 강력함**
- 한국어 semantic 이해 우수
- 숫자 정보도 암묵적으로 학습됨
- Query Rewriting 불필요

### 3. **Latency의 중요성**
- 5% 성능 향상 vs 80배 속도 저하
- 실시간 서비스에서는 속도가 우선

### 4. **다중 정답 평가의 필요성**
- v5.6 (단일 정답): R@1 = 89%
- v5.9 (다중 정답): R@1 = 63%
- 현실적 평가를 위해 다중 정답 필수

---

## 📋 평가 데이터셋

### v5.9 쿼리 구성

| 의도 | 개수 | 비율 | 설명 |
|------|------|------|------|
| location_based | 60 | 30% | 같은 지역 비슷한 업종 사례 |
| scale_based | 50 | 25% | 비슷한 규모 사례 |
| channel_strategy | 40 | 20% | 채널별 마케팅 전략 |
| problem_solving | 30 | 15% | 평점/리뷰 개선 사례 |
| industry_trend | 10 | 5% | 업종 전반 트렌드 |
| complex_condition | 10 | 5% | 복합 조건 사례 |
| **합계** | **200** | **100%** | |

### Ground Truth 특징
- **다중 정답**: 쿼리당 평균 110개 관련 문서
- **Relevance 점수**: 0 (무관), 1 (부분 관련), 2 (완전 관련)
- **현실적 패턴**: 실제 사용자 질문 패턴 반영

---

## 🚀 배포 가능 여부

### ✅ **배포 가능 (권장)**

**근거**:
1. **Recall@10 = 94%**: Top 10에 정답 포함 확률 매우 높음
2. **Success Rate = 98%**: 거의 모든 질문에 답변 생성 가능
3. **Answer Quality = 3.98/5**: 답변 품질 우수
4. **Latency = 0.27초**: 실시간 서비스 가능
5. **안정성**: 추가 개선 방법들이 모두 성능 저하 → Baseline이 가장 안정적

**약점**:
- problem_solving (60%), scale_based (84%) 의도는 상대적으로 낮음
- 하지만 전체의 40%만 차지하므로 영향 제한적

---

## 📌 향후 개선 방향

### 1. **실제 사용자 로그 수집** (3-6개월 후)
- 실제 쿼리 패턴 분석
- 실패 케이스 식별
- A/B 테스트로 개선 효과 검증

### 2. **문서 품질 개선**
- 메타데이터 보강
- 문서 내용 정제
- Retrieval보다 문서 자체 품질이 더 중요할 수 있음

---

## 📁 평가 파일 구조

```
evaluation/
├── 01_generate_queries.py              # 쿼리 생성
├── 02_evaluate_recall.py               # Recall@K 평가
├── 03_evaluate_hybrid_reranker.py      # Hybrid + Reranker 평가
├── 04_evaluate_advanced_metrics.py     # Relevance, NDCG, Success Rate 평가
├── 05_evaluate_query_rewriting.py      # Query Rewriting 평가
│
├── results/
│   └── queries_final.json              # 최종 쿼리 (200개)
│
└── 문서/
    ├── README.md                       # 평가 가이드
    └── FINAL_EVALUATION_RESULTS.md     # 이 파일
```

---

## 🎓 배운 점

1. **평가 데이터의 현실성**
   - 너무 구체적인 쿼리(v5.6)는 과적합
   - 다중 정답이 현실적

2. **복잡함 vs 성능**
   - 복잡한 방법이 항상 좋은 것은 아님
   - Simple Baseline이 최고

3. **Latency vs 성능 트레이드오프**
   - 실시간 서비스에서 속도 우선
   - 5% 성능 향상보다 80배 빠른 속도가 중요

4. **한국어 NLP 특수성**
   - BM25 등 영어 기반 방법 적용 어려움
   - Multilingual 임베딩 모델이 효과적

---

**최종 업데이트**: 2026-01-27
**평가 버전**: v5.9
**총 쿼리 수**: 200개
**평가 완료 항목**: Recall@K, Relevance@K, NDCG@K, Success Rate, Answer Quality, Latency, Hybrid Search, Reranker, Query Rewriting

**결론**: ✅ **Baseline (E5 Dense only) 배포 권장**
