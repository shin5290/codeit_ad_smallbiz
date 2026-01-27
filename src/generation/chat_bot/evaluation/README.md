# RAG Evaluation Framework

> 소상공인 마케팅 상담 챗봇을 위한 RAG 시스템 평가 프레임워크

---

## 📁 디렉토리 구조

```
evaluation/
├── 01_generate_queries.py              # 평가 쿼리 생성 (200개)
├── 02_evaluate_recall.py               # Recall@K 평가
├── 03_evaluate_hybrid_reranker.py      # Hybrid Search + Reranker 평가
├── 04_evaluate_advanced_metrics.py     # Relevance@K, NDCG@K, Success Rate 평가
├── 05_evaluate_query_rewriting.py      # Query Rewriting 평가
│
├── results/
│   └── queries_final.json              # 최종 평가 쿼리 (200개)
│
└── 문서/
    ├── README.md                       # 이 파일
    └── FINAL_EVALUATION_RESULTS.md     # 최종 평가 결과 요약
```

---

## 🚀 빠른 시작

### 1. 평가 쿼리 생성 (선택)

```bash
# 이미 생성된 queries_final.json 사용 가능
# 새로 생성하려면:
python 01_generate_queries.py
```

### 2. Recall 평가 (기본)

```bash
python 02_evaluate_recall.py
```

### 3. 고급 메트릭 평가 (LLM 필요)

```bash
OPENAI_API_KEY=xxx python 04_evaluate_advanced_metrics.py
```

### 4. Hybrid Search + Reranker 평가

```bash
# 시간이 오래 걸림 (Reranker: 쿼리당 ~23초)
python 03_evaluate_hybrid_reranker.py
```

### 5. Query Rewriting 평가

```bash
OPENAI_API_KEY=xxx python 05_evaluate_query_rewriting.py
```

---

## 📊 최종 평가 결과

### ✅ **배포 결정: Baseline (Dense Embedding only)**

**성능**:
```
Recall@1  = 63.0%
Recall@5  = 88.5%
Recall@10 = 94.0%
Success Rate = 98.0%
Answer Quality = 3.98/5
Latency = 0.27초
```

**구성**:
- 임베딩: `intfloat/multilingual-e5-large`
- VectorDB: ChromaDB
- Top K: 5개 문서

상세 결과는 [FINAL_EVALUATION_RESULTS.md](FINAL_EVALUATION_RESULTS.md) 참조

---

## 🔬 실험한 개선 방법 (모두 실패)

| 방법 | Recall@1 | Recall@5 | Latency | 결과 |
|------|----------|----------|---------|------|
| **Baseline** | **63.0%** | **88.5%** | **0.27초** | ✅ 채택 |
| Metadata Filtering | 79.5% | 79.5% | 0.27초 | ❌ R@5 하락 |
| Hybrid Search (α=0.5) | 34.5% | 66.0% | 0.28초 | ❌ 성능 저하 |
| BGE Reranker | 68.5% | 89.0% | 22.85초 | ❌ Latency 80배 |
| Query Rewriting | 62.0% | 83.0% | 1.23초 | ❌ 성능 저하 |

---

## 🎯 평가 데이터셋

### queries_final.json (200개 쿼리)

| 의도 | 개수 | 비율 | 설명 |
|------|------|------|------|
| location_based | 60 | 30% | 같은 지역 비슷한 업종 사례 |
| scale_based | 50 | 25% | 비슷한 규모 (리뷰/평점) 사례 |
| channel_strategy | 40 | 20% | 채널별 마케팅 전략 사례 |
| problem_solving | 30 | 15% | 평점/리뷰 개선 사례 |
| industry_trend | 10 | 5% | 업종 전반 트렌드 |
| complex_condition | 10 | 5% | 복합 조건 사례 |

**특징**:
- 다중 정답: 쿼리당 평균 110개 관련 문서
- Relevance 점수: 0 (무관), 1 (부분 관련), 2 (완전 관련)
- 현실적 패턴: 실제 사용자 질문 패턴 반영

---

## 📈 평가 지표

### Recall@K
- **R@1**: Top 1에 정답이 있는 비율
- **R@3**: Top 3에 정답이 있는 비율
- **R@5**: Top 5에 정답이 있는 비율
- **R@10**: Top 10에 정답이 있는 비율

### Relevance@K
- Top K 문서의 평균 관련성 점수 (0-2)
- 2: 완전 관련, 1: 부분 관련, 0: 무관

### NDCG@K (Normalized Discounted Cumulative Gain)
- 순위를 고려한 검색 품질 (0-1)
- 1에 가까울수록 관련성 높은 문서가 상위에 위치

### MRR (Mean Reciprocal Rank)
```
MRR = Average(1 / rank of first correct answer)

예시:
- 1위에 정답: 1/1 = 1.0
- 3위에 정답: 1/3 = 0.33
- 없음: 0
```

### Success Rate
- Top K 문서로 질문에 답변 생성 가능한 비율
- LLM이 평가 (유의미한 답변 생성 가능 여부)

### Answer Quality
- LLM-as-Judge로 답변 품질 평가 (1-5점)
- 5점: 완벽, 4점: 우수, 3점: 양호, 2점: 제한적, 1점: 부적절

### Latency
- 쿼리 실행 시간 (초)
- 평균, 중앙값(P50), P95 측정

---

## 🛠️ 개발 환경

### 요구사항

```bash
# Python 패키지
chromadb
sentence-transformers
openai
numpy
rank-bm25  # Hybrid Search 평가용
```

### 환경 변수

```bash
# LLM 평가 시 필요 (04, 05번 스크립트)
export OPENAI_API_KEY=sk-...

# CPU 스레드 제한 (선택)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

---

## 💡 주요 인사이트

### 1. Simple is Best
- 복잡한 개선 방법들이 모두 실패
- Baseline (Dense only)이 최고 성능

### 2. E5 임베딩의 강력함
- 한국어 semantic 이해 우수
- 숫자 정보도 암묵적으로 학습
- Query Rewriting 불필요

### 3. Latency의 중요성
- 5% 성능 향상 vs 80배 속도 저하
- 실시간 서비스에서는 속도 우선

### 4. 다중 정답 평가의 필요성
- 단일 정답(v5.6): R@1 = 89%
- 다중 정답(v5.9): R@1 = 63%
- 현실적 평가를 위해 다중 정답 필수

---

## 📚 문서

- **[FINAL_EVALUATION_RESULTS.md](FINAL_EVALUATION_RESULTS.md)**
  - 최종 평가 결과 요약 (v5.9)
  - 배포 결정 근거
  - 실패한 개선 방법들 (Hybrid Search, Reranker, Query Rewriting)
  - 의도별 성능 분석

---

## 🔧 스크립트 설명

### 01_generate_queries.py
- **목적**: 평가용 synthetic 쿼리 생성
- **출력**: results/queries_final.json
- **쿼리 수**: 200개 (의도별 분포)
- **특징**: 다중 정답, relevance 점수 포함

### 02_evaluate_recall.py
- **목적**: Baseline Recall@K 평가
- **측정**: Recall@1/3/5/10, MRR, Latency
- **소요 시간**: 1-2분

### 03_evaluate_hybrid_reranker.py
- **목적**: Hybrid Search + Reranker 성능 비교
- **평가 대상**:
  - Baseline (Dense)
  - Hybrid Search (α=0.1~0.9)
  - Dense + BGE Reranker
  - Hybrid + Reranker
- **소요 시간**: 30-40분 (Reranker 느림)

### 04_evaluate_advanced_metrics.py
- **목적**: 고급 평가 지표 측정
- **측정**: Relevance@K, NDCG@K, Success Rate, Answer Quality
- **요구사항**: OPENAI_API_KEY
- **소요 시간**: 15-20분 (LLM 호출)

### 05_evaluate_query_rewriting.py
- **목적**: Query Rewriting 효과 검증
- **방법**: 숫자 조건 → semantic 표현 변환
- **요구사항**: OPENAI_API_KEY
- **소요 시간**: 10-15분 (LLM 호출)

---

## 🚧 알려진 이슈 및 제한사항

### 평가 데이터
- Synthetic 쿼리로 생성 (실제 사용자 로그 없음)
- 실제 성능은 배포 후 검증 필요

### BM25 한국어 지원
- 단순 공백 토큰화로는 한국어 처리 부족
- 형태소 분석기 필요 (konlpy 등)
- 현재는 Dense only로 충분

### Reranker Latency
- Cross-Encoder는 CPU에서 매우 느림
- GPU 사용 시 개선 가능하나 인프라 복잡도 증가

---

## 📞 문의

이슈 및 개선 제안: [GitHub Issues](https://github.com/...)

---

**Last Updated**: 2026-01-27
**Version**: v5.9 (Final)
**Status**: ✅ 평가 완료, 배포 가능
