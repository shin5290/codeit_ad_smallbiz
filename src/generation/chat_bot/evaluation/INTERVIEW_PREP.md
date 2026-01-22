# Retrieval 평가 면접 대비 가이드

> 이 문서는 RAG 시스템의 Retrieval 평가에 대한 면접 질문과 답변을 정리한 것입니다.

---

## 1. R@1, R@3는 어떻게 구했나요?

### 답변

**Recall@K (R@K)**는 "상위 K개 검색 결과 안에 정답 문서가 포함되어 있는 비율"입니다.

```
R@K = (정답이 상위 K에 포함된 쿼리 수) / (전체 쿼리 수)
```

**측정 방법 (코드 기반):**
```python
# evaluation/04_evaluate_embeddings.py 라인 209-216
for i, q in enumerate(queries):
    target = q["doc_id"]  # 정답 문서 ID
    retrieved = indices[i].tolist()  # 검색된 문서 ID 순위 리스트

    for k in k_values:  # k = 1, 3, 5, 10
        # 정답이 상위 k개 안에 있으면 1, 없으면 0
        recalls[k].append(1 if target in retrieved[:k] else 0)

# 평균 계산
results[f"Recall@{k}"] = float(np.mean(recalls[k]))
```

**실제 측정 과정:**
1. 쿼리를 임베딩 → 벡터 검색으로 상위 K개 문서 추출
2. 정답 문서(doc_id)가 상위 K개에 있는지 확인
3. 전체 쿼리에서의 적중률을 평균

**결과 해석:**
- R@1 = 0.8533 → 첫 번째 검색 결과가 정답인 비율 85.33%
- R@3 = 0.9033 → 상위 3개 안에 정답이 있는 비율 90.33%

---

## 2. 정답 문서는 어떻게 마련했나요?

### 답변

**합성 쿼리(Synthetic Query) 방식**을 사용했습니다.

**방법:**
1. 592개 매장 문서 중 100개를 랜덤 샘플링
2. 각 문서에 대해 GPT-4o-mini로 3개의 쿼리 생성 (총 300개 쿼리)
3. **쿼리를 생성한 문서 = 해당 쿼리의 정답 문서**

**코드 (evaluation/04_evaluate_embeddings.py 라인 82-127):**
```python
class SyntheticQueryGenerator:
    def generate_queries(self, doc, num_queries=3):
        prompt = f"""다음은 {industry} 업종 매장 정보입니다.
        제목: {title}
        지역: {location}
        내용: {text[:600]}

        사용자가 이 매장을 찾기 위해 입력할 질문 {num_queries}개를 생성하세요.
        - 한국어, 짧고 구어체
        - 지역+특징이 드러나게
        - 서로 다른 관점 (위치/메뉴/후기/평점 등)
        """
```

**예시 (synthetic_queries_v5.json):**
```json
{
  "query": "연남에서 분위기 좋은 카페 어디 있어?",
  "doc_id": 494,
  "doc_title": "코리코카페 연남점"
}
```
→ "코리코카페 연남점" 문서에서 생성된 쿼리이므로, 정답 = doc_id 494

---

## 3. 실제 테스트에서 정답 문서와 어떻게 비교했나요?

### 답변

**FAISS 벡터 인덱스**를 사용해 유사도 검색 후 순위를 비교했습니다.

**과정:**
```python
# 1. 모든 문서를 임베딩 (592개)
doc_embeddings = model.embed(doc_texts, is_query=False)

# 2. 모든 쿼리를 임베딩 (300개)
query_embeddings = model.embed(query_texts, is_query=True)

# 3. FAISS 인덱스 생성 (Inner Product = 코사인 유사도)
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

# 4. 각 쿼리에 대해 상위 K개 문서 검색
_, indices = index.search(query_embeddings, max_k)
# indices[i] = 쿼리 i에 대한 검색 결과 (문서 ID 순위 리스트)

# 5. 정답 문서 순위 확인
for i, q in enumerate(queries):
    target = q["doc_id"]  # 정답
    retrieved = indices[i].tolist()  # 검색 결과 순위

    # target이 retrieved 몇 번째에 있는지 확인
    rank = retrieved.index(target) + 1  # 1-indexed
```

---

## 4. 정답 문서의 정답은 진짜 사실인가요? 어떻게 검증했나요?

### 답변

**합성 쿼리 방식의 특성상 "자동 검증"이 됩니다.**

**왜 신뢰할 수 있는가:**
1. **인과관계**: 쿼리가 특정 문서에서 생성됨 → 해당 쿼리의 정답은 당연히 그 문서
2. **LLM 가이드**: GPT-4o-mini에게 "이 매장을 찾기 위한 질문"을 요청 → 문서 내용과 직접 연결
3. **다양성 확보**: 위치/메뉴/후기/평점 등 다른 관점의 쿼리 생성

**한계와 보완점:**
- 합성 쿼리가 실제 사용자 쿼리와 다를 수 있음
- 보완: 실제 사용자 로그 수집 후 Human Evaluation 필요
- 추가 검증: RAGAS, LLM-as-a-Judge 등 자동 평가 프레임워크 활용

**현재 검증 방식:**
```python
# LLM-as-a-Judge로 응답 품질 평가 (별도 진행)
# evaluation/evaluate_prompts.py
# - Specificity: 구체성 (숫자, 예시)
# - Evidence: 근거 (출처 명시)
# - Structure: 구조 일관성
# - Safety: 과장/허위 정보 없음
```

---

## 5. 정답 문서의 쿼리는 어떤 것이었나요? 왜 그것이었나요?

### 답변

**쿼리 설계 원칙:**
```
1. 짧고 구어체 (실제 사용자처럼)
2. 지역 + 특징이 드러나게
3. 서로 다른 관점 (다양성)
```

**예시:**
| 문서 | 쿼리 | 관점 |
|------|------|------|
| 코리코카페 연남점 | "연남에서 분위기 좋은 카페 어디 있어?" | 위치 + 분위기 |
| 코리코카페 연남점 | "코리코카페 연남점 인기 메뉴 뭐야?" | 메뉴 |
| 코리코카페 연남점 | "연남에 있는 카페 중에서 평점 높은 곳?" | 평점 |

**왜 이런 쿼리인가:**
1. **실제 사용 시나리오 모방**: 소상공인 마케팅 상담에서 사용자는 "OO 지역 OO 업종 추천해줘" 형태로 질문
2. **검색 난이도 조절**: 지역만 언급 vs 매장명 직접 언급 → 다양한 난이도
3. **멀티 관점 테스트**: 같은 문서라도 메뉴/위치/후기 등 다른 관점으로 검색 가능한지 평가

---

## 6. MRR은 어떤 평가 지표이고 어떻게 측정했나요?

### 답변

**MRR (Mean Reciprocal Rank)** = 정답 문서 순위의 역수 평균

```
MRR = (1/N) × Σ(1/rank_i)

- rank_i: i번째 쿼리에서 정답 문서의 순위
- N: 전체 쿼리 수
```

**예시:**
| 쿼리 | 정답 순위 | Reciprocal Rank |
|------|-----------|-----------------|
| Q1 | 1위 | 1/1 = 1.0 |
| Q2 | 3위 | 1/3 = 0.33 |
| Q3 | 2위 | 1/2 = 0.5 |
| **MRR** | | **(1.0 + 0.33 + 0.5) / 3 = 0.61** |

**코드:**
```python
# evaluation/04_evaluate_embeddings.py 라인 217-220
for i, q in enumerate(queries):
    target = q["doc_id"]
    retrieved = indices[i].tolist()

    try:
        rank = retrieved.index(target) + 1  # 1-indexed
        mrrs.append(1.0 / rank)  # Reciprocal Rank
    except ValueError:
        mrrs.append(0.0)  # 정답이 없으면 0

results["MRR"] = float(np.mean(mrrs))
```

**MRR vs Recall 차이:**
- **Recall@K**: "상위 K개 안에 정답이 있는가?" (이진)
- **MRR**: "정답이 몇 번째에 있는가?" (순위 반영)

**결과 해석:**
- MRR = 0.8858 → 평균적으로 정답이 1~2위 사이에 위치
- MRR이 높을수록 정답이 상위에 랭크됨

---

## 7. 추가 질문 대비

### Q: 왜 multilingual-e5-large를 선택했나요?

**답변:**
1. **한국어 성능**: 다국어 모델 중 한국어 벤치마크 성능 우수
2. **prefix 지원**: "query:", "passage:" prefix로 비대칭 검색 최적화
3. **비교 평가 결과**:
   - bge-m3: R@1 = 0.82
   - multilingual-e5-large: R@1 = 0.85 (선택)
   - text-embedding-3-small: R@1 = 0.79

### Q: Reranker를 왜 선택적으로 사용하나요?

**답변:**
1. **성능 향상**: R@1 0.8533 → 0.8733 (+2%)
2. **비용/시간 트레이드오프**: Reranker 추가 시 latency 증가
3. **사용 시점**: 높은 정확도가 필요한 경우에만 활성화

### Q: 평가 데이터셋 크기가 충분한가요?

**답변:**
- 현재: 100개 문서 × 3개 쿼리 = 300개 쿼리
- 통계적으로 유의미한 결과를 위해서는 최소 100개 이상 권장
- 추후 개선: 전체 592개 문서로 확대, 실제 사용자 쿼리 로그 추가

---

## 8. 핵심 요약 (30초 답변용)

> **"합성 쿼리 기반 Retrieval 평가를 진행했습니다.**
>
> 592개 매장 문서 중 100개를 샘플링하고, GPT-4o-mini로 각 문서에서 3개씩 쿼리를 생성했습니다. 총 300개 쿼리로 평가했고, 쿼리를 생성한 문서가 정답 문서입니다.
>
> FAISS 벡터 검색으로 상위 K개를 추출하고, 정답 문서가 포함되어 있는지 확인해서 Recall@K를 계산했습니다. MRR은 정답 순위의 역수 평균입니다.
>
> 결과: R@1=85.33%, R@3=90.33%, MRR=0.886
>
> multilingual-e5-large 모델이 한국어 매장 데이터에서 가장 좋은 성능을 보였고, Reranker 추가 시 2% 추가 향상을 확인했습니다."

---

**작성일:** 2025-01-21
**목적:** 포트폴리오 면접 대비
