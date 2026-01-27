"""
평가 스크립트: Query Rewriting으로 숫자 조건 개선

목적:
- 숫자 조건 (리뷰 수, 평점)을 semantic 표현으로 변환
- problem_solving, scale_based 의도 성능 개선
- Baseline vs Query Rewriting 비교

실행:
  OPENAI_API_KEY=xxx python evaluation/11_query_rewriting_eval.py

출력:
  evaluation/results/v59_query_rewriting_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CPU 스레드 설정
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# --------------------------------------------
# 설정
# --------------------------------------------
QUERIES_PATH = PROJECT_ROOT / "evaluation" / "results" / "synthetic_queries_v59.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "v59_query_rewriting_results.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

# 평가할 K 값들
K_VALUES = [1, 3, 5, 10]

# LLM 설정
LLM_MODEL = "gpt-4o-mini"


# --------------------------------------------
# RAG 시스템 초기화
# --------------------------------------------
def init_rag():
    """RAG 시스템 초기화"""
    print("🔧 RAG 시스템 초기화 중...")

    from rag.chain import SmallBizRAG

    rag = SmallBizRAG(use_reranker=False)
    print(f"✅ RAG 로드 완료: {rag.vectorstore._collection.count()}개 문서\n")

    return rag


# --------------------------------------------
# Query Rewriter 클래스
# --------------------------------------------
class QueryRewriter:
    """Query Rewriting: 숫자 조건을 semantic 표현으로 변환"""

    def __init__(self, api_key: str, model: str = LLM_MODEL):
        print("🔧 Query Rewriter 초기화 중...")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"✅ Query Rewriter 초기화 완료 (모델: {model})\n")

    def rewrite(self, query: str) -> str:
        """쿼리 재작성: 숫자 조건을 semantic 표현으로 변환"""

        prompt = f"""다음은 소상공인 마케팅 상담 질문입니다.

질문: {query}

이 질문에 숫자 조건(리뷰 수, 평점 등)이 포함되어 있다면, 그 숫자를 의미적(semantic) 표현으로 바꿔서 질문을 다시 작성하세요.

변환 규칙:
- "리뷰 1000개 이상" → "리뷰가 매우 많은 인기 있는"
- "리뷰 100-500개" → "적당한 리뷰를 받고 있는"
- "리뷰 100개 미만" → "신생 또는 리뷰가 적은"
- "평점 4.5점 이상" → "평점이 매우 높은"
- "평점 4점대" → "평점이 높은"
- "평점 3점대" → "평점이 보통이거나 개선이 필요한"

숫자 조건이 없다면 원래 질문을 그대로 반환하세요.

재작성된 질문만 출력하세요:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            rewritten = response.choices[0].message.content.strip()

            # 따옴표 제거
            rewritten = rewritten.strip('"\'')

            return rewritten

        except Exception as e:
            print(f"  ⚠️  Query Rewriting 실패: {e}")
            return query  # 실패 시 원본 반환


# --------------------------------------------
# 검색 함수들
# --------------------------------------------
def search_baseline(rag, query: str, k: int = 10) -> Tuple[List[str], float]:
    """Baseline: 원본 쿼리로 검색"""
    start = time.time()
    docs = rag.vectorstore.similarity_search(query, k=k)
    latency = time.time() - start

    doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
    return doc_ids, latency


def search_with_rewriting(
    rag,
    rewriter: QueryRewriter,
    query: str,
    k: int = 10
) -> Tuple[List[str], str, float]:
    """Query Rewriting 적용 검색"""
    start = time.time()

    # 1. Query Rewriting
    rewritten_query = rewriter.rewrite(query)

    # 2. 검색
    docs = rag.vectorstore.similarity_search(rewritten_query, k=k)

    latency = time.time() - start

    doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
    return doc_ids, rewritten_query, latency


# --------------------------------------------
# 평가 지표 계산
# --------------------------------------------
def calculate_recall_at_k(retrieved_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """Recall@K 계산 (Hit@K)"""
    if not relevant_doc_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_doc_ids))
    return 1.0 if hits > 0 else 0.0


def calculate_mrr(retrieved_ids: List[str], relevant_doc_ids: List[str]) -> float:
    """MRR (Mean Reciprocal Rank) 계산"""
    if not relevant_doc_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def calculate_relevance_at_k(
    retrieved_ids: List[str],
    ground_truth: Dict[str, int],
    k: int
) -> float:
    """Relevance@K 계산"""
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevances = [ground_truth.get(doc_id, 0) for doc_id in top_k]
    return np.mean(relevances) if relevances else 0.0


# --------------------------------------------
# 평가 실행
# --------------------------------------------
def evaluate_method(
    method_name: str,
    search_fn,
    queries: List[Dict[str, Any]],
    k_values: List[int] = K_VALUES,
    save_examples: bool = False
) -> Dict[str, Any]:
    """단일 방법 평가"""
    print(f"🔍 [{method_name}] 평가 시작...")

    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"relevance@{k}": [] for k in k_values})
    results["mrr"] = []
    latencies = []

    # 의도별 결과
    intent_results = defaultdict(lambda: {
        **{f"recall@{k}": [] for k in k_values},
        **{f"relevance@{k}": [] for k in k_values},
        "mrr": [],
    })

    # 예시 저장 (Query Rewriting 전후 비교)
    examples = []

    for idx, query_item in enumerate(queries):
        query = query_item["query"]
        intent = query_item.get("intent", "unknown")
        relevant_docs = query_item.get("relevant_docs", [])

        # relevant_doc_ids 및 ground_truth
        relevant_doc_ids = [doc["doc_id"] for doc in relevant_docs if doc.get("relevance", 0) >= 1]
        ground_truth = {doc["doc_id"]: doc.get("relevance", 0) for doc in relevant_docs}

        if not relevant_doc_ids:
            continue

        # 검색 실행
        search_result = search_fn(query, k=max(k_values))

        # Baseline vs Rewriting 처리
        if len(search_result) == 2:  # Baseline
            retrieved_ids, latency = search_result
            rewritten_query = query
        else:  # Rewriting
            retrieved_ids, rewritten_query, latency = search_result

        latencies.append(latency)

        # 예시 저장 (처음 5개만)
        if save_examples and len(examples) < 5 and rewritten_query != query:
            examples.append({
                "intent": intent,
                "original": query,
                "rewritten": rewritten_query,
            })

        # Recall@K 계산
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_ids, relevant_doc_ids, k)
            results[f"recall@{k}"].append(recall)
            intent_results[intent][f"recall@{k}"].append(recall)

        # Relevance@K 계산
        for k in k_values:
            relevance = calculate_relevance_at_k(retrieved_ids, ground_truth, k)
            results[f"relevance@{k}"].append(relevance)
            intent_results[intent][f"relevance@{k}"].append(relevance)

        # MRR 계산
        mrr = calculate_mrr(retrieved_ids, relevant_doc_ids)
        results["mrr"].append(mrr)
        intent_results[intent]["mrr"].append(mrr)

        # 진행 상황
        if (idx + 1) % 20 == 0:
            print(f"  [{method_name}] 진행: {idx + 1}/{len(queries)}")

    print(f"  [{method_name}] ✅ 평가 완료\n")

    # 평균 계산
    avg_results = {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in results.items()
    }

    # 의도별 평균
    intent_avg = {}
    for intent, metrics_dict in intent_results.items():
        intent_avg[intent] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in metrics_dict.items()
        }

    # Latency 통계
    avg_results["latency_mean"] = np.mean(latencies) if latencies else 0.0
    avg_results["latency_p50"] = np.percentile(latencies, 50) if latencies else 0.0
    avg_results["latency_p95"] = np.percentile(latencies, 95) if latencies else 0.0

    return {
        "overall": avg_results,
        "by_intent": intent_avg,
        "examples": examples,
    }


# --------------------------------------------
# 결과 출력
# --------------------------------------------
def print_results(baseline_results: Dict, rewriting_results: Dict):
    """결과 출력"""
    print("=" * 90)
    print("📊 평가 결과: Baseline vs Query Rewriting")
    print("=" * 90)

    baseline = baseline_results["overall"]
    rewriting = rewriting_results["overall"]

    # 1. Recall@K 비교
    print("\n### Recall@K 비교")
    print(f"{'Metric':<15} {'Baseline':>12} {'Rewriting':>12} {'Improvement':>12}")
    print("-" * 52)

    for k in K_VALUES:
        b_val = baseline.get(f"recall@{k}", 0.0)
        r_val = rewriting.get(f"recall@{k}", 0.0)
        improvement = ((r_val - b_val) / b_val * 100) if b_val > 0 else 0.0

        print(f"Recall@{k:<2}       {b_val:>11.1%} {r_val:>11.1%} {improvement:>+10.1f}%")

    b_mrr = baseline.get("mrr", 0.0)
    r_mrr = rewriting.get("mrr", 0.0)
    improvement = ((r_mrr - b_mrr) / b_mrr * 100) if b_mrr > 0 else 0.0
    print(f"MRR            {b_mrr:>11.1%} {r_mrr:>11.1%} {improvement:>+10.1f}%")

    # 2. Relevance@K 비교
    print("\n### Relevance@K 비교 (0-2)")
    print(f"{'Metric':<15} {'Baseline':>12} {'Rewriting':>12} {'Improvement':>12}")
    print("-" * 52)

    for k in K_VALUES:
        b_val = baseline.get(f"relevance@{k}", 0.0)
        r_val = rewriting.get(f"relevance@{k}", 0.0)
        improvement = ((r_val - b_val) / b_val * 100) if b_val > 0 else 0.0

        print(f"Relevance@{k:<2}    {b_val:>12.3f} {r_val:>12.3f} {improvement:>+10.1f}%")

    # 3. Latency 비교
    print("\n### Latency 비교 (초)")
    print(f"{'Metric':<15} {'Baseline':>12} {'Rewriting':>12}")
    print("-" * 40)

    b_lat = baseline.get("latency_mean", 0.0)
    r_lat = rewriting.get("latency_mean", 0.0)
    print(f"평균           {b_lat:>12.4f} {r_lat:>12.4f}")

    b_p50 = baseline.get("latency_p50", 0.0)
    r_p50 = rewriting.get("latency_p50", 0.0)
    print(f"중앙값(P50)    {b_p50:>12.4f} {r_p50:>12.4f}")

    # 4. 의도별 개선 (문제 의도 중심)
    print("\n### 의도별 성능 (Recall@5)")
    print(f"{'Intent':<20} {'Baseline':>12} {'Rewriting':>12} {'Improvement':>12}")
    print("-" * 57)

    for intent in ["problem_solving", "scale_based", "location_based", "channel_strategy",
                   "industry_trend", "complex_condition"]:
        if intent in baseline_results["by_intent"] and intent in rewriting_results["by_intent"]:
            b_val = baseline_results["by_intent"][intent].get("recall@5", 0.0)
            r_val = rewriting_results["by_intent"][intent].get("recall@5", 0.0)
            improvement = ((r_val - b_val) / b_val * 100) if b_val > 0 else 0.0

            print(f"{intent:<20} {b_val:>11.1%} {r_val:>11.1%} {improvement:>+10.1f}%")

    # 5. Query Rewriting 예시
    if rewriting_results.get("examples"):
        print("\n### Query Rewriting 예시")
        print("-" * 90)
        for i, example in enumerate(rewriting_results["examples"], 1):
            print(f"\n{i}. [{example['intent']}]")
            print(f"   원본:   {example['original']}")
            print(f"   변환:   {example['rewritten']}")

    print("\n" + "=" * 90)


# --------------------------------------------
# 메인
# --------------------------------------------
def main():
    # 쿼리 로드
    print("📋 쿼리 로드 중...")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    queries = queries_data.get("queries", [])
    print(f"✅ {len(queries)}개 쿼리 로드 완료\n")

    # RAG 초기화
    rag = init_rag()

    # Query Rewriter 초기화
    rewriter = QueryRewriter(api_key=OPENAI_API_KEY)

    # 평가 실행
    print("=" * 80)
    print("🔍 평가 시작")
    print("=" * 80)
    print()

    # 1. Baseline
    baseline_results = evaluate_method(
        "Baseline",
        lambda q, k: search_baseline(rag, q, k),
        queries
    )

    # 2. Query Rewriting
    rewriting_results = evaluate_method(
        "Query Rewriting",
        lambda q, k: search_with_rewriting(rag, rewriter, q, k),
        queries,
        save_examples=True
    )

    # 결과 출력
    print_results(baseline_results, rewriting_results)

    # 결과 저장
    output = {
        "version": "v5.9",
        "total_queries": len(queries),
        "k_values": K_VALUES,
        "baseline": baseline_results["overall"],
        "rewriting": rewriting_results["overall"],
        "baseline_by_intent": baseline_results["by_intent"],
        "rewriting_by_intent": rewriting_results["by_intent"],
        "examples": rewriting_results["examples"],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {OUTPUT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
