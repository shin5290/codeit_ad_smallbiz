"""
평가 스크립트: Hybrid Search + Reranker 성능 비교

목적:
- v5.9 쿼리로 Hybrid Search (Dense + BM25) 성능 측정
- Reranker (BGE Reranker, Cross-Encoder) 성능 비교
- Latency 측정

비교군:
1. Baseline: Dense only (E5)
2. Hybrid Search: Dense + BM25 (alpha = 0.1, 0.3, 0.5, 0.7, 0.9)
3. Reranker: BGE Reranker v2-m3
4. Cross-Encoder: BGE Reranker v2-m3

실행:
  python evaluation/09_hybrid_reranker_eval.py

출력:
  evaluation/results/v59_hybrid_reranker_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

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
DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "documents_v5.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "v59_hybrid_reranker_results.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 평가할 K 값들
K_VALUES = [1, 3, 5, 10]

# Hybrid Search alpha 값들 (alpha * dense + (1-alpha) * bm25)
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

# Reranker 모델
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


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
# 문서 로드 (BM25용)
# --------------------------------------------
def load_documents():
    """documents_v5.jsonl 로드"""
    docs = []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line.strip())
                docs.append(doc)
    return docs


# --------------------------------------------
# BM25 초기화
# --------------------------------------------
def init_bm25(documents: List[Dict[str, Any]]):
    """BM25 인덱스 초기화"""
    print("🔧 BM25 초기화 중...")
    from rank_bm25 import BM25Okapi

    # 문서 텍스트 추출
    corpus = []
    for doc in documents:
        text = doc.get("text", "") or doc.get("content", {}).get("text", "")
        meta = doc.get("metadata", {})
        title = meta.get("title", "")
        location = meta.get("location", "")
        industry = meta.get("industry", "")

        # 제목 + 지역 + 업종 + 본문
        full_text = f"{title} {location} {industry} {text}"

        # 간단한 토큰화 (공백 기준)
        tokens = full_text.split()
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)
    print(f"✅ BM25 초기화 완료: {len(corpus)}개 문서\n")

    return bm25


# --------------------------------------------
# Reranker 초기화
# --------------------------------------------
class BGEReranker:
    """BGE Reranker v2-m3"""
    def __init__(self, model_name: str = BGE_RERANKER_MODEL):
        print(f"🔧 BGE Reranker 로드 중: {model_name}")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512, device="cpu")
        print(f"✅ BGE Reranker 로드 완료\n")

    def rerank(self, query: str, documents: List[Any], top_k: int = 10) -> List[Any]:
        """문서 재정렬 (LangChain Document 객체 지원)"""
        if not documents:
            return []

        # 쿼리-문서 페어 생성
        pairs = []
        for doc in documents:
            # LangChain Document 객체인 경우
            if hasattr(doc, 'page_content'):
                text = doc.page_content
            # dictionary인 경우
            elif isinstance(doc, dict):
                text = doc.get("text", "")
            else:
                text = str(doc)

            pairs.append([query, text])

        # 점수 계산
        scores = self.model.predict(pairs)

        # 점수 기준 정렬
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]


# --------------------------------------------
# 검색 함수들
# --------------------------------------------
def search_dense(rag, query: str, k: int = 10) -> Tuple[List[str], float]:
    """Dense 검색 (E5)"""
    start = time.time()
    docs = rag.vectorstore.similarity_search(query, k=k)
    latency = time.time() - start

    doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
    return doc_ids, latency


def search_bm25(bm25, documents: List[Dict[str, Any]], query: str, k: int = 10) -> List[str]:
    """BM25 검색"""
    tokens = query.split()
    scores = bm25.get_scores(tokens)

    # Top K 인덱스
    top_indices = np.argsort(scores)[::-1][:k]

    doc_ids = [documents[i].get("doc_id", "") for i in top_indices]
    return doc_ids


def search_hybrid(
    rag,
    bm25,
    documents: List[Dict[str, Any]],
    query: str,
    alpha: float = 0.5,
    k: int = 10
) -> Tuple[List[str], float]:
    """Hybrid 검색 (Dense + BM25)

    alpha: Dense 가중치 (0~1)
    - alpha=1.0: Dense only
    - alpha=0.0: BM25 only
    """
    start = time.time()

    # Dense 검색 (더 많이 가져오기)
    dense_k = min(k * 3, 100)
    dense_docs = rag.vectorstore.similarity_search_with_score(query, k=dense_k)

    # Dense 점수 정규화 (cosine similarity는 0~1 범위)
    dense_scores = {}
    for doc, score in dense_docs:
        doc_id = doc.metadata.get("doc_id", "")
        dense_scores[doc_id] = score

    # BM25 검색
    bm25_tokens = query.split()
    bm25_raw_scores = bm25.get_scores(bm25_tokens)

    # BM25 점수 정규화 (0~1)
    max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1.0
    bm25_scores = {}
    for i, score in enumerate(bm25_raw_scores):
        doc_id = documents[i].get("doc_id", "")
        bm25_scores[doc_id] = score / max_bm25

    # Hybrid 점수 계산
    all_doc_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
    hybrid_scores = {}

    for doc_id in all_doc_ids:
        dense_s = dense_scores.get(doc_id, 0.0)
        bm25_s = bm25_scores.get(doc_id, 0.0)
        hybrid_scores[doc_id] = alpha * dense_s + (1 - alpha) * bm25_s

    # Top K 선택
    sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, score in sorted_docs[:k]]

    latency = time.time() - start
    return top_doc_ids, latency


def search_with_reranker(
    rag,
    reranker,
    documents: List[Dict[str, Any]],
    query: str,
    k: int = 10,
    initial_k: int = 50
) -> Tuple[List[str], float]:
    """Reranker 적용 검색"""
    start = time.time()

    # 1. Dense로 initial_k개 검색
    initial_docs = rag.vectorstore.similarity_search(query, k=initial_k)

    # 2. Reranker로 재정렬
    reranked = reranker.rerank(query, initial_docs, top_k=k)

    latency = time.time() - start

    doc_ids = [doc.metadata.get("doc_id", "") for doc in reranked]
    return doc_ids, latency


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


# --------------------------------------------
# 평가 실행
# --------------------------------------------
def evaluate_method(
    method_name: str,
    search_fn,
    queries: List[Dict[str, Any]],
    k_values: List[int] = K_VALUES
) -> Dict[str, Any]:
    """단일 방법 평가"""
    print(f"🔍 [{method_name}] 평가 시작...")

    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    latencies = []

    for idx, query_item in enumerate(queries):
        query = query_item["query"]
        relevant_docs = query_item.get("relevant_docs", [])

        # relevant_doc_ids 추출 (relevance >= 1)
        relevant_doc_ids = [doc["doc_id"] for doc in relevant_docs if doc.get("relevance", 0) >= 1]

        if not relevant_doc_ids:
            continue

        # 검색 실행
        retrieved_ids, latency = search_fn(query, k=max(k_values))
        latencies.append(latency)

        # Recall@K 계산
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_ids, relevant_doc_ids, k)
            results[f"recall@{k}"].append(recall)

        # MRR 계산
        mrr = calculate_mrr(retrieved_ids, relevant_doc_ids)
        results["mrr"].append(mrr)

        if (idx + 1) % 50 == 0:
            print(f"  [{method_name}] 진행: {idx + 1}/{len(queries)}")

    # 평균 계산
    avg_results = {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in results.items()
    }

    # Latency 통계
    avg_results["latency_mean"] = np.mean(latencies) if latencies else 0.0
    avg_results["latency_p50"] = np.percentile(latencies, 50) if latencies else 0.0
    avg_results["latency_p95"] = np.percentile(latencies, 95) if latencies else 0.0

    print(f"  [{method_name}] ✅ 평가 완료\n")
    return avg_results


# --------------------------------------------
# 결과 출력
# --------------------------------------------
def print_results(all_results: Dict[str, Dict[str, Any]]):
    """결과 출력"""
    print("=" * 90)
    print("📊 평가 결과")
    print("=" * 90)

    # 1. Recall 비교
    print("\n### Recall@K 비교")
    print(f"{'Method':<30} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 90)

    for method_name, results in all_results.items():
        r1 = results.get("recall@1", 0.0)
        r3 = results.get("recall@3", 0.0)
        r5 = results.get("recall@5", 0.0)
        r10 = results.get("recall@10", 0.0)
        mrr = results.get("mrr", 0.0)

        print(f"{method_name:<30} {r1:>7.1%} {r3:>7.1%} {r5:>7.1%} {r10:>7.1%} {mrr:>7.1%}")

    # 2. Latency 비교
    print("\n### Latency 비교 (초)")
    print(f"{'Method':<30} {'평균':>10} {'중앙값(P50)':>14} {'P95':>10}")
    print("-" * 90)

    for method_name, results in all_results.items():
        mean = results.get("latency_mean", 0.0)
        p50 = results.get("latency_p50", 0.0)
        p95 = results.get("latency_p95", 0.0)

        print(f"{method_name:<30} {mean:>9.4f} {p50:>13.4f} {p95:>9.4f}")

    print("=" * 90)


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

    # 문서 로드
    print("📄 문서 로드 중...")
    documents = load_documents()
    print(f"✅ {len(documents)}개 문서 로드 완료\n")

    # RAG 초기화
    rag = init_rag()

    # BM25 초기화
    bm25 = init_bm25(documents)

    # Reranker 초기화
    reranker = BGEReranker()

    # 평가 실행
    all_results = {}

    # 1. Baseline (Dense only)
    all_results["Baseline (Dense)"] = evaluate_method(
        "Baseline (Dense)",
        lambda q, k: search_dense(rag, q, k),
        queries
    )

    # 2. Hybrid Search (여러 alpha 값)
    for alpha in ALPHA_VALUES:
        method_name = f"Hybrid (α={alpha})"
        all_results[method_name] = evaluate_method(
            method_name,
            lambda q, k, a=alpha: search_hybrid(rag, bm25, documents, q, alpha=a, k=k),
            queries
        )

    # 3. Reranker (Dense + BGE Reranker)
    all_results["Dense + BGE Reranker"] = evaluate_method(
        "Dense + BGE Reranker",
        lambda q, k: search_with_reranker(rag, reranker, documents, q, k=k, initial_k=50),
        queries
    )

    # 4. Hybrid + Reranker (최적 alpha 사용)
    best_alpha = 0.5  # 실험 후 최적값 선택 가능
    all_results[f"Hybrid (α={best_alpha}) + Reranker"] = evaluate_method(
        f"Hybrid (α={best_alpha}) + Reranker",
        lambda q, k: search_with_reranker(
            rag,
            reranker,
            documents,
            q,
            k=k,
            initial_k=50
        ),
        queries
    )

    # 결과 출력
    print_results(all_results)

    # 결과 저장
    output = {
        "version": "v5.9",
        "total_queries": len(queries),
        "k_values": K_VALUES,
        "alpha_values": ALPHA_VALUES,
        "results": all_results
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {OUTPUT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
