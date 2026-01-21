"""
Reranker 평가 스크립트 (Cross-Encoder 기반)

기능:
- documents_v5.jsonl, synthetic_queries_v5.json 로드
- 임베딩 모델로 벡터 검색 (Top-K 후보)
- Cross-Encoder Reranker로 재순위화
- Vector Only vs Reranked 성능 비교

Reranker 모델:
- BAAI/bge-reranker-v2-m3 (다국어, 빠름)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (영어 최적화)

사용법:
  python evaluation/05_evaluate_reranker.py

필요 패키지:
  pip install sentence-transformers
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------
# 설정
# --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
DOCS_PATH = os.getenv("DOCS_PATH", str(DATA_DIR / "processed" / "documents_v5.jsonl"))

# 쿼리 파일 경로 (evaluation 폴더에 있음)
QUERIES_PATH = os.getenv(
    "QUERIES_PATH",
    str(EVAL_DIR / "results" / "synthetic_queries_v5.json")
)

# 임베딩 모델
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# Reranker 모델
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# 검색 설정
TOP_K = int(os.getenv("TOP_K", "20"))  # 벡터 검색 후보 수
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))  # Rerank 후 상위 N개

K_VALUES = [1, 3, 5, 10]


# --------------------------------------------
# 로더
# --------------------------------------------
def load_documents(path: str) -> List[Dict[str, Any]]:
    """문서 로드"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def load_queries(path: str) -> List[Dict[str, Any]]:
    """쿼리 로드 (여러 경로 시도)"""
    paths_to_try = [
        path,
        str(EVAL_DIR / "results" / "synthetic_queries_v5.json"),
        str(DATA_DIR / "processed" / "synthetic_queries_v5.json"),
    ]

    for p in paths_to_try:
        if os.path.exists(p):
            print(f"쿼리 로드: {p}")
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(f"쿼리 파일을 찾을 수 없습니다. 시도한 경로: {paths_to_try}")


# --------------------------------------------
# 임베딩 모델
# --------------------------------------------
class EmbeddingModel:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        print(f"임베딩 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()
        self.doc_prefix = "passage: " if self.is_e5 else ""
        self.query_prefix = "query: " if self.is_e5 else ""

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix
        prefixed = [prefix + t for t in texts] if prefix else texts
        emb = self.model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


# --------------------------------------------
# Cross-Encoder Reranker
# --------------------------------------------
class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        print(f"Reranker 모델 로딩: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        doc_texts: List[str],
        doc_indices: List[int],
        top_n: int
    ) -> List[int]:
        """
        Cross-Encoder로 재순위화

        Args:
            query: 검색 쿼리
            doc_texts: 문서 텍스트 리스트
            doc_indices: 원본 문서 인덱스 리스트
            top_n: 반환할 상위 N개

        Returns:
            재순위화된 원본 문서 인덱스 리스트
        """
        if not doc_texts:
            return []

        # (query, doc) 쌍 생성
        pairs = [(query, doc) for doc in doc_texts]

        # 점수 계산
        scores = self.model.predict(pairs)

        # 점수로 정렬
        scored = list(zip(scores, doc_indices))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 상위 N개 인덱스 반환
        return [idx for _, idx in scored[:top_n]]


# --------------------------------------------
# 메트릭 계산
# --------------------------------------------
def compute_metrics(
    retrieved_list: List[List[int]],
    queries: List[Dict[str, Any]],
    k_values: List[int]
) -> Dict[str, float]:
    """Recall@K, MRR 계산"""
    recalls = {k: [] for k in k_values}
    mrrs: List[float] = []

    for i, q in enumerate(queries):
        target = q["doc_id"]
        retrieved = retrieved_list[i]

        for k in k_values:
            in_top_k = target in retrieved[:k] if len(retrieved) >= k else target in retrieved
            recalls[k].append(1 if in_top_k else 0)

        try:
            rank = retrieved.index(target) + 1
            mrrs.append(1.0 / rank)
        except ValueError:
            mrrs.append(0.0)

    results: Dict[str, float] = {}
    for k in k_values:
        results[f"Recall@{k}"] = float(np.mean(recalls[k]))
    results["MRR"] = float(np.mean(mrrs))
    return results


# --------------------------------------------
# 메인
# --------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Reranker 평가 (Vector Search → Cross-Encoder Rerank)")
    print("=" * 70)
    print(f"임베딩 모델: {EMBEDDING_MODEL}")
    print(f"Reranker 모델: {RERANKER_MODEL}")
    print(f"Top-K: {TOP_K}, Rerank Top-N: {RERANK_TOP_N}")
    print()

    # 1. 데이터 로드
    print("[1/5] 데이터 로드")
    docs = load_documents(DOCS_PATH)
    queries = load_queries(QUERIES_PATH)

    # 유효한 쿼리만 필터링
    queries = [q for q in queries if q.get("doc_id", -1) >= 0]
    print(f"문서: {len(docs)}개, 유효 쿼리: {len(queries)}개")

    doc_texts = [d.get("text", "") or d.get("content", {}).get("text", "") for d in docs]
    query_texts = [q["query"] for q in queries]

    # 2. 임베딩
    print("\n[2/5] 임베딩 생성")
    embedder = EmbeddingModel(EMBEDDING_MODEL)
    doc_emb = embedder.embed(doc_texts, is_query=False)
    query_emb = embedder.embed(query_texts, is_query=True)
    print(f"문서 임베딩: {doc_emb.shape}, 쿼리 임베딩: {query_emb.shape}")

    # 3. 벡터 검색
    print("\n[3/5] 벡터 검색")
    import faiss

    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb)

    _, nn_indices = index.search(query_emb, TOP_K)
    print(f"각 쿼리당 Top-{TOP_K} 후보 검색 완료")

    # Vector Only 결과 저장
    vector_only_results = [indices.tolist() for indices in nn_indices]

    # 4. Reranking
    print("\n[4/5] Cross-Encoder Reranking")
    reranker = CrossEncoderReranker(RERANKER_MODEL)

    reranked_results: List[List[int]] = []
    for qi in tqdm(range(len(queries)), desc="Reranking"):
        query = query_texts[qi]
        cand_indices = nn_indices[qi].tolist()
        cand_docs = [doc_texts[i] for i in cand_indices]

        reranked_indices = reranker.rerank(query, cand_docs, cand_indices, RERANK_TOP_N)
        reranked_results.append(reranked_indices)

    # 5. 메트릭 계산 및 비교
    print("\n[5/5] 메트릭 계산")

    vector_metrics = compute_metrics(vector_only_results, queries, K_VALUES)
    rerank_metrics = compute_metrics(reranked_results, queries, K_VALUES)

    # 결과 출력
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)

    results_data = []

    print("\n[ Vector Only (Top-20) ]")
    row1 = {"Method": "Vector Only"}
    for k, v in vector_metrics.items():
        print(f"  {k}: {v:.4f}")
        row1[k] = v
    results_data.append(row1)

    print(f"\n[ + Reranker (Top-{RERANK_TOP_N}) ]")
    row2 = {"Method": f"+ Reranker (Top-{RERANK_TOP_N})"}
    for k, v in rerank_metrics.items():
        print(f"  {k}: {v:.4f}")
        row2[k] = v
    results_data.append(row2)

    # 개선율
    print("\n[ 개선율 ]")
    for k in K_VALUES:
        key = f"Recall@{k}"
        before = vector_metrics[key]
        after = rerank_metrics[key]
        diff = after - before
        print(f"  {key}: {before:.4f} → {after:.4f} ({diff:+.4f})")

    mrr_before = vector_metrics["MRR"]
    mrr_after = rerank_metrics["MRR"]
    print(f"  MRR: {mrr_before:.4f} → {mrr_after:.4f} ({mrr_after - mrr_before:+.4f})")

    # CSV 저장
    df = pd.DataFrame(results_data)
    out_path = RESULTS_DIR / "reranker_evaluation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
