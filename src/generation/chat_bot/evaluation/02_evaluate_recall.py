"""
평가 스크립트: v5.9 Recall@K 측정

목적:
- v5.9 쿼리 (다중 정답, 상담 의도 반영)에 대한 Recall@K 측정
- Baseline (필터 없음) vs Filtered (메타데이터 필터) 비교
- 의도별 성능 분석

실행:
  python evaluation/08_evaluate_v59_recall.py

출력:
  evaluation/results/v59_recall_results.json
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

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
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "v59_recall_results.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 평가할 K 값들
K_VALUES = [1, 3, 5, 10]

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
# Location 매핑 (부모-자식 관계)
# --------------------------------------------
LOCATION_GROUPS = {
    "강남": ["강남", "역삼", "선릉", "삼성", "청담", "신사", "논현", "압구정"],
    "홍대": ["홍대", "연남", "상수", "합정", "망원"],
    "신촌": ["신촌", "이대"],
    "건대": ["건대", "구의", "광진"],
    "신림": ["신림", "봉천"],
    "성수": ["성수", "뚝섬"],
    "서울대": ["신림"],
    "이태원": ["이태원", "한남", "경리단길"],
    "가로수길": ["신사", "청담"],
    # 기타 지역은 자기 자신만
}

def expand_location(location: str) -> List[str]:
    """Location을 확장 (예: 홍대 → [홍대, 연남, 상수, ...])"""
    for parent, children in LOCATION_GROUPS.items():
        if location in children or location == parent:
            return children
    return [location]


# --------------------------------------------
# 검색 함수
# --------------------------------------------
def search_baseline(rag, query: str, k: int = 10) -> List[str]:
    """Baseline: 필터 없이 semantic search만"""
    docs = rag.vectorstore.similarity_search(query, k=k)
    doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
    return doc_ids


def search_with_filters(rag, query: str, filters: Dict[str, Any], k: int = 10) -> List[str]:
    """Filtered: 메타데이터 필터 적용"""
    where_filter = {}
    where_conditions = []

    # Location 필터
    if "location" in filters:
        locations = expand_location(filters["location"])
        if len(locations) == 1:
            where_conditions.append({"location": {"$eq": locations[0]}})
        else:
            where_conditions.append({"location": {"$in": locations}})

    # Industry 필터
    if "industry" in filters:
        where_conditions.append({"industry": {"$eq": filters["industry"]}})

    # Review count 필터
    if "review_count_range" in filters:
        min_r, max_r = filters["review_count_range"]
        where_conditions.append({"review_count": {"$gte": min_r}})
        where_conditions.append({"review_count": {"$lte": max_r}})
    elif "review_count_min" in filters:
        where_conditions.append({"review_count": {"$gte": filters["review_count_min"]}})

    # Rating 필터
    if "rating_range" in filters:
        min_r, max_r = filters["rating_range"]
        where_conditions.append({"rating": {"$gte": min_r}})
        where_conditions.append({"rating": {"$lte": max_r}})
    elif "rating_min" in filters:
        where_conditions.append({"rating": {"$gte": filters["rating_min"]}})

    # 조합
    if len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}
    elif len(where_conditions) == 1:
        where_filter = where_conditions[0]
    else:
        where_filter = None

    # 검색
    try:
        docs = rag.vectorstore.similarity_search(
            query,
            k=k,
            filter=where_filter if where_filter else None,
        )
        doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
        return doc_ids
    except Exception as e:
        # 필터에 맞는 문서가 없으면 빈 리스트 반환
        return []


# --------------------------------------------
# 평가 지표 계산
# --------------------------------------------
def calculate_recall_at_k(retrieved_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """Recall@K 계산"""
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
def evaluate(rag, queries: List[Dict[str, Any]]):
    """전체 쿼리에 대해 평가 실행"""
    print("=" * 80)
    print("🔍 평가 시작")
    print("=" * 80)

    # 결과 저장용
    baseline_results = {f"recall@{k}": [] for k in K_VALUES}
    baseline_results["mrr"] = []

    filtered_results = {f"recall@{k}": [] for k in K_VALUES}
    filtered_results["mrr"] = []

    # 의도별 결과
    intent_baseline = defaultdict(lambda: {f"recall@{k}": [] for k in K_VALUES})
    intent_filtered = defaultdict(lambda: {f"recall@{k}": [] for k in K_VALUES})

    # 각 쿼리 평가
    for idx, query_item in enumerate(queries):
        query = query_item["query"]
        intent = query_item.get("intent", "unknown")
        relevant_docs = query_item.get("relevant_docs", [])
        filters = query_item.get("filters", {})

        # relevant_doc_ids 추출 (relevance >= 1인 문서들)
        relevant_doc_ids = [doc["doc_id"] for doc in relevant_docs if doc.get("relevance", 0) >= 1]

        if not relevant_doc_ids:
            continue

        # Baseline 검색
        baseline_ids = search_baseline(rag, query, k=max(K_VALUES))

        # Filtered 검색
        filtered_ids = search_with_filters(rag, query, filters, k=max(K_VALUES))

        # Recall@K 계산
        for k in K_VALUES:
            baseline_recall = calculate_recall_at_k(baseline_ids, relevant_doc_ids, k)
            filtered_recall = calculate_recall_at_k(filtered_ids, relevant_doc_ids, k)

            baseline_results[f"recall@{k}"].append(baseline_recall)
            filtered_results[f"recall@{k}"].append(filtered_recall)

            intent_baseline[intent][f"recall@{k}"].append(baseline_recall)
            intent_filtered[intent][f"recall@{k}"].append(filtered_recall)

        # MRR 계산
        baseline_mrr = calculate_mrr(baseline_ids, relevant_doc_ids)
        filtered_mrr = calculate_mrr(filtered_ids, relevant_doc_ids)

        baseline_results["mrr"].append(baseline_mrr)
        filtered_results["mrr"].append(filtered_mrr)

        # 진행 상황
        if (idx + 1) % 50 == 0:
            print(f"진행: {idx + 1}/{len(queries)}")

    print(f"✅ 평가 완료: {len(queries)}개 쿼리\n")

    # 평균 계산
    baseline_avg = {metric: sum(values) / len(values) if values else 0.0
                    for metric, values in baseline_results.items()}

    filtered_avg = {metric: sum(values) / len(values) if values else 0.0
                    for metric, values in filtered_results.items()}

    # 의도별 평균
    intent_baseline_avg = {}
    intent_filtered_avg = {}

    for intent in intent_baseline.keys():
        intent_baseline_avg[intent] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in intent_baseline[intent].items()
        }
        intent_filtered_avg[intent] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in intent_filtered[intent].items()
        }

    return {
        "baseline": baseline_avg,
        "filtered": filtered_avg,
        "intent_baseline": intent_baseline_avg,
        "intent_filtered": intent_filtered_avg,
        "raw_baseline": baseline_results,
        "raw_filtered": filtered_results,
    }


# --------------------------------------------
# 결과 출력
# --------------------------------------------
def print_results(results: Dict[str, Any], total_queries: int):
    """결과 출력"""
    print("=" * 80)
    print("📊 평가 결과")
    print("=" * 80)

    baseline = results["baseline"]
    filtered = results["filtered"]

    print("\n### 전체 성능 (Baseline vs Filtered)")
    print(f"{'Metric':<15} {'Baseline':>12} {'Filtered':>12} {'Improvement':>12}")
    print("-" * 52)

    for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]:
        b_val = baseline[metric]
        f_val = filtered[metric]
        improvement = ((f_val - b_val) / b_val * 100) if b_val > 0 else 0.0

        print(f"{metric:<15} {b_val:>11.1%} {f_val:>11.1%} {improvement:>+10.1f}%")

    # 의도별 결과
    print("\n### 의도별 성능 (Filtered)")
    print(f"{'Intent':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 60)

    for intent, metrics in results["intent_filtered"].items():
        r1 = metrics.get("recall@1", 0.0)
        r3 = metrics.get("recall@3", 0.0)
        r5 = metrics.get("recall@5", 0.0)
        r10 = metrics.get("recall@10", 0.0)

        print(f"{intent:<20} {r1:>7.1%} {r3:>7.1%} {r5:>7.1%} {r10:>7.1%}")

    print(f"\n총 쿼리 수: {total_queries}")
    print("=" * 80)


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

    # 평가 실행
    results = evaluate(rag, queries)

    # 결과 출력
    print_results(results, len(queries))

    # 결과 저장
    output = {
        "version": "v5.9",
        "total_queries": len(queries),
        "k_values": K_VALUES,
        "baseline": results["baseline"],
        "filtered": results["filtered"],
        "intent_baseline": results["intent_baseline"],
        "intent_filtered": results["intent_filtered"],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
