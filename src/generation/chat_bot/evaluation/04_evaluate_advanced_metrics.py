"""
평가 스크립트: 고급 평가 지표 (Relevance@K, NDCG@K, Success Rate, Answer Quality)

목적:
- v5.9 쿼리로 Recall 이외의 평가 지표 측정
- Relevance@K: 검색된 문서의 평균 관련성 점수
- NDCG@K: 순위를 고려한 검색 품질
- Success Rate: 검색 결과로 답변 생성 가능 비율
- Answer Quality: LLM-as-Judge로 답변 품질 평가 (1-5점)

실행:
  OPENAI_API_KEY=xxx python evaluation/10_advanced_metrics_eval.py

출력:
  evaluation/results/v59_advanced_metrics_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional

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
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "v59_advanced_metrics_results.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("Success Rate와 Answer Quality 평가를 건너뜁니다.")

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
# LLM 클라이언트 초기화
# --------------------------------------------
def init_llm():
    """OpenAI 클라이언트 초기화"""
    if not OPENAI_API_KEY:
        return None

    print("🔧 LLM 클라이언트 초기화 중...")
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"✅ LLM 클라이언트 초기화 완료 (모델: {LLM_MODEL})\n")

    return client


# --------------------------------------------
# 검색 함수
# --------------------------------------------
def search_and_get_docs(rag, query: str, k: int = 10) -> List[Any]:
    """검색 실행 및 문서 객체 반환"""
    docs = rag.vectorstore.similarity_search(query, k=k)
    return docs


# --------------------------------------------
# Relevance@K 계산
# --------------------------------------------
def calculate_relevance_at_k(
    retrieved_docs: List[Any],
    ground_truth: Dict[str, int],
    k: int
) -> float:
    """Relevance@K 계산

    Args:
        retrieved_docs: 검색된 문서 리스트 (LangChain Document)
        ground_truth: {doc_id: relevance} 매핑 (relevance: 0-2)
        k: Top K

    Returns:
        평균 relevance 점수 (0~2)
    """
    if not retrieved_docs:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    relevances = []

    for doc in top_k_docs:
        doc_id = doc.metadata.get("doc_id", "")
        relevance = ground_truth.get(doc_id, 0)
        relevances.append(relevance)

    return np.mean(relevances) if relevances else 0.0


# --------------------------------------------
# NDCG@K 계산
# --------------------------------------------
def calculate_dcg(relevances: List[float]) -> float:
    """DCG (Discounted Cumulative Gain) 계산"""
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        dcg += rel / np.log2(i + 1)
    return dcg


def calculate_ndcg_at_k(
    retrieved_docs: List[Any],
    ground_truth: Dict[str, int],
    k: int
) -> float:
    """NDCG@K 계산

    NDCG = DCG / IDCG
    - DCG: 실제 검색 결과의 점수
    - IDCG: 이상적인 순서의 점수 (relevance 높은 순)
    """
    if not retrieved_docs:
        return 0.0

    top_k_docs = retrieved_docs[:k]

    # 실제 검색 결과의 relevance 순서
    actual_relevances = []
    for doc in top_k_docs:
        doc_id = doc.metadata.get("doc_id", "")
        relevance = ground_truth.get(doc_id, 0)
        actual_relevances.append(relevance)

    # 이상적인 relevance 순서 (높은 순으로 정렬)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]

    # DCG 계산
    dcg = calculate_dcg(actual_relevances)
    idcg = calculate_dcg(ideal_relevances)

    # NDCG 계산
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# --------------------------------------------
# Success Rate 평가
# --------------------------------------------
def evaluate_success_rate(
    llm_client,
    query: str,
    retrieved_docs: List[Any]
) -> bool:
    """Success Rate 평가: 검색 결과로 질문에 답변 가능한가?"""
    if not llm_client or not retrieved_docs:
        return False

    # 문서 내용 요약
    doc_summaries = []
    for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5만 사용
        title = doc.metadata.get("title", "")
        location = doc.metadata.get("location", "")
        industry = doc.metadata.get("industry", "")
        text_snippet = doc.page_content[:200]  # 처음 200자

        summary = f"{i}. {title} ({location}, {industry})\n{text_snippet}..."
        doc_summaries.append(summary)

    docs_text = "\n\n".join(doc_summaries)

    # 프롬프트
    prompt = f"""다음은 소상공인 마케팅 상담 질문과 검색된 매장 정보입니다.

질문: {query}

검색된 매장 정보:
{docs_text}

이 매장 정보들을 활용하여 위 질문에 유의미한 답변을 제공할 수 있나요?

답변 기준:
- 질문과 관련된 매장 정보가 1개 이상 있으면 "예"
- 질문의 의도를 파악하고 도움이 되는 정보 제공 가능하면 "예"
- 검색 결과가 질문과 전혀 무관하면 "아니오"

답변: "예" 또는 "아니오"만 출력하세요."""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        answer = response.choices[0].message.content.strip()
        return "예" in answer or "yes" in answer.lower()

    except Exception as e:
        print(f"  ⚠️  Success Rate 평가 실패: {e}")
        return False


# --------------------------------------------
# Answer Quality 평가
# --------------------------------------------
def evaluate_answer_quality(
    llm_client,
    query: str,
    retrieved_docs: List[Any]
) -> Optional[int]:
    """Answer Quality 평가: LLM-as-Judge로 답변 품질 평가 (1-5점)"""
    if not llm_client or not retrieved_docs:
        return None

    # 1. 답변 생성
    doc_summaries = []
    for i, doc in enumerate(retrieved_docs[:5], 1):
        title = doc.metadata.get("title", "")
        location = doc.metadata.get("location", "")
        industry = doc.metadata.get("industry", "")
        review_count = doc.metadata.get("review_count", 0)
        rating = doc.metadata.get("rating", 0.0)
        text_snippet = doc.page_content[:300]

        summary = f"{i}. {title} ({location}, {industry})\n   리뷰: {review_count}개, 평점: {rating}\n   {text_snippet}..."
        doc_summaries.append(summary)

    docs_text = "\n\n".join(doc_summaries)

    generation_prompt = f"""다음은 소상공인 마케팅 상담 질문과 검색된 매장 정보입니다.

질문: {query}

검색된 매장 정보:
{docs_text}

위 정보를 바탕으로 질문에 대한 마케팅 조언을 제공하세요.
- 구체적인 사례와 함께 설명
- 실행 가능한 팁 제공
- 2-3문장으로 간결하게"""

    try:
        # 답변 생성
        generation_response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": generation_prompt}],
            temperature=0.7,
            max_tokens=300,
        )

        generated_answer = generation_response.choices[0].message.content.strip()

        # 2. 답변 품질 평가 (LLM-as-Judge)
        judge_prompt = f"""다음은 소상공인 마케팅 상담 질문과 AI가 생성한 답변입니다.

질문: {query}

AI 답변:
{generated_answer}

이 답변을 다음 기준으로 1-5점으로 평가하세요:
- 5점: 질문에 완벽히 답하며, 구체적이고 실행 가능한 조언 제공
- 4점: 질문에 잘 답하며, 유용한 정보 포함
- 3점: 질문에 답하지만 다소 일반적이거나 부족함
- 2점: 질문과 관련은 있으나 도움이 제한적
- 1점: 질문과 무관하거나 잘못된 정보

점수만 출력하세요 (1, 2, 3, 4, 또는 5):"""

        judge_response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=5,
        )

        score_text = judge_response.choices[0].message.content.strip()

        # 점수 추출
        for char in score_text:
            if char.isdigit():
                score = int(char)
                if 1 <= score <= 5:
                    return score

        return None

    except Exception as e:
        print(f"  ⚠️  Answer Quality 평가 실패: {e}")
        return None


# --------------------------------------------
# 평가 실행
# --------------------------------------------
def evaluate_all_metrics(rag, llm_client, queries: List[Dict[str, Any]]):
    """모든 메트릭 평가"""
    print("=" * 80)
    print("🔍 평가 시작")
    print("=" * 80)

    results = {
        f"relevance@{k}": [] for k in K_VALUES
    }
    results.update({
        f"ndcg@{k}": [] for k in K_VALUES
    })
    results["success_rate"] = []
    results["answer_quality"] = []

    # 의도별 결과
    intent_results = defaultdict(lambda: {
        **{f"relevance@{k}": [] for k in K_VALUES},
        **{f"ndcg@{k}": [] for k in K_VALUES},
        "success_rate": [],
        "answer_quality": [],
    })

    for idx, query_item in enumerate(queries):
        query = query_item["query"]
        intent = query_item.get("intent", "unknown")
        relevant_docs = query_item.get("relevant_docs", [])

        # Ground truth: {doc_id: relevance}
        ground_truth = {
            doc["doc_id"]: doc.get("relevance", 0)
            for doc in relevant_docs
        }

        if not ground_truth:
            continue

        # 검색 실행
        retrieved_docs = search_and_get_docs(rag, query, k=max(K_VALUES))

        # Relevance@K 계산
        for k in K_VALUES:
            rel_k = calculate_relevance_at_k(retrieved_docs, ground_truth, k)
            results[f"relevance@{k}"].append(rel_k)
            intent_results[intent][f"relevance@{k}"].append(rel_k)

        # NDCG@K 계산
        for k in K_VALUES:
            ndcg_k = calculate_ndcg_at_k(retrieved_docs, ground_truth, k)
            results[f"ndcg@{k}"].append(ndcg_k)
            intent_results[intent][f"ndcg@{k}"].append(ndcg_k)

        # Success Rate (LLM 호출)
        if llm_client:
            success = evaluate_success_rate(llm_client, query, retrieved_docs)
            results["success_rate"].append(1.0 if success else 0.0)
            intent_results[intent]["success_rate"].append(1.0 if success else 0.0)

            # Answer Quality (LLM 호출)
            quality = evaluate_answer_quality(llm_client, query, retrieved_docs)
            if quality is not None:
                results["answer_quality"].append(quality)
                intent_results[intent]["answer_quality"].append(quality)

            # Rate limiting
            time.sleep(0.1)

        # 진행 상황
        if (idx + 1) % 20 == 0:
            print(f"진행: {idx + 1}/{len(queries)}")

    print(f"✅ 평가 완료: {len(queries)}개 쿼리\n")

    # 평균 계산
    avg_results = {}
    for metric, values in results.items():
        if values:
            avg_results[metric] = sum(values) / len(values)
        else:
            avg_results[metric] = 0.0

    # 의도별 평균
    intent_avg = {}
    for intent, metrics_dict in intent_results.items():
        intent_avg[intent] = {}
        for metric, values in metrics_dict.items():
            if values:
                intent_avg[intent][metric] = sum(values) / len(values)
            else:
                intent_avg[intent][metric] = 0.0

    return {
        "overall": avg_results,
        "by_intent": intent_avg,
        "raw": results,
    }


# --------------------------------------------
# 결과 출력
# --------------------------------------------
def print_results(results: Dict[str, Any], total_queries: int):
    """결과 출력"""
    print("=" * 80)
    print("📊 평가 결과")
    print("=" * 80)

    overall = results["overall"]

    # 1. Relevance@K
    print("\n### Relevance@K (평균 관련성 점수, 0-2)")
    print(f"{'Metric':<15} {'Score':>10}")
    print("-" * 26)
    for k in K_VALUES:
        score = overall.get(f"relevance@{k}", 0.0)
        print(f"Relevance@{k:<2}    {score:>10.3f}")

    # 2. NDCG@K
    print("\n### NDCG@K (순위 품질, 0-1)")
    print(f"{'Metric':<15} {'Score':>10}")
    print("-" * 26)
    for k in K_VALUES:
        score = overall.get(f"ndcg@{k}", 0.0)
        print(f"NDCG@{k:<2}         {score:>10.3f}")

    # 3. Success Rate & Answer Quality
    if overall.get("success_rate", 0.0) > 0:
        print("\n### Success Rate & Answer Quality")
        print(f"{'Metric':<20} {'Score':>10}")
        print("-" * 31)

        success_rate = overall.get("success_rate", 0.0)
        print(f"Success Rate        {success_rate:>9.1%}")

        if overall.get("answer_quality", 0.0) > 0:
            quality = overall.get("answer_quality", 0.0)
            print(f"Answer Quality      {quality:>10.2f}/5")

    # 4. 의도별 결과
    print("\n### 의도별 성능")
    print(f"{'Intent':<20} {'Rel@5':>8} {'NDCG@5':>8} {'Success':>8} {'Quality':>8}")
    print("-" * 62)

    by_intent = results["by_intent"]
    for intent, metrics in by_intent.items():
        rel5 = metrics.get("relevance@5", 0.0)
        ndcg5 = metrics.get("ndcg@5", 0.0)
        success = metrics.get("success_rate", 0.0)
        quality = metrics.get("answer_quality", 0.0)

        quality_str = f"{quality:.2f}" if quality > 0 else "-"
        print(f"{intent:<20} {rel5:>7.3f} {ndcg5:>7.3f} {success:>7.1%} {quality_str:>8}")

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

    # LLM 초기화
    llm_client = init_llm()

    # 평가 실행
    results = evaluate_all_metrics(rag, llm_client, queries)

    # 결과 출력
    print_results(results, len(queries))

    # 결과 저장
    output = {
        "version": "v5.9",
        "total_queries": len(queries),
        "k_values": K_VALUES,
        "overall": results["overall"],
        "by_intent": results["by_intent"],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
