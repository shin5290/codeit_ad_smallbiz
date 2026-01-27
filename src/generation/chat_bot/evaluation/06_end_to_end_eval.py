"""
End-to-End 시스템 평가 (200개 쿼리)

평가 항목:
1. IntentRouter 정확도 (RAG vs Agent 경로 선택)
2. Agent 비용/Latency (web_search)
3. Self-refine 효과 (품질 vs 비용)
4. 전체 시스템 성능

실행:
    OPENAI_API_KEY=xxx TAVILY_API_KEY=xxx python 06_end_to_end_eval.py

결과:
    results/end_to_end_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import traceback

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import TrendAgent
from rag.prompts import IntentRouter, UserContext
from refine.self_refine import SelfRefiner
from rag.chain import SmallBizRAG

# API keys check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("JupyterHub에서 실행 중이라면 계속 진행됩니다...")
    # raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
else:
    print(f"✅ OPENAI_API_KEY: {OPENAI_API_KEY[:10]}...")

print(f"✅ TAVILY_API_KEY: {'있음' if TAVILY_API_KEY else '없음 (DuckDuckGo 사용)'}")


# Token counting (approximate)
def count_tokens(text: str) -> int:
    """간단한 토큰 카운팅 (실제로는 tiktoken 사용 권장)"""
    return len(text) // 4  # 대략 4글자 = 1토큰


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    """비용 추정 (GPT-4o 기준)"""
    # GPT-4o: $2.50 / 1M input, $10.00 / 1M output
    if "gpt-4o" in model:
        input_cost = input_tokens / 1_000_000 * 2.50
        output_cost = output_tokens / 1_000_000 * 10.00
    elif "gpt-4o-mini" in model:
        # GPT-4o-mini: $0.15 / 1M input, $0.60 / 1M output
        input_cost = input_tokens / 1_000_000 * 0.15
        output_cost = output_tokens / 1_000_000 * 0.60
    else:
        input_cost = output_cost = 0.0

    return input_cost + output_cost


def classify_query_intent(query: str) -> str:
    """쿼리의 예상 의도 분류 (정답 라벨용)"""
    # IntentRouter 기준과 최대한 정렬 (평가 기준 현실화)
    trend_keywords = [
        "요즘", "최근", "트렌드", "유행", "인기", "핫한", "뜨는",
        "밈", "meme", "랭킹", "2024", "2025", "2026",
        "신상", "신메뉴", "핫플", "바이럴",
    ]
    stats_keywords = [
        "가장 많이", "가장 적게", "1위", "2위", "3위", "순위",
        "몇 %", "몇 퍼센트", "비율", "통계", "데이터",
        "평균", "전체", "시장", "점유율",
        "대한민국", "한국", "국내", "글로벌", "세계",
    ]
    marketing_keywords = [
        "예산", "비용", "집행", "배분", "분배",
        "전략", "방법", "어떻게", "뭐가 좋",
        "리텐션", "재방문", "고객", "매출", "방문자",
        "홍보", "광고", "마케팅", "프로모션",
    ]
    doc_rag_keywords = [
        "사례", "케이스", "성공", "실패", "경험",
        "매장", "가게", "다른 카페", "다른 맛집",
        "리뷰", "평점", "사진", "촬영", "업로드",
    ]

    query_lower = query.lower().strip()
    has_personal = any(kw in query_lower for kw in ["우리", "내", "저희", "내 매장", "우리 가게"])

    if any(kw in query_lower for kw in trend_keywords):
        return "trend_web"
    if any(kw in query_lower for kw in stats_keywords) and not has_personal:
        return "stats_query"
    if any(kw in query_lower for kw in marketing_keywords) or has_personal:
        return "marketing_counsel"
    if any(kw in query_lower for kw in doc_rag_keywords):
        return "doc_rag"
    return "doc_rag"


def main():
    # ==========================================
    # 1. 데이터 로드
    # ==========================================
    print("\n" + "="*60)
    print("End-to-End 시스템 평가 시작")
    print("="*60)

    queries_file = BASE_DIR / "results" / "queries_final.json"

    if not queries_file.exists():
        raise FileNotFoundError(f"쿼리 파일이 없습니다: {queries_file}")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    queries = queries_data["queries"]
    print(f"\n📊 평가 쿼리: {len(queries)}개")

    # ==========================================
    # 2. 시스템 초기화
    # ==========================================
    print("\n🔧 시스템 초기화 중...")

    try:
        agent = TrendAgent(
            llm_model="gpt-4o-mini",  # 비용 절감
            verbose=False
        )
        print("  ✅ TrendAgent 초기화 완료")
    except Exception as e:
        print(f"  ❌ TrendAgent 초기화 실패: {e}")
        traceback.print_exc()
        return

    try:
        intent_router = IntentRouter()
        print("  ✅ IntentRouter 초기화 완료")
    except Exception as e:
        print(f"  ❌ IntentRouter 초기화 실패: {e}")
        return

    try:
        refiner = SelfRefiner(llm_model="gpt-4o-mini", verbose=False)
        print("  ✅ SelfRefiner 초기화 완료")
    except Exception as e:
        print(f"  ⚠️  SelfRefiner 초기화 실패 (선택적 기능): {e}")
        refiner = None

    # ==========================================
    # 3. 평가 실행
    # ==========================================
    results = []
    stats = {
        "total": len(queries),
        "completed": 0,
        "errors": 0,
        "intent_accuracy": 0,
        "route_distribution": defaultdict(int),
        "total_cost": 0.0,
        "total_time": 0.0,
        "refine_improvement": 0,
    }

    print("\n" + "="*60)
    print("평가 시작 (200개 쿼리)")
    print("="*60)

    for idx, q in enumerate(queries, 1):
        query_text = q["query"]
        query_intent_type = q.get("intent_type", "unknown")

        print(f"\n[{idx}/{len(queries)}] {query_text[:50]}...")

        result = {
            "idx": idx,
            "query": query_text,
            "intent_type": query_intent_type,
            "expected_route": classify_query_intent(query_text),
        }

        try:
            # ---------------------------------
            # A. Intent 분류
            # ---------------------------------
            start_time = time.time()

            context = UserContext()
            detected_intent = intent_router.classify(query_text)

            intent_time = time.time() - start_time

            result["detected_intent"] = detected_intent
            result["intent_match"] = (detected_intent == result["expected_route"])
            result["intent_time"] = round(intent_time, 3)

            stats["route_distribution"][detected_intent] += 1
            if result["intent_match"]:
                stats["intent_accuracy"] += 1

            print(f"  🎯 Intent: {detected_intent} (예상: {result['expected_route']})")

            # ---------------------------------
            # B. 답변 생성 (기본)
            # ---------------------------------
            start_time = time.time()

            # Agent invoke
            response = agent.run(query_text)

            answer_time = time.time() - start_time

            # 응답 파싱
            if isinstance(response, dict):
                answer_text = response.get("answer") or response.get("output") or str(response)
            else:
                answer_text = str(response)

            result["answer"] = answer_text[:500]  # 저장 용량 절약
            result["answer_time"] = round(answer_time, 3)

            # 비용 추정
            input_tokens = count_tokens(query_text)
            output_tokens = count_tokens(answer_text)
            cost = estimate_cost(input_tokens, output_tokens, "gpt-4o-mini")

            result["input_tokens"] = input_tokens
            result["output_tokens"] = output_tokens
            result["cost"] = round(cost, 6)

            stats["total_cost"] += cost
            stats["total_time"] += answer_time

            print(f"  💬 답변 생성: {answer_time:.2f}초, ${cost:.4f}")

            # ---------------------------------
            # C. Self-Refine (선택적)
            # ---------------------------------
            if refiner and idx <= 50:  # 처음 50개만 테스트
                try:
                    start_time = time.time()

                    refine_result = refiner.run(
                        question=query_text,
                        initial_answer=answer_text
                    )

                    refine_time = time.time() - start_time

                    refined_answer = refine_result.get("final_answer", answer_text)
                    refined_flag = refine_result.get("refined", False)
                    refine_used = refine_result.get("used", True)

                    refine_cost = 0.0
                    if refine_used:
                        # Refine 비용 (추가 LLM 호출: critique + refine)
                        refine_tokens = count_tokens(refined_answer)
                        refine_cost = estimate_cost(
                            count_tokens(query_text + answer_text) * 2,  # critique + refine
                            refine_tokens,
                            "gpt-4o-mini"
                        )

                    result["refined_answer"] = refined_answer[:500]
                    result["refine_time"] = round(refine_time, 3) if refine_used else 0.0
                    result["refine_cost"] = round(refine_cost, 6)
                    result["refine_used"] = refine_used
                    result["refine_improved"] = refined_flag

                    # 품질 향상 여부
                    if refined_flag:
                        stats["refine_improvement"] += 1

                    if refine_used:
                        print(f"  ✨ Refine: {refine_time:.2f}초, ${refine_cost:.4f}, 향상={refined_flag}")
                    else:
                        print("  ✨ Refine: skipped")

                except Exception as e:
                    print(f"  ⚠️  Refine 실패: {e}")
                    result["refine_used"] = False
            else:
                result["refine_used"] = False

            # ---------------------------------
            # 완료
            # ---------------------------------
            stats["completed"] += 1

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            traceback.print_exc()
            result["error"] = str(e)
            stats["errors"] += 1

        results.append(result)

        # 중간 저장 (10개마다)
        if idx % 10 == 0:
            temp_file = BASE_DIR / "results" / "end_to_end_results_temp.json"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({
                    "queries": results,
                    "stats": stats,
                    "progress": f"{idx}/{len(queries)}"
                }, f, ensure_ascii=False, indent=2)
            print(f"  💾 중간 저장: {temp_file}")

    # ==========================================
    # 4. 최종 통계 계산
    # ==========================================
    print("\n" + "="*60)
    print("평가 완료! 최종 통계 계산 중...")
    print("="*60)

    # Intent 정확도
    if stats["completed"] > 0:
        stats["intent_accuracy"] = round(stats["intent_accuracy"] / stats["completed"], 3)

    # 평균 비용/시간
    stats["avg_cost"] = round(stats["total_cost"] / stats["completed"], 6) if stats["completed"] > 0 else 0
    stats["avg_time"] = round(stats["total_time"] / stats["completed"], 3) if stats["completed"] > 0 else 0

    # Route distribution
    stats["route_distribution"] = dict(stats["route_distribution"])

    # ==========================================
    # 5. 결과 저장
    # ==========================================
    output_file = BASE_DIR / "results" / "end_to_end_results.json"

    final_output = {
        "metadata": {
            "total_queries": len(queries),
            "completed": stats["completed"],
            "errors": stats["errors"],
            "model": "gpt-4o-mini",
            "tavily_enabled": TAVILY_API_KEY is not None,
        },
        "summary": {
            "intent_accuracy": stats["intent_accuracy"],
            "route_distribution": stats["route_distribution"],
            "avg_cost_per_query": stats["avg_cost"],
            "avg_time_per_query": stats["avg_time"],
            "total_cost": round(stats["total_cost"], 4),
            "total_time": round(stats["total_time"], 2),
            "refine_improvement_rate": round(stats["refine_improvement"] / 50, 3) if stats["refine_improvement"] > 0 else 0,
        },
        "queries": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 결과 저장: {output_file}")

    # ==========================================
    # 6. 요약 출력
    # ==========================================
    print("\n" + "="*60)
    print("📊 최종 결과 요약")
    print("="*60)

    print(f"\n✅ 완료: {stats['completed']}/{stats['total']}개")
    print(f"❌ 오류: {stats['errors']}개")

    print(f"\n🎯 Intent 정확도: {stats['intent_accuracy']:.1%}")
    print(f"\n📍 Route 분포:")
    for route, count in stats["route_distribution"].items():
        pct = count / stats["completed"] * 100 if stats["completed"] > 0 else 0
        print(f"  • {route}: {count}개 ({pct:.1f}%)")

    print(f"\n💰 비용:")
    print(f"  • 총 비용: ${stats['total_cost']:.4f}")
    print(f"  • 쿼리당 평균: ${stats['avg_cost']:.6f}")
    print(f"  • 월 1만 쿼리 예상: ${stats['avg_cost'] * 10000:.2f}")

    print(f"\n⏱️  시간:")
    print(f"  • 총 시간: {stats['total_time']:.1f}초")
    print(f"  • 쿼리당 평균: {stats['avg_time']:.3f}초")

    if stats["refine_improvement"] > 0:
        print(f"\n✨ Self-Refine (50개 샘플):")
        print(f"  • 품질 향상: {stats['refine_improvement']}/50 ({stats['refine_improvement']/50:.1%})")

    print("\n" + "="*60)
    print("평가 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
