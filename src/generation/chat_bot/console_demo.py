"""
간단 터미널용 데모 스크립트.

사용 방법 (JupyterHub 터미널):
  python console_demo.py

옵션:
  --method auto|rag|agent   강제 라우팅(기본 auto)
  --k 5                     RAG 검색 문서 수

종료: 빈 줄 입력 또는 Ctrl+C
리셋: "리셋", "초기화" 입력 시 업종/지역 정보 초기화
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트 추가 (어디서 실행해도 import 가능하도록)
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
REPO_ROOT = PROJECT_ROOT.parent
# 패키지 로드를 위해 상위 디렉터리를 sys.path에 추가
sys.path.insert(0, str(REPO_ROOT))

from chat_bot import SmallBizConsultant, SmallBizRAG, TrendAgent, UserContext


# ============================================
# UserContext 동적 추출
# ============================================

INDUSTRY_KEYWORDS = {
    "카페": "cafe",
    "커피": "cafe",
    "커피숍": "cafe",
    "음식점": "restaurant",
    "식당": "restaurant",
    "맛집": "restaurant",
    "베이커리": "bakery",
    "빵집": "bakery",
    "디저트": "dessert",
    # 단글자 '바'는 오탐이 많아 제외, 주점/술집만 허용
    "술집": "bar",
    "주점": "bar",
}

LOCATION_KEYWORDS = [
    "강남", "홍대", "신사", "압구정", "청담", "이태원", "성수", "연남",
    "종로", "명동", "을지로", "광화문", "여의도", "잠실", "건대",
    "신촌", "이대", "합정", "망원", "연희", "서촌", "북촌",
    "판교", "분당", "수원", "인천", "부산", "대구", "제주", "순천",
]


def match_keyword(text: str, keyword: str) -> bool:
    """
    한글 키워드가 다른 단어 내부에 섞여있는 오탐을 줄이기 위한 매칭
    - 조사(에서/에/은/는/이/가/을/를 등)까지 허용하여 '순천에서', '카페를'도 인식
    - 한글 앞/뒤가 다른 한글이면 매칭하지 않음
    """
    import re
    pattern = rf"(?<![가-힣]){re.escape(keyword)}(?:에서|에|은|는|이|가|을|를|으로|로|이요|이에요|예요|야|이야)?(?![가-힣])"
    return re.search(pattern, text) is not None


def extract_context_from_query(query: str, current_ctx: Optional[UserContext]) -> UserContext:
    """
    대화에서 업종/지역/예산을 추출하여 UserContext 업데이트
    """
    if current_ctx is None:
        current_ctx = UserContext()

    query_lower = query.lower()

    # 1) 업종 추출
    for keyword, industry_code in INDUSTRY_KEYWORDS.items():
        if match_keyword(query, keyword):
            current_ctx.industry = industry_code
            break

    # 2) 지역 추출
    for location in LOCATION_KEYWORDS:
        if match_keyword(query, location):
            current_ctx.location = location
            break

    # 3) 예산 추출 (숫자 + 만원/원)
    budget_patterns = [
        r"(\d+)\s*만\s*원",  # 30만원, 30 만 원
        r"예산\s*(\d+)\s*만",  # 예산 30만
        r"(\d+)만\s*원",  # 30만원
        r"예산이?\s*(\d{4,})\s*원?",  # 예산 300000원
    ]
    for pattern in budget_patterns:
        match = re.search(pattern, query)
        if match:
            num = int(match.group(1))
            # "만원" 단위면 10000 곱하기
            if "만" in pattern or num < 1000:
                current_ctx.budget = num * 10000
            else:
                current_ctx.budget = num
            break

    # 4) 목표 추출 (간단한 패턴)
    goal_patterns = {
        "신규 고객": "신규 고객 유치",
        "재방문": "재방문율 증가",
        "매출": "매출 증대",
        "인지도": "브랜드 인지도 향상",
        "홍보": "매장 홍보",
    }
    for keyword, goal in goal_patterns.items():
        if keyword in query:
            current_ctx.goal = goal
            break

    return current_ctx


def run_once(
    question: str,
    method: str,
    k: int,
    consultant: SmallBizConsultant,
    user_ctx: Optional[UserContext],
) -> dict:
    """질문을 한 번 실행하고 결과 dict를 반환."""
    try:
        if method == "rag":
            result = consultant.rag.query(question=question, k=k, user_context=user_ctx)
            result["method"] = "rag"
            result["intent"] = result.get("intent", "doc_rag")
        elif method == "agent":
            result = consultant.agent.run(query=question, user_context=user_ctx)
            result["method"] = "agent"
            result["intent"] = result.get("intent", "trend_web")
        else:
            result = consultant.consult(query=question, user_context=user_ctx)

        return result
    except KeyboardInterrupt:
        raise
    except Exception as e:  # pragma: no cover - 데모용 방어
        return {
            "question": question,
            "answer": f"[에러] {type(e).__name__}: {e}",
            "method": "error",
            "intent": "error",
        }


def format_response(result: dict) -> str:
    """결과 dict를 출력 문자열로 포맷."""
    method_used = result.get("method", "unknown")
    intent = result.get("intent", "unknown")
    question = result.get("question", "")
    body = result.get("answer", "응답이 없습니다.")

    # Slot-filling인 경우 특별 표시
    if method_used == "slot_filling":
        header = f"[SLOT_FILLING] (intent: {intent})"
    else:
        header = f"[{method_used.upper()}] (intent: {intent})"

    return f"{header}\n\n{body}"


def format_context(ctx: Optional[UserContext]) -> str:
    """UserContext를 간단한 문자열로 포맷"""
    if ctx is None:
        return "미설정"
    parts = []
    if ctx.industry:
        parts.append(f"업종:{ctx.industry}")
    if ctx.location:
        parts.append(f"지역:{ctx.location}")
    if ctx.budget:
        parts.append(f"예산:{ctx.budget:,}원")
    return " | ".join(parts) if parts else "미설정"


def main():
    parser = argparse.ArgumentParser(description="터미널 인터랙티브 데모")
    parser.add_argument("--method", default="auto", choices=["auto", "rag", "agent"], help="강제 라우팅")
    parser.add_argument("--k", type=int, default=5, help="RAG 검색 문서 수")
    args = parser.parse_args()

    # UserContext를 None으로 시작 (대화 중 동적 추출)
    user_ctx: Optional[UserContext] = None
    consultant = SmallBizConsultant()

    # Slot-Filling 상태 관리
    pending_query: Optional[str] = None  # 슬롯 채워진 후 재실행할 질문

    # 리셋 명령어 패턴
    RESET_PATTERNS = ["리셋", "초기화", "reset", "clear", "처음부터"]

    print("=" * 70)
    print("ChatBot 콘솔 데모 (종료: 빈 줄 또는 Ctrl+C)")
    print(f"라우팅: {args.method}")
    print("(업종/지역/예산은 대화 중 자동 인식됩니다)")
    print("(리셋/초기화 입력 시 컨텍스트 초기화)")
    print("=" * 70)

    try:
        while True:
            # 현재 컨텍스트 표시
            ctx_str = format_context(user_ctx)

            # 대기 중인 질문이 있으면 표시
            if pending_query:
                prompt = f"\n[{ctx_str}] (대기 질문: {pending_query[:30]}...)\n답변> "
            else:
                prompt = f"\n[{ctx_str}]\n질문> "

            q = input(prompt).strip()
            if not q:
                print("종료합니다.")
                break

            # 리셋 명령어 체크
            q_lower = q.lower()
            if any(reset_word in q_lower for reset_word in RESET_PATTERNS):
                user_ctx = None
                pending_query = None
                print("\n[시스템] 업종/지역/예산 정보가 초기화되었습니다. 새로운 정보를 입력해주세요.")
                continue

            # 대화에서 컨텍스트 추출
            user_ctx = extract_context_from_query(q, user_ctx)

            # 대기 중인 질문이 있고, 슬롯 정보만 입력받은 경우
            # (짧은 답변 = 슬롯 정보로 간주)
            if pending_query and len(q) < 30:
                # 슬롯 정보가 추출됐으면 원래 질문 재실행
                if user_ctx and (user_ctx.industry or user_ctx.location):
                    print(f"\n→ 원래 질문 재실행: {pending_query}")
                    result = run_once(
                        question=pending_query,
                        method=args.method,
                        k=args.k,
                        consultant=consultant,
                        user_ctx=user_ctx,
                    )
                    pending_query = None  # 대기 질문 해제
                    print("\n" + format_response(result))
                    continue

            # 일반 질문 실행
            result = run_once(
                question=q,
                method=args.method,
                k=args.k,
                consultant=consultant,
                user_ctx=user_ctx,
            )

            # Slot-Filling 응답인 경우 대기 질문 저장
            if result.get("method") == "slot_filling":
                pending_query = result.get("pending_query", q)
            else:
                pending_query = None  # 정상 응답이면 대기 질문 해제

            print("\n" + format_response(result))

    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
