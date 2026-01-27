"""
Self-Refine 모듈 (답변 품질 자동 개선)

기능:
- 초안 생성 → 자체 비평 → 개선 (2단계 루프)
- 평가 기준: 구체성, 근거, 안전성, 구조
- 점수 7점 미만 시 자동 리파인

사용법:
  python refine/self_refine.py

필요 패키지:
  pip install langchain langchain-openai
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add project root to path for rag.prompts import
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.prompts import UserContext

# --------------------------------------------
# 설정
# --------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
REFINE_THRESHOLD = 7.0  # 이 점수 미만이면 리파인


# --------------------------------------------
# Critique 프롬프트
# --------------------------------------------

CRITIQUE_SYSTEM = """당신은 소상공인 마케팅 상담 답변을 평가하는 전문 평가자입니다.

## 평가 기준 (각 0~10점)

1) **구체성 (specificity)**
   - 실행 가능한 구체적 액션인가?
   - 숫자는 **원문에 근거가 있을 때만** 긍정 평가 (근거 없는 숫자 추가는 감점)
   - 0점: 추상적 조언만 / 10점: 모든 항목이 구체적

2) **근거 (evidence)**
   - 출처가 명시되어 있는가? (매장명, 지역, 웹 링크)
   - 검색 결과를 인용했는가?
   - 0점: 출처 없음 / 10점: 각 조언마다 출처 있음

3) **안전성 (safety)**
   - 결과 보장 표현이 없는가? ("반드시", "100%", "확실히")
   - 평점/별점 언급이 없는가?
   - 0점: 보장 표현 다수 / 10점: 안전한 표현만

4) **구조 (structure)**
   - 요청된 섹션을 따랐는가?
   - 간결하고 읽기 쉬운가?
   - 0점: 구조 없음 / 10점: 완벽한 구조

## 주의
- **새로운 숫자/예산/성과 수치 추가를 제안하지 말 것**
- 출처가 없으면 "근거 부족"으로 지적하되, 임의 수치 추가를 권하지 말 것

## 출력 형식 (JSON만 출력)
```json
{
  "scores": {
    "specificity": 8,
    "evidence": 6,
    "safety": 9,
    "structure": 7
  },
  "avg_score": 7.5,
  "issues": [
    "2번 항목에 예산 수치가 없음",
    "출처가 1개만 있음"
  ],
  "suggestions": [
    "각 실행 아이디어에 예산 범위 추가",
    "사례 출처 2개 이상 추가"
  ]
}
```"""


CRITIQUE_USER = """[질문]
{question}

[답변]
{answer}

위 답변을 평가해주세요. JSON 형식으로만 응답하세요."""


# --------------------------------------------
# Refine 프롬프트
# --------------------------------------------

REFINE_SYSTEM = """당신은 소상공인 마케팅 상담 답변을 개선하는 전문가입니다.

## 개선 규칙
1) 지적된 문제점을 모두 수정
2) **새로운 숫자/예산/성과 수치 추가 금지** (원문에 있는 수치만 유지)
3) 출처가 부족하면 **원문에 있는 출처만 재정리** (새 출처 생성 금지)
4) 결과 보장 표현 제거 ("반드시" → "~할 수 있습니다")
5) 평점/별점 언급 제거
6) 원래 답변의 핵심 내용은 유지
7) 출처가 없으면 "출처: 제공된 자료에서 확인 불가"로 명시

## 출력
개선된 답변만 출력하세요. 설명이나 메타 정보 없이 바로 답변 내용만."""


REFINE_USER = """[원래 질문]
{question}

[원래 답변]
{answer}

[평가 결과]
- 점수: {avg_score}/10
- 문제점: {issues}
- 개선 제안: {suggestions}

위 피드백을 반영하여 답변을 개선해주세요."""


# --------------------------------------------
# SelfRefiner 클래스
# --------------------------------------------

class SelfRefiner:
    """
    Self-Refine 파이프라인

    Flow:
    1. 초안 입력
    2. Critique (평가)
    3. 점수 < threshold → Refine (개선)
    4. 최종 답변 반환
    """

    def __init__(
        self,
        llm_model: str = LLM_MODEL,
        threshold: float = REFINE_THRESHOLD,
        max_iterations: int = 2,
        verbose: bool = True,
    ):
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.3,
            api_key=OPENAI_API_KEY,
        )
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.verbose = verbose

    def critique(self, question: str, answer: str) -> Dict[str, Any]:
        """
        답변 평가

        Returns:
            {
                "scores": {...},
                "avg_score": float,
                "issues": [...],
                "suggestions": [...]
            }
        """
        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM),
            HumanMessage(content=CRITIQUE_USER.format(
                question=question,
                answer=answer,
            )),
        ]

        response = self.llm.invoke(messages)

        # JSON 파싱
        try:
            # JSON 블록 추출
            content = response.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)

            # avg_score 계산 (없으면)
            if "avg_score" not in result and "scores" in result:
                scores = result["scores"]
                result["avg_score"] = sum(scores.values()) / len(scores)

            return result
        except Exception as e:
            if self.verbose:
                print(f"Critique 파싱 오류: {e}")
            return {
                "scores": {"specificity": 5, "evidence": 5, "safety": 5, "structure": 5},
                "avg_score": 5.0,
                "issues": ["평가 파싱 실패"],
                "suggestions": ["재시도 필요"],
                "error": str(e),
            }

    def refine(
        self,
        question: str,
        answer: str,
        critique_result: Dict[str, Any],
    ) -> str:
        """
        답변 개선

        Returns:
            개선된 답변 텍스트
        """
        messages = [
            SystemMessage(content=REFINE_SYSTEM),
            HumanMessage(content=REFINE_USER.format(
                question=question,
                answer=answer,
                avg_score=critique_result.get("avg_score", 5),
                issues=", ".join(critique_result.get("issues", [])),
                suggestions=", ".join(critique_result.get("suggestions", [])),
            )),
        ]

        response = self.llm.invoke(messages)
        return response.content

    @staticmethod
    def _should_refine(answer: str) -> bool:
        """
        조건부 리파인: 명확한 품질 이슈가 있을 때만 실행
        - 출처 누락
        - 결과 보장 표현 포함
        - 너무 짧은 답변
        """
        if not answer:
            return False
        text = answer.strip()
        if len(text) < 200:
            return True
        if "출처" not in text:
            return True
        forbidden = ["반드시", "확실히", "100%"]
        if any(k in text for k in forbidden):
            return True
        return False

    def run(
        self,
        question: str,
        initial_answer: str,
    ) -> Dict[str, Any]:
        """
        Self-Refine 실행

        Args:
            question: 원래 질문
            initial_answer: 초안 답변

        Returns:
            {
                "question": 질문,
                "initial_answer": 초안,
                "final_answer": 최종 답변,
                "refined": 리파인 여부,
                "iterations": 반복 횟수,
                "critique_history": [평가 결과들],
            }
        """
        current_answer = initial_answer
        critique_history = []

        if not self._should_refine(initial_answer):
            return {
                "question": question,
                "initial_answer": initial_answer,
                "final_answer": initial_answer,
                "refined": False,
                "used": False,
                "iterations": 0,
                "final_score": None,
                "critique_history": [],
            }

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Self-Refine Iteration {iteration + 1} ---")

            # 1. Critique
            critique = self.critique(question, current_answer)
            critique_history.append(critique)
            avg_score = critique.get("avg_score", 0)

            if self.verbose:
                print(f"평가 점수: {avg_score:.1f}/10")
                print(f"문제점: {critique.get('issues', [])}")

            # 2. 점수 체크
            if avg_score >= self.threshold:
                if self.verbose:
                    print(f"점수 {avg_score:.1f} >= {self.threshold} → 리파인 불필요")
                return {
                    "question": question,
                    "initial_answer": initial_answer,
                    "final_answer": current_answer,
                    "refined": iteration > 0,
                    "used": True,
                    "iterations": iteration + 1,
                    "final_score": avg_score,
                    "critique_history": critique_history,
                }

            # 3. Refine
            if self.verbose:
                print(f"점수 {avg_score:.1f} < {self.threshold} → 리파인 실행")

            current_answer = self.refine(question, current_answer, critique)

        # Max iterations 도달
        final_critique = self.critique(question, current_answer)
        critique_history.append(final_critique)

        return {
            "question": question,
            "initial_answer": initial_answer,
            "final_answer": current_answer,
            "refined": True,
            "used": True,
            "iterations": self.max_iterations,
            "final_score": final_critique.get("avg_score", 0),
            "critique_history": critique_history,
        }


# --------------------------------------------
# 통합 Consultant (Self-Refine 포함)
# --------------------------------------------

class SmallBizConsultantWithRefine:
    """
    Self-Refine이 통합된 상담 시스템

    Flow:
    1. IntentRouter → RAG 또는 Agent
    2. 초안 생성
    3. Self-Refine (점수 < 7이면 개선)
    4. 최종 답변
    """

    def __init__(
        self,
        llm_model: str = LLM_MODEL,
        use_reranker: bool = False,
        refine_threshold: float = REFINE_THRESHOLD,
        verbose: bool = True,
    ):
        # 기존 Consultant 로드
        from importlib.util import spec_from_file_location, module_from_spec

        agent_path = BASE_DIR / "08_agent.py"
        spec = spec_from_file_location("agent", agent_path)
        agent_module = module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        self.consultant = agent_module.SmallBizConsultant(
            llm_model=llm_model,
            use_reranker=use_reranker,
            verbose=False,  # 내부 verbose 끔
        )

        # Self-Refiner
        self.refiner = SelfRefiner(
            llm_model=llm_model,
            threshold=refine_threshold,
            verbose=verbose,
        )

        self.verbose = verbose

    def consult(
        self,
        query: str,
        user_context: Optional[UserContext] = None,
        force_method: Optional[str] = None,
        skip_refine: bool = False,
    ) -> Dict[str, Any]:
        """
        상담 실행 (Self-Refine 포함)

        Args:
            query: 사용자 질문
            user_context: 사용자 컨텍스트
            force_method: 강제 지정 ("rag" 또는 "agent")
            skip_refine: True면 Self-Refine 스킵

        Returns:
            {
                "question": 질문,
                "answer": 최종 답변,
                "method": 사용된 방법,
                "refined": 리파인 여부,
                "score": 최종 점수,
            }
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"질문: {query}")
            print(f"{'='*50}")

        # 1. 초안 생성 (RAG 또는 Agent)
        result = self.consultant.consult(
            query=query,
            user_context=user_context,
            force_method=force_method,
        )

        initial_answer = result.get("answer", "")
        method = result.get("method", "unknown")

        if self.verbose:
            print(f"\n방법: {method}")
            print(f"초안 길이: {len(initial_answer)}자")

        # 2. Self-Refine (옵션)
        if skip_refine:
            return {
                "question": query,
                "answer": initial_answer,
                "method": method,
                "refined": False,
                "score": None,
            }

        refine_result = self.refiner.run(
            question=query,
            initial_answer=initial_answer,
        )

        if self.verbose:
            print(f"\n리파인 여부: {refine_result['refined']}")
            print(f"최종 점수: {refine_result.get('final_score', 'N/A')}")

        return {
            "question": query,
            "answer": refine_result["final_answer"],
            "initial_answer": initial_answer,
            "method": method,
            "refined": refine_result["refined"],
            "score": refine_result.get("final_score"),
            "iterations": refine_result.get("iterations", 0),
        }


# --------------------------------------------
# 테스트
# --------------------------------------------

def main():
    print("=" * 70)
    print("Self-Refine 시스템 테스트")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY 환경변수를 설정하세요")
        return

    # 테스트 1: SelfRefiner 단독 테스트
    print("\n" + "-" * 50)
    print("테스트 1: SelfRefiner 단독 테스트")
    print("-" * 50)

    refiner = SelfRefiner(verbose=True)

    # 일부러 품질 낮은 답변으로 테스트
    test_question = "카페 인스타그램 마케팅 어떻게 해야해?"
    test_answer = """인스타그램 마케팅을 하려면 사진을 예쁘게 찍어서 올리면 됩니다.
해시태그도 잘 달아야 하고, 스토리도 자주 올리세요.
그러면 반드시 팔로워가 늘어날 거예요."""

    result = refiner.run(test_question, test_answer)

    print(f"\n[초안]")
    print(test_answer)
    print(f"\n[최종 답변]")
    print(result["final_answer"])
    print(f"\n리파인됨: {result['refined']}, 최종점수: {result.get('final_score', 'N/A')}")

    # 테스트 2: 통합 시스템 테스트
    print("\n" + "-" * 50)
    print("테스트 2: SmallBizConsultantWithRefine 통합 테스트")
    print("-" * 50)

    user_ctx = UserContext(
        industry="cafe",
        location="강남",
        budget=300000,
        goal="신규 고객 유치",
    )

    consultant = SmallBizConsultantWithRefine(verbose=True)

    result = consultant.consult(
        query="카페 신메뉴 출시 마케팅 전략 알려줘",
        user_context=user_ctx,
    )

    print(f"\n[최종 답변]")
    print(result["answer"][:800])
    print(f"\n...")
    print(f"\n방법: {result['method']}, 리파인: {result['refined']}, 점수: {result.get('score', 'N/A')}")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
