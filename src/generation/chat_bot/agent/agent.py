"""
LangChain Agent 시스템 (트렌드 검색 + RAG 통합)

기능:
- Tool Calling 기반 Agent (LangChain 1.x 호환)
- web_search: 실시간 트렌드/뉴스 검색 (Tavily 우선, DuckDuckGo 폴백)
- rag_search: 소상공인 마케팅 사례 검색 (기존 RAG)
- 의도 기반 라우팅 (trend → Agent, 일반 → RAG)

사용법:
  python 08_agent.py

필요 패키지:
  pip install langchain langchain-openai langchain-community duckduckgo-search
"""

from __future__ import annotations

import json
import os
import sys
import re
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.chain import SmallBizRAG
from rag.prompts import UserContext, IntentRouter, SlotChecker

# --------------------------------------------
# 설정
# --------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore" / "chroma_db"


# --------------------------------------------
# Tools 정의 (함수 기반)
# --------------------------------------------

def _sanitize_text(text: str) -> str:
    """서로게이트 문자 및 인코딩 문제 해결"""
    if not text:
        return ""
    # 서로게이트 문자 제거 (UnicodeEncodeError 방지)
    return text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')


def web_search(query: str) -> str:
    """
    웹 검색 실행 (Tavily 우선, 실패 시 DuckDuckGo 폴백)

    Args:
        query: 검색할 쿼리 (예: "2024 카페 마케팅 트렌드")

    Returns:
        검색 결과 텍스트
    """
    # 1) Tavily (API 키 필요)
    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=TAVILY_API_KEY)
            results = client.search(query, max_results=5)

            items = results.get("results", []) if isinstance(results, dict) else []
            if items:
                formatted = []
                for i, r in enumerate(items, 1):
                    title = _sanitize_text(r.get('title', '제목 없음'))
                    content = _sanitize_text(r.get('content', ''))[:200]
                    url = r.get('url', '')
                    formatted.append(
                        f"[{i}] {title}\n"
                        f"    {content}...\n"
                        f"    링크: {url}"
                    )
                return "\n\n".join(formatted)
        except Exception:
            # fall back to DuckDuckGo
            pass

    # 2) DuckDuckGo fallback
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "검색 결과가 없습니다. (DuckDuckGo)"

        formatted = []
        for i, r in enumerate(results, 1):
            title = _sanitize_text(r.get('title', '제목 없음'))
            body = _sanitize_text(r.get('body', ''))[:200]
            href = r.get('href', '')
            formatted.append(
                f"[{i}] {title}\n"
                f"    {body}...\n"
                f"    링크: {href}"
            )

        return "\n\n".join(formatted)
    except ImportError:
        return "DuckDuckGo 패키지가 설치되지 않았습니다. pip install duckduckgo-search"
    except Exception as e:
        return f"검색 오류: {str(e)}"


def rag_search(query: str) -> str:
    """
    RAG 검색 실행 (소상공인 마케팅 사례)

    Args:
        query: 검색할 쿼리 (예: "카페 인스타그램 마케팅 사례")

    Returns:
        검색된 사례 텍스트
    """
    try:
        # RAG 인스턴스 생성 (싱글톤)
        if not hasattr(rag_search, "_rag_instance"):
            rag_search._rag_instance = SmallBizRAG(use_reranker=False)

        rag = rag_search._rag_instance

        # 검색 실행
        retrieved = rag.retrieve(query, k=7)

        if not retrieved:
            return "관련 사례를 찾지 못했습니다."

        # 결과 포맷팅
        results = []
        for i, doc in enumerate(retrieved, 1):
            meta = doc.get("metadata", {})
            content = doc.get("content", "")[:400]
            # 평점/리뷰 등 노이즈 제거
            content = re.sub(r"\d+(\.\d+)?점", "", content)
            content = re.sub(r"\d+개", "", content)
            content = content.replace("평점", "").replace("리뷰", "")
            results.append(
                f"[사례 {i}] {meta.get('title', '무제')} ({meta.get('location', '위치 미상')})\n"
                f"업종: {meta.get('industry', '미분류')}\n"
                f"내용: {content}..."
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"RAG 검색 오류: {str(e)}"


def hybrid_search(query: str) -> str:
    """
    웹 검색 + RAG 검색 결과를 묶어서 반환
    """
    web = web_search(query)
    rag = rag_search(query)
    return f"[웹 검색]\n{web}\n\n[사례 검색]\n{rag}"


# Tool 정의 (OpenAI Function Calling 형식)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "최신 트렌드, 뉴스, 실시간 정보를 웹에서 검색합니다. '요즘', '최근', '트렌드', '유행' 관련 질문에 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 쿼리 (예: '2024 카페 마케팅 트렌드')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "소상공인 마케팅 성공 사례 데이터베이스를 검색합니다. 카페, 맛집, 베이커리 등 실제 매장의 마케팅 전략을 찾습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 쿼리 (예: '카페 인스타그램 마케팅 사례')",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# Tool 함수 매핑
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "rag_search": rag_search,
}


# --------------------------------------------
# Agent 시스템 프롬프트
# --------------------------------------------

import datetime
CURRENT_YEAR = datetime.datetime.now().year

AGENT_SYSTEM_PROMPT = f"""당신은 소상공인 마케팅 전문 컨설턴트입니다.

## 중요: 연도 처리 규칙
- 현재 연도: {CURRENT_YEAR}년
- **사용자가 특정 연도를 명시하면 그 연도를 사용** (예: "2025년 밈" → "2025 밈")
- 연도 미명시 + "최신/요즘/유행" → 현재 연도({CURRENT_YEAR}) 사용

## 도구 사용 규칙
1. web_search 쿼리 작성 시:
   - 사용자 질문의 핵심 키워드를 반드시 유지 (밈→밈, 트렌드→트렌드)
   - **사용자가 "2025년"이라고 말하면 "2025"로 검색** (현재 연도로 바꾸지 말 것)
   - 연도 미명시 시에만 "{CURRENT_YEAR}" 추가
   - 예: "2025년 밈" → "2025 밈 트렌드" (O)
   - 예: "2025년 밈" → "2026 밈 트렌드" (X - 연도 변경 금지)
   - 예: "최신 밈" → "{CURRENT_YEAR} 최신 밈 트렌드" (O)
   - ❌ 잘못된 예: "밈" → "카페 마케팅 트렌드" (키워드 변경 금지)

2. "요즘", "최근", "트렌드", "유행", "밈" 키워드 → web_search 1회 + rag_search 1회
3. "사례", "성공", "매장" 관련 질문 → rag_search
4. 복합 질문은 두 도구 모두 활용 (중복 호출 금지)

## 답변 규칙 (엄격 적용)
1) 결과 보장 금지 ("반드시", "100%", "확실히" 사용 금지)
2) 숫자는 검색 결과에서 확인된 것만 사용
   - 검색 결과에 숫자가 없으면 "구체적 수치는 검색 결과에서 확인되지 않음"이라고 명시
   - **절대 "20% 증가", "30% 상승" 같은 숫자를 지어내지 말 것**
3) 참고 매장 표기 (환각 금지):
   - **사례 검색 결과에 실제로 있는 매장만 사용**
   - 형식: {{매장명}}({{지역}})만 허용
   - **가짜 매장 생성 절대 금지** (예: "사례[1]" 같은 플레이스홀더 금지)
   - 참고 매장이 없으면 섹션을 생략
4) 평점/별점 언급 금지, 프랜차이즈 지양(로컬 우선)
5) 한국어, 간결/친근, 상식적 조언 생략
6) **검색 도구를 사용하지 않고 답변할 때는 "검색 없이 보유 지식으로 답변합니다"라고 먼저 명시**

## 환각 방지 체크리스트 (매 답변 전 확인)
- [ ] 내가 언급하는 숫자가 검색 결과에 있는가?
- [ ] 내가 언급하는 참고 매장이 실제 사례 결과인가?
- [ ] 지어낸 정보가 포함되어 있지 않은가?

## 출력 형식
- 요약 2줄
- 실행 아이디어 3개 (검색 결과 기반)
- 주의/리스크 2개
- 참고 매장 (있을 때만)"""


# --------------------------------------------
# TrendAgent 클래스 (Tool Calling 기반)
# --------------------------------------------

class TrendAgent:
    """
    트렌드 검색 Agent (LangChain 1.x 호환)

    Features:
    - OpenAI Tool Calling 기반
    - 웹 검색 + RAG 검색 통합
    - 멀티턴 대화 지원
    """

    def __init__(
        self,
        llm_model: str = LLM_MODEL,
        max_iterations: int = 5,
        verbose: bool = False,
    ):
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.3,
            api_key=OPENAI_API_KEY,
        )
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.intent_router = IntentRouter()

    def run(
        self,
        query: str,
        user_context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        """
        Agent 실행

        Args:
            query: 사용자 질문
            user_context: 사용자 컨텍스트

        Returns:
            {"question": ..., "answer": ..., "intent": ..., "steps": [...]}
        """
        intent = self.intent_router.classify(query)

        # 컨텍스트 추가
        enhanced_query = query
        if user_context:
            ctx_parts = []
            if user_context.industry:
                ctx_parts.append(f"업종: {user_context.industry}")
            if user_context.location:
                ctx_parts.append(f"위치: {user_context.location}")
            if user_context.budget:
                ctx_parts.append(f"예산: {user_context.budget:,}원")
            if ctx_parts:
                enhanced_query = f"[사용자 정보: {', '.join(ctx_parts)}]\n{query}"

        # 메시지 초기화
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=enhanced_query),
        ]

        steps = []
        # 도구 실행 결과 캐시 (중복 실행 방지)
        web_result = None
        rag_result = None

        # Tool Calling 루프 (최대 2회 시도 후 종료)
        for iteration in range(min(self.max_iterations, 2)):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            response = self.llm.invoke(messages, tools=TOOLS)

            # Tool Call이 없으면 최종 답변
            if not response.tool_calls:
                if self.verbose:
                    print("Final Answer Generated")
                return {
                    "question": query,
                    "answer": response.content,
                    "intent": intent,
                    "steps": steps,
                }

            # Tool Call 처리
            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if self.verbose:
                    print(f"Tool: {tool_name}")
                    print(f"Args: {tool_args}")

                tool_func = TOOL_FUNCTIONS.get(tool_name)
                # 중복 실행 방지: 동일 도구는 1회만 실제 호출
                if tool_name == "web_search" and web_result is not None:
                    result = "중복 web_search 호출이 감지되어 스킵합니다."
                elif tool_name == "rag_search" and rag_result is not None:
                    result = "중복 rag_search 호출이 감지되어 스킵합니다."
                elif tool_func:
                    result = tool_func(**tool_args)
                    if tool_name == "web_search":
                        web_result = result
                    if tool_name == "rag_search":
                        rag_result = result
                else:
                    result = f"Unknown tool: {tool_name}"

                if self.verbose:
                    print(f"Result: {result[:200]}...")

                steps.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )

            # 트렌드 의도라면 웹+RAG 근거를 보강하고 인용을 강제
            if intent == "agent":
                # 필요시 보충 실행 (중복 없이)
                if web_result is None:
                    web_result = web_search(query)
                    steps.append({"tool": "web_search", "args": {"query": query}, "result": web_result})
                if rag_result is None:
                    rag_result = rag_search(query)
                    steps.append({"tool": "rag_search", "args": {"query": query}, "result": rag_result})

                hybrid_result = f"[웹 검색]\n{web_result}\n\n[사례 검색]\n{rag_result}"
                steps.append({
                    "tool": "hybrid_search",
                    "args": {"query": query},
                    "result": hybrid_result,
                })
                messages.append(
                    HumanMessage(
                        content=(
                            "[추가 근거] 아래 웹/사례 결과를 참고해 답변하세요. "
                            "참고 매장 섹션에는 사례 검색에서 실제로 참고한 매장만 bullet로 작성하세요.\n\n"
                            f"{hybrid_result}"
                        )
                    )
                )

                # 보강 후 바로 최종 답변 생성
                final_response = self.llm.invoke(messages)
                return {
                    "question": query,
                    "answer": final_response.content,
                    "intent": intent,
                    "steps": steps,
                }

        # Max iterations 도달
        final_response = self.llm.invoke(messages)
        return {
            "question": query,
            "answer": final_response.content,
            "intent": intent,
            "steps": steps,
        }

    def should_use_agent(self, query: str) -> bool:
        """Agent 사용 여부 판단 (trend_web 인텐트일 때)"""
        intent = self.intent_router.classify(query)
        return intent == "trend_web"


# --------------------------------------------
# CHITCHAT 응답 프롬프트
# --------------------------------------------

CHITCHAT_SYSTEM_PROMPT = """당신은 소상공인 마케팅 컨설턴트 챗봇입니다.
사용자가 일상적인 대화를 하면 친근하게 응답하세요.

규칙:
- 간결하고 친근하게 (1-2문장)
- 마케팅 관련 질문이 있으면 언제든 도와드릴 수 있다고 안내
- 이모지 최소화
- 한국어로 응답"""


CLARIFY_SYSTEM_PROMPT = """당신은 소상공인 마케팅 컨설턴트 챗봇입니다.
사용자가 도움을 요청했지만 구체적인 질문이 없습니다.

역할:
- 친근하게 인사하고, 어떤 고민인지 구체적으로 물어보세요
- 예시를 들어 질문을 유도하세요

예시 응답:
"안녕하세요! 네, 기꺼이 도와드릴게요. 어떤 부분이 고민이세요?
예를 들어:
- 광고 예산을 어떻게 써야 할지
- 인스타그램/네이버 중 어디에 집중할지
- 신메뉴 홍보 방법
등 구체적으로 말씀해주시면 맞춤 조언 드릴 수 있어요!"

규칙:
- 1-3문장으로 간결하게
- 구체적 질문 예시 2-3개 제시
- 이모지 최소화
- 한국어로 응답"""


# --------------------------------------------
# 통합 라우터 (5개 인텐트 지원)
# --------------------------------------------

class SmallBizConsultant:
    """
    통합 상담 시스템 (RAG + Agent + LLM)

    5개 인텐트에 따라 라우팅:
    - chitchat → LLM 직접 응답
    - trend_web → Agent (웹 검색 + RAG)
    - task_action → 생성 모듈 (현재는 RAG fallback)
    - marketing_counsel → RAG
    - doc_rag → RAG
    """

    def __init__(
        self,
        llm_model: str = LLM_MODEL,
        use_reranker: bool = False,
        verbose: bool = True,
    ):
        from langchain_openai import ChatOpenAI

        # RAG 초기화
        print("RAG 초기화 중...")
        self.rag = SmallBizRAG(
            llm_model=llm_model,
            use_reranker=use_reranker,
        )

        # Agent 초기화
        print("Agent 초기화 중...")
        self.agent = TrendAgent(
            llm_model=llm_model,
            verbose=verbose,
        )

        # Chitchat용 LLM
        self.chat_llm = ChatOpenAI(
            model=llm_model,
            temperature=0.8,
            api_key=OPENAI_API_KEY,
        )

        self.intent_router = IntentRouter()
        self.slot_checker = SlotChecker()
        self.verbose = verbose

        # 현재 질문 저장 (슬롯 채워진 후 재실행용)
        self.pending_query: Optional[str] = None
        self.pending_intent: Optional[str] = None
        self.pending_slot: Optional[str] = None
        self.pending_ambiguous_query: Optional[str] = None
        # 세션 내 컨텍스트 유지(외부에서 user_context를 넘기지 않는 경우 대비)
        self.session_context: Optional[UserContext] = None

    def _build_filter(self, user_context: Optional[UserContext], intent: str) -> Optional[Dict[str, Any]]:
        """슬롯에서 추출된 업종/지역을 기반으로 검색 필터 생성"""
        if not user_context:
            return None

        clauses: List[Dict[str, Any]] = []
        if user_context.industry:
            clauses.append({"industry": user_context.industry})

        # 지역 고정이 필요한 인텐트에만 지역 필터 적용
        if user_context.location and intent in ("marketing_counsel", "doc_rag", "task_action"):
            clauses.append({"location": user_context.location})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _handle_chitchat(self, query: str) -> Dict[str, Any]:
        """일상 대화 처리 (LLM 직접 응답)"""
        messages = [
            {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response = self.chat_llm.invoke(messages)
        return {
            "question": query,
            "answer": response.content,
            "intent": "chitchat",
            "method": "llm",
        }

    def _handle_clarify(self, query: str) -> Dict[str, Any]:
        """구체적 질문 유도 (대화 시작)"""
        messages = [
            {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response = self.chat_llm.invoke(messages)
        return {
            "question": query,
            "answer": response.content,
            "intent": "conversation_start",
            "method": "clarify",
        }

    def _handle_slot_question(
        self,
        query: str,
        missing_slot: str,
        intent: str
    ) -> Dict[str, Any]:
        """누락된 슬롯에 대한 질문 생성"""
        # 원래 질문 저장 (슬롯 채워진 후 재실행용)
        self.pending_query = query
        self.pending_intent = intent
        self.pending_slot = missing_slot

        # 친근한 질문 생성
        slot_question = self.slot_checker.get_slot_question(missing_slot)

        # 맥락을 포함한 응답 생성
        context_response = f"좋은 질문이에요! 더 정확한 답변을 드리기 위해 한 가지만 여쭤볼게요.\n\n{slot_question}"

        return {
            "question": query,
            "answer": context_response,
            "intent": intent,
            "method": "slot_filling",
            "missing_slot": missing_slot,
            "pending_query": query,
        }

    def _is_ambiguous(self, query: str) -> bool:
        """모호성 판단을 비활성화 (사용자에게 추가 입력을 맡김)"""
        return False

    def _build_ambiguity_prompt(self, query: str) -> str:
        """모호한 발화에 대한 재질문 생성 (상황별로 분기)"""
        change_keywords = ["바꿔", "변경", "바꾸", "바꾸려", "변경하려", "옮기"]
        if any(kw in query for kw in change_keywords):
            return (
                "지금 말씀하신 내용이 두 가지로 해석될 수 있어요.\n"
                "1) 기존 업종/지역을 다른 걸로 바꾸려는 건가요?\n"
                "2) 새 업종/지역을 새로 검토하려는 건가요?\n"
                "선택하거나, 업종/지역을 구체적으로 적어주세요. 예: '카페→술집, 순천→강남' 또는 '새 업종/지역을 고민 중'."
            )

        # 일반적인 애매 발화일 때는 범용 확인 질문
        return (
            "어떤 점이 궁금하신지 조금 더 구체히 말씀해 주세요. 예:\n"
            "- 프로모션 아이디어 알려줘 (업종/지역/예산 포함 가능)\n"
            "- 예산 50만 원일 때 채널 별로 어떻게 집행할까?\n"
            "- 어떤 고객을 타겟으로 할지 추천해줘"
        )

    def _is_vague_marketing_issue(self, query: str, intent: str) -> bool:
        """마케팅 관련 고민/어려움 표현이지만 구체 주제가 없는지 판단"""
        if intent not in ("marketing_counsel", "doc_rag"):
            return False

        text = query.lower().strip()
        vague_signals = [
            "고민", "어려움", "힘들", "막막", "모르겠", "감이 안",
            "어떻게 할지", "어찌", "걱정",
        ]
        marketing_signals = ["광고", "마케팅", "홍보", "프로모션"]
        specific_signals = [
            "예산", "비용", "광고비", "채널", "플랫폼", "인스타", "네이버",
            "검색", "키워드", "콘텐츠", "사진", "카피", "문구",
            "전략", "방법", "타겟", "고객", "매출", "방문자",
            "리뷰", "후기", "전환", "roas", "cpa", "쿠폰", "이벤트",
            "브랜딩", "도달", "유입",
        ]

        has_vague = any(kw in text for kw in vague_signals)
        has_marketing = any(kw in text for kw in marketing_signals)
        has_specific = any(kw in text for kw in specific_signals)

        return has_vague and has_marketing and not has_specific

    def _handle_issue_clarify(self, query: str, intent: str) -> Dict[str, Any]:
        """구체적인 고민 포인트를 묻는 확인 질문"""
        prompt = (
            "도움 드릴게요! 구체적으로 어떤 부분이 가장 고민인가요?\n"
            "예: 채널 선택, 예산 배분, 콘텐츠/카피, 타겟 설정, 매출/유입 개선 등"
        )
        return {
            "question": query,
            "answer": prompt,
            "intent": intent,
            "method": "clarify_issue",
        }

    def _resolve_ambiguity_selection(self, query: str) -> Optional[str]:
        """재질문에 대한 단답(1번/2번 등)을 해석"""
        stripped = query.strip()
        option1 = ["1", "1번", "첫번째", "첫 번째", "첫번", "첫 번"]
        option2 = ["2", "2번", "두번째", "두 번째", "두번", "두 번"]
        if stripped in option1:
            return "option1"
        if stripped in option2:
            return "option2"
        return None

    def consult(
        self,
        query: str,
        user_context: Optional[UserContext] = None,
        force_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        상담 실행 (Slot-Filling 포함)

        Args:
            query: 사용자 질문
            user_context: 사용자 컨텍스트
            force_method: 강제 지정 ("rag", "agent", "llm")

        Returns:
            {
                "question": 질문,
                "answer": 답변,
                "method": 사용된 방법,
                "intent": 분류된 의도,
            }
        """
        # 입력 텍스트를 안전하게 정제 (서로게이트 제거)
        query = _sanitize_text(query)

        # 1. 쿼리에서 슬롯 추출하여 UserContext 업데이트
        if user_context is None:
            user_context = self.session_context
        if user_context is None:
            user_context = UserContext()

        prev_context = copy.deepcopy(user_context) if user_context else UserContext()
        user_context = self.slot_checker.update_context_from_query(query, user_context)
        self.session_context = user_context

        # 슬롯 질문 직후라면, 느슨한 지역 추정으로라도 채워보기
        if self.pending_slot == "location" and user_context and not user_context.location:
            coarse = self.slot_checker.coarse_location_guess(query)
            if coarse:
                user_context.location = coarse

        resolved_ambiguity = False
        # 직전 턴이 모호성 확인용이었다면, 추가 설명/선택을 원 질문에 합쳐 재실행
        if self.pending_ambiguous_query:
            selection = self._resolve_ambiguity_selection(query)
            if self.verbose:
                print("모호성 해소 → 원 질문 재실행")
            if selection == "option1":
                extra = "[사용자 선택]: 기존 업종/지역을 다른 것으로 바꾸려는 중"
            elif selection == "option2":
                extra = "[사용자 선택]: 새 업종/지역을 새로 검토 중"
            else:
                extra = f"[추가 설명]: {query}"
            query = _sanitize_text(f"{self.pending_ambiguous_query}\n{extra}")
            self.pending_ambiguous_query = None
            resolved_ambiguity = True

        # 2. 이전 턴에서 슬롯을 물었다면, 채워졌는지 확인 후 원 질문 재실행
        if self.pending_query and self.pending_intent:
            # 누락 슬롯 재확인 (업데이트된 컨텍스트 사용)
            missing_slots = self.slot_checker.check_required_slots(
                intent=self.pending_intent,
                user_context=user_context,
                query=self.pending_query,
            )
            if missing_slots:
                first_missing = missing_slots[0]
                if self.verbose:
                    print(f"슬롯 보충 필요(pending): {missing_slots} → 질문: {first_missing}")
                return self._handle_slot_question(self.pending_query, first_missing, self.pending_intent)

            # 모든 필수 슬롯이 채워졌으면 원 질문으로 이어서 처리
            if self.verbose:
                print(f"슬롯 충족 → 원 질문 재실행: {self.pending_query}")
            query = self.pending_query
            intent = self.pending_intent
            routing = self.intent_router.get_routing(intent)
            # pending 상태 해제
            self.pending_query = None
            self.pending_intent = None
            self.pending_slot = None
        else:
            # 2. 의도 분류
            intent = self.intent_router.classify(query)
            routing = self.intent_router.get_routing(intent)

        # 1-A. 새롭게 들어온 발화가 모호하면 재질문
        if (
            not self.pending_query  # 슬롯 채우기 중이 아닐 때만
            and self.pending_ambiguous_query is None
            and not resolved_ambiguity
            and self._is_ambiguous(query)
        ):
            if self.verbose:
                print("모호한 발화 감지 → 재질문")
            self.pending_ambiguous_query = query
            return {
                "question": query,
                "answer": self._build_ambiguity_prompt(query),
                "intent": "clarify_ambiguous",
                "method": "clarify_ambiguous",
            }

        # 질문/명령 신호 감지
        has_question_signal = any(
            kw in query
            for kw in ["?", "어떻게", "방법", "알려", "줘", "궁금", "추천", "뭐", "무엇", "프로모션", "마케팅", "전략"]
        )
        slot_update_signal = any(kw in query for kw in ["지역", "동네", "업종", "예산", "목표"])
        newly_set = []
        if not prev_context.industry and user_context.industry:
            newly_set.append("industry")
        if not prev_context.location and user_context.location:
            newly_set.append("location")
        if not prev_context.budget and user_context.budget:
            newly_set.append("budget")
        if not prev_context.goal and user_context.goal:
            newly_set.append("goal")

        # 2-A. 슬롯만 제공한 경우: 짧게 확인하고 질문을 유도
        if (
            not self.pending_query  # 슬롯 채우기 중이 아닐 때만
            and not has_question_signal
            and (slot_update_signal or newly_set)
        ):
            if self.verbose:
                print("컨텍스트 업데이트 감지 → 질문 요청")
            ack = "업종/지역 정보를 업데이트했어요. 궁금한 점을 말씀해 주세요! 예: '프로모션 아이디어 알려줘', '예산 50만 원 배분 어떻게 해?'"
            return {
                "question": query,
                "answer": ack,
                "intent": "slot_update",
                "method": "slot_ack",
            }

        # 2-B. 질문 신호가 전혀 없는 서술형 입력은 일반 답변을 내지 않고 명확한 질문을 요청
        if (
            not self.pending_query
            and not has_question_signal
            and intent not in ("chitchat", "conversation_start")
            and not resolved_ambiguity  # 방금 모호성 해소된 경우는 건너뜀
        ):
            if self.verbose:
                print("질문 신호 없음 → 명확한 질문 요청")
            ack = "어떤 점이 궁금하신가요? 예: '프로모션 아이디어 알려줘', '예산 50만 원 배분 어떻게 해?'처럼 질문 형태로 말씀해 주세요."
            return {
                "question": query,
                "answer": ack,
                "intent": intent,
                "method": "ask_for_question",
            }

        # 강제 지정이 있으면 우선
        if force_method:
            routing = force_method

        if self.verbose:
            print(f"\n의도: {intent} → 라우팅: {routing}")
            if user_context:
                ctx_parts = []
                if user_context.industry:
                    ctx_parts.append(f"업종:{user_context.industry}")
                if user_context.location:
                    ctx_parts.append(f"지역:{user_context.location}")
                if user_context.budget:
                    ctx_parts.append(f"예산:{user_context.budget:,}원")
                if ctx_parts:
                    print(f"컨텍스트: {' | '.join(ctx_parts)}")

        # 3. Slot-Filling: 필수 슬롯 확인 (RAG/Generator 라우팅일 때만)
        if routing in ("rag", "generator"):
            missing_slots = self.slot_checker.check_required_slots(
                intent=intent,
                user_context=user_context,
                query=query,
            )

            if missing_slots:
                # 누락된 슬롯이 있으면 질문
                first_missing = missing_slots[0]
                if self.verbose:
                    print(f"누락 슬롯: {missing_slots} → 질문: {first_missing}")
                return self._handle_slot_question(query, first_missing, intent)

        # 3-A. 슬롯이 충족되었지만 고민이 모호한 경우 구체화 질문
        if self._is_vague_marketing_issue(query, intent):
            if self.verbose:
                print("모호한 마케팅 고민 → 구체 질문 요청")
            return self._handle_issue_clarify(query, intent)

        # 4. 라우팅별 처리
        if routing == "llm":
            # CHITCHAT: LLM 직접 응답
            return self._handle_chitchat(query)

        elif routing == "clarify":
            # CONVERSATION_START: 구체적 질문 유도
            return self._handle_clarify(query)

        elif routing == "agent":
            # TREND_WEB: Agent (웹 검색)
            result = self.agent.run(query, user_context)
            result["method"] = "agent"
            result["intent"] = intent
            return result

        elif routing == "generator":
            # TASK_ACTION: 생성 작업 (현재는 RAG로 fallback)
            # TODO: 별도 생성 모듈 연결
            filter_kwargs = self._build_filter(user_context, intent)
            result = self.rag.query(
                question=query,
                k=7,
                filter=filter_kwargs,
                user_context=user_context,
            )
            result["method"] = "rag"
            result["intent"] = intent
            return result

        else:
            # MARKETING_COUNSEL, DOC_RAG: RAG
            filter_kwargs = self._build_filter(user_context, intent)
            result = self.rag.query(
                question=query,
                k=5,
                filter=filter_kwargs,
                user_context=user_context,
            )
            result["method"] = "rag"
            result["intent"] = intent
            return result


# --------------------------------------------
# 테스트
# --------------------------------------------

def main():
    print("=" * 70)
    print("LangChain Agent 시스템 테스트 (Tool Calling 기반)")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY 환경변수를 설정하세요")
        return

    # 사용자 컨텍스트
    user_ctx = UserContext(
        industry="cafe",
        location="강남",
        budget=300000,
        goal="신규 고객 유치",
    )

    # 테스트 1: Agent만 테스트 (트렌드 질문)
    print("\n" + "-" * 50)
    print("테스트 1: TrendAgent 직접 테스트")
    print("-" * 50)

    agent = TrendAgent(verbose=False)
    result = agent.run(
        query="요즘 유행하는 카페 마케팅 트렌드가 뭐야?",
        user_context=user_ctx,
    )

    print(f"\n질문: {result['question']}")
    print(f"의도: {result.get('intent', 'N/A')}")
    print(f"사용된 도구: {[s['tool'] for s in result.get('steps', [])]}")
    print(f"\n답변:\n{result['answer']}")

    # 테스트 2: 통합 시스템 테스트
    print("\n" + "-" * 50)
    print("테스트 2: SmallBizConsultant 통합 테스트")
    print("-" * 50)

    consultant = SmallBizConsultant(verbose=True)

    # 트렌드 질문 (Agent 경로)
    print("\n[트렌드 질문]")
    result = consultant.consult(
        query="2024년 소상공인 SNS 마케팅 최신 트렌드 알려줘",
        user_context=user_ctx,
    )
    print(f"방법: {result['method']}")
    print(f"답변: {result['answer'][:500]}...")

    # 사례 질문 (RAG 경로)
    print("\n[사례 질문]")
    result = consultant.consult(
        query="카페 인스타그램 마케팅 성공 사례 알려줘",
        user_context=user_ctx,
    )
    print(f"방법: {result['method']}")
    print(f"답변: {result['answer'][:500]}...")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
