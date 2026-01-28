"""
프롬프트 엔지니어링 모듈

소상공인 마케팅 상담 챗봇을 위한 프롬프트 템플릿 정의

구성:
1. SYSTEM_PROMPT: 기본 시스템 프롬프트 (역할, 규칙)
2. CONTEXT_TEMPLATE: 컨텍스트 조립 템플릿
3. TASK_PROMPTS: 태스크별 프롬프트 (추천, 광고문구, 전략)
4. PromptBuilder: 프롬프트 동적 생성 클래스
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================
# 1. SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """당신은 소상공인 온라인 마케팅 컨설턴트입니다.

## 역할
- 상황을 파악하고 실행 가능한 전략을 제안합니다
- 제공된 사례/데이터 범위 안에서만 근거를 사용합니다

## 절대 규칙
1) 결과 보장 금지: "반드시/확실히/100%" 등 금지
2) 근거 표기: 참고한 매장은 {제목}({지역}) 형식으로만 명시
3) 컨텍스트 준수: 주어진 사례와 사용자 정보 밖으로 추측하지 않음
4) 구체성: 실행 가능한 값 포함. **숫자는 근거가 있을 때만 사용**
5) 채널 현실성: 인스타그램/틱톡/네이버 등 최신 채널을 우선하며, 페이스북 페이지 광고처럼 트렌드에서 밀린 채널은 근거가 있을 때만 제한적으로 제안
6) 프랜차이즈 최소화: 프랜차이즈 브랜드는 가급적 언급/추천하지 말고, 로컬/개인 매장을 우선 제안
7) 평점/별점 언급 금지: 리뷰 점수나 별점은 언급하지 않음
8) 상식 배제: 누구나 아는 상식적 조언은 생략하고, 플랫폼/실행 특화 지침에 집중
9) 숫자/근거: 근거가 있는 항목에만 숫자(빈도/예산/기간 등) 포함. 근거가 없으면 숫자 없이 서술
10) 톤: 친근하지만 간결, 불필요한 장식/이모지 최소화

## 출력 형식
- 핵심 요약 → 세부 실행 → 참고 매장(있을 때만) 순서로 제시"""


# ============================================
# 2. CONTEXT TEMPLATES
# ============================================

CONTEXT_TEMPLATE = """## 참고 매장 정보
{retrieved_cases}

## 사용자 정보
{user_context}

## 대화 히스토리
{chat_history}

## 현재 질문
{query}"""


CASE_TEMPLATE = """### [{rank}] {title}
- 위치: {location} | 업종: {industry}
- 요약: {description}"""


# ============================================
# 3. TASK-SPECIFIC PROMPTS
# ============================================

TASK_PROMPTS = {
    # 일반 정보 질문 (순위, 통계, 개념 설명 등)
    "info_query": """사용자가 일반적인 정보를 물어봤습니다.
질문에 직접적으로 답변하세요.

형식:
1) 핵심 답변: 질문에 대한 직접적인 답 (1-2문장)
2) 상세 설명: 필요한 경우 추가 설명

주의:
- 매장 추천이나 방문 팁은 제공하지 마세요
- 질문과 관련 없는 정보는 포함하지 마세요""",

    # 매장 추천
    "recommend": """다음 형식을 지켜 답하세요:
1) 한눈에 보기: 추천 요약 2줄
2) 추천 매장: 1~3곳, "{제목} - 이유(사용자 니즈 매칭)" 형태
3) 방문 팁: 시간대/좌석/메뉴 등 2~3개
제한: 평점/별점 언급 없이 작성""",

    # 광고 문구 생성
    "ad_copy": """광고 문구 생성은 별도 전담 모듈에서 처리합니다.

지금은 다음 정보를 정리해 전달하세요:
- 업종/지역/타겟/목표/예산
- 강조할 메뉴나 행사
- 참고 사례 {제목}({지역})

간단히 요구사항을 bullet로 정리해 드리고, 광고팀에 넘기도록 안내하세요.""",

    # 마케팅 전략 상담
    "strategy": """형식:
1) 현재 상황 요약 (사용자 정보 반영)
2) 실행 전략 3가지: 각 항목마다 **근거가 있는 경우에만** 숫자(빈도/예산/기간/목표 도달 등) 포함. 근거가 없으면 숫자 없이 서술
3) 주의사항 2개
4) 다음 단계 체크리스트 3개
채널 우선순위: 인스타그램/틱톡/네이버 중심으로 제안하고, 페이스북 등 노후 채널은 최신 근거가 있을 때만 제한적으로 언급
섹션 제목을 그대로 사용하세요: "현재 상황", "실행", "주의사항", "다음 단계", "참고 매장".
참고 매장: 하단에 bullet로 {제목}({지역})만 명시 (실제로 참고한 매장만)
제한: 평점/별점 언급 없이 작성""",

    # 트렌드 분석 (Agent 연동용)
    "trend": """최신 트렌드 요약 후 적용 포인트를 제안하세요:
1) 트렌드 요약 3줄
2) 우리 업종 적용 아이디어 3개: **근거가 있는 경우에만** 숫자(빈도/예산/기간 등) 포함. 근거가 없으면 숫자 없이 서술
3) 리스크/주의 2개
4) 참고 매장: {제목}({지역})만 bullet로 명시 (실제로 참고한 매장만)""",

    # 매장 사진 촬영/업로드 가이드 (네이버/소셜 최적화)
    "photo_guide": """상식적 조언은 생략하고, 실행 가능한 촬영·업로드 가이드만 제시하세요.

구성:
1) 촬영 세트리스트 (8~12컷): 간판/입구, 베스트 좌석, 인기 메뉴 접사, 바/카운터, 화장실·콘센트, 동선/주차, 전체 전경, 고객 실루엣(초상권 주의), 인테리어 디테일
2) 설정/포맷: 권장 비율(네이버 대표 3:4, 메뉴 1:1), 해상도 ≥2000px, 수평/왜곡 방지, RAW→JPEG 변환 시 품질 85~90%
3) 연출 팁: 네온/간판 글자 노출, 브랜드 컬러 소품 배치, 반사·지저분 배경 제거 체크리스트, 실루엣/손만 등장
4) 업로드 팁: 파일명 규칙(업종_지역_키워드), 대표→메뉴→실내→외부 순서, 과도한 필터 금지, 노출 잘 된 컷만 사용
5) 네이버 플레이스 최적화: 대표사진은 정면/간판 3:4, 메뉴는 1:1, 업로드 시점은 점심/퇴근 전(노출량↑), 텍스트/워터마크 없는 원본 사용
6) 검수 체크리스트 5개: 흐림/노이즈, 색 온도, 수평, 개인정보 노출, 저작권 문제""",

    # 일반 질문
    "general": """질문에 직접 답하고, 필요 시 사례 근거를 덧붙입니다.
구성: 답변 요약 2줄 → 핵심 내용 → 참고 매장(있을 때만)."""
}


# ============================================
# 4. PROMPT BUILDER CLASS
# ============================================

@dataclass
class UserContext:
    """사용자 컨텍스트"""
    industry: Optional[str] = None  # 업종
    location: Optional[str] = None  # 지역
    budget: Optional[int] = None    # 예산
    goal: Optional[str] = None      # 목표
    platform: Optional[str] = None  # 타겟 플랫폼

    def to_string(self) -> str:
        parts = []
        if self.industry:
            parts.append(f"- 업종: {self.industry}")
        if self.location:
            parts.append(f"- 지역: {self.location}")
        if self.budget:
            parts.append(f"- 예산: {self.budget:,}원")
        if self.goal:
            parts.append(f"- 목표: {self.goal}")
        if self.platform:
            parts.append(f"- 플랫폼: {self.platform}")
        return "\n".join(parts) if parts else "정보 없음"


class PromptBuilder:
    """프롬프트 동적 생성 클래스"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        user_context: Optional[UserContext] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """컨텍스트 조립"""

        # 검색된 사례 포맷팅
        cases_text = self._format_cases(retrieved_docs)
        if cases_text == "관련 사례 없음":
            cases_text = "관련 사례 없음 (일반 원칙 기반으로 답변)"

        # 사용자 컨텍스트
        user_text = user_context.to_string() if user_context else "정보 없음"

        # 대화 히스토리
        history_text = self._format_history(chat_history) if chat_history else "없음"

        return CONTEXT_TEMPLATE.format(
            retrieved_cases=cases_text,
            user_context=user_text,
            chat_history=history_text,
            query=query,
        )

    def _format_cases(self, docs: List[Dict[str, Any]]) -> str:
        """검색된 문서를 포맷팅"""
        if not docs:
            return "관련 사례 없음"

        formatted = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            content = doc.get("content", "")

            # 설명 추출 (내용의 첫 200자)
            description = content[:200] + "..." if len(content) > 200 else content
            location = meta.get("location") or "미상"
            # 불필요한 괄호/공백 제거
            location = str(location).strip("(){}[] ").strip()

            case_text = CASE_TEMPLATE.format(
                rank=i,
                title=meta.get("title", "제목 없음"),
                location=location,
                industry=self._translate_industry(meta.get("industry", "")),
                description=description,
            )
            formatted.append(case_text)

        return "\n\n".join(formatted)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """대화 히스토리 포맷팅 (최근 5턴)"""
        if not history:
            return "없음"

        recent = history[-10:]  # 최근 5턴 (user + assistant)
        formatted = []
        for msg in recent:
            role = "사용자" if msg["role"] == "user" else "상담사"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            formatted.append(f"**{role}**: {content}")

        return "\n".join(formatted)

    def _translate_industry(self, industry: str) -> str:
        """업종 코드를 한글로 변환"""
        mapping = {
            "cafe": "카페",
            "restaurant": "음식점",
            "bakery": "베이커리",
            "dessert": "디저트",
            "bar": "바/주점",
            "": "기타",
        }
        return mapping.get(industry, industry)

    def _format_sources(self, docs: List[Dict[str, Any]]) -> str:
        """출처 리스트 포맷팅 (중복 제거, 제목/지역만)"""
        if not docs:
            return "없음"

        seen = set()
        sources = []
        for doc in docs:
            meta = doc.get("metadata", {})
            title = meta.get("title")
            location = meta.get("location")
            if not title:
                continue

            location = (location or "미상")
            location = str(location).strip("(){}[] ").strip()
            key = (title, location)
            if key in seen:
                continue
            seen.add(key)
            sources.append(f"- {title} ({location})")

        return "\n".join(sources) if sources else "없음"

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        task: str = "general",
        user_context: Optional[UserContext] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        최종 프롬프트 생성 (OpenAI messages 형식)

        Args:
            query: 사용자 질문
            retrieved_docs: 검색된 문서 리스트
            task: 태스크 유형 (recommend, ad_copy, strategy, trend, general)
            user_context: 사용자 컨텍스트
            chat_history: 대화 히스토리

        Returns:
            OpenAI chat messages 형식의 리스트
        """
        # 컨텍스트 조립
        context = self.build_context(
            retrieved_docs=retrieved_docs,
            query=query,
            user_context=user_context,
            chat_history=chat_history,
        )

        # 태스크별 지시사항
        task_instruction = TASK_PROMPTS.get(task, TASK_PROMPTS["general"])

        # 출처 포맷팅
        sources_text = self._format_sources(retrieved_docs)

        # 사용자 메시지 구성
        if sources_text != "없음":
            sources_block = f"\n\n---\n\n참고 매장 후보:\n{sources_text}"
        else:
            sources_block = ""
        user_message = f"{context}\n\n---\n\n{task_instruction}{sources_block}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def classify_task(self, query: str) -> str:
        """
        쿼리에서 태스크 유형 자동 분류

        Returns:
            task type: recommend, strategy, trend, general, photo_guide, info_query
        """
        query_lower = query.lower()

        # 사진/촬영 가이드
        if any(kw in query_lower for kw in ["사진", "촬영", "찍어", "사진을", "올릴", "업로드", "포토"]):
            return "photo_guide"

        # 트렌드
        if any(kw in query_lower for kw in ["요즘", "최근", "트렌드", "유행", "인기"]):
            return "trend"

        # 일반 정보 질문 (매장 추천이 아닌 개념/정의/비교 질문)
        # "뭐야", "뭐지", "무엇", "차이" 등이 포함되고 업종/지역 키워드가 없으면 info_query
        info_keywords = ["뭐야", "뭐지", "무엇", "차이", "비교", "장단점", "설명", "정의", "개념", "종류"]
        place_keywords = ["카페", "음식점", "식당", "맛집", "베이커리", "빵집", "바", "술집", "가게", "매장", "가볼만한", "방문"]

        if any(kw in query_lower for kw in info_keywords):
            # 장소/매장 관련 키워드가 없으면 정보 질문
            if not any(pk in query_lower for pk in place_keywords):
                return "info_query"

        # 전략/예산/캠페인/리텐션
        if any(kw in query_lower for kw in [
            "전략", "방법", "어떻게", "마케팅", "홍보",
            "예산", "캠페인", "배분", "분배", "리텐션",
            "재방문", "lifetime", "ltv", "구독", "재구매",
        ]):
            return "strategy"

        # 추천 (장소/매장 관련)
        if any(kw in query_lower for kw in ["추천", "어디", "어떤", "찾아"]):
            # "알려줘"는 info_query로 갈 수 있으므로 제외
            return "recommend"

        return "general"


# ============================================
# 5. INTENT ROUTER (의도 분류) - 5개 인텐트
# ============================================

class IntentRouter:
    """
    사용자 의도 분류 및 라우팅 (5개 인텐트)

    인텐트:
    - CHITCHAT: 인사/잡담/감정/가벼운 대화
    - MARKETING_COUNSEL: 광고 고민 상담/전략/카피 아이디어
    - TREND_WEB: 최신 트렌드/밈/요즘/순위 (웹 검색 필요)
    - DOC_RAG: 문서/사례/데이터 기반 질의
    - TASK_ACTION: 생성 작업 (배너, 문구, 이미지 프롬프트)
    """

    INTENT_KEYWORDS = {
        # 일상 대화 / 잡담
        "chitchat": [
            "안녕", "하이", "헬로", "반가워", "고마워", "감사", "수고",
            "뭐해", "뭐야", "왜", "그래", "응", "ㅇㅇ", "ㅋㅋ", "ㅎㅎ",
            "힘들", "피곤", "좋아", "싫어", "재밌", "심심",
            "오늘", "내일", "어제",  # 단독 사용 시
            "없네", "있네", "그렇구나", "알겠어", "몰라",
            "누구", "어디서", "언제",  # 단독 질문
        ],
        # 일반 정보/통계 질문 (웹 검색 필요) - 순위, 비율, 통계 등
        "stats_query": [
            "가장 많이", "가장 적게", "1위", "2위", "3위", "순위",
            "몇 %", "몇 퍼센트", "비율", "통계", "데이터",
            "평균", "전체", "시장", "점유율",
            "대한민국", "한국", "국내", "글로벌", "세계",
            "어디야", "뭐야",  # "광고 플랫폼이 뭐야" 같은 질문
        ],
        # 트렌드/실시간 (웹 검색 필요)
        "trend_web": [
            "요즘", "최근", "트렌드", "유행", "인기", "핫한", "뜨는",
            "밈", "meme", "랭킹", "2024", "2025", "2026",
            "신상", "신메뉴", "핫플", "바이럴",
        ],
        # 생성 작업
        "task_action": [
            "만들어", "생성", "작성해", "써줘", "뽑아",
            "배너", "문구", "카피", "슬로건", "해시태그",
            "이미지", "프롬프트", "템플릿",
            "10개", "5개", "3개",  # 개수 요청
        ],
        # 마케팅 상담 (우리 매장 관련)
        "marketing_counsel": [
            "광고", "마케팅", "홍보",
            "예산", "비용", "집행", "배분", "분배",
            "전략", "방법", "어떻게", "뭐가 좋",
            "리텐션", "재방문", "고객", "매출", "방문자",
            "우리", "내", "저희",  # 우리 매장 관련
        ],
        # 문서/사례 RAG
        "doc_rag": [
            "사례", "케이스", "성공", "실패", "경험",
            "매장", "가게", "다른 카페", "다른 맛집",
            "사진", "촬영", "업로드", "올려", "찍어",
        ],
    }

    # 짧은 일상 대화 패턴 (정규식 또는 exact match)
    CHITCHAT_PATTERNS = [
        "ㅋ", "ㅎ", "ㅇㅇ", "ㄴㄴ", "ㄱㄱ", "ㅇㅋ",
        "네", "응", "아", "오", "음", "흠",
        "뭐", "왜", "어",
    ]

    # 대화 시작 패턴 (구체적 질문 없이 도움 요청)
    # 주의: "어떻게 해야"는 구체적 질문에도 자주 쓰이므로 제외
    CONVERSATION_START_PATTERNS = [
        "고민이야", "고민있어", "고민 있어", "고민이 있어",
        "고민이 많아", "고민 많아", "고민이많아", "고민많아",
        "도와줘", "도움이 필요해", "상담하고 싶어", "상담받고 싶어",
        "모르겠어", "힘들어", "막막해", "잘 모르겠",
        # "어떻게 해야", "뭘 해야" → 구체적 질문에도 사용되므로 제외
    ]

    # 구체적 질문 신호 (이 키워드가 있으면 conversation_start가 아님)
    SPECIFIC_QUESTION_KEYWORDS = [
        "얼마", "어떤", "뭐가 좋", "추천해", "알려줘", "예산",
        "어떻게", "방법", "효과", "광고", "마케팅", "홍보",
        "플랫폼", "채널", "SNS", "인스타", "네이버", "틱톡",
        "배달", "가게", "카페", "음식점", "매장",
    ]

    def classify(self, query: str) -> str:
        """
        의도 분류

        Returns:
            - "chitchat": 일상 대화 → LLM 직접 응답
            - "conversation_start": 대화 시작 (구체적 질문 없음) → clarification
            - "trend_web": 트렌드/실시간 → Agent (웹 검색)
            - "task_action": 생성 작업 → 생성 모듈/LLM
            - "marketing_counsel": 마케팅 상담 → RAG
            - "doc_rag": 문서/사례 질의 → RAG
        """
        query_lower = query.lower().strip()
        query_len = len(query_lower)

        # 1) 아주 짧은 입력 (5자 이하) → chitchat 우선
        if query_len <= 5:
            # 짧은 패턴 체크
            if any(p in query_lower for p in self.CHITCHAT_PATTERNS):
                return "chitchat"
            # 인사/감정 키워드
            if any(kw in query_lower for kw in ["안녕", "하이", "고마", "뭐", "왜", "응", "네"]):
                return "chitchat"

        # 2) 대화 시작 패턴 체크 (인사 + 고민/도움 요청, 구체적 질문 없음)
        has_greeting = any(kw in query_lower for kw in ["안녕", "하이", "헬로", "반가워"])
        has_help_request = any(p in query_lower for p in self.CONVERSATION_START_PATTERNS)

        # 구체적 질문 여부 체크 (질문 키워드 + 길이)
        has_specific_question = (
            query_len > 30 or
            any(kw in query_lower for kw in self.SPECIFIC_QUESTION_KEYWORDS)
        )

        # 도움 요청이지만 구체적 질문/주제가 없을 때만 conversation_start
        if has_help_request and not has_specific_question:
            return "conversation_start"

        # 3) chitchat 키워드 (마케팅 키워드 없을 때만)
        has_marketing = any(kw in query_lower for kw in self.INTENT_KEYWORDS["marketing_counsel"])
        has_chitchat = any(kw in query_lower for kw in self.INTENT_KEYWORDS["chitchat"])

        if has_chitchat and not has_marketing and query_len < 20:
            return "chitchat"

        # 3) stats_query (일반 정보/통계 질문 → 웹 검색)
        has_stats = any(kw in query_lower for kw in self.INTENT_KEYWORDS["stats_query"])
        # "우리/내/저희" 없이 일반적인 정보 질문이면 stats_query
        has_personal = any(kw in query_lower for kw in ["우리", "내", "저희", "내 매장", "우리 가게"])
        if has_stats and not has_personal:
            return "stats_query"

        # 4) trend_web (웹 검색 필요)
        if any(kw in query_lower for kw in self.INTENT_KEYWORDS["trend_web"]):
            return "trend_web"

        # 5) task_action (생성 작업)
        if any(kw in query_lower for kw in self.INTENT_KEYWORDS["task_action"]):
            return "task_action"

        # 6) marketing_counsel (우리 매장 관련 상담)
        if has_marketing or has_personal:
            return "marketing_counsel"

        # 7) doc_rag (문서/사례)
        if any(kw in query_lower for kw in self.INTENT_KEYWORDS["doc_rag"]):
            return "doc_rag"

        # 8) 기본값: 일반 정보 질문 → Agent
        return "stats_query"

    def get_routing(self, intent: str) -> str:
        """
        인텐트 → 라우팅 대상 반환

        Returns:
            - "llm": LLM 직접 응답
            - "clarify": clarification 요청 (구체적 질문 유도)
            - "agent": Agent (웹 검색 + RAG)
            - "rag": RAG only
            - "generator": 생성 모듈
        """
        routing_map = {
            "chitchat": "llm",
            "conversation_start": "clarify",  # 구체적 질문 유도
            "stats_query": "agent",  # 일반 정보/통계 → 웹 검색
            "trend_web": "agent",
            "task_action": "generator",
            "marketing_counsel": "rag",
            "doc_rag": "rag",
        }
        return routing_map.get(intent, "agent")


# ============================================
# 6. SLOT CHECKER (필수 정보 확인)
# ============================================

class SlotChecker:
    """
    인텐트별 필수 슬롯 확인 및 질문 생성

    Slot-Filling 로직:
    1. 인텐트에 필요한 필수 슬롯 정의
    2. UserContext에서 채워진 슬롯 확인
    3. 누락된 슬롯이 있으면 질문 생성
    """

    # 인텐트별 필수 슬롯 정의
    REQUIRED_SLOTS = {
        # 지역까지 확보한 뒤에만 답변하도록 필수 슬롯을 확대
        "marketing_counsel": ["industry", "location"],  # 업종+지역 필수
        "doc_rag": ["industry", "location"],            # 업종+지역 필수
        "task_action": ["industry"],                    # 업종 필수
        "trend_web": [],                                # 필수 없음
        "stats_query": [],                              # 필수 없음
        "chitchat": [],                                 # 필수 없음
        "conversation_start": [],                       # 필수 없음
    }

    # 슬롯별 질문 템플릿
    SLOT_QUESTIONS = {
        "industry": "어떤 업종을 운영하고 계세요? (예: 카페, 음식점, 베이커리, 술집 등)",
        "location": "어느 동네에서 운영하고 계세요? (말씀해주시면 그 동네 사례만 참고해서 답변드릴게요)",
        "budget": "마케팅 예산은 어느 정도로 생각하고 계세요? (예: 30만원, 50만원)",
        "goal": "주요 목표가 무엇인가요? (예: 신규 고객 유치, 재방문율 증가, 매출 증대)",
    }

    # 쿼리에서 슬롯 추출을 위한 키워드 매핑
    INDUSTRY_KEYWORDS = {
        "카페": "cafe", "커피": "cafe", "커피숍": "cafe",
        "음식점": "restaurant", "식당": "restaurant", "맛집": "restaurant",
        "베이커리": "bakery", "빵집": "bakery",
        "디저트": "dessert",
        # 단글자 키워드(바)는 오탐이 많아 주점/술집만 남김
        "술집": "bar", "주점": "bar",
    }

    LOCATION_KEYWORDS = [
        "강남", "홍대", "신사", "압구정", "청담", "이태원", "성수", "연남",
        "종로", "명동", "을지로", "광화문", "여의도", "잠실", "건대",
        "신촌", "이대", "합정", "망원", "연희", "서촌", "북촌",
        "판교", "분당", "수원", "인천", "부산", "대구", "제주",
        "순천",  # 자주 등장하는 지역 추가
    ]

    @staticmethod
    def _match_keyword(text: str, keyword: str) -> bool:
        """
        한글 키워드가 다른 단어 내부에 섞여있는 오탐을 줄이기 위한 매칭
        - 조사(에서/에/은/는/이/가/을/를 등)까지 허용하여 '순천에서', '카페를'도 인식
        - 키워드 앞뒤가 다른 한글인 경우는 매칭하지 않음
        """
        import re
        pattern = rf"(?<![가-힣]){re.escape(keyword)}(?:에서|에|은|는|이|가|을|를|으로|로|이요|이에요|예요|야|이야)?(?![가-힣])"
        return re.search(pattern, text) is not None

    @staticmethod
    def _is_general_question_pattern(text: str, keyword: str) -> bool:
        """
        업종 키워드가 일반적인 질문 패턴(다른 업종에 대한 질문)인지 확인
        예: "음식점 광고는 사진이 더 중요한가요?" → True (일반 질문)
        예: "카페야", "카페에서 일해" → False (자기 업종 선언)
        """
        import re
        # 일반 질문 패턴: 키워드 + "은/는/의" + 광고/마케팅/...
        general_patterns = [
            rf"{re.escape(keyword)}(?:\s*)(?:광고|마케팅|홍보|사진|메뉴|가격)",  # 음식점 광고, 카페 마케팅
            rf"{re.escape(keyword)}(?:은|는|의)\s",  # 음식점은, 카페의
        ]
        for pattern in general_patterns:
            if re.search(pattern, text):
                return True
        return False

    def extract_slots_from_query(self, query: str) -> dict:
        """
        쿼리에서 슬롯 정보 추출

        Returns:
            {"industry": "cafe", "location": "강남", ...} 또는 빈 dict
        """
        extracted = {}
        query_lower = query.lower()
        query_text = query

        # 업종 추출 (단, 일반 질문 패턴이면 추출하지 않음)
        for keyword, industry_code in self.INDUSTRY_KEYWORDS.items():
            if self._match_keyword(query_text, keyword):
                # "음식점 광고는~", "카페의 마케팅~" 같은 일반 질문은 건너뜀
                if self._is_general_question_pattern(query_text, keyword):
                    continue
                extracted["industry"] = industry_code
                break

        # 지역 추출
        for location in self.LOCATION_KEYWORDS:
            if self._match_keyword(query_text, location):
                extracted["location"] = location
                break

        # 예산 추출
        import re
        budget_patterns = [
            r"(\d+)\s*만\s*원",
            r"예산\s*(\d+)\s*만",
            r"(\d+)만\s*원",
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, query)
            if match:
                num = int(match.group(1))
                extracted["budget"] = num * 10000
                break

        return extracted

    def coarse_location_guess(self, query: str) -> Optional[str]:
        """
        키워드 매칭에 실패했을 때 사용하는 느슨한 지역 추정
        - 행정동/시/군/구/도 접미사를 우선
        - 없으면 마지막 단어를 후처리해 사용
        """
        import re

        cleaned = re.sub(r"[^가-힣0-9 ]", " ", query)
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if not tokens:
            return None

        suffixes = ("시", "군", "구", "읍", "면", "동", "리", "도")
        for tok in reversed(tokens):
            if tok.endswith(suffixes):
                return re.sub(r"(이요|여요|이에요|예요|요|이야)$", "", tok)

        # 접미사가 없으면 마지막 토큰을 후처리
        last = tokens[-1]
        last = re.sub(r"(이요|여요|이에요|예요|요|이야)$", "", last)
        return last or None

    def check_required_slots(
        self,
        intent: str,
        user_context: Optional[UserContext],
        query: str = ""
    ) -> list:
        """
        필수 슬롯 중 누락된 것 확인

        Args:
            intent: 분류된 인텐트
            user_context: 현재 사용자 컨텍스트
            query: 현재 쿼리 (슬롯 추출용)

        Returns:
            누락된 슬롯 리스트 (예: ["industry", "location"])
        """
        required = self.REQUIRED_SLOTS.get(intent, [])

        if not required:
            return []

        # 쿼리에서 슬롯 추출
        query_slots = self.extract_slots_from_query(query)

        # 누락 체크
        missing = []
        for slot in required:
            # UserContext에 있는지 확인
            has_in_context = (
                user_context and
                getattr(user_context, slot, None) is not None
            )
            # 쿼리에서 추출됐는지 확인
            has_in_query = slot in query_slots

            if not has_in_context and not has_in_query:
                missing.append(slot)

        return missing

    def get_slot_question(self, slot: str, context_hint: str = "") -> str:
        """
        슬롯에 대한 질문 생성

        Args:
            slot: 슬롯 이름
            context_hint: 맥락 힌트 (예: 이전 질문 요약)

        Returns:
            사용자에게 보여줄 질문
        """
        base_question = self.SLOT_QUESTIONS.get(slot, f"{slot} 정보를 알려주세요.")

        if context_hint:
            return f"{context_hint}\n\n우선, {base_question}"

        return base_question

    def update_context_from_query(
        self,
        query: str,
        user_context: Optional[UserContext]
    ) -> UserContext:
        """
        쿼리에서 추출한 슬롯으로 UserContext 업데이트

        Args:
            query: 사용자 쿼리
            user_context: 기존 UserContext (없으면 생성)

        Returns:
            업데이트된 UserContext
        """
        if user_context is None:
            user_context = UserContext()

        extracted = self.extract_slots_from_query(query)

        if "industry" in extracted and not user_context.industry:
            user_context.industry = extracted["industry"]
        if "location" in extracted and not user_context.location:
            user_context.location = extracted["location"]
        if "budget" in extracted and not user_context.budget:
            user_context.budget = extracted["budget"]

        return user_context


# ============================================
# 7. EXAMPLE USAGE
# ============================================

def example_usage():
    """사용 예시"""
    builder = PromptBuilder()

    # 샘플 검색 결과
    retrieved_docs = [
        {
            "content": "강남역 근처 조용한 분위기의 카페입니다. 넓은 좌석과 콘센트가 많아 작업하기 좋습니다.",
            "metadata": {
                "title": "카페 로얄마카롱",
                "location": "강남",
                "industry": "cafe",
                "rating": 4.5,
            },
        },
        {
            "content": "수제 마카롱이 유명한 디저트 카페입니다. 계절 한정 메뉴가 인기 있습니다.",
            "metadata": {
                "title": "달콤한 하루",
                "location": "신사동",
                "industry": "dessert",
                "rating": 4.8,
            },
        },
    ]

    # 사용자 컨텍스트
    user_ctx = UserContext(
        industry="cafe",
        location="강남",
        budget=100000,
        goal="신규 고객 유치",
    )

    # 프롬프트 생성
    query = "강남역 근처 분위기 좋은 카페 추천해줘"
    messages = builder.build_prompt(
        query=query,
        retrieved_docs=retrieved_docs,
        task="recommend",
        user_context=user_ctx,
    )

    print("=== System Prompt ===")
    print(messages[0]["content"][:500])
    print("\n=== User Message ===")
    print(messages[1]["content"])


if __name__ == "__main__":
    example_usage()
