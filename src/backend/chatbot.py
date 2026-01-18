"""
RAG 기반 챗봇 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 기반 챗봇의 핵심 로직을 구현합니다.
- ConversationManager: 대화 히스토리 관리 (PostgreSQL)
- LLMOrchestrator: LLM 호출 관리 (의도 분석, 상담 응답 생성)
- RAGChatbot: RAG 챗봇 메인 클래스 (Intent 분석 후 분기 처리)
"""

from typing import Optional, List, Dict, AsyncIterator
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


def _format_chat_history(chat_history: List[Dict]) -> str:
    if not chat_history:
        return "(없음)"
    lines = []
    for idx, msg in enumerate(chat_history, start=1):
        role = msg.get("role") or "unknown"
        content = (msg.get("content") or "").strip()
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines)


def _format_generation_history(generation_history: List[Dict]) -> str:
    if not generation_history:
        return "(없음)"
    lines = []
    for gen in generation_history:
        gen_id = gen.get("id")
        content_type = gen.get("content_type") or "unknown"
        input_text = (gen.get("input_text") or "").strip()
        input_image = gen.get("input_image") or {}
        output_image = gen.get("output_image") or {}
        input_hash = input_image.get("file_hash")
        output_hash = output_image.get("file_hash")
        image_note = ""
        if input_hash or output_hash:
            image_note = f", input_image={input_hash or '-'}, output_image={output_hash or '-'}"
        lines.append(
            f"- ID: {gen_id}, 타입: {content_type}, 입력: \"{input_text}\"{image_note}"
        )
    return "\n".join(lines)


def _image_payload(image) -> Optional[Dict]:
    if not image:
        return None
    return {
        "file_hash": image.file_hash,
        "file_directory": image.file_directory,
    }


def _last_user_message(chat_history: List[Dict]) -> str:
    for msg in reversed(chat_history or []):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return ""


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _select_generation_input(
    generation_history: List[Dict],
    target_generation_id: Optional[int],
) -> str:
    if target_generation_id:
        for gen in generation_history:
            if gen.get("id") == target_generation_id:
                return (gen.get("input_text") or "").strip()
    for gen in reversed(generation_history or []):
        candidate = (gen.get("input_text") or "").strip()
        if candidate:
            return candidate
    return ""


def _is_copy_of_previous_output(
    refined_input: str,
    chat_history: List[Dict],
    generation_history: List[Dict],
) -> bool:
    normalized = _normalize_text(refined_input)
    if len(normalized) < 10:
        return False
    for msg in chat_history or []:
        if msg.get("role") == "assistant":
            if _normalize_text(msg.get("content") or "") == normalized:
                return True
    for gen in generation_history or []:
        if _normalize_text(gen.get("output_text") or "") == normalized:
            return True
    return False


# =====================================================
# ConversationManager: 대화 히스토리 관리
# =====================================================

class ConversationManager:
    """
    대화 히스토리 관리 클래스
    PostgreSQL 기반 대화 및 생성 이력 관리
    """

    def add_message(
        self,
        db: Session,
        session_id: str,
        role: str,
        content: str,
        image_id: Optional[int] = None,
    ) -> int:
        """
        메시지 저장 (PostgreSQL)

        Args:
            db: SQLAlchemy session
            session_id: 세션 ID
            role: 메시지 역할 (user/assistant)
            content: 메시지 내용
            image_id: 이미지 ID (선택)
        Returns:
            저장된 메시지 ID
        """
        from src.backend import process_db

        # DB 저장
        chat_row = process_db.save_chat_message(
            db,
            session_id,
            role,
            content,
            image_id=image_id,
        )

        logger.info(
            f"ConversationManager: saved message with id={chat_row.id}"
        )
        return chat_row.id

    def get_recent_messages(
        self,
        db: Session,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """최근 대화 조회"""
        from src.backend import process_db

        messages = process_db.get_chat_history_by_session(db, session_id, limit)
        messages_reversed = list(reversed(messages))

        result = [
            {
                "role": msg.role,
                "content": msg.content,
                "image_id": msg.image_id,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in messages_reversed
        ]

        logger.info(f"ConversationManager: retrieved {len(result)} recent messages")
        return result

    def get_full_messages(
        self,
        db: Session,
        session_id: str,
    ) -> List[Dict]:
        """전체 대화 조회"""
        from src.backend import process_db

        messages = process_db.get_chat_history_by_session(db, session_id, limit=None)
        messages_reversed = list(reversed(messages))

        result = [
            {
                "role": msg.role,
                "content": msg.content,
                "image_id": msg.image_id,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in messages_reversed
        ]

        logger.info(f"ConversationManager: retrieved {len(result)} full messages")
        return result

    def get_generation_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 5,
    ) -> List[Dict]:
        """생성 이력 조회"""
        from src.backend import process_db

        generations = process_db.get_generation_history_by_session(db, session_id, limit)

        result = [
            {
                "id": gen.id,
                "content_type": gen.content_type,
                "input_text": gen.input_text,
                "output_text": gen.output_text,
                "prompt": gen.prompt,
                "style": gen.style,
                "industry": gen.industry,
                "input_image": _image_payload(gen.input_image),
                "output_image": _image_payload(gen.output_image),
                "timestamp": gen.created_at.isoformat(),
            }
            for gen in generations
        ]

        logger.info(f"ConversationManager: retrieved {len(result)} generation history entries")
        return result

    def get_full_generation_history(
        self,
        db: Session,
        session_id: str,
    ) -> List[Dict]:
        """전체 생성 이력 조회"""
        from src.backend import process_db

        generations = process_db.get_generation_history_by_session(db, session_id, limit=None)
        generations_reversed = list(reversed(generations))

        result = [
            {
                "id": gen.id,
                "content_type": gen.content_type,
                "input_text": gen.input_text,
                "output_text": gen.output_text,
                "prompt": gen.prompt,
                "style": gen.style,
                "industry": gen.industry,
                "input_image": _image_payload(gen.input_image),
                "output_image": _image_payload(gen.output_image),
                "timestamp": gen.created_at.isoformat(),
            }
            for gen in generations_reversed
        ]

        logger.info(f"ConversationManager: retrieved {len(result)} full generation history entries")
        return result

# =====================================================
# LLMOrchestrator: LLM 호출 관리
# =====================================================

class LLMOrchestrator:
    """
    LLM 호출 관리 클래스
    - analyze_intent(): 의도 분석 (모든 경우에 사용)
    - refine_generation_input(): 생성 입력 정제 (생성/수정 파이프라인용)
    - generate_consulting_response(): 상담 응답 생성 (상담 intent만)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    async def analyze_intent(
        self,
        user_message: str,
        context: Dict
    ) -> Dict:
        """의도 분석 (생성/수정/상담 구분 + 생성 타입)"""
        import openai
        import json

        system_prompt = """
당신은 광고 제작 어시스턴트입니다. 사용자 메시지를 분석하여 의도를 파악하세요.

의도 분류:
- generation: 새로운 광고를 만들고 싶어하는 경우 (예: "광고 만들어줘", "이미지 생성해줘", "카페 광고 만들기")
- modification: 기존 광고를 수정하고 싶어하는 경우 (예: "더 밝게", "텍스트 바꿔줘", "색상 변경")
- consulting: 광고 제작 방법이나 조언을 구하는 경우 (예: "어떻게 만들어?", "팁 알려줘", "뭐가 좋을까?")

의도 판단 가이드:
- "어때?", "괜찮을까?", "해볼까?", "어떻게 생각해?"처럼 의견/아이디어를 묻는 문장은 consulting
- 명시적으로 "만들어줘/생성해줘/그려줘/작성해줘"가 없으면 consulting 쪽을 우선

생성 타입 (intent가 generation일 때만 해당):
- image: 이미지/사진/그림/배너/포스터/피드 등 시각적 콘텐츠를 원하는 경우 (예: "이미지 생성", "사진 만들어줘", "인스타 피드용", "배너 만들어줘")
- text: 광고 문구/카피/슬로건만 원하는 경우 (예: "문구 만들어줘", "카피 작성", "슬로건")

수정 대상 선택 (intent가 modification일 때만 해당):
- 제공된 생성 이력은 recent_rank=1이 가장 최신입니다
- 사용자 메시지와 내용이 가장 맞는 항목을 선택해 target_generation_id로 반환하세요 (id 값을 반환)
- "방금/최근/마지막"처럼 최신을 지칭하면 recent_rank=1을 선택하세요
- 번호 지칭(예: 2번, 두 번째)이 있으면 해당 recent_rank를 선택하세요
- "처음/첫 요청/맨 처음"이면 가장 오래된 recent_rank를 선택하세요
- 명확하지 않으면 null

generation_type 규칙:
- intent=generation이면 image/text 중 하나를 선택하세요
- intent=modification/consulting이면 null (명확히 알 수 있으면 text/image 가능)

JSON 형식으로만 응답하세요:
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "generation_type": "image|text|null",
  "target_generation_id": 123 or null
}
"""

        recent_conversations = context.get("recent_conversations", [])
        generation_history = context.get("generation_history", [])

        context_str = ""
        if recent_conversations:
            conv_summary = "\n".join([
                f"{msg['role']}: {(msg.get('content') or '')[:200]}"
                for msg in recent_conversations[-5:]
            ])
            context_str += f"\n\n최근 대화:\n{conv_summary}"

        generation_summary = []
        for idx, gen in enumerate(generation_history[:5], start=1):
            generation_summary.append(
                {
                    "recent_rank": idx,
                    "id": gen.get("id"),
                    "content_type": gen.get("content_type"),
                    "input_text": (gen.get("input_text") or "")[:120],
                    "timestamp": gen.get("timestamp"),
                }
            )
        if generation_summary:
            context_str += "\n\n이전 생성 이력 (recent_rank=1이 최신):\n"
            context_str += json.dumps(generation_summary, ensure_ascii=False, indent=2)

        user_prompt = f"사용자 메시지: {user_message}{context_str}"

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            logger.info(f"LLMOrchestrator: intent analysis raw response: {content}")

            result = json.loads(content)
            intent = result.get("intent")
            if intent not in ("generation", "modification", "consulting"):
                intent = "consulting"

            confidence = result.get("confidence")
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.5

            gen_type = result.get("generation_type")
            if gen_type not in ("image", "text"):
                gen_type = None

            target_id = result.get("target_generation_id")
            if isinstance(target_id, str) and target_id.isdigit():
                target_id = int(target_id)
            if not isinstance(target_id, int):
                target_id = None
            logger.info(
                "LLMOrchestrator: analyzed intent=%s, generation_type=%s, target_generation_id=%s",
                intent,
                gen_type,
                target_id,
            )
            return {
                "intent": intent,
                "confidence": confidence,
                "generation_type": gen_type,
                "target_generation_id": target_id,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"LLMOrchestrator: JSON decode error: {e}")
            return {
                "intent": "consulting",
                "confidence": 0.5,
                "generation_type": None,
                "target_generation_id": None,
            }
        except Exception as e:
            logger.error(f"LLMOrchestrator: intent analysis failed: {e}", exc_info=True)
            return {
                "intent": "consulting",
                "confidence": 0.5,
                "generation_type": None,
                "target_generation_id": None,
            }

    async def refine_generation_input(
        self,
        *,
        intent: str,
        generation_type: Optional[str],
        target_generation_id: Optional[int],
        chat_history: List[Dict],
        generation_history: List[Dict],
    ) -> str:
        """전체 맥락을 반영해 생성 입력을 정제"""
        import openai
        import json

        system_prompt = """
당신은 광고 생성 요청을 정제하는 맥락 분석기입니다.
반드시 refined_input 텍스트만 출력하세요. JSON 형식이나 코드블록은 절대 쓰지 마세요.
"""

        prompt = f"""
# 전체 대화 히스토리
{_format_chat_history(chat_history)}

# 생성 이력
{_format_generation_history(generation_history)}

# 사용자 요청
intent: {intent}
generation_type: {generation_type or "null"}
target_generation_id: {target_generation_id if target_generation_id is not None else "null"}

# 작업: refined_input 작성

## refined_input 작성 규칙

### 1. 번호 참조 해석
- "2번으로 해줘" → 대화에서 2번 찾아서 실제 내용으로 치환
- Assistant의 consulting 응답(추천 문구 등)도 확인할 것
- 예: Assistant가 "1) A 2) 우리집 필수템 3) C"라고 했으면 "2번" → "우리집 필수템"

### 2. 대명사 해석
- "그거", "저거", "그 문구" 등은 구체적 대상으로 치환

### 3. modification 요청 처리
- 절대 이전 생성물을 복사하지 말 것
- 대신 다음을 요약:
  1) 처음 생성 요청 내용
  2) 제품/서비스 정보
  3) 누적된 모든 수정 요구사항

### 4. generation 요청 처리
- 대화에서 언급된 모든 관련 정보 요약
- 이전 consulting 응답에서 선택한 내용 포함
- 구체적이고 실행 가능한 명령으로 작성

### 5. 추가 금지 사항
- 이미지 비율, 플랫폼, 스타일 관련 분석은 아직 하지 말 것

## 예시

### 예시 1: consulting 응답 참조
대화:
- User: "문구 추천해줘"
- Assistant (consulting): "1) 깨끗한 마무리 2) 우리집 필수템 3) 프리미엄 행주"
- User: "2번으로 이미지 만들어줘"

refined_input: "대나무 행주 광고 이미지. '우리집 필수템' 텍스트 포함"

### 예시 2: 연속 수정 (이전 생성물 복사 금지)
대화:
- User: "네이버 스토어 소개글 써줘. 행주 파는 곳, 기름 안 묻음"
- Assistant: "대나무 행주로 깨끗한 주방"
- User: "행주라는 단어 넣어줘"
- Assistant: "대나무 행주로 깨끗한 주방"
- User: "주방이라는 단어는 빼줘"

refined_input: "네이버 스토어 소개글. 대나무 행주 판매(기름 안 묻음). '행주' 단어 포함 필수, '주방' 단어 제외"

### 예시 3: 관련 컨텐츠 생성
대화:
- User: "네이버 소개글 써줘"
- Assistant: "자연의 손길로 깨끗함을"
- User: "관련 이미지도 생성해줄 수 있어?"

refined_input: "네이버 스토어용 대나무 행주 광고 이미지. 자연 친화적, 깨끗한 느낌"

# 반환
refined_input만 텍스트로 반환 (JSON 형식 아님, 따옴표 없이)
"""

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content or ""
            refined = content.strip()
            if refined.startswith("{"):
                try:
                    parsed = json.loads(refined)
                    refined = (parsed.get("refined_input") or "").strip()
                except json.JSONDecodeError:
                    pass
            if len(refined) >= 2 and refined[0] == refined[-1] and refined[0] in ("\"", "'"):
                refined = refined[1:-1].strip()
            if refined.lower() == "null":
                refined = ""

            if not refined:
                refined = _last_user_message(chat_history)

            if intent == "modification" and _is_copy_of_previous_output(
                refined,
                chat_history,
                generation_history,
            ):
                base_input = _select_generation_input(
                    generation_history,
                    target_generation_id,
                )
                fallback_user = _last_user_message(chat_history)
                fallback_parts = [part for part in [base_input, fallback_user] if part]
                if fallback_parts:
                    refined = " / ".join(fallback_parts)
                    logger.info("LLMOrchestrator: adjusted refined_input to avoid output copy")

            logger.info(
                "LLMOrchestrator: refined_input=%s",
                refined[:200],
            )
            return refined

        except Exception as e:
            logger.error(f"LLMOrchestrator: refinement failed: {e}", exc_info=True)
            return _last_user_message(chat_history)

    async def generate_consulting_response(
        self,
        user_message: str,
        context: Dict
    ) -> str:
        """상담 응답 생성 (상담 intent에만 사용)"""
        import openai

        system_prompt = self._build_consulting_system_prompt()
        user_prompt = self._build_consulting_user_prompt(user_message, context)

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )

            assistant_message = response.choices[0].message.content
            logger.info(f"LLMOrchestrator: consulting response generated")
            return assistant_message

        except Exception as e:
            logger.error(f"LLMOrchestrator: consulting response generation failed: {e}", exc_info=True)
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요."

    async def stream_consulting_response(
        self,
        user_message: str,
        context: Dict
    ) -> AsyncIterator[str]:
        """상담 응답 스트리밍 (상담 intent에만 사용)"""
        import openai

        system_prompt = self._build_consulting_system_prompt()
        user_prompt = self._build_consulting_user_prompt(user_message, context)

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        except Exception as e:
            logger.error(f"LLMOrchestrator: consulting stream failed: {e}", exc_info=True)
            yield "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요."

    def _build_consulting_system_prompt(self) -> str:
        return """
당신은 광고 제작 전문 상담사입니다. 사용자의 질문에 대해 친절하고 전문적으로 답변하세요.

답변 시 고려사항:
1. 사용자의 과거 대화 맥락을 참고하세요.
2. 이전 생성 이력이 있다면 개인화된 조언을 제공하세요.
3. 지식베이스의 정보를 활용하되, 자연스럽게 통합하세요.
4. 구체적이고 실행 가능한 조언을 제공하세요.
"""

    def _build_consulting_user_prompt(self, user_message: str, context: Dict) -> str:
        import json

        recent_conversations = context.get("recent_conversations", [])
        generation_history = context.get("generation_history", [])
        knowledge_base = context.get("knowledge_base", [])

        conv_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_conversations[-5:]
        ])

        kb_str = "\n".join([
            f"- {doc.get('content', '')[:200]}..."
            for doc in knowledge_base[:3]
        ])

        return f"""
사용자 질문: "{user_message}"

최근 대화 맥락:
{conv_str}

이전 생성 이력:
{json.dumps(generation_history[:2], ensure_ascii=False, indent=2)}

관련 지식베이스:
{kb_str}
"""


# =====================================================
# RAGChatbot: RAG 챗봇 메인 클래스
# =====================================================

class RAGChatbot:
    """
    RAG 기반 챗봇 메인 클래스
    Intent 분석 후 분기 처리:
    - 생성/수정: 생성 파이프라인 호출 (정제 단계에서 LLM 사용)
    - 상담: LLM 응답 생성
    """

    def __init__(
        self,
        llm_orchestrator: LLMOrchestrator,
        conversation_manager: ConversationManager,
        knowledge_base: Optional["ConsultingKnowledgeBase"] = None
    ):
        self.llm = llm_orchestrator
        self.conv = conversation_manager
        self.knowledge = knowledge_base

    async def process_message(
        self,
        db: Session,
        session_id: str,
        user_message: str,
        image_id: Optional[int] = None
    ) -> Dict:
        """
        메시지 처리 (RAG 파이프라인)

        플로우:
        1. Intent 분석 (LLM 1회 호출) - 유사 대화 활용
        2. Intent별 분기:
           - 생성/수정: PostgreSQL 검색 → 생성 파이프라인 호출
           - 상담: PostgreSQL + 정적파일 검색 → LLM 응답 생성
        """
        logger.info(f"RAGChatbot.process_message: session={session_id}")

        # 1. 사용자 메시지 저장
        self.conv.add_message(db, session_id, "user", user_message, image_id)

        # 2. PostgreSQL 검색 (의도 분석은 최근 대화 기준)
        recent_conversations = self.conv.get_recent_messages(
            db,
            session_id,
            limit=5,
        )
        generation_history = self.conv.get_generation_history(db, session_id, limit=5)

        context = {
            "recent_conversations": recent_conversations,
            "generation_history": generation_history,
        }

        logger.info(f"RAGChatbot: loaded {len(recent_conversations)} recent conversations for intent context")

        # 3. 의도 분석
        intent_result = await self.llm.analyze_intent(user_message, context)
        intent = intent_result.get("intent", "consulting")

        logger.info(f"RAGChatbot: intent={intent}, confidence={intent_result.get('confidence')}")

        # 4. Intent별 분기 처리
        if intent in ["generation", "modification"]:
            logger.info("RAGChatbot: redirecting to generation pipeline")
            # generation_type 결정 (LLM 분석 결과 또는 기본값)
            generation_type = intent_result.get("generation_type") or "image"
            target_generation_id = intent_result.get("target_generation_id")
            logger.info(f"RAGChatbot: generation_type={generation_type}")
            return {
                "intent": intent,
                "redirect_to_pipeline": True,
                "ready_to_generate": False,
                "assistant_message": "광고를 생성하겠습니다. 잠시만 기다려주세요.",
                "generation_type": generation_type,
                "target_generation_id": target_generation_id,
            }

        elif intent == "consulting":
            logger.info(
                "RAGChatbot: consulting intent, using recent conversations only"
            )
            # 지식베이스 검색 (있는 경우)
            if self.knowledge:
                try:
                    knowledge_results = self.knowledge.search(
                        query=user_message,
                        category="faq",
                        limit=3
                    )
                    context["knowledge_base"] = knowledge_results
                    logger.info(f"RAGChatbot: knowledge search returned {len(knowledge_results)} results")
                except Exception as e:
                    logger.warning(f"RAGChatbot: knowledge search failed: {e}")
                    context["knowledge_base"] = []

            # LLM 호출 (상담 응답 생성)
            assistant_message = await self.llm.generate_consulting_response(
                user_message, context
            )

            # 어시스턴트 응답 저장
            self.conv.add_message(db, session_id, "assistant", assistant_message)

            logger.info("RAGChatbot: consulting response generated")
            return {
                "intent": "consulting",
                "assistant_message": assistant_message,
                "redirect_to_pipeline": False,
                "ready_to_generate": False,
            }

        else:
            assistant_message = "무엇을 도와드릴까요?"
            self.conv.add_message(db, session_id, "assistant", assistant_message)

            logger.info("RAGChatbot: unknown intent, default response")
            return {
                "intent": "consulting",
                "assistant_message": assistant_message,
                "redirect_to_pipeline": False,
                "ready_to_generate": False,
            }


# =====================================================
# 싱글톤 인스턴스
# =====================================================

_chatbot_instance: Optional[RAGChatbot] = None


def get_chatbot() -> RAGChatbot:
    """챗봇 인스턴스 반환 (싱글톤)"""
    global _chatbot_instance
    if _chatbot_instance is None:
        from src.utils.config import settings
        from src.backend.consulting_knowledge_base import get_knowledge_base

        llm = LLMOrchestrator(api_key=settings.OPENAI_API_KEY)
        conv = ConversationManager()
        knowledge = get_knowledge_base()

        _chatbot_instance = RAGChatbot(llm, conv, knowledge)
        logger.info("RAGChatbot singleton instance created with knowledge_base")
    return _chatbot_instance
