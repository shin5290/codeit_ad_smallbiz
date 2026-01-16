"""
RAG 기반 챗봇 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 기반 챗봇의 핵심 로직을 구현합니다.
- ConversationManager: 대화 히스토리 관리 (PostgreSQL)
- LLMOrchestrator: LLM 호출 관리 (의도 분석, 상담 응답 생성)
- RAGChatbot: RAG 챗봇 메인 클래스 (Intent 분석 후 분기 처리)
"""

from typing import Optional, List, Dict
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


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
        generate_embedding: bool = True
    ) -> int:
        """
        메시지 저장 (PostgreSQL) + 임베딩 생성

        Args:
            db: SQLAlchemy session
            session_id: 세션 ID
            role: 메시지 역할 (user/assistant)
            content: 메시지 내용
            image_id: 이미지 ID (선택)
            generate_embedding: 임베딩 생성 여부 (기본값: True)

        Returns:
            저장된 메시지 ID
        """
        from src.backend import process_db, models
        from src.utils.embedding import generate_embedding as create_embedding
        from src.utils.config import settings

        # 1. 임베딩 생성 (user 메시지만)
        embedding = None
        if generate_embedding and role == "user" and content.strip():
            try:
                embedding = create_embedding(
                    text=content,
                    api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-3-small"
                )
                if embedding:
                    logger.info(f"ConversationManager: embedding generated (dim={len(embedding)})")
                else:
                    logger.warning("ConversationManager: embedding generation returned None")
            except Exception as e:
                logger.error(f"ConversationManager: embedding generation error: {e}", exc_info=True)
                embedding = None

        # 2. DB 저장
        chat_row = models.ChatHistory(
            session_id=session_id,
            role=role,
            content=content,
            image_id=image_id,
            embedding=embedding
        )
        db.add(chat_row)
        db.commit()
        db.refresh(chat_row)

        logger.info(
            f"ConversationManager: saved message with id={chat_row.id}, "
            f"has_embedding={embedding is not None}"
        )
        return chat_row.id

    def get_recent_messages(
        self,
        db: Session,
        session_id: str,
        limit: int = 10
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
                "timestamp": msg.created_at.isoformat()
            }
            for msg in messages_reversed
        ]

        logger.info(f"ConversationManager: retrieved {len(result)} recent messages")
        return result

    def get_generation_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """생성 이력 조회"""
        from src.backend import process_db

        generations = process_db.get_generation_history_by_session(db, session_id, limit)

        result = [
            {
                "content_type": gen.content_type,
                "output_text": gen.output_text,
                "prompt": gen.prompt,
                "style": gen.style,
                "industry": gen.industry,
                "timestamp": gen.created_at.isoformat()
            }
            for gen in generations
        ]

        logger.info(f"ConversationManager: retrieved {len(result)} generation history entries")
        return result

    def search_similar_messages(
        self,
        db: Session,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        유사 대화 검색 (pgvector 기반 벡터 유사도 검색)

        Args:
            db: SQLAlchemy session
            query: 검색 쿼리
            session_id: 세션 ID (None이면 전체 검색)
            limit: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)

        Returns:
            유사한 메시지 리스트 (유사도 순)
        """
        from src.utils.embedding import generate_embedding as create_embedding
        from src.utils.config import settings
        from src.backend import models

        # 1. 쿼리 임베딩 생성
        try:
            query_embedding = create_embedding(
                text=query,
                api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )

            if not query_embedding:
                logger.warning("ConversationManager: query embedding generation failed, fallback to recent messages")
                if session_id:
                    return self.get_recent_messages(db, session_id, limit)
                return []

        except Exception as e:
            logger.error(f"ConversationManager: embedding generation error: {e}", exc_info=True)
            if session_id:
                return self.get_recent_messages(db, session_id, limit)
            return []

        # 2. pgvector 유사도 검색
        try:
            from sqlalchemy import text

            # pgvector cosine distance 검색
            # <-> : 코사인 거리 (0에 가까울수록 유사)
            # 유사도 = 1 - distance
            sql_query = text("""
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    created_at,
                    1 - (embedding <-> :query_embedding) AS similarity
                FROM chat_history
                WHERE
                    role = 'user'
                    AND embedding IS NOT NULL
                    AND (:session_id IS NULL OR session_id = :session_id)
                    AND 1 - (embedding <-> :query_embedding) >= :similarity_threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """)

            # 파라미터 바인딩
            # pgvector는 Python list를 자동으로 vector 타입으로 변환
            result = db.execute(
                sql_query,
                {
                    "query_embedding": str(query_embedding),  # pgvector는 문자열로 전달
                    "session_id": session_id,
                    "similarity_threshold": similarity_threshold,
                    "limit": limit
                }
            ).fetchall()

            # 결과 변환
            messages = [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.created_at.isoformat(),
                    "similarity": float(row.similarity)
                }
                for row in result
            ]

            logger.info(
                f"ConversationManager: vector search completed, "
                f"found {len(messages)} messages (threshold={similarity_threshold})"
            )

            return messages

        except Exception as e:
            logger.error(f"ConversationManager: vector search error: {e}", exc_info=True)
            # 에러 시 최근 메시지로 폴백
            if session_id:
                return self.get_recent_messages(db, session_id, limit)
            return []


# =====================================================
# LLMOrchestrator: LLM 호출 관리
# =====================================================

class LLMOrchestrator:
    """
    LLM 호출 관리 클래스
    - analyze_intent(): 의도 분석 (모든 경우에 사용)
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

생성 타입 (intent가 generation일 때만 해당):
- image: 이미지/사진/그림/배너/포스터/피드 등 시각적 콘텐츠를 원하는 경우 (예: "이미지 생성", "사진 만들어줘", "인스타 피드용", "배너 만들어줘")
- text: 광고 문구/카피/슬로건만 원하는 경우 (예: "문구 만들어줘", "카피 작성", "슬로건")

JSON 형식으로만 응답하세요:
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "generation_type": "image|text"
}
"""

        recent_conversations = context.get("recent_conversations", [])
        generation_history = context.get("generation_history", [])

        context_str = ""
        if recent_conversations:
            conv_summary = "\n".join([
                f"{msg['role']}: {msg['content'][:100]}"
                for msg in recent_conversations[-3:]
            ])
            context_str += f"\n\n최근 대화:\n{conv_summary}"

        if generation_history:
            context_str += f"\n\n이전 생성 이력: {len(generation_history)}개"

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
            logger.info(f"LLMOrchestrator: analyzed intent={result.get('intent')}")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLMOrchestrator: JSON decode error: {e}")
            return {"intent": "consulting", "confidence": 0.5}
        except Exception as e:
            logger.error(f"LLMOrchestrator: intent analysis failed: {e}", exc_info=True)
            return {"intent": "consulting", "confidence": 0.5}

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
    - 생성/수정: 생성 파이프라인 호출 (백엔드에서 LLM 호출 없음)
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

        # 2. PostgreSQL 검색 (유사 대화 검색 우선)
        # 의도 분석 시: 현재 세션의 유사 대화를 검색
        similar_conversations = self.conv.search_similar_messages(
            db,
            query=user_message,
            session_id=session_id,
            limit=5,
            similarity_threshold=0.6
        )
        generation_history = self.conv.get_generation_history(db, session_id, limit=5)

        context = {
            "recent_conversations": similar_conversations,  # 유사 대화 사용
            "generation_history": generation_history,
        }

        logger.info(f"RAGChatbot: loaded {len(similar_conversations)} similar conversations for context")

        # 3. 의도 분석
        intent_result = await self.llm.analyze_intent(user_message, context)
        intent = intent_result.get("intent", "consulting")

        logger.info(f"RAGChatbot: intent={intent}, confidence={intent_result.get('confidence')}")

        # 4. Intent별 분기 처리
        if intent in ["generation", "modification"]:
            logger.info("RAGChatbot: redirecting to generation pipeline")
            # generation_type 결정 (LLM 분석 결과 또는 기본값)
            generation_type = intent_result.get("generation_type", "image")
            logger.info(f"RAGChatbot: generation_type={generation_type}")
            return {
                "intent": intent,
                "redirect_to_pipeline": True,
                "ready_to_generate": False,
                "workflow_state": {},
                "assistant_message": "광고를 생성하겠습니다. 잠시만 기다려주세요.",
                "generation_type": generation_type,
            }

        elif intent == "consulting":
            # 상담 응답 생성 시: 전체 세션에서 유사 대화 검색 (세션 범위 확장)
            consulting_similar_conversations = self.conv.search_similar_messages(
                db,
                query=user_message,
                session_id=session_id,  # 현재 세션 내에서만 검색
                limit=5,
                similarity_threshold=0.6
            )

            # 컨텍스트 업데이트 (상담용 유사 대화 사용)
            context["recent_conversations"] = consulting_similar_conversations
            logger.info(f"RAGChatbot: loaded {len(consulting_similar_conversations)} similar conversations for consulting")

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
                "workflow_state": {},
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
                "workflow_state": {},
            }

    def get_workflow_state(self, session_id: str) -> "WorkflowState":
        """워크플로우 상태 조회 (스텁)"""
        logger.warning("RAGChatbot.get_workflow_state() not implemented (stub)")
        return WorkflowState(session_id=session_id)

    def reset_workflow(self, session_id: str):
        """워크플로우 초기화"""
        logger.warning("RAGChatbot.reset_workflow() not implemented (stub)")
        pass


# =====================================================
# WorkflowState: 워크플로우 상태 (스텁)
# =====================================================

class WorkflowState:
    """광고 생성 워크플로우 상태 관리 (스텁)"""

    def __init__(
        self,
        session_id: str,
        ad_type: Optional[str] = None,
        business_type: Optional[str] = None,
        user_input: Optional[str] = None,
        style: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
    ):
        self.session_id = session_id
        self.ad_type = ad_type
        self.business_type = business_type
        self.user_input = user_input
        self.style = style
        self.aspect_ratio = aspect_ratio
        self.is_complete = False

    def get_missing_info(self) -> List[str]:
        """누락된 필수 정보 목록"""
        return []


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
