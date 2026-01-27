"""RAG 모듈"""
from .chain import SmallBizRAG
from .prompts import PromptBuilder, UserContext, IntentRouter
from .knowledge_base import (
    ConsultingKnowledgeBase,
    SmallBizKnowledgeBase,
    SmallBizConsultingKnowledgeBase,
    get_knowledge_base,
    reset_knowledge_base,
)

__all__ = [
    # RAG Chain
    "SmallBizRAG",
    "PromptBuilder",
    "UserContext",
    "IntentRouter",
    # Knowledge Base (백엔드 연동용)
    "ConsultingKnowledgeBase",
    "SmallBizKnowledgeBase",
    "SmallBizConsultingKnowledgeBase",
    "get_knowledge_base",
    "reset_knowledge_base",
]
