"""
RAG Preloader
서버 시작 시 상담 RAG(벡터스토어/임베딩 모델)를 백그라운드로 로드
"""

import threading
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

_RAG_STATE = "idle"  # idle | loading | ready | failed
_RAG_ERROR: Optional[str] = None
_RAG_STATE_LOCK = threading.Lock()
_RAG_READY_EVENT = threading.Event()


def _set_rag_state(state: str, error: Optional[str] = None) -> None:
    global _RAG_STATE, _RAG_ERROR
    with _RAG_STATE_LOCK:
        _RAG_STATE = state
        _RAG_ERROR = error
        if state == "loading":
            _RAG_READY_EVENT.clear()
        elif state in ("ready", "failed"):
            _RAG_READY_EVENT.set()


def is_rag_ready() -> bool:
    """RAG 로딩 완료 여부."""
    with _RAG_STATE_LOCK:
        return _RAG_STATE == "ready"


def start_rag_preload() -> None:
    """RAG 프리로드를 백그라운드로 시작."""
    if is_rag_ready():
        return
    with _RAG_STATE_LOCK:
        if _RAG_STATE == "loading":
            return
    thread = threading.Thread(
        target=preload_rag,
        daemon=True,
    )
    thread.start()


def preload_rag() -> None:
    """벡터스토어/임베딩 모델을 백그라운드로 로드."""
    global _RAG_STATE, _RAG_ERROR
    try:
        if is_rag_ready():
            return
        with _RAG_STATE_LOCK:
            if _RAG_STATE == "loading":
                return
            _RAG_STATE = "loading"
            _RAG_ERROR = None
            _RAG_READY_EVENT.clear()

        logger.info("[RAG Preload] Starting RAG preload in background...")
        from src.backend.chatbot import get_consulting_rag

        get_consulting_rag()
        _set_rag_state("ready")
        logger.info("[RAG Preload] RAG preload complete!")

    except Exception as exc:
        _set_rag_state("failed", str(exc))
        logger.error(f"[RAG Preload] RAG preload failed: {exc}")
