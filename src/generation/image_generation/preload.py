"""
Model Preloader
서버 시작 시 이미지 생성 모델을 미리 GPU에 로드

목적:
- 첫 사용자 요청 시 60초 대기 제거
- UX 개선 (모든 요청이 일관된 응답 시간)
"""

import asyncio
import threading
from typing import Optional

from .shared_cache import load_shared_components, is_cache_loaded
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_STATE = "idle"  # idle | loading | ready | failed
_MODEL_ERROR: Optional[str] = None
_MODEL_STATE_LOCK = threading.Lock()
_MODEL_READY_EVENT = threading.Event()


def _set_model_state(state: str, error: Optional[str] = None) -> None:
    global _MODEL_STATE, _MODEL_ERROR
    with _MODEL_STATE_LOCK:
        _MODEL_STATE = state
        _MODEL_ERROR = error
        if state == "loading":
            _MODEL_READY_EVENT.clear()
        elif state in ("ready", "failed"):
            _MODEL_READY_EVENT.set()


def get_model_load_state() -> str:
    """현재 모델 로딩 상태 반환."""
    with _MODEL_STATE_LOCK:
        return _MODEL_STATE


def get_model_load_error() -> Optional[str]:
    """모델 로딩 실패 메시지 반환."""
    with _MODEL_STATE_LOCK:
        return _MODEL_ERROR


def is_model_ready() -> bool:
    """모델 로딩 완료 여부."""
    if is_cache_loaded():
        _set_model_state("ready")
        return True
    with _MODEL_STATE_LOCK:
        return _MODEL_STATE == "ready"


def start_model_preload(device: str = "cuda") -> None:
    """모델 프리로드를 백그라운드로 시작."""
    if is_model_ready():
        return
    with _MODEL_STATE_LOCK:
        if _MODEL_STATE == "loading":
            return
    thread = threading.Thread(
        target=preload_models,
        kwargs={"device": device},
        daemon=True,
    )
    thread.start()


async def wait_for_model_ready(timeout: Optional[float] = None) -> bool:
    """모델 로딩 완료까지 대기 (비동기)."""
    if is_model_ready():
        return True
    start_model_preload()
    completed = await asyncio.to_thread(_MODEL_READY_EVENT.wait, timeout)
    if not completed:
        return False
    with _MODEL_STATE_LOCK:
        return _MODEL_STATE == "ready"


def preload_models(device: str = "cuda"):
    """
    서버 시작 시 Z-Image Turbo 모델을 GPU에 미리 로드

    Args:
        device: 디바이스 (cuda 또는 cpu)

    Returns:
        None

    Example:
        # main.py의 lifespan에서 호출
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            preload_models()  # 모델 로드
            yield
    """
    global _MODEL_STATE, _MODEL_ERROR
    try:
        if is_model_ready():
            return
        with _MODEL_STATE_LOCK:
            if _MODEL_STATE == "loading":
                return
            _MODEL_STATE = "loading"
            _MODEL_ERROR = None
            _MODEL_READY_EVENT.clear()

        logger.info(f"[Preload] Starting model preload on {device}...")
        logger.info("[Preload] This may take 60-90 seconds on first load...")

        # shared_cache를 통해 모델 로드
        # 이후 모든 Text2ImageNode/Image2ImageNode 호출 시 즉시 사용 가능
        load_shared_components(device=device)

        _set_model_state("ready")
        logger.info("[Preload] Model preload complete!")
        logger.info("[Preload] Image generation requests will now be fast (~8 seconds)")

    except Exception as e:
        _set_model_state("failed", str(e))
        logger.error(f"[Preload] Model preload failed: {e}")
        logger.error("[Preload] Server will still start, but first request will be slow")
        import traceback
        logger.error(traceback.format_exc())
