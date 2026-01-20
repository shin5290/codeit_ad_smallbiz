"""
Model Preloader
서버 시작 시 이미지 생성 모델을 미리 GPU에 로드

목적:
- 첫 사용자 요청 시 60초 대기 제거
- UX 개선 (모든 요청이 일관된 응답 시간)
"""

from .shared_cache import load_shared_components
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
    try:
        logger.info(f"[Preload] Starting model preload on {device}...")
        logger.info("[Preload] This may take 60-90 seconds on first load...")

        # shared_cache를 통해 모델 로드
        # 이후 모든 Text2ImageNode/Image2ImageNode 호출 시 즉시 사용 가능
        load_shared_components(device=device)

        logger.info("[Preload] Model preload complete!")
        logger.info("[Preload] Image generation requests will now be fast (~8 seconds)")

    except Exception as e:
        logger.error(f"[Preload] Model preload failed: {e}")
        logger.error("[Preload] Server will still start, but first request will be slow")
        import traceback
        logger.error(traceback.format_exc())
