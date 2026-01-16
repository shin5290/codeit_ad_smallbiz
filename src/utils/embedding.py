"""
임베딩 생성 유틸리티

OpenAI Embedding API를 사용하여 텍스트를 벡터로 변환합니다.
pgvector와 함께 사용하여 의미 기반 유사도 검색을 지원합니다.
"""

import logging
from typing import List, Optional
import openai

logger = logging.getLogger(__name__)


def generate_embedding(
    text: str,
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small"
) -> Optional[List[float]]:
    """
    텍스트를 임베딩 벡터로 변환

    Args:
        text: 임베딩할 텍스트
        api_key: OpenAI API 키 (None이면 환경 변수에서 로드)
        model: 임베딩 모델 이름
            - text-embedding-3-small: 1536차원, 빠르고 저렴
            - text-embedding-3-large: 3072차원, 더 정확하지만 비쌈
            - text-embedding-ada-002: 1536차원, 구버전 (deprecated)

    Returns:
        임베딩 벡터 (List[float]) 또는 에러 시 None

    Examples:
        >>> embedding = generate_embedding("안녕하세요")
        >>> len(embedding)
        1536
    """
    if not text or not text.strip():
        logger.warning("generate_embedding: empty text provided")
        return None

    try:
        # API 키 설정
        if api_key:
            client = openai.OpenAI(api_key=api_key)
        else:
            # 환경 변수에서 로드 (OPENAI_API_KEY)
            client = openai.OpenAI()

        # 임베딩 생성
        response = client.embeddings.create(
            model=model,
            input=text.strip(),
            encoding_format="float"
        )

        # 벡터 추출
        embedding = response.data[0].embedding

        logger.info(
            f"generate_embedding: success, text_len={len(text)}, "
            f"vector_dim={len(embedding)}, model={model}"
        )

        return embedding

    except openai.AuthenticationError as e:
        logger.error(f"generate_embedding: authentication error: {e}")
        return None
    except openai.RateLimitError as e:
        logger.error(f"generate_embedding: rate limit error: {e}")
        return None
    except openai.APIError as e:
        logger.error(f"generate_embedding: API error: {e}")
        return None
    except Exception as e:
        logger.error(f"generate_embedding: unexpected error: {e}", exc_info=True)
        return None


async def generate_embedding_async(
    text: str,
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small"
) -> Optional[List[float]]:
    """
    텍스트를 임베딩 벡터로 변환 (비동기)

    Args:
        text: 임베딩할 텍스트
        api_key: OpenAI API 키 (None이면 환경 변수에서 로드)
        model: 임베딩 모델 이름

    Returns:
        임베딩 벡터 (List[float]) 또는 에러 시 None

    Examples:
        >>> embedding = await generate_embedding_async("안녕하세요")
        >>> len(embedding)
        1536
    """
    if not text or not text.strip():
        logger.warning("generate_embedding_async: empty text provided")
        return None

    try:
        # API 키 설정
        if api_key:
            client = openai.AsyncOpenAI(api_key=api_key)
        else:
            # 환경 변수에서 로드 (OPENAI_API_KEY)
            client = openai.AsyncOpenAI()

        # 임베딩 생성
        response = await client.embeddings.create(
            model=model,
            input=text.strip(),
            encoding_format="float"
        )

        # 벡터 추출
        embedding = response.data[0].embedding

        logger.info(
            f"generate_embedding_async: success, text_len={len(text)}, "
            f"vector_dim={len(embedding)}, model={model}"
        )

        return embedding

    except openai.AuthenticationError as e:
        logger.error(f"generate_embedding_async: authentication error: {e}")
        return None
    except openai.RateLimitError as e:
        logger.error(f"generate_embedding_async: rate limit error: {e}")
        return None
    except openai.APIError as e:
        logger.error(f"generate_embedding_async: API error: {e}")
        return None
    except Exception as e:
        logger.error(f"generate_embedding_async: unexpected error: {e}", exc_info=True)
        return None


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터의 코사인 유사도 계산

    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터

    Returns:
        코사인 유사도 (-1.0 ~ 1.0)

    Examples:
        >>> vec1 = [1.0, 0.0, 0.0]
        >>> vec2 = [0.0, 1.0, 0.0]
        >>> calculate_cosine_similarity(vec1, vec2)
        0.0
    """
    import numpy as np

    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
