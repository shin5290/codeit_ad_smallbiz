"""
Shared Model Component Cache for Z-Image Turbo
T2I와 I2I가 동일한 모델 컴포넌트를 공유하여 메모리 절약

메모리 효율:
- 분리된 캐시: T2I (20.5GB) + I2I (20.5GB) = 41GB ❌
- 공유 캐시: Transformer + VAE + Text Encoder = 20.5GB ✅
"""

import os
import gc
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
from diffusers import (
    ZImagePipeline,
    ZImageImg2ImgPipeline,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL
)

# ==============================================================================
# 전역 공유 컴포넌트 캐시
# ==============================================================================
_GLOBAL_TRANSFORMER = None
_GLOBAL_VAE = None
_GLOBAL_TEXT_ENCODER = None
_GLOBAL_TOKENIZER = None
_GLOBAL_SCHEDULER = None
_CACHE_LOCK = threading.Lock()

# 모델 경로
ZIT_MODELS_DIR = Path(os.getenv("ZIT_MODELS_DIR", "/opt/ai-models/zit"))
ZIT_BASE_MODEL = ZIT_MODELS_DIR / "Z-Image-Turbo-BF16"


def load_shared_components(device: str = "cuda") -> Tuple:
    """
    공유 모델 컴포넌트 로드 (한 번만)

    Returns:
        (transformer, vae, text_encoder, tokenizer, scheduler)
    """
    global _GLOBAL_TRANSFORMER, _GLOBAL_VAE, _GLOBAL_TEXT_ENCODER
    global _GLOBAL_TOKENIZER, _GLOBAL_SCHEDULER

    with _CACHE_LOCK:
        # 이미 로드되었으면 반환
        if _GLOBAL_TRANSFORMER is not None:
            print("[SharedCache] ✅ Using cached components")
            return (
                _GLOBAL_TRANSFORMER,
                _GLOBAL_VAE,
                _GLOBAL_TEXT_ENCODER,
                _GLOBAL_TOKENIZER,
                _GLOBAL_SCHEDULER
            )

        print(f"[SharedCache] 🚀 Loading ZIT components (20.5GB)...")

        # Scheduler 로드
        _GLOBAL_SCHEDULER = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(ZIT_BASE_MODEL), subfolder="scheduler"
        )

        # 임시 파이프라인 로드 (컴포넌트 추출용)
        temp_pipe = ZImagePipeline.from_pretrained(
            str(ZIT_BASE_MODEL),
            scheduler=_GLOBAL_SCHEDULER,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True
        )

        # 컴포넌트 추출
        _GLOBAL_TRANSFORMER = temp_pipe.transformer
        _GLOBAL_VAE = temp_pipe.vae
        _GLOBAL_TEXT_ENCODER = temp_pipe.text_encoder
        _GLOBAL_TOKENIZER = temp_pipe.tokenizer

        # GPU로 이동
        _GLOBAL_TRANSFORMER.to(device)
        _GLOBAL_VAE.to(device)
        _GLOBAL_TEXT_ENCODER.to(device)

        # FlashAttention 최적화
        try:
            if hasattr(_GLOBAL_TRANSFORMER, "set_attn_processor"):
                from diffusers.models.attention_processor import AttnProcessor2_0
                _GLOBAL_TRANSFORMER.set_attn_processor(AttnProcessor2_0())
                print(f"[SharedCache] FlashAttention enabled")
        except Exception as e:
            print(f"[SharedCache] Could not enable attention optimization: {e}")

        # VAE 최적화
        _GLOBAL_VAE.enable_tiling()
        _GLOBAL_VAE.enable_slicing()
        print(f"[SharedCache] VAE tiling/slicing enabled")

        # 임시 파이프라인 삭제
        del temp_pipe
        gc.collect()

        print(f"[SharedCache] ✅ Components loaded on {device}")

        return (
            _GLOBAL_TRANSFORMER,
            _GLOBAL_VAE,
            _GLOBAL_TEXT_ENCODER,
            _GLOBAL_TOKENIZER,
            _GLOBAL_SCHEDULER
        )


def get_t2i_pipeline(device: str = "cuda") -> ZImagePipeline:
    """
    공유 컴포넌트를 사용하는 T2I 파이프라인 생성

    Returns:
        ZImagePipeline (메모리 추가 사용 없음)
    """
    transformer, vae, text_encoder, tokenizer, scheduler = load_shared_components(device)

    pipe = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer
    )

    # 파이프라인을 명시적으로 GPU로 이동
    pipe.enable_attention_slicing()
    pipe.to(device)

    print("[SharedCache] T2I pipeline created (using shared components)")
    return pipe


def get_i2i_pipeline(device: str = "cuda") -> ZImageImg2ImgPipeline:
    """
    공유 컴포넌트를 사용하는 I2I 파이프라인 생성

    Returns:
        ZImageImg2ImgPipeline (메모리 추가 사용 없음)
    """
    transformer, vae, text_encoder, tokenizer, scheduler = load_shared_components(device)

    pipe = ZImageImg2ImgPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer
    )

    # 파이프라인을 명시적으로 GPU로 이동
    pipe.enable_attention_slicing()
    pipe.to(device)

    print("[SharedCache] I2I pipeline created (using shared components)")
    return pipe


def flush_shared_cache():
    """전역 캐시 완전 초기화"""
    global _GLOBAL_TRANSFORMER, _GLOBAL_VAE, _GLOBAL_TEXT_ENCODER
    global _GLOBAL_TOKENIZER, _GLOBAL_SCHEDULER

    with _CACHE_LOCK:
        if _GLOBAL_TRANSFORMER is not None:
            del _GLOBAL_TRANSFORMER
            del _GLOBAL_VAE
            del _GLOBAL_TEXT_ENCODER
            del _GLOBAL_TOKENIZER
            del _GLOBAL_SCHEDULER

            _GLOBAL_TRANSFORMER = None
            _GLOBAL_VAE = None
            _GLOBAL_TEXT_ENCODER = None
            _GLOBAL_TOKENIZER = None
            _GLOBAL_SCHEDULER = None

            gc.collect()
            torch.cuda.empty_cache()
            print("[SharedCache] ✅ Cache flushed")


def is_cache_loaded() -> bool:
    """캐시 로드 상태 확인"""
    return _GLOBAL_TRANSFORMER is not None
