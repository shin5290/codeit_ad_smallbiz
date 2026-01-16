"""
Z-Image Turbo Text2Image Node
Z-Image Turbo 모델을 사용한 고속 이미지 생성 노드

특징:
- 8 steps로 고품질 이미지 생성 (~1-2초)
- 긴 프롬프트 지원 (CLIP 77 토큰 제한 없음, T5 기반)
- LoRA를 통한 스타일 전환 지원
- Negative Prompt 미지원 (CFG 미사용)
"""

import os
import sys

# [전략 1] CUDA 메모리 설정: torch import 전에 설정해야 효력이 있음
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:256"

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import gc
from PIL import Image
import threading # [전략 2] 동시성 제어용

from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL
)

from .base import BaseNode
from ..config import (
    model_config,
    generation_config,
    aspect_ratio_templates,
)

# Z-Image Turbo 모델 경로
ZIT_MODELS_DIR = Path(os.getenv("ZIT_MODELS_DIR", "/opt/ai-models/zit"))
ZIT_BASE_MODEL = ZIT_MODELS_DIR / "Z-Image-Turbo-Base"
ZIT_LORA_DIR = ZIT_MODELS_DIR / "lora"

# 스타일별 LoRA 매핑
STYLE_LORA_MAP = {
    "realistic": None,  # 베이스 모델 그대로 사용
    "ultra_realistic": None,  # 베이스 모델 그대로 사용
    "semi_realistic": "OB半写实肖像画2.0 OB Semi-Realistic Portraits z- image turbo(1).safetensors",
    "anime": "Anime-Z.safetensors",
}

# ==============================================================================
# ★ 전역 상태 관리 (캐시 + 락 + 실행 카운터)
# ==============================================================================
_GLOBAL_PIPE = None
_CURRENT_LORA = "init"
_EXECUTION_LOCK = threading.Lock() # [전략 2] GPU는 한 번에 하나의 작업만 수행
_EXECUTION_COUNT = 0               # [전략 4] 주기적 청소를 위한 카운터

class Text2ImageNode(BaseNode):
    def __init__(self, device: Optional[str] = None, auto_unload: bool = False):
        super().__init__("Text2ImageNode")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_unload = auto_unload

    def load_pipeline(self):
        global _GLOBAL_PIPE
        
        if _GLOBAL_PIPE is not None:
            return _GLOBAL_PIPE

        print(f"[{self.node_name}] 🚀 Initializing Pipeline (Enterprise Settings)...")
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(ZIT_BASE_MODEL), subfolder="scheduler"
            )

            pipe = DiffusionPipeline.from_pretrained(
                str(ZIT_BASE_MODEL),
                scheduler=scheduler,
                torch_dtype=torch.bfloat16,  # ZIT는 bfloat16 필수 (float16은 NaN 생성)
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 로딩 시 RAM 스파이크 방지
            )

            # ComfyUI 스타일 최적화
            # Model CPU Offload (Sequential보다 빠름, ComfyUI와 유사)
            pipe.enable_model_cpu_offload(gpu_id=0)

            # Attention 최적화 (ComfyUI의 pytorch attention)
            try:
                # PyTorch 2.0+ scaled dot product attention (FlashAttention)
                if hasattr(pipe, "unet") and hasattr(pipe.unet, "set_attn_processor"):
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    pipe.unet.set_attn_processor(AttnProcessor2_0())
                elif hasattr(pipe, "transformer") and hasattr(pipe.transformer, "set_attn_processor"):
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    pipe.transformer.set_attn_processor(AttnProcessor2_0())
            except Exception as e:
                print(f"[{self.node_name}] Could not enable attention optimization: {e}")

            # VAE Optimization (고해상도 대응)
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

            print(f"[{self.node_name}] ✅ Model loaded with optimized CPU offload + attention")
            
            _GLOBAL_PIPE = pipe
            return pipe

        except Exception as e:
            raise RuntimeError(f"Pipeline Load Failed: {e}")

    def safe_unload_lora(self, pipe):
        """[전략 4] 안전하게 LoRA 언로드 (버전 호환성 체크)"""
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as e:
                print(f"⚠️ Warning: unload_lora_weights failed: {e}")
        else:
            # unload가 없는 구버전 등에서는 fuse 해제 등을 고려해야 하나 ZIT는 최신이므로 패스
            print(f"⚠️ Warning: Pipeline has no unload_lora_weights method.")

    def switch_lora(self, pipe, style):
        global _CURRENT_LORA
        target_lora_file = STYLE_LORA_MAP.get(style)
        
        if _CURRENT_LORA == style: return

        # 1. Base Model로 복귀
        if target_lora_file is None:
            if _CURRENT_LORA is not None:
                print(f"[{self.node_name}] 🔄 Switching to Base Model")
                self.safe_unload_lora(pipe)
                _CURRENT_LORA = style
            return

        # 2. 새로운 LoRA 로드
        lora_path = ZIT_LORA_DIR / target_lora_file
        if lora_path.exists():
            print(f"[{self.node_name}] 🔄 Loading LoRA: {style}")
            self.safe_unload_lora(pipe) # 기존 제거
            try:
                pipe.load_lora_weights(str(lora_path))
                _CURRENT_LORA = style
            except Exception as e:
                print(f"⚠️ LoRA Load Error (Check filename/encoding): {e}")
                # 로드 실패 시 상태를 알 수 없으므로 초기화 표시
                _CURRENT_LORA = "error" 
        else:
            print(f"⚠️ LoRA file missing: {lora_path}")

    def get_generator_device(self, pipe):
        """[전략 5] Generator를 위한 올바른 디바이스 탐색"""
        # 1순위: 파이프라인의 실행 디바이스 (CPU Offload 시 'cuda'가 숨겨져 있을 수 있음)
        if hasattr(pipe, "_execution_device"):
            return pipe._execution_device
        
        # 2순위: 모델의 device 속성
        if hasattr(pipe, "device"):
            # CPU Offload 상태면 pipe.device가 'cpu'일 수 있음. 
            # 하지만 실제 연산은 가속기에서 하므로 CUDA가 있다면 CUDA를 우선해야 함.
            if pipe.device.type == "cpu" and torch.cuda.is_available():
                return torch.device("cuda")
            return pipe.device

        # 3순위: 초기화 시 설정된 디바이스
        return torch.device(self.device)

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        global _EXECUTION_COUNT

        # [전략 2] 쓰레드 락: 동시에 여러 요청이 와도 순차 처리 보장
        with _EXECUTION_LOCK:
            prompt = inputs["prompt"]
            aspect_ratio = inputs.get("aspect_ratio", generation_config.DEFAULT_ASPECT_RATIO)
            style = inputs.get("style", "realistic")
            num_inference_steps = inputs.get("num_inference_steps", 9)
            seed = inputs.get("seed", None)
            
            # [전략 3] 해상도 보정: 내림(floor) 대신 반올림(round) 사용
            w, h = aspect_ratio_templates.get_size(aspect_ratio)
            width = int(round(w / 16) * 16)
            height = int(round(h / 16) * 16)
            
            if width != w or height != h:
                # 로그 노이즈를 줄이기 위해 차이가 클 때만 출력하거나 생략 가능
                pass

            # 1. 파이프라인 준비
            pipe = self.load_pipeline()
            #self.switch_lora(pipe, style)

            # 2. 제너레이터 생성
            generator = None
            if seed is not None:
                exec_device = self.get_generator_device(pipe)
                generator = torch.Generator(device=exec_device).manual_seed(seed)

            print(f"[{self.node_name}] Generating ({width}x{height})...")
            
            # 3. 생성
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0, 
                    generator=generator,
                    output_type="pil"
                ).images[0]

            # [전략 4] 주기적 메모리 정리 (매번 하면 느림, 5회마다 실행)
            _EXECUTION_COUNT += 1
            if _EXECUTION_COUNT % 5 == 0:
                print(f"[{self.node_name}] 🧹 Periodic Memory Cleanup (Count: {_EXECUTION_COUNT})")
                gc.collect()
                torch.cuda.empty_cache()

            # auto_unload가 True일 때만 강제 종료 (보통은 False로 둠)
            if self.auto_unload:
                self.flush_global()

            return {"image": image, "seed": seed, "width": width, "height": height}
    
    def flush_global(self):
        """전역 캐시 완전 초기화"""
        global _GLOBAL_PIPE, _CURRENT_LORA
        if _GLOBAL_PIPE is not None:
            del _GLOBAL_PIPE
            _GLOBAL_PIPE = None
        _CURRENT_LORA = "init"
        gc.collect()
        torch.cuda.empty_cache()

    def get_required_inputs(self): return ["prompt"]
    def get_output_keys(self): return ["image", "seed", "width", "height"]