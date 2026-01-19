"""
Image Generator (Z-Image Turbo)
Z-Image Turbo 모델을 사용한 고속 이미지 생성

특징:
- 8 steps로 고품질 이미지 생성 (~1-2초)
- 긴 프롬프트 지원 (CLIP 77 토큰 제한 없음, T5 기반)
- LoRA를 통한 스타일 전환
- Negative Prompt 미지원 (CFG 미사용)

흐름:
1. Backend → 프롬프트 + 설정 생성
2. Image Generator (이 모듈) → 이미지 생성 + 저장
3. Backend → 저장 경로/URL 반환
"""

from typing import Optional, Literal, Dict, Any
from pathlib import Path
import io
import hashlib

from PIL import Image

from src.utils.config import PROJECT_ROOT as _PROJECT_ROOT
from .workflow import ImageGenerationWorkflow
from .nodes.text2image import Text2ImageNode
from .prompt import PromptTemplateManager


PROJECT_ROOT = Path(_PROJECT_ROOT)

# 기본 저장 경로
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "data" / "generated"
DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def generate_and_save_image(
    user_input: str,
    style: Literal["realistic", "ultra_realistic", "semi_realistic", "anime"] = "realistic",
    aspect_ratio: Literal["1:1", "3:4", "4:3", "16:9", "9:16"] = "1:1",
    industry: Optional[Literal["cafe", "restaurant", "retail", "service"]] = None,
    num_inference_steps: int = 8,  # Z-Image Turbo 기본값
    seed: Optional[int] = None,
    filename: Optional[str] = None,
    storage_dir: Optional[Path] = None,
    reference_image: Optional[Image.Image] = None,
    control_type: Literal["canny", "depth", "pose"] = "canny",
    controlnet_conditioning_scale: float = 0.8,
) -> Dict[str, Any]:
    """
    한글 사용자 입력으로 자동 프롬프트 생성 후 이미지 생성 및 저장

    Z-Image Turbo 사용으로 SDXL 대비:
    - 5배 빠른 생성 (8 steps vs 40 steps)
    - 긴 프롬프트 지원 (토큰 제한 없음)
    - LoRA로 스타일 전환

    Args:
        user_input: 한글 사용자 입력 (자동으로 영어 프롬프트로 변환)
            예: "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌"
        style: 이미지 스타일 (LoRA로 적용)
            - "realistic": 사실적 사진 스타일 (베이스 모델)
            - "ultra_realistic": realistic과 동일
            - "semi_realistic": 반사실적 (LoRA)
            - "anime": 애니메이션/일러스트 스타일 (LoRA)
        aspect_ratio: 이미지 비율
            - "1:1": 정사각형 (1024x1024)
            - "3:4": 세로 (896x1152)
            - "4:3": 가로 (1152x896)
            - "16:9": 와이드 (1344x768)
            - "9:16": 세로 (768x1344)
        industry: 업종 (참고용 키워드 제공)
        num_inference_steps: 생성 스텝 수 (기본: 8, Z-Image Turbo 권장값)
        seed: 랜덤 시드 (재현성)
        filename: 커스텀 파일명 (선택, 미사용 - 해시 기반 자동 생성)
        storage_dir: 커스텀 저장 디렉토리 (선택)
    Returns:
        Dict[str, Any]: 생성 결과
            {
                "success": bool,
                "image_path": str,
                "filename": str,
                "width": int,
                "height": int,
                "style": str,
                "seed": int or None,
                "generation_time": float,
                "prompt": str,
                "error": str or None
            }

    Example:
        >>> result = generate_and_save_image(
        ...     user_input="카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
        ...     style="realistic",
        ...     aspect_ratio="16:9"
        ... )
        >>> print(result["image_path"])
    """
    import time
    start_time = time.time()

    try:
        # 1. 한글 입력 → 상세 영어 프롬프트 생성 (GPT)
        prompt_generator = PromptTemplateManager()
        prompt_result = prompt_generator.generate_detailed_prompt(
            user_input=user_input,
            style=style
        )

        prompt = prompt_result["positive"]
        # Z-Image Turbo는 negative_prompt 미지원 (CFG 미사용)
        detected_style = prompt_result.get("style", style)

        # GPT가 감지한 스타일로 업데이트
        if detected_style in ["realistic", "ultra_realistic", "semi_realistic", "anime"]:
            style = detected_style

        # 2. I2I 분기 처리 (reference_image가 있으면 ControlNet 사용)
        if reference_image is not None:
            # TODO: Z-Image Turbo ControlNet 구현 필요
            # 현재는 미구현 - 에러 반환
            _ = control_type  # 향후 사용 예정
            _ = controlnet_conditioning_scale  # 향후 사용 예정
            raise NotImplementedError(
                "Z-Image Turbo ControlNet I2I는 아직 구현되지 않았습니다. "
                "SDXL 버전을 사용하려면 generator_sdxl.py를 import하세요."
            )

        # 3. 저장 디렉토리 설정
        if storage_dir is None:
            storage_dir = DEFAULT_STORAGE_DIR
        storage_dir.mkdir(parents=True, exist_ok=True)

        # 4. 워크플로우 생성 (Z-Image Turbo)
        workflow = ImageGenerationWorkflow(name=f"ZIT_Generate_{style}")
        workflow.add_node(Text2ImageNode(auto_unload=False))

        # 5. 입력 데이터 준비 (negative_prompt 없음 - ZIT 미지원)
        inputs = {
            "prompt": prompt,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
        }

        if seed is not None:
            inputs["seed"] = seed

        # 5. 이미지 생성
        result = workflow.run(inputs)
        image = result["image"]

        # 6. 해시 기반 파일명 생성 및 저장
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        filename = hashlib.sha256(image_bytes).hexdigest()
        subdir = filename[:2]

        save_path = storage_dir / subdir / f"{filename}.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path, format='JPEG', quality=95)

        generation_time = time.time() - start_time

        return {
            "success": True,
            "image_path": str(save_path.absolute()),
            "filename": filename,
            "width": result["width"],
            "height": result["height"],
            "style": style,
            "seed": result.get("seed"),
            "generation_time": generation_time,
            "prompt": prompt,
            "error": None
        }

    except Exception as e:
        generation_time = time.time() - start_time
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

        return {
            "success": False,
            "image_path": None,
            "filename": filename,
            "width": None,
            "height": None,
            "style": style,
            "seed": seed,
            "generation_time": generation_time,
            "prompt": prompt if 'prompt' in dir() else None,
            "error": error_msg
        }


def generate_batch_images(
    prompts: list[str],
    style: Literal["realistic", "ultra_realistic", "semi_realistic", "anime"] = "realistic",
    aspect_ratio: Literal["1:1", "3:4", "4:3", "16:9", "9:16"] = "1:1",
    industry: Optional[Literal["cafe", "restaurant", "retail", "service"]] = None,
    num_inference_steps: int = 8,
    seeds: Optional[list[int]] = None,
    storage_dir: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    """
    여러 프롬프트에 대해 배치로 이미지 생성 및 저장

    Args:
        prompts: 프롬프트 리스트
        style: 이미지 스타일
        aspect_ratio: 이미지 비율
        industry: 업종
        num_inference_steps: 생성 스텝 수
        seeds: 각 이미지의 시드 리스트
        storage_dir: 저장 디렉토리

    Returns:
        생성 결과 리스트
    """
    if seeds is not None and len(seeds) != len(prompts):
        raise ValueError(f"seeds 길이({len(seeds)})와 prompts 길이({len(prompts)})가 다릅니다")

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[Batch {i+1}/{len(prompts)}] Generating: {prompt[:50]}...")

        seed = seeds[i] if seeds is not None else None

        result = generate_and_save_image(
            user_input=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            industry=industry,
            num_inference_steps=num_inference_steps,
            seed=seed,
            storage_dir=storage_dir,
        )

        results.append(result)

        if result["success"]:
            print(f"✅ Saved: {result['image_path']}")
        else:
            print(f"❌ Failed: {result['error'][:100]}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\n✅ Batch complete: {success_count}/{len(results)} images generated")

    return results
