"""
Image Generator (Z-Image Turbo)
Z-Image Turbo 모델을 사용한 고속 이미지 생성

노드 기반 아키텍처:
- PromptProcessorNode: 한글 → 영어 프롬프트 변환
- Text2ImageNode / Image2ImageNode: 이미지 생성
- SaveImageNode: 이미지 저장

특징:
- 8 steps로 고품질 이미지 생성 (~1-2초)
- 긴 프롬프트 지원 (CLIP 77 토큰 제한 없음, T5 기반)
- LoRA를 통한 스타일 전환
- Negative Prompt 미지원 (CFG 미사용)

흐름:
1. PromptProcessorNode → 한글 입력 → 영어 프롬프트
2. Text2ImageNode/Image2ImageNode → 이미지 생성
3. SaveImageNode → 이미지 저장
4. Backend → 저장 경로/URL 반환
"""

from typing import Optional, Literal, Dict, Any, Callable
from pathlib import Path
import time

from PIL import Image

from src.utils.logging import get_logger

from .workflow import ImageGenerationWorkflow
from .nodes.text2image import Text2ImageNode
from .nodes.image2image import Image2ImageNode
from .nodes.prompt_processor import PromptProcessorNode
from .nodes.save_image import SaveImageNode
from .nodes.gpt_layout_analyzer import GPTLayoutAnalyzerNode
from .nodes.text_overlay import TextOverlayNode

logger = get_logger(__name__)


# 기본 저장 경로: /mnt/data/generated
DEFAULT_STORAGE_DIR = Path("/mnt/data/generated")
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
    strength: float = 0.6,  # I2I 변형 강도 (0.3~0.7 권장)
    control_type: Literal["canny", "depth", "pose"] = "canny",
    controlnet_conditioning_scale: float = 0.8,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    한글 사용자 입력으로 자동 프롬프트 생성 후 이미지 생성 및 저장

    노드 기반 워크플로우:
    1. PromptProcessorNode: 한글 → 영어 프롬프트 변환 + 텍스트 추출
    2. Text2ImageNode/Image2ImageNode: 이미지 생성
    3. GPTLayoutAnalyzerNode: GPT-4V 이미지 분석 및 레이아웃 결정 (조건부)
    4. TextOverlayNode: 텍스트 오버레이 (조건부)
    5. SaveImageNode: 이미지 저장

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
        reference_image: I2I용 참조 이미지 (선택)
        strength: I2I 변형 강도 (0.3~0.7 권장)
        control_type: ControlNet 타입 (현재 미사용)
        controlnet_conditioning_scale: ControlNet 강도 (현재 미사용)

    Returns:
        Dict[str, Any]: 생성 결과
            {
                "success": bool,
                "image_path": str,
                "filename": str,
                "width": int,
                "height": int,
                "style": str,
                "industry": str or None,
                "seed": int or None,
                "generation_time": float,
                "prompt": str,
                "generation_method": str,
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
    start_time = time.time()

    # ControlNet 파라미터는 현재 미사용 (ZIT는 ControlNet 미지원)
    _ = control_type
    _ = controlnet_conditioning_scale

    try:
        generation_method = "i2i" if reference_image is not None else "t2i"
        # 저장 디렉토리 설정
        if storage_dir is None:
            storage_dir = DEFAULT_STORAGE_DIR

        def emit_progress(event: str, node_name: str) -> None:
            if progress_callback:
                progress_callback({"event": event, "node": node_name})

        def on_node_start(node, _data) -> None:
            if node.node_name in ("Text2ImageNode", "Image2ImageNode"):
                emit_progress("image_generation_start", node.node_name)
            elif node.node_name == "SaveImageNode":
                emit_progress("image_save_start", node.node_name)

        def on_node_end(node, _data) -> None:
            if node.node_name == "PromptProcessorNode":
                emit_progress("prompt_done", node.node_name)

        # 워크플로우 구성
        if reference_image is not None:
            # I2I 워크플로우: Prompt → I2I → (Text Overlay) → Save
            workflow = ImageGenerationWorkflow(name=f"ZIT_I2I_{style}")
            workflow.add_node(PromptProcessorNode(default_style=style))
            workflow.add_node(Image2ImageNode(auto_unload=False))

            # 텍스트 오버레이 노드 (조건부 실행: text_data 있을 때만)
            workflow.add_node(GPTLayoutAnalyzerNode())
            workflow.add_node(TextOverlayNode())

            workflow.add_node(SaveImageNode(storage_dir=storage_dir))

            # 입력 데이터
            inputs = {
                "user_input": user_input,
                "style": style,
                "reference_image": reference_image,
                "strength": strength,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": num_inference_steps,
                "industry": industry,
            }
        else:
            # T2I 워크플로우: Prompt → T2I → Save(원본) → (Text Overlay) → Save(최종)
            workflow = ImageGenerationWorkflow(name=f"ZIT_Generate_{style}")
            workflow.add_node(PromptProcessorNode(default_style=style))
            workflow.add_node(Text2ImageNode(auto_unload=False))

            # 원본 이미지 저장 (subdir/origin/)
            workflow.add_node(SaveImageNode(storage_dir=storage_dir, is_origin=True))

            # 텍스트 오버레이 노드 (조건부 실행: text_data 있을 때만)
            workflow.add_node(GPTLayoutAnalyzerNode())
            workflow.add_node(TextOverlayNode())

            # 최종 이미지 저장 (subdir/)
            workflow.add_node(SaveImageNode(storage_dir=storage_dir, is_origin=False))

            # 입력 데이터
            inputs = {
                "user_input": user_input,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": num_inference_steps,
                "industry": industry,
            }

        # seed 추가 (선택적)
        if seed is not None:
            inputs["seed"] = seed

        # 워크플로우 실행
        result = workflow.run(
            inputs,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
        )

        generation_time = time.time() - start_time

        return {
            "success": True,
            "image_path": result["image_path"],
            "filename": result["filename"],
            "width": result["width"],
            "height": result["height"],
            "style": result.get("detected_style", style),
            "industry": result.get("industry"),
            "seed": result.get("seed"),
            "generation_time": generation_time,
            "prompt": result["prompt"],
            "generation_method": generation_method,
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
            "industry": None,
            "seed": seed,
            "generation_time": generation_time,
            "prompt": None,
            "generation_method": "i2i" if reference_image is not None else "t2i",
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
        prompts: 프롬프트 리스트 (한글 사용자 입력)
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
        logger.info(f"[Batch {i+1}/{len(prompts)}] Generating: {prompt[:50]}...")

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
            logger.info(f"Saved: {result['image_path']}")
        else:
            logger.error(f"Failed: {result['error'][:100]}")

    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Batch complete: {success_count}/{len(results)} images generated")

    return results
