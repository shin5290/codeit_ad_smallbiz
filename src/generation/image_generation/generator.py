"""
Image Generator
Text Generator에서 받은 프롬프트와 설정으로 이미지 생성 후 저장

흐름:
1. Backend → Text Generator → 프롬프트 + 설정 생성
2. Text Generator → Image Generator (이 모듈) → 이미지 생성 + 저장
3. Image Generator → Backend → 저장 경로/URL 반환
"""

from typing import Optional, Literal, Dict, Any
from pathlib import Path
from datetime import datetime
import uuid

from .workflow import ImageGenerationWorkflow
from .nodes.generation import Text2ImageNode


# 기본 저장 경로 (나중에 config로 분리 가능)
DEFAULT_STORAGE_DIR = Path("storage/generated_images")
DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# 스타일별 모델 매핑
STYLE_MODEL_MAP = {
    "ultra_realistic": "SG161222/RealVisXL_V4.0",
    "semi_realistic": "John6666/bss-equinox-il-semi-realistic-model-v25-sdxl",
    "anime": "cagliostrolab/animagine-xl-3.1",
}


def generate_and_save_image(
    prompt: str,
    style: Literal["ultra_realistic", "semi_realistic", "anime"] = "ultra_realistic",
    aspect_ratio: Literal["1:1", "3:4", "4:3", "16:9", "9:16"] = "1:1",
    industry: Optional[Literal["cafe", "restaurant", "retail", "service"]] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    business_id: Optional[str] = None,
    filename: Optional[str] = None,
    storage_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Text Generator에서 받은 프롬프트로 이미지 생성 후 저장

    이 함수가 Image Generation 모듈의 메인 진입점입니다.
    Backend → Text Generator → 이 함수 → 이미지 생성 + 저장 → 경로 반환

    Args:
        prompt: 생성할 이미지 설명 (Text Generator에서 생성된 프롬프트)
            예: "modern cafe interior with wooden furniture and warm lighting"
        style: 이미지 스타일
            - "ultra_realistic": 초사실적 사진 스타일
            - "semi_realistic": 반사실적 (사진 + 일러스트 중간)
            - "anime": 애니메이션/일러스트 스타일
        aspect_ratio: 이미지 비율
            - "1:1": 정사각형 (1024x1024) - SNS 프로필, 썸네일
            - "3:4": 세로 (896x1152) - Instagram 피드, 포스터
            - "4:3": 가로 (1152x896) - 프레젠테이션, 배너
            - "16:9": 와이드 (1344x768) - 유튜브 썸네일, 웹 배너
            - "9:16": 세로 (768x1344) - Instagram Story, 모바일
        industry: 업종 (config.py의 INDUSTRY_STYLES에 정의된 스타일 자동 적용)
            - "cafe": 카페
            - "restaurant": 레스토랑
            - "retail": 소매점
            - "service": 서비스업
        num_inference_steps: 생성 스텝 수 (기본: 40)
            - 빠른 생성: 20-30
            - 고품질: 50-60
        guidance_scale: CFG 스케일 (기본: 7.5)
            - 낮음(5-6): 창의적
            - 높음(9-12): 프롬프트에 충실
        seed: 랜덤 시드 (재현성, 선택)
        business_id: 사업체 ID (저장 경로 구성용, 선택)
            - 제공 시: storage/generated_images/{business_id}/{filename}
            - 미제공 시: storage/generated_images/{filename}
        filename: 파일명 (선택)
            - 제공 시: 그대로 사용
            - 미제공 시: {timestamp}_{uuid}.png 형식 자동 생성
        storage_dir: 커스텀 저장 디렉토리 (선택)
            - 기본: storage/generated_images/

    Returns:
        Dict[str, Any]: 생성 결과
            {
                "success": bool,           # 성공 여부
                "image_path": str,         # 저장된 이미지 절대 경로
                "relative_path": str,      # 상대 경로 (URL 생성용)
                "filename": str,           # 파일명
                "width": int,              # 이미지 너비
                "height": int,             # 이미지 높이
                "style": str,              # 사용된 스타일
                "seed": int or None,       # 사용된 시드
                "generation_time": float,  # 생성 소요 시간 (초)
                "error": str or None       # 에러 메시지 (실패 시)
            }

    Example:
        >>> # Text Generator 결과를 받아서 이미지 생성
        >>> result = generate_and_save_image(
        ...     prompt="modern cafe interior with wooden furniture",  # Text Generator 출력
        ...     style="ultra_realistic",
        ...     aspect_ratio="16:9",
        ...     industry="cafe",
        ...     business_id="user123"
        ... )
        >>> print(result["image_path"])  # Backend에 경로 반환
        >>> # "storage/generated_images/user123/20260105_143022_a1b2c3d4.png"
    """
    import time
    start_time = time.time()

    try:
        # 저장 디렉토리 설정
        if storage_dir is None:
            storage_dir = DEFAULT_STORAGE_DIR

        # business_id가 있으면 하위 폴더 생성
        if business_id:
            save_dir = storage_dir / business_id
        else:
            save_dir = storage_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.png"

        # 전체 저장 경로
        save_path = save_dir / filename

        # 스타일에 맞는 모델 선택
        model_id = STYLE_MODEL_MAP.get(style, STYLE_MODEL_MAP["ultra_realistic"])

        # 워크플로우 생성
        workflow = ImageGenerationWorkflow(name=f"Generate_{style}")
        workflow.add_node(Text2ImageNode(
            model_id=model_id,
            auto_unload=True  # 메모리 자동 관리
        ))

        # 입력 데이터 준비
        inputs = {
            "prompt": prompt,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        # 선택적 파라미터
        if industry is not None:
            inputs["industry"] = industry
        if seed is not None:
            inputs["seed"] = seed

        # 이미지 생성
        result = workflow.run(inputs)
        image = result["image"]

        # 이미지 저장
        image.save(save_path)

        # 생성 시간 계산
        generation_time = time.time() - start_time

        # 상대 경로 계산 (storage/ 기준)
        relative_path = str(save_path.relative_to(Path("storage")))

        return {
            "success": True,
            "image_path": str(save_path.absolute()),
            "relative_path": relative_path,
            "filename": filename,
            "width": result["width"],
            "height": result["height"],
            "style": style,
            "seed": result.get("seed"),
            "generation_time": generation_time,
            "error": None
        }

    except Exception as e:
        # 에러 발생 시
        generation_time = time.time() - start_time
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

        return {
            "success": False,
            "image_path": None,
            "relative_path": None,
            "filename": filename,
            "width": None,
            "height": None,
            "style": style,
            "seed": seed,
            "generation_time": generation_time,
            "error": error_msg
        }


def generate_batch_images(
    prompts: list[str],
    style: Literal["ultra_realistic", "semi_realistic", "anime"] = "ultra_realistic",
    aspect_ratio: Literal["1:1", "3:4", "4:3", "16:9", "9:16"] = "1:1",
    industry: Optional[Literal["cafe", "restaurant", "retail", "service"]] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    seeds: Optional[list[int]] = None,
    business_id: Optional[str] = None,
    storage_dir: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    """
    여러 프롬프트에 대해 배치로 이미지 생성 및 저장

    Args:
        prompts: 프롬프트 리스트 (Text Generator에서 생성된 프롬프트들)
        style: 이미지 스타일 (모든 이미지에 동일하게 적용)
        aspect_ratio: 이미지 비율 (모든 이미지에 동일하게 적용)
        industry: 업종
        num_inference_steps: 생성 스텝 수
        guidance_scale: CFG 스케일
        seeds: 각 이미지의 시드 리스트 (선택, prompts와 길이 같아야 함)
        business_id: 사업체 ID
        storage_dir: 저장 디렉토리

    Returns:
        생성 결과 리스트 (각 항목은 generate_and_save_image의 반환값과 동일)

    Example:
        >>> prompts = [
        ...     "modern cafe interior",
        ...     "cozy bookstore with reading corner",
        ...     "minimalist hair salon"
        ... ]
        >>> results = generate_batch_images(
        ...     prompts=prompts,
        ...     style="semi_realistic",
        ...     aspect_ratio="4:3",
        ...     business_id="user123"
        ... )
        >>> for r in results:
        ...     if r["success"]:
        ...         print(f"Saved: {r['relative_path']}")
    """
    if seeds is not None and len(seeds) != len(prompts):
        raise ValueError(f"seeds 길이({len(seeds)})와 prompts 길이({len(prompts)})가 다릅니다")

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[Batch {i+1}/{len(prompts)}] Generating: {prompt[:50]}...")

        # 시드 설정
        seed = seeds[i] if seeds is not None else None

        # 파일명 생성 (배치 인덱스 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"batch_{i}_{timestamp}_{unique_id}.png"

        # 이미지 생성 및 저장
        result = generate_and_save_image(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            industry=industry,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            business_id=business_id,
            filename=filename,
            storage_dir=storage_dir,
        )

        results.append(result)

        if result["success"]:
            print(f"✅ Saved: {result['relative_path']}")
        else:
            print(f"❌ Failed: {result['error'][:100]}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\n✅ Batch complete: {success_count}/{len(results)} images generated")

    return results
