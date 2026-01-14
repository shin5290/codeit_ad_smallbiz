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
import io
import uuid
import hashlib

from PIL import Image

from src.utils.config import PROJECT_ROOT as _PROJECT_ROOT
from .workflow import ImageGenerationWorkflow
from .nodes.text2image import Text2ImageNode
from .nodes.controlnet import ControlNetPreprocessorNode, ControlNetLoaderNode
from .nodes.image2image import Image2ImageControlNetNode
from .prompt import PromptGenerator, PromptTemplateManager


PROJECT_ROOT = Path(_PROJECT_ROOT)

# 기본 저장 경로 (나중에 config로 분리 가능)
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "data" / "generated"
DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# 스타일별 모델 매핑
# Note: "realistic"과 "ultra_realistic" 모두 같은 모델 사용
STYLE_MODEL_MAP = {
    "realistic": "SG161222/RealVisXL_V4.0",
    "ultra_realistic": "SG161222/RealVisXL_V4.0",  # alias
    "semi_realistic": "John6666/bss-equinox-il-semi-realistic-model-v25-sdxl",
    "anime": "cagliostrolab/animagine-xl-3.1",
}


def generate_and_save_image(
    user_input: str,
    style: Literal["ultra_realistic", "semi_realistic", "anime"] = "ultra_realistic",
    aspect_ratio: Literal["1:1", "3:4", "4:3", "16:9", "9:16"] = "1:1",
    industry: Optional[Literal["cafe", "restaurant", "retail", "service"]] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
    storage_dir: Optional[Path] = None,
    reference_image: Optional[Image.Image] = None,
    control_type: Literal["canny", "depth", "openpose"] = "canny",
    controlnet_conditioning_scale: float = 0.8,
) -> Dict[str, Any]:
    """
    한글 사용자 입력으로 자동 프롬프트 생성 후 이미지 생성 및 저장

    이 함수가 Image Generation 모듈의 메인 진입점입니다.
    Backend → 이 함수 → 프롬프트 자동 생성 → 이미지 생성 + 저장 → 경로 반환

    **자동 분기 처리**:
    - reference_image=None: Text-to-Image (T2I)
    - reference_image 제공: Image-to-Image (I2I) with ControlNet

    Args:
        user_input: 한글 사용자 입력 (자동으로 영어 프롬프트로 변환)
            예: "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌"
            → GPT-4o로 영어 키워드 추출 → SDXL 프롬프트 자동 생성
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
        filename: 커스텀 파일명 (선택, 사용 안 함 - 해시 기반 자동 생성)
        storage_dir: 커스텀 저장 디렉토리 (선택)
            - 기본: {PROJECT_ROOT}/data/generated
        reference_image: 레퍼런스 이미지 (선택, PIL.Image)
            - None: Text-to-Image 실행
            - 제공: Image-to-Image (ControlNet) 실행
            - 제품 사진, 로고 등의 형태를 유지하면서 스타일 변환
        control_type: ControlNet 타입 (reference_image 제공 시만 사용)
            - "canny": 윤곽선 기반 (제품 형태 유지에 최적) **추천**
            - "depth": 깊이 기반 (공간 구조 유지)
            - "openpose": 포즈 기반 (사람 이미지용)
        controlnet_conditioning_scale: ControlNet 강도 (0.0~1.0, reference_image 제공 시만 사용)
            - 높을수록 reference_image의 구조를 더 충실히 따름
            - 기본: 0.8 (권장)

    Returns:
        Dict[str, Any]: 생성 결과
            {
                "success": bool,           # 성공 여부
                "image_path": str,         # 저장된 이미지 절대 경로
                "filename": str,           # 파일명 (해시값)
                "width": int,              # 이미지 너비
                "height": int,             # 이미지 높이
                "style": str,              # 사용된 스타일
                "seed": int or None,       # 사용된 시드
                "generation_time": float,  # 생성 소요 시간 (초)
                "control_type": str,       # I2I인 경우 ControlNet 타입
                "controlnet_scale": float, # I2I인 경우 강도값
                "error": str or None       # 에러 메시지 (실패 시)
            }

    Example:
        >>> # Text-to-Image (한글 입력)
        >>> result = generate_and_save_image(
        ...     user_input="카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
        ...     style="ultra_realistic",
        ...     aspect_ratio="16:9",
        ...     industry="cafe"
        ... )
        >>> print(result["image_path"])
        >>> # "/home/spai0415/codeit_ad_smallbiz/data/generated/a1/a1b2c3d4e5f6...hash.jpg"

        >>> # Image-to-Image (제품 이미지 + 한글 입력)
        >>> from PIL import Image
        >>> ref_img = Image.open("product.jpg")
        >>> result = generate_and_save_image(
        ...     user_input="전문적인 제품 사진, 깔끔한 배경",
        ...     reference_image=ref_img,
        ...     control_type="canny",
        ...     style="ultra_realistic"
        ... )
    """
    # 1. 한글 입력 → 영어 키워드 추출 (GPT-4o)
    keyword_extractor = PromptTemplateManager()
    keywords = keyword_extractor.extract_keywords_english(user_input)

    if not keywords:
        # Fallback: 기본 키워드
        keywords = {"product": "item", "theme": "professional"}

    # 2. 키워드 → SDXL 프롬프트 생성
    prompt_gen = PromptGenerator()
    prompt_result = prompt_gen.generate(
        industry=industry or "general",
        user_input=keywords  # Dict 타입
    )
    prompt = prompt_result["positive"]
    negative_prompt = prompt_result["negative"]

    # 3. 분기 처리: reference_image가 있으면 I2I, 없으면 T2I
    if reference_image is not None:
        # Image-to-Image (ControlNet) 실행
        return generate_with_controlnet(
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
            control_type=control_type,
            style=style,
            aspect_ratio=aspect_ratio,
            industry=industry,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
            filename=filename,
            storage_dir=storage_dir,
        )

    # Text-to-Image 실행
    import time
    start_time = time.time()

    try:
        # 저장 디렉토리 설정
        if storage_dir is None:
            storage_dir = DEFAULT_STORAGE_DIR

        storage_dir.mkdir(parents=True, exist_ok=True)

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
            "negative_prompt": negative_prompt,
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

        # PIL Image를 bytes로 변환 (해시 생성용)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # 해시 기반 파일명 생성
        filename = hashlib.sha256(image_bytes).hexdigest()
        subdir = filename[:2]

        # 전체 저장 경로
        save_path = storage_dir / subdir / f"{filename}.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 이미지 저장
        image.save(save_path, format='JPEG', quality=95)

        # 생성 시간 계산
        generation_time = time.time() - start_time

        return {
            "success": True,
            "image_path": str(save_path.absolute()),  # 절대 경로 (PROJECT_ROOT 기반)
            "filename": filename,
            "width": result["width"],
            "height": result["height"],
            "style": style,
            "seed": result.get("seed"),
            "generation_time": generation_time,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
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
            "prompt": prompt if 'prompt' in dir() else None,
            "negative_prompt": negative_prompt if 'negative_prompt' in dir() else None,
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


def generate_with_controlnet(
    prompt: str,
    reference_image: Image.Image,
    negative_prompt: Optional[str] = None,
    control_type: Literal["canny", "depth", "openpose"] = "canny",
    style: Literal["ultra_realistic", "semi_realistic", "anime"] = "ultra_realistic",
    aspect_ratio: Literal["1:1", "3:4", "4:3", "16:9", "9:16"] = "1:1",
    industry: Optional[Literal["cafe", "restaurant", "retail", "service"]] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 0.8,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
    storage_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    ControlNet을 사용한 Image-to-Image 생성 후 저장

    제품 이미지나 레퍼런스 이미지의 구조(형태)를 유지하면서
    선택된 스타일로 재생성합니다.

    사용 예시:
    - 실사 제품 사진 → ultra_realistic 광고 이미지
    - 실사 제품 사진 → semi_realistic 스타일
    - 실사 제품 사진 → anime 스타일

    Args:
        prompt: 생성할 이미지 설명
            예: "professional product photo of coffee cup on wooden table"
        reference_image: 레퍼런스 이미지 (PIL.Image)
            - 제품 사진, 로고, 기존 이미지 등
            - ControlNet이 이 이미지의 구조를 추출함
        control_type: ControlNet 타입
            - "canny": 윤곽선 기반 (제품 형태 유지에 최적) **추천**
            - "depth": 깊이 기반 (공간 구조 유지)
            - "openpose": 포즈 기반 (사람 이미지용)
        style: 이미지 스타일
            - "ultra_realistic": 초사실적
            - "semi_realistic": 반사실적
            - "anime": 애니메이션
        aspect_ratio: 이미지 비율
        industry: 업종
        num_inference_steps: 생성 스텝 수
        guidance_scale: CFG 스케일
        controlnet_conditioning_scale: ControlNet 강도 (0.0~1.0)
            - 높을수록 reference_image의 구조를 더 충실히 따름
            - 기본: 0.8 (권장)
            - 형태 강하게 유지: 0.9~1.0
            - 형태 약하게 유지: 0.5~0.7
        seed: 랜덤 시드
        filename: 커스텀 파일명 (선택, 사용 안 함 - 해시 기반 자동 생성)
        storage_dir: 커스텀 저장 디렉토리 (선택)
            - 기본: {PROJECT_ROOT}/data/generated

    Returns:
        Dict[str, Any]: 생성 결과 (generate_and_save_image와 동일)
            {
                "success": bool,
                "image_path": str,             # 절대 경로
                "filename": str,               # 파일명 (해시값)
                "width": int,
                "height": int,
                "style": str,
                "seed": int or None,
                "generation_time": float,
                "control_type": str,           # 사용된 ControlNet 타입
                "controlnet_scale": float,     # ControlNet 강도
                "error": str or None
            }

    Example:
        >>> # 실사 커피잔 사진을 ultra_realistic 광고로 재생성
        >>> from PIL import Image
        >>> product_photo = Image.open("coffee_cup.jpg")
        >>> result = generate_with_controlnet(
        ...     prompt="professional product photo of artisan coffee cup on wooden cafe table, warm lighting",
        ...     reference_image=product_photo,
        ...     control_type="canny",
        ...     style="ultra_realistic",
        ...     aspect_ratio="1:1",
        ...     industry="cafe"
        ... )
        >>> print(result["image_path"])
        >>> # "/home/spai0415/codeit_ad_smallbiz/data/generated/a1/a1b2c3d4e5f6...hash.jpg"
    """
    import time
    start_time = time.time()

    try:
        # 저장 디렉토리 설정
        if storage_dir is None:
            storage_dir = DEFAULT_STORAGE_DIR

        storage_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 생성 먼저 수행 (해시 계산을 위해)

        # 스타일에 맞는 모델 선택
        model_id = STYLE_MODEL_MAP.get(style, STYLE_MODEL_MAP[style])

        # 워크플로우 생성
        workflow = ImageGenerationWorkflow(name=f"ControlNet_{control_type}_{style}")

        # 1. ControlNet Preprocessor 노드 추가 (윤곽선/깊이 추출)
        workflow.add_node(ControlNetPreprocessorNode(control_type=control_type))

        # 2. ControlNet 모델 로더 노드 추가
        workflow.add_node(ControlNetLoaderNode(control_type=control_type))

        # 3. Image2Image 생성 노드 추가
        workflow.add_node(Image2ImageControlNetNode(
            model_id=model_id,
            auto_unload=True
        ))

        # 입력 데이터 준비
        inputs = {
            "image": reference_image,  # Preprocessor에 전달
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        # 선택적 파라미터
        if industry is not None:
            inputs["industry"] = industry
        if seed is not None:
            inputs["seed"] = seed

        # 이미지 생성
        result = workflow.run(inputs)
        image = result["image"]

        # PIL Image를 bytes로 변환 (해시 생성용)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # 해시 기반 파일명 생성
        filename = hashlib.sha256(image_bytes).hexdigest()
        subdir = filename[:2]

        # 전체 저장 경로
        save_path = storage_dir / subdir / f"{filename}.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 이미지 저장
        image.save(save_path, format='JPEG', quality=95)

        # 생성 시간 계산
        generation_time = time.time() - start_time

        return {
            "success": True,
            "image_path": str(save_path.absolute()),  # 절대 경로 (PROJECT_ROOT 기반)
            "filename": filename,
            "width": result["width"],
            "height": result["height"],
            "style": style,
            "seed": result.get("seed"),
            "generation_time": generation_time,
            "control_type": control_type,
            "controlnet_scale": controlnet_conditioning_scale,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
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
            "control_type": control_type,
            "controlnet_scale": controlnet_conditioning_scale,
            "prompt": prompt if 'prompt' in dir() else None,
            "negative_prompt": negative_prompt if 'negative_prompt' in dir() else None,
            "error": error_msg
        }
