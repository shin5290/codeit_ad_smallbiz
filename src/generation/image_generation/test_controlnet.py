"""
ControlNet 기능 테스트 스크립트

사용법:
    python test_controlnet.py

테스트 항목:
1. Canny ControlNet으로 제품 이미지 스타일 변환
2. 생성된 이미지 저장 및 검증
"""

from PIL import Image
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.image_generation.generator import generate_with_controlnet


def test_controlnet_basic():
    """
    기본 ControlNet 테스트

    테스트 이미지가 필요합니다:
    - src/generation/image_generation/test_images/product_sample.jpg
    """
    print("=" * 60)
    print("ControlNet 기본 테스트 시작")
    print("=" * 60)

    # 테스트 이미지 경로
    test_image_path = Path(__file__).parent / "test_images" / "product_sample.jpg"

    if not test_image_path.exists():
        print(f"[ERROR] Test image not found: {test_image_path}")
        print("\nPlease prepare test image:")
        print("1. Create directory: src/generation/image_generation/test_images/")
        print("2. Save product photo as product_sample.jpg")
        return

    # 테스트 이미지 로드
    print(f"\n[1/3] 테스트 이미지 로드: {test_image_path.name}")
    reference_image = Image.open(test_image_path)
    print(f"  - 이미지 크기: {reference_image.size}")
    print(f"  - 이미지 모드: {reference_image.mode}")

    # ControlNet으로 이미지 생성
    print("\n[2/3] ControlNet 이미지 생성 시작...")
    print("  - Control Type: canny (윤곽선 기반)")
    print("  - Style: ultra_realistic")
    print("  - Aspect Ratio: 1:1")

    # 소금빵 광고 이미지 프롬프트
    prompt = (
        "professional food photography of Korean salt bread roll, "
        "oval-shaped golden brown bread with white salt crystals on top, "
        "soft fluffy texture, glossy surface, fresh from bakery, "
        "warm lighting, wooden table, clean background, "
        "high quality detailed commercial photo, appetizing presentation"
    )

    print(f"\n사용할 프롬프트: {prompt}")

    result = generate_with_controlnet(
        prompt=prompt,
        reference_image=reference_image,
        control_type="canny",
        style="ultra_realistic",
        aspect_ratio="1:1",
        num_inference_steps=40,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        business_id="test",
    )

    # 결과 확인
    print("\n[3/3] 생성 결과:")
    if result["success"]:
        print(f"  ✅ 성공!")
        print(f"  - 저장 경로: {result['image_path']}")
        print(f"  - 상대 경로: {result['relative_path']}")
        print(f"  - 이미지 크기: {result['width']}x{result['height']}")
        print(f"  - 생성 시간: {result['generation_time']:.2f}초")
        print(f"  - ControlNet 타입: {result['control_type']}")
        print(f"  - ControlNet 강도: {result['controlnet_scale']}")
    else:
        print(f"  ❌ 실패!")
        print(f"  - 에러: {result['error']}")

    print("\n" + "=" * 60)


def test_controlnet_multi_style():
    """
    여러 스타일로 테스트
    """
    print("=" * 60)
    print("ControlNet 멀티 스타일 테스트")
    print("=" * 60)

    test_image_path = Path(__file__).parent / "test_images" / "product_sample.jpg"

    if not test_image_path.exists():
        print(f"❌ 테스트 이미지를 찾을 수 없습니다: {test_image_path}")
        return

    reference_image = Image.open(test_image_path)

    styles = ["ultra_realistic", "semi_realistic", "anime"]

    # 스타일별 프롬프트
    style_prompts = {
        "ultra_realistic": (
            "professional food photography of Korean salt bread roll, "
            "oval-shaped golden brown bread with white salt crystals on top, "
            "soft fluffy texture, glossy surface, fresh from bakery, "
            "warm lighting, wooden table, clean background, "
            "high quality detailed commercial photo, appetizing presentation"
        ),
        "semi_realistic": (
            "semi-realistic illustration of Korean salt bread roll, "
            "oval-shaped golden bread with white salt on top, "
            "soft texture, warm colors, wooden table background, "
            "artistic food illustration, painterly style, clean rendering"
        ),
        "anime": (
            "anime style food illustration of Korean salt bread, "
            "oval-shaped golden bread with sparkly white salt crystals, "
            "soft fluffy appearance, warm colors, wooden table, "
            "detailed anime art style, clean linework, vibrant appetizing colors"
        )
    }

    for i, style in enumerate(styles, 1):
        print(f"\n[{i}/{len(styles)}] {style} 스타일 테스트...")

        prompt = style_prompts.get(style, f"professional photo of salt bread, {style} style")
        print(f"  Prompt: {prompt[:80]}...")

        result = generate_with_controlnet(
            prompt=prompt,
            reference_image=reference_image,
            control_type="canny",
            style=style,
            aspect_ratio="1:1",
            num_inference_steps=30,  # 빠른 테스트를 위해 30 스텝
            business_id="test_multi",
        )

        if result["success"]:
            print(f"  ✅ {style}: {result['filename']} ({result['generation_time']:.2f}초)")
        else:
            print(f"  ❌ {style}: 실패")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nControlNet Test Script")
    print("=" * 60)

    # 테스트 선택
    print("\n테스트 옵션:")
    print("1. 기본 테스트 (Canny + Ultra Realistic)")
    print("2. 멀티 스타일 테스트 (3가지 스타일)")
    print("3. 모두 실행")

    choice = input("\n선택 (1-3): ").strip()

    if choice == "1":
        test_controlnet_basic()
    elif choice == "2":
        test_controlnet_multi_style()
    elif choice == "3":
        test_controlnet_basic()
        print("\n")
        test_controlnet_multi_style()
    else:
        print("[ERROR] Invalid choice.")

    print("\n[SUCCESS] Test completed!")
