#!/usr/bin/env python3
"""
터미널 테스트 스크립트
GCP 상에서 터미널로 이미지 생성 테스트

사용법:
    python -m src.generation.image_generation.test_generate

    또는 직접 실행:
    cd /path/to/codeit_ad_smallbiz
    python src/generation/image_generation/test_generate.py
"""

import sys
import os

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# DATABASE_URL이 없으면 더미값 설정 (테스트용)
if not os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from src.generation.image_generation.generator import generate_and_save_image


def print_banner():
    print("\n" + "=" * 60)
    print("   이미지 생성 테스트 (Image Generation Test)")
    print("=" * 60)
    print("\n스타일 옵션:")
    print("  1. ultra_realistic (실사)")
    print("  2. semi_realistic (반실사)")
    print("  3. anime (애니메이션)")
    print("\n비율 옵션:")
    print("  1:1, 3:4, 4:3, 16:9, 9:16")
    print("\n종료하려면 'q' 또는 'quit' 입력")
    print("=" * 60 + "\n")


def get_user_input():
    """사용자 입력 받기"""
    print("\n" + "-" * 40)

    # 한글 입력
    user_input = input("🎨 한글 입력 (예: 카페 신메뉴 딸기라떼 홍보): ").strip()
    if user_input.lower() in ['q', 'quit', 'exit']:
        return None

    if not user_input:
        print("⚠️  입력이 비어있습니다.")
        return get_user_input()

    # 스타일 선택
    print("\n스타일 선택 (1=ultra_realistic, 2=semi_realistic, 3=anime)")
    style_input = input("스타일 [1]: ").strip() or "1"
    style_map = {"1": "ultra_realistic", "2": "semi_realistic", "3": "anime"}
    style = style_map.get(style_input, "ultra_realistic")

    # 비율 선택
    print("\n비율 선택 (1:1, 3:4, 4:3, 16:9, 9:16)")
    aspect_ratio = input("비율 [1:1]: ").strip() or "1:1"
    if aspect_ratio not in ["1:1", "3:4", "4:3", "16:9", "9:16"]:
        aspect_ratio = "1:1"

    return {
        "user_input": user_input,
        "style": style,
        "aspect_ratio": aspect_ratio
    }


def run_generation(params):
    """이미지 생성 실행"""
    print("\n" + "=" * 40)
    print("🚀 이미지 생성 시작...")
    print(f"   입력: {params['user_input']}")
    print(f"   스타일: {params['style']}")
    print(f"   비율: {params['aspect_ratio']}")
    print("=" * 40 + "\n")

    result = generate_and_save_image(
        user_input=params["user_input"],
        style=params["style"],
        aspect_ratio=params["aspect_ratio"],
        num_inference_steps=30,  # 테스트용으로 빠르게
        guidance_scale=7.5
    )

    print("\n" + "=" * 40)
    if result["success"]:
        print("✅ 생성 성공!")
        print(f"   경로: {result['image_path']}")
        print(f"   크기: {result['width']}x{result['height']}")
        print(f"   스타일: {result['style']}")
        print(f"   시간: {result['generation_time']:.2f}초")
    else:
        print("❌ 생성 실패!")
        print(f"   에러: {result['error'][:200]}...")
    print("=" * 40)

    return result


def main():
    """메인 루프"""
    print_banner()

    while True:
        try:
            params = get_user_input()

            if params is None:
                print("\n👋 테스트 종료")
                break

            run_generation(params)

            # 계속 여부
            cont = input("\n계속하시겠습니까? (Enter=계속, q=종료): ").strip()
            if cont.lower() in ['q', 'quit', 'exit']:
                print("\n👋 테스트 종료")
                break

        except KeyboardInterrupt:
            print("\n\n👋 테스트 종료 (Ctrl+C)")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
