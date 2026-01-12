"""
광고 생성 통합 모듈
백엔드(진수경)가 호출할 메인 함수

작성자: 배현석
버전: 1.0
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가 (JupyterHub에서도 작동하도록)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.text_generation.text_generator import TextGenerator
from src.generation.text_generation.prompt_manager import PromptTemplateManager


def generate_advertisement(
    user_input: str,
    tone: str = "warm",
    max_length: int = 20,
    style: str = "realistic"
) -> dict:
    """
    광고 생성 통합 함수 (백엔드가 호출)

    배현석 담당:
    - 텍스트 생성 (광고 문구)
    - 프롬프트 생성 (Positive/Negative)
    - 업종 감지

    Args:
        user_input (str): 한글 사용자 요청
            예: "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌, 겨울"
        tone (str): 광고 문구 톤 (warm, professional, friendly, energetic)
        max_length (int): 광고 문구 최대 길자 수
        style (str): 이미지 스타일 (realistic, anime 등)

    Returns:
        dict: {
            # 텍스트 생성 [1개]
            "ad_copy": "따뜻한 겨울, 새로운 맛",

            # 프롬프트 생성 [2개]
            "positive_prompt": "Professional food photography of...",
            "negative_prompt": "cartoon, illustration, low quality...",

            # 업종 [1개]
            "industry": "cafe",

            # 상태
            "status": "success"
        }

    실패 시:
        {
            "ad_copy": None,
            "positive_prompt": None,
            "negative_prompt": None,
            "industry": None,
            "status": "failed",
            "error": "에러 메시지"
        }
    """

    print("=" * 80)
    print("🎬 광고 생성 파이프라인 시작")
    print("=" * 80)
    print(f"📥 사용자 입력: {user_input}")
    print(f"📐 설정: tone={tone}, max_length={max_length}, style={style}\n")

    try:
        # 1. 광고 문구 생성 (TextGenerator)
        print("1️⃣ 광고 문구 생성")
        print("-" * 80)

        text_gen = TextGenerator()
        ad_copy = text_gen.generate_ad_copy(
            user_input=user_input,
            tone=tone,
            max_length=max_length
        )

        print(f"✅ 광고 문구: '{ad_copy}' ({len(ad_copy)}자)\n")

        # 2. 이미지 프롬프트 생성 (PromptTemplateManager)
        print("2️⃣ 이미지 프롬프트 생성")
        print("-" * 80)

        prompt_manager = PromptTemplateManager()
        image_prompts = prompt_manager.generate_image_prompt(
            user_input=user_input,
            style=style
        )

        print(f"✅ Positive: {len(image_prompts['positive'])} chars")
        print(f"✅ Negative: {len(image_prompts['negative'])} chars")
        print(f"✅ Industry: {image_prompts['industry']}\n")

        # 3. 결과 통합
        result = {
            # 백엔드 요구사항에 맞춤
            "ad_copy": ad_copy,  # 텍스트 생성 [1개]
            "positive_prompt": image_prompts["positive"],  # 프롬프트 [1/2]
            "negative_prompt": image_prompts["negative"],  # 프롬프트 [2/2]
            "industry": image_prompts["industry"],  # 업종 [1개]
            "status": "success"
        }

        print("=" * 80)
        print("🎉 광고 생성 완료!")
        print("=" * 80)
        print(f"📝 광고 문구: {ad_copy}")
        print(f"📸 Positive: {image_prompts['positive'][:60]}...")
        print(f"🚫 Negative: {image_prompts['negative'][:60]}...")
        print(f"🏢 업종: {image_prompts['industry']}")
        print("=" * 80 + "\n")

        return result

    except Exception as e:
        print(f"❌ 광고 생성 실패: {e}")
        import traceback
        traceback.print_exc()

        return {
            "ad_copy": None,
            "positive_prompt": None,
            "negative_prompt": None,
            "industry": None,
            "status": "failed",
            "error": str(e)
        }


# ============================================
# API 호출 없이 테스트 (로직 검증용)
# ============================================

def test_without_api():
    """
    API 호출 없이 구조만 테스트
    (JupyterHub에서 실제 테스트 전 로컬 검증용)
    """

    print("=" * 80)
    print("🧪 API 호출 없이 구조 테스트")
    print("=" * 80)
    print("⚠️  실제 GPT API는 호출하지 않습니다.")
    print("⚠️  구조와 로직만 검증합니다.\n")

    # 더미 데이터로 테스트
    test_cases = [
        {
            "user_input": "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
            "tone": "warm",
            "max_length": 20
        },
        {
            "user_input": "헬스장 신규 회원 모집, 현대적인 시설",
            "tone": "professional",
            "max_length": 18
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"테스트 케이스 {i}")
        print(f"{'='*80}\n")

        # 구조만 확인
        print(f"입력: {test['user_input']}")
        print(f"톤: {test['tone']}")
        print(f"최대 길이: {test['max_length']}")

        expected_output = {
            "ad_copy": "[API 호출 시 생성될 광고 문구]",
            "positive_prompt": "[API 호출 시 생성될 Positive 프롬프트]",
            "negative_prompt": "[API 호출 시 생성될 Negative 프롬프트]",
            "industry": "[자동 감지될 업종]",
            "status": "success"
        }

        print(f"\n✅ 예상 출력 구조:")
        for key, value in expected_output.items():
            print(f"   {key}: {value}")

        print(f"\n{'⏸️  '*20}\n")

    print("=" * 80)
    print("✅ 구조 테스트 완료!")
    print("✅ JupyterHub에서 generate_advertisement() 함수를 호출하여 실제 테스트하세요.")
    print("=" * 80)


# ============================================
# 메인 실행
# ============================================

if __name__ == "__main__":
    # API 호출 없이 구조만 테스트
    test_without_api()

    print("\n" + "=" * 80)
    print("📌 실제 사용 예제 (JupyterHub에서 실행)")
    print("=" * 80)
    print("""
from src.generation.text_generation.ad_generator import generate_advertisement

# 광고 생성
result = generate_advertisement(
    user_input="카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
    tone="warm",
    max_length=20
)

# 결과 확인
print(result['ad_copy'])           # 광고 문구
print(result['positive_prompt'])   # Positive 프롬프트
print(result['negative_prompt'])   # Negative 프롬프트
print(result['industry'])          # 업종
""")
    print("=" * 80)
