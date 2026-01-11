"""
통합 테스트: ad_generator.py
백엔드(진수경) 요구사항에 맞춘 출력 포맷 검증

백엔드 요구사항:
- 텍스트 생성: ad_copy [1개]
- 프롬프트 생성: positive_prompt, negative_prompt, industry [3개]
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ad_generator import generate_advertisement


def test_backend_integration():
    """
    백엔드 통합 테스트
    진수경님이 services.py에서 호출할 방식 시뮬레이션
    """

    print("\n" + "=" * 80)
    print("🧪 백엔드 통합 테스트")
    print("=" * 80)
    print("📋 백엔드 요구사항:")
    print("   - 텍스트 생성: ad_copy [1개]")
    print("   - 프롬프트 생성: positive_prompt, negative_prompt [2개]")
    print("   - 업종: industry [1개]")
    print("=" * 80 + "\n")

    # 테스트 케이스들
    test_cases = [
        {
            "name": "카페 - 딸기라떼",
            "user_input": "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌, 겨울",
            "tone": "warm",
            "max_length": 20
        },
        {
            "name": "헬스장 - 신규 회원",
            "user_input": "헬스장 신규 회원 모집, 현대적인 시설, 전문 트레이너",
            "tone": "professional",
            "max_length": 18
        },
        {
            "name": "베이커리 - 크루아상",
            "user_input": "베이커리 갓 구운 크루아상 홍보, 따뜻한 아침",
            "tone": "friendly",
            "max_length": 15
        }
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"테스트 케이스 {i}/{len(test_cases)}: {test['name']}")
        print(f"{'='*80}\n")

        # 광고 생성 (배현석 파트)
        result = generate_advertisement(
            user_input=test["user_input"],
            tone=test["tone"],
            max_length=test["max_length"]
        )

        results.append(result)

        # 백엔드 포맷 검증
        print(f"\n{'='*80}")
        print(f"📦 백엔드에 전달할 데이터")
        print(f"{'='*80}")

        if result["status"] == "success":
            print(f"""
✅ 성공!

{{
    "ad_copy": "{result['ad_copy']}",
    "positive_prompt": "{result['positive_prompt'][:60]}...",
    "negative_prompt": "{result['negative_prompt'][:60]}...",
    "industry": "{result['industry']}",
    "status": "success"
}}

📊 상세 정보:
   - 광고 문구: {len(result['ad_copy'])}자
   - Positive 프롬프트: {len(result['positive_prompt'])} chars
   - Negative 프롬프트: {len(result['negative_prompt'])} chars
   - 감지된 업종: {result['industry']}
""")
        else:
            print(f"❌ 실패: {result.get('error', 'Unknown error')}")

        print(f"\n{'⏸️  '*20}\n")

    # 최종 요약
    print("=" * 80)
    print("📊 테스트 요약")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")

    print(f"\n총 테스트: {len(test_cases)}개")
    print(f"성공: {success_count}개")
    print(f"실패: {len(test_cases) - success_count}개\n")

    if success_count == len(test_cases):
        print("🎉 모든 테스트 통과!")
        print("\n✅ 백엔드 통합 준비 완료!")
        print("✅ 진수경님이 services.py에서 generate_advertisement() 호출 가능!")
        print("✅ 이현석님한테 프롬프트 전달 준비 완료!")
    else:
        print("⚠️  일부 테스트 실패")

    print("=" * 80)

    return results


def show_usage_example():
    """
    사용법 예제 출력
    """

    print("\n" + "=" * 80)
    print("📌 백엔드(진수경님) 사용 예제")
    print("=" * 80)
    print("""
# services.py에서 호출 방법

from src.generation.text_generation.ad_generator import generate_advertisement

def create_advertisement(user_input: str):
    '''광고 생성 API 엔드포인트'''

    # 배현석 파트 호출
    result = generate_advertisement(
        user_input=user_input,
        tone="warm",           # optional (기본값: "warm")
        max_length=20,         # optional (기본값: 20)
        style="realistic"      # optional (기본값: "realistic")
    )

    if result["status"] == "success":
        # 성공 시 처리
        ad_copy = result["ad_copy"]              # 광고 문구
        positive_prompt = result["positive_prompt"]  # 이미지 생성용
        negative_prompt = result["negative_prompt"]  # 이미지 생성용
        industry = result["industry"]            # 감지된 업종

        # 이현석님한테 프롬프트 전달
        image_result = generate_image(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt
        )

        # 최종 광고 결과 반환
        return {
            "ad_copy": ad_copy,
            "image_path": image_result["path"],
            "industry": industry
        }
    else:
        # 실패 시 처리
        return {"error": result["error"]}
""")
    print("=" * 80)


if __name__ == "__main__":
    # 통합 테스트 실행
    results = test_backend_integration()

    # 사용법 예제 출력
    show_usage_example()
