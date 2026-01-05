"""
통합 테스트: TextGenerator + PromptTemplateManager
진수경님이 사용할 방식 시뮬레이션
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from text_generator import TextGenerator
from prompt_manager import PromptTemplateManager


def simulate_advertisement_creation(user_input):
    """
    광고 생성 프로세스 시뮬레이션
    (진수경님의 services.py에서 호출할 방식)
    """
    
    print("=" * 80)
    print(f"🎬 광고 생성 시작")
    print("=" * 80)
    print(f"📥 사용자 입력: {user_input}\n")
    
    try:
        # 1. 프롬프트 생성 (배현석 → 이현석)
        print("1️⃣ 이미지 프롬프트 생성 (배현석 파트)")
        print("-" * 80)
        pm = PromptTemplateManager()
        image_prompts = pm.generate_image_prompt(user_input, style="realistic")
        print(f"✅ 완료: Positive {len(image_prompts['positive'].split(','))}개 태그\n")
        print(f"✅ 완료: Negative 프롬프트 생성\n")
        
        # 2. 광고 문구 생성 (배현석)
        print("2️⃣ 광고 문구 생성 (배현석 파트)")
        print("-" * 80)
        tg = TextGenerator()
        ad_copy = tg.generate_ad_copy(user_input, tone="warm", max_length=20)
        print(f"✅ 완료: {len(ad_copy)}자 생성\n")
        
        # 3. 결과 통합 (진수경이 할 부분)
        print("3️⃣ 결과 통합 (진수경 파트)")
        print("-" * 80)
        result = {
            "positive_prompt": image_prompts["positive"],
            "negative_prompt": image_prompts["negative"],
            "ad_copy": ad_copy,
            "status": "success"
        }
        
        # 4. 최종 결과 출력
        print("=" * 80)
        print("🎉 광고 생성 완료!")
        print("=" * 80)
        print(f"\n📸 이미지 Positive 프롬프트:")
        print(f"   {image_prompts['positive']}\n")
        print(f"🚫 이미지 Negative 프롬프트:")
        print(f"   {image_prompts['negative']}\n")
        print(f"📝 광고 문구:")
        print(f"   '{ad_copy}' ({len(ad_copy)}자)\n")
        
        # 5. 진수경님한테 전달할 형태
        print("=" * 80)
        print("📦 진수경님한테 전달할 데이터 (JSON 형태)")
        print("=" * 80)
        print(f"""{{
    "positive_prompt": "{image_prompts['positive'][:50]}...",
    "negative_prompt": "{image_prompts['negative'][:50]}...",
    "ad_copy": "{ad_copy}",
    "status": "success"
}}""")
        
        return result
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return {
            "image_prompt": None,
            "ad_copy": None,
            "status": "failed",
            "error": str(e)
        }


def main():
    """메인 테스트"""
    
    print("\n" + "=" * 80)
    print("🧪 통합 테스트: 진수경님 사용 시나리오")
    print("=" * 80 + "\n")
    
    # 테스트 케이스들
    test_cases = [
        "카페 신메뉴 홍보, 따뜻한 느낌, 겨울 시즌",
        "식당 가족 모임 이벤트, 편안한 분위기",
        "헬스장 신규 회원 모집, 현대적인 시설"
    ]
    
    results = []
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"테스트 케이스 {i}/3")
        print(f"{'='*80}\n")
        
        result = simulate_advertisement_creation(user_input)
        results.append(result)
        
        print("\n" + "⏸️  " * 20 + "\n")
    
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
        print("\n✅ 진수경님과 통합 준비 완료!")
        print("✅ 이현석님한테 프롬프트 전달 가능!")
    else:
        print("⚠️  일부 테스트 실패")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
