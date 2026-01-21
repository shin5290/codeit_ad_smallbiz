"""
광고 문구 생성 모듈 (v2.0.0)
industries.yaml 기반 업종별 최적화 프롬프트 지원

작성자: 배현석
버전: 2.0.0 (industries.yaml 연동)
"""
import os
from typing import Optional, Dict

from dotenv import load_dotenv
from openai import OpenAI

# PromptTemplateManager import (선택적, 모듈 의존성 분리 목적)
try:
    from src.generation.text_generation.prompt_manager import PromptTemplateManager
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    PROMPT_MANAGER_AVAILABLE = False

load_dotenv()


class TextGenerator:
    """
    광고 문구 생성 클래스 (v2.0.0)

    v2.0.0 변경사항:
    - industries.yaml 기반 업종별 최적화 프롬프트 지원
    - AIDA 프레임워크 적용
    - 추천 톤 자동 설정
    """

    def __init__(self, use_industry_config: bool = True):
        """
        초기화: OpenAI 클라이언트 및 PromptTemplateManager 설정

        Args:
            use_industry_config: industries.yaml 사용 여부 (기본: True)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # PromptTemplateManager 초기화 (선택적)
        self.use_industry_config = use_industry_config and PROMPT_MANAGER_AVAILABLE
        if self.use_industry_config:
            try:
                self.prompt_manager = PromptTemplateManager()
            except Exception as e:
                print(f"⚠️ PromptTemplateManager 초기화 실패: {e}")
                self.prompt_manager = None
                self.use_industry_config = False
        else:
            self.prompt_manager = None

    def generate_ad_copy(
        self,
        user_input: str,
        tone: str = "auto",
        max_length: int = 20,
        industry: Optional[str] = None
    ) -> str:
        """
        광고 문구 생성

        Args:
            user_input (str): 사용자 요청 텍스트
                예: "카페 신메뉴 홍보, 따뜻한 느낌, 겨울"
            tone (str): 톤 앤 매너
                - "auto": 업종에 맞는 톤 자동 선택 (v2.0.0 신규)
                - "warm", "professional", "friendly", "energetic", "practical", "respectful"
            max_length (int): 최대 글자 수 (기본 20자)
            industry (str): 업종 코드 (None이면 자동 감지)

        Returns:
            str: 생성된 광고 문구
                예: "따뜻한 겨울, 새로운 맛"
        """
        print("📝 광고 문구 생성 중...")
        print(f"   입력: {user_input}")

        # 업종 감지 및 톤 자동 설정 (v2.0.0)
        detected_industry = None
        if self.use_industry_config and self.prompt_manager:
            detected_industry = industry or self.prompt_manager.detect_industry(user_input)
            print(f"   감지된 업종: {detected_industry}")

            # tone이 "auto"이면 업종에 맞는 톤 자동 선택
            if tone == "auto":
                tone = self.prompt_manager.get_recommended_tone(detected_industry)
                print(f"   추천 톤 적용: {tone}")

        # tone이 여전히 "auto"이면 기본값 사용
        if tone == "auto":
            tone = "warm"

        print(f"   톤: {tone}, 최대 {max_length}자")

        try:
            # v2.0.0: 업종별 최적화 프롬프트 사용
            if self.use_industry_config and self.prompt_manager:
                prompts = self.prompt_manager.get_ad_copy_prompt(
                    user_input=user_input,
                    tone=tone,
                    max_length=max_length,
                    industry=detected_industry
                )
                system_prompt = prompts["system_prompt"]
                user_prompt = prompts["user_prompt"]
            else:
                # 레거시 방식
                system_prompt = self._get_system_prompt(tone, max_length)
                user_prompt = self._build_user_prompt(user_input, max_length)

            # GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )

            # 응답 추출
            ad_copy = response.choices[0].message.content.strip()

            # 후처리
            ad_copy = self._postprocess(ad_copy, max_length)

            print(f"✅ 생성 완료: {ad_copy}")
            return ad_copy

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return self._get_fallback_copy()

    def generate_ad_copy_with_info(
        self,
        user_input: str,
        tone: str = "auto",
        max_length: int = 20,
        industry: Optional[str] = None
    ) -> Dict:
        """
        광고 문구 생성 + 메타데이터 반환 (v2.0.0 신규)

        Args:
            user_input: 사용자 입력
            tone: 톤 ("auto"이면 자동 선택)
            max_length: 최대 글자 수
            industry: 업종 코드

        Returns:
            Dict: {
                "ad_copy": "생성된 광고 문구",
                "industry": "s1_hot_cooking",
                "tone": "warm",
                "examples": ["예시1", "예시2"]
            }
        """
        # 업종 감지
        detected_industry = None
        if self.use_industry_config and self.prompt_manager:
            detected_industry = industry or self.prompt_manager.detect_industry(user_input)

            if tone == "auto":
                tone = self.prompt_manager.get_recommended_tone(detected_industry)

        if tone == "auto":
            tone = "warm"

        # 광고 문구 생성
        ad_copy = self.generate_ad_copy(
            user_input=user_input,
            tone=tone,
            max_length=max_length,
            industry=detected_industry
        )

        # 예시 문구
        examples = []
        if self.use_industry_config and self.prompt_manager:
            info = self.prompt_manager.get_industry_info(detected_industry)
            examples = info.get("example_copies", [])

        return {
            "ad_copy": ad_copy,
            "industry": detected_industry,
            "tone": tone,
            "examples": examples
        }

    def _get_system_prompt(self, tone: str, max_length: int) -> str:
        """톤에 따른 시스템 프롬프트 반환 (레거시)"""

        base_prompt = f"""당신은 소상공인을 위한 전문 광고 카피라이터입니다.
짧고 임팩트 있는 광고 문구를 만들어주세요.

규칙:
- {max_length}자 이내 (공백 포함)
- 번호, 특수문자 없이 문구만 작성
- 사용자 별다른 요청 없을시 무조건 한국어로 작성
- 사용자 요청시 요청한 언어로 작성
- 광고 문구 1개만 생성"""

        tone_styles = {
            "warm": "따뜻하고 감성적인 톤으로 작성하세요. 편안하고 아늑한 느낌을 주세요.",
            "professional": "전문적이고 신뢰감 있는 톤으로 작성하세요. 격식 있고 세련된 느낌을 주세요.",
            "friendly": "친근하고 편안한 톤으로 작성하세요. 대화하듯 자연스러운 느낌을 주세요.",
            "energetic": "활기차고 역동적인 톤으로 작성하세요. 열정적이고 긍정적인 느낌을 주세요.",
            "practical": "실용적이고 명확한 톤으로 작성하세요. 구체적인 혜택을 강조하세요.",
            "respectful": "정중하고 신뢰감 있는 톤으로 작성하세요. 전문성을 강조하세요."
        }

        tone_guide = tone_styles.get(tone, tone_styles["warm"])

        return f"{base_prompt}\n\n톤 앤 매너:\n{tone_guide}"

    def _build_user_prompt(self, user_input: str, max_length: int) -> str:
        """사용자 프롬프트 구성 (레거시)"""

        return f"""다음 내용으로 광고 문구를 만들어주세요:

{user_input}

요구사항:
- {max_length}자 이내
- 광고 문구만 작성 (설명, 번호 등 불필요한 내용 제외)
- 감성적이면서도 명확한 메시지 전달

광고 문구:"""

    def _postprocess(self, text, max_length):
        """텍스트 후처리"""

        # 1. 불필요한 문자 제거
        text = text.replace("1. ", "").replace("2. ", "").replace("- ", "")
        text = text.replace('"', '').replace("'", "").replace('「', '').replace('」', '')
        text = text.strip()

        # 2. 길이 제한
        if len(text) > max_length:
            text = text[:max_length].strip()

        # 3. 빈 문자열 체크
        if not text:
            return self._get_fallback_copy()

        return text

    def _get_fallback_copy(self):
        """GPT 실패 시 기본 문구 반환"""
        return "특별한 순간을 함께하세요"


# ============================================
# 테스트 코드
# ============================================
# if __name__ == "__main__":
#     print("=" * 60)
#     print("📝 TextGenerator 테스트")
#     print("=" * 60)

#     generator = TextGenerator()

#     # 테스트 케이스들
#     test_cases = [
#         {
#             "input": "카페 신메뉴 홍보, 따뜻한 느낌, 겨울 시즌",
#             "tone": "warm"
#         },
#         {
#             "input": "식당 가족 모임 이벤트, 주말 특가",
#             "tone": "friendly"
#         },
#         {
#             "input": "헬스장 신규 회원 모집, 전문 트레이너",
#             "tone": "professional"
#         }
#     ]

#     for i, test in enumerate(test_cases, 1):
#         print(f"\n{'='*60}")
#         print(f"테스트 {i}")
#         print(f"{'='*60}")

#         result = generator.generate_ad_copy(
#             user_input=test["input"],
#             tone=test["tone"]
#         )

#         print(f"\n결과: '{result}'")
#         print(f"길이: {len(result)}자")

#     print(f"\n{'='*60}")
#     print("✅ 모든 테스트 완료!")
#     print(f"{'='*60}")
