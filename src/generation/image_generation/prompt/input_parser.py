"""
Input Parser - 한글 입력 → 영어 키워드 추출 모듈

GPT API를 활용하여 사용자의 한글 광고 문구에서
이미지 생성에 필요한 영어 키워드를 추출합니다.
GPT_API_KEY 필수

사용 예시:
    from src.generation.image_generation.prompt.input_parser import InputParser

    parser = InputParser(api_key="your-api-key")
    keywords = parser.parse("삼겹살 맛집 홍보, 불판에서 지글지글 굽는 모습")
    # 결과: {
    #     "product": "grilled pork belly",
    #     "setting": "Korean BBQ restaurant",
    #     "mood": "appetizing",
    #     "action": "sizzling on hot grill plate",
    #     "details": "charcoal grill marks, juicy fat rendering",
    #     "style": "realistic"
    # }
"""

import json
from typing import Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class InputParser:
    """
    한글 입력을 영어 프롬프트 요소로 변환

    GPT API를 사용하여 한글 광고 문구에서 이미지 생성에 필요한
    영어 키워드를 추출합니다.
    """

    SYSTEM_PROMPT = """You are an expert at extracting visual elements from Korean advertising text for AI image generation.

Given a Korean ad description, extract these elements in English:
- product: Main product/service to be photographed (e.g., "grilled pork belly", "strawberry latte", "hair styling result")
- setting: Location/environment for the photo (e.g., "Korean BBQ restaurant with warm ambiance", "cozy cafe interior")
- mood: Atmosphere/feeling (e.g., "warm", "appetizing", "cozy", "professional", "energetic")
- action: Key visual action or state (e.g., "sizzling on hot grill plate", "steam rising", "being styled")
- details: Specific visual details to include (e.g., "charcoal grill marks, juicy texture", "creamy foam art")

IMPORTANT RULES:
1. Return ONLY a valid JSON object with these exact keys: product, setting, mood, action, details
2. Use descriptive English terms optimized for photorealistic image generation
3. Focus on VISUAL elements that can be photographed
4. Keep each value concise but descriptive (5-15 words max)
5. If information is not available, use empty string ""
6. Do NOT include any Korean text in the output

Example input: "삼겹살 맛집 홍보, 불판에서 지글지글 굽는 모습"
Example output:
{
    "product": "thick sliced pork belly",
    "setting": "Korean BBQ restaurant with warm wooden interior",
    "mood": "appetizing and inviting",
    "action": "sizzling on hot charcoal grill plate with smoke rising",
    "details": "glistening fat, charcoal grill marks, golden brown color"
}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 로드)
            model: 사용할 GPT 모델 (기본값: gpt-4o-mini - 비용 효율적)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai 패키지가 설치되지 않았습니다. "
                "'pip install openai'로 설치해주세요."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def parse(
        self,
        korean_input: str,
        detected_industry: Optional[str] = None,
        style: str = "realistic"
    ) -> Dict[str, str]:
        """
        한글 입력에서 영어 키워드 추출

        Args:
            korean_input: 한글 광고 문구
            detected_industry: 감지된 업종 코드 (힌트용, 선택)
            style: 이미지 스타일 (기본값: realistic)

        Returns:
            Dict: {
                "product": "grilled pork belly",
                "setting": "Korean BBQ restaurant",
                "mood": "warm and appetizing",
                "action": "sizzling on hot plate",
                "details": "charcoal grill marks, juicy texture",
                "style": "realistic"
            }
        """
        # 사용자 프롬프트 구성
        user_prompt = f"Korean ad text: {korean_input}"
        if detected_industry:
            user_prompt += f"\nDetected industry category: {detected_industry}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # 일관된 결과를 위해 낮은 temperature
                max_tokens=500
            )

            # JSON 파싱
            result = json.loads(response.choices[0].message.content)

            # 필수 키 확인 및 기본값 설정
            required_keys = ["product", "setting", "mood", "action", "details"]
            for key in required_keys:
                if key not in result:
                    result[key] = ""

            # 스타일 추가
            result["style"] = style

            return result

        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 기본 구조 반환
            return self._fallback_parse(korean_input, style)
        except Exception as e:
            # API 호출 실패 시 기본 구조 반환
            print(f"⚠️ GPT API 호출 실패: {e}")
            return self._fallback_parse(korean_input, style)

    def _fallback_parse(self, korean_input: str, style: str) -> Dict[str, str]:
        """
        API 실패 시 기본 파싱 (한글 그대로 반환 - 권장하지 않음)

        실제 사용 시에는 이 fallback이 호출되지 않도록
        API 키와 네트워크 연결을 확인하세요.
        """
        return {
            "product": "product",  # 기본값
            "setting": "",
            "mood": "",
            "action": "",
            "details": "",
            "style": style,
            "_warning": "GPT API 호출 실패로 기본값 사용됨"
        }

    def parse_batch(
        self,
        korean_inputs: list,
        detected_industries: Optional[list] = None,
        style: str = "realistic"
    ) -> list:
        """
        여러 한글 입력을 일괄 처리

        Args:
            korean_inputs: 한글 광고 문구 리스트
            detected_industries: 감지된 업종 코드 리스트 (선택)
            style: 이미지 스타일

        Returns:
            list: 추출된 키워드 딕셔너리 리스트
        """
        results = []
        industries = detected_industries or [None] * len(korean_inputs)

        for korean_input, industry in zip(korean_inputs, industries):
            result = self.parse(korean_input, industry, style)
            results.append(result)

        return results


# ============================================
# 테스트 코드
# ============================================

# if __name__ == "__main__":
#     # 테스트용 예시
#     test_inputs = [
#         "삼겹살 맛집 홍보, 불판에서 지글지글 굽는 모습",
#         "카페 봄 시즌 딸기 음료 출시 홍보",
#         "미용실 봄 시즌 헤어컬러 이벤트",
#         "헬스장 PT 프로그램 홍보, 운동하는 모습",
#     ]

#     print("=== InputParserLight 테스트 (GPT API 없이) ===")
#     light_parser = InputParserLight()

#     for test_input in test_inputs:
#         result = light_parser.parse(test_input)
#         print(f"\n입력: {test_input}")
#         print(f"결과: {result}")

#     print("\n" + "="*50)
#     print("GPT API를 사용하려면:")
#     print("  parser = InputParser(api_key='your-api-key')")
#     print("  result = parser.parse('한글 입력')")
