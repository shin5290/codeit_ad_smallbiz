"""
광고 문구 생성 모듈
작성자: 배현석
버전: 1.0
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class TextGenerator:
    """광고 문구 생성 클래스"""
    
    def __init__(self):
        """초기화: OpenAI 클라이언트 설정"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def generate_ad_copy(self, user_input, tone="warm", max_length=100):
        """
        광고 문구 생성

        Args:
            user_input (str): 사용자 요청 텍스트
                예: "카페 신메뉴 홍보, 따뜻한 느낌, 겨울"
            tone (str): 톤 앤 매너 ("warm", "professional", "friendly")
            max_length (int): 최대 글자 수 (기본 20자)
        Returns:
            str: 생성된 광고 문구
                예: "따뜻한 겨울, 새로운 맛"
        """

        print(f"📝 광고 문구 생성 중...")
        print(f"   입력: {user_input}")
        print(f"   톤: {tone}, 최대 {max_length}자")
        try:
            # 1. 시스템 프롬프트 선택
            system_prompt = self._get_system_prompt(tone, max_length)

            # 2. 사용자 프롬프트 구성
            user_prompt = self._build_user_prompt(user_input, max_length)
            
            # 3. GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # 4. 응답 추출
            ad_copy = response.choices[0].message.content.strip()
            
            # 5. 후처리
            ad_copy = self._postprocess(ad_copy, max_length)
            
            print(f"✅ 생성 완료: {ad_copy}")
            return ad_copy
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return self._get_fallback_copy()
    
    def _get_system_prompt(self, tone, max_length):
        """톤에 따른 시스템 프롬프트 반환"""
        
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
            "energetic": "활기차고 역동적인 톤으로 작성하세요. 열정적이고 긍정적인 느낌을 주세요."
        }
        
        tone_guide = tone_styles.get(tone, tone_styles["warm"])
        
        return f"{base_prompt}\n\n톤 앤 매너:\n{tone_guide}"
    
    def _build_user_prompt(self, user_input, max_length):
        """사용자 프롬프트 구성"""

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


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("📝 TextGenerator 테스트")
    print("=" * 60)
    
    generator = TextGenerator()
    
    # 테스트 케이스들
    test_cases = [
        {
            "input": "카페 신메뉴 홍보, 따뜻한 느낌, 겨울 시즌",
            "tone": "warm"
        },
        {
            "input": "식당 가족 모임 이벤트, 주말 특가",
            "tone": "friendly"
        },
        {
            "input": "헬스장 신규 회원 모집, 전문 트레이너",
            "tone": "professional"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}")
        print(f"{'='*60}")
        
        result = generator.generate_ad_copy(
            user_input=test["input"],
            tone=test["tone"]
        )
        
        print(f"\n결과: '{result}'")
        print(f"길이: {len(result)}자")
    
    print(f"\n{'='*60}")
    print("✅ 모든 테스트 완료!")
    print(f"{'='*60}")
