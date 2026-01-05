"""
이미지 생성 프롬프트 관리 모듈
작성자: 배현석
버전: 1.1
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class PromptTemplateManager:
    """이미지 생성용 프롬프트 관리 클래스"""
    
    def __init__(self):
        """초기화: OpenAI 클라이언트 설정"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def generate_image_prompt(self, user_input, style="realistic"):
        """
        이미지 생성용 프롬프트(태그) 생성
        
        Args:
            user_input (str): 사용자 요청
                예: "카페 신메뉴 홍보, 따뜻한 느낌, 겨울"
            style (str): 이미지 스타일 ("realistic", "illustration", "minimal")
        
        Returns:
            dict: {"positive": str, "negative": str}
                예: {
                    "positive": "cafe interior, new menu board, warm lighting, ...",
                    "negative": "low quality, blurry, text, ..."
                }
        """
        
        print(f"🎨 이미지 프롬프트 생성 중...")
        print(f"   입력: {user_input}")
        print(f"   스타일: {style}")
        
        try:
            # 1. 시스템 프롬프트
            system_prompt = self._get_system_prompt(style)
            
            # 2. 사용자 프롬프트
            user_prompt = self._build_user_prompt(user_input)
            
            # 3. GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            
            # 4. 응답 추출
            prompt = response.choices[0].message.content.strip()
            
            # 5. 후처리
            positive_prompt = self._postprocess(prompt, style)
            
            # 6. Negative 프롬프트 생성
            negative_prompt = self._get_negative_prompt(style)
            
            print(f"✅ 생성 완료")
            print(f"   Positive: {positive_prompt[:60]}...")
            print(f"   Negative: {negative_prompt[:60]}...")
            
            return {
                "positive": positive_prompt,
                "negative": negative_prompt
            }
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return {
                "positive": self._get_fallback_prompt(style),
                "negative": self._get_negative_prompt(style)
            }
    
    def _get_system_prompt(self, style):
        """스타일에 따른 시스템 프롬프트"""
        
        base_prompt = """You are an expert in creating image generation prompts for Stable Diffusion.
Convert Korean user input into English tags that AI image generators can understand.

CRITICAL RULES:
1. Output ONLY English tags
2. Separate tags with commas
3. Focus on VISUAL elements only (no abstract concepts)
4. Maximum 20 tags
5. Include: subject, setting, atmosphere, lighting, style, quality

Output format example:
cafe interior, new menu board, warm lighting, cozy atmosphere, winter season, coffee cups, wooden table, soft focus, professional photography, high quality"""

        style_guides = {
            "realistic": """
Style focus: Photorealistic, professional photography
Include: natural lighting, detailed textures, realistic colors, sharp focus
Avoid: cartoon, anime, illustration, painting""",
            
            "illustration": """
Style focus: Hand-drawn, artistic illustration
Include: soft colors, artistic style, illustrated, painted, creative
Avoid: photorealistic, photograph, 3D render""",
            
            "minimal": """
Style focus: Clean, simple, minimalist design
Include: minimal, clean, simple, white background, modern, elegant
Avoid: cluttered, busy, complex, detailed"""
        }
        
        style_guide = style_guides.get(style, style_guides["realistic"])
        
        return f"{base_prompt}\n\n{style_guide}"
    
    def _build_user_prompt(self, user_input):
        """사용자 프롬프트 구성"""
        
        return f"""Convert this Korean description into English image generation tags:

{user_input}

Remember:
- ONLY English tags
- Comma-separated
- Visual elements only
- 20 tags maximum

Tags:"""
    
    def _postprocess(self, prompt, style):
        """프롬프트 후처리"""
        
        # 1. 한글 제거 (혹시 있다면)
        prompt = ''.join(char for char in prompt if ord(char) < 0x3131 or ord(char) > 0x318e)
        prompt = ''.join(char for char in prompt if ord(char) < 0xac00 or ord(char) > 0xd7a3)
        
        # 2. 불필요한 문자 정리
        prompt = prompt.replace('"', '').replace("'", "").strip()
        
        # 3. 품질 태그 추가
        quality_tags = self._get_quality_tags(style)
        
        # 이미 품질 태그가 있는지 확인
        if "high quality" not in prompt.lower():
            prompt = f"{prompt}, {quality_tags}"
        
        # 4. 중복 제거
        tags = [tag.strip() for tag in prompt.split(',')]
        unique_tags = []
        seen = set()
        
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen and tag:
                unique_tags.append(tag)
                seen.add(tag_lower)
        
        # 5. 20개 제한
        if len(unique_tags) > 20:
            unique_tags = unique_tags[:20]
        
        return ', '.join(unique_tags)
    
    def _get_quality_tags(self, style):
        """스타일별 품질 태그"""
        
        quality_tags = {
            "realistic": "high quality, detailed, professional photography, sharp focus, 4k",
            "illustration": "high quality, detailed artwork, professional illustration, artistic",
            "minimal": "high quality, clean design, professional, elegant, modern"
        }
        
        return quality_tags.get(style, quality_tags["realistic"])
    
    def _get_fallback_prompt(self, style):
        """GPT 실패 시 기본 프롬프트"""
        
        fallback = {
            "realistic": "professional photography, high quality, detailed, sharp focus, natural lighting",
            "illustration": "artistic illustration, hand-drawn style, colorful, creative, high quality",
            "minimal": "minimal design, clean, simple, modern, elegant, white background"
        }
        
        return fallback.get(style, fallback["realistic"])
    
    def _get_negative_prompt(self, style):
        """스타일별 Negative 프롬프트 생성"""
        
        # 모든 스타일 공통 negative
        base_negative = "low quality, blurry, text, watermark, bad anatomy, distorted, deformed"
        
        # 스타일별 추가 negative
        style_negatives = {
            "realistic": ", cartoon, anime, illustration, painting, drawing, sketch, 3d render",
            "illustration": ", photorealistic, photograph, photo, realistic, 3d render, cgi",
            "minimal": ", cluttered, busy, complex, detailed background, ornate, messy, crowded"
        }
        
        additional = style_negatives.get(style, "")
        
        return base_negative + additional


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("🎨 PromptTemplateManager 테스트")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    # 테스트 케이스들
    test_cases = [
        {
            "input": "카페 신메뉴 홍보, 따뜻한 느낌, 겨울 시즌, 라떼 아트",
            "style": "realistic"
        },
        {
            "input": "식당 가족 모임, 편안한 분위기, 한식",
            "style": "realistic"
        },
        {
            "input": "헬스장 홍보, 현대적인 시설, 운동 기구",
            "style": "minimal"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"테스트 {i}")
        print(f"{'='*80}")
        
        result = manager.generate_image_prompt(
            user_input=test["input"],
            style=test["style"]
        )
        
        # 검증
        positive_tags = result["positive"].split(',')
        has_korean = any(
            '\uac00' <= char <= '\ud7a3' or '\u3131' <= char <= '\u318e' 
            for char in result["positive"]
        )
        
        print(f"\n📊 검증 결과:")
        print(f"   ✅ Positive 태그: {len(positive_tags)}개")
        print(f"   ✅ 한글 포함: {'❌ 있음' if has_korean else '✅ 없음'}")
        print(f"   ✅ Positive 프롬프트:\n   {result['positive']}")
        print(f"   🚫 Negative 프롬프트:\n   {result['negative']}")
    
    print(f"\n{'='*80}")
    print("✅ 모든 테스트 완료!")
    print(f"{'='*80}")