"""
키워드 추출 모듈 (GPT-4o 기반)
작성자: 배현석
버전: 4.0 - 키워드 추출 전용

역할: 한글 사용자 입력 → 영어 키워드 추출
이후 prompt_templates.py에서 최종 프롬프트 생성
"""

import sys
import io

# UTF-8 인코딩 강제 설정
if sys.platform == 'win32':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from config_loader import industry_config

load_dotenv()


class PromptTemplateManager:
    """한글 입력 → 영어 키워드 추출 (GPT-4o)"""
    
    def __init__(self):
        """초기화: OpenAI 클라이언트 설정"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def extract_keywords_english(self, user_input: str) -> dict:
        """
        한글 사용자 입력 → 영어 키워드 추출
        
        Args:
            user_input (str): 한글 사용자 요청
                예: "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌"
        
        Returns:
            dict: 영어 키워드
                예: {
                    "product": "strawberry latte",
                    "activity": "promotion", 
                    "theme": "warm",
                    "mood": "cozy"
                }
        """
        
        print(f"🔍 키워드 추출 중...")
        print(f"   입력: {user_input}")
        
        try:
            # 1. 업종 자동 감지
            industry = self._detect_industry(user_input)
            print(f"   감지된 업종: {industry}")
            
            # 2. 시스템 프롬프트 (키워드 추출용)
            system_prompt = self._get_system_prompt_for_extraction(industry)
            
            # 3. 사용자 프롬프트
            user_prompt = self._build_user_prompt_for_extraction(user_input, industry)
            
            # 4. GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 일관성 최대
                max_tokens=150    # 짧은 JSON만
            )
            
            # 5. 응답 추출
            result = response.choices[0].message.content.strip()
            
            # 6. JSON 파싱 (```json``` 제거)
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            keywords = json.loads(result)
            
            print(f"✅ 추출 완료: {keywords}")
            
            return keywords
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: 빈 딕셔너리 반환
            return {}
    
    def _detect_industry(self, user_input: str) -> str:
        """
        사용자 입력에서 업종 자동 감지 (YAML 기반)
        
        Args:
            user_input: 사용자 입력 텍스트
        
        Returns:
            str: 감지된 업종 ("cafe", "gym", ...) 또는 "general"
        """
        if industry_config is None:
            return "general"
        
        return industry_config.detect_industry(user_input)
    
    def _get_system_prompt_for_extraction(self, industry: str) -> str:
        """
        키워드 추출용 시스템 프롬프트
        
        구조: Base (공통) + Specialized (업종별 특화)
        
        Args:
            industry: 감지된 업종
        
        Returns:
            str: 시스템 프롬프트
        """
        
        # ====================================================================
        # Base Prompt (모든 업종 공통)
        # ====================================================================
        base_prompt = """You are a keyword extraction expert for image generation prompts.

Your task: Extract keywords from Korean user input and translate them to English.

CRITICAL RULES:
1. Output ONLY English keywords (NEVER Korean characters - 절대 한글 금지!)
2. Extract visual elements only (no abstract marketing concepts)
3. Translate product/service names accurately
4. Output ONLY valid JSON format
5. Be specific with names (not generic terms)

COMMON FIELDS (extract if present in input):
- product/item/dish: Main product/item name (구체적으로!)
- activity/service: Action or service being performed
- person_type: Subject person (if person involved)
- state: Condition (fresh, warm, cold, clean, etc)
- presentation: Display method (on board, in glass, etc)
- surface: Surface type (marble table, wooden counter, etc)
- theme: Overall mood (warm, minimal, cozy, etc)
- mood: Atmosphere (energetic, calm, professional, etc)
- time: Time of day (morning, afternoon, evening)
- focus: What to emphasize (texture, color, etc)

Output format example:
{
  "product": "strawberry latte",
  "activity": "promotion",
  "theme": "warm",
  "surface": "marble table"
}

IMPORTANT:
- Only include fields that are clearly mentioned in input
- Translate ALL Korean to English
- Use simple, descriptive English words
- Do NOT include marketing language (translate core meaning only)"""

        # ====================================================================
        # Specialized Guides (복잡한 업종만)
        # ====================================================================
        specialized_guides = {
            "cafe": """

CAFE SPECIALIZATION:
- product: Exact beverage name (예: "strawberry latte", "iced americano", "cappuccino")
  ⚠️  NOT generic: "beverage", "drink" (too vague!)
- Common states: "iced", "hot", "fresh"
- Common presentations: "in tall glass", "with latte art", "topped with cream"
- Common surfaces: "marble table", "wooden counter", "cafe table\"""",
            
            "gym": """

GYM SPECIALIZATION:
- person_type: Describe fitness level (예: "athletic man", "fitness woman", "muscular person")
  ⚠️  NOT generic: "person" (be specific!)
- activity: Specific exercise (예: "barbell squat", "bench press", "deadlift", "running")
  ⚠️  NOT generic: "workout", "exercise" (name the exercise!)
- focus: What to highlight (예: "muscle definition", "form", "strength", "power")""",
            
            "bakery": """

BAKERY SPECIALIZATION:
- product: Exact baked good (예: "croissant", "baguette", "sourdough bread", "chocolate cake")
  ⚠️  NOT generic: "bread", "pastry" (be specific!)
- state: Freshness indicator (예: "freshly baked", "warm", "golden brown", "crispy")
- presentation: Display method (예: "on wooden board", "in wicker basket", "on display shelf")""",
            
            "restaurant": """

RESTAURANT SPECIALIZATION:
- dish: Complete dish name (예: "pasta carbonara", "grilled ribeye steak", "caesar salad")
  ⚠️  NOT generic: "pasta", "meat" (include full dish name!)
- plating: Plating style (예: "elegantly plated", "rustic presentation", "modern plating")
- cuisine_style: Cuisine type (예: "italian", "french", "japanese", "korean")"""
        }
        
        # ====================================================================
        # 조합: Base + Specialized (있으면)
        # ====================================================================
        # laundry, hair_salon, nail_salon 등은 base만으로 충분
        specialized = specialized_guides.get(industry, "")
        
        return base_prompt + specialized
    
    def _build_user_prompt_for_extraction(self, user_input: str, industry: str) -> str:
        """
        키워드 추출용 사용자 프롬프트
        
        Args:
            user_input: 사용자 입력
            industry: 감지된 업종
        
        Returns:
            str: 사용자 프롬프트
        """
        
        return f"""Extract keywords from this Korean input and translate to English.

User input: {user_input}
Detected industry: {industry}

Output ONLY valid JSON with English values.
Include only the fields that are clearly present in the input.

JSON:"""


# ============================================
# 유틸리티 함수
# ============================================

def clean_input(text):
    """
    입력 텍스트 정제 - surrogate 문자 제거
    """
    if not text:
        return ""
    
    try:
        cleaned = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t ')
        return cleaned.strip()
    except Exception as e:
        print(f"⚠️  입력 정제 중 오류: {e}")
        return ''.join(char for char in text if ord(char) < 128).strip()


# ============================================
# 테스트 코드
# ============================================

if __name__ == "__main__":
    print("=" * 80)
    print("🔍 Keyword Extraction Module (GPT-4o)")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    # 테스트 케이스
    test_cases = [
        "카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌",
        "헬스장 근육맨 스쿼트하는 모습",
        "빵집 갓 구운 크루아상 나무 보드에 올린 사진",
        "레스토랑 파스타 까르보나라 예쁘게 플레이팅"
    ]
    
    print("\n📝 테스트 케이스:")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test}")
        print(f"{'='*80}")
        
        result = manager.extract_keywords_english(test)
        
        print(f"\n결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*80}")
    print("✅ 테스트 완료")
    print(f"{'='*80}")