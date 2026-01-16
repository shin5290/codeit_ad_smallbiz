"""
프롬프트 생성 모듈 (Z-Image Turbo용)
작성자: 배현석
버전: 5.0 - Z-Image Turbo 최적화

Z-Image Turbo 특징:
- Negative Prompt 미지원 (CFG 미사용)
- 긴 프롬프트 권장 (512-1024 토큰, T5 인코더)
- 자연어 문장 형태 선호 (카메라 지시처럼)
- 가중치 문법 미지원
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
from .config_loader import industry_config

load_dotenv()


class PromptTemplateManager:
    """한글 입력 → Z-Image Turbo용 영어 프롬프트 생성 (GPT)"""

    def __init__(self):
        """초기화: OpenAI 클라이언트 설정"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5-mini"

    def generate_detailed_prompt(self, user_input: str, style: str = "realistic", conversation_history: list = None) -> dict:
        """
        한글 사용자 입력 → Z-Image Turbo용 상세 영어 프롬프트 생성

        Z-Image Turbo 특징:
        - Negative Prompt 미지원 → positive만 생성
        - 긴 자연어 프롬프트 권장 (80-250 단어)
        - 카메라 지시 스타일 (구체적, 명확)

        Args:
            user_input (str): 한글 사용자 요청
            style (str): 스타일 힌트 (realistic, semi_realistic, anime)
                         ※ LoRA로 적용되므로 프롬프트에는 반영 안함
            conversation_history (list): 대화 히스토리 (선택사항)
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            dict: {
                "positive": "상세한 prompt...",
                "negative": "",  # ZIT는 negative 미지원
                "style": "detected style",
                "industry": "detected industry"
            }
        """
        print(f"\n{'='*80}")
        print(f"🎨 Z-Image Turbo 프롬프트 생성 중...")
        print(f"   입력: {user_input}")
        if conversation_history:
            print(f"   대화 히스토리: {len(conversation_history)}개 메시지")
        print(f"{'='*80}")

        try:
            # 1. 업종 자동 감지
            industry = self._detect_industry(user_input)
            print(f"   감지된 업종: {industry}")

            # 2. GPT 시스템 프롬프트 (ZIT 최적화)
            system_prompt = self._get_system_prompt_for_zit()

            # 3. 업종별 참고 키워드 가져오기 (YAML에서)
            reference_keywords = self._get_industry_reference_keywords(industry)

            # 4. 대화 히스토리 컨텍스트 구성
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                # 최근 5개 메시지만 사용 (토큰 절약)
                recent_messages = conversation_history[-5:]
                conv_str = "\n".join([
                    f"- {msg['role']}: {msg['content'][:100]}"
                    for msg in recent_messages
                ])
                conversation_context = f"""
===== CONVERSATION CONTEXT =====
The user has been chatting with an AI assistant. Consider this context when generating the prompt:
{conv_str}

Use this context to understand what the user wants (e.g., style preferences, specific details mentioned earlier).
=================================
"""

            # 5. 사용자 프롬프트
            user_prompt = f"""Generate a detailed Z-Image Turbo prompt for this request:

User Input (Korean): {user_input}
Style Hint: {style}
Detected Industry: {industry}

{conversation_context}

===== REFERENCE KEYWORDS (FOR INSPIRATION ONLY) =====
{reference_keywords}

⚠️ CRITICAL WARNING ⚠️
These keywords are ONLY for inspiration!
DO NOT copy them directly!
Create your OWN unique, creative description inspired by these concepts.
=================================================

Remember:
- Write LONG, DETAILED natural language description (80-250 words)
- Describe like you're directing a camera crew
- Include: subject, action, setting, lighting, atmosphere, textures, colors
- NO negative prompts (Z-Image Turbo doesn't support them)
- Output valid JSON with "positive" and "style" fields only
- If conversation context provided, incorporate any specific details or preferences mentioned"""

            # 6. GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            # 7. 응답 추출
            result = response.choices[0].message.content.strip()

            # 8. JSON 파싱
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            prompt_data = json.loads(result)

            # 9. 결과 검증
            positive = prompt_data.get("positive", "")
            detected_style = prompt_data.get("style", style)

            print(f"\n✅ 프롬프트 생성 완료!")
            print(f"   Style: {detected_style}")
            print(f"   Positive: {len(positive)} chars (~{len(positive.split())} words)")
            print(f"{'='*80}\n")

            return {
                "positive": positive,
                "negative": "",  # Z-Image Turbo는 negative 미지원
                "style": detected_style,
                "industry": industry
            }

        except Exception as e:
            print(f"❌ 프롬프트 생성 오류: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: 기본 프롬프트 생성
            print("⚠️  Fallback: 기본 프롬프트 사용")
            return self._fallback_prompt_generation(user_input, style)

    def _get_system_prompt_for_zit(self) -> str:
        """
        Z-Image Turbo 최적화 시스템 프롬프트

        핵심 원칙:
        1. 긴 자연어 설명 (80-250 단어)
        2. 카메라 지시 스타일 (구체적, 명확)
        3. Negative prompt 없음
        4. 가중치 문법 없음
        5. 주요 키워드는 프롬프트 앞부분에 배치
        """
        return """You are a Z-Image Turbo prompt engineer. Convert Korean requests into detailed English image prompts.

## Z-IMAGE TURBO CHARACTERISTICS
- T5 text encoder: supports LONG prompts (80-250 words optimal)
- NO negative prompts supported (classifier-free guidance disabled)
- NO weight syntax like (keyword:1.3)
- Prefers NATURAL LANGUAGE over keyword lists
- Put most important keywords (Subject + Text) at the VERY START

## OUTPUT FORMAT
{"positive": "detailed natural language prompt...", "style": "realistic|semi_realistic|anime"}

## STYLE DETECTION (for LoRA selection)
- anime: 캐릭터, 애니, 만화, 마스코트, 귀여운, 일러스트
- semi_realistic: 반실사, 디지털아트, 일러스트풍
- realistic: 사진, 상품, 음식, 포토 (DEFAULT)

## PROMPT STRUCTURE (4-Step Formula)
1. **Subject & Action** (FIRST - most important): Who/what, doing what, in what pose
2. **Visual Style & Medium**: Photography style, camera specs, or artistic medium
3. **Lighting & Atmosphere**: Light source, mood, time of day
4. **Details & Textures**: Skin texture, fabric detail, film grain, imperfections

## IMAGE TYPE DETECTION
- MULTI (이모티콘/스티커/여러포즈): "character sheet showing multiple poses and expressions"
- SINGLE (default): Focus on ONE clear action, ONE composition

## STYLE-SPECIFIC GUIDELINES

### REALISTIC (Photography)
Write like camera direction:
"Shot on [camera] with [lens], [lighting description], [subject] [action] in [setting], [atmosphere], [technical details like depth of field, color grading]"

Example: "Shot on a Canon EOS R5 with 85mm f/1.4 lens, natural window light creating soft shadows. A freshly brewed strawberry latte in a tall glass sits centered on a white marble table, steam gently rising. The drink features beautiful pink and white layers with fresh strawberry slices. Shallow depth of field with creamy bokeh background, warm color tones, professional commercial food photography aesthetic."

### ANIME/ILLUSTRATION
Focus on artistic elements:
"[Style] illustration of [subject] [action], [artistic details], [color palette], [mood]"

Example: "Vibrant anime style illustration of a cheerful bear mascot character lifting dumbbells at a colorful gym. The character has round cute features, determined expression, wearing a red headband. Flat color shading with clean bold outlines, pastel background with energetic sparkle effects, kawaii aesthetic, dynamic pose suggesting movement and energy."

### SEMI-REALISTIC
Blend photography with artistic:
"Highly detailed digital artwork of [subject], [cinematic elements], [artistic touches]"

## CRITICAL RULES
1. Start with the MAIN SUBJECT immediately
2. Write in flowing sentences, not comma-separated keywords
3. Include TEXTURE descriptions (skin pores, fabric weave, condensation drops)
4. Specify LIGHTING source and quality
5. If text/words needed in image, put them in "quotes"
6. 80-250 words is optimal (model attention fades after ~75 tokens for key elements)

## COMMON MISTAKES TO AVOID
- Don't mix contradictory styles ("photorealistic anime")
- Don't use generic terms ("beautiful", "amazing") - be SPECIFIC
- Don't forget texture keywords (images look plastic without them)"""

    def _get_industry_reference_keywords(self, industry: str) -> str:
        """
        YAML에서 업종별 참고 키워드 추출 (GPT 참고용)

        Args:
            industry: 업종 코드 (cafe, gym 등)

        Returns:
            str: 참고용 키워드 문자열
        """
        try:
            if industry_config is None:
                return "No reference keywords available."

            industry_data = industry_config.get_industry(industry)
            if not industry_data or "prompt_template" not in industry_data:
                return "No reference keywords available."

            template = industry_data["prompt_template"]
            keywords = []

            # 주요 키워드 카테고리 추출
            if "lighting_phrases" in template:
                keywords.append(f"Lighting: {', '.join(template['lighting_phrases'][:3])}")
            if "composition_keywords" in template:
                keywords.append(f"Composition: {', '.join(template['composition_keywords'][:3])}")
            if "color_phrases" in template:
                keywords.append(f"Colors: {', '.join(template['color_phrases'][:3])}")
            if "style_keywords" in template:
                keywords.append(f"Style: {', '.join(template['style_keywords'][:3])}")
            if "details_keywords" in template:
                keywords.append(f"Details: {', '.join(template['details_keywords'][:3])}")

            if not keywords:
                return "No reference keywords available."

            return "\n".join(keywords)

        except Exception as e:
            print(f"⚠️ 참고 키워드 로드 실패: {e}")
            return "No reference keywords available."

    def _fallback_prompt_generation(self, user_input: str, style: str) -> dict:
        """Fallback: 기본 프롬프트 생성"""
        industry = self._detect_industry(user_input)

        # 스타일별 기본 프롬프트
        if style == "anime":
            prompt = f"Anime style illustration of {user_input}, vibrant colors, clean lineart, detailed artwork, professional quality"
        elif style == "semi_realistic":
            prompt = f"Highly detailed digital artwork of {user_input}, cinematic lighting, semi-realistic style, professional quality"
        else:
            prompt = f"Professional commercial photography of {user_input}, natural lighting, sharp focus, high quality, detailed textures"

        return {
            "positive": prompt,
            "negative": "",  # Z-Image Turbo는 negative 미지원
            "style": style,
            "industry": industry
        }

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
    print("🔍 Z-Image Turbo Prompt Generator")
    print("=" * 80)

    manager = PromptTemplateManager()

    # 테스트 케이스
    test_cases = [
        ("카페 신메뉴 딸기라떼 홍보, 따뜻한 느낌", "realistic"),
        ("귀여운 곰 캐릭터가 헬스장에서 운동하는 광고", "anime"),
        ("빵집 갓 구운 크루아상 나무 보드에 올린 사진", "realistic"),
    ]

    print("\n📝 테스트 케이스:")
    for i, (test_input, test_style) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test_input} (style: {test_style})")
        print(f"{'='*80}")

        result = manager.generate_detailed_prompt(test_input, test_style)

        print(f"\n결과:")
        print(f"  Style: {result['style']}")
        print(f"  Industry: {result['industry']}")
        print(f"  Prompt ({len(result['positive'])} chars):")
        print(f"  {result['positive'][:200]}...")

    print(f"\n{'='*80}")
    print("✅ 테스트 완료")
    print(f"{'='*80}")
