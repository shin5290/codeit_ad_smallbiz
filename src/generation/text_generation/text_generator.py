"""
광고 문구 생성 모듈
작성자: 배현석
버전: 1.1 (신승목, 로깅 추가)
"""

import os
import re
from typing import Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from src.generation.text_generation.prompt_manager import PromptTemplateManager
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


class TextGenerator:
    """광고 문구 생성 클래스"""

    def __init__(self, use_industry_config: bool = True):
        """
        초기화: OpenAI 클라이언트 설정

        Args:
            use_industry_config (bool): True이면 PromptTemplateManager를 사용하여
                                        업종별 최적화된 프롬프트 생성
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.use_industry_config = use_industry_config
        self.prompt_manager = PromptTemplateManager()

    def generate_ad_copy(
        self,
        user_input,
        tone="warm",
        max_length=100,
        chat_history=None,
        generation_history=None,
        industry=None,
    ):
        """
        광고 문구 생성

        Args:
            user_input (str): 사용자 요청 텍스트
                예: "카페 신메뉴 홍보, 따뜻한 느낌, 겨울"
            tone (str): 톤 앤 매너 ("warm", "professional", "friendly", "energetic")
            max_length (int): 최대 글자 수 (기본 100자)
            chat_history (list): 최근 대화 히스토리 (선택)
            generation_history (list): 이전 생성 이력 (선택)
            industry (str): 업종 정보 (cafe, restaurant, gym 등, 선택)
        Returns:
            str: 생성된 광고 문구
                예: "따뜻한 겨울, 새로운 맛"
        """

        logger.info("📝 광고 문구 생성 중...")
        logger.info(f"   입력: {user_input}")
        logger.info(f"   톤: {tone}, 최대 {max_length}자, 업종: {industry or 'general'}")
        logger.info(f"   업종 설정 사용: {self.use_industry_config}")
        try:
            length_spec = self._extract_length_expectation(
                self._merge_length_source_text(user_input)
            )
            target_min = length_spec.min_len
            target_max = length_spec.max_len
            target_source = length_spec.source
            wants_long = length_spec.is_long

            safe_max_length = 100
            if max_length is not None:
                try:
                    safe_max_length = int(max_length)
                except (TypeError, ValueError):
                    safe_max_length = 100
            elif target_max is not None:
                safe_max_length = int(target_max)

            if safe_max_length < 10:
                safe_max_length = 10

            if target_source in ("range", "approx", "exact") and target_max is not None:
                if max_length is not None and int(max_length) < target_max:
                    logger.warning(
                        "length override: spec length=%s exceeds intent max_length=%s",
                        target_max,
                        max_length,
                    )
                safe_max_length = max(safe_max_length, target_max)

            if wants_long and target_source != "exact":
                if max_length is not None and int(max_length) < 600:
                    logger.warning(
                        "length override: user wants long-form but intent max_length=%s",
                        max_length,
                    )
                target_max = min(800, max(600, target_max or 0))
                if target_min is None:
                    target_min = int(target_max * 0.7)
                safe_max_length = max(safe_max_length, target_max)

            if not wants_long and target_max is not None and target_max > safe_max_length:
                logger.warning(
                    "length conflict: target_max=%s exceeds max_length=%s (source=%s)",
                    target_max,
                    safe_max_length,
                    target_source,
                )
                target_max = safe_max_length

            if target_min is not None and target_min > safe_max_length:
                logger.warning(
                    "length conflict: target_min=%s exceeds max_length=%s (source=%s)",
                    target_min,
                    safe_max_length,
                    target_source,
                )
                target_min = max(10, int(safe_max_length * 0.9))

            min_length = target_min if target_min is not None else max(10, int(safe_max_length * 0.7))

            require_hashtags, hashtag_count = self._detect_hashtag_requirement(
                user_input=user_input,
                chat_history=chat_history,
            )

            # 업종별 설정 사용 시 PromptTemplateManager 활용
            history_context = self._build_history_context(
                chat_history=chat_history,
                generation_history=generation_history,
            )

            for attempt in range(2):
                if self.use_industry_config and self.prompt_manager:
                    prompts = self.prompt_manager.get_ad_copy_prompt(
                        user_input=user_input,
                        tone=tone,
                        max_length=safe_max_length,
                        industry=industry
                    )
                    system_prompt = self._append_length_emphasis(
                        prompts["system_prompt"], safe_max_length, min_length
                    )
                    if require_hashtags:
                        system_prompt = self._append_hashtag_instruction(
                            system_prompt, hashtag_count
                        )
                    user_prompt = prompts["user_prompt"]
                    user_prompt = self._append_user_requirements(
                        user_prompt,
                        max_length=safe_max_length,
                        min_length=min_length,
                        hashtag_count=hashtag_count if require_hashtags else None,
                    )
                    if history_context:
                        user_prompt = f"{user_prompt}\n\n{history_context}"
                    # 자동 감지된 업종 정보 출력
                    detected_industry = prompts.get("industry", industry)
                    logger.info(f"   감지된 업종: {detected_industry}")
                else:
                    # 1. 시스템 프롬프트 선택 (대화 히스토리와 업종 정보 포함)
                    system_prompt = self._get_system_prompt(
                        tone,
                        safe_max_length,
                        chat_history=chat_history,
                        generation_history=generation_history,
                        industry=industry,
                    )
                    if require_hashtags:
                        system_prompt = self._append_hashtag_instruction(
                            system_prompt, hashtag_count
                        )

                    # 2. 사용자 프롬프트 구성
                    user_prompt = self._build_user_prompt(
                        user_input,
                        safe_max_length,
                        min_length=min_length,
                        history_context=history_context,
                        hashtag_count=hashtag_count if require_hashtags else None,
                    )

                # 3. GPT API 호출
                if wants_long:
                    max_tokens = min(1500, max(900, int(safe_max_length * 2)))
                else:
                    max_tokens = min(900, max(120, int(safe_max_length * 1.5)))
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )

                # 4. 응답 추출
                ad_copy = response.choices[0].message.content.strip()

                # 5. 후처리
                ad_copy = self._postprocess(ad_copy, safe_max_length)

                too_short = len(ad_copy) < min_length
                missing_hashtags = require_hashtags and ("#" not in ad_copy)
                if too_short or missing_hashtags:
                    logger.debug(
                        "postprocess check failed (attempt=%s): too_short=%s, missing_hashtags=%s",
                        attempt + 1,
                        too_short,
                        missing_hashtags,
                    )
                    if attempt == 0:
                        continue

                logger.info(f"✅ 생성 완료: {ad_copy}")
                return ad_copy

            logger.info(f"✅ 생성 완료: {ad_copy}")
            return ad_copy

        except Exception as e:
            logger.error(f"❌ 오류 발생: {e}")
            return self._get_fallback_copy()

    def _get_system_prompt(
        self,
        tone,
        max_length,
        chat_history=None,
        generation_history=None,
        industry=None,
    ):
        """톤, 대화 히스토리, 업종에 따른 시스템 프롬프트 반환"""

        base_prompt = f"""당신은 소상공인을 위한 전문 광고 카피라이터입니다.
짧고 임팩트 있는 광고 문구를 만들어주세요.

규칙:
- 반드시 {max_length}자 이내 (공백 포함, 초과 금지)
- 너무 짧게 쓰지 말고 최소한의 길이를 확보할 것
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

        # 업종별 가이드라인 추가 (30개)
        industry_guides = {
            # 1-10: 기존 업종
            "cafe": "카페는 '휴식', '여유', '감성', '따뜻함', '일상 속 특별한 순간'을 강조하세요. 커피 향, 아늑한 공간, 힐링의 느낌을 살립니다.",
            "restaurant": "음식점은 '맛', '오감', '경험', '만족감', '특별한 한 끼'를 강조하세요. 신선함, 정성, 풍미를 표현합니다.",
            "bakery": "베이커리는 '갓 구운', '신선함', '달콤함', '행복', '일상의 작은 사치'를 강조하세요. 빵 굽는 향기와 따뜻함을 전합니다.",
            "gym": "헬스장은 '변화', '도전', '성취', '건강', '새로운 나'를 강조하세요. 동기부여와 긍정적 변화를 표현합니다.",
            "beauty": "뷰티/미용은 '아름다움', '자신감', '케어', '스타일', '나를 위한 투자'를 강조하세요. 변화와 자기관리의 가치를 전합니다.",
            "fashion": "패션/의류는 '스타일', '트렌드', '개성', '멋', '나만의 표현'을 강조하세요. 감각과 차별화를 살립니다.",
            "hair_salon": "미용실은 '스타일 변신', '헤어 케어', '전문성', '트렌디함'을 강조하세요. 새로운 이미지와 만족감을 전합니다.",
            "nail_salon": "네일샵은 '섬세함', '아름다운 디테일', '감각', '자기 표현'을 강조하세요. 손끝의 예술과 케어를 표현합니다.",
            "flower_shop": "꽃집은 '감성', '특별한 날', '마음 전달', '아름다움', '자연'을 강조하세요. 꽃의 의미와 감동을 살립니다.",
            "laundry": "세탁소는 '깨끗함', '편리함', '신뢰', '전문 케어'를 강조하세요. 옷의 수명 연장과 안심을 전합니다.",

            # 11-20: 추가 생활 서비스
            "convenience_store": "편의점은 '편리함', '24시간', '빠른 해결', '일상 필수'를 강조하세요. 언제나 가까운 곳에서의 편의성을 전합니다.",
            "pharmacy": "약국은 '건강', '전문 상담', '신뢰', '정확함', '케어'를 강조하세요. 건강 지킴이로서의 전문성을 표현합니다.",
            "hospital": "병원/의원은 '건강', '전문 진료', '신뢰', '정확한 진단', '안심'을 강조하세요. 의료 전문성과 환자 케어를 전합니다.",
            "dental_clinic": "치과는 '건강한 미소', '전문 진료', '무통 치료', '예방', '자신감'을 강조하세요. 치아 건강과 심미성을 표현합니다.",
            "pet_shop": "애완동물샵은 '반려동물', '사랑', '케어', '행복', '가족'을 강조하세요. 반려동물과의 특별한 유대감을 살립니다.",
            "bookstore": "서점은 '지식', '여유', '탐험', '새로운 세계', '힐링'을 강조하세요. 책을 통한 경험과 성장을 표현합니다.",
            "stationery": "문구점은 '창의성', '학습', '준비', '설렘', '새 학기'를 강조하세요. 문구를 통한 즐거움과 준비성을 전합니다.",
            "pc_cafe": "PC방은 '게임', '몰입', '편안함', '최신 시설', '친구와 함께'를 강조하세요. 게임 환경과 즐거움을 표현합니다.",
            "karaoke": "노래방은 '스트레스 해소', '즐거움', '추억', '자유로움', '신나는 시간'을 강조하세요. 노래를 통한 힐링과 즐거움을 전합니다.",
            "academy": "학원은 '성적 향상', '전문 강사', '체계적 학습', '미래 준비', '성공'을 강조하세요. 교육의 질과 성과를 표현합니다.",

            # 21-30: 전문 서비스 및 기타
            "yoga": "요가/필라테스는 '균형', '유연성', '힐링', '건강한 몸과 마음', '명상'을 강조하세요. 몸과 마음의 조화를 표현합니다.",
            "massage": "마사지/스파는 '힐링', '릴랙스', '피로 회복', '프리미엄 케어', '재충전'을 강조하세요. 깊은 휴식과 회복을 전합니다.",
            "real_estate": "부동산은 '신뢰', '전문성', '최적의 선택', '정확한 정보', '내 집 마련'을 강조하세요. 부동산 전문가로서의 신뢰를 표현합니다.",
            "car_wash": "세차장은 '깨끗함', '광택', '프리미엄 케어', '새차 같은 느낌', '편리함'을 강조하세요. 차량 관리의 완성도를 전합니다.",
            "car_repair": "자동차정비는 '안전', '전문 기술', '신뢰', '정확한 진단', '정직함'을 강조하세요. 차량 정비의 전문성과 신뢰를 표현합니다.",
            "optical_shop": "안경점은 '시력 보호', '스타일', '전문 검안', '편안함', '명품 안경'을 강조하세요. 시력과 스타일의 조화를 전합니다.",
            "jewelry": "주얼리/액세서리는 '특별함', '빛나는 순간', '프리미엄', '선물', '품격'을 강조하세요. 특별한 순간의 가치를 표현합니다.",
            "furniture": "가구점은 '편안함', '인테리어', '품질', '나만의 공간', '실용성'을 강조하세요. 공간을 완성하는 가구의 가치를 전합니다.",
            "interior": "인테리어는 '공간 변신', '맞춤 디자인', '감각', '실용성과 미학', '꿈의 공간'을 강조하세요. 공간의 완전한 변화를 표현합니다.",
            "cleaning_service": "청소/이사는 '깨끗함', '편리함', '전문성', '신뢰', '새로운 시작'을 강조하세요. 청결과 편의성을 전합니다.",
        }

        industry_guide = ""
        if industry and industry in industry_guides:
            industry_guide = f"\n\n업종 특화 가이드라인:\n{industry_guides[industry]}"

        history_context = self._build_history_context(
            chat_history=chat_history,
            generation_history=generation_history,
        )
        history_section = f"\n\n{history_context}" if history_context else ""

        return f"{base_prompt}\n\n톤 앤 매너:\n{tone_guide}{industry_guide}{history_section}"

    def _build_user_prompt(
        self,
        user_input,
        max_length,
        min_length=None,
        history_context=None,
        hashtag_count=None,
    ):
        """사용자 프롬프트 구성"""

        history_note = ""
        if history_context:
            history_note = f"\n\n{history_context}"

        requirements = [f"반드시 {max_length}자 이내 (초과 금지)"]
        if min_length is not None:
            requirements.append(f"최소 {min_length}자 이상")
        if hashtag_count is not None:
            requirements.append(
                f"해시태그 {hashtag_count}개를 문장 끝에 포함 (각 해시태그는 #으로 시작, 공백으로 구분)"
            )
        requirements_text = "\n- ".join(requirements)

        return f"""다음 내용으로 광고 문구를 만들어주세요:

{user_input}{history_note}

요구사항:
- {requirements_text}
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
            trimmed = text[:max_length]
            last_space = trimmed.rfind(" ")
            if last_space >= int(max_length * 0.6):
                trimmed = trimmed[:last_space]
            trimmed = trimmed.strip()
            logger.debug(
                "postprocess: trimmed from %s to %s chars",
                len(text),
                len(trimmed),
            )
            text = trimmed

        # 3. 빈 문자열 체크
        if not text:
            return self._get_fallback_copy()

        return text

    def _get_fallback_copy(self):
        """GPT 실패 시 기본 문구 반환"""
        return "특별한 순간을 함께하세요"

    def _append_length_emphasis(self, prompt: str, max_length: int, min_length: int) -> str:
        return (
            f"{prompt}\n\n중요: 반드시 {max_length}자 이내로 작성하세요. "
            f"너무 짧게 쓰지 말고 최소 {min_length}자 이상을 지키세요."
        )

    def _build_history_context(self, chat_history=None, generation_history=None) -> str:
        sections = []
        chat_part = self._format_chat_history(chat_history)
        if chat_part:
            sections.append(f"최근 대화 맥락(참고):\n{chat_part}")
        gen_part = self._format_generation_history(generation_history)
        if gen_part:
            sections.append(f"이전에 생성된 문구(참고):\n{gen_part}")
        return "\n\n".join(sections)

    def _append_hashtag_instruction(self, prompt: str, hashtag_count: int) -> str:
        return (
            f"{prompt}\n\n해시태그 규칙: 문장 끝에 해시태그 {hashtag_count}개를 포함하고, "
            "각 해시태그는 #으로 시작하며 공백으로 구분하세요."
        )

    def _append_user_requirements(
        self,
        user_prompt: str,
        max_length: int,
        min_length: int,
        hashtag_count: Optional[int],
    ) -> str:
        requirements = [
            f"- 반드시 {max_length}자 이내 (초과 금지)",
            f"- 최소 {min_length}자 이상",
        ]
        if hashtag_count:
            requirements.append(
                f"- 문장 끝에 해시태그 {hashtag_count}개 포함 (#으로 시작, 공백으로 구분)"
            )
        return f"{user_prompt}\n\n요구사항:\n" + "\n".join(requirements)

    def _detect_hashtag_requirement(self, user_input, chat_history=None) -> tuple[bool, int]:
        combined = " ".join(
            [user_input or ""]
            + [msg.get("content", "") for msg in (chat_history or [])[-5:]]
        )
        text = combined.lower()
        if "해시태그" in text or "hash tag" in text or "hashtag" in text or "#" in text:
            count = self._extract_hashtag_count(combined)
            return True, count
        return False, 0

    def _extract_hashtag_count(self, text: str) -> int:
        match = re.search(r"해시\s*태그\s*(\d+)\s*개|해시태그\s*(\d+)\s*개|해시태그\s*(\d+)", text)
        if match:
            for group in match.groups():
                if group and group.isdigit():
                    value = int(group)
                    return max(1, min(value, 10))
        return 3

    def _extract_length_expectation(self, user_input: str) -> "LengthSpec":
        if not user_input:
            return LengthSpec(None, None, "none", False)

        text = user_input.replace(" ", "")

        range_match = re.search(r"(\d{2,4})[~\-](\d{2,4})자", text)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start > end:
                start, end = end, start
            return LengthSpec(start, end, "range", end >= 400)

        approx_match = re.search(r"약(\d{2,4})자", text)
        if approx_match:
            value = int(approx_match.group(1))
            return LengthSpec(int(value * 0.8), value, "approx", value >= 400)

        exact_match = re.search(r"(\d{2,4})자", text)
        if exact_match:
            value = int(exact_match.group(1))
            return LengthSpec(int(value * 0.8), value, "exact", value >= 400)

        if "길게" in text:
            return LengthSpec(400, 600, "long", True)
        if "짧게" in text:
            return LengthSpec(20, 40, "short", False)

        return LengthSpec(None, None, "none", False)

    def _merge_length_source_text(self, user_input) -> str:
        return user_input or ""

    def _format_chat_history(self, chat_history, max_items: int = 5, max_len: int = 120) -> str:
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history[-max_items:]:
            role = msg.get("role", "unknown")
            content = " ".join((msg.get("content") or "").split())
            if not content:
                continue
            snippet = content[:max_len]
            if len(content) > max_len:
                snippet += "..."
            lines.append(f"- {role}: {snippet}")
        return "\n".join(lines)

    def _format_generation_history(
        self,
        generation_history,
        max_items: int = 3,
        max_len: int = 120,
    ) -> str:
        if not generation_history:
            return ""
        lines = []
        for gen in generation_history:
            if gen.get("content_type") != "text":
                continue
            output_text = " ".join((gen.get("output_text") or "").split())
            if not output_text:
                continue
            snippet = output_text[:max_len]
            if len(output_text) > max_len:
                snippet += "..."
            lines.append(f"- {snippet}")
            if len(lines) >= max_items:
                break
        return "\n".join(lines)


class LengthSpec:
    def __init__(self, min_len: Optional[int], max_len: Optional[int], source: str, is_long: bool):
        self.min_len = min_len
        self.max_len = max_len
        self.source = source
        self.is_long = is_long


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
