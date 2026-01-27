"""
RAG 기반 챗봇 모듈

역할 분리:
- ConversationManager: DB 접근 레이어 (대화/생성 이력 CRUD)
- LLMOrchestrator: LLM 호출 레이어 (intent 분석, refinement)
- ConsultingService: 상담 전용 비즈니스 로직
"""

import json, openai, re
from typing import Any, Optional, List, Dict, AsyncIterator
from sqlalchemy.orm import Session

from src.utils.image import image_payload 
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================
# Helper Functions
# =====================================================

def _format_chat_history(chat_history: List[Dict]) -> str:
    """대화 히스토리를 문자열로 포맷"""
    if not chat_history:
        return "(없음)"
    lines = []
    for idx, msg in enumerate(chat_history, start=1):
        role = msg.get("role") or "unknown"
        content = (msg.get("content") or "").strip()
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines)


def _format_generation_history(generation_history: List[Dict]) -> str:
    """생성 이력을 recent_rank 포함하여 포맷"""
    if not generation_history:
        return "(없음)"
    
    lines = []
    for idx, gen in enumerate(generation_history, start=1):
        gen_id = gen.get("id")
        content_type = gen.get("content_type") or "unknown"
        input_text = (gen.get("input_text") or "").strip()
        timestamp = gen.get("timestamp", "")
        has_image = "Y" if gen.get("output_image") or gen.get("input_image") else "N"
        chrono_rank = gen.get("_chronological_rank")
        chrono_part = f", chrono_rank={chrono_rank}" if chrono_rank is not None else ""
        prompt_text = (gen.get("prompt") or "").strip()
        prompt_part = f", 프롬프트=\"{prompt_text[:80]}\"" if prompt_text else ""
        
        lines.append(
            f"recent_rank={idx}{chrono_part}, ID={gen_id}, 타입={content_type}, "
            f"이미지={has_image}, 입력=\"{input_text[:100]}\"{prompt_part}, 시간={timestamp}"
        )
    
    return "\n".join(lines)


def _summarize_chat_history(
    chat_history: List[Dict],
    *,
    keep_last: int = 10,
    summary_items: int = 6,
    snippet_len: int = 80,
) -> tuple[List[Dict], str]:
    """대화 히스토리를 최근 N개로 제한하고 이전 내용은 요약 문자열로 반환."""
    if not chat_history or len(chat_history) <= keep_last:
        return chat_history, ""

    recent = chat_history[-keep_last:]
    older = chat_history[:-keep_last]
    summary_lines: List[str] = []
    for msg in older:
        role = msg.get("role") or "unknown"
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        snippet = content[:snippet_len]
        summary_lines.append(f"{role}: {snippet}")
        if len(summary_lines) >= summary_items:
            break

    omitted = len(older) - len(summary_lines)
    if summary_lines:
        summary = " | ".join(summary_lines)
        if omitted > 0:
            summary += f" ... 외 {omitted}개"
        return recent, summary

    return recent, f"이전 대화 {len(older)}개 생략"


def _summarize_generation_history(
    generation_history: List[Dict],
    *,
    keep_latest: int = 5,
    keep_oldest: int = 2,
    summary_items: int = 4,
    snippet_len: int = 80,
) -> tuple[List[Dict], str]:
    """생성 이력을 최신/최초 일부만 유지하고 중간은 요약 문자열로 반환."""
    if not generation_history:
        return generation_history, ""

    total = len(generation_history)
    if total <= keep_latest + keep_oldest:
        return generation_history, ""

    latest = generation_history[:keep_latest]
    oldest = generation_history[-keep_oldest:]
    middle = generation_history[keep_latest:-keep_oldest]

    summary_lines: List[str] = []
    for gen in middle:
        gen_id = gen.get("id")
        content_type = gen.get("content_type") or "unknown"
        input_text = (gen.get("input_text") or "").strip().replace("\n", " ")
        if not input_text:
            continue
        snippet = input_text[:snippet_len]
        summary_lines.append(f"ID={gen_id}, 타입={content_type}, 입력={snippet}")
        if len(summary_lines) >= summary_items:
            break

    omitted = len(middle) - len(summary_lines)
    summary = ""
    if summary_lines:
        summary = " | ".join(summary_lines)
        if omitted > 0:
            summary += f" ... 외 {omitted}개"
    else:
        summary = f"중간 생성 이력 {len(middle)}개 생략"

    return latest + oldest, summary


def _extract_json_dict(content: Optional[str]) -> Optional[Dict]:
    """LLM 응답에서 JSON dict를 안전하게 추출."""
    if not content:
        return None

    cleaned = content.strip()

    # 코드펜스 제거
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = cleaned[start:end + 1].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 흔한 trailing comma 보정
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


# =====================================================
# ConversationManager: 대화 히스토리 관리
# =====================================================

class ConversationManager:
    """대화 히스토리 관리 클래스 (DB 접근 레이어)"""

    def add_message(
        self,
        db: Session,
        session_id: str,
        role: str,
        content: str,
        image_id: Optional[int] = None,
    ) -> int:
        """메시지 저장"""
        from src.backend import process_db

        chat_row = process_db.save_chat_message(
            db, session_id, role, content, image_id=image_id
        )
        logger.debug("ConversationManager: saved message id=%s", chat_row.id)
        return chat_row.id

    def get_recent_messages(
        self, db: Session, session_id: str, limit: int = 10
    ) -> List[Dict]:
        """최근 대화 조회"""
        from src.backend import process_db

        messages = process_db.get_chat_history_by_session(db, session_id, limit)
        messages_reversed = list(reversed(messages))

        return [
            {
                "role": msg.role,
                "content": msg.content,
                "image_id": msg.image_id,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in messages_reversed
        ]

    def get_full_messages(self, db: Session, session_id: str) -> List[Dict]:
        """전체 대화 조회"""
        from src.backend import process_db

        messages = process_db.get_chat_history_by_session(db, session_id, limit=None)
        messages_reversed = list(reversed(messages))

        return [
            {
                "role": msg.role,
                "content": msg.content,
                "image_id": msg.image_id,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in messages_reversed
        ]

    def get_generation_history(
        self, db: Session, session_id: str, limit: int = 5
    ) -> List[Dict]:
        """생성 이력 조회"""
        from src.backend import process_db

        generations = process_db.get_generation_history_by_session(db, session_id, limit)

        return [
            {
                "id": gen.id,
                "content_type": gen.content_type,
                "input_text": gen.input_text,
                "output_text": gen.output_text,
                "prompt": gen.prompt,
                "style": gen.style,
                "industry": gen.industry,
                "strength": gen.strength,
                "input_image": image_payload(gen.input_image),
                "output_image": image_payload(gen.output_image),
                "timestamp": gen.created_at.isoformat(),
            }
            for gen in generations
        ]

    def get_full_generation_history(
        self, db: Session, session_id: str
    ) -> List[Dict]:
        """전체 생성 이력 조회"""
        from src.backend import process_db

        generations = process_db.get_generation_history_by_session(
            db, session_id, limit=None
        )

        return [
            {
                "id": gen.id,
                "content_type": gen.content_type,
                "input_text": gen.input_text,
                "output_text": gen.output_text,
                "prompt": gen.prompt,
                "style": gen.style,
                "industry": gen.industry,
                "strength": gen.strength,
                "input_image": image_payload(gen.input_image),
                "output_image": image_payload(gen.output_image),
                "timestamp": gen.created_at.isoformat(),
            }
            for gen in generations
        ]


# =====================================================
# LLMOrchestrator: LLM 호출 관리
# =====================================================

class LLMOrchestrator:
    """LLM 호출 관리 클래스"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model


    async def analyze_intent(
        self,
        user_message: str,
        recent_conversations: Optional[List[Dict]] = None
    ) -> Dict:
        """
        의도 분석 + 생성 파라미터 자동 결정
        
        Args:
            user_message: 현재 사용자 메시지
            recent_conversations: 최근 대화 (선택, 맥락 파악용)
        
        Returns:
        {
            "intent": "generation|modification|consulting",
            "confidence": 0.0-1.0,
            "generation_type": "image|text|null",
            "aspect_ratio": "1:1|16:9|9:16|4:3",
            "style": "ultra_realistic|semi_realistic|anime",
            "need_rmbg": true|false,
            "strength": 0.0-1.0 or null
        }
        """

        system_prompt = """
당신은 광고 제작 어시스턴트입니다. 사용자 메시지를 분석하여 의도와 생성 파라미터를 파악하세요.

## 1. 의도 분류
- generation: 새로운 광고를 만들고 싶어하는 경우
- modification: 기존 광고를 수정하고 싶어하는 경우
- consulting: 광고 제작 방법이나 조언을 구하는 경우

## 2. 생성 타입 결정 (intent=generation일 때)
- image: 이미지/사진/그림/배너/포스터/피드 등 시각적 콘텐츠
- text: 광고 문구/카피/슬로건만 필요한 경우

## 3. 수정 강도(strength) 감지 (intent=modification일 때)
사용자의 수정 요청 표현을 분석하여 0.0~1.0 범위의 strength 값을 결정:

### 약한 수정 (strength: 0.3~0.4)
표현: "살짝", "약간", "조금", "미세하게", "아주 조금만"
예시: "색상 살짝만 밝게", "약간 더 따뜻한 느낌으로"
→ strength: 0.35

### 보통 수정 (strength: 0.5~0.6)
표현: "좀", "적당히", "어느정도", 수정 정도를 명시하지 않은 일반적 표현
예시: "밝게 해줘", "색상 바꿔줘", "분위기 따뜻하게"
→ strength: 0.55

### 강한 수정 (strength: 0.7~0.8)
표현: "확실하게", "많이", "크게", "완전히", "대폭"
예시: "훨씬 밝게", "색상 완전히 바꿔줘", "분위기 대폭 변경"
→ strength: 0.75

### 매우 강한 수정 (strength: 0.85~0.95)
표현: "완전 다르게", "전체적으로", "처음부터 다시", "거의 새로 만들 듯이"
예시: "완전히 다른 느낌으로", "전체 분위기를 바꿔줘"
→ strength: 0.9

수정 종류에 따른 기본값:
- 색상 조정 (밝기, 채도, 색온도): 0.5~0.6
- 부분 수정 (특정 요소만): 0.4~0.5
- 스타일/분위기 변경: 0.6~0.8
- 구도/레이아웃 변경: 0.7~0.9

명시적 표현이 없으면 수정 종류의 기본값 사용

## 3.5 배경 제거 요청 감지 (need_rmbg)
- 사용자가 "배경 제거", "누끼", "배경 없애줘", "투명 배경" 등을 명시하면 need_rmbg: true
- 그렇지 않으면 need_rmbg: false

## 4. 플랫폼 감지 및 비율 결정
플랫폼별 기본 비율:
- instagram, 인스타그램, 인스타, 피드, 인스타피드: platform="instagram", aspect_ratio="1:1"
- youtube, 유튜브, 썸네일, 배너: platform="youtube", aspect_ratio="16:9"
- story, 스토리, 릴스, 인스타스토리, 인스타릴스: platform="story", aspect_ratio="9:16"
- facebook, 페이스북, 페북: platform="facebook", aspect_ratio="4:3"
- naver, 네이버, 스마트스토어, 네이버지도: platform="naver", aspect_ratio="1:1"
- 당근, 당근마켓: platform="carrot", aspect_ratio="1:1"
- 배달, 배민, 배달의민족, 쿠팡이츠: platform="delivery", aspect_ratio="16:9"
- banner, 배너, 웹배너, 프로모션배너: platform="banner", aspect_ratio="16:9"
- poster, 포스터, 전단지, 리플렛, flyer: platform="poster", aspect_ratio="4:3"

명시적 비율 지정이 있으면 우선:
- "정사각형", "1대1", "1:1" → "1:1"
- "가로", "16:9" → "16:9"
- "세로", "9:16" → "9:16"

플랫폼이 명확하지 않으면 platform=null, aspect_ratio=null

## 5. 스타일 자동 결정
업종/분위기 키워드 기반:

### ultra_realistic (고급/프로페셔널)
- 키워드: 고급, 프리미엄, 레스토랑, 호텔, 리조트, 의료, 법률, 금융, 부동산
- 음식 사진 (음식, 요리, 메뉴, 식당)
- 제품 사진 (명품, 럭셔리, 고급 제품)

### semi_realistic (친근/웜톤)
- 키워드: 카페, 베이커리, 디저트, 일상, 따뜻한, 친근한
- 소규모 로컬 비즈니스
- 힐링, 감성, 아늑한 분위기

### anime (캐주얼/브랜드/이벤트)
- 키워드: 브랜드, 패션, 이벤트, 공지, 할인, 프로모션
- 젊은층 타겟, 캐주얼한 분위기
- 템플릿, 포스터, 안내문
- 캐릭터, 귀여운, 일러스트

사용자가 명시적으로 스타일을 지정하면 우선 적용
스타일이 명확하지 않으면 style=null (기본값 사용)

반드시 JSON 객체만 출력하고, 설명/코드펜스/추가 텍스트는 금지.

## JSON 응답 형식
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "generation_type": "image|text|null",
  "aspect_ratio": "1:1|16:9|9:16|4:3" or null,
  "style": "ultra_realistic|semi_realistic|anime" or null,
  "need_rmbg": true|false,
  "strength": 0.0-1.0 or null
}

## 분석 예시

입력: "인스타 피드용 카페 광고 만들어줘"
출력:
{
  "intent": "generation",
  "confidence": 0.95,
  "generation_type": "image",
  "aspect_ratio": "1:1",
  "style": "semi_realistic",
  "need_rmbg": false,
  "strength": null
}

입력: "유튜브 썸네일로 쓸 고급 레스토랑 이미지"
출력:
{
  "intent": "generation",
  "confidence": 0.95,
  "generation_type": "image",
  "aspect_ratio": "16:9",
  "style": "ultra_realistic",
  "need_rmbg": false,
  "strength": null
}

입력: "인스타 스토리용 할인 이벤트 포스터"
출력:
{
  "intent": "generation",
  "confidence": 0.95,
  "generation_type": "image",
  "aspect_ratio": "9:16",
  "style": "anime",
  "need_rmbg": false,
  "strength": null
}

입력: "방금 만든 거 색상 좀 밝게 해줘"
출력:
{
  "intent": "modification",
  "confidence": 0.9,
  "generation_type": "image",
  "aspect_ratio": null,
  "style": null,
  "need_rmbg": false,
  "strength": 0.55
}

입력: "이미지 살짝만 더 따뜻한 느낌으로 바꿔줘"
출력:
{
  "intent": "modification",
  "confidence": 0.95,
  "generation_type": "image",
  "aspect_ratio": null,
  "style": null,
  "need_rmbg": false,
  "strength": 0.35
}

입력: "전체적으로 분위기를 완전히 다르게 만들어줘"
출력:
{
  "intent": "modification",
  "confidence": 0.95,
  "generation_type": "image",
  "aspect_ratio": null,
  "style": null,
  "need_rmbg": false,
  "strength": 0.9
}

입력: "배경만 좀 바꿔줘"
출력:
{
  "intent": "modification",
  "confidence": 0.85,
  "generation_type": "image",
  "aspect_ratio": null,
  "style": null,
  "need_rmbg": false,
  "strength": 0.45
}

입력: "카페 광고 문구 한 줄로 써줘"
출력:
{
  "intent": "generation",
  "confidence": 0.9,
  "generation_type": "text",
  "aspect_ratio": null,
  "style": null,
  "need_rmbg": false,
  "strength": null
}
"""

        # 최근 대화만 컨텍스트로 사용 (선택적)
        context_str = ""
        if recent_conversations:
            conv_summary = "\n".join([
                f"{msg['role']}: {(msg.get('content') or '')[:200]}"
                for msg in recent_conversations[-3:]  # 3개만으로 충분
            ])
            context_str = f"\n\n최근 대화 (참고용):\n{conv_summary}"

        user_prompt = f"사용자 메시지: {user_message}{context_str}"

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            logger.info(f"Intent analysis raw response: {content}")

            result = _extract_json_dict(content)
            if result is None:
                logger.warning("Intent analysis JSON parse failed")
                return {
                    "intent": "consulting",
                    "confidence": 0.5,
                    "generation_type": None,
                    "aspect_ratio": None,
                    "style": None,
                    "need_rmbg": False,
                    "strength": None,
                }

            # 기본값 처리
            intent = result.get("intent")
            if isinstance(intent, str):
                intent = intent.strip()
            if intent not in ("generation", "modification", "consulting"):
                intent = "consulting"

            confidence = result.get("confidence", 0.5)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.5

            gen_type = result.get("generation_type")
            if isinstance(gen_type, str):
                gen_type = gen_type.strip()
            if gen_type not in ("image", "text"):
                gen_type = None

            # 플랫폼/비율/스타일/업종/strength 추출
            aspect_ratio = result.get("aspect_ratio")
            style = result.get("style")
            strength = result.get("strength")
            need_rmbg = result.get("need_rmbg")
            # 유효성 검사
            valid_aspect_ratios = ["1:1", "16:9", "9:16", "4:3"]
            if isinstance(aspect_ratio, str):
                aspect_ratio = aspect_ratio.strip()
            if aspect_ratio not in valid_aspect_ratios:
                aspect_ratio = None

            valid_styles = ["ultra_realistic", "semi_realistic", "anime"]
            if isinstance(style, str):
                style = style.strip()
            if style not in valid_styles:
                style = None

            if strength is not None:
                try:
                    strength = float(strength)
                except (TypeError, ValueError):
                    strength = None
            if strength is not None and not (0.0 <= strength <= 1.0):
                strength = None

            if isinstance(need_rmbg, str):
                normalized = need_rmbg.strip().lower()
                if normalized in ("true", "yes", "y", "1"):
                    need_rmbg = True
                elif normalized in ("false", "no", "n", "0"):
                    need_rmbg = False
                else:
                    need_rmbg = None
            if isinstance(need_rmbg, (int, float)) and not isinstance(need_rmbg, bool):
                need_rmbg = bool(need_rmbg)
            if not isinstance(need_rmbg, bool):
                need_rmbg = False

            if intent == "modification" and gen_type in ("text", None):
                msg_lower = user_message.lower()
                image_keywords = [
                    "image", "photo", "picture", "background", "color", "font", "text",
                    "이미지", "사진", "그림", "배경", "색상", "색깔", "폰트", "글씨",
                    "포스터", "배너", "구도", "레이아웃",
                ]
                if any(keyword in msg_lower for keyword in image_keywords):
                    gen_type = "image"

            if aspect_ratio is None:
                msg_lower = user_message.lower()
                poster_keywords = ["poster", "포스터", "전단지", "리플렛", "flyer"]
                banner_keywords = ["banner", "배너", "웹배너", "프로모션배너"]
                if any(k in msg_lower for k in poster_keywords):
                    aspect_ratio = "4:3"
                elif any(k in msg_lower for k in banner_keywords):
                    aspect_ratio = "16:9"

            if gen_type != "image":
                need_rmbg = False

            logger.info(
                f"Intent: {intent}, type: {gen_type}, strength: {strength}, "
                f"ratio: {aspect_ratio}, style: {style}, need_rmbg: {need_rmbg}"
            )

            return {
                "intent": intent,
                "confidence": confidence,
                "generation_type": gen_type,
                "aspect_ratio": aspect_ratio,
                "style": style,
                "need_rmbg": need_rmbg,
                "strength": strength,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return {
                "intent": "consulting",
                "confidence": 0.5,
                "generation_type": None,
                "aspect_ratio": None,
                "style": None,
                "need_rmbg": False,
                "strength": None,
            }
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}", exc_info=True)
            return {
                "intent": "consulting",
                "confidence": 0.5,
                "generation_type": None,
                "aspect_ratio": None,
                "style": None,
                "need_rmbg": False,
                "strength": None,
            }


    async def refine_generation_input(
        self,
        *,
        intent: str,
        generation_type: Optional[str],
        chat_history: List[Dict],
        generation_history: List[Dict],
    ) -> Dict:
        """
        전체 맥락을 반영해 생성 입력을 정제하고 수정 대상 ID 찾기
        
        Returns:
            {
                "refined_input": str,  # 정제된 입력 텍스트
                "target_generation_id": int or None,  # 수정 대상 ID (modification일 때만)
                "is_text_modification": bool  # 텍스트만 수정 요청 여부
            }
        """

        system_prompt = """
# Role: Ad Request Context Analyzer
당신은 사용자의 발화와 생성 이력을 분석하여, 1) 수정 대상이 되는 특정 생성물의 ID를 찾고, 2) 최종적으로 반영해야 할 정제된 요청 사항(refined_input)을 도출하는 전문가입니다.

---

# Input Data Structure
1. **Dialogue History:** 사용자와의 대화 내용
2. **Generation History (List):**
   - `id`: 고유 ID
   - `recent_rank`: 1 (최신) ~ N (가장 오래됨)
   - `chrono_rank`: 1 (가장 오래됨) ~ N (최신)
   - `input_text`: 당시 요청 내용
   - `is_image`: Y/N

---

# Task 1: Target Identification Logic (수정 대상 찾기)

사용자의 발화가 `intent=modification`일 때, 아래 우선순위 로직에 따라 단 **하나의 target_id**를 결정하십시오.

### [우선순위 판단 로직]
1. **명시적 순서 지칭 (Chronological):**
   - "처음", "첫 번째", "맨 처음" → `chrono_rank=1`
   - "두 번째", "세 번째" → `chrono_rank=n`
2. **상대적 순서 지칭 (Recency):**
   - "방금", "마지막", "이거", "지금 나온 거", "최근" → `recent_rank=1`
   - "이전 거", "아까 거", "그 전 거" → `recent_rank=2`
3. **내용 기반 지칭 (Content Matching):**
   - "파란색 배경인 거", "고양이 나온 거" → `input_text`나 맥락상 해당 속성을 가진 항목 중 가장 최신(`recent_rank`가 낮은) 항목
4. **대명사 (Implicit):**
   - "그거", "저거" → 직전 대화에서 특정 이미지를 언급했다면 그 ID, 없다면 `recent_rank=1`
5. **Fallback (예외 처리):**
   - 사용자가 대상을 전혀 특정하지 않고 수정만 요청함 ("글자만 좀 바꿔줘") → `recent_rank=1`

**주의:** 사용자가 "새로 만들어줘", "다시 해봐" 등 `generation` 의도를 보이면 `target_generation_id`는 `null`입니다.

---

# Task 2: Input Refinement Logic (요구사항 정제)

단순히 사용자의 마지막 말만 번역하지 말고, **(기존 문맥 + 새로운 요청)**을 통합하여 **정제된 요구사항(spec)**을 만드십시오.

1. **상속 (Inheritance):** 선택된 `target_generation_id`의 `input_text`를 베이스로 가져옵니다. (target이 없으면 대화 내 제품 정보가 베이스)
2. **수정 (Modification):** 사용자의 새로운 요청(추가/삭제/변경)을 반영합니다.
   - "A 빼줘" → A 삭제
   - "B를 C로 바꿔줘" → B를 C로 치환
3. **구체화 (Specification):** 대명사("그 문구")나 모호한 표현을 구체적인 값으로 치환합니다.
4. **형식 (Output Format):** 문장 형태보다는, 이미지 생성/광고 문구 생성에 즉시 투입 가능한 **요구사항 형태**로 요약합니다.

추가 원칙:
- **현재 사용자 요청이 우선**입니다. 이전 지시사항은 기본값으로만 유지하고, 현재 요청과 충돌하면 제거/수정하십시오.
- 사용자가 특정 처리(예: 어떤 단계나 효과)를 **하지 않겠다는 의도**를 보이면, 이전에 포함되었더라도 refined_input에서 제외하십시오.
- 사용자가 그 부분을 **명시적으로 유지/변경**하는 경우에만 refined_input에 남기십시오.
- 사용자가 **직접 제시한 문구/문장**은 누락하지 말고 refined_input에 **그대로 포함**하십시오.
- 사용자가 이미지에 텍스트를 넣어달라고 하면, **넣어야 할 정확한 텍스트**를 refined_input에 명확히 반영하십시오.
- 해당 텍스트가 **현재 메시지에 없고 이전 대화/생성 이력에만 있는 경우**, 문맥(지시어·대명사)을 근거로 **가장 관련 있는 텍스트를 찾아** refined_input에 포함하십시오.
- refined_input에는 **최종 광고 문구를 완성형으로 작성하지 마십시오.**
  - 예: "엄마의 정성이..." 같은 완성 문장은 금지
  - 대신: "문구는 따뜻한 톤, 길게, 해시태그 포함"처럼 **요구사항(spec)**으로 작성
- 길이 기대치는 사용자의 표현을 사람 기준으로 해석해 포함하십시오.
  - 예: "길게" → "약 400~600자 수준"
  - 예: "짧게" → "약 20~40자 수준"
- 해시태그 요청이 있으면 **문장 끝에 해시태그 N개 포함**을 요구사항에 명시하십시오.


---

---

# Task 3: Text-only Modification Detection
사용자가 **텍스트만** 수정하겠다는 의도가 명확한지 판단하십시오.
아래 조건에 해당하면 `is_text_modification = true`:
- "문구만", "글자만", "텍스트만", "카피만" 등 텍스트 수정만 명시
- "이미지는 그대로", "배경은 그대로"처럼 이미지 변경 제외를 명시

그 외에는 `is_text_modification = false`.

Output Format (JSON Only)
반드시 아래 JSON 포맷으로만 응답하며, 마크다운 코드 블록(``json)을 사용하지 마십시오.
**reasoning` 필드에 당신의 판단 근거를 먼저 작성해야 정확도가 올라갑니다.**

{{
  "reasoning": "사용자가 '두 번째 거'라고 했으므로 chrono_rank=2인 ID 105를 선택함. 기존 요청 '시원한 느낌'에 '빨간색 강조'를 추가함.",
  "intent_type": "modification" | "generation",
  "target_generation_id": 123 | null,
  "refined_input": "여기에 정제된 텍스트 작성",
  "is_text_modification": true|false
}}


"""

        latest_user_message = ""
        for msg in reversed(chat_history or []):
            if msg.get("role") == "user":
                latest_user_message = (msg.get("content") or "").strip()
                break

        total_generations = len(generation_history or [])
        if total_generations:
            for idx, gen in enumerate(generation_history):
                gen["_chronological_rank"] = total_generations - idx

        condensed_chat_history, chat_summary = _summarize_chat_history(chat_history)
        condensed_generation_history, generation_summary = _summarize_generation_history(
            generation_history
        )
        condensed_image_history = [
            gen for gen in condensed_generation_history
            if gen.get("output_image") or gen.get("input_image") or gen.get("content_type") == "image"
        ]

        prompt = f"""
# 현재 사용자 메시지
{latest_user_message or "(없음)"}

# 이전 대화 요약
{chat_summary or "(없음)"}

# 전체 대화 히스토리
{_format_chat_history(condensed_chat_history)}

# 이전 생성 요약
{generation_summary or "(없음)"}

# 생성 이력 (recent_rank=1이 최신)
{_format_generation_history(condensed_generation_history)}

# 이미지 생성 이력 (이미지=Y만)
{_format_generation_history(condensed_image_history) if condensed_image_history else "(없음)"}

# 사용자 요청
intent: {intent}
generation_type: {generation_type or "null"}

# 작업
1. modification이면 target_generation_id 찾기
2. refined_input 작성

JSON 형식으로 응답:
{{"refined_input": "...", "target_generation_id": 123 or null, "is_text_modification": true|false}}
"""

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            content = response.choices[0].message.content or ""

            result = _extract_json_dict(content)
            if result is None:
                logger.warning("Refinement JSON parse failed: %s", content.strip())
                refined_input = content.strip() or latest_user_message
                return {
                    "refined_input": refined_input,
                    "target_generation_id": None,
                }

            refined_input = (result.get("refined_input") or "").strip()
            target_id = result.get("target_generation_id")
            is_text_modification = result.get("is_text_modification")

            # target_id 유효성 검사
            if isinstance(target_id, str) and target_id.isdigit():
                target_id = int(target_id)
            if not isinstance(target_id, int):
                target_id = None

            if isinstance(is_text_modification, str):
                normalized = is_text_modification.strip().lower()
                if normalized in ("true", "yes", "y", "1"):
                    is_text_modification = True
                elif normalized in ("false", "no", "n", "0"):
                    is_text_modification = False
                else:
                    is_text_modification = None
            if isinstance(is_text_modification, (int, float)) and not isinstance(is_text_modification, bool):
                is_text_modification = bool(is_text_modification)
            if not isinstance(is_text_modification, bool):
                is_text_modification = False

            # refined_input이 비어있으면 마지막 사용자 메시지 사용
            if not refined_input:
                refined_input = latest_user_message

            logger.info(
                "Refinement: original=%s, refined=%s, target_id=%s",
                latest_user_message,
                refined_input,
                target_id,
            )

            return {
                "refined_input": refined_input,
                "target_generation_id": target_id,
                "is_text_modification": is_text_modification,
            }

        except Exception as e:
            logger.error(f"Refinement failed: {e}", exc_info=True)

            return {
                "refined_input": latest_user_message,
                "target_generation_id": None,
                "is_text_modification": False,
            }

# =====================================================
# ConsultingService: 상담 전용 비즈니스 로직
# =====================================================

class ConsultingService:
    """상담 관련 비즈니스 로직"""

    def __init__(
        self,
        llm_orchestrator: LLMOrchestrator,
        conversation_manager: ConversationManager,
        rag: Optional[Any] = None,
    ):
        from src.generation.chat_bot.rag.prompts import SlotChecker, UserContext

        self.llm = llm_orchestrator
        self.conv = conversation_manager
        self.rag = rag
        self._consultant_bots: Dict[str, Any] = {}
        self.slot_checker = SlotChecker()
        self._session_contexts: Dict[str, UserContext] = {}

    def _get_consultant_bot(self, session_id: str):
        """세션별 SmallBizConsultant 인스턴스 반환 (슬롯 상태 분리)"""
        bot = self._consultant_bots.get(session_id)
        if bot is not None:
            return bot

        try:
            from src.generation.chat_bot.agent.agent import SmallBizConsultant
        except Exception as e:
            logger.error(f"SmallBizConsultant import failed: {e}", exc_info=True)
            return None

        try:
            bot = SmallBizConsultant(
                llm_model=self.llm.model,
                use_reranker=False,
                verbose=False,
            )
            self._consultant_bots[session_id] = bot
            logger.info(f"SmallBizConsultant initialized for session={session_id}")
            return bot
        except Exception as e:
            logger.error(f"SmallBizConsultant init failed: {e}", exc_info=True)
            return None

    def build_context(
        self, db: Session, session_id: str, message: str, recent_limit: int = 5
    ) -> Dict:
        """상담에 필요한 컨텍스트 구성"""
        recent_conversations = self.conv.get_recent_messages(
            db, session_id, limit=recent_limit
        )
        generation_history = self.conv.get_generation_history(db, session_id, limit=5)

        context = {
            "recent_conversations": recent_conversations,
            "generation_history": generation_history,
        }

        return context

    def _build_rag_filter(self, user_context) -> Optional[Dict[str, str]]:
        clauses: List[Dict[str, str]] = []
        if user_context and user_context.industry:
            clauses.append({"industry": user_context.industry})
        if user_context and user_context.location:
            clauses.append({"location": user_context.location})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _build_user_context(
        self,
        session_id: str,
        recent_conversations: List[Dict],
    ):
        user_context = self._session_contexts.get(session_id)
        if user_context is None:
            from src.generation.chat_bot.rag.prompts import UserContext

            user_context = UserContext()
            self._session_contexts[session_id] = user_context

        for msg in recent_conversations:
            if msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if content:
                    user_context = self.slot_checker.update_context_from_query(
                        content, user_context
                    )

        self._session_contexts[session_id] = user_context
        return user_context

    async def _stream_consulting_answer(
        self,
        message: str,
        recent_conversations: List[Dict],
        session_id: str,
    ) -> AsyncIterator[str]:
        chat_history = [
            {"role": msg.get("role"), "content": msg.get("content")}
            for msg in recent_conversations
            if msg.get("role") and msg.get("content") is not None
        ]
        user_context = self._build_user_context(session_id, recent_conversations)
        task = self.rag.prompt_builder.classify_task(message)
        filter_kwargs = self._build_rag_filter(user_context)
        retrieved_docs = self.rag.retrieve(message, k=5, filter=filter_kwargs)

        async for chunk in self.rag.generate_stream(
            query=message,
            retrieved_docs=retrieved_docs,
            task=task,
            user_context=user_context,
            chat_history=chat_history,
        ):
            yield chunk

    async def stream_response(
        self, db: Session, session_id: str, message: str
    ) -> AsyncIterator[Dict]:
        """상담 응답을 스트리밍으로 생성하며 청크/완료 이벤트를 반환"""
        bot = self._get_consultant_bot(session_id)
        assistant_chunks: List[str] = []
        assistant_message = ""

        if bot is not None:
            context = self.build_context(db, session_id, message)
            recent_conversations = context.get("recent_conversations", [])
            user_context = self._build_user_context(session_id, recent_conversations)
            result = bot.consult(query=message, user_context=user_context)
            assistant_message = (result.get("answer") or "").strip() or "무엇을 도와드릴까요?"
            yield {"type": "chunk", "content": assistant_message}
        else:
            context = self.build_context(db, session_id, message)
            recent_conversations = context.get("recent_conversations", [])

            if self.rag is not None:
                try:
                    async for chunk in self._stream_consulting_answer(
                        message,
                        recent_conversations,
                        session_id,
                    ):
                        if chunk:
                            assistant_chunks.append(chunk)
                            yield {"type": "chunk", "content": chunk}
                except Exception as e:
                    logger.error(f"Consulting RAG response failed: {e}", exc_info=True)

            if not assistant_chunks and self.llm is not None:
                async for chunk in self.llm.stream_consulting_response(message, context):
                    if chunk:
                        assistant_chunks.append(chunk)
                        yield {"type": "chunk", "content": chunk}

            assistant_message = "".join(assistant_chunks).strip() or "무엇을 도와드릴까요?"

        self.conv.add_message(db, session_id, "assistant", assistant_message)

        yield {
            "type": "done",
            "session_id": session_id,
            "intent": "consulting",
            "assistant_message": assistant_message,
        }


# =====================================================
# 싱글톤 인스턴스
# =====================================================

_conversation_manager: Optional[ConversationManager] = None
_llm_orchestrator: Optional[LLMOrchestrator] = None
_consulting_service: Optional[ConsultingService] = None
_consulting_rag = None


def get_conversation_manager() -> ConversationManager:
    """ConversationManager 싱글톤"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


def get_llm_orchestrator() -> LLMOrchestrator:
    """LLMOrchestrator 싱글톤"""
    global _llm_orchestrator
    if _llm_orchestrator is None:
        from src.utils.config import settings
        _llm_orchestrator = LLMOrchestrator(api_key=settings.OPENAI_API_KEY)
    return _llm_orchestrator


def get_consulting_service() -> ConsultingService:
    """ConsultingService 싱글톤"""
    global _consulting_service
    if _consulting_service is None:
        llm = get_llm_orchestrator()
        conv = get_conversation_manager()
        rag = get_consulting_rag()
        _consulting_service = ConsultingService(llm, conv, rag=rag)
    return _consulting_service


def get_consulting_rag():
    """SmallBizRAG 싱글톤 (consulting 전용)"""
    global _consulting_rag
    if _consulting_rag is None:
        from src.generation.chat_bot.rag.chain import SmallBizRAG

        _consulting_rag = SmallBizRAG(use_reranker=False)
    return _consulting_rag
