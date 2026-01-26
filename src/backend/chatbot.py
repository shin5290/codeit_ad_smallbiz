"""
RAG 기반 챗봇 모듈

역할 분리:
- ConversationManager: DB 접근 레이어 (대화/생성 이력 CRUD)
- LLMOrchestrator: LLM 호출 레이어 (intent 분석, refinement, consulting 응답)
- ConsultingService: 상담 전용 비즈니스 로직
"""

import json, openai, re
from typing import Any, Optional, List, Dict, AsyncIterator
from sqlalchemy.orm import Session

from src.backend.consulting_knowledge_base import ConsultingKnowledgeBase
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
        
        lines.append(
            f"recent_rank={idx}, ID={gen_id}, 타입={content_type}, "
            f"입력=\"{input_text[:100]}\", 시간={timestamp}"
        )
    
    return "\n".join(lines)


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
        logger.info(f"ConversationManager: saved message id={chat_row.id}")
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
        generations_reversed = list(reversed(generations))

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
            for gen in generations_reversed
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
            "industry": "cafe|restaurant|fashion|...",
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

## 6. 업종 감지
- cafe: 카페, 커피, 카페테리아
- restaurant: 레스토랑, 음식점, 식당
- bakery: 베이커리, 빵집, 제과점
- fashion: 패션, 의류, 옷
- beauty: 뷰티, 미용, 화장품
- event: 이벤트, 행사, 프로모션
- 기타 명확한 업종이 있으면 영문으로
- 업종 불명확 시 general로 출력

## 7. 텍스트 생성 파라미터 (generation_type=text일 때)
톤(text_tone): warm|professional|friendly|energetic 중 하나
- 사용자가 톤을 직접 지정하면 우선 적용
- "전문적", "신뢰", "격식", "고급" → professional
- "친근", "편안", "부드럽게" → friendly
- "활기", "에너지", "역동" → energetic
- 그 외는 warm

길이(text_max_length): 숫자 (10~200)
- "한 줄", "짧게", "슬로건", "캐치프레이즈" → 15~30
- "제목", "헤드라인" → 30~40
- "소개", "설명", "상세" → 80~120
- 배너/썸네일 → 15~25, 포스터 → 30~50, 인스타 피드 → 20~40
- 사용자가 "50자" 등 명시하면 그대로 반영

반드시 JSON 객체만 출력하고, 설명/코드펜스/추가 텍스트는 금지.

## JSON 응답 형식
{
  "intent": "generation|modification|consulting",
  "confidence": 0.0-1.0,
  "generation_type": "image|text|null",
  "aspect_ratio": "1:1|16:9|9:16|4:3" or null,
  "style": "ultra_realistic|semi_realistic|anime" or null,
  "industry": "cafe|restaurant|..." or null,
  "strength": 0.0-1.0 or null,
  "text_tone": "warm|professional|friendly|energetic" or null,
  "text_max_length": 10-200 or null
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
  "industry": "cafe",
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
  "industry": "restaurant",
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
  "industry": "event",
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
  "industry": null,
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
  "industry": null,
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
  "industry": null,
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
  "industry": null,
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
  "industry": "cafe",
  "strength": null,
  "text_tone": "warm",
  "text_max_length": 20
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
                    "industry": None,
                    "strength": None,
                    "text_tone": None,
                    "text_max_length": None,
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
            industry = result.get("industry")
            strength = result.get("strength")
            text_tone = result.get("text_tone")
            text_max_length = result.get("text_max_length")

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
            if isinstance(industry, str):
                industry = industry.strip() or None

            if strength is not None:
                try:
                    strength = float(strength)
                except (TypeError, ValueError):
                    strength = None
            if strength is not None and not (0.0 <= strength <= 1.0):
                strength = None

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

            valid_tones = ["warm", "professional", "friendly", "energetic"]
            if isinstance(text_tone, str):
                text_tone = text_tone.strip()
            if text_tone not in valid_tones:
                text_tone = None

            if text_max_length is not None:
                try:
                    text_max_length = int(text_max_length)
                except (TypeError, ValueError):
                    text_max_length = None
            if text_max_length is not None:
                if text_max_length < 10:
                    text_max_length = 10
                elif text_max_length > 200:
                    text_max_length = 200

            if gen_type != "text":
                text_tone = None
                text_max_length = None

            logger.info(
                f"Intent: {intent}, type: {gen_type}, strength: {strength}, "
                f"ratio: {aspect_ratio}, style: {style}, industry: {industry}, "
                f"text_tone: {text_tone}, text_max_length: {text_max_length}"
            )

            return {
                "intent": intent,
                "confidence": confidence,
                "generation_type": gen_type,
                "aspect_ratio": aspect_ratio,
                "style": style,
                "industry": industry,
                "strength": strength,
                "text_tone": text_tone,
                "text_max_length": text_max_length,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return {
                "intent": "consulting",
                "confidence": 0.5,
                "generation_type": None,
                "aspect_ratio": None,
                "style": None,
                "industry": None,
                "strength": None,
                "text_tone": None,
                "text_max_length": None,
            }
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}", exc_info=True)
            return {
                "intent": "consulting",
                "confidence": 0.5,
                "generation_type": None,
                "aspect_ratio": None,
                "style": None,
                "industry": None,
                "strength": None,
                "text_tone": None,
                "text_max_length": None,
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
                "target_generation_id": int or None  # 수정 대상 ID (modification일 때만)
            }
        """

        system_prompt = """
당신은 광고 생성 요청을 정제하고 수정 대상을 찾는 맥락 분석기입니다.

## 역할 1: 수정 대상 특정 (intent=modification일 때만)

생성 이력 정보:
- recent_rank=1이 가장 최신
- 각 항목에 id, content_type, input_text, timestamp가 있음

대상 선택 규칙:
1. "방금", "최근", "마지막", "지금" → recent_rank=1 선택
2. "2번", "두 번째" → recent_rank=2 선택
3. "첫 번째", "처음", "맨 처음" → 가장 오래된 것 (마지막 rank)
4. "그거", "저거" → 대화 맥락에서 가장 관련 있는 것
5. 명확하지 않으면 recent_rank=1 (최신)

## 역할 2: 입력 텍스트 정제

### 번호 참조 해석
- "2번으로 해줘" → 대화에서 2번 찾아서 실제 내용으로 치환
- Assistant의 consulting 응답(추천 문구 등)도 확인

### 대명사 해석
- "그거", "저거", "그 문구" → 구체적 대상으로 치환

### modification 요청 처리
- **절대 이전 생성물을 복사하지 말 것**
- 다음을 요약:
  1) 처음 생성 요청 내용
  2) 제품/서비스 정보
  3) 누적된 모든 수정 요구사항

예시:
대화:
- User: "네이버 스토어 소개글 써줘. 행주 파는 곳, 기름 안 묻음"
- Assistant: "대나무 행주로 깨끗한 주방"
- User: "행주라는 단어 넣어줘"
- Assistant: "대나무 행주로 깨끗한 주방"
- User: "주방이라는 단어는 빼줘"

refined_input: "네이버 스토어 소개글. 대나무 행주 판매(기름 안 묻음). '행주' 단어 포함 필수, '주방' 단어 제외"

### generation 요청 처리
- 대화에서 언급된 모든 관련 정보 요약
- 이전 consulting 응답에서 선택한 내용 포함
- 구체적이고 실행 가능한 명령으로 작성

## JSON 응답 형식
{
  "refined_input": "정제된 입력 텍스트",
  "target_generation_id": 123 or null
}

target_generation_id는 modification일 때만 필요, generation이면 null
"""

        prompt = f"""
# 전체 대화 히스토리
{_format_chat_history(chat_history)}

# 생성 이력 (recent_rank=1이 최신)
{_format_generation_history(generation_history)}

# 사용자 요청
intent: {intent}
generation_type: {generation_type or "null"}

# 작업
1. modification이면 target_generation_id 찾기
2. refined_input 작성

JSON 형식으로 응답:
{{"refined_input": "...", "target_generation_id": 123 or null}}
"""

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content or ""
            
            # JSON 파싱
            try:
                result = json.loads(content.strip())
                refined_input = result.get("refined_input", "").strip()
                target_id = result.get("target_generation_id")
                
                # target_id 유효성 검사
                if isinstance(target_id, str) and target_id.isdigit():
                    target_id = int(target_id)
                if not isinstance(target_id, int):
                    target_id = None
                
                # refined_input이 비어있으면 마지막 사용자 메시지 사용
                if not refined_input:
                    for msg in reversed(chat_history or []):
                        if msg.get("role") == "user":
                            refined_input = msg.get("content", "").strip()
                            break
                
                logger.info(
                    f"Refinement: input={refined_input[:100]}, target_id={target_id}"
                )
                
                return {
                    "refined_input": refined_input,
                    "target_generation_id": target_id,
                }
                
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트 그대로 사용
                refined_input = content.strip()
                if not refined_input:
                    for msg in reversed(chat_history or []):
                        if msg.get("role") == "user":
                            refined_input = msg.get("content", "").strip()
                            break
                
                return {
                    "refined_input": refined_input,
                    "target_generation_id": None,
                }

        except Exception as e:
            logger.error(f"Refinement failed: {e}", exc_info=True)
            
            # 폴백: 마지막 사용자 메시지
            fallback = ""
            for msg in reversed(chat_history or []):
                if msg.get("role") == "user":
                    fallback = msg.get("content", "").strip()
                    break
            
            return {
                "refined_input": fallback,
                "target_generation_id": None,
            }



    async def stream_consulting_response(
        self, user_message: str, context: Dict
    ) -> AsyncIterator[str]:
        """상담 응답 스트리밍"""
        import openai

        system_prompt = """당신은 광고 제작 전문 상담사입니다."""

        recent_conversations = context.get("recent_conversations", [])
        conv_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_conversations[-5:]
        ])

        user_prompt = f"""
사용자 질문: "{user_message}"

최근 대화:
{conv_str}
"""

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        except Exception as e:
            logger.error(f"Consulting stream failed: {e}", exc_info=True)
            yield "죄송합니다. 일시적인 오류가 발생했습니다."


# =====================================================
# ConsultingService: 상담 전용 비즈니스 로직
# =====================================================

class ConsultingService:
    """상담 관련 비즈니스 로직"""

    def __init__(
        self,
        llm_orchestrator: LLMOrchestrator,
        conversation_manager: ConversationManager,
        knowledge_base: Optional[ConsultingKnowledgeBase] = None,
    ):
        self.llm = llm_orchestrator
        self.conv = conversation_manager
        self.knowledge = knowledge_base
        self._consultant_bots: Dict[str, Any] = {}

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
            "knowledge_base": [],
        }

        if self.knowledge:
            try:
                knowledge_results = self.knowledge.search(
                    query=message, category="faq", limit=3
                )
                context["knowledge_base"] = knowledge_results
                logger.info(f"Knowledge search returned {len(knowledge_results)} results")
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")

        return context

    async def stream_response(
        self, db: Session, session_id: str, message: str
    ) -> AsyncIterator[Dict]:
        """상담 응답을 스트리밍으로 생성하며 청크/완료 이벤트를 반환"""
        bot = self._get_consultant_bot(session_id)

        if bot is None:
            context = self.build_context(db, session_id, message)
            assistant_chunks: List[str] = []
            async for chunk in self.llm.stream_consulting_response(message, context):
                if chunk:
                    assistant_chunks.append(chunk)
                    yield {"type": "chunk", "content": chunk}
            assistant_message = "".join(assistant_chunks).strip() or "무엇을 도와드릴까요?"
        else:
            result = bot.consult(query=message, user_context=None)
            assistant_message = (result.get("answer") or "").strip() or "무엇을 도와드릴까요?"
            yield {"type": "chunk", "content": assistant_message}

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
        from src.backend.consulting_knowledge_base import get_knowledge_base
        
        llm = get_llm_orchestrator()
        conv = get_conversation_manager()
        knowledge = get_knowledge_base()
        
        _consulting_service = ConsultingService(llm, conv, knowledge)
    return _consulting_service
