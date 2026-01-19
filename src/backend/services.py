import logging, os, re
from dataclasses import dataclass
from fastapi import Cookie, Depends, HTTPException, Response, UploadFile
from jose import JWTError
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, AsyncIterator

from src.backend import process_db, schemas, models
from src.backend.chatbot import get_chatbot
from src.utils.config import PROJECT_ROOT
from src.utils.security import verify_password, create_access_token, decode_token
from src.utils.session import normalize_session_id, ensure_chat_session
from src.utils.image import save_uploaded_image, load_image_from_payload
from src.generation.text_generation.text_generator import TextGenerator
from src.generation.image_generation.generator import generate_and_save_image


logger = logging.getLogger(__name__)
_TEXT_GENERATOR = None  # 싱글톤 인스턴스

def get_text_generator() -> TextGenerator:
    """TextGenerator 싱글톤 인스턴스 반환"""
    global _TEXT_GENERATOR
    if _TEXT_GENERATOR is None:
        _TEXT_GENERATOR = TextGenerator()
    return _TEXT_GENERATOR

def get_current_user(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(process_db.get_db),
):
    """
    JWT token으로 현재 로그인 유저 정보 가져오기
    - 쿠키에 토큰이 없으면 None 반환
    """
    if not access_token:
        return None

    try:
        payload = decode_token(access_token)
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="토큰 오류")

    user = process_db.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")

    return user

def get_current_user_optional(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(process_db.get_db),
):
    """
    JWT 토큰이 유효하지 않아도 게스트로 처리
    - 토큰이 없거나 잘못된 경우 None 반환
    """
    if not access_token:
        return None
    try:
        payload = decode_token(access_token)
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        logger.info("get_current_user_optional: 토큰이 유효하지 않아 게스트로 처리")
        return None

    user = process_db.get_user_by_id(db, user_id)
    if not user:
        logger.info("get_current_user_optional: 사용자를 찾지 못해 게스트로 처리")
        return None

    return user

def register_user(db, signup: schemas.SignupRequest):
    """
    회원가입 서비스
    - 중복 아이디 체크
    - 비밀번호 해싱 및 저장
    - 사용자 정보 반환
    """
    if process_db.check_duplicate_id(db, signup.login_id):
        raise HTTPException(400, "이미 사용 중인 아이디입니다")

    user = process_db.create_user(
        db=db,
        login_id=signup.login_id,
        login_pw=signup.login_pw,
        name=signup.name,
    )
    return user


def authenticate_user(db, login_id: str, login_pw: str, response: Response) -> None:
    """
    사용자 인증(로그인) 서비스
    - 아이디 존재 여부 확인, 비밀번호 검증
    - JWT access 토큰을 HttpOnly 쿠키로 설정
    """
    user = process_db.get_user_by_login_id(db, login_id)
    if not user or not verify_password(login_pw, user.login_pw):
        raise HTTPException(400, "아이디 또는 비밀번호가 일치하지 않습니다.")

    token = create_access_token(str(user.user_id))
    logger.info(f"authenticate_user: 로그인 성공 user_id={user.user_id}, login_id={user.login_id}")
    secure_cookie = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )


def update_user(db: Session, current_user, update: schemas.UpdateUserRequest):
    """
    회원 정보 수정 서비스
    - 이름 변경
    - 비밀번호 변경
    - 모든 수정에 현재 비밀번호 검증 필요
    """
    # 현재 비밀번호 검증 (모든 수정에 필수)
    if not verify_password(update.current_password, current_user.login_pw):
        raise HTTPException(400, "현재 비밀번호가 올바르지 않습니다.")

    updated = process_db.update_user(
        db=db,
        user=current_user,
        name=update.name if update.name is not None else None,
        new_login_pw=update.new_password if update.new_password else None,
    )
    return updated


def delete_user(db: Session, current_user, login_pw: str):
    """
    회원 삭제 서비스
    - 현재 비밀번호 검증 후 삭제
    """
    if not verify_password(login_pw, current_user.login_pw):
        raise HTTPException(400, "비밀번호가 올바르지 않습니다.")
    process_db.delete_user(db, current_user)


async def _save_uploaded_image_payload(
    *,
    db: Session,
    image: Optional[UploadFile],
) -> tuple[Optional[int], Optional[dict]]:
    """
    업로드 이미지를 저장하고 
    DB 레코드 ID와 메타데이터(file_hash, file_directory)를 반환.
    """
    if not image:
        return None, None

    logger.info("save_uploaded_image_payload: 업로드 이미지 디스크 + DB 저장 시작")

    base_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
    image_data = await save_uploaded_image(image=image, base_dir=base_dir)
    if not image_data:
        logger.warning("save_uploaded_image_payload: save_uploaded_image가 None 반환")
        return None, None

    image_row = process_db.save_image_from_hash(
        db=db,
        file_hash=image_data["file_hash"],
        file_directory=image_data["file_directory"],
    )
    logger.info(f"save_uploaded_image_payload: 이미지 저장 완료 id={image_row.id}")
    return image_row.id, image_data


def _resolve_target_generation(
    *,
    db: Session,
    session_id: str,
    target_generation_id: Optional[int],
) -> Optional[models.GenerationHistory]:
    """
    세션 내 생성 이력을 조회해 타깃 생성물(또는 최신)을 반환.
    """
    def _snippet(text: Optional[str], limit: int = 80) -> str:
        """로그용으로 긴 텍스트를 축약."""
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}..."

    if target_generation_id:
        candidate = process_db.get_generation_by_session_and_id(
            db=db,
            session_id=session_id,
            generation_id=target_generation_id,
        )
        if candidate:
            logger.info(
                "resolve_target_generation: target_generation_id=%s 사용, input=%s",
                candidate.id,
                _snippet(candidate.input_text),
            )
            return candidate
        logger.warning(
            "resolve_target_generation: target_generation_id 없음, 최신 이력으로 대체"
        )

    latest = process_db.get_latest_generation(db, session_id)
    if latest:
        logger.info(
            "resolve_target_generation: latest_generation_id=%s 사용, input=%s",
            latest.id,
            _snippet(latest.input_text),
        )
    return latest

@dataclass
class IngestResult:
    session_id: str
    chat_history_id: int
    input_image: Optional[dict]
    
async def ingest_user_message(
    *,
    db,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    image: Optional[UploadFile] = None,
) -> IngestResult:
    """
    입력 수집/저장 레이어
    - 세션 확보/생성/유저귀속
    - 유저 텍스트 저장
    - 업로드 이미지 디스크 저장 + DB 연결
    - 반환: session_key, chat_row_id(또는 chat_row), saved_payload(필요하면)
    """

    session_id = normalize_session_id(session_id) # 프론트에서 받아온 값 정규화
    logger.info(f"ingest_user_message: session_id={session_id}, user_id={user_id}, has_image={image is not None}")

    # 1) 세션 확보/생성/귀속
    session_key = ensure_chat_session(db, session_id, user_id)
    logger.info(f"ingest_user_message: session_key={session_key}")

    # 2) 이미지 디스크 저장 + image_maching DB 저장
    image_id, input_image = await _save_uploaded_image_payload(db=db, image=image)

    # 3) chat_history DB 저장 (image_id 포함)
    chatbot = get_chatbot()
    chat_history_id = chatbot.conv.add_message(
        db,
        session_key,
        "user",
        input_text,
        image_id,
    )
    logger.info(f"ingest_user_message: 채팅 메시지 저장 완료 id={chat_history_id}")

    result = IngestResult(
        session_id=session_key,
        chat_history_id=chat_history_id,
        input_image=input_image,
    )
    logger.info(f"ingest_user_message: 결과 반환 input_image={input_image is not None}")
    return result

@dataclass
class GeneratedContent:
    content_type: str
    input_text: str
    input_image: Optional[dict]
    output_text: Optional[str] = None
    output_image: Optional[dict] = None
    prompt: Optional[str] = None  # 이미지 생성용 프롬프트
    generation_method: Optional[str] = None  # control_type (canny, depth, openpose)
    style: Optional[str] = None
    industry: Optional[str] = None
    seed: Optional[int] = None
    aspect_ratio: Optional[str] = None

    def to_public_dict(self) -> dict:
        """클라이언트 응답용 최소 필드를 딕셔너리로 변환."""
        image = None
        if self.output_image:
            if "file_hash" in self.output_image:
                image = self.output_image["file_hash"]
            elif "file_directory" in self.output_image:
                image = self.output_image["file_directory"]
        return {
            "content_type": self.content_type,
            "output_text": self.output_text,
            "image": image,
        }



async def generate_contents(
    *,
    input_text: str,
    input_image: Optional[dict]=None,
    generation_type: str,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> GeneratedContent:
    """
    콘텐츠 생성

    Args:
        input_text: 사용자 입력 텍스트
        input_image: 사용자 업로드 이미지(단일)
        generation_type: 생성 타입 (text, image)
        style: 이미지 스타일 (ultra_realistic, semi_realistic, anime)
        aspect_ratio: 이미지 비율 (1:1, 16:9, 9:16, 4:3)
    Returns:
        GeneratedContent: 생성된 광고 콘텐츠
    """

    logger.info(
        f"generate_contents: generation_type={generation_type}, has_input_image={input_image is not None}"
    )

    output_text = ""
    output_image = None
    gen_prompt = None
    gen_method = None
    gen_industry = None
    gen_seed = None
    reference_image = None

    # input_image 로드
    if input_image:
        logger.info(f"generate_contents: 참조 이미지 로딩 시작 {input_image}")
        reference_image = load_image_from_payload(input_image)
        if reference_image:
            logger.info(f"generate_contents: 참조 이미지 로드 완료 size={reference_image.size}")
        else:
            logger.warning("generate_contents: 입력 데이터에서 참조 이미지 로드 실패")

    try:
        if generation_type == "text":
            # 텍스트 생성만 (text_generation 모듈 사용)
            logger.info("generate_contents: 텍스트 생성 시작")
            text_gen = get_text_generator()

            ad_copy = text_gen.generate_ad_copy(
                user_input=input_text,
                tone="warm",
                max_length=100,
            )

            output_text = ad_copy
            logger.info(f"generate_contents: 텍스트 생성 완료 - {ad_copy}")

        elif generation_type == "image":
            # 이미지 생성 (프롬프트 생성 + 이미지 생성 모두 image_generation 모듈에서 처리)
            logger.info("generate_contents: 이미지 생성 시작")

            # generate_and_save_image가 내부적으로 PromptTemplateManager를 사용하여
            # user_input으로부터 프롬프트를 자동 생성함
            img_result = generate_and_save_image(
                user_input=input_text,
                style=style or "ultra_realistic",
                aspect_ratio=aspect_ratio or "1:1",
                reference_image=reference_image,
            )

            if img_result["success"]:
                output_image = {
                    "file_hash": img_result["filename"],
                    "file_directory": img_result["image_path"],
                }
                gen_prompt = img_result.get("prompt")
                gen_seed = img_result.get("seed")
                gen_method = img_result.get("control_type")  # I2I인 경우

                logger.info("generate_contents: 이미지 생성 성공")
            else:
                output_text = f"이미지 생성 실패: {img_result.get('error', '알 수 없는 오류')}"
                logger.error(f"generate_contents: 이미지 생성 실패 - {output_text}")

    except Exception as exc:
        output_text = f"생성 실패: {exc}"
        logger.error(f"생성 실패: {exc}", exc_info=True)

    return GeneratedContent(
        content_type=generation_type,
        input_text=input_text,
        input_image=input_image,
        output_text=output_text,
        output_image=output_image,
        prompt=gen_prompt,
        generation_method=gen_method,
        style=style,
        industry=gen_industry,
        seed=gen_seed,
        aspect_ratio=aspect_ratio,
    )


def persist_generation_result(
    *,
    db: Session,
    session_id: str,
    gen: GeneratedContent,
):
    """생성 결과를 DB에 저장"""

    output_image_id = None
    if gen.output_image:
        output_row = process_db.save_image_from_hash(
            db=db,
            file_hash=gen.output_image["file_hash"],
            file_directory=gen.output_image["file_directory"],
        )
        output_image_id = output_row.id

    # Chat History에 텍스트 결과 저장
    chatbot = get_chatbot()
    chatbot.conv.add_message(
        db,
        session_id,
        "assistant",
        gen.output_text,
        output_image_id,
    )

    input_image_id = None
    if gen.input_image:
        input_row = process_db.save_image_from_hash(
            db=db,
            file_hash=gen.input_image["file_hash"],
            file_directory=gen.input_image["file_directory"],
        )
        input_image_id = input_row.id

    # Generation History 이력 저장
    process_db.save_generation_history(
        db=db,
        data={
            "session_id": session_id,
            "content_type": gen.content_type,
            "input_text": gen.input_text,
            "output_text": gen.output_text,
            "prompt": gen.prompt,
            "input_image_id": input_image_id,
            "output_image_id": output_image_id,
            "generation_method": gen.generation_method,
            "style": gen.style,
            "industry": gen.industry,
            "seed": gen.seed,
            "aspect_ratio": gen.aspect_ratio,
        }
    )


async def _analyze_user_intent(
    db: Session,
    input_text: str,
    session_id: Optional[str],
) -> Dict:
    """
    사용자 입력의 의도를 분석 (generation/modification/consulting)
    """
    chatbot = get_chatbot()

    # 최근 대화 히스토리 조회
    context = {}
    if session_id:
        recent_conversations = chatbot.conv.get_recent_messages(db, session_id, limit=5)
        generation_history = chatbot.conv.get_generation_history(db, session_id, limit=5)
        context = {
            "recent_conversations": recent_conversations,
            "generation_history": generation_history,
        }

    # LLM을 통한 의도 분석
    intent_result = await chatbot.llm.analyze_intent(input_text, context)
    return intent_result


async def _refine_generation_input(
    *,
    db: Session,
    session_id: Optional[str],
    user_message: str,
    intent: str,
    generation_type: Optional[str],
    target_generation_id: Optional[int],
) -> str:
    """
    전체 히스토리를 활용해 생성 입력을 정제
    """
    if not session_id:
        return user_message

    chatbot = get_chatbot()
    chat_history = chatbot.conv.get_full_messages(db, session_id)
    if user_message:
        if (
            not chat_history
            or chat_history[-1].get("role") != "user"
            or chat_history[-1].get("content") != user_message
        ):
            chat_history.append(
                {
                    "role": "user",
                    "content": user_message,
                    "image_id": None,
                    "timestamp": "pending",
                }
            )

    generation_history = chatbot.conv.get_full_generation_history(db, session_id)

    try:
        refined = await chatbot.llm.refine_generation_input(
            intent=intent,
            generation_type=generation_type,
            target_generation_id=target_generation_id,
            chat_history=chat_history,
            generation_history=generation_history,
        )
        return refined or user_message
    except Exception as exc:
        logger.error(f"_refine_generation_input: 실패 - {exc}", exc_info=True)
        return user_message


async def _execute_generation_pipeline(
    *,
    db: Session,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    generation_input: str,
    generation_type: str,
    style: Optional[str],
    aspect_ratio: Optional[str],
    ingest: Optional[IngestResult] = None,
    image: Optional[UploadFile] = None,
) -> Dict:
    """
    생성 파이프라인 실행 (ingest 결과 재사용 가능)
    """
    if ingest is None:
        ingest = await ingest_user_message(
            db=db,
            input_text=input_text,
            session_id=session_id,
            user_id=user_id,
            image=image,
        )

    gen_result = await generate_contents(
        input_text=generation_input,
        input_image=ingest.input_image,
        generation_type=generation_type,
        style=style,
        aspect_ratio=aspect_ratio,
    )

    persist_generation_result(
        db=db,
        session_id=ingest.session_id,
        gen=gen_result,
    )

    result = {
        "session_id": ingest.session_id,
        "output": gen_result.to_public_dict(),
    }
    return result


def _build_consulting_context(
    *,
    chatbot,
    db: Session,
    session_id: str,
    message: str,
    recent_limit: int = 5,
) -> Dict:
    """
    상담 응답에 필요한 대화/생성/지식베이스 컨텍스트를 구성.
    """
    recent_conversations = chatbot.conv.get_recent_messages(
        db,
        session_id,
        limit=recent_limit,
    )

    generation_history = chatbot.conv.get_generation_history(db, session_id, limit=5)
    context = {
        "recent_conversations": recent_conversations,
        "generation_history": generation_history,
        "knowledge_base": [],
    }

    if chatbot.knowledge:
        try:
            knowledge_results = chatbot.knowledge.search(
                query=message,
                category="faq",
                limit=3,
            )
            context["knowledge_base"] = knowledge_results
            logger.info(
                "_build_consulting_context: 지식 검색 결과 %s건",
                len(knowledge_results),
            )
        except Exception as exc:
            logger.warning(f"_build_consulting_context: 지식 검색 실패: {exc}")

    return context


async def _dispatch_consulting_intent(
    *,
    chatbot,
    db: Session,
    session_id: str,
    message: str,
) -> AsyncIterator[Dict]:
    """
    상담 응답을 스트리밍으로 생성하며 청크/완료 이벤트를 순차 반환.
    """
    context = _build_consulting_context(
        chatbot=chatbot,
        db=db,
        session_id=session_id,
        message=message,
        recent_limit=5,
    )

    assistant_chunks: List[str] = []
    async for chunk in chatbot.llm.stream_consulting_response(message, context):
        if chunk:
            assistant_chunks.append(chunk)
            yield {"type": "chunk", "content": chunk}
    assistant_message = "".join(assistant_chunks).strip() or "무엇을 도와드릴까요?"

    chatbot.conv.add_message(db, session_id, "assistant", assistant_message)

    yield {
        "type": "done",
        "session_id": session_id,
        "intent": "consulting",
        "assistant_message": assistant_message,
        "redirect_to_pipeline": False,
        "ready_to_generate": False,
    }


# =====================================================
# RAG 챗봇 서비스
# =====================================================

async def handle_chat_message_stream(
    *,
    db: Session,
    session_id: Optional[str],
    user_id: Optional[int],
    message: str,
    image: Optional[UploadFile] = None,
) -> AsyncIterator[Dict]:
    """
    RAG 챗봇 메시지 처리 스트리밍 서비스 (SSE용)
    - consulting: 응답을 스트리밍으로 전송
    - generation/modification: 생성 결과를 스트리밍으로 반환
    """
    logger.info(f"handle_chat_message_stream: session_id={session_id}, user_id={user_id}")
    chatbot = get_chatbot()

    ingest = await ingest_user_message(
        db=db,
        input_text=message,
        session_id=session_id,
        user_id=user_id,
        image=image,
    )
    session_key = ingest.session_id

    recent_conversations = chatbot.conv.get_recent_messages(db, session_key, limit=5)
    generation_history = chatbot.conv.get_generation_history(db, session_key, limit=5)
    context = {
        "recent_conversations": recent_conversations,
        "generation_history": generation_history,
    }

    intent_result = await chatbot.llm.analyze_intent(message, context)
    intent = intent_result.get("intent", "consulting")
    generation_type = intent_result.get("generation_type") or "image"
    target_generation_id = intent_result.get("target_generation_id")

    yield {"type": "meta", "session_id": session_key, "intent": intent}

    if intent == "consulting":
        async for payload in _dispatch_consulting_intent(
            chatbot=chatbot,
            db=db,
            session_id=session_key,
            message=message,
        ):
            yield payload
        return

    yield {
        "type": "progress",
        "session_id": session_key,
        "intent": intent,
        "stage": "analyzing",
        "message": "요청을 정리하고 있습니다.",
    }

    generation_input = await _refine_generation_input(
        db=db,
        session_id=session_key,
        user_message=message,
        intent=intent,
        generation_type=generation_type,
        target_generation_id=target_generation_id,
    )
    generation_input = generation_input or message

    try:
        if intent == "modification":
            yield {
                "type": "progress",
                "session_id": session_key,
                "intent": intent,
                "stage": "generating",
                "message": "광고를 수정하고 있습니다.",
            }

            result = await handle_chat_revise(
                db=db,
                session_id=session_key,
                user_id=user_id,
                revision_request=message,
                target_generation_id=target_generation_id,
                generation_input=generation_input,
                save_user_message=False,
            )
            output = result.get("output") or {}
            assistant_message = output.get("output_text") or "광고가 수정되었습니다."

            yield {
                "type": "done",
                "session_id": session_key,
                "intent": intent,
                "assistant_message": assistant_message,
                "output": output,
                "generation_id": result.get("generation_id"),
            }
            return

        yield {
            "type": "progress",
            "session_id": session_key,
            "intent": intent,
            "stage": "generating",
            "message": "광고를 생성하고 있습니다.",
        }

        result = await _execute_generation_pipeline(
            db=db,
            input_text=message,
            session_id=session_key,
            user_id=user_id,
            generation_input=generation_input,
            generation_type=generation_type,
            style=None,
            aspect_ratio=None,
            ingest=ingest,
            image=image,
        )
        output = result.get("output") or {}
        assistant_message = output.get("output_text") or "광고가 생성되었습니다."

        yield {
            "type": "done",
            "session_id": session_key,
            "intent": intent,
            "assistant_message": assistant_message,
            "output": output,
        }
    except HTTPException as exc:
        if exc.status_code == 404 and intent == "modification":
            assistant_message = "수정할 광고가 없습니다. 먼저 광고를 생성해주세요. 어떤 광고를 만들어 드릴까요?"
            chatbot.conv.add_message(db, session_key, "assistant", assistant_message)
            yield {
                "type": "done",
                "session_id": session_key,
                "intent": intent,
                "assistant_message": assistant_message,
                "needs_generation_first": True,
                "output": {
                    "content_type": "consulting",
                    "output_text": assistant_message,
                    "image": None,
                },
            }
        else:
            raise
    except Exception as exc:
        logger.error(f"handle_chat_message_stream: 실패 - {exc}", exc_info=True)
        yield {"type": "error", "message": "요청 처리 중 오류가 발생했습니다."}


async def handle_chat_revise(
    *,
    db: Session,
    session_id: str,
    user_id: Optional[int],
    revision_request: str,
    target_generation_id: Optional[int] = None,
    generation_input: Optional[str] = None,
    save_user_message: bool = False,
):
    """
    광고 수정 요청 처리
    """
    logger.info(f"handle_chat_revise: session_id={session_id}, 요청={revision_request}")

    # 1. 대상 생성 이력 선택
    chatbot = get_chatbot()
    latest_generation = _resolve_target_generation(
        db=db,
        session_id=session_id,
        target_generation_id=target_generation_id,
    )

    if not latest_generation:
        raise HTTPException(
            status_code=404,
            detail="수정할 광고를 찾을 수 없습니다. 먼저 광고를 생성해주세요."
        )

    logger.info(f"handle_chat_revise: 생성 이력 발견 id={latest_generation.id}")

    # 2. 수정 요청 파싱
    updated_params = await _parse_revision_request(
        chatbot=chatbot,
        db=db,
        session_id=session_id,
        revision_request=revision_request,
        latest_generation=latest_generation,
        generation_input=generation_input,
    )

    reference_payload = None
    reference_image_id = None
    if updated_params["generation_type"] == "image":
        reference_image = latest_generation.output_image or latest_generation.input_image
        if reference_image:
            reference_payload = {
                "file_hash": reference_image.file_hash,
                "file_directory": reference_image.file_directory,
            }
            reference_image_id = reference_image.id
    # TODO: i2i 기능 완성 시 input_image=reference_payload로 전달하고
    # input_image_id=reference_image_id로 저장하도록 변경하세요.
    i2i_payload = None
    i2i_reference_image_id = None

    # 3. 수정 메시지 저장
    if save_user_message:
        chatbot.conv.add_message(db, session_id, "user", revision_request)

    try:
        gen_result = await generate_contents(
            input_text=updated_params["input_text"],
            input_image=i2i_payload,
            generation_type=updated_params["generation_type"],
            style=updated_params.get("style"),
            aspect_ratio=updated_params.get("aspect_ratio"),
        )

        output_image_id = None
        if gen_result.output_image:
            output_row = process_db.save_image_from_hash(
                db=db,
                file_hash=gen_result.output_image["file_hash"],
                file_directory=gen_result.output_image["file_directory"],
            )
            output_image_id = output_row.id

        assistant_message = gen_result.output_text or "광고가 수정되었습니다."
        chatbot.conv.add_message(
            db,
            session_id,
            "assistant",
            assistant_message,
            output_image_id,
        )

        new_gen = process_db.save_generation_history(
            db=db,
            data={
                "session_id": session_id,
                "content_type": gen_result.content_type,
                "input_text": gen_result.input_text,
                "output_text": gen_result.output_text,
                "prompt": gen_result.prompt,
                "input_image_id": i2i_reference_image_id,
                "output_image_id": output_image_id,
                "generation_method": gen_result.generation_method,
                "style": gen_result.style,
                "industry": gen_result.industry,
                "seed": gen_result.seed,
                "aspect_ratio": gen_result.aspect_ratio,
            },
        )

        result = {
            "session_id": session_id,
            "generation_id": new_gen.id,
            "output": gen_result.to_public_dict(),
        }

        logger.info(f"handle_chat_revise: 완료 - new_id={new_gen.id}")
        return result

    except Exception as exc:
        logger.error(f"handle_chat_revise: 실패 - {exc}", exc_info=True)
        raise


async def _parse_revision_request(
    chatbot,
    db: Session,
    session_id: str,
    revision_request: str,
    latest_generation: models.GenerationHistory,
    generation_input: Optional[str] = None,
) -> Dict:
    """
    수정 요청을 파싱하여 업데이트된 파라미터 반환
    """
    updated_params = {
        "input_text": latest_generation.input_text or "",
        "generation_type": latest_generation.content_type,
        "style": latest_generation.style,
        "aspect_ratio": latest_generation.aspect_ratio,
    }

    if generation_input:
        cleaned = generation_input.strip()
        cleaned = re.sub(
            r"이전 생성물\s*\(ID:\s*\d+\)\s*을?\s*기준으로\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"이전 생성물\s*ID\s*\d+\s*을?\s*기준으로\s*",
            "",
            cleaned,
        ).strip()
        if cleaned != generation_input and (latest_generation.input_text or ""):
            cleaned = f"{latest_generation.input_text} {cleaned}".strip()
            logger.info("_parse_revision_request: 기본 input_text로 generation_input 정규화")
        updated_params["input_text"] = cleaned
        return updated_params

    return updated_params
