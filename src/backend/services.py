import asyncio
import logging, os
from dataclasses import dataclass
from fastapi import Cookie, Depends, HTTPException, Response, UploadFile
from jose import JWTError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from typing import List, Optional, Dict, AsyncIterator, Callable, Any

from src.backend import process_db, schemas, models
from src.backend.chatbot import get_conversation_manager, get_llm_orchestrator, get_consulting_service
from src.generation.text_generation.text_generator import TextGenerator
from src.generation.image_generation.generator import generate_and_save_image
from src.generation.image_generation.preload import (
    get_model_load_error,
    is_model_ready,
    start_model_preload,
    wait_for_model_ready,
)
from src.utils.security import verify_password, create_access_token, decode_token
from src.utils.session import normalize_session_id, ensure_chat_session
from src.utils.image import save_uploaded_image, load_image_from_payload, image_payload
from src.utils.logging import get_logger

logger = get_logger(__name__)
_TEXT_GENERATOR = None  # 싱글톤 인스턴스
_IMAGE_PROGRESS_HINTS = {
    "prompt_done": {
        "percent": 70,
        "message": "이미지 프롬프트 생성을 완료했습니다.",
    },
    "image_generation_start": {
        "percent": 82,
        "message": "이미지를 생성중입니다.",
    },
    "image_save_start": {
        "percent": 96,
        "message": "이미지를 저장중입니다.",
    },
}


def _build_generation_progress_payload(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hint = _IMAGE_PROGRESS_HINTS.get(event.get("event"))
    if not hint:
        return None
    return {
        "type": "progress",
        "stage": "generation_update",
        "message": hint["message"],
        "percent": hint["percent"],
    }

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


def authenticate_user(db, login_id: str, login_pw: str, response: Response) -> models.User:
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

    return user


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

    STORAGE_BASE = "/mnt/data"
    base_dir = os.path.join(STORAGE_BASE, "uploads")

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


def _resolve_industry_for_revision(
    *,
    db: Session,
    session_id: str,
    target_generation: Optional[models.GenerationHistory],
    override_industry: Optional[str],
) -> Optional[str]:
    def _normalize(value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return value
        value = value.strip()
        return value or None

    normalized = _normalize(override_industry)
    if normalized:
        return normalized

    if target_generation:
        normalized = _normalize(target_generation.industry)
        if normalized:
            return normalized

    for gen in process_db.get_generation_history_by_session(db, session_id, limit=20):
        normalized = _normalize(gen.industry)
        if normalized:
            return normalized

    return None

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
    logger.debug(
        "ingest_user_message: session_id=%s, user_id=%s, has_image=%s",
        session_id,
        user_id,
        image is not None,
    )

    # 1) 세션 확보/생성/귀속
    session_key = ensure_chat_session(db, session_id, user_id)
    logger.debug("ingest_user_message: session_key=%s", session_key)

    # 2) 이미지 디스크 저장 + image_maching DB 저장
    image_id, input_image = await _save_uploaded_image_payload(db=db, image=image)

    # 3) chat_history DB 저장 (image_id 포함)
    conv_manager = get_conversation_manager()
    chat_history_id = conv_manager.add_message(
        db,
        session_key,
        "user",
        input_text,
        image_id,
    )
    result = IngestResult(
        session_id=session_key,
        chat_history_id=chat_history_id,
        input_image=input_image,
    )
    logger.info(
        "ingest_user_message: session_id=%s, user_id=%s, has_image=%s, chat_id=%s",
        session_key,
        user_id,
        input_image is not None,
        chat_history_id,
    )
    return result

@dataclass
class GeneratedContent:
    content_type: str
    input_text: str
    input_image: Optional[dict]
    output_text: Optional[str] = None
    output_image: Optional[dict] = None
    prompt: Optional[str] = None  # 이미지 생성용 프롬프트
    generation_method: Optional[str] = None  # t2i or i2i
    style: Optional[str] = None
    industry: Optional[str] = None
    seed: Optional[int] = None
    aspect_ratio: Optional[str] = None
    strength: Optional[float] = None

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
    input_image: Optional[dict] = None,
    generation_type: str,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    industry: Optional[str] = None,
    strength: Optional[float] = None,
    text_tone: Optional[str] = None,
    text_max_length: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> GeneratedContent:
    """
    콘텐츠 생성

    Args:
        input_text: 사용자 입력 텍스트
        input_image: 사용자 업로드 이미지(단일)
        generation_type: 생성 타입 (text, image)
        style: 이미지 스타일 (ultra_realistic, semi_realistic, anime)
        aspect_ratio: 이미지 비율 (1:1, 16:9, 9:16, 4:3)
        industry: 업종 (cafe, restaurant 등)
        strength: 수정 강도 (0.0~1.0)
        text_tone: 텍스트 톤 (warm, professional, friendly, energetic)
        text_max_length: 텍스트 최대 길이 (10~200)
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
    gen_seed = None
    reference_image = None
    effective_strength = strength
    resolved_industry = industry.strip() if isinstance(industry, str) else industry
    if resolved_industry == "":
        resolved_industry = None

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

            safe_max_length = 100
            if text_max_length is not None:
                try:
                    safe_max_length = int(text_max_length)
                except (TypeError, ValueError):
                    safe_max_length = 100
            if safe_max_length < 10:
                safe_max_length = 10
            elif safe_max_length > 200:
                safe_max_length = 200

            ad_copy = text_gen.generate_ad_copy(
                user_input=input_text,
                tone=text_tone or "warm",
                max_length=safe_max_length,
            )

            output_text = ad_copy
            logger.info(f"generate_contents: 텍스트 생성 완료 - {ad_copy}")

        elif generation_type == "image":
            # 이미지 생성 (프롬프트 생성 + 이미지 생성 모두 image_generation 모듈에서 처리)
            logger.info("generate_contents: 이미지 생성 시작")

            # generate_and_save_image가 내부적으로 PromptTemplateManager를 사용하여
            # user_input으로부터 프롬프트를 자동 생성함
            if reference_image is not None and effective_strength is None:
                effective_strength = 0.6
            img_result = await run_in_threadpool(
                generate_and_save_image,
                user_input=input_text,
                style=style or "ultra_realistic",
                aspect_ratio=aspect_ratio or "1:1",
                industry=resolved_industry,
                reference_image=reference_image,
                strength=effective_strength,
                progress_callback=progress_callback,
            )

            gen_method = img_result.get("generation_method")
            if not gen_method:
                gen_method = "i2i" if reference_image is not None else "t2i"

            if not resolved_industry:
                detected_industry = img_result.get("industry")
                if isinstance(detected_industry, str):
                    detected_industry = detected_industry.strip() or None
                if detected_industry == "unknown":
                    detected_industry = None
                resolved_industry = detected_industry

            if img_result["success"]:
                output_image = {
                    "file_hash": img_result["filename"],
                    "file_directory": img_result["image_path"],
                }
                gen_prompt = img_result.get("prompt")
                gen_seed = img_result.get("seed")

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
        industry=resolved_industry,
        strength=effective_strength if generation_type == "image" else None,
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

    # Chat History에 결과 저장
    conv_manager = get_conversation_manager()
    conv_manager.add_message(
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
    gen_history = process_db.save_generation_history(
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
            "strength": gen.strength,
            "aspect_ratio": gen.aspect_ratio,
        }
    )
    return gen_history


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
    """RAG 챗봇 메시지 처리 스트리밍"""
    
    conv_manager = get_conversation_manager()
    llm = get_llm_orchestrator()
    consulting = get_consulting_service()

    # 1. 입력 수집
    ingest = await ingest_user_message(
        db=db,
        input_text=message,
        session_id=session_id,
        user_id=user_id,
        image=image,
    )
    session_key = ingest.session_id

    yield {
        "type": "progress",
        "stage": "analyzing",
        "message": "메시지를 분석중입니다.",
    }

    # 2. Intent 분석 (플랫폼/스타일 자동 결정 - generation_history 불필요)
    recent_conversations = conv_manager.get_recent_messages(db, session_key, limit=3)
    
    intent_result = await llm.analyze_intent(
        user_message=message,
        recent_conversations=recent_conversations
    )
    intent = intent_result.get("intent", "consulting")
    
    # Intent 분석 결과에서 생성 파라미터 추출
    generation_type = intent_result.get("generation_type") or "image"
    aspect_ratio = intent_result.get("aspect_ratio")
    style = intent_result.get("style")
    industry = intent_result.get("industry")
    strength = intent_result.get("strength")
    text_tone = intent_result.get("text_tone")
    text_max_length = intent_result.get("text_max_length")

    yield {
        "type": "meta",
        "session_id": session_key,
        "intent": intent,
        "generation_type": generation_type,
        "aspect_ratio": aspect_ratio,
        "style": style,
        "industry": industry,
        "strength": strength,
        "text_tone": text_tone,
        "text_max_length": text_max_length,
    }

    # 3. Consulting 분기
    if intent == "consulting":
        async for payload in consulting.stream_response(db, session_key, message):
            yield payload
        return

    if generation_type == "image":
        start_model_preload(device="cuda")
        if not is_model_ready():
            yield {
                "type": "progress",
                "stage": "preloading",
                "message": "이미지 생성 모델을 로드 중입니다. 잠시만 기다려주세요.",
                "session_id": session_key,
            }
            if not await wait_for_model_ready():
                error_detail = get_model_load_error()
                if error_detail:
                    logger.error(f"Image model preload failed: {error_detail}")
                yield {
                    "type": "error",
                    "message": "이미지 생성 모델 로드에 실패했습니다. 잠시 후 다시 시도해주세요.",
                }
                return

    # 4. Generation/Modification 분기
    # Refinement (텍스트 정제 + 수정 대상 ID 찾기)
    chat_history = conv_manager.get_full_messages(db, session_key)
    generation_history = conv_manager.get_full_generation_history(db, session_key)
    
    refinement_result = await llm.refine_generation_input(
        intent=intent,
        generation_type=generation_type,
        chat_history=chat_history,
        generation_history=generation_history,
    )
    
    generation_input = refinement_result.get("refined_input") or message
    target_generation_id = refinement_result.get("target_generation_id")  # Refinement에서 찾음

    progress_queue: Optional[asyncio.Queue] = None
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    if generation_type == "image":
        loop = asyncio.get_running_loop()
        progress_queue = asyncio.Queue()

        def progress_callback(event: Dict[str, Any]) -> None:
            payload = _build_generation_progress_payload(event)
            if payload:
                loop.call_soon_threadsafe(progress_queue.put_nowait, payload)

    try:
        if intent == "modification":
            yield {
                "type": "progress",
                "stage": "generating",
                "message": "광고를 수정하고 있습니다.",
                "generation_type": generation_type,
            }

            task = asyncio.create_task(
                handle_chat_revise(
                    db=db,
                    session_id=session_key,
                    target_generation_id=target_generation_id,
                    generation_input=generation_input,
                    style=style,  # Intent에서 결정된 스타일 전달
                    aspect_ratio=aspect_ratio,  # Intent에서 결정된 비율 전달
                    industry=industry,
                    strength=strength,  # Intent에서 결정된 강도 전달
                    text_tone=text_tone,
                    text_max_length=text_max_length,
                    progress_callback=progress_callback,
                )
            )
            if progress_queue is not None:
                while True:
                    if task.done() and progress_queue.empty():
                        break
                    try:
                        payload = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    yield payload
            result = await task
            if progress_queue is not None:
                while not progress_queue.empty():
                    yield progress_queue.get_nowait()
            output = result.get("output", {})

            yield {
                "type": "done",
                "intent": intent,
                "output": output,
                "generation_id": result.get("generation_id"),
            }
            return

        # Generation
        yield {
            "type": "progress",
            "stage": "generating",
            "message": f"광고를 생성하고 있습니다. (비율: {aspect_ratio or '기본'})",
            "generation_type": generation_type,
        }

        task = asyncio.create_task(
            _execute_generation_pipeline(
                db=db,
                generation_input=generation_input,
                generation_type=generation_type,
                aspect_ratio=aspect_ratio,
                style=style,
                industry=industry,
                strength=strength,
                text_tone=text_tone,
                text_max_length=text_max_length,
                ingest=ingest,
                progress_callback=progress_callback,
            )
        )
        if progress_queue is not None:
            while True:
                if task.done() and progress_queue.empty():
                    break
                try:
                    payload = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                yield payload
        result = await task
        if progress_queue is not None:
            while not progress_queue.empty():
                yield progress_queue.get_nowait()
        output = result.get("output", {})

        yield {
            "type": "done",
            "intent": intent,
            "output": output,
        }

    except Exception as exc:
        logger.error(f"Stream failed: {exc}", exc_info=True)
        yield {"type": "error", "message": "요청 처리 중 오류가 발생했습니다."}


async def _execute_generation_pipeline(
    *,
    db: Session,
    generation_input: str,
    generation_type: str,
    aspect_ratio: Optional[str] = None,
    style: Optional[str] = None,
    industry: Optional[str] = None,
    strength: Optional[float] = None,
    text_tone: Optional[str] = None,
    text_max_length: Optional[int] = None,
    ingest: Optional[IngestResult] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict:
    """
    생성 파이프라인 실행
    
    파라미터 우선순위:
    1. 명시적으로 전달된 값 (aspect_ratio, style)
    2. Intent 분석에서 자동 결정된 값
    3. 기본값
    """
    # aspect_ratio 최종 결정 (Intent 결과 or 기본값)
    final_aspect_ratio = aspect_ratio or "1:1"

    # style 최종 결정 (Intent 결과 or 기본값)
    final_style = style or "ultra_realistic"

    logger.info(
        f"Generation pipeline: type={generation_type}, "
        f"ratio={final_aspect_ratio}, style={final_style}, industry={industry}"
    )

    # 콘텐츠 생성
    gen_result = await generate_contents(
        input_text=generation_input,
        input_image=ingest.input_image,
        generation_type=generation_type,
        style=final_style,
        aspect_ratio=final_aspect_ratio,
        industry=industry,  # 업종 정보도 전달 (프롬프트 생성 시 활용 가능)
        strength=strength,
        text_tone=text_tone,
        text_max_length=text_max_length,
        progress_callback=progress_callback,
    )

    # 결과 저장
    persist_generation_result(
        db=db,
        session_id=ingest.session_id,
        gen=gen_result,
    )

    return {
        "session_id": ingest.session_id,
        "output": gen_result.to_public_dict(),
    }


async def handle_chat_revise(
    *,
    db: Session,
    session_id: str,
    target_generation_id: Optional[int] = None,
    generation_input: Optional[str] = None,
    style: Optional[str] = None,  # Intent에서 전달받음
    aspect_ratio: Optional[str] = None,  # Intent에서 전달받음
    industry: Optional[str] = None,
    strength: Optional[float] = None,  # Intent에서 전달받음
    text_tone: Optional[str] = None,
    text_max_length: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    광고 수정 요청 처리 서비스
    - 수정 시에도 Intent 분석 결과의 스타일/비율 적용 가능
    """

    # 대상 생성 이력 선택
    latest_generation = _resolve_target_generation(
        db=db,
        session_id=session_id,
        target_generation_id=target_generation_id,
    )

    if not latest_generation:
        raise HTTPException(404, "수정할 광고를 찾을 수 없습니다.")

    resolved_industry = _resolve_industry_for_revision(
        db=db,
        session_id=session_id,
        target_generation=latest_generation,
        override_industry=industry,
    )

    # 수정 파라미터 구성 (Intent 결과 우선, 없으면 기존값 유지)
    reference_payload = (
        image_payload(latest_generation.output_image)
        or image_payload(latest_generation.input_image)
    )
    updated_params = {
        "input_text": generation_input or latest_generation.input_text or "",
        "generation_type": latest_generation.content_type,
        "style": style or latest_generation.style,  # Intent 결과 or 기존 스타일
        "aspect_ratio": aspect_ratio or latest_generation.aspect_ratio,  # Intent 결과 or 기존 비율
        "industry": resolved_industry,
        "reference_image": reference_payload,
        "strength": strength,  # Intent 결과 우선
    }


    # 콘텐츠 생성
    gen_result = await generate_contents(
        input_text=updated_params["input_text"],
        input_image=updated_params["reference_image"],  # 기존 생성물 또는 입력 이미지 사용
        generation_type=updated_params["generation_type"],
        style=updated_params["style"],
        aspect_ratio=updated_params["aspect_ratio"],
        industry=updated_params["industry"],
        strength=updated_params["strength"],
        text_tone=text_tone,
        text_max_length=text_max_length,
        progress_callback=progress_callback,
    )

    # 결과 저장
    gen_history = persist_generation_result(
        db=db,
        session_id=session_id,
        gen=gen_result,
    )

    return {
        "session_id": session_id,
        "generation_id": gen_history.id,
        "output": gen_result.to_public_dict(),
    }
