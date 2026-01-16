import logging, os
from dataclasses import dataclass
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from PIL import Image

from src.backend import process_db, schemas, models
from src.backend.task import TaskStatus, create_task, update_task_progress, complete_task, fail_task
from src.utils.config import PROJECT_ROOT
from src.utils.security import verify_password, create_access_token, decode_token
from src.utils.session import normalize_session_id, ensure_chat_session
from src.utils.image import save_uploaded_image, load_image_from_payload
from src.generation.text_generation.text_generator import TextGenerator
from src.generation.image_generation.generator import generate_and_save_image
from src.backend.chatbot import get_chatbot


logger = logging.getLogger(__name__)
task_storage = {}
_TEXT_GENERATOR = None  # 싱글톤 인스턴스

def get_text_generator() -> TextGenerator:
    """TextGenerator 싱글톤 인스턴스 반환"""
    global _TEXT_GENERATOR
    if _TEXT_GENERATOR is None:
        _TEXT_GENERATOR = TextGenerator()
    return _TEXT_GENERATOR

# 클라이언트는 Bearer 헤더로 전달
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False) 

def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(process_db.get_db),
):
    """
    JWT token으로 현재 로그인 유저 정보 가져오기
    - 토큰이 없으면 None 반환
    """
    if not token:
        return None

    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="토큰 오류")

    user = process_db.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")

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


def authenticate_user(db, login_id: str, login_pw: str) -> str:
    """
    사용자 인증(로그인) 서비스
    - 아이디 존재 여부 확인, 비밀번호 검증
    - JWT access 토큰 생성 및 반환
    """
    user = process_db.get_user_by_login_id(db, login_id)
    if not user or not verify_password(login_pw, user.login_pw):
        raise HTTPException(400, "아이디 또는 비밀번호가 일치하지 않습니다.")
    return create_access_token(str(user.user_id))


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

    # 2) 이미지 디스크 저장 + DB 저장
    input_image = None
    if image:
        logger.info(f"ingest_user_message: processing image upload")
        base_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        logger.info(f"ingest_user_message: base_dir={base_dir}")

        input_image = await save_uploaded_image(image=image, base_dir=base_dir)

        if input_image:
            logger.info(f"ingest_user_message: image saved to disk: {input_image}")
        else:
            logger.warning("ingest_user_message: save_uploaded_image returned None")

    image_row = None
    if input_image:
        logger.info(f"ingest_user_message: saving image to DB")
        image_row = process_db.save_image_from_hash(
            db=db,
            file_hash=input_image["file_hash"],
            file_directory=input_image["file_directory"],
        )
        logger.info(f"ingest_user_message: image saved to DB with id={image_row.id}")
    else:
        logger.info("ingest_user_message: no image to save to DB")

    # 3) 텍스트 DB 저장 (image_id 포함)
    chat_row = process_db.save_chat_message(
        db,
        {
            "session_id": session_key,
            "role": "user",
            "content": input_text,
            "image_id": image_row.id if image_row else None,
        },
    )
    logger.info(f"ingest_user_message: chat message saved with id={chat_row.id}")

    result = IngestResult(
        session_id=session_key,
        chat_history_id=chat_row.id,
        input_image=input_image,
    )
    logger.info(f"ingest_user_message: returning result with input_image={input_image is not None}")
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
    conversation_history: Optional[List[Dict]] = None,
) -> GeneratedContent:
    """
    콘텐츠 생성

    Args:
        input_text: 사용자 입력 텍스트
        input_image: 사용자 업로드 이미지(단일)
        generation_type: 생성 타입 (text, image)
        style: 이미지 스타일 (ultra_realistic, semi_realistic, anime)
        aspect_ratio: 이미지 비율 (1:1, 16:9, 9:16, 4:3)
        conversation_history: 대화 히스토리 (선택사항)

    Returns:
        GeneratedContent: 생성된 광고 콘텐츠
    """

    logger.info(f"generate_contents: generation_type={generation_type}, has_input_image={input_image is not None}, has_conversation_history={conversation_history is not None}")

    output_text = ""
    output_image = None
    gen_prompt = None
    gen_method = None
    gen_industry = None
    gen_seed = None
    reference_image = None

    # input_image 로드
    if input_image:
        logger.info(f"generate_contents: loading reference image from {input_image}")
        reference_image = load_image_from_payload(input_image)
        if reference_image:
            logger.info(f"generate_contents: reference image loaded successfully: size={reference_image.size}")
        else:
            logger.warning("generate_contents: failed to load reference image from payload")

    try:
        if generation_type == "text":
            # 텍스트 생성만 (text_generation 모듈 사용)
            logger.info("generate_contents: text generation started")
            text_gen = get_text_generator()

            ad_copy = text_gen.generate_ad_copy(
                user_input=input_text,
                tone="warm",
                max_length=20,
                conversation_history=conversation_history
            )

            output_text = ad_copy
            logger.info(f"generate_contents: text generated - {ad_copy}")

        elif generation_type == "image":
            # 이미지 생성 (프롬프트 생성 + 이미지 생성 모두 image_generation 모듈에서 처리)
            logger.info("generate_contents: image generation started")

            # generate_and_save_image가 내부적으로 PromptTemplateManager를 사용하여
            # user_input으로부터 프롬프트를 자동 생성함
            img_result = generate_and_save_image(
                user_input=input_text,
                style=style or "ultra_realistic",
                aspect_ratio=aspect_ratio or "1:1",
                reference_image=reference_image,
                conversation_history=conversation_history,
            )

            if img_result["success"]:
                output_image = {
                    "file_hash": img_result["filename"],
                    "file_directory": img_result["image_path"],
                }
                gen_prompt = img_result.get("prompt")
                gen_seed = img_result.get("seed")
                gen_method = img_result.get("control_type")  # I2I인 경우

                # 광고 문구도 함께 생성
                text_gen = get_text_generator()
                ad_copy = text_gen.generate_ad_copy(
                    user_input=input_text,
                    tone="warm",
                    max_length=20,
                    conversation_history=conversation_history
                )
                output_text = ad_copy

                logger.info(f"generate_contents: image generated successfully")
            else:
                output_text = f"이미지 생성 실패: {img_result.get('error', '알 수 없는 오류')}"
                logger.error(f"generate_contents: image generation failed - {output_text}")

    except Exception as exc:
        output_text = f"생성 실패: {exc}"
        logger.error(f"Generation failed: {exc}", exc_info=True)

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
    assistant_row = process_db.save_chat_message(
        db,
        {
            "session_id": session_id,
            "role": "assistant",
            "content": gen.output_text,
            "image_id": output_image_id,
        },
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





async def handle_generate_pipeline(
    *,
    db: Session,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    image: Optional[UploadFile]=None,
    task_id: str,
    create_task_entry: bool = True,
    generation_type: Optional[str] = None,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    skip_intent_analysis: bool = False,
):
    """
    광고 생성 파이프라인 통합 서비스
    - task를 통해 작업 상태 관리
    - Intent 분석을 통해 수정/상담 요청을 적절히 처리
    """
    # 작업 생성
    if create_task_entry:
        create_task(task_id)

    try:
        # 0. Intent 분석 (skip_intent_analysis=False일 때만)
        # UI에서 generation_type을 지정해도 사용자 입력 의도를 먼저 분석
        if not skip_intent_analysis:
            update_task_progress(task_id, 2, TaskStatus.INGESTING)
            intent_result = await _analyze_user_intent(db, input_text, session_id)
            intent = intent_result.get("intent", "generation")
            logger.info(f"handle_generate_pipeline: analyzed intent={intent}")

            # Intent별 분기 처리
            if intent == "modification":
                # 수정 요청 → revise 플로우로 라우팅
                logger.info("handle_generate_pipeline: routing to modification flow")
                return await _handle_modification_in_pipeline(
                    db=db,
                    input_text=input_text,
                    session_id=session_id,
                    user_id=user_id,
                    task_id=task_id,
                )

            elif intent == "consulting":
                # 상담 요청 → 챗봇 응답 생성
                logger.info("handle_generate_pipeline: routing to consulting flow")
                return await _handle_consulting_in_pipeline(
                    db=db,
                    input_text=input_text,
                    session_id=session_id,
                    user_id=user_id,
                    task_id=task_id,
                )

        # 1. 입력 수집/저장
        update_task_progress(task_id, 5, TaskStatus.INGESTING)
        ingest = await ingest_user_message(
            db=db,
            input_text=input_text,
            session_id=session_id,
            user_id=user_id,
            image=image,
        )

        # 2. 대화 히스토리 조회
        conversation_history = []
        if ingest.session_id:
            chatbot = get_chatbot()
            conversation_history = chatbot.conv.get_recent_messages(db, ingest.session_id, limit=10)
            logger.info(f"handle_generate_pipeline: retrieved {len(conversation_history)} conversation messages")

        # 3. 콘텐츠 생성
        update_task_progress(task_id, 30, TaskStatus.GENERATING)
        gen_result = await generate_contents(
            input_text=input_text,
            input_image=ingest.input_image,
            generation_type=generation_type,
            style=style,
            aspect_ratio=aspect_ratio,
            conversation_history=conversation_history,
        )

        # 4. 결과 DB 저장
        update_task_progress(task_id, 80, TaskStatus.PERSISTING)
        persist_generation_result(
            db=db,
            session_id=ingest.session_id,
            gen=gen_result,
        )

        # 5. 완료
        result = {
            "session_id": ingest.session_id,
            "output": gen_result.to_public_dict(),
        }
        complete_task(task_id, result)

        return result

    except Exception as exc:
        fail_task(task_id, str(exc))
        raise


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
        generation_history = chatbot.conv.get_generation_history(db, session_id, limit=3)
        context = {
            "recent_conversations": recent_conversations,
            "generation_history": generation_history,
        }

    # LLM을 통한 의도 분석
    intent_result = await chatbot.llm.analyze_intent(input_text, context)
    return intent_result


async def _handle_modification_in_pipeline(
    *,
    db: Session,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    task_id: str,
) -> Dict:
    """
    생성 파이프라인 내에서 수정 요청 처리
    """
    from src.utils.session import normalize_session_id, ensure_chat_session

    # 세션 확보
    session_key = normalize_session_id(session_id)
    session_key = ensure_chat_session(db, session_key, user_id)

    # 최근 생성 이력 확인
    latest_generation = process_db.get_latest_unconfirmed_generation(db, session_key)
    if not latest_generation:
        latest_generation = process_db.get_latest_generation(db, session_key)

    if not latest_generation:
        # 수정할 광고가 없으면 새로 생성하도록 안내
        update_task_progress(task_id, 10, TaskStatus.GENERATING)

        # 사용자 메시지 저장
        process_db.save_chat_message(
            db,
            {
                "session_id": session_key,
                "role": "user",
                "content": input_text,
                "image_id": None,
            }
        )

        assistant_message = "수정할 광고가 없습니다. 먼저 광고를 생성해주세요. 어떤 광고를 만들어 드릴까요?"
        process_db.save_chat_message(
            db,
            {
                "session_id": session_key,
                "role": "assistant",
                "content": assistant_message,
                "image_id": None,
            }
        )

        result = {
            "session_id": session_key,
            "output": {
                "content_type": "consulting",
                "output_text": assistant_message,
                "image": None,
            },
            "intent": "modification",
            "needs_generation_first": True,
        }
        complete_task(task_id, result)
        return result

    # 수정 플로우 실행
    logger.info(f"_handle_modification_in_pipeline: revising generation id={latest_generation.id}")

    return await handle_chat_revise(
        db=db,
        session_id=session_key,
        user_id=user_id,
        revision_request=input_text,
        task_id=task_id,
        create_task_entry=False,  # 이미 task가 생성되어 있으므로
    )


async def _handle_consulting_in_pipeline(
    *,
    db: Session,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    task_id: str,
) -> Dict:
    """
    생성 파이프라인 내에서 상담 요청 처리
    """
    from src.utils.session import normalize_session_id, ensure_chat_session

    # 세션 확보
    session_key = normalize_session_id(session_id)
    session_key = ensure_chat_session(db, session_key, user_id)

    update_task_progress(task_id, 10, TaskStatus.GENERATING)

    # 챗봇을 통한 상담 응답 생성
    chatbot = get_chatbot()

    # 사용자 메시지 저장
    chatbot.conv.add_message(db, session_key, "user", input_text)

    # 컨텍스트 구성
    recent_conversations = chatbot.conv.get_recent_messages(db, session_key, limit=10)
    generation_history = chatbot.conv.get_generation_history(db, session_key, limit=5)

    context = {
        "recent_conversations": recent_conversations,
        "generation_history": generation_history,
        "knowledge_base": [],
    }

    # 지식베이스 검색 (있는 경우)
    if chatbot.knowledge:
        try:
            knowledge_results = chatbot.knowledge.search(
                query=input_text,
                category="faq",
                limit=3
            )
            context["knowledge_base"] = knowledge_results
            logger.info(f"_handle_consulting_in_pipeline: knowledge search returned {len(knowledge_results)} results")
        except Exception as e:
            logger.warning(f"_handle_consulting_in_pipeline: knowledge search failed: {e}")

    update_task_progress(task_id, 50, TaskStatus.GENERATING)

    # LLM 상담 응답 생성
    assistant_message = await chatbot.llm.generate_consulting_response(input_text, context)

    # 어시스턴트 응답 저장
    chatbot.conv.add_message(db, session_key, "assistant", assistant_message)

    update_task_progress(task_id, 90, TaskStatus.PERSISTING)

    result = {
        "session_id": session_key,
        "output": {
            "content_type": "consulting",
            "output_text": assistant_message,
            "image": None,
        },
        "intent": "consulting",
    }
    complete_task(task_id, result)

    logger.info("_handle_consulting_in_pipeline: consulting response generated")
    return result


# =====================================================
# RAG 챗봇 서비스
# =====================================================

async def handle_chat_message(
    *,
    db: Session,
    session_id: Optional[str],
    user_id: Optional[int],
    message: str,
    image: Optional[UploadFile] = None,
    background_tasks=None,
):
    """
    RAG 챗봇 메시지 처리 서비스
    - Intent 분석 후 분기 처리
    - generation/modification: 생성 파이프라인 시작 + task_id 반환
    - consulting: LLM 상담 응답 반환
    """
    import uuid

    logger.info(f"handle_chat_message: session_id={session_id}, user_id={user_id}")

    # 1. 세션 확보
    session_id = normalize_session_id(session_id)
    session_key = ensure_chat_session(db, session_id, user_id)

    # 2. 이미지 처리 (있는 경우)
    image_id = None
    image_data = None
    if image:
        logger.info(f"handle_chat_message: processing image upload")
        base_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        image_data = await save_uploaded_image(image=image, base_dir=base_dir)

        if image_data:
            image_row = process_db.save_image_from_hash(
                db=db,
                file_hash=image_data["file_hash"],
                file_directory=image_data["file_directory"],
            )
            image_id = image_row.id
            logger.info(f"handle_chat_message: image saved with id={image_id}")

    # 3. 챗봇 처리 (Intent 분석 + 분기 처리)
    chatbot = get_chatbot()
    result = await chatbot.process_message(
        db=db,
        session_id=session_key,
        user_message=message,
        image_id=image_id,
    )

    # 4. Intent별 분기 처리
    intent = result.get("intent", "consulting")
    logger.info(f"handle_chat_message: intent={intent}")

    if intent in ["generation", "modification"] and result.get("redirect_to_pipeline"):
        # 생성/수정 intent: 백그라운드에서 파이프라인 시작
        task_id = str(uuid.uuid4())
        create_task(task_id)

        logger.info(f"handle_chat_message: starting pipeline for intent={intent}, task_id={task_id}")

        if background_tasks:
            # LLM 분석 결과에서 generation_type 가져오기 (기본값: image)
            generation_type = result.get("generation_type", "image")
            logger.info(f"handle_chat_message: generation_type from intent analysis={generation_type}")

            background_tasks.add_task(
                _run_generation_for_intent,
                db=db,
                session_id=session_key,
                user_id=user_id,
                message=message,
                image_data=image_data,
                task_id=task_id,
                intent=intent,
                generation_type=generation_type,
            )

        response = {
            "session_id": session_key,
            "intent": intent,
            "assistant_message": result.get("assistant_message", "광고를 생성하겠습니다."),
            "redirect_to_pipeline": True,
            "task_id": task_id,
            "ready_to_generate": True,
            "workflow_state": result.get("workflow_state", {}),
        }
    else:
        # 상담 intent: 바로 응답 반환
        response = {
            "session_id": session_key,
            "intent": intent,
            "assistant_message": result.get("assistant_message", "무엇을 도와드릴까요?"),
            "redirect_to_pipeline": False,
            "ready_to_generate": False,
            "workflow_state": result.get("workflow_state", {}),
        }

    return response


async def _run_generation_for_intent(
    *,
    db: Session,
    session_id: str,
    user_id: Optional[int],
    message: str,
    image_data: Optional[dict],
    task_id: str,
    intent: str,
    generation_type: str,
):
    """
    Intent 분석 후 실제 생성/수정 파이프라인 실행
    """
    try:
        if intent == "modification":
            # 수정 요청
            logger.info(f"_run_generation_for_intent: running modification for task_id={task_id}")
            await _handle_modification_in_pipeline(
                db=db,
                input_text=message,
                session_id=session_id,
                user_id=user_id,
                task_id=task_id,
            )
        else:
            # 생성 요청
            logger.info(f"_run_generation_for_intent: running generation for task_id={task_id}")

            update_task_progress(task_id, 10, TaskStatus.GENERATING)

            # 대화 히스토리 조회
            chatbot = get_chatbot()
            conversation_history = chatbot.conv.get_recent_messages(db, session_id, limit=10)
            logger.info(f"_run_generation_for_intent: retrieved {len(conversation_history)} conversation messages")

            gen_result = await generate_contents(
                input_text=message,
                input_image=image_data,
                generation_type=generation_type,
                style=None,
                aspect_ratio=None,
                conversation_history=conversation_history,
            )

            update_task_progress(task_id, 70, TaskStatus.PERSISTING)

            persist_generation_result(
                db=db,
                session_id=session_id,
                gen=gen_result,
            )

            result = {
                "session_id": session_id,
                "output": gen_result.to_public_dict(),
                "intent": intent,
            }
            complete_task(task_id, result)
            logger.info(f"_run_generation_for_intent: generation complete for task_id={task_id}")

    except Exception as exc:
        logger.error(f"_run_generation_for_intent: failed - {exc}", exc_info=True)
        fail_task(task_id, str(exc))


async def handle_chat_generate(
    *,
    db: Session,
    session_id: str,
    user_id: Optional[int],
    task_id: str,
):
    """
    챗봇을 통해 수집된 정보로 광고 생성
    """
    logger.info(f"handle_chat_generate: session_id={session_id}, task_id={task_id}")

    chatbot = get_chatbot()
    workflow_state = chatbot.get_workflow_state(session_id)

    if not workflow_state.is_complete:
        raise HTTPException(
            status_code=400,
            detail=f"필수 정보가 부족합니다: {', '.join(workflow_state.get_missing_info())}"
        )

    result = await handle_generate_pipeline(
        db=db,
        input_text=workflow_state.user_input,
        session_id=session_id,
        user_id=user_id,
        image=None,
        task_id=task_id,
        create_task_entry=True,
        generation_type=workflow_state.ad_type,
        style=workflow_state.style,
        aspect_ratio=workflow_state.aspect_ratio,
    )

    chatbot.reset_workflow(session_id)
    logger.info(f"handle_chat_generate: generation complete, workflow reset")

    return result


async def handle_chat_revise(
    *,
    db: Session,
    session_id: str,
    user_id: Optional[int],
    revision_request: str,
    task_id: str,
    create_task_entry: bool = True,
):
    """
    광고 수정 요청 처리
    """
    logger.info(f"handle_chat_revise: session_id={session_id}, request={revision_request}")

    # 1. 최근 생성 이력 조회
    latest_generation = process_db.get_latest_unconfirmed_generation(db, session_id)

    if not latest_generation:
        latest_generation = process_db.get_latest_generation(db, session_id)

    if not latest_generation:
        raise HTTPException(
            status_code=404,
            detail="수정할 광고를 찾을 수 없습니다. 먼저 광고를 생성해주세요."
        )

    logger.info(f"handle_chat_revise: found generation id={latest_generation.id}")

    # 2. 수정 요청 파싱
    chatbot = get_chatbot()
    updated_params = await _parse_revision_request(
        chatbot=chatbot,
        revision_request=revision_request,
        latest_generation=latest_generation,
    )

    # 3. 수정 메시지 저장
    process_db.save_chat_message(
        db,
        {
            "session_id": session_id,
            "role": "user",
            "content": f"[수정 요청] {revision_request}",
            "image_id": None,
        }
    )

    # 4. 재생성
    if create_task_entry:
        create_task(task_id)

    try:
        update_task_progress(task_id, 10, TaskStatus.GENERATING)

        # 대화 히스토리 조회
        chatbot_inst = get_chatbot()
        conversation_history = chatbot_inst.conv.get_recent_messages(db, session_id, limit=10)
        logger.info(f"handle_chat_revise: retrieved {len(conversation_history)} conversation messages")

        gen_result = await generate_contents(
            input_text=updated_params["input_text"],
            input_image=None,
            generation_type=updated_params["generation_type"],
            style=updated_params.get("style"),
            aspect_ratio=updated_params.get("aspect_ratio"),
            conversation_history=conversation_history,
        )

        update_task_progress(task_id, 70, TaskStatus.GENERATING)

        output_image_id = None
        if gen_result.output_image:
            output_row = process_db.save_image_from_hash(
                db=db,
                file_hash=gen_result.output_image["file_hash"],
                file_directory=gen_result.output_image["file_directory"],
            )
            output_image_id = output_row.id

        assistant_message = f"광고가 수정되었습니다. (수정 #{latest_generation.revision_number + 1})"
        process_db.save_chat_message(
            db,
            {
                "session_id": session_id,
                "role": "assistant",
                "content": assistant_message,
                "image_id": output_image_id,
            }
        )

        new_gen = process_db.save_generation_history_with_revision(
            db=db,
            data={
                "session_id": session_id,
                "content_type": gen_result.content_type,
                "input_text": gen_result.input_text,
                "output_text": gen_result.output_text,
                "prompt": gen_result.prompt,
                "input_image_id": None,
                "output_image_id": output_image_id,
                "generation_method": gen_result.generation_method,
                "style": gen_result.style,
                "industry": gen_result.industry,
                "seed": gen_result.seed,
                "aspect_ratio": gen_result.aspect_ratio,
            },
            revision_of_id=latest_generation.id,
        )

        update_task_progress(task_id, 90, TaskStatus.PERSISTING)

        result = {
            "session_id": session_id,
            "generation_id": new_gen.id,
            "revision_number": new_gen.revision_number,
            "revision_of_id": latest_generation.id,
            "output": gen_result.to_public_dict(),
        }

        complete_task(task_id, result)
        logger.info(f"handle_chat_revise: complete - new_id={new_gen.id}")

        return result

    except Exception as exc:
        logger.error(f"handle_chat_revise: failed - {exc}", exc_info=True)
        fail_task(task_id, str(exc))
        raise


async def _parse_revision_request(
    chatbot,
    revision_request: str,
    latest_generation: models.GenerationHistory,
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

    revision_lower = revision_request.lower()

    # 스타일 변경 감지
    if "사실적" in revision_lower or "realistic" in revision_lower:
        updated_params["style"] = "ultra_realistic"
    elif "애니" in revision_lower or "anime" in revision_lower:
        updated_params["style"] = "anime"
    elif "세미" in revision_lower or "semi" in revision_lower:
        updated_params["style"] = "semi_realistic"

    # 비율 변경 감지
    if "정사각" in revision_lower or "1:1" in revision_lower:
        updated_params["aspect_ratio"] = "1:1"
    elif "가로" in revision_lower or "16:9" in revision_lower:
        updated_params["aspect_ratio"] = "16:9"
    elif "세로" in revision_lower or "9:16" in revision_lower:
        updated_params["aspect_ratio"] = "9:16"

    return updated_params


async def handle_chat_confirm(
    *,
    db: Session,
    session_id: str,
):
    """
    최종 광고 확정 처리
    """
    logger.info(f"handle_chat_confirm: session_id={session_id}")

    latest_generation = process_db.get_latest_unconfirmed_generation(db, session_id)

    if not latest_generation:
        raise HTTPException(
            status_code=404,
            detail="확정할 광고를 찾을 수 없습니다. 먼저 광고를 생성해주세요."
        )

    confirmed_gen = process_db.confirm_generation(db, latest_generation.id)

    if not confirmed_gen:
        raise HTTPException(
            status_code=500,
            detail="광고 확정 중 오류가 발생했습니다."
        )

    assistant_message = "광고가 최종 확정되었습니다!"
    process_db.save_chat_message(
        db,
        {
            "session_id": session_id,
            "role": "assistant",
            "content": assistant_message,
            "image_id": confirmed_gen.output_image_id,
        }
    )

    chatbot = get_chatbot()
    chatbot.reset_workflow(session_id)

    logger.info(f"handle_chat_confirm: confirmed id={confirmed_gen.id}")

    result = {
        "session_id": session_id,
        "generation_id": confirmed_gen.id,
        "message": "광고가 최종 확정되었습니다.",
        "content_type": confirmed_gen.content_type,
        "output_text": confirmed_gen.output_text,
        "revision_number": confirmed_gen.revision_number,
    }

    if confirmed_gen.output_image:
        result["image"] = confirmed_gen.output_image.file_hash

    return result
