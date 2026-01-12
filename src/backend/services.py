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
from src.generation.text_generation.ad_generator import generate_advertisement
from src.generation.image_generation.generator import generate_and_save_image


logger = logging.getLogger(__name__)
task_storage = {}
_TEXT_GENERATOR = None

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
) -> GeneratedContent:
    """
    콘텐츠 생성

    Args:
        input_text: 사용자 입력 텍스트
        input_image: 사용자 업로드 이미지(단일)
        reference_image: 이미지 생성에 사용하는 PIL 이미지
        generation_type: 생성 타입 (text, image)
        style: 이미지 스타일 (ultra_realistic, semi_realistic, anime)
        aspect_ratio: 이미지 비율 (1:1, 16:9, 9:16, 4:3)

    Returns:
        GeneratedContent: 생성된 광고 콘텐츠
    """

    logger.info(f"generate_contents: generation_type={generation_type}, has_input_image={input_image is not None}")

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
        ad_result = generate_advertisement(
            user_input=input_text,
            tone="warm",           # optional (기본값: "warm")
            max_length=20,         # optional (기본값: 20)
            style="realistic"      # optional (기본값: "realistic")
        )
        gen_industry = ad_result.get("industry")

        if generation_type == "text":
            # 텍스트 생성 (광고 문구 + 업종)
            
            output_text = ad_result.get("ad_copy", "광고 문구 생성에 실패했습니다.")
            

        elif generation_type == "image":
            # 프롬프트 생성
            gen_prompt = ad_result.get("positive_prompt")


            if ad_result["status"] != "success":
                output_text = f"프롬프트 생성 실패: {ad_result.get('error', '알 수 없는 오류')}"
            else:
                gen_prompt = ad_result["positive_prompt"]

                # 2. 이미지 생성
                img_result = generate_and_save_image(
                    prompt=gen_prompt,
                    style=style or "ultra_realistic",
                    aspect_ratio=aspect_ratio or "1:1",
                    industry=gen_industry,
                    reference_image=reference_image,
                )

                if img_result["success"]:
                    output_image = {
                        "file_hash": img_result["filename"],
                        "file_directory": img_result["image_path"],
                    }
                    gen_seed = img_result.get("seed")
                    gen_method = img_result.get("control_type")  # I2I인 경우
                    output_text = ad_result["ad_copy"]  # 광고 문구도 함께 반환
                else:
                    output_text = f"이미지 생성 실패: {img_result.get('error', '알 수 없는 오류')}"

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
):
    """
    광고 생성 파이프라인 통합 서비스
    - task를 통해 작업 상태 관리
    """
    # 작업 생성
    if create_task_entry:
        create_task(task_id)

    try:
        # 1. 입력 수집/저장
        update_task_progress(task_id, 5, TaskStatus.INGESTING)
        ingest = await ingest_user_message(
            db=db,
            input_text=input_text,
            session_id=session_id,
            user_id=user_id,
            image=image,
        )

        # 2. 콘텐츠 생성
        update_task_progress(task_id, 30, TaskStatus.GENERATING)
        gen_result = await generate_contents(
            input_text=input_text,
            input_image=ingest.input_image,
            generation_type=generation_type,
            style=style,
            aspect_ratio=aspect_ratio,
        )

        # 3. 결과 DB 저장
        update_task_progress(task_id, 80, TaskStatus.PERSISTING)
        persist_generation_result(
            db=db,
            session_id=ingest.session_id,
            gen=gen_result,
        )

        # 4. 완료
        result = {
            "session_id": ingest.session_id,
            "output": gen_result.to_public_dict(),
        }
        complete_task(task_id, result)
        
        return result

    except Exception as exc:
        fail_task(task_id, str(exc))
        raise
