import logging, os
from dataclasses import dataclass
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from typing import List, Optional, Dict

from src.backend import process_db, schemas, models
from src.backend.task import TaskStatus, create_task, update_task_progress, complete_task, fail_task
from src.utils.config import PROJECT_ROOT
from src.utils.security import verify_password, create_access_token, decode_token
from src.utils.session import normalize_session_id, ensure_chat_session
from src.utils.image import save_uploaded_images, resolve_image_reference
from src.utils.analyze import detect_content_type , detect_image_references


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
    - 비밀번호 변경(현재 비밀번호 검증 필요)
    """
    if update.new_password:
        if not update.current_password or not verify_password(update.current_password, current_user.login_pw):
            raise HTTPException(400, "비밀번호가 올바르지 않습니다.")

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
    input_images: List[dict]
    
async def ingest_user_message(
    *,
    db,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    images: List[UploadFile],
    image_payloads: Optional[List[dict]] = None,
) -> IngestResult:
    """
    입력 수집/저장 레이어
    - 세션 확보/생성/유저귀속
    - 유저 텍스트 저장
    - 업로드 이미지 디스크 저장 + DB 연결
    - 반환: session_key, chat_row_id(또는 chat_row), saved_payloads(필요하면)
    """

    session_id = normalize_session_id(session_id) # 프론트에서 받아온 값 정규화

    # 1) 세션 확보/생성/귀속
    session_key = ensure_chat_session(db, session_id, user_id)

    # 2) 텍스트 DB 저장
    chat_row = process_db.save_chat_message(
        db,
        {"session_id": session_key, "role": "user", "content": input_text},
    )

    # 3) 이미지 디스크 저장 + DB 저장
    payloads = list(image_payloads or [])
    if not payloads and images:
        base_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        payloads = await save_uploaded_images(images=images, base_dir=base_dir)

        if payloads:
            process_db.attach_images_to_chat(
                db=db,
                chat_history_id=chat_row.id,
                images=payloads,
                role="input",
            )

    return IngestResult(
        session_id=session_key,
        chat_history_id=chat_row.id,
        input_images=payloads,
    )

@dataclass
class ParsedIntent:
    input_text: str
    content_type: str   # "text", "image", "both"
    input_images: List[Dict] = None # 사용자 업로드 이미지 + 참조 이미지


async def analyze_intent(
    *,
    db: Session,
    user_id: int,
    input_text: str,
    uploaded_images: List[dict],
    before_chat_history_id: Optional[int] = None,
) -> ParsedIntent:
    """
    사용자 입력 텍스트를 분석하여 의도를 파악하고 필요한 이미지를 수집
    
    Args:
        db: 데이터베이스 세션
        user_id: 사용자 ID
        input_text: 사용자 입력 텍스트
        uploaded_images: 사용자가 현재 업로드한 이미지들
        before_chat_history_id: 현재 메시지 이전의 채팅 히스토리 ID
        
    Returns:
        ParsedIntent: 파싱된 의도 정보 (업로드 이미지 + 참조 이미지 포함)
    """
    if not input_text:
        raise ValueError("Input text cannot be empty")
    
    normalized_text = input_text.lower()
    
    # 1. 콘텐츠 타입 감지
    content_type = detect_content_type(normalized_text)
    
    # 2. 이미지 참조 감지
    image_references = detect_image_references(normalized_text)
    
    # 3. 참조된 이미지 resolve (DB에서 실제 이미지 가져오기)
    images = list(uploaded_images or [])
    for img in image_references:
        referenced_image = resolve_image_reference(
            db=db,
            user_id=user_id,
            image_reference=img,
            before_chat_history_id=before_chat_history_id,
        )
        if referenced_image:
            images.append({
                "file_hash": referenced_image.file_hash,
                "file_directory": referenced_image.file_directory,
            })

    return ParsedIntent(
        input_text=input_text,
        content_type=content_type,
        input_images=images,
    )



@dataclass
class GeneratedContent:
    content_type: str
    input_text: str
    input_images: List[dict] = None
    output_text: Optional[str] = None
    output_images: Optional[List[dict]] = None
    generation_method: Optional[str] = None
    style: Optional[str] = None
    industry: Optional[str] = None
    seed: Optional[int] = None
    aspect_ratio: Optional[str] = None

    def to_public_dict(self) -> dict:
        images = []
        for img in self.output_images or []:
            if "file_hash" in img:
                images.append(img["file_hash"])
            elif "file_directory" in img:
                images.append(img["file_directory"])
        return {
            "content_type": self.content_type,
            "output_text": self.output_text,
            "images": images,
        }



async def generate_contents(parsed: ParsedIntent) -> GeneratedContent:
    """
    분석 결과를 기반으로 실제 콘텐츠 생성
    
    Args:
        parsed: 분석된 의도 (텍스트, 이미지 포함)
        
    Returns:
        GeneratedContent: 생성된 광고 콘텐츠
    """

    output_text = ""
    output_images = []
    gen_method = None
    gen_style = None
    gen_industry = None
    gen_aspect_ratio = None
    gen_seed = None
    
    if parsed.content_type in ["text", "both"]:
        print("텍스트 생성")
        output_text = "생성된 광고 문구 예시입니다."
    if parsed.content_type in ["image", "both"]:
        print("이미지 생성")

    if not output_text:
        output_text = "이미지/텍스트 생성이 완료되었습니다."

    return GeneratedContent(
        content_type=parsed.content_type,
        input_text=parsed.input_text,
        input_images=parsed.input_images,
        output_text=output_text,
        output_images=output_images,
        generation_method=gen_method,
        style=gen_style,
        industry=gen_industry,
        seed=gen_seed,
        aspect_ratio=gen_aspect_ratio,
    )


def persist_generation_result(
    *,
    db: Session,
    session_id: str,
    gen: GeneratedContent,
):
    """생성 결과를 DB에 저장"""

    # Chat History에 텍스트 결과 저장
    assistant_row = process_db.save_chat_message(
        db,
        {"session_id": session_id, "role": "assistant", "content": gen.output_text},
    )

    # Chat History에 이미지 저장(history_image + image_matching)
    if gen.output_images:
        process_db.attach_images_to_chat(
            db=db,
            chat_history_id=assistant_row.id,
            images=gen.output_images,
            role="output",
        )

    # Generation History 이력 저장
    generation_row = process_db.save_generation_history(
        db=db,
        data={
            "session_id": session_id,
            "content_type": gen.content_type,
            "input_text": gen.input_text,
            "output_text": gen.output_text,
            "generation_method": gen.generation_method,
            "style": gen.style,
            "industry": gen.industry,
            "seed": gen.seed,
            "aspect_ratio": gen.aspect_ratio,
        }
    )

    # Generation History - History image 저장
    if gen.input_images:
        process_db.attach_images_to_generation(
            db=db,
            generation_history_id=generation_row.id,
            images=gen.input_images,
            role="input",
        )

    if gen.output_images:
        process_db.attach_images_to_generation(
            db=db,
            generation_history_id=generation_row.id,
            images=gen.output_images,
            role="output",
        )





async def handle_generate_pipeline(
    *,
    db: Session,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    images: List[UploadFile],
    task_id: str,
    image_payloads: Optional[List[dict]] = None,
    create_task_entry: bool = True,
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
            images=images,
            image_payloads=image_payloads,
        )

        # 2. 의도 파악 + 이미지 수집
        update_task_progress(task_id, 20, TaskStatus.ANALYZING)
        parsed = await analyze_intent(
            db=db,
            user_id=user_id,
            input_text=input_text,
            uploaded_images=ingest.input_images,
            before_chat_history_id=ingest.chat_history_id,
        )

        # 3. 콘텐츠 생성
        update_task_progress(task_id, 50, TaskStatus.GENERATING)
        gen_result = await generate_contents(parsed)

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
