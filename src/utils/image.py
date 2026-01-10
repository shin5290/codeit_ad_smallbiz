import os, hashlib
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse
from typing import Optional, List
from sqlalchemy.orm import Session

from src.backend import process_db
from .analyze import ImageReference

def sha256_hex(data: bytes) -> str:
    """
    bytes 데이터를 SHA-256 해시로 변환
    """
    return hashlib.sha256(data).hexdigest()

def ext_from_content_type(ct: Optional[str]) -> str:
    """
    확장자 매핑
    """
    if ct == "image/png":
        return ".png"
    if ct in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if ct == "image/webp":
        return ".webp"
    return ".bin"

async def save_uploaded_images(*, images: List[UploadFile], base_dir: str) -> List[dict]:
    """
    업로드 이미지 디스크 저장
    - 입력값: UploadFile 리스트(fastapi), 저장 베이스 디렉토리(/data/uploads 등)
    - 반환: [{"file_hash": ..., "file_directory": ...}]
    """
    if not images:
        return []

    os.makedirs(base_dir, exist_ok=True)

    payloads = []
    for f in images:
        contents = await f.read()
        if not contents:
            continue

        file_hash = sha256_hex(contents)
        ext = ext_from_content_type(getattr(f, "content_type", None))

        subdir = os.path.join(base_dir, file_hash[:2])
        os.makedirs(subdir, exist_ok=True)

        filename = f"{file_hash}{ext}"
        file_directory = os.path.join(subdir, filename)

        if not os.path.exists(file_directory):
            with open(file_directory, "wb") as out:
                out.write(contents)

        payloads.append({"file_hash": file_hash, "file_directory": file_directory})

    return payloads


def get_image_file_response(db: Session, file_hash: str) -> FileResponse:
    """
    파일 경로 조회 및 FileResponse 반환(프론트용)
    """
    img = process_db.get_image_by_hash(db, file_hash)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    path = img.file_directory
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(path)


def resolve_image_reference(
    *,
    db: Session,
    user_id: int,
    image_reference: Optional[ImageReference],
    before_chat_history_id: Optional[int] = None,
):
    """
    ImageReference를 실제 이미지로 해석한다.
    - 유저 전체 히스토리 기준
    - before_chat_history_id가 있으면 이전 이미지 판단에 사용
    """
    if not image_reference:
        return None

    return process_db.resolve_image_reference(
        db=db,
        user_id=user_id,
        role=image_reference.role,
        position=image_reference.position,
        index=image_reference.index,
        before_chat_history_id=before_chat_history_id,
    )