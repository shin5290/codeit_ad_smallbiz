import os, hashlib
from fastapi import HTTPException, FileResponse, UploadFile
from typing import Optional, List
from sqlalchemy.orm import Session

import backend.process_db as process_db

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
    - 반환: [{"bytes": ..., "path": ...}]
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
        disk_path = os.path.join(subdir, filename)

        if not os.path.exists(disk_path):
            with open(disk_path, "wb") as out:
                out.write(contents)

        payloads.append({"bytes": contents, "path": disk_path})

    return payloads


def get_image_file_response(db: Session, file_hash: str) -> FileResponse:
    """
    파일 경로 조회 및 FileResponse 반환
    """
    img = process_db.get_image_by_hash(db, file_hash)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    path = img.file_directory
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(path)

