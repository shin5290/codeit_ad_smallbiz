import os, hashlib, logging
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse
from typing import Optional, Dict
from sqlalchemy.orm import Session

from PIL import Image
from src.backend import process_db

logger = logging.getLogger(__name__)

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

async def save_uploaded_image(*, image: UploadFile, base_dir: str) -> Optional[dict]:
    """
    업로드 이미지 디스크 저장
    - 입력값: UploadFile (fastapi), 저장 베이스 디렉토리(/data/uploads 등)
    - 반환: {"file_hash": ..., "file_directory": ...}
    """
    if not image:
        logger.warning("save_uploaded_image: image is None")
        return None

    try:
        # 베이스 디렉토리 생성
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"save_uploaded_image: base_dir={base_dir}")

        # 이미지 데이터 읽기
        contents = await image.read()
        if not contents:
            logger.warning("save_uploaded_image: image contents is empty")
            return None

        logger.info(f"save_uploaded_image: read {len(contents)} bytes from {getattr(image, 'filename', 'unknown')}")

        # 해시 계산 및 파일명 생성
        file_hash = sha256_hex(contents)
        ext = ext_from_content_type(getattr(image, "content_type", None))
        logger.info(f"save_uploaded_image: file_hash={file_hash}, ext={ext}, content_type={getattr(image, 'content_type', None)}")

        # 서브디렉토리 생성 (해시 앞 2자리)
        subdir = os.path.join(base_dir, file_hash[:2])
        os.makedirs(subdir, exist_ok=True)
        logger.info(f"save_uploaded_image: created subdir={subdir}")

        # 전체 파일 경로
        filename = f"{file_hash}{ext}"
        file_directory = os.path.join(subdir, filename)

        # 파일 저장
        if not os.path.exists(file_directory):
            with open(file_directory, "wb") as out:
                out.write(contents)
            logger.info(f"save_uploaded_image: saved file to {file_directory}")
        else:
            logger.info(f"save_uploaded_image: file already exists at {file_directory}")

        result = {"file_hash": file_hash, "file_directory": file_directory}
        logger.info(f"save_uploaded_image: returning {result}")
        return result

    except Exception as e:
        logger.error(f"save_uploaded_image: error occurred: {e}", exc_info=True)
        raise


def load_image_from_payload(payload: Optional[Dict]) -> Optional[Image.Image]:
    """
    payload를 PIL Image로 변환
    payload: {"file_hash": ..., "file_directory": ...}
    """
    if not payload:
        return None
    image_path = payload.get("file_directory")
    if not image_path or not os.path.exists(image_path):
        return None
    return Image.open(image_path)


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


def image_payload(image) -> Optional[Dict]:
    """이미지 객체를 payload dict로 변환"""
    if not image:
        return None
    return {
        "file_hash": image.file_hash,
        "file_directory": image.file_directory,
    }