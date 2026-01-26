import os, hashlib
from datetime import datetime, timezone
from email.utils import formatdate, parsedate_to_datetime
from fastapi import HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, Response
from typing import Optional, Dict
from sqlalchemy.orm import Session
from PIL import Image

from src.backend import process_db
from .logging import get_logger

logger = get_logger(__name__)

CACHE_CONTROL_HEADER = "private, max-age=3600"
THUMB_DIRNAME = "thumbs"
THUMB_MAX_SIZE = 256

def _stat_cache_headers(path: str) -> tuple[str, str]:
    stat = os.stat(path)
    etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    last_modified = formatdate(stat.st_mtime, usegmt=True)
    return etag, last_modified

def _etag_matches(if_none_match: str, etag: str) -> bool:
    tags = [tag.strip() for tag in if_none_match.split(",") if tag.strip()]
    if "*" in tags:
        return True
    return etag in tags

def _is_not_modified(request: Optional[Request], path: str, etag: str) -> bool:
    if not request:
        return False
    if_none_match = request.headers.get("if-none-match")
    if if_none_match:
        return _etag_matches(if_none_match, etag)
    if_modified_since = request.headers.get("if-modified-since")
    if not if_modified_since:
        return False
    try:
        since = parsedate_to_datetime(if_modified_since)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        modified_at = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        return modified_at <= since
    except Exception:
        return False

def _thumbnail_path(path: str) -> str:
    directory = os.path.dirname(path)
    thumb_dir = os.path.join(directory, THUMB_DIRNAME)
    return os.path.join(thumb_dir, os.path.basename(path))

def _guess_format(path: str, img_format: Optional[str]) -> str:
    if img_format:
        return img_format
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".png":
        return "PNG"
    if ext == ".webp":
        return "WEBP"
    return "PNG"

def _ensure_thumbnail(path: str) -> str:
    thumb_path = _thumbnail_path(path)
    if os.path.exists(thumb_path):
        return thumb_path
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    with Image.open(path) as img:
        img.thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE))
        img_format = _guess_format(path, img.format)
        if img_format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        save_kwargs: dict[str, object] = {}
        if img_format.upper() == "JPEG":
            save_kwargs.update({"quality": 82, "optimize": True, "progressive": True})
        elif img_format.upper() == "PNG":
            save_kwargs["optimize"] = True
        img.save(thumb_path, format=img_format, **save_kwargs)
    return thumb_path

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


def get_image_file_response(
    db: Session,
    file_hash: str,
    request: Optional[Request] = None,
    size: Optional[str] = None,
) -> FileResponse | Response:
    """
    파일 경로 조회 및 FileResponse 반환(프론트용)
    """
    img = process_db.get_image_by_hash(db, file_hash)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    path = img.file_directory
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing on disk")

    resolved_path = path
    if size:
        size_key = size.lower()
        if size_key in ("thumb", "thumbnail", "small"):
            try:
                resolved_path = _ensure_thumbnail(path)
            except Exception as exc:
                logger.warning("thumbnail generation failed: %s", exc, exc_info=True)
                resolved_path = path
        elif size_key not in ("full", "original"):
            raise HTTPException(status_code=400, detail="Invalid image size")

    etag, last_modified = _stat_cache_headers(resolved_path)
    headers = {
        "Cache-Control": CACHE_CONTROL_HEADER,
        "ETag": etag,
        "Last-Modified": last_modified,
    }
    if _is_not_modified(request, resolved_path, etag):
        return Response(status_code=304, headers=headers)
    return FileResponse(resolved_path, headers=headers)


def image_payload(image) -> Optional[Dict]:
    """이미지 객체를 payload dict로 변환"""
    if not image:
        return None
    return {
        "file_hash": image.file_hash,
        "file_directory": image.file_directory,
    }
