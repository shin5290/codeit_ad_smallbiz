import asyncio
import json
import os
from datetime import date, datetime, timedelta
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.backend import process_db, schemas
from src.backend import services
from src.utils.admin_logs import (
    LOG_STREAM_POLL_INTERVAL,
    LOG_STREAM_TIMEOUT_SECONDS,
    LOG_TAIL_DEFAULT_LINES,
    LOG_TAIL_MAX_LINES,
    LOG_STREAM_SEMAPHORE,
    _acquire_log_stream_slot,
    _filter_log_line,
    _is_valid_log_date,
    _is_within_root,
    _mask_sensitive,
    _read_full_file,
    _resolve_log_dir,
    _resolve_log_file,
    _tail_lines,
)
from src.utils.logging import get_current_log_file, get_log_root, get_run_id

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(current_user=Depends(services.get_current_user)):
    if not current_user or not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return current_user


@router.get("/users", response_model=schemas.AdminUserListResponse)
def list_users(
    limit: int = Query(15, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: int | None = Query(None),
    login_id: str | None = Query(None),
    name: str | None = Query(None),
    is_admin: bool | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    start_at = None
    end_at = None
    if start_date:
        start_at = datetime.combine(start_date, datetime.min.time())
    if end_date:
        end_at = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)

    users, total = process_db.list_users(
        db,
        limit=limit,
        offset=offset,
        user_id=user_id,
        login_id=login_id,
        name=name,
        is_admin=is_admin,
        start_at=start_at,
        end_at=end_at,
    )
    return {"users": users, "total": total}


@router.post("/users/delete", response_model=schemas.AdminDeleteUsersResponse)
def delete_users(
    payload: schemas.AdminDeleteUsersRequest,
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    user_ids = payload.user_ids or []
    if not user_ids:
        raise HTTPException(status_code=400, detail="삭제할 사용자가 없습니다.")

    deleted_ids, skipped_ids = process_db.delete_users_by_ids(
        db,
        user_ids=user_ids,
        exclude_admin=True,
        exclude_user_id=current_user.user_id,
    )
    return {"deleted_ids": deleted_ids, "skipped_ids": skipped_ids}


@router.get("/generations", response_model=schemas.AdminGenerationPage)
def list_generations(
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1, le=50),
    user_id: int | None = Query(None),
    login_id: str | None = Query(None),
    session_id: str | None = Query(None),
    content_type: str | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    offset = (page - 1) * limit

    start_at = None
    end_at = None
    if start_date:
        start_at = datetime.combine(start_date, datetime.min.time())
    if end_date:
        end_at = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)

    items, total = process_db.get_admin_generation_page(
        db,
        limit=limit,
        offset=offset,
        user_id=user_id,
        login_id=login_id,
        session_id=session_id,
        content_type=content_type,
        start_at=start_at,
        end_at=end_at,
    )

    result_items: list[schemas.AdminGenerationItem] = []
    for gen in items:
        session = gen.session
        user = session.user if session else None
        latest_input = process_db.get_latest_user_input_before(
            db,
            session_id=gen.session_id,
            before_at=gen.created_at,
        )
        raw_input_text = latest_input.content if latest_input else None
        refined_input_text = gen.input_text

        result_items.append(
            schemas.AdminGenerationItem(
                id=gen.id,
                session_id=gen.session_id,
                user_id=session.user_id if session else None,
                login_id=user.login_id if user else None,
                name=user.name if user else None,
                content_type=gen.content_type,
                input_text=raw_input_text,
                refined_input_text=refined_input_text,
                output_text=gen.output_text,
                prompt=gen.prompt,
                input_image=(
                    schemas.AdminImageRef(file_hash=gen.input_image.file_hash)
                    if gen.input_image
                    else None
                ),
                output_image=(
                    schemas.AdminImageRef(file_hash=gen.output_image.file_hash)
                    if gen.output_image
                    else None
                ),
                generation_method=gen.generation_method,
                style=gen.style,
                industry=gen.industry,
                seed=gen.seed,
                strength=gen.strength,
                aspect_ratio=gen.aspect_ratio,
                created_at=gen.created_at,
            )
        )

    return {
        "items": result_items,
        "total": total,
        "page": page,
        "limit": limit,
    }


@router.get("/sessions", response_model=schemas.AdminSessionPage)
def list_sessions(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    user_id: int | None = Query(None),
    login_id: str | None = Query(None),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    start_at = None
    end_at = None
    if from_date:
        start_at = datetime.combine(from_date, datetime.min.time())
    if to_date:
        end_at = datetime.combine(to_date, datetime.min.time()) + timedelta(days=1)

    rows, total = process_db.get_admin_session_page(
        db,
        limit=limit,
        offset=offset,
        query=query,
        user_id=user_id,
        login_id=login_id,
        start_at=start_at,
        end_at=end_at,
    )

    items: list[schemas.AdminSessionItem] = []
    for row in rows:
        items.append(
            schemas.AdminSessionItem(
                session_id=row.session_id,
                user_id=row.user_id,
                login_id=row.login_id,
                created_at=row.created_at,
                last_message_at=row.last_message_at,
                message_count=row.message_count or 0,
                generation_count=row.generation_count or 0,
            )
        )

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/sessions/{session_id}", response_model=schemas.AdminSessionDetail)
def get_session_detail(
    session_id: str,
    message_limit: int = Query(200, ge=1, le=500),
    message_offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    role: str | None = Query(None),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    has_image: bool | None = Query(None),
    generation_limit: int = Query(5, ge=0, le=50),
    mask: bool = Query(True),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    overview = process_db.get_admin_session_overview(db, session_id=session_id)
    if not overview:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    start_at = None
    end_at = None
    if from_date:
        start_at = datetime.combine(from_date, datetime.min.time())
    if to_date:
        end_at = datetime.combine(to_date, datetime.min.time()) + timedelta(days=1)

    messages, message_total = process_db.get_admin_session_messages(
        db,
        session_id=session_id,
        limit=message_limit,
        offset=message_offset,
        query=query,
        role=role,
        start_at=start_at,
        end_at=end_at,
        has_image=has_image,
    )

    message_items: list[schemas.AdminSessionMessage] = []
    for msg in messages:
        content = msg.content
        if mask:
            content = _mask_sensitive(content)
        message_items.append(
            schemas.AdminSessionMessage(
                id=msg.id,
                role=msg.role,
                content=content,
                created_at=msg.created_at,
                image=(
                    schemas.AdminImageRef(file_hash=msg.image.file_hash)
                    if msg.image
                    else None
                ),
            )
        )

    generation_items: list[schemas.AdminGenerationSummary] = []
    if generation_limit > 0:
        generations = process_db.get_generation_history_by_session(
            db,
            session_id=session_id,
            limit=generation_limit,
        )
        for gen in generations:
            generation_items.append(
                schemas.AdminGenerationSummary(
                    id=gen.id,
                    content_type=gen.content_type,
                    output_text=gen.output_text,
                    output_image=(
                        schemas.AdminImageRef(file_hash=gen.output_image.file_hash)
                        if gen.output_image
                        else None
                    ),
                    created_at=gen.created_at,
                    task_id=None,
                )
            )

    log_hint = None
    log_file = get_current_log_file()
    if log_file:
        log_root = get_log_root().resolve()
        resolved = log_file.resolve()
        if _is_within_root(resolved, log_root):
            log_hint = schemas.AdminLogHint(
                date=resolved.parent.name,
                file=resolved.name,
            )

    return {
        "session_id": overview.session_id,
        "user_id": overview.user_id,
        "login_id": overview.login_id,
        "created_at": overview.created_at,
        "last_message_at": overview.last_message_at,
        "message_count": message_total,
        "run_id": get_run_id(),
        "log_hint": log_hint,
        "messages": message_items,
        "generations": generation_items,
    }


@router.get("/messages", response_model=schemas.AdminMessagePage)
def search_messages(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    level: str | None = Query(None),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    mask: bool = Query(True),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    start_at = None
    end_at = None
    if from_date:
        start_at = datetime.combine(from_date, datetime.min.time())
    if to_date:
        end_at = datetime.combine(to_date, datetime.min.time()) + timedelta(days=1)

    rows, total = process_db.search_admin_messages(
        db,
        query=query,
        level=level,
        start_at=start_at,
        end_at=end_at,
        limit=limit,
        offset=offset,
    )

    items: list[schemas.AdminMessageItem] = []
    for msg, user_id, login_id in rows:
        content = msg.content
        if mask:
            content = _mask_sensitive(content)
        items.append(
            schemas.AdminMessageItem(
                id=msg.id,
                session_id=msg.session_id,
                user_id=user_id,
                login_id=login_id,
                role=msg.role,
                content=content,
                created_at=msg.created_at,
            )
        )

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/logs/dates", response_model=schemas.AdminLogDatesResponse)
def list_log_dates(
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    log_root = get_log_root()
    if not log_root.exists() or not log_root.is_dir():
        return {"dates": []}

    dates = [
        entry.name
        for entry in log_root.iterdir()
        if entry.is_dir() and _is_valid_log_date(entry.name)
    ]
    dates.sort(reverse=True)
    return {"dates": dates}


@router.get("/logs/files", response_model=schemas.AdminLogFilesResponse)
def list_log_files(
    date: str = Query(...),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    log_dir = _resolve_log_dir(date)
    current_log = get_current_log_file()
    current_log_name = current_log.name if current_log else None
    
    files = []
    for entry in log_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".log":
            continue
        # 현재 실행 중인 로그 파일은 이전 로그 목록에서 제외
        if current_log_name and entry.name == current_log_name:
            continue
        stat = entry.stat()
        files.append(
            schemas.AdminLogFileItem(
                name=entry.name,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    files.sort(key=lambda item: item.modified_at, reverse=True)
    return {"files": files}


@router.get("/logs/tail", response_model=schemas.AdminLogTailResponse)
def tail_log_file(
    date: str = Query(...),
    file: str = Query(...),
    lines: int = Query(LOG_TAIL_DEFAULT_LINES, ge=10, le=LOG_TAIL_MAX_LINES),
    query: str | None = Query(None),
    level: str | None = Query(None),
    mask: bool = Query(True),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    target = _resolve_log_file(date, file)
    raw_lines = _tail_lines(target, lines)
    filtered: list[str] = []
    for line in raw_lines:
        if not _filter_log_line(line, query, level):
            continue
        filtered.append(_mask_sensitive(line) if mask else line)
    return {"lines": filtered}


@router.get("/logs/stream")
async def stream_log_file(
    date: str = Query(...),
    file: str = Query(...),
    query: str | None = Query(None),
    level: str | None = Query(None),
    mask: bool = Query(True),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    target = _resolve_log_file(date, file)
    await _acquire_log_stream_slot()

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        try:
            with target.open("r", encoding="utf-8", errors="replace") as file_obj:
                file_obj.seek(0, os.SEEK_END)
                while True:
                    elapsed = loop.time() - start_time
                    if elapsed > LOG_STREAM_TIMEOUT_SECONDS:
                        break
                    line = file_obj.readline()
                    if not line:
                        await asyncio.sleep(LOG_STREAM_POLL_INTERVAL)
                        continue
                    line = line.rstrip("\n")
                    if not _filter_log_line(line, query, level):
                        continue
                    if mask:
                        line = _mask_sensitive(line)
                    payload = {"line": line}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            LOG_STREAM_SEMAPHORE.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/logs/current", response_model=schemas.AdminCurrentLogResponse)
def get_current_log_info(
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    """현재 실행 중인 로그 파일 정보를 반환합니다."""
    current_log = get_current_log_file()
    if not current_log:
        return {"date": None, "file": None, "run_id": get_run_id()}
    
    log_root = get_log_root().resolve()
    resolved = current_log.resolve()
    if not _is_within_root(resolved, log_root):
        return {"date": None, "file": None, "run_id": get_run_id()}
    
    return {
        "date": resolved.parent.name,
        "file": resolved.name,
        "run_id": get_run_id(),
    }


@router.get("/logs/full", response_model=schemas.AdminLogFullResponse)
def read_full_log_file(
    date: str = Query(...),
    file: str = Query(...),
    query: str | None = Query(None),
    level: str | None = Query(None),
    mask: bool = Query(True),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    """전체 로그 파일을 읽어서 반환합니다."""
    target = _resolve_log_file(date, file)
    lines = _read_full_file(target, query, level, mask)
    return {"lines": lines, "total_lines": len(lines)}


@router.get("/logs/stream/current")
async def stream_current_log(
    query: str | None = Query(None),
    level: str | None = Query(None),
    mask: bool = Query(True),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(require_admin),
):
    """현재 실행 중인 로그 파일을 실시간으로 스트리밍합니다."""
    current_log = get_current_log_file()
    if not current_log or not current_log.exists():
        raise HTTPException(status_code=404, detail="현재 실행 중인 로그가 없습니다.")
    
    await _acquire_log_stream_slot()

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        try:
            with current_log.open("r", encoding="utf-8", errors="replace") as file_obj:
                file_obj.seek(0, os.SEEK_END)
                while True:
                    elapsed = loop.time() - start_time
                    if elapsed > LOG_STREAM_TIMEOUT_SECONDS:
                        break
                    line = file_obj.readline()
                    if not line:
                        await asyncio.sleep(LOG_STREAM_POLL_INTERVAL)
                        continue
                    line = line.rstrip("\n")
                    if not _filter_log_line(line, query, level):
                        continue
                    if mask:
                        line = _mask_sensitive(line)
                    payload = {"line": line}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            LOG_STREAM_SEMAPHORE.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
