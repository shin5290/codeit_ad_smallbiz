from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.backend import process_db, schemas
from src.backend import services

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
