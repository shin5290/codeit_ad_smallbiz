from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import src.backend.process_db as process_db
import src.backend.schemas as schemas
import src.backend.services as services

router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/me")
def me(current_user = Depends(services.get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    return {"name": current_user.name}

@router.post("/signup", response_model=schemas.UserResponse)
def signup(user: schemas.SignupRequest, db: Session = Depends(process_db.get_db)):
    try:
        return services.register_user(db, user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/login", response_model=schemas.AuthResponse)
def login(
    response: Response,
    db: Session = Depends(process_db.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    services.authenticate_user(db, form_data.username, form_data.password, response)
    return {"ok": True}


@router.post("/logout", response_model=schemas.AuthResponse)
def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return {"ok": True}

@router.put("/user", response_model=schemas.UserResponse)
def update_user(
    user_data: schemas.UpdateUserRequest,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    return services.update_user(db, current_user, user_data)

@router.delete("/user", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    delete_data: schemas.DeleteUserRequest,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    services.delete_user(db, current_user, delete_data.login_pw)
    return Response(status_code=status.HTTP_204_NO_CONTENT)