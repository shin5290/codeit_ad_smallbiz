"""
FastAPI 엔드포인트

상담 챗봇 API 서버
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.consultant_bot import AdvancedConsultantBot
from ..config.settings import get_settings


# Request/Response 모델
class UserProfile(BaseModel):
    """사용자 프로필"""
    industry: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[int] = None
    platforms: list[str] = Field(default_factory=list)


class ConsultRequest(BaseModel):
    """상담 요청"""
    query: str = Field(..., min_length=1, description="사용자 질문")
    user_profile: Optional[UserProfile] = None
    session_id: Optional[str] = None


class ConsultResponse(BaseModel):
    """상담 응답"""
    answer: str
    method: str
    sources: list[dict] = Field(default_factory=list)
    improved: bool = False
    improvement_details: Optional[dict] = None
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    available_methods: list[str]
    index_size: int


# 전역 봇 인스턴스 (싱글톤)
_bot_instance: Optional[AdvancedConsultantBot] = None


def get_bot() -> AdvancedConsultantBot:
    """봇 인스턴스 반환"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = AdvancedConsultantBot()
    return _bot_instance


def create_app() -> FastAPI:
    """FastAPI 앱 생성"""
    app = FastAPI(
        title="AI 마케팅 상담 챗봇 API",
        description="RAG + Agent + MCP + Self-Refine 통합 상담 시스템",
        version="0.1.0",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """헬스체크"""
        try:
            bot = get_bot()
            return HealthResponse(
                status="healthy",
                available_methods=bot.available_methods,
                index_size=bot.rag.index_size if bot.rag else 0,
            )
        except Exception as e:
            return HealthResponse(
                status=f"unhealthy: {str(e)}",
                available_methods=[],
                index_size=0,
            )

    @app.post("/consult", response_model=ConsultResponse)
    async def consult(request: ConsultRequest):
        """
        상담 수행

        Args:
            request: 상담 요청

        Returns:
            상담 응답
        """
        try:
            bot = get_bot()

            # 프로필 변환
            user_profile = None
            if request.user_profile:
                user_profile = {
                    "industry": request.user_profile.industry,
                    "location": request.user_profile.location,
                    "budget": request.user_profile.budget,
                    "platforms": request.user_profile.platforms,
                }

            # 상담 수행
            result = bot.consult(
                query=request.query,
                user_profile=user_profile,
            )

            return ConsultResponse(**result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reset")
    async def reset_session():
        """세션 초기화"""
        try:
            bot = get_bot()
            bot.reset_session()
            return {"status": "success", "message": "Session reset"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/session")
    async def get_session_info():
        """세션 정보 조회"""
        try:
            bot = get_bot()
            return bot.get_session_summary()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# 메인 실행
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "endpoints:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
