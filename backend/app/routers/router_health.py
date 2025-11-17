from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()


@router.get(
    "/",
    summary="기본 헬스 체크 / Basic health check",
)
async def get_health_status() -> dict[str, str]:
    """
    서버 상태를 확인하는 헬스 체크 엔드포인트.
    Health check endpoint to verify server status.
    """
    settings = get_settings()

    return {
        "status": "ok",
        "app_version": settings.app_version,
        "environment": settings.environment,
    }
