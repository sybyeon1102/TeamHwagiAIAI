from typing import Final

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.routers.router_behavior import router as router_behavior
from app.routers.router_event import router as router_event
from app.routers.router_health import router as router_health


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성한다.
    Create and configure a FastAPI application instance.
    """
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title=resolved_settings.app_title,
        version=resolved_settings.app_version,
        description=resolved_settings.app_description,
    )

    _include_routers(app)

    return app


def _include_routers(app: FastAPI) -> None:
    """
    애플리케이션에 모든 라우터를 등록한다.
    Include all routers into the FastAPI application.
    """
    # 헬스 체크 / Health check
    app.include_router(router_health, prefix="/health", tags=["health"])

    # 행동 분석 / Behavior analysis
    app.include_router(router_behavior, prefix="/behavior", tags=["behavior"])

    # 이벤트 로깅 / Event logging
    app.include_router(router_event, prefix="/event", tags=["event"])


APP: Final[FastAPI] = create_app()
app = APP
