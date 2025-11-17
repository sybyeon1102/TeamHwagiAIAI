from fastapi import APIRouter, HTTPException, status

from project_core import Err, Ok

from app.models.model_io_event import EventLogRequest, EventLogResponse
from app.services.service_event import record_event_log

router = APIRouter()


@router.post(
    "/",
    summary="이벤트 로그 기록 / Record event log",
    response_model=EventLogResponse,
)
async def record_event_log_endpoint(
    request: EventLogRequest,
) -> EventLogResponse:
    """
    단일 이벤트 로그를 기록하고 결과를 반환한다.
    Record a single event log and return the result.
    """
    result = record_event_log(request)

    match result:
        case Ok(value=response):
            return response
        case Err(error=message):
            # 문자열 에러 메시지를 HTTP 500으로 래핑 / Wrap string error as HTTP 500
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message,
            )
        case _:
            # Result 패턴이 아닌 값이 돌아온 경우 / Non-Result value returned
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected result type.",
            )
