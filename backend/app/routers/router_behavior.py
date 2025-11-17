from fastapi import APIRouter, HTTPException, status

from project_core import Err, Ok

from app.errors import map_inference_error_to_http_exception
from app.models.model_io_behavior import (
    BehaviorAnalyzeRequest,
    BehaviorAnalyzeResponse,
)
from app.services.service_behavior import analyze_behavior

router = APIRouter()


@router.post(
    "/",
    summary="포즈/행동 시퀀스 이상 분석 / Analyze pose/behavior sequence for anomaly",
    response_model=BehaviorAnalyzeResponse,
)
async def analyze_behavior_endpoint(
    request: BehaviorAnalyzeRequest,
) -> BehaviorAnalyzeResponse:
    """
    포즈/행동 시퀀스를 받아 이상 행동 여부를 분석한다.
    Analyze a pose/behavior sequence and decide whether it's anomalous.
    """
    result = analyze_behavior(request)

    match result:
        case Ok(value=response):
            return response
        case Err(error=error):
            # 도메인 에러를 HTTP 응답으로 매핑 / Map domain error to HTTP response
            raise map_inference_error_to_http_exception(error)
        case _:
            # 여기에 도달하면 내부 불변식이 깨진 것 / Internal invariant is broken if reached
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected result type.",
            )
