# backend/app/routers/router_behavior.py

from fastapi import APIRouter, HTTPException

from project_core import Err, Ok

from app.errors import map_inference_error_to_http_exception
from app.models.model_io_behavior import (
    BehaviorAnalyzeRequest,
    BehaviorAnalyzeResponse,
)
from app.services.service_behavior import analyze_behavior


router = APIRouter(
    prefix="/behavior",
    tags=["behavior"],
)


@router.post(
    "/analyze",
    response_model=BehaviorAnalyzeResponse,
    summary="포즈 기반 정상/이상 행동 분석",
    description=(
        "포즈 프레임 시퀀스를 입력 받아 LSTM 기반으로 정상/이상 행동을 분석합니다.\n"
        "Analyze a sequence of pose frames using an LSTM-based model "
        "to determine whether the behavior is normal or anomalous."
    ),
)
def analyze_behavior_endpoint(
    request: BehaviorAnalyzeRequest,
) -> BehaviorAnalyzeResponse:
    """
    BehaviorAnalyzeRequest를 받아 LSTM 추론 레이어를 호출하고,
    결과를 BehaviorAnalyzeResponse로 반환합니다.

    Takes BehaviorAnalyzeRequest, calls the LSTM inference layer,
    and returns BehaviorAnalyzeResponse.
    """
    result = analyze_behavior(request)

    match result:
        case Ok(value=response):
            return response

        case Err(error=inference_error):
            # 도메인 InferenceError를 HTTPException으로 변환해 raise.
            # Convert domain-level InferenceError into HTTPException.
            raise map_inference_error_to_http_exception(inference_error)

        case _:
            # Result 타입이 아닌 예기치 못한 값이 온 경우.
            # Unexpected result type (invariant broken).
            raise HTTPException(
                status_code=500,
                detail="Unexpected result type from analyze_behavior.",
            )
