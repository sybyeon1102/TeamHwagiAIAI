from project_core import Err, Ok

from app.errors import InferenceError, InferenceErrorCode, InferenceResult
from app.models.model_io_behavior import (
    BehaviorAnalyzeRequest,
    BehaviorAnalyzeResponse,
)


def analyze_behavior(
    request: BehaviorAnalyzeRequest,
) -> InferenceResult[BehaviorAnalyzeResponse]:
    """
    LSTM/모델링 레이어를 호출해 이상 행동 여부를 계산한다.
    Call the modeling/inference layer to compute anomaly scores.
    """
    # TODO: modeling 패키지에서 실제 LSTM 추론 헬퍼를 가져온다.
    # TODO: Import actual LSTM inference helper from the `modeling` package.
    #
    # 예시:
    # from modeling.inference.behavior import run_behavior_inference
    # is_anomaly, normal_score, anomaly_score = run_behavior_inference(request.frames)

    try:
        # 여기서는 더미 구현 / Dummy implementation for now
        anomaly_score = 0.1
        normal_score = 1.0 - anomaly_score
    except Exception as exc:  # 외부 라이브러리 예외 래핑 / Wrap external library exceptions
        return Err(
            InferenceError(
                code=InferenceErrorCode.INTERNAL_ERROR,
                message=f"Inference failed: {exc}",
            )
        )

    response = BehaviorAnalyzeResponse(
        is_anomaly=anomaly_score > 0.5,
        normal_score=normal_score,
        anomaly_score=anomaly_score,
    )

    return Ok(response)
