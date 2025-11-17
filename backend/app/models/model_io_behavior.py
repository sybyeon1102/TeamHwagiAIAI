from collections.abc import Sequence

from pydantic import BaseModel


class PoseFrame(BaseModel):
    """
    단일 포즈 프레임의 특징 벡터를 나타낸다.
    Represents a single pose frame feature vector.
    """

    features: list[float]


class BehaviorAnalyzeRequest(BaseModel):
    """
    여러 포즈 프레임으로 구성된 시퀀스를 입력으로 받는다.
    Accepts a sequence of pose frames as input.
    """

    frames: Sequence[PoseFrame]


class BehaviorAnalyzeResponse(BaseModel):
    """
    이상 여부와 normal/anomaly 점수를 응답으로 반환한다.
    Returns anomaly flag and normal/anomaly scores as response.
    """

    is_anomaly: bool
    normal_score: float
    anomaly_score: float
