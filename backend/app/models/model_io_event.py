from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EventLog(BaseModel):
    """
    단일 이벤트 로그를 나타낸다.
    Represents a single event log.
    """

    timestamp: datetime
    event_type: str
    session_id: str | None = None
    user_id: str | None = None
    data: dict[str, Any] | None = None


class EventLogRequest(BaseModel):
    """
    이벤트 로그를 기록하기 위한 요청 본문.
    Request body for recording an event log.
    """

    event: EventLog


class EventLogResponse(BaseModel):
    """
    이벤트 로그 기록 결과 응답.
    Response for event log recording result.
    """

    success: bool
    event_id: str | None = None
