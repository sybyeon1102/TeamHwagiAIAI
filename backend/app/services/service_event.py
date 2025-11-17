from project_core import Err, Ok, Result

from app.models.model_io_event import EventLogRequest, EventLogResponse


def record_event_log(
    request: EventLogRequest,
) -> Result[EventLogResponse, str]:
    """
    이벤트 로그를 기록하고 결과를 반환한다.
    Record an event log and return the result.
    """
    # TODO: 실제 저장소(DB, 메시지 큐, 파일 등)에 이벤트를 기록한다.
    # TODO: Persist the event to a real store (DB, message queue, file, etc.).

    try:
        # 여기서는 더미 구현 / Dummy implementation for now
        # 나중에 실제 event_id 를 생성해 채운다.
        # Later, generate a real event_id.
        event_id = "dummy-event-id"

        response = EventLogResponse(
            success=True,
            event_id=event_id,
        )
        return Ok(response)
    except Exception as exc:  # noqa: BLE001
        # 간단히 문자열 에러로 감싼다 (추후 EventError 타입으로 확장 가능).
        # Wrap error as a simple string (can be replaced with EventError later).
        return Err(f"Failed to record event log: {exc}")
