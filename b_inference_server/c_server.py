# server.py
import os, json, time
from typing import Literal, Optional, Dict, Any, List
from collections import deque
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# ===== 설정 =====
LOG_JSONL = os.getenv("LOG_JSONL", "events_log.jsonl")  # ""로 비우면 파일 저장 안 함
KEEP_RECENT = int(os.getenv("KEEP_RECENT", "200"))      # 최근 메모리 보관 건수

app = FastAPI(title="Event Receiver (Test)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===== 데이터 모델 =====
EventType = Literal["START", "END", "PROBS", "HEARTBEAT"]

class EventPayload(BaseModel):
    # 공통 필드
    type: EventType
    ts: float = Field(default_factory=lambda: time.time())
    frame_idx: Optional[int] = None

    # START 전송 예: {"event":"fire","prob":0.83}
    event: Optional[str] = None
    prob: Optional[float] = None

    # END 전송 예: {"event":"fire","duration_sec":5.45}
    duration_sec: Optional[float] = None

    # HEARTBEAT/PROBS 보조 정보
    state: Optional[str] = None
    top_event: Optional[str] = None
    top_prob: Optional[float] = None
    probs: Optional[Dict[str, float]] = None

    # 기타 필드도 허용 (확장성)
    extra: Optional[Dict[str, Any]] = None

# ===== 메모리 버퍼 =====
recent_events: deque[dict] = deque(maxlen=KEEP_RECENT)

def append_and_maybe_dump(d: dict):
    recent_events.append(d)
    if LOG_JSONL:
        try:
            with open(LOG_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] write jsonl failed: {e}")

# ===== 라우트 =====
@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.post("/events")
async def receive_events(payload: EventPayload):
    d = payload.model_dump()
    print(f"[RECV] {d['type']} | event={d.get('event')} | top={d.get('top_event')} "
          f"| p={d.get('prob') or d.get('top_prob')} | frame={d.get('frame_idx')}")
    append_and_maybe_dump(d)
    # 간단한 응답 (필요시 서버 측에서 추가 로직/트리거 가능)
    return {"ok": True, "received_type": d["type"]}

@app.get("/events/recent")
def list_recent(limit: int = 50):
    """최근 수신 이벤트 조회 (기본 50개)"""
    data = list(recent_events)[-limit:]
    return {"count": len(data), "items": data}

@app.post("/echo")  # 자유 테스트용
async def echo(req: Request):
    body = await req.json()
    return {"echo": body, "ts": time.time()}

# ===== 로컬 실행 =====
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("c_server:app", host="0.0.0.0", port=port, reload=True)
