# server_kakao_direct_full.py
# 목적: END 이벤트를 수신하면 Kakao "나에게 보내기" API로 알림 전송
# 특징:
#  - .env 자동 로드 (python-dotenv)
#  - 단독 실행 가능 (python server_kakao_direct_full.py)
#  - /health, /token/check, /token/send_test 제공(바로 점검 가능)

import os
import json
import time
from typing import Optional, Literal

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# ========= .env 로드 =========
load_dotenv()

KAKAO_ACCESS_TOKEN = os.getenv("KAKAO_ACCESS_TOKEN", "").strip()
ALERT_LINK_URL = os.getenv("ALERT_LINK_URL", "").strip()
BTN_TITLE = (os.getenv("ALERT_BUTTON_TITLE", "자세히 보기") or "자세히 보기").strip()

KAKAO_MEMO_ENDPOINT = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
KAKAO_ME_ENDPOINT   = "https://kapi.kakao.com/v2/user/me"  # 토큰 점검용

app = FastAPI(title="Anomaly → Kakao Memo (.env, run-direct)")

class EventPayload(BaseModel):
    type: Literal["START","END","HEARTBEAT"]
    ts: float
    video_ts: Optional[float] = None
    event: Optional[str] = None
    prob: Optional[float] = None
    duration_sec: Optional[float] = None
    frame_idx: Optional[int] = None
    class Config: extra = "allow"

def fmt_hhmmss(sec: Optional[float]) -> str:
    if sec is None: return "-"
    s = max(0, int(round(sec)))
    h = s // 3600; m = (s % 3600) // 60; ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"

def build_text(payload: EventPayload) -> str:
    lines = [
        f"[이상행동탐지] 최종 이벤트 : {payload.event or '-'}",
        f"- 지속시간 : {fmt_hhmmss(payload.duration_sec)}"
        + (f" ({payload.duration_sec:.1f}s)" if payload.duration_sec is not None else ""),
    ]
    if payload.video_ts is not None:
        lines.append(f"- 영상 시간 : {payload.video_ts:.2f}s")
    lines.append(f"- 프레임 : {payload.frame_idx or '-'}")
    lines.append(time.strftime("- 수신시각 : %Y-%m-%d / %H:%M:%S", time.localtime()))
    return "\n".join(lines)

def send_kakao_memo(text: str) -> dict:
    import urllib.parse

    if not KAKAO_ACCESS_TOKEN:
        return {"ok": False, "err": "KAKAO_ACCESS_TOKEN not set (.env 필요)"}

    template_object = {
        "object_type": "text",
        "text": text,
        "link": {},
        "button_title": BTN_TITLE,
    }
    if ALERT_LINK_URL:
        template_object["link"] = {
            "web_url": ALERT_LINK_URL,
            "mobile_web_url": ALERT_LINK_URL,
        }

    headers = {
        "Authorization": f"Bearer {KAKAO_ACCESS_TOKEN}",
        # charset 지정이 없는 환경에서 깨지는 경우가 있어 명시
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }

    # ✅ URL-encode를 직접 수행 (중요)
    tmpl = json.dumps(template_object, ensure_ascii=False, separators=(",", ":"))
    body = "template_object=" + urllib.parse.quote_plus(tmpl)

    try:
        r = requests.post(KAKAO_MEMO_ENDPOINT, headers=headers, data=body, timeout=8)
        try:
            body_json = r.json()
        except Exception:
            body_json = {"text": r.text[:1000]}

        ok = (r.status_code == 200) and (isinstance(body_json, dict) and body_json.get("result_code") == 0)

        # 디버깅용 힌트 제공
        if not ok:
            return {
                "ok": False,
                "status": r.status_code,
                "hint": "If you see 'template_id can't be null', Kakao didn't parse template_object. "
                        "We now send urlencoded form explicitly.",
                "body": body_json,
            }
        return {"ok": True, "status": r.status_code, "body": body_json}
    except Exception as e:
        return {"ok": False, "err": str(e)}


# ===== 점검용 엔드포인트 =====

@app.get("/health")
def health():
    return {
        "status":"ok",
        "has_token": bool(KAKAO_ACCESS_TOKEN),
        "link": ALERT_LINK_URL or None,
        "button": BTN_TITLE,
    }

@app.get("/token/check")
def token_check():
    """카카오 액세스 토큰 유효성 확인 (/v2/user/me)"""
    if not KAKAO_ACCESS_TOKEN:
        return {"ok": False, "err": "KAKAO_ACCESS_TOKEN not set"}
    try:
        r = requests.get(
            KAKAO_ME_ENDPOINT,
            headers={"Authorization": f"Bearer {KAKAO_ACCESS_TOKEN}"},
            timeout=8
        )
        body = {}
        try: body = r.json()
        except Exception: body = {"text": r.text[:1000]}
        return {"ok": r.status_code == 200, "status": r.status_code, "body": body}
    except Exception as e:
        return {"ok": False, "err": str(e)}

@app.post("/token/send_test")
def token_send_test():
    """템플릿 없이 test 메시지 전송(template_object)"""
    text = "[EG-WAY] 테스트 메시지\n- 이 메시지가 보이면 token/talk_message 권한 OK"
    res = send_kakao_memo(text)
    return {"ok": res.get("ok", False), "kakao": res, "preview": text}

# ===== 실제 이벤트 수신 =====

@app.post("/events")
def events(ep: EventPayload):
    print("[EVENT]", ep.dict())
    if ep.type != "END" or not ep.event:
        return {"ok": True, "skipped": ep.type}
    text = build_text(ep)
    res = send_kakao_memo(text)
    return {"ok": res.get("ok", False), "kakao": res, "preview": text}

# ===== 단독 실행 지원 =====
if __name__ == "__main__":
    # 여기서 바로 uvicorn 실행 → python server_kakao_direct_full.py 만으로 기동됨
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    print(f"[BOOT] Loading .env ... KAKAO_ACCESS_TOKEN set? {bool(KAKAO_ACCESS_TOKEN)}")
    print(f"[BOOT] Starting server on http://{host}:{port}")
    uvicorn.run("kakao_test:app", host=host, port=port, reload=False, workers=1)
