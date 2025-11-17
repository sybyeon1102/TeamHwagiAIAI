# Backend 서브 프로젝트

이 디렉터리는 **이상/정상 행동 분석용 FastAPI 서버**를 담당하는
`backend` 서브 프로젝트이다.
모든 실행 예시는 **이 디렉터리(`backend/`) 기준**으로 작성되어 있다.

- HTTP API 기반 헬스 체크 (`/health`)
- 행동 분석 엔드포인트 (`/behavior`)
- 이벤트 로깅 엔드포인트 (`/event`)

현재 1차 라운드에서는 **서버 구조/레이아웃을 정리한 상태**이며,
`/behavior`의 추론 로직은 **더미 구현**으로 되어 있고
나중에 `modeling` 서브 프로젝트의 실제 LSTM 추론 코드로 교체될 예정이다.

---

## 0. 환경 구축 (uv)

이 서브 프로젝트는 `uv` 기반으로 Python 환경을 관리한다.
Python 버전은 프로젝트 전역과 동일하게 **3.12.x**를 사용한다.

### 0-1. 기본 환경 구축

`backend/pyproject.toml` 이 있는 디렉터리에서:

```bash
cd backend

# 의존성 설치 / Install dependencies
uv sync
```

`pyproject.toml`에는 다음과 같은 의존성이 포함되어 있다:

- `fastapi[standard]` : FastAPI + uvicorn 실행 환경
- `pydantic`, `pydantic-settings` : 설정/IO 모델
- `project-core` : 공용 Result 타입 및 유틸
- `modeling` : (추후) LSTM 추론 로직을 사용할 서브 프로젝트
  *(현재는 전처리/훈련만 구현되어 있으며, 추론 API는 아직 연결되지 않음)*

---

## 1. 개발 서버 실행

### 1-1. 현재 backend 서버 실행 (FastAPI CLI)

FastAPI 공식 CLI와 `uv`를 사용해 개발 서버를 실행한다.
우리 프로젝트에서는 **`python` / `uvicorn` 명령어 대신**
**`uv run fastapi`** 형식을 기본으로 사용한다.

```bash
cd backend

# FastAPI 개발 서버 실행 / Run FastAPI development server
uv run fastapi dev
```

- 기본적으로 `app/main.py`의 `app` 변수를 자동으로 찾는다.
- 기본 포트는 `8000`이며, 필요하면 `--port` 옵션으로 변경할 수 있다.

예:

```bash
uv run fastapi dev --host 0.0.0.0 --port 8000
```

서버가 뜬 후 브라우저에서 다음 주소로 접속할 수 있다:

- 기본 문서: `http://localhost:8000/docs`
- 헬스 체크: `http://localhost:8000/health/`

---

### 1-2. (참고) 기존 레거시 서버 명령어와의 대응

원래 루트 `README.md`에는 다음과 같은 명령어가 있었다:

```bash
uvicorn c_server:app --host 0.0.0.0 --port 8000 --reload
```

이 명령어는 레거시 `c_server.py` 기반 FastAPI 서버를 직접 띄우는 방식이었다.
현재 구조에서는 **이 역할을 `backend` 서브 프로젝트가 대체**하며,
위 명령어는 다음과 같이 대응된다고 볼 수 있다:

```bash
# 레거시:
# uvicorn c_server:app --host 0.0.0.0 --port 8000 --reload

# 현재:
cd backend
uv sync
uv run fastapi dev --host 0.0.0.0 --port 8000
```

마찬가지로 테스트용 앱이었던:

```bash
uvicorn stream_lstm_app:app --host 0.0.0.0 --port 8000 --reload
```

은 앞으로 `backend` + (추후) `frontend` 조합으로 대체될 예정이다.
지금 시점에서는 `backend`만 독립적으로 실행 가능한 상태이다.

---

## 2. 디렉터리 구조 (backend/app)

`backend/app` 패키지는 다음과 같이 구성된다:

```text
backend/
  pyproject.toml
  app/
    __init__.py
    main.py          # FastAPI 엔트리포인트 (create_app / app)
    config.py        # 설정/환경 로딩 (pydantic-settings)
    dependencies.py  # FastAPI Depends용 공용 의존성
    errors.py        # 도메인 에러 타입 + HTTP 매핑
    models/
      __init__.py
      model_io_behavior.py  # 행동 분석 요청/응답 IO 모델
      model_io_event.py     # 이벤트 로깅 요청/응답 IO 모델
    services/
      __init__.py
      service_behavior.py   # 행동 분석 도메인 서비스 (Result 기반)
      service_event.py      # 이벤트 로깅 도메인 서비스
    routers/
      __init__.py
      router_health.py      # /health 헬스 체크
      router_behavior.py    # /behavior 행동 분석 API
      router_event.py       # /event 이벤트 로깅 API
    internal/
      __init__.py
      internal_admin.py     # (옵션) 내부/관리자용 엔드포인트 자리
```

---

## 3. 주요 엔드포인트 개요

### 3-1. 헬스 체크: `/health/`

- 메서드: `GET /health/`
- 반환 예시:

```json
{
  "status": "ok",
  "app_version": "0.1.0",
  "environment": "local"
}
```

서버 상태 및 환경 정보를 단순히 확인하기 위한 엔드포인트이다.

---

### 3-2. 행동 분석: `/behavior/`

- 메서드: `POST /behavior/`
- 요청 바디: `BehaviorAnalyzeRequest`

```json
{
  "frames": [
    {
      "features": [0.1, 0.2, 0.3]
    },
    {
      "features": [0.0, 0.5, 0.9]
    }
  ]
}
```

- 응답 바디: `BehaviorAnalyzeResponse`

```json
{
  "is_anomaly": false,
  "normal_score": 0.9,
  "anomaly_score": 0.1
}
```

현재는 `service_behavior.py`에서 **더미 점수**를 반환하며,
`normal_score` / `anomaly_score` 값은 실제 LSTM 추론과 무관하다.
나중에 `modeling` 서브 프로젝트의 추론 모듈이 준비되면,
이 엔드포인트의 내부 구현만 실제 추론 호출로 교체할 예정이다.

---

### 3-3. 이벤트 로깅: `/event/`

- 메서드: `POST /event/`
- 요청 바디: `EventLogRequest`

```json
{
  "event": {
    "timestamp": "2025-11-17T12:34:56.789Z",
    "event_type": "HEARTBEAT",
    "session_id": "session-123",
    "user_id": "user-456",
    "data": {
      "note": "optional extra metadata"
    }
  }
}
```

- 응답 바디: `EventLogResponse`

```json
{
  "success": true,
  "event_id": "dummy-event-id"
}
```

현재는 실제 저장소(DB, 파일, 메시지 큐 등)에는 기록하지 않고,
`service_event.py`에서 **더미 event_id**와 `success=true`만 반환한다.
향후 구현에서 DB 또는 로그 스토리지와 연동할 예정이며,
그때에도 HTTP 스펙은 최대한 유지하는 것을 목표로 한다.

---

## 4. 전형적인 개발 워크플로 예시

1. **backend 환경 구축**

   ```bash
   cd backend
   uv sync
   ```

2. **개발 서버 실행**

   ```bash
   uv run fastapi dev --host 0.0.0.0 --port 8000
   ```

3. **헬스 체크 확인**

   - 브라우저 또는 HTTP 클라이언트에서:

     - `GET http://localhost:8000/health/`

4. **행동 분석 테스트**

   - `POST http://localhost:8000/behavior/`
     위 예시 JSON 바디를 전송하여 응답 구조를 확인한다.

5. **이벤트 로깅 테스트**

   - `POST http://localhost:8000/event/`
     `EventLogRequest` 예시 바디를 전송해 `success` / `event_id` 응답을 확인한다.

---

## 5. 앞으로의 계획 (요약)

- `modeling` 서브 프로젝트의 LSTM 추론 로직을
  `modeling.inference` 모듈 형태로 정리한 뒤,
- `service_behavior.py`의 더미 구현을
  해당 추론 모듈 호출로 교체한다.
- 레거시 `c_server.py` / `c_realtime_client.py`의 역할은
  - **HTTP/API 부분**: `backend` / `frontend`로,
  - **추론/모델 부분**: `modeling`으로
  천천히 분리·이전하는 것을 목표로 한다.
