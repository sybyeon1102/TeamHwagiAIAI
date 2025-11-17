# Backend 서브 프로젝트

이 디렉터리는 **이상/정상 행동 분석용 FastAPI 서버**를 담당하는
`backend` 서브 프로젝트이다.
모든 실행 예시는 **이 디렉터리(`backend/`) 기준**으로 작성되어 있다.

- HTTP API 기반 헬스 체크 (`/health`)
- 포즈 기반 행동 분석 엔드포인트 (`/behavior/analyze`)
- 이벤트 로깅 엔드포인트 (`/event`)

모델 추론은 `modeling` 서브 프로젝트의 **LSTM 기반 inference 레이어**
(`modeling.inference.inference_lstm`)를 사용하며,
추론 오류/모델 상태는 `project_core.result` 기반의 Result/에러 타입으로 표현된다.

---

## 0. 환경 구축 (uv + pyproject)

`backend` 서브 프로젝트는 **pyproject.toml + uv**를 사용해
Python 환경과 의존성을 관리한다.

### 0-1. Python 버전 및 의존성

- Python: `>= 3.12`
- 주요 의존성:
  - `fastapi[standard]`
  - `pydantic-settings`
  - **로컬 서브 프로젝트**:
    - `project-core` (core/python)
    - `modeling`

### 0-2. uv 환경 동기화

```bash
cd backend

# 의존성 설치 / Sync dependencies
uv sync
```

---

## 1. 설정(.env) 및 Settings

backend 설정은 `app.config.Settings` 클래스로 관리되며,
`pydantic-settings` 를 사용해 **환경 변수 및 `.env`** 에서 값을 읽어온다.

```python
# backend/app/config.py (발췌)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BACKEND_",
        extra="ignore",
    )

    app_title: str = APP_TITLE_DEFAULT
    app_version: str = APP_VERSION_DEFAULT
    app_description: str = APP_DESCRIPTION_DEFAULT

    model_checkpoint_path: Path = Field(
        ...,
        description=(
            "LSTM 모델 체크포인트(.pt) 파일 경로.\n"
            "Path to the LSTM model checkpoint (.pt) file."
        ),
    )

    environment: str = Field(default="local", ...)
    debug: bool = Field(default=False, ...)
```

### 1-1. `.env` 위치

- backend용 `.env` 파일은 **`backend/.env`** 에 둔다.
- `env_prefix="BACKEND_"` 로 설정되어 있으므로,
  예를 들면 다음과 같이 작성한다:

```env
# backend/.env

BACKEND_MODEL_CHECKPOINT_PATH=/abs/path/to/lstm_checkpoint.pt
BACKEND_APP_TITLE=Behavior Inference API
BACKEND_APP_VERSION=0.1.0
BACKEND_ENVIRONMENT=local
BACKEND_DEBUG=true
```

---

## 2. 디렉터리 구조

요약된 구조는 다음과 같다:

```text
backend/
  pyproject.toml
  README.md
  .env                # (선택) Settings용 환경 변수 파일

  app/
    main.py           # FastAPI 앱 생성, 라우터 등록
    config.py         # Settings (pydantic-settings)
    dependencies.py   # FastAPI 의존성 주입 helper
    errors.py         # InferenceError, InferenceResult, HTTP 매핑
    models/
      model_io_behavior.py  # /behavior 입출력 모델
      model_io_event.py     # /event 입출력 모델
    routers/
      router_health.py      # /health
      router_behavior.py    # /behavior/analyze
      router_event.py       # /event
    services/
      service_behavior.py   # 모델 추론 호출 (modeling.inference)
      service_event.py      # 이벤트 로깅 (현재는 더미 구현)
```

---

## 3. 서버 실행 방법

### 3-1. 개발용 실행 (uv + fastapi)

현재는 별도 엔트리 스크립트를 두기보다는,
`uv run fastapi ...` 형식으로 개발용 서버를 실행한다.

```bash
cd backend

# 기본 개발 실행 예시
uv run fastapi app.main:app --host 0.0.0.0 --port 8000 --reload
```

나중에는 `pyproject.toml` 의 `[project.scripts]` 에
예를 들어 `run-backend-dev = "app.main:app"` 형태로
스크립트 엔트리를 등록해 둘 수 있다.
([INV-061] “uv run + 엔트리” 철학을 따르기 위함)

### 3-2. 기본 헬스 체크

서버가 뜬 뒤, 다음과 같이 헬스 체크를 할 수 있다:

```bash
# 기본 헬스 체크
curl http://localhost:8000/health/
```

응답 예시:

```json
{
  "status": "ok",
  "app_version": "0.1.0",
  "environment": "local"
}
```

---

## 4. API 개요

### 4-1. 헬스 체크 `/health/`

- **메서드**: `GET /health/`
- **목적**:
  - 서버가 살아 있는지 확인.
  - 현재 앱 버전, environment 확인.

응답 예:

```json
{
  "status": "ok",
  "app_version": "0.1.0",
  "environment": "local"
}
```

### 4-2. 행동 분석 `/behavior/analyze`

포즈 기반 프레임 시퀀스를 입력으로 받아,
LSTM 기반 이상/정상 판단을 수행한다.

- **메서드**: `POST /behavior/analyze`
- **Request Body**: `BehaviorAnalyzeRequest`
- **Response Body**: `BehaviorAnalyzeResponse`

#### Request 모델

```jsonc
{
  "frames": [
    {
      "index": 0,
      "features": [0.12, 1.34, -0.56, 0.01]
    },
    {
      "index": 1,
      "features": [0.10, 1.30, -0.53, 0.02]
    }
  ]
}
```

- `frames`: 시간순으로 정렬된 포즈 프레임 배열
- 각 `features`: 전처리/학습 단계에서 사용한 feature 벡터와 동일한 차원(F)을 가져야 한다.
- `index` 필드는 선택(optional)이며, 단순 디버깅용이다.

#### Response 모델

```jsonc
{
  "is_anomaly": true,
  "normal_score": 0.23,
  "anomaly_score": 0.77,
  "events": [
    "event_walk",
    "event_run",
    "event_jump"
  ],
  "scores": [
    0.12,
    0.77,
    0.05
  ],
  "thresholds": [
    0.5,
    0.6,
    0.4
  ]
}
```

- `is_anomaly`
  - 하나라도 임계값을 넘는 이벤트가 있으면 `true`.
- `normal_score`
  - `anomaly_score` (이벤트별 점수 중 최대값)을 기준으로 계산된 정상 점수 (대략 0~1).
- `anomaly_score`
  - 모든 이벤트 중 최대 sigmoid 점수 (0~1).
- `events`
  - 클래스/이벤트 이름 리스트. `scores`, `thresholds`와 인덱스로 정렬.
- `scores`
  - 각 이벤트별 sigmoid 점수 리스트.
- `thresholds`
  - 각 이벤트별 의사결정 임계값 리스트.

에러가 발생하면, 내부 도메인 에러(`InferenceError`)가
`app.errors.map_inference_error_to_http_exception()` 을 통해
적절한 HTTP 상태 코드 및 메시지로 변환된다.

### 4-3. 이벤트 로깅 `/event/`

사용자의 행동/세션 등에 대한 이벤트를 기록하기 위한 엔드포인트이다.

- **메서드**: `POST /event/`
- **Request Body**: `EventLogRequest`
- **Response Body**: `EventLogResponse`

#### Request 모델

```jsonc
{
  "event": {
    "timestamp": "2025-11-17T12:34:56Z",
    "event_type": "behavior_analyzed",
    "session_id": "session-123",
    "user_id": "user-42",
    "payload": {
      "some": "json",
      "extra": 123
    }
  }
}
```

- `event_type`: 이벤트 종류(자유 형식 문자열)
- `timestamp`: ISO 8601 형식의 시각
- `session_id`, `user_id`: 선택(optional)
- `payload`: 추가 메타데이터를 담는 임의의 JSON 객체

#### Response 모델

```json
{
  "success": true,
  "event_id": "dummy-event-id"
}
```

현재 구현은 **데모용 더미 구현**으로,
`event_id` 를 고정 값 `"dummy-event-id"` 로 반환하며
실제 저장소(DB, 메시지 큐 등)에 기록하지는 않는다.
(향후 `service_event.py` 에서 실제 저장소로 확장 가능)

---

## 5. modeling.inference 연동 구조

### 5-1. LSTM inference 컨텍스트 로딩

`service_behavior._get_lstm_context()` 는
`Settings.model_checkpoint_path`를 기반으로 LSTM 모델을 로드한다:

- checkpoint에서:
  - `state_dict` (LSTM 모델 파라미터)
  - `meta` (`events`, `win`, `norm_mean`, `norm_std` 등 전처리 메타데이터)
  - `thresholds` (클래스별 임계값; 없으면 0.5로 채움)
- 를 읽어서 `LstmInferenceContext` 를 생성한다.

이 컨텍스트는 `functools.lru_cache` 로 캐시되어,
서버가 떠 있는 동안 같은 checkpoint를 재사용한다.

### 5-2. inference 호출 플로우

1. `/behavior/analyze` → `router_behavior.analyze_behavior_endpoint()`
2. `service_behavior.analyze_behavior(request)` 호출
3. 내부 플로우:
   - `_get_lstm_context()` 로 모델/메타 로드
   - `BehaviorAnalyzeRequest.frames` → `list[list[float]]` 로 변환
   - `run_inference_lstm_single(context, frames)` 호출
   - 결과(`LstmInferenceOutput`)를 `BehaviorAnalyzeResponse`로 매핑
   - `Ok(response)` / `Err(InferenceError)` 형태로 반환
4. 라우터에서 `Ok`/`Err`를 패턴 매칭(`match`)하여
   - 성공 → HTTP 200 + `BehaviorAnalyzeResponse`
   - 실패 → `InferenceError` → `HTTPException` 변환 후 에러 응답

### 5-3. modeling README와의 대응

`modeling/README.md` 의 **6. LSTM 추론(inference) 단독 실행** 섹션에서
설명하는 CLI (`run-inference-lstm`)와
backend의 `/behavior/analyze` 엔드포인트는
**동일한 inference 레이어**를 사용한다.

- CLI:
  - `uv run run-inference-lstm --checkpoint ... --input-json ...`
- Backend:
  - `POST /behavior/analyze` + `BehaviorAnalyzeRequest` JSON

입력/출력 포맷은 서로 호환되도록 설계되어 있어,
동일한 시퀀스 JSON을 CLI와 HTTP에 모두 넣고 결과를 비교할 수 있다.

---

## 6. 앞으로의 작업 아이디어

backend 1차 라운드(레이아웃 정리 + LSTM 연동)는 완료된 상태이며,
향후에는 다음과 같은 작업을 고려할 수 있다:

- `/health` 에 **모델 로드 상태** 추가
  - `_get_lstm_context()` 를 호출해,
    모델이 정상 로드되지 않은 경우 `status: "degraded"` 등으로 표현.
- `service_event.py` 를 실제 저장소(DB/메시지 큐/파일 등)에 연결
- 공통 에러/응답 포맷 정리
  - 예: `{ "code": "...", "message": "..." }` 통일
- `[project.scripts]` 에 backend 서버 실행 엔트리 추가
  - 예: `run-backend-dev = "app.main:app"`
    → `uv run run-backend-dev` 형식으로 실행.
- pytest 기반의 간단한 엔드투엔드 테스트
  - `/health` / `/behavior/analyze` 에 대한 최소 검증.
