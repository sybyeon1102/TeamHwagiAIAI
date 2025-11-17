# Modeling 서브 프로젝트

이 디렉터리는 **정상/이상 행동 데이터 전처리 + LSTM 모델 학습**을 담당하는
`modeling` 서브 프로젝트이다.
모든 실행 예시는 **이 디렉터리(`modeling/`) 기준**으로 작성되어 있다.

---

## 0. 환경 구축 (uv + optional extras)

이 서브 프로젝트는 `uv` 기반으로 Python 환경을 관리한다.
PyTorch는 **optional extra**를 통해 CPU / CUDA 버전을 선택해서 설치한다.

### 0-1. 지원하는 optional 목록

`uv sync --extra <optional>` 에서 `<optional>`로 사용할 수 있는 값은 다음과 같다:

- `cpu`
- `cu130`
- `cu129`
- `cu128`
- `cu126`
- `cu124`
- `cu121`

> 실제 extra 이름은 pyproject 설정에 따라 약간 다를 수 있다
> (예: cuda-12.1 등).
> 아래 내용은 **개념적으로 “CUDA 12.1 / 12.4 / … / 13.0용 extra”**를 의미한다.

### 0-2. CUDA 버전 선택 규칙

사용자 시스템에 설치된 CUDA 런타임/드라이버 버전이 **V**라고 할 때:

- 반드시 “지원 버전 중, V 이하의 버전”을 골라야 한다.
- 예를 들어,
  - 시스템 CUDA가 12.6이라면, 아래 중에서 선택 가능:
    - `cu126`
    - `cu124`
    - `cu121`
  - 시스템 CUDA가 13.0이라면, 아래 중에서 선택 가능:
    - `cu130`
    - `cu129`
    - `cu128`
    - `cu126`
    - `cu124`
    - `cu121`

CUDA 버전이 헷갈리면, 일반적으로 다음처럼 확인할 수 있다 (리눅스/윈도우):

    nvidia-smi

CUDA 버전을 모르면 **일단 `cpu` extra로 시작**하는 것을 권장한다.

### 0-3. 플랫폼별 지원 매트릭스

> “지원한다”는 말은 **해당 조합으로 `uv sync --extra ...`가 성공한다**는 뜻이다.

- **Linux**
  - 지원:
    - `cpu`
    - `cu121`, `cu124`, `cu126`, `cu128`, `cu129`, `cu130`
  - 즉, **CPU + 모든 CUDA optional**을 사용할 수 있다
    (단, 시스템 CUDA 버전이 해당 optional 버전 이상이어야 실제로 동작).

- **Windows**
  - 지원:
    - `cpu`
    - `cu121`, `cu124`, `cu126`, `cu128`, `cu130`
  - 미지원(가정):
    - `cu129` (윈도우용 wheel이 없다고 가정)

  따라서 윈도우에서는:

  - CPU만 쓰려면 → `uv sync --extra cpu`
  - CUDA를 쓰려면 → `cu121 / 124 / 126 / 128 / 130` 중에서
    본인 시스템 CUDA 버전 이하를 선택.

- **macOS**
  - 지원:
    - `cpu` 만 지원
  - 모든 `cu...` optional은 미지원
    → 시도 시 wheel이 없어서 설치가 실패하는 것이 정상이다.

### 0-4. 기본적인 환경 구축 예시

1. **CPU 전용 (모든 플랫폼 공통, 가장 안전한 선택)**

       uv sync --extra cpu

   - GPU 설정을 신경 쓰기 싫거나, CUDA 버전을 모르는 경우
     → 일단 이 조합으로 시작하는 것을 추천.

2. **Linux + CUDA 12.6 환경 (예시)**

   시스템 CUDA가 12.6인 경우:

   - 가장 잘 맞는 extra:

         uv sync --extra cu126

   - 혹은 좀 더 하위 버전 wheel을 쓰고 싶다면:

         uv sync --extra cu124
         uv sync --extra cu121

3. **Windows + CUDA 13.0 환경 (예시)**

   시스템 CUDA가 13.0이고, 윈도우용 13.0 wheel이 있다고 가정하면:

       uv sync --extra cu130

   - 하위 버전을 쓰려면:

         uv sync --extra cu128
         uv sync --extra cu126
         uv sync --extra cu124
         uv sync --extra cu121

4. **macOS 환경**

   - CPU 전용만 지원:

         uv sync --extra cpu

   - `cu121` 등 CUDA optional을 선택하면
     → 해당 플랫폼용 wheel이 없어서 설치 실패가 나는 것이 정상이다.

### 0-5. 지원하지 않는 조합을 선택했을 때

- 예: macOS에서 `uv sync --extra cu126`
- 예: Windows에서 `uv sync --extra cu129` (지원하지 않는다고 가정)

이런 경우 `uv`는:

- 해당 extra에 맞는 PyTorch wheel을 찾지 못하고,
- “current platform에 대한 wheel이 없다”는 식의 에러를 내며 **설치에 실패**한다.

이것은 의도된 동작이며,
**“지원하지 않는 조합은 조용히 CPU로 떨어지지 않고, 아예 실패로 드러난다”**고 이해하면 된다.

---

## 1. Modeling CLI 개요

`pyproject.toml`의 `[project.scripts]`에 다음 엔트리가 등록되어 있다:

    [project.scripts]
    build-dataset-normal  = "modeling.preprocessing.dataset_normal:main"
    build-dataset-anomaly = "modeling.preprocessing.dataset_anomaly:main"
    train-lstm-model      = "modeling.training.trainer_lstm:main"

따라서 uv 환경에서 다음과 같이 실행한다:

- 정상 행동 데이터셋: `uv run build-dataset-normal ...`
- 이상 행동 데이터셋: `uv run build-dataset-anomaly ...`
- LSTM 학습: `uv run train-lstm-model ...`

---

## 2. 정상(구매/일상) 행동 데이터셋 생성

`build-dataset-normal` (→ `modeling.preprocessing.dataset_normal:main`)

CVAT XML + 영상(.mp4)을 이용해 **정상 행동(normal / purchase / regular)용 LSTM 데이터셋**
`X.npy`, `Y.npy`, `meta.json`을 생성한다.

### 2-1. 기본 사용법

    uv run build-dataset-normal \
      --video-root /path/to/normal/videos \
      --xml-root   /path/to/normal/xmls \
      --out-dir    /path/to/out/normal_lstm

### 2-2. 전체 옵션

    uv run build-dataset-normal --help

옵션 의미와 기본값은 이상 행동용과 동일하다:

- `--video-root` : 입력 영상 루트 디렉터리
  root directory containing input videos
- `--xml-root` : CVAT XML 루트 디렉터리
  root directory containing CVAT XML annotations
- `--out-dir` : 출력 디렉터리 (X/Y/meta 저장)
  output directory for X.npy / Y.npy / meta.json
- `--window-size` / `--stride` / `--overlap`
- `--resize-w`
- `--model-complexity`

차이점은 내부에서 사용하는 **이벤트 라벨 셋**과
`moving` 이벤트에 대한 구간 처리 로직(정상 행동 전용 규칙)이 다르다는 점이다.

---

## 3. 이상 행동 데이터셋 생성

`build-dataset-anomaly` (→ `modeling.preprocessing.dataset_anomaly:main`)

CVAT XML + 영상(.mp4)을 이용해 **이상 행동(anomaly)용 LSTM 데이터셋**
`X.npy`, `Y.npy`, `meta.json`을 생성한다.

### 3-1. 기본 사용법

    uv run build-dataset-anomaly \
      --video-root /path/to/anomaly/videos \
      --xml-root   /path/to/anomaly/xmls \
      --out-dir    /path/to/out/anomaly_lstm

### 3-2. 전체 옵션

    uv run build-dataset-anomaly --help

- `--video-root PATH`        영상(.mp4)들이 들어 있는 루트 디렉터리
                             root directory containing input videos
- `--xml-root PATH`          CVAT XML 어노테이션 파일들이 들어 있는 루트 디렉터리
                             root directory containing CVAT XML annotations
- `--out-dir PATH`           생성된 X.npy / Y.npy / meta.json을 저장할 디렉터리
                             output directory for X.npy / Y.npy / meta.json
- `--window-size INT`        슬라이딩 윈도우 길이 (프레임 수, 기본값: 16)
                             sliding window length in frames (default: 16)
- `--stride INT`             슬라이딩 윈도우 이동 간격 (프레임 수, 기본값: 4)
                             sliding window stride in frames (default: 4)
- `--overlap FLOAT`          윈도우와 이벤트 구간의 최소 겹침 비율 (0.0~1.0, 기본값: 0.25)
                             minimum overlap ratio between window and event interval
                             (default: 0.25)
- `--resize-w INT`           영상 가로 리사이즈 크기 (0이면 리사이즈하지 않음, 기본값: 640)
                             target video width; 0 means no resize (default: 640)
- `--model-complexity INT`   MediaPipe Pose model_complexity 값 (0, 1, 2 중 하나, 기본값: 0)
                             MediaPipe Pose model_complexity: 0, 1 or 2 (default: 0)

생성 결과:

- `X.npy` : `(N, T, F)` 형태의 윈도우 피처
- `Y.npy` : `(N, C)` 멀티라벨(one-hot) 타깃
- `meta.json` : 이벤트 라벨, 정규화 정보 등 메타데이터

---

## 4. LSTM 모델 학습

`train-lstm-model` (→ `modeling.training.trainer_lstm:main`)

전처리 결과(`X.npy`, `Y.npy`, `meta.json`)가 있는 디렉터리를 입력으로 받아
LSTM 기반 멀티라벨 분류 모델을 학습하고,
최적 검증 손실 및 클래스별 threshold를 포함한 체크포인트를 저장한다.

### 4-1. 기본 사용법 (CPU 예시)

    uv run train-lstm-model \
      --data_dir /path/to/ds_lstm_all \
      --epochs 40 \
      --batch 64 \
      --lr 0.002 \
      --sampler_pos_boost 4.0 \
      --val_ratio 0.2 \
      --device auto \
      --save /path/to/out/lstm_multilabel.pt

### 4-2. 전체 옵션

    uv run train-lstm-model --help

- `--data_dir PATH`          전처리 결과(X.npy, Y.npy, meta.json)가 저장된 디렉터리 (필수)
                             directory containing X.npy, Y.npy and meta.json
- `--epochs INT`             학습 epoch 수 (기본값: 40)
                             number of training epochs (default: 40)
- `--batch INT`              배치 크기 (기본값: 64)
                             batch size (default: 64)
- `--lr FLOAT`               학습률 learning rate (기본값: 2e-3)
                             learning rate (default: 2e-3)
- `--sampler_pos_boost FLOAT`
                             positive 샘플 가중치 부스트 배율 (기본값: 4.0)
                             multiplier to up-weight positive samples (default: 4.0)
- `--val_ratio FLOAT`        검증 세트 비율 (0.0~1.0, 기본값: 0.2)
                             validation split ratio (default: 0.2)
- `--num_workers INT`        DataLoader num_workers (기본값: 2; 0이면 메인 프로세스에서 로딩)
                             number of worker processes for DataLoader (default: 2)
- `--device {auto,cpu,cuda}`
                             학습에 사용할 디바이스 선택 (기본값: auto)
                             auto: cuda 가능 시 cuda, 아니면 cpu
- `--seed INT`               난수 시드 (기본값: 42)
                             random seed (default: 42)
- `--save PATH`              학습된 모델 체크포인트 저장 경로
                             (기본값: ./lstm_multilabel.pt)
                             path to save the trained model checkpoint

출력(저장) 파일:

- `lstm_multilabel.pt` :
  - `state_dict` : LSTM 모델 가중치
  - `meta` : 전처리 단계에서 만든 `meta.json` 내용
  - `thresholds` : 클래스별 decision threshold 리스트

---

## 5. 전형적인 워크플로 예시

1. **환경 구축 (예: CPU 전용)**

       uv sync --extra cpu

2. **정상/이상 데이터셋 각각 생성**

       # 정상 행동
       uv run build-dataset-normal \
         --video-root /data/normal/videos \
         --xml-root   /data/normal/xmls \
         --out-dir    /data/ds_normal_lstm

       # 이상 행동
       uv run build-dataset-anomaly \
         --video-root /data/anomaly/videos \
         --xml-root   /data/anomaly/xmls \
         --out-dir    /data/ds_anomaly_lstm

3. **필요하다면 정상/이상 X/Y를 합쳐서 `ds_lstm_all` 디렉터리 구성**
   (간단한 스크립트나 노트북에서 `np.concatenate`로 합치는 식)

4. **합쳐진 데이터셋으로 LSTM 학습**

       uv run train-lstm-model \
         --data_dir /data/ds_lstm_all \
         --epochs 40 \
         --batch 64 \
         --lr 0.002 \
         --device auto \
         --save /models/lstm_multilabel.pt

## 6. LSTM 추론(inference) 단독 실행

이 섹션은 학습이 끝난 LSTM 체크포인트를 사용해, backend 없이도 단일 시퀀스에 대해 추론을 수행하는 방법을 정리한다.

### 6-1. 엔트리 이름과 기본 사용법

`modeling` 서브 프로젝트의 `pyproject.toml` 에는 다음과 같은 엔트리가 정의되어 있다:

    [project.scripts]
    run-inference-lstm = "modeling.inference.inference_lstm:main"

따라서 `modeling/` 디렉터리 기준으로 LSTM 추론을 실행할 수 있다:

    # modeling/ 디렉터리에서 실행
    # Run from the modeling/ directory
    uv run run-inference-lstm \
      --checkpoint /models/lstm_multilabel.pt \
      --input-json /data/sample_frames.json \
      --device auto \
      --pretty

옵션 설명:

- `--checkpoint, -c`
  - 학습이 완료된 LSTM 체크포인트(.pt) 파일 경로
- `--input-json, -i`
  - 하나의 시퀀스(윈도우)에 대한 프레임들이 들어 있는 JSON 파일 경로
- `--device, -d`
  - `auto` / `cpu` / `cuda` 중 하나를 선택 (`auto` 기본)
- `--pretty`
  - 결과 JSON을 사람이 읽기 좋은 형태로 들여쓰기해서 출력

### 6-2. 입력 JSON 형식

입력 JSON은 하나의 시퀀스(윈도우)에 대한 프레임 리스트를 표현한다.

두 가지 최상위 형식을 지원한다:

1. 최상위가 바로 프레임 리스트인 경우

       [
         [0.12, 1.34, -0.56, 0.01],
         [0.10, 1.30, -0.53, 0.02],
         [0.08, 1.27, -0.50, 0.03]
       ]

2. "frames" 키 아래에 프레임 리스트가 있는 경우

       {
         "frames": [
           [0.12, 1.34, -0.56, 0.01],
           [0.10, 1.30, -0.53, 0.02],
           [0.08, 1.27, -0.50, 0.03]
         ]
       }

각 프레임은 `features` 벡터에 대응되며,
전처리/학습 단계에서 사용한 feature 차원(F)과 동일한 길이를 가져야 한다.

이 형식은 backend의 `BehaviorAnalyzeRequest` 와도 대응되며,
backend 에서는 다음과 같은 구조를 사용한다:

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

여기서 `index` 필드는 선택(optional)이며,
이 README에서 사용하는 CLI 입력은 `features` 부분만을 직접 리스트로 전달하는 축약형이다.

### 6-3. 출력 JSON 형식

성공적으로 추론이 완료되면, 다음과 같은 JSON이 표준 출력(stdout)에 출력된다.

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

필드 의미:

- `is_anomaly`
  - 하나라도 임계값을 넘는 이벤트가 있으면 `true`
- `normal_score`
  - 이상 점수의 최대값을 기준으로 계산된 “정상일 가능성” 점수 (대략 0~1 범위)
- `anomaly_score`
  - 전체 이벤트 중 최대 sigmoid 점수 (0~1)
- `events`
  - 클래스(이벤트) 이름 리스트. `scores` / `thresholds`와 인덱스로 정렬.
- `scores`
  - 각 이벤트에 대한 sigmoid 점수 리스트.
- `thresholds`
  - 각 이벤트에 대한 의사결정 임계값 리스트.

backend의 `/behavior/analyze` 엔드포인트는 내부적으로 이와 동일한 LSTM 추론 레이어를 사용하며,
HTTP 응답에서는 `BehaviorAnalyzeResponse` 모델을 통해 동일한 정보(`is_anomaly`, `normal_score`, `anomaly_score`, `events`, `scores`, `thresholds`)를 전달할 수 있다.
