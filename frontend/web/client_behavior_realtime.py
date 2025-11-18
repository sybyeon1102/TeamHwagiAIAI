"""
실시간 웹캠/RTSP 영상을 읽어 포즈를 추출하고,
백엔드 /behavior/analyze API를 호출해 정상/이상 행동을 분석하는 클라이언트입니다.

This script reads webcam/RTSP video, extracts pose keypoints with MediaPipe,
builds LSTM input features, and sends them to the backend /behavior/analyze API.
Optionally, it also sends PROBS-style event logs to /event/payload.
"""

import argparse
import os
import time
from collections import deque
from typing import Deque

import cv2
import mediapipe as mp
import numpy as np
import requests


# ----------------------------
# 전처리 유틸 (modeling.preprocessing.dataset_common 로직을 독립 복제)
# Preprocessing utilities (logic copied from modeling.preprocessing.dataset_common)
# ----------------------------


def fill_missing_values(features: np.ndarray) -> np.ndarray:
    """NaN을 앞/뒤 방향으로 채운 뒤, 남은 NaN/inf는 0으로 채웁니다.
    Forward/backward fill NaNs and replace remaining NaN/inf with zeros.
    """
    time_steps, dim = features.shape
    output = features.copy()

    last_values = np.zeros(dim, np.float32)
    has_value = np.zeros(dim, bool)

    # forward fill
    for t in range(time_steps):
        not_nan = ~np.isnan(output[t])
        last_values[not_nan] = output[t, not_nan]
        has_value |= not_nan
        missing = np.isnan(output[t]) & has_value
        output[t, missing] = last_values[missing]

    # backward fill
    last_values[:] = 0.0
    has_value[:] = False

    for t in range(time_steps - 1, -1, -1):
        not_nan = ~np.isnan(output[t])
        last_values[not_nan] = output[t, not_nan]
        has_value |= not_nan
        missing = np.isnan(output[t]) & has_value
        output[t, missing] = last_values[missing]

    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def build_features_from_keypoints(sequence_keypoints: np.ndarray) -> np.ndarray:
    """(T, 33, 4) 형태의 키포인트 배열에서 LSTM 입력용 피처를 구성합니다.
    Build LSTM input features from pose keypoints (T, 33, 4).

    - T: 시간 축 길이 (프레임 수) / number of frames
    - 33: MediaPipe Pose keypoints
    - 4: (x, y, z, visibility)
    """
    if sequence_keypoints.size == 0:
        return np.zeros((0, 169), np.float32)

    time_steps = sequence_keypoints.shape[0]

    xy = sequence_keypoints[:, :, :2].reshape(time_steps, -1)
    visibility = sequence_keypoints[:, :, 3:4].reshape(time_steps, -1)

    xy = fill_missing_values(xy).reshape(time_steps, 33, 2)
    visibility = fill_missing_values(visibility).reshape(time_steps, 33, 1)

    hip_center = np.mean(xy[:, [23, 24], :], axis=1)
    shoulder_center = np.mean(xy[:, [11, 12], :], axis=1)
    scale = np.linalg.norm(shoulder_center - hip_center, axis=1, keepdims=True)
    scale[scale < 1e-3] = 1.0

    xy_normalized = (xy - hip_center[:, None, :]) / scale[:, None, :]
    velocity = np.diff(xy_normalized, axis=0, prepend=xy_normalized[:1])

    def compute_angle(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> np.ndarray:
        vector_1 = point_a - point_b
        vector_2 = point_c - point_b
        norm_1 = np.linalg.norm(vector_1, axis=-1)
        norm_2 = np.linalg.norm(vector_2, axis=-1)
        norm_1[norm_1 == 0] = 1e-6
        norm_2[norm_2 == 0] = 1e-6
        cosine = (vector_1 * vector_2).sum(-1) / (norm_1 * norm_2)
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    def select_joint(index: int) -> np.ndarray:
        return xy_normalized[:, index, :]

    joint_angles = np.stack(
        [
            compute_angle(select_joint(11), select_joint(13), select_joint(15)),
            compute_angle(select_joint(12), select_joint(14), select_joint(16)),
            compute_angle(select_joint(23), select_joint(25), select_joint(27)),
            compute_angle(select_joint(24), select_joint(26), select_joint(28)),
        ],
        axis=1,
    )

    features = np.concatenate(
        [
            xy_normalized.reshape(time_steps, -1),
            velocity.reshape(time_steps, -1),
            joint_angles,
            visibility.reshape(time_steps, -1),
        ],
        axis=1,
    )

    return np.clip(features.astype(np.float32), -10.0, 10.0)


# ----------------------------
# Pose 추출 유틸 / Pose extraction helpers
# ----------------------------


def extract_keypoints_from_frame(
    frame_bgr: np.ndarray,
    pose_estimator: mp.solutions.pose.Pose,
) -> np.ndarray:
    """단일 프레임에서 MediaPipe Pose 키포인트를 추출합니다.
    Extract 33 pose keypoints (x, y, z, visibility) from a single frame.

    반환 shape: (33, 4) – 포즈가 없으면 NaN으로 채움.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose_estimator.process(frame_rgb)

    if result and result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        points = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
    else:
        points = [[np.nan] * 4] * 33

    return np.asarray(points, dtype=np.float32)


# ----------------------------
# 백엔드 호출 / Backend calls
# ----------------------------


def call_backend_behavior_analyze(
    backend_url: str,
    features: np.ndarray,
) -> dict:
    """backend /behavior/analyze 엔드포인트를 호출해 결과를 반환합니다.
    Call backend /behavior/analyze endpoint with feature sequence.

    요청 형식은 BehaviorAnalyzeRequest 와 동일해야 합니다.
    The request body must follow the BehaviorAnalyzeRequest schema.
    """
    frames_payload = []
    for frame_index, row in enumerate(features):
        frames_payload.append(
            {
                "index": frame_index,
                "features": [float(value) for value in row],
            },
        )

    request_body = {"frames": frames_payload}

    url = backend_url.rstrip("/") + "/behavior/analyze"
    response = requests.post(url, json=request_body, timeout=2.0)
    response.raise_for_status()
    return response.json()


def send_event_probs_payload(
    backend_url: str,
    frame_index: int,
    events: list[str],
    scores: list[float],
    top_event: str | None,
    top_probability: float | None,
) -> None:
    """PROBS 타입 EventPayload를 /event/payload 에 전송합니다.
    Send a PROBS-type EventPayload to /event/payload.
    """
    if not events or not scores:
        return

    probability_map = {
        event_name: float(scores[i])
        for i, event_name in enumerate(events)
        if i < len(scores)
    }

    payload = {
        "type": "PROBS",
        "ts": time.time(),
        "frame_idx": frame_index,
        "top_event": top_event,
        "top_prob": float(top_probability) if top_probability is not None else None,
        "probs": probability_map,
    }

    url = backend_url.rstrip("/") + "/event/payload"
    try:
        requests.post(url, json=payload, timeout=1.0)
    except requests.RequestException as exc:  # noqa: PERF203
        print(f"[WARN] Failed to send PROBS event: {exc}")


# ----------------------------
# 메인 루프 / Main loop
# ----------------------------


def parse_arguments_realtime_client() -> argparse.Namespace:
    """실시간 클라이언트용 명령행 인자를 파싱합니다.
    Parse command-line arguments for the realtime client.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Realtime pose-based anomaly client using backend /behavior/analyze.\n"
            "웹캠/RTSP에서 포즈를 추출하고 백엔드 API로 이상행동을 분석합니다."
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (camera index or RTSP/file URL). 기본값: '0' (웹캠)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
        help="Backend base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="LSTM 윈도우 크기 (프레임 수) / Window size in frames (default: 60)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="분석 호출 주기 (프레임 단위) / Analysis stride in frames (default: 5)",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="입력 프레임을 리사이즈할 가로 폭(px). 0이면 원본 유지.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose model_complexity (0/1/2). 기본: 1",
    )
    parser.add_argument(
        "--no-event-log",
        action="store_true",
        help="/event/payload 로 PROBS 이벤트를 보내지 않습니다.",
    )

    return parser.parse_args()


def open_video_capture(source: str) -> cv2.VideoCapture:
    """source 문자열을 바탕으로 VideoCapture를 연다.
    Open cv2.VideoCapture from a source string.
    """
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    return capture


def run_client_realtime() -> None:
    """실시간 클라이언트 메인 루프.
    Main loop for realtime client.
    """
    arguments = parse_arguments_realtime_client()

    backend_url = arguments.backend_url
    window_size = max(1, arguments.window_size)
    stride = max(1, arguments.stride)

    print(f"[INFO] Using backend: {backend_url}")
    print(f"[INFO] window_size={window_size}, stride={stride}")

    capture = open_video_capture(arguments.source)

    pose_estimator = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=arguments.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoints_buffer: Deque[np.ndarray] = deque(maxlen=window_size)

    is_current_anomaly = False
    current_top_event: str | None = None
    current_top_probability: float | None = None

    frame_index = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[INFO] end of stream or camera error.")
                break

            frame_index += 1

            if arguments.resize_width:
                scale = arguments.resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (arguments.resize_width, int(frame.shape[0] * scale)),
                )

            keypoints = extract_keypoints_from_frame(frame, pose_estimator)
            keypoints_buffer.append(keypoints)

            if len(keypoints_buffer) == window_size and (frame_index % stride == 0):
                sequence_keypoints = np.stack(keypoints_buffer, axis=0)
                features = build_features_from_keypoints(sequence_keypoints)

                try:
                    result = call_backend_behavior_analyze(backend_url, features)
                except requests.RequestException as exc:  # noqa: PERF203
                    print(f"[WARN] backend analyze failed: {exc}")
                else:
                    events = result.get("events") or []
                    scores = result.get("scores") or []
                    is_current_anomaly = bool(result.get("is_anomaly", False))

                    if events and scores and len(events) == len(scores):
                        scores_array = np.asarray(scores, dtype=float)
                        top_index = int(scores_array.argmax())
                        current_top_event = events[top_index]
                        current_top_probability = float(scores_array[top_index])
                    else:
                        current_top_event = None
                        current_top_probability = None

                    if (
                        not arguments.no_event_log
                        and events
                        and scores
                    ):
                        send_event_probs_payload(
                            backend_url=backend_url,
                            frame_index=frame_index,
                            events=list(events),
                            scores=list(scores),
                            top_event=current_top_event,
                            top_probability=current_top_probability,
                        )

            elapsed = time.time() - start_time
            status_text = "ANOMALY" if is_current_anomaly else "NORMAL"
            status_color = (0, 0, 255) if is_current_anomaly else (0, 255, 0)

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                status_color,
                2,
                cv2.LINE_AA,
            )

            if current_top_event is not None and current_top_probability is not None:
                cv2.putText(
                    frame,
                    f"{current_top_event}: {current_top_probability:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"t={elapsed:.1f}s  frame={frame_index}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Realtime Behavior Client (backend /behavior/analyze)",
                frame,
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[INFO] interrupted by user.")
                break

    finally:
        capture.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
        print("[INFO] finished.")


if __name__ == "__main__":
    # Tensorflow 등이 섞여 있을 수 있는 환경을 조용히 만들기 위한 예전 습관을 유지.
    # Keep some legacy-style env tweaks for quiet logs if TF is present in the env.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    run_client_realtime()
