# 필요한 라이브러리 임포트
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, APIRouter, WebSocketDisconnect
from collections import deque
import cv2
import numpy as np
import torch
import asyncio
import gc
import requests
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from goCam import *
from video_utils import *
from starlette.websockets import WebSocketState

# 서버 앱 초기화
router = APIRouter()

# 모델 및 설정 로드
LABELS = ["v", "d", "f"]
NORMAL_LABEL_IDX = 1
LSTM_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/models/transformer_augment.keras"
yolo_model = YOLO("yolo11n.pt")
lstm_model = load_model(LSTM_MODEL_PATH)
SEQUENCE_LENGTH = 80
# LINE_TOKEN = "owva2Uxp1YB1BeKxE31Ji8E1gy7DFwyZwQYd0UKsPRV"
SUSPICION_THRESHOLD = 4
EMERGENCY_THRESHOLD = 5

# 전역 변수 설정
CAM_SERVER = "http://localhost:9000"
HOST_URL = "localhost:8000"
is_streaming = False
recent_labels = deque(maxlen=10)

# WebSocket을 통한 비동기 스트림 전송
async def send_frames():
    global is_streaming
    async with websockets.connect(f'ws://{HOST_URL}/camera/stream') as websocket:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and is_streaming:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send(buffer.tobytes())
        cap.release()

# WebSocket 엔드포인트
@router.websocket("/camera/stream")
async def video_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence_data1 = deque(maxlen=SEQUENCE_LENGTH)
    sequence_data2 = deque(maxlen=SEQUENCE_LENGTH)
    sequence_data3 = deque(maxlen=SEQUENCE_LENGTH)
    prediction_label = "프레임 수집중"

    try:
        while True:
            # 클라이언트로부터 프레임 데이터를 수신
            frame_bytes = await websocket.receive_bytes()
            frame = process_frame(frame_bytes)

            # 바운딩 박스 및 포즈 데이터 추출
            bounding_boxes = detect_bounding_boxes(frame)
            if bounding_boxes:
                pose_data = process_pose_data(frame, bounding_boxes)
                if pose_data is not None:
                    flattened_pose = pose_data.reshape(-1)  # (132,) 형상으로 변경
                    sequence_data1.append(flattened_pose)
                    sequence_data2.append(flattened_pose)
                    sequence_data3.append(flattened_pose)

                # 80개의 프레임이 각 시퀀스에 쌓이면 예측 수행
                if len(sequence_data1) == SEQUENCE_LENGTH:
                    input_1 = np.array(sequence_data1).reshape(1, SEQUENCE_LENGTH, 132)
                    input_2 = np.array(sequence_data2).reshape(1, SEQUENCE_LENGTH, 132)
                    input_3 = np.array(sequence_data3).reshape(1, SEQUENCE_LENGTH, 132)

                    # 모델 예측
                    pred_idx = np.argmax(lstm_model.predict([input_1, input_2, input_3], verbose=0))
                    prediction_label = LABELS[pred_idx]

                    # 예측 결과 전송 및 의심 상황 감지
                    await websocket.send_text(f"Prediction: {prediction_label}")
                    recent_labels.append(prediction_label)
                    if recent_labels.count(prediction_label) == SUSPICION_THRESHOLD:
                        send_line_notify(f"{prediction_label} 의심 상황 발생")
                    if recent_labels.count(prediction_label) == EMERGENCY_THRESHOLD:
                        send_line_notify(f"{prediction_label} 상황 발생")

            # 프레임에 예측 결과 표시
            if frame is not None:
                cv2.putText(
                    frame,
                    f"Prediction: {prediction_label}",  # 예측 결과 출력
                    (10, 30),  # 위치 (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
                    1,  # 크기
                    (0, 255, 0),  # 색상 (초록색)
                    2,  # 두께
                    cv2.LINE_AA  # 라인 타입
                )

                # 수신된 프레임을 화면에 표시
                cv2.imshow("수신된 영상", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("영상 표시 종료")
                    break

            gc.collect()
            torch.cuda.empty_cache()

    except WebSocketDisconnect:
        print("WebSocket 연결이 닫혔습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # WebSocket 연결이 이미 닫혔는지 확인하고 닫지 않았을 경우에만 닫기
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

# 카메라 제어 API
@router.post("/camera/start")
async def cam_start():
    try:
        response = requests.post(f"{CAM_SERVER}/start")
        print(response.json() if response.status_code == 200 else f"스트림 시작 요청 실패: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"스트림 시작 요청 중 오류 발생: {e}")

@router.post("/camera/stop")
async def cam_stop():
    try:
        response = requests.post(f"{CAM_SERVER}/stop")
        print(response.json() if response.status_code == 200 else f"스트림 중지 요청 실패: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"스트림 중지 요청 중 오류 발생: {e}")

@router.get("/camera/check")
async def cam_check():
    response = requests.get(f"{CAM_SERVER}/check")
    if response.json().get('state'):
        return True
    else:
        return False