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
from threading import Thread
import time

# 서버 앱 초기화
router = APIRouter()

# 모델 및 설정 로드
LABELS = ["폭력", "일상", "쓰러짐"]
NORMAL_LABEL_IDX = 1
LSTM_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/models/transformer_augment.keras"
yolo_model = YOLO("yolo11n.pt")
lstm_model = load_model(LSTM_MODEL_PATH)
SEQUENCE_LENGTH = 80
# LINE_TOKEN = "owva2Uxp1YB1BeKxE31Ji8E1gy7DFwyZwQYd0UKsPRV"
SUSPICION_THRESHOLD = 4
EMERGENCY_THRESHOLD = 10
DAILY_THRESHOLD = 30 * 60 * 5 - 60 

# 전역 변수 설정
# CAM_SERVER = "http://localhost:9000"
HOST_URL = "localhost:8000"
is_streaming = False
streaming_task = None
recent_labels = deque(maxlen=30)
daily_detect_labels = deque(maxlen=30 * 60 * 5) # 1초에 30 프레임 & 5분

# 스트림 종료를 위한 타이머 설정
last_suspicion_time = time.time()  # 초기화
stop_video_task = None

# 비동기 함수: 프레임을 서버에 전송하는 작업
async def send_frames():
    global is_streaming
    async with websockets.connect(F'ws://{HOST_URL}/camera/stream') as websocket:
        cap = cv2.VideoCapture(1)  # 1번 카메라 사용
        fc = 0  # 프레임 카운터

        while cap.isOpened() and is_streaming:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            fc += 1

            # 30 프레임에 한 번씩 서버에 전송
            if fc % 1 == 0:  # 조정 가능
                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send(buffer.tobytes())
                print(f"{fc}번째 프레임 전송 완료")

                try:
                    # 서버로부터 응답 수신
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"서버 응답: {response}")
                except asyncio.TimeoutError:
                    print("서버 응답 대기 시간이 초과되었습니다.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("사용자가 스트림을 종료했습니다.")
                break

        cap.release()
        print("카메라 스트림 종료")

# 백그라운드로 실행할 스레드용 함수
def run_stream():
    global streaming_task
    asyncio.run(send_frames())

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

                    # 예측 결과 WebSocket을 통해 클라이언트로 전송
                    await websocket.send_text(f"Prediction: {prediction_label}")
                    recent_labels.append(prediction_label)
                    daily_detect_labels.append(prediction_label)

                    # 상황 처리 로직
                    if prediction_label in ["폭력", "쓰러짐"]:
                        if recent_labels.count(prediction_label) == SUSPICION_THRESHOLD:
                            send_line_notify(f"{prediction_label} 의심 상황 발생")
                        if recent_labels.count(prediction_label) == EMERGENCY_THRESHOLD:
                            send_line_notify(f"{prediction_label} 상황 발생. 즉시 신고")
                            await cam_stop()
                            break
                    elif prediction_label == "일상":
                        if daily_detect_labels.count(prediction_label) == DAILY_THRESHOLD:
                            # 5분 동안 "일상" 상태 유지 시 종료
                            await cam_stop()
                            break
                        
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

# 스트림 시작 API 엔드포인트
@router.post("/camera/start")
async def cam_start(background_tasks: BackgroundTasks):
    global is_streaming, streaming_task
    if not is_streaming:
        is_streaming = True
        streaming_task = Thread(target=run_stream)
        streaming_task.start()
        return {"message": "스트림 시작됨"}
    else:
        return {"message": "스트림이 이미 실행 중입니다."}

# 스트림 중지 API 엔드포인트
@router.post("/camera/stop")
async def cam_stop():
    global is_streaming
    if is_streaming:
        is_streaming = False
        return {"message": "스트림 중지됨"}
    else:
        return {"message": "스트림이 실행 중이 아닙니다."}

# 스트림 상태 확인 API 엔드포인트
@router.get("/camera/check")
async def cam_check():
    return {"state": is_streaming}