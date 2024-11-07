import cv2
import websockets
import asyncio
from fastapi import FastAPI, BackgroundTasks
from threading import Thread
from fastapi import FastAPI, BackgroundTasks
from threading import Thread
from collections import deque
from video_utils import * 
import numpy as np
import cv2
import gc
import torch
import requests
import time
from tensorflow.keras.models import load_model
from fastapi import FastAPI, BackgroundTasks
from threading import Thread
from collections import deque
from video_utils import * 
import numpy as np
import cv2
import gc
import torch
import requests
import time
from tensorflow.keras.models import load_model

app = FastAPI()
is_streaming = False
streaming_task = None

MAIN_SERVER_URL = "http://192.168.0.184:8000/get_predict"

SEQUENCE_LENGTH = 80
LABELS = ["daily", "danger", "falldown"]
SUSPICION_THRESHOLD = 5
EMERGENCY_THRESHOLD = 10
DAILY_THRESHOLD = 300
LSTM_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/models/transformer_bestacc.keras"
lstm_model = load_model(LSTM_MODEL_PATH)

recent_labels = deque(maxlen=100)

# 노티 전송 여부를 저장하는 플래그
notification_sent = {"danger": False, "falldown": False}

def reset_notifications():
    global notification_sent
    # 노티 전송 플래그 초기화
    notification_sent = {key: False for key in notification_sent.keys()}

# 모델 및 예측 함수
def predict_with_model(input):
    pred_idx = np.argmax(lstm_model.predict([input], verbose=0))
    return LABELS[pred_idx]

def run_stream():
    global is_streaming
    cap = cv2.VideoCapture(1)  # 카메라 시작
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)
    prediction_label = "Frame incoming"

    while is_streaming and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 바운딩 박스 및 포즈 데이터 추출
        bounding_boxes = detect_bounding_boxes(frame)
        if bounding_boxes:
            pose_data = process_pose_data(frame, bounding_boxes)
            if pose_data is not None:
                flattened_pose = pose_data.reshape(-1)
                sequence_data.append(flattened_pose)

            if len(sequence_data) == SEQUENCE_LENGTH:
                input_data = np.array(sequence_data).reshape(1, SEQUENCE_LENGTH, 396)

                prediction_label = predict_with_model(input_data)
                recent_labels.append(prediction_label)

                try:
                    requests.post(MAIN_SERVER_URL, json={"prediction": prediction_label})
                except Exception as e:
                    print(f"Error sending prediction: {e}")

                # 상황 처리 로직 및 노티 전송 관리
                if prediction_label in ["danger", "falldown"]:
                    if not notification_sent[prediction_label]:  # 각 상황에 대해 1회 노티 전송
                        if recent_labels.count(prediction_label) >= SUSPICION_THRESHOLD:
                            send_line_notify(f"{prediction_label} 의심 상황 발생")
                            notification_sent[prediction_label] = True
                        if recent_labels.count(prediction_label) >= EMERGENCY_THRESHOLD:
                            send_line_notify(f"{prediction_label} 상황 발생. 즉시 신고")
                            notification_sent[prediction_label] = True

        # 프레임에 예측 결과 표시
        if frame is not None:
            cv2.putText(
                frame,
                f"Prediction: {prediction_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("수신된 영상", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("영상 표시 종료")
                print("영상 표시 종료")
                break

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

@app.post("/start")
def start_stream(background_tasks: BackgroundTasks):
    global is_streaming, streaming_task, sequence_data

    if not is_streaming:
        is_streaming = True
        reset_notifications()  # 스트림 시작 시 노티 플래그 초기화

        # deque 초기화
        sequence_data = deque(maxlen=SEQUENCE_LENGTH)

        # 스트림을 새로 시작
        streaming_task = Thread(target=run_stream)
        streaming_task.start()
        return {"message": "스트림 시작됨"}
    else:
        return {"message": "스트림이 이미 실행 중입니다."}

# 스트림 중지 API 엔드포인트
@app.post("/stop")
def stop_stream():
    global is_streaming
    if is_streaming:
        is_streaming = False
        return {"message": "스트림 중지됨"}
    else:
        return {"message": "스트림이 실행 중이 아닙니다."}

    
@app.get("/check")
def check():
    if is_streaming:
        print("true##################################################################3")
        return {"state": True }
    else:
        print("false##################################################################3")
        return {"state":False}

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)