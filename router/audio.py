# 필요한 라이브러리 임포트
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, APIRouter
from collections import deque
import numpy as np
import torch
import asyncio
import gc
import requests
from tensorflow.keras.models import load_model
import librosa
from router.video import *
import speech_recognition as sr
import time
from datetime import datetime, timedelta, timezone
from fastapi.responses import JSONResponse

router = APIRouter()

# 모델 및 설정 로드
AUDIO_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/models/jhyaudio2.h5"
audio_model = load_model(AUDIO_MODEL_PATH)
SUSPICION_THRESHOLD = 3

# 전역 변수 설정
CAM_SERVER = "http://192.168.0.226:9000"
BUFFER_DURATION = 5  # 5초 분량의 오디오 데이터
SLIDING_INTERVAL = 1  # 1초마다 슬라이딩
SAMPLE_RATE = 16000  # 16kHz 샘플링
audio_buffer: deque[np.ndarray] = deque(maxlen=int(BUFFER_DURATION * SAMPLE_RATE))
recent_labels = deque(maxlen=30)
import librosa
import numpy as np
import cv2

# WebSocket 엔드포인트로 모든 음성 처리 통합
@router.websocket("/ws/audio")
async def audio_processing(websocket: WebSocket):
    await websocket.accept()
    recognizer = sr.Recognizer()
    print("WebSocket 연결됨: 실시간 음성 데이터 처리 시작")

    try:
        while True:
            # 클라이언트로부터 음성 데이터 수신
            audio_chunk = await websocket.receive_bytes()
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            audio_buffer.extend(audio_array)

            # 5초 분량의 오디오 데이터가 쌓이면 예측 수행
            if len(audio_buffer) == BUFFER_DURATION * SAMPLE_RATE:
                buffer_array = np.array(audio_buffer)
                
                # MFCC 추출 및 변환
                segments = extract_mfcc_segments(buffer_array, sr=SAMPLE_RATE)
                
                # 각 세그먼트를 모델의 input shape에 맞게 변환하고 예측
                for segment in segments:
                    # 128x128로 리사이즈, 3 채널 유지
                    data_resized = cv2.resize(segment, (128, 128))
                    data_resized = np.expand_dims(data_resized, axis=0)  # 배치 차원 추가
                    data_resized = np.expand_dims(data_resized, axis=-1)  # 채널 차원 추가하여 (1, 128, 128, 3)로 맞춤
                    
                    # 모델 예측
                    prediction = audio_model.predict(data_resized, verbose=0)
                    predicted_class = int(np.argmax(prediction))
                    prediction_label = "일상" if predicted_class == 0 else "위험"
                    print(f"Prediction class at auidio1: {predicted_class}")

                    # 예측 결과에 따른 처리
                    recent_labels.append(prediction_label)

                    if prediction_label == "위험":
                        label_count = recent_labels.count(prediction_label)
                        if label_count == SUSPICION_THRESHOLD:
                            await websocket.send_json({"status": 200,  "code": 200100, "message": "위험 상황 발생. 연결 종료"})
                            # await websocket.send_text("play_warning_message")
                            # await websocket.close()
                            print("위험 상황 발생 - WebSocket 연결 종료")
                            return

                # 슬라이딩 윈도우 방식으로 버퍼 일부 제거 (1초 분량)
                if len(audio_buffer) == int(BUFFER_DURATION * SAMPLE_RATE):
                    del list(audio_buffer)[:int(SLIDING_INTERVAL * SAMPLE_RATE)]

    except WebSocketDisconnect:
        print("WebSocket 연결이 닫혔습니다.")
    finally:
        await websocket.close()

def extract_mfcc_segments(audio_data, sr=16000, duration=5):
    """
    오디오 데이터를 5초씩 잘라서 MFCC 추출
    """
    segments = []
    total_length = librosa.get_duration(y=audio_data, sr=sr)
    
    for start in range(0, int(total_length), duration):
        end = start + duration
        if end <= total_length:
            # 5초 구간 오디오 추출
            y_segment = audio_data[start * sr:end * sr]
            
            # MFCC 추출 (157개 MFCC, 3채널)
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=157)
            mfcc_3ch = np.stack([mfcc] * 3, axis=-1)  # 3채널로 맞춤
            segments.append(mfcc_3ch)
    
    return segments

@router.post("/handle_abnormal_situation_file")
async def handle_abnormal_situation_file(file: UploadFile = File(...)):
    # test_cam_server_connection()

    recognizer = sr.Recognizer()
    audio_data = await file.read()
    audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
    recognized = False

    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        print(f"Recognized Text: {text}")

        # 위험 상황에 따른 클라이언트 알림
        if "도와줘" in text:
            recognized = True
            send_line_notify("위험 상황 발생 - 도와줘 감지")
            recent_labels.clear()
            return JSONResponse(content={"code": 200101, "message": "play danger message"}, status_code=200)
        elif "괜찮아" in text:
            recognized = True
            recent_labels.clear()
            return JSONResponse(content={"code": 200102, "message": "play fine message"}, status_code=200)

    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
        # return JSONResponse(content={"status": "error", "message": "음성을 인식하지 못했습니다."}, status_code=400)
    except sr.RequestError as e:
        print(f"Google Speech Recognition 서비스에 접근할 수 없습니다. 오류: {e}")
        # return JSONResponse(content={"status": "error", "message": f"음성 인식 서비스 오류: {e}"}, status_code=500)

    # 타임아웃 또는 인식 실패 시 무응답 처리
    if not recognized:
        print("10초 동안 응답이 없어 무응답으로 처리합니다.")
        send_line_notify("위험 예상 상황 발생 not recognized - 도움 요청")
        requests.post(f"{CAM_SERVER}/start")
        return JSONResponse(content={"code": 200103, "message": "play no response message"}, status_code=200)
