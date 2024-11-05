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

# 서버 앱 초기화
router = APIRouter()

# 모델 및 설정 로드
AUDIO_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/models/model_image_10sec.keras"
audio_model = load_model(AUDIO_MODEL_PATH)
SUSPICION_THRESHOLD = 10

# 전역 변수 설정
CAM_SERVER = "http://localhost:9000"
HOST_URL = "localhost:8000"
BUFFER_DURATION = 5
SLIDING_INTERVAL = 1

SAMPLE_RATE = 16000  # assuming 16kHz sampling rate
audio_buffer: deque[np.ndarray] = deque(maxlen=int(BUFFER_DURATION * SAMPLE_RATE))
recent_labels = deque(maxlen=30)

# WebSocket 엔드포인트: 음성 메시지 및 실시간 오디오 데이터 처리
@router.websocket("/ws/handle_abnormal_situation")
async def handle_abnormal_situation(websocket: WebSocket):
    await websocket.accept()

    # 경고 메시지 요청
    await websocket.send_text("play_warning_message")
    recognizer = sr.Recognizer()
    recognized = False

    try:
        print("Listening for '도와줘' or '괜찮아'...")

        # 실시간 음성 데이터를 WebSocket으로 수신하여 처리
        while True:
            # 클라이언트로부터 음성 데이터 수신
            audio_chunk = await websocket.receive_bytes()
            audio_data = sr.AudioData(audio_chunk, sample_rate=16000, sample_width=2)

            try:
                # Google Speech Recognition으로 음성 인식
                text = recognizer.recognize_google(audio_data, language="ko-KR")
                print(f"Recognized Text: {text}")

                # 특정 문구에 따라 상황을 처리하고 클라이언트에 알림
                if "도와줘" in text:
                    recognized = True
                    await websocket.send_text("play_danger_message")
                    send_line_notify("위험 상황 발생 - 도와줘 감지")
                    recent_labels.clear()
                    break

                elif "괜찮아" in text:
                    recognized = True
                    await websocket.send_text("play_fine_message")
                    recent_labels.clear()
                    break

            except sr.UnknownValueError:
                print("음성을 인식하지 못했습니다.")
            except sr.RequestError as e:
                print(f"Google Speech Recognition 서비스에 접근할 수 없습니다. 오류: {e}")

        # 음성을 인식하지 못했거나 응답이 없을 경우 처리
        if not recognized:
            await websocket.send_text("play_no_response_message")
            send_line_notify("위험 상황 예측 - 응답 없음")
            requests.post(f"{CAM_SERVER}/start")

    except WebSocketDisconnect:
        print("WebSocket 연결이 닫혔습니다.")
    finally:
        await websocket.close()

# 오디오 분류 API 엔드포인트
@router.post("/classify-audio")
async def classify_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    audio_data = await file.read()
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    audio_buffer.extend(audio_array)

    # deque에 5초 분량이 쌓였는지 확인
    if len(audio_buffer) == BUFFER_DURATION * SAMPLE_RATE:
        buffer_array = np.array(audio_buffer)

        # 음성 데이터를 MFCC로 변환하여 (타임스텝, 13) 형상으로 맞춤
        mfcc = librosa.feature.mfcc(y=buffer_array, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc = mfcc.T

        # (400, 1000) 크기로 맞춤
        if mfcc.shape[0] < 400:
            padding = np.zeros((400 - mfcc.shape[0], 13))
            mfcc = np.vstack([mfcc, padding])
        elif mfcc.shape[0] > 400:
            mfcc = mfcc[:400]

        # MFCC를 (400, 1000, 1) 크기로 변환
        mfcc = np.tile(mfcc, (1, int(np.ceil(1000 / mfcc.shape[1]))))[:400, :1000]
        mfcc = mfcc[:, :, np.newaxis]  # (400, 1000, 1)

        # 모델 예측
        prediction = audio_model.predict(np.expand_dims(mfcc, axis=0))
        predicted_class = int(np.argmax(prediction))
        prediction_label = "일상" if predicted_class == 1 else "위험"
        print(f"Prediction class: {predicted_class}")

        # 예측 결과 처리
        recent_labels.append(prediction_label)
        if prediction_label == "위험":
            label_count = recent_labels.count(prediction_label)
            if label_count == SUSPICION_THRESHOLD:
                background_tasks.add_task(handle_abnormal_situation, prediction_label)

        # 슬라이딩 윈도우 방식으로 버퍼 일부 제거
        if len(audio_buffer) == int(BUFFER_DURATION * SAMPLE_RATE):
            del list(audio_buffer)[:int(SLIDING_INTERVAL * SAMPLE_RATE)]

        return {"status": "성공", "classification": prediction_label}

    return {"status": "처리중", "message": "5초 정도 기다려주세요"}