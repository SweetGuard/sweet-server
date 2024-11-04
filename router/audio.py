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

# 서버 앱 초기화
router = APIRouter()

# 모델 및 설정 로드
AUDIO_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/sound_test.h5"
audio_model = load_model(AUDIO_MODEL_PATH)
SUSPICION_THRESHOLD = 10
EMERGENCY_THRESHOLD = 20



# 전역 변수 설정
CAM_SERVER = "http://localhost:9000"
HOST_URL = "localhost:8000"
BUFFER_DURATION = 5
SLIDING_INTERVAL = 1
  
SAMPLE_RATE = 16000  # assuming 16kHz sampling rate
audio_buffer: deque[np.ndarray] = deque(maxlen=int(BUFFER_DURATION * SAMPLE_RATE))
recent_labels = deque(maxlen=10)
required_timesteps = 13

# 음성 분석 API
@router.post("/classify-audio")
async def classify_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    audio_data = await file.read()
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    audio_buffer.extend(audio_array)

    # deque에 5초 분량이 쌓였는지 확인
    if len(audio_buffer) == BUFFER_DURATION * SAMPLE_RATE:
        # 5초 분량의 오디오 데이터를 numpy array로 변환
        buffer_array = np.array(audio_buffer)
        
        # 음성 데이터를 MFCC로 변환하여 (타임스텝, 13) 형상으로 맞춤
        mfcc = librosa.feature.mfcc(y=audio_array, sr=16000, n_mfcc=13)
        mfcc = mfcc.T  # (타임스텝, 13) 형상으로 전치하여 (타임스텝, 13)

        # 타임스텝을 13으로 맞춤
        if mfcc.shape[0] < required_timesteps:
            # 부족한 타임스텝을 0으로 패딩
            padding = np.zeros((required_timesteps - mfcc.shape[0], 13))
            mfcc = np.vstack([mfcc, padding])
        elif mfcc.shape[0] > required_timesteps:
            # 타임스텝이 초과된 경우 자르기
            mfcc = mfcc[:required_timesteps]

        # 최종 형상 조정: (1, 13, 1, 1)
        mfcc = mfcc.reshape((13, 13))  # (타임스텝, 13) 형상 유지
        mfcc = mfcc[:, :, np.newaxis, np.newaxis]  # (13, 13

        # 모델 예측
        prediction = audio_model.predict(mfcc)
        predicted_class = int(np.argmax(prediction))
        print(f"Prediction class: {predicted_class}")
        prediction_label = "일상"

    # # 예측된 클래스에 따라 상황별 핸들링 수행
    # if predicted_class == 0:
    #     result = "범죄"
    #     await handle_crime_situation("crime")
    # elif predicted_class == 1:
    #     result = "쓰러짐"
    #     await handle_fall_situation(background_tasks, "fall")
    # elif predicted_class == 2:
    #     result = "도와줘"
    #     await handle_help_situation("help")
    # else:
    #     result = "일상"

        if predicted_class == 1:
            prediction_label = "일상"
        elif predicted_class == 0: # "비정상"범주임 (아직 비정상이 분류되지 않음)
            prediction_label = "범죄"
            await handle_crime_situation(prediction_label)

        recent_labels.append(prediction_label)

        # 슬라이딩 윈도우 방식으로 버퍼 일부 제거
        if len(audio_buffer) == int(BUFFER_DURATION * SAMPLE_RATE):
            del list(audio_buffer)[:int(SLIDING_INTERVAL * SAMPLE_RATE)]

        return {"status": "success", "classification": prediction_label}
    
    return {"status": "waiting", "message": "wait at least 5 secs"}

# 상황별 함수
async def handle_crime_situation(prediction_label):
    requests.post(f"{CAM_SERVER}/start")
    process_notification("crime", prediction_label, recent_labels)

async def handle_fall_situation(background_tasks: BackgroundTasks, prediction_label):
    if not await ask_user_if_ok():
        background_tasks.add_task(send_frames)
        process_notification("fall", prediction_label, recent_labels)

async def handle_help_situation(prediction_label):
    process_notification("help", prediction_label, recent_labels)

# 음성 어시스턴트가 사용자에게 괜찮은지 묻는 함수 (아직 implement x)
async def ask_user_if_ok():
    print("음성 어시스턴트: '괜찮으십니까? 예/아니오로 응답해 주세요.'")
    await asyncio.sleep(3)
    return False  # 예시: 기본적으로 응답 없음

# 메시지 전송 함수
def process_notification(situation_type: str, prediction_label: str, recent_labels):
    label_count = recent_labels.count(prediction_label)
    if label_count == SUSPICION_THRESHOLD:
        send_line_notify(f"{situation_type} 의심 상황 발생 ({label_count}회)")
    elif label_count == EMERGENCY_THRESHOLD:
        send_line_notify(f"{situation_type} 상황 발생 ({label_count}회)")

