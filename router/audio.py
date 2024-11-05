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
from playsound import playsound

# 서버 앱 초기화
router = APIRouter()

# 모델 및 설정 로드
AUDIO_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/models/sound_test2.h5"
audio_model = load_model(AUDIO_MODEL_PATH)
SUSPICION_THRESHOLD = 3
# EMERGENCY_THRESHOLD = 20



# 전역 변수 설정
CAM_SERVER = "http://localhost:9000"
HOST_URL = "localhost:8000"
BUFFER_DURATION = 5
SLIDING_INTERVAL = 1
  
SAMPLE_RATE = 16000  # assuming 16kHz sampling rate
audio_buffer: deque[np.ndarray] = deque(maxlen=int(BUFFER_DURATION * SAMPLE_RATE))
recent_labels = deque(maxlen=30)
required_timesteps = 128

@router.post("/classify-audio")
async def classify_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    audio_data = await file.read()
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    audio_buffer.extend(audio_array)

    # deque에 5초 분량이 쌓였는지 확인
    if len(audio_buffer) == BUFFER_DURATION * SAMPLE_RATE:
        buffer_array = np.array(audio_buffer)

        # 음성 데이터를 MFCC로 변환하여 (타임스텝, 13) 형상으로 맞춤
        mfcc = librosa.feature.mfcc(y=audio_array, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc = mfcc.T  # (타임스텝, 13) 형상으로 전치하여 (타임스텝, 13)

        # (128, 128) 크기로 맞춤
        if mfcc.shape[0] < required_timesteps:
            # 부족한 타임스텝을 0으로 패딩
            padding = np.zeros((required_timesteps - mfcc.shape[0], 13))
            mfcc = np.vstack([mfcc, padding])
        elif mfcc.shape[0] > required_timesteps:
            # 타임스텝이 초과된 경우 자르기
            mfcc = mfcc[:required_timesteps]

        # MFCC를 (128, 128, 1) 크기로 변환
        mfcc = np.tile(mfcc, (1, int(np.ceil(128 / mfcc.shape[1]))))[:128, :128]  # (128, 128)
        mfcc = mfcc[:, :, np.newaxis]  # (128, 128, 1)

        # 모델 예측
        prediction = audio_model.predict(np.expand_dims(mfcc, axis=0))  # 배치 차원 추가
        predicted_class = int(np.argmax(prediction))
        print(f"Prediction class: {predicted_class}")
        prediction_label = "일상" if predicted_class == 1 else "위험"

        # recent_labels에 예측 결과 추가
        recent_labels.append(prediction_label)

        # 비정상 상황일 때만 비동기 처리
        if prediction_label == "위험":
            label_count = recent_labels.count(prediction_label)
            if label_count == SUSPICION_THRESHOLD:
                background_tasks.add_task(handle_abnormal_situation, prediction_label)

        # 슬라이딩 윈도우 방식으로 버퍼 일부 제거
        if len(audio_buffer) == int(BUFFER_DURATION * SAMPLE_RATE):
            del list(audio_buffer)[:int(SLIDING_INTERVAL * SAMPLE_RATE)]

        return {"status": "성공", "classification": prediction_label}

    return {"status": "처리중", "message": "5초 정도 기다려주세요"}

# 상황별 함수
async def handle_abnormal_situation(prediction_label):
    # 괜찮냐고 물어봄 
    # -> 괜찮다면 상황 종료 / 도와달라하면 바로 신고 
    # 대답이 없으면 우선 위험이 예상되는 상황임을 연락하고 캠을 켬 -> 캠으로 위험 상황 감지되면 위험 상황 연락
    playsound("/Users/trispark/summer2024/sweet_guard/server/voice_assistant/warning_message_korean.mp3")

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for '도와줘' or '괜찮아'...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # 주변 소음 조절
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # 대기 시간 설정

    recognized = False
    try:
        text = recognizer.recognize_google(audio, language="ko-KR")  # 한국어 인식
        print(f"Recognized Text: {text}")

        if "도와줘" in text:
            # 도와달라하면 바로 신고 
            recognized = True
            playsound("/Users/trispark/summer2024/sweet_guard/server/voice_assistant/danger.mp3")
            send_line_notify(f"{prediction_label} 상황 발생")
            recent_labels.clear()
            
        elif "괜찮아" in text:
            # 괜찮다면 상황 종료
            recognized = True
            playsound('/Users/trispark/summer2024/sweet_guard/server/voice_assistant/fine.mp3')
            recent_labels.clear()

    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
    except sr.RequestError as e:
        print(f"Google Speech Recognition 서비스에 접근할 수 없습니다. 오류: {e}")
    
    # 음성을 인식하지 못하거나 원하는 문구가 없을 때 처리
    if not recognized:
        recognized = False
        # 대답이 없으면 우선 위험이 예상되는 상황임을 연락하고 캠을 켬 -> 캠으로 위험 상황 감지되면 위험 상황 연락
        playsound("/Users/trispark/summer2024/sweet_guard/server/voice_assistant/no_response.mp3")
        send_line_notify(f"{prediction_label} 예상 상황 발생. 도움 요청.")
        requests.post(f"{CAM_SERVER}/start")

# def process_abnormal_situation(prediction_label, recent_labels):
#     label_count = recent_labels.count(prediction_label)
#     if label_count == SUSPICION_THRESHOLD:
#         handle_abnormal_situation(prediction_label)
        

