# 필요한 라이브러리 임포트
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, APIRouter, WebSocketDisconnect, Request
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
CAM_SERVER = "http://localhost:9000"
# 예측 결과 저장소
prediction_results = deque(maxlen=100)

@router.post("/get_predict")
async def get_predict(request: Request):
    data = await request.json()
    prediction = data.get("prediction")
    
    if prediction:
        prediction_results.append(prediction)
        return {"message": "Prediction received"}
    else:
        return {"message": "No prediction received"}, 400

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