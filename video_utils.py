from collections import deque
import cv2
import numpy as np
import torch
import asyncio
import gc
import requests
import mediapipe as mp
from ultralytics import YOLO
from goCam import *

LINE_TOKEN = "owva2Uxp1YB1BeKxE31Ji8E1gy7DFwyZwQYd0UKsPRV"
LSTM_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/transformer_augment.keras"
yolo_model = YOLO("yolo11n.pt")

# LINE 알림 전송 함수
def send_line_notify(message, token=LINE_TOKEN):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': message}
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print('메시지 전송 성공')
    else:
        print(f'메시지 전송 실패: {response.status_code}')


# YOLO 객체 탐지 및 포즈 데이터 처리 함수들
def process_frame(frame_bytes):
    frame = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)

def detect_bounding_boxes(frame):
    with torch.no_grad():
        results = yolo_model(frame, verbose=False)
    return [r.boxes.xywh[j].cpu().numpy() for r in results if 0 in r.boxes.cls.cpu().numpy() for j, c in enumerate(r.boxes.cls.cpu().numpy()) if c == 0]

def process_pose_data(frame, bounding_boxes):
    frame_to_pose = FrameToPoseArray(frame, bounding_boxes)
    pose_data = frame_to_pose.get_pose_data()
    return pose_data if pose_data.shape == (99, 4) else None

class FrameToPoseArray:
    def __init__(self, frame, bounding_boxes):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.frame = frame
        self.bounding_boxes = bounding_boxes
        self.cropped_images = []
        self.pose_array = []

    def crop_images(self):
        for bbox in self.bounding_boxes:
            cx, cy, w, h = map(int, bbox)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            self.cropped_images.append(self.frame[y1:y2, x1:x2])

    def extract_pose_landmarks(self):
        for cropped_image in self.cropped_images:
            result = self.pose.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            self.pose_array.append(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]) if result.pose_landmarks else np.zeros((33, 4)))

    def pad_pose_data(self):
        while len(self.pose_array) < 3:
            self.pose_array.append(np.zeros((33, 4)))
        return np.array(self.pose_array[:3])

    def get_pose_data(self):
        self.crop_images()
        self.extract_pose_landmarks()
        self.pad_pose_data()
        return np.concatenate(self.pose_array)