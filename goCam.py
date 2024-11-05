import cv2
import websockets
import asyncio
from fastapi import FastAPI, BackgroundTasks
from threading import Thread

# uvicorn goCam:app --host 0.0.0.0 --port 9000 --reload

app = FastAPI()

# 전역 변수 설정
streaming_task = None
is_streaming = False
HOST_URL = "localhost:8000"

# 비동기 함수: 프레임을 서버에 전송하는 작업
async def send_frames():
    global is_streaming
    async with websockets.connect(F'ws://{HOST_URL}/camera/stream') as websocket:
        cap = cv2.VideoCapture(0)  # 1번 카메라 사용
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

# 스트림 시작 API 엔드포인트
@app.post("/start")
def start_stream(background_tasks: BackgroundTasks):
    global is_streaming, streaming_task
    if not is_streaming:
        is_streaming = True
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
        return {"state": True }
    else:
        return {"state":False}

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
