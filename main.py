# main.py
from fastapi import FastAPI
from router.audio import router as audio_router
from router.video import router as video_router

app = FastAPI()

# 각 모듈의 라우터를 메인 앱에 추가
app.include_router(audio_router)
app.include_router(video_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio and Video Processing API"}

# 서버 실행 (main.py로 직접 실행할 때만 작동)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

