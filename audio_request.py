import requests
import time
from pydub import AudioSegment

url = "http://localhost:8000/classify-audio"  # 서버의 classify_audio 엔드포인트 URL
file_path = "/Users/trispark/summer2024/sweet_guard/server/test_sounds/낙상_combined.wav"  # 업로드할 파일 경로

# 오디오 파일 로드
audio = AudioSegment.from_wav(file_path)

# 오디오 파일을 1초(1000ms)씩 분할하여 전송
segment_duration = 1000  # 1초 = 1000ms
for i in range(0, len(audio), segment_duration):
    segment = audio[i:i + segment_duration]
    
    # 분할된 1초 분량의 오디오를 임시 파일로 저장
    segment_path = "/tmp/temp_segment.wav"
    segment.export(segment_path, format="wav")
    
    # 서버에 파일 전송
    with open(segment_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        
        # 응답 확인
        if response.ok:
            print(f"Segment {i // segment_duration + 1}: {response.json()}")
        else:
            print(f"Segment {i // segment_duration + 1}: 오류 발생:", response.text)

    # 1초 대기
    time.sleep(1)