from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import torch
import subprocess
import sys

# sys.path에 model 폴더 경로를 강제로 추가하여 모듈을 찾을 수 있도록 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

# 모델 관련 모듈 임포트
from ECAPAModel import ECAPAModel  # model 폴더에서 ECAPAModel 임포트
from dataLoader import *  # model 폴더에서 dataLoader 임포트
from tools import *  # model 폴더에서 tools 임포트

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드된 파일을 저장할 폴더 설정
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 모델 로드 함수
def load_model():
    model_path = os.path.abspath("./pretrain.model")  # 절대 경로로 모델 파일 경로 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU 또는 CUDA 사용

    # ECAPAModel 초기화 및 파라미터 로드
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=1863, m=0.2, s=30, test_step=10, device=device).to(
        device)
####
    # 모델 파라미터 로드 (오류 발생 시 로그 출력)
    try:
        model.load_parameters(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed.")

    model.eval()  # 평가 모드로 전환
    return model, device


# 모델 로드
model, device = load_model()


# MFCC 추출 함수
def extract_mfcc(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File not found")

        # m4a 또는 mp4 파일 변환
        if file_path.endswith('.m4a') or file_path.endswith('.mp4'):
            if file_path.endswith('.mp4'):
                video = VideoFileClip(file_path)
                audio_path = file_path.replace('.mp4', '.wav')
                video.audio.write_audiofile(audio_path)
                file_path = audio_path

            elif file_path.endswith('.m4a'):
                wav_path = file_path.replace('.m4a', '.wav')
                command = ['ffmpeg', '-i', file_path, wav_path]
                subprocess.run(command, check=True)
                file_path = wav_path

        y, sr = librosa.load(file_path, sr=16000)
        max_audio = 300 * 160 + 240

        if len(y) < max_audio:
            y = np.pad(y, (0, max_audio - len(y)), 'wrap')
        else:
            y = y[:max_audio]

        audio_tensor = torch.FloatTensor(np.stack([y], axis=0)).to(device)
        return audio_tensor

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MFCC 추출 실패 이유 : {str(e)}")


# 업로드 및 처리 경로
@app.post("/upload-proper-speaker/")
async def upload_proper_speaker(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    audio_tensor = extract_mfcc(file_path)
    return {"info": "적합한 화자의 데이터가 처리되었습니다.", "mfcc": audio_tensor.tolist()}


@app.post("/upload-compare-speaker/")
async def upload_compare_speaker(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    audio_tensor = extract_mfcc(file_path)
    return {"info": "비교 대상 화자의 데이터가 처리되었습니다.", "mfcc": audio_tensor.tolist()}


@app.post("/compare-speakers/")
async def compare_speakers(mfcc_file1: UploadFile = File(...), mfcc_file2: UploadFile = File(...)):
    try:
        # 두 오디오 임베딩 비교
        audio1 = extract_mfcc(os.path.join(UPLOAD_FOLDER, mfcc_file1.filename))
        audio2 = extract_mfcc(os.path.join(UPLOAD_FOLDER, mfcc_file2.filename))

        with torch.no_grad():
            embedding1 = model(audio1, aug=False)
            embedding2 = model(audio2, aug=False)

        # 코사인 유사도 계산
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)

        # 유사도 기준에 따라 동일 여부 판단
        if similarity.item() >= 0.6:
            return {"result": "same", "similarity": similarity.item()}
        else:
            return {"result": "different", "similarity": similarity.item()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"화자 비교 중 오류 발생: {str(e)}")
