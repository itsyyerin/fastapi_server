from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import librosa
from pydub import AudioSegment
import numpy as np
import logging  # 로그 모듈 추가
from model import ECAPAModel
from model.ECAPAModel import *
from model.model import ECAPA_TDNN  # 모델 임포트
from model.dataLoader import *  # 데이터 로더 임포트
from model.tools import *  # 툴 임포트

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

MODEL_PATH = "./model/exps/pretrain.model"  # pretrain.model 경로
device = torch.device("cpu")  # CPU 사용
model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=1863, m=0.2, s=30, test_step=10, device=device).to(device)

# 모델 가중치를 선택적으로 로드하는 코드
state_dict = torch.load(MODEL_PATH, map_location=device)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()


# 파일을 WAV로 변환하는 함수
def convert_to_wav(file: UploadFile, save_path: str):
    file_path = os.path.join(save_path, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # MP4 파일이면 WAV로 변환
    if file_path.endswith('.mp4'):
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(".mp4", ".wav")
        audio.export(wav_path, format="wav")
        logging.info(f"Converted {file.filename} to WAV format.")  # 변환 로그
        return wav_path
    return file_path


def preprocess_audio(wav_path: str):
    logging.info(f"Extracting features from {wav_path}...")  # 특징 추출 시작 로그
    audio_data, sr = librosa.load(wav_path, sr=16000)
    max_audio = 300 * 160 + 240
    if len(audio_data) < max_audio:
        audio_data = np.pad(audio_data, (0, max_audio - len(audio_data)), 'wrap')
    else:
        audio_data = audio_data[:max_audio]
    audio_tensor = torch.FloatTensor(np.stack([audio_data], axis=0)).to(device)  # CPU에서 처리
    logging.info(f"Feature extraction completed for {wav_path}.")  # 특징 추출 완료 로그
    return audio_tensor


@app.post("/upload/")
async def process_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    save_dir = "./uploads"
    os.makedirs(save_dir, exist_ok=True)

    # 파일을 받았다는 로그 기록
    logging.info(f"Received files: {file1.filename}, {file2.filename}")

    wav1_path = convert_to_wav(file1, save_dir)
    wav2_path = convert_to_wav(file2, save_dir)

    audio1 = preprocess_audio(wav1_path)
    audio2 = preprocess_audio(wav2_path)

    with torch.no_grad():
        embedding1 = model(audio1)
        embedding2 = model(audio2)
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)

    # AI 결과가 나왔다는 로그 기록
    logging.info("AI processing completed.")

    # 유사도 로그 기록
    logging.info(f"Similarity score: {similarity.item()}")

    # 동일인으로 판단된 경우 점수 변환 적용
    if similarity.item() >= 0.5:
        original_score = similarity.item()

        # 유사도 점수 변환: 0.5 ~ 1.0 사이의 값을 0.8 ~ 0.9 사이로 변환
        min_score = 0.5  # 변환할 최소 유사도
        max_score = 1.0  # 변환할 최대 유사도

        # 유사도 값을 0 ~ 1 범위로 정규화 후, 0.8 ~ 0.9 사이로 변환
        adjusted_score = (original_score - min_score) / (max_score - min_score)  # 0 ~ 1로 정규화
        adjusted_score = adjusted_score * 0.1 + 0.8  # 0.8 ~ 0.9로 변환


        # 동일인으로 판단된 경우 유사도 점수를 0.2씩 올림
        final_score = adjusted_score  #+ 0.2

        # 유사도 점수는 최대 1.0을 넘지 않도록 제한
        if final_score > 1.0:
            final_score = 1.0

        # 유사도 로그 기록
        logging.info(f"Adjusted similarity score (same): {final_score}")

        return {"similarity_score": final_score, "result": "same"}  # 변환된 유사도 점수 반환
    else:
        logging.info("Result: different")
        return {"similarity_score": similarity.item(), "result": "different"}  # 동일인이 아니면 원래 점수 반환
