###fastapi 서버 실행 방법
# 1. 안드로이드스튜디오 home.dart에서 서버ip로 바꾸고
# 2. anaconda에서 base가상환경에서 이 프로젝트 경로 찾아간 다음,
# 서버랑 같은 컴에서 접속할거면 uvicorn main:app --reload
# 서버랑 다른 컴에서 접속할거면 uvicorn main:app --host 0.0.0.0 --port 8000
# 하면됨. application startup complete. uvicorn running on 되면 서버실행중.







#
# # 두 MFCC 비교 API
# @app.post("/compare-mfcc/")
# async def compare_mfcc(mfcc1: list, mfcc2: list):
#     mfcc1 = np.array(mfcc1)
#     mfcc2 = np.array(mfcc2)
#
#     # 코사인 유사도 계산
#     similarity = np.dot(mfcc1.mean(axis=0), mfcc2.mean(axis=0)) / (
#             np.linalg.norm(mfcc1.mean(axis=0)) * np.linalg.norm(mfcc2.mean(axis=0)))
#     is_same_speaker = similarity > 0.8  # 유사도 임계값
#
#     return {"is_same_speaker": is_same_speaker, "similarity_score": similarity}
#
#
# # 두 파일을 업로드받아 MFCC 비교
# @app.post("/compare_mfcc_files/")
# async def compare_mfcc_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
#     file2_path = os.path.join(UPLOAD_FOLDER, file2.filename)
#
#     # 파일 저장
#     with open(file1_path, "wb") as f1, open(file2_path, "wb") as f2:
#         f1.write(await file1.read())
#         f2.write(await file2.read())
#
#     # MFCC 추출
#     mfcc1 = extract_mfcc(file1_path)
#     mfcc2 = extract_mfcc(file2_path)
#
#     # 코사인 유사도 비교
#     similarity = np.dot(np.array(mfcc1).mean(axis=0), np.array(mfcc2).mean(axis=0)) / (
#             np.linalg.norm(np.array(mfcc1).mean(axis=0)) * np.linalg.norm(np.array(mfcc2).mean(axis=0)))
#     is_same_speaker = similarity > 0.8  # 임계값을 통해 동일 화자인지 확인
#
#     # 파일 삭제 (선택 사항)
#     os.remove(file1_path)
#     os.remove(file2_path)
#
#     return {"is_same_speaker": is_same_speaker, "similarity_score": similarity}
