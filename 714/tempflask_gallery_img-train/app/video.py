import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)

# InsightFace 모델 로드
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


# 학습할 얼굴 이미지 로드 및 얼굴 임베딩
def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return faces[0].normed_embedding


# 학습할 얼굴 이미지 경로 리스트
learning_image_paths = ['static/train/train_image.jpg']  # 예시로 단일 학습 이미지 경로를 설정

# 학습할 얼굴 임베딩들
learning_embeddings = []
for path in learning_image_paths:
    learning_embeddings.append(get_face_embedding(path).reshape(1, -1))
learning_embeddings = np.concatenate(learning_embeddings, axis=0)

# 비디오 파일 열기
video_path = 'static/input/input_video.mp4'
cap = cv2.VideoCapture(video_path)

# 비디오 프레임의 너비와 높이
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 저장을 위한 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
temp_video_path = 'static/output/output_video.mp4'

# 기존 파일 삭제
if os.path.exists(temp_video_path):
    os.remove(temp_video_path)
    print(f"Deleted existing file: {temp_video_path}")

out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

# 초기 임계값 설정
cosine_similarity_threshold = 0.4  # 임계값을 조정하여 정확도를 높이기

# tqdm을 사용하여 진행 상황 표시
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=frame_count, desc="Processing frames", ncols=100) as pbar:  # tqdm 사용 시 with 블록 사용
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb_frame)

        for face in faces:
            embedding = face.normed_embedding.reshape(1, -1)

            # 학습된 얼굴 임베딩과 비교하여 가장 높은 유사도 찾기
            similarities = cosine_similarity(embedding, learning_embeddings)
            max_similarity = similarities.max()

            # 얼굴 영역이 화면의 경계를 넘지 않도록 조정
            x_min, y_min, x_max, y_max = face.bbox.astype(int)
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, frame_width)
            y_max = min(y_max, frame_height)

            # 학습된 얼굴이 아닌 경우 모자이크 처리
            if max_similarity < cosine_similarity_threshold:
                roi = frame[y_min:y_max, x_min:x_max].copy()  # 복사본 생성
                roi_small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                roi_mosaic = cv2.resize(roi_small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

                # 원본 프레임에 모자이크 적용
                frame[y_min:y_max, x_min:x_max] = roi_mosaic

        # 모자이크 처리된 프레임을 저장
        out.write(frame)
        pbar.update(1)  # tqdm 업데이트

# 비디오 파일 처리 완료 후 객체 릴리스
cap.release()
out.release()


# 동영상과 오디오 결합
def combine_audio_video(original_video_path, temp_video_path, output_video_path):
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
        print(f"Deleted existing file: {output_video_path}")

    original_clip = VideoFileClip(original_video_path)
    temp_clip = VideoFileClip(temp_video_path)

    # 오디오 추출
    audio = original_clip.audio

    # 오디오를 모자이크 처리된 비디오에 결합
    final_clip = temp_clip.set_audio(audio)
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')


# 오디오가 결합된 비디오 저장
combine_audio_video(video_path, temp_video_path, 'static/output/output_video_with_audio.mp4')

print("Video processing completed.")
cv2.destroyAllWindows()
