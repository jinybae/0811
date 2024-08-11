import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import sounddevice as sd
from scipy.io.wavfile import write
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
import threading
import os

# InsightFace 모델 로드
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 학습할 얼굴 이미지 로드 및 얼굴 임베딩
def get_face_embeddings(image_paths):
    embeddings = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue
        faces = app.get(img)
        if len(faces) == 0:
            print(f"No face detected in the image: {image_path}")
            continue
        for face in faces:
            embeddings.append(face.normed_embedding)
    if not embeddings:
        raise ValueError("No faces detected in the provided images.")
    return np.array(embeddings)

# 학습할 얼굴 이미지 경로 리스트
learning_image_paths = ['./static/train/train_image.jpg']  # 단일 학습 이미지 경로

# 학습할 얼굴 임베딩들
learning_embeddings = get_face_embeddings(learning_image_paths)

# 오디오 녹음 함수
def record_audio(sample_rate, output_file, stop_event, start_event):
    try:
        print("Recording audio...")
        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_data.extend(indata.copy())

            if stop_event.is_set():
                raise sd.CallbackStop

        start_event.set()  # 비디오 녹화 시작을 알림
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            stop_event.wait()  # 오디오 녹음을 중지할 때까지 대기

        write(output_file, sample_rate, np.array(audio_data))
        print("Recording finished.")
    except Exception as e:
        print("Error in recording audio:", e)

# 비디오와 오디오 결합 함수
def combine_audio_video(video_path, audio_path, output_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        # 비디오 클립의 속도를 조정하여 오디오 길이에 맞춤
        video_duration = video_clip.duration
        audio_duration = audio_clip.duration
        speed_factor = video_duration / audio_duration
        adjusted_video_clip = video_clip.fx(vfx.speedx, speed_factor)

        final_clip = adjusted_video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print("Video and audio have been successfully combined.")
    except Exception as e:
        print("Failed to combine video and audio:", e)
    finally:
        if 'video_clip' in locals():
            video_clip.close()
        if 'audio_clip' in locals():
            audio_clip.close()

# 출력 디렉토리 확인 및 생성
output_dir = 'static/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 웹캡 캡처 시작
cap = cv2.VideoCapture(0)

# 웹캠 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 6)  # FPS 6으로 설정

# 초기 임계값 설정
cosine_similarity_threshold = 0.5  # 임계값을 조정하여 정확도를 높이기

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out_path = os.path.join(output_dir, "output_webcam_video.mp4")
audio_out_path = os.path.join(output_dir, "output_webcam_audio.wav")
output_path = os.path.join(output_dir, "output_webcam_with_audio.mp4")
out = cv2.VideoWriter(video_out_path, fourcc, 6.0, (640, 480))  # FPS 6.0으로 설정

# 오디오 녹음 설정
stop_event = threading.Event()
start_event = threading.Event()
audio_thread = threading.Thread(target=record_audio, args=(44100, audio_out_path, stop_event, start_event))
audio_thread.start()

# 오디오 녹음 시작 대기
start_event.wait()
print("Recording video...")

# 프레임 스킵 설정 및 얼굴 위치 저장 변수
frame_skip = 5  # 5프레임마다 1프레임 처리
frame_count = 0
last_faces = []  # 마지막으로 감지된 얼굴 위치

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count > 1 and frame_count % frame_skip != 0:
        # 이전에 감지된 얼굴 위치에 모자이크 적용
        for (x_min, y_min, x_max, y_max) in last_faces:
            roi = frame[y_min:y_max, x_min:x_max]
            roi_small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
            roi_mosaic = cv2.resize(roi_small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
            frame[y_min:y_max, x_min:x_max] = roi_mosaic

        out.write(frame)  # 프레임 저장
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_frame)

    last_faces = []
    for face in faces:
        embedding = face.normed_embedding.reshape(1, -1)  # 2차원 배열로 변환

        # 학습된 얼굴 임베딩과 비교하여 가장 높은 유사도 찾기
        similarities = cosine_similarity(embedding, learning_embeddings)
        max_similarity = similarities.max()

        # 얼굴 영역이 화면의 경계를 넘지 않도록 조정
        x_min, y_min, x_max, y_max = face.bbox.astype(int)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, frame.shape[1])
        y_max = min(y_max, frame.shape[0])

        # 학습된 얼굴이 아닌 경우 모자이크 처리
        if max_similarity < cosine_similarity_threshold:
            roi = frame[y_min:y_max, x_min:x_max]  # 복사하지 않음
            roi_small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
            roi_mosaic = cv2.resize(roi_small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

            # 원본 프레임에 모자이크 적용
            frame[y_min:y_max, x_min:x_max] = roi_mosaic

            # 마지막 얼굴 위치 저장
            last_faces.append((x_min, y_min, x_max, y_max))
        else:
            print(f"Detected known face with similarity: {max_similarity}")

    # 결과 프레임 출력 및 저장
    out.write(frame)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 릴리스 및 윈도우 종료
cap.release()
out.release()
cv2.destroyAllWindows()

# 오디오 녹음 종료
stop_event.set()
audio_thread.join()

# 비디오와 오디오 결합
combine_audio_video(video_out_path, audio_out_path, output_path)
