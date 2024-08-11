from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Response
import os
import base64
import cv2
import threading
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io.wavfile import write
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
import sounddevice as sd
import image  # Assuming image.py contains the process_images function

app = Flask(__name__)

# 파일 업로드를 위한 디렉토리 설정
TRAIN_FOLDER = './static/train/'
INPUT_FOLDER = './static/input/'
OUTPUT_FOLDER = './static/output/'
app.config['TRAIN_FOLDER'] = TRAIN_FOLDER
app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# InsightFace 모델 로드
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 학습할 얼굴 이미지 로드 및 얼굴 임베딩
def get_face_embeddings(image_paths):
    embeddings = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue
        faces = face_app.get(img)
        if len(faces) == 0:
            print(f"No face detected in the image: {image_path}")
            continue
        for face in faces:
            embeddings.append(face.normed_embedding)
    if not embeddings:
        raise ValueError("No faces detected in the provided images.")
    return np.array(embeddings)

# 학습할 얼굴 이미지 경로 리스트
learning_image_paths = [os.path.join(app.config['TRAIN_FOLDER'], 'train_image.jpg')]
learning_embeddings = get_face_embeddings(learning_image_paths)

# 이미지 변환 라우트
@app.route('/convert_image', methods=['POST'])
def convert_image():
    image_data = request.files['image']
    image_path = os.path.join(app.config['INPUT_FOLDER'], 'input_image.jpg')
    image_data.save(image_path)

    try:
        learning_image_path = os.path.join(app.config['TRAIN_FOLDER'], 'train_image.jpg')
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_image.jpg')
        image.process_images(learning_image_path, image_path, output_image_path)
    except ValueError as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Face not detected. Mosaic process failed.'}), 500
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': f"An error occurred while processing the image: {e}"}), 500

    return redirect(url_for('gallery_convert'))

@app.route('/save_camera_image', methods=['POST'])
def save_camera_image():
    try:
        image_data = request.form['image']
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = base64.b64decode(image_data)

        image_dir = app.config['INPUT_FOLDER']
        image_path = os.path.join(image_dir, 'camera_image.jpg')

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        with open(image_path, 'wb') as f:
            f.write(image_data)

        try:
            learning_image_path = os.path.join(app.config['TRAIN_FOLDER'], 'train_image.jpg')
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_image.jpg')
            image.process_images(learning_image_path, image_path, output_image_path)
        except ValueError:
            return jsonify({'error': 'Face not detected. Mosaic process failed.'})
        except Exception as e:
            print(f"Error occurred: {e}")
            return jsonify({'error': f"An error occurred while processing the image: {e}"}), 500

        return jsonify({'redirect_url': url_for('camera_convert')})

    except Exception as e:
        print(f"Error occurred in save_camera_image: {e}")
        return jsonify({'error': f"An error occurred while saving the image: {e}"}), 500

# 웹캠 비디오 스트림 생성기
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 6)
    cosine_similarity_threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_frame)

        for face in faces:
            embedding = face.normed_embedding.reshape(1, -1)
            similarities = cosine_similarity(embedding, learning_embeddings)
            max_similarity = similarities.max()
            x_min, y_min, x_max, y_max = face.bbox.astype(int)
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, frame.shape[1])
            y_max = min(y_max, frame.shape[0])

            if max_similarity < cosine_similarity_threshold:
                roi = frame[y_min:y_max, x_min:x_max]
                roi_small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                roi_mosaic = cv2.resize(roi_small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                frame[y_min:y_max, x_min:x_max] = roi_mosaic
            else:
                print(f"Detected known face with similarity: {max_similarity}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/train_camera')
def train_camera():
    return render_template('train_camera.html')

@app.route('/train_gallery')
def train_gallery():
    return render_template('train_gallery.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/gallery_convert')
def gallery_convert():
    return render_template('gallery_convert.html')

@app.route('/camera_convert')
def camera_convert():
    return render_template('camera_convert.html')

# 새로운 save_image 라우트 추가
@app.route('/save_image', methods=['POST'])
def save_image():
    image_data = request.form['image']
    image_data = image_data.replace('data:image/png;base64,', '')
    image_data = base64.b64decode(image_data)

    # 이미지 저장 경로 설정
    image_dir = app.config['TRAIN_FOLDER']
    image_path = os.path.join(image_dir, 'train_image.jpg')

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 이미지 저장
    with open(image_path, 'wb') as f:
        f.write(image_data)

    return redirect(url_for('index'))

# 새로운 save_exclusion_image 라우트 추가
@app.route('/save_exclusion_image', methods=['POST'])
def save_exclusion_image():
    if 'gallery_photo' not in request.files:
        return jsonify({'error': 'No photo part in the request'}), 400

    photo = request.files['gallery_photo']
    if photo.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = 'train_image.jpg'
    save_path = os.path.join(app.config['TRAIN_FOLDER'], filename)
    app.logger.info(f'Saving photo to {save_path}')
    try:
        photo.save(save_path)
        app.logger.info('Photo successfully saved')
        return jsonify({'message': 'File successfully uploaded', 'redirect_url': url_for('index')}), 200
    except Exception as e:
        app.logger.error(f'Error saving photo: {e}')
        return jsonify({'error': 'Failed to save file'}), 500

# 동영상 녹화 시작 라우트 추가
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global stop_event, start_event, audio_thread, cap, out

    output_dir = app.config['OUTPUT_FOLDER']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out_path = os.path.join(output_dir, "output_webcam_video.mp4")
    audio_out_path = os.path.join(output_dir, "output_webcam_audio.wav")
    output_path = os.path.join(output_dir, "output_webcam_with_audio.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 6)
    out = cv2.VideoWriter(video_out_path, fourcc, 6.0, (640, 480))

    stop_event = threading.Event()
    start_event = threading.Event()
    audio_thread = threading.Thread(target=record_audio, args=(44100, audio_out_path, stop_event, start_event))
    audio_thread.start()
    start_event.wait()

    return jsonify({'message': 'Recording started'})

# 동영상 녹화 중지 라우트 추가
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global cap, out, stop_event, audio_thread

    if cap.isOpened():
        cap.release()
    if out is not None:
        out.release()
    stop_event.set()
    audio_thread.join()

    output_dir = app.config['OUTPUT_FOLDER']
    video_out_path = os.path.join(output_dir, "output_webcam_video.mp4")
    audio_out_path = os.path.join(output_dir, "output_webcam_audio.wav")
    output_path = os.path.join(output_dir, "output_webcam_with_audio.mp4")

    combine_audio_video(video_out_path, audio_out_path, output_path)

    return jsonify({'message': 'Recording stopped', 'redirect_url': url_for('camera_webcam_show')})

@app.route('/output/<filename>')
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/camera_webcam')
def camera_webcam():
    return render_template('camera_webcam.html')

@app.route('/camera_webcam_show')
def camera_webcam_show():
    return render_template('camera_webcam_show.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

        start_event.set()
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            stop_event.wait()

        write(output_file, sample_rate, np.array(audio_data))
        print("Recording finished.")
    except Exception as e:
        print("Error in recording audio:", e)

# 비디오와 오디오 결합 함수
def combine_audio_video(video_path, audio_path, output_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

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

if __name__ == '__main__':
    stop_event = None
    start_event = None
    audio_thread = None
    cap = None
    out = None
    app.run(debug=True, host='0.0.0.0', port=5001)
