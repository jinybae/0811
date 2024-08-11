import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os

# InsightFace 모델 로드
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 학습할 얼굴 이미지 로드 및 얼굴 임베딩
def get_face_embedding(image_path):
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return faces[0].normed_embedding

def process_images(learning_image_path, comparison_image_path, output_image_path):
    try:
        # 학습할 얼굴 임베딩
        learning_embedding = get_face_embedding(learning_image_path).reshape(1, -1)

        # 비교할 이미지 로드
        comparison_image = cv2.imread(comparison_image_path)
        if comparison_image is None:
            raise ValueError(f"Error loading image: {comparison_image_path}")
        rgb_comparison_image = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB)
        img_height, img_width = comparison_image.shape[:2]

        # 비교할 이미지에서 얼굴 인식 및 모자이크 처리
        faces = app.get(rgb_comparison_image)
        print(f"Detected {len(faces)} faces")

        for face in faces:
            x_min, y_min, x_max, y_max = face.bbox.astype(int)
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, img_width)
            y_max = min(y_max, img_height)

            # 얼굴 임베딩 계산
            face_embedding = face.normed_embedding.reshape(1, -1)

            # 코사인 유사도로 얼굴 비교
            similarity = cosine_similarity(learning_embedding, face_embedding)[0][0]
            print(f"Similarity: {similarity}")

            # 학습된 얼굴과 다르면 모자이크 처리
            if similarity < 0.4:  # 임계값 조정
                roi = comparison_image[y_min:y_max, x_min:x_max]
                roi_small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                roi_mosaic = cv2.resize(roi_small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                comparison_image[y_min:y_max, x_min:x_max] = roi_mosaic

        # 기존 파일이 존재하면 삭제
        if os.path.exists(output_image_path):
            os.remove(output_image_path)

        # 결과 이미지 저장
        cv2.imwrite(output_image_path, comparison_image)
        print(f"Output image saved to: {output_image_path}")

    except Exception as e:
        print(f"Error: {e}")
        raise

