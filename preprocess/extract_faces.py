import os
import cv2
from mtcnn import MTCNN
from pathlib import Path

def extract_faces_from_video(video_path, output_dir, margin=20):
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = detector.detect_faces(frame)
        for i, res in enumerate(results):
            x, y, w, h = res['box']
            x, y = max(0, x - margin), max(0, y - margin)
            face = frame[y:y+h+margin, x:x+w+margin]
            face_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_{i}.jpg")
            cv2.imwrite(face_path, face)
    cap.release()

if __name__ == "__main__":
    video = r"C:\deepcheck\original_sequences\youtube\c23\videos\sample.mp4"
    out = r"C:\deepcheck\faces\sample"
    extract_faces_from_video(video, out)
