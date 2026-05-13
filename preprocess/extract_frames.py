import cv2
import os

# Paths
video_base = r"C:\deepcheck\dataset_split"
frame_base = r"C:\deepcheck\frames"

splits = ["train", "val", "test"]
categories = ["real", "fake"]
max_frames_long = 50  # max frames for long videos

# Smart frame extraction
for split in splits:
    for category in categories:
        video_path = os.path.join(video_base, split, category)
        frame_path = os.path.join(frame_base, split, category)
        os.makedirs(frame_path, exist_ok=True)

        for video_file in os.listdir(video_path):
            if not video_file.endswith(".mp4"):
                continue

            video_full_path = os.path.join(video_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_frame_folder = os.path.join(frame_path, video_name)
            os.makedirs(video_frame_folder, exist_ok=True)

            cap = cv2.VideoCapture(video_full_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration_sec = total_frames / video_fps

            # Determine FPS to extract based on video length
            if duration_sec < 10:
                extract_fps = video_fps  # all frames
            elif duration_sec < 30:
                extract_fps = 2  # 2 frames per second
            else:
                extract_fps = 1  # 1 frame per second for long videos

            # Interval between frames to save
            frame_interval = max(int(video_fps / extract_fps), 1)

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame at interval
                if frame_count % frame_interval == 0:
                    if duration_sec >= 30 and saved_count >= max_frames_long:
                        break  # limit frames for long videos
                    frame_file = os.path.join(video_frame_folder, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_file, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"{video_file}: duration {duration_sec:.1f}s, saved {saved_count} frames")

print("Smart frame extraction complete!")
