import cv2
import os

video_path = "original_video.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps * (240 / 240))  

# 4 phút = 240s, muốn ~240 frames → lấy mỗi 1 giây
interval = int(fps * 1)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        filename = f"{output_folder}/frame_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"FPS: {fps} | Interval: {interval} frames | Saved: {saved_count} frames")
