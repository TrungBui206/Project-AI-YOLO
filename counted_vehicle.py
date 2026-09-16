import cv2
import csv
from ultralytics import YOLO

# Load model đã train
model = YOLO("runs/detect/train/weights/best.pt")

# Mở video
video_path = "original_video.mp4"
cap = cv2.VideoCapture(video_path)

# Setup ghi video output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_counted.mp4", fourcc, fps, (w, h))

# Tạo file CSV
csv_file = open("vehicle_counts.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "vehicle_count", "timestamp_sec"])

frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Detect
    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes

    # Đếm số xe trong frame này
    vehicle_count = len(boxes)

    # Vẽ bounding box
    annotated = results[0].plot()

    # Hiển thị số xe lên frame
    cv2.rectangle(annotated, (10, 10), (280, 60), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"Vehicles: {vehicle_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    # Ghi CSV
    timestamp = frame_num / fps
    writer.writerow([frame_num, vehicle_count, round(timestamp, 2)])

    print(f"Frame {frame_num}: {vehicle_count} vehicles")

    out.write(annotated)

cap.release()
out.release()
csv_file.close()
print("Done! Saved to output_counted.mp4 and vehicle_counts.csv")