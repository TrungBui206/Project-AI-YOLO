import cv2
import json
import numpy as np
import os
from ultralytics import YOLO
 
try:
    from ocsort import OCSort
except ImportError:
    print("Cảnh báo: Chưa tìm thấy module OC-SORT. Vui lòng đảm bảo đã setup thư viện tracking này.")
 
 
class TrafficCounter:
    def __init__(self, roi_config_path, yolo_model_path="best.pt", conf_threshold=0.3):
        """
        Khởi tạo hệ thống đếm xe: Load ROIs, mô hình YOLOv8 custom và bộ theo dõi OC-SORT.
        """
        # 1. Khởi tạo cấu hình ROI từ file JSON
        self.rois = self._load_rois(roi_config_path)
 
        # 2. Khởi tạo Model YOLOv8 với file weights đã train
        self.model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
 
        # CHỈ ĐỊNH CLASS CUSTOM: 0 = Car, 1 = Truck
        self.vehicle_classes = [0, 1]
 
        # 3. Khởi tạo OC-SORT Tracker
        self.tracker = OCSort(det_thresh=conf_threshold, max_age=30, min_hits=3, iou_threshold=0.3)
 
    def _load_rois(self, path):
        """Đọc tọa độ 9 vùng ROI từ file JSON và chuyển về định dạng numpy mảng của OpenCV."""
        with open(path, "r") as f:
            rois_raw = json.load(f)
 
        rois = {}
        for name, points in rois_raw.items():
            rois[name] = np.array(points, np.int32).reshape((-1, 1, 2))
        return rois
 
    def get_vehicle_counts(self, frame):
        """
        Xử lý 1 frame ảnh, chạy Detection + Tracking và trả về số lượng xe trong mỗi vùng ROI.
        """
        counts = {name: 0 for name in self.rois.keys()}
        vehicles_in_rois = {name: [] for name in self.rois.keys()}
 
        # 1. Chạy YOLOv8 Detection
        results = self.model.predict(frame, conf=self.conf_threshold,
                                     classes=self.vehicle_classes, verbose=False)
 
        dets = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                dets.append([box[0], box[1], box[2], box[3], conf, cls])

        import torch
        dets = torch.tensor(dets, dtype=torch.float32) if len(dets) > 0 else torch.empty((0, 6))

        # 2. Chạy OC-SORT Tracking
        if dets.shape[0] > 0:
            tracked_objects = self.tracker.update(dets, frame.shape)
        else:
            tracked_objects = np.empty((0, 5))
 
        # 3. Phân loại xe vào các ROI dựa trên điểm giữa cạnh dưới bounding box
        # OCSort.update() trả về [x1, y1, x2, y2, obj_id] — 5 cột
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj[:5]
 
            bottom_center_x = int((x1 + x2) / 2)
            bottom_center_y = int(y2)
            pt = (bottom_center_x, bottom_center_y)
 
            for roi_name, polygon in self.rois.items():
                if cv2.pointPolygonTest(polygon, pt, False) >= 0:
                    counts[roi_name] += 1
                    vehicles_in_rois[roi_name].append(int(obj_id))
                    break  # 1 xe chỉ thuộc 1 ROI
 
        return counts, tracked_objects, vehicles_in_rois
 
    def find_first_active_frame(self, cap):
        """
        Tua video từ đầu, trả về index của frame đầu tiên có xe xuất hiện.
        Dùng để bỏ qua phần intro/cảnh flycam bay lên chưa có xe.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=self.conf_threshold,
                                         classes=self.vehicle_classes, verbose=False)
            if len(results[0].boxes) > 0:
                print(f"Phát hiện xe đầu tiên tại frame: {frame_idx}")
                # Reset lại đúng vị trí frame này
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                return frame_idx
            frame_idx += 1
        return 0
 
 
# ==========================================
# KHỐI CHẠY THỬ (TEST BLOCK)
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "best.pt"
    ROI_PATH   = "roi_config.json"
    VIDEO_PATH = "original_video.mp4"
 
    # Kiểm tra file tồn tại
    for path in [MODEL_PATH, ROI_PATH, VIDEO_PATH]:
        if not os.path.exists(path):
            print(f"Lỗi: Không tìm thấy file '{path}'")
            exit()
 
    counter = TrafficCounter(roi_config_path=ROI_PATH, yolo_model_path=MODEL_PATH)
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video '{VIDEO_PATH}'")
        exit()
 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames | {fps:.1f} FPS")
 
    # Tìm frame đầu tiên có xe rồi chạy test 30 frame từ đó
    start_frame = counter.find_first_active_frame(cap)
 
    print(f"\nChạy test 30 frames bắt đầu từ frame {start_frame}:\n")
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        counts, tracks, details = counter.get_vehicle_counts(frame)
        print(f"Frame {start_frame + i + 1:4d} | Tracks: {len(tracks):3d} | "
              f"N_S:{counts['North_Straight']} N_L:{counts['North_Left']} "
              f"S_S:{counts['South_Straight']} S_L:{counts['South_Left']} "
              f"E_S:{counts['East_Straight']} E_L:{counts['East_Left']} "
              f"W_S:{counts['West_Straight']} W_L:{counts['West_Left']} "
              f"C:{counts['Center']}")
 
    cap.release()
