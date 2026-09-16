import cv2
import json
import numpy as np

from traffic_counter import TrafficCounter
from phase_controller import PhaseController


# ======================================================================
# CẤU HÌNH MÀU SẮC TRỰC QUAN
# ======================================================================
COLOR = {
    'GREEN':     (0,   255,  0),   # Xanh lá - Làn được đi
    'YELLOW':    (0,   220, 255),  # Vàng - Làn chuẩn bị dừng
    'RED':       (0,   0,   255),  # Đỏ - Làn dừng lại
    'EMERGENCY': (0,   0,   180),  # Đỏ sẫm khẩn cấp
    'WHITE':     (255, 255, 255),
    'BLACK':     (0,   0,   0),
    'GRAY':      (80,  80,  80),
    'BG_PANEL':  (20,  20,  20),
}


# ======================================================================
# VISUALIZER
# ======================================================================
class Visualizer:
    def __init__(
        self,
        video_path,
        roi_config_path,
        model_path,
        output_path="output_video.mp4",
        fps=30,
        conf_threshold=0.3,
        gridlock_threshold=8,
        roi_alpha=0.25,         # Độ trong suốt fill ROI
        show_roi_labels=True,   # Hiện tên ROI
        show_bbox=True,         # Hiện bounding box xe
        panel_width=380,        # Độ rộng panel thông tin bên phải
    ):
        self.video_path        = video_path
        self.output_path       = output_path
        self.fps               = fps
        self.roi_alpha         = roi_alpha
        self.show_roi_labels   = show_roi_labels
        self.show_bbox         = show_bbox
        self.panel_width       = panel_width

        # Load ROI config
        with open(roi_config_path, 'r', encoding='utf-8') as f:
            rois_raw = json.load(f)
        self.rois = {
            name: np.array(pts, np.int32).reshape((-1, 1, 2))
            for name, pts in rois_raw.items()
        }

        # Khởi tạo các module
        self.counter = TrafficCounter(
            roi_config_path=roi_config_path,
            yolo_model_path=model_path,
            conf_threshold=conf_threshold,
        )
        self.adaptive = PhaseController(
            fps=fps,
            min_green_straight=15,
            min_green_left=5,
            max_green=60,
            yellow_time=3,
            sec_per_veh=2.5,
            gridlock_threshold=gridlock_threshold,
        )

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video '{self.video_path}'")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video: thêm panel bên phải
        out_w = w + self.panel_width
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (out_w, h))

        print(f"Video: {total_frames} frames | {w}x{h} | {self.fps} FPS")
        print(f"Output: {self.output_path} ({out_w}x{h})")
        print("Đang render... (có thể mất vài phút)")

        # Tìm frame đầu tiên có xe và đồng bộ con trỏ video độc lập
        start_frame = self.counter.find_first_active_frame(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 
        print(f"Bắt đầu kết xuất từ frame: {start_frame}")

        frame_idx = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Đếm xe từ Traffic Counter
            counts, tracks, _ = self.counter.get_vehicle_counts(frame)

            # 2. Cập nhật pha đèn từ bộ điều phối Adaptive mẫu
            status = self.adaptive.update(counts)
            lights = status['lights']

            # 3. Vẽ overlay trạng thái động lên frame video gốc
            vis_frame = self._draw_frame(frame, counts, tracks, lights, status, frame_idx)

            # 4. Ghép panel thông tin trực quan bên phải
            panel = self._draw_panel(counts, status, h)
            output_frame = np.hstack([vis_frame, panel])

            writer.write(output_frame)

            frame_idx += 1
            if (frame_idx - start_frame) % 300 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Tiến độ: {progress:.1f}% | Frame {frame_idx}/{total_frames}")

        cap.release()
        writer.release()
        print(f"\nHoàn tất! Output đã lưu tại: {self.output_path}")

    # ------------------------------------------------------------------
    # Vẽ overlay lên frame chính
    # ------------------------------------------------------------------
    def _draw_frame(self, frame, counts, tracks, lights, status, frame_idx):
        vis = frame.copy()

        # --- Vẽ ROI fill (Đổi màu động theo trạng thái đèn thực tế để tránh xung đột trực quan) ---
        overlay = vis.copy()
        for roi_name, polygon in self.rois.items():
            if roi_name == 'Center':
                continue
            
            light = lights.get(roi_name, 'RED')
            
            if status.get('current_state') == 'EMERGENCY' or status.get('emergency_clearance', False):
                fill_color = COLOR['EMERGENCY']
            elif light == 'GREEN':
                fill_color = COLOR['GREEN']
            elif light == 'YELLOW':
                fill_color = COLOR['YELLOW']
            else:
                fill_color = COLOR['RED']  # Đèn ĐỎ tô đỏ mờ đồng bộ cấu trúc thực tế nút giao

            cv2.fillPoly(overlay, [polygon], fill_color)

        cv2.addWeighted(overlay, self.roi_alpha, vis, 1 - self.roi_alpha, 0, vis)

        # --- Vẽ viền đường biên ROI ---
        for roi_name, polygon in self.rois.items():
            if roi_name == 'Center':
                continue
            
            light = lights.get(roi_name, 'RED')
            
            if light == 'GREEN':
                border_color = COLOR['GREEN']
                thickness = 3
            elif light == 'YELLOW':
                border_color = COLOR['YELLOW']
                thickness = 3
            else:
                border_color = (0, 0, 150)  # Đường viền đỏ sẫm cho làn đang đỏ
                thickness = 1

            cv2.polylines(vis, [polygon], isClosed=True, color=border_color, thickness=thickness)

            # Label tên ROI + số xe hiện tại trong vùng tách biệt
            if self.show_roi_labels:
                M = polygon.reshape(-1, 2)
                cx, cy = int(M[:, 0].mean()), int(M[:, 1].mean())
                n = counts.get(roi_name, 0)
                short_display_name = roi_name.replace('_Straight', ' Thẳng').replace('_Left', ' Trái')
                label = f"{short_display_name}: {n} xe"
                self._draw_text_with_bg(vis, label, (cx - 60, cy), scale=0.4)

        # --- Vẽ bounding box xe của mô hình định danh ---
        if self.show_bbox:
            for obj in tracks:
                x1, y1, x2, y2, obj_id = [int(v) for v in obj[:5]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(vis, f"ID: {int(obj_id)}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # --- Header hiển thị động trạng thái góc trên bên trái ---
        emergency = status.get('emergency_clearance', False)
        header_color = COLOR['EMERGENCY'] if emergency else COLOR['WHITE']
        header_text  = "!!! EMERGENCY CLEARANCE !!!" if emergency else \
                       f"Pha hien tai: {status['current_phase']} [{status['current_state']}] | Con lai: {status['time_left']}s"
        
        self._draw_text_with_bg(vis, header_text, (20, 35), scale=0.65, color=header_color, thickness=2)
        self._draw_text_with_bg(vis, f"Video Frame Index: {frame_idx}", (20, 65), scale=0.45)

        return vis

    # ------------------------------------------------------------------
    # Panel thông tin chi tiết bên phải
    # ------------------------------------------------------------------
    def _draw_panel(self, counts, status, h):
        panel = np.full((h, self.panel_width, 3), 25, dtype=np.uint8)

        lights    = status['lights']
        phase     = status['current_phase']
        state     = status['current_state']
        time_left = status['time_left']
        emergency = status.get('emergency_clearance', False)

        y = 30
        # Tiêu đề bảng điều khiển trung tâm
        cv2.putText(panel, "ADAPTIVE CONTROLLER PANEL", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR['WHITE'], 2)
        y += 30
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (80, 80, 80), 1)
        y += 20

        # Trạng thái pha hiện hành
        if emergency:
            state_color = COLOR['EMERGENCY']
            state_text  = "HỆ THỐNG KHẨN CẤP"
        elif state == 'GREEN':
            state_color = COLOR['GREEN']
            state_text  = f"PHA {phase} - ĐÈN XANH"
        elif state == 'YELLOW':
            state_color = COLOR['YELLOW']
            state_text  = f"PHA {phase} - ĐÈN VÀNG"
        else:
            state_color = COLOR['RED']
            state_text  = f"PHA {phase} - ĐÈN ĐỎ"

        cv2.putText(panel, state_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
        y += 25
        cv2.putText(panel, f"Thoi gian con lai: {time_left}s", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 1)
        y += 35

        cv2.line(panel, (10, y), (self.panel_width - 10, y), (80, 80, 80), 1)
        y += 20

        # Trạng thái đèn tín hiệu chi tiết từng nhánh ROI
        cv2.putText(panel, "TÍN HIỆU ĐÈN CHI TIẾT LÀN ĐƯỜNG:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += 22

        roi_display_order = [
            'North_Straight', 'North_Left',
            'South_Straight', 'South_Left',
            'East_Straight',  'East_Left',
            'West_Straight',  'West_Left',
        ]

        for roi in roi_display_order:
            light = lights.get(roi, 'RED')
            n     = counts.get(roi, 0)

            if light == 'GREEN':
                dot_color = COLOR['GREEN']
            elif light == 'YELLOW':
                dot_color = COLOR['YELLOW']
            else:
                dot_color = COLOR['RED']

            # Chấm tròn hiển thị màu đèn LED đồng bộ
            cv2.circle(panel, (18, y - 5), 6, dot_color, -1)

            short_name = roi.replace('_Straight', '_Thẳng').replace('_Left', '_Trái')
            label = f"{short_name:<15} {n:>2} xe"
            cv2.putText(panel, label, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR['WHITE'], 1)
            y += 22

        y += 10
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (80, 80, 80), 1)
        y += 20

        # Thống kê tổng lượng xe nền giao lộ
        total_vehicles = sum(counts.get(r, 0) for r in roi_display_order)
        center_count   = counts.get('Center', 0)
        cv2.putText(panel, f"Tong xe cho tai vach: {total_vehicles} xe", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR['WHITE'], 1)
        y += 22
        cv2.putText(panel, f"Xe dang trong giao lo:  {center_count} xe", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR['WHITE'], 1)
        y += 25

        # Cảnh báo xung đột Gridlock khu trung tâm hình học
        if status.get('gridlock_alert', False):
            cv2.putText(panel, "!!! CANH BAO UN TAC TRUNG TAM !!!", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['EMERGENCY'], 2)
        y += 30

        cv2.line(panel, (10, y), (self.panel_width - 10, y), (80, 80, 80), 1)
        y += 20

        # Bản đồ chú thích thứ tự chu kỳ pha của mô hình đề xuất
        cv2.putText(panel, "CHU KY PHA DIEU KHIEN SẴN CO:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += 20
        phase_desc = [
            "Pha 1: Bac - Nam Di Thang",
            "Pha 2: Bac Re Trai + Thang",
            "Pha 3: Dong - Tay Re Trai",
            "Pha 4: Dong - Tay Di Thang",
            "Pha 5: Nam Re Trai + Thang",
        ]
        for i, desc in enumerate(phase_desc):
            color = COLOR['GREEN'] if (i + 1) == phase and not emergency else (120, 120, 120)
            cv2.putText(panel, desc, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 18

        return panel

    # ------------------------------------------------------------------
    # Helper hỗ trợ vẽ Text sắc nét có kèm nền tối scannable
    # ------------------------------------------------------------------
    def _draw_text_with_bg(self, img, text, pos, scale=0.6, color=COLOR['WHITE'], thickness=1, padding=4):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = pos
        cv2.rectangle(img,
                      (x - padding, y - th - padding),
                      (x + tw + padding, y + padding),
                      COLOR['BG_PANEL'], -1)
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)


# ======================================================================
# KHỞI CHẠY THỰC THI 
# ======================================================================
if __name__ == "__main__":
    viz = Visualizer(
        video_path      = "original_video.mp4",
        roi_config_path = "roi_config.json",
        model_path      = "best.pt",
        output_path     = "adaptive_simulation_output.mp4",
        fps             = 30,
        conf_threshold  = 0.3,
        gridlock_threshold = 8,
        roi_alpha       = 0.25,
        show_roi_labels = True,
        show_bbox       = True,
        panel_width     = 380,
    )
    viz.run()
