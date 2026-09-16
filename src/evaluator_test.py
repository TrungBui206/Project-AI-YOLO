import cv2
import csv

from traffic_counter import TrafficCounter
from phase_controller_test import PhaseController


# ======================================================================
# FIXED-TIME CONTROLLER
# ======================================================================
class FixedTimeController:
    """
    Bộ điều phối đèn cố định truyền thống.
    Mỗi pha có thời gian xanh cố định, không phụ thuộc mật độ xe.
    """

    PHASE_SEQUENCE = [1, 2, 3, 4, 5]

    FIXED_GREEN = {
        1: 30,  # Bắc-Nam thẳng
        2: 15,  # Bắc trái + Bắc thẳng
        3: 15,  # Đông-Tây trái
        4: 30,  # Đông-Tây thẳng
        5: 15,  # Nam trái + Nam thẳng
    }

    FIXED_YELLOW = 3

    def __init__(self, fps=30):
        self.fps           = fps
        self.phase_index   = 0
        self.current_phase = 1
        self.current_state = "GREEN"
        self.frame_counter = 0
        self.current_allocated_time = self.FIXED_GREEN[1]

    def update(self):
        self.frame_counter += 1
        elapsed = self.frame_counter / self.fps

        if elapsed >= self.current_allocated_time:
            self._transition()

        time_left = max(0.0, self.current_allocated_time - elapsed)
        return self._build_status(time_left)

    def _transition(self):
        self.frame_counter = 0
        if self.current_state == "GREEN":
            self.current_state          = "YELLOW"
            self.current_allocated_time = self.FIXED_YELLOW
        else:
            self.current_state  = "GREEN"
            self.phase_index    = (self.phase_index + 1) % len(self.PHASE_SEQUENCE)
            self.current_phase  = self.PHASE_SEQUENCE[self.phase_index]
            self.current_allocated_time = self.FIXED_GREEN[self.current_phase]

    def _build_status(self, time_left):
        lights = {
            'North_Straight': 'RED', 'North_Left': 'RED',
            'South_Straight': 'RED', 'South_Left': 'RED',
            'East_Straight':  'RED', 'East_Left':  'RED',
            'West_Straight':  'RED', 'West_Left':  'RED',
            'Center':         'N/A',
        }

        state = self.current_state

        if self.current_phase == 1:
            lights['North_Straight'] = state
            lights['South_Straight'] = state
        elif self.current_phase == 2:
            lights['North_Left']     = state
            lights['North_Straight'] = state
        elif self.current_phase == 3:
            lights['East_Left']  = state
            lights['West_Left']  = state
        elif self.current_phase == 4:
            lights['East_Straight'] = state
            lights['West_Straight'] = state
        elif self.current_phase == 5:
            lights['South_Left']     = state
            lights['South_Straight'] = state

        return {
            'current_phase': self.current_phase,
            'current_state': self.current_state,
            'time_left':     round(time_left, 1),
            'lights':        lights,
        }


# ======================================================================
# METRICS TRACKER
# ======================================================================
class MetricsTracker:
    """
    Theo dõi và tính toán các chỉ số đánh giá hiệu năng:
        1. Waiting Load          — Tải trọng chờ tích lũy (xe-giây) khi đèn đỏ
        2. Average Queue Length  — Độ dài hàng đợi trung bình (xe dừng đỏ)
        3. Estimated Throughput  — Số xe ước lượng đã được giải phóng khỏi ROI
        4. Congestion Rate       — Tỷ lệ ùn tắc (Center > threshold)
    """

    ROI_NAMES = [
        'North_Straight', 'North_Left',
        'South_Straight', 'South_Left',
        'East_Straight',  'East_Left',
        'West_Straight',  'West_Left',
    ]

    def __init__(self, fps=30, gridlock_threshold=8):
        self.fps                = fps
        self.gridlock_threshold = gridlock_threshold

        self.total_wait_load      = 0.0
        self.total_queue_sum      = 0.0
        self.frame_count          = 0
        self.estimated_throughput = 0
        self.congestion_frames    = 0
        self.prev_counts          = {}

    def update(self, counts, lights):
        self.frame_count += 1
        dt = 1.0 / self.fps
        queue_this_frame = 0

        for roi in self.ROI_NAMES:
            current_count = counts.get(roi, 0)
            light         = lights.get(roi, 'RED')

            # Waiting Load + Queue khi đèn ĐỎ
            if light == 'RED':
                self.total_wait_load += current_count * dt
                queue_this_frame     += current_count

            # Estimated Throughput: xe giảm trong ROI khi đèn XANH/VÀNG
            prev_count = self.prev_counts.get(roi, current_count)
            if light in ('GREEN', 'YELLOW'):
                served = max(prev_count - current_count, 0)
                self.estimated_throughput += served

            self.prev_counts[roi] = current_count

        self.total_queue_sum += queue_this_frame

        if counts.get('Center', 0) > self.gridlock_threshold:
            self.congestion_frames += 1

    def get_summary(self):
        avg_queue = (self.total_queue_sum / self.frame_count
                     if self.frame_count else 0)

        congestion_rate = (self.congestion_frames / self.frame_count * 100
                           if self.frame_count else 0)

        return {
            'waiting_load':         round(self.total_wait_load,       2),
            'avg_queue_length':     round(avg_queue,                   2),
            'estimated_throughput': int(self.estimated_throughput),
            'congestion_rate_pct':  round(congestion_rate,             2),
        }


# ======================================================================
# EVALUATOR
# ======================================================================
class Evaluator:
    def __init__(
        self,
        video_path,
        roi_config_path,
        model_path,
        fps=30,
        conf_threshold=0.3,
        gridlock_threshold=8,
        output_csv_counts="vehicle_counts.csv",
        output_csv_results="signal_results.csv",
    ):
        self.video_path         = video_path
        self.fps                = fps
        self.output_csv_counts  = output_csv_counts
        self.output_csv_results = output_csv_results

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
        self.fixed = FixedTimeController(fps=fps)

        self.metrics_adaptive = MetricsTracker(fps=fps, gridlock_threshold=gridlock_threshold)
        self.metrics_fixed    = MetricsTracker(fps=fps, gridlock_threshold=gridlock_threshold)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Loi: Khong the mo video '{self.video_path}'")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {total_frames} frames | {self.fps} FPS")
        print("Dang xu ly phan tich...")

        start_frame = self.counter.find_first_active_frame(cap)
        print(f"Bat dau xu ly tinh toan tu frame: {start_frame}")

        counts_rows  = []
        results_rows = []
        frame_idx    = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            counts, tracks, _ = self.counter.get_vehicle_counts(frame)

            status_adaptive = self.adaptive.update(counts)
            status_fixed    = self.fixed.update()

            self.metrics_adaptive.update(counts, status_adaptive['lights'])
            self.metrics_fixed.update(counts,    status_fixed['lights'])

            counts_rows.append({
                'frame':          frame_idx,
                'North_Straight': counts.get('North_Straight', 0),
                'North_Left':     counts.get('North_Left',     0),
                'South_Straight': counts.get('South_Straight', 0),
                'South_Left':     counts.get('South_Left',     0),
                'East_Straight':  counts.get('East_Straight',  0),
                'East_Left':      counts.get('East_Left',      0),
                'West_Straight':  counts.get('West_Straight',  0),
                'West_Left':      counts.get('West_Left',      0),
                'Center':         counts.get('Center',         0),
            })

            results_rows.append({
                'frame':              frame_idx,
                'adaptive_phase':     status_adaptive['current_phase'],
                'adaptive_state':     status_adaptive['current_state'],
                'adaptive_time_left': status_adaptive['time_left'],
                'fixed_phase':        status_fixed['current_phase'],
                'fixed_state':        status_fixed['current_state'],
                'fixed_time_left':    status_fixed['time_left'],
            })

            frame_idx += 1

            if (frame_idx - start_frame) % 300 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Tien do: {progress:.1f}% | Frame {frame_idx}/{total_frames}")

        cap.release()

        self._save_csv(self.output_csv_counts,  counts_rows)
        self._save_csv(self.output_csv_results, results_rows)
        self._print_comparison()

    def _save_csv(self, path, rows):
        if not rows:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Da luu thanh cong: {path}")

    def _print_comparison(self):
        a = self.metrics_adaptive.get_summary()
        f = self.metrics_fixed.get_summary()

        def improvement(adaptive_val, fixed_val, lower_is_better=True):
            if fixed_val == 0:
                return 0.0
            delta = (fixed_val - adaptive_val) / fixed_val * 100
            return round(delta if lower_is_better else -delta, 1)

        wait_imp       = improvement(a['waiting_load'],         f['waiting_load'])
        queue_imp      = improvement(a['avg_queue_length'],     f['avg_queue_length'])
        throughput_imp = improvement(a['estimated_throughput'], f['estimated_throughput'], lower_is_better=False)
        congestion_imp = improvement(a['congestion_rate_pct'],  f['congestion_rate_pct'])

        print("\n" + "=" * 65)
        print("  KET QUA THUC NGHIEM: ADAPTIVE vs FIXED-TIME")
        print("=" * 65)
        print(f"{'Chi so danh gia':<30} {'Fixed-Time':>12} {'Adaptive':>12} {'Cai thien':>10}")
        print("-" * 65)
        print(f"{'Tai trong cho (xe-giay)':<30} "
              f"{f['waiting_load']:>12.2f} "
              f"{a['waiting_load']:>12.2f} "
              f"{wait_imp:>+9.1f}%")
        print(f"{'Hang doi trung binh (xe)':<30} "
              f"{f['avg_queue_length']:>12.2f} "
              f"{a['avg_queue_length']:>12.2f} "
              f"{queue_imp:>+9.1f}%")
        print(f"{'Luu luong uoc luong (xe)':<30} "
              f"{f['estimated_throughput']:>12d} "
              f"{a['estimated_throughput']:>12d} "
              f"{throughput_imp:>+9.1f}%")
        print(f"{'Ty le un tac trung tam (%)':<30} "
              f"{f['congestion_rate_pct']:>12.2f} "
              f"{a['congestion_rate_pct']:>12.2f} "
              f"{congestion_imp:>+9.1f}%")
        print("=" * 65)

        summary = [
            {'metric': 'Tai trong cho (xe-giay)',    'fixed': f['waiting_load'],         'adaptive': a['waiting_load'],         'improvement_pct': wait_imp},
            {'metric': 'Hang doi trung binh (xe)',   'fixed': f['avg_queue_length'],     'adaptive': a['avg_queue_length'],     'improvement_pct': queue_imp},
            {'metric': 'Luu luong uoc luong (xe)',   'fixed': f['estimated_throughput'], 'adaptive': a['estimated_throughput'], 'improvement_pct': throughput_imp},
            {'metric': 'Ty le un tac trung tam (%)', 'fixed': f['congestion_rate_pct'],  'adaptive': a['congestion_rate_pct'],  'improvement_pct': congestion_imp},
        ]
        self._save_csv("evaluation_summary.csv", summary)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    evaluator = Evaluator(
        video_path         = "original_video.mp4",
        roi_config_path    = "roi_config.json",
        model_path         = "best.pt",
        fps                = 30,
        conf_threshold     = 0.3,
        gridlock_threshold = 8,
    )
    evaluator.run()
