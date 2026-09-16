import time


class PhaseController:
    """
    Bộ điều phối đèn giao thông thông minh theo 5 sub-pha.

    Chu kỳ đèn:
        ① Bắc-Nam thẳng
           T = max(North_Straight, South_Straight) * sec_per_veh

        ② Bắc trái + Bắc thẳng  (bỏ qua nếu North_Left < 2)
           T = max(North_Left, int(North_Straight * straight_share_factor)) * sec_per_veh

        ③ Đông-Tây trái          (bỏ qua nếu East_Left + West_Left < 2)
           T = max(East_Left, West_Left) * sec_per_veh

        ④ Đông-Tây thẳng         (bỏ qua nếu East_Straight = West_Straight = 0)
           T = max(East_Straight, West_Straight) * sec_per_veh

        ⑤ Nam trái + Nam thẳng   (bỏ qua nếu South_Left < 2)
           T = max(South_Left, int(South_Straight * straight_share_factor)) * sec_per_veh

        → Quay lại ①

    Nguyên tắc:
        - Counts REALTIME tại thời điểm kích hoạt pha
        - Pha rẽ trái cần >= 2 xe mới kích hoạt
        - Pha thẳng (1,4): clamp [min_green_straight, max_green]
        - Pha rẽ trái (2,3,5): clamp [min_green_left, max_green]
        - Hệ số 0.3 cho thẳng ở pha 2/5
        - Bắc trái và Nam trái luôn cách nhau ít nhất 2 sub-pha
    """

    PHASE_SEQUENCE = [1, 2, 3, 4, 5]

    def __init__(
        self,
        fps=30,
        min_green_straight=15,
        min_green_left=5,
        max_green=60,
        yellow_time=3,
        sec_per_veh=2.5,
        gridlock_threshold=3,
        straight_share_factor=0.3,
        max_emergency_time=10.0,
        gridlock_cooldown_time=10.0
    ):
        self.fps                    = fps
        self.min_green_straight     = min_green_straight
        self.min_green_left         = min_green_left
        self.max_green              = max_green
        self.yellow_time            = yellow_time
        self.sec_per_veh            = sec_per_veh
        self.gridlock_threshold     = gridlock_threshold
        self.straight_share_factor  = straight_share_factor
        self.max_emergency_time     = max_emergency_time
        self.gridlock_cooldown_time = gridlock_cooldown_time

        # Trạng thái máy
        self.phase_index              = 0
        self.current_phase            = 1
        self.current_state            = "GREEN"
        self.frame_counter            = 0
        self.current_allocated_time   = min_green_straight

        self._init_pending = True

       
    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def update(self, current_counts):
        """Gọi mỗi frame. Trả về dict trạng thái đèn hiện tại."""

        if self._init_pending:
            self._init_pending = False
            self.current_allocated_time = self._calc_green_time(
                self.current_phase, current_counts
            )
        
        # --- VÒNG LẶP CHU KỲ BÌNH THƯỜNG ---
        self.frame_counter += 1
        elapsed = self.frame_counter / self.fps

        if elapsed >= self.current_allocated_time:
            self._transition(current_counts)

        time_left = max(0.0, self.current_allocated_time - elapsed)
        return self._build_status(time_left, current_counts)

    # ------------------------------------------------------------------
    # PRIVATE — Chuyển trạng thái
    # ------------------------------------------------------------------

    def _transition(self, counts):
        self.frame_counter = 0

        if self.current_state == "GREEN":
            self.current_state          = "YELLOW"
            self.current_allocated_time = self.yellow_time
        else:
            self.current_state = "GREEN"
            for _ in range(len(self.PHASE_SEQUENCE)):
                self.phase_index = (self.phase_index + 1) % len(self.PHASE_SEQUENCE)
                next_phase = self.PHASE_SEQUENCE[self.phase_index]

                print("\n========== PHASE CHECK ==========")
                print(f"Current phase : {self.current_phase}")
                print(f"Checking phase: {next_phase}")

                print(
                    f"NS={counts.get('North_Straight',0)} "
                    f"NL={counts.get('North_Left',0)} "
                    f"SS={counts.get('South_Straight',0)} "
                    f"SL={counts.get('South_Left',0)} "
                    f"ES={counts.get('East_Straight',0)} "
                    f"EL={counts.get('East_Left',0)} "
                    f"WS={counts.get('West_Straight',0)} "
                    f"WL={counts.get('West_Left',0)}"
                )

                result = self._phase_has_vehicles(next_phase, counts)

                print(f"Phase {next_phase} valid = {result}")

                if result:
                    print(f"SELECT PHASE {next_phase}")
                    print("===============================\n")

                    self.current_phase = next_phase
                    self.current_allocated_time = self._calc_green_time(
                        next_phase, counts
                    )

                    return

            self.current_phase          = 1
            self.phase_index            = 0
            self.current_allocated_time = self.min_green_straight

    # ------------------------------------------------------------------
    # PRIVATE — Kiểm tra sub-pha có đủ xe không
    # ------------------------------------------------------------------

    def _phase_has_vehicles(self, phase, counts):
        if phase == 1:
            return (counts.get('North_Straight', 0) > 0 or
                    counts.get('South_Straight', 0) > 0)
        elif phase == 2:
            # Cần ít nhất 2 xe rẽ trái Bắc
            return counts.get('North_Left', 0) >= 2
        elif phase == 3:
            # Cần tổng ít nhất 2 xe rẽ trái Đông-Tây
            return (counts.get('East_Left', 0) +
                    counts.get('West_Left', 0)) >= 2
        elif phase == 4:
            return (counts.get('East_Straight', 0) > 0 or
                    counts.get('West_Straight', 0) > 0)
        elif phase == 5:
            # Cần ít nhất 2 xe rẽ trái Nam
            return counts.get('South_Left', 0) >= 2
        return False

    # ------------------------------------------------------------------
    # PRIVATE — Tính thời gian đèn xanh động
    # ------------------------------------------------------------------

    def _calc_green_time(self, phase, counts):
        n_s = counts.get('North_Straight', 0)
        s_s = counts.get('South_Straight', 0)
        n_l = counts.get('North_Left',     0)
        s_l = counts.get('South_Left',     0)
        e_s = counts.get('East_Straight',  0)
        w_s = counts.get('West_Straight',  0)
        e_l = counts.get('East_Left',      0)
        w_l = counts.get('West_Left',      0)

        if phase == 1:
            vehicles = max(n_s, s_s)
            min_time = self.min_green_straight
        elif phase == 2:
            vehicles = max(n_l, int(n_s * self.straight_share_factor))
            min_time = self.min_green_left
        elif phase == 3:
            vehicles = max(e_l, w_l)
            min_time = self.min_green_left
        elif phase == 4:
            vehicles = max(e_s, w_s)
            min_time = self.min_green_straight
        elif phase == 5:
            vehicles = max(s_l, int(s_s * self.straight_share_factor))
            min_time = self.min_green_left
        else:
            vehicles = 0
            min_time = self.min_green_straight

        raw_time = vehicles * self.sec_per_veh
        return max(min_time, min(raw_time, self.max_green))

    # ------------------------------------------------------------------
    # PRIVATE — Tạo báo cáo trạng thái đèn
    # ------------------------------------------------------------------

    def _build_status(self, time_left, counts):
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

        gridlock_alert = counts.get('Center', 0) > self.gridlock_threshold

        return {
            'current_phase':       self.current_phase,
            'current_state':       self.current_state,
            'time_left':           round(time_left, 1),
            'lights':              lights,
            'gridlock_alert':      gridlock_alert,
        }


# ======================================================================
# KHỐI CHẠY THỬ MÔ PHỎNG
# ======================================================================
if __name__ == "__main__":
    controller = PhaseController(
        fps=10,
        min_green_straight=15,
        min_green_left=5,
        max_green=40,
        sec_per_veh=2.5,
        gridlock_threshold=3,
        straight_share_factor=0.3,
        max_emergency_time=10.0,
        gridlock_cooldown_time=10.0,
    )

    counts = {
        'North_Straight': 10, 'North_Left': 3,
        'South_Straight': 8,  'South_Left': 2,
        'East_Straight':  5,  'East_Left':  1,
        'West_Straight':  4,  'West_Left':  1,
        'Center':         0,
    }

    print("=" * 70)
    print("  MÔ PHỎNG BỘ ĐIỀU PHỐI ĐÈN THÔNG MINH 5 PHA")
    print("=" * 70)

    prev_phase = None
    prev_state = None

    for frame in range(2000):
        if frame == 150:
            counts['Center'] = 5
        if frame == 220:
            counts['Center'] = 0
        if frame == 350:
            counts['North_Left'] = 0
        if frame == 500:
            counts['North_Left'] = 6

        status = controller.update(counts)

        p = status['current_phase']
        s = status['current_state']
        if p != prev_phase or s != prev_state:
            prev_phase = p
            prev_state = s
            t = round(frame / 10, 1)
            green_dirs = [k for k, v in status['lights'].items()
                          if v in ('GREEN', 'YELLOW')]
            print(f"\n[{t:5.1f}s] Pha {p} [{s}]"
                  f" | Con: {status['time_left']}s"
            )
            print(f"         Huong xanh/vang: "
                  f"{green_dirs if green_dirs else '--- TAT CA DO ---'}")

    print("\n" + "=" * 70)
    print("  KET THUC MO PHONG")
    print("=" * 70)
