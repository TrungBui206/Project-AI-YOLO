import pandas as pd

# Đọc file kết quả
df = pd.read_csv("signal_results.csv")

# Lọc ra các thời điểm mà pha chuyển sang GREEN và lấy thời gian được cấp ban đầu
# (Lấy giá trị max của time_left mỗi khi chuyển pha)
adaptive_green = df[df['adaptive_state'] == 'GREEN']
fixed_green = df[df['fixed_state'] == 'GREEN']

# Đếm tổng thời gian đèn xanh đã cấp
total_adaptive_time = adaptive_green.groupby((adaptive_green['adaptive_phase'] != adaptive_green['adaptive_phase'].shift()).cumsum())['adaptive_time_left'].max().sum()
total_fixed_time = fixed_green.groupby((fixed_green['fixed_phase'] != fixed_green['fixed_phase'].shift()).cumsum())['fixed_time_left'].max().sum()

print(f"Tổng thời gian cấp đèn xanh (Fixed-time): {total_fixed_time:.1f} giây")
print(f"Tổng thời gian cấp đèn xanh (Adaptive):   {total_adaptive_time:.1f} giây")

if total_fixed_time > 0:
    saved_time = total_fixed_time - total_adaptive_time
    efficiency = (saved_time / total_fixed_time) * 100
    print(f"\n=> Thuật toán Adaptive đã TIẾT KIỆM được: {saved_time:.1f} giây")
    print(f"=> Hiệu quả cắt giảm thời gian chờ vô ích: {efficiency:.1f}%")
else:
    print("Chưa đủ dữ liệu chu kỳ.")