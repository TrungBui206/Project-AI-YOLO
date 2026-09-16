import cv2
import json
import numpy as np

# Cấu hình ban đầu
image_path = "frame_sample.jpg" 
roi_names = [
    "North_Straight", "North_Left", 
    "South_Straight", "South_Left",
    "East_Straight", "East_Left", 
    "West_Straight", "West_Left", 
    "Center"
]

# Bảng màu riêng biệt cho từng ROI 
COLORS = [
    (0, 255, 0),     # North_Straight - Xanh lá
    (0, 128, 255),   # North_Left - Cam
    (255, 0, 0),     # South_Straight - Xanh dương
    (0, 255, 255),   # South_Left - Vàng
    (255, 0, 255),   # East_Straight - Hồng
    (255, 165, 0),   # East_Left - Cam nhạt
    (128, 0, 128),   # West_Straight - Tím
    (0, 0, 255),     # West_Left - Đỏ
    (128, 128, 0)    # Center - Xanh Olive
]

rois_dict = {}
current_points = []
current_index = 0
user_quit = False  

img = cv2.imread(image_path)
if img is None:
    print("Không tìm thấy ảnh!")
    exit()

preview_img = img.copy()

def draw_state():
    """Hàm cập nhật UI realtime"""
    global clone, current_points, roi_names, current_index
    clone = img.copy()
    
    roi_name = roi_names[current_index]
    current_color = COLORS[current_index % len(COLORS)]
    
    cv2.putText(clone, f"Draw: {roi_name}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2, cv2.LINE_AA)
    cv2.putText(clone, "Enter: Save | c: Clear | b: Back | q: Quit", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    if len(current_points) >= 3:
        pts_array = np.array(current_points, np.int32).reshape((-1, 1, 2))
        
        overlay = clone.copy()
        cv2.fillPoly(overlay, [pts_array], color=current_color)
        cv2.addWeighted(overlay, 0.3, clone, 0.7, 0, clone)
        cv2.polylines(clone, [pts_array], isClosed=True, color=current_color, thickness=2)
        
        for pt in current_points:
            cv2.circle(clone, pt, 4, current_color, -1)
            
    elif len(current_points) > 0:
        if len(current_points) == 2:
            pts_array = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(clone, [pts_array], isClosed=False, color=current_color, thickness=2)
            
        for pt in current_points:
            cv2.circle(clone, pt, 4, current_color, -1)

def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        draw_state()
        cv2.imshow("Draw ROI", clone)

cv2.namedWindow("Draw ROI", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw ROI", mouse_callback)

while current_index < len(roi_names) and not user_quit:
    draw_state() 
    cv2.imshow("Draw ROI", clone)
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13: # Phím Enter
            if len(current_points) >= 3:
                rois_dict[roi_names[current_index]] = current_points
                current_color = COLORS[current_index % len(COLORS)]
                pts_preview = np.array(current_points, np.int32).reshape((-1, 1, 2))
                
                overlay_preview = preview_img.copy()
                cv2.fillPoly(overlay_preview, [pts_preview], color=current_color)
                cv2.addWeighted(overlay_preview, 0.3, preview_img, 0.7, 0, preview_img)
                cv2.polylines(preview_img, [pts_preview], isClosed=True, color=current_color, thickness=2)
                
                cx = int(np.mean([p[0] for p in current_points]))
                cy = int(np.mean([p[1] for p in current_points]))
                
                cv2.putText(preview_img, roi_names[current_index], (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview_img, roi_names[current_index], (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                print(f"-> Đã lưu {roi_names[current_index]}!")
                current_points = []
                current_index += 1
                break
            else:
                print("Cảnh báo: Phải có ít nhất 3 điểm!")

        elif key == ord("c"): # Phím C (Clear)
            current_points = []
            draw_state() 
            cv2.imshow("Draw ROI", clone)
            print("Đã xóa nét vẽ hiện tại.")
            
        elif key == ord("b"): # Phím B (Back) - MỚI BỔ SUNG
            if current_index > 0:
                current_index -= 1 # Lùi lại 1 bước
                last_roi = roi_names[current_index]
                
                if last_roi in rois_dict:
                    del rois_dict[last_roi] # Xóa dữ liệu cũ
                current_points = []
                
                # Khôi phục ảnh preview từ ảnh gốc và vẽ lại các ROI đã đúng
                preview_img = img.copy()
                for i in range(current_index):
                    prev_name = roi_names[i]
                    prev_pts_raw = rois_dict[prev_name]
                    prev_pts = np.array(prev_pts_raw, np.int32).reshape((-1, 1, 2))
                    prev_color = COLORS[i % len(COLORS)]
                    
                    overlay_preview = preview_img.copy()
                    cv2.fillPoly(overlay_preview, [prev_pts], color=prev_color)
                    cv2.addWeighted(overlay_preview, 0.3, preview_img, 0.7, 0, preview_img)
                    cv2.polylines(preview_img, [prev_pts], isClosed=True, color=prev_color, thickness=2)
                    
                    cx = int(np.mean([p[0] for p in prev_pts_raw]))
                    cy = int(np.mean([p[1] for p in prev_pts_raw]))
                    cv2.putText(preview_img, prev_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(preview_img, prev_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                draw_state()
                cv2.imshow("Draw ROI", clone)
                print(f"Đã hoàn tác! Quay lại vẽ: {roi_names[current_index]}")
            else:
                print("Bạn đang ở ROI đầu tiên, không thể quay lại!")

        elif key == ord("q"):
            if rois_dict:  
                print(f"\nCẢNH BÁO: Bạn đã vẽ {len(rois_dict)}/{len(roi_names)} ROI.")
                print("Bấm 'q' lần nữa để LƯU PHẦN ĐÃ VẼ và THOÁT.")
                print("Bấm phím bất kỳ khác để QUAY LẠI vẽ tiếp.")
                
                confirm = cv2.waitKey(0) & 0xFF
                if confirm == ord("q"):
                    user_quit = True  
                    break
                else:
                    draw_state()
                    cv2.imshow("Draw ROI", clone)
            else:
                print("Chưa có dữ liệu. Đã thoát.")
                exit()

# Khối xử lý File I/O
if rois_dict:
    with open("roi_config.json", "w") as f:
        json.dump(rois_dict, f, indent=4)
    
    cv2.imwrite("roi_preview.jpg", preview_img)
    print("\nHOÀN TẤT!")

cv2.destroyAllWindows()
