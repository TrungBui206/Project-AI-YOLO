import os

label_dir = r"C:\project 2 - YOLOv8 remake\labels"

for i in range(155, 200):  # frame_0155 -> frame_0199
    file_path = os.path.join(label_dir, f"frame_{i:04d}.txt")

    if not os.path.isfile(file_path):
        print(f"Không tìm thấy: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()

        if not parts:
            continue

        if parts[0] == "0":
            parts[0] = "1"
        elif parts[0] == "1":
            parts[0] = "0"

        lines[idx] = " ".join(parts) + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Đã sửa: frame_{i:04d}.txt")

print("Hoàn thành!")
