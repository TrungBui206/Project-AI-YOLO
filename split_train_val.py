import os
import random
import shutil

image_dir = "frames"
label_dir = "labels"
out = "dataset"
SPLIT = 0.8

# Xóa dataset cũ nếu có
if os.path.exists(out):
    shutil.rmtree(out)

for split in ["train", "val"]:
    os.makedirs(f"{out}/images/{split}")
    os.makedirs(f"{out}/labels/{split}")

# Lấy danh sách file có cả ảnh lẫn label
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt") and f != "classes.txt"]
image_files = [f.replace(".txt", ".jpg") for f in label_files if os.path.exists(f"{image_dir}/{f.replace('.txt', '.jpg')}")]

random.seed(42)  # cố định seed để kết quả chia luôn nhất quán
random.shuffle(image_files)

split_idx = int(SPLIT * len(image_files))
train = image_files[:split_idx]
val = image_files[split_idx:]

assert len(set(train) & set(val)) == 0, "Trùng lặp train/val!"

def copy_files(files, split):
    for f in files:
        shutil.copy(f"{image_dir}/{f}", f"{out}/images/{split}/{f}")
        shutil.copy(f"{label_dir}/{f.replace('.jpg', '.txt')}", f"{out}/labels/{split}/{f.replace('.jpg', '.txt')}")

copy_files(train, "train")
copy_files(val, "val")

print(f"Done! Train: {len(train)} | Val: {len(val)}")