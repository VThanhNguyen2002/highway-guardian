#!/usr/bin/env python3
"""
Kaggle Training Script với Auto-disable WandB

Script này tự động setup và chạy training trên Kaggle mà không cần nhập thủ công WandB options.
Sử dụng code được cung cấp bởi người dùng với các cải tiến tự động hóa.

Usage:
    # Copy toàn bộ nội dung script này vào Kaggle notebook
    # Hoặc chạy từng cell một cách tuần tự

Author: Highway Guardian Team
Date: 2024
"""

# Cell 1: Tự động disable WandB để tránh phải nhập thủ công
import os
os.environ['WANDB_MODE'] = 'disabled'  # Tự động chọn option 3: Don't visualize my results
os.environ['WANDB_DISABLED'] = 'true'   # Backup disable
print("✅ WandB đã được tự động disable - không cần nhập thủ công!")

# Cell 2: Đứng đúng thư mục dự án
%cd /kaggle/working/highway-guardian

# Cell 3: (Nếu chưa giải nén) Giải nén dataset vào data/car_detection
!mkdir -p data/car_detection
!unzip -o car-detection-dataset.zip -d data/car_detection > /dev/null

# Cell 4: Kiểm tra nhanh cấu trúc
!find data/car_detection -maxdepth 3 -type d | sort | sed -n '1,120p'

# Cell 5: Auto-detect dataset root và patch configs/car_detection_colab.yaml
import os, glob, re, pathlib

BASE = "/kaggle/working/highway-guardian/data/car_detection"

# Tìm thư mục chứa train/valid/test theo 2 pattern phổ biến
cands = set()
for pat in ("train/images", "images/train"):
    for p in glob.glob(f"{BASE}/**/{pat}", recursive=True):
        cands.add(os.path.dirname(os.path.dirname(p)))  # về root của dataset

if not cands:
    raise SystemExit("❌ Không tìm thấy train/images (hoặc images/train) dưới data/car_detection")

DATA_ROOT = sorted(cands)[0]
print("✅ Detected dataset root:", DATA_ROOT)

# Chuẩn hoá các nhánh train/val/test tương ứng với thực tế
def pick_rel(root, pref_a, pref_b):
    a = os.path.join(root, pref_a); b = os.path.join(root, pref_b)
    if os.path.isdir(a): return pref_a
    if os.path.isdir(b): return pref_b
    return None

train_rel = pick_rel(DATA_ROOT, "train/images", "images/train")
val_rel   = pick_rel(DATA_ROOT, "valid/images", "val/images")
test_rel  = pick_rel(DATA_ROOT, "test/images",  "images/test")

# Nếu thiếu 'valid', mà có 'val' -> dùng 'val'
# Nếu thiếu 'test' -> fallback sang 'valid/val'
if not val_rel:
    raise SystemExit("❌ Không tìm thấy valid/images hoặc val/images")
if not test_rel:
    test_rel = val_rel
    print("ℹ️  test -> dùng chung thư mục với val/valid:", test_rel)

print("train:", train_rel, "| val:", val_rel, "| test:", test_rel)

# Patch file config
cfg = pathlib.Path("configs/car_detection_colab.yaml")
text = cfg.read_text()
text = re.sub(r"path:\s*'.*?'",          f"path: '{DATA_ROOT}'", text)
text = re.sub(r"train:\s*'.*?'",         f"train: '{train_rel}'", text)
text = re.sub(r"val:\s*'.*?'",           f"val: '{val_rel}'",     text)
text = re.sub(r"test:\s*'.*?'",          f"test: '{test_rel}'",    text)
cfg.write_text(text)

print("✅ Patched config:")
print(cfg.read_text().splitlines()[10:25])  # in ra block data:

# Cell 6: Chạy training với config file đã tạo cho Colab
print("🚀 Bắt đầu training với WandB đã được tự động disable...")
!python src/training/scripts/train_car_detection.py --config configs/car_detection_colab.yaml

print("\n🎉 Training hoàn tất! Kiểm tra kết quả trong thư mục runs/")