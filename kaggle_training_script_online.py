#!/usr/bin/env python3
"""
Kaggle Training Script với Internet ON - WandB Enabled

Script này sử dụng internet để cài đặt packages và enable WandB logging trên Kaggle.
Phiên bản này cho phép sử dụng đầy đủ tính năng WandB để theo dõi training.

Usage:
    # Bật Internet trong Kaggle settings trước khi chạy
    # Copy toàn bộ nội dung script này vào Kaggle notebook
    # Hoặc chạy từng cell một cách tuần tự

Author: Highway Guardian Team
Date: 2024
"""

# Cell 1: Cài đặt packages với internet enabled
print("🌐 Internet Mode: Cài đặt packages từ PyPI...")

# Cài đặt các packages cần thiết
!pip install ultralytics==8.0.196 --quiet
!pip install wandb --quiet
!pip install opencv-python --quiet
!pip install torch torchvision --quiet

print("✅ Packages đã được cài đặt thành công!")

# Cell 2: Setup WandB với tùy chọn
print("\n🔧 WandB Setup Options:")
print("1. Tự động disable WandB (không cần account)")
print("2. Enable WandB logging (cần account)")

# Tùy chọn 1: Disable WandB (mặc định)
WANDB_MODE = "disabled"  # Thay đổi thành "online" nếu muốn enable WandB

if WANDB_MODE == "disabled":
    import os
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'
    print("✅ WandB đã được disable - không cần account")
else:
    print("🔑 WandB enabled - cần đăng nhập account")
    import wandb
    # wandb.login()  # Uncomment nếu cần đăng nhập

# Cell 3: Đứng đúng thư mục dự án
%cd /kaggle/working/highway-guardian

# Cell 4: Download và setup dataset
print("\n📦 Dataset Setup...")

# Tạo thư mục data
!mkdir -p data/car_detection

# Option A: Sử dụng dataset có sẵn từ Kaggle Input
if os.path.exists('/kaggle/input/car-detection-dataset'):
    print("✅ Sử dụng dataset từ Kaggle Input")
    !cp -r /kaggle/input/car-detection-dataset/* data/car_detection/
else:
    # Option B: Download từ Kaggle API (cần internet)
    print("📥 Downloading dataset từ Kaggle API...")
    !kaggle datasets download -d seyeon040768/car-detection-dataset
    !unzip -o car-detection-dataset.zip -d data/car_detection > /dev/null
    !rm car-detection-dataset.zip

# Cell 5: Kiểm tra cấu trúc dataset
print("\n🔍 Kiểm tra cấu trúc dataset:")
!find data/car_detection -maxdepth 3 -type d | sort | head -20

# Cell 6: Auto-detect dataset root và tạo config
import os, glob, re, pathlib, yaml

BASE = "/kaggle/working/highway-guardian/data/car_detection"

# Tìm thư mục chứa train/valid/test
cands = set()
for pat in ("train/images", "images/train"):
    for p in glob.glob(f"{BASE}/**/{pat}", recursive=True):
        cands.add(os.path.dirname(os.path.dirname(p)))

if not cands:
    raise SystemExit("❌ Không tìm thấy train/images (hoặc images/train) dưới data/car_detection")

DATA_ROOT = sorted(cands)[0]
print(f"✅ Detected dataset root: {DATA_ROOT}")

# Chuẩn hoá các nhánh train/val/test
def pick_rel(root, pref_a, pref_b):
    a = os.path.join(root, pref_a); b = os.path.join(root, pref_b)
    if os.path.isdir(a): return pref_a
    if os.path.isdir(b): return pref_b
    return None

train_rel = pick_rel(DATA_ROOT, "train/images", "images/train")
val_rel   = pick_rel(DATA_ROOT, "valid/images", "val/images")
test_rel  = pick_rel(DATA_ROOT, "test/images",  "images/test")

if not val_rel:
    raise SystemExit("❌ Không tìm thấy valid/images hoặc val/images")
if not test_rel:
    test_rel = val_rel
    print(f"ℹ️  test -> dùng chung thư mục với val/valid: {test_rel}")

print(f"train: {train_rel} | val: {val_rel} | test: {test_rel}")

# Tạo config file cho training
config_data = {
    'path': DATA_ROOT,
    'train': train_rel,
    'val': val_rel,
    'test': test_rel,
    'nc': 1,
    'names': ['car']
}

config_path = 'configs/car_detection_online.yaml'
os.makedirs('configs', exist_ok=True)

with open(config_path, 'w') as f:
    yaml.dump(config_data, f, default_flow_style=False)

print(f"\n✅ Config file created: {config_path}")
print("Config content:")
with open(config_path, 'r') as f:
    print(f.read())

# Cell 7: Training với internet enabled
from ultralytics import YOLO
import torch

print("\n🚀 Bắt đầu training với internet enabled...")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"WandB Mode: {WANDB_MODE}")

# Initialize model
model = YOLO('yolov8n.pt')

# Training parameters optimized cho Kaggle với internet
train_params = {
    'data': config_path,
    'epochs': 50,  # Tăng epochs vì có internet
    'batch': 16,
    'imgsz': 640,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'workers': 4,  # Tăng workers vì có internet
    'project': 'runs_detect',
    'name': 'car_detection_online',
    'save': True,
    'save_period': 10,
    'patience': 15,
    'plots': True,
    'verbose': True,
    'val': True,
    'cache': True,  # Enable cache vì có internet
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5
}

print("\nTraining parameters:")
for key, value in train_params.items():
    print(f"  {key}: {value}")

# Start training
try:
    print("\n🎯 Starting training...")
    results = model.train(**train_params)
    print("\n🎉 Training completed successfully!")
    
    # Display results
    print(f"\n📊 Training Results:")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    print("Checking for saved checkpoints...")
    
    checkpoint_dir = '/kaggle/working/runs_detect/car_detection_online/weights'
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        print(f"Available checkpoints: {checkpoints}")
    else:
        print("No checkpoints found.")

# Cell 8: Validation và test
print("\n🔍 Model Validation...")

best_model_path = '/kaggle/working/runs_detect/car_detection_online/weights/best.pt'
if os.path.exists(best_model_path):
    print(f"✅ Best model found: {best_model_path}")
    
    # Load best model
    best_model = YOLO(best_model_path)
    
    # Validate
    val_results = best_model.val(data=config_path)
    print(f"\n📈 Validation Metrics:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    
else:
    print("❌ Best model not found")

print("\n✅ Training pipeline completed!")
print("📁 Results saved in: /kaggle/working/runs_detect/car_detection_online/")

if WANDB_MODE != "disabled":
    print("📊 Check WandB dashboard for detailed metrics and visualizations")