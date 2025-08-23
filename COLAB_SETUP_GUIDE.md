# Hướng dẫn Setup Highway Guardian trên Google Colab

## 🚀 Bước 1: Clone Repository

```python
# Clone repository từ GitHub
!git clone https://github.com/your-username/highway-guardian.git
%cd highway-guardian
```

## 📦 Bước 2: Cài đặt Dependencies

```python
# Cài đặt tất cả packages cần thiết
!pip install -r requirements.txt

# Hoặc cài đặt từng package quan trọng
!pip install torch torchvision ultralytics opencv-python kaggle wandb
```

## 🔑 Bước 3: Setup Kaggle API (Tùy chọn)

```python
# Upload file kaggle.json của bạn lên Colab
from google.colab import files
uploaded = files.upload()  # Chọn file kaggle.json

# Tạo thư mục .kaggle và copy file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Kiểm tra kết nối
!kaggle datasets list
```

## 📊 Bước 4: Tải Dataset

### Option A: Sử dụng Dataset từ Kaggle (Khuyến nghị)
```python
# Tải dataset car detection
!kaggle datasets download -d seyeon040768/car-detection-dataset
!unzip car-detection-dataset.zip -d data/car_detection/

# Tạo file config cho Colab
!mkdir -p configs
with open('configs/car_detection_colab.yaml', 'w') as f:
    f.write('''# Car Detection Training Configuration for Colab
# Dataset: Real Kaggle Car Detection Dataset

model:
  weights: 'yolov8n.pt'  # pretrained model
  architecture: 'yolov8n'

data:
  path: '/content/highway-guardian/data/car_detection/car_dataset-master'
  train: 'train/images'
  val: 'valid/images'
  test: 'test/images'
  nc: 1  # number of classes
  names: ['car']  # class names

training:
  epochs: 5
  batch_size: 2
  image_size: 320
  device: 'cuda'  # Use GPU on Colab
  workers: 2
  optimizer: 'AdamW'
  lr0: 0.001
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  box: 7.5
  cls: 0.5
  dfl: 1.5

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

validation:
  save_period: 1
  patience: 10
  conf: 0.001
  iou: 0.6

output:
  project: 'runs/detect'
  name: 'car_detection_colab'
  plots: true
  verbose: true
''')

print("✅ Config file created for Colab!")
```

### Option B: Sử dụng Demo Dataset (Nhanh hơn)
```python
# Tạo demo dataset nhỏ để test nhanh
!python create_demo_dataset.py

# Demo dataset đã có đường dẫn tương đối, không cần sửa
print("✅ Demo dataset ready!")
```

### Option C: Sử dụng Dataset Manager (Tự động)
```python
# Sử dụng script có sẵn (cần sửa đường dẫn)
!python scripts/dataset_manager.py
```

## 🎯 Bước 5: Training Model

### ⚠️ Lưu ý quan trọng:
Nếu gặp lỗi về đường dẫn dataset, hãy đảm bảo:
1. Đã tạo file `data.yaml` với đường dẫn Colab đúng (theo Bước 4)
2. Dataset đã được tải và giải nén vào đúng thư mục
3. Sử dụng tham số `--data` để chỉ định file data.yaml mới

### Training với Dataset từ Kaggle (Khuyến nghị)
```python
# Chạy training với config file đã tạo cho Colab
!python src/training/scripts/train_car_detection.py --config configs/car_detection_colab.yaml
```

### Training nhanh với Demo Dataset
```python
# Training với demo dataset (nhỏ, nhanh)
!python src/training/scripts/train_car_detection.py --config configs/demo_training_config.yaml
```

### Training với Config có sẵn (Cần sửa đường dẫn)
```python
# Lưu ý: Config này có thể cần sửa đường dẫn dataset
!python src/training/scripts/train_car_detection.py --config configs/car_detection_real.yaml
```

## 📈 Bước 6: Theo dõi Training

```python
# Xem logs training
!tail -f runs/detect/*/train.log

# Hiển thị kết quả
from IPython.display import Image, display
display(Image('runs/detect/train/results.png'))
display(Image('runs/detect/train/confusion_matrix.png'))
```

## 🔍 Bước 7: Test Model

```python
# Load model đã train
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')

# Test với ảnh mẫu
results = model('path/to/test/image.jpg')
results[0].show()
```

## 💾 Bước 8: Lưu kết quả

```python
# Tải về model đã train
from google.colab import files
files.download('runs/detect/train/weights/best.pt')
files.download('runs/detect/train/weights/last.pt')

# Tải về kết quả training
files.download('runs/detect/train/results.png')
files.download('runs/detect/train/confusion_matrix.png')
```

## ⚡ Tips tối ưu cho Colab

### 1. Sử dụng GPU
```python
# Kiểm tra GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2. Mount Google Drive (Tùy chọn)
```python
# Mount Drive để lưu kết quả lâu dài
from google.colab import drive
drive.mount('/content/drive')

# Copy kết quả sang Drive
!cp -r runs/detect/train /content/drive/MyDrive/highway-guardian-results/
```

### 3. Cấu hình tối ưu cho Colab
```python
# Sửa file config cho phù hợp với Colab
# Giảm batch_size nếu bị out of memory
# Tăng workers nếu có GPU mạnh
```

## 🚨 Troubleshooting

### ❌ Lỗi Dataset Path (Phổ biến nhất)
**Triệu chứng:** 
- `Dataset 'xxx/data.yaml' images not found ⚠️, missing path '/content/highway-guardian/datasets/C:/PersonalProject/...'`
- `train_car_detection.py: error: the following arguments are required: --config`
- `Training failed: 'output'` (KeyError khi thiếu section output trong config)

**Nguyên nhân:** 
1. Script train_car_detection.py chỉ nhận tham số `--config` chứ không nhận `--data`
2. File config vẫn chứa đường dẫn Windows local thay vì đường dẫn Colab
3. File config thiếu section `output` mà script yêu cầu

**Cách khắc phục:**
```python
# Bước 1: Kiểm tra cấu trúc dataset hiện tại
import os
print("📁 Dataset structure:")
for root, dirs, files in os.walk("data"):
    level = root.replace("data", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:  # Chỉ hiển thị 3 level đầu
        subindent = " " * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f"{subindent}{file}")

# Bước 2: Tạo file config với đường dẫn đúng cho Colab
with open('configs/car_detection_colab.yaml', 'w') as f:
    f.write('''# Car Detection Training Configuration for Colab
model:
  weights: 'yolov8n.pt'
  architecture: 'yolov8n'

data:
  path: '/content/highway-guardian/data/car_detection/car_dataset-master'
  train: 'train/images'
  val: 'valid/images'
  test: 'test/images'
  nc: 1
  names: ['car']

training:
  epochs: 5
  batch_size: 2
  image_size: 320
  device: 'cuda'
  workers: 2
  optimizer: 'AdamW'
  lr0: 0.001
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  box: 7.5
  cls: 0.5
  dfl: 1.5

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

validation:
  save_period: 1
  patience: 10
  conf: 0.001
  iou: 0.6

output:
  project: 'runs/detect'
  name: 'car_detection_colab'
  plots: true
  verbose: true
''')

# Bước 3: Kiểm tra file config đã được tạo
with open('configs/car_detection_colab.yaml', 'r') as f:
    print("\n📄 Content of car_detection_colab.yaml:")
    print(f.read())

# Bước 4: Kiểm tra đường dẫn dataset có tồn tại không
dataset_path = "/content/highway-guardian/data/car_detection/car_dataset-master"
print(f"\n📍 Dataset path exists: {os.path.exists(dataset_path)}")
if os.path.exists(dataset_path):
    print(f"📍 Train images exist: {os.path.exists(os.path.join(dataset_path, 'train/images'))}")
    print(f"📍 Valid images exist: {os.path.exists(os.path.join(dataset_path, 'valid/images'))}")
```

### Lỗi Out of Memory
```python
# Giảm batch_size trong config
batch_size: 4  # thay vì 8 hoặc 16
image_size: 320  # thay vì 640
```

### Lỗi Kaggle API
```python
# Kiểm tra file kaggle.json
!cat ~/.kaggle/kaggle.json
!ls -la ~/.kaggle/
```

### Lỗi Dependencies
```python
# Cài đặt lại packages
!pip install --upgrade ultralytics
!pip install --force-reinstall torch torchvision
```

## 📋 Checklist hoàn thành

- [ ] Clone repository thành công
- [ ] Cài đặt dependencies
- [ ] Setup Kaggle API (nếu cần)
- [ ] Tải dataset
- [ ] Chạy training
- [ ] Kiểm tra kết quả
- [ ] Lưu model và results

## 🎉 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:
- Model YOLOv8 đã được train cho car detection
- Các file weights (.pt)
- Biểu đồ kết quả training
- Confusion matrix
- Validation metrics

---

**Lưu ý**: Google Colab có giới hạn thời gian sử dụng. Đối với training lâu, hãy sử dụng Colab Pro hoặc chia nhỏ quá trình training.