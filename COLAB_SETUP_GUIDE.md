# Hướng dẫn Setup Highway Guardian trên Google Colab

## 🚀 Bước 1: Clone Repository

```python
# Clone repository từ GitHub
!git clone https://github.com/VThanhNguyen2002/highway-guardian.git
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

### Option A: Sử dụng Dataset Manager
```python
# Sử dụng script có sẵn
!python scripts/dataset_manager.py
```

### Option B: Tải thủ công từ Kaggle
```python
# Tải dataset car detection
!kaggle datasets download -d seyeon040768/car-detection-dataset
!unzip car-detection-dataset.zip -d data/car_detection/
```

### Option C: Sử dụng Demo Dataset
```python
# Tạo demo dataset nhỏ để test nhanh
!python create_demo_dataset.py
```

## 🎯 Bước 5: Training Model

### Training với Dataset thực từ Kaggle
```python
# Chạy training với cấu hình tối ưu cho Colab
!python src/training/scripts/train_car_detection.py --config configs/car_detection_real.yaml
```

### Training với Demo Dataset
```python
# Training nhanh với demo dataset
!python src/training/scripts/train_car_detection.py --config configs/demo_training_config.yaml
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