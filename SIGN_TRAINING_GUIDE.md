# 🚦 Hướng Dẫn Training Sign Detection - Khắc Phục Lỗi CUDA

## 🚨 Vấn Đề
Khi training mô hình sign detection trên Google Colab, bạn gặp lỗi:
```
CUDA error: device-side assert triggered
```

## 💡 Giải Pháp
Thay vì training trên Google Colab, chúng ta sẽ training **local** trên máy tính của bạn.

## 🛠️ Các File Đã Tạo

### 1. Script Training Đơn Giản
- **File**: `src/train_sign_simple.py`
- **Mô tả**: Script Python đơn giản, tối ưu cho training local
- **Ưu điểm**: 
  - Tránh lỗi CUDA của Colab
  - Batch size nhỏ (4) để tránh OOM
  - Epochs ít hơn (80) để test nhanh
  - Tự động tìm mô hình xe để transfer learning

### 2. Jupyter Notebook Local
- **File**: `src/Sign_Detection_Local_Training.ipynb`
- **Mô tả**: Notebook chi tiết với visualization
- **Ưu điểm**:
  - Có thể chạy từng cell
  - Hiển thị kết quả training
  - Có confusion matrix và validation plots

### 3. Batch File Tự Động
- **File**: `run_sign_training_simple.bat`
- **Mô tả**: File batch để chạy training một cách dễ dàng
- **Cách dùng**: Double-click để chạy

## 🚀 Cách Sử Dụng

### Phương Pháp 1: Chạy Script Đơn Giản (Khuyến Nghị)
```bash
# Mở Command Prompt hoặc PowerShell
cd d:\Projects\highway-guardian

# Chạy batch file
run_sign_training_simple.bat
```

### Phương Pháp 2: Chạy Trực Tiếp Python
```bash
cd d:\Projects\highway-guardian\src
python train_sign_simple.py
```

### Phương Pháp 3: Sử Dụng Jupyter Notebook
```bash
# Mở Jupyter Notebook
jupyter notebook src/Sign_Detection_Local_Training.ipynb
```

## ⚙️ Cấu Hình Training

### Parameters Đã Tối Ưu:
- **Epochs**: 80 (thay vì 150)
- **Batch Size**: 4 (thay vì 8-16)
- **Image Size**: 640 (thay vì 960)
- **Workers**: 2 (thay vì 4-8)
- **Cache**: False (tiết kiệm RAM)
- **AMP**: True (mixed precision)

### Yêu Cầu Hệ Thống:
- **RAM**: Tối thiểu 8GB
- **GPU**: NVIDIA với CUDA (khuyến nghị)
- **Disk**: 5GB trống
- **Python**: 3.8+

## 📊 Kết Quả Mong Đợi

### Mục Tiêu Performance:
- **mAP50**: > 80%
- **mAP50-95**: > 50%
- **Model Size**: < 50MB

### Output Files:
```
runs/detect/sign_simple/
├── weights/
│   ├── best.pt          # Mô hình tốt nhất
│   └── last.pt          # Mô hình cuối cùng
├── results.png          # Biểu đồ training
├── confusion_matrix.png # Ma trận nhầm lẫn
└── val_batch0_labels.jpg # Ảnh validation
```

## 🔧 Troubleshooting

### Lỗi CUDA OOM (Out of Memory):
```python
# Giảm batch size trong train_sign_simple.py
'batch': 2,  # hoặc 1
'imgsz': 416,  # giảm image size
```

### Lỗi không tìm thấy data:
```python
# Kiểm tra đường dẫn data trong config
path: '../data/traffic_signs/extracted/train_data'
```

### Lỗi thiếu dependencies:
```bash
pip install ultralytics torch torchvision pyyaml matplotlib
```

## 🎯 Sử Dụng Model Đã Training

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/sign_simple/weights/best.pt')

# Inference
results = model('path/to/image.jpg')

# Hiển thị kết quả
for r in results:
    r.show()
```

## 📝 So Sánh với Google Colab

| Aspect | Google Colab | Local Training |
|--------|--------------|----------------|
| CUDA Error | ❌ Có | ✅ Không |
| Control | ❌ Hạn chế | ✅ Đầy đủ |
| Time Limit | ❌ 12h | ✅ Không giới hạn |
| GPU Memory | ❌ Chia sẻ | ✅ Dedicated |
| Data Access | ❌ Upload | ✅ Local |
| Debugging | ❌ Khó | ✅ Dễ |

## 🎉 Kết Luận

Việc training local sẽ:
- ✅ Tránh được lỗi CUDA của Colab
- ✅ Có control tốt hơn
- ✅ Không bị giới hạn thời gian
- ✅ Dễ debug và tối ưu

**Khuyến nghị**: Sử dụng `run_sign_training_simple.bat` để bắt đầu!