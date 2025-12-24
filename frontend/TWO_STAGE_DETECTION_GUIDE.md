# Hướng dẫn Sử dụng Hệ thống Nhận diện 2 Giai đoạn

## 🎯 Tổng quan

Hệ thống Highway Guardian hiện hỗ trợ **2 chế độ nhận diện**:

### 1. **YOLO Only** (Nhanh - Real-time)
- Chỉ sử dụng model YOLO
- Phát hiện và phân loại trực tiếp
- Tốc độ cao, phù hợp cho real-time
- Độ chính xác: Tốt

### 2. **YOLO + CNN** (Chính xác - 2 Stage Pipeline)
- **Giai đoạn 1**: YOLO phát hiện và crop vùng biển báo
- **Giai đoạn 2**: CNN phân loại chi tiết biển báo đã crop
- Độ chính xác cao hơn
- Tốc độ chậm hơn một chút

---

## 📁 Cấu trúc Models

```
models/
├── yolo/
│   └── best.pt                    # YOLO model cho detection
└── cnn/
    └── bien_bao_mobilenetv2_MERGED_BALANCED_model (1).h5  # CNN model cho classification
```

---

## 🚀 Cách sử dụng

### A. Trang Detect (Upload Ảnh)

1. **Chọn loại model**:
   - `YOLO (Detection)`: Nhận diện nhanh
   - `CNN (Classification)`: Phân loại toàn bộ ảnh

2. **Chọn model cụ thể** từ dropdown

3. **Upload ảnh** và nhấn "Bắt đầu Nhận diện"

### B. Trang Camera (Live Detection)

1. **Chọn chế độ nhận diện**:
   - `YOLO Only (Nhanh)`: Chỉ dùng YOLO
   - `YOLO + CNN (Chính xác)`: Pipeline 2 giai đoạn

2. **Chọn models**:
   - Nếu chọn "YOLO Only": Chỉ cần chọn 1 YOLO model
   - Nếu chọn "YOLO + CNN": Chọn cả YOLO và CNN model

3. **Bật Camera** và xem kết quả real-time

4. **Theo dõi thống kê**:
   - FPS: Số khung hình/giây
   - Phát hiện: Số lượng biển báo phát hiện được

---

## 🔧 API Endpoints

### 1. GET `/models`
Lấy danh sách tất cả models

**Response:**
```json
{
  "yolo": ["best.pt"],
  "cnn": ["bien_bao_mobilenetv2_MERGED_BALANCED_model (1).h5"]
}
```

### 2. POST `/predict`
Nhận diện với 1 model (YOLO hoặc CNN)

**Parameters:**
- `file`: File ảnh
- `model_name`: Tên model
- `model_type`: "yolo" hoặc "cnn"

**Response (YOLO):**
```json
{
  "predictions": [
    {
      "box_coordinates": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_name": "Giới hạn tốc độ"
    }
  ]
}
```

**Response (CNN):**
```json
{
  "predictions": [
    {
      "class_name": "Class_5",
      "confidence": 0.92,
      "class_id": 5
    }
  ]
}
```

### 3. POST `/predict_two_stage`
Pipeline 2 giai đoạn (YOLO + CNN)

**Parameters:**
- `file`: File ảnh
- `yolo_model`: Tên YOLO model
- `cnn_model`: Tên CNN model
- `confidence_threshold`: Ngưỡng confidence (default: 0.25)

**Response:**
```json
{
  "predictions": [
    {
      "box_coordinates": [x1, y1, x2, y2],
      "yolo_confidence": 0.95,
      "cnn_confidence": 0.92,
      "class_name": "Class_5",
      "class_id": 5
    }
  ]
}
```

---

## 🎨 Giao diện

### Màu sắc Bounding Box
- **YOLO Only**: Gradient xanh dương-tím (#667eea → #764ba2)
- **YOLO + CNN**: Cùng màu, nhưng hiển thị cả 2 confidence scores

### Thông tin hiển thị
- **YOLO Only**: `Tên biển báo (95%)`
- **YOLO + CNN**: `Tên biển báo (CNN: 92%)`

---

## ⚙️ Cấu hình

### Backend (src/main.py)
```python
YOLO_MODELS_DIR = "/app/models/yolo"
CNN_MODELS_DIR = "/app/models/cnn"
```

### Frontend
- **Detect.vue**: Upload và nhận diện ảnh
- **Camera.vue**: Real-time detection với webcam

---

## 📊 So sánh Hiệu suất

| Chế độ | Tốc độ | Độ chính xác | Use Case |
|--------|--------|--------------|----------|
| YOLO Only | ⚡⚡⚡ Rất nhanh | ⭐⭐⭐ Tốt | Real-time, demo |
| YOLO + CNN | ⚡⚡ Nhanh | ⭐⭐⭐⭐ Rất tốt | Production, độ chính xác cao |

---

## 🐛 Troubleshooting

### Lỗi: "Model not found"
- Kiểm tra file model có tồn tại trong thư mục `models/yolo` hoặc `models/cnn`
- Đảm bảo tên file chính xác

### Lỗi: "Cannot load CNN model"
- Cần cài đặt TensorFlow: `pip install tensorflow`
- Kiểm tra định dạng file (.h5 hoặc .keras)

### Camera không hoạt động
- Cho phép trình duyệt truy cập camera
- Kiểm tra camera không bị ứng dụng khác sử dụng

### FPS thấp
- Giảm resolution camera
- Sử dụng chế độ "YOLO Only"
- Tăng interval giữa các frame (hiện tại: 200ms)

---

## 🔮 Tính năng Tương lai

- [ ] Hỗ trợ nhiều CNN models khác nhau
- [ ] Mapping class_id sang tên biển báo tiếng Việt cho CNN
- [ ] Export kết quả detection
- [ ] Batch processing cho nhiều ảnh
- [ ] Video processing
- [ ] Model comparison tool

---

*Cập nhật lần cuối: 2025-01-21*
