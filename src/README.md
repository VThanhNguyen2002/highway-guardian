# Highway Guardian Backend API

## 📁 Cấu trúc Project (Refactored)

```
src/
├── main.py                      # Main FastAPI application
├── requirements.txt             # Python dependencies
│
├── config/                      # Configuration files
│   ├── __init__.py
│   └── settings.py             # App settings and constants
│
├── services/                    # Business logic services
│   ├── __init__.py
│   └── detection_service.py    # Detection and prediction logic
│
└── utils/                       # Utility modules
    ├── __init__.py
    ├── model_manager.py         # Model loading and caching
    └── traffic_sign_mapping.py  # Sign name translations and mappings
```

## 🎯 Kiến trúc Mới

### 1. **Separation of Concerns**
- `main.py`: Chỉ chứa API endpoints và routing
- `services/`: Business logic và xử lý chính
- `utils/`: Helper functions và utilities
- `config/`: Configuration và settings

### 2. **Model Management**
- **Caching**: Models được cache để tránh load lại
- **LRU Eviction**: Tự động xóa models ít dùng khi cache đầy
- **Error Handling**: Xử lý lỗi rõ ràng cho từng loại model

### 3. **Traffic Sign Mapping**
- **VIETNAMESE_SIGN_MAP**: Mapping YOLO class names sang tiếng Việt
- **CNN_CLASS_NAMES**: Mapping CNN class IDs sang tên biển báo
- **SIGN_CATEGORIES**: Phân loại biển báo (cấm, hiệu lệnh, cảnh báo, chỉ dẫn)

## 🚀 Cách sử dụng

### Cài đặt Dependencies:
```bash
cd src
pip install -r requirements.txt
```

### Chạy Server:
```bash
# Option 1: Trực tiếp
python main.py

# Option 2: Dùng batch file (Windows)
..\start_backend.bat

# Option 3: Uvicorn manual
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test API:
```bash
# Health check
curl http://localhost:8000/

# Get models
curl http://localhost:8000/models
```

## 📝 Cập nhật CNN Class Names

Để cập nhật mapping cho CNN model, chỉnh sửa file `utils/traffic_sign_mapping.py`:

```python
CNN_CLASS_NAMES = {
    0: "Cấm rẽ trái",
    1: "Cấm rẽ phải",
    2: "Giới hạn tốc độ 50",
    3: "Giới hạn tốc độ 60",
    # ... thêm các class khác
}
```

## 🔧 Configuration

Chỉnh sửa `config/settings.py` để thay đổi:
- Model directories
- CORS origins
- Default thresholds
- Cache settings

## 📊 API Endpoints

### GET `/`
Health check

### GET `/models`
Lấy danh sách models

### POST `/predict`
Single model prediction (YOLO hoặc CNN)

### POST `/predict_two_stage`
Two-stage pipeline (YOLO + CNN)

## 🎨 Tính năng

### Model Caching
- Tự động cache models đã load
- LRU eviction khi cache đầy
- Giảm thời gian load cho requests tiếp theo

### Error Handling
- HTTPException cho lỗi rõ ràng
- Try-catch cho từng service
- Logging chi tiết

### Extensibility
- Dễ dàng thêm model types mới
- Dễ dàng thêm preprocessing steps
- Dễ dàng thêm post-processing logic

## 🔮 Tương lai

- [ ] Add more sign categories
- [ ] Implement sign detection history
- [ ] Add model performance metrics
- [ ] Implement batch processing
- [ ] Add model versioning
- [ ] Add A/B testing for models

---

*Refactored: 2025-01-21*
