# Hướng dẫn Debug Camera Detection

## Các thay đổi đã thực hiện:

### 1. Frontend (Camera.vue)
- ✅ Thêm **timeout 5 giây** cho mỗi request (tránh bị treo)
- ✅ Thêm **error handling** tốt hơn với error counter
- ✅ Giảm tần suất gửi request: **500ms/request** thay vì mỗi frame
- ✅ Thêm tùy chọn **điều chỉnh tốc độ detection** (200ms - 2000ms)
- ✅ Tự động **tắt camera** sau 3 lỗi liên tiếp
- ✅ Hiển thị **số lỗi** trên UI
- ✅ Không block vòng lặp khi gửi request

### 2. Backend (main.py)
- ✅ Luôn trả về format đúng: `{"predictions": [], "success": bool, "error": str}`
- ✅ Thêm traceback để debug lỗi dễ hơn

## Cách kiểm tra:

### Bước 1: Kiểm tra Backend đang chạy
```bash
# Mở terminal và chạy:
curl http://localhost:8000/

# Kết quả mong đợi:
{"status":"running","service":"Highway Guardian API","version":"1.0.0"}
```

### Bước 2: Kiểm tra Models có load được không
```bash
curl http://localhost:8000/models

# Kết quả mong đợi:
{"yolo":["best.pt"],"cnn":["mobilenetv2_traffic_signs.h5"]}
```

### Bước 3: Test upload ảnh thủ công
```bash
# Tạo file test (hoặc dùng ảnh có sẵn)
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "model_name=best.pt" \
  -F "model_type=yolo"

# Nếu thành công sẽ thấy:
{"predictions":[...],"success":true}
```

### Bước 4: Mở Browser Console khi test Camera
1. Mở trang Camera
2. Nhấn F12 để mở DevTools
3. Vào tab Console
4. Bật camera và xem log

**Log bình thường:**
```
Camera đang chạy...
FPS: 2
Phát hiện: 1
```

**Log có lỗi:**
```
Detect error: AbortError (timeout)
hoặc
Detect error: Server error: 500
```

## Các vấn đề thường gặp:

### 1. FPS = 0, Phát hiện = 0
**Nguyên nhân:**
- Backend không chạy
- Model chưa load
- Request bị timeout

**Giải pháp:**
- Kiểm tra backend: `curl http://localhost:8000/`
- Xem log backend trong terminal
- Tăng timeout lên 10s nếu model load chậm

### 2. Camera bật nhưng không detect
**Nguyên nhân:**
- Chưa chọn model
- Model file không tồn tại
- CORS error

**Giải pháp:**
- Kiểm tra đã chọn model chưa
- Xem Console có lỗi CORS không
- Kiểm tra file model trong thư mục `models/`

### 3. Detect chậm, lag
**Nguyên nhân:**
- Model quá nặng
- CPU/GPU yếu
- Gửi request quá nhanh

**Giải pháp:**
- Chọn "YOLO Only" thay vì "YOLO + CNN"
- Tăng Detection Interval lên 1000ms hoặc 2000ms
- Giảm resolution camera

### 4. Lỗi "Quá nhiều lỗi"
**Nguyên nhân:**
- Backend crash
- Model load lỗi
- Network issue

**Giải pháp:**
- Restart backend
- Kiểm tra log backend
- Kiểm tra file model còn nguyên vẹn không

## Tùy chỉnh thêm:

### Thay đổi timeout (trong Camera.vue):
```javascript
// Dòng ~210
const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s
// Đổi thành 10s nếu cần:
const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s
```

### Thay đổi số lỗi tối đa:
```javascript
// Trong data():
maxErrors: 3  // Đổi thành 5 hoặc 10 nếu muốn
```

### Thay đổi tốc độ mặc định:
```javascript
// Trong data():
detectionInterval: 500  // Đổi thành 1000 để chậm hơn
```

## Debug Backend:

### Xem log chi tiết:
```bash
# Nếu chạy bằng Python:
python src/main.py

# Nếu chạy bằng Docker:
docker logs -f <container_name>
```

### Test model load:
```python
# Tạo file test_model.py
from utils.model_manager import load_yolo_model, load_cnn_model

try:
    yolo = load_yolo_model("best.pt", "models/yolo")
    print("✅ YOLO loaded")
except Exception as e:
    print(f"❌ YOLO error: {e}")

try:
    cnn = load_cnn_model("mobilenetv2_traffic_signs.h5", "models/cnn")
    print("✅ CNN loaded")
except Exception as e:
    print(f"❌ CNN error: {e}")
```

## Liên hệ nếu vẫn lỗi:
- Gửi screenshot Console (F12)
- Gửi log Backend
- Mô tả chi tiết bước tái hiện lỗi
