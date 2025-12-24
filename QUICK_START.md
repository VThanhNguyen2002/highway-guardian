# 🚀 Quick Start - Highway Guardian

## Khởi động nhanh trong 3 bước

### Bước 1: Chuẩn bị Models

Đảm bảo bạn có models trong thư mục:
```
models/
├── yolo/
│   └── best.pt
└── cnn/
    └── bien_bao_mobilenetv2_MERGED_BALANCED_model (1).h5
```

### Bước 2: Start Backend

```bash
# Cài đặt dependencies (lần đầu)
pip install fastapi uvicorn ultralytics tensorflow pillow python-multipart

# Chạy server
cd src
python main.py
```

Backend sẽ chạy tại: **http://localhost:8000**

### Bước 3: Start Frontend

```bash
# Cài đặt dependencies (lần đầu)
cd frontend
npm install

# Chạy dev server
npm run dev
```

Frontend sẽ chạy tại: **http://localhost:5173**

---

## ✨ Sử dụng

### 1. Trang Login
- Email: (Firebase account)
- Password: (Firebase password)
- Popup thông báo khi đăng nhập thành công

### 2. Trang Detect (Upload Ảnh)
1. Chọn "Loại Model": YOLO hoặc CNN
2. Chọn model cụ thể
3. Upload ảnh
4. Nhấn "Bắt đầu Nhận diện"

### 3. Trang Camera (Real-time)
1. Chọn chế độ:
   - **YOLO Only**: Nhanh, real-time
   - **YOLO + CNN**: Chính xác hơn
2. Chọn models
3. Nhấn "Bật Camera"
4. Xem kết quả real-time với FPS counter

---

## 🎯 Test nhanh

### Test Backend API:
```bash
# Kiểm tra server
curl http://localhost:8000/

# Lấy danh sách models
curl http://localhost:8000/models
```

### Test Frontend:
1. Mở http://localhost:5173
2. Login với Firebase account
3. Vào trang Detect hoặc Camera
4. Test nhận diện

---

## 🐛 Troubleshooting nhanh

**Backend không start:**
```bash
# Kiểm tra port 8000 có bị chiếm không
netstat -ano | findstr :8000

# Kill process nếu cần
taskkill /PID <PID> /F
```

**Frontend không start:**
```bash
# Xóa node_modules và cài lại
rm -rf node_modules package-lock.json
npm install
```

**Models không load:**
- Kiểm tra đường dẫn models trong `src/main.py`
- Đảm bảo file models tồn tại
- Kiểm tra quyền đọc file

---

## 📚 Tài liệu chi tiết

- [TWO_STAGE_DETECTION_GUIDE.md](frontend/TWO_STAGE_DETECTION_GUIDE.md) - Hướng dẫn chi tiết
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Tóm tắt implementation

---

*Happy Coding! 🎉*
