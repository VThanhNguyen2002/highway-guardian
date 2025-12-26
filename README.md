# Highway Guardian - Hướng Dẫn Dự Án Toàn Tập

## 1. Tổng quan Dự án
Highway Guardian là hệ thống AI nhận diện biển báo giao thông và phân loại xe sử dụng Deep Learning (YOLOv8 & CNN).
- **Tính năng chính**:
    - Nhận diện biển báo giao thông (với mã ID và loại biển).
    - Phân loại xe (xe hơi, xe tải, xe máy, v.v.).
    - Nhận diện Real-time qua Camera.
    - Dashboard quản lý đẹp mắt.

## 2. Cài đặt và Chạy Nhanh (Quick Start)

### Yêu cầu
- Python 3.8+
- Node.js & npm
- Docker (Khuyến nghị, tùy chọn)

### Chạy bằng Script (Khuyến nghị)
Sử dụng file `run.bat` ở thư mục gốc (sau khi dọn dẹp xong):
```bash
.\run.bat
```
Chọn các tùy chọn:
1. **Start ALL (Dev Mode)**: Chạy cả Backend và Frontend.
2. **Start Backend Only**: Chạy server Python (Port 8000).
3. **Start Frontend Only**: Chạy giao diện Web (Port 5173).

### Chạy Thủ công
**Backend:**
```bash
cd src
pip install -r requirements.txt
python main.py
```
Backend: http://localhost:8000

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

## 3. Cấu trúc Dự án
```
highway-guardian/
├── data/                    # Tập dữ liệu (biển báo, xe)
├── models/                  # File Weights (YOLO .pt, CNN .h5)
├── src/                     # Source code Backend (Python/FastAPI)
├── frontend/                # Source code Frontend (Vue/React/Vite)
├── scripts/                 # Các script tiện ích (.bat, .py)
├── docs/                    # Tài liệu chi tiết khác
└── run.bat                  # Script chạy chính
```

## 4. Training Model
Các script training đã được chuyển vào thư mục `scripts/` hoặc `src/training/ scripts/`.
- Để train model nhận diện xe: `python src/training/scripts/train_car_detection.py`
- Để train model biển báo: `python src/training/scripts/train_sign_detection.py`

## 5. Troubleshooting (Sửa lỗi thường gặp)
- **Backend không chạy**: Kiểm tra port 8000 có bị chiếm không (`netstat -ano | findstr :8000`). Kill process nếu cần.
- **Frontend lỗi node_modules**: Xóa thư mục `node_modules` và chạy lại `npm install`.
- **Lỗi thiếu model**: Đảm bảo file `.pt` và `.h5` nằm đúng vị trí trong thư mục `models/`.

## 6. Liên hệ
- **Tác giả**: VThanhNguyen2002
- **Email**: vietthanhnguyen2006@gmail.com
- **GitHub**: https://github.com/VThanhNguyen2002
