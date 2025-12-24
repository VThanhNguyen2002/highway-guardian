# Lộ trình 5 Giai đoạn: Xây dựng Ứng dụng Nhận diện Biển báo Giao thông (YOLOv8)

Dưới đây là kế hoạch chi tiết, từng bước để xây dựng một hệ thống nhận diện biển báo giao thông hoàn chỉnh, từ việc chuẩn bị dữ liệu trên Kaggle đến triển khai một ứng dụng web có khả năng xử lý video trực tiếp.

---

## 🧠 Giai đoạn 1: Chuẩn bị dữ liệu (Data Foundation)

**Mục tiêu:** Khắc phục vấn đề **Recall thấp** (bỏ sót) của các lớp `yield` và `speed_limit` bằng cách bổ sung dữ liệu đa dạng và thực tế.

### 1. Chọn Bộ Dữ Liệu
*   **Dữ liệu gốc (Việt Nam):** `jaydenguyenx/vietnamese-traffic-signs-detection-and-recognition`
*   **Dữ liệu Bổ sung (Quốc tế):** `mapillary/mapillary-traffic-sign-dataset-in-yolo-format`. Bộ dữ liệu này rất lớn và đa dạng (ảnh nhỏ, mờ, lóa, khuất), lý tưởng để cải thiện Recall.

### 2. Tải Dữ Liệu trên Kaggle
Sử dụng Kaggle Notebook (với GPU) để tải dữ liệu:

```bash
# Tải dữ liệu gốc (Việt Nam)
!kaggle datasets download -d jaydenguyenx/vietnamese-traffic-signs-detection-and-recognition -p /kaggle/working/data/vietnam --unzip

# Tải dữ liệu bổ sung (Mapillary)
!kaggle datasets download -d mapillary/mapillary-traffic-sign-dataset-in-yolo-format -p /kaggle/working/data/mapillary --unzip
```

### 3. Tiền xử lý: Đồng bộ hóa Lớp (Class Harmonization)
Đây là bước **QUAN TRỌNG NHẤT**. Chúng ta phải viết một script Python để xử lý các tệp nhãn (`.txt`) của bộ Mapillary:

*   Đọc file `data.yaml` của Mapillary để tìm chỉ số lớp (ví dụ: `speed_limit_30` là `50`, `yield` là `80`).
*   Quét tất cả các tệp `.txt` trong `mapillary/train/labels` và `mapillary/valid/labels`.
*   Viết lại các chỉ số lớp:
    *   Nếu lớp là `50` (hoặc lớp tốc độ khác) -> đổi thành `0` (lớp `speed_limit` mới).
    *   Nếu là `80` (`yield`) -> đổi thành `1` (lớp `yield` mới).
    *   Nếu là `mandatory_...` -> đổi thành `2` (lớp `mandatory` mới).
    *   ...
*   Xóa bỏ các dòng (bounding box) có lớp không liên quan đến 4 lớp mục tiêu của chúng ta.

### 4. Tạo Tệp YAML Hợp nhất
Tạo file `combined_data.yaml` để báo cho YOLO huấn luyện trên cả hai bộ dữ liệu đã chuẩn hóa.

```yaml
# combined_data.yaml
path: /kaggle/working/data

# Chỉ định cả hai thư mục huấn luyện
train:
  - vietnam/train/images
  - mapillary/train/images

# Chỉ định cả hai thư mục validation
val:
  - vietnam/valid/images
  - mapillary/valid/images

# 4 lớp mục tiêu của chúng ta
nc: 4
names: ['speed_limit', 'yield', 'mandatory', 'other']
```

---

## 🚀 Giai đoạn 2: Huấn luyện & Phân tích (Train & Analyze)

**Mục tiêu:** Huấn luyện một mô hình mạnh mẽ và hiểu rõ hiệu suất của nó.

### 1. Lựa chọn Mô hình
*   `yolov8n.pt` (Nano): Nhanh nhất (cho live video trên thiết bị yếu).
*   `yolov8s.pt` (Small): Lựa chọn cân bằng (Recommended). Tốt cho web app.
*   `yolov8m.pt` (Medium): Chính xác nhất (ưu tiên phát hiện).

Chúng ta sẽ dùng `yolov8s.pt` để cân bằng giữa độ chính xác và tốc độ.

### 2. Huấn luyện (trên Kaggle)
Chạy huấn luyện với mô hình 'small' trên bộ dữ liệu hợp nhất.

```python
from ultralytics import YOLO

# Tải mô hình yolov8s (small)
model = YOLO('yolov8s.pt') 

# Bắt đầu huấn luyện
results = model.train(
    data='combined_data.yaml',
    epochs=75,         # Tăng epochs cho bộ dữ liệu lớn
    imgsz=640,
    batch=32,          # Điều chỉnh nếu 'out of memory'
    project='runs/detect',
    name='sign_detector_v3',
    exist_ok=True,
    device=[0, 1]      # Dùng tất cả GPU có sẵn
)
```

### 3. Phân tích Kết quả (Visualize)
Sau khi huấn luyện, kiểm tra thư mục `runs/detect/sign_detector_v3/`:

*   **Ma trận nhầm lẫn (`confusion_matrix.png`):** Kiểm tra hàng `background`. Các số FN (False Negatives) của `yield` và `speed_limit` phải giảm mạnh so với phiên bản trước.
*   **Đường cong (`results.csv`):** `metrics/mAP50-95` và `metrics/recall` phải cao hơn bản cũ một cách rõ rệt.
*   **Ảnh Dự đoán (`val_batch*_pred.jpg`):** Kiểm tra trực quan xem mô hình có bắt được các biển báo nhỏ/mờ mà trước đây nó bỏ sót không.

### 4. Tải về "Thành phẩm"
Tải về tệp quan trọng nhất: `runs/detect/sign_detector_v3/weights/best.pt`. Đây chính là "bộ não" cho ứng dụng web.

---

## 🏗️ Giai đoạn 3: Xây dựng Ứng dụng (Detect Ảnh)

**Mục tiêu:** Xây dựng một trang web Flask cơ bản cho phép người dùng tải ảnh lên và nhận diện.

### 1. Backend (Flask)
*   Tạo `app.py`.
*   Tải mô hình `best.pt` **MỘT LẦN** khi khởi động.
*   Tạo endpoint `/predict` (nhận tệp ảnh).
*   Xử lý ảnh qua `model(image)`.
*   Trả về kết quả (tọa độ box, tên lớp, độ tự tin) dưới dạng JSON.

### 2. Frontend (HTML/JS)
*   Tạo `templates/index.html`.
*   Dùng `fetch()` API của JavaScript để gửi ảnh đến `/predict`.
*   Nhận JSON trả về.
*   Dùng Canvas để vẽ ảnh gốc và các bounding box kết quả lên trên.

---

## 🎥 Giai đoạn 4: Mở rộng (Detect Video & Live)

**Mục tiêu:** Nâng cấp ứng dụng để xử lý nhiều loại phương tiện hơn.

### 1. Xử lý Video Tải lên (File Video)
*   Tạo endpoint mới, ví dụ `/predict_video`.
*   Backend nhận file video, dùng OpenCV để đọc video từng khung hình.
*   Cho từng khung hình qua `model.predict()`.
*   Dùng OpenCV để vẽ lại các box lên khung hình, ghi ra file video mới.
*   Frontend cho phép người dùng tải video kết quả về.

### 2. Xử lý Video Trực tiếp (Live Webcam)
#### Hướng 1 (Dùng Server): WebSockets
*   **Frontend (JS):** Gửi khung hình webcam đến server.
*   **Backend (Flask):** Xử lý YOLO, gửi JSON kết quả về.
*   **Frontend (JS):** Nhận JSON, vẽ box.
*   **Nhược điểm:** Tốn tài nguyên server, có độ trễ (latency).

#### Hướng 2 (Khuyến khích - Dùng Trình duyệt): TensorFlow.js
*   **Chuẩn bị (1 lần):** Bạn cần xuất (export) mô hình `best.pt` của mình sang định dạng web:
    ```python
    model = YOLO('best.pt')
    model.export(format='tfjs') # Sẽ tạo ra 1 thư mục với file model.json
    ```
*   **Frontend (HTML/JS):** Tải TensorFlow.js và tải file `model.json`.
*   **Frontend (JS):** Lấy video từ webcam và chạy mô hình ngay trong trình duyệt của người dùng.
*   **Ưu điểm:** Cực nhanh (real-time), không tốn tài nguyên server.

---

## 📦 Giai đoạn 5: Hoàn thiện (Conclusion & Theory)

**Mục tiêu:** Đóng gói và ghi lại dự án một cách chuyên nghiệp.

### 1. Kết luận & Lý thuyết (README.md)
*   Viết một file `README.md` mới (giống như file này) cho dự án của bạn.
*   Trình bày vấn đề ban đầu (Recall thấp).
*   Giải thích giải pháp (bổ sung dữ liệu Mapillary).
*   Trình bày kết quả (dán ảnh `confusion_matrix.png` mới, bảng so sánh mAP trước và sau).
*   Phân tích các hạn chế của mô hình (ví dụ: vẫn nhận diện sai trong trời tối...).

### 2. Sắp xếp & Đóng gói
*   **Docker:** Viết một `Dockerfile` để đóng gói toàn bộ ứng dụng Flask (`app.py`, `model.pt`, `templates`). Điều này giúp bất kỳ ai cũng có thể chạy ứng dụng của bạn.
*   **Tệp `requirements.txt`:** Liệt kê tất cả các thư viện Python cần thiết (`flask`, `ultralytics`, `opencv-python`, ...).
