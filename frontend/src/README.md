# Highway Guardian 🛡️
Dự án nhận diện biển báo giao thông sử dụng YOLOv8, FastAPI và Vue.js.

## Giới thiệu
Highway Guardian là một hệ thống web cho phép người dùng tải lên hình ảnh hoặc sử dụng webcam để phát hiện và phân loại các biển báo giao thông trong thời gian thực. Hệ thống được xây dựng với kiến trúc microservice, bao gồm:
* **Backend:** API FastAPI (Python) để xử lý logic AI/ML với model YOLOv8.
* **Frontend:** Giao diện người dùng (UI) tương tác được xây dựng bằng Vue.js.
* **Deployment:** Toàn bộ ứng dụng được đóng gói và quản lý bằng Docker.

## Yêu cầu cài đặt
Để chạy dự án, bạn cần cài đặt các công cụ sau:
1.  [**Node.js**](https://nodejs.org/) (phiên bản 18+): Môi trường để chạy và build Vue.js.
2.  [**Docker**](https://www.docker.com/products/docker-desktop/): Để chạy ứng dụng trong môi trường production.
3.  [**Python**](https://www.python.org/) (phiên bản 3.9+): Chỉ cần nếu bạn muốn chạy backend ở chế độ development (không qua Docker).

---

## 1. Hướng dẫn chạy (Phát triển - Development)
Chế độ này cho phép "Hot Reload" (tự động cập nhật khi bạn sửa code), lý tưởng cho việc lập trình. Bạn sẽ cần chạy 2 terminal riêng biệt.

### A. Chạy Backend (API)
(Giả sử bạn đã cài Python và các thư viện trong `requirements.txt`)

1.  Mở terminal 1 tại thư mục gốc `HIGHWAY-GUARDIAN`:
    ```bash
    # Cài đặt thư viện (chỉ làm lần đầu)
    pip install -r requirements.txt

    # Khởi động server FastAPI
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
2.  API sẽ chạy tại `http://localhost:8000`.

### B. Chạy Frontend (Vue.js)

1.  Mở terminal 2, di chuyển vào thư mục `frontend`:
    ```bash
    cd frontend
    ```
2.  Cài đặt thư viện (chỉ làm lần đầu):
    ```bash
    npm install
    ```
3.  Khởi động server phát triển Vite:
    ```bash
    npm run dev
    ```
4.  Trang web sẽ chạy tại `http://localhost:5173` (hoặc một port khác do Vite chỉ định).

---

## 2. Hướng dẫn chạy (Production - Docker)
Chế độ này sẽ build (biên dịch) code của bạn thành phiên bản tối ưu và chạy chúng bên trong các container biệt lập. Đây là cách bạn deploy ứng dụng.

**Yêu cầu:** Đảm bảo **Docker Desktop** đang chạy.

1.  Mở terminal tại thư mục gốc `HIGHWAY-GUARDIAN`.
2.  Chạy lệnh `docker-compose` để build và khởi động tất cả dịch vụ:
    ```bash
    docker-compose up --build
    ```
    * `--build`: Báo cho Docker build lại image nếu có thay đổi trong `Dockerfile` hoặc source code. (Bạn có thể bỏ `--build` ở những lần chạy sau nếu không thay đổi code).

3.  Đợi Docker build xong.
4.  Truy cập ứng dụng tại: **`http://localhost:8080`**

### Các lệnh Docker hữu ích khác

* Để tắt ứng dụng (tắt và xóa container):
    ```bash
    docker-compose down
    ```
* Để chạy ngầm (không chiếm terminal):
    ```bash
    docker-compose up -d
    ```
* Để xem log (khi chạy ngầm):
    ```bash
    docker-compose logs -f frontend
    docker-compose logs -f backend
    ```