# Highway Guardian - Hệ thống Nhận diện Biển báo Giao thông và Phân loại Xe

## Tổng quan Dự án

Highway Guardian là một hệ thống AI tiên tiến được thiết kế để nhận diện biển báo giao thông và phân loại các loại xe trên đường. Dự án sử dụng các mô hình deep learning hiện đại để đảm bảo độ chính xác cao trong việc phát hiện và phân loại.

## Mục tiêu Chính

### 1. Nhận diện Biển báo Giao thông
- **Mô hình sử dụng**: CNN và YOLOv8 (có thể nâng cấp lên các phiên bản YOLOv khác)
- **Tập dữ liệu**: Biển báo giao thông Việt Nam và quốc tế
- **Phương pháp**: 
  - Phát hiện và nhận diện mã ID của từng biển báo
  - Chuyển đổi mã ID thành loại biển báo cụ thể
  - Ví dụ: P.12 → Biển báo cấm
- **Output cuối**: Phân loại loại biển báo (cấm, báo hiệu, chỉ dẫn, etc.)

### 2. Phân loại Xe
- **Phân loại cơ bản**: Xe hơi, xe tải, xe máy, xe buýt
- **Phân loại nâng cao**: Nhận diện thương hiệu xe (Toyota, BMW, Honda, etc.)
- **Đặc điểm nhận diện**: Hình dáng, kích thước, logo thương hiệu

### 3. Roadmap Phát triển
- **Phase 1**: Xây dựng mô hình cơ bản cho biển báo
- **Phase 2**: Tích hợp phân loại xe
- **Phase 3**: Tối ưu hóa và triển khai
- **Phase 4**: Mở rộng tính năng và cải thiện độ chính xác

### 4. Quản lý Môi trường
- **Docker**: Sử dụng containerization để quản lý dependencies
- **Lợi ích**: Tiết kiệm dung lượng ổ đĩa, dễ dàng triển khai
- **Hỗ trợ**: Laptop cấu hình thấp

## Cấu trúc Dự án

```
highway-guardian/
├── data/                    # Tập dữ liệu
│   ├── traffic_signs/      # Dữ liệu biển báo
│   └── vehicles/           # Dữ liệu xe
├── models/                 # Các mô hình AI
│   ├── cnn/               # Mô hình CNN
│   └── yolo/              # Mô hình YOLO
├── src/                   # Source code
│   ├── detection/         # Module phát hiện
│   ├── classification/    # Module phân loại
│   └── utils/             # Tiện ích
├── tests/                 # Test cases
├── roadmap/              # Roadmap và test structures
├── docker/               # Docker configurations
├── docs/                 # Tài liệu
└── world.md              # Lý thuyết tổng hợp
```

## Công nghệ Sử dụng

- **Deep Learning**: TensorFlow, PyTorch
- **Computer Vision**: OpenCV, PIL
- **Object Detection**: YOLOv8, YOLO variants
- **CNN Frameworks**: Keras, TensorFlow
- **Containerization**: Docker
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## Yêu cầu Hệ thống

- Python 3.8+
- Docker (khuyến nghị)
- GPU support (tùy chọn, để tăng tốc training)
- Minimum 8GB RAM
- 10GB+ dung lượng ổ đĩa

## Cài đặt và Sử dụng

### Sử dụng Docker (Khuyến nghị)
```bash
# Clone repository
git clone https://github.com/VThanhNguyen2002/highway-guardian.git
cd highway-guardian

# Build Docker image
docker build -t highway-guardian .

# Run container
docker run -it highway-guardian
```

### Cài đặt Local
```bash
# Clone repository
git clone https://github.com/VThanhNguyen2002/highway-guardian.git
cd highway-guardian

# Install dependencies
pip install -r requirements.txt

# Run application
python src/main.py
```

## Đóng góp

Chúng tôi hoan nghênh mọi đóng góp cho dự án. Vui lòng đọc [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem [LICENSE](LICENSE) để biết thêm thông tin.

## Liên hệ

- **Tác giả**: VThanhNguyen2002
- **Email**: [vietthanhnguyen2006@gmail.com]
- **GitHub**: [https://github.com/VThanhNguyen2002](https://github.com/VThanhNguyen2002)

## Tài liệu Tham khảo

Xem [world.md](world.md) để có cái nhìn tổng quan về lý thuyết và các khái niệm liên quan đến nhận diện biển báo giao thông và phân loại xe.