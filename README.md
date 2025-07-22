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
│   ├── vehicles/           # Dữ liệu xe
│   └── runs/               # Kết quả training
│       └── detect/         # Kết quả YOLO detection
│           ├── car_yolo112/    # Mô hình phát hiện xe
│           ├── sign_yolo85/    # Mô hình phát hiện biển báo
│           └── sign_yolo85_ft/ # Mô hình fine-tuned
├── models/                 # Các mô hình AI
│   ├── cnn/               # Mô hình CNN
│   └── yolo/              # Mô hình YOLO
├── src/                   # Source code
│   ├── detection/         # Module phát hiện
│   ├── classification/    # Module phân loại
│   ├── training/          # Module training
│   │   ├── notebooks/     # Jupyter notebooks
│   │   ├── scripts/       # Python training scripts
│   │   └── configs/       # Cấu hình training
│   └── utils/             # Tiện ích
├── tests/                 # Test cases
├── roadmap/              # Roadmap và test structures
├── docker/               # Docker configurations
├── docs/                 # Tài liệu
└── world.md              # Lý thuyết tổng hợp
```

## Kết quả Training và Mô hình

### Mô hình Phát hiện Xe (Car Detection)
- **Mô hình**: YOLOv8n
- **Epochs**: 112
- **Dataset**: Custom car dataset
- **Kết quả**: Mô hình đã được training thành công
- **Lưu trữ**: `src/data/runs/detect/car_yolo112/`

### Mô hình Phát hiện Biển báo (Traffic Sign Detection)
- **Mô hình**: YOLOv8s
- **Epochs**: 120 (base) + 20 (fine-tuning)
- **Dataset**: 25 classes traffic signs
- **Kết quả**:
  - **mAP50**: 0.863 (86.3%)
  - **mAP50-95**: 0.593 (59.3%)
- **Lưu trữ**: 
  - Base model: `src/data/runs/detect/sign_yolo85/`
  - Fine-tuned: `src/data/runs/detect/sign_yolo85_ft/`

### Training Scripts và Configs
- **Notebooks**: `src/training/notebooks/Car_Traffic_Detection.ipynb`
- **Python Scripts**: 
  - `src/training/scripts/train_car_detection.py`
  - `src/training/scripts/train_sign_detection.py`
  - `src/training/scripts/evaluate_model.py`
- **Configs**: 
  - `src/training/configs/car_detection_config.yaml`
  - `src/training/configs/sign_detection_config.yaml`

## Công nghệ Sử dụng

- **Deep Learning**: TensorFlow, PyTorch, Ultralytics
- **Computer Vision**: OpenCV, PIL
- **Object Detection**: YOLOv8, YOLOv11
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

### Training Mô hình

#### Sử dụng Python Scripts
```bash
# Training car detection
python src/training/scripts/train_car_detection.py --config src/training/configs/car_detection_config.yaml

# Training sign detection
python src/training/scripts/train_sign_detection.py --config src/training/configs/sign_detection_config.yaml

# Resume training từ checkpoint
python src/training/scripts/train_sign_detection.py --config src/training/configs/sign_detection_config.yaml --resume
```

#### Sử dụng Jupyter Notebook
```bash
# Mở notebook
jupyter notebook src/training/notebooks/Car_Traffic_Detection.ipynb
```

### Đánh giá Mô hình
```bash
# Đánh giá mô hình đơn lẻ
python src/training/scripts/evaluate_model.py --model_path src/data/runs/detect/sign_yolo85/weights/best.pt --data_path src/data/config_yaml/sign_det.yaml

# Đánh giá chi tiết với visualization
python src/training/scripts/evaluate_model.py --model_path src/data/runs/detect/sign_yolo85/weights/best.pt --data_path src/data/config_yaml/sign_det.yaml --detailed

# So sánh nhiều mô hình
python src/training/scripts/evaluate_model.py --compare model1.pt model2.pt --data_path data.yaml
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

Xem [BaseKnowledge.md](BaseKnowledge.md) để có cái nhìn tổng quan về lý thuyết và các khái niệm liên quan đến nhận diện biển báo giao thông và phân loại xe.