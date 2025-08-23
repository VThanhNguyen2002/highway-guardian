# Highway Guardian - Docker Setup Guide

## Yêu cầu hệ thống

### GPU Support (Khuyến nghị)
- NVIDIA GPU với CUDA 11.8+
- NVIDIA Docker runtime
- Docker Compose v3.8+

### Cài đặt NVIDIA Docker (Windows)
```bash
# Cài đặt NVIDIA Container Toolkit
# Tham khảo: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

## Khởi động nhanh

### 1. Build và chạy container chính
```bash
# Build image
docker-compose build highway-guardian

# Chạy container chính (khởi động một lần)
docker-compose up highway-guardian
```

### 2. Chạy các service riêng biệt

#### Jupyter Notebook
```bash
docker-compose up jupyter
# Truy cập: http://localhost:8888
```

#### TensorBoard
```bash
docker-compose up tensorboard
# Truy cập: http://localhost:6006
```

#### Chạy tất cả services
```bash
docker-compose up -d
```

## Sử dụng container

### Truy cập container
```bash
# Vào container đang chạy
docker exec -it highway-guardian-app bash

# Hoặc chạy lệnh trực tiếp
docker exec -it highway-guardian-app python3 src/scripts/main.py
```

### Training YOLO models
```bash
# Car detection
docker exec -it highway-guardian-app python3 src/training/scripts/train_car_detection.py

# Sign detection
docker exec -it highway-guardian-app python3 src/training/scripts/train_sign_detection.py
```

### Chạy inference
```bash
docker exec -it highway-guardian-app python3 src/scripts/main.py --input /path/to/video.mp4
```

## Cấu trúc volumes

- `./src` → `/app/src` - Source code
- `./data` → `/app/data` - Datasets
- `./models` → `/app/models` - Trained models
- `./outputs` → `/app/outputs` - Results
- `./src/data/runs` → `/app/src/data/runs` - Training logs

## Ports

- `8000` - Main application
- `8501` - Streamlit UI
- `8888` - Jupyter Notebook
- `6006` - TensorBoard

## Troubleshooting

### GPU không được nhận diện
```bash
# Kiểm tra NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### Container không khởi động
```bash
# Xem logs
docker-compose logs highway-guardian

# Rebuild image
docker-compose build --no-cache highway-guardian
```

### Lỗi permissions
```bash
# Fix permissions (Linux/WSL)
sudo chown -R $USER:$USER ./src/data/runs
```

## Development workflow

1. **Khởi động development environment:**
   ```bash
   docker-compose up jupyter tensorboard
   ```

2. **Develop trong Jupyter:** http://localhost:8888

3. **Monitor training:** http://localhost:6006

4. **Test changes:**
   ```bash
   docker exec -it highway-guardian-app python3 src/scripts/main.py
   ```

## Production deployment

```bash
# Chạy trong background
docker-compose up -d highway-guardian

# Scale services nếu cần
docker-compose up -d --scale highway-guardian=2
```

## Cleanup

```bash
# Dừng tất cả services
docker-compose down

# Xóa volumes (cẩn thận!)
docker-compose down -v

# Xóa images
docker rmi $(docker images highway-guardian* -q)
```