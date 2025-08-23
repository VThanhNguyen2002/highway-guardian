# Highway Guardian Scripts

Bộ scripts hỗ trợ cho dự án Highway Guardian - Hệ thống phát hiện xe cộ và biển báo giao thông.

## 📁 Cấu trúc Scripts

```
scripts/
├── README.md                    # Tài liệu này
├── quick_setup.bat             # Setup nhanh cho Windows
├── quick_setup.sh              # Setup nhanh cho Linux/Mac
├── setup_environment.py        # Script setup môi trường
├── dataset_manager.py          # Quản lý dataset
├── training_manager.py         # Quản lý training jobs
├── model_validator.py          # Validation và đánh giá model
└── deploy_model.py             # Deploy model thành API
```

## 🚀 Quick Start

### 1. Setup Môi Trường (Một dòng - Một lần - Tối ưu)

**Windows:**
```bash
.\scripts\quick_setup.bat
```

**Linux/Mac:**
```bash
bash scripts/quick_setup.sh
```

**Hoặc setup thủ công:**
```bash
python scripts/setup_environment.py --mode full
```

### 2. Quản lý Dataset

```bash
# Download dataset từ Kaggle
python scripts/dataset_manager.py download --dataset username/dataset-name --output data/raw

# Validate dataset YOLO
python scripts/dataset_manager.py validate --path data/processed/car_detection

# Split dataset
python scripts/dataset_manager.py split --input data/raw --output data/processed --train 0.7 --val 0.2 --test 0.1

# Thống kê dataset
python scripts/dataset_manager.py stats --path data/processed/car_detection
```

### 3. Training Management

```bash
# Chạy training experiment
python scripts/training_manager.py run --config src/configs/car_det.yaml

# Monitor training
python scripts/training_manager.py monitor --experiment car_detection_20241201_120000

# So sánh experiments
python scripts/training_manager.py compare --experiments exp1,exp2,exp3

# Hyperparameter tuning
python scripts/training_manager.py tune --config src/configs/sign_det_improved.yaml --params '{"lr0": [0.01, 0.001], "batch_size": [16, 32]}'

# List tất cả experiments
python scripts/training_manager.py list
```

### 4. Model Validation

```bash
# Validate single model
python scripts/model_validator.py validate --model runs/detect/exp1/weights/best.pt --data src/configs/car_det.yaml

# So sánh multiple models
python scripts/model_validator.py compare --models model1.pt,model2.pt --data src/configs/sign_det.yaml

# Benchmark performance
python scripts/model_validator.py benchmark --model best.pt --data test_data.yaml --runs 5

# Tạo validation report
python scripts/model_validator.py report --model best.pt --data src/configs/car_det.yaml --output reports/
```

### 5. Model Deployment

```bash
# Export model sang ONNX
python scripts/deploy_model.py export --model best.pt --format onnx

# Start API server
python scripts/deploy_model.py api --model best.pt --host 0.0.0.0 --port 8000

# Tạo Docker image
python scripts/deploy_model.py docker --model best.pt --tag highway-guardian:latest

# Test deployed API
python scripts/deploy_model.py test --url http://localhost:8000 --image test.jpg
```

## 📊 Workflow Hoàn Chỉnh

### Bước 1: Setup Môi Trường
```bash
# Windows
.\scripts\quick_setup.bat

# Chọn mode: full (GPU + dev tools)
```

### Bước 2: Chuẩn Bị Dataset
```bash
# Download và validate dataset
python scripts/dataset_manager.py download --dataset car-traffic-dataset --output data/raw
python scripts/dataset_manager.py validate --path data/raw
python scripts/dataset_manager.py split --input data/raw --output data/processed
```

### Bước 3: Training
```bash
# Train car detection model
python scripts/training_manager.py run --config src/configs/car_det.yaml --name car_detection_v1

# Train improved sign detection model
python scripts/training_manager.py run --config src/configs/sign_det_improved.yaml --name sign_detection_v1

# Monitor training progress
python scripts/training_manager.py monitor
```

### Bước 4: Validation & Comparison
```bash
# Validate trained models
python scripts/model_validator.py validate --model runs/detect/car_detection_v1/weights/best.pt --data src/configs/car_det.yaml
python scripts/model_validator.py validate --model runs/detect/sign_detection_v1/weights/best.pt --data src/configs/sign_det_improved.yaml

# Compare models
python scripts/model_validator.py compare --models runs/detect/car_detection_v1/weights/best.pt,runs/detect/sign_detection_v1/weights/best.pt --data src/configs/car_det.yaml

# Generate detailed report
python scripts/model_validator.py report --model runs/detect/car_detection_v1/weights/best.pt --data src/configs/car_det.yaml
```

### Bước 5: Deployment
```bash
# Export model for production
python scripts/deploy_model.py export --model runs/detect/car_detection_v1/weights/best.pt --format onnx

# Deploy as API
python scripts/deploy_model.py api --model runs/detect/car_detection_v1/weights/best.pt

# Test API
python scripts/deploy_model.py test --url http://localhost:8000 --image test_images/sample.jpg
```

## 🔧 Configuration Files

### Car Detection Config (`src/configs/car_det.yaml`)
```yaml
path: data/processed/car_detection
train: images/train
val: images/val
test: images/test

nc: 4  # number of classes
names: ['car', 'truck', 'bus', 'motorcycle']

# Training parameters
epochs: 100
batch_size: 16
imgsz: 640
device: 0
```

### Improved Sign Detection Config (`src/configs/sign_det_improved.yaml`)
```yaml
path: data/processed/sign_detection
train: images/train
val: images/val
test: images/test

nc: 10  # number of classes
names: ['stop', 'yield', 'speed_limit', 'no_entry', 'warning', 'info', 'mandatory', 'prohibition', 'priority', 'other']

# Optimized training parameters
epochs: 150
batch_size: 32
imgsz: 640
device: 0

# Advanced settings
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights (optimized for signs)
box: 7.5
cls: 0.5
dfl: 1.5

# Class weights for imbalanced dataset
class_weights: [1.0, 1.2, 0.8, 1.5, 1.0, 1.0, 1.1, 1.3, 1.0, 1.4]

# Advanced augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.1
copy_paste: 0.1
```

## 📈 Performance Monitoring

### Weights & Biases Integration
```bash
# Login to W&B
wandb login

# Training sẽ tự động log metrics lên W&B
python scripts/training_manager.py run --config src/configs/car_det.yaml
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir runs/detect

# Mở browser: http://localhost:6006
```

## 🐳 Docker Deployment

### Build và Run Container
```bash
# Build Docker image
python scripts/deploy_model.py docker --model best.pt --tag highway-guardian:v1.0

# Run container
docker run -p 8000:8000 highway-guardian:v1.0

# Test API
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"
```

### Docker Compose (Production)
```yaml
# docker-compose.yml
version: '3.8'
services:
  highway-guardian-api:
    image: highway-guardian:v1.0
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pt
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Giảm batch size
   python scripts/training_manager.py run --config config.yaml --batch-size 8
   ```

2. **Dataset Not Found**
   ```bash
   # Validate dataset structure
   python scripts/dataset_manager.py validate --path data/processed/car_detection
   ```

3. **Model Loading Error**
   ```bash
   # Check model file
   python -c "from ultralytics import YOLO; model = YOLO('model.pt'); print('Model loaded successfully')"
   ```

4. **API Connection Error**
   ```bash
   # Test API health
   curl http://localhost:8000/health
   ```

### Performance Optimization

1. **Training Speed**
   - Sử dụng mixed precision: `--half`
   - Tăng batch size nếu GPU memory đủ
   - Sử dụng multiple GPUs: `--device 0,1,2,3`

2. **Inference Speed**
   - Export sang ONNX: `--format onnx`
   - Sử dụng TensorRT: `--format engine`
   - Optimize image size: `--imgsz 416`

3. **Memory Usage**
   - Giảm batch size
   - Sử dụng gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/new-script`
3. Commit changes: `git commit -am 'Add new script'`
4. Push branch: `git push origin feature/new-script`
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

---

**Highway Guardian Team** 🚗🛡️

*Protecting roads with AI-powered traffic monitoring*