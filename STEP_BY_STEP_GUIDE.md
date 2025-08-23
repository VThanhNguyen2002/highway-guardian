# 🚀 Hướng Dẫn Chạy Từng Bước - Highway Guardian

Hướng dẫn chi tiết để chạy từng bước workflow cải thiện Highway Guardian project.

## 📋 Tổng Quan Workflow

```
1. Setup Môi Trường → 2. Chuẩn Bị Dataset → 3. Training → 4. Validation → 5. Deployment
```

---

## 🔧 Bước 1: Setup Môi Trường (Một dòng - Một lần - Tối ưu)

### Option A: Quick Setup (Khuyến nghị)

**Windows:**
```powershell
# Mở PowerShell as Administrator
cd C:\PersonalProject\highway-guardian
.\scripts\quick_setup.bat
```

**Linux/Mac:**
```bash
cd /path/to/highway-guardian
bash scripts/quick_setup.sh
```

### Option B: Manual Setup

```powershell
# Setup với GPU support và dev tools
python scripts/setup_environment.py --mode full

# Hoặc chỉ GPU
python scripts/setup_environment.py --mode gpu

# Hoặc CPU only
python scripts/setup_environment.py --mode basic
```

### ✅ Kiểm Tra Setup

```powershell
# Kiểm tra Python và packages
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8 ready!')"
```

**Expected Output:**
```
Python 3.9.x
PyTorch: 2.1.x
CUDA available: True
YOLOv8 ready!
```

---

## 📊 Bước 2: Chuẩn Bị Dataset

### 2.1 Download Dataset (nếu chưa có)

```powershell
# Setup Kaggle API (nếu chưa có)
kaggle --version

# Download car detection dataset
python scripts/dataset_manager.py download --dataset "car-detection-dataset" --output "src/data/raw/car_detection"

# Download traffic sign dataset
python scripts/dataset_manager.py download --dataset "traffic-signs-dataset" --output "src/data/raw/sign_detection"
```

### 2.2 Validate Dataset Structure

```powershell
# Validate car detection dataset
python scripts/dataset_manager.py validate --path "src/data/raw/car_detection"

# Validate sign detection dataset
python scripts/dataset_manager.py validate --path "src/data/raw/sign_detection"
```

### 2.3 Split Dataset

```powershell
# Split car detection dataset (70% train, 20% val, 10% test)
python scripts/dataset_manager.py split --input "src/data/raw/car_detection" --output "src/data/processed/car_detection" --train 0.7 --val 0.2 --test 0.1

# Split sign detection dataset
python scripts/dataset_manager.py split --input "src/data/raw/sign_detection" --output "src/data/processed/sign_detection" --train 0.7 --val 0.2 --test 0.1
```

### 2.4 Generate Dataset Statistics

```powershell
# Thống kê car detection dataset
python scripts/dataset_manager.py stats --path "src/data/processed/car_detection"

# Thống kê sign detection dataset
python scripts/dataset_manager.py stats --path "src/data/processed/sign_detection"
```

**Expected Output:**
```
📊 Dataset Statistics:
- Total images: 5000
- Train: 3500 (70%)
- Val: 1000 (20%)
- Test: 500 (10%)
- Classes: 4 ['car', 'truck', 'bus', 'motorcycle']
- Annotations: 15000
```

---

## 🎯 Bước 3: Training Models

### 3.1 Setup Weights & Biases (Optional)

```powershell
# Login to W&B for monitoring
wandb login
# Paste your API key when prompted
```

### 3.2 Train Car Detection Model

```powershell
# Train car detection với existing config
python scripts/training_manager.py run --config "src/configs/car_det.yaml" --name "car_detection_improved_v1"
```

### 3.3 Train Improved Sign Detection Model

```powershell
# Train sign detection với improved config
python scripts/training_manager.py run --config "src/configs/sign_det_improved.yaml" --name "sign_detection_improved_v1"
```

### 3.4 Monitor Training Progress

**Option A: Real-time monitoring**
```powershell
# Monitor specific experiment
python scripts/training_manager.py monitor --experiment "sign_detection_improved_v1"

# Monitor all active trainings
python scripts/training_manager.py monitor
```

**Option B: TensorBoard**
```powershell
# Start TensorBoard
tensorboard --logdir "src/data/runs/detect"
# Mở browser: http://localhost:6006
```

**Option C: Weights & Biases**
- Mở https://wandb.ai/your-username/highway-guardian

### 3.5 List All Experiments

```powershell
python scripts/training_manager.py list
```

**Expected Output:**
```
📋 Found 2 experiments:
======================================================================
Name                                     Date                 Status    
----------------------------------------------------------------------
sign_detection_improved_v1               2024-12-01 15:30:25  ✅ Done   
car_detection_improved_v1                2024-12-01 14:20:15  ✅ Done   
```

---

## 📈 Bước 4: Validation và Comparison

### 4.1 Validate Individual Models

```powershell
# Validate car detection model
python scripts/model_validator.py validate --model "src/data/runs/detect/car_detection_improved_v1/weights/best.pt" --data "src/configs/car_det.yaml"

# Validate sign detection model
python scripts/model_validator.py validate --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --data "src/configs/sign_det_improved.yaml"
```

**Expected Output:**
```
🔍 Validating model: best.pt
📊 Data config: sign_det_improved.yaml
🚀 Running validation...
✅ Validation completed
📊 mAP50: 0.756
📊 mAP50-95: 0.445
📊 Precision: 0.721
📊 Recall: 0.689
```

### 4.2 Compare Multiple Models

```powershell
# So sánh car detection models
python scripts/model_validator.py compare --models "src/data/runs/detect/car_detection_improved_v1/weights/best.pt,src/data/runs/detect/car_detection_v2/weights/best.pt" --data "src/configs/car_det.yaml"

# So sánh sign detection models
python scripts/model_validator.py compare --models "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt,src/data/runs/detect/sign_detection_v2/weights/best.pt" --data "src/configs/sign_det_improved.yaml"
```

### 4.3 Benchmark Performance

```powershell
# Benchmark với 5 runs để đánh giá stability
python scripts/model_validator.py benchmark --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --data "src/configs/sign_det_improved.yaml" --runs 5
```

### 4.4 Generate Detailed Report

```powershell
# Tạo HTML report chi tiết
python scripts/model_validator.py report --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --data "src/configs/sign_det_improved.yaml" --output "validation_results/reports"
```

**Report sẽ được tạo tại:** `validation_results/reports/validation_report_best_YYYYMMDD_HHMMSS.html`

---

## 🚀 Bước 5: Model Deployment

### 5.1 Export Models for Production

```powershell
# Export car detection model to ONNX
python scripts/deploy_model.py export --model "src/data/runs/detect/car_detection_improved_v1/weights/best.pt" --format onnx --imgsz 640

# Export sign detection model to ONNX
python scripts/deploy_model.py export --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --format onnx --imgsz 640

# Export to TensorRT (nếu có NVIDIA GPU)
python scripts/deploy_model.py export --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --format engine --imgsz 640
```

### 5.2 Start API Server

```powershell
# Start API server cho sign detection
python scripts/deploy_model.py api --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --host 0.0.0.0 --port 8000
```

**API sẽ chạy tại:** http://localhost:8000

### 5.3 Test API

**Terminal mới:**
```powershell
# Test API với sample image
python scripts/deploy_model.py test --url "http://localhost:8000" --image "test_images/sample.jpg"
```

**Hoặc test bằng curl:**
```powershell
# Test health endpoint
curl http://localhost:8000/health

# Test prediction với image
curl -X POST "http://localhost:8000/predict" -F "file=@test_images/sample.jpg"
```

### 5.4 Create Docker Image

```powershell
# Tạo Docker image
python scripts/deploy_model.py docker --model "src/data/runs/detect/sign_detection_improved_v1/weights/best.pt" --tag "highway-guardian:v1.0"

# Run Docker container
docker run -p 8000:8000 highway-guardian:v1.0
```

---

## 🔧 Bước 6: Advanced Features

### 6.1 Hyperparameter Tuning

```powershell
# Tune hyperparameters cho sign detection
python scripts/training_manager.py tune --config "src/configs/sign_det_improved.yaml" --params '{"lr0": [0.01, 0.001, 0.0001], "batch_size": [16, 32], "epochs": [100, 150]}'
```

### 6.2 Compare Tuning Results

```powershell
# So sánh kết quả tuning
python scripts/training_manager.py compare --experiments "tune_001_20241201_120000,tune_002_20241201_130000,tune_003_20241201_140000"
```

---

## 📊 Bước 7: Đánh Giá Kết Quả

### 7.1 Expected Improvements

**Before (Original):**
- Car Detection: mAP50-95 = 0.849
- Sign Detection: mAP50-95 = 0.289

**After (Improved):**
- Car Detection: mAP50-95 = 0.860+ (1.3% improvement)
- Sign Detection: mAP50-95 = 0.450+ (55% improvement)

### 7.2 Key Improvements

1. **Sign Detection:**
   - ✅ Weighted loss cho class imbalance
   - ✅ Advanced data augmentation
   - ✅ Early stopping
   - ✅ Learning rate scheduling
   - ✅ Better validation metrics

2. **Code Structure:**
   - ✅ Modular training scripts
   - ✅ Configuration management
   - ✅ Comprehensive logging
   - ✅ Error handling

3. **Deployment:**
   - ✅ Production-ready API
   - ✅ Docker containerization
   - ✅ Model export options
   - ✅ Performance monitoring

---

## 🔍 Troubleshooting

### Common Issues và Solutions

**1. CUDA Out of Memory**
```powershell
# Giảm batch size
python scripts/training_manager.py run --config "src/configs/sign_det_improved.yaml" --batch-size 8
```

**2. Dataset Not Found**
```powershell
# Kiểm tra dataset structure
python scripts/dataset_manager.py validate --path "src/data/processed/sign_detection"
```

**3. Model Loading Error**
```powershell
# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('src/data/runs/detect/sign_detection_improved_v1/weights/best.pt'); print('Model loaded successfully')"
```

**4. API Connection Error**
```powershell
# Test API health
curl http://localhost:8000/health
```

**5. Training Stuck**
```powershell
# Check GPU usage
nvidia-smi

# Check training logs
python scripts/training_manager.py monitor --experiment "your_experiment_name"
```

---

## 📈 Performance Monitoring

### Real-time Monitoring

1. **Weights & Biases:** https://wandb.ai/your-username/highway-guardian
2. **TensorBoard:** http://localhost:6006
3. **Training Manager:** `python scripts/training_manager.py monitor`

### Key Metrics to Watch

- **mAP50-95:** Overall detection accuracy
- **Precision:** False positive rate
- **Recall:** False negative rate
- **Loss curves:** Training convergence
- **Learning rate:** Optimization progress

---

## 🎯 Next Steps

1. **Optimize Further:**
   - Experiment với different model sizes (YOLOv8s, YOLOv8m, YOLOv8l)
   - Try ensemble methods
   - Implement test-time augmentation

2. **Production Deployment:**
   - Setup CI/CD pipeline
   - Implement model versioning
   - Add monitoring và alerting

3. **Data Enhancement:**
   - Collect more diverse data
   - Implement active learning
   - Add synthetic data generation

---

## 📞 Support

Nếu gặp vấn đề, hãy:
1. Kiểm tra logs trong `validation_results/`
2. Xem TensorBoard cho training metrics
3. Check GPU memory với `nvidia-smi`
4. Validate dataset structure

**Happy Training! 🚗🛡️**