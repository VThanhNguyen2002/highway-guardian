# Hướng dẫn sử dụng Car Detection Training trên Kaggle

## Tổng quan

Hướng dẫn này cung cấp code hoàn chỉnh để training car detection trên Kaggle với chế độ offline, không cần kết nối internet.

## Code hoàn chỉnh cho Kaggle Notebook

### Cell 1: Environment Setup - Check GPU
```python
# Check GPU availability
import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### Cell 2: Install Required Packages (Offline-Safe)
```python
# Install required packages with offline fallback
import subprocess
import sys
import os

def install_package_safe(package, fallback_msg=None):
    """Install package with network error handling for Kaggle offline mode"""
    try:
        __import__(package.replace('-', '_'))  # Handle package name differences
        print(f"✅ {package} already available")
        return True
    except ImportError:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                timeout=60)  # Add timeout
            print(f"✅ {package} installed successfully")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"❌ Failed to install {package}: {e}")
            if fallback_msg:
                print(f"💡 {fallback_msg}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error installing {package}: {e}")
            return False

# Try to install packages with fallbacks
print("=== PACKAGE INSTALLATION ===")

# Ultralytics - most important
ultralytics_ok = install_package_safe('ultralytics', 
    "Fallback: You can manually upload ultralytics wheel file or use pre-installed version")

# OpenCV - usually available in Kaggle
opencv_ok = install_package_safe('opencv-python', 
    "Fallback: OpenCV is usually pre-installed in Kaggle environment")

# Wandb - optional for logging
wandb_ok = install_package_safe('wandb', 
    "Fallback: WandB is optional, training will work without it")

# Check critical packages
if not ultralytics_ok:
    print("\n⚠️  WARNING: Ultralytics not available!")
    print("Solutions:")
    print("1. Enable internet in Kaggle notebook settings")
    print("2. Upload ultralytics wheel file as dataset")
    print("3. Use Kaggle's pre-installed packages")
    
print(f"\n📊 Installation Summary:")
print(f"  Ultralytics: {'✅' if ultralytics_ok else '❌'}")
print(f"  OpenCV: {'✅' if opencv_ok else '❌'}")
print(f"  WandB: {'✅' if wandb_ok else '❌'}")
```

### Cell 3: Import Libraries (with Error Handling)
```python
import os
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Image, display
import numpy as np

# Import with error handling
try:
    import yaml
except ImportError:
    print("⚠️ PyYAML not available, using basic dict for config")
    yaml = None

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics imported successfully")
except ImportError as e:
    print(f"❌ Ultralytics import failed: {e}")
    print("💡 Please check package installation or enable internet")
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    import torch
    TORCH_AVAILABLE = True
    print(f"✅ PyTorch {torch.__version__} available")
except ImportError:
    print("❌ PyTorch not available")
    TORCH_AVAILABLE = False
    torch = None

try:
    import cv2
    CV2_AVAILABLE = True
    print(f"✅ OpenCV {cv2.__version__} available")
except ImportError:
    print("❌ OpenCV not available")
    CV2_AVAILABLE = False
    cv2 = None

# Set device
if TORCH_AVAILABLE:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("⚠️ PyTorch not available, defaulting to CPU")

# Check critical dependencies
print("\n📋 Dependency Check:")
print(f"  PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
print(f"  Ultralytics: {'✅' if ULTRALYTICS_AVAILABLE else '❌'}")
print(f"  OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
print(f"  PyYAML: {'✅' if yaml else '❌'}")

if not ULTRALYTICS_AVAILABLE:
    print("\n🛑 CRITICAL: Cannot proceed without Ultralytics!")
    print("Please resolve package installation issues before continuing.")
```

### Cell 4: Dataset Setup - Check Dataset
```python
# Dataset paths for Kaggle environment
# Assuming dataset is added as input data
DATASET_PATH = '/kaggle/input/car-detection-dataset'
WORK_DIR = '/kaggle/working'

# Check if dataset exists
if os.path.exists(DATASET_PATH):
    print(f"Dataset found at: {DATASET_PATH}")
    print(f"Dataset contents: {os.listdir(DATASET_PATH)}")
else:
    print("Dataset not found! Please add 'car-detection-dataset' as input data.")
    print("Available input data:")
    if os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            print(f"  - {item}")
```

### Cell 5: Extract and Organize Dataset
```python
# Extract and organize dataset
import zipfile

# Create working directories
data_dir = os.path.join(WORK_DIR, 'data', 'car_detection')
os.makedirs(data_dir, exist_ok=True)

# Find and extract zip file
zip_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith('.zip'):
            zip_files.append(os.path.join(root, file))

if zip_files:
    print(f"Found zip files: {zip_files}")
    
    # Extract the first zip file
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print(f"Dataset extracted to: {data_dir}")
    print(f"Extracted contents: {os.listdir(data_dir)}")
else:
    print("No zip files found. Checking for direct dataset structure...")
    # Copy dataset structure if already extracted
    if os.path.exists(DATASET_PATH):
        shutil.copytree(DATASET_PATH, data_dir, dirs_exist_ok=True)
        print(f"Dataset copied to: {data_dir}")
```

### Cell 6: Find Dataset Structure
```python
# Find dataset structure
def find_dataset_structure(base_path):
    """Find train, val, test directories in dataset"""
    structure = {}
    
    for root, dirs, files in os.walk(base_path):
        for dir_name in ['train', 'val', 'test', 'valid']:
            if dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                if 'images' in os.listdir(full_path) or any(f.endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(full_path)):
                    structure[dir_name] = full_path
    
    return structure

dataset_structure = find_dataset_structure(data_dir)
print(f"Dataset structure found: {dataset_structure}")
```

### Cell 7: Create YAML Configuration
```python
# Create YAML configuration for YOLO
config = {
    'path': data_dir,
    'train': 'train/images' if 'train' in dataset_structure else 'images',
    'val': 'val/images' if 'val' in dataset_structure else ('valid/images' if 'valid' in dataset_structure else 'images'),
    'test': 'test/images' if 'test' in dataset_structure else 'images',
    
    'nc': 1,  # number of classes
    'names': ['car']  # class names
}

# Save configuration
config_path = os.path.join(WORK_DIR, 'car_detection_kaggle.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Configuration saved to: {config_path}")
print("Configuration content:")
with open(config_path, 'r') as f:
    print(f.read())
```

### Cell 8: Model Training Setup
```python
# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Start with nano model for faster training

# Training parameters optimized for Kaggle
train_params = {
    'data': config_path,
    'epochs': 30,  # Reduced epochs to avoid timeout
    'batch': 16,   # Optimized batch size for GPU memory
    'imgsz': 640,
    'device': device,
    'workers': 2,  # Reduced workers for Kaggle
    'project': 'runs_detect',
    'name': 'car_detection_kaggle',
    'save': True,
    'save_period': 5,  # Save checkpoint every 5 epochs
    'patience': 10,    # Early stopping patience
    'plots': True,
    'verbose': True,
    'val': True,
    'cache': False,    # Disable cache to save memory
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5
}

print("Training parameters:")
for key, value in train_params.items():
    print(f"  {key}: {value}")
```

### Cell 9: Start Training
```python
# Start training with error handling
try:
    print("Starting training...")
    results = model.train(**train_params)
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    print("Checking for saved checkpoints...")
    
    # List available checkpoints
    checkpoint_dir = os.path.join(WORK_DIR, 'runs_detect', 'car_detection_kaggle', 'weights')
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        print(f"Available checkpoints: {checkpoints}")
    else:
        print("No checkpoints found.")
```

### Cell 10: Display Training Results
```python
# Display training results
results_dir = os.path.join(WORK_DIR, 'runs_detect', 'car_detection_kaggle')

if os.path.exists(results_dir):
    print(f"Results directory: {results_dir}")
    print(f"Contents: {os.listdir(results_dir)}")
    
    # Display training plots
    plot_files = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png', 'PR_curve.png']
    
    for plot_file in plot_files:
        plot_path = os.path.join(results_dir, plot_file)
        if os.path.exists(plot_path):
            print(f"\n{plot_file}:")
            display(Image(plot_path))
        else:
            print(f"Plot not found: {plot_file}")
else:
    print("Results directory not found.")
```

### Cell 11: Load Best Model and Validate
```python
# Load best model and show metrics
best_model_path = os.path.join(WORK_DIR, 'runs_detect', 'car_detection_kaggle', 'weights', 'best.pt')

if os.path.exists(best_model_path):
    print(f"Best model found at: {best_model_path}")
    
    # Load the best model
    best_model = YOLO(best_model_path)
    
    # Validate the model
    print("\nValidating best model...")
    val_results = best_model.val(data=config_path)
    
    print(f"\nValidation Results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
else:
    print("Best model not found. Training may have failed or is incomplete.")
```

### Cell 12: Test Model on Sample Images
```python
# Test model on sample images
if os.path.exists(best_model_path):
    # Find test images
    test_images = []
    
    # Look for test images in dataset
    for root, dirs, files in os.walk(data_dir):
        for file in files[:5]:  # Limit to 5 images
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
    
    if test_images:
        print(f"Testing on {len(test_images)} sample images...")
        
        # Run inference
        results = best_model(test_images[:3])  # Test on first 3 images
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nImage {i+1}: {test_images[i]}")
            print(f"Detections: {len(result.boxes) if result.boxes is not None else 0}")
            
            # Save annotated image
            annotated = result.plot()
            output_path = os.path.join(WORK_DIR, f'result_{i+1}.jpg')
            cv2.imwrite(output_path, annotated)
            
            # Display
            display(Image(output_path))
    else:
        print("No test images found in dataset.")
```

### Cell 13: Export Model
```python
# Export model to different formats
if os.path.exists(best_model_path):
    print("Exporting model to different formats...")
    
    try:
        # Export to ONNX
        onnx_path = best_model.export(format='onnx')
        print(f"ONNX model exported to: {onnx_path}")
        
        # Export to TensorRT (if available)
        # trt_path = best_model.export(format='engine')
        # print(f"TensorRT model exported to: {trt_path}")
        
    except Exception as e:
        print(f"Export error: {e}")
        
    # Copy best model to output
    output_model_path = os.path.join(WORK_DIR, 'best_car_detection_model.pt')
    shutil.copy2(best_model_path, output_model_path)
    print(f"Best model copied to: {output_model_path}")
```

### Cell 14: Training Summary
```python
# Training summary
print("=== TRAINING SUMMARY ===")
print(f"Device used: {device}")
print(f"Dataset path: {data_dir}")
print(f"Configuration: {config_path}")

if os.path.exists(best_model_path):
    print(f"✅ Training completed successfully!")
    print(f"Best model: {best_model_path}")
    print(f"Model size: {os.path.getsize(best_model_path) / (1024*1024):.1f} MB")
    
    print("\n=== NEXT STEPS ===")
    print("1. Download the trained model from Kaggle output")
    print("2. Test the model on new images")
    print("3. Integrate into your highway-guardian project")
    print("4. Consider fine-tuning with more data if needed")
else:
    print("❌ Training incomplete or failed")
    print("Check the error messages above and retry with adjusted parameters")
```

## Các cải tiến so với Colab

### ✅ Đã khắc phục
- **TPU Engine timeout**: Chuyển sang sử dụng GPU thay vì TPU
- **Session timeout**: Thêm checkpoint saving mỗi 5 epochs
- **Internet dependency**: Sử dụng dataset có sẵn trên Kaggle
- **Memory issues**: Tối ưu batch size và workers
- **Training interruption**: Thêm early stopping và error handling

### 🚀 Tối ưu hóa
- **Device**: GPU thay vì TPU (ổn định hơn)
- **Epochs**: Giảm từ 50 xuống 30 để tránh timeout
- **Batch size**: 16 (tối ưu cho GPU memory)
- **Workers**: 2 (phù hợp với Kaggle)
- **Checkpoint**: Lưu mỗi 5 epochs
- **Early stopping**: Patience = 10 epochs

## Cách sử dụng trên Kaggle

### Bước 1: Tạo Notebook mới
1. Đăng nhập vào [Kaggle](https://www.kaggle.com)
2. Tạo notebook mới: **New Notebook**
3. Chọn **GPU** trong settings (bắt buộc)
4. Tắt **Internet** trong settings (để chạy offline)

### Bước 2: Thêm Dataset
1. Trong notebook, click **+ Add Data**
2. Tìm kiếm dataset: `seyeon040768/car-detection-dataset`
3. Click **Add** để thêm dataset vào notebook
4. Dataset sẽ có sẵn tại `/kaggle/input/car-detection-dataset`

### Bước 3: Upload Notebook
1. Copy nội dung từ `Car_Detection_Kaggle_Offline.ipynb`
2. Paste vào Kaggle notebook
3. Hoặc upload file `.ipynb` trực tiếp

### Bước 4: Chạy Training
1. **Run All** để chạy toàn bộ notebook
2. Hoặc chạy từng cell một cách tuần tự
3. Training sẽ mất khoảng 2-3 giờ (30 epochs)

## Cấu trúc Notebook

### 1. Environment Setup
- Kiểm tra GPU availability
- Cài đặt packages cần thiết

### 2. Dataset Setup
- Tự động detect và extract dataset
- Tạo cấu trúc thư mục phù hợp

### 3. YAML Configuration
- Tự động tạo file config cho YOLO
- Detect train/val/test splits

### 4. Model Training
- Khởi tạo YOLOv8n model
- Training với parameters tối ưu
- Checkpoint saving và early stopping

### 5. Results Visualization
- Hiển thị training plots
- Validation metrics
- Sample predictions

### 6. Model Export
- Export sang ONNX format
- Copy model để download

## Troubleshooting

### Lỗi thường gặp

#### 1. Lỗi kết nối mạng (Network Connection Errors)
**Triệu chứng:** `Temporary failure in name resolution`, `NewConnectionError`

**Nguyên nhân:** Kaggle notebook ở chế độ offline hoặc kết nối internet bị hạn chế

**Giải pháp:**
```python
# Kiểm tra kết nối internet
import urllib.request

def check_internet():
    try:
        urllib.request.urlopen('http://google.com', timeout=5)
        return True
    except:
        return False

if check_internet():
    print("✅ Internet available")
else:
    print("❌ No internet - using offline mode")
```

**Các bước khắc phục:**
1. **Bật Internet trong Kaggle:**
   - Settings → Internet → ON
   - Restart notebook

2. **Sử dụng packages có sẵn:**
   ```python
   # Kiểm tra packages có sẵn
   import pkg_resources
   installed = [d.project_name for d in pkg_resources.working_set]
   print("Available packages:", sorted(installed))
   ```

3. **Upload wheel files thủ công:**
   - Download `.whl` files locally
   - Upload as dataset
   - Install from local path

#### 2. "Dataset not found"
**Nguyên nhân**: Chưa add dataset vào notebook
**Giải pháp**: 
- Add dataset `seyeon040768/car-detection-dataset`
- Kiểm tra path `/kaggle/input/car-detection-dataset`

#### 3. "CUDA out of memory"
**Nguyên nhân**: Batch size quá lớn
**Giải pháp**:
```python
# Giảm batch size trong train_params
train_params['batch'] = 8  # thay vì 16
```

#### 4. "Training timeout"
**Nguyên nhân**: Kaggle session timeout (9 giờ)
**Giải pháp**:
```python
# Giảm epochs
train_params['epochs'] = 20  # thay vì 30
```

#### 5. "No GPU available"
**Nguyên nhân**: Chưa bật GPU trong settings
**Giải pháp**:
- Settings → Accelerator → GPU T4 x2

#### 6. Package import errors
**Ultralytics không import được:**
```python
# Fallback: Sử dụng YOLOv5 thay thế
!git clone https://github.com/ultralytics/yolov5
sys.path.append('/kaggle/working/yolov5')
from models.experimental import attempt_load
```

**OpenCV không có:**
```python
# Fallback: Sử dụng PIL
from PIL import Image
import matplotlib.image as mpimg
```

### Performance Tips

#### Tăng tốc training
```python
# Tăng batch size nếu GPU memory đủ
train_params['batch'] = 32

# Sử dụng mixed precision
train_params['amp'] = True

# Cache images (nếu memory đủ)
train_params['cache'] = 'ram'
```

#### Giảm memory usage
```python
# Giảm image size
train_params['imgsz'] = 416  # thay vì 640

# Giảm workers
train_params['workers'] = 1

# Tắt plots
train_params['plots'] = False
```

## Kết quả mong đợi

### Training metrics
- **mAP50**: ~0.99 (rất tốt)
- **mAP50-95**: ~0.87-0.90
- **Training time**: 2-3 giờ (30 epochs)
- **Model size**: ~6MB (YOLOv8n)

### Output files
- `best_car_detection_model.pt`: Model tốt nhất
- `results.png`: Training curves
- `confusion_matrix.png`: Confusion matrix
- `*.onnx`: ONNX export (nếu thành công)

## Download Model

1. Sau khi training xong, vào **Output** tab
2. Download file `best_car_detection_model.pt`
3. Sử dụng trong project highway-guardian

```python
# Sử dụng model đã train
from ultralytics import YOLO
model = YOLO('best_car_detection_model.pt')
results = model('path/to/image.jpg')
```

## So sánh với Colab

| Aspect | Colab (TPU) | Kaggle (GPU) |
|--------|-------------|---------------|
| **Stability** | ❌ Timeout issues | ✅ Ổn định |
| **Internet** | ✅ Required | ✅ Offline mode |
| **Training time** | ❌ Bị dừng ở 97% | ✅ Hoàn thành |
| **Memory** | ❌ Limited | ✅ Đủ dùng |
| **Checkpoints** | ❌ Không có | ✅ Auto save |
| **GPU hours** | ❌ Limited | ✅ 30h/week |

## Lưu ý quan trọng

1. **Bắt buộc bật GPU**: Notebook không chạy được với CPU
2. **Dataset size**: ~2GB, cần đủ storage
3. **Training time**: 2-3 giờ, đảm bảo không đóng browser
4. **Weekly limit**: Kaggle có giới hạn 30 giờ GPU/tuần
5. **Save progress**: Download model ngay sau khi training xong

## Liên hệ

Nếu gặp vấn đề, hãy kiểm tra:
1. GPU settings đã bật chưa
2. Dataset đã add chưa
3. Error messages trong output
4. Kaggle GPU quota còn lại

---

**Chúc bạn training thành công! 🚀**