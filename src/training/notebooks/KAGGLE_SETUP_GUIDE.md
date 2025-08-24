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
# Install required packages with enhanced offline fallback
import subprocess
import sys
import os
import time

def install_package_safe(package, timeout=180, fallback_msg=None):
    """Install package with enhanced network error handling for Kaggle offline mode"""
    # Check if already installed
    try:
        __import__(package.replace('-', '_'))  # Handle package name differences
        print(f"✅ {package} already available")
        return True
    except ImportError:
        pass
    
    # Try installation with multiple strategies
    strategies = [
        ([sys.executable, "-m", "pip", "install", package], timeout),
        ([sys.executable, "-m", "pip", "install", package, "--no-deps"], 120),
        ([sys.executable, "-m", "pip", "install", package, "--user"], 90)
    ]
    
    for i, (cmd, cmd_timeout) in enumerate(strategies):
        try:
            strategy_name = ["standard", "no-deps", "user-install"][i]
            print(f"📦 Installing {package} ({strategy_name})...")
            
            subprocess.check_call(cmd, timeout=cmd_timeout)
            print(f"✅ {package} installed successfully ({strategy_name})")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after {cmd_timeout}s ({strategy_name})")
            if i < len(strategies) - 1:
                print(f"🔄 Trying next strategy...")
            continue
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed ({strategy_name}): {e}")
            if i < len(strategies) - 1:
                print(f"🔄 Trying next strategy...")
            continue
            
        except Exception as e:
            print(f"❌ Unexpected error ({strategy_name}): {e}")
            continue
    
    # All strategies failed
    print(f"❌ All installation strategies failed for {package}")
    if fallback_msg:
        print(f"💡 {fallback_msg}")
    return False

# Try to install packages with enhanced fallbacks
print("=== ENHANCED PACKAGE INSTALLATION ===")
print("⚠️  Note: Installation may take 3-5 minutes per package")
print("🌐 Make sure Internet is enabled in Kaggle settings\n")

# Ultralytics - most important (try longer timeout)
print("🎯 Installing Ultralytics (critical for YOLO training)...")
ultralytics_ok = install_package_safe('ultralytics', timeout=300,  # 5 minutes
    """Fallback options:
    1. Upload ultralytics wheel file as dataset
    2. Use YOLOv5 from torch.hub: torch.hub.load('ultralytics/yolov5', 'yolov5s')
    3. Enable internet and restart notebook""")

# OpenCV - usually available in Kaggle
print("\n🖼️  Installing OpenCV...")
opencv_ok = install_package_safe('opencv-python', timeout=120,
    "Fallback: OpenCV is usually pre-installed in Kaggle environment")

# Wandb - optional for logging
print("\n📊 Installing WandB (optional)...")
wandb_ok = install_package_safe('wandb', timeout=120,
    "Fallback: WandB is optional, training will work without it")

# Final status report
print("\n" + "="*50)
print("📊 INSTALLATION SUMMARY")
print("="*50)
print(f"  🎯 Ultralytics: {'✅ Ready' if ultralytics_ok else '❌ Failed'}")
print(f"  🖼️  OpenCV:     {'✅ Ready' if opencv_ok else '❌ Failed'}")
print(f"  📊 WandB:      {'✅ Ready' if wandb_ok else '❌ Failed'}")

if not ultralytics_ok:
    print("\n⚠️  CRITICAL: Ultralytics not available!")
    print("🔧 Quick fixes:")
    print("   1. Settings → Internet → ON → Restart notebook")
    print("   2. Try: !pip install ultralytics --no-deps")
    print("   3. Upload ultralytics wheel file as dataset")
    print("   4. Use alternative: torch.hub.load('ultralytics/yolov5', 'yolov5s')")
else:
    print("\n🎉 All critical packages ready for training!")
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

#### 1. Lỗi timeout khi cài đặt Ultralytics

**Triệu chứng:**
```
❌ Failed to install ultralytics: Command '[...] timed out after 60 seconds
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) 
after connection broken by 'NewConnectionError'...
```

**Nguyên nhân:**
- Kết nối mạng chậm hoặc không ổn định
- Kaggle notebook ở chế độ offline
- Package ultralytics có dung lượng lớn

**Giải pháp:**

**Option 1: Bật internet và tăng timeout**
```python
def install_package_safe(package, timeout=180):  # Tăng timeout lên 3 phút
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after {timeout}s - trying with --no-deps")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                 package, "--no-deps"], timeout=120)
            return True
        except:
            return False
```

**Option 2: Sử dụng pre-installed packages**
```python
# Kiểm tra ultralytics có sẵn không
try:
    import ultralytics
    print("✅ Ultralytics available (pre-installed)")
except ImportError:
    print("❌ Ultralytics not available")
    # Fallback to manual installation
```

**Option 3: Upload wheel file thủ công**
1. Download `ultralytics-*.whl` từ [PyPI](https://pypi.org/project/ultralytics/#files)
2. Upload as Kaggle dataset
3. Install từ local:
```python
import os
wheel_path = "/kaggle/input/ultralytics-wheel/ultralytics-8.0.0-py3-none-any.whl"
if os.path.exists(wheel_path):
    !pip install {wheel_path}
```

#### 2. Lỗi kết nối mạng (Network Connection Errors)
**Triệu chứng:** `Temporary failure in name resolution`, `NewConnectionError`

**Nguyên nhân:** Kaggle notebook ở chế độ offline hoặc kết nối internet bị hạn chế

**Giải pháp:**
1. **Kiểm tra và bật internet:**
   - Settings → Internet → ON
   - Restart notebook
   
2. **Kiểm tra kết nối:**
   ```python
   import requests
   try:
       response = requests.get('https://pypi.org', timeout=10)
       print(f"✅ PyPI accessible (status: {response.status_code})")
   except Exception as e:
       print(f"❌ Cannot reach PyPI: {e}")
   ```

3. **Sử dụng packages có sẵn:**
   ```python
   # Kiểm tra packages có sẵn
   import pkg_resources
   installed = [d.project_name for d in pkg_resources.working_set]
   print("Available packages:", sorted(installed))
   ```

4. **Upload wheel files thủ công:**
   - Download `.whl` files locally
   - Upload as dataset
   - Install from local path

#### 2. Lỗi import gói (Package Import Errors)

**Triệu chứng:**
- `ModuleNotFoundError: No module named 'ultralytics'`
- `ImportError: cannot import name 'YOLO'`

**Nguyên nhân:**
- Package chưa được cài đặt hoặc cài đặt thất bại
- Phiên bản không tương thích
- Lỗi dependencies

**Giải pháp khi Ultralytics không có:**

**Option 1: Sử dụng YOLOv8 alternative**
```python
# Thử các package YOLO khác
try:
    # Thử ultralytics trước
    from ultralytics import YOLO
    print("✅ Using ultralytics YOLO")
except ImportError:
    try:
        # Fallback: sử dụng yolov5
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("✅ Using YOLOv5 from torch.hub")
    except Exception as e:
        print(f"❌ Cannot load any YOLO model: {e}")
        print("Please upload ultralytics wheel file manually")
```

**Option 2: Cài đặt từ GitHub (nếu có internet)**
```python
# Cài đặt trực tiếp từ GitHub
try:
    !pip install git+https://github.com/ultralytics/ultralytics.git
    from ultralytics import YOLO
    print("✅ Installed from GitHub")
except Exception as e:
    print(f"❌ GitHub installation failed: {e}")
```

**Option 3: Sử dụng pre-trained model có sẵn**
```python
# Kiểm tra model có sẵn trong Kaggle
import os
kaggle_models = [
    '/kaggle/input/yolov8-models/yolov8n.pt',
    '/kaggle/input/yolo-models/yolov8s.pt',
    '/opt/conda/lib/python3.10/site-packages/ultralytics/'
]

for model_path in kaggle_models:
    if os.path.exists(model_path):
        print(f"✅ Found model at: {model_path}")
        break
else:
    print("❌ No pre-trained models found")
```

#### 3. "Dataset not found"
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

## So sánh Kaggle vs Google Colab

### Kaggle Accelerator Options

| Accelerator | Miễn phí | Yêu cầu xác thực | Hiệu năng | Giới hạn |
|-------------|----------|------------------|-----------|----------|
| **None (CPU)** | ✅ Có | ❌ Không | Thấp | Không giới hạn |
| **GPU T4 x2** | ✅ Có | ✅ **Phone verification** | Cao | 30h/tuần |
| **GPU P100** | ✅ Có | ✅ **Phone verification** | Trung bình | 30h/tuần |
| **TPU VM v3-8** | ✅ Có | ✅ **Phone verification** | Rất cao | 20h/tuần |

### Xác thực tài khoản Kaggle

**Để sử dụng GPU/TPU, bạn PHẢI:**
1. **Phone Verification**: Xác thực số điện thoại
2. **Account Settings** → **Phone Verification** → Nhập số điện thoại
3. Nhận SMS code và xác nhận

**Lưu ý quan trọng:**
- ⚠️ **CPU (None)** chỉ phù hợp cho testing, không đủ mạnh để train YOLO
- 🚀 **GPU T4 x2** là lựa chọn tốt nhất cho car detection
- 📱 **Bắt buộc** phải xác thực phone để dùng GPU/TPU

### So sánh chi tiết

| Tính năng | Kaggle (CPU) | Kaggle (GPU) | Google Colab |
|-----------|--------------|--------------|-------------|
| **Yêu cầu xác thực** | ❌ Không | ✅ Phone | ❌ Không |
| **GPU miễn phí** | ❌ Không | ✅ 30h/tuần T4 | ✅ 12h/ngày (giới hạn) |
| **Thời gian training** | 🐌 Rất chậm | 🚀 2-3 giờ | 🚀 2-3 giờ |
| **Thời gian session** | 12 giờ | 12 giờ | 12 giờ (có thể ngắt sớm) |
| **Dataset** | ✅ Tích hợp | ✅ Tích hợp | ❌ Cần upload |
| **Internet** | Tùy chọn | Tùy chọn | Luôn có |
| **Ổn định** | ✅ Cao | ✅ Cao | ⚠️ Trung bình |
| **Phù hợp cho YOLO** | ❌ Không | ✅ Có | ✅ Có |

### Khuyến nghị

**Nếu chưa xác thực phone:**
- 🔄 Sử dụng Google Colab cho training
- 📱 Xác thực phone Kaggle để dùng lâu dài

**Nếu đã xác thực phone:**
- 🎯 **Kaggle GPU T4 x2** là lựa chọn tốt nhất
- ⏰ 30h/tuần đủ cho nhiều lần training
- 🔒 Ổn định hơn Colab, ít bị ngắt session

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