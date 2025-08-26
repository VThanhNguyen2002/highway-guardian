# Kaggle Training Setup Guide - Online Mode

## Overview
This guide provides comprehensive instructions for setting up and running YOLO training on Kaggle with **Internet ON (Online Mode)** for maximum flexibility and latest features.

## Quick Start

### Prerequisites
- Kaggle account with phone verification
- Kaggle API token (kaggle.json)
- Basic understanding of YOLO and object detection

### Internet ON (Online Mode) - Recommended Setup

**Benefits:**
- Install latest packages and dependencies
- Enable/disable WandB logging as needed
- Download datasets directly from Kaggle API
- Use caching for better performance
- Higher workers and epochs for optimal training
- Real-time monitoring and logging capabilities

## Complete Setup Process

### Step 1: Initial Environment Setup

```python
# Clone the highway-guardian repository
!git clone https://github.com/your-username/highway-guardian.git
%cd highway-guardian

# Install required packages with latest versions
!pip install --upgrade torch torchvision ultralytics opencv-python kaggle wandb

# Alternative: Install from requirements if available
# !pip install -r requirements.txt
```

### Step 2: Kaggle API Configuration

```python
# Upload kaggle.json file
from google.colab import files
uploaded = files.upload()  # Select and upload your kaggle.json file

# Setup Kaggle API credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Verify Kaggle connection
!kaggle datasets list --max-size 1000000
```

### Step 3: Dataset Download and Setup

```python
# Download car detection dataset from Kaggle
!kaggle datasets download -d seyeon040768/car-detection-dataset
!unzip car-detection-dataset.zip -d data/car_detection/

# Create necessary directories
!mkdir -p configs
!mkdir -p runs_detect
```

### Step 4: Training Configuration

```python
# Create optimized config file for Kaggle online mode
config_content = '''
# Car Detection Training Configuration for Kaggle Online Mode
# Dataset: Real Kaggle Car Detection Dataset

model:
  weights: 'yolov8n.pt'  # pretrained model
  architecture: 'yolov8n'

data:
  path: '/kaggle/working/highway-guardian/data/car_detection/car_dataset-master'
  train: 'train/images'
  val: 'valid/images'
  test: 'test/images'
  nc: 1  # number of classes
  names: ['car']  # class names

training:
  epochs: 50
  batch_size: 16
  image_size: 640
  device: 'auto'  # Auto-detect GPU/CPU
  workers: 4
  optimizer: 'AdamW'
  lr0: 0.01
  lrf: 0.1
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 5
  box: 7.5
  cls: 0.5
  dfl: 1.5

augmentation:
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
  mixup: 0.0

validation:
  save_period: 1
  patience: 20
  conf: 0.001
  iou: 0.6

output:
  project: 'runs_detect'
  name: 'car_detection_kaggle'
  plots: true
  verbose: true
'''

with open('configs/car_detection_kaggle.yaml', 'w') as f:
    f.write(config_content)

print("✅ Config file created for Kaggle Online Mode!")
```

### Step 5: WandB Setup (Optional)

```python
# Option 1: Enable WandB logging
import wandb
wandb.login()  # Follow prompts to login

# Option 2: Disable WandB (for faster training)
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'
print("WandB disabled for faster training")
```

### Step 6: Run Training

```python
# Method 1: Direct YOLO training
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Start training with optimized parameters
results = model.train(
    data='configs/car_detection_kaggle.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device='auto',
    workers=4,
    project='runs_detect',
    name='car_detection_kaggle',
    cache=True,  # Enable caching for faster data loading
    plots=True,
    verbose=True
)

print("✅ Training completed!")
```

```python
# Method 2: Using custom training script (if available)
!python src/training/scripts/train_car_detection.py --config configs/car_detection_kaggle.yaml
```

### Step 7: Monitor Training Results

```python
# Display training results
from IPython.display import Image, display
import os

# Find the latest training run
runs_dir = 'runs_detect'
latest_run = max([d for d in os.listdir(runs_dir) if d.startswith('car_detection')], 
                key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))

results_path = f'{runs_dir}/{latest_run}'

# Display training plots
if os.path.exists(f'{results_path}/results.png'):
    print("📊 Training Results:")
    display(Image(f'{results_path}/results.png'))

if os.path.exists(f'{results_path}/confusion_matrix.png'):
    print("🎯 Confusion Matrix:")
    display(Image(f'{results_path}/confusion_matrix.png'))

if os.path.exists(f'{results_path}/labels.jpg'):
    print("🏷️ Dataset Labels Distribution:")
    display(Image(f'{results_path}/labels.jpg'))

print(f"📁 All results saved in: {results_path}")
```

## Advanced Features

### Auto-detect Dataset Structure

```python
# Auto-detect dataset root directory
import os

def find_dataset_root(base_path):
    """Find the actual dataset root directory"""
    for root, dirs, files in os.walk(base_path):
        if 'train' in dirs and 'valid' in dirs:
            return root
    return base_path

# Update config with detected path
dataset_root = find_dataset_root('data/car_detection')
print(f"📂 Dataset root detected: {dataset_root}")
```

### Performance Optimization

```python
# Check available resources
import torch

print(f"🖥️ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Optimize batch size based on available memory
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb >= 15:
        recommended_batch = 32
    elif gpu_memory_gb >= 8:
        recommended_batch = 16
    else:
        recommended_batch = 8
else:
    recommended_batch = 4

print(f"💡 Recommended batch size: {recommended_batch}")
```

## Troubleshooting

### Common Issues and Solutions:

1. **Dataset Download Fails**
   ```python
   # Verify Kaggle API setup
   !kaggle datasets list --max-size 100000
   ```

2. **Out of Memory Errors**
   ```python
   # Reduce batch size and image size
   model.train(batch=8, imgsz=416)
   ```

3. **Slow Training**
   ```python
   # Enable GPU in Kaggle settings
   # Reduce workers if CPU bottleneck
   model.train(workers=2)
   ```

4. **WandB Issues**
   ```python
   # Disable WandB completely
   os.environ['WANDB_MODE'] = 'disabled'
   ```

## Performance Tips

1. **Enable GPU/TPU** in Kaggle notebook settings
2. **Use caching** for faster data loading: `cache=True`
3. **Optimize batch size** based on available memory
4. **Monitor resource usage** during training
5. **Save checkpoints** regularly for long training runs

## Best Practices

1. **Start with small epochs** (5-10) to test setup
2. **Monitor training metrics** in real-time
3. **Use version control** for configuration files
4. **Document experiments** with clear naming
5. **Backup important models** and results

## Support

For issues and questions:
- Check [Ultralytics Documentation](https://docs.ultralytics.com/)
- Review [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- Post in [Kaggle Community Forums](https://www.kaggle.com/discussions)

---

*Last updated: 2024 - Optimized for Kaggle Online Mode*