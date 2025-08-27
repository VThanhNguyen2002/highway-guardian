# Traffic Signs Detection Training Guide

## Overview

Notebook `update-training-traffic-signs.ipynb` được thiết kế để training mô hình YOLOv8 cho việc phát hiện biển báo giao thông trên Kaggle với setup tối ưu.

## Dataset

### Kaggle Dataset
- **Dataset**: `valentynsichkar/traffic-signs-dataset-in-yolo-format` <mcreference link="https://github.com/Balakishan77/yolov5_custom_trained_traffic_sign_detector" index="3">3</mcreference>
- **Format**: YOLO bounding box annotations
- **Classes**: 4 categories
  - `speed_limit`: Biển báo giới hạn tốc độ
  - `yield`: Biển báo nhường đường
  - `mandatory`: Biển báo bắt buộc
  - `other`: Các biển báo khác

### Dataset Structure
```
traffic-signs-dataset/
├── train/
│   ├── images/     # Training images (.jpg)
│   └── labels/     # Training labels (.txt)
├── valid/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── test/
    ├── images/     # Test images
    └── labels/     # Test labels
```

## Training Configuration

### Hardware Setup
- **GPUs**: 2x Tesla T4 (Kaggle)
- **Memory**: ~30GB GPU memory total
- **CPU**: Multi-core with 4 workers

### Model Parameters
- **Base Model**: YOLOv8n (nano - fastest)
- **Input Size**: 640x640
- **Batch Size**: 32 (16 per GPU)
- **Epochs**: 50 (more for smaller objects)

### Optimizer Settings
- **Optimizer**: AdamW (better than SGD for detection)
- **Learning Rate**: 0.001 (lower for stability)
- **Weight Decay**: 0.0005 (L2 regularization)
- **LR Schedule**: Cosine annealing (lrf=0.01)

### Data Augmentation (Tuned for Traffic Signs)
- **HSV**: Reduced hue (0.01), moderate saturation (0.5), value (0.3)
- **Geometric**: Small rotation (5°), minimal translation (0.05)
- **Scale**: 0.3 scale variation
- **Flip**: 50% horizontal flip
- **Mosaic**: 80% mosaic augmentation
- **Mixup**: 10% mixup augmentation

## Usage Instructions

### 1. Kaggle Setup

1. **Create New Notebook** trên Kaggle
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Enable Internet**: Settings → Internet → On
4. **Add Dataset**: 
   - Search: `valentynsichkar/traffic-signs-dataset-in-yolo-format`
   - Click "Add Data"

### 2. Upload Notebook

1. Upload `update-training-traffic-signs.ipynb` to Kaggle
2. Hoặc copy-paste nội dung từ file

### 3. Run Training

1. **Run All Cells** hoặc chạy từng cell
2. **Monitor Progress**: Xem training logs và metrics
3. **Check Results**: Xem plots và confusion matrix
4. **Download Archive**: Tải về file zip kết quả

### 4. Expected Training Time

- **Setup**: ~2-3 minutes
- **Training**: ~45-60 minutes (50 epochs)
- **Validation**: ~2-3 minutes
- **Total**: ~1 hour

## Expected Results

### Performance Targets
- **mAP50**: >0.85 (target >85%)
- **mAP50-95**: >0.60 (target >60%)
- **Training Speed**: ~3-4 it/s with 2 GPUs
- **Convergence**: Early convergence around epoch 30-40

### Output Files
```
runs_detect/traffic_signs_detection/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Final epoch model
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── val_batch0_pred.jpg  # Validation predictions
├── labels.jpg           # Dataset labels distribution
└── args.yaml           # Training arguments
```

## Comparison with Car Detection

| Aspect | Car Detection | Traffic Signs |
|--------|---------------|---------------|
| **Classes** | 1 (car) | 4 (speed_limit, yield, mandatory, other) |
| **Object Size** | Large | Small to Medium |
| **Epochs** | 30 | 50 |
| **Augmentation** | Standard | Reduced (signs are rigid) |
| **Learning Rate** | 0.001 | 0.001 |
| **Expected mAP50** | >0.99 | >0.85 |
| **Difficulty** | Easy | Medium |

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Ensure dataset is added to Kaggle notebook
   - Check dataset name: `valentynsichkar/traffic-signs-dataset-in-yolo-format`

2. **GPU Memory Error**
   - Reduce batch size: `batch = 16` instead of 32
   - Use single GPU: `device = "0"`

3. **Slow Training**
   - Ensure 2 GPUs are enabled
   - Check `workers=4` for data loading

4. **Poor Performance**
   - Increase epochs to 100
   - Adjust learning rate: try 0.0005
   - Check dataset quality

### Performance Optimization

1. **Faster Training**
   - Use YOLOv8s instead of YOLOv8n (if memory allows)
   - Increase batch size to 48-64
   - Use mixed precision (already enabled)

2. **Better Accuracy**
   - Increase input size to 832
   - Use more epochs (100+)
   - Add more augmentation
   - Use ensemble methods

## Advanced Features

### 1. Weights & Biases Integration
```python
# Enable W&B logging
os.environ['WANDB_DISABLED'] = 'false'
import wandb
wandb.login()  # Enter API key
```

### 2. Custom Classes
```python
# Modify classes in config
traffic_signs_classes = [
    "stop", "yield", "speed_limit_30", "speed_limit_50",
    "no_entry", "mandatory_right", "warning", "other"
]
```

### 3. Transfer Learning
```python
# Use custom pretrained model
model = YOLO("path/to/custom_pretrained.pt")
```

## Next Steps

1. **Test on Real Data**: Validate model on real traffic sign images
2. **Deploy Model**: Convert to ONNX/TensorRT for production
3. **Integration**: Integrate with highway monitoring system
4. **Continuous Learning**: Retrain with new data periodically

---

**Note**: Notebook này được tối ưu hóa cho Kaggle environment với 2x Tesla T4 GPUs. Điều chỉnh parameters nếu sử dụng hardware khác.