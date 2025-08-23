# Sign Detection Training Results (sign_yolo85)

## Training Configuration
- **Model**: YOLOv8s
- **Dataset**: Vietnamese Traffic Signs Detection
- **Epochs**: 120
- **Batch Size**: 16
- **Image Size**: 960x960
- **Optimizer**: AdamW
- **Learning Rate**: 0.002

## Dataset Statistics
- **Total Images**: 1,170
- **Training**: 900 images
- **Validation**: 180 images
- **Testing**: 90 images
- **Classes**: 25 (Vietnamese traffic signs)

## Training Results

### Final Metrics (Epoch 120)
- **mAP50**: 0.863
- **mAP50-95**: 0.593
- **Precision**: 0.845
- **Recall**: 0.742

### Class-Specific Performance

#### Best Performing Classes:
1. **Cấm đỗ xe**: P=0.975, R=0.920, mAP50=0.992
2. **Cấm dừng & đỗ**: P=0.920, R=0.929, mAP50=0.929
3. **Cấm xe tải**: P=1.000, R=0.976, mAP50=0.995
4. **Hướng đi**: P=0.954, R=0.941, mAP50=0.990
5. **Vạch sang đường**: P=0.914, R=0.800, mAP50=0.885

#### Classes Needing Improvement:
1. **Giới hạn chiều cao**: P=1.000, R=0.000, mAP50=0.0495
2. **Công trường**: P=0.415, R=0.829, mAP50=0.497
3. **Hạn chế tốc độ**: P=0.802, R=0.556, mAP50=0.699
4. **Đi chậm**: P=1.000, R=0.533, mAP50=0.790
5. **Cảnh báo khác**: P=0.743, R=0.500, mAP50=0.786

#### All Classes Performance:
- **Cấm đi ngược chiều**: P=0.917, R=0.737, mAP50=0.846
- **Cấm rẽ trái**: P=0.398, R=1.000, mAP50=0.995
- **Cấm quay đầu**: P=1.000, R=0.000, mAP50=0.995
- **Cấm quay đầu & rẽ trái**: P=0.919, R=1.000, mAP50=0.995
- **Cấm xe máy**: P=0.802, R=1.000, mAP50=0.995
- **Cấm ô-tô**: P=0.674, R=1.000, mAP50=0.995
- **Biển cấm khác**: P=0.854, R=0.684, mAP50=0.866
- **Chỉ dẫn**: P=0.853, R=0.767, mAP50=0.794
- **Giới hạn tải trọng**: P=1.000, R=0.000, mAP50=0.995
- **Nguy hiểm giao nhau**: P=0.814, R=0.915, mAP50=0.937
- **Nguy hiểm đường trơn**: P=0.926, R=0.826, mAP50=0.883
- **Nguy hiểm người đi bộ**: P=0.841, R=0.833, mAP50=0.898
- **Làn xe được phép**: P=0.800, R=0.722, mAP50=0.814
- **Làn xe & tốc độ**: P=0.856, R=1.000, mAP50=0.978
- **Hết cấm**: P=0.783, R=1.000, mAP50=0.995
- **Khác**: P=0.822, R=0.833, mAP50=0.844

## Training Progress
- Training completed successfully in 120 epochs
- Model showed steady improvement throughout training
- Best performance achieved around epoch 110-120
- No significant overfitting observed

## Model Files
- `weights/best.pt`: Best performing model (mAP50=0.863)
- `weights/last.pt`: Final epoch model
- `results.csv`: Detailed metrics per epoch
- Various visualization plots and analysis

## Performance Analysis

### Strengths:
- Excellent performance on prohibition signs (Cấm đỗ xe, Cấm dừng & đỗ)
- Good detection of directional signs (Hướng đi)
- Strong performance on warning signs (Nguy hiểm giao nhau)

### Areas for Improvement:
- Height restriction signs need more training data
- Construction zone signs require better feature learning
- Speed limit signs could benefit from data augmentation
- Some rare classes have insufficient training samples

### Recommendations:
1. Collect more data for underperforming classes
2. Apply class-specific data augmentation
3. Consider ensemble methods for rare classes
4. Fine-tune with additional epochs for specific classes

## Usage
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/sign_yolo85/weights/best.pt')

# Run inference
results = model('path/to/traffic_sign_image.jpg')

# Get predictions
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        print(f"Detected: {class_name} (confidence: {confidence:.2f})")
```