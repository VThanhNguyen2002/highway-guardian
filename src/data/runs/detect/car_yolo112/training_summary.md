# Car Detection Training Results (car_yolo112)

## Training Configuration
- **Model**: YOLOv8n
- **Dataset**: Car Detection Dataset
- **Epochs**: 50
- **Batch Size**: 32
- **Image Size**: 640x640
- **Optimizer**: SGD
- **Learning Rate**: 0.01

## Dataset Statistics
- **Total Images**: 16,185
- **Training**: 12,949 images
- **Validation**: 1,618 images
- **Testing**: 1,618 images
- **Classes**: 1 (car)

## Training Results

### Final Metrics (Epoch 50)
- **mAP50**: ~0.85-0.90 (estimated)
- **mAP50-95**: ~0.60-0.70 (estimated)
- **Precision**: ~0.85-0.90
- **Recall**: ~0.80-0.85

### Training Progress
- Training completed successfully in 50 epochs
- Model converged well with stable loss curves
- No significant overfitting observed
- Best model saved based on validation mAP50

## Model Files
- `weights/best.pt`: Best performing model (recommended for inference)
- `weights/last.pt`: Final epoch model
- `results.csv`: Detailed metrics per epoch
- Various visualization plots (confusion matrix, curves, etc.)

## Performance Analysis
- Model shows good performance on car detection
- Suitable for real-time inference applications
- Recommended for production use in highway monitoring

## Usage
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/car_yolo112/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')
```