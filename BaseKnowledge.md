# World Knowledge Base - Lý thuyết Tổng hợp

## Mục lục
1. [Lý thuyết Nhận diện Biển báo Giao thông](#1-lý-thuyết-nhận-diện-biển-báo-giao-thông)
2. [Lý thuyết Phân loại Xe](#2-lý-thuyết-phân-loại-xe)
3. [Deep Learning và Computer Vision](#3-deep-learning-và-computer-vision)
4. [Mô hình CNN](#4-mô-hình-cnn)
5. [Mô hình YOLO](#5-mô-hình-yolo)
6. [Xử lý Dữ liệu và Tiền xử lý](#6-xử-lý-dữ-liệu-và-tiền-xử-lý)
7. [Đánh giá Mô hình](#7-đánh-giá-mô-hình)
8. [Triển khai và Tối ưu hóa](#8-triển-khai-và-tối-ưu-hóa)

---

## 1. Lý thuyết Nhận diện Biển báo Giao thông

### 1.1 Phân loại Biển báo Giao thông

#### Theo Luật Giao thông Việt Nam:
- **Biển báo cấm (P)**: P.1 - P.30
  - P.1: Cấm đi ngược chiều
  - P.12: Cấm xe ô tô
  - P.15: Cấm xe máy
  - P.21: Cấm rẽ trái

- **Biển báo nguy hiểm (W)**: W.1 - W.35
  - W.1: Chỗ ngoặt nguy hiểm
  - W.15: Đường giao nhau
  - W.20: Trẻ em

- **Biển báo hiệu lệnh (R)**: R.1 - R.20
  - R.1: Đường bắt buộc
  - R.5: Hướng đi thẳng
  - R.10: Làn đường dành cho xe buýt

- **Biển báo chỉ dẫn (I)**: I.1 - I.50
  - I.1: Chỉ dẫn địa danh
  - I.15: Bệnh viện
  - I.20: Trạm xăng

### 1.2 Đặc điểm Hình học

#### Hình dạng:
- **Tam giác**: Biển báo nguy hiểm
- **Hình tròn**: Biển báo cấm và hiệu lệnh
- **Hình vuông/chữ nhật**: Biển báo chỉ dẫn
- **Hình thoi**: Biển báo ưu tiên

#### Màu sắc:
- **Đỏ**: Cấm, dừng, nguy hiểm
- **Xanh**: Chỉ dẫn, thông tin
- **Vàng**: Cảnh báo, tạm thời
- **Trắng**: Nền chung
- **Đen**: Chữ và ký hiệu

### 1.3 Thuật toán Nhận diện

#### Pipeline Xử lý:
1. **Tiền xử lý ảnh**
   - Resize ảnh về kích thước chuẩn
   - Chuẩn hóa pixel values [0,1]
   - Data augmentation

2. **Phát hiện Object**
   - Sử dụng YOLO để detect bounding box
   - Non-Maximum Suppression (NMS)
   - Confidence threshold filtering

3. **Phân loại**
   - Extract features bằng CNN
   - Classification layer
   - Softmax activation

4. **Post-processing**
   - Mapping ID → Loại biển báo
   - Confidence score
   - Visualization

---

## 2. Lý thuyết Phân loại Xe

### 2.1 Phân loại theo Kích thước

#### Xe con (Passenger Cars):
- **Sedan**: 4 cửa, khoang hành lý riêng
- **Hatchback**: 3-5 cửa, khoang hành lý liền
- **SUV**: Sport Utility Vehicle, gầm cao
- **Coupe**: 2 cửa, thể thao
- **Convertible**: Mui trần

#### Xe thương mại (Commercial Vehicles):
- **Pickup Truck**: Thùng xe mở
- **Van**: Xe tải nhỏ, kín
- **Truck**: Xe tải lớn
- **Bus**: Xe buýt, xe khách

### 2.2 Phân loại theo Thương hiệu

#### Thương hiệu Nhật Bản:
- **Toyota**: Logo oval, thiết kế conservative
- **Honda**: Logo chữ H, thiết kế sporty
- **Nissan**: Logo tròn, thiết kế modern
- **Mazda**: Logo cánh chim, thiết kế elegant

#### Thương hiệu Đức:
- **BMW**: Logo propeller, kidney grille
- **Mercedes-Benz**: Logo ngôi sao 3 cánh
- **Audi**: Logo 4 vòng tròn
- **Volkswagen**: Logo VW

#### Thương hiệu Hàn Quốc:
- **Hyundai**: Logo chữ H nghiêng
- **Kia**: Logo KIA stylized

### 2.3 Đặc điểm Nhận diện

#### Visual Features:
- **Grille Design**: Lưới tản nhiệt đặc trưng
- **Headlight Shape**: Hình dạng đèn pha
- **Body Proportions**: Tỷ lệ thân xe
- **Logo Position**: Vị trí và hình dạng logo
- **Wheel Design**: Thiết kế mâm xe

#### Technical Features:
- **Aspect Ratio**: Tỷ lệ chiều dài/rộng
- **Ground Clearance**: Khoảng sáng gầm xe
- **Roof Line**: Đường nóc xe
- **Window Shape**: Hình dạng cửa sổ

---

## 3. Deep Learning và Computer Vision

### 3.1 Convolutional Neural Networks (CNN)

#### Kiến trúc Cơ bản:
```
Input → Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Dense → Output
```

#### Các Layer chính:
- **Convolutional Layer**: Trích xuất features
- **Pooling Layer**: Giảm kích thước, tăng invariance
- **Activation Layer**: Non-linearity (ReLU, Sigmoid, Tanh)
- **Fully Connected Layer**: Classification
- **Dropout Layer**: Regularization

#### Hyperparameters:
- **Filter Size**: 3x3, 5x5, 7x7
- **Stride**: Bước nhảy của filter
- **Padding**: Same, Valid
- **Learning Rate**: 0.001, 0.01, 0.1
- **Batch Size**: 16, 32, 64, 128

### 3.2 Transfer Learning

#### Pre-trained Models:
- **VGG16/VGG19**: Deep architecture, small filters
- **ResNet50/ResNet101**: Residual connections
- **InceptionV3**: Multi-scale features
- **MobileNet**: Lightweight, mobile-optimized
- **EfficientNet**: Compound scaling

#### Fine-tuning Strategies:
1. **Feature Extraction**: Freeze pre-trained layers
2. **Fine-tuning**: Unfreeze some layers
3. **Full Training**: Train all layers

---

## 4. Mô hình CNN

### 4.1 Kiến trúc cho Traffic Sign Recognition

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### 4.2 Data Augmentation

```python
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,  # Không flip biển báo
    fill_mode='nearest'
)
```

### 4.3 Loss Functions

- **Categorical Crossentropy**: Multi-class classification
- **Sparse Categorical Crossentropy**: Integer labels
- **Focal Loss**: Imbalanced datasets
- **Center Loss**: Feature learning

---

## 5. Mô hình YOLO

### 5.1 YOLO Architecture

#### YOLOv8 Components:
- **Backbone**: CSPDarknet53
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Detection head với anchor-free

#### Key Features:
- **Single Shot Detection**: Một lần forward pass
- **Grid-based Prediction**: Chia ảnh thành grid
- **Bounding Box Regression**: Dự đoán tọa độ box
- **Class Prediction**: Phân loại object

### 5.2 Loss Function

```
Total Loss = λ₁ × Box Loss + λ₂ × Object Loss + λ₃ × Class Loss
```

- **Box Loss**: IoU loss, GIoU loss, DIoU loss
- **Object Loss**: Binary crossentropy
- **Class Loss**: Categorical crossentropy

### 5.3 Non-Maximum Suppression (NMS)

```python
def nms(boxes, scores, iou_threshold=0.5):
    # Sort by confidence scores
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        # Calculate IoU with remaining boxes
        ious = calculate_iou(boxes[current], boxes[indices[1:]])
        
        # Remove boxes with high IoU
        indices = indices[1:][ious < iou_threshold]
    
    return keep
```

---

## 6. Xử lý Dữ liệu và Tiền xử lý

### 6.1 Dataset Preparation

#### Traffic Signs Dataset:
- **GTSRB**: German Traffic Sign Recognition Benchmark
- **BTSD**: Belgian Traffic Sign Dataset
- **Vietnam Traffic Signs**: Custom dataset

#### Vehicle Dataset:
- **COCO**: Common Objects in Context
- **ImageNet**: Large-scale image database
- **Stanford Cars**: Fine-grained car classification

### 6.2 Annotation Formats

#### YOLO Format:
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
```

#### COCO Format:
```json
{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "area": 1000,
    "iscrowd": 0
}
```

### 6.3 Data Preprocessing Pipeline

```python
def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Resize
    image = cv2.resize(image, (640, 640))
    
    # Normalize
    image = image / 255.0
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
```

---

## 7. Đánh giá Mô hình

### 7.1 Metrics cho Classification

#### Confusion Matrix:
```
                Predicted
              P    N
Actual   P   TP   FN
         N   FP   TN
```

#### Derived Metrics:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### 7.2 Metrics cho Object Detection

#### Intersection over Union (IoU):
```
IoU = Area of Overlap / Area of Union
```

#### Average Precision (AP):
- **AP@0.5**: IoU threshold = 0.5
- **AP@0.75**: IoU threshold = 0.75
- **mAP**: Mean Average Precision across all classes

#### Mean Average Precision (mAP):
```
mAP = (1/N) × Σ(AP_i) for i = 1 to N classes
```

### 7.3 Validation Strategies

- **Hold-out Validation**: 70-20-10 split
- **K-Fold Cross Validation**: K=5 or K=10
- **Stratified Sampling**: Maintain class distribution
- **Time-based Split**: For temporal data

---

## 8. Triển khai và Tối ưu hóa

### 8.1 Model Optimization

#### Quantization:
- **INT8 Quantization**: Giảm từ 32-bit xuống 8-bit
- **Dynamic Quantization**: Runtime quantization
- **Static Quantization**: Calibration dataset

#### Pruning:
- **Magnitude-based Pruning**: Remove small weights
- **Structured Pruning**: Remove entire channels/layers
- **Gradual Pruning**: Progressive weight removal

#### Knowledge Distillation:
```python
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    
    soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
    hard_loss = F.cross_entropy(student_logits, true_labels)
    
    return 0.7 * soft_loss + 0.3 * hard_loss
```

### 8.2 Deployment Strategies

#### Edge Deployment:
- **TensorRT**: NVIDIA GPU optimization
- **OpenVINO**: Intel CPU/GPU optimization
- **TensorFlow Lite**: Mobile deployment
- **ONNX**: Cross-platform inference

#### Cloud Deployment:
- **Docker Containers**: Containerized deployment
- **Kubernetes**: Orchestration
- **AWS SageMaker**: Managed ML platform
- **Google Cloud AI Platform**: Scalable inference

### 8.3 Real-time Inference

#### Optimization Techniques:
- **Batch Processing**: Process multiple images
- **Pipeline Parallelism**: Overlap computation
- **Memory Management**: Efficient memory usage
- **GPU Utilization**: CUDA optimization

#### Performance Monitoring:
```python
import time

def benchmark_model(model, test_data, num_runs=100):
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(test_data)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    return avg_time, fps
```

---

## Tài liệu Tham khảo

### Papers:
1. "You Only Look Once: Unified, Real-Time Object Detection" - Redmon et al.
2. "Deep Residual Learning for Image Recognition" - He et al.
3. "ImageNet Classification with Deep Convolutional Neural Networks" - Krizhevsky et al.
4. "Focal Loss for Dense Object Detection" - Lin et al.

### Datasets:
1. German Traffic Sign Recognition Benchmark (GTSRB)
2. COCO Dataset
3. ImageNet
4. Stanford Cars Dataset

### Libraries và Frameworks:
1. TensorFlow/Keras
2. PyTorch
3. OpenCV
4. Ultralytics YOLOv8
5. Albumentations

### Tools:
1. LabelImg - Annotation tool
2. Roboflow - Dataset management
3. Weights & Biases - Experiment tracking
4. TensorBoard - Visualization

---

*Tài liệu này được cập nhật thường xuyên để phản ánh những tiến bộ mới nhất trong lĩnh vực Computer Vision và Deep Learning.*