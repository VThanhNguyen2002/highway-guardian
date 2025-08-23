# Training Results Directory

Thư mục này chứa kết quả training từ các experiment khác nhau.

## Cấu trúc thư mục

```
runs/
├── detect/                      # Kết quả detection training
│   ├── car_yolo112/            # Car detection experiment
│   │   ├── weights/
│   │   │   ├── best.pt         # Model tốt nhất
│   │   │   └── last.pt         # Model cuối cùng
│   │   ├── results.csv         # Training metrics
│   │   ├── confusion_matrix.png
│   │   ├── F1_curve.png
│   │   ├── P_curve.png
│   │   ├── PR_curve.png
│   │   ├── R_curve.png
│   │   └── labels.jpg          # Label distribution
│   ├── sign_yolo85/            # Sign detection experiment
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── results.csv
│   │   ├── confusion_matrix.png
│   │   └── ...
│   └── sign_yolo85_ft/         # Fine-tuned sign detection
│       └── ...
└── classify/                    # Kết quả classification training (nếu có)
    └── ...
```

## Mô tả các file kết quả

### Weights
- `best.pt`: Model có performance tốt nhất trên validation set
- `last.pt`: Model ở epoch cuối cùng

### Metrics và Plots
- `results.csv`: Chi tiết metrics qua từng epoch
- `confusion_matrix.png`: Ma trận nhầm lẫn
- `F1_curve.png`: Đường cong F1-score
- `P_curve.png`: Đường cong Precision
- `PR_curve.png`: Đường cong Precision-Recall
- `R_curve.png`: Đường cong Recall
- `labels.jpg`: Phân bố nhãn trong dataset

## Kết quả Training từ Notebook

Dựa trên kết quả từ `Car_Traffic_Detection.ipynb`:

### Car Detection (car_yolo112)
- **Model**: YOLOv8n
- **Epochs**: 50
- **Dataset**: 16,185 images (12,949 train, 1,618 val, 1,618 test)
- **Final mAP50**: ~0.85-0.90 (ước tính từ training log)
- **Classes**: 1 (car)

### Sign Detection (sign_yolo85)
- **Model**: YOLOv8s
- **Epochs**: 120
- **Dataset**: 1,170 images (900 train, 180 val, 90 test)
- **Final mAP50**: 0.863
- **Final mAP50-95**: 0.593
- **Final Precision**: 0.845
- **Final Recall**: 0.742
- **Classes**: 25 (các loại biển báo giao thông Việt Nam)

### Class Performance (Sign Detection)
Các class có performance tốt nhất:
- Cấm đỗ xe: Precision=0.975, Recall=0.92
- Cấm dừng & đỗ: Precision=0.92, Recall=0.929
- Cấm xe tải: Precision=1.0, Recall=0.976
- Hướng đi: Precision=0.954, Recall=0.941

Các class cần cải thiện:
- Giới hạn chiều cao: Precision=1.0, Recall=0.0
- Công trường: Precision=0.415, Recall=0.829
- Hạn chế tốc độ: Precision=0.802, Recall=0.556

## Lưu ý

- Kết quả training được lưu tự động bởi YOLO
- Mỗi experiment có tên unique với timestamp
- Best model được chọn dựa trên validation mAP50
- Có thể resume training từ last.pt checkpoint