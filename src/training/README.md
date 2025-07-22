# Training Module

Thư mục này chứa tất cả các file và script liên quan đến việc training model.

## Cấu trúc thư mục

```
training/
├── README.md                    # File hướng dẫn này
├── notebooks/                   # Jupyter notebooks cho training
│   ├── Car_Traffic_Detection.ipynb
│   └── model_evaluation.ipynb
├── scripts/                     # Python scripts cho training
│   ├── train_car_detection.py
│   ├── train_sign_detection.py
│   ├── evaluate_model.py
│   └── utils.py
├── configs/                     # File cấu hình training
│   ├── car_detection_config.yaml
│   ├── sign_detection_config.yaml
│   └── training_params.yaml
├── runs/                        # Kết quả training
│   ├── car_detection/
│   │   ├── experiment_1/
│   │   ├── experiment_2/
│   │   └── best_model/
│   └── sign_detection/
│       ├── experiment_1/
│       ├── experiment_2/
│       └── best_model/
└── models/                      # Pretrained models và checkpoints
    ├── pretrained/
    ├── checkpoints/
    └── final_models/
```

## Mô tả các thư mục

### notebooks/
Chứa các Jupyter notebook để thực hiện training và phân tích:
- `Car_Traffic_Detection.ipynb`: Notebook chính cho training cả car detection và sign detection
- `model_evaluation.ipynb`: Notebook để đánh giá và so sánh các model

### scripts/
Chứa các Python script để training:
- `train_car_detection.py`: Script training cho car detection
- `train_sign_detection.py`: Script training cho sign detection  
- `evaluate_model.py`: Script đánh giá model
- `utils.py`: Các utility functions

### configs/
Chứa các file cấu hình YAML:
- `car_detection_config.yaml`: Cấu hình cho car detection
- `sign_detection_config.yaml`: Cấu hình cho sign detection
- `training_params.yaml`: Các tham số training chung

### runs/
Chứa kết quả training được tổ chức theo từng experiment:
- Mỗi experiment có folder riêng với timestamp
- Chứa weights, logs, metrics, plots
- Folder `best_model/` chứa model tốt nhất

### models/
Chứa các model files:
- `pretrained/`: Pretrained models (YOLOv8, YOLOv11)
- `checkpoints/`: Model checkpoints trong quá trình training
- `final_models/`: Model cuối cùng đã training xong

## Hướng dẫn sử dụng

1. **Training với Jupyter Notebook:**
   ```bash
   cd src/training/notebooks
   jupyter notebook Car_Traffic_Detection.ipynb
   ```

2. **Training với Python Script:**
   ```bash
   cd src/training/scripts
   python train_car_detection.py --config ../configs/car_detection_config.yaml
   python train_sign_detection.py --config ../configs/sign_detection_config.yaml
   ```

3. **Đánh giá Model:**
   ```bash
   python evaluate_model.py --model_path ../runs/car_detection/best_model/weights/best.pt
   ```