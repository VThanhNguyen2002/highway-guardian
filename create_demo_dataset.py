#!/usr/bin/env python3
"""
Create Demo Dataset for Testing Training Pipeline
Tạo dataset demo nhỏ để test training pipeline mà không cần Kaggle API

Author: Highway Guardian Team
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import random
import yaml

def create_demo_images(output_dir, num_images=50):
    """Tạo ảnh demo với bounding boxes đơn giản"""
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Tạo ảnh 640x640 với background ngẫu nhiên
        img = Image.new('RGB', (640, 640), color=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
        draw = ImageDraw.Draw(img)
        
        # Tạo 1-3 objects ngẫu nhiên
        num_objects = random.randint(1, 3)
        labels = []
        
        for j in range(num_objects):
            # Tạo bounding box ngẫu nhiên
            x1 = random.randint(50, 400)
            y1 = random.randint(50, 400)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            
            # Đảm bảo không vượt quá kích thước ảnh
            x2 = min(x2, 590)
            y2 = min(y2, 590)
            
            # Vẽ rectangle (giả lập xe hoặc biển báo)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=3)
            
            # Convert to YOLO format (normalized)
            center_x = (x1 + x2) / 2 / 640
            center_y = (y1 + y2) / 2 / 640
            width = (x2 - x1) / 640
            height = (y2 - y1) / 640
            
            # Class 0 (car/vehicle)
            labels.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Lưu ảnh
        img_path = images_dir / f"demo_{i:04d}.jpg"
        img.save(img_path)
        
        # Lưu label
        label_path = labels_dir / f"demo_{i:04d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"✅ Created {num_images} demo images in {output_dir}")

def create_dataset_structure():
    """Tạo cấu trúc dataset hoàn chỉnh"""
    base_dir = Path("data/demo_dataset")
    
    # Tạo cấu trúc thư mục
    splits = ['train', 'val', 'test']
    for split in splits:
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Tạo ảnh cho từng split
    create_demo_images(base_dir / "train", 100)  # 100 ảnh train
    create_demo_images(base_dir / "val", 20)     # 20 ảnh val
    create_demo_images(base_dir / "test", 10)    # 10 ảnh test
    
    # Tạo data.yaml
    data_config = {
        'path': str(base_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': 1,
        'names': ['vehicle']
    }
    
    with open(base_dir / "data.yaml", 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created complete dataset structure at {base_dir}")
    print(f"📁 Structure:")
    print(f"   {base_dir}/")
    print(f"   ├── train/ (100 images)")
    print(f"   ├── val/ (20 images)")
    print(f"   ├── test/ (10 images)")
    print(f"   └── data.yaml")
    
    return base_dir

def create_demo_config(dataset_path):
    """Tạo config file cho demo training"""
    config = {
        'data': {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['vehicle']
        },
        'model': {
            'architecture': 'yolov8n',
            'pretrained': True,
            'weights': 'yolov8n.pt'
        },
        'training': {
            'epochs': 10,  # Ít epochs cho demo
            'batch_size': 8,  # Batch size nhỏ
            'image_size': 640,
            'device': 'auto',
            'workers': 2,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 1.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        },
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        },
        'validation': {
            'save_period': -1,
            'patience': 100,
            'conf': 0.25,
            'iou': 0.7,
            'max_det': 300
        },
        'output': {
            'project': 'runs/demo_training',
            'name': 'demo_experiment',
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'plots': True,
            'verbose': True
        }
    }
    
    config_path = Path("configs/demo_training_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created demo config at {config_path}")
    return config_path

if __name__ == "__main__":
    print("🚀 Creating demo dataset for training pipeline testing...")
    
    # Tạo dataset
    dataset_path = create_dataset_structure()
    
    # Tạo config
    config_path = create_demo_config(dataset_path)
    
    print("\n🎯 Next steps:")
    print(f"1. Run training: python src/training/scripts/train_car_detection.py --config {config_path}")
    print(f"2. Or use simple YOLO: from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='{dataset_path}/data.yaml', epochs=10)")
    print("\n✨ Demo dataset ready for testing!")