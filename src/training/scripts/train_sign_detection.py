#!/usr/bin/env python3
"""
Sign Detection Training Script

Script này thực hiện training model YOLOv8 cho việc phát hiện biển báo giao thông Việt Nam.
Sử dụng cấu hình từ file YAML và có thể chạy độc lập.

Usage:
    python train_sign_detection.py --config ../configs/sign_detection_config.yaml
    python train_sign_detection.py --config ../configs/sign_detection_config.yaml --resume

Author: Highway Guardian Team
Date: 2024
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, create_experiment_dir, save_training_config


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_data_yaml(config, output_path):
    """Create YOLO data configuration file."""
    data_config = {
        'path': config['data']['path'],
        'train': config['data']['train'],
        'val': config['data']['val'],
        'test': config['data']['test'],
        'nc': config['data']['nc'],
        'names': config['data']['names']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    return output_path


def train_sign_detection(config_path, resume=False):
    """Main training function for sign detection."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging('sign_detection_training')
    logger.info(f"Starting sign detection training with config: {config_path}")
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(
        config['output']['project'], 
        config['output']['name']
    )
    
    # Save training configuration
    save_training_config(config, experiment_dir)
    
    # Create data YAML file
    data_yaml_path = os.path.join(experiment_dir, 'data.yaml')
    create_data_yaml(config, data_yaml_path)
    
    # Initialize model
    if resume and os.path.exists(os.path.join(experiment_dir, 'weights', 'last.pt')):
        model_path = os.path.join(experiment_dir, 'weights', 'last.pt')
        logger.info(f"Resuming training from: {model_path}")
    else:
        model_path = config['model']['weights']
        logger.info(f"Starting new training with: {model_path}")
    
    model = YOLO(model_path)
    
    # Training arguments
    train_args = {
        'data': data_yaml_path,
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch_size'],
        'imgsz': config['training']['image_size'],
        'device': config['training']['device'],
        'workers': config['training']['workers'],
        'optimizer': config['training']['optimizer'],
        'lr0': config['training']['lr0'],
        'lrf': config['training']['lrf'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'box': config['training']['box'],
        'cls': config['training']['cls'],
        'dfl': config['training']['dfl'],
        'project': config['output']['project'],
        'name': config['output']['name'],
        'save_period': config['validation']['save_period'],
        'patience': config['validation']['patience'],
        'plots': config['output']['plots'],
        'verbose': config['output']['verbose'],
        'resume': resume
    }
    
    # Add augmentation parameters
    for key, value in config['augmentation'].items():
        train_args[key] = value
    
    logger.info("Starting training...")
    logger.info(f"Training arguments: {train_args}")
    
    # Start training
    try:
        results = model.train(**train_args)
        logger.info("Training completed successfully!")
        
        # Log final metrics
        logger.info(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # Log class-specific metrics
        logger.info("\n=== Class-specific Performance ===")
        for i, class_name in enumerate(config['data']['names']):
            precision = results.results_dict.get(f'metrics/precision(B)_class_{i}', 'N/A')
            recall = results.results_dict.get(f'metrics/recall(B)_class_{i}', 'N/A')
            logger.info(f"{class_name}: Precision={precision}, Recall={recall}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train sign detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Start training
    try:
        results = train_sign_detection(args.config, args.resume)
        print("\n=== Training Summary ===")
        print(f"Training completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()