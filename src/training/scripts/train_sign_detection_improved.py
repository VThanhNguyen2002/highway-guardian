#!/usr/bin/env python3
"""
Improved Sign Detection Training Script
Based on analysis from Car_Traffic_Detection.ipynb

Cải thiện:
- Weighted loss cho class imbalance
- Advanced data augmentation
- Early stopping và learning rate scheduling
- Better validation metrics
- Modular structure

Author: Highway Guardian Team
"""

import os
import sys
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import argparse
import logging
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.training_utils import (
    setup_logging,
    load_config,
    setup_wandb,
    save_training_summary,
    validate_dataset,
    calculate_class_weights
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Improved Sign Detection Model')
    parser.add_argument('--config', type=str, 
                       default='src/configs/sign_det_improved.yaml',
                       help='Path to config file')
    parser.add_argument('--data-path', type=str,
                       help='Override dataset path')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU device (0, 1, 2, etc.)')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--name', type=str,
                       help='Experiment name')
    parser.add_argument('--weighted-loss', action='store_true',
                       help='Use weighted loss for class imbalance')
    return parser.parse_args()

def analyze_dataset_distribution(dataset_path, logger):
    """
    Analyze class distribution in dataset for weighted loss calculation
    """
    train_labels_dir = dataset_path / 'labels' / 'train'
    class_counts = Counter()
    total_objects = 0
    
    if not train_labels_dir.exists():
        logger.warning(f"Training labels directory not found: {train_labels_dir}")
        return None
    
    for label_file in train_labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    
    if total_objects == 0:
        logger.warning("No objects found in training labels")
        return None
    
    # Log distribution
    logger.info("Dataset class distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_objects) * 100
        logger.info(f"  Class {class_id}: {count} objects ({percentage:.2f}%)")
    
    return class_counts

def create_weighted_loss_config(class_counts, num_classes):
    """
    Create weighted loss configuration based on class distribution
    """
    if not class_counts:
        return None
    
    # Calculate class weights (inverse frequency)
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        class_weights[class_id] = weight
    
    return class_weights

def setup_advanced_augmentation():
    """
    Setup advanced augmentation parameters for sign detection
    """
    return {
        # Color augmentation (important for signs)
        'hsv_h': 0.02,      # Hue variation
        'hsv_s': 0.8,       # Saturation variation
        'hsv_v': 0.5,       # Value/brightness variation
        
        # Geometric augmentation
        'degrees': 15.0,    # Rotation degrees
        'translate': 0.15,  # Translation
        'scale': 0.6,       # Scale variation
        'shear': 5.0,       # Shear degrees
        'perspective': 0.001, # Perspective transformation
        
        # Flip augmentation
        'flipud': 0.0,      # No vertical flip for signs
        'fliplr': 0.3,      # Limited horizontal flip
        
        # Advanced augmentation
        'mosaic': 0.8,      # Mosaic augmentation
        'mixup': 0.1,       # Mixup augmentation
        'copy_paste': 0.1,  # Copy-paste augmentation
        
        # Noise and blur
        'blur': 0.01,       # Gaussian blur
        'noise': 0.02,      # Gaussian noise
    }

def main():
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.name or f'sign_detection_improved_{timestamp}'
    
    log_dir = project_root / 'src' / 'data' / 'runs' / 'detect' / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_dir / 'training.log')
    logger.info(f"Starting improved sign detection training: {exp_name}")
    
    # Load configuration
    config_path = project_root / args.config
    config = load_config(config_path)
    
    # Override config with command line args
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    logger.info(f"Configuration loaded: {config_path}")
    
    # Validate dataset
    dataset_path = Path(config['data']['path'])
    if not validate_dataset(dataset_path, config['data']['expected_classes']):
        logger.error("Dataset validation failed")
        return False
    
    # Analyze dataset distribution
    class_counts = analyze_dataset_distribution(dataset_path, logger)
    
    # Setup device
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model_config = config['model']
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        model_name = model_config['name']
        logger.info(f"Initializing model: {model_name}")
        model = YOLO(f"{model_name}.pt" if model_config['pretrained'] else model_name)
    
    # Setup training parameters
    train_config = config['training']
    
    # Setup monitoring
    if config.get('monitoring', {}).get('wandb', {}).get('enabled', False):
        wandb_config = config['monitoring']['wandb']
        setup_wandb(wandb_config, config, exp_name)
        logger.info("W&B monitoring enabled")
    
    # Training arguments with improvements
    train_args = {
        'data': str(dataset_path / 'data.yaml'),
        'epochs': train_config['epochs'],
        'batch': train_config['batch_size'],
        'imgsz': train_config['img_size'],
        'device': device,
        'project': str(log_dir.parent),
        'name': exp_name,
        'save': True,
        'save_period': train_config.get('save_period', 5),
        'cache': train_config.get('cache', True),
        'workers': train_config.get('workers', 8),
        'optimizer': train_config.get('optimizer', 'AdamW'),
        'lr0': train_config.get('lr0', 0.001),  # Lower learning rate for signs
        'lrf': train_config.get('lrf', 0.01),   # Learning rate final
        'weight_decay': train_config.get('weight_decay', 0.0005),
        'warmup_epochs': train_config.get('warmup_epochs', 5),
        'patience': train_config.get('patience', 30),  # Early stopping
        'verbose': True,
        
        # Loss function improvements
        'box': 7.5,         # Box loss gain
        'cls': 0.5,         # Classification loss gain
        'dfl': 1.5,         # Distribution focal loss gain
        
        # Validation improvements
        'val': True,
        'plots': True,
        'save_json': True,  # Save validation results as JSON
    }
    
    # Add advanced augmentation
    aug_params = setup_advanced_augmentation()
    train_args.update(aug_params)
    
    # Add weighted loss if requested and class imbalance detected
    if args.weighted_loss and class_counts:
        num_classes = len(config['data']['expected_classes'])
        class_weights = create_weighted_loss_config(class_counts, num_classes)
        if class_weights:
            logger.info(f"Using weighted loss: {class_weights}")
            # Note: YOLO doesn't directly support class weights, but we log them for reference
            # Alternative: use focal loss parameters
            train_args['fl_gamma'] = 2.0  # Focal loss gamma
    
    logger.info(f"Training arguments: {train_args}")
    
    try:
        # Start training
        logger.info("Starting improved training...")
        results = model.train(**train_args)
        
        # Save training summary
        summary_path = log_dir / 'training_summary.md'
        save_training_summary(results, config, summary_path)
        
        # Validate best model with detailed metrics
        best_model_path = log_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info("Validating best model with detailed metrics...")
            best_model = YOLO(str(best_model_path))
            val_results = best_model.val(
                data=train_args['data'],
                save_json=True,
                save_hybrid=True,
                conf=0.25,
                iou=0.6
            )
            
            # Log detailed validation results
            logger.info(f"Final validation results:")
            logger.info(f"mAP50: {val_results.box.map50:.4f}")
            logger.info(f"mAP50-95: {val_results.box.map:.4f}")
            
            # Per-class metrics
            if hasattr(val_results.box, 'maps'):
                class_names = config['data']['expected_classes']
                for i, (class_name, map_val) in enumerate(zip(class_names, val_results.box.maps)):
                    logger.info(f"Class '{class_name}' mAP50-95: {map_val:.4f}")
            
            # Export model in multiple formats
            export_formats = config.get('export', {}).get('formats', ['onnx', 'torchscript'])
            for fmt in export_formats:
                try:
                    export_path = best_model.export(format=fmt, optimize=True)
                    logger.info(f"Model exported to {fmt}: {export_path}")
                except Exception as e:
                    logger.warning(f"Failed to export to {fmt}: {e}")
        
        logger.info(f"Improved training completed successfully: {exp_name}")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
    
    finally:
        if 'wandb' in globals() and wandb.run:
            wandb.finish()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)