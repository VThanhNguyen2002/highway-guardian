#!/usr/bin/env python3
"""
Training Utility Functions
Support functions for training scripts

Author: Highway Guardian Team
"""

import os
import yaml
import logging
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import torch
import json

def setup_logging(log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('highway_guardian')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def setup_wandb(wandb_config: Dict, training_config: Dict, experiment_name: str) -> None:
    """
    Setup Weights & Biases monitoring
    
    Args:
        wandb_config: W&B configuration
        training_config: Training configuration
        experiment_name: Experiment name
    """
    wandb.init(
        project=wandb_config.get('project', 'highway-guardian'),
        name=experiment_name,
        config=training_config,
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        save_code=True
    )

def validate_dataset(dataset_path: Path, expected_classes: List[str]) -> bool:
    """
    Validate dataset structure and content
    
    Args:
        dataset_path: Path to dataset
        expected_classes: List of expected class names
    
    Returns:
        True if dataset is valid
    """
    logger = logging.getLogger('highway_guardian')
    
    # Check basic structure
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            return False
    
    # Check data.yaml file
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        logger.warning(f"data.yaml not found: {data_yaml}")
        # Create basic data.yaml
        create_data_yaml(dataset_path, expected_classes)
    
    # Check if images and labels exist
    train_images = list((dataset_path / 'images/train').glob('*'))
    train_labels = list((dataset_path / 'labels/train').glob('*.txt'))
    
    if len(train_images) == 0:
        logger.error("No training images found")
        return False
    
    if len(train_labels) == 0:
        logger.error("No training labels found")
        return False
    
    logger.info(f"Dataset validation passed: {len(train_images)} images, {len(train_labels)} labels")
    return True

def create_data_yaml(dataset_path: Path, class_names: List[str]) -> None:
    """
    Create data.yaml file for YOLO training
    
    Args:
        dataset_path: Path to dataset
        class_names: List of class names
    """
    data_config = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    data_yaml_path = dataset_path / 'data.yaml'
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

def calculate_class_weights(dataset_path: Path, num_classes: int) -> Optional[Dict[int, float]]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        dataset_path: Path to dataset
        num_classes: Number of classes
    
    Returns:
        Dictionary of class weights
    """
    from collections import Counter
    
    train_labels_dir = dataset_path / 'labels' / 'train'
    if not train_labels_dir.exists():
        return None
    
    class_counts = Counter()
    total_objects = 0
    
    for label_file in train_labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    
    if total_objects == 0:
        return None
    
    # Calculate inverse frequency weights
    class_weights = {}
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weight = total_objects / (num_classes * count)
        class_weights[class_id] = weight
    
    return class_weights

def save_training_summary(results, config: Dict, summary_path: Path) -> None:
    """
    Save training summary to markdown file
    
    Args:
        results: Training results from YOLO
        config: Training configuration
        summary_path: Path to save summary
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    summary_content = f"""# Training Summary

**Date:** {timestamp}
**Model:** {config.get('model', {}).get('name', 'Unknown')}
**Dataset:** {config.get('data', {}).get('path', 'Unknown')}

## Configuration

### Model Settings
- Architecture: {config.get('model', {}).get('name', 'Unknown')}
- Pretrained: {config.get('model', {}).get('pretrained', 'Unknown')}
- Image Size: {config.get('training', {}).get('img_size', 'Unknown')}

### Training Settings
- Epochs: {config.get('training', {}).get('epochs', 'Unknown')}
- Batch Size: {config.get('training', {}).get('batch_size', 'Unknown')}
- Learning Rate: {config.get('training', {}).get('lr0', 'Unknown')}
- Optimizer: {config.get('training', {}).get('optimizer', 'Unknown')}
- Device: {config.get('training', {}).get('device', 'Unknown')}

## Results

### Final Metrics
"""
    
    # Add results if available
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        summary_content += f"""
- mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}
- mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}
- Precision: {metrics.get('metrics/precision(B)', 'N/A')}
- Recall: {metrics.get('metrics/recall(B)', 'N/A')}
"""
    
    summary_content += f"""

### Training Details
- Save Directory: {getattr(results, 'save_dir', 'Unknown')}
- Best Weights: {getattr(results, 'save_dir', 'Unknown')}/weights/best.pt
- Last Weights: {getattr(results, 'save_dir', 'Unknown')}/weights/last.pt

## Files Generated
- Training plots: results.png, confusion_matrix.png
- Validation results: val_batch*.jpg
- Training curves: results.csv

---
*Generated automatically by Highway Guardian training pipeline*
"""
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

def get_device_info() -> Dict[str, Union[str, int]]:
    """
    Get device information for training
    
    Returns:
        Device information dictionary
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    }
    
    return device_info

def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of experiment
    
    Returns:
        Path to created experiment directory
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def save_training_config(config: Dict, experiment_dir: Path) -> None:
    """
    Save training configuration to experiment directory
    
    Args:
        config: Training configuration
        experiment_dir: Experiment directory
    """
    config_path = experiment_dir / 'training_config.yaml'
    save_config(config, config_path)
    
    # Also save as JSON for easier parsing
    json_path = experiment_dir / 'training_config.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def monitor_gpu_usage() -> Optional[Dict[str, float]]:
    """
    Monitor GPU usage during training
    
    Returns:
        GPU usage statistics or None if not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        return {
            'memory_used_mb': mem_info.used / 1024**2,
            'memory_total_mb': mem_info.total / 1024**2,
            'memory_percent': (mem_info.used / mem_info.total) * 100,
            'gpu_percent': gpu_util.gpu
        }
    except ImportError:
        # pynvml not available
        return {
            'memory_used_mb': torch.cuda.memory_allocated() / 1024**2,
            'memory_cached_mb': torch.cuda.memory_reserved() / 1024**2
        }
    except Exception:
        return None