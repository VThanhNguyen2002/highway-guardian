#!/usr/bin/env python3
"""
Highway Guardian - Main Application Entry Point

Hệ thống nhận diện biển báo giao thông và phân loại xe
Sử dụng CNN và YOLOv8 cho detection và classification

Author: VThanhNguyen2002
Date: 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.config import load_config
from detection.traffic_sign_detector import TrafficSignDetector
from detection.vehicle_detector import VehicleDetector
from classification.sign_classifier import SignClassifier
from classification.vehicle_classifier import VehicleClassifier


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Highway Guardian - Traffic Sign and Vehicle Detection System"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "evaluate", "demo"],
        default="demo",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--model",
        choices=["traffic_signs", "vehicles", "both"],
        default="both",
        help="Model type to use"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input image/video path or camera index (0 for webcam)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        help="Model weights path"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup logging and environment"""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("highway_guardian", level=log_level)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    return logger, config


def initialize_models(config, args):
    """Initialize detection and classification models"""
    models = {}
    
    if args.model in ["traffic_signs", "both"]:
        # Initialize traffic sign detector and classifier
        models["traffic_sign_detector"] = TrafficSignDetector(
            weights_path=args.weights or config.get("traffic_signs", {}).get("detector_weights"),
            device=args.device,
            confidence=args.confidence
        )
        
        models["sign_classifier"] = SignClassifier(
            weights_path=config.get("traffic_signs", {}).get("classifier_weights"),
            device=args.device
        )
    
    if args.model in ["vehicles", "both"]:
        # Initialize vehicle detector and classifier
        models["vehicle_detector"] = VehicleDetector(
            weights_path=args.weights or config.get("vehicles", {}).get("detector_weights"),
            device=args.device,
            confidence=args.confidence
        )
        
        models["vehicle_classifier"] = VehicleClassifier(
            weights_path=config.get("vehicles", {}).get("classifier_weights"),
            device=args.device
        )
    
    return models


def run_training(args, config, models, logger):
    """Run training mode"""
    logger.info("Starting training mode...")
    
    if args.model == "traffic_signs":
        from training.train_traffic_signs import train_traffic_signs
        train_traffic_signs(config, args.output)
    
    elif args.model == "vehicles":
        from training.train_vehicles import train_vehicles
        train_vehicles(config, args.output)
    
    elif args.model == "both":
        from training.train_traffic_signs import train_traffic_signs
        from training.train_vehicles import train_vehicles
        
        logger.info("Training traffic signs model...")
        train_traffic_signs(config, args.output)
        
        logger.info("Training vehicles model...")
        train_vehicles(config, args.output)
    
    logger.info("Training completed!")


def run_inference(args, config, models, logger):
    """Run inference mode"""
    logger.info(f"Starting inference on: {args.input}")
    
    from inference.predictor import Predictor
    
    predictor = Predictor(models, config)
    results = predictor.predict(args.input, args.output)
    
    logger.info(f"Inference completed. Results saved to: {args.output}")
    return results


def run_evaluation(args, config, models, logger):
    """Run evaluation mode"""
    logger.info("Starting evaluation mode...")
    
    from evaluation.evaluator import Evaluator
    
    evaluator = Evaluator(models, config)
    metrics = evaluator.evaluate(args.input, args.output)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results: {metrics}")
    
    return metrics


def run_demo(args, config, models, logger):
    """Run demo mode with web interface"""
    logger.info("Starting demo mode...")
    
    try:
        from demo.web_app import create_app
        app = create_app(models, config)
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except ImportError:
        logger.warning("Web dependencies not available. Running CLI demo...")
        
        from demo.cli_demo import run_cli_demo
        run_cli_demo(models, config, args.input or 0)


def main():
    """Main application entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup environment
        logger, config = setup_environment(args)
        
        logger.info("Highway Guardian - Traffic Sign and Vehicle Detection System")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {args.device}")
        
        # Initialize models
        models = initialize_models(config, args)
        
        # Run based on mode
        if args.mode == "train":
            run_training(args, config, models, logger)
        
        elif args.mode == "inference":
            if not args.input:
                raise ValueError("Input path is required for inference mode")
            run_inference(args, config, models, logger)
        
        elif args.mode == "evaluate":
            if not args.input:
                raise ValueError("Input path is required for evaluation mode")
            run_evaluation(args, config, models, logger)
        
        elif args.mode == "demo":
            run_demo(args, config, models, logger)
        
        logger.info("Application completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()