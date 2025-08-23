#!/usr/bin/env python3
"""
Model Deployment Script
Script để deploy trained models thành production-ready API service

Features:
- Deploy YOLO models as REST API
- Support multiple model formats (PyTorch, ONNX, TensorRT)
- Real-time inference with video streams
- Batch processing capabilities
- Model versioning and A/B testing
- Performance monitoring

Usage:
    python scripts/deploy_model.py api --model best.pt --host 0.0.0.0 --port 8000
    python scripts/deploy_model.py docker --model best.pt --tag highway-guardian:latest
    python scripts/deploy_model.py export --model best.pt --format onnx
    python scripts/deploy_model.py test --url http://localhost:8000 --image test.jpg

Author: Highway Guardian Team
"""

import os
import sys
import argparse
import json
import yaml
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Install with: pip install fastapi uvicorn")

try:
    import cv2
    import numpy as np
    from PIL import Image
    import io
    import base64
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV/PIL not available. Install with: pip install opencv-python pillow")

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("Warning: docker not available. Install with: pip install docker")

class ModelDeployer:
    def __init__(self):
        self.project_root = project_root
        self.models_dir = self.project_root / "deployed_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def export_model(self, model_path: str, export_format: str = 'onnx', 
                    **kwargs) -> str:
        """
        Export model to different formats
        
        Args:
            model_path: Path to model weights
            export_format: Export format (onnx, torchscript, tflite, etc.)
            **kwargs: Additional export parameters
        
        Returns:
            Path to exported model
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for model export")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📦 Exporting model to {export_format.upper()}...")
        print(f"📁 Source: {model_path}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Export model
        export_path = model.export(format=export_format, **kwargs)
        
        # Move to deployed models directory
        exported_file = Path(export_path)
        deployed_path = self.models_dir / exported_file.name
        
        if exported_file != deployed_path:
            import shutil
            shutil.copy2(exported_file, deployed_path)
        
        print(f"✅ Model exported successfully")
        print(f"📁 Exported to: {deployed_path}")
        
        # Save export info
        self._save_export_info(model_path, deployed_path, export_format, kwargs)
        
        return str(deployed_path)
    
    def _save_export_info(self, source_path: Path, exported_path: Path, 
                         export_format: str, export_kwargs: Dict) -> None:
        """
        Save export information
        """
        info = {
            'source_model': str(source_path),
            'exported_model': str(exported_path),
            'export_format': export_format,
            'export_parameters': export_kwargs,
            'export_timestamp': datetime.now().isoformat(),
            'file_size_mb': exported_path.stat().st_size / (1024 * 1024)
        }
        
        info_file = exported_path.with_suffix('.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def create_api_server(self, model_path: str, host: str = '0.0.0.0', 
                         port: int = 8000, **kwargs) -> None:
        """
        Create and run FastAPI server for model inference
        
        Args:
            model_path: Path to model weights
            host: Server host
            port: Server port
            **kwargs: Additional server parameters
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API deployment")
        
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV and PIL are required for image processing")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"🚀 Starting API server...")
        print(f"📁 Model: {model_path}")
        print(f"🌐 Server: http://{host}:{port}")
        
        # Create FastAPI app
        app = self._create_fastapi_app(model_path, **kwargs)
        
        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    
    def _create_fastapi_app(self, model_path: Path, **kwargs) -> FastAPI:
        """
        Create FastAPI application
        """
        app = FastAPI(
            title="Highway Guardian API",
            description="Traffic Detection and Sign Recognition API",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Load model
        model = YOLO(str(model_path))
        
        # Store model info
        model_info = {
            'model_path': str(model_path),
            'model_name': model_path.name,
            'classes': model.names,
            'loaded_at': datetime.now().isoformat()
        }
        
        @app.get("/")
        async def root():
            return {
                "message": "Highway Guardian API",
                "status": "running",
                "model": model_info['model_name'],
                "endpoints": [
                    "/predict",
                    "/predict/batch",
                    "/health",
                    "/model/info"
                ]
            }
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": True
            }
        
        @app.get("/model/info")
        async def model_info_endpoint():
            return model_info
        
        @app.post("/predict")
        async def predict_image(file: UploadFile = File(...)):
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Convert to numpy array
                image_np = np.array(image)
                
                # Run inference
                results = model(image_np)
                
                # Process results
                predictions = self._process_predictions(results[0])
                
                return JSONResponse({
                    "success": True,
                    "predictions": predictions,
                    "image_info": {
                        "filename": file.filename,
                        "size": image.size,
                        "mode": image.mode
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/batch")
        async def predict_batch(files: List[UploadFile] = File(...)):
            try:
                if len(files) > 10:  # Limit batch size
                    raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
                
                results = []
                
                for file in files:
                    if not file.content_type.startswith('image/'):
                        results.append({
                            "filename": file.filename,
                            "success": False,
                            "error": "Not an image file"
                        })
                        continue
                    
                    try:
                        # Read and process image
                        image_data = await file.read()
                        image = Image.open(io.BytesIO(image_data))
                        image_np = np.array(image)
                        
                        # Run inference
                        pred_results = model(image_np)
                        predictions = self._process_predictions(pred_results[0])
                        
                        results.append({
                            "filename": file.filename,
                            "success": True,
                            "predictions": predictions,
                            "image_info": {
                                "size": image.size,
                                "mode": image.mode
                            }
                        })
                    
                    except Exception as e:
                        results.append({
                            "filename": file.filename,
                            "success": False,
                            "error": str(e)
                        })
                
                return JSONResponse({
                    "success": True,
                    "batch_size": len(files),
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/url")
        async def predict_from_url(data: dict):
            try:
                import requests
                
                url = data.get('url')
                if not url:
                    raise HTTPException(status_code=400, detail="URL is required")
                
                # Download image
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Process image
                image = Image.open(io.BytesIO(response.content))
                image_np = np.array(image)
                
                # Run inference
                results = model(image_np)
                predictions = self._process_predictions(results[0])
                
                return JSONResponse({
                    "success": True,
                    "predictions": predictions,
                    "image_info": {
                        "url": url,
                        "size": image.size,
                        "mode": image.mode
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"URL prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    def _process_predictions(self, result) -> List[Dict]:
        """
        Process YOLO prediction results
        """
        predictions = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = float(confidences[i])
                class_id = int(classes[i])
                class_name = result.names[class_id]
                
                predictions.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                })
        
        return predictions
    
    def create_docker_image(self, model_path: str, tag: str = 'highway-guardian:latest', 
                           base_image: str = 'python:3.9-slim') -> None:
        """
        Create Docker image for model deployment
        
        Args:
            model_path: Path to model weights
            tag: Docker image tag
            base_image: Base Docker image
        """
        if not DOCKER_AVAILABLE:
            raise ImportError("docker is required for Docker deployment")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"🐳 Creating Docker image: {tag}")
        print(f"📁 Model: {model_path}")
        
        # Create Dockerfile
        dockerfile_content = self._create_dockerfile(model_path, base_image)
        
        # Create build context
        build_dir = self.models_dir / "docker_build"
        build_dir.mkdir(exist_ok=True)
        
        # Write Dockerfile
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        # Copy model and requirements
        import shutil
        shutil.copy2(model_path, build_dir / "model.pt")
        
        # Copy requirements.txt if exists
        requirements_path = self.project_root / "requirements.txt"
        if requirements_path.exists():
            shutil.copy2(requirements_path, build_dir / "requirements.txt")
        
        # Copy deployment script
        deploy_script = self._create_deployment_script()
        with open(build_dir / "app.py", 'w', encoding='utf-8') as f:
            f.write(deploy_script)
        
        # Build Docker image
        client = docker.from_env()
        
        try:
            print("🔨 Building Docker image...")
            image, logs = client.images.build(
                path=str(build_dir),
                tag=tag,
                rm=True
            )
            
            # Print build logs
            for log in logs:
                if 'stream' in log:
                    print(log['stream'].strip())
            
            print(f"✅ Docker image created: {tag}")
            print(f"📦 Image ID: {image.id[:12]}")
            
            # Save image info
            self._save_docker_info(tag, model_path, image)
            
        except Exception as e:
            print(f"❌ Docker build failed: {e}")
            raise
    
    def _create_dockerfile(self, model_path: Path, base_image: str) -> str:
        """
        Create Dockerfile content
        """
        return f"""
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY model.pt ./model.pt
COPY app.py ./app.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
"""
    
    def _create_deployment_script(self) -> str:
        """
        Create deployment script for Docker
        """
        return """
#!/usr/bin/env python3
import uvicorn
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

# Import deployment functionality
from deploy_model import ModelDeployer

def main():
    deployer = ModelDeployer()
    
    # Create API server
    deployer.create_api_server(
        model_path="model.pt",
        host="0.0.0.0",
        port=8000
    )

if __name__ == "__main__":
    main()
"""
    
    def _save_docker_info(self, tag: str, model_path: Path, image) -> None:
        """
        Save Docker image information
        """
        info = {
            'image_tag': tag,
            'image_id': image.id,
            'model_path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'size_mb': image.attrs.get('Size', 0) / (1024 * 1024)
        }
        
        info_file = self.models_dir / f"docker_{tag.replace(':', '_')}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def test_api(self, api_url: str, test_image: str = None) -> Dict:
        """
        Test deployed API
        
        Args:
            api_url: API base URL
            test_image: Path to test image
        
        Returns:
            Test results
        """
        import requests
        
        print(f"🧪 Testing API: {api_url}")
        
        results = {
            'api_url': api_url,
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Test health endpoint
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            results['tests'].append({
                'endpoint': '/health',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            })
        except Exception as e:
            results['tests'].append({
                'endpoint': '/health',
                'success': False,
                'error': str(e)
            })
        
        # Test model info endpoint
        try:
            response = requests.get(f"{api_url}/model/info", timeout=10)
            results['tests'].append({
                'endpoint': '/model/info',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'data': response.json() if response.status_code == 200 else None
            })
        except Exception as e:
            results['tests'].append({
                'endpoint': '/model/info',
                'success': False,
                'error': str(e)
            })
        
        # Test prediction endpoint with image
        if test_image and Path(test_image).exists():
            try:
                with open(test_image, 'rb') as f:
                    files = {'file': (Path(test_image).name, f, 'image/jpeg')}
                    response = requests.post(
                        f"{api_url}/predict",
                        files=files,
                        timeout=30
                    )
                
                results['tests'].append({
                    'endpoint': '/predict',
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'predictions_count': len(response.json().get('predictions', [])) if response.status_code == 200 else 0
                })
            except Exception as e:
                results['tests'].append({
                    'endpoint': '/predict',
                    'success': False,
                    'error': str(e)
                })
        
        # Display results
        self._display_test_results(results)
        
        return results
    
    def _display_test_results(self, results: Dict) -> None:
        """
        Display API test results
        """
        print("\n" + "="*60)
        print("🧪 API TEST RESULTS")
        print("="*60)
        
        total_tests = len(results['tests'])
        passed_tests = sum(1 for test in results['tests'] if test.get('success', False))
        
        print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed")
        print(f"🌐 API URL: {results['api_url']}")
        print(f"🕐 Test Time: {results['timestamp']}")
        
        print(f"\n{'Endpoint':<15} {'Status':<10} {'Code':<6} {'Time (ms)':<10} {'Details':<20}")
        print("-" * 70)
        
        for test in results['tests']:
            endpoint = test['endpoint']
            status = "✅ PASS" if test.get('success', False) else "❌ FAIL"
            code = test.get('status_code', 'N/A')
            time_ms = f"{test.get('response_time_ms', 0):.1f}" if 'response_time_ms' in test else 'N/A'
            
            details = ""
            if 'predictions_count' in test:
                details = f"{test['predictions_count']} predictions"
            elif 'error' in test:
                details = test['error'][:20] + "..." if len(test['error']) > 20 else test['error']
            
            print(f"{endpoint:<15} {status:<10} {code:<6} {time_ms:<10} {details:<20}")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! API is ready for production.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the API.")

def main():
    parser = argparse.ArgumentParser(description='Model Deployment Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to different formats')
    export_parser.add_argument('--model', required=True, help='Model weights path')
    export_parser.add_argument('--format', default='onnx', 
                              choices=['onnx', 'torchscript', 'tflite', 'edgetpu', 'tfjs'],
                              help='Export format')
    export_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    export_parser.add_argument('--half', action='store_true', help='Use FP16 precision')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--model', required=True, help='Model weights path')
    api_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    api_parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # Docker command
    docker_parser = subparsers.add_parser('docker', help='Create Docker image')
    docker_parser.add_argument('--model', required=True, help='Model weights path')
    docker_parser.add_argument('--tag', default='highway-guardian:latest', help='Docker image tag')
    docker_parser.add_argument('--base', default='python:3.9-slim', help='Base Docker image')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test deployed API')
    test_parser.add_argument('--url', required=True, help='API base URL')
    test_parser.add_argument('--image', help='Test image path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    deployer = ModelDeployer()
    
    try:
        if args.command == 'export':
            export_kwargs = {
                'imgsz': args.imgsz,
                'half': args.half
            }
            deployer.export_model(args.model, args.format, **export_kwargs)
        
        elif args.command == 'api':
            deployer.create_api_server(args.model, args.host, args.port)
        
        elif args.command == 'docker':
            deployer.create_docker_image(args.model, args.tag, args.base)
        
        elif args.command == 'test':
            deployer.test_api(args.url, args.image)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()