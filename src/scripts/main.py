"""
Highway Guardian API - Main Application
WebSocket support for real-time camera detection
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import io
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config.settings import (
    API_TITLE, API_VERSION, CORS_ORIGINS,
    YOLO_MODELS_DIR, CNN_MODELS_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD, API_HOST, API_PORT
)
from services.detection_service import (
    yolo_predict, cnn_predict, two_stage_predict
)
from utils.model_manager import get_available_models

# ============================================================================
# WEBSOCKET CONFIG - Hardcoded model names for real-time detection
# ============================================================================
WS_YOLO_MODEL = "best.pt"
WS_CNN_MODEL = "bien_bao_mobilenetv2_AUGMENTED_BALANCED_model.h5"
WS_CONFIDENCE_THRESHOLD = 0.25

# ============================================================================
# LIFESPAN
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("🚀 STARTUP CHECK: Checking Models...")
    if os.path.exists(YOLO_MODELS_DIR):
        print(f"✅ YOLO Directory: {os.listdir(YOLO_MODELS_DIR)}")
    if os.path.exists(CNN_MODELS_DIR):
        print(f"✅ CNN Directory: {os.listdir(CNN_MODELS_DIR)}")
    print(f"📡 WebSocket models: YOLO={WS_YOLO_MODEL}, CNN={WS_CNN_MODEL}")
    print("="*50 + "\n")
    yield
    print("🛑 Server shutting down...")

# ============================================================================
# APP INITIALIZATION
# ============================================================================
app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HTTP ENDPOINTS (Existing)
# ============================================================================
@app.get("/")
def read_root():
    return {"status": "running", "service": API_TITLE}

@app.get("/models")
def get_models():
    return get_available_models(YOLO_MODELS_DIR, CNN_MODELS_DIR)

@app.post("/predict")
async def predict(model_name: str = Form(...), model_type: str = Form("yolo"), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if model_type == "yolo":
            predictions = yolo_predict(image, model_name, YOLO_MODELS_DIR)
        elif model_type == "cnn":
            prediction = cnn_predict(image, model_name, CNN_MODELS_DIR)
            predictions = [prediction]
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        return {"predictions": predictions, "success": True}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "predictions": [], "success": False}

@app.post("/predict_two_stage")
async def predict_two_stage_endpoint(yolo_model: str = Form(...), cnn_model: str = Form(...), 
                                   file: UploadFile = File(...), confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        predictions = two_stage_predict(image, yolo_model, cnn_model, YOLO_MODELS_DIR, CNN_MODELS_DIR, confidence_threshold)
        return {"predictions": predictions, "success": True}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "predictions": [], "success": False}

# ============================================================================
# WEBSOCKET ENDPOINT - Real-time Camera Detection
# ============================================================================
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera detection.
    
    Protocol:
    - Client sends: raw image bytes (JPEG/PNG)
    - Server responds: JSON with predictions
    
    Uses hardcoded models for consistent performance.
    """
    await websocket.accept()
    print(f"📡 WebSocket connected: {websocket.client}")
    
    frame_count = 0
    
    try:
        while True:
            # Receive raw image bytes from client
            data = await websocket.receive_bytes()
            frame_count += 1
            
            try:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(data))
                
                # Convert to RGB if needed (handle PNG with alpha channel)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Run two-stage prediction (YOLO + CNN)
                predictions = two_stage_predict(
                    image=image,
                    yolo_model_name=WS_YOLO_MODEL,
                    cnn_model_name=WS_CNN_MODEL,
                    yolo_dir=YOLO_MODELS_DIR,
                    cnn_dir=CNN_MODELS_DIR,
                    conf_threshold=WS_CONFIDENCE_THRESHOLD
                )
                
                # Send results back to client
                await websocket.send_json({
                    "success": True,
                    "predictions": predictions,
                    "frame": frame_count
                })
                
            except Exception as e:
                # Send error but keep connection alive
                print(f"⚠️ Frame {frame_count} error: {str(e)}")
                await websocket.send_json({
                    "success": False,
                    "error": str(e),
                    "predictions": [],
                    "frame": frame_count
                })
                
    except WebSocketDisconnect:
        print(f"📴 WebSocket disconnected: {websocket.client} (processed {frame_count} frames)")
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
        traceback.print_exc()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
