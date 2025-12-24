"""
Model Manager - Handle loading and caching of ML models
Strategy: Reconstruct Architecture -> Load Weights (Fixes Config Errors)
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from typing import Dict
from ultralytics import YOLO
from fastapi import HTTPException
import tensorflow as tf
from tensorflow import keras

# =========================================================================
# 🏗️ RECONSTRUCT ARCHITECTURE (Tái tạo kiến trúc MobileNetV2)
# =========================================================================
def build_mobilenetv2_structure(num_classes=57):
    """
    Tạo lại kiến trúc MobileNetV2 sạch sẽ để nạp trọng số
    """
    print(f"🏗️ Reconstructing MobileNetV2 architecture with {num_classes} classes...")
    
    # Base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    
    # Head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model
# =========================================================================

class ModelCache:
    def __init__(self, max_size: int = 5):
        self.cache: Dict = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
    
    def get(self, key: str):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value):
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        self.cache[key] = value
        self.access_count[key] = 1

yolo_cache = ModelCache()
cnn_cache = ModelCache()

def load_yolo_model(model_name: str, models_dir: str) -> YOLO:
    cached = yolo_cache.get(model_name)
    if cached: return cached
    
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"YOLO model not found at {model_path}")
    
    try:
        print(f"Loading YOLO model: {model_name}")
        model = YOLO(model_path)
        yolo_cache.set(model_name, model)
        print(f"✓ YOLO model loaded")
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading YOLO: {str(e)}")

def load_cnn_model(model_name: str, models_dir: str):
    cached = cnn_cache.get(model_name)
    if cached: return cached
    
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"CNN model not found at {model_path}")
    
    print(f"🔄 Loading CNN model from: {model_path}")
    model = None
    
    try:
        # CÁCH 1: Thử load chuẩn
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        print(f"⚠️ Config error detected. Switching to Weights-Only loading...")
        try:
            # CÁCH 2: Tái tạo kiến trúc & Load Weights (An toàn nhất)
            model = build_mobilenetv2_structure(num_classes=57)
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
        except Exception as e_fatal:
            raise HTTPException(status_code=500, detail=f"CRITICAL: Cannot load CNN model. Error: {str(e_fatal)}")

    cnn_cache.set(model_name, model)
    print(f"✓ CNN model '{model_name}' loaded successfully!")
    return model

def get_available_models(yolo_dir: str, cnn_dir: str):
    models = {"yolo": [], "cnn": []}
    if os.path.exists(yolo_dir):
        models["yolo"] = [f for f in os.listdir(yolo_dir) if f.endswith(".pt")]
    if os.path.exists(cnn_dir):
        models["cnn"] = [f for f in os.listdir(cnn_dir) if f.endswith((".h5", ".keras"))]
    return models