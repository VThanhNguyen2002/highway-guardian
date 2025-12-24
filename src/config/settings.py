"""
Configuration settings for Highway Guardian API
"""
import os
import sys

# API Settings
API_TITLE = "Highway Guardian API"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# CORS Settings
CORS_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:3000",
    "*", # Cho phép tất cả trong môi trường dev để tránh lỗi kết nối
]

# --- MODEL DIRECTORIES CONFIGURATION ---
# Lấy đường dẫn gốc của file này
CURRENT_FILE = os.path.abspath(__file__)
# Config dir
CONFIG_DIR = os.path.dirname(CURRENT_FILE)
# Src dir
SRC_DIR = os.path.dirname(CONFIG_DIR)
# Project root
BASE_DIR = os.path.dirname(SRC_DIR)

# Mặc định (cho chạy local)
MODELS_BASE_DIR = os.path.join(BASE_DIR, "models")

# Cấu hình cho Docker
# Kiểm tra xem folder /app/models có tồn tại không (do Volume mount)
if os.path.exists("/app/models"):
    print("🐳 DETECTED DOCKER ENVIRONMENT: Using /app/models")
    MODELS_BASE_DIR = "/app/models"
else:
    print(f"💻 DETECTED LOCAL ENVIRONMENT: Using {MODELS_BASE_DIR}")

# Đường dẫn chi tiết
YOLO_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "yolo")
CNN_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "cnn")

# Detection Settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
CNN_INPUT_SIZE = (224, 224)

# Cache Settings
ENABLE_MODEL_CACHE = True
MAX_CACHE_SIZE = 5

# Debug Print (Sẽ hiện trong log Docker)
print(f"📂 CONFIG: YOLO DIR = {YOLO_MODELS_DIR}")
print(f"📂 CONFIG: CNN DIR  = {CNN_MODELS_DIR}")