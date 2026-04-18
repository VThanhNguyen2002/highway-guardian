# Backend — FastAPI Inference API

Production-grade FastAPI backend powering the Highway Guardian two-stage inference pipeline.

---

## Tech Stack

| Package | Role |
|---------|------|
| FastAPI + Uvicorn | ASGI web server & API framework |
| PyTorch + Ultralytics | ML inference (YOLO + MobileNetV2) |
| ensemble-boxes | Weighted Boxes Fusion (WBF) |
| Pillow + OpenCV | Image preprocessing |
| SQLModel + aiosqlite | SQLite ORM (async) |
| Firebase Admin SDK | Firestore sync (`sync_detection`) |
| python-dotenv | Environment variable loading |

## Directory Structure

```
backend/
├── main.py           # FastAPI app factory + lifespan manager
├── requirements.txt  # Python dependencies
├── api/
│   └── v1/
│       └── router.py # API route registrations
├── config/
│   └── settings.py   # Pydantic Settings (reads .env)
├── core/
│   ├── inference_pipeline.py  # Two-stage YOLO + MobileNetV2 detector
│   ├── model_loader.py        # Singleton model cache (YOLO + CNN)
│   ├── pdf_parser.py          # QCVN 41:2019 PDF rule validator
│   └── task_manager.py        # Async task queue
└── db/
    └── database.py            # SQLModel async engine
```

## Setup & Running

```bash
# 1. Create virtual environment (from project root)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in: FIREBASE_KEY_PATH, model paths, etc.

# 4. Place model weights (see models/README.md)
#    models/yolov8_v2.pt
#    models/best_mobilenet_v2.pth

# 5. Start the API server (run from project root)
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Key API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `POST` | `/api/v1/detect` | Submit image for inference → returns `task_id` |
| `GET`  | `/api/v1/detect/{task_id}` | Poll task status (`PENDING` / `COMPLETED` / `FAILED`) |
| `GET`  | `/docs` | Swagger UI |
| `GET`  | `/redoc` | ReDoc UI |

## Inference Flow

```
POST /api/v1/detect  (multipart image + mode)
   → Task queued (task_id returned immediately)
       → TrafficSignDetector.detect_ensemble()
           Stage 1: YOLOv8 tiling → bounding boxes
           WBF: Merge overlapping detections
           Stage 2: MobileNetV2 classify each crop (7 Zalo classes)
           sync_detection() → Firestore 'detections' collection
   → GET /api/v1/detect/{task_id} → COMPLETED + results[]
```
